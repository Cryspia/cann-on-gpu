# metal/ — CANN / AscendCL API 的 Apple Metal 后端

> English: [README.md](README.md)

本目录用 Apple Metal 生态在 Apple 芯片 GPU 上重新实现 [`../include/`](../include/) 声明的 CANN / AscendCL / aclnn / HCCL API 契约。
产物为 `lib/libascendcl.dylib` + `lib/libhccl.dylib`(+ `lib/default.metallib`);应用像链接真正的昇腾库一样链接它们——只暴露 API 表面,不暴露 Metal。`../tests/`、`../tools/` 里后端无关的客户端无需改源码即可链接。

## 目录结构

- `src/` —— 实现。`runtime.mm`(`aclrt*` 设备/流/事件/内存)、`meta.mm`(aclTensor / aclScalar / aclOpExecutor / aclTensorList)、
  `aclop.mm`(旧版单算子接口,转发到 aclnn)、`hccl.mm`(单 rank 集合通信 stub)、`ops/*.{mm,metal}`(各算子族)、
  `internal.h`(内部结构 + pipeline 缓存 + metallib 加载器)、低精度编解码 `ops/{subfp.h,hif8_table.inc}`。
- `Makefile` —— `.mm` 主机层用 `clang++ -ObjC++`,`.metal` 用 Metal Toolchain 编译器编进一个 `default.metallib`。`-I../include`(共享契约)+ `$ACL_INCLUDE`(acl/aclnn 元契约头)。
- `lib/` —— 构建产物(`libascendcl.dylib`、`libhccl.dylib`、`default.metallib`)。

## 构建

```bash
source ../env.sh                  # BACKEND=metal;用 DEVELOPER_DIR 做项目内 Xcode 作用域(不改全局 xcode-select)
make                              # → lib/libascendcl.dylib + lib/libhccl.dylib + lib/default.metallib
make -C ../tests BACKEND=../metal run
```

---

## 设计要点

### API 层拦截(从不触碰昇腾二进制)

应用调用 `aclrtMalloc` / `aclnnXxx` / `HcclXxx` 时,链接边界解析到本库的 Metal 实现;昇腾厂商算子二进制从不加载。
只复刻**每个算子的数学语义**,而非昇腾内部的算法。昇腾侧唯一的编译期依赖是 API **头文件**(契约)。

### 主机 = Objective-C++,kernel = MSL

主机/派发层是 Objective-C++(`.mm`),以驱动 MPS / MPSGraph / Accelerate(仅 Objective-C 的框架);计算 kernel 是手写 MSL(`.metal`),编进一个在 `aclInit` 时加载的 `default.metallib`。kernel 的 dtype/op 特化用 MSL 模板加按函数名缓存的 `MTLComputePipelineState`。

### aclnn 两段式执行器

aclnn 算子是两段式:`aclnnXxxGetWorkspaceSize(...)` 先在主机端构造 `aclOpExecutor`(记录张量、形状、轴、属性)并返回所需 workspace 大小;`aclnnXxx(workspace, ..., executor, stream)` 再执行并销毁执行器。本后端与 CUDA 后端实现同一契约,因此 `aclTensor` / `aclScalar` / `aclOpExecutor` 的算子 plan 原样可移植,只是 Execute 实现不同。

### 统一内存

用 `MTLStorageModeShared` 时,`MTLBuffer.contents` 指针主机可寻址且与设备端一致,于是 `aclTensor` 的设备指针主机可直接使用,`aclrtMemcpy`(主机↔设备)退化成 `memcpy`。主机侧容差比对无需额外拷贝,主机侧算子实现也能直接读写设备张量。

### 算子后端映射

| 算子族 | 实现 | 选型理由 |
|---|---|---|
| 逐元素 / 激活 / 归约 / softmax / norm / 比较 / cast / 排序 | 手写 **MSL** kernel | 语义简单但需广播 + 混合 dtype + 非连续视图;用 MSL 模板 + pipeline 缓存特化 |
| matmul / GEMM / BatchMatMul | **MPS** | 厂商调优 matmul;fp32/fp16 原生、**bf16 经无损 fp32 加宽**、**子字节 fp8/fp4/fp6/HiF8 经无损 fp16 加宽**、**W8A8 int8 经 MPS 原生 Int8 GEMM**。全程 fp32 累加;仅融合激活 / 二维 bias / 混合精度回退主机 GEMM |
| 卷积 / 池化 | 在 GPU 上:conv2d/3d 前向(im2col + MPS GEMM)、转置与反向卷积 + 池化 2D/3D/自适应/反向(MSL gather kernel)。ConvTbc / 可变形卷积走主机端 | 前向/反向/3D/转置/自适应全覆盖 |
| 注意力 | 批量 **MPS** QKᵀ/PV + MSL softmax(按尺寸门控)+ 每线程 MSL kernel / 主机回退 | 见下文"GPU 快路径" |
| 稠密线性代数 | **Accelerate / LAPACK**(统一内存上走 CPU) | inverse/solve/det/cholesky/qr/svd/eigh/lu;无设备拷贝故成本低 |
| FFT | **MPSGraph / vDSP** | RFFT/FFT |
| 低精度 fp8/fp4/fp6/HiF8/MX*(子字节) | 主机端按昇腾边界表解码;**matmul → 原生 MPS fp16**,其余算子走主机端 | Apple GPU 无原生子字节类型,但每个码字在 fp16 下精确,故 GEMM 走原生;解码留在主机端以保昇腾保真(见"低精度保真") |
| 通信 `Hccl*` | 单 rank 恒等 stub | 一台 Mac = 一个设备 |
| dtype | `half` 原生;`bfloat` 原生(M 系列 Metal 3.1+);子字节走位运算编解码 | 移植自 CUDA 后端的 `subfp`/`hif8` 逻辑 |

### 数值精度策略(对齐昇腾,非逐位一致)

- **一律 fp32 累加**:matmul 走 fp32 累加;手写归约(RMSNorm / softmax / LayerNorm)内部用 `float`/`double` 累加、写回时转换——与昇腾(及 PyTorch)的 bf16/fp16 语义一致。
- **bf16 matmul**:`MPSMatrixMultiplication` 无 bf16 类型,后端用 MSL cast kernel 把 bf16 无损加宽到 fp32(bf16 本就是截断的 fp32),走原生 MPS fp32 GEMM,再窄化回 bf16(fp32 累加与参考一致)。注意力 kernel 另有原生 bf16(`bfloat`)MSL 路径。
- **特殊函数精度到 ~1e-6**:手写 kernel 用 `-fno-fast-math` 编译,`erf` 用 Kahan 级数、`erfc` 用连分式、selu/elu/celu 基于 `expm1`、`gelu` 经 `erfc`、`lgamma` 相对精确——均移植自 CUDA functor。

### GPU 快路径

热点在 GPU 上跑,其余走统一内存上的主机端(正确,且因无拷贝而成本低):
- **逐元素 / softmax / norm**:手写 MSL kernel(合并访存,归约用每行一个 threadgroup)。
- **matmul / BatchMatMul**:MPS。
- **卷积 / 池化**:conv2d/3d 前向经 im2col(主机 gather)+ MPS fp32 GEMM;转置与反向卷积、池化(2D/3D、自适应、反向)经 MSL gather kernel(每输出一个线程、逆向索引映射、无原子)。保留主机回退。
- **注意力**:QKᵀ 和 PV 走批量 **MPS** GEMM、中间夹一个 MSL 带掩码 softmax(fp32 内部,故 fp16/bf16 加宽以保证 scores 精度;GQA/MQA 经复制头的 gather;causal/mask)。由**尺寸门控**派发;极小/decode/特异形状回退到每线程在线 softmax MSL kernel,再回退主机端。回退始终可用,正确性不依赖门控。

### 低精度保真

**W8A8 int8 matmul 原生运行**:走 `MPSMatrixMultiplication` 的 Int8 路径(int8·int8 → fp32 原始和,再乘 per-channel 反量化 scale)—— 只要每个输出的和落在 fp32 的 2²⁴ 整数范围内即精确(覆盖典型 W8A8;K 很大时舍入,在量化容差内),另有精确的主机回退。

**子字节格式** —— fp8(e4m3/e5m2)、fp4(e2m1/e1m2)、fp6(e2m3/e3m2)、HiFloat8、MXFP8、MXFP4 —— Apple GPU 无原生类型,故**主机端按昇腾边界表解码**。这是**昇腾保真所必需**:昇腾这些格式的编码/舍入边界与任何原生低精度格式都不同,喂给原生低精度单元(即便有)也会验证错误的数值目标。**对于 matmul,每个解码值在 fp16 下都精确(幅度 ≤ 32768,尾数 ≤ 3 位 ≤ fp16 的 10 位),故解码以 MSL kernel 在 GPU 上完成、GEMM 走原生 MPS fp16 路径**(fp32 累加);解码表由同一套边界表编解码器生成,故逐位一致。其余算子族在统一内存上走主机端。编解码 golden 值用纯位运算复现。

---

## 已实现内容

- **与 CUDA 后端算子完全对齐**——CUDA 后端导出的每个 `aclnn*` 算子这里都已实现(用两个动态库的导出符号双向核对验证)。
- **所有后端无关测试二进制通过**(`make -C ../tests BACKEND=../metal run`),外加各算子族的对齐测试——与 CUDA 后端同一根标准。
- **真实模型端到端**:Qwen3-0.6B(fp32)前向 + 贪心解码对齐 HuggingFace——prefill logits L2 **3.82e-6**,贪心 **8/8 token 一致**(与 CUDA 后端同一数字)。混合架构 **Qwen3-Next("Qwen3.5")** 前向(GatedDeltaNet 线性注意力 + 门控全注意力 + 逐层稀疏 MoE)也对齐 HuggingFace——逐阶段/逐层 ~1e-6,logits **4.34e-6**,贪心下一 token 一致(`../tools/qwen35/`)。
- **正确性**由仓库后端无关的容差测试,以及**与 CUDA 后端的对拍**(在一台 NVIDIA GB10 参考机上)保证,另有独立的 PyTorch oracle(`../tools/torch_golden.sh`、`../tools/torch_check.sh`)。

## 边界(单台 Mac 上超范围)

多设备集合通信(一台 Mac = 一个设备;HCCL 为单 rank stub)、真昇腾 golden / 逐位 NZ-fractal 保真、以及 FlashAttention-3 级 kernel,均超出本后端范围——见 [`TODO.zh-CN.md`](TODO.zh-CN.md)。这与项目定位一致:**功能 / 容差级**验证 + 跨架构后端探索,不追求与昇腾逐位一致,也不与 CUDA 比性能。

## 文档

- **[`TODO.zh-CN.md`](TODO.zh-CN.md)** —— 剩余 / 可优化的工作(开放路线图;已完成的工作不在此追踪)。
- **[`BENCHMARK.zh-CN.md`](BENCHMARK.zh-CN.md)** —— 基准测量方法、M 系列结果,以及与 CUDA 后端的跨后端对比。
- **[`../include/README.md`](../include/README.zh-CN.md)** —— API 契约头的内容与"谁实现、谁使用"。
