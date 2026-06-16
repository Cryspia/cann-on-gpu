# cuda/ —— CANN / AscendCL API 的 NVIDIA CUDA 后端

> English: [README.md](README.md)

本目录把顶层 [`../include/`](../include/) 声明的 CANN/AscendCL/aclnn/HCCL API 契约，用 CUDA 生态在 NVIDIA GPU 上重新实现。
输出 `lib/libascendcl.so` + `lib/libhccl.so`；应用像链昇腾原库一样链它们，只见 API、不见 CUDA。

## 目录

- `src/` —— 实现。`runtime.cu`（`aclrt*` 设备/流/事件/内存）、`meta.cu`（aclTensor / aclOpExecutor / aclTensorList）、
  `aclop.cu`（旧式单算子接口路由到 aclnn）、`hccl.cu`（HCCL→NCCL）、`ops/*.cu`（各算子族）、`internal.h`（内部结构）、
  低精度编解码 `ops/{hif8.cuh,hif8_table.inc,subfp.cuh}`。
- `Makefile` —— `nvcc -arch=$(GPU_ARCH)`，`-I../include`（共享契约）+ `$ACL_INCLUDE`（acl/aclnn 元契约头）+ cuDNN/NCCL。`cudart` 静态链入。
- `lib/` —— 构建产物。
- `tools/` —— CUDA 专有工具：`nccl_allreduce.cu`（裸 NCCL）、`ptx_module_demo`（CUDA Driver/PTX/nvrtc）及对应脚本。

## 构建

```bash
source ../env.sh
make                       # → lib/libascendcl.so + lib/libhccl.so
make GPU_ARCH=sm_90        # 换 NVIDIA 架构重编（见末节"跨架构可移植"）
```

---

## 设计要点

### API 层拦截（不碰昇腾二进制）

应用调 `aclrtMalloc` / `aclnnXxx` / `HcclXxx` 时，链接边界落到本库的 CUDA 实现，昇腾原厂算子二进制从不被加载。
只需复刻**每个算子的数学语义**，不需复刻昇腾**怎么算**。运行期禁止链接昇腾任何 runtime（会符号冲突）；
昇腾侧的唯一编译期依赖是 API **头**（契约）。

### aclnn 两段式执行器

aclnn 算子是两段式：`aclnnXxxGetWorkspaceSize(...)` 先在 host 构造一个 `aclOpExecutor`（记录张量、形状、轴、属性等，
**不分配 device 内存**），返回所需 workspace 大小；`aclnnXxx(workspace, ..., executor, stream)` 再用调用方预分配的 workspace 执行并销毁执行器。
本后端如实实现这一契约——这也使整条算子链可被 CUDA Graph 捕获（GetWorkspaceSize 是 host-only、捕获安全；执行段复用预分配 workspace，捕获期不 `cudaMalloc`）。

### 算子后端映射与选型理由

| 算子族 | 实现 | 选型理由 |
|---|---|---|
| 逐元素 / 激活 / 归约 / 形变 / 比较 / 排序 | 手写 CUDA kernel | 语义简单、要支持广播 + 混合 dtype + 非连续视图（按真实 stride/offset 寻址），手写最灵活；归约用 block/warp 归约 |
| matmul / GEMM | **cuBLASLt** | 厂商高度调优、近峰值；行主序经"列主序映射"对接；fp32 用 fp32 累加，fp16/bf16 用 fp32 累加（与昇腾 Cube 一致） |
| 卷积 / 池化 / BatchNorm | **cuDNN** | 卷积算法选择与张量核利用交给 cuDNN；前向/反向/3D/转置/池化齐全 |
| 注意力 | 手写朴素 + 手写 flash + cuBLASLt 批量 | 见下"注意力" |
| 通信 HCCL | **NCCL** | 集合通信语义 1:1；难点在控制面（ranktable/uniqueId 翻译） |
| 低精度 fp8/fp4/fp6/HiF8/MX* | 解码到 fp16（或 fp8）再算 | 见下"低精度的保真做法" |

### 数值精度策略（与昇腾对齐，非 bit-exact）

- **累加恒用 fp32**：GEMM 走 `CUBLAS_COMPUTE_32F`，手写归约（RMSNorm/softmax/LayerNorm）内部 `float`/`double` 累加，写回时 cast。
  这与昇腾（以及 PyTorch）的 bf16/fp16 语义一致——**张量按低精度存储、算子内部 fp32 累加**。
- **bf16 全算子覆盖**：matmul/MatmulBias/BatchMatMul/GroupedMatmul/卷积全家/LayerNorm/RmsNorm/RoPE/FlashAttention/PagedAttention 等均支持 bf16。
  真实模型跑纯 bf16 时，输出与 HuggingFace 的 bf16 输出逐 token 一致（验证 bf16 语义忠实）。
- **fp16 乘法对拍**用 atol+rtol（次正规区相对误差发散）；**GEMM 对拍**用归一化误差 `max|err|/max|ref|`（避免近零参考元素假性放大相对误差）。

### 低精度的保真做法

fp8(e4m3/e5m2)、fp4(e2m1/e1m2)、fp6(e2m3/e3m2)、HiFloat8、MXFP8、MXFP4：默认走**软件解码到 fp16/fp8 再做 GEMM**的功能路径。
这不是性能妥协，而是**对昇腾保真的必要做法**——昇腾这些格式的编码/舍入边界与 NVIDIA 原生低精度不同，
必须先按昇腾语义解码（边界表）再算；直接喂 NVIDIA 原生 fp4/fp8 tensor core 会算成 NVIDIA 的数值，验证的是错误对象。
低精度 codec 的 golden 由 CANN 参考编解码离线生成验证（纯位运算）。
Blackwell 的 MXFP8 原生微缩放路径（`aclnnMatmulMxFp8Hw`，VEC32_UE8M0 + 128×4 super-tile swizzle）已实现，
并在非 Blackwell 架构上运行时自动回退功能路径。
另有**可选的原生 NVFP4 fp4 路径**（`NVFP4_HW=1`）：在线把昇腾的 E8M0/block-32 缩放转成 E4M3/block-16 以打到 Blackwell fp4 张量核（`VEC16_UE4M3`），实测**达功能路径的 3.0×**（见 `../BENCHMARK.md`）。当每块缩放指数落在 E4M3 范围 [2⁻⁶, 2⁸] 内时精确、超出则钳位,故为可选的高速/降保真档;功能解码→fp16 路径仍为默认与保真基准。

---

## 性能优化手段

所有优化保持容差对拍全绿，并保留环境变量开关可回退做对照。性能数字见 [`../BENCHMARK.md`](../BENCHMARK.zh-CN.md)。

- **真 flash attention（在线 softmax）**：warp-per-row，每 lane 持 D/32 个头维分量，点积用 `__shfl` 归约（无 `__syncthreads`），
  running max/sum/acc 全在寄存器 → **免物化 S/P**（workspace 归零，大 S 不再 O(S²) 显存爆炸）。覆盖 GQA/MQA/causal/mask/fp16/bf16/fp4。
  fp16+标准 MHA 时 QKᵀ 上 WMMA tensor-core（fp32 累加）；PV 保持 fp32 标量（fp16 量化 P 做 PV 误差过大）。
- **缓存显存分配器**：`aclrtMalloc/Free` 走分桶复用，单次 malloc+free 从直通 cudaMalloc 的数百 µs 降到亚 µs，是推理后端地基。
- **PagedAttention warp-per-row**：寄存器 `acc[D/32]` 替代本地内存 `acc[256]` + 合并 KV 访存，达 ~bandwidth-bound。
- **权重量化小 M 融合**：W8A16/W4A16 在小 M（解码/小批）直读 int8/int4 权重融合 GEMM，不物化 fp16，省 4×/8× 权重带宽；大 M（prefill）仍解量化 + cuBLASLt 张量核。
- **访存合并归一化**：RMSNorm/Softmax/GroupNorm 以及融合 add-norm 族（`AddRmsNorm`/`AddLayerNorm`/`AddRmsNormCast`/`DeepNorm`,被 MoE/融合 matmul/mc2 调用）均用"一 block 一行 + warp 归约"做合并访存。
- **Foreach 多张量融合**：算术类 `aclnnForeach*` 族把整个张量列表融成单次 grid-stride kernel（每张量元信息放调用方 workspace,`grid.y` = 张量下标,仿 PyTorch `MultiTensorApply`），取代逐张量一次启动 —— 对"大量小张量"的优化器状态列表 **43–70×**（`FOREACH_NO_FUSE=1` 回退）。
- **W8A8 原生 int8 GEMM**：cuBLASLt `CUDA_R_8I` + int32 累加 + per-channel scale epilogue；失败回退 fp16。
- **MoE 分组 GEMM**：变长分组跨多流重叠，或原生 grouped GEMM（开关）。
- **CUDA Graph**：单 token decode 步捕获成图、整循环重放，省逐 kernel launch 开销（decode 是 launch-bound）。
- **128-bit 向量化访存**：elementwise 16B 向量化快路径 + 标量回退（`ELTWISE_NO_VEC=1` 回退）。同卡收益分区间 —— L2 常驻时 1.5–3×,纯 DRAM 带宽受限后无差别（见 `../BENCHMARK.md`）。

---

## 跨 NVIDIA 架构可移植

源码是通用 CUDA（无 `__CUDA_ARCH__` 分支、无按设备属性 gating），架构绑定**仅来自构建标志** `-arch=$(GPU_ARCH)`。
换架构只改 `GPU_ARCH` 重编，源码不动：

| 目标 | `GPU_ARCH` | 低精度 |
|---|---|---|
| GB10（默认） | `sm_121` | 全部，含 MXFP8 原生微缩放 |
| RTX 50（GB202） | `sm_120` | 全部，含 MXFP8 原生 |
| B200 / GB200 | `sm_100` | 全部，含 MXFP8 原生 |
| Hopper H100/H200 | `sm_90` | fp8/fp4/fp6/MX* 均可（MX 走功能路径） |
| Ada / RTX 40 | `sm_89` | 同上 |

- **fp4/fp6 不需要硬件 fp4**：查表解码到 fp16 再算，任何有 fp16 的架构都能跑（且是昇腾保真的正确做法）。
- 动态依赖 cuBLASLt/cuDNN/NCCL 需目标架构对应版本。

## 边界（A 路径结构性不覆盖）

非计算语义、或绑死昇腾专有格式/硬件的 API 不在本后端范围：GE 图引擎 / Graph 模式编译、`aclmdl`（`.om` 离线模型）、
AOL 预编译算子黑盒、ATB、DVPP/HIXL/SiP 专用硬件单元、`aclprof*`/`acldump*` 调试采集、以及 bit-exact 数值一致。
其中"加载模型→执行"这条控制流提供了**等价能力**：`../tools/cann_graph.h` 一个极简算子图 IR + 序列化 + 顺序执行器（非昇腾 `.om` 的等价容器）。
