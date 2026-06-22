# 性能基准 —— Metal 后端

> English: [BENCHMARK.md](BENCHMARK.md)

cann-on-gpu **Apple Metal** 后端的测量方法与结果。

> 数据采集于 **Apple M4 Max**(40 核 GPU,128GB 统一内存 LPDDR5X,Metal 4)。跨后端对照列为 **NVIDIA GB10**(`sm_121`,UMA LPDDR5X,CUDA 13),用已验证的 CUDA 后端跑*同一套*契约。
> 二者是**不同芯片、不同软件栈**——该对比展示同一 ACL 工作负载在两个后端上的表现,**不是**优化 A/B。数字不跨机外推;换机请重测。

## 价值主张

与 CUDA 后端一样,Metal 后端的目的是拿到可对拍的昇腾算子结果,**比官方 CPU 无卡仿真快几个数量级**(指令级、约 40–85 秒/算子、O(work) 增长)。在 Apple 芯片上,统一内存设计还彻底消除了主机/设备拷贝,所以走主机端回退的算子族不付传输成本。

## 测量方法

- 工具:`tools/bench.cpp`(纯 ACL 客户端,只链 `libascendcl.dylib`,与昇腾应用的链接方式一致),针对 `metal/lib` 构建。
- 计时:`std::chrono` 墙钟/次,**含 GetWorkspaceSize 执行器构建**(真实的每次调用成本);warmup 后多次迭代取平均;workspace 预分配复用。
- 派生指标:GEMM / 注意力用 GFLOP/s(注意力按 QKᵀ+PV ≈ 4·B·N·S²·D);访存型算子用有效 GB/s。
- **无基于开关的优化 A/B。** 与 CUDA 后端(暴露 `ELTWISE_NO_VEC` / `MXFP8_NO_HW` / … 环境开关)不同,Metal 后端没有优化开关,因此本文给出绝对吞吐 + 跨后端对比,而非优化-vs-朴素扫描。
- 复现:

```bash
source ../env.sh
cd ../tools && mkdir -p bin
clang++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include bench.cpp \
    -L../metal/lib -lascendcl -Wl,-rpath,../metal/lib -o bin/bench
./bin/bench
```

> 在 Apple 芯片上,fp16 / bf16 matmul、W8A8 int8 matmul、子字节 fp8/fp4/fp6/HiF8 matmul、注意力、卷积与池化(前向 / 反向 / 转置 / 3D / 自适应)、逐元素、softmax、norm 全在 GPU 上跑(MPS 或 MSL)。bf16 经无损加宽走 MPS 的 fp32 路径,子字节经无损 decode→fp16 加宽走 MPS 的 fp16 路径,int8 走 MPS 原生 Int8 GEMM;注意力把 QKᵀ/PV 走批量 MPS GEMM;conv2d/3d 前向走 im2col + MPS GEMM,卷积/池化的反向与转置走 MSL gather kernel。PagedAttention 的 scalar-vs-warp 是 CUDA 专有 A/B、在此走主机端,已从下表略去。

## 结果 —— Apple M4 Max(Metal)

| 算子(配置) | 结果 | 备注 |
|---|---|---|
| 设备 malloc+free / 对(缓存) | 0.0034 ms(3.36 µs) | 基于 `MTLBuffer` 复用的缓存分配器 |
| GEMM fp16 1024³ | 0.522 ms / **4115 GFLOP/s** | MPS matmul |
| GEMM fp16 4096³ | 9.14 ms / **15042 GFLOP/s** | MPS matmul(~15 TFLOP/s) |
| GEMM bf16 1024³ | 0.55 ms / 3939 GFLOP/s | GPU bf16↔fp32 加宽 kernel + MPS fp32(此尺寸受开销主导,≈ fp16) |
| GEMM bf16 4096³ | 14.6 ms / **9444 GFLOP/s** | GPU bf16↔fp32 加宽 kernel + MPS fp32 GEMM |
| GEMM fp8 e4m3 1024³ | 0.74 ms / 2886 GFLOP/s | GPU 子字节 decode→fp16 kernel + MPS fp16(受开销主导,≈ fp16) |
| GEMM fp8 e4m3 4096³ | 13.0 ms / **10552 GFLOP/s** | GPU decode→fp16 kernel + MPS fp16 GEMM(fp4/fp6/HiF8 同路径) |
| 注意力 S=128(B4 N8 D64,fp16) | 0.88 ms / 153 GFLOP/s | 批量 MPS QKᵀ/PV + MSL softmax(fp32 内部);回退为每线程在线 softmax kernel |
| └ S=512 / 1024 / 2048 | 1.62 / 5.77 / 21.7 ms | 1324 / 1488 / 1584 GFLOP/s |
| Conv2d N8 C64 64² Co64 k3(fp32) | 21.7 ms / 111 GFLOP/s | im2col(主机 gather)+ MPS fp32 GEMM(conv3d 前向走同一路径) |
| 卷积反向/转置、池化(2D/3D、自适应、反向) | MSL gather kernel,每输出一个线程 | 输出端逆映射;无原子操作 |
| Add 4M(fp32) | 0.246 ms / **205 GB/s** | MSL 逐元素 |
| RMSNorm 4096×1024(fp32) | 0.453 ms / 74 GB/s | MSL,每行一个 threadgroup |
| Softmax 末维 4096×1024(fp32) | 0.339 ms / 99 GB/s | MSL 合并访存 |

(`ReduceSum` 全归约略去:该算子通过容差测试,但基准里 100 次流水线化全归约会卡住本后端的 harness。)

## 跨后端对比(同一工作负载、同一契约)

同一份 `tools/bench.cpp` 的用例,分别在 Metal 后端(M4 Max)与 CUDA 后端(NVIDIA GB10)上跑。这是**后端 + 硬件**对比,不是优化测量——二者是不同芯片、计算与访存上限不同。

| 工作负载 | M4 Max — Metal | GB10 — CUDA | GB10 / M4 |
|---|---|---|---|
| malloc+free / 对(缓存) | 3.36 µs | 0.073 µs | 46× |
| GEMM fp16 1024³ | 4115 GFLOP/s | 69442 GFLOP/s | 16.9× |
| GEMM fp16 4096³ | 15042 GFLOP/s | 81613 GFLOP/s | 5.4× |
| 注意力 S=2048(B4 N8 D64) | 1584 GFLOP/s | 1656 GFLOP/s | 1.05× |
| Add 4M(fp32) | 205 GB/s | 235 GB/s | 1.15× |
| RMSNorm 4096×1024(fp32) | 74 GB/s | 241 GB/s | 3.3× |
| Softmax 4096×1024(fp32) | 99 GB/s | 232 GB/s | 2.3× |

**结论。**
- **流式逐元素接近**(Add 1.15×):二者都是统一内存 LPDDR5X 部件,纯访存型算子在两边落在同一档。
- **计算密集的 GEMM 偏向 GB10**(4096³ 5.4×):NVIDIA GPU 张量吞吐上限更高;MPS 仍在 Apple GPU 上给出了正确的厂商 matmul(fp16 ~15 TFLOP/s)。
- **注意力基本持平**(1.05×):两边都把矩阵乘交给厂商 GEMM(Metal 用批量 MPS QKᵀ/PV,CUDA 用 cuBLASLt)、中间夹 softmax,故注意力在两边落在同一档。
- **RMSNorm/Softmax**(2.3–3.3×):尽管访存带宽相近,Metal 的 MSL 归约 kernel 不如 CUDA 的合并/warp 归约版本调优充分。

## 潜力在哪

剩余的 GPU 加速潜力,与 [`TODO.zh-CN.md`](TODO.zh-CN.md) 的路线一致:

- **不 materialize scores 的长序列注意力。** 注意力把 QKᵀ/PV 走批量 MPS GEMM,会 materialize [B·H, Sq, Skv] 的 scores,因而受内存上限约束,超过即回退到每线程在线 softmax kernel。一个 `simdgroup_matrix` 的 FlashAttention 式 kernel 可保住 matmul 硬件速度且不 materialize S²。
- **少数刻意留在主机端的算子族。** 稠密线性代数(走 Accelerate/LAPACK、统一内存上)与个别 niche 卷积(ConvTbc、可变形卷积)仍在主机端——都不是瓶颈。(conv2d 前向走 im2col + MPS GEMM,故当前瓶颈是主机 im2col gather 而非 GEMM。)

> Metal 后端的目标是与 CUDA 后端达到**功能 / 容差级**对齐(已达到——算子完全对齐、所有测试全绿、Qwen3-0.6B 端到端 3.82e-6),而非性能竞赛。这里的数字量化的是 GPU 加速的潜力所在,而非正确性上的不足。
