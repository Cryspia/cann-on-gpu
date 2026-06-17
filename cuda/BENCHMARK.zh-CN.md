# 性能基准

> English: [BENCHMARK.md](BENCHMARK.md)

cann-on-gpu CUDA 后端的性能测量方法与结果。

> 数据采集于 **NVIDIA RTX PRO 6000 Blackwell Workstation Edition**（`sm_120`，96 GB GDDR7 ≈1.8 TB/s，独立显卡），CUDA 13.3，cuBLASLt / cuDNN 后端。
> 这些是**独显 Blackwell 的数字，与此前 GB10（`sm_121`，UMA LPDDR5X ~273 GB/s）基线不可直接比较**——高带宽 GDDR7 卡把 GB10 的 UMA 所掩盖的访存与 tensor-core 收益暴露了出来。数字仍不可跨卡外推，换卡须重测。见 [`TODO.md`](TODO.zh-CN.md)。

## 价值主张

GPU 后端的意义在于：**比官方 CPU 无卡仿真快几个数量级地拿到可对拍的结果**。

- 单算子验证：无卡仿真 **40–85 s/算子**（构建 + 指令级仿真）vs GPU 后端 **µs–ms**。
- 整网前向：单层 transformer（~20 算子），GPU **0.21 ms/次**；无卡仿真连单个算子都要数十秒。
- 量级差 **~10⁵–10⁶×**。且指令级仿真复杂度 O(work)，大张量不可行；GPU 随硬件吞吐扩展。

| 算子 | 无卡仿真 wall-clock | 仿真器件时间 |
|---|---|---|
| Transpose（259 条指令） | 39.6 s | 1.7 µs |
| ReduceSum（363） | 45.5 s | 2.2 µs |
| Add（30899） | 84.6 s | 6.7 µs |

turnaround 由"构建 + 指令级仿真"主导并随工作量上升；器件时间本身仅 µs 级。

## 测量方法

- 工具：`tools/bench.cpp`（纯 ACL 客户端，仅链 `libascendcl.so`，模拟昇腾应用的链接方式）。
- 计时：`std::chrono` wall-clock，**含 GetWorkspaceSize 执行器构建**（即真实每次调用成本）；warmup 5–10 次后多迭代取平均；workspace 缓冲预分配复用。
- 派生指标：GEMM / Attention 报 GFLOP/s（attention 按 QKᵀ+PV ≈ 4·B·N·S²·D）；访存型报有效带宽 GB/s。
- 复现：

```bash
source env.sh
cd tools
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include bench.cpp \
    -L../cuda/lib -lascendcl -Wl,-rpath,../cuda/lib -o bench
./bench
```

- decode-loop CUDA Graph 延迟：`tools/decode_graph/build_decode_graph.sh`。

## 结果

下表在 RTX PRO 6000 Blackwell（`sm_120`）上实测。"加速"列在本次运行直接测了优化路径与朴素/旧路径两者（通过 env 开关）时填写;各项优化的同卡 A/B（128-bit 向量化、融合、MXFP8、NVFP4、开关扫描、vs 原生 CUDA）见下方 "Optimization A/B" 各节。

| 算子（配置） | 结果 | 加速 | 说明 |
|---|---|---|---|
| GEMM fp16 4096³ | 0.315 ms / **436 TFLOP/s** | — | cuBLASLt 近峰值（1024³：200 TFLOP/s） |
| GEMM bf16 4096³ | 0.315 ms / **436 TFLOP/s** | — | cuBLASLt 近峰值 |
| Attention 默认路径（flash）S=2048（B4 N8 D64） | 5.29 ms / 6500 GFLOP/s | — | 在线 softmax flash（免物化 S/P）+ WMMA QKᵀ tensor-core；workspace 536 MB→0 |
| └ S=1024 / 512 / 128 | 1.59 / 0.37 / 0.064 ms | — | 5403 / 5841 / 2089 GFLOP/s |
| Attention 性能路径 S=2048 | 1.55 ms / **22102 GFLOP/s** | — | cuBLASLt 批量 tensor-core，标准 MHA 吞吐最高（物化 S/P） |
| └ S=1024 / 512 / 128 | 0.31 / 0.084 / 0.019 ms | — | 27654 / 25492 / 7088 GFLOP/s |
| PagedAttention 解码 B64 N8 L4096 D64 | 3.28 ms / 164 GFLOP/s | **15.3×** | warp-per-row + 寄存器 online softmax，对比标量旧路径（50.2 ms） |
| └ L1024 | 0.67 ms / 201 GFLOP/s | **18.4×** | 对比标量旧路径（12.3 ms） |
| RMSNorm 4096×1024（fp32） | 0.0078 ms / 4286 GB/s † | — | 一行一 block 合并访存 + warp 归约 |
| Softmax 末维 4096×1024（fp32） | 0.0081 ms / 4153 GB/s † | — | 合并访存快路径 |
| Add 4M（fp32） | 0.0062 ms / 8148 GB/s † | — | 向量化逐元素 |
| ReduceSum 4M（fp32） | 0.0159 ms / 1057 GB/s | — | 树形归约 |
| tiny LLM 前向（单层，~20 算子） | **0.209 ms/次** | — | 端到端组合；logits 误差 2.2e-7 |
| Foreach addcmul，256 张量 × 4096 | 0.0162 ms | **43.5×** | 融合多张量 kernel vs 逐张量循环（0.705 ms，`FOREACH_NO_FUSE=1`） |
| └ 512 张量 × 256（启动受限） | 0.0203 ms | **70.1×** | vs 逐张量循环（1.42 ms） |
| 显存 malloc+free / 对 | 0.040 µs | **~2958×** | 缓存分配器 vs 直通（118.3 µs，`ACL_NO_CACHE_ALLOC=1`） |
| decode-loop（单 token，2 层 GQA） | 0.164 → 0.117 ms/token | **1.40×** | 整循环 CUDA Graph 重放，省逐 kernel launch 开销 |

† 这些小工作集（≤48 MB）在 Blackwell 的大 L2 上**常驻 L2**，所报 GB/s 反映的是 L2 带宽，而非 GDDR7 DRAM 上限。实测 DRAM-bound 流式（Add，≥256M 元素工作集）上限约 **~1.75 TB/s**（见下方 128-bit A/B）。

### cann-on-gpu vs 原生 CUDA（同进程，`aclnnMatmul` vs `cublasGemmEx`）

ACL 抽象把 GEMM 分派到 cuBLAS(Lt) 张量核,所以与直接 cuBLAS 调用的唯一差别就是每次调用的执行器构建 + 分派。同一二进制内实测 fp16 GEMM:

| GEMM fp16 | cann-on-gpu | 原生 cuBLAS | cann/原生 |
|---|---|---|---|
| 512³ | 55.6 TFLOP/s | 50.9 TFLOP/s | 1.09× |
| 2048³ | 403 TFLOP/s | 406 TFLOP/s | 0.99× |
| 4096³ | 435 TFLOP/s | 435 TFLOP/s | 1.00× |
| 8192³ | 514 TFLOP/s | 528 TFLOP/s | 0.97× |

**结论**:cann-on-gpu 与原生 CUDA **持平**(0–3% 以内,512³ 处更快是其 cuBLASLt 启发式恰好选了更优算法)。"算力密集区差距"实际为**零** —— 后端跑的是同一套张量核内核,ACL 执行器/缓存分配器开销即便在小尺寸也可忽略。(此对比只在 compute-bound 卡上有意义;带宽受限卡上主导的是 GEMM 内核选择而非包装层。)

## 关于本卡（独显 Blackwell，sm_120）的说明

这些数字仅对本卡有效。**不要拿绝对吞吐与此前的 GB10 基线对比**——GB10 与 RTX PRO 6000 是不同硅片、算力与带宽上限本就不同，跨卡差值衡量的是硬件而非优化。上表里每个「加速」都是**同卡 A/B**：优化路径 vs 关掉开关后的朴素/旧路径（`FOREACH_NO_FUSE` / `PAGED_SCALAR` / `ACL_NO_CACHE_ALLOC` 等），两者都在本卡实测。

本卡改变的是「哪些优化值得测」（均须在本卡用开关做同卡前/后对比来量化）：

- **访存优化在本卡值得测**：这是独显高带宽 GDDR7 卡，所以 128-bit 向量化访存、降访存流量的融合能测出真实的同卡前/后差值（在低带宽 UMA 卡上测不出来）—— 见下方 "Optimization A/B" 各节。
- **本卡有原生低精度 tensor-core**（fp4/fp8/MXFP8），所以原生 vs 功能路径是有意义的同卡对比（`MXFP8_NO_HW` 等）。
- **launch-bound 优化**（CUDA Graph 重放、缓存分配器、foreach 融合）的同卡收益与带宽无关——见上表。
- 所有数字换卡须重测；绝不跨硅片搬运数字或比例。

## 优化 A/B（同卡，开关对照）

### 128-bit 向量化访存（`ELTWISE_NO_VEC=1` 强制走标量路径）

16 字节（`LDG.128`）向量化逐元素路径 vs 单元素/线程标量路径，均在本卡，流式 `Add`（触及 3 个数组）按尺寸扫描：

| 工作集（Add） | 区间 | fp32 向量 | fp32 标量 | fp16 向量 | fp16 标量 |
|---|---|---|---|---|---|
| 48 MB（4M 元素） | L2 常驻 | **7899 GB/s** | 5284 GB/s（1.50×） | **6993 GB/s** | 2649 GB/s（2.64×） |
| 192 MB / 96 MB | L2 边缘 | ~2189 | ~2138（≈） | **8774 GB/s** | 2871 GB/s（3.06×） |
| ≥768 MB（≥64M 元素） | DRAM-bound | 1745 | 1750（≈） | 1741 | 1789（≈） |

**结论**:收益真实但**分区间** —— 在 L2 常驻区间,高 L2 带宽让标量路径变成发射受限,128-bit 加载带来 **1.5×(fp32)~3×(fp16)**;一旦工作集超出 L2、算子纯 DRAM-bound,两条路径都把 GDDR7 打满到 **~1.75 TB/s**,向量化无差别。它从不劣化,故默认开启。(GB10 的低带宽 UMA 没有「L2 常驻高带宽」这个区间可利用,所以该优化在那里看不出收益。)

### 算子融合 —— 降访存流量（`AddRmsNorm` vs `Add`+`RmsNorm`）

融合的「残差加 + RMSNorm」（1 个 kernel，中间结果 `x+res` 留在片上 / L2 里）vs 非融合两 kernel（`Add` 把和写回 DRAM，`RmsNorm` 再读回），fp16：

| rows × D（fp16） | 区间 | 融合 | 非融合（add+rms） | 融合加速 |
|---|---|---|---|---|
| 4096 × 4096 | L2 常驻 | 0.0360 ms | 0.0328 ms | 0.91× |
| 4096 × 8192 | L2 边缘 | 0.1592 ms | 0.1539 ms | 0.97× |
| 8192 × 8192 | DRAM-bound | 0.3205 ms | 0.3820 ms | **1.19×** |

**结论**:融合收益**随工作集增大而增大** —— 当中间结果会往返 DRAM（大 rows×D）时,融合省掉这部分流量(+ 一次启动)得到约 **1.2×**;当工作集 L2 常驻、没有 DRAM 往返可省时,融合 ≈ 非融合。*(融合 add-norm 族 —— `AddRmsNorm`/`AddLayerNorm`/`AddRmsNormCast`/`DeepNorm`,被 MoE / 融合 matmul / mc2 内部调用 —— 用一行一 block + 合并访存 + warp 归约,与 `k_rms_fast` 一致:如 DeepNorm 0.026 ms、AddRmsNormCast 0.094 ms @ 4096² fp16,约为逐行单线程基线的 40–140×。)*

### 融合 QKV 投影（一个 `[K,3N]` GEMM vs 三个 `[K,N]` GEMM）

把 Q/K/V 投影权重拼成一个 matmul(总 FLOPs 相同;输入激活只读一次、一次启动、一个更大的 GEMM)。fp16,K=N=4096:

| M(token 数) | 区间 | 3× 独立 | 融合 [K,3N] | 加速 |
|---|---|---|---|---|
| 8 | decode | 0.0260 ms | 0.0216 ms | 1.21× |
| 64 | 小批 decode | 0.0478 ms | 0.0291 ms | **1.64×** |
| 512 | — | 0.1427 ms | 0.1356 ms | 1.05× |
| 4096 | prefill | 0.8747 ms | 0.8707 ms | 1.00× |

**结论**:融合在 **decode / 小 M 区间**收益最大(最高 **1.64×**),此时三个独立投影是启动 + 权重读取受限;prefill 时是算力受限(FLOPs 相同)收益消失。这不需要后端内核 —— 是模型层变换(把三个投影权重打包、发一个 `aclnnMatmul`),故指导是:**decode 密集型推理应拼接 QKV 权重**。

### 原生 MXFP8 微缩放 GEMM（`MXFP8_NO_HW=1` 强制走功能路径）

原生 Blackwell 微缩放 tensor-core（cuBLASLt `VEC32_UE8M0` 块缩放）vs 功能路径（块反量化 fp8→fp16 再做 fp16 GEMM）：

| MXFP8 GEMM | 原生（HW） | 功能路径 | 原生加速 |
|---|---|---|---|
| 2048³ | 236 TFLOP/s | 288 TFLOP/s | 0.82× |
| 4096³ | 430 TFLOP/s | 357 TFLOP/s | 1.21× |
| 8192³ | **601 TFLOP/s** | 437 TFLOP/s | **1.38×** |

**结论**:原生路径的 fp8 tensor-core 在 8192³ 达 **601 TFLOP/s** —— 是功能路径的 **1.38×**,而功能路径上限约 437(≈ fp16 GEMM 峰值,因为它先反量化到 fp16 再算)。原生路径有固定前处理(转置 B 操作数 + 把 E8M0 缩放重排成 128×4 超块),所以小尺寸(2048³)反而更慢,约在 4096³ 交叉;在本 Blackwell 卡上对大矩阵应作默认,非 Blackwell 架构自动回退功能路径。

### 开关类优化扫描（同卡）

| 优化 | 开关 | 配置 | 关 | 开 | 加速 |
|---|---|---|---|---|---|
| TF32 快路径 GEMM | `CANN_FAST_TF32=1` | fp32 GEMM 8192³ | 94.3 TFLOP/s | 245.0 TFLOP/s | **2.60×** |
| Flash WMMA QKᵀ | `FLASH_NO_WMMA`（关闭它） | fp16 attn S=2048 | 4395 GFLOP/s | 6645 GFLOP/s | **1.51×** |
| Flash 快 PV | `FLASH_FAST_PV=1` | fp16 attn S=2048 | 6645（WMMA） | 7555 GFLOP/s | **1.14×**（对 no-WMMA 1.72×） |
| Grouped GEMM 原生 | `CANN_GMM_NATIVE=1` | E8 每专家=128，启动受限 | 187 TFLOP/s | 200 TFLOP/s | 1.07× |
| └ 同上 | | E8 每专家=512，计算受限 | 389 TFLOP/s | 301 TFLOP/s | 0.77×（多流胜） |
| Conv 算法 autotune | `CANN_CONV_AUTOTUNE=1` | 3×3 N32 C256 56² K256 | 325 TFLOP/s | 323 TFLOP/s | ≈1.00× |

**结论**:TF32 给 fp32 GEMM 干净的 **2.6×**(降精度,opt-in)。WMMA tensor-core QKᵀ 在 S≥512 给 flash attention **1.5×**(S=128 太小,WMMA 前处理摊不开会更慢);`FLASH_FAST_PV` 再加 1.14×。Grouped-GEMM 原生只在启动受限(大量小专家)有益,专家足够大时默认多流路径重叠更好,故 opt-in。Conv autotune 对标准卷积**中性**,因为 cuDNN v7 启发式已选到最优算法(只对异形 shape 有用)。大 L 排序自动走 bitonic(按尺寸选择,非开关)。

### 原生低精度 GEMM 路径(fp4/fp6/MXFP4)—— 覆盖与一个注意点

原生张量核低精度 GEMM 路径均已存在并通过容差校验:MXFP8(`aclnnMatmulMxFp8Hw`,见上)、MXFP4(`aclnnMatmulMxFp4Hw`,`CUDA_R_4F_E2M1` + `VEC32_UE8M0`)、以及普通 fp4/fp6 matmul(无损解码到 fp8-e4m3 走原生 fp8 GEMM,`SUBFP_NO_FP8` 回退 fp16)。同卡 MXFP4 原生 vs 功能(解包→fp16):

| MXFP4 GEMM | 原生(E8M0/blk-32) | 功能(→fp16) | 原生加速 |
|---|---|---|---|
| 4096³ | 322 TFLOP/s | 372 TFLOP/s | 0.87× |
| 8192³ | 398 TFLOP/s | 438 TFLOP/s | 0.91× |

**fp6(E2M3/E3M2):** fp6 普通 matmul 无损解码到 fp8-e4m3、走原生 fp8 GEMM,达 **554 TFLOP/s @ 8192³**(fp6 与 fp8 共用 Blackwell 张量核数据通路,已是 fp8 上限的约 92%)。

**注意**:`aclnnMatmulMxFp4Hw` 请求的是 `VEC32_UE8M0`,而**本 cuBLASLt 对 fp4 不接受该模式**(只接受 NVFP4 `VEC16_UE4M3`),于是静默回退到功能路径 —— 这就是上表"原生" MXFP4 数字等于功能路径的原因。真正的原生 fp4 速率需要下面的 NVFP4 路径。

### NVFP4 原生 fp4 路径 —— 速度与保真(`NVFP4_HW=1`)

新建了原生 NVFP4 GEMM(`CUDA_R_4F_E2M1` + `VEC16_UE4M3` block-16),在线把 Ascend 的 E8M0/block-32 缩放转成 E4M3/block-16(每个 block-32 → 两个同 2^exp 的 block-16)。这才是真正打到 Blackwell fp4 张量核的路径:

| GEMM @ 8192³ | TFLOP/s | vs 功能 fp16 | vs 原生 MXFP8 |
|---|---|---|---|
| 功能(解包→fp16) | 437 | 1.0× | — |
| 原生 MXFP8(`VEC32_UE8M0`) | 601 | 1.38× | 1.0× |
| **原生 NVFP4(`VEC16_UE4M3`)** | **1318** | **3.0×** | **2.2×** |

(NVFP4 扫描:2048³/4096³/8192³ = 491 / 999 / 1318 TFLOP/s。)

**保真**:E8M0→E4M3 缩放转换在每块缩放指数落在 E4M3 可表示的 2 的幂范围 **[2⁻⁶, 2⁸]**(15 个二进制档)内时**精确** —— 已验证:`NVFP4_HW=1` 下 MXFP4 容差测试以 `maxrel = 0.00e+00` 通过。超出该范围的指数被钳位,故 NVFP4 是面向"每块动态范围超过 15 档"张量的**高速、降保真档**;精确功能路径(完整 E8M0 范围,解包→fp16)仍为保真基准与默认。用 `NVFP4_HW=1` 开启。

> 所有性能优化均保持容差对拍全绿（不牺牲正确性），且保留环境变量开关可回退到未优化路径做对照
> （`ACL_NO_CACHE_ALLOC` / `ATTN_NO_PERF` / `PAGED_SCALAR` / `FLASH_NO_WMMA` / `MXFP8_NO_HW` / `ELTWISE_NO_VEC` / `FOREACH_NO_FUSE` 等）。
