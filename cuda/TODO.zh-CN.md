# cann-on-gpu 路线图

> English: [TODO.md](TODO.md)

**算子覆盖(原 A 部分的首要目标)在本机已功能性完成。** 权威开源 aclnn 二级接口共 **1036** 个(`cann/{opbase,ops-math,ops-transformer,ops-cv,ops-nn}` 的 `9.1.0-beta.1` 分支,冻结于 `tools/official_aclnn.txt`),其中 **1014 个(~98%)** 已实现并在本机做了容差验证——对 PyTorch(forward + autograd backward)、CPU/闭式参考、结构不变量(QR/SVD/eigh 重构、RoPE/FFT/cast/MoE round-trip)、以及真实 2 机 RoCE/HCCL 通路。汇总与分族明细见 `README.md` 的"算子覆盖基线"。

凡是在这台单卡 GB10 上能做、能验证的都已做完。**下面剩的都绑定本机没有的硬件**——真昇腾卡(做 golden 对照与位级格式保真)、高带宽/原生低精度独显(做那些被 GB10 UMA 掩盖收益的性能工作)、或多卡/多节点集群(做超出已验证 2 机基线的扩展)。

---

## A. 绑定真昇腾卡

这些是关于*对昇腾的保真度*、而非 GPU 算力——需要真昇腾硬件(或慢的免卡 CANN Simulator)作参考,本机没有。

- **无 torch 参照算子的真昇腾 golden 对照。** 约 600 个已实现算子在本机没有外部数值 oracle(昇腾专有融合注意力 MLA/NSA/Paged、量化 matmul 矩阵、分布式融合通信、NZ 格式算子),本机只能靠结构不变量或退化等价验证。用真昇腾 golden(经 `tools/cannsim_golden.sh`,需上游 cann-api-explorer + cann-sim)做 second opinion 可加固——但 cann-sim 慢(~40–85 秒/算子)且这些算子大多还没有 explorer 单元。torch/CPU/不变量验证(`tools/torch_golden.sh` + `*_check` 工具)是快速默认通路,保持主位。
- **NZ / fractal 权重格式的位级保真。** `*WeightNz` matmul 族做的是*逻辑*等价(NZ 权重当逻辑行主序 → ND cuBLASLt 核,零 de-swizzle 开销、构造上最快)——**不是**位级 fractal 布局复刻。这是 GPU 上的永久设计 limitation;确认 fractal 快路径与昇腾原生布局一致需真昇腾硬件。
- **仍未实现的 22 个算子**(`tools/gap.txt`)——对 GPU 数值参考而言永久 out-of-scope:框架生命周期(`aclnnInit`/`aclnnFinalize`)、分布式控制面(`aclnnDistributeBarrier`/`V2`)、调试/RAS 基建(`aclnnSilentCheck`/`V2`、`aclnnPrecisionCompare`)、设备编解码(`aclnnHansEncode`/`Decode`)、DVPP 图像 custom 算子(`aclnnRasterizer`、`aclnnResize`、`aclnnBackgroundReplace`、`aclnnBlendImagesCustom`、`aclnnMrgbaCustom`)、RNG sim-thread(`aclnnSimThreadExponential`)、以及几个无外部参照的大融合 / tensor-list 组合(`aclnnConfusionTranspose`、`aclnnSplitTensor`、`aclnnAddLora`、`aclnnCoalesceSparse`、`aclnnExpandIntoJaggedPermute`、`aclnnFusedCrossEntropyLossWithMaxSum`、`aclnnFusedLinearOnlineMaxSum`)。本机均无可验证的 GPU 算力语义。

## B. 绑定其它硬件的性能工作

当前性能数据采集于 GB10:统一内存(UMA)、低带宽(LPDDR5X ~273 GB/s)、大 L2、单架构 SASS、单卡。
这套硬件特性会**掩盖访存类优化的收益**、且缺少某些原生低精度硬件单元。
下列工作的正确性已验证或实现已就绪,但**性能/原生路径须在高带宽独显或多卡上重测、调优、或新做**。
目标卡:H100/H200(`sm_90`)、RTX 40(`sm_89`)、B200/GB200(`sm_100`)、RTX 50(`sm_120`)等。

### 需要重测(实现已就绪,收益被 GB10 掩盖)

- **全部性能数字重测**:`BENCHMARK.md` 的所有数字是 GB10 专属、不可外推,换卡须整体重跑。
- **128-bit 向量化访存**:在高带宽(GDDR7/HBM)才显效,GB10 上被低带宽掩盖。
- **降访存流量的算子融合**:部分在 UMA 也有益,但完整收益须高带宽 + tensor-core 饱和时才显现。
- **低精度原生 tensor-core 吞吐**:MXFP8 原生微缩放路径已在 Blackwell 跑通;在各架构上的原生路径需各自重测(Hopper/Ada 无原生 MX,会走功能路径)。
- **开关类优化的加速比**:TF32 快路径 GEMM、fp8-flash、大 L bitonic 排序、原生 grouped GEMM、conv 算法 autotune(cuDNN find / Winograd)等,收益在 GB10 UMA 下普遍被掩盖,须高带宽 / 原生低精度卡重测。
- **多机通信扩展性**:当前只在 2 机 × 1 卡验证了正确性,未证扩展性;大规模 ring/tree、非 2 幂次 rank、节点内 NVLink 均待多卡/多节点集群验证。

### 值得新做(高带宽卡上才有意义或才显效)

1. **原生低精度 GEMM 内核路径**:fp4/fp6/MXFP4 直喂 tensor-core,而非现有"解包→fp16 GEMM"的功能路径。实打实的吞吐/显存收益,但只在原生 fp4/fp8 卡(Blackwell)有意义;功能路径已保真,原生路径作可选高速档。
2. **FlashAttention-3 级注意力内核**:当前是 WMMA 自写 flash,质量不错但非 FA3 级。注意力是 compute-bound 区的主要可优化点,重写/调优须在高带宽卡上 profile 才能定向。
3. **降访存流量的融合落地**:fused QKV 投影、residual+norm+下一投影、attention 进一步免物化等——须高带宽卡量化完整收益。
4. **128-bit 向量化访存全面铺开**:提升 memory-level parallelism,只在高带宽显存上显效。
5. **cann-on-gpu vs 原生 CUDA 的对比基准**:GB10 多数场景是访存-bound 而非 compute-bound,对比结论失真;须在高带宽卡上测,才能给出可信的"算力密集区差距"数字。
6. **多卡张量并行 / NVLink / 多节点扩展**:TP/PP 切分、节点内 NVLink、大规模集合通信,需多卡集群。
7. **Foreach 多张量融合**:foreach 族(`src/ops/foreach_ext.cu`)目前对列表中每个张量各发一次 kernel。可将整个列表融成单次 grid-stride 多张量 kernel(一次启动遍历打包的张量元信息数组,类似 PyTorch 的 `MultiTensorApply`)以消除逐张量启动开销 —— 在含大量小参数张量的优化器状态场景收益最明显。

### 原生低精度缩放格式的保真问题(待原生 fp4 卡验证)

MXFP4 的原生 NVFP4 路径需把昇腾的 E8M0/32 块缩放转成 E4M3/16 块缩放:受 E4M3 动态范围限制、且非昇腾原格式,只能作"可选高速低保真"档,须在原生 fp4 卡(Blackwell)上验证。昇腾 MXFP4 的**保真**计算仍用功能路径(块反量化→fp16,全 E8M0 范围、精确)。

## C. 可选的未来结构工作(非硬件绑定)

- **`ops*` 按库分编——已评估,不采纳。** 把构建拆成五个上游 `.so`(`opbase` / `ops-math` / `ops-transformer` / `ops-cv` / `ops-nn`)与当前源码布局**不兼容**:`cuda/src/ops/*.cu` 是按**功能族**(blas、norm、attention……)组织的,与上游**库**分组是正交的两种切法——多数族文件混了 2–4 个库的算子(如 `misc_ext.cu` 横跨四库;`index_ext.cu`/`matmul.cu`/`loss.cu` 跨三库)。一个 `.cu` → 一个 `.o` → 只能进一个 `.so`,所以忠实分库须按 `tools/op_map.tsv` 把每个算子重新切分(撤销刚做的功能合并、文件数又涨),还要建 `.so` 依赖 DAG(`opbase ← ops-math ← ops-nn ← ops-transformer/ops-cv`),因为存在真实跨库耦合:共享助手(`ew_run_one`/`ew_exec`/`check_same` 在 `elementwise.cu`,被 `inplace`/`foreach` 调用)、嵌套执行器调用(`mc2` 调 `aclnnMatmul`/`aclnnAddRmsNorm`)。**决定:维持单一 `libascendcl.so`**——它完全可用、对参考/CI 后端最简单、也是 `tests/`/`tools/` 的链接对象。仅当下游确有"只链子集"的实际需求时再重启。

- **HCCL 层的厂商中立互联(未来,等 API 面更大时做)。** 现在 HCCL 集合通信(`cuda/src/hccl.cu`)直接建在 **NCCL** 上——这是 NVIDIA 专有传输。CUDA 是唯一后端时没问题,但把厂商焊死进了通信路径。随着 API 面和 GPU 后端增多,应把集合通信/互联实现重构到一个**后端无关的传输抽象**之后:让 `include/hccl/hccl.h` 契约保持稳定,各后端插入自己的原生传输(如 ROCm 的 RCCL、Intel 的 oneCCL/Level-Zero),或者更好——用单一厂商中立的 fabric(UCX / libfabric / 薄 RDMA-verbs 层)跨架构共享,使跨 GPU 架构互联不再依赖 NCCL。`include/hccl/hccl.h` 这套 API 和 `Hccl*` 语义已经是正确的接缝;要做的是把 `hccl.cu` 的传输做成可插拔而非硬绑 NCCL。
