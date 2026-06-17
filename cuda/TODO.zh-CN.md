# cann-on-gpu 路线图

> English: [TODO.md](TODO.md)

**算子覆盖(原 A 部分的首要目标)在本机已功能性完成。** 权威开源 aclnn 二级接口共 **1036** 个(`cann/{opbase,ops-math,ops-transformer,ops-cv,ops-nn}` 的 `9.1.0-beta.1` 分支,冻结于 `tools/official_aclnn.txt`),其中 **1014 个(~98%)** 已实现并在本机做了容差验证——对 PyTorch(forward + autograd backward)、CPU/闭式参考、结构不变量(QR/SVD/eigh 重构、RoPE/FFT/cast/MoE round-trip)、以及真实 2 机 RoCE/HCCL 通路。汇总与分族明细见 `README.md` 的"算子覆盖基线"。

**需要高带宽 / 原生低精度独显的性能工作现已完成并实测** —— 在 RTX PRO 6000 Blackwell(`sm_120`,GDDR7,原生 fp4/fp8)上,每项优化都做了同卡 A/B,并新建了原生 NVFP4 fp4 GEMM 路径。数据与方法见 [`BENCHMARK.md`](BENCHMARK.md)("Optimization A/B"、"Toggle-style optimization sweep"、"NVFP4 native fp4 path"、"cann-on-gpu vs native CUDA")。

**下面剩的都绑定本机没有的硬件**——真昇腾卡(做 golden 对照与位级格式保真)、或多卡/多节点集群(做超出已验证 2 机基线的扩展)——外加一项大型 compute-bound 内核重写和几项可选结构重构。

---

## A. 绑定真昇腾卡

这些是关于*对昇腾的保真度*、而非 GPU 算力——需要真昇腾硬件(或慢的免卡 CANN Simulator)作参考,本机没有。

- **无 torch 参照算子的真昇腾 golden 对照。** 约 600 个已实现算子在本机没有外部数值 oracle(昇腾专有融合注意力 MLA/NSA/Paged、量化 matmul 矩阵、分布式融合通信、NZ 格式算子),本机只能靠结构不变量或退化等价验证。用真昇腾 golden(经 `tools/cannsim_golden.sh`,需上游 cann-api-explorer + cann-sim)做 second opinion 可加固——但 cann-sim 慢(~40–85 秒/算子)且这些算子大多还没有 explorer 单元。torch/CPU/不变量验证(`tools/torch_golden.sh` + `*_check` 工具)是快速默认通路,保持主位。
- **NZ / fractal 权重格式的位级保真。** `*WeightNz` matmul 族做的是*逻辑*等价(NZ 权重当逻辑行主序 → ND cuBLASLt 核,零 de-swizzle 开销、构造上最快)——**不是**位级 fractal 布局复刻。这是 GPU 上的永久设计 limitation;确认 fractal 快路径与昇腾原生布局一致需真昇腾硬件。
- **仍未实现的 22 个算子**(`tools/gap.txt`)——对 GPU 数值参考而言永久 out-of-scope:框架生命周期(`aclnnInit`/`aclnnFinalize`)、分布式控制面(`aclnnDistributeBarrier`/`V2`)、调试/RAS 基建(`aclnnSilentCheck`/`V2`、`aclnnPrecisionCompare`)、设备编解码(`aclnnHansEncode`/`Decode`)、DVPP 图像 custom 算子(`aclnnRasterizer`、`aclnnResize`、`aclnnBackgroundReplace`、`aclnnBlendImagesCustom`、`aclnnMrgbaCustom`)、RNG sim-thread(`aclnnSimThreadExponential`)、以及几个无外部参照的大融合 / tensor-list 组合(`aclnnConfusionTranspose`、`aclnnSplitTensor`、`aclnnAddLora`、`aclnnCoalesceSparse`、`aclnnExpandIntoJaggedPermute`、`aclnnFusedCrossEntropyLossWithMaxSum`、`aclnnFusedLinearOnlineMaxSum`)。本机均无可验证的 GPU 算力语义。

## B. 剩余的性能 / 扩展工作

同卡优化基准测试与原生低精度 GEMM 构建均**已完成**(见 `BENCHMARK.md`)。剩下:

- **FlashAttention-3 级注意力内核。** `ncu` 显示 `k_flash_wmma` 占用率受限(实测 2.84%、1 warp/block、每 block 12.42 KB shared),收益需要完整 FA3 级重写——把每 warp 的 shared 砍到约 4 KB(O 放寄存器、缩 tile)+ warp 特化 `cp.async`/TMA/ping-pong——不是小改。cuBLASLt 批量 perf 路径(约 flash 的 3×)在 S/P 物化放得下时已是默认;flash 留作大 S/causal 的省内存路径。(另:本卡 cuBLASLt 给批量注意力 GEMM 选的是 `cutlass_80_tensorop` Ampere 内核——值得在更新 cuBLAS 上复查。)
- **降访存流量的生产级融合——剩余算子。** residual+norm(`AddRmsNorm`/`AddLayerNorm`)已构建、修正确性并量化;fused QKV 投影已量化(decode/小 M 区间最高 1.64× —— 是模型层权重拼接,不需后端内核;见 BENCHMARK)。仍未做:融合注意力路径内的进一步免物化。
- **多卡 / 多节点扩展。** 集合通信正确性仅在 2 机 × 1 卡验证。大规模 ring/tree、非 2 幂 rank、张量/流水并行切分、节点内 NVLink 均须多卡集群。

## C. 可选的未来结构工作(非硬件绑定)

- **`ops*` 按库分编——已评估,不采纳。** 把构建拆成五个上游 `.so`(`opbase` / `ops-math` / `ops-transformer` / `ops-cv` / `ops-nn`)与当前源码布局**不兼容**:`cuda/src/ops/*.cu` 是按**功能族**(blas、norm、attention……)组织的,与上游**库**分组是正交的两种切法——多数族文件混了 2–4 个库的算子(如 `misc_ext.cu` 横跨四库;`index_ext.cu`/`matmul.cu`/`loss.cu` 跨三库)。一个 `.cu` → 一个 `.o` → 只能进一个 `.so`,所以忠实分库须按 `tools/op_map.tsv` 把每个算子重新切分(撤销刚做的功能合并、文件数又涨),还要建 `.so` 依赖 DAG(`opbase ← ops-math ← ops-nn ← ops-transformer/ops-cv`),因为存在真实跨库耦合:共享助手(`ew_run_one`/`ew_exec`/`check_same` 在 `elementwise.cu`,被 `inplace`/`foreach` 调用)、嵌套执行器调用(`mc2` 调 `aclnnMatmul`/`aclnnAddRmsNorm`)。**决定:维持单一 `libascendcl.so`**——它完全可用、对参考/CI 后端最简单、也是 `tests/`/`tools/` 的链接对象。仅当下游确有"只链子集"的实际需求时再重启。

- **HCCL 层的厂商中立互联(未来,待 API 面更大时)。** HCCL 集合通信(`cuda/src/hccl.cu`)目前直接建在 **NCCL** 上——NVIDIA 专有传输。在 CUDA 是唯一后端时无妨,但把厂商焊死进了通信通路。随着 API 面与 GPU 后端集合增长,应把集合/互联实现重构到**后端无关的传输抽象**之后,让 `include/hccl/hccl.h` 契约保持稳定、各后端插各自原生传输(ROCm 上 RCCL、Intel 上 oneCCL/Level-Zero)——或更好,用单一厂商中立 fabric(UCX / libfabric / 薄 RDMA-verbs 层)跨架构共享,使跨 GPU 架构互联不依赖 NCCL。`include/hccl/hccl.h` 接口与 `Hccl*` 语义已是正确的接缝;工作是把 `hccl.cu` 的传输做成可插拔而非 NCCL 硬连。
