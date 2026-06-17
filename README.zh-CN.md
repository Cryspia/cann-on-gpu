# cann-on-gpu

> English: [README.md](README.md)

在 NVIDIA GPU 上重新实现华为昇腾 **CANN / AscendCL** 的 API 表面（`acl*` 运行时、`aclnn*` 算子、`Hccl*` 通信），
让面向昇腾 API 编写的程序**无需改动**即可在普通 NVIDIA GPU 上编译、运行、验证。

## 这个项目解决什么

- **没有昇腾卡也能开发和验证 CANN 算子。** 面向昇腾 API 写的代码可直接链本项目的 `libascendcl.so` 跑在 NVIDIA GPU 上。
  相比官方的无卡 CPU 仿真（指令级、单算子分钟级、随张量规模 O(work) 增长、大张量不可行），GPU 路径**快几个数量级**拿到可对拍的结果，
  且随硬件吞吐扩展——把"验证一个昇腾算子/一张网"的周转从分钟级压到毫秒级。
- **可执行的语义参考与回归基线。** 一份按 CANN 数学语义实现、可读的开源算子库，等于昇腾算子语义的"可执行文档"；
  并能与**真实昇腾 golden**（经无卡仿真产出）做容差对拍，把昇腾算子的功能正确性放进**普通 GPU 的 CI**，不再依赖稀缺昂贵的昇腾硬件。
- **探索跨架构的 AI Infra 融合。** 把"算子/通信的 API 契约"与"具体芯片"解耦：同一份应用二进制，换一个 GPU 后端即可运行。
  本项目把契约（`include/`）与后端（`cuda/`）分层，为将来扩展到其他 GPU（如 AMD ROCm、Intel SYCL）预留——证明 AI 基础设施可以是**厂商可移植**的。
- **不止于玩具的推理后端雏形。** 已能端到端加载真实开源模型（Qwen3-0.6B）跑前向并贪心续写，逐 token 复现 HuggingFace 参考输出。
  混合架构 **Qwen3-Next（“Qwen3.5”）** —— GatedDeltaNet 线性注意力 / SSM 层、门控全注意力、逐层稀疏 MoE —— 也已在 shim 上端到端跑通并与 HuggingFace logits 一致（`tools/qwen35/`）。

> 定位：**功能 / 容差级验证**与跨架构后端探索，**不**追求与昇腾 bit-exact 的数值保真，也**不**对标昇腾硬件性能。

## 当前进度

**较完整的 CUDA 后端 —— 已实现并容差验证约 1014 个算子**（约占 aclnn 全集的 98%；权威目标与剩余 out-of-scope 集合见下文"算子覆盖基准"）。概览：

- **算子覆盖**：逐元素全家（含广播 / 混合 dtype / 非连续视图）、任意维归约、≈17 种激活与初等函数、形变与索引、比较与逻辑、排序与扫描；
  matmul（含 batched / 转置 / bias 融合 / 量化 W4A16·W8A16·W8A8 / 低精度 fp8·fp4·fp6·MXFP8·MXFP4 / 分组 MoE）、
  卷积（2D/3D 前向与反向、转置卷积、池化含 3D 与自适应）、注意力（朴素 + flash + cuBLASLt 性能版、GQA/MQA、mask、KV-cache 的 Paged/Incre/Prompt、RoPE）、
  归一化（Softmax/LayerNorm/RMSNorm/GroupNorm/BatchNorm/InstanceNorm/LRN/RmsNormGated 及融合 AddRmsNorm·SwiGlu·GeGlu）、Embedding、损失、优化器、随机数族、显式量化、低精度编解码、分形格式转换。
  覆盖 19 个算子族的完整 aclnn 面：逐元素数学与激活补全、选择类归约（median/kthvalue/quantile/mode）与扫描（cummax/cummin/logcumsumexp）、索引（tril/triu/diag/bincount/searchsorted/scatter-max·min·mul/take）、形变（im2col/pixel-shuffle/各类 pad）、视觉（upsample/grid-sample/自适应池化/NMS）含反向、BLAS 扩展（mm/bmm/addmm/dot/kron）、**线性代数（inverse/solve/det/cholesky/qr/svd/eigh/lu/pinverse/matrix-exp，经 cuSOLVER）**、**FFT/RFFT（cuFFT）**、RNN（LSTM/GRU）、**线性注意力 / SSM（causal-conv1d、Mamba 选择性扫描、门控 delta 规则）**、以及 **MoE 路由（gating-topk-softmax / token permute-unpermute）+ 多潜在注意力 + RoPE 反向**。
- **另覆盖**：`foreach` 多张量族（70 个）、完整的 `Inplace*` 族（约 95 个，复用逐元素/比较核）、张量生成器（Eye/Linspace/LogSpace/Range）、Normal/Bernoulli 随机变体、注意力版本变体（FlashAttention/IncreFlashAttention/PromptFlashAttention/FusedInferAttention V2–V5）、**MoE 路由 dispatch/combine 及其梯度**（InitRouting/FinalizeRouting/TokenPermute round-trip）、**量化 matmul 版本变体**（QuantMatmul V2–V5、WeightQuantBatchMatmul、GroupedMatmulAdd）与**融合 norm+量化**（AddRmsNormQuant/DynamicQuant、RmsNormQuant、SwiGluQuant、AdaLayerNorm 及反向、GemmaRmsNorm、GroupNormSilu/Swish）、GLU 梯度（SwiGlu/GeGlu 反向）、`Gemm`/`GatherNd`/`ScatterNd`、以及 **NZ 权重格式 matmul 族**（`MatmulWeightNz` 等，做逻辑等价——格式 limitation 见范围说明）。
- **分布式融合集合通信，已在真实 2 机 RoCE/HCCL 链路上验证**（双网卡 200G）：`mc2` 族——`MatmulAllReduce`/`MatmulReduceScatter`/`MatmulAllGather`（K 切分或行切分本地 matmul + 对应 HCCL 集合通信）、`QuantMatmulAllReduce`、`MatmulAllReduceAddRmsNorm`（尾部融合 norm）、以及 MoE 专家并行 `MoeDistributeDispatch`/`Combine`（按 capacity 布局做 HcclAlltoAll，round-trip 验证）——全部走非退化的多 rank 真实通路，而非单 rank 退化。
- **dtype**：fp32 / fp16 / bf16 全算子覆盖；int8/int4/fp8/fp4/fp6/hifloat8 在量化与低精度路径覆盖。
- **通信**：HCCL→NCCL，双机全集合通信（AllReduce/AllGather/ReduceScatter/Broadcast/Send-Recv/AllToAll）+ ClusterInfo 端到端验证通过。
- **验证**：单进程容差对拍 **742 项检查、31 个测试程序全绿、0 fail**，外加 2 机 HCCL 集合通信 / mc2 套件；**74 个算子与真实昇腾 golden 对拍通过**（经无卡仿真经 `tools/cannsim_golden.sh` 产出昇腾输出，多数误差 ~1e-7，fp32 反三角 ~1e-5）；
  真实模型 Qwen3-0.6B 端到端（`tools/qwen3/`），prefill logits 与 HF 误差 ~5.4e-6、贪心续写 8/8 token 一致；
  混合架构 Qwen3-Next（“Qwen3.5”，`tools/qwen35/`）前向（GatedDeltaNet 线性注意力 + 门控注意力 + 稀疏 MoE）对 HF：单层 GatedDeltaNet ~8e-7、tiny 整模逐层 ~1e-6、**logits ~3.8e-6、贪心下一 token 一致**。

详见各子文档（见下"文档")。

## 算子覆盖基准

权威算子集以 gitcode 上的**开源 CANN 算子库**为准 —— [`cann/opbase`](https://gitcode.com/cann/opbase)、[`cann/ops-math`](https://gitcode.com/cann/ops-math)、[`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer)、[`cann/ops-cv`](https://gitcode.com/cann/ops-cv)、[`cann/ops-nn`](https://gitcode.com/cann/ops-nn) —— 锁定 **`9.1.0-beta.1`** 分支（与 CANN 9.1 toolkit 对应）。这些库面向 Atlas A2/A3，**并通过 CANN Simulator 支持 Ascend 950PR/950DT**；本项目以 **950** 接口为基准。

- **权威清单**：**1036** 个 `aclnn*` level-2 API（`tools/official_aclnn.txt`），与 A3 `ops` run 包交叉核对一致率 97.4% —— 证实 aclnn host 接口跨 SoC 统一，950/A3 之分几乎不影响"有哪些函数"。
- **当前覆盖**：已实现并容差验证 **1014 / 1036 ≈ 98%**。剩余 **22**（`tools/gap.txt`）为无 GPU-compute 语义的昇腾设备/编解码/调试基建算子 —— 如 Hans 编解码、Rasterizer、图像自定义（Blend/Mrgba/BackgroundReplace）、分布式 barrier、SilentCheck、PrecisionCompare、Init/Finalize、NpuFormatCast 类布局转换,以及少数需 tensor-list 输出或大型融合的（ConfusionTranspose、SplitTensor、AddLora、CoalesceSparse、Fused*OnlineMaxSum）。这些在 `cuda/TODO.md` 标记为永久 out-of-scope。
- **数据格式保真**：每个算子的官方 dtype/shape 矩阵可从各库 `tests/st/aclnn*/all_*.json` 系统测试用例恢复（`tools/op_map.tsv` 给出 API→库/类别 映射），用于保证我方实现接受原版支持的所有 dtype。

## 环境要求与安装

**硬件**：一张 NVIDIA GPU（Ampere/Ada/Hopper/Blackwell 均可；默认构建目标 `sm_121` = GB10，换卡改 `GPU_ARCH` 重编即可，源码不动）。
单卡即可开发与对拍；集合通信的真多 rank 验证需要两台机器。

**软件**：Linux、NVIDIA 驱动 + CUDA Toolkit 13、conda。后端依赖 cuDNN / NCCL / CUTLASS，
以及 CANN 的 ACL/aclnn/HCCL 头文件（仅**头**，编译期契约；运行期不链昇腾任何库）。

**一键安装**（幂等、装进独立 conda env、不写全局 PATH）：

```bash
./deploy.sh            # 建 env、装 cuDNN/NCCL/CUTLASS、取 ACL 头、生成 env.sh
./deploy.sh test       # 冒烟测试

source env.sh          # 导出 GPU_ARCH / ACL_INCLUDE / CUDNN_DIR / NCCL_DIR 等
make -C cuda           # 构建后端 → cuda/lib/libascendcl.so + libhccl.so
make -C tests run      # 跑全部容差对拍
```

## 目录结构

| 目录 | 作用 |
|---|---|
| `include/` | **后端无关的 API 契约头**（`aclnnop/`、`hccl/`）。任何 GPU 后端实现这同一套；客户端只依赖它。 |
| `cuda/` | **NVIDIA CUDA 后端**：`src/*.cu` 实现、`Makefile`、产物 `lib/`、CUDA 专有工具 `tools/`。实现 `include/` 的契约。 |
| `tests/` | 后端无关的 ACL 客户端测试（容差对拍）。`make`（默认 `BACKEND=../cuda`，将来可 `BACKEND=../rocm`）。 |
| `tools/` | 后端无关的客户端与编排脚本：真实模型端到端（`qwen3/`）、混合 Qwen3-Next/“Qwen3.5” 适配（`qwen35/`）、模型容器 demo（`graph_demo`）、decode-loop CUDA Graph（`decode_graph/`）、昇腾 golden 对拍闭环（`cannsim_golden*`）、HCCL 双机脚本、性能基准（`bench`）。 |
| `third_party/` | 外部依赖（CUTLASS 头文件）。可选的真昇腾 golden 对照（`tools/cannsim_golden.sh`）另需 clone 上游 `cann-api-explorer` 并设置 `EXPLORER_DIR` —— 正常使用不需要,默认验证走本地快速 torch oracle。 |
| `deploy.sh` / `env.sh` | 环境部署与变量导出。 |

## 文档

- **[`cuda/BENCHMARK.md`](cuda/BENCHMARK.zh-CN.md)** —— 性能基准的测量方法与结果,含在 RTX PRO 6000 Blackwell(`sm_120`)上一整套**同卡 A/B** 优化实测:TF32 GEMM 2.6×、原生 MXFP8 1.38×、从零构建的**原生 NVFP4 fp4 GEMM 达功能路径的 3.0×(1318 TFLOP/s)**、foreach 多张量融合 43–70×、以及与原生 cuBLAS **持平**(0.97–1.09×)。每个加速都是同一张卡上经 env 开关的优化-vs-朴素对比 —— 绝非跨卡比较。
- **[`cuda/README.md`](cuda/README.zh-CN.md)** —— CUDA 后端的实现要点、性能优化手段、技术选型理由。
- **[`cuda/TODO.md`](cuda/TODO.zh-CN.md)** —— 剩余路线图。算子覆盖已功能性完成(~98%),高带宽 GPU 的性能工作也已完成并实测(见 `cuda/BENCHMARK.md`),故该文件现只追踪本机硬件够不到的部分:真昇腾 golden 对照与位级格式保真、完整的 FlashAttention-3 级内核重写(需 Nsight Compute)、以及多卡/多节点扩展。
- **[`include/README.md`](include/README.zh-CN.md)** —— API 契约头的内容与"谁实现、谁使用"。
