# cann-on-gpu

> English: [README.md](README.md)

在**通用 GPU** 上重新实现华为昇腾 **CANN / AscendCL** 的 API 表面(`acl*` 运行时、`aclnn*` 算子、`Hccl*` 通信),
让面向昇腾 API 编写的程序**无需改动**即可在普通 GPU 上编译、运行、验证。
API 契约与具体芯片解耦:同一份应用二进制,换一个 GPU 后端即可运行。**目前提供两个后端**——**NVIDIA CUDA**(`cuda/`)与 **Apple Metal**(`metal/`)——都实现 `include/` 里的同一套契约。

## 这个项目解决什么

- **没有昇腾卡也能开发和验证 CANN 算子。** 面向昇腾 API 写的代码可直接链本项目的 `libascendcl` 跑在普通 GPU(NVIDIA 或 Apple)上。
  相比官方的无卡 CPU 仿真(指令级、单算子分钟级、随张量规模 O(work) 增长、大张量不可行),GPU 路径**快几个数量级**拿到可对拍的结果,
  且随硬件吞吐扩展——把"验证一个昇腾算子/一张网"的周转从分钟级压到毫秒级。
- **可执行的语义参考与回归基线。** 一份按 CANN 数学语义实现、可读的开源算子库,等于昇腾算子语义的"可执行文档";
  并能与**真实昇腾 golden**(经无卡仿真产出)做容差对拍,把昇腾算子的功能正确性放进**普通 GPU 的 CI**,不再依赖稀缺昂贵的昇腾硬件。
- **跨架构的 AI Infra 可移植性。** 通过把"算子/通信的 API 契约"(`include/`)与"具体芯片"(`cuda/`、`metal/`)分层,本项目证明 AI 基础设施可以是**厂商可移植**的:
  同一套契约如今既在 NVIDIA 独显/UMA GPU 上满足,也在 Apple 芯片 GPU 上满足,并为扩展到其他 GPU(如 AMD ROCm、Intel SYCL)预留空间。
- **不止于玩具的推理后端雏形。** 已能端到端加载真实开源模型(Qwen3-0.6B)跑前向并贪心续写,逐 token 复现 HuggingFace 参考输出——在**任一后端**上。
  混合架构 **Qwen3-Next("Qwen3.5")**——GatedDeltaNet 线性注意力 / SSM 层、门控全注意力、逐层稀疏 MoE——也已在**两个后端上**端到端跑通并与 HuggingFace logits 一致(`tools/qwen35/`)。

> 定位:**功能 / 容差级验证**与跨架构后端探索,**不**追求与昇腾 bit-exact 的数值保真,也**不**对标昇腾硬件性能。

## 当前进度

两个后端共享同一套契约、`tests/` 里同一批后端无关测试、`tools/` 里同一批客户端工具。

### CUDA 后端(NVIDIA)—— 较完整覆盖,容差验证

**已实现并容差验证约 1014 个算子**(约占 aclnn 全集的 98%;见"算子覆盖基准")。概览:

- **算子覆盖**:逐元素全家(广播 / 混合 dtype / 非连续视图)、任意维归约、激活与初等函数、形变与索引、比较与逻辑、排序与扫描;
  matmul(batched / 转置 / bias 融合 / 量化 W4A16·W8A16·W8A8 / 低精度 fp8·fp4·fp6·MXFP8·MXFP4 / 分组 MoE)、
  卷积(2D/3D 前向与反向、转置、池化含 3D 与自适应)、注意力(朴素 + flash + cuBLASLt 性能版、GQA/MQA、mask、KV-cache 的 Paged/Incre/Prompt、RoPE)、
  归一化(Softmax/LayerNorm/RMSNorm/GroupNorm/BatchNorm/InstanceNorm/LRN/RmsNormGated 及融合 AddRmsNorm·SwiGlu·GeGlu)、Embedding、损失、优化器、随机数族、显式量化、低精度编解码、分形格式转换、
  线性代数(inverse/solve/det/cholesky/qr/svd/eigh/lu/pinverse/matrix-exp,经 cuSOLVER)、FFT/RFFT(cuFFT)、RNN(LSTM/GRU)、线性注意力 / SSM(causal-conv1d、Mamba 选择性扫描、门控 delta 规则)、以及 MoE 路由 + 多潜在注意力 + RoPE 反向。
- **分布式融合集合通信,已在真实 2 机 RoCE/HCCL 链路上验证**(双网卡 200G):`mc2` 族——`MatmulAllReduce`/`MatmulReduceScatter`/`MatmulAllGather`、`QuantMatmulAllReduce`、`MatmulAllReduceAddRmsNorm`、以及 MoE 专家并行 `MoeDistributeDispatch`/`Combine`——全部走非退化的多 rank 真实通路。
- **验证**:单进程容差对拍 **742 项检查、31 个测试程序全绿**,外加 2 机 HCCL/mc2 套件;**74 个算子与真实昇腾 golden 对拍通过**;Qwen3-0.6B 端到端对 HF(prefill logits ~5.4e-6、贪心 8/8)。同卡性能优化实测见 [`cuda/BENCHMARK.md`](cuda/BENCHMARK.zh-CN.md)。

### Metal 后端(Apple)—— 算子完全对齐

**与 CUDA 后端算子完全对齐**——CUDA 后端导出的每个 `aclnn*` 算子在 Metal 上都已实现(用两个动态库的导出符号双向核对验证)。

- **实现**:主机层用 Objective-C++ 驱动 MPS/MPSGraph/Accelerate;elementwise/softmax/norm/sort 族用手写 MSL kernel;**统一内存**让 `aclTensor` 设备指针主机可寻址,于是 H2D/D2H 即 `memcpy`、长尾算子族走主机端零拷贝。`Hccl*` 为单 rank stub(一台 Mac = 一个设备)。
- **验证**:所有后端无关测试二进制通过(`make -C tests BACKEND=../metal run`),外加各算子族对齐测试;Qwen3-0.6B(fp32)端到端对齐 HF——prefill logits **3.82e-6**、贪心 **8/8 token 一致**(与 CUDA 相同)。正确性另由**与 CUDA 后端对拍**及独立 PyTorch oracle 交叉验证。

各后端详情见下文"文档"。

## 算子覆盖基准

权威算子集以 gitcode 上的**开源 CANN 算子库**为准 —— [`cann/opbase`](https://gitcode.com/cann/opbase)、[`cann/ops-math`](https://gitcode.com/cann/ops-math)、[`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer)、[`cann/ops-cv`](https://gitcode.com/cann/ops-cv)、[`cann/ops-nn`](https://gitcode.com/cann/ops-nn) —— 锁定 **`9.1.0-beta.1`** 分支(与 CANN 9.1 toolkit 对应)。这些库面向 Atlas A2/A3,**并通过 CANN Simulator 支持 Ascend 950PR/950DT**;本项目以 **950** 接口为基准。

- **权威清单**:**1036** 个 `aclnn*` level-2 API(`tools/official_aclnn.txt`),与 A3 `ops` run 包交叉核对一致率 97.4% —— 证实 aclnn host 接口跨 SoC 统一。
- **CUDA 覆盖**:已实现并容差验证 **1014 / 1036 ≈ 98%**。剩余 **22**(`tools/gap.txt`)为无 GPU-compute 语义的昇腾设备/编解码/调试基建算子(在 `cuda/TODO.md` 标记为永久 out-of-scope)。
- **Metal 覆盖**:与 CUDA 后端的导出算子集完全对齐。
- **数据格式保真**:每个算子的官方 dtype/shape 矩阵可从各库 `tests/st/aclnn*/all_*.json` 系统测试用例恢复(`tools/op_map.tsv` 给出 API→库/类别 映射),用于保证我方实现接受原版支持的所有 dtype。

## 环境要求与安装

`deploy.sh` 按主机 OS 自动选后端(可用 `BACKEND=cuda|metal` 覆盖)。幂等、装进独立 conda/miniforge env、不写全局 PATH。

**NVIDIA / CUDA(Linux)**:一张 NVIDIA GPU(Ampere/Ada/Hopper/Blackwell;默认构建目标 `sm_121` = GB10,换卡改 `GPU_ARCH` 重编)。NVIDIA 驱动 + CUDA Toolkit 13;后端依赖 cuDNN / NCCL / CUTLASS。

```bash
./deploy.sh            # 建 env、装 cuDNN/NCCL/CUTLASS、取 ACL 头、生成 env.sh
source env.sh          # 导出 GPU_ARCH / ACL_INCLUDE / CUDNN_DIR / NCCL_DIR 等
make -C cuda           # → cuda/lib/libascendcl.so + libhccl.so
make -C tests run      # 跑全部容差对拍(默认 BACKEND=../cuda)
```

**Apple / Metal(macOS)**:一台 Apple 芯片 Mac + 完整 Xcode + 可下载的 Metal Toolchain。`deploy.sh` 会走 Metal 分支(forge 环境、用 `DEVELOPER_DIR` 做项目内 Xcode 作用域——**不改全局 `xcode-select`**)。

```bash
./deploy.sh                          # macOS → Metal 分支(forge 环境 + Xcode/Metal Toolchain 检查)
source env.sh                        # BACKEND=metal;用 DEVELOPER_DIR 做 Xcode 作用域
make -C metal                        # → metal/lib/libascendcl.dylib + libhccl.dylib + default.metallib
make -C tests BACKEND=../metal run   # 用 Metal 后端跑全部容差对拍
```

两条路径仅共享 ACL/aclnn/HCCL **头文件**(编译期契约;运行期不链昇腾任何库)。

## 目录结构

| 目录 | 作用 |
|---|---|
| `include/` | **后端无关的 API 契约头**(`aclnnop/`、`hccl/`)。任何 GPU 后端实现这同一套;客户端只依赖它。 |
| `cuda/` | **NVIDIA CUDA 后端**:`src/*.cu`、`Makefile`、产物 `lib/`、CUDA 专有工具与基准文档。实现 `include/` 的契约。 |
| `metal/` | **Apple Metal 后端**:`src/*.{mm,metal}`、`Makefile`、产物 `lib/`、后端文档。在 Apple 芯片 GPU 上实现同一套 `include/` 契约。 |
| `tests/` | 后端无关的 ACL 客户端测试(容差对拍)。`make`(默认 `BACKEND=../cuda`;Apple 用 `BACKEND=../metal`;将来可 `BACKEND=../rocm`)。 |
| `tools/` | 后端无关的客户端与编排脚本:真实模型端到端(`qwen3/`)、混合 Qwen3-Next/"Qwen3.5"(`qwen35/`)、模型容器 demo、decode-loop 图、昇腾 golden 对拍闭环(`cannsim_golden*`)、PyTorch 差分 oracle(`torch_golden.sh` / `torch_check.sh`)、跨后端对拍(`cann_metal_diff.sh`)、HCCL 多机脚本、性能基准(`bench`)。 |
| `third_party/` | 外部依赖(CUDA 后端的 CUTLASS 头)。可选的真昇腾 golden 对照另需上游 `cann-api-explorer`——正常使用不需要,默认验证走本地快速 torch oracle。 |
| `deploy.sh` / `env.sh` | 环境部署与变量导出(按 OS 选后端)。 |

## 文档

- **[`cuda/README.md`](cuda/README.zh-CN.md)** —— CUDA 后端的实现要点、性能优化手段、技术选型理由。
- **[`cuda/BENCHMARK.md`](cuda/BENCHMARK.zh-CN.md)** —— CUDA 基准方法与同卡优化实测。
- **[`cuda/TODO.md`](cuda/TODO.zh-CN.md)** —— CUDA 后端剩余路线图(硬件受限项 + 扩展)。
- **[`metal/README.md`](metal/README.zh-CN.md)** —— Metal 后端概览、算子-后端映射、数值策略。
- **[`metal/BENCHMARK.md`](metal/BENCHMARK.zh-CN.md)** —— Metal 基准方法、M 系列结果、跨后端对比。
- **[`metal/TODO.md`](metal/TODO.zh-CN.md)** —— Metal 后端剩余 / 可优化路线图。
- **[`include/README.md`](include/README.zh-CN.md)** —— API 契约头的内容与"谁实现、谁使用"。
