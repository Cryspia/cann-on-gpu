# cann-on-gpu

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

Re-implements the Huawei Ascend **CANN / AscendCL** API surface on NVIDIA GPUs (`acl*` runtime, `aclnn*` operators, `Hccl*` collectives),
so that programs written against the Ascend API can be compiled, run, and validated on ordinary NVIDIA GPUs **without any code changes**.

## What Problem This Solves

- **Develop and validate CANN operators without an Ascend card.** Code written against the Ascend API can be linked directly against this project's `libascendcl.so` and run on an NVIDIA GPU.
  Compared with the official card-free CPU simulation (instruction-level, minutes per operator, O(work) growth with tensor size, infeasible for large tensors), the GPU path delivers comparable results **orders of magnitude faster**
  and scales with hardware throughput — compressing the turnaround for "validate one Ascend operator / one network" from minutes down to milliseconds.
- **Executable semantic reference and regression baseline.** A readable, open-source operator library implemented to CANN's mathematical semantics serves as "executable documentation" for Ascend operator semantics.
  It can also be tolerance-compared against **real Ascend golden outputs** (produced via card-free simulation), bringing Ascend operator functional correctness into **ordinary GPU CI** without depending on scarce, expensive Ascend hardware.
- **Exploring cross-architecture AI infrastructure convergence.** Decouples the "operator/collective API contract" from the "specific chip": the same application binary runs by swapping in a different GPU backend.
  This project layers the contract (`include/`) separately from the backend (`cuda/`), leaving room to extend to other GPUs (e.g., AMD ROCm, Intel SYCL) in the future — demonstrating that AI infrastructure can be **vendor-portable**.
- **A non-toy inference backend prototype.** Already capable of end-to-end loading of a real open-source model (Qwen3-0.6B), running a forward pass and greedy decoding, reproducing HuggingFace reference output token by token.
  The hybrid **Qwen3-Next ("Qwen3.5")** architecture — GatedDeltaNet linear-attention / SSM layers, gated full-attention, and per-layer sparse MoE — also runs end-to-end on the shim and matches HuggingFace logits (`tools/qwen35/`).

> Scope: **functional / tolerance-level validation** and cross-architecture backend exploration. This project does **not** pursue bit-exact numerical fidelity with Ascend, nor does it benchmark against Ascend hardware performance.

## Current Status

**Broad CUDA backend — ~1014 operators implemented and tolerance-verified** (≈98% of the full aclnn surface; see "Operator Coverage Baseline" below for the authoritative target and the remaining out-of-scope set). Overview:

- **Operator coverage**: full elementwise family (with broadcasting / mixed dtype / non-contiguous views), arbitrary-dim reduction, ~17 activation and elementary functions, reshape and indexing, comparison and logic, sort and scan;
  matmul (batched / transpose / bias fusion / quantized W4A16·W8A16·W8A8 / low-precision fp8·fp4·fp6·MXFP8·MXFP4 / grouped MoE),
  convolution (2D/3D forward and backward, transposed convolution, pooling including 3D and adaptive), attention (naive + flash + cuBLASLt performance variant, GQA/MQA, mask, KV-cache Paged/Incre/Prompt, RoPE),
  normalization (Softmax/LayerNorm/RMSNorm/GroupNorm/BatchNorm/InstanceNorm/LRN/RmsNormGated and fused AddRmsNorm·SwiGlu·GeGlu), Embedding, loss, optimizer, RNG family, explicit quantization, low-precision encode/decode, fractal format conversion.
  Covers the full aclnn surface across 19 operator families: elementwise math & activation completions, selection reductions (median/kthvalue/quantile/mode) and scans (cummax/cummin/logcumsumexp), indexing (tril/triu/diag/bincount/searchsorted/scatter-max·min·mul/take), shape (im2col/pixel-shuffle/pad variants), vision (upsample/grid-sample/adaptive-pool/NMS) with backward, BLAS extras (mm/bmm/addmm/dot/kron), **linear algebra (inverse/solve/det/cholesky/qr/svd/eigh/lu/pinverse/matrix-exp via cuSOLVER)**, **FFT/RFFT (cuFFT)**, RNN (LSTM/GRU), **linear-attention / SSM (causal-conv1d, Mamba selective-scan, gated-delta-rule)**, and **MoE routing (gating-topk-softmax / token permute-unpermute) + multi-latent attention + RoPE backward**.
- **Also covered**: the `foreach` multi-tensor family (70 ops), a comprehensive `Inplace*` family (~95 ops sharing the elementwise/compare cores), tensor generators (Eye/Linspace/LogSpace/Range), Normal/Bernoulli RNG variants, attention version variants (FlashAttention/IncreFlashAttention/PromptFlashAttention/FusedInferAttention V2–V5), **MoE routing dispatch/combine + their gradients** (InitRouting/FinalizeRouting/TokenPermute round-trips), **quant-matmul version variants** (QuantMatmul V2–V5, WeightQuantBatchMatmul, GroupedMatmulAdd) and **fused norm+quant** (AddRmsNormQuant/DynamicQuant, RmsNormQuant, SwiGluQuant, AdaLayerNorm + backward, GemmaRmsNorm, GroupNormSilu/Swish), GLU gradients (SwiGlu/GeGlu backward), `Gemm`/`GatherNd`/`ScatterNd`, and the **NZ weight-format matmul family** (`MatmulWeightNz` etc., implemented to logical equivalence — see the format limitation in scope).
- **Distributed fused collectives, verified on a real 2-node RoCE/HCCL link** (dual-NIC 200G): the `mc2` family — `MatmulAllReduce`/`MatmulReduceScatter`/`MatmulAllGather` (K-split or row-shard local matmul + the matching HCCL collective), `QuantMatmulAllReduce`, `MatmulAllReduceAddRmsNorm` (tail-fused norm), and MoE expert-parallel `MoeDistributeDispatch`/`Combine` (HcclAlltoAll over the capacity layout, round-trip verified) — all run the non-degenerate multi-rank path, not a single-rank fallback.
- **dtypes**: fp32 / fp16 / bf16 full operator coverage; int8/int4/fp8/fp4/fp6/hifloat8 covered in quantization and low-precision paths.
- **Collectives**: HCCL→NCCL, two-machine full collective suite (AllReduce/AllGather/ReduceScatter/Broadcast/Send-Recv/AllToAll) + ClusterInfo end-to-end validated.
- **Validation**: single-process tolerance tests **all green (742 checks across 31 test binaries, 0 fail)** plus the 2-node HCCL collective/mc2 suite; **74 operators tolerance-matched against real Ascend golden outputs** (Ascend outputs produced via card-free simulation through `tools/cannsim_golden.sh`, most errors ~1e-7, fp32 inverse-trig ~1e-5);
  real model Qwen3-0.6B end-to-end (`tools/qwen3/`): prefill logits error vs. HF ~5.4e-6, greedy decoding 8/8 tokens identical;
  hybrid Qwen3-Next ("Qwen3.5") forward (`tools/qwen35/`): standalone GatedDeltaNet layer ~8e-7, tiny full model (GatedDeltaNet linear-attention + gated attention + sparse MoE) vs. HF per-layer ~1e-6, **logits ~3.8e-6, greedy next token identical**.

See the sub-documents listed under "Documentation" below for details.

## Operator Coverage Baseline

The authoritative operator set is the **open-source CANN operator libraries** on gitcode — [`cann/opbase`](https://gitcode.com/cann/opbase), [`cann/ops-math`](https://gitcode.com/cann/ops-math), [`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer), [`cann/ops-cv`](https://gitcode.com/cann/ops-cv), [`cann/ops-nn`](https://gitcode.com/cann/ops-nn) — pinned to the **`9.1.0-beta.1`** branch (matching the CANN 9.1 toolkit). These libraries target Atlas A2/A3 **and Ascend 950PR/950DT through the CANN Simulator**; the **950** interface is this project's baseline.

- **Authoritative list**: **1036** `aclnn*` level-2 APIs (`tools/official_aclnn.txt`), cross-checked against the A3 `ops` run package (97.4% identical — confirming the aclnn host API is uniform across SoCs, so the 950/A3 distinction barely affects *which* functions exist).
- **Current coverage**: **1014 / 1036 ≈ 98%** implemented and tolerance-verified. The remaining **22** (`tools/gap.txt`) are Ascend device/codec/debug infrastructure ops with no GPU-compute meaning — e.g. Hans encode/decode, Rasterizer, image-custom ops (Blend/Mrgba/BackgroundReplace), distributed barriers, SilentCheck, PrecisionCompare, Init/Finalize, NpuFormatCast-style layout casts, and a few that need tensor-list outputs or large fused composites (ConfusionTranspose, SplitTensor, AddLora, CoalesceSparse, Fused*OnlineMaxSum). These are tracked as permanent out-of-scope in `cuda/TODO.md`.
- **dtype fidelity**: every operator's official dtype/shape matrix is recoverable from each library's `tests/st/aclnn*/all_*.json` system-test cases (`tools/op_map.tsv` maps each API → library/category), used to ensure our implementations accept every dtype the original supports.

## Requirements and Installation

**Hardware**: one NVIDIA GPU (Ampere/Ada/Hopper/Blackwell all supported; default build target `sm_121` = GB10 — change `GPU_ARCH` and recompile for a different card, no source changes needed).
A single card is sufficient for development and tolerance testing; true multi-rank collective validation requires two machines.

**Software**: Linux, NVIDIA driver + CUDA Toolkit 13, conda. The backend depends on cuDNN / NCCL / CUTLASS,
as well as the CANN ACL/aclnn/HCCL header files (headers **only** — compile-time contract; no Ascend library is linked at runtime).

**One-command install** (idempotent, installs into a dedicated conda env, does not modify global PATH):

```bash
./deploy.sh            # create env, install cuDNN/NCCL/CUTLASS, fetch ACL headers, generate env.sh
./deploy.sh test       # smoke test

source env.sh          # export GPU_ARCH / ACL_INCLUDE / CUDNN_DIR / NCCL_DIR etc.
make -C cuda           # build backend → cuda/lib/libascendcl.so + libhccl.so
make -C tests run      # run all tolerance tests
```

## Directory Layout

| Directory | Purpose |
|---|---|
| `include/` | **Backend-agnostic API contract headers** (`aclnnop/`, `hccl/`). Any GPU backend implements this same set; clients depend only on it. |
| `cuda/` | **NVIDIA CUDA backend**: `src/*.cu` implementations, `Makefile`, build artifacts in `lib/`, CUDA-specific utilities in `tools/`. Implements the `include/` contract. |
| `tests/` | Backend-agnostic ACL client tests (tolerance comparisons). `make` (default `BACKEND=../cuda`; in the future `BACKEND=../rocm`). |
| `tools/` | Backend-agnostic client code and orchestration scripts: real model end-to-end (`qwen3/`), hybrid Qwen3-Next/"Qwen3.5" adaptation (`qwen35/`), model-graph container demo (`graph_demo`), decode-loop CUDA Graph (`decode_graph/`), Ascend golden comparison loop (`cannsim_golden*`), HCCL two-machine scripts, performance benchmarks (`bench`). |
| `third_party/` | External dependencies (CUTLASS headers). The optional real-Ascend golden cross-check (`tools/cannsim_golden.sh`) additionally needs the upstream `cann-api-explorer` cloned and `EXPLORER_DIR` set — not required for normal use, the default verification is the fast local torch oracle. |
| `deploy.sh` / `env.sh` | Environment setup and variable export. |

## Documentation

- **[`BENCHMARK.md`](BENCHMARK.md)** — Benchmark methodology and results, including a full set of **same-card A/B** optimization measurements on an RTX PRO 6000 Blackwell (`sm_120`): TF32 GEMM 2.6×, native MXFP8 1.38×, a built-from-scratch **native NVFP4 fp4 GEMM at 3.0× the functional path (1318 TFLOP/s)**, foreach multi-tensor fusion 43–70×, and **parity with native cuBLAS** (0.92–1.03×). Every speedup is optimized-vs-naive on the *same* card via env toggles — never a cross-card comparison.
- **[`cuda/README.md`](cuda/README.md)** — CUDA backend implementation highlights, performance optimizations, and design rationale.
- **[`cuda/TODO.md`](cuda/TODO.md)** — Remaining roadmap. Operator coverage is functionally complete (~98%) and the high-bandwidth-GPU performance work is done and measured (see `BENCHMARK.md`), so the file now tracks only what this box's hardware can't reach: real-Ascend golden cross-checks and bit-exact format fidelity, a full FlashAttention-3-grade kernel rewrite (needs Nsight Compute), and multi-card / multi-node scaling.
- **[`include/README.md`](include/README.md)** — Contents of the API contract headers and the "who implements / who consumes" split.
