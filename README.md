# cann-on-gpu

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

Re-implements the Huawei Ascend **CANN / AscendCL** API surface (`acl*` runtime, `aclnn*` operators, `Hccl*` collectives) on **commodity GPUs**,
so that programs written against the Ascend API can be compiled, run, and validated on ordinary GPUs **without any code changes**.
The API contract is decoupled from the chip: the same application binary runs by swapping in a different GPU backend. **Two backends ship today** — **NVIDIA CUDA** (`cuda/`) and **Apple Metal** (`metal/`) — both implementing the single contract in `include/`.

## What Problem This Solves

- **Develop and validate CANN operators without an Ascend card.** Code written against the Ascend API links directly against this project's `libascendcl` and runs on an ordinary GPU (NVIDIA or Apple).
  Compared with the official card-free CPU simulation (instruction-level, minutes per operator, O(work) growth with tensor size, infeasible for large tensors), the GPU path delivers comparable results **orders of magnitude faster**
  and scales with hardware throughput — compressing the turnaround for "validate one Ascend operator / one network" from minutes down to milliseconds.
- **Executable semantic reference and regression baseline.** A readable, open-source operator library implemented to CANN's mathematical semantics serves as "executable documentation" for Ascend operator semantics.
  It can also be tolerance-compared against **real Ascend golden outputs** (produced via card-free simulation), bringing Ascend operator functional correctness into **ordinary GPU CI** without depending on scarce, expensive Ascend hardware.
- **Cross-architecture AI-infrastructure portability.** By separating the "operator/collective API contract" (`include/`) from the "specific chip" (`cuda/`, `metal/`), the project demonstrates that AI infrastructure can be **vendor-portable**:
  the same contract is satisfied today on an NVIDIA discrete/UMA GPU and on an Apple-silicon GPU, with room to extend to other GPUs (e.g. AMD ROCm, Intel SYCL).
- **A non-toy inference backend prototype.** Already capable of end-to-end loading of a real open-source model (Qwen3-0.6B), running a forward pass and greedy decoding, reproducing HuggingFace reference output token by token — on **either backend**.
  The hybrid **Qwen3-Next ("Qwen3.5")** architecture — GatedDeltaNet linear-attention / SSM layers, gated full-attention, and per-layer sparse MoE — also runs end-to-end and matches HuggingFace logits **on both backends** (`tools/qwen35/`).

> Scope: **functional / tolerance-level validation** and cross-architecture backend exploration. This project does **not** pursue bit-exact numerical fidelity with Ascend, nor does it benchmark against Ascend hardware performance.

## Current Status

The two backends share the same contract, the same backend-agnostic tests in `tests/`, and the same client tools in `tools/`.

### CUDA backend (NVIDIA) — broad coverage, tolerance-verified

**~1014 operators implemented and tolerance-verified** (≈98% of the full aclnn surface; see "Operator Coverage Baseline"). Highlights:

- **Operator coverage**: full elementwise family (broadcasting / mixed dtype / non-contiguous views), arbitrary-dim reduction, activation and elementary functions, reshape and indexing, comparison and logic, sort and scan;
  matmul (batched / transpose / bias fusion / quantized W4A16·W8A16·W8A8 / low-precision fp8·fp4·fp6·MXFP8·MXFP4 / grouped MoE),
  convolution (2D/3D forward and backward, transposed, pooling incl. 3D and adaptive), attention (naive + flash + cuBLASLt performance variant, GQA/MQA, mask, KV-cache Paged/Incre/Prompt, RoPE),
  normalization (Softmax/LayerNorm/RMSNorm/GroupNorm/BatchNorm/InstanceNorm/LRN/RmsNormGated and fused AddRmsNorm·SwiGlu·GeGlu), Embedding, loss, optimizer, RNG family, explicit quantization, low-precision encode/decode, fractal format conversion,
  linear algebra (inverse/solve/det/cholesky/qr/svd/eigh/lu/pinverse/matrix-exp via cuSOLVER), FFT/RFFT (cuFFT), RNN (LSTM/GRU), linear-attention / SSM (causal-conv1d, Mamba selective-scan, gated-delta-rule), and MoE routing + multi-latent attention + RoPE backward.
- **Distributed fused collectives, verified on a real 2-node RoCE/HCCL link** (dual-NIC 200G): the `mc2` family — `MatmulAllReduce`/`MatmulReduceScatter`/`MatmulAllGather`, `QuantMatmulAllReduce`, `MatmulAllReduceAddRmsNorm`, and MoE expert-parallel `MoeDistributeDispatch`/`Combine` — all on the non-degenerate multi-rank path.
- **Validation**: single-process tolerance tests **all green (742 checks across 31 test binaries)** plus the 2-node HCCL/mc2 suite; **74 operators tolerance-matched against real Ascend golden outputs**; Qwen3-0.6B end-to-end vs HF (prefill logits ~5.4e-6, greedy 8/8). Same-card performance optimizations are measured in [`cuda/BENCHMARK.md`](cuda/BENCHMARK.md).

### Metal backend (Apple) — full operator parity

**Full operator parity with the CUDA backend** — every `aclnn*` operator the CUDA backend exports is implemented on Metal (verified by comparing exported symbols on both shared libraries).

- **Implementation**: host layer in Objective-C++ driving MPS/MPSGraph/Accelerate; hand-written MSL kernels for the elementwise/softmax/norm/sort families; **unified memory** makes the `aclTensor` device pointer host-addressable, so H2D/D2H is `memcpy` and the long-tail families run host-side without copies. `Hccl*` is a single-rank stub (one Mac = one device).
- **Validation**: all backend-agnostic test binaries pass (`make -C tests BACKEND=../metal run`), plus per-family parity tests; Qwen3-0.6B (fp32) end-to-end matches HF — prefill logits **3.82e-6**, greedy **8/8 identical tokens** (same as CUDA). Correctness is additionally cross-checked by **differential testing (对拍) against the CUDA backend** and the independent PyTorch oracle.

See the per-backend documents under "Documentation" below.

## Operator Coverage Baseline

The authoritative operator set is the **open-source CANN operator libraries** on gitcode — [`cann/opbase`](https://gitcode.com/cann/opbase), [`cann/ops-math`](https://gitcode.com/cann/ops-math), [`cann/ops-transformer`](https://gitcode.com/cann/ops-transformer), [`cann/ops-cv`](https://gitcode.com/cann/ops-cv), [`cann/ops-nn`](https://gitcode.com/cann/ops-nn) — pinned to the **`9.1.0-beta.1`** branch (matching the CANN 9.1 toolkit). These libraries target Atlas A2/A3 **and Ascend 950PR/950DT through the CANN Simulator**; the **950** interface is this project's baseline.

- **Authoritative list**: **1036** `aclnn*` level-2 APIs (`tools/official_aclnn.txt`), cross-checked against the A3 `ops` run package (97.4% identical — confirming the aclnn host API is uniform across SoCs).
- **CUDA coverage**: **1014 / 1036 ≈ 98%** implemented and tolerance-verified. The remaining **22** (`tools/gap.txt`) are Ascend device/codec/debug infrastructure ops with no GPU-compute meaning (tracked as permanent out-of-scope in `cuda/TODO.md`).
- **Metal coverage**: full parity with the CUDA backend's exported operator set.
- **dtype fidelity**: every operator's official dtype/shape matrix is recoverable from each library's `tests/st/aclnn*/all_*.json` system-test cases (`tools/op_map.tsv` maps each API → library/category), used to ensure our implementations accept every dtype the original supports.

## Requirements and Installation

`deploy.sh` auto-selects the backend by host OS (override with `BACKEND=cuda|metal`). It is idempotent, installs into a dedicated conda/miniforge env, and does not modify global PATH.

**NVIDIA / CUDA (Linux):** one NVIDIA GPU (Ampere/Ada/Hopper/Blackwell; default build target `sm_121` = GB10 — change `GPU_ARCH` and recompile for another card). NVIDIA driver + CUDA Toolkit 13; the backend depends on cuDNN / NCCL / CUTLASS.

```bash
./deploy.sh            # create env, install cuDNN/NCCL/CUTLASS, fetch ACL headers, generate env.sh
source env.sh          # export GPU_ARCH / ACL_INCLUDE / CUDNN_DIR / NCCL_DIR etc.
make -C cuda           # → cuda/lib/libascendcl.so + libhccl.so
make -C tests run      # run all tolerance tests (default BACKEND=../cuda)
```

**Apple / Metal (macOS):** an Apple-silicon Mac with full Xcode + the downloadable Metal Toolchain. `deploy.sh` provisions the Metal path (forge env, project-scoped Xcode via `DEVELOPER_DIR` — **no global `xcode-select` change**).

```bash
./deploy.sh                          # macOS → Metal path (forge env + Xcode/Metal Toolchain checks)
source env.sh                        # BACKEND=metal; scopes Xcode via DEVELOPER_DIR
make -C metal                        # → metal/lib/libascendcl.dylib + libhccl.dylib + default.metallib
make -C tests BACKEND=../metal run   # run all tolerance tests against the Metal backend
```

Both paths share the ACL/aclnn/HCCL **headers only** (compile-time contract; no Ascend library is linked at runtime).

## Directory Layout

| Directory | Purpose |
|---|---|
| `include/` | **Backend-agnostic API contract headers** (`aclnnop/`, `hccl/`). Every GPU backend implements this same set; clients depend only on it. |
| `cuda/` | **NVIDIA CUDA backend**: `src/*.cu`, `Makefile`, artifacts in `lib/`, CUDA-specific utilities + benchmark docs. Implements the `include/` contract. |
| `metal/` | **Apple Metal backend**: `src/*.{mm,metal}`, `Makefile`, artifacts in `lib/`, backend docs. Implements the same `include/` contract on Apple-silicon GPUs. |
| `tests/` | Backend-agnostic ACL client tests (tolerance comparisons). `make` (default `BACKEND=../cuda`; `BACKEND=../metal` for Apple; in the future `BACKEND=../rocm`). |
| `tools/` | Backend-agnostic client code and orchestration: real model end-to-end (`qwen3/`), hybrid Qwen3-Next/"Qwen3.5" (`qwen35/`), model-graph container demo, decode-loop graph, Ascend golden comparison loop (`cannsim_golden*`), PyTorch differential oracle (`torch_golden.sh` / `torch_check.sh`), cross-backend 对拍 (`cann_metal_diff.sh`), HCCL multi-node scripts, performance benchmark (`bench`). |
| `third_party/` | External dependencies (CUTLASS headers for the CUDA backend). The optional real-Ascend golden cross-check additionally needs the upstream `cann-api-explorer` — not required for normal use; the default verification is the fast local torch oracle. |
| `deploy.sh` / `env.sh` | Environment setup and variable export (OS-selected backend). |

## Documentation

- **[`cuda/README.md`](cuda/README.md)** — CUDA backend implementation highlights, performance optimizations, and design rationale.
- **[`cuda/BENCHMARK.md`](cuda/BENCHMARK.md)** — CUDA benchmark methodology and same-card optimization measurements.
- **[`cuda/TODO.md`](cuda/TODO.md)** — CUDA backend remaining roadmap (hardware-bound items + scaling).
- **[`metal/README.md`](metal/README.md)** — Metal backend overview, operator-backend mapping, and numerical strategy.
- **[`metal/BENCHMARK.md`](metal/BENCHMARK.md)** — Metal benchmark methodology, M-series results, and a cross-backend comparison.
- **[`metal/TODO.md`](metal/TODO.md)** — Metal backend remaining / optimizable roadmap.
- **[`include/README.md`](include/README.md)** — Contents of the API contract headers and the "who implements / who consumes" split.
