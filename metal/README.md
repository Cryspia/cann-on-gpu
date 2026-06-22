# metal/ — Apple Metal Backend for CANN / AscendCL APIs

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

This directory re-implements the CANN / AscendCL / aclnn / HCCL API contracts declared in [`../include/`](../include/) using the Apple Metal ecosystem on Apple-silicon GPUs.
It produces `lib/libascendcl.dylib` + `lib/libhccl.dylib` (+ `lib/default.metallib`); applications link against them exactly as they would against the genuine Ascend libraries — only the API surface is visible, not Metal. The backend-agnostic clients in `../tests/` and `../tools/` link it without source changes.

## Directory Layout

- `src/` — Implementations. `runtime.mm` (`aclrt*` device/stream/event/memory), `meta.mm` (aclTensor / aclScalar / aclOpExecutor / aclTensorList),
  `aclop.mm` (legacy single-operator interface routing to aclnn), `hccl.mm` (single-rank collective stub), `ops/*.{mm,metal}` (per-operator families),
  `internal.h` (internal structures + pipeline cache + metallib loader), low-precision codecs `ops/{subfp.h,hif8_table.inc}`.
- `Makefile` — `clang++ -ObjC++` for the `.mm` host layer, the Metal Toolchain compiler for `.metal` → one `default.metallib`. `-I../include` (shared contract) + `$ACL_INCLUDE` (acl/aclnn meta-contract headers).
- `lib/` — Build artifacts (`libascendcl.dylib`, `libhccl.dylib`, `default.metallib`).

## Build

```bash
source ../env.sh                  # BACKEND=metal; scopes Xcode via DEVELOPER_DIR (no global xcode-select change)
make                              # → lib/libascendcl.dylib + lib/libhccl.dylib + lib/default.metallib
make -C ../tests BACKEND=../metal run
```

---

## Design Highlights

### API-Layer Interception (Ascend Binaries Never Touched)

When an application calls `aclrtMalloc` / `aclnnXxx` / `HcclXxx`, the link boundary resolves to this library's Metal implementation; Ascend vendor operator binaries are never loaded.
Only the **mathematical semantics of each operator** are replicated — not how Ascend computes them internally. The only compile-time dependency on the Ascend side is the API **headers** (the contract).

### Host = Objective-C++, kernels = MSL

The host/dispatch layer is Objective-C++ (`.mm`) so it can drive MPS / MPSGraph / Accelerate (Objective-C-only frameworks); compute kernels are hand-written MSL (`.metal`) compiled into one `default.metallib` loaded at `aclInit`. Kernel dtype/op specialization uses MSL templates plus a `MTLComputePipelineState` cache keyed by function name.

### aclnn Two-Stage Executor

aclnn operators follow a two-stage protocol: `aclnnXxxGetWorkspaceSize(...)` first constructs an `aclOpExecutor` on the host (recording tensors, shapes, axes, attributes) and returns the required workspace size; `aclnnXxx(workspace, ..., executor, stream)` then executes and destroys the executor. This backend implements the contract identically to the CUDA backend, so the `aclTensor` / `aclScalar` / `aclOpExecutor` plans port unchanged — only the Execute body differs.

### Unified Memory

With `MTLStorageModeShared`, an `MTLBuffer`'s `.contents` pointer is host-addressable and identical device-side, so the `aclTensor` device pointer is directly host-usable and `aclrtMemcpy` (host↔device) degenerates to `memcpy`. Host-side tolerance comparison needs no extra copies, and host-side operator implementations read/write device tensors directly.

### Operator Backend Mapping

| Operator Family | Implementation | Rationale |
|---|---|---|
| Elementwise / activation / reduction / softmax / norm / compare / cast / sort | Hand-written **MSL** kernels | Simple semantics with broadcast + mixed dtype + non-contiguous views; specialized via MSL templates + pipeline cache |
| matmul / GEMM / BatchMatMul | **MPS** | Vendor-tuned matmul; fp32/fp16 native, **bf16 via lossless fp32 widen**, **sub-byte fp8/fp4/fp6/HiF8 via lossless fp16 widen**, **W8A8 int8 via native MPS Int8 GEMM**. fp32 accumulation throughout; host-GEMM fallback only for fused activation / 2-D bias / mixed-precision |
| Convolution / pooling | On GPU: conv2d/3d forward (im2col + MPS GEMM), transposed & backward conv + pooling 2D/3D/adaptive/backward (MSL gather kernels). ConvTbc / deformable conv host-side | Full forward/backward/3D/transposed/adaptive coverage |
| Attention | Batched **MPS** QKᵀ/PV + MSL softmax (size-gated) + per-thread MSL kernel / host fallback | See "GPU fast paths" below |
| Dense linear algebra | **Accelerate / LAPACK** (CPU over unified memory) | inverse/solve/det/cholesky/qr/svd/eigh/lu; cheap because there is no device copy |
| FFT | **MPSGraph / vDSP** | RFFT/FFT |
| Low-precision fp8/fp4/fp6/HiF8/MX* (sub-byte) | Host decode (Ascend boundary tables); **matmul → native MPS fp16**, other ops host-side | No native Apple-GPU sub-byte type, but every code is exact in fp16 so GEMM runs native; decode stays host for Ascend fidelity (see "Low-Precision Fidelity") |
| Communication `Hccl*` | Single-rank identity stub | One Mac = one device |
| dtypes | `half` native; `bfloat` native (Metal 3.1+ on M-series); sub-byte via bit-twiddle codecs | Ported from the CUDA backend's `subfp`/`hif8` logic |

### Numerical Precision Strategy (Aligned with Ascend, Not Bit-Exact)

- **Always accumulate in fp32**: matmul uses fp32 accumulation; hand-written reductions (RMSNorm / softmax / LayerNorm) accumulate internally in `float`/`double` and cast on write-back — consistent with Ascend's (and PyTorch's) bf16/fp16 semantics.
- **bf16 matmul**: `MPSMatrixMultiplication` has no bf16 type, so the backend widens bf16→fp32 (lossless — bf16 is a truncated fp32) with MSL cast kernels, runs the native MPS fp32 GEMM, and narrows the result back to bf16; fp32 accumulation matches the reference. The attention kernel also has a native bf16 (`bfloat`) MSL path.
- **Accurate special functions to ~1e-6**: the hand-written kernels are compiled `-fno-fast-math`, with Kahan-summed series `erf`, continued-fraction `erfc`, `expm1`-based selu/elu/celu, `gelu` via `erfc`, and a relatively-accurate `lgamma` — ported from the CUDA functors.

### GPU Fast Paths

Hot paths run on the GPU; everything else runs host-side over unified memory (correct, and cheap because there is no copy):
- **Elementwise / softmax / norm**: hand-written MSL kernels (coalesced, one threadgroup per row for reductions).
- **matmul / BatchMatMul**: MPS.
- **Convolution / pooling**: conv2d/3d forward via im2col (host gather) + MPS fp32 GEMM; transposed & backward conv and pooling (2D/3D, adaptive, backward) via MSL gather kernels (one thread per output, inverse index map, no atomics). Host fallback retained.
- **Attention**: QKᵀ and PV run as batched **MPS** GEMM with an MSL masked-softmax between (fp32-internal, so fp16/bf16 widen for accurate scores; GQA/MQA via a head-replicating gather; causal/mask). Dispatched by a **size gate**; small/decode/exotic shapes fall back to a per-thread online-softmax MSL kernel, then the host path. The fallback is always available, so correctness never depends on the gate.

### Low-Precision Fidelity

**W8A8 int8 matmul runs natively** on `MPSMatrixMultiplication`'s Int8 path (int8·int8 → fp32 raw sums, then the per-channel dequant scale) — exact while the per-output sum stays in fp32's 2²⁴ integer range, which covers typical W8A8 (larger K rounds within quant tolerance), with an exact host fallback.

**Sub-byte formats** — fp8 (e4m3/e5m2), fp4 (e2m1/e1m2), fp6 (e2m3/e3m2), HiFloat8, MXFP8, MXFP4 — have no native Apple-GPU type, so they are **decoded host-side by Ascend's boundary tables**. This decode is the **necessary approach for Ascend fidelity**: Ascend's encoding/rounding boundaries differ from any native low-precision format, so feeding a native low-precision unit (even if one existed) would validate the wrong numerical target. **For matmul, every decoded value is exact in fp16 (magnitude ≤ 32768, mantissa ≤ 3 bits ≤ fp16's 10), so the decode runs as an MSL kernel on the GPU and the GEMM runs on the native MPS fp16 path** (fp32 accumulation); the decode table is built from the same boundary-table codec, so it stays bit-exact. Other operator families compute host-side over unified memory. Codec golden values are reproduced with pure bitwise operations.

---

## What's Implemented

- **Full operator parity with the CUDA backend** — every `aclnn*` operator the CUDA backend exports is implemented here (verified by comparing exported symbols on both shared libraries).
- **All backend-agnostic test binaries pass** against `libascendcl.dylib` (`make -C ../tests BACKEND=../metal run`), plus per-family parity tests — the same bar the CUDA backend meets.
- **Real model end-to-end**: Qwen3-0.6B (fp32) forward + greedy decode matches HuggingFace — prefill logits L2 **3.82e-6**, greedy **8/8 identical tokens** (the same figure as the CUDA backend). The hybrid **Qwen3-Next ("Qwen3.5")** forward (GatedDeltaNet linear-attention + gated full-attention + per-layer sparse MoE) also matches HuggingFace — per-stage/per-layer ~1e-6, logits **4.34e-6**, greedy next-token identical (`../tools/qwen35/`).
- **Correctness** is checked by the repo's backend-agnostic tolerance tests and by **differential testing (对拍) against the CUDA backend** on an NVIDIA GB10 reference host, plus the independent PyTorch oracle (`../tools/torch_golden.sh`, `../tools/torch_check.sh`).

## Boundaries (Out of Scope on a Single Mac)

Multi-device collectives (one Mac = one device; HCCL is a single-rank stub), real-Ascend golden / bit-exact NZ-fractal fidelity, and a FlashAttention-3-grade kernel are out of scope here — see [`TODO.md`](TODO.md). This mirrors the project scope: **functional / tolerance-level** validation and cross-architecture backend exploration, not bit-exact Ascend fidelity or a performance contest against CUDA.

## Documentation

- **[`TODO.md`](TODO.md)** — Remaining / optimizable work (the open roadmap; completed work is not tracked here).
- **[`BENCHMARK.md`](BENCHMARK.md)** — Benchmark methodology, M-series results, and a cross-backend comparison against the CUDA backend.
- **[`../include/README.md`](../include/README.md)** — The API contract headers and the "who implements / who consumes" split.
