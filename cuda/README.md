# cuda/ — NVIDIA CUDA Backend for CANN / AscendCL APIs

> 中文版 / Chinese: [README.zh-CN.md](README.zh-CN.md)

This directory re-implements the CANN/AscendCL/aclnn/HCCL API contracts declared in [`../include/`](../include/) using the CUDA ecosystem on NVIDIA GPUs.
It produces `lib/libascendcl.so` + `lib/libhccl.so`; applications link against them exactly as they would against the genuine Ascend libraries — only the API surface is visible, not CUDA.

## Directory Layout

- `src/` — Implementations. `runtime.cu` (`aclrt*` device/stream/event/memory), `meta.cu` (aclTensor / aclOpExecutor / aclTensorList),
  `aclop.cu` (legacy single-operator interface routing to aclnn), `hccl.cu` (HCCL→NCCL), `ops/*.cu` (per-operator families), `internal.h` (internal structures),
  low-precision codecs `ops/{hif8.cuh,hif8_table.inc,subfp.cuh}`.
- `Makefile` — `nvcc -arch=$(GPU_ARCH)`, `-I../include` (shared contract) + `$ACL_INCLUDE` (acl/aclnn meta-contract headers) + cuDNN/NCCL. `cudart` statically linked.
- `lib/` — Build artifacts.
- `tools/` — CUDA-specific utilities: `nccl_allreduce.cu` (bare NCCL), `ptx_module_demo` (CUDA Driver/PTX/nvrtc), and accompanying scripts.

## Build

```bash
source ../env.sh
make                       # → lib/libascendcl.so + lib/libhccl.so
make GPU_ARCH=sm_90        # recompile for a different NVIDIA architecture (see "Cross-Architecture Portability" below)
```

---

## Design Highlights

### API-Layer Interception (Ascend Binaries Never Touched)

When an application calls `aclrtMalloc` / `aclnnXxx` / `HcclXxx`, the link boundary resolves to this library's CUDA implementation; Ascend vendor operator binaries are never loaded.
Only the **mathematical semantics of each operator** need to be replicated — not how Ascend computes them internally. Linking any Ascend runtime at run time is prohibited (symbol conflicts);
the only compile-time dependency on the Ascend side is the API **headers** (the contract).

### aclnn Two-Stage Executor

aclnn operators follow a two-stage protocol: `aclnnXxxGetWorkspaceSize(...)` first constructs an `aclOpExecutor` on the host (recording tensors, shapes, axes, attributes, etc.,
**without allocating device memory**) and returns the required workspace size; `aclnnXxx(workspace, ..., executor, stream)` then executes using caller-preallocated workspace and destroys the executor.
This backend faithfully implements the contract — which also makes the entire operator chain capturable into a CUDA Graph (GetWorkspaceSize is host-only and capture-safe; the execution stage reuses preallocated workspace and does not call `cudaMalloc` during capture).

### Operator Backend Mapping and Rationale

| Operator Family | Implementation | Rationale |
|---|---|---|
| Elementwise / activation / reduction / reshape / comparison / sort | Hand-written CUDA kernels | Simple semantics, must support broadcast + mixed dtype + non-contiguous views (addressed via real stride/offset); hand-written is most flexible; reductions use block/warp reduction |
| matmul / GEMM | **cuBLASLt** | Vendor-tuned, near-peak performance; row-major interfaced via column-major mapping; fp32 with fp32 accumulation, fp16/bf16 with fp32 accumulation (consistent with Ascend Cube) |
| Convolution / pooling / BatchNorm | **cuDNN** | Convolution algorithm selection and tensor-core utilization delegated to cuDNN; forward/backward/3D/transposed/pooling all covered |
| Attention | Hand-written naive + hand-written flash + cuBLASLt batched | See "Attention" below |
| Communication HCCL | **NCCL** | Collective communication semantics 1:1; the challenge is the control plane (ranktable/uniqueId translation) |
| Low-precision fp8/fp4/fp6/HiF8/MX* | Decode to fp16 (or fp8) then compute | See "Low-Precision Fidelity" below |

### Numerical Precision Strategy (Aligned with Ascend, Not Bit-Exact)

- **Always accumulate in fp32**: GEMM uses `CUBLAS_COMPUTE_32F`; hand-written reductions (RMSNorm/softmax/LayerNorm) accumulate internally in `float`/`double` and cast on write-back.
  This is consistent with Ascend's (and PyTorch's) bf16/fp16 semantics — **tensors stored at low precision, operators accumulate internally in fp32**.
- **bf16 full operator coverage**: matmul/MatmulBias/BatchMatMul/GroupedMatmul/full convolution family/LayerNorm/RmsNorm/RoPE/FlashAttention/PagedAttention all support bf16.
  When running a real model in pure bf16, outputs match HuggingFace bf16 outputs token-for-token (validating bf16 semantic fidelity).
- **fp16 multiplication verification** uses atol+rtol (subnormal-region relative errors diverge); **GEMM verification** uses normalized error `max|err|/max|ref|` (avoids spurious relative-error inflation from near-zero reference elements).

### Low-Precision Fidelity

fp8 (e4m3/e5m2), fp4 (e2m1/e1m2), fp6 (e2m3/e3m2), HiFloat8, MXFP8, MXFP4: the default path is **software decode to fp16/fp8 followed by GEMM**.
This is not a performance compromise — it is the **necessary approach for Ascend fidelity**: Ascend's encoding/rounding boundaries for these formats differ from NVIDIA's native low-precision formats.
The data must first be decoded according to Ascend semantics (boundary tables) before computation; feeding NVIDIA's native fp4/fp8 tensor cores directly would produce NVIDIA numerical results, validating the wrong target.
Low-precision codec golden values are generated offline from the CANN reference codec and verified with pure bitwise operations.
The Blackwell MXFP8 native microscaling path (`aclnnMatmulMxFp8Hw`, VEC32_UE8M0 + 128×4 super-tile swizzle) is implemented and automatically falls back to the functional path at run time on non-Blackwell architectures.
An **optional native NVFP4 fp4 path** (`NVFP4_HW=1`) converts Ascend's E8M0/block-32 scale to E4M3/block-16 on the fly to reach Blackwell's fp4 tensor cores (`VEC16_UE4M3`) — measured at **3.0× the functional path** (see `BENCHMARK.md`). It is exact while each block's scale exponent stays in E4M3's range [2⁻⁶, 2⁸] and clamps outside, so it is an opt-in fast/reduced-fidelity tier; the functional decode→fp16 path remains the default and the fidelity reference.

---

## Performance Optimizations

All optimizations keep tolerance-based comparison tests fully passing and retain environment-variable switches for fall-back comparisons. Performance numbers are in [`BENCHMARK.md`](BENCHMARK.md).

- **True flash attention (online softmax)**: warp-per-row, each lane holds D/32 head-dimension components, dot products reduced via `__shfl` (no `__syncthreads`),
  running max/sum/acc entirely in registers → **no S/P materialization** (workspace zeroed; large S no longer causes O(S²) memory explosion). Covers GQA/MQA/causal/mask/fp16/bf16/fp4.
  For fp16 + standard MHA, QKᵀ uses WMMA tensor-core (fp32 accumulation); PV remains fp32 scalar (fp16-quantized P for PV introduces too much error).
- **Caching device-memory allocator**: `aclrtMalloc/Free` uses bucketed reuse; a single malloc+free round-trip drops from hundreds of µs (direct cudaMalloc) to sub-µs — the foundation of the inference backend.
- **PagedAttention warp-per-row**: register array `acc[D/32]` replaces local-memory `acc[256]` + coalesced KV access, achieving ~bandwidth-bound throughput.
- **Weight-quantized small-M fused GEMM**: W8A16/W4A16 at small M (decode/small batch) reads int8/int4 weights directly into a fused GEMM without materializing fp16, saving 4×/8× weight bandwidth; large M (prefill) still dequantizes + cuBLASLt tensor-core.
- **Coalesced-access normalization**: RMSNorm/Softmax/GroupNorm and the fused add-norm family (`AddRmsNorm`/`AddLayerNorm`/`AddRmsNormCast`/`DeepNorm`, used by MoE/fused-matmul/mc2) all use "one block per row + warp reduction" for coalesced access.
- **Foreach multi-tensor fusion**: the arithmetic `aclnnForeach*` family fuses the whole tensor list into a single grid-strided kernel (per-tensor metadata staged in the caller workspace, `grid.y` = tensor index, à la PyTorch `MultiTensorApply`) instead of one launch per tensor — **43–70×** on optimizer-state-style lists of many small tensors (`FOREACH_NO_FUSE=1` reverts).
- **W8A8 native int8 GEMM**: cuBLASLt `CUDA_R_8I` + int32 accumulation + per-channel scale epilogue; falls back to fp16 on failure.
- **MoE grouped GEMM**: variable-length groups overlapped across multiple streams, or native grouped GEMM (switchable).
- **CUDA Graph**: single-token decode steps captured into a graph and replayed in a full loop, eliminating per-kernel launch overhead (decode is launch-bound).
- **128-bit vectorized memory access**: elementwise 16B vectorized fast path + scalar fallback (`ELTWISE_NO_VEC=1` reverts). Same-card win is regime-dependent — 1.5–3× when L2-resident, a wash once DRAM-bandwidth-bound (see `BENCHMARK.md`).

---

## Cross-NVIDIA-Architecture Portability

The source code is generic CUDA (no `__CUDA_ARCH__` branches, no device-property gating); architecture binding comes **only from the build flag** `-arch=$(GPU_ARCH)`.
Switching architectures only requires changing `GPU_ARCH` and recompiling — no source changes:

| Target | `GPU_ARCH` | Low-Precision |
|---|---|---|
| GB10 (default) | `sm_121` | Full, including native MXFP8 microscaling |
| RTX 50 (GB202) | `sm_120` | Full, including native MXFP8 |
| B200 / GB200 | `sm_100` | Full, including native MXFP8 |
| Hopper H100/H200 | `sm_90` | fp8/fp4/fp6/MX* all supported (MX via functional path) |
| Ada / RTX 40 | `sm_89` | Same as above |

- **fp4/fp6 requires no hardware fp4**: lookup-table decode to fp16 then compute; runs on any architecture with fp16 (and is the correct approach for Ascend fidelity).
- Dynamic dependencies on cuBLASLt/cuDNN/NCCL require versions corresponding to the target architecture.

## Boundaries (Structurally Out of Scope for the A-Path)

APIs that are non-computational in semantics or that are tightly coupled to Ascend-proprietary formats/hardware are out of scope for this backend: GE graph engine / Graph-mode compilation, `aclmdl` (`.om` offline models),
AOL precompiled operator black-boxes, ATB, DVPP/HIXL/SiP dedicated hardware units, `aclprof*`/`acldump*` debug collection, and bit-exact numerical equivalence.
For the "load model → execute" control flow, **equivalent capability** is provided: `../tools/cann_graph.h` — a minimal operator-graph IR + serialization + sequential executor (a non-Ascend-`.om` equivalent container).
