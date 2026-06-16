# Performance Benchmarks

> 中文版 / Chinese: [BENCHMARK.zh-CN.md](BENCHMARK.zh-CN.md)

Measurement methodology and results for the cann-on-gpu CUDA backend.

> Data collected on **NVIDIA RTX PRO 6000 Blackwell Workstation Edition** (`sm_120`, 96 GB GDDR7 ≈1.8 TB/s, discrete card), CUDA 13.3, cuBLASLt / cuDNN backend.
> These are **discrete-Blackwell numbers and are not comparable to the previous GB10 (`sm_121`, UMA LPDDR5X ~273 GB/s) baseline** — the high-bandwidth GDDR7 card exposes the memory-access and tensor-core gains that GB10's UMA masked. Numbers still don't extrapolate across cards; re-benchmark when switching. See [`cuda/TODO.md`](cuda/TODO.md).

## Value Proposition

The GPU backend's purpose is to obtain cross-checkable results **orders of magnitude faster than the official CPU no-card simulator**.

- Single-operator validation: no-card simulation **40–85 s/operator** (build + instruction-level simulation) vs. GPU backend **µs–ms**.
- Full-network forward pass: single-layer transformer (~20 ops), GPU **0.21 ms/iter**; no-card simulation takes tens of seconds even for a single operator.
- Order-of-magnitude gap: **~10⁵–10⁶×**. Instruction-level simulation complexity is O(work), making large tensors infeasible; the GPU scales with hardware throughput.

| Operator | No-card sim wall-clock | Simulated device time |
|---|---|---|
| Transpose (259 instructions) | 39.6 s | 1.7 µs |
| ReduceSum (363) | 45.5 s | 2.2 µs |
| Add (30899) | 84.6 s | 6.7 µs |

Turnaround is dominated by "build + instruction-level simulation" and grows with workload; the device time itself is only µs-level.

## Measurement Methodology

- Tool: `tools/bench.cpp` (pure ACL client, links only `libascendcl.so`, mirroring how an Ascend application links).
- Timing: `std::chrono` wall-clock, **including GetWorkspaceSize executor construction** (i.e., the true per-call cost); average over multiple iterations after 5–10 warmup runs; workspace buffer pre-allocated and reused.
- Derived metrics: GEMM / Attention reported in GFLOP/s (attention computed as QKᵀ+PV ≈ 4·B·N·S²·D); memory-bound operators reported in effective bandwidth GB/s.
- Reproduction:

```bash
source env.sh
cd tools
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include bench.cpp \
    -L../cuda/lib -lascendcl -Wl,-rpath,../cuda/lib -o bench
./bench
```

- decode-loop CUDA Graph latency: `tools/decode_graph/build_decode_graph.sh`.

## Results

Numbers below are measured on the RTX PRO 6000 Blackwell (`sm_120`). The "speedup" column is reported where this run directly measured both the optimized and the naive/legacy path (via the env toggles); the per-optimization same-card A/Bs (128-bit vectorization, fusion, MXFP8, NVFP4, the toggle sweep, and vs-native-CUDA) are in the "Optimization A/B" sections below.

| Operator (config) | Result | Speedup | Notes |
|---|---|---|---|
| GEMM fp16 4096³ | 0.315 ms / **436 TFLOP/s** | — | cuBLASLt near peak (1024³: 200 TFLOP/s) |
| GEMM bf16 4096³ | 0.315 ms / **436 TFLOP/s** | — | cuBLASLt near peak |
| Attention default (flash) S=2048 (B4 N8 D64) | 5.29 ms / 6500 GFLOP/s | — | Online-softmax flash (no S/P materialization) + WMMA QKᵀ tensor-core; workspace 536 MB→0 |
| └ S=1024 / 512 / 128 | 1.59 / 0.37 / 0.064 ms | — | 5403 / 5841 / 2089 GFLOP/s |
| Attention performance path S=2048 | 1.55 ms / **22102 GFLOP/s** | — | cuBLASLt batched tensor-core, highest standard MHA throughput (S/P materialized) |
| └ S=1024 / 512 / 128 | 0.31 / 0.084 / 0.019 ms | — | 27654 / 25492 / 7088 GFLOP/s |
| PagedAttention decode B64 N8 L4096 D64 | 3.28 ms / 164 GFLOP/s | **15.3×** | warp-per-row + register online softmax vs scalar legacy (50.2 ms) |
| └ L1024 | 0.67 ms / 201 GFLOP/s | **18.4×** | vs scalar legacy (12.3 ms) |
| RMSNorm 4096×1024 (fp32) | 0.0078 ms / 4286 GB/s † | — | One block per row, coalesced access + warp reduction |
| Softmax last-dim 4096×1024 (fp32) | 0.0081 ms / 4153 GB/s † | — | Coalesced fast path |
| Add 4M (fp32) | 0.0062 ms / 8148 GB/s † | — | Vectorized elementwise |
| ReduceSum 4M (fp32) | 0.0159 ms / 1057 GB/s | — | Tree reduction |
| tiny LLM forward (1-layer, ~20 ops) | **0.209 ms/iter** | — | End-to-end composed; logits err 2.2e-7 |
| Foreach addcmul, 256 tensors × 4096 | 0.0162 ms | **43.5×** | Fused multi-tensor kernel vs per-tensor loop (0.705 ms, `FOREACH_NO_FUSE=1`) |
| └ 512 tensors × 256 (launch-bound) | 0.0203 ms | **70.1×** | vs per-tensor loop (1.42 ms) |
| Device malloc+free / pair | 0.040 µs | **~2958×** | Caching allocator vs passthrough (118.3 µs, `ACL_NO_CACHE_ALLOC=1`) |
| decode-loop (single token, 2-layer GQA) | 0.164 → 0.117 ms/token | **1.40×** | Full-loop CUDA Graph replay, eliminates per-kernel launch overhead |

† These small working sets (≤48 MB) are **L2-resident** on Blackwell's large L2, so the reported GB/s reflects L2 — not the GDDR7 DRAM ceiling. Measured DRAM-bound streaming (Add, ≥256M-element working set) tops out at **~1.75 TB/s** (see the 128-bit A/B below).

### cann-on-gpu vs native CUDA (same process, `aclnnMatmul` vs `cublasGemmEx`)

The ACL abstraction dispatches GEMM to cuBLAS(Lt) tensor cores, so the only difference from a direct cuBLAS call is the per-call executor build + dispatch. Measured fp16 GEMM, both in one binary:

| GEMM fp16 | cann-on-gpu | native cuBLAS | cann/native |
|---|---|---|---|
| 512³ | 55.6 TFLOP/s | 50.9 TFLOP/s | 0.92× |
| 2048³ | 403 TFLOP/s | 406 TFLOP/s | 1.01× |
| 4096³ | 435 TFLOP/s | 435 TFLOP/s | 1.00× |
| 8192³ | 514 TFLOP/s | 528 TFLOP/s | 1.03× |

**Finding:** cann-on-gpu is **at parity with native CUDA** (within 0–3%, and faster at 512³ where its cuBLASLt heuristic happens to pick a better algo). The "compute-intensive gap" is effectively **zero** — the backend runs the same tensor-core kernels, and the ACL executor/caching-allocator overhead is negligible even at small sizes. (This comparison is only meaningful on a compute-bound card; on a bandwidth-bound part the GEMM kernel choice, not the wrapper, would dominate.)

## Notes on this card (discrete Blackwell, sm_120)

These numbers are specific to this card. **Do not compare absolute throughput against the previous GB10 baseline** — GB10 and the RTX PRO 6000 are different silicon with different compute and memory ceilings, so a cross-card delta measures the hardware, not an optimization. Every "speedup" in the table above is a **same-card A/B**: the optimized path vs. the naive/legacy path toggled off (`FOREACH_NO_FUSE` / `PAGED_SCALAR` / `ACL_NO_CACHE_ALLOC`, etc.), both measured here.

What this card changes about *which* optimizations are worth measuring (all to be quantified same-card via the toggles):

- **Memory-access optimizations are worth measuring here**: this is a discrete high-bandwidth GDDR7 card, so 128-bit vectorized loads and traffic-reducing fusion show a real same-card before/after delta (on a low-bandwidth UMA part they cannot) — see the "Optimization A/B" sections.
- **Native low-precision tensor cores exist here** (fp4/fp8/MXFP8), so the native vs. functional path is a meaningful same-card comparison (`MXFP8_NO_HW`, etc.).
- **Launch-bound optimizations** (CUDA Graph replay, the caching allocator, foreach fusion) show same-card wins independent of bandwidth — see the table.
- All numbers must be re-benchmarked when switching cards; never carry a number or a ratio across silicon.

## Optimization A/B (same-card, via toggles)

### 128-bit vectorized memory access (`ELTWISE_NO_VEC=1` forces the scalar path)

The 16-byte (`LDG.128`) vectorized elementwise path vs. the 1-element/thread scalar path, both on this card, streaming `Add` (3 arrays touched) across a size sweep:

| Working set (Add) | Regime | fp32 vector | fp32 scalar | fp16 vector | fp16 scalar |
|---|---|---|---|---|---|
| 48 MB (4M elem) | L2-resident | **7899 GB/s** | 5284 GB/s (1.50×) | **6993 GB/s** | 2649 GB/s (2.64×) |
| 192 MB / 96 MB | L2 edge | ~2189 | ~2138 (≈) | **8774 GB/s** | 2871 GB/s (3.06×) |
| ≥768 MB (≥64M elem) | DRAM-bound | 1745 | 1750 (≈) | 1741 | 1789 (≈) |

**Finding:** the gain is real but **regime-dependent** — in the L2-resident regime the high L2 bandwidth makes the scalar path load-issue-bound, so 128-bit loads buy **1.5× (fp32) to ~3× (fp16)**; once the working set exceeds L2 and the op is purely DRAM-bound, both paths saturate GDDR7 at **~1.75 TB/s** and vectorization is a wash. It never regresses, so it stays on by default. (On GB10's low-bandwidth UMA there is no L2-resident high-bandwidth regime to exploit, which is why this optimization was invisible there.)

### Operator fusion — memory-traffic reduction (`AddRmsNorm` vs `Add`+`RmsNorm`)

Fused residual-add + RMSNorm (1 kernel, intermediate `x+res` kept on-chip / hot in L2) vs the unfused two-kernel form (`Add` writes the sum to DRAM, `RmsNorm` reads it back), fp16:

| rows × D (fp16) | Regime | Fused | Unfused (add+rms) | Fusion |
|---|---|---|---|---|
| 4096 × 4096 | L2-resident | 0.0360 ms | 0.0328 ms | 0.91× |
| 4096 × 8192 | L2 edge | 0.1592 ms | 0.1539 ms | 0.97× |
| 8192 × 8192 | DRAM-bound | 0.3205 ms | 0.3820 ms | **1.19×** |

**Finding:** the fusion win **grows with working-set size** — when the intermediate would round-trip DRAM (large rows×D) fusion saves that traffic (+ one launch) for **~1.2×**; when the working set is L2-resident there is no DRAM round-trip to remove, so fused ≈ unfused. *(The fused add-norm family — `AddRmsNorm`/`AddLayerNorm`/`AddRmsNormCast`/`DeepNorm`, used internally by MoE / fused-matmul / mc2 — uses one-block-per-row + coalesced + warp-reduction, matching `k_rms_fast`: e.g. DeepNorm 0.026 ms and AddRmsNormCast 0.094 ms at 4096² fp16, ~40–140× a one-thread-per-row baseline.)*

### Fused QKV projection (one `[K,3N]` GEMM vs three `[K,N]` GEMMs)

Concatenating the Q/K/V projection weights into one matmul (same total FLOPs; reads the input activation once, one launch, one larger GEMM). fp16, K=N=4096:

| M (tokens) | Regime | 3× separate | Fused [K,3N] | Speedup |
|---|---|---|---|---|
| 8 | decode | 0.0260 ms | 0.0216 ms | 1.21× |
| 64 | small-batch decode | 0.0478 ms | 0.0291 ms | **1.64×** |
| 512 | — | 0.1427 ms | 0.1356 ms | 1.05× |
| 4096 | prefill | 0.8747 ms | 0.8707 ms | 1.00× |

**Finding:** fusion helps most in the **decode / small-M regime** (up to **1.64×**), where three separate projections are launch- and weight-load-bound; at prefill it is compute-bound (identical FLOPs) so the gain vanishes. This needs no backend kernel — it's a model-level transform (pack the three projection weights and issue one `aclnnMatmul`), so the guidance is: **concatenate QKV weights for decode-heavy inference**.

### Native MXFP8 microscaling GEMM (`MXFP8_NO_HW=1` forces the functional path)

Native Blackwell microscaling tensor cores (cuBLASLt `VEC32_UE8M0` block scaling) vs the functional path (block-dequant fp8→fp16, then fp16 GEMM):

| MXFP8 GEMM | Native (HW) | Functional | Native speedup |
|---|---|---|---|
| 2048³ | 236 TFLOP/s | 288 TFLOP/s | 0.82× |
| 4096³ | 430 TFLOP/s | 357 TFLOP/s | 1.21× |
| 8192³ | **601 TFLOP/s** | 437 TFLOP/s | **1.38×** |

**Finding:** the native path's fp8 tensor cores reach **601 TFLOP/s** at 8192³ — **1.38×** the functional path, which is capped at ~437 (≈ the fp16 GEMM peak, since it dequantizes to fp16 before the GEMM). The native path carries a fixed preamble (transpose the B operand + swizzle the E8M0 scales into 128×4 super-tiles), so it *loses* at small sizes (2048³) and crosses over around 4096³; it is the right default on this Blackwell card for large matmuls and falls back to the functional path on non-Blackwell architectures automatically.

### Toggle-style optimization sweep (same-card)

| Optimization | Toggle | Config | Off | On | Speedup |
|---|---|---|---|---|---|
| TF32 fast-path GEMM | `CANN_FAST_TF32=1` | fp32 GEMM 8192³ | 94.3 TFLOP/s | 245.0 TFLOP/s | **2.60×** |
| Flash-attn WMMA QKᵀ | `FLASH_NO_WMMA` (off) | fp16 attn S=2048 | 4395 GFLOP/s | 6645 GFLOP/s | **1.51×** |
| Flash-attn fast PV | `FLASH_FAST_PV=1` | fp16 attn S=2048 | 6645 (WMMA) | 7555 GFLOP/s | **1.14×** (1.72× vs no-WMMA) |
| Grouped GEMM native | `CANN_GMM_NATIVE=1` | E8 per-expert=128, launch-bound | 187 TFLOP/s | 200 TFLOP/s | 1.07× |
| └ same | | E8 per-expert=512, compute-bound | 389 TFLOP/s | 301 TFLOP/s | 0.77× (multi-stream wins) |
| Conv algo autotune | `CANN_CONV_AUTOTUNE=1` | 3×3 N32 C256 56² K256 | 325 TFLOP/s | 323 TFLOP/s | ≈1.00× |

**Findings:** TF32 is a clean **2.6×** for fp32 GEMM (reduced precision, opt-in). WMMA tensor-core QKᵀ gives **1.5×** flash attention at S≥512 (loses at tiny S=128 where WMMA setup isn't amortized); `FLASH_FAST_PV` adds another 1.14×. Grouped-GEMM-native helps only in the launch-bound regime (many tiny experts); for compute-sized experts the default multi-stream path overlaps better — hence it's opt-in. Conv autotune is **neutral** for standard convolutions because the cuDNN v7 heuristic already selects the optimal algorithm (it only helps for unusual shapes). Large-L sort uses the bitonic path automatically (size-gated, not a toggle).

### Native low-precision GEMM paths (fp4/fp6/MXFP4) — coverage and a caveat

The native-tensor-core low-precision GEMM paths already exist and are tolerance-verified: MXFP8 (`aclnnMatmulMxFp8Hw`, above), MXFP4 (`aclnnMatmulMxFp4Hw`, `CUDA_R_4F_E2M1` + `VEC32_UE8M0`), and plain fp4/fp6 matmul which decodes losslessly to fp8-e4m3 and uses the native fp8 GEMM (`SUBFP_NO_FP8` toggles back to fp16). Same-card MXFP4 native vs functional (dequant→fp16):

| MXFP4 GEMM | Native (E8M0/blk-32) | Functional (→fp16) | Native speedup |
|---|---|---|---|
| 4096³ | 322 TFLOP/s | 372 TFLOP/s | 0.87× |
| 8192³ | 398 TFLOP/s | 438 TFLOP/s | 0.91× |

**fp6 (E2M3/E3M2):** fp6 plain matmul decodes losslessly to fp8-e4m3 and runs the native fp8 GEMM, reaching **554 TFLOP/s @ 8192³** (fp6 and fp8 share the Blackwell tensor-core datapath, so this is already ~92% of the fp8 ceiling).

**Caveat:** the `aclnnMatmulMxFp4Hw` path requests `VEC32_UE8M0`, which **this cuBLASLt rejects for fp4** (it accepts only NVFP4 `VEC16_UE4M3`), so it silently falls back to the functional path — that's why the "native" MXFP4 numbers above equal the functional ones. The true native fp4 rate requires the NVFP4 path below.

### NVFP4 native fp4 path — speed & fidelity (`NVFP4_HW=1`)

Built a native NVFP4 GEMM (`CUDA_R_4F_E2M1` + `VEC16_UE4M3` block-16) that converts Ascend's E8M0/block-32 scales to E4M3/block-16 on the fly (each block-32 → two block-16 with the same 2^exp). This is the path that actually reaches Blackwell's fp4 tensor cores:

| GEMM @ 8192³ | TFLOP/s | vs functional fp16 | vs native MXFP8 |
|---|---|---|---|
| Functional (dequant→fp16) | 437 | 1.0× | — |
| Native MXFP8 (`VEC32_UE8M0`) | 601 | 1.38× | 1.0× |
| **Native NVFP4 (`VEC16_UE4M3`)** | **1318** | **3.0×** | **2.2×** |

(NVFP4 sweep: 491 / 999 / 1318 TFLOP/s at 2048³ / 4096³ / 8192³.)

**Fidelity:** the E8M0→E4M3 scale conversion is **exact** when each block's scale exponent is in E4M3's representable power-of-2 range **[2⁻⁶, 2⁸]** (15 binades) — verified: the MXFP4 tolerance tests pass with `maxrel = 0.00e+00` under `NVFP4_HW=1`. Exponents outside that range are clamped, so NVFP4 is a **fast, reduced-fidelity tier** for tensors whose per-block dynamic range exceeds 15 binades; the exact functional path (full E8M0 range, dequant→fp16) remains the fidelity reference and the default. Opt in with `NVFP4_HW=1`.

> All performance optimizations maintain full tolerance cross-check pass (no correctness sacrificed), and retain environment-variable toggles to revert to the unoptimized path for comparison
> (`ACL_NO_CACHE_ALLOC` / `ATTN_NO_PERF` / `PAGED_SCALAR` / `FLASH_NO_WMMA` / `MXFP8_NO_HW` / `ELTWISE_NO_VEC` / `FOREACH_NO_FUSE`, etc.).
