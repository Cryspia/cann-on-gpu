# Performance Benchmarks — Metal Backend

> 中文版 / Chinese: [BENCHMARK.zh-CN.md](BENCHMARK.zh-CN.md)

Measurement methodology and results for the cann-on-gpu **Apple Metal** backend.

> Data collected on **Apple M4 Max** (40-core GPU, 128 GB unified memory LPDDR5X, Metal 4). The cross-backend column is **NVIDIA GB10** (`sm_121`, UMA LPDDR5X, CUDA 13) running the *same* contract with the verified CUDA backend.
> These are **different silicon and different software stacks** — the comparison shows the same ACL workload on two backends, **not** an optimization A/B. Numbers do not extrapolate across machines; re-benchmark when switching.

## Value Proposition

Like the CUDA backend, the Metal backend's purpose is to obtain cross-checkable Ascend-operator results **orders of magnitude faster than the official CPU no-card simulator** (instruction-level, ~40–85 s/operator, O(work) growth). On Apple silicon the unified-memory design also removes host/device copies entirely, so the host-side fallback families pay no transfer cost.

## Measurement Methodology

- Tool: `tools/bench.cpp` (pure ACL client, links only `libascendcl.dylib`, mirroring how an Ascend application links), built against `metal/lib`.
- Timing: `std::chrono` wall-clock per call, **including the GetWorkspaceSize executor build** (the true per-call cost); average over many iterations after warmup; workspace pre-allocated and reused.
- Derived metrics: GEMM / attention in GFLOP/s (attention as QKᵀ+PV ≈ 4·B·N·S²·D); memory-bound operators in effective GB/s.
- **No toggle-based optimization A/B.** Unlike the CUDA backend (which exposes `ELTWISE_NO_VEC` / `MXFP8_NO_HW` / … env switches), the Metal backend has no optimization toggles, so this document reports absolute throughput plus the cross-backend comparison rather than optimized-vs-naive sweeps.
- Reproduction:

```bash
source ../env.sh
cd ../tools && mkdir -p bin
clang++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include bench.cpp \
    -L../metal/lib -lascendcl -Wl,-rpath,../metal/lib -o bin/bench
./bin/bench
```

> On Apple silicon, fp16 / bf16 matmul, W8A8 int8 matmul, sub-byte fp8/fp4/fp6/HiF8 matmul, attention, convolution and pooling (forward / backward / transposed / 3D / adaptive), elementwise, softmax and norm all run on the GPU (MPS or MSL). bf16 uses the MPS fp32 path via a lossless widen; sub-byte formats use the MPS fp16 path via a lossless decode→fp16 widen; int8 uses MPS's native Int8 GEMM; attention runs QKᵀ/PV as batched MPS GEMM; conv2d/3d forward use im2col + MPS GEMM and the conv/pool backward + transposed paths use MSL gather kernels. The PagedAttention scalar-vs-warp case is a CUDA-specific A/B that runs host-side and is omitted below.

## Results — Apple M4 Max (Metal)

| Operator (config) | Result | Notes |
|---|---|---|
| Device malloc+free / pair (cached) | 0.0034 ms (3.36 µs) | Caching allocator over `MTLBuffer` reuse |
| GEMM fp16 1024³ | 0.522 ms / **4115 GFLOP/s** | MPS matmul |
| GEMM fp16 4096³ | 9.14 ms / **15042 GFLOP/s** | MPS matmul (~15 TFLOP/s) |
| GEMM bf16 1024³ | 0.55 ms / 3939 GFLOP/s | GPU bf16↔fp32 widen kernels + MPS fp32 (overhead-bound at this size, ≈ fp16) |
| GEMM bf16 4096³ | 14.6 ms / **9444 GFLOP/s** | GPU bf16↔fp32 widen kernels + MPS fp32 GEMM |
| GEMM fp8 e4m3 1024³ | 0.74 ms / 2886 GFLOP/s | GPU sub-byte decode→fp16 kernel + MPS fp16 (overhead-bound, ≈ fp16) |
| GEMM fp8 e4m3 4096³ | 13.0 ms / **10552 GFLOP/s** | GPU decode→fp16 kernel + MPS fp16 GEMM (fp4/fp6/HiF8 same route) |
| Attention S=128 (B4 N8 D64, fp16) | 0.88 ms / 153 GFLOP/s | batched MPS QKᵀ/PV + MSL softmax (fp32-internal); per-thread online-softmax kernel as fallback |
| └ S=512 / 1024 / 2048 | 1.62 / 5.77 / 21.7 ms | 1324 / 1488 / 1584 GFLOP/s |
| Conv2d N8 C64 64² Co64 k3 (fp32) | 21.7 ms / 111 GFLOP/s | im2col (host gather) + MPS fp32 GEMM (conv3d forward takes the same route) |
| Conv backward / transposed, pooling (2D/3D, adaptive, backward) | MSL gather kernels, one thread per output | output-centric inverse map; no atomics |
| Add 4M (fp32) | 0.246 ms / **205 GB/s** | MSL elementwise |
| RMSNorm 4096×1024 (fp32) | 0.453 ms / 74 GB/s | MSL, one threadgroup per row |
| Softmax last-dim 4096×1024 (fp32) | 0.339 ms / 99 GB/s | MSL coalesced |

(`ReduceSum` full-reduction is omitted: the op passes the tolerance suite, but the benchmark's 100-iteration pipelined full reduction stalls the harness on this backend.)

## Cross-Backend Comparison (same workload, same contract)

The same `tools/bench.cpp` cases on the Metal backend (M4 Max) and the CUDA backend (NVIDIA GB10). This is a **backend + hardware** comparison, not an optimization measurement — the two are different chips with different compute and memory ceilings.

| Workload | M4 Max — Metal | GB10 — CUDA | GB10 / M4 |
|---|---|---|---|
| malloc+free / pair (cached) | 3.36 µs | 0.073 µs | 46× |
| GEMM fp16 1024³ | 4115 GFLOP/s | 69442 GFLOP/s | 16.9× |
| GEMM fp16 4096³ | 15042 GFLOP/s | 81613 GFLOP/s | 5.4× |
| Attention S=2048 (B4 N8 D64) | 1584 GFLOP/s | 1656 GFLOP/s | 1.05× |
| Add 4M (fp32) | 205 GB/s | 235 GB/s | 1.15× |
| RMSNorm 4096×1024 (fp32) | 74 GB/s | 241 GB/s | 3.3× |
| Softmax 4096×1024 (fp32) | 99 GB/s | 232 GB/s | 2.3× |

**Findings.**
- **Streaming elementwise is close** (Add 1.15×): both are unified-memory LPDDR5X parts, so a bandwidth-bound op that just streams memory lands in the same class on both.
- **Compute-heavy GEMM favors the GB10** (5.4× at 4096³): the NVIDIA GPU has the larger tensor-throughput ceiling; MPS still delivers the right vendor matmul on the Apple GPU (~15 TFLOP/s fp16).
- **Attention is on par** (1.05×): both run the matmul through vendor GEMM (MPS batched QKᵀ/PV on Metal, cuBLASLt on CUDA) with a softmax in between, so attention lands in the same class on both parts.
- **RMSNorm/Softmax** (2.3–3.3×): the Metal MSL reduction kernels are less tuned than the CUDA coalesced/warp-reduction versions, despite similar memory bandwidth.

## Where the Headroom Is

The remaining GPU-acceleration headroom, matching the roadmap in [`TODO.md`](TODO.md):

- **Long-sequence attention without materialized scores.** The attention path runs QKᵀ and PV as batched MPS GEMM, which materializes the [B·H, Sq, Skv] scores; it is memory-capped and falls back to the per-thread online-softmax kernel beyond that. A `simdgroup_matrix` FlashAttention-style kernel would keep the matmul-hardware speed without the S² materialization.
- **A few intentionally host-side families.** Dense linear algebra (Accelerate/LAPACK over unified memory) and a couple of niche conv ops (ConvTbc, deformable conv) remain on the host — not bottlenecks. (conv2d forward goes through im2col + MPS GEMM, so its current limit is the host im2col gather, not the GEMM.)

> The Metal backend's goal is **functional / tolerance-level** parity with the CUDA backend (which it meets — full operator parity, all tests green, Qwen3-0.6B end-to-end at 3.82e-6), not a performance contest. The figures here quantify where the GPU-acceleration headroom is, not a deficiency in correctness.
