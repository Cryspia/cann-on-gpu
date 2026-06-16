# Performance Benchmarks

> 中文版 / Chinese: [BENCHMARK.zh-CN.md](BENCHMARK.zh-CN.md)

Measurement methodology and results for the cann-on-gpu CUDA backend.

> Data collected on NVIDIA GB10 (`sm_121`, unified memory UMA, LPDDR5X ~273 GB/s), CUDA 13, cuBLASLt / cuDNN backend.
> **These numbers are GB10-specific and cannot be extrapolated** — GB10's low-bandwidth UMA masks the gains of many memory-access optimizations. Switching to a high-bandwidth discrete GPU (HBM/GDDR7) requires a full re-benchmark; see [`cuda/TODO.md`](cuda/TODO.md).

## Value Proposition

The GPU backend's purpose is to obtain cross-checkable results **orders of magnitude faster than the official CPU no-card simulator**.

- Single-operator validation: no-card simulation **40–85 s/operator** (build + instruction-level simulation) vs. GPU backend **µs–ms**.
- Full-network forward pass: 20-operator single-layer transformer, GPU **1.67 ms/iter**; no-card simulation takes tens of seconds even for a single operator.
- Order-of-magnitude gap: **~10⁴–10⁵×**. Instruction-level simulation complexity is O(work), making large tensors infeasible; the GPU scales with hardware throughput.

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

The table below shows numbers achieved after optimization; "speedup" is relative to the naive unoptimized implementation.

| Operator (config) | Result | Speedup | Notes |
|---|---|---|---|
| GEMM fp16 / bf16 4096³ | ~90–94 TFLOP/s | — | cuBLASLt near peak |
| Attention default path S=2048 (B4 N8 D64) | 20.9 ms / 1647 GFLOP/s | 2.7× | Online softmax flash (no S/P materialization) + WMMA QKᵀ tensor-core; workspace 536 MB→0 |
| └ S=1024 / 512 / 128 | 5.08 / 1.39 / 0.11 ms | 2.6–2.7× | |
| Attention performance path S=2048 | 12.0 ms / 2863 GFLOP/s | — | cuBLASLt batched tensor-core, highest standard MHA throughput (S/P materialized) |
| PagedAttention decode B64 N8 L4096 D64 | 4.4 ms / 122 GFLOP/s | 12.5× | warp-per-row + register online softmax |
| RMSNorm 4096×1024 | 309 GB/s | 5.3× | One block per row, coalesced memory access + warp reduction |
| Softmax last-dim 4096×1024 | 324 GB/s | — | Coalesced fast path |
| Add / ReduceSum 4M (fp32) | 247 / 292 GB/s | — | Near bandwidth peak |
| tiny LLM forward (20 ops/layer) | 1.67 ms/iter | — | End-to-end composed |
| Device malloc+free / pair | 0.075 µs | ~3700× | Caching allocator (`ACL_NO_CACHE_ALLOC=1` as baseline) |
| decode-loop (single token, 2-layer GQA) | 0.173 → 0.100 ms/token | 1.72× | Full-loop CUDA Graph replay, eliminates per-kernel launch overhead |

## Notes on GB10

GB10 uses unified memory, low bandwidth, single-architecture SASS, single card. As a result:

- **Memory-access optimization gains are masked**: e.g., 128-bit vectorized loads and operator fusion that reduces memory traffic show no measurable difference on GB10; they only take effect on HBM/GDDR7 high-bandwidth cards.
- **Launch-bound optimizations are measurable**: decode is launch-bound, so CUDA Graph replay gains are not masked by bandwidth — one of the few optimization directions that shows real benefit on GB10.
- All numbers must be re-benchmarked when switching cards; a cann vs. native CUDA comparison benchmark is only meaningful on a high-bandwidth card (see `cuda/TODO.md`).

> All performance optimizations maintain full tolerance cross-check pass (no correctness sacrificed), and retain environment-variable toggles to revert to the unoptimized path for comparison
> (`ACL_NO_CACHE_ALLOC` / `ATTN_NO_PERF` / `PAGED_SCALAR` / `FLASH_NO_WMMA` / `MXFP8_NO_HW`, etc.).
