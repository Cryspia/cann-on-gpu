# cann-on-gpu Roadmap

> 中文版 / Chinese: [TODO.zh-CN.md](TODO.zh-CN.md)

**Operator coverage (the former Part A primary goal) is functionally complete on this machine.** Of the **1036** authoritative open-source aclnn level-2 APIs (`cann/{opbase,ops-math,ops-transformer,ops-cv,ops-nn}` at branch `9.1.0-beta.1`, frozen in `tools/official_aclnn.txt`), **1014 (~98%)** are implemented and tolerance-verified locally — against PyTorch (forward + autograd backward), CPU/closed-form references, structural invariants (QR/SVD/eigh reconstruction, RoPE/FFT/cast/MoE round-trips), and a real 2-node RoCE/HCCL path. See `README.md` → "Operator Coverage Baseline" for the summary and the per-family breakdown.

Everything that could be done and verified on this single GB10 box is done. **What is left below is bound to hardware this machine does not have** — a real Ascend card (for golden cross-checks and bit-exact format fidelity), a high-bandwidth / native-low-precision discrete GPU (for the performance work whose gains GB10's UMA masks), or a multi-card / multi-node cluster (for scaling beyond the verified 2-node baseline).

---

## A. Bound to a real Ascend card

These are about *fidelity against Ascend*, not GPU compute — they need real Ascend hardware (or the slow card-free CANN Simulator) as the reference, which this machine does not have.

- **Real-Ascend golden cross-check of the no-torch-oracle ops.** ~600 implemented ops have no external numerical oracle on this box (Ascend-specific fused attention MLA/NSA/Paged, quant-matmul matrices, distributed-fused comm, NZ-format ops). Locally they are verified only by structural invariants or degenerate-equivalence. A second opinion against real Ascend golden output (via `tools/cannsim_golden.sh`, which needs the upstream cann-api-explorer + cann-sim) would harden them — but cann-sim is slow (~40–85 s/op) and the bulk of these ops have no explorer unit yet. The torch/CPU/invariant verification (`tools/torch_golden.sh` + the `*_check` harnesses) is the fast default and stays primary.
- **Bit-exact NZ / fractal weight-format fidelity.** The `*WeightNz` matmul family is implemented to *logical* equivalence (NZ weight treated as logically row-major → ND cuBLASLt cores, zero de-swizzle overhead, maximally performant) — **not** bit-level fractal layout reproduction. This is a permanent design limitation on GPU; confirming the fractal fast path matches Ascend's native layout requires real Ascend hardware.
- **The 22 still-unimplemented ops** (`tools/gap.txt`) — permanent out-of-scope for a GPU numeric reference: framework lifecycle (`aclnnInit`/`aclnnFinalize`), distributed control-plane (`aclnnDistributeBarrier`/`V2`), debug/RAS infrastructure (`aclnnSilentCheck`/`V2`, `aclnnPrecisionCompare`), device codecs (`aclnnHansEncode`/`Decode`), DVPP image-custom ops (`aclnnRasterizer`, `aclnnResize`, `aclnnBackgroundReplace`, `aclnnBlendImagesCustom`, `aclnnMrgbaCustom`), RNG sim-thread (`aclnnSimThreadExponential`), and a few large fused / tensor-list composites with no external reference (`aclnnConfusionTranspose`, `aclnnSplitTensor`, `aclnnAddLora`, `aclnnCoalesceSparse`, `aclnnExpandIntoJaggedPermute`, `aclnnFusedCrossEntropyLossWithMaxSum`, `aclnnFusedLinearOnlineMaxSum`). None have GPU-compute meaning verifiable on this box.

## B. Performance work bound to other hardware

Current performance data was collected on GB10: unified memory (UMA), low bandwidth (LPDDR5X ~273 GB/s), large L2, single-architecture SASS, single card.
This hardware profile **masks the gains from memory-bandwidth optimizations** and lacks certain native low-precision hardware units.
The correctness of the items below is verified or the implementations are ready, but **performance / native paths must be re-benchmarked, tuned, or built from scratch on high-bandwidth discrete GPUs or multi-card setups**.
Target cards: H100/H200 (`sm_90`), RTX 40 (`sm_89`), B200/GB200 (`sm_100`), RTX 50 (`sm_120`), etc.

### Needs Re-benchmarking (Implementation Ready; Gains Masked by GB10)

- **All performance numbers**: Every number in `BENCHMARK.md` is GB10-specific and cannot be extrapolated; a full re-run is required when switching cards.
- **128-bit vectorized memory access**: Only shows benefit on high-bandwidth (GDDR7/HBM) memory; masked by low bandwidth on GB10.
- **Operator fusion to reduce memory traffic**: Partially beneficial even on UMA, but the full gain only emerges when high bandwidth and tensor-core saturation coincide.
- **Native low-precision tensor-core throughput**: The MXFP8 native microscaling path already runs on Blackwell; the native path on each architecture needs its own re-benchmark
  (Hopper/Ada have no native MX support and will take the functional path).
- **Speedup of toggle-style optimizations**: TF32 fast-path GEMM, fp8-flash, large-L bitonic sort, native grouped GEMM, conv algorithm autotune (cuDNN find / Winograd), etc. —
  gains are broadly masked under GB10 UMA and must be re-measured on high-bandwidth / native low-precision cards.
- **Multi-node communication scalability**: Correctness has only been verified on 2 nodes × 1 card; scalability is unproven. Large-scale ring/tree, non-power-of-2 ranks, and intra-node NVLink all await multi-card/multi-node cluster validation.

### Worth Building New (Only Meaningful or Visible on High-Bandwidth Cards)

1. **Native low-precision GEMM kernel path**: Feed fp4/fp6/MXFP4 directly to tensor-cores instead of the current functional "unpack → fp16 GEMM" path.
   Delivers real throughput/memory savings, but only meaningful on native fp4/fp8 cards (Blackwell); the functional path is already accurate and the native path serves as an optional fast tier.
2. **FlashAttention-3-class attention kernel**: The current implementation is a hand-written WMMA flash kernel — solid quality but not FA3 grade. Attention is the primary optimization target in the compute-bound region;
   rewriting/tuning requires profiling on a high-bandwidth card to be actionable.
3. **Bandwidth-reducing fusion in production**: Fused QKV projection, residual+norm+next-projection, further attention materialization elimination, etc. — full gains must be quantified on a high-bandwidth card.
4. **Pervasive 128-bit vectorized memory access**: Improves memory-level parallelism; only effective on high-bandwidth VRAM.
5. **cann-on-gpu vs. native CUDA comparison benchmark**: On GB10 most scenarios are memory-bound rather than compute-bound, distorting the comparison; benchmarking on a high-bandwidth card is required for credible "compute-intensive gap" numbers.
6. **Multi-card tensor parallelism / NVLink / multi-node scaling**: TP/PP sharding, intra-node NVLink, and large-scale collective communication all require a multi-card cluster.
7. **Foreach multi-tensor fusion**: the foreach family (`src/ops/foreach_ext.cu`) currently issues one kernel launch per tensor in the list. Fuse the whole list into a single grid-strided multi-tensor kernel (one launch over a packed tensor-meta array, as in PyTorch's `MultiTensorApply`) to remove per-tensor launch overhead — most visible for optimizer states with many small parameter tensors.

### Fidelity of Native Low-Precision Scaling Formats (Pending Native fp4 Card Validation)

The native NVFP4 path for MXFP4 requires converting Ascend's E8M0/block-32 scaling to E4M3/block-16 scaling: constrained by E4M3's dynamic range and being non-native to the Ascend format,
this can only serve as an "optional fast, reduced-fidelity" tier and must be validated on a native fp4 card (Blackwell). **Fidelity** computation for Ascend MXFP4 continues to use the functional path (block dequantize → fp16, full E8M0 range, exact).

## C. Possible future structure work (optional, not hardware-bound)

- **`ops*` per-library build — evaluated, not pursued.** Splitting the build into five upstream `.so`s (`opbase` / `ops-math` / `ops-transformer` / `ops-cv` / `ops-nn`) is *not* compatible with the current source layout: `cuda/src/ops/*.cu` is organized by **functional family** (blas, norm, attention, …), an orthogonal partition to the upstream **library** grouping — most family files mix ops from 2–4 libraries (e.g. `misc_ext.cu` spans all four; `index_ext.cu`/`matmul.cu`/`loss.cu` span three). Since one `.cu` → one `.o` → one `.so`, a faithful per-library split would require re-partitioning every op by `tools/op_map.tsv` (undoing the functional consolidation and growing the file count again), plus an inter-`.so` dependency DAG (`opbase ← ops-math ← ops-nn ← ops-transformer/ops-cv`) because of real cross-library coupling: shared helpers (`ew_run_one`/`ew_exec`/`check_same` in `elementwise.cu`, used by `inplace`/`foreach`) and nested-executor calls (`mc2` calling `aclnnMatmul`/`aclnnAddRmsNorm`). **Decision: keep the single `libascendcl.so`** — it is fully functional, simplest for a reference/CI backend, and what `tests/`/`tools/` link against. Revisit only if a downstream consumer concretely needs to link a library subset.

- **Vendor-neutral interconnect for the HCCL layer (future, once the API surface is larger).** The HCCL collectives (`cuda/src/hccl.cu`) are currently implemented directly on **NCCL** — an NVIDIA-specific transport. This is fine while CUDA is the only backend, but it bakes a vendor into the communication path. As the API surface and the set of GPU backends grow, refactor the collective/interconnect implementation behind a **backend-agnostic transport abstraction** so the `include/hccl/hccl.h` contract stays stable while each backend plugs in its native transport (e.g. RCCL on ROCm, oneCCL/Level-Zero on Intel) — or, better, a single vendor-neutral fabric (UCX / libfabric / a thin RDMA-verbs layer) shared across architectures so cross-GPU-architecture interconnect does not depend on NCCL. The `include/hccl/hccl.h` API and the `Hccl*` semantics already form the right seam; the work is to make `hccl.cu`'s transport pluggable rather than NCCL-hardwired.
