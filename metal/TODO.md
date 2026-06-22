# Metal Backend — Roadmap

> 中文版 / Chinese: [TODO.zh-CN.md](TODO.zh-CN.md)

The Apple Metal backend implements the *same* `acl*` runtime + `aclnn*` operator + `Hccl*` contract as the CUDA backend, so the backend-agnostic clients in `../tests/` and `../tools/` link against it without source changes (`make -C ../tests BACKEND=../metal`).

**Operator coverage is complete** — full parity with the CUDA backend (every exported `aclnn*` operator), all backend-agnostic test binaries pass, and Qwen3-0.6B runs end-to-end matching HuggingFace (logits 3.82e-6, greedy 8/8). What this file tracks is what is **left or optimizable**: items bound to hardware this machine does not have, GPU-acceleration headroom (much of the long tail runs host-side for correctness), and a multi-device interconnect for the collectives.

> Scope mirrors the project as a whole: **functional / tolerance-level** validation and cross-architecture backend exploration — not bit-exact Ascend fidelity, not a performance contest against CUDA.

Target machine: **Apple M4 Max** (40-core GPU, 128 GB unified memory, Metal 4).

---

## A. Bound to hardware this machine does not have

- **Apple M5 GPU Neural Accelerator.** The M5 generation adds a per-GPU-core Neural Accelerator (matrix/tensor units) that accelerates GEMM; this box is an **M4 Max**, which has no such unit, so it cannot be tested or used here. When an M5-class machine is available, the matmul path (MPS GEMM and the quant-GEMM decode→GEMM tail) is the natural place to route through it.
- **Real-Ascend golden cross-check + bit-exact NZ / fractal fidelity.** The Ascend-specific fused / quant / distributed / NZ-format ops are verified locally by structural invariants, degenerate-equivalence, and **differential testing against the CUDA backend** on an NVIDIA GB10 reference host. A second opinion against real Ascend golden output, and confirming the NZ fractal layout matches Ascend's native layout bit-for-bit, both need real Ascend hardware, which this machine does not have. This is the same permanent out-of-scope item the CUDA backend carries.

## B. GPU-acceleration headroom (performance, not correctness)

The matmul, attention, convolution / pooling, elementwise, softmax and norm families run on the GPU (MPS or MSL). What remains:

- **Long-sequence attention without materialized scores.** Attention runs QKᵀ and PV as batched MPS GEMM, which materializes the [B·H, Sq, Skv] scores; it is therefore memory-capped and falls back to the per-thread online-softmax kernel beyond the cap. A `simdgroup_matrix` FlashAttention-style kernel would keep the matmul-hardware speed without the S² materialization.
- **A few intentionally host-side families.** Dense linear algebra (inverse/solve/det/cholesky/qr/svd/eigh/lu) runs on Accelerate/LAPACK over unified memory (no device copy), and a couple of niche conv ops (ConvTbc, deformable conv) stay on the host loop. These are not bottlenecks.

## C. Multi-device collectives / interconnect

`Hccl*` is a single-rank identity stub (one Mac = one device; multi-rank reduction is degenerate). A genuine multi-Mac path would implement the collectives over:

- a **Thunderbolt interconnect** (a Thunderbolt bridge directly linking two or more Macs), or
- a **Thunderbolt-attached NIC providing RDMA**,

behind the same `include/hccl/hccl.h` contract — the analogue of the CUDA backend's NCCL-over-RoCE transport. The `Hccl*` API and semantics already form the right seam; the work is to plug in a real Thunderbolt / RDMA transport in place of the single-rank stub, so the contract stays stable while the backend supplies its native fabric.
