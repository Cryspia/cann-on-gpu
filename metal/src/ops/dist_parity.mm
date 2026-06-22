// Distributed / comm-fused operators (MC2 + MoE-distribute) — SINGLE-RANK (rank 0 of 1).
//
// This backend is single-device: HCCL collectives degenerate to identity on one rank, so each fused op
// reduces to its COMPUTE part with the collective as a no-op. Semantics mirror cuda/src/ops/mc2_ext.cu and
// the local MoE init/finalize-routing in moe.cu:
//   MatmulAllReduce / AllGatherMatmul / MatmulReduceScatter on 1 rank  == plain Matmul (+bias).
//   QuantMatmulAllReduce on 1 rank                                     == int8 dequant matmul.
//   WeightQuantMatmulAllReduce on 1 rank                               == antiquant (weight-only) matmul.
//   GroupedMatMulAllReduce on 1 rank                                   == per-group local matmul.
//   *MatmulAllReduceAddRmsNorm                                         == Matmul (+(de)quant) then AddRmsNorm.
//   MoeDistributeDispatch/Combine (+ Setup/Teardown) on 1 rank         == identity copy (local rearrange).
//   MoeDistributeCombineAddRmsNorm on 1 rank                           == AddRmsNorm(x, residual)·gamma.
//   AlltoAll*/BatchMatMul* on 1 rank                                   == local (batch) matmul.
//   QuantAllReduce / AllGatherAdd on 1 rank                            == identity / (x + bias).
//
// All math is host-side over unified memory (MTLStorageModeShared ⇒ device ptr == host ptr), exact fp64
// accumulation, mirroring the host-GEMM fallback in matmul.mm / conv.mm. Ops here are declared in
// aclnnop/aclnn_mc2.h (signatures depend on HcclComm); we only DEFINE them (no Metal op is duplicated).
// We never invoke any Hccl* function — the collective is a no-op for the single rank, so libascendcl needs
// no link dependency on libhccl.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "hccl/hccl.h"          // HcclComm (= void*); collectives are no-ops on the single rank (not called).
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// We declare our own prototypes for the 38 ops below rather than including aclnnop/aclnn_mc2.h: that header
// re-declares aclnnMatmulAlltoAll / aclnnAlltoAllMatmul (and friends) with an HcclComm-bearing signature that
// conflicts with the no-comm declarations already in aclnn_ops.h (defined in blas_parity.mm). The prototypes
// here match the aclnn_mc2.h signatures exactly for the ops we implement; tests use the same signatures.
extern "C" {
#define MC2_MM_PROTO(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**); aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define MC2_QMM_PROTO(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**); aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define MC2_A2A_PROTO(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**); aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
MC2_MM_PROTO(aclnnMatmulAllReduce) MC2_MM_PROTO(aclnnMatmulAllReduceV2)
MC2_MM_PROTO(aclnnMatmulReduceScatter) MC2_MM_PROTO(aclnnMatmulReduceScatterV2)
MC2_MM_PROTO(aclnnMatmulAllGather) MC2_MM_PROTO(aclnnAllGatherMatmul) MC2_MM_PROTO(aclnnAllGatherMatmulV2)
MC2_MM_PROTO(aclnnAlltoAllAllGatherBatchMatMul) MC2_MM_PROTO(aclnnBatchMatMulReduceScatterAlltoAll)
MC2_QMM_PROTO(aclnnQuantMatmulAllReduce) MC2_QMM_PROTO(aclnnQuantMatmulAllReduceV2)
MC2_QMM_PROTO(aclnnQuantMatmulAllReduceV3) MC2_QMM_PROTO(aclnnQuantMatmulAllReduceV4)
MC2_QMM_PROTO(aclnnQuantReduceScatter)
MC2_A2A_PROTO(aclnnQuantAllReduce)
MC2_A2A_PROTO(aclnnMoeDistributeDispatch) MC2_A2A_PROTO(aclnnMoeDistributeDispatchV2)
MC2_A2A_PROTO(aclnnMoeDistributeDispatchV3) MC2_A2A_PROTO(aclnnMoeDistributeDispatchV4)
MC2_A2A_PROTO(aclnnMoeDistributeCombine) MC2_A2A_PROTO(aclnnMoeDistributeCombineV2)
MC2_A2A_PROTO(aclnnMoeDistributeCombineV3) MC2_A2A_PROTO(aclnnMoeDistributeCombineV4)
MC2_A2A_PROTO(aclnnMoeDistributeDispatchSetup) MC2_A2A_PROTO(aclnnMoeDistributeDispatchTeardown)
MC2_A2A_PROTO(aclnnMoeDistributeCombineSetup) MC2_A2A_PROTO(aclnnMoeDistributeCombineTeardown)
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor*, const aclTensor*, const char*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAllGatherAdd(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnWeightQuantMatmulAllReduce(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclIntArray*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedMatMulAllReduce(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, HcclComm, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
#undef MC2_MM_PROTO
#undef MC2_QMM_PROTO
#undef MC2_A2A_PROTO
} // extern "C"

namespace {

// ---- generic dtype read/write over unified memory (fp32/fp16/bf16/int8/int4) ----
inline float dg_h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; std::memcpy(&x, &f, 4); x |= sign; }
    else if (e == 31) x = sign | 0x7F800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
inline uint16_t dg_f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t r = m & 0x1FFF, h = sign | (e << 10) | (m >> 13);
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}
inline float dg_bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }
inline uint16_t dg_f2bf(float f) { uint32_t b; std::memcpy(&b, &f, 4); b += 0x7FFF + ((b >> 16) & 1); return (uint16_t)(b >> 16); }

// read int4 (two nibbles per byte, low nibble first), sign-extended
inline int rd_i4(const uint8_t *base, int64_t i) {
    uint8_t byte = base[i >> 1]; int nib = (i & 1) ? (byte >> 4) : (byte & 0xF);
    return (nib & 0x8) ? (nib - 16) : nib;
}
inline float dg_rd(const aclTensor *t, int64_t i) {
    const void *base = t->data; int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:   return ((const float *)base)[off + i];
        case ACL_FLOAT16: return dg_h2f(((const uint16_t *)base)[off + i]);
        case ACL_BF16:    return dg_bf2f(((const uint16_t *)base)[off + i]);
        case ACL_INT8:    return (float)((const int8_t *)base)[off + i];
        case ACL_INT32:   return (float)((const int32_t *)base)[off + i];
        default: break;
    }
    if (t->dtype == ACL_INT4) return (float)rd_i4((const uint8_t *)base + off / 2, (off & 1) + i);
    return 0.f;
}
inline void dg_wr(aclTensor *t, int64_t i, float v) {
    void *base = t->data; int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:   ((float *)base)[off + i] = v; return;
        case ACL_FLOAT16: ((uint16_t *)base)[off + i] = dg_f2h(v); return;
        case ACL_BF16:    ((uint16_t *)base)[off + i] = dg_f2bf(v); return;
        case ACL_INT8:    { int q = (int)std::lround(v); ((int8_t *)base)[off + i] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q)); return; }
        case ACL_INT32:   ((int32_t *)base)[off + i] = (int32_t)std::lround(v); return;
        default: return;
    }
}
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ---- GEMM kinds the executor records (e->op reused as a local tag) ----
enum DistKind {
    DK_MM = 1,      // out = x[M,K] @ W[K,N] (+ bias[N])
    DK_QMM,         // out = (int8 x @ int8 W) * scale[N]   (dequant)
    DK_WQMM,        // out = x @ ((W - offset[N]) * scale[N])  (antiquant, weight-only)
    DK_GMM,         // grouped matmul over groupList (row groups, shared single weight here)
    DK_BMM,         // batched matmul x[B,M,K] @ W[B,K,N] (+ bias[N])
    DK_COPY,        // identity copy x -> out  (alltoall / dispatch / combine / quant-allreduce on 1 rank)
    DK_ADD,         // out = x + bias[N]  (allgather-add)
    DK_MM_RMS,      // out=matmul into temp; then y = rms(temp+residual)*gamma; residualSum = temp+residual
    DK_QMM_RMS,     // quant matmul + AddRmsNorm
    DK_WQMM_RMS,    // antiquant matmul + AddRmsNorm
    DK_INPLACE_MM_RMS,    // selfRef holds x; matmul(selfRef,W)->temp; AddRmsNorm in place into selfRef
    DK_INPLACE_QMM_RMS,
    DK_INPLACE_WQMM_RMS,
    DK_COMBINE_RMS, // moe combine + AddRmsNorm: y = rms(x+residual)*gamma
};

// dims helper: trailing two dims are [M,K]/[K,N]; treat anything before as a single M (flatten rows).
inline void mk_dims(const aclTensor *x, const aclTensor *w, int64_t &M, int64_t &K, int64_t &N) {
    K = x->viewDims.back();
    M = x->numel() / K;
    N = w->viewDims.back();   // weight is [K,N]
}

// out[M,N] = x[M,K] @ W[K,N] (+ bias[N])
void gemm_mm(const aclTensor *x, const aclTensor *w, const aclTensor *bias, aclTensor *out) {
    int64_t M, K, N; mk_dims(x, w, M, K, N);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (double)dg_rd(x, m * K + k) * dg_rd(w, k * N + n);
            if (bias) acc += dg_rd(bias, n);
            dg_wr(out, m * N + n, (float)acc);
        }
}
// out[M,N] = (int8 x[M,K] @ int8 W[K,N]) * scale[N]
void gemm_qmm(const aclTensor *x, const aclTensor *w, const aclTensor *scale, aclTensor *out) {
    int64_t M, K, N; mk_dims(x, w, M, K, N);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            long long acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (long long)dg_rd(x, m * K + k) * (long long)dg_rd(w, k * N + n);
            dg_wr(out, m * N + n, (float)((double)acc * dg_rd(scale, n)));
        }
}
// out[M,N] = x[M,K] @ ((W[K,N] - offset[N]) * scale[N])   (weight-only antiquant; W int8/int4)
void gemm_wqmm(const aclTensor *x, const aclTensor *w, const aclTensor *scale, const aclTensor *offset, aclTensor *out) {
    int64_t M, K, N; mk_dims(x, w, M, K, N);
    for (int64_t m = 0; m < M; ++m)
        for (int64_t n = 0; n < N; ++n) {
            double acc = 0, sc = dg_rd(scale, n), of = offset ? dg_rd(offset, n) : 0.0;
            for (int64_t k = 0; k < K; ++k) {
                double wf = ((double)dg_rd(w, k * N + n) - of) * sc;
                acc += (double)dg_rd(x, m * K + k) * wf;
            }
            dg_wr(out, m * N + n, (float)acc);
        }
}
// batched: x[B,M,K] @ W[B,K,N] (+ bias[N]); W may be [K,N] (shared, B==1 broadcast handled by caller dims)
void gemm_bmm(const aclTensor *x, const aclTensor *w, const aclTensor *bias, aclTensor *out) {
    int nd = (int)x->viewDims.size();
    int64_t K = x->viewDims[nd - 1], M = x->viewDims[nd - 2];
    int64_t B = x->numel() / (M * K);
    int wd = (int)w->viewDims.size();
    int64_t N = w->viewDims[wd - 1];
    bool wBatched = (wd >= 3);
    for (int64_t b = 0; b < B; ++b)
        for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
                double acc = 0;
                int64_t xBase = (b * M + m) * K, wBase = wBatched ? b * K * N : 0;
                for (int64_t k = 0; k < K; ++k) acc += (double)dg_rd(x, xBase + k) * dg_rd(w, wBase + k * N + n);
                if (bias) acc += dg_rd(bias, n);
                dg_wr(out, (b * M + m) * N + n, (float)acc);
            }
}

// tail AddRmsNorm: y[r,d] = rms_r(s)*gamma[d], s = temp + residual; residualSum = s.
// temp holds the (de)quantized matmul output [M,N]; gamma/residual length D == N.
void add_rms_norm(const std::vector<double> &temp, const aclTensor *residual, const aclTensor *gamma,
                  double eps, aclTensor *y, aclTensor *residualSum, int64_t M, int64_t N) {
    std::vector<double> s(N);
    for (int64_t r = 0; r < M; ++r) {
        double ss = 0;
        for (int64_t d = 0; d < N; ++d) {
            double sv = temp[r * N + d] + (residual ? dg_rd(residual, r * N + d) : 0.0);
            s[d] = sv; ss += sv * sv;
            if (residualSum) dg_wr(residualSum, r * N + d, (float)sv);
        }
        double inv = 1.0 / std::sqrt(ss / N + eps);
        for (int64_t d = 0; d < N; ++d) dg_wr(y, r * N + d, (float)(s[d] * inv * dg_rd(gamma, d)));
    }
}

aclnnStatus run_dist(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    int64_t M, K, N;
    switch (e->op) {
        case DK_MM:   gemm_mm(e->a, e->b, e->c, e->out); break;
        case DK_QMM:  gemm_qmm(e->a, e->b, e->c, e->out); break;
        case DK_WQMM: gemm_wqmm(e->a, e->b, e->c, e->out2, e->out); break;
        case DK_BMM:  gemm_bmm(e->a, e->b, e->c, e->out); break;
        case DK_ADD: {
            int64_t n = e->a->numel(), bn = e->c ? e->c->numel() : 0;
            for (int64_t i = 0; i < n; ++i)
                dg_wr(e->out, i, (float)(dg_rd(e->a, i) + (e->c && bn ? dg_rd(e->c, i % bn) : 0.0)));
            break;
        }
        case DK_COPY: {
            int64_t n = e->a->numel();
            // same-dtype fast path is a byte copy; else value copy through fp.
            if (e->a->dtype == e->out->dtype)
                std::memcpy((char *)e->out->data + e->out->offset * (int64_t)dtype_size(e->out->dtype),
                            (char *)e->a->data + e->a->offset * (int64_t)dtype_size(e->a->dtype),
                            (size_t)n * dtype_size(e->a->dtype));
            else for (int64_t i = 0; i < n; ++i) dg_wr(e->out, i, dg_rd(e->a, i));
            break;
        }
        case DK_GMM: {
            // grouped matmul: x rows split by groupList (cumulative row counts in e->axes); single shared W [K,N].
            K = e->a->viewDims.back(); N = e->b->viewDims.back();
            int64_t total = e->a->numel() / K, start = 0;
            const std::vector<int64_t> &gl = e->axes;
            for (size_t g = 0; g <= gl.size(); ++g) {
                int64_t end = (g < gl.size()) ? gl[g] : total;
                if (end > total) end = total;
                for (int64_t m = start; m < end; ++m)
                    for (int64_t n = 0; n < N; ++n) {
                        double acc = 0;
                        for (int64_t k = 0; k < K; ++k) acc += (double)dg_rd(e->a, m * K + k) * dg_rd(e->b, k * N + n);
                        dg_wr(e->out, m * N + n, (float)acc);
                    }
                start = end; if (start >= total) break;
            }
            break;
        }
        case DK_MM_RMS: case DK_QMM_RMS: case DK_WQMM_RMS:
        case DK_INPLACE_MM_RMS: case DK_INPLACE_QMM_RMS: case DK_INPLACE_WQMM_RMS: {
            mk_dims(e->a, e->b, M, K, N);
            std::vector<double> temp((size_t)M * N);
            // compute (de)quant matmul into temp
            for (int64_t m = 0; m < M; ++m)
                for (int64_t n = 0; n < N; ++n) {
                    double v;
                    if (e->op == DK_QMM_RMS) {
                        // out-of-place quant: int8 x @ int8 W, dequant by scale[n].
                        long long acc = 0;
                        for (int64_t k = 0; k < K; ++k) acc += (long long)dg_rd(e->a, m * K + k) * (long long)dg_rd(e->b, k * N + n);
                        v = (double)acc * dg_rd(e->c, n);          // c = scale
                    } else if (e->op == DK_WQMM_RMS || e->op == DK_INPLACE_QMM_RMS) {
                        // weight-only (anti)quant: float x @ (int8 W · scale[n]). The inplace-quant variant uses
                        // this form so selfRef stays a float buffer (in == out dtype) under inplace semantics.
                        double acc = 0, sc = e->c ? dg_rd(e->c, n) : 1.0;
                        for (int64_t k = 0; k < K; ++k) acc += (double)dg_rd(e->a, m * K + k) * ((double)dg_rd(e->b, k * N + n) * sc);
                        v = acc;
                    } else {
                        // plain matmul (DK_MM_RMS, DK_INPLACE_MM_RMS, DK_INPLACE_WQMM_RMS with scale=1)
                        double acc = 0, sc = (e->op == DK_INPLACE_WQMM_RMS && e->c) ? dg_rd(e->c, n) : 1.0;
                        for (int64_t k = 0; k < K; ++k) acc += (double)dg_rd(e->a, m * K + k) * ((double)dg_rd(e->b, k * N + n) * sc);
                        v = acc;
                    }
                    temp[m * N + n] = v;
                }
            // residual stashed in e->mask, gamma in e->mean, y in e->out, residualSum in e->out2.
            // Inplace variants write y back into selfRef (e->a). Use a mutable view.
            aclTensor *y = e->out ? e->out : const_cast<aclTensor *>(e->a);
            add_rms_norm(temp, e->mask, e->mean, e->eps, y, e->out2, M, N);
            break;
        }
        case DK_COMBINE_RMS: {
            // y = rms(x + residual)*gamma; residualSum = x+residual.  x=e->a, residual=e->mask, gamma=e->mean.
            int64_t D = e->mean->numel(), R = e->a->numel() / D;
            std::vector<double> temp((size_t)R * D);
            for (int64_t i = 0; i < R * D; ++i) temp[i] = dg_rd(e->a, i);
            add_rms_norm(temp, e->mask, e->mean, e->eps, e->out, e->out2, R, D);
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_dist(e, s); } delete e; return st; }

} // namespace

extern "C" {

// ================= plain Matmul-collective family (single-rank == x@W (+bias)) =================
// out = AllReduce_SUM(x@W) (+ bias) -> local x@W (+bias)
aclnnStatus aclnnMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_MM; e->a = x; e->b = weight; e->c = bias; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMatmulAllReduce)
aclnnStatus aclnnMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulAllReduceGetWorkspaceSize(x, weight, bias, comm, out, ws, ex);
}
RUN(aclnnMatmulAllReduceV2)

// out[M/nranks,N] = ReduceScatter(x@W); single rank: nranks=1 so out == full x@W.
aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulAllReduceGetWorkspaceSize(x, weight, bias, comm, out, ws, ex);
}
RUN(aclnnMatmulReduceScatter)
aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulAllReduceGetWorkspaceSize(x, weight, bias, comm, out, ws, ex);
}
RUN(aclnnMatmulReduceScatterV2)

// out[M,N] = AllGather over row-shards(x_shard@W); single rank: just x@W.
aclnnStatus aclnnMatmulAllGatherGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulAllReduceGetWorkspaceSize(x, weight, bias, comm, out, ws, ex);
}
RUN(aclnnMatmulAllGather)

// the MC2_MM block in aclnn_mc2.h: (x, weight, bias, comm, out)
#define MC2_MM_DEF(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, \
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnMatmulAllReduceGetWorkspaceSize(x, weight, bias, comm, out, ws, ex); } \
RUN(NAME)
MC2_MM_DEF(aclnnAllGatherMatmul)
MC2_MM_DEF(aclnnAllGatherMatmulV2)

// AlltoAll+AllGather then BatchMatMul / BatchMatMul then ReduceScatter+AlltoAll — single rank: local batch matmul.
aclnnStatus aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_BMM; e->a = x; e->b = weight; e->c = bias; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAlltoAllAllGatherBatchMatMul)
aclnnStatus aclnnBatchMatMulReduceScatterAlltoAllGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize(x, weight, bias, comm, out, ws, ex);
}
RUN(aclnnBatchMatMulReduceScatterAlltoAll)

// AllGatherAdd: out = AllGather(x) + bias -> single rank x + bias. (group/rankSize ignored.)
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *x, const aclTensor *bias, const char *group,
        int64_t rankSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)group; (void)rankSize; if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_ADD; e->a = x; e->c = bias; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAllGatherAdd)

// ================= quant matmul family (single-rank == int8 dequant matmul) =================
aclnnStatus aclnnQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_QMM; e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantMatmulAllReduce)
#define MC2_QMM_DEF(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, \
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnQuantMatmulAllReduceGetWorkspaceSize(x, weight, scale, comm, out, ws, ex); } \
RUN(NAME)
MC2_QMM_DEF(aclnnQuantMatmulAllReduceV2)
MC2_QMM_DEF(aclnnQuantMatmulAllReduceV3)
MC2_QMM_DEF(aclnnQuantMatmulAllReduceV4)
MC2_QMM_DEF(aclnnQuantReduceScatter)   // ReduceScatter(dequant(x@W)) -> single rank: dequant matmul

// WeightQuantMatmulAllReduce: weight-only antiquant. (x, weight, antiquantScale, antiquantOffset, comm, out)
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, HcclComm comm, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !antiquantScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_WQMM; e->a = x; e->b = weight; e->c = antiquantScale;
    e->out2 = (aclTensor *)antiquantOffset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnWeightQuantMatmulAllReduce)

// QuantAllReduce: AllReduce of a quantized buffer -> single rank: identity copy.
aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_COPY; e->a = x; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantAllReduce)

// GroupedMatMulAllReduce: per-group local matmul (groupList = cumulative row counts), shared weight [K,N].
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_GMM; e->a = x; e->b = weight; e->out = out;
    if (groupList) e->axes = groupList->v; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnGroupedMatMulAllReduce)

// ================= AddRmsNorm tail-fused (matmul (+ (de)quant) then AddRmsNorm) =================
// y = RmsNorm(AllReduce(x@W) + residual)·gamma; residualSum = that sum.
aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y,
        aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_MM_RMS; e->a = x; e->b = weight; e->mask = residual; e->mean = gamma;
    e->out = y; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMatmulAllReduceAddRmsNorm)

aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm,
        aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !scale || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_QMM_RMS; e->a = x; e->b = weight; e->c = scale; e->mask = residual;
    e->mean = gamma; e->out = y; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantMatmulAllReduceAddRmsNorm)

aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm,
        aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !weight || !scale || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_WQMM_RMS; e->a = x; e->b = weight; e->c = scale; e->mask = residual;
    e->mean = gamma; e->out = y; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnWeightQuantMatmulAllReduceAddRmsNorm)

// Inplace variants: selfRef holds x in, y out (written back into selfRef). out=nullptr signals inplace.
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight,
        const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!selfRef || !weight || !residual || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_INPLACE_MM_RMS; e->a = selfRef; e->b = weight; e->mask = residual;
    e->mean = gamma; e->out = nullptr; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnInplaceMatmulAllReduceAddRmsNorm)

aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight,
        const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm,
        aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!selfRef || !weight || !scale || !residual || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_INPLACE_QMM_RMS; e->a = selfRef; e->b = weight; e->c = scale;
    e->mask = residual; e->mean = gamma; e->out = nullptr; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnInplaceQuantMatmulAllReduceAddRmsNorm)

aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight,
        const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!selfRef || !weight || !residual || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    // weight-only antiquant with no explicit scale in this (degenerate) signature: scale=1 via c=nullptr handled? use scale-of-1
    auto *e = new aclOpExecutor(); e->op = DK_INPLACE_WQMM_RMS; e->a = selfRef; e->b = weight; e->c = nullptr;
    e->mask = residual; e->mean = gamma; e->out = nullptr; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm)

// ================= MoE distribute (single-rank == identity copy / local AddRmsNorm) =================
#define MOE_A2A_DEF(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)comm; if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = DK_COPY; e->a = x; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
MOE_A2A_DEF(aclnnMoeDistributeDispatch)
MOE_A2A_DEF(aclnnMoeDistributeDispatchV2)
MOE_A2A_DEF(aclnnMoeDistributeDispatchV3)
MOE_A2A_DEF(aclnnMoeDistributeDispatchV4)
MOE_A2A_DEF(aclnnMoeDistributeCombine)
MOE_A2A_DEF(aclnnMoeDistributeCombineV2)
MOE_A2A_DEF(aclnnMoeDistributeCombineV3)
MOE_A2A_DEF(aclnnMoeDistributeCombineV4)
MOE_A2A_DEF(aclnnMoeDistributeDispatchSetup)
MOE_A2A_DEF(aclnnMoeDistributeDispatchTeardown)
MOE_A2A_DEF(aclnnMoeDistributeCombineSetup)
MOE_A2A_DEF(aclnnMoeDistributeCombineTeardown)

// MoeDistributeCombineAddRmsNorm: y = RmsNorm(combine(x) + residual)·gamma; combine == identity on 1 rank.
aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual,
        const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)comm; if (!x || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = DK_COMBINE_RMS; e->a = x; e->mask = residual; e->mean = gamma;
    e->out = y; e->out2 = residualSum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMoeDistributeCombineAddRmsNorm)
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual,
        const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum,
        uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(x, residual, gamma, eps, comm, y, residualSum, ws, ex);
}
RUN(aclnnMoeDistributeCombineAddRmsNormV2)

} // extern "C"
