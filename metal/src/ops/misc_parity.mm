// Mixed "gap" operators (CUDA parity) the Metal backend had not yet exposed: ~39 leftover/odd ops spread
// across norm variants, forward/backward losses, two fused optimizers, a few GEMM/quant matmul variants,
// vision (Col2Im / MaxUnpool3d / NonMaxSuppression / CIoU), and misc (Logdet / PdistForward /
// NpuFormatCast / AdvanceStep / Dropout{Gen,Do}Mask / RReluWithNoise / TransSparse4to2Para / ...).
//
// All host-side over unified memory (MTLStorageModeShared: device pointer == host pointer). cblas backs
// the GEMM/FFN paths; LAPACK (self-declared, classic interface — same pattern as linalg.mm/blas_parity.mm to
// avoid the vecLib __LAPACK_int header conflict) backs Logdet's LU. Every op follows the standard two-phase
// aclnn contract: GetWorkspaceSize stashes the plan in aclOpExecutor and sets *ws=0; Execute drains the
// stream, computes, deletes the executor. Semantics mirror the CUDA reference (cuda/src/ops/*.cu).
//
// Declarations: the great majority of these symbols are declared in the canonical aclnnop/aclnn_ops.h and
// are NOT re-declared here. The three distributed/sparse fusions whose canonical headers are not shipped
// locally (AlltoAllvGroupedMatMul, GroupedMatMulAlltoAllv, TransSparse4to2Para) are self-declared extern "C"
// below from their CUDA semantics — aclnn_ops.h is left untouched.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "hccl/hccl.h"   // HcclComm for the canonical MC2 grouped-matmul signatures (ignored on a single rank)
#include <vecLib/cblas.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

// Self-declared classic LAPACK (Accelerate provides the symbols; self-declaring avoids the __LAPACK_int
// prototype clash that the vecLib lapack.h umbrella would introduce — identical approach to linalg.mm).
extern "C" {
void sgetrf_(const int*, const int*, float*, const int*, int*, int*);
}

// ---- self-declared CUDA-only ops (NOT in aclnn_ops.h) ----
extern "C" {
aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnAlltoAllvGroupedMatMul(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnGroupedMatMulAlltoAllv(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnTransSparse4to2ParaGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnTransSparse4to2Para(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
}

namespace {
float   *FP (const aclTensor *t) { return t ? (float   *)t->data + t->offset : nullptr; }
int64_t *I64(const aclTensor *t) { return t ? (int64_t *)t->data + t->offset : nullptr; }
uint8_t *U8 (const aclTensor *t) { return t ? (uint8_t *)t->data + t->offset : nullptr; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// counter-based hash RNG mirroring cuda/src/ops/random.cu & misc_ext.cu (reproducible: seed+index -> [0,1)).
inline uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
inline float u01(uint64_t seed, int64_t i) {
    uint32_t h = hash32((uint32_t)seed ^ hash32((uint32_t)i ^ (uint32_t)(i >> 32) ^ (uint32_t)(seed >> 32)));
    return (h >> 8) * (1.0f / 16777216.0f);
}
inline float hashu(uint64_t x) { return (hash32((uint32_t)x ^ hash32((uint32_t)(x >> 32))) >> 8) * (1.0f / 16777216.0f); }

enum {
    // norm
    K_ADALN = 1, K_BNREDUCE, K_GNACT, K_LNIMPL, K_VECNORM, K_NORMAL, K_SYNCBN_GATHER,
    // loss
    K_KLGRAD, K_MSE, K_MLML, K_NLL2D, K_SCE,
    // optim
    K_ADAMW, K_EMAADAM,
    // matmul / quant
    K_GMM, K_BMM, K_FFN, K_TBMM, K_COPY,
    // vision
    K_COL2IM, K_MAXUNPOOL3D, K_NMS, K_IOU,
    // misc
    K_ADVANCE, K_GENMASK, K_DOMASK, K_DROPOUT, K_LOGDET, K_PDIST, K_RRELU,
};

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    // ============================== NORM ==============================
    case K_ADALN: {   // AdaLayerNormV2: y = LayerNorm(x over last dim D)*(1+scale) + shift; per-row [rows,D]
        const aclTensor *X = e->a; int nd = (int)X->viewDims.size(); int64_t D = X->viewDims[nd - 1];
        int64_t rows = X->numel() / D; const float *x = FP(X), *sc = FP(e->b), *sh = FP(e->c); float *y = FP(e->out);
        double eps = e->eps;
        for (int64_t r = 0; r < rows; r++) { const float *xr = x + r * D;
            double mean = 0; for (int64_t d = 0; d < D; d++) mean += xr[d]; mean /= D;
            double var = 0; for (int64_t d = 0; d < D; d++) { double u = xr[d] - mean; var += u * u; } var /= D;
            double inv = 1.0 / std::sqrt(var + eps);
            for (int64_t d = 0; d < D; d++) { double n = (xr[d] - mean) * inv;
                y[r * D + d] = (float)(n * (1.0 + (sc ? sc[r * D + d] : 0.0)) + (sh ? sh[r * D + d] : 0.0)); }
        }
        return ACLNN_SUCCESS;
    }
    case K_BNREDUCE: {   // BatchNormReduce: per-channel sumDy / sumDyXmu / gradWeight / gradBias over [N,C,HW]
        const aclTensor *GO = e->a, *X = e->b; int64_t N = X->viewDims[0], C = X->viewDims[1];
        int64_t HW = X->numel() / (N * C); const float *go = FP(GO), *x = FP(X), *mean = FP(e->mean), *inv = FP(e->rstd);
        float *sumDy = FP(e->out), *sumDyXmu = FP(e->out2);
        float *gradW = e->inputs.size() > 0 ? FP(e->inputs[0]) : nullptr;
        float *gradB = e->inputs.size() > 1 ? FP(e->inputs[1]) : nullptr;
        for (int64_t c = 0; c < C; c++) { double a = 0, b = 0;
            for (int64_t n = 0; n < N; n++) for (int64_t h = 0; h < HW; h++) { int64_t idx = (n * C + c) * HW + h;
                double dy = go[idx]; a += dy; b += dy * ((double)x[idx] - mean[c]); }
            if (sumDy) sumDy[c] = (float)a; if (sumDyXmu) sumDyXmu[c] = (float)b;
            if (gradW) gradW[c] = (float)(b * inv[c]); if (gradB) gradB[c] = (float)a;
        }
        return ACLNN_SUCCESS;
    }
    case K_GNACT: {   // GroupNormSiluV2 / GroupNormSwish: GroupNorm over [N,C,HW] then optional swish y*sig(act*y)
        const aclTensor *X = e->a; int64_t C = X->viewDims[1], G = e->reduceCount, Cg = C / G;
        int64_t N = X->viewDims[0], HW = X->numel() / (N * C), cnt = Cg * HW; double eps = e->eps, act = e->alpha;
        const float *x = FP(X), *g = FP(e->b), *b = FP(e->c); float *o = FP(e->out);
        float *meanO = e->mean ? FP((aclTensor *)e->mean) : nullptr, *rstdO = e->rstd ? FP((aclTensor *)e->rstd) : nullptr;
        for (int64_t blk = 0; blk < N * G; blk++) { int64_t n = blk / G, grp = blk % G, base = (n * C + grp * Cg) * HW;
            double sum = 0, sq = 0; for (int64_t i = 0; i < cnt; i++) { double v = x[base + i]; sum += v; sq += v * v; }
            double mean = sum / cnt, var = sq / cnt - mean * mean, inv = 1.0 / std::sqrt(var + eps);
            if (meanO) meanO[blk] = (float)mean; if (rstdO) rstdO[blk] = (float)inv;
            for (int64_t i = 0; i < cnt; i++) { int64_t c = grp * Cg + i / HW; double gg = g ? g[c] : 1.0, bb = b ? b[c] : 0.0;
                double y = ((double)x[base + i] - mean) * inv * gg + bb;
                if (act >= 0.0) y = y / (1.0 + std::exp(-act * y));   // swish; act==1 == SiLU; act<0 == no activation
                o[base + i] = (float)y; }
        }
        return ACLNN_SUCCESS;
    }
    case K_LNIMPL: {   // LayerNormWithImplMode: LayerNorm over last dim, optional gamma/beta; implMode ignored (fp32)
        const aclTensor *X = e->a; int64_t D = e->k; int64_t rows = X->numel() / D; double eps = e->eps;
        const float *x = FP(X), *g = FP(e->b), *b = FP(e->c); float *o = FP(e->out);
        float *meanO = e->mean ? FP((aclTensor *)e->mean) : nullptr, *rstdO = e->rstd ? FP((aclTensor *)e->rstd) : nullptr;
        for (int64_t r = 0; r < rows; r++) { const float *xr = x + r * D;
            double mean = 0; for (int64_t d = 0; d < D; d++) mean += xr[d]; mean /= D;
            double var = 0; for (int64_t d = 0; d < D; d++) { double u = xr[d] - mean; var += u * u; } var /= D;
            double inv = 1.0 / std::sqrt(var + eps); if (meanO) meanO[r] = (float)mean; if (rstdO) rstdO[r] = (float)inv;
            for (int64_t d = 0; d < D; d++) o[r * D + d] = (float)(((xr[d] - mean) * inv) * (g ? g[d] : 1.0) + (b ? b[d] : 0.0));
        }
        return ACLNN_SUCCESS;
    }
    case K_VECNORM: {   // LinalgVectorNorm: out = (sum |x|^p over reduce dims)^(1/p). axes=reduce dims, p in eps.
        const aclTensor *X = e->a; aclTensor *O = e->out; double p = e->eps; int nd = (int)X->viewDims.size();
        const float *x = FP(X); float *o = FP(O); int64_t outN = O->numel();
        // reduce-mask over input dims
        std::vector<char> red(nd, 0); if (e->axes.empty()) for (int i = 0; i < nd; i++) red[i] = 1;
        else for (int64_t a : e->axes) { int d = (int)a; if (d < 0) d += nd; red[d] = 1; }
        // strides (contiguous input)
        std::vector<int64_t> st(nd); { int64_t acc = 1; for (int i = nd - 1; i >= 0; i--) { st[i] = acc; acc *= X->viewDims[i]; } }
        std::vector<double> acc(outN, 0.0);
        int64_t total = X->numel();
        for (int64_t lin = 0; lin < total; lin++) {
            // decode multi-index, build output linear index (skip reduced dims)
            int64_t rem = lin, oidx = 0;
            for (int d = 0; d < nd; d++) { int64_t coord = rem / st[d]; rem %= st[d]; if (!red[d]) oidx = oidx * X->viewDims[d] + coord; }
            acc[oidx] += std::pow(std::fabs((double)x[lin]), p);
        }
        for (int64_t i = 0; i < outN; i++) o[i] = (float)std::pow(acc[i], 1.0 / p);
        return ACLNN_SUCCESS;
    }
    case K_NORMAL: {   // NormalFloatTensor / NormalTensorFloat: out = m + s*z, z~Box-Muller; m/s scalar or tensor (b=mean,c=std)
        aclTensor *O = e->out; int64_t n = O->numel(); float *o = FP(O);
        double meanS = e->dscalars[0], stdS = e->dscalars[1]; const float *meanT = FP(e->b), *stdT = FP(e->c);
        uint64_t seed = (uint64_t)e->m;
        for (int64_t i = 0; i < n; i++) { float u1 = std::max(u01(seed, 2 * i), 1e-7f), u2 = u01(seed, 2 * i + 1);
            double z = std::sqrt(-2.0 * std::log(u1)) * std::cos(6.2831853 * u2);
            double m = meanT ? meanT[i] : meanS, sd = stdT ? stdT[i] : stdS; o[i] = (float)(m + sd * z); }
        return ACLNN_SUCCESS;
    }
    case K_SYNCBN_GATHER: {   // SyncBatchNormGatherStats: combine per-group (mean,invstd,count)[L,C] -> global mean/invstd[C]
        const aclTensor *MA = e->a, *IA = e->b, *CN = e->c; int64_t L = MA->viewDims[0], C = MA->viewDims[1];
        const float *means = FP(MA), *invstds = FP(IA), *counts = FP(CN); float *mean = FP(e->out), *invstd = FP(e->out2);
        double eps = e->eps;
        for (int64_t c = 0; c < C; c++) { double tot = 0, mAcc = 0;
            for (int64_t l = 0; l < L; l++) { double cnt = counts[l]; tot += cnt; mAcc += cnt * means[l * C + c]; }
            double m = tot > 0 ? mAcc / tot : 0.0, m2 = 0;
            for (int64_t l = 0; l < L; l++) { double cnt = counts[l], mi = means[l * C + c];
                double iv = invstds[l * C + c]; double vi = 1.0 / (iv * iv) - eps; m2 += cnt * (vi + (mi - m) * (mi - m)); }
            double var = tot > 0 ? m2 / tot : 0.0; mean[c] = (float)m; invstd[c] = (float)(1.0 / std::sqrt(var + eps));
        }
        return ACLNN_SUCCESS;
    }
    // ============================== LOSS ==============================
    case K_KLGRAD: {   // Dense/SparseLightningIndexerGradKLLoss: gradScore[q,k] = softmax(indexScore[q,:])_k - target
        const aclTensor *S = e->a, *T = e->b; aclTensor *G = e->out; int nd = (int)S->viewDims.size();
        int64_t K = S->viewDims[nd - 1], Q = S->numel() / K; const float *p = FP(S), *t = FP(T); float *g = FP(G);
        for (int64_t q = 0; q < Q; q++) { const float *pr = p + q * K, *tr = t + q * K; float *gr = g + q * K;
            double mx = -1e30; for (int64_t k = 0; k < K; k++) mx = std::max(mx, (double)pr[k]);
            double sm = 0; for (int64_t k = 0; k < K; k++) sm += std::exp(pr[k] - mx);
            for (int64_t k = 0; k < K; k++) gr[k] = (float)(std::exp(pr[k] - mx) / sm - tr[k]); }
        return ACLNN_SUCCESS;
    }
    case K_MSE: {   // MseLossOut: (pred-target)^2 elementwise (reduction 0) or reduced (1=mean,2=sum) in m
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); int red = (int)e->m; float *o = FP(e->out);
        if (red == 0) { for (int64_t i = 0; i < n; i++) { double d = (double)p[i] - t[i]; o[i] = (float)(d * d); } }
        else { double s = 0; for (int64_t i = 0; i < n; i++) { double d = (double)p[i] - t[i]; s += d * d; }
            o[0] = (float)(red == 1 ? s / n : s); }
        return ACLNN_SUCCESS;
    }
    case K_MLML: {   // MultilabelMarginLoss: per-sample sum_violations max(0,1-(score[pos]-score[neg]))/C, target padded -1
        const aclTensor *X = e->a; int64_t N = X->viewDims[0], C = X->viewDims[1]; const float *x = FP(X);
        const int64_t *t = I64(e->b); aclTensor *O = e->out; int red = (int)e->m; std::vector<double> per(N);
        for (int64_t n = 0; n < N; n++) { const float *pr = x + n * C; const int64_t *tt = t + n * C; double loss = 0;
            for (int64_t j = 0; j < C; j++) { int64_t tj = tt[j]; if (tj < 0) break;
                for (int64_t i = 0; i < C; i++) { bool isT = false;
                    for (int64_t k = 0; k < C; k++) { if (tt[k] < 0) break; if (tt[k] == i) { isT = true; break; } }
                    if (isT) continue; double z = 1.0 - ((double)pr[tj] - pr[i]); if (z > 0) loss += z; } }
            per[n] = loss / C; }
        float *o = FP(O);
        if (red == 0) { for (int64_t n = 0; n < N; n++) o[n] = (float)per[n]; }
        else { double s = 0; for (int64_t n = 0; n < N; n++) s += per[n]; o[0] = (float)(red == 1 ? s / N : s); }
        return ACLNN_SUCCESS;
    }
    case K_NLL2D: {   // NLLLoss2d: out[n] = -weight[target[n]] * logProb[n,target[n]] (per-sample; reduction in m)
        const aclTensor *X = e->a; int64_t N = X->viewDims[0], C = X->viewDims[1]; const float *x = FP(X);
        const int64_t *t = I64(e->b); const float *w = FP(e->c); aclTensor *O = e->out; int red = (int)e->m;
        int64_t ignore = e->n; float *o = FP(O); double sumLoss = 0, sumW = 0; std::vector<double> per(N, 0.0);
        for (int64_t n = 0; n < N; n++) { int64_t tt = t[n]; if (tt == ignore) { per[n] = 0; continue; }
            double wt = w ? w[tt] : 1.0; per[n] = -wt * x[n * C + tt]; sumLoss += per[n]; sumW += wt; }
        if (red == 0) { for (int64_t n = 0; n < N; n++) o[n] = (float)per[n]; }
        else o[0] = (float)(red == 1 ? (sumW > 0 ? sumLoss / sumW : 0.0) : sumLoss);
        if (e->out2) FP(e->out2)[0] = (float)sumW;   // totalWeight
        return ACLNN_SUCCESS;
    }
    case K_SCE: {   // SoftmaxCrossEntropyWithLogits: loss[n]=-Σ labels*logsoftmax(logits); backprop=softmax-labels
        const aclTensor *X = e->a, *Tg = e->b; int64_t N = X->viewDims[0], C = X->viewDims[1];
        const float *x = FP(X), *t = FP(Tg); float *loss = FP(e->out), *bp = FP(e->out2);
        for (int64_t n = 0; n < N; n++) { const float *pr = x + n * C, *tr = t + n * C;
            double mx = -1e30; for (int64_t c = 0; c < C; c++) mx = std::max(mx, (double)pr[c]);
            double sm = 0; for (int64_t c = 0; c < C; c++) sm += std::exp(pr[c] - mx);
            double l = 0; for (int64_t c = 0; c < C; c++) { double lsm = (pr[c] - mx) - std::log(sm); l += -(double)tr[c] * lsm; }
            loss[n] = (float)l;
            for (int64_t c = 0; c < C; c++) bp[n * C + c] = (float)(std::exp(pr[c] - mx) / sm - tr[c]); }
        return ACLNN_SUCCESS;
    }
    // ============================== OPTIM ==============================
    case K_ADAMW: {   // ApplyAdamWV2: in-place AdamW with decoupled weight decay; param=a,m=out,v=out2,grad=b
        float *p = FP((aclTensor *)e->a), *m = FP(e->out), *v = FP(e->out2); const float *g = FP(e->b);
        int64_t n = e->a->numel(); auto &d = e->dscalars; double lr = d[0], b1 = d[1], b2 = d[2], eps = d[3], wd = d[4]; int step = (int)e->m;
        double bc1 = 1.0 - std::pow(b1, step), bc2 = 1.0 - std::pow(b2, step);
        for (int64_t i = 0; i < n; i++) { double gi = g[i]; double mi = b1 * m[i] + (1 - b1) * gi, vi = b2 * v[i] + (1 - b2) * gi * gi;
            m[i] = (float)mi; v[i] = (float)vi; double mh = mi / bc1, vh = vi / bc2;
            p[i] = (float)(p[i] - lr * (mh / (std::sqrt(vh) + eps) + wd * p[i])); }
        return ACLNN_SUCCESS;
    }
    case K_EMAADAM: {   // ApplyFusedEmaAdam: AdamW step then ema = emaDecay*ema + (1-emaDecay)*param. ema=c.
        float *p = FP((aclTensor *)e->a), *m = FP(e->out), *v = FP(e->out2), *ema = FP((aclTensor *)e->c); const float *g = FP(e->b);
        int64_t n = e->a->numel(); auto &d = e->dscalars; double lr = d[0], b1 = d[1], b2 = d[2], eps = d[3], wd = d[4], ed = d[5]; int step = (int)e->m;
        double bc1 = 1.0 - std::pow(b1, step), bc2 = 1.0 - std::pow(b2, step);
        for (int64_t i = 0; i < n; i++) { double gi = g[i]; double mi = b1 * m[i] + (1 - b1) * gi, vi = b2 * v[i] + (1 - b2) * gi * gi;
            m[i] = (float)mi; v[i] = (float)vi; double pp = p[i] - lr * ((mi / bc1) / (std::sqrt(vi / bc2) + eps) + wd * p[i]);
            p[i] = (float)pp; ema[i] = (float)(ed * ema[i] + (1 - ed) * pp); }
        return ACLNN_SUCCESS;
    }
    // ============================== MATMUL / QUANT ==============================
    case K_GMM: {   // grouped matmul: x[M,K] partitioned by groupList @ weight[E,K,N] -> out[M,N]
        const aclTensor *X = e->a, *W = e->b; aclTensor *O = e->out;
        int K = (int)X->viewDims[1], N = (int)W->viewDims[2]; int E = (int)e->axes.size(); int64_t off = 0;
        const float *x = FP(X), *w = FP(W); float *o = FP(O);
        for (int gi = 0; gi < E; gi++) { int rows = (int)e->axes[gi];
            if (rows > 0) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, N, K,
                1.0f, x + off * K, K, w + (size_t)gi * K * N, N, 0.0f, o + off * N, N);
            off += rows; }
        return ACLNN_SUCCESS;
    }
    case K_BMM: {   // BatchMatMulWeightNz: x[B,M,K] @ weight[B,K,N] -> out[B,M,N]
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
        int Bn = (int)A->viewDims[0], M = (int)A->viewDims[1], K = (int)A->viewDims[2], N = (int)B->viewDims[2];
        const float *a = FP(A), *b = FP(B); float *o = FP(O);
        for (int bi = 0; bi < Bn; bi++) cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
            1.0f, a + (size_t)bi * M * K, K, b + (size_t)bi * K * N, N, 0.0f, o + (size_t)bi * M * N, N);
        return ACLNN_SUCCESS;
    }
    case K_TBMM: {   // TransposeBatchMatMulWeightNZ: self @ other^T (transpose other's last two dims).
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int rb = (int)B->viewDims.size();
        int64_t P = B->viewDims[rb - 2], Q = B->viewDims[rb - 1]; int64_t Bn = 1; for (int i = 0; i < rb - 2; i++) Bn *= B->viewDims[i];
        int ra = (int)A->viewDims.size(); int M = (int)A->viewDims[ra - 2], Kk = (int)A->viewDims[ra - 1];
        // other^T: [Bn,Q,P], so out = self[*,M,Kk] @ otherT[*,Kk,?]. Kk must equal Q; N = P.
        const float *a = FP(A), *b = FP(B); float *o = FP(O); int64_t Ba = (ra == 2) ? 1 : Bn;
        for (int64_t bi = 0; bi < Ba; bi++) {
            const float *ai = a + (size_t)bi * M * Kk; const float *bi_ = b + (size_t)bi * P * Q; float *oi = o + (size_t)bi * M * P;
            // out[m,p] = sum_k self[m,k]*other[p,k]  (other row-major [P,Q], Q==Kk) -> use CblasTrans on B
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, (int)P, Kk, 1.0f, ai, Kk, bi_, (int)Q, 0.0f, oi, (int)P);
        }
        return ACLNN_SUCCESS;
    }
    case K_FFN: {   // FFNV2/V3: out = act(x@W1+b1) @ W2 + b2; x[M,K] W1[K,Hd] b1[Hd] W2[Hd,N] b2[N]; act 0 relu 1 gelu 2 silu
        // inputs = {weight2, bias1, bias2} (bias1/bias2 may be nullptr).
        const aclTensor *X = e->a, *W1 = e->b, *W2 = e->inputs[0]; aclTensor *O = e->out;
        const aclTensor *B1 = e->inputs[1], *B2 = e->inputs[2];
        int M = (int)X->viewDims[0], K = (int)X->viewDims[1], Hd = (int)W1->viewDims[1], N = (int)W2->viewDims[1];
        int act = (int)e->m; const float *x = FP(X), *w1 = FP(W1), *w2 = FP(W2), *b1 = FP(B1), *b2 = FP(B2); float *o = FP(O);
        std::vector<float> h((size_t)M * Hd);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, Hd, K, 1.0f, x, K, w1, Hd, 0.0f, h.data(), Hd);
        for (int r = 0; r < M; r++) for (int j = 0; j < Hd; j++) { double acc = h[(size_t)r * Hd + j] + (b1 ? b1[j] : 0.0); double a;
            if (act == 0) a = std::max(acc, 0.0); else if (act == 1) a = 0.5 * acc * std::erfc(-acc * 0.70710678); else a = acc * sigm(acc);
            h[(size_t)r * Hd + j] = (float)a; }
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, Hd, 1.0f, h.data(), Hd, w2, N, 0.0f, o, N);
        if (b2) for (int r = 0; r < M; r++) for (int n = 0; n < N; n++) o[(size_t)r * N + n] += b2[n];
        return ACLNN_SUCCESS;
    }
    case K_COPY: {   // NpuFormatCast / TransSparse4to2Para: value-preserving copy (layout cast, no value change)
        const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = std::min(A->numel(), O->numel());
        std::memcpy(O->data, A->data, (size_t)n * dtype_size(A->dtype));
        return ACLNN_SUCCESS;
    }
    // ============================== VISION ==============================
    case K_COL2IM: {   // Col2Im: fold columns [N, C*kh*kw, oH*oW] -> image out[N,C,H,W]; overlap accumulates
        aclTensor *O = e->out; int64_t N = O->viewDims[0], C = O->viewDims[1], H = O->viewDims[2], W = O->viewDims[3];
        int64_t kh = e->axes[0], kw = e->axes[1], sh = e->axes[2], sw = e->axes[3], ph = e->axes[4], pw = e->axes[5], dh = e->axes[6], dw = e->axes[7];
        int64_t oH = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1, oW = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1;
        int64_t L = oH * oW, Kr = C * kh * kw; const float *cols = FP(e->a); float *o = FP(O);
        std::memset(o, 0, (size_t)O->numel() * sizeof(float));
        int64_t total = N * Kr * L;
        for (int64_t i = 0; i < total; i++) { int64_t l = i % L, krow = (i / L) % Kr, n = i / (L * Kr);
            int64_t ow = l % oW, oh = l / oW; int64_t kj = krow % kw, ki = (krow / kw) % kh, c = krow / (kh * kw);
            int64_t ih = oh * sh - ph + ki * dh, iw = ow * sw - pw + kj * dw;
            if (ih >= 0 && ih < H && iw >= 0 && iw < W) o[((n * C + c) * H + ih) * W + iw] += cols[i]; }
        return ACLNN_SUCCESS;
    }
    case K_MAXUNPOOL3D: {   // MaxUnpool3d: scatter self -> out (zeros elsewhere) at per-(N*C) flat spatial indices
        const aclTensor *S = e->a, *I = e->b; aclTensor *O = e->out; int64_t NC = O->viewDims[0] * O->viewDims[1];
        int64_t srcSp = S->numel() / NC, outSp = O->numel() / NC; const float *s = FP(S); const int64_t *idx = I64(I); float *o = FP(O);
        std::memset(o, 0, (size_t)O->numel() * sizeof(float));
        for (int64_t nc = 0; nc < NC; nc++) for (int64_t j = 0; j < srcSp; j++) { int64_t id = idx[nc * srcSp + j];
            if (id >= 0 && id < outSp) o[nc * outSp + id] = s[nc * srcSp + j]; }
        return ACLNN_SUCCESS;
    }
    case K_NMS: {   // NonMaxSuppression: greedy NMS on boxes[M,4]=(x1,y1,x2,y2) by descending score; keep idx + count
        const aclTensor *Boxes = e->a, *Sc = e->b; aclTensor *Keep = e->out, *Cnt = e->out2;
        int64_t M = Boxes->viewDims[0]; double iou = e->alpha; const float *boxes = FP(Boxes), *scores = FP(Sc);
        int64_t *keep = I64(Keep); int64_t *cnt = I64(Cnt);
        std::vector<int64_t> order(M); for (int64_t i = 0; i < M; i++) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](int64_t a, int64_t b) { return scores[a] > scores[b]; });
        std::vector<char> removed(M, 0); int64_t kc = 0;
        for (int64_t a = 0; a < M; a++) { int64_t i = order[a]; if (removed[i]) continue; keep[kc++] = i;
            float ax1 = boxes[i*4], ay1 = boxes[i*4+1], ax2 = boxes[i*4+2], ay2 = boxes[i*4+3]; float aarea = (ax2-ax1)*(ay2-ay1);
            for (int64_t bb = a + 1; bb < M; bb++) { int64_t j = order[bb]; if (removed[j]) continue;
                float bx1 = boxes[j*4], by1 = boxes[j*4+1], bx2 = boxes[j*4+2], by2 = boxes[j*4+3];
                float ix1 = std::max(ax1,bx1), iy1 = std::max(ay1,by1), ix2 = std::min(ax2,bx2), iy2 = std::min(ay2,by2);
                float iw = std::max(0.f, ix2-ix1), ih = std::max(0.f, iy2-iy1), inter = iw*ih;
                float barea = (bx2-bx1)*(by2-by1), u = aarea + barea - inter;
                if (u > 0 && inter / u > iou) removed[j] = 1; } }
        cnt[0] = kc;
        return ACLNN_SUCCESS;
    }
    case K_IOU: {   // CIoU: per box-pair complete IoU on boxes[N,4]=(x1,y1,x2,y2). e->dim=1 -> CIoU penalty
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int64_t N = A->viewDims[0]; int ciou = (int)e->dim;
        const float *a = FP(A), *b = FP(B); float *o = FP(O);
        for (int64_t i = 0; i < N; i++) { const float *A4 = a + i*4, *B4 = b + i*4;
            float ix1 = std::max(A4[0],B4[0]), iy1 = std::max(A4[1],B4[1]), ix2 = std::min(A4[2],B4[2]), iy2 = std::min(A4[3],B4[3]);
            float iw = std::max(ix2-ix1,0.f), ih = std::max(iy2-iy1,0.f), inter = iw*ih;
            float ua = (A4[2]-A4[0])*(A4[3]-A4[1]) + (B4[2]-B4[0])*(B4[3]-B4[1]) - inter; float v = ua > 0 ? inter/ua : 0.f;
            if (ciou) { float cxa=(A4[0]+A4[2])*.5f, cya=(A4[1]+A4[3])*.5f, cxb=(B4[0]+B4[2])*.5f, cyb=(B4[1]+B4[3])*.5f;
                float d2 = (cxa-cxb)*(cxa-cxb)+(cya-cyb)*(cya-cyb);
                float cx1=std::min(A4[0],B4[0]), cy1=std::min(A4[1],B4[1]), cx2=std::max(A4[2],B4[2]), cy2=std::max(A4[3],B4[3]);
                float c2 = (cx2-cx1)*(cx2-cx1)+(cy2-cy1)*(cy2-cy1); v = v - (c2 > 0 ? d2/c2 : 0.f); }
            o[i] = v; }
        return ACLNN_SUCCESS;
    }
    // ============================== MISC ==============================
    case K_ADVANCE: {   // AdvanceStep(+V2): int64 positions += 1
        const int64_t *in = I64(e->a); int64_t *o = I64(e->out); int64_t n = e->out->numel();
        for (int64_t i = 0; i < n; i++) o[i] = in[i] + 1;
        return ACLNN_SUCCESS;
    }
    case K_GENMASK: {   // DropoutGenMask(+V2/V2Tensor): uint8 mask[i] = u01(seed,i) < keep ? 1 : 0; keep in alpha
        uint8_t *o = U8(e->out); int64_t n = e->out->numel(); float keep = (float)e->alpha; uint64_t seed = (uint64_t)e->m;
        for (int64_t i = 0; i < n; i++) o[i] = u01(seed, i) < keep ? 1 : 0;
        return ACLNN_SUCCESS;
    }
    case K_DOMASK: {   // DropoutDoMask: out[i] = mask[i] ? x[i]/keep : 0; keep in alpha
        const float *x = FP(e->a); const uint8_t *m = U8((aclTensor *)e->mask); float *o = FP(e->out); int64_t n = e->out->numel(); float keep = (float)e->alpha;
        for (int64_t i = 0; i < n; i++) o[i] = m[i] ? x[i] / keep : 0.f;
        return ACLNN_SUCCESS;
    }
    case K_DROPOUT: {   // DropoutV3: fused; keep=u01<1-p; out=keep?x/(1-p):0; mask=keep. keep in alpha.
        const float *x = FP(e->a); float *o = FP(e->out); uint8_t *m = e->out2 ? U8(e->out2) : nullptr;
        int64_t n = e->out->numel(); float keepp = (float)e->alpha; uint64_t seed = (uint64_t)e->m;
        for (int64_t i = 0; i < n; i++) { int k = u01(seed, i) < keepp ? 1 : 0; if (m) m[i] = (uint8_t)k; o[i] = k ? x[i] / keepp : 0.f; }
        return ACLNN_SUCCESS;
    }
    case K_LOGDET: {   // Logdet: out = log|det(A)| via LU (sum log|U_ii|). A square [n,n] (or batched [...,n,n]).
        const aclTensor *A = e->a; aclTensor *O = e->out; int nd = (int)A->viewDims.size();
        int n = (int)A->viewDims[nd - 1]; int64_t batch = A->numel() / ((int64_t)n * n);
        const float *a = FP(A); float *o = FP(O);
        for (int64_t bi = 0; bi < batch; bi++) {
            std::vector<float> col(n * n);   // row-major[n,n] -> column-major
            const float *ab = a + bi * n * n; for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) col[i + j * n] = ab[i * n + j];
            std::vector<int> piv(n); int info = 0; sgetrf_(&n, &n, col.data(), &n, piv.data(), &info);
            double s = 0; if (info < 0) return ACLNN_ERR_PARAM_INVALID;
            for (int i = 0; i < n; i++) s += std::log(std::fabs((double)col[i + i * n]));
            o[bi] = (float)s;
        }
        return ACLNN_SUCCESS;
    }
    case K_PDIST: {   // PdistForward: condensed pairwise Lp distances within X[N,D] -> out[N*(N-1)/2], p in eps
        const aclTensor *X = e->a; int64_t N = X->viewDims[0], D = X->viewDims[1]; double p = e->eps;
        const float *x = FP(X); float *o = FP(e->out); int64_t idx = 0;
        for (int64_t i = 0; i < N; i++) for (int64_t j = i + 1; j < N; j++) {
            double s = 0; for (int64_t d = 0; d < D; d++) s += std::pow(std::fabs((double)x[i*D+d] - x[j*D+d]), p);
            o[idx++] = (float)std::pow(s, 1.0 / p); }
        return ACLNN_SUCCESS;
    }
    case K_RRELU: {   // RReluWithNoise: x>=0 -> x (noise 1); x<0 -> x*nz, nz=training? U[lo,hi):0.5(lo+hi)
        const float *x = FP(e->a); float *o = FP(e->out); float *noise = e->b ? FP((aclTensor *)e->b) : nullptr;
        int64_t n = e->out->numel(); float lo = (float)e->dscalars[0], hi = (float)e->dscalars[1]; uint64_t seed = (uint64_t)e->dscalars[2]; int tr = (int)e->dim;
        for (int64_t i = 0; i < n; i++) { float v = x[i];
            if (v >= 0.f) { if (noise) noise[i] = 1.f; o[i] = v; continue; }
            float nz = tr ? (lo + (hi - lo) * hashu(seed + (uint64_t)i * 2654435761ULL)) : 0.5f * (lo + hi);
            if (noise) noise[i] = nz; o[i] = v * nz; }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
} // namespace

extern "C" {

// ============================== NORM ==============================
aclnnStatus aclnnAdaLayerNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_ADALN; e->a = x; e->b = scale; e->c = shift; e->out = out; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdaLayerNormV2)

aclnnStatus aclnnBatchNormReduceGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !self || !mean || !invstd || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_BNREDUCE; e->a = gradOut; e->b = self; e->mean = mean; e->rstd = invstd; e->out = sumDy; e->out2 = sumDyXmu;
    if (gradWeight) e->inputs.push_back(gradWeight); if (gradBias) e->inputs.push_back(gradBias); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBatchNormReduce)

aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, bool activateSilu, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || group <= 0 || self->viewDims[1] % group != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_GNACT; e->a = self; e->b = gamma; e->c = beta; e->out = out; e->mean = meanOut; e->rstd = rstdOut;
    e->reduceCount = group; e->eps = eps; e->alpha = activateSilu ? 1.0 : -1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupNormSiluV2)
aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, double swishScale, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || group <= 0 || self->viewDims[1] % group != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_GNACT; e->a = self; e->b = gamma; e->c = beta; e->out = out; e->mean = meanOut; e->rstd = rstdOut;
    e->reduceCount = group; e->eps = eps; e->alpha = swishScale; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupNormSwish)

aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, int64_t implMode, uint64_t *ws, aclOpExecutor **ex) {
    (void)implMode; if (!input || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = 1; if (normalizedShape) for (auto d : normalizedShape->v) D *= d; else D = input->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = K_LNIMPL; e->a = input; e->b = weight; e->c = bias; e->out = out; e->mean = meanOut; e->rstd = rstdOut; e->k = D; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLayerNormWithImplMode)

aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)dtype; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_VECNORM; e->a = self; e->out = out; e->eps = p; if (dim) e->axes = dim->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLinalgVectorNorm)

aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(double mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!std || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_NORMAL; e->out = out; e->c = std; e->dscalars = {mean, 0.0}; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNormalFloatTensor)
aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(const aclTensor *mean, double std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!mean || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_NORMAL; e->out = out; e->b = mean; e->dscalars = {0.0, std}; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNormalTensorFloat)

aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex) {
    if (!meanAll || !invstdAll || !counts || !mean || !invstd || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_SYNCBN_GATHER; e->a = meanAll; e->b = invstdAll; e->c = counts; e->out = mean; e->out2 = invstd; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSyncBatchNormGatherStats)

// ============================== LOSS ==============================
aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *ws, aclOpExecutor **ex) {
    if (!indexScore || !target || !gradScore || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_KLGRAD; e->a = indexScore; e->b = target; e->out = gradScore; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDenseLightningIndexerGradKLLoss)
aclnnStatus aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *ws, aclOpExecutor **ex) {
    if (!indexScore || !target || !gradScore || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_KLGRAD; e->a = indexScore; e->b = target; e->out = gradScore; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSparseLightningIndexerGradKLLoss)

aclnnStatus aclnnMseLossOutGetWorkspaceSize(const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!pred || !target || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_MSE; e->a = pred; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMseLossOut)

aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, aclTensor *isTarget, uint64_t *ws, aclOpExecutor **ex) {
    (void)isTarget; if (!self || !target || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_MLML; e->a = self; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMultilabelMarginLoss)

aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, aclTensor *out, aclTensor *totalWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !target || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_NLL2D; e->a = self; e->b = target; e->c = weight; e->out = out; e->out2 = totalWeight; e->m = reduction; e->n = ignoreIndex; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNLLLoss2d)

aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor *features, const aclTensor *labels, aclTensor *loss, aclTensor *backprop, uint64_t *ws, aclOpExecutor **ex) {
    if (!features || !labels || !loss || !backprop || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_SCE; e->a = features; e->b = labels; e->out = loss; e->out2 = backprop; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSoftmaxCrossEntropyWithLogits)

// ============================== OPTIM ==============================
aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !grad || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_ADAMW; e->a = param; e->b = grad; e->out = m; e->out2 = v; e->dscalars = {lr, beta1, beta2, eps, weightDecay}; e->m = step; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnApplyAdamWV2)
aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, aclTensor *ema, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, double emaDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !ema || !grad || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_EMAADAM; e->a = param; e->b = grad; e->c = ema; e->out = m; e->out2 = v; e->dscalars = {lr, beta1, beta2, eps, weightDecay, emaDecay}; e->m = step; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnApplyFusedEmaAdam)

// ============================== MATMUL / QUANT ==============================
aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_BMM; e->a = x; e->b = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBatchMatMulWeightNz)
aclnnStatus aclnnTransposeBatchMatMulWeightNZGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_TBMM; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTransposeBatchMatMulWeightNZ)

#define FFN_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !weight1 || !weight2 || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = K_FFN; e->a = x; e->b = weight1; e->out = out; e->m = activation; \
    e->inputs = {weight2, bias1, bias2}; \
    *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
FFN_OP(aclnnFFNV2) FFN_OP(aclnnFFNV3)

// grouped matmul + all-to-all (single-rank: all-to-all == identity, so plain grouped GEMM). Self-declared.
#define GMM_AA(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)comm; if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = K_GMM; e->a = x; e->b = weight; e->out = out; \
    if (groupList) e->axes = groupList->v; else e->axes = {x->viewDims[0]}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
GMM_AA(aclnnAlltoAllvGroupedMatMul) GMM_AA(aclnnGroupedMatMulAlltoAllv)

aclnnStatus aclnnTransSparse4to2ParaGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_COPY; e->a = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTransSparse4to2Para)

// ============================== VISION ==============================
aclnnStatus aclnnCol2ImGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, const aclIntArray *kernel, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)outputSize; if (!self || !kernel || !dilation || !padding || !stride || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_COL2IM; e->a = self; e->out = out;
    e->axes = { kernel->v[0], kernel->v[1], stride->v[0], stride->v[1], padding->v[0], padding->v[1], dilation->v[0], dilation->v[1] };
    *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCol2Im)

aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)outputSize; if (!self || !indices || !out || !ex || indices->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_MAXUNPOOL3D; e->a = self; e->b = indices; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaxUnpool3d)

aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!boxes || !scores || !keepOut || !countOut || !ex || keepOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_NMS; e->a = boxes; e->b = scores; e->out = keepOut; e->out2 = countOut; e->alpha = iouThreshold; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNonMaxSuppression)

aclnnStatus aclnnCIoUGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!boxes1 || !boxes2 || !out || !ex || boxes1->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_IOU; e->a = boxes1; e->b = boxes2; e->out = out; e->dim = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCIoU)

// ============================== MISC ==============================
aclnnStatus aclnnAdvanceStepGetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)numSeqs; (void)blockSize; if (!positions || !out || !ex || positions->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_ADVANCE; e->a = positions; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdvanceStep)
aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)numSeqs; (void)blockSize; if (!positions || !out || !ex || positions->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_ADVANCE; e->a = positions; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdvanceStepV2)

aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    (void)shape; if (!mask || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_GENMASK; e->out = mask; e->alpha = 1.0 - p; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutGenMask)
aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    (void)shape; if (!mask || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_GENMASK; e->out = mask; e->alpha = 1.0 - p; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutGenMaskV2)
aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(const aclTensor *shapeRef, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    (void)shapeRef; if (!mask || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_GENMASK; e->out = mask; e->alpha = 1.0 - p; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutGenMaskV2Tensor)

aclnnStatus aclnnDropoutDoMaskGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !mask || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_DOMASK; e->a = x; e->mask = mask; e->out = out; e->alpha = 1.0 - p; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutDoMask)

aclnnStatus aclnnDropoutV3GetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_DROPOUT; e->a = x; e->out = out; e->out2 = mask; e->alpha = 1.0 - p; e->m = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutV3)

aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_LOGDET; e->a = A; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLogdet)

aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor *self, int64_t format, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)format; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_COPY; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNpuFormatCast)

aclnnStatus aclnnPdistForwardGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = K_PDIST; e->a = self; e->out = out; e->eps = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnPdistForward)

aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(const aclTensor *self, aclTensor *noise, double lower, double upper, bool training, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_RRELU; e->a = self; e->b = noise; e->out = out; e->dscalars = {lower, upper, (double)seed}; e->dim = training ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRReluWithNoise)

} // extern "C"
