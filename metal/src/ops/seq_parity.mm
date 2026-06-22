// Sequence / expert operator gap fill (CUDA parity), host-side over unified memory.
// All ops here are small index-manipulation / recurrence / fused-norm math, run on the CPU directly
// against the shared device pointers (device ptr == host ptr under MTLStorageModeShared), after draining
// the stream. Semantics mirror cuda/src/ops/{moe,attention,rnn,ssm,misc_ext}.cu.
//
// Implemented (23):
//   MoE/MLA: MhcPre, MhcPost, MhcSinkhorn, MlaPreprocess, MlaPreprocessV2, MlaProlog, MlaPrologV2WeightNz,
//            MlaPrologV3WeightNz, MoeFinalizeRoutingV2, MoeFinalizeRoutingV3, MoeFusedTopk, MoeGatingTopK,
//            MoeGatingTopKSoftmaxV2, MoeInitRoutingV2, MoeInitRoutingV3, MoeTokenPermuteWithRoutingMap,
//            MoeTokenUnpermuteWithEp, MoeTokenUnpermuteWithRoutingMap
//   RNN/SSM: LSTM, BidirectionLSTM, BidirectionLSTMV2, RecurrentGatedDeltaRule
//   Attention: TransformBiasRescaleQkv
//
// All 23 ops ARE declared in aclnnop/aclnn_ops.h (the shipped API header), so it is #included and the
// definitions below bind to those C-linkage declarations. None are defined elsewhere in the Metal backend
// (verified: no duplicate symbols).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cstring>

namespace {
const float *FP(const aclTensor *t) { return (const float *)t->data + t->offset; }
float *FPW(const aclTensor *t) { return (float *)t->data + t->offset; }
int32_t *IP(const aclTensor *t) { return (int32_t *)t->data + t->offset; }
int64_t *LP(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
double sigf(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ============================ MhcPre / MhcPost: identity copy ============================
aclnnStatus run_copy(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *in = e->a; aclTensor *o = e->out;
    if (!in || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t n = std::min(in->numel(), o->numel());
    std::memcpy((char *)o->data + (int64_t)o->offset * dtype_size(o->dtype),
                (const char *)in->data + (int64_t)in->offset * dtype_size(in->dtype),
                (size_t)n * dtype_size(in->dtype));
    return ACLNN_SUCCESS;
}

// ============================ MhcSinkhorn: row/col-normalize iterations ============================
// out = cost; for `iters`: row-normalize (each row sums to 1 if >0), then col-normalize. tau unused (parity).
aclnnStatus run_sinkhorn(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *cost = e->a; aclTensor *o = e->out;
    if (!cost || !o || o->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t R = o->viewDims[0], C = o->viewDims[1]; int iters = (int)e->m;
    const float *src = FP(cost); float *m = FPW(o);
    for (int64_t i = 0; i < R * C; ++i) m[i] = src[i];
    for (int it = 0; it < iters; ++it) {
        for (int64_t r = 0; r < R; ++r) { double sum = 0; for (int64_t c = 0; c < C; ++c) sum += m[r * C + c]; if (sum > 0) for (int64_t c = 0; c < C; ++c) m[r * C + c] = (float)(m[r * C + c] / sum); }
        for (int64_t c = 0; c < C; ++c) { double sum = 0; for (int64_t r = 0; r < R; ++r) sum += m[r * C + c]; if (sum > 0) for (int64_t r = 0; r < R; ++r) m[r * C + c] = (float)(m[r * C + c] / sum); }
    }
    return ACLNN_SUCCESS;
}

// ============================ MLA preprocess/prolog: per-row RMSNorm + RoPE ============================
// x[rows,D], gamma[D] (optional), cos/sin[rows,D]. n = x*rms*gamma; rope rotate-half (mode 0) or interleaved (mode 1).
aclnnStatus run_rmsnorm_rope(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *gamma = e->b, *cosb = e->c, *sinb = e->mask; aclTensor *o = e->out;
    if (!x || !cosb || !sinb || !o || x->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back(), rows = x->numel() / D, half = D / 2; float eps = (float)e->eps; int mode = (int)e->dim;
    const float *X = FP(x), *G = gamma ? FP(gamma) : nullptr, *C = FP(cosb), *SN = FP(sinb); float *O = FPW(o);
    for (int64_t r = 0; r < rows; ++r) {
        const float *p = X + r * D; float *op = O + r * D; const float *cr = C + r * D, *sr = SN + r * D;
        float ss = 0; for (int64_t d = 0; d < D; ++d) ss += p[d] * p[d];
        float rms = 1.f / std::sqrt(ss / (float)D + eps);
        auto nrm = [&](int64_t d) { return p[d] * rms * (G ? G[d] : 1.f); };
        for (int64_t d = 0; d < D; ++d) {
            float n = nrm(d), c = cr[d], sn = sr[d], np;
            if (mode == 0) {
                if (d < half) np = n * c - nrm(d + half) * sn;
                else          np = n * c + nrm(d - half) * sn;
            } else {
                int64_t k = d / 2;
                if (d % 2 == 0) np = n * c - nrm(2 * k + 1) * sn;
                else            np = n * c + nrm(2 * k) * sn;
            }
            op[d] = np;
        }
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE GatingTopKSoftmax: softmax over E, top-K, renormalize ============================
// logits[T,E] -> weights[T,K], indices[T,K] (int32). Top-K by selection (descending logit, lower index on tie).
aclnnStatus run_gating(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *lg = e->a; aclTensor *w = e->out, *idx = e->out2; int64_t K = e->k;
    if (!lg || !w || !idx) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)lg->viewDims.size(); int64_t E = lg->viewDims[rank - 1], T = lg->numel() / E;
    const float *L = FP(lg); float *W = FPW(w); int32_t *I = IP(idx);
    std::vector<char> used(E);
    for (int64_t t = 0; t < T; ++t) {
        const float *row = L + t * E;
        double mx = -1e30; for (int64_t x = 0; x < E; ++x) mx = std::max(mx, (double)row[x]);
        double se = 0; for (int64_t x = 0; x < E; ++x) se += std::exp((double)row[x] - mx);
        std::fill(used.begin(), used.end(), 0); double wsum = 0;
        for (int64_t k = 0; k < K; ++k) {
            int64_t best = -1; double bv = -1e30;
            for (int64_t x = 0; x < E; ++x) { if (used[x]) continue; if ((double)row[x] > bv) { bv = row[x]; best = x; } }
            used[best] = 1; double p = std::exp((double)row[best] - mx) / se;
            I[t * K + k] = (int32_t)best; W[t * K + k] = (float)p; wsum += p;
        }
        for (int64_t k = 0; k < K; ++k) W[t * K + k] = (float)(W[t * K + k] / wsum);
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE InitRouting: expand (token,expert) pairs grouped by expert ============================
// x[T,H], expertIdx[T,K] (int32) -> expandedX[T*K,H], expandedRowIdx[T*K] (int32), expandedExpertIdx[T*K] (int32).
// Grouped (stable) by expert id; rowIdx[p] = original flat index t*K+k of expanded row p.
aclnnStatus run_init_routing(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *ex = e->b; aclTensor *exX = e->out, *rowI = e->out2, *expI = (aclTensor *)e->mask;
    if (!x || !ex || !exX || !rowI || !expI) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t T = x->viewDims[0], H = x->viewDims[1], K = ex->viewDims[1], M = T * K;
    const float *X = FP(x); const int32_t *EID = IP(ex);
    float *EX = FPW(exX); int32_t *RI = IP(rowI), *EI = IP(expI);
    std::vector<int64_t> perm(M); std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return EID[a] < EID[b]; });
    for (int64_t p = 0; p < M; ++p) {
        int64_t flat = perm[p], t = flat / K;
        RI[p] = (int32_t)flat; EI[p] = EID[flat];
        for (int64_t h = 0; h < H; ++h) EX[p * H + h] = X[t * H + h];
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE FinalizeRouting: weighted combine ============================
// expandedY[T*K,H], expandedRowIdx[T*K] (int32), scales[T,K] (optional) -> out[T,H].
aclnnStatus run_finalize_routing(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *ey = e->a, *rowI = e->b, *sc = e->c; aclTensor *o = e->out; int64_t K = e->k;
    if (!ey || !rowI || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t P = ey->viewDims[0], H = ey->viewDims[1], T = o->viewDims[0];
    const float *EY = FP(ey); const int32_t *RI = IP(rowI); const float *SC = sc ? FP(sc) : nullptr; float *O = FPW(o);
    for (int64_t i = 0; i < T * H; ++i) O[i] = 0.f;
    for (int64_t p = 0; p < P; ++p) {
        int64_t flat = RI[p], t = flat / K, k = flat % K; float w = SC ? SC[t * K + k] : 1.f;
        for (int64_t h = 0; h < H; ++h) O[t * H + h] += w * EY[p * H + h];
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE TokenPermute: group tokens by expert ============================
// x[T,H], expertId[T] (int32) -> permX[T,H] grouped by expert, srcIdx[T] (int64 permuted->original).
aclnnStatus run_permute(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *ex = e->b; aclTensor *px = e->out, *si = e->out2;
    if (!x || !ex || !px || !si) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t T = x->viewDims[0], H = x->numel() / T;
    const float *X = FP(x); const int32_t *EID = IP(ex); float *PX = FPW(px); int64_t *SI = LP(si);
    std::vector<int64_t> perm(T); std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return EID[a] < EID[b]; });
    for (int64_t p = 0; p < T; ++p) {
        int64_t src = perm[p]; SI[p] = src;
        for (int64_t h = 0; h < H; ++h) PX[p * H + h] = X[src * H + h];
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE TokenUnpermute: inverse scatter ============================
// permY[P,H], srcIdx[P] (int64), weight[P] (optional) -> out[T,H] (scatter, scaled by weight).
aclnnStatus run_unpermute(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *py = e->a, *si = e->b, *wt = e->c; aclTensor *o = e->out;
    if (!py || !si || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t P = py->viewDims[0], H = py->numel() / P, T = o->numel() / H;
    const float *PY = FP(py); const int64_t *SI = LP(si); const float *W = wt ? FP(wt) : nullptr; float *O = FPW(o);
    for (int64_t i = 0; i < T * H; ++i) O[i] = 0.f;
    for (int64_t p = 0; p < P; ++p) {
        int64_t dst = SI[p]; float w = W ? W[p] : 1.f;
        for (int64_t h = 0; h < H; ++h) O[dst * H + h] += PY[p * H + h] * w;
    }
    return ACLNN_SUCCESS;
}

// ============================ LSTM: single-layer forward (gates i,f,g,o; rows g*H+hh) ============================
aclnnStatus run_lstm(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->inputs[0], *wih = e->inputs[1], *whh = e->inputs[2], *bih = e->inputs[3], *bhh = e->inputs[4];
    int T = (int)x->viewDims[0], B = (int)x->viewDims[1], I = (int)x->viewDims[2], H = (int)whh->viewDims[1];
    const float *X = FP(x), *Wih = FP(wih), *Whh = FP(whh), *Bih = FP(bih), *Bhh = FP(bhh);
    const float *h0 = FP(e->inputs[5]), *c0 = FP(e->inputs[6]);
    float *Y = FPW((aclTensor *)e->inputs[7]), *hN = FPW((aclTensor *)e->inputs[8]), *cN = FPW((aclTensor *)e->inputs[9]);
    std::vector<double> h(B * H), c(B * H); for (int i = 0; i < B * H; ++i) { h[i] = h0[i]; c[i] = c0[i]; }
    for (int t = 0; t < T; ++t) {
        std::vector<double> hn(B * H), cn(B * H);
        for (int b = 0; b < B; ++b) for (int hh = 0; hh < H; ++hh) {
            double g[4];
            for (int gate = 0; gate < 4; ++gate) {
                int row = gate * H + hh; double acc = Bih[row] + Bhh[row];
                for (int i = 0; i < I; ++i) acc += (double)Wih[row * I + i] * X[(t * B + b) * I + i];
                for (int kk = 0; kk < H; ++kk) acc += (double)Whh[row * H + kk] * h[b * H + kk];
                g[gate] = acc;
            }
            double gi = sigf(g[0]), gf = sigf(g[1]), gg = std::tanh(g[2]), go = sigf(g[3]);
            double cc = gf * c[b * H + hh] + gi * gg; cn[b * H + hh] = cc;
            double hc = go * std::tanh(cc); hn[b * H + hh] = hc; Y[(t * B + b) * H + hh] = (float)hc;
        }
        h = hn; c = cn;
    }
    for (int i = 0; i < B * H; ++i) { hN[i] = (float)h[i]; cN[i] = (float)c[i]; }
    return ACLNN_SUCCESS;
}

// ============================ GatedDeltaRule: per (b*Hd) state S[Dk*Dv] ============================
// q,k[B,Hd,L,Dk], v[B,Hd,L,Dv], beta,g[B,Hd,L] -> y[B,Hd,L,Dv].
aclnnStatus run_gated_delta(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *Q = e->inputs[0], *Kk = e->inputs[1], *V = e->inputs[2], *Be = e->inputs[3], *G = e->inputs[4];
    aclTensor *Y = (aclTensor *)e->inputs[5];
    int B = (int)Q->viewDims[0], Hd = (int)Q->viewDims[1], L = (int)Q->viewDims[2], Dk = (int)Q->viewDims[3], Dv = (int)V->viewDims[3];
    const float *q = FP(Q), *k = FP(Kk), *v = FP(V), *beta = FP(Be), *g = FP(G); float *y = FPW(Y);
    for (int bh = 0; bh < B * Hd; ++bh) {
        std::vector<double> S(Dk * Dv, 0);
        for (int t = 0; t < L; ++t) {
            int64_t base = (int64_t)bh * L + t; const float *kt = k + base * Dk, *qt = q + base * Dk, *vt = v + base * Dv;
            double gt = g[base], bt = beta[base];
            for (int i = 0; i < Dk * Dv; ++i) S[i] *= gt;
            for (int j = 0; j < Dv; ++j) { double sk = 0; for (int i = 0; i < Dk; ++i) sk += S[i * Dv + j] * kt[i]; double kv = vt[j] - sk; for (int i = 0; i < Dk; ++i) S[i * Dv + j] += bt * kt[i] * kv; }
            for (int j = 0; j < Dv; ++j) { double yt = 0; for (int i = 0; i < Dk; ++i) yt += qt[i] * S[i * Dv + j]; y[base * Dv + j] = (float)yt; }
        }
    }
    return ACLNN_SUCCESS;
}

// ============================ TransformBiasRescaleQkv: add bias, scale q-segment ============================
// qkv flattened [3, seg, D] (q,k,v stacked); bias[3,D] (optional). out[which,seg,d] = qkv + bias; q (which==0) *= 1/sqrt(headDim).
aclnnStatus run_qkv_rescale(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *qkv = e->a, *bias = e->b; aclTensor *o = e->out;
    if (!qkv || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t seg = e->m, D = e->k; float qscale = (float)e->alpha;
    const float *X = FP(qkv), *Bs = bias ? FP(bias) : nullptr; float *O = FPW(o);
    int64_t tot = 3 * seg * D;
    for (int64_t i = 0; i < tot; ++i) {
        int64_t which = i / (seg * D), d = i % D;
        float vv = X[i] + (Bs ? Bs[which * D + d] : 0.f);
        O[i] = (which == 0) ? vv * qscale : vv;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {

// ---- MhcPre / MhcPost: identity copy ----
aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMhcPre, run_copy)
aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMhcPost, run_copy)

// ---- MhcSinkhorn ----
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)tau; if (!cost || !out || !ws || !ex || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = cost; e->out = out; e->m = iters > 0 ? iters : 5; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMhcSinkhorn, run_sinkhorn)

// ---- MLA preprocess/prolog (rmsnorm + rope) ----
#define MLA_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = cos; e->mask = sin; e->out = out; e->eps = eps; e->dim = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_rmsnorm_rope)
MLA_OP(aclnnMlaPreprocess)
MLA_OP(aclnnMlaPreprocessV2)
MLA_OP(aclnnMlaProlog)
MLA_OP(aclnnMlaPrologV2WeightNz)
MLA_OP(aclnnMlaPrologV3WeightNz)

// ---- MoE gating variants (softmax top-K) ----
#define GATING_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) { \
    if (!logits || !weights || !indices || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = logits; e->out = weights; e->out2 = indices; e->k = k; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_gating)
GATING_OP(aclnnMoeFusedTopk)
GATING_OP(aclnnMoeGatingTopK)
GATING_OP(aclnnMoeGatingTopKSoftmaxV2)

// ---- MoE InitRouting variants ----
#define INIT_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex) { \
    (void)numExperts; if (!x || !expertIdx || !expandedX || !expandedRowIdx || !expandedExpertIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = x; e->b = expertIdx; e->out = expandedX; e->out2 = expandedRowIdx; e->mask = expandedExpertIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_init_routing)
INIT_OP(aclnnMoeInitRoutingV2)
INIT_OP(aclnnMoeInitRoutingV3)

// ---- MoE FinalizeRouting variants ----
#define FINAL_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!expandedY || !expandedRowIdx || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = expandedY; e->b = expandedRowIdx; e->c = scales; e->k = k; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_finalize_routing)
FINAL_OP(aclnnMoeFinalizeRoutingV2)
FINAL_OP(aclnnMoeFinalizeRoutingV3)

// ---- MoE TokenPermuteWithRoutingMap (single-rank: == base permute) ----
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex) {
    (void)numExperts; if (!x || !expertId || !permX || !srcIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = expertId; e->out = permX; e->out2 = srcIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeTokenPermuteWithRoutingMap, run_permute)

// ---- MoE TokenUnpermute variants (single-rank: == base unpermute) ----
#define UNPERM_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!permY || !srcIdx || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = permY; e->b = srcIdx; e->c = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_unpermute)
UNPERM_OP(aclnnMoeTokenUnpermuteWithEp)
UNPERM_OP(aclnnMoeTokenUnpermuteWithRoutingMap)

// ---- LSTM / BidirectionLSTM (forward direction) ----
#define LSTM_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->inputs = {x, wih, whh, bih, bhh, h0, c0, y, hN, cN}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_lstm)
LSTM_OP(aclnnLSTM)
LSTM_OP(aclnnBidirectionLSTM)
LSTM_OP(aclnnBidirectionLSTMV2)

// ---- RecurrentGatedDeltaRule ----
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->inputs = {q, k, v, beta, g, y}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnRecurrentGatedDeltaRule, run_gated_delta)

// ---- TransformBiasRescaleQkv ----
aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(const aclTensor *qkv, const aclTensor *bias, int64_t headDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!qkv || !out || !ws || !ex || qkv->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = qkv->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = qkv; e->b = bias; e->out = out; e->k = D; e->m = qkv->numel() / (3 * D);
    e->alpha = 1.0 / std::sqrt((double)(headDim > 0 ? headDim : D)); *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnTransformBiasRescaleQkv, run_qkv_rescale)

} // extern "C"
