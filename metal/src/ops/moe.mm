// MoE routing (P15) + a couple of attention completions, host-side over unified memory.
// All ops here are pure index manipulation (gather/scatter/sort/top-k) or small attention math, so they
// run on the CPU directly against the shared device pointers (device ptr == host ptr under MTLStorageModeShared).
// Implements: MoeGatingTopKSoftmax, MoeComputeExpertTokens, MoeTokenPermute, MoeTokenUnpermute,
//   MoeInitRouting, MoeFinalizeRouting, MoeFinalizeRoutingV2Grad, MoeInitRoutingV2Grad, MoeUpdateExpert,
//   MoeTokenPermuteWithEp, ApplyRotaryPosEmbGrad, FusedInferAttentionScore, PromptFlashAttention,
//   MultiLatentAttention.
// (ApplyRotaryPosEmb fwd / FlashAttentionScore live in attention.mm and are NOT redefined here.)
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int32_t *IP(const aclTensor *t) { return (int32_t *)t->data + t->offset; }
int64_t *LP(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ---- MoeGatingTopKSoftmax: logits[T,E] -> weights[T,K], indices[T,K] ----
// top-K by logit (descending); weight = softmax_prob[k] renormalized over the K selected.
aclnnStatus run_gating(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *lg = e->a; aclTensor *w = e->out, *idx = e->out2; int64_t K = e->k;
    int64_t T = lg->viewDims[0], E = lg->viewDims[1];
    const float *L = FP(lg); float *W = FP(w); int32_t *I = IP(idx);
    std::vector<int> ord(E);
    for (int64_t t = 0; t < T; ++t) {
        const float *row = L + t * E;
        std::iota(ord.begin(), ord.end(), 0);
        // stable descending sort by logit (ties -> lower index first), matching the reference std::sort comparator
        std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return row[a] > row[b]; });
        double mx = -1e300; for (int64_t x = 0; x < E; ++x) mx = std::max(mx, (double)row[x]);
        double se = 0; for (int64_t x = 0; x < E; ++x) se += std::exp((double)row[x] - mx);
        std::vector<double> p(K); double wsum = 0;
        for (int64_t k = 0; k < K; ++k) { p[k] = std::exp((double)row[ord[k]] - mx) / se; wsum += p[k]; }
        for (int64_t k = 0; k < K; ++k) { W[t * K + k] = (float)(p[k] / wsum); I[t * K + k] = ord[k]; }
    }
    return ACLNN_SUCCESS;
}

// ---- MoeComputeExpertTokens: indices[M] (int32) -> tokensPerExpert[E] (int64), offsets[E] (int64) ----
aclnnStatus run_compute_tokens(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *idx = e->a; aclTensor *cnt = e->out, *off = e->out2; int64_t E = e->k;
    int64_t M = idx->numel(); const int32_t *I = IP(idx);
    int64_t *C = LP(cnt), *O = LP(off);
    for (int64_t x = 0; x < E; ++x) C[x] = 0;
    for (int64_t m = 0; m < M; ++m) { int x = I[m]; if (x >= 0 && x < E) C[x]++; }
    int64_t acc = 0; for (int64_t x = 0; x < E; ++x) { O[x] = acc; acc += C[x]; }
    return ACLNN_SUCCESS;
}

// stable argsort of token rows by expert id (non-decreasing); preserves source order within an expert group.
std::vector<int64_t> group_by_expert(const int32_t *eid, int64_t T) {
    std::vector<int64_t> perm(T); std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return eid[a] < eid[b]; });
    return perm;
}

// ---- MoeTokenPermute / MoeTokenPermuteWithEp: x[T,H], expertId[T] -> permX[T,H], srcIdx[T] (int64) ----
// permX[p] = x[srcIdx[p]], srcIdx = tokens sorted (stably) by expert id.
aclnnStatus run_permute(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *ex = e->b; aclTensor *px = e->out, *si = e->out2;
    int64_t T = x->viewDims[0], H = x->viewDims[1];
    const float *X = FP(x); const int32_t *EID = IP(ex);
    float *PX = FP(px); int64_t *SI = LP(si);
    auto perm = group_by_expert(EID, T);
    for (int64_t p = 0; p < T; ++p) {
        int64_t src = perm[p]; SI[p] = src;
        for (int64_t h = 0; h < H; ++h) PX[p * H + h] = X[src * H + h];
    }
    return ACLNN_SUCCESS;
}

// ---- MoeTokenUnpermute: permY[T,H], srcIdx[T] (int64), weight(opt) -> out[T,H] ----
// inverse scatter: out[srcIdx[p]] = permY[p] (scaled by weight[p] if provided).
aclnnStatus run_unpermute(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *py = e->a, *si = e->b, *wt = e->c; aclTensor *o = e->out;
    int64_t T = py->viewDims[0], H = py->viewDims[1];
    const float *PY = FP(py); const int64_t *SI = LP(si); float *O = FP(o);
    const float *W = wt ? FP(wt) : nullptr;
    for (int64_t i = 0; i < T * H; ++i) O[i] = 0.f;
    for (int64_t p = 0; p < T; ++p) {
        int64_t dst = SI[p]; float w = W ? W[p] : 1.f;
        for (int64_t h = 0; h < H; ++h) O[dst * H + h] = PY[p * H + h] * w;
    }
    return ACLNN_SUCCESS;
}

// ---- MoeInitRouting: x[T,H], expertIdx[T,K] -> expandedX[M,H], expandedRowIdx[M] (int32), expandedExpertIdx[M] (int32) ----
// M = T*K. Expand each (token,expert) pair, then group by expert id (stable). expandedRowIdx = flat pair index (0..M-1).
aclnnStatus run_init_routing(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *ex = e->b; aclTensor *exX = e->out, *rowI = e->out2, *expI = (aclTensor *)e->c;
    int64_t T = x->viewDims[0], H = x->viewDims[1];
    int64_t K = ex->viewDims[1], M = T * K;
    const float *X = FP(x); const int32_t *EID = IP(ex);
    float *EX = FP(exX); int32_t *RI = IP(rowI), *EI = IP(expI);
    std::vector<int64_t> perm(M); std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) { return EID[a] < EID[b]; });
    for (int64_t p = 0; p < M; ++p) {
        int64_t flat = perm[p]; int64_t t = flat / K;
        RI[p] = (int32_t)flat; EI[p] = EID[flat];
        for (int64_t h = 0; h < H; ++h) EX[p * H + h] = X[t * H + h];
    }
    return ACLNN_SUCCESS;
}

// ---- MoeFinalizeRouting: expandedY[M,H], expandedRowIdx[M] (int32), scales[T,K] -> out[T,H] ----
// out[t] += scales[t,k] * expandedY[p], where flat = rowIdx[p], t = flat/K, k = flat%K.
aclnnStatus run_finalize_routing(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *ey = e->a, *rowI = e->b, *sc = e->c; aclTensor *o = e->out; int64_t K = e->k;
    int64_t M = ey->viewDims[0], H = ey->viewDims[1], T = o->viewDims[0];
    const float *EY = FP(ey); const int32_t *RI = IP(rowI); const float *SC = FP(sc); float *O = FP(o);
    for (int64_t i = 0; i < T * H; ++i) O[i] = 0.f;
    for (int64_t p = 0; p < M; ++p) {
        int64_t flat = RI[p], t = flat / K, k = flat % K; float w = SC[t * K + k];
        for (int64_t h = 0; h < H; ++h) O[t * H + h] += w * EY[p * H + h];
    }
    return ACLNN_SUCCESS;
}

// ---- MoeFinalizeRoutingV2Grad: gradOut[T,H], expandedY[M,H], rowIdx[M] (int32), scales[T,K]
//        -> gradExpandedY[M,H], gradScales[T,K] ----
// gradExpY[p] = scales[t,k]*gradOut[t]; gradScales[t,k] = dot(gradOut[t], expandedY[p]).
aclnnStatus run_finalize_grad(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *go = e->a, *ey = e->b, *rowI = e->c, *sc = e->mean;
    aclTensor *gey = e->out, *gsc = e->out2; int64_t K = e->k;
    int64_t M = ey->viewDims[0], H = ey->viewDims[1];
    const float *GO = FP(go); const float *EY = FP(ey); const int32_t *RI = IP(rowI); const float *SC = FP(sc);
    float *GEY = FP(gey), *GSC = FP(gsc);
    for (int64_t p = 0; p < M; ++p) {
        int64_t flat = RI[p], t = flat / K, k = flat % K; float w = SC[t * K + k]; double dot = 0;
        for (int64_t h = 0; h < H; ++h) { GEY[p * H + h] = w * GO[t * H + h]; dot += (double)GO[t * H + h] * EY[p * H + h]; }
        GSC[t * K + k] = (float)dot;
    }
    return ACLNN_SUCCESS;
}

// ---- MoeInitRoutingV2Grad: gradExpandedX[M,H], rowIdx[M] (int32) -> gradX[T,H] ----
// gradX[t] = sum over p with rowIdx[p]/K == t of gradExpandedX[p].
aclnnStatus run_init_grad(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *gex = e->a, *rowI = e->b; aclTensor *gx = e->out; int64_t K = e->k;
    int64_t M = gex->viewDims[0], H = gex->viewDims[1], T = gx->viewDims[0];
    const float *GEX = FP(gex); const int32_t *RI = IP(rowI); float *GX = FP(gx);
    for (int64_t i = 0; i < T * H; ++i) GX[i] = 0.f;
    for (int64_t p = 0; p < M; ++p) {
        int64_t t = (int64_t)RI[p] / K;
        for (int64_t h = 0; h < H; ++h) GX[t * H + h] += GEX[p * H + h];
    }
    return ACLNN_SUCCESS;
}

// ---- MoeUpdateExpert: identity copy of expert ids (single-rank EP) ----
aclnnStatus run_update_expert(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *in = e->a; aclTensor *o = e->out; int64_t n = in->numel();
    const int32_t *I = IP(in); int32_t *O = IP(o);
    for (int64_t i = 0; i < n; ++i) O[i] = I[i];
    return ACLNN_SUCCESS;
}

// ---- ApplyRotaryPosEmbGrad: un-rotate q,k by negating sin (valid inverse for symmetric half-pair cos/sin) ----
// gradQ/gradK [B,N,S,D]; cos/sin [S,D]. out = x*cos - rotate_half(x)*sin, rotate_half = [-x[D/2:], x[:D/2]].
void rope_inverse(const aclTensor *x, const aclTensor *cosb, const aclTensor *sinb, aclTensor *o) {
    int64_t N = x->viewDims[1], S = x->viewDims[2], D = x->viewDims[3], half = D / 2;
    const float *X = FP(x), *C = FP(cosb), *SN = FP(sinb); float *O = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) {
        const float *xr = X + (n * S + sp) * D; float *orow = O + (n * S + sp) * D;
        const float *cr = C + sp * D, *sr = SN + sp * D;
        for (int64_t d = 0; d < D; ++d) {
            float rot = (d < half) ? -xr[d + half] : xr[d - half];
            orow[d] = xr[d] * cr[d] - rot * sr[d];   // inverse rotation: subtract instead of add
        }
    }
}
aclnnStatus run_rope_grad(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *gq = e->a, *gk = e->b, *cosb = e->c, *sinb = e->mean; aclTensor *gqo = e->out, *gko = e->out2;
    if (!gq || !gk || !cosb || !sinb || !gqo || !gko) return ACLNN_ERR_PARAM_NULLPTR;
    rope_inverse(gq, cosb, sinb, gqo); rope_inverse(gk, cosb, sinb, gko);
    return ACLNN_SUCCESS;
}

// ---- standard MHA over BNSD (one online softmax per (head, query pos)); GQA via Hq%Hkv==0 ----
void mha(const float *Q, const float *K, const float *V, float *O,
         int64_t B, int64_t Hq, int64_t Hkv, int64_t S, int64_t Skv, int64_t D, float scale, bool causal) {
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Hq; ++h) {
        int64_t kvh = h / (Hq / Hkv);
        for (int64_t si = 0; si < S; ++si) {
            const float *qr = Q + ((b * Hq + h) * S + si) * D;
            int64_t lim = causal ? (Skv - S + si + 1) : Skv; if (lim < 0) lim = 0; if (lim > Skv) lim = Skv;
            double m = -1e300, l = 0; std::vector<double> acc(D, 0);
            for (int64_t sj = 0; sj < lim; ++sj) {
                const float *kr = K + ((b * Hkv + kvh) * Skv + sj) * D;
                double sc = 0; for (int64_t d = 0; d < D; ++d) sc += (double)qr[d] * kr[d]; sc *= scale;
                double mn = std::max(m, sc), corr = std::exp(m - mn), p = std::exp(sc - mn);
                l = l * corr + p; const float *vr = V + ((b * Hkv + kvh) * Skv + sj) * D;
                for (int64_t d = 0; d < D; ++d) acc[d] = acc[d] * corr + p * vr[d];
                m = mn;
            }
            float *orow = O + ((b * Hq + h) * S + si) * D;
            for (int64_t d = 0; d < D; ++d) orow[d] = (float)(acc[d] / (l + 1e-20));
        }
    }
}
aclnnStatus run_attn_host(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *q = e->a, *k = e->b, *v = e->c; aclTensor *o = e->out;
    if (!q || !k || !v || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t B = q->viewDims[0], Hq = q->viewDims[1], S = q->viewDims[2], D = q->viewDims[3];
    int64_t Hkv = k->viewDims[1], Skv = k->viewDims[2];
    if (Hkv == 0 || (Hq % Hkv) != 0) return ACLNN_ERR_PARAM_INVALID;
    mha(FP(q), FP(k), FP(v), FP(o), B, Hq, Hkv, S, Skv, D, (float)e->alpha, e->causal);
    return ACLNN_SUCCESS;
}

// ---- MultiLatentAttention: up-project compressed latent KV per head, then standard causal MHA ----
// q[B,Nh,S,D], cKV[B,S,Lc], wUK/wUV[Lc, Nh*D] -> out[B,Nh,S,D].
aclnnStatus run_mla(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *q = e->a, *cKV = e->b, *wUK = e->c, *wUV = e->mean; aclTensor *o = e->out;
    if (!q || !cKV || !wUK || !wUV || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t B = q->viewDims[0], Nh = q->viewDims[1], S = q->viewDims[2], D = q->viewDims[3];
    int64_t Lc = cKV->viewDims[2], WN = Nh * D;
    const float *Q = FP(q), *C = FP(cKV), *WK = FP(wUK), *WV = FP(wUV); float *O = FP(o);
    // materialize per-head K,V: K[b,h,s,d] = sum_l cKV[b,s,l]*wUK[l, h*D+d]
    std::vector<float> Kbuf(B * Nh * S * D), Vbuf(B * Nh * S * D);
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Nh; ++h) for (int64_t sp = 0; sp < S; ++sp) {
        const float *c = C + (b * S + sp) * Lc;
        for (int64_t d = 0; d < D; ++d) {
            double ak = 0, av = 0; int64_t wc = h * D + d;
            for (int64_t l = 0; l < Lc; ++l) { float cl = c[l]; ak += (double)cl * WK[l * WN + wc]; av += (double)cl * WV[l * WN + wc]; }
            int64_t kidx = ((b * Nh + h) * S + sp) * D + d; Kbuf[kidx] = (float)ak; Vbuf[kidx] = (float)av;
        }
    }
    mha(Q, Kbuf.data(), Vbuf.data(), O, B, Nh, Nh, S, S, D, (float)e->alpha, e->causal);
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    if (!logits || !weights || !indices || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = logits; e->out = weights; e->out2 = indices; e->k = k; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeGatingTopKSoftmax, run_gating)

aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(const aclTensor *indices, int64_t numExperts, aclTensor *tokensPerExpert, aclTensor *offsets, uint64_t *ws, aclOpExecutor **ex) {
    if (!indices || !tokensPerExpert || !offsets || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = indices; e->out = tokensPerExpert; e->out2 = offsets; e->k = numExperts; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeComputeExpertTokens, run_compute_tokens)

aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex) {
    (void)numExperts; if (!x || !expertId || !permX || !srcIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = expertId; e->out = permX; e->out2 = srcIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeTokenPermute, run_permute)

aclnnStatus aclnnMoeTokenPermuteWithEpGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex) {
    (void)numExperts; if (!x || !expertId || !permX || !srcIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = expertId; e->out = permX; e->out2 = srcIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeTokenPermuteWithEp, run_permute)

aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!permY || !srcIdx || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = permY; e->b = srcIdx; e->c = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeTokenUnpermute, run_unpermute)

aclnnStatus aclnnMoeInitRoutingGetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex) {
    (void)numExperts; if (!x || !expertIdx || !expandedX || !expandedRowIdx || !expandedExpertIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = expertIdx; e->out = expandedX; e->out2 = expandedRowIdx; e->c = expandedExpertIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeInitRouting, run_init_routing)

aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!expandedY || !expandedRowIdx || !scales || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = expandedY; e->b = expandedRowIdx; e->c = scales; e->k = k; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeFinalizeRouting, run_finalize_routing)

aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *expandedY, const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *gradExpandedY, aclTensor *gradScales, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !expandedY || !expandedRowIdx || !scales || !gradExpandedY || !gradScales || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = expandedY; e->c = expandedRowIdx; e->mean = scales; e->k = k; e->out = gradExpandedY; e->out2 = gradScales; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeFinalizeRoutingV2Grad, run_finalize_grad)

aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(const aclTensor *gradExpandedX, const aclTensor *expandedRowIdx, int64_t k, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradExpandedX || !expandedRowIdx || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradExpandedX; e->b = expandedRowIdx; e->k = k; e->out = gradX; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeInitRoutingV2Grad, run_init_grad)

aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(const aclTensor *expertIds, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!expertIds || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = expertIds; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMoeUpdateExpert, run_update_expert)

aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(const aclTensor *gradQ, const aclTensor *gradK, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *gradQOut, aclTensor *gradKOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)mode; if (!gradQ || !gradK || !cos || !sin || !gradQOut || !gradKOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradQ; e->b = gradK; e->c = cos; e->mean = sin; e->out = gradQOut; e->out2 = gradKOut; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnApplyRotaryPosEmbGrad, run_rope_grad)

aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)attenMask; (void)headNum; if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->out = out; e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnFusedInferAttentionScore, run_attn_host)

aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)attenMask; (void)headNum; if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->out = out; e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnPromptFlashAttention, run_attn_host)

aclnnStatus aclnnMultiLatentAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *cKV, const aclTensor *wUK, const aclTensor *wUV, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)headNum; if (!q || !cKV || !wUK || !wUV || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = cKV; e->c = wUK; e->mean = wUV; e->out = out; e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMultiLatentAttention, run_mla)

} // extern "C"
