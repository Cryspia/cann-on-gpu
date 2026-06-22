// Linear-attention / SSM family: CausalConv1d, SelectiveScan (Mamba), GatedDeltaRule.
// Sequential state recurrences host-side over unified memory (exact) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>

namespace {
const float *fp(const aclTensor *t) { return (const float *)t->data + t->offset; }
float *fpw(const aclTensor *t) { return (float *)t->data + t->offset; }

aclnnStatus run_ssm(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    if (e->m == 0) {   // CausalConv1d: x[B,C,L], w[C,K], bias[C] -> out[B,C,L] (depthwise, left-padded)
        const aclTensor *X = e->inputs[0], *W = e->inputs[1], *Bs = e->inputs[2]; aclTensor *O = (aclTensor *)e->inputs[3];
        int B = (int)X->viewDims[0], C = (int)X->viewDims[1], L = (int)X->viewDims[2], K = (int)W->viewDims[1];
        const float *x = fp(X), *w = fp(W), *bias = Bs ? fp(Bs) : nullptr; float *o = fpw(O);   // bias is optional
        int act = (int)e->n;   // activation: 1 = SiLU (HF GatedDeltaNet conv path), 0 = none
        for (int b = 0; b < B; ++b) for (int c = 0; c < C; ++c) for (int t = 0; t < L; ++t) {
            double acc = bias ? bias[c] : 0.0;
            for (int k = 0; k < K; ++k) { int ti = t - (K - 1) + k; if (ti >= 0) acc += (double)w[c * K + k] * x[(b * C + c) * L + ti]; }
            if (act == 1) acc = acc / (1.0 + std::exp(-acc));   // SiLU
            o[(b * C + c) * L + t] = (float)acc;
        }
    } else if (e->m == 1) {   // SelectiveScan (Mamba): per (b,d) state h[N]
        const aclTensor *U = e->inputs[0], *Dl = e->inputs[1], *A = e->inputs[2], *Bm = e->inputs[3], *Cm = e->inputs[4], *Ds = e->inputs[5]; aclTensor *Y = (aclTensor *)e->inputs[6];
        int B = (int)U->viewDims[0], L = (int)U->viewDims[1], D = (int)U->viewDims[2], N = (int)A->viewDims[1];
        const float *u = fp(U), *dl = fp(Dl), *a = fp(A), *bm = fp(Bm), *cm = fp(Cm), *ds = Ds ? fp(Ds) : nullptr; float *y = fpw(Y);   // D (skip) is optional
        for (int b = 0; b < B; ++b) for (int d = 0; d < D; ++d) {
            std::vector<double> h(N, 0);
            for (int t = 0; t < L; ++t) {
                double dt = dl[(b * L + t) * D + d], ut = u[(b * L + t) * D + d], yt = 0;
                for (int n = 0; n < N; ++n) { double dA = std::exp(dt * a[d * N + n]); h[n] = dA * h[n] + dt * bm[(b * L + t) * N + n] * ut; yt += cm[(b * L + t) * N + n] * h[n]; }
                y[(b * L + t) * D + d] = (float)(yt + (ds ? ds[d] : 0.0) * ut);
            }
        }
    } else {   // GatedDeltaRule: per (b*Hd) state S[Dk*Dv]
        const aclTensor *Q = e->inputs[0], *Kk = e->inputs[1], *V = e->inputs[2], *Be = e->inputs[3], *G = e->inputs[4]; aclTensor *Y = (aclTensor *)e->inputs[5];
        int B = (int)Q->viewDims[0], Hd = (int)Q->viewDims[1], L = (int)Q->viewDims[2], Dk = (int)Q->viewDims[3], Dv = (int)V->viewDims[3];
        const float *q = fp(Q), *k = fp(Kk), *v = fp(V), *beta = fp(Be), *g = fp(G); float *y = fpw(Y);
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
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_ssm(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 0; e->n = activation; e->inputs = {x, weight, bias, out}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnCausalConv1d)
aclnnStatus aclnnSelectiveScanGetWorkspaceSize(const aclTensor *u, const aclTensor *delta, const aclTensor *A, const aclTensor *B, const aclTensor *C, const aclTensor *D, aclTensor *y, uint64_t *ws, aclOpExecutor **ex) {
    if (!u || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 1; e->inputs = {u, delta, A, B, C, D, y}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnSelectiveScan)
aclnnStatus aclnnGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 2; e->inputs = {q, k, v, beta, g, y}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnGatedDeltaRule)
} // extern "C"
