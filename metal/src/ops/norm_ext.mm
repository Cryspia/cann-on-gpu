// Normalization extensions (P7) — host-side over unified memory (MTLStorageModeShared ⇒ device ptr == host ptr).
// Covers the test_norm spec: InstanceNorm, LpNormalize, LocalResponseNorm, RmsNormGated, BatchNorm{Stats,Elemt,
// Backward,GatherStatsWithCounts}, GemmaRmsNorm, GroupNormSilu/Backward, FastLayerNorm, AddRmsNormCast,
// InplaceAddRmsNorm, AdaLayerNorm(+Backward), AddLayerNormGrad, and the quant-fused norms (RmsNormQuant,
// LayerNormQuant, AddRmsNormDynamicQuant, RmsNormDynamicMxQuant, SwiGluQuant, ClippedSwiglu, DequantSwigluQuant).
// Math/semantics mirror the CUDA reference (cuda/src/ops/{norm_ext,glu_ext,layernorm}.cu); only ops that are
// currently undefined for test_norm are defined here — plain aclnnRmsNorm etc. live elsewhere.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int8_t *I8(const aclTensor *t) { return (int8_t *)t->data + t->offset; }
uint16_t *U16(const aclTensor *t) { return (uint16_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// fp16 encode (round-to-nearest-even), matching the harness f2h.
uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7C00);
    uint32_t r = m & 0x1FFF, h = s | (e << 10) | (m >> 13); if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++; return (uint16_t)h;
}
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
inline int8_t clip8(float v) { int q = (int)std::lround(v); return (int8_t)(q < -127 ? -127 : (q > 127 ? 127 : q)); }

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }

// ---------- InstanceNorm: x[N,C,S...], gamma/beta[C]; normalize each (n,c) over the spatial dims ----------
aclnnStatus run_instance(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *o = e->out; double eps = e->eps;
    int nd = (int)x->viewDims.size(); if (nd < 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x), *gp = FP(g), *bp = FP(b); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) {
        int64_t base = (n * C + c) * S; double m = 0; for (int64_t i = 0; i < S; ++i) m += xp[base + i]; m /= S;
        double v = 0; for (int64_t i = 0; i < S; ++i) { double d = xp[base + i] - m; v += d * d; } v /= S;
        double inv = 1.0 / std::sqrt(v + eps);
        for (int64_t i = 0; i < S; ++i) op[base + i] = (float)((xp[base + i] - m) * inv * gp[c] + bp[c]);
    }
    return ACLNN_SUCCESS;
}
// ---------- LpNormalize over the last dim ----------
aclnnStatus run_lpnorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a; aclTensor *o = e->out; double p = e->alpha, eps = e->eps;
    int64_t D = x->viewDims.back(); int64_t R = x->numel() / D; const float *xp = FP(x); float *op = FP(o);
    for (int64_t r = 0; r < R; ++r) { double n = 0; for (int64_t i = 0; i < D; ++i) n += std::pow(std::fabs((double)xp[r * D + i]), p);
        n = std::pow(n, 1.0 / p); double den = n < eps ? eps : n;
        for (int64_t i = 0; i < D; ++i) op[r * D + i] = (float)(xp[r * D + i] / den); }
    return ACLNN_SUCCESS;
}
// ---------- LocalResponseNorm across channels: x[N,C,S...] ----------
aclnnStatus run_lrn(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a; aclTensor *o = e->out;
    int sz = (int)e->dim; double alpha = e->dscalars[0], beta = e->dscalars[1], k = e->dscalars[2];
    int nd = (int)x->viewDims.size(); int64_t N = nd > 0 ? x->viewDims[0] : 1, C = nd > 1 ? x->viewDims[1] : 1, S = 1;
    for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x); float *op = FP(o); int half = sz / 2;
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t spat = 0; spat < S; ++spat) {
        double acc = 0; for (int64_t j = c - half; j <= c + half; ++j) { if (j < 0 || j >= C) continue; double v = xp[(n * C + j) * S + spat]; acc += v * v; }
        double den = std::pow(k + alpha / sz * acc, beta);
        op[(n * C + c) * S + spat] = (float)(xp[(n * C + c) * S + spat] / den);
    }
    return ACLNN_SUCCESS;
}
// ---------- RmsNormGated: y = (x · silu(gate)) normalized over last dim · weight ----------
aclnnStatus run_rmsgated(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *gate = e->b, *w = e->c; aclTensor *o = e->out; double eps = e->eps;
    int64_t D = w->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gp = FP(gate), *wp = FP(w); float *op = FP(o);
    std::vector<double> h(D);
    for (int64_t r = 0; r < R; ++r) { double ss = 0; for (int64_t i = 0; i < D; ++i) { double hi = xp[r * D + i] * silu((float)gp[r * D + i]); h[i] = hi; ss += hi * hi; }
        double inv = 1.0 / std::sqrt(ss / D + eps); for (int64_t i = 0; i < D; ++i) op[r * D + i] = (float)(h[i] * inv * wp[i]); }
    return ACLNN_SUCCESS;
}
// ---------- GemmaRmsNorm: y = x·inv·(1+gamma), rstd = inv ----------
aclnnStatus run_gemma(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b; aclTensor *y = e->out, *rstd = e->out2; double eps = e->eps;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gp = FP(g); float *yp = FP(y), *rp = rstd ? FP(rstd) : nullptr;
    for (int64_t r = 0; r < R; ++r) { double ss = 0; for (int64_t i = 0; i < D; ++i) ss += (double)xp[r * D + i] * xp[r * D + i];
        double inv = 1.0 / std::sqrt(ss / D + eps); if (rp) rp[r] = (float)inv;
        for (int64_t i = 0; i < D; ++i) yp[r * D + i] = (float)(xp[r * D + i] * inv * (1.0 + gp[i])); }
    return ACLNN_SUCCESS;
}
// ---------- GroupNormSilu: x[N,C,S...]; per (n,group) normalize, affine per channel, then SiLU ----------
aclnnStatus run_gnsilu(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *o = e->out; double eps = e->eps; int G = (int)e->dim;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    int64_t Cg = C / G, cnt = Cg * S; const float *xp = FP(x), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr; float *op = FP(o);
    aclTensor *meanO = e->out2 ? const_cast<aclTensor *>(e->out2) : nullptr; aclTensor *rstdO = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr;
    float *mp = meanO ? FP(meanO) : nullptr, *rp = rstdO ? FP(rstdO) : nullptr;
    for (int64_t n = 0; n < N; ++n) for (int grp = 0; grp < G; ++grp) {
        double m = 0, q = 0; for (int64_t cc = 0; cc < Cg; ++cc) for (int64_t spat = 0; spat < S; ++spat) { double v = xp[(n * C + grp * Cg + cc) * S + spat]; m += v; q += v * v; }
        m /= cnt; double var = q / cnt - m * m; double inv = 1.0 / std::sqrt(var + eps);
        if (mp) mp[n * G + grp] = (float)m; if (rp) rp[n * G + grp] = (float)inv;
        for (int64_t cc = 0; cc < Cg; ++cc) { int64_t c = grp * Cg + cc; double gv = gp ? gp[c] : 1.0, bv = bp ? bp[c] : 0.0;
            for (int64_t spat = 0; spat < S; ++spat) { double y = (xp[(n * C + c) * S + spat] - m) * inv * gv + bv; op[(n * C + c) * S + spat] = (float)silu((float)y); } }
    }
    return ACLNN_SUCCESS;
}
// ---------- FastLayerNorm over the last dim ----------
aclnnStatus run_fastln(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *o = e->out; double eps = e->eps;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gp = FP(g), *bp = FP(b); float *op = FP(o);
    aclTensor *meanO = e->out2 ? const_cast<aclTensor *>(e->out2) : nullptr; aclTensor *rstdO = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr;
    float *mp = meanO ? FP(meanO) : nullptr, *rp = rstdO ? FP(rstdO) : nullptr;
    for (int64_t r = 0; r < R; ++r) { double m = 0; for (int64_t i = 0; i < D; ++i) m += xp[r * D + i]; m /= D;
        double v = 0; for (int64_t i = 0; i < D; ++i) { double d = xp[r * D + i] - m; v += d * d; } v /= D; double inv = 1.0 / std::sqrt(v + eps);
        if (mp) mp[r] = (float)m; if (rp) rp[r] = (float)inv;
        for (int64_t i = 0; i < D; ++i) op[r * D + i] = (float)((xp[r * D + i] - m) * inv * gp[i] + bp[i]); }
    return ACLNN_SUCCESS;
}
// ---------- AddRmsNorm family: s = x + residual; y = rms(s)·gamma. m: 0 cast (y fp32, yCast fp16, residualSum),
//            1 inplace (write y back into x), 2 dynamic int8 quant (per-row absmax/127). ----------
aclnnStatus run_addrms(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *res = e->b, *g = e->c; double eps = e->eps; int mode = (int)e->dim;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *rp = FP(res), *gp = FP(g);
    aclTensor *y = e->out, *yCast = e->out2, *sumO = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr, *scaleO = e->rstd ? const_cast<aclTensor *>(e->rstd) : nullptr;
    float *yp = (mode == 1) ? (float *)x->data + x->offset : (y && mode != 2 ? FP(y) : nullptr);
    uint16_t *ycp = (yCast && mode == 0) ? U16(yCast) : nullptr;
    float *sp = sumO ? FP(sumO) : nullptr; int8_t *qp = (mode == 2 && y) ? I8(y) : nullptr; float *scp = scaleO ? FP(scaleO) : nullptr;
    std::vector<double> sval(D), yval(D);
    for (int64_t r = 0; r < R; ++r) {
        double ss = 0; for (int64_t i = 0; i < D; ++i) { double sv = (double)xp[r * D + i] + rp[r * D + i]; sval[i] = sv; ss += sv * sv; if (sp) sp[r * D + i] = (float)sv; }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        double amax = 0; for (int64_t i = 0; i < D; ++i) { double yv = sval[i] * inv * gp[i]; yval[i] = yv; amax = std::max(amax, std::fabs(yv)); }
        if (mode == 2) { double sc = amax > 0 ? amax / 127.0 : 1.0; if (scp) scp[r] = (float)sc; double qi = 1.0 / sc;
            for (int64_t i = 0; i < D; ++i) qp[r * D + i] = clip8((float)(yval[i] * qi)); }
        else { for (int64_t i = 0; i < D; ++i) { if (yp) yp[r * D + i] = (float)yval[i]; if (ycp) ycp[r * D + i] = f2h((float)yval[i]); } }
    }
    return ACLNN_SUCCESS;
}
// ---------- AdaLayerNorm: y = norm(x)·(1+scale) + shift, all per-element over last dim ----------
aclnnStatus run_adaln(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *sc = e->b, *sh = e->c; aclTensor *o = e->out; double eps = e->eps;
    int64_t D = x->viewDims.back(); int64_t R = x->numel() / D; const float *xp = FP(x), *scp = FP(sc), *shp = FP(sh); float *op = FP(o);
    for (int64_t r = 0; r < R; ++r) { double m = 0; for (int64_t d = 0; d < D; ++d) m += xp[r * D + d]; m /= D;
        double v = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - m; v += u * u; } v /= D; double inv = 1.0 / std::sqrt(v + eps);
        for (int64_t d = 0; d < D; ++d) op[r * D + d] = (float)((xp[r * D + d] - m) * inv * (1.0 + scp[r * D + d]) + shp[r * D + d]); }
    return ACLNN_SUCCESS;
}
// ---------- SwiGluQuant / ClippedSwiglu / DequantSwigluQuant: in[...,2D] -> [...,D] ----------
// m: 0 SwiGluQuant (int8 + per-row scale), 1 ClippedSwiglu (clip gate, fp32 out), 2 DequantSwigluQuant (optional dq, int8 + scale)
aclnnStatus run_swiglu(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *in = e->a; aclTensor *o = e->out; int mode = (int)e->m; double clip = e->alpha;
    int64_t D = o->viewDims.back(); int64_t R = in->numel() / (2 * D); const float *xp = FP(in);
    const float *dq = e->b ? FP(e->b) : nullptr;
    aclTensor *scaleO = e->rstd ? const_cast<aclTensor *>(e->rstd) : nullptr; float *scp = scaleO ? FP(scaleO) : nullptr;
    int8_t *qp = (mode != 1) ? I8(o) : nullptr; float *fp = (mode == 1) ? FP(o) : nullptr;
    std::vector<double> gv(D);
    for (int64_t r = 0; r < R; ++r) {
        double scl = dq ? dq[r] : 1.0, amax = 0;
        for (int64_t d = 0; d < D; ++d) { double a = xp[r * 2 * D + d] * scl, b = xp[r * 2 * D + D + d] * scl;
            if (mode == 1) a = a < -clip ? -clip : (a > clip ? clip : a);
            double g = silu((float)a) * b; gv[d] = g; amax = std::max(amax, std::fabs(g)); }
        if (mode == 1) { for (int64_t d = 0; d < D; ++d) fp[r * D + d] = (float)gv[d]; }
        else { double sc = amax > 0 ? amax / 127.0 : 1.0; if (scp) scp[r] = (float)sc; double qi = 1.0 / sc;
            for (int64_t d = 0; d < D; ++d) qp[r * D + d] = clip8((float)(gv[d] * qi)); }
    }
    return ACLNN_SUCCESS;
}
// ---------- RmsNormQuant: static scale+offset; yq = round(rms(x)·g / scale + offset) ----------
aclnnStatus run_rmsquant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b; aclTensor *o = e->out; double eps = e->eps, scale = e->alpha, off = e->dscalars[0];
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gp = FP(g); int8_t *qp = I8(o);
    for (int64_t r = 0; r < R; ++r) { double ss = 0; for (int64_t i = 0; i < D; ++i) ss += (double)xp[r * D + i] * xp[r * D + i];
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int64_t i = 0; i < D; ++i) { double y = xp[r * D + i] * inv * gp[i]; int q = (int)std::lround(y / scale + off); q = q < -127 ? -127 : (q > 127 ? 127 : q); qp[r * D + i] = (int8_t)q; } }
    return ACLNN_SUCCESS;
}
// ---------- LayerNormQuant: y = norm(x)·g + b; per-row absmax/127 dynamic int8 + scaleOut ----------
aclnnStatus run_lnquant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->c, *b = e->mask; aclTensor *o = e->out, *scaleO = e->out2; double eps = e->eps;
    int64_t D = e->reduceCount; int64_t R = e->outerCount; const float *xp = FP(x), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr;
    int8_t *qp = I8(o); float *scp = FP(scaleO); std::vector<double> yv(D);
    for (int64_t r = 0; r < R; ++r) { double m = 0; for (int64_t d = 0; d < D; ++d) m += xp[r * D + d]; m /= D;
        double v = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - m; v += u * u; } v /= D; double inv = 1.0 / std::sqrt(v + eps);
        double amax = 0; for (int64_t d = 0; d < D; ++d) { double y = (xp[r * D + d] - m) * inv * (gp ? gp[d] : 1.0) + (bp ? bp[d] : 0.0); yv[d] = y; amax = std::max(amax, std::fabs(y)); }
        double sc = amax > 0 ? amax / 127.0 : 1.0; scp[r] = (float)sc; double qi = 1.0 / sc;
        for (int64_t d = 0; d < D; ++d) qp[r * D + d] = clip8((float)(yv[d] * qi)); }
    return ACLNN_SUCCESS;
}
// ---------- RmsNormDynamicMxQuant: y = rms(x)·g; per-row MX power-of-2 scale = exp2(ceil(log2(amax/127))) ----------
aclnnStatus run_mxquant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b; aclTensor *o = e->out, *scaleO = e->out2; double eps = e->eps;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gp = FP(g); int8_t *qp = I8(o); float *scp = FP(scaleO);
    std::vector<double> yv(D);
    for (int64_t r = 0; r < R; ++r) { double ms = 0; for (int64_t d = 0; d < D; ++d) ms += (double)xp[r * D + d] * xp[r * D + d]; ms /= D; double rr = 1.0 / std::sqrt(ms + eps);
        double amax = 0; for (int64_t d = 0; d < D; ++d) { double y = xp[r * D + d] * rr * gp[d]; yv[d] = y; amax = std::max(amax, std::fabs(y)); }
        double sc = amax > 0 ? std::exp2(std::ceil(std::log2(amax / 127.0))) : 1.0; scp[r] = (float)sc; double qi = 1.0 / sc;
        for (int64_t d = 0; d < D; ++d) { int q = (int)std::lround(yv[d] * qi); q = q < -128 ? -128 : (q > 127 ? 127 : q); qp[r * D + d] = (int8_t)q; } }
    return ACLNN_SUCCESS;
}
// ---------- BatchNormStats: per-channel mean & invstd over [N, *S] ----------
aclnnStatus run_bnstats(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a; aclTensor *mean = e->out, *invstd = e->out2; double eps = e->eps;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x); float *mp = FP(mean), *ip = FP(invstd); int64_t M = N * S;
    for (int64_t c = 0; c < C; ++c) { double m = 0; for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) m += xp[(n * C + c) * S + sp]; m /= M;
        double v = 0; for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) { double u = xp[(n * C + c) * S + sp] - m; v += u * u; } v /= M;
        mp[c] = (float)m; ip[c] = (float)(1.0 / std::sqrt(v + eps)); }
    return ACLNN_SUCCESS;
}
// ---------- BatchNormElemt: y = (x - mean)·invstd·weight + bias (per channel) ----------
aclnnStatus run_bnelemt(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *w = e->b, *bias = e->c, *mean = e->mean, *invstd = e->rstd; aclTensor *o = e->out;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x), *wp = w ? FP(w) : nullptr, *bp = bias ? FP(bias) : nullptr, *mp = FP(mean), *ip = FP(invstd); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t sp = 0; sp < S; ++sp) {
        int64_t idx = (n * C + c) * S + sp; op[idx] = (float)((xp[idx] - mp[c]) * ip[c] * (wp ? wp[c] : 1.0) + (bp ? bp[c] : 0.0)); }
    return ACLNN_SUCCESS;
}
// ---------- BatchNormBackward: per-channel gradGamma, gradBeta, gradX from saved mean/invstd ----------
aclnnStatus run_bnbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gy = e->a, *x = e->b, *gamma = e->c, *mean = e->mean, *invstd = e->rstd;
    aclTensor *gx = e->out, *gg = e->out2, *gb = e->mask ? const_cast<aclTensor *>(e->mask) : nullptr;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *gyp = FP(gy), *xp = FP(x), *gam = FP(gamma), *mp = FP(mean), *ip = FP(invstd);
    float *gxp = gx ? FP(gx) : nullptr, *ggp = gg ? FP(gg) : nullptr, *gbp = gb ? FP(gb) : nullptr; double M = N * S;
    for (int64_t c = 0; c < C; ++c) { double sumg = 0, sumgx = 0;
        for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) { int64_t idx = (n * C + c) * S + sp; double xhat = (xp[idx] - mp[c]) * ip[c]; sumgx += gyp[idx] * xhat; sumg += gyp[idx]; }
        if (ggp) ggp[c] = (float)sumgx; if (gbp) gbp[c] = (float)sumg;
        if (gxp) for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) { int64_t idx = (n * C + c) * S + sp; double xhat = (xp[idx] - mp[c]) * ip[c];
            gxp[idx] = (float)(gam[c] * ip[c] / M * (M * gyp[idx] - sumg - xhat * sumgx)); } }
    return ACLNN_SUCCESS;
}
// ---------- BatchNormGatherStatsWithCounts: combine partition stats [P,C] + counts[P] into [C] ----------
aclnnStatus run_bngather(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *means = e->a, *invs = e->b, *counts = e->c; aclTensor *meanO = e->out, *invstdO = e->out2; double eps = e->eps;
    int64_t P = means->viewDims[0], C = means->viewDims[1]; const float *mp = FP(means), *ip = FP(invs), *cp = FP(counts);
    float *omp = FP(meanO), *oip = FP(invstdO);
    for (int64_t c = 0; c < C; ++c) { double tot = 0, msum = 0; for (int64_t p = 0; p < P; ++p) { tot += cp[p]; msum += cp[p] * mp[p * C + c]; }
        double m = msum / tot, varsum = 0;
        for (int64_t p = 0; p < P; ++p) { double vp = 1.0 / ((double)ip[p * C + c] * ip[p * C + c]) - eps; double mm = mp[p * C + c]; varsum += cp[p] * (vp + (mm - m) * (mm - m)); }
        double var = varsum / tot; omp[c] = (float)m; oip[c] = (float)(1.0 / std::sqrt(var + eps)); }
    return ACLNN_SUCCESS;
}
// ---------- GroupNormBackward: gradX from gradOut, x, mean[N*G], rstd[N*G], numGroups ----------
aclnnStatus run_gnbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gy = e->a, *x = e->b, *mean = e->mean, *rstd = e->rstd, *gamma = e->c; aclTensor *gx = e->out; int G = (int)e->dim;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    int64_t Cg = C / G, Dn = Cg * S; const float *gyp = FP(gy), *xp = FP(x), *mp = FP(mean), *rp = FP(rstd), *gam = gamma ? FP(gamma) : nullptr; float *gxp = FP(gx);
    for (int64_t n = 0; n < N; ++n) for (int grp = 0; grp < G; ++grp) { double m = mp[n * G + grp], rs = rp[n * G + grp], sa = 0, sb = 0;
        for (int64_t cc = 0; cc < Cg; ++cc) for (int64_t sp = 0; sp < S; ++sp) { int64_t c = grp * Cg + cc; int64_t idx = (n * C + c) * S + sp;
            double dy = gyp[idx] * (gam ? gam[c] : 1.0); double xhat = (xp[idx] - m) * rs; sa += dy; sb += dy * xhat; }
        for (int64_t cc = 0; cc < Cg; ++cc) for (int64_t sp = 0; sp < S; ++sp) { int64_t c = grp * Cg + cc; int64_t idx = (n * C + c) * S + sp;
            double dy = gyp[idx] * (gam ? gam[c] : 1.0); double xhat = (xp[idx] - m) * rs; gxp[idx] = (float)(rs * (dy - sa / Dn - xhat * sb / Dn)); } }
    return ACLNN_SUCCESS;
}
// ---------- AdaLayerNormBackward: gradX, gradScale, gradShift over last dim. g = gradY·(1+scale). ----------
aclnnStatus run_adalnbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gy = e->a, *x = e->b, *sc = e->c; aclTensor *gx = e->out, *gs = e->out2, *gsh = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr; double eps = e->eps;
    int64_t D = x->viewDims.back(); int64_t R = x->numel() / D; const float *gyp = FP(gy), *xp = FP(x), *scp = FP(sc);
    float *gxp = gx ? FP(gx) : nullptr, *gsp = gs ? FP(gs) : nullptr, *gshp = gsh ? FP(gsh) : nullptr; std::vector<double> n(D), g(D);
    for (int64_t r = 0; r < R; ++r) { double mean = 0; for (int64_t d = 0; d < D; ++d) mean += xp[r * D + d]; mean /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - mean; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        double sg = 0, sgn = 0; for (int64_t d = 0; d < D; ++d) { n[d] = (xp[r * D + d] - mean) * inv; g[d] = gyp[r * D + d] * (1.0 + scp[r * D + d]); sg += g[d]; sgn += g[d] * n[d]; }
        double mg = sg / D, mgn = sgn / D;
        for (int64_t d = 0; d < D; ++d) { if (gxp) gxp[r * D + d] = (float)(inv * (g[d] - mg - n[d] * mgn));
            if (gsp) gsp[r * D + d] = (float)(gyp[r * D + d] * n[d]); if (gshp) gshp[r * D + d] = gyp[r * D + d]; } }
    return ACLNN_SUCCESS;
}
// ---------- AddLayerNormGrad: gradX/gradResidual (equal), gradGamma, gradBeta. norm over (x+res). g=gradY·gamma. ----------
aclnnStatus run_addlngrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gy = e->a, *x = e->b, *res = e->c, *gamma = e->mean; double eps = e->eps;
    aclTensor *gx = e->out, *gr = e->out2, *gg = e->rstd ? const_cast<aclTensor *>(e->rstd) : nullptr; aclTensor *gb = e->inputs.empty() ? nullptr : const_cast<aclTensor *>(e->inputs[0]);
    int64_t D = gamma->numel(); int64_t R = x->numel() / D; const float *gyp = FP(gy), *xp = FP(x), *rp = FP(res), *gam = FP(gamma);
    float *gxp = gx ? FP(gx) : nullptr, *grp = gr ? FP(gr) : nullptr, *ggp = gg ? FP(gg) : nullptr, *gbp = gb ? FP(gb) : nullptr;
    std::vector<double> rgg(D, 0), rgb(D, 0), n(D), g(D);
    for (int64_t r = 0; r < R; ++r) { double mean = 0; for (int64_t d = 0; d < D; ++d) mean += (double)xp[r * D + d] + rp[r * D + d]; mean /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double u = (double)xp[r * D + d] + rp[r * D + d] - mean; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        double sg = 0, sgn = 0; for (int64_t d = 0; d < D; ++d) { n[d] = ((double)xp[r * D + d] + rp[r * D + d] - mean) * inv; g[d] = gyp[r * D + d] * gam[d]; sg += g[d]; sgn += g[d] * n[d]; }
        double mg = sg / D, mgn = sgn / D;
        for (int64_t d = 0; d < D; ++d) { double gs = inv * (g[d] - mg - n[d] * mgn); if (gxp) gxp[r * D + d] = (float)gs; if (grp) grp[r * D + d] = (float)gs;
            rgg[d] += gyp[r * D + d] * n[d]; rgb[d] += gyp[r * D + d]; } }
    if (ggp) for (int64_t d = 0; d < D; ++d) ggp[d] = (float)rgg[d]; if (gbp) for (int64_t d = 0; d < D; ++d) gbp[d] = (float)rgb[d];
    return ACLNN_SUCCESS;
}

RUN(ex_instance, run_instance)
RUN(ex_lpnorm, run_lpnorm)
RUN(ex_lrn, run_lrn)
RUN(ex_rmsgated, run_rmsgated)
RUN(ex_gemma, run_gemma)
RUN(ex_gnsilu, run_gnsilu)
RUN(ex_fastln, run_fastln)
RUN(ex_addrms, run_addrms)
RUN(ex_adaln, run_adaln)
RUN(ex_swiglu, run_swiglu)
RUN(ex_rmsquant, run_rmsquant)
RUN(ex_lnquant, run_lnquant)
RUN(ex_mxquant, run_mxquant)
RUN(ex_bnstats, run_bnstats)
RUN(ex_bnelemt, run_bnelemt)
RUN(ex_bnbwd, run_bnbwd)
RUN(ex_bngather, run_bngather)
RUN(ex_gnbwd, run_gnbwd)
RUN(ex_adalnbwd, run_adalnbwd)
RUN(ex_addlngrad, run_addlngrad)
} // namespace

extern "C" {
aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !gamma || !beta || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->out = out; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInstanceNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_instance(w, wss, e, s); }

aclnnStatus aclnnLpNormalizeGetWorkspaceSize(const aclTensor *self, double p, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->alpha = p; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLpNormalize(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_lpnorm(w, wss, e, s); }

aclnnStatus aclnnLocalResponseNormGetWorkspaceSize(const aclTensor *self, int64_t size, double alpha, double beta, double k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = size; e->dscalars = {alpha, beta, k}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLocalResponseNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_lrn(w, wss, e, s); }

aclnnStatus aclnnRmsNormGatedGetWorkspaceSize(const aclTensor *self, const aclTensor *gate, const aclTensor *weight, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !gate || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gate; e->c = weight; e->out = out; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormGated(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_rmsgated(w, wss, e, s); }

aclnnStatus aclnnGemmaRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *yOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !yOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->out = yOut; e->out2 = rstdOut; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGemmaRmsNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gemma(w, wss, e, s); }

aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->dim = group; e->eps = eps; e->out = out; e->out2 = meanOut; e->mean = rstdOut; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNormSilu(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gnsilu(w, wss, e, s); }

aclnnStatus aclnnFastLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !beta || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = beta; e->out = out; e->out2 = meanOut; e->mean = rstdOut; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFastLayerNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_fastln(w, wss, e, s); }

aclnnStatus aclnnAddRmsNormCastGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *y, aclTensor *yCast, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = y; e->out2 = yCast; e->mean = residualSum; e->dim = 0; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormCast(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms(w, wss, e, s); }

aclnnStatus aclnnInplaceAddRmsNormGetWorkspaceSize(aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->mean = residualSum; e->dim = 1; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceAddRmsNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms(w, wss, e, s); }

aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *yQuant, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !yQuant || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = yQuant; e->rstd = scaleOut; e->mean = residualSum; e->dim = 2; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormDynamicQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms(w, wss, e, s); }

aclnnStatus aclnnAdaLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !shift || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = shift; e->out = out; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaLayerNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_adaln(w, wss, e, s); }

aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->rstd = scaleOut; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSwiGluQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swiglu(w, wss, e, s); }

aclnnStatus aclnnClippedSwigluGetWorkspaceSize(const aclTensor *x, double clipValue, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->alpha = clipValue; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnClippedSwiglu(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swiglu(w, wss, e, s); }

aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = dequantScale; e->out = out; e->rstd = scaleOut; e->m = 2; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDequantSwigluQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swiglu(w, wss, e, s); }

aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double scale, double offset, double eps, aclTensor *yQuant, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !yQuant || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->out = yQuant; e->alpha = scale; e->dscalars = {offset}; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_rmsquant(w, wss, e, s); }

aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = x->viewDims.back(); auto *e = new aclOpExecutor(); e->a = x; e->c = gamma; e->mask = beta; e->out = out; e->out2 = scaleOut;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLayerNormQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_lnquant(w, wss, e, s); }

aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->out = out; e->out2 = scaleOut; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormDynamicMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mxquant(w, wss, e, s); }

aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mean || !invstd || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = mean; e->out2 = invstd; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormStats(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_bnstats(w, wss, e, s); }

aclnnStatus aclnnBatchNormElemtGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mean || !invstd || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = weight; e->c = bias; e->mean = mean; e->rstd = invstd; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormElemt(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_bnelemt(w, wss, e, s); }

aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, const aclTensor *savedMean, const aclTensor *savedInvStd, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !savedMean || !savedInvStd || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gamma; e->mean = savedMean; e->rstd = savedInvStd; e->out = gradX; e->out2 = gradGamma; e->mask = gradBeta; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_bnbwd(w, wss, e, s); }

aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex) {
    if (!meanAll || !invstdAll || !counts || !mean || !invstd || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = meanAll; e->b = invstdAll; e->c = counts; e->out = mean; e->out2 = invstd; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormGatherStatsWithCounts(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_bngather(w, wss, e, s); }

aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma, int64_t numGroups, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    (void)gradGamma; (void)gradBeta; if (!gradOut || !self || !mean || !rstd || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->mean = mean; e->rstd = rstd; e->c = gamma; e->dim = numGroups; e->out = gradX; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNormBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gnbwd(w, wss, e, s); }

aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *scale, double eps, aclTensor *gradX, aclTensor *gradScale, aclTensor *gradShift, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !scale || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = scale; e->out = gradX; e->out2 = gradScale; e->mean = gradShift; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaLayerNormBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_adalnbwd(w, wss, e, s); }

aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *gradX, aclTensor *gradResidual, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !residual || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = residual; e->mean = gamma; e->out = gradX; e->out2 = gradResidual; e->rstd = gradGamma;
    if (gradBeta) e->inputs.push_back(gradBeta); e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddLayerNormGrad(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addlngrad(w, wss, e, s); }
} // extern "C"
