// Neural-net ops for test_nn (P8) — host-side over unified memory (MTLStorageModeShared ⇒ device ptr == host ptr).
// Covers the symbols test_nn needs that are not defined elsewhere: LayerNorm(+Backward), RmsNormBackward, GroupNorm,
// BatchNorm(+Training), DeepNorm, AddRmsNorm/AddLayerNorm, GeGlu, the loss family (Mse/BCEWithLogits/NLL/
// CrossEntropy(+Backward)/MseBackward), ApplyAdamW, and the attention family
// (FlashAttentionScoreHighPerf, FlashAttentionScoreBackward, PagedAttention).
// Math/semantics mirror the CUDA reference (cuda/src/ops/{loss,norm_ext,layernorm,attention,optim}.cu) and the
// CPU cross-checks in tests/test_nn.cpp. Norm/attention variants already in norm_ext.mm / attention.mm are NOT
// redefined here. Attention here is naive batched (BNSD), dtype-aware (fp32/fp16/bf16), matching the test reference.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// fp16 / bf16 conversions matching the harness (round-to-nearest-even).
uint16_t f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7C00);
    uint32_t r = m & 0x1FFF, h = s | (e << 10) | (m >> 13); if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++; return (uint16_t)h;
}
float h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; memcpy(&x, &f, 4); x |= s; }
    else if (e == 31) x = s | 0x7F800000 | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
uint16_t f2bf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x >> 16) + ((x >> 15) & 1)); }
float bf2f(uint16_t b) { uint32_t x = ((uint32_t)b) << 16; float f; memcpy(&f, &x, 4); return f; }

// Read/write tensor element i honoring dtype (fp32 / fp16 / bf16).
double ld(const aclTensor *t, int64_t i) {
    switch (t->dtype) {
        case ACL_FLOAT16: return (double)h2f(((uint16_t *)t->data + t->offset)[i]);
        case ACL_BF16:    return (double)bf2f(((uint16_t *)t->data + t->offset)[i]);
        default:          return (double)((float *)t->data + t->offset)[i];
    }
}
void st(aclTensor *t, int64_t i, double v) {
    switch (t->dtype) {
        case ACL_FLOAT16: ((uint16_t *)t->data + t->offset)[i] = f2h((float)v); break;
        case ACL_BF16:    ((uint16_t *)t->data + t->offset)[i] = f2bf((float)v); break;
        default:          ((float *)t->data + t->offset)[i] = (float)v; break;
    }
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }

// ======================= Norms =======================

// LayerNorm: normalize over the last normalizedShape dims (here last dim D); affine optional; meanOut/rstdOut optional.
aclnnStatus run_layernorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *o = e->out;
    aclTensor *meanO = e->out2, *rstdO = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr; double eps = e->eps;
    int64_t D = e->reduceCount; int64_t R = x->numel() / D;
    float *mp = meanO ? FP(meanO) : nullptr, *rp = rstdO ? FP(rstdO) : nullptr;
    for (int64_t r = 0; r < R; ++r) {
        double m = 0; for (int64_t i = 0; i < D; ++i) m += ld(x, r * D + i); m /= D;
        double v = 0; for (int64_t i = 0; i < D; ++i) { double d = ld(x, r * D + i) - m; v += d * d; } v /= D;
        double inv = 1.0 / std::sqrt(v + eps); if (mp) mp[r] = (float)m; if (rp) rp[r] = (float)inv;
        for (int64_t i = 0; i < D; ++i) {
            double gv = g ? ld(g, i) : 1.0, bv = b ? ld(b, i) : 0.0;
            st(o, r * D + i, (ld(x, r * D + i) - m) * inv * gv + bv);
        }
    }
    return ACLNN_SUCCESS;
}
// LayerNormBackward (fp32): dx + dgamma + dbeta over last dim.
aclnnStatus run_layernorm_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *dy = e->a, *x = e->b, *g = e->c; aclTensor *dx = e->out, *dgm = e->out2,
        *dbt = e->mask ? const_cast<aclTensor *>(e->mask) : nullptr; double eps = e->eps;
    int64_t D = e->reduceCount; int64_t R = x->numel() / D;
    const float *dyp = FP(dy), *xp = FP(x), *gp = g ? FP(g) : nullptr;
    float *dxp = dx ? FP(dx) : nullptr, *dgp = dgm ? FP(dgm) : nullptr, *dbp = dbt ? FP(dbt) : nullptr;
    std::vector<double> rgg(D, 0), rgb(D, 0);
    for (int64_t r = 0; r < R; ++r) {
        double mean = 0; for (int64_t d = 0; d < D; ++d) mean += xp[r * D + d]; mean /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - mean; var += u * u; } var /= D;
        double rstd = 1.0 / std::sqrt(var + eps), mg = 0, mgx = 0;
        for (int64_t d = 0; d < D; ++d) { double xhat = (xp[r * D + d] - mean) * rstd, gi = dyp[r * D + d] * (gp ? gp[d] : 1.0); mg += gi; mgx += gi * xhat; }
        mg /= D; mgx /= D;
        for (int64_t d = 0; d < D; ++d) { double xhat = (xp[r * D + d] - mean) * rstd, gi = dyp[r * D + d] * (gp ? gp[d] : 1.0);
            if (dxp) dxp[r * D + d] = (float)(rstd * (gi - mg - xhat * mgx)); rgg[d] += dyp[r * D + d] * xhat; rgb[d] += dyp[r * D + d]; }
    }
    if (dgp) for (int64_t d = 0; d < D; ++d) dgp[d] = (float)rgg[d];
    if (dbp) for (int64_t d = 0; d < D; ++d) dbp[d] = (float)rgb[d];
    return ACLNN_SUCCESS;
}
// RmsNormBackward (fp32): dx + dgamma over last dim.
aclnnStatus run_rmsnorm_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *dy = e->a, *x = e->b, *g = e->c; aclTensor *dx = e->out, *dgm = e->out2; double eps = e->eps;
    int64_t D = g->numel(); int64_t R = x->numel() / D;
    const float *dyp = FP(dy), *xp = FP(x), *gp = FP(g); float *dxp = dx ? FP(dx) : nullptr, *dgp = dgm ? FP(dgm) : nullptr;
    std::vector<double> rgg(D, 0);
    for (int64_t r = 0; r < R; ++r) {
        double ss = 0; for (int64_t d = 0; d < D; ++d) ss += (double)xp[r * D + d] * xp[r * D + d];
        double rcp = 1.0 / std::sqrt(ss / D + eps), r3 = rcp * rcp * rcp, dot = 0;
        for (int64_t d = 0; d < D; ++d) dot += (double)dyp[r * D + d] * gp[d] * xp[r * D + d];
        for (int64_t d = 0; d < D; ++d) { if (dxp) dxp[r * D + d] = (float)(rcp * (dyp[r * D + d] * gp[d]) - (r3 / D) * xp[r * D + d] * dot);
            rgg[d] += dyp[r * D + d] * xp[r * D + d] * rcp; }
    }
    if (dgp) for (int64_t d = 0; d < D; ++d) dgp[d] = (float)rgg[d];
    return ACLNN_SUCCESS;
}
// GroupNorm: x[N,C,*S]; per (n,group) normalize, affine per channel.
aclnnStatus run_groupnorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *o = e->out; double eps = e->eps; int G = (int)e->dim;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    int64_t Cg = C / G, cnt = Cg * S; const float *xp = FP(x), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr; float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int grp = 0; grp < G; ++grp) {
        double sum = 0, sq = 0; for (int64_t cc = 0; cc < Cg; ++cc) for (int64_t sp = 0; sp < S; ++sp) { double v = xp[(n * C + grp * Cg + cc) * S + sp]; sum += v; sq += v * v; }
        double m = sum / cnt, var = sq / cnt - m * m, inv = 1.0 / std::sqrt(var + eps);
        for (int64_t cc = 0; cc < Cg; ++cc) { int64_t c = grp * Cg + cc; double gv = gp ? gp[c] : 1.0, bv = bp ? bp[c] : 0.0;
            for (int64_t sp = 0; sp < S; ++sp) op[(n * C + c) * S + sp] = (float)((xp[(n * C + c) * S + sp] - m) * inv * gv + bv); }
    }
    return ACLNN_SUCCESS;
}
// BatchNorm inference: y = (x - mean)/sqrt(var+eps)·gamma + beta (per channel, running stats).
aclnnStatus run_batchnorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c, *mean = e->mean, *var = e->rstd; aclTensor *o = e->out; double eps = e->eps;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr, *mp = FP(mean), *vp = FP(var); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) { double inv = 1.0 / std::sqrt((double)vp[c] + eps);
        for (int64_t sp = 0; sp < S; ++sp) { int64_t idx = (n * C + c) * S + sp; op[idx] = (float)((xp[idx] - mp[c]) * inv * (gp ? gp[c] : 1.0) + (bp ? bp[c] : 0.0)); } }
    return ACLNN_SUCCESS;
}
// BatchNormTraining: per-channel batch stats; y; savedMean; savedInvStd; updates running mean/var (momentum).
aclnnStatus run_batchnorm_train(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *g = e->b, *b = e->c; aclTensor *rm = const_cast<aclTensor *>(e->mean), *rv = const_cast<aclTensor *>(e->rstd);
    aclTensor *o = e->out, *sm = e->out2, *si = e->mask ? const_cast<aclTensor *>(e->mask) : nullptr; double eps = e->eps, mom = e->alpha;
    int nd = (int)x->viewDims.size(); int64_t N = x->viewDims[0], C = x->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= x->viewDims[d];
    const float *xp = FP(x), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr;
    float *op = FP(o), *smp = sm ? FP(sm) : nullptr, *sip = si ? FP(si) : nullptr, *rmp = rm ? FP(rm) : nullptr, *rvp = rv ? FP(rv) : nullptr;
    int64_t M = N * S;
    for (int64_t c = 0; c < C; ++c) {
        double mean = 0; for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) mean += xp[(n * C + c) * S + sp]; mean /= M;
        double var = 0; for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) { double u = xp[(n * C + c) * S + sp] - mean; var += u * u; } var /= M;
        double inv = 1.0 / std::sqrt(var + eps); if (smp) smp[c] = (float)mean; if (sip) sip[c] = (float)inv;
        if (rmp) rmp[c] = (float)((1.0 - mom) * rmp[c] + mom * mean);
        if (rvp) { double ub = M > 1 ? var * M / (M - 1) : var; rvp[c] = (float)((1.0 - mom) * rvp[c] + mom * ub); }
        for (int64_t n = 0; n < N; ++n) for (int64_t sp = 0; sp < S; ++sp) { int64_t idx = (n * C + c) * S + sp; op[idx] = (float)((xp[idx] - mean) * inv * (gp ? gp[c] : 1.0) + (bp ? bp[c] : 0.0)); }
    }
    return ACLNN_SUCCESS;
}
// DeepNorm: y = LayerNorm(alpha·x + gx)·gamma + beta, over last dim.
aclnnStatus run_deepnorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *gx = e->b, *g = e->c, *b = e->mask; aclTensor *o = e->out; double alpha = e->alpha, eps = e->eps;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *gxp = FP(gx), *gp = FP(g), *bp = b ? FP(b) : nullptr; float *op = FP(o);
    std::vector<double> t(D);
    for (int64_t r = 0; r < R; ++r) {
        double mean = 0; for (int64_t d = 0; d < D; ++d) { t[d] = alpha * xp[r * D + d] + gxp[r * D + d]; mean += t[d]; } mean /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double u = t[d] - mean; var += u * u; } var /= D; double rstd = 1.0 / std::sqrt(var + eps);
        for (int64_t d = 0; d < D; ++d) op[r * D + d] = (float)((t[d] - mean) * rstd * gp[d] + (bp ? bp[d] : 0.0));
    }
    return ACLNN_SUCCESS;
}
// AddRmsNorm / AddLayerNorm: s = x + residual; y = norm(s)·gamma (+beta for layer); outputs y + residualSum.
aclnnStatus run_addnorm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *res = e->b, *g = e->c, *b = e->mask; aclTensor *y = e->out, *sumO = e->out2; double eps = e->eps; bool layer = e->causal;
    int64_t D = g->numel(); int64_t R = x->numel() / D; const float *xp = FP(x), *rp = FP(res), *gp = FP(g), *bp = b ? FP(b) : nullptr;
    float *yp = FP(y), *sp = sumO ? FP(sumO) : nullptr; std::vector<double> t(D);
    for (int64_t r = 0; r < R; ++r) {
        for (int64_t d = 0; d < D; ++d) { t[d] = (double)xp[r * D + d] + rp[r * D + d]; if (sp) sp[r * D + d] = (float)t[d]; }
        if (layer) { double mean = 0; for (auto v : t) mean += v; mean /= D; double var = 0; for (auto v : t) var += (v - mean) * (v - mean); var /= D; double rstd = 1.0 / std::sqrt(var + eps);
            for (int64_t d = 0; d < D; ++d) yp[r * D + d] = (float)((t[d] - mean) * rstd * gp[d] + (bp ? bp[d] : 0.0)); }
        else { double ss = 0; for (auto v : t) ss += v * v; double inv = 1.0 / std::sqrt(ss / D + eps);
            for (int64_t d = 0; d < D; ++d) yp[r * D + d] = (float)(t[d] * inv * gp[d]); }
    }
    return ACLNN_SUCCESS;
}
// GeGlu: in[...,2D] -> out[...,D]; out = gelu(a)·b where a,b are the two halves.
aclnnStatus run_geglu(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *in = e->a; aclTensor *o = e->out;
    int64_t D = o->viewDims.back(); int64_t R = in->numel() / (2 * D); const float *xp = FP(in); float *op = FP(o);
    for (int64_t r = 0; r < R; ++r) for (int64_t d = 0; d < D; ++d) {
        double a = xp[r * 2 * D + d], b = xp[r * 2 * D + D + d];
        double act = 0.5 * a * (1.0 + std::erf(a * 0.70710678118654752)); op[r * D + d] = (float)(act * b);
    }
    return ACLNN_SUCCESS;
}

// ======================= Losses =======================
// reduction: 0 none, 1 mean, 2 sum (test passes 1 = mean).
double reduce_div(int64_t reduction, int64_t n) { return reduction == 1 ? (double)n : 1.0; }

aclnnStatus run_mseloss(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *p = e->a, *t = e->b; aclTensor *o = e->out; int64_t red = (int64_t)e->dim;
    int64_t n = p->numel(); const float *pp = FP(p), *tp = FP(t); float *op = FP(o);
    if (red == 0) { for (int64_t i = 0; i < n; ++i) { double d = (double)pp[i] - tp[i]; op[i] = (float)(d * d); } return ACLNN_SUCCESS; }  // none: per-element
    double sum = 0; for (int64_t i = 0; i < n; ++i) { double d = (double)pp[i] - tp[i]; sum += d * d; }
    if (red == 1) sum /= n; op[0] = (float)sum; return ACLNN_SUCCESS;
}
aclnnStatus run_mseloss_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *p = e->b, *t = e->c; aclTensor *gp = e->out; int64_t red = (int64_t)e->dim;
    int64_t n = p->numel(); double div = (red == 1) ? (double)n : 1.0;   // mean divides by n; none/sum don't
    const float *gop = FP(go), *pp = FP(p), *tp = FP(t); float *o = FP(gp); double g0 = gop[0]; bool gscalar = (go->numel() == 1);
    for (int64_t i = 0; i < n; ++i) { double gg = gscalar ? g0 : gop[i]; o[i] = (float)(gg * 2.0 * (pp[i] - tp[i]) / div); }   // none: per-element gradOut
    return ACLNN_SUCCESS;
}
aclnnStatus run_bce_logits(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *t = e->b; aclTensor *o = e->out; int64_t red = (int64_t)e->dim;
    int64_t n = x->numel(); const float *xp = FP(x), *tp = FP(t);
    double sum = 0; for (int64_t i = 0; i < n; ++i) sum += std::max((double)xp[i], 0.0) - (double)xp[i] * tp[i] + std::log1p(std::exp(-std::fabs((double)xp[i])));
    if (red == 1) sum /= n; FP(o)[0] = (float)sum; return ACLNN_SUCCESS;
}
// NLLLoss: input is logProb[N,C], target[N] (int64); loss = -mean(logProb[n, target[n]]).
aclnnStatus run_nllloss(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *lp = e->a, *t = e->b; aclTensor *o = e->out; int64_t red = (int64_t)e->dim;
    int64_t N = lp->viewDims[0], C = lp->viewDims[1]; const float *lpp = FP(lp); const int64_t *tp = (const int64_t *)t->data + t->offset; float *op = FP(o);
    if (red == 0) { for (int64_t r = 0; r < N; ++r) op[r] = (float)(-lpp[r * C + tp[r]]); return ACLNN_SUCCESS; }  // none: per-sample [N]
    double sum = 0; for (int64_t r = 0; r < N; ++r) sum += -lpp[r * C + tp[r]];
    if (red == 1) sum /= N; op[0] = (float)sum; return ACLNN_SUCCESS;
}
// CrossEntropyLoss: logits[N,C], target[N]; = NLL(logsoftmax(logits)).
aclnnStatus run_celoss(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *lg = e->a, *t = e->b; aclTensor *o = e->out; int64_t red = (int64_t)e->dim;
    int64_t N = lg->viewDims[0], C = lg->viewDims[1]; const float *lgp = FP(lg); const int64_t *tp = (const int64_t *)t->data + t->offset;
    double sum = 0;
    for (int64_t r = 0; r < N; ++r) { double mx = -1e30; for (int64_t c = 0; c < C; ++c) mx = std::max(mx, (double)lgp[r * C + c]);
        double se = 0; for (int64_t c = 0; c < C; ++c) se += std::exp(lgp[r * C + c] - mx);
        sum += -(lgp[r * C + tp[r]] - mx - std::log(se)); }
    if (red == 1) sum /= N; FP(o)[0] = (float)sum; return ACLNN_SUCCESS;
}
// CrossEntropyLossBackward: gradLogits = gradOut·(softmax(logits) - onehot(target)) / N (mean).
aclnnStatus run_celoss_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *lg = e->b, *t = e->c; aclTensor *gl = e->out; int64_t red = (int64_t)e->dim;
    int64_t N = lg->viewDims[0], C = lg->viewDims[1]; double g0 = FP(go)[0], div = reduce_div(red, N);
    const float *lgp = FP(lg); const int64_t *tp = (const int64_t *)t->data + t->offset; float *o = FP(gl);
    for (int64_t r = 0; r < N; ++r) { double mx = -1e30; for (int64_t c = 0; c < C; ++c) mx = std::max(mx, (double)lgp[r * C + c]);
        double se = 0; for (int64_t c = 0; c < C; ++c) se += std::exp(lgp[r * C + c] - mx);
        for (int64_t c = 0; c < C; ++c) { double sm = std::exp(lgp[r * C + c] - mx) / se; o[r * C + c] = (float)(g0 * (sm - (c == tp[r] ? 1.0 : 0.0)) / div); } }
    return ACLNN_SUCCESS;
}

// ======================= Optimizer =======================
// ApplyAdamW: single-step in-place param/m/v update.
aclnnStatus run_adamw(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *p = const_cast<aclTensor *>(e->a), *m = const_cast<aclTensor *>(e->b), *v = const_cast<aclTensor *>(e->c); const aclTensor *g = e->mask;
    int64_t n = p->numel(); double lr = e->dscalars[0], b1 = e->dscalars[1], b2 = e->dscalars[2], eps = e->dscalars[3], wd = e->dscalars[4]; int64_t step = (int64_t)e->dim;
    float *pp = FP(p), *mp = FP(m), *vp = FP(v); const float *gp = FP(g);
    double bc1 = 1.0 - std::pow(b1, step), bc2 = 1.0 - std::pow(b2, step);
    for (int64_t i = 0; i < n; ++i) {
        double mi = b1 * mp[i] + (1.0 - b1) * gp[i], vi = b2 * vp[i] + (1.0 - b2) * (double)gp[i] * gp[i];
        double mh = mi / bc1, vh = vi / bc2; pp[i] = (float)(pp[i] - lr * (mh / (std::sqrt(vh) + eps) + wd * pp[i])); mp[i] = (float)mi; vp[i] = (float)vi;
    }
    return ACLNN_SUCCESS;
}

// ======================= Attention =======================
// Naive batched BNSD attention, dtype-aware (fp32/fp16/bf16). GQA via Hq%Hkv==0. causal = bottom-right aligned.
aclnnStatus run_attn_perf(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *q = e->a, *k = e->b, *v = e->c, *mask = e->mask; aclTensor *o = e->out;
    int64_t B = q->viewDims[0], Nq = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3];
    int64_t Nkv = k->viewDims[1], Skv = k->viewDims[2]; double scale = e->alpha; bool causal = e->causal; int64_t off = Skv - Sq;
    if (Nkv == 0 || (Nq % Nkv) != 0) return ACLNN_ERR_PARAM_INVALID;
    const uint8_t *mp = mask ? ((const uint8_t *)mask->data + mask->offset) : nullptr;
    std::vector<double> sc(Skv);
    for (int64_t bi = 0; bi < B; ++bi) for (int64_t h = 0; h < Nq; ++h) {
        int64_t kvh = h / (Nq / Nkv), qf = bi * Nq + h, kf = bi * Nkv + kvh;
        for (int64_t i = 0; i < Sq; ++i) {
            double mx = -1e30;
            for (int64_t j = 0; j < Skv; ++j) { double d = 0; for (int64_t t = 0; t < D; ++t) d += ld(q, (qf * Sq + i) * D + t) * ld(k, (kf * Skv + j) * D + t);
                d *= scale; bool blk = (causal && j > i + off) || (mp && mp[(bi * Sq + i) * Skv + j]); if (blk) d = -1e30; sc[j] = d; mx = std::max(mx, d); }
            double sum = 0; for (int64_t j = 0; j < Skv; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            for (int64_t t = 0; t < D; ++t) { double acc = 0; for (int64_t j = 0; j < Skv; ++j) acc += sc[j] / sum * ld(v, (kf * Skv + j) * D + t);
                st(o, (qf * Sq + i) * D + t, acc); }
        }
    }
    return ACLNN_SUCCESS;
}
// FlashAttentionScoreBackward (fp32, standard MHA): dQ/dK/dV.
aclnnStatus run_attn_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *q = e->a, *k = e->b, *v = e->c, *dy = e->mean; aclTensor *dq = e->out, *dk = e->out2, *dv = e->mask ? const_cast<aclTensor *>(e->mask) : nullptr;
    int64_t B = q->viewDims[0], N = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3], Skv = k->viewDims[2];
    double scale = e->alpha; bool causal = e->causal; int64_t off = Skv - Sq;
    const float *Q = FP(q), *K = FP(k), *V = FP(v), *dO = FP(dy); float *dQ = FP(dq), *dK = FP(dk), *dV = FP(dv);
    std::vector<double> P(Sq * Skv), dP(Sq * Skv);
    for (int64_t bn = 0; bn < B * N; ++bn) {
        for (int64_t i = 0; i < Sq; ++i) { double mx = -1e30;
            for (int64_t j = 0; j < Skv; ++j) { double sv = 0; for (int64_t d = 0; d < D; ++d) sv += (double)Q[(bn * Sq + i) * D + d] * K[(bn * Skv + j) * D + d]; sv *= scale; if (causal && j > i + off) sv = -1e30; P[i * Skv + j] = sv; mx = std::max(mx, sv); }
            double sm = 0; for (int64_t j = 0; j < Skv; ++j) { P[i * Skv + j] = std::exp(P[i * Skv + j] - mx); sm += P[i * Skv + j]; } for (int64_t j = 0; j < Skv; ++j) P[i * Skv + j] /= sm; }
        for (int64_t i = 0; i < Sq; ++i) for (int64_t j = 0; j < Skv; ++j) { double dp = 0; for (int64_t d = 0; d < D; ++d) dp += (double)dO[(bn * Sq + i) * D + d] * V[(bn * Skv + j) * D + d]; dP[i * Skv + j] = dp; }
        for (int64_t j = 0; j < Skv; ++j) for (int64_t d = 0; d < D; ++d) { double a = 0; for (int64_t i = 0; i < Sq; ++i) a += P[i * Skv + j] * dO[(bn * Sq + i) * D + d]; dV[(bn * Skv + j) * D + d] = (float)a; }
        for (int64_t i = 0; i < Sq; ++i) { double dot = 0; for (int64_t j = 0; j < Skv; ++j) dot += P[i * Skv + j] * dP[i * Skv + j]; for (int64_t j = 0; j < Skv; ++j) dP[i * Skv + j] = P[i * Skv + j] * (dP[i * Skv + j] - dot); }
        for (int64_t i = 0; i < Sq; ++i) for (int64_t d = 0; d < D; ++d) { double a = 0; for (int64_t j = 0; j < Skv; ++j) a += dP[i * Skv + j] * K[(bn * Skv + j) * D + d]; dQ[(bn * Sq + i) * D + d] = (float)(a * scale); }
        for (int64_t j = 0; j < Skv; ++j) for (int64_t d = 0; d < D; ++d) { double a = 0; for (int64_t i = 0; i < Sq; ++i) a += dP[i * Skv + j] * Q[(bn * Sq + i) * D + d]; dK[(bn * Skv + j) * D + d] = (float)(a * scale); }
    }
    return ACLNN_SUCCESS;
}
// PagedAttention (GQA decode): q[B,Nq,Sq,D]; kCache/vCache[blocks,blockSize,Nkv,D]; blockTable[B,maxBlocks]; contextLens[B].
aclnnStatus run_paged(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *q = e->a, *kc = e->b, *vc = e->c, *bt = e->mean, *cl = e->rstd; aclTensor *o = e->out;
    int64_t B = q->viewDims[0], Nq = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3];
    int64_t blockSize = kc->viewDims[1], Nkv = kc->viewDims[2], maxBlocks = bt->viewDims[1]; double scale = e->alpha;
    if (Nkv == 0 || (Nq % Nkv) != 0) return ACLNN_ERR_PARAM_INVALID;
    const int32_t *btp = (const int32_t *)bt->data + bt->offset, *clp = (const int32_t *)cl->data + cl->offset;
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Nq; ++h) { int64_t kvh = h / (Nq / Nkv); int64_t L = clp[b];
        for (int64_t i = 0; i < Sq; ++i) { std::vector<double> sc(L); double mx = -1e30;
            for (int64_t p = 0; p < L; ++p) { int64_t blk = btp[b * maxBlocks + p / blockSize], offb = p % blockSize; double d = 0;
                for (int64_t t = 0; t < D; ++t) d += ld(q, (((b * Nq + h) * Sq) + i) * D + t) * ld(kc, ((blk * blockSize + offb) * Nkv + kvh) * D + t); d *= scale; sc[p] = d; mx = std::max(mx, d); }
            double sum = 0; for (int64_t p = 0; p < L; ++p) { sc[p] = std::exp(sc[p] - mx); sum += sc[p]; }
            for (int64_t t = 0; t < D; ++t) { double acc = 0; for (int64_t p = 0; p < L; ++p) { int64_t blk = btp[b * maxBlocks + p / blockSize], offb = p % blockSize; acc += sc[p] / sum * ld(vc, ((blk * blockSize + offb) * Nkv + kvh) * D + t); }
                st(o, (((b * Nq + h) * Sq) + i) * D + t, acc); } } }
    return ACLNN_SUCCESS;
}

RUN(ex_layernorm, run_layernorm)
RUN(ex_layernorm_bwd, run_layernorm_bwd)
RUN(ex_rmsnorm_bwd, run_rmsnorm_bwd)
RUN(ex_groupnorm, run_groupnorm)
RUN(ex_batchnorm, run_batchnorm)
RUN(ex_batchnorm_train, run_batchnorm_train)
RUN(ex_deepnorm, run_deepnorm)
RUN(ex_addnorm, run_addnorm)
RUN(ex_geglu, run_geglu)
RUN(ex_mseloss, run_mseloss)
RUN(ex_mseloss_bwd, run_mseloss_bwd)
RUN(ex_bce, run_bce_logits)
RUN(ex_nllloss, run_nllloss)
RUN(ex_celoss, run_celoss)
RUN(ex_celoss_bwd, run_celoss_bwd)
RUN(ex_adamw, run_adamw)
RUN(ex_attn_perf, run_attn_perf)
RUN(ex_attn_bwd, run_attn_bwd)
RUN(ex_paged, run_paged)
} // namespace

extern "C" {
aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight,
        const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = 1; if (normalizedShape) for (auto d : normalizedShape->v) D *= d; else D = input->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = input; e->b = weight; e->c = bias; e->out = out; e->out2 = meanOut; e->mean = rstdOut;
    e->reduceCount = D; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLayerNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_layernorm(w, wss, e, s); }

aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        const aclIntArray *normalizedShape, double eps, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = 1; if (normalizedShape) for (auto d : normalizedShape->v) D *= d; else D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gamma; e->out = gradX; e->out2 = gradGamma; e->mask = gradBeta;
    e->reduceCount = D; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLayerNormBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_layernorm_bwd(w, wss, e, s); }

aclnnStatus aclnnRmsNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gamma || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gamma; e->out = gradX; e->out2 = gradGamma; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_rmsnorm_bwd(w, wss, e, s); }

aclnnStatus aclnnGroupNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
        int64_t numGroups, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->dim = numGroups; e->eps = eps; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_groupnorm(w, wss, e, s); }

aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
        const aclTensor *mean, const aclTensor *var, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mean || !var || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->mean = mean; e->rstd = var; e->eps = eps; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_batchnorm(w, wss, e, s); }

aclnnStatus aclnnBatchNormTrainingGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta,
        aclTensor *runningMean, aclTensor *runningVar, double momentum, double eps, aclTensor *out, aclTensor *savedMean, aclTensor *savedInvStd,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = beta; e->mean = runningMean; e->rstd = runningVar;
    e->out = out; e->out2 = savedMean; e->mask = savedInvStd; e->alpha = momentum; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormTraining(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_batchnorm_train(w, wss, e, s); }

aclnnStatus aclnnDeepNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gx, const aclTensor *gamma, const aclTensor *beta,
        double alpha, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gx || !gamma || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gx; e->c = gamma; e->mask = beta; e->alpha = alpha; e->eps = eps; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDeepNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_deepnorm(w, wss, e, s); }

aclnnStatus aclnnAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
        double eps, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = y; e->out2 = residualSum; e->causal = false; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addnorm(w, wss, e, s); }

aclnnStatus aclnnAddLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
        const aclTensor *beta, double eps, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->mask = beta; e->out = y; e->out2 = residualSum; e->causal = true; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddLayerNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addnorm(w, wss, e, s); }

aclnnStatus aclnnGeGluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGeGlu(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_geglu(w, wss, e, s); }

aclnnStatus aclnnMseLossGetWorkspaceSize(const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!pred || !target || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = pred; e->b = target; e->out = out; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMseLoss(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mseloss(w, wss, e, s); }

aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *gradPred, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !pred || !target || !gradPred || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = pred; e->c = target; e->out = gradPred; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMseLossBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mseloss_bwd(w, wss, e, s); }

aclnnStatus aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!logits || !target || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = logits; e->b = target; e->out = out; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBinaryCrossEntropyWithLogits(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_bce(w, wss, e, s); }

aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor *logProb, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!logProb || !target || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = logProb; e->b = target; e->out = out; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnNLLLoss(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_nllloss(w, wss, e, s); }

aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!logits || !target || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = logits; e->b = target; e->out = out; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCrossEntropyLoss(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_celoss(w, wss, e, s); }

aclnnStatus aclnnCrossEntropyLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *gradLogits, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !logits || !target || !gradLogits || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = logits; e->c = target; e->out = gradLogits; e->dim = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCrossEntropyLossBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_celoss_bwd(w, wss, e, s); }

aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
        double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = param; e->b = m; e->c = v; e->mask = grad; e->dscalars = {lr, beta1, beta2, eps, weightDecay}; e->dim = step; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdamW(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_adamw(w, wss, e, s); }

aclnnStatus aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)headNum; if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->mask = attenMask; e->out = out; e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionScoreHighPerf(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_attn_perf(w, wss, e, s); }

aclnnStatus aclnnFlashAttentionScoreBackwardGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal,
        aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *ws, aclOpExecutor **ex) {
    (void)attenMask; (void)headNum; if (!q || !k || !v || !dy || !dq || !dk || !dv || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->mean = dy; e->out = dq; e->out2 = dk; e->mask = dv; e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionScoreBackward(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_attn_bwd(w, wss, e, s); }

aclnnStatus aclnnPagedAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *kCache, const aclTensor *vCache,
        const aclTensor *blockTable, const aclTensor *contextLens, double scaleValue, int64_t numHeads, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)numHeads; if (!query || !kCache || !vCache || !blockTable || !contextLens || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = query; e->b = kCache; e->c = vCache; e->mean = blockTable; e->rstd = contextLens; e->out = out; e->alpha = scaleValue; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPagedAttention(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_paged(w, wss, e, s); }
} // extern "C"
