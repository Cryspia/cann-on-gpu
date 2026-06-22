// Activation / pooling / upsample / gather backward family. Host-side over unified memory
// (device ptr == host ptr under MTLStorageModeShared). Each op = the standard two-phase aclnn
// contract: GetWorkspaceSize stashes plan state in aclOpExecutor and returns *ws=0; the Execute
// entry drains the stream, computes, then deletes the executor. Math/semantics mirror the test
// spec (tests/test_actbwd.cpp) and the CUDA reference (cuda/src/ops/backward.cu et al.).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
constexpr double INV_SQRT2 = 0.7071067811865476;   // 1/sqrt(2)
constexpr double INV_SQRT2PI = 0.3989422804014327; // 1/sqrt(2*pi)

// ---- elementwise activation backward: gradInput = gradOutput * f'(val). m selects f'. ----
// val is `self` (raw x) for most ops; for Sigmoid/Tanh `self` already carries the activation output y.
double act_deriv(int m, double v, double p) {
    switch (m) {
        case 0:  return v > 0 ? 1.0 : 0.0;                                            // Relu
        case 1:  return 0.5 * (1 + std::erf(v * INV_SQRT2)) + v * INV_SQRT2PI * std::exp(-0.5 * v * v); // Gelu (erf)
        case 2:  { double s = sigm(v); return s * (1 + v * (1 - s)); }                // Silu
        case 3:  return sigm(v);                                                      // Softplus
        case 4:  return v < -3 ? 0.0 : (v > 3 ? 1.0 : (2 * v + 3) / 6);               // Hardswish
        case 5:  return v * (1 - v);                                                  // Sigmoid (v=y)
        case 6:  return 1 - v * v;                                                    // Tanh (v=y)
        case 7:  { double g = 0.7978845608028654, u = g * (v + 0.044715 * v * v * v), t = std::tanh(u), du = g * (1 + 0.134145 * v * v); return 0.5 * (1 + t) + 0.5 * v * (1 - t * t) * du; } // FastGelu / tanh-Gelu
        case 8:  return (v > -3 && v < 3) ? 1.0 / 6 : 0.0;                            // Hardsigmoid
        case 9:  return 1.0 / (1.0 + std::exp(v));                                    // LogSigmoid: d/dx log(sigmoid(x)) = sigmoid(-x)
        case 10: { double sp = v > 0 ? v + std::log1p(std::exp(-v)) : std::log1p(std::exp(v)); double w = std::tanh(sp), s = sigm(v); return w + v * (1 - w * w) * s; } // Mish
        case 11: { const double sc = 1.0507009873554805, al = 1.6732632423543772; return sc * (v > 0 ? 1.0 : al * std::exp(v)); } // Selu
        case 12: return (v > p || v < -p) ? 1.0 : 0.0;                                // Hardshrink / Softshrink (lambda=p)
        case 13: return v > p ? 1.0 : 0.0;                                            // Threshold (threshold=p)
        case 14: return v > 0 ? 1.0 : p;                                              // LeakyRelu (negativeSlope=p)
        default: return 0.0;
    }
}
aclnnStatus run_actbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *g = e->a, *v = e->b; aclTensor *o = e->out;
    int64_t n = o->numel(); const float *gp = FP(g), *vp = FP(v); float *op = FP(o);
    int m = (int)e->m; double p = e->alpha, al = e->eps;
    for (int64_t i = 0; i < n; ++i) {
        double d = (m == 15) ? (vp[i] > 0 ? 1.0 : al * std::exp(vp[i])) : act_deriv(m, vp[i], p);  // 15: Elu (alpha on exp branch)
        op[i] = (float)(gp[i] * d);
    }
    return ACLNN_SUCCESS;
}
// ---- hardtanh backward: 1 inside (min,max) else 0 ----
aclnnStatus run_hardtanhbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *g = e->a, *v = e->b; aclTensor *o = e->out; int64_t n = o->numel();
    const float *gp = FP(g), *vp = FP(v); float *op = FP(o); double lo = e->alpha, hi = e->eps;
    for (int64_t i = 0; i < n; ++i) op[i] = (vp[i] > lo && vp[i] < hi) ? gp[i] : 0.f;
    return ACLNN_SUCCESS;
}
// ---- softmax / logsoftmax backward over dim ----
aclnnStatus run_softmaxbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *g = e->a, *y = e->b; aclTensor *o = e->out; bool logsm = (e->m != 0);
    int nd = (int)y->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd;
    int64_t outer = 1, D = y->viewDims[dim], inner = 1;
    for (int d = 0; d < dim; ++d) outer *= y->viewDims[d];
    for (int d = dim + 1; d < nd; ++d) inner *= y->viewDims[d];
    const float *gp = FP(g), *yp = FP(y); float *op = FP(o);
    for (int64_t oo = 0; oo < outer; ++oo) for (int64_t ii = 0; ii < inner; ++ii) {
        int64_t base = oo * D * inner + ii; double tot = 0;
        for (int64_t d = 0; d < D; ++d) { int64_t idx = base + d * inner; tot += logsm ? gp[idx] : (double)gp[idx] * yp[idx]; }
        for (int64_t d = 0; d < D; ++d) { int64_t idx = base + d * inner;
            op[idx] = (float)(logsm ? gp[idx] - std::exp((double)yp[idx]) * tot : (double)yp[idx] * ((double)gp[idx] - tot)); }
    }
    return ACLNN_SUCCESS;
}
// ---- gather backward (scatter-add): gradInput[index[...],j] += gradOut[...,j] along dim ----
// Test uses dim=0, index shape [L], gradOut [L,B], gradInput [A,B]: ref[idx[l]*B+b]+=go[l*B+b].
aclnnStatus run_gatherbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *idx = e->b; aclTensor *gi = e->out;
    int nd = (int)go->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd;
    int64_t outer = 1, D = go->viewDims[dim], inner = 1;
    for (int d = 0; d < dim; ++d) outer *= go->viewDims[d];
    for (int d = dim + 1; d < nd; ++d) inner *= go->viewDims[d];
    int64_t Din = gi->viewDims[dim];
    const float *gop = FP(go); float *gip = FP(gi); const int64_t *ix = (const int64_t *)idx->data + idx->offset;
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    // index_select-style backward: index has one entry per dim-slice (shape matches gradOut along `dim`),
    // applied across all inner positions. ref[idx[d]*inner+ii] += gradOut[d*inner+ii] (per outer block).
    bool perElement = (idx->numel() == go->numel());   // full-shape index (PyTorch gather) vs per-slice index
    for (int64_t oo = 0; oo < outer; ++oo) for (int64_t d = 0; d < D; ++d) for (int64_t ii = 0; ii < inner; ++ii) {
        int64_t srcoff = (oo * D + d) * inner + ii;
        int64_t tgt = perElement ? ix[srcoff] : ix[d];
        int64_t dstoff = (oo * Din + tgt) * inner + ii;
        gip[dstoff] += gop[srcoff];
    }
    return ACLNN_SUCCESS;
}
// ---- upsample nearest 2d backward: scatter gradOut to the source pixel it sampled ----
// src index: ih=min(oh*H/oH,H-1), iw=min(ow*W/oW,W-1). Layout [N,C,oH,oW] -> [N,C,H,W].
aclnnStatus run_upnearestbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int64_t N = go->viewDims[0], C = go->viewDims[1], oH = go->viewDims[2], oW = go->viewDims[3];
    int64_t H = gi->viewDims[2], W = gi->viewDims[3]; int64_t NC = N * C;
    const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        int64_t ih = std::min(oh * H / oH, H - 1), iw = std::min(ow * W / oW, W - 1);
        gip[(nc * H + ih) * W + iw] += gop[(nc * oH + oh) * oW + ow];
    }
    return ACLNN_SUCCESS;
}
// ---- upsample bilinear 2d backward: distribute gradOut to its 4 source neighbors ----
aclnnStatus run_upbilinearbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out; bool align = (e->m != 0);
    int64_t N = go->viewDims[0], C = go->viewDims[1], oH = go->viewDims[2], oW = go->viewDims[3];
    int64_t H = gi->viewDims[2], W = gi->viewDims[3]; int64_t NC = N * C;
    const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        float fh, fw;
        if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
        else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw; }
        int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, (int)H - 1), w1 = std::min(w0 + 1, (int)W - 1);
        float dh = fh - h0, dw = fw - w0, gv = gop[(nc * oH + oh) * oW + ow]; float *pp = &gip[nc * H * W];
        pp[h0 * W + w0] += gv * (1 - dh) * (1 - dw); pp[h0 * W + w1] += gv * (1 - dh) * dw;
        pp[h1 * W + w0] += gv * dh * (1 - dw);       pp[h1 * W + w1] += gv * dh * dw;
    }
    return ACLNN_SUCCESS;
}
// ---- adaptive avg pool 2d backward: each output cell spreads its grad uniformly over its source window ----
aclnnStatus run_adaptpoolbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int64_t N = go->viewDims[0], C = go->viewDims[1], oH = go->viewDims[2], oW = go->viewDims[3];
    int64_t H = gi->viewDims[2], W = gi->viewDims[3]; int64_t NC = N * C;
    const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        int64_t hs = oh * H / oH, he = (oh + 1) * H / oH + ((oh + 1) * H % oH ? 1 : 0);
        int64_t ws = ow * W / oW, we = (ow + 1) * W / oW + ((ow + 1) * W % oW ? 1 : 0);
        double share = gop[(nc * oH + oh) * oW + ow] / (double)((he - hs) * (we - ws));
        for (int64_t h = hs; h < he; ++h) for (int64_t w = ws; w < we; ++w) gip[(nc * H + h) * W + w] += (float)share;
    }
    return ACLNN_SUCCESS;
}
// ---- max pool 2d with indices (forward): out = max over window; indices = flattened src offset (ih*W+iw) ----
aclnnStatus run_maxpool(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a; aclTensor *o = e->out, *id = e->out2;
    int64_t N = x->viewDims[0], C = x->viewDims[1], H = x->viewDims[2], W = x->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3]; int64_t NC = N * C;
    int kh = (int)e->axes[0], kw = (int)e->axes[1], sh = (int)e->stride[0], sw = (int)e->stride[1], ph = (int)e->pad[0], pw = (int)e->pad[1];
    const float *xp = FP(x); float *op = FP(o); int64_t *ip = (int64_t *)id->data + id->offset;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        float best = -INFINITY; int64_t bi = 0;
        for (int a = 0; a < kh; ++a) for (int b = 0; b < kw; ++b) {
            int64_t ih = oh * sh - ph + a, iw = ow * sw - pw + b;
            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
            float val = xp[(nc * H + ih) * W + iw];
            if (val > best) { best = val; bi = ih * W + iw; }
        }
        op[(nc * oH + oh) * oW + ow] = best; ip[(nc * oH + oh) * oW + ow] = bi;
    }
    return ACLNN_SUCCESS;
}
// ---- max unpool 2d: scatter input values to indices recorded by max pool ----
aclnnStatus run_maxunpool(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *id = e->b; aclTensor *o = e->out;
    int64_t N = x->viewDims[0], C = x->viewDims[1], iH = x->viewDims[2], iW = x->viewDims[3];
    int64_t H = (int64_t)e->m, W = (int64_t)e->n; int64_t NC = N * C, plane = H * W;
    const float *xp = FP(x); const int64_t *ip = (const int64_t *)id->data + id->offset; float *op = FP(o);
    for (int64_t i = 0, n = o->numel(); i < n; ++i) op[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t p = 0; p < iH * iW; ++p) {
        int64_t src = nc * iH * iW + p; op[nc * plane + ip[src]] = xp[src];
    }
    return ACLNN_SUCCESS;
}
// ---- GLU backward: in[...,2D] split into a=in[...,d], b=in[...,D+d]; gradOut[...,D]. ----
// gradIn[a] = g * act'(a) * b ; gradIn[b] = g * act(a). m: 0 swish/silu, 1 gelu(erf).
aclnnStatus run_glubwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *in = e->b; aclTensor *gi = e->out; bool gelu = (e->m != 0);
    int nd = (int)in->viewDims.size(); int64_t twoD = in->viewDims[nd - 1], D = twoD / 2;
    int64_t rows = in->numel() / twoD;
    const float *gop = FP(go), *inp = FP(in); float *gip = FP(gi);
    for (int64_t r = 0; r < rows; ++r) for (int64_t d = 0; d < D; ++d) {
        double a = inp[r * twoD + d], b = inp[r * twoD + D + d], g = gop[r * D + d], act, dact;
        if (gelu) { double cdf = 0.5 * (1 + std::erf(a * INV_SQRT2)); act = a * cdf; dact = cdf + a * INV_SQRT2PI * std::exp(-0.5 * a * a); }
        else { double sg = sigm(a); act = a * sg; dact = sg + a * sg * (1 - sg); }
        gip[r * twoD + d] = (float)(g * dact * b);
        gip[r * twoD + D + d] = (float)(g * act);
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
// ---- simple activation backward: gradOutput, self, gradInput ----
#define ABWD(NAME, M) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->m = M; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(NAME, run_actbwd)
ABWD(aclnnReluBackward, 0) ABWD(aclnnGeluBackward, 1) ABWD(aclnnSiluBackward, 2) ABWD(aclnnSoftplusBackward, 3)
ABWD(aclnnHardswishBackward, 4) ABWD(aclnnSigmoidBackward, 5) ABWD(aclnnTanhBackward, 6)
ABWD(aclnnFastGeluBackward, 7) ABWD(aclnnHardsigmoidBackward, 8) ABWD(aclnnHardswishBackwardV2, 4)
ABWD(aclnnLogSigmoidBackward, 9) ABWD(aclnnMishBackward, 10) ABWD(aclnnSeluBackward, 11)
#undef ABWD
// GeluBackwardV2: approximate flag selects erf (0) vs tanh (1); test uses tanh form -> m=7
aclnnStatus aclnnGeluBackwardV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, int64_t approximate, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->m = approximate ? 7 : 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnGeluBackwardV2, run_actbwd)
// scalar-param activation backward: shrink lambda / threshold
#define ABWD_S(NAME, M) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *p, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->m = M; e->alpha = p ? p->v : 0.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(NAME, run_actbwd)
ABWD_S(aclnnHardshrinkBackward, 12) ABWD_S(aclnnSoftshrinkBackward, 12) ABWD_S(aclnnThresholdBackward, 13)
#undef ABWD_S
aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double negativeSlope, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->m = 14; e->alpha = negativeSlope; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnLeakyReluBackward, run_actbwd)
aclnnStatus aclnnEluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double alpha, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->m = 15; e->eps = alpha; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnEluBackward, run_actbwd)
aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *minVal, const aclScalar *maxVal, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->alpha = minVal->v; e->eps = maxVal->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnHardtanhBackward, run_hardtanhbwd)
// softmax / logsoftmax backward
aclnnStatus aclnnSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t dim, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = output; e->out = gradInput; e->dim = dim; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnSoftmaxBackward, run_softmaxbwd)
aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t dim, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = output; e->out = gradInput; e->dim = dim; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnLogSoftmaxBackward, run_softmaxbwd)
// gather backward
aclnnStatus aclnnGatherBackwardGetWorkspaceSize(const aclTensor *gradOut, int64_t dim, const aclTensor *index, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = index; e->out = gradInput; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnGatherBackward, run_gatherbwd)
// upsample / pool backward
aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnUpsampleNearest2dBackward, run_upnearestbwd)
aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradInput; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnUpsampleBilinear2dBackward, run_upbilinearbwd)
aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAdaptiveAvgPool2dBackward, run_adaptpoolbwd)
aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->out2 = indices;
    e->axes = {kernel->v[0], kernel->v.size() > 1 ? kernel->v[1] : kernel->v[0]};
    e->stride[0] = stride && !stride->v.empty() ? stride->v[0] : kernel->v[0];
    e->stride[1] = stride && stride->v.size() > 1 ? stride->v[1] : e->stride[0];
    e->pad[0] = padding && !padding->v.empty() ? padding->v[0] : 0;
    e->pad[1] = padding && padding->v.size() > 1 ? padding->v[1] : e->pad[0];
    *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnMaxPool2dWithIndices, run_maxpool)
aclnnStatus aclnnMaxUnpool2dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->b = indices; e->out = out; e->m = H; e->n = W; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnMaxUnpool2d, run_maxunpool)
// GLU gradients
aclnnStatus aclnnSwiGluGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->out = gradIn; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnSwiGluGrad, run_glubwd)
aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->out = gradIn; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnGeGluBackward, run_glubwd)
} // extern "C"
