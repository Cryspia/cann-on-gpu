// Pooling / vision / conv "gap" family — the ~23 ops still missing from the Metal backend versus the
// CUDA reference. Host-side over unified memory (device ptr == host ptr under MTLStorageModeShared),
// matching the rest of the Metal backend (pool.mm / conv.mm). Standard two-phase aclnn contract:
// GetWorkspaceSize news the executor + stashes plan state + sets *ws=0; Execute drains the stream,
// computes, then deletes the executor.
//
// Semantics mirror the CUDA backend:
//   upsample_ext.cu (nearest 1d/3d, nearest-exact 1d/2d/3d, linear1d, trilinear3d, GlobalMaxPool),
//   pool_ext.cu     (MaxPool generic = MaxPool2d, MaxPool2dWithMask = max+flat-indices,
//                    RoiPoolingGradWithArgMax = scatter-add by argmax),
//   conv.cu / ssm.cu (ConvDepthwise2d = grouped conv, FusedCausalConv1d = causal depthwise conv1d,
//                    MultiScaleDeformableAttnFunction, TransConvolutionWeight = copy),
//   loss.cu         (GridSampler2D/3D = bilinear/trilinear zeros-pad; *AA upsample fall back to non-AA).
//
// This file MUST include the canonical header so the compiler validates every op signature against the
// macros there; op prototypes are NOT hand-written.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace pcg {  // file-local namespace so helpers cannot collide with other translation units

float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *IP(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline float silu(float x) { return x / (1.f + std::exp(-x)); }
inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// IntArray accessor with default fallback.
inline int64_t IA(const aclIntArray *a, int i, int64_t def) { return (a && (int)a->v.size() > i) ? a->v[i] : def; }

// nearest source index. exact => half-pixel sampling (floor((o+0.5)*isz/osz)), else floor(o*isz/osz).
inline int nsrc(int o, int isz, int osz, bool exact) {
    int i = exact ? (int)std::floor((o + 0.5f) * isz / osz) : (int)std::floor((float)o * isz / osz);
    return i < 0 ? 0 : (i >= isz ? isz - 1 : i);
}
// linear coord mapping (align-corners aware) -> lo/hi index + fraction.
inline void lcoord(int o, int isz, int osz, bool align, int &lo, int &hi, float &fr) {
    float s;
    if (align) s = osz > 1 ? (float)o * (isz - 1) / (osz - 1) : 0.f;
    else       s = osz > 0 ? ((o + 0.5f) * isz / osz - 0.5f) : 0.f;
    if (s < 0) s = 0;
    int b = (int)std::floor(s); fr = s - b;
    lo = b < 0 ? 0 : (b >= isz ? isz - 1 : b);
    hi = lo + 1 >= isz ? isz - 1 : lo + 1;
}
inline float cubicw(float t) { t = std::fabs(t); float a = -0.75f;
    if (t <= 1) return ((a + 2) * t - (a + 3)) * t * t + 1;
    if (t < 2)  return (((t - 5) * t + 8) * t - 4) * a;
    return 0.f; }
// bilinear sample of plane p[H,W] at fractional (fy,fx); zero padding outside [-1,H]/[-1,W].
inline float bilin(const float *p, int H, int W, float fy, float fx) {
    if (fy < -1 || fy > H || fx < -1 || fx > W) return 0.f;
    fy = std::max(fy, 0.f); fx = std::max(fx, 0.f);
    int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
    float dy = fy - y0, dx = fx - x0;
    return (1 - dy) * (1 - dx) * p[y0 * W + x0] + (1 - dy) * dx * p[y0 * W + x1]
         + dy * (1 - dx) * p[y1 * W + x0] + dy * dx * p[y1 * W + x1];
}

// ====================================================================================================
// Run helpers. Each Execute computes from executor plan state stashed by GetWorkspaceSize.
// ====================================================================================================

// ---- Generic N-D nearest / linear upsample. nsp spatial dims (1/2/3); interp 0 nearest, 1 nearest-exact,
//      2 linear/bilinear/trilinear. Output spatial sizes taken from `out`. ----
aclnnStatus run_upsample(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    int nsp = (int)e->reduceCount; bool exactOrAlign = (e->m != 0); int interp = (int)e->dim;
    const auto &S = e->a->viewDims, &O = e->out->viewDims; int rank = (int)S.size(), sp0 = rank - nsp;
    int is[3] = {1, 1, 1}, os[3] = {1, 1, 1};
    for (int d = 0; d < nsp; d++) { is[3 - nsp + d] = (int)S[sp0 + d]; os[3 - nsp + d] = (int)O[sp0 + d]; }
    int64_t NC = 1; for (int d = 0; d < sp0; d++) NC *= S[d];
    int i0 = is[0], i1 = is[1], i2 = is[2], o0 = os[0], o1 = os[1], o2 = os[2];
    const float *xp = FP(e->a); float *op = FP(e->out);
    int64_t isp = (int64_t)i0 * i1 * i2, osp = (int64_t)o0 * o1 * o2;
    for (int64_t nc = 0; nc < NC; ++nc) {
        const float *p = xp + nc * isp; float *q = op + nc * osp;
        for (int c0 = 0; c0 < o0; ++c0) for (int c1 = 0; c1 < o1; ++c1) for (int c2 = 0; c2 < o2; ++c2) {
            int64_t oi = ((int64_t)c0 * o1 + c1) * o2 + c2;
            if (interp == 2) {
                int l0, h0, l1, h1, l2, h2; float f0, f1, f2;
                lcoord(c0, i0, o0, exactOrAlign, l0, h0, f0);
                lcoord(c1, i1, o1, exactOrAlign, l1, h1, f1);
                lcoord(c2, i2, o2, exactOrAlign, l2, h2, f2);
                float acc = 0;
                for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
                    int z = a ? h0 : l0, y = b ? h1 : l1, x = c ? h2 : l2;
                    float w = (a ? f0 : 1 - f0) * (b ? f1 : 1 - f1) * (c ? f2 : 1 - f2);
                    acc += w * p[((int64_t)z * i1 + y) * i2 + x];
                }
                q[oi] = acc;
            } else {
                int s0 = nsrc(c0, i0, o0, exactOrAlign), s1 = nsrc(c1, i1, o1, exactOrAlign), s2 = nsrc(c2, i2, o2, exactOrAlign);
                q[oi] = p[((int64_t)s0 * i1 + s1) * i2 + s2];
            }
        }
    }
    return ACLNN_SUCCESS;
}

// ---- 2D nearest / bilinear (NCHW) used by Nearest2dV2 / Bilinear2D / *AA aliases. ----
aclnnStatus run_up2d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *a = e->a; aclTensor *o = e->out; bool align = (e->m != 0); int interp = (int)e->dim;
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) { const float *p = xp + nc * H * W; float *q = op + nc * oH * oW;
        for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
            int64_t oi = oh * oW + ow;
            if (interp == 0) {                       // nearest (floor mapping, matches CUDA k_upsample_nearest)
                int64_t ih = std::min((int64_t)(oh * H / oH), H - 1), iw = std::min((int64_t)(ow * W / oW), W - 1);
                q[oi] = p[ih * W + iw];
            } else if (interp == 2) {                // bilinear
                float fh, fw;
                if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
                else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
                fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw;
                int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, (int)H - 1), w1 = std::min(w0 + 1, (int)W - 1);
                float dh = fh - h0, dw = fw - w0;
                q[oi] = p[h0 * W + w0] * (1 - dh) * (1 - dw) + p[h0 * W + w1] * (1 - dh) * dw
                      + p[h1 * W + w0] * dh * (1 - dw) + p[h1 * W + w1] * dh * dw;
            } else {                                 // bicubic (interp==3)
                float fh, fw;
                if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
                else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
                int64_t y0 = (int64_t)std::floor(fh), x0 = (int64_t)std::floor(fw); float dy = fh - y0, dx = fw - x0;
                double acc = 0;
                for (int m = -1; m <= 2; m++) { float wy = cubicw(dy - m); int64_t yy = clampi((int)(y0 + m), 0, (int)H - 1);
                    for (int n = -1; n <= 2; n++) { float wx = cubicw(dx - n); int64_t xx = clampi((int)(x0 + n), 0, (int)W - 1);
                        acc += (double)wy * wx * p[yy * W + xx]; } }
                q[oi] = (float)acc;
            }
        }
    }
    return ACLNN_SUCCESS;
}

// ---- UpsampleBilinear2dBackwardV2: scatter bilinear weights of gradOut into gradIn (NCHW). ----
aclnnStatus run_upbilinear2d_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *go = e->a; aclTensor *gi = e->out; bool align = (e->m != 0);
    int64_t N = gi->viewDims[0], C = gi->viewDims[1], H = gi->viewDims[2], W = gi->viewDims[3];
    int64_t oH = go->viewDims[2], oW = go->viewDims[3], NC = N * C;
    const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) { const float *p = gop + nc * oH * oW; float *q = gip + nc * H * W;
        for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
            float fh, fw;
            if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
            else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
            fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw;
            int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, (int)H - 1), w1 = std::min(w0 + 1, (int)W - 1);
            float dh = fh - h0, dw = fw - w0, g = p[oh * oW + ow];
            q[h0 * W + w0] += (1 - dh) * (1 - dw) * g; q[h0 * W + w1] += (1 - dh) * dw * g;
            q[h1 * W + w0] += dh * (1 - dw) * g;       q[h1 * W + w1] += dh * dw * g;
        }
    }
    return ACLNN_SUCCESS;
}

// ---- GlobalMaxPool: self[N,C,*] -> out[N,C,1,..]; reduce all spatial with max. ----
aclnnStatus run_globalmax(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    int64_t NC = e->m, sp = e->n; const float *xp = FP(e->a); float *op = FP(e->out);
    for (int64_t nc = 0; nc < NC; ++nc) { const float *p = xp + nc * sp; float best = -INFINITY;
        for (int64_t i = 0; i < sp; ++i) best = std::max(best, p[i]); op[nc] = best; }
    return ACLNN_SUCCESS;
}

// ---- MaxPool2d (generic MaxPool): NCHW window max, no indices. kernel/stride/pad in axes. ----
aclnnStatus run_maxpool2d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *a = e->a; aclTensor *o = e->out;
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    int kh = (int)e->axes[0], kw = (int)e->axes[1], sh = (int)e->axes[2], sw = (int)e->axes[3], ph = (int)e->axes[4], pw = (int)e->axes[5];
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) { const float *p = xp + nc * H * W; float *q = op + nc * oH * oW;
        for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
            float best = -INFINITY;
            for (int a2 = 0; a2 < kh; ++a2) for (int b2 = 0; b2 < kw; ++b2) {
                int64_t ih = oh * sh - ph + a2, iw = ow * sw - pw + b2;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                best = std::max(best, p[ih * W + iw]); }
            q[oh * oW + ow] = best;
        }
    }
    return ACLNN_SUCCESS;
}

// ---- MaxPool2dWithMask (= MaxPool2dWithIndices): NCHW window max + flat indices (ih*W+iw). ----
aclnnStatus run_maxpool2d_mask(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *a = e->a; aclTensor *o = e->out, *id = e->out2;
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    int kh = (int)e->axes[0], kw = (int)e->axes[1], sh = (int)e->axes[2], sw = (int)e->axes[3], ph = (int)e->axes[4], pw = (int)e->axes[5];
    const float *xp = FP(a); float *op = FP(o); int64_t *ip = IP(id);
    for (int64_t nc = 0; nc < NC; ++nc) { const float *p = xp + nc * H * W; float *q = op + nc * oH * oW; int64_t *qi = ip + nc * oH * oW;
        for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
            float best = -INFINITY; int64_t bi = 0;
            for (int a2 = 0; a2 < kh; ++a2) for (int b2 = 0; b2 < kw; ++b2) {
                int64_t ih = oh * sh - ph + a2, iw = ow * sw - pw + b2;
                if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                float v = p[ih * W + iw]; if (v > best) { best = v; bi = ih * W + iw; } }
            int64_t oi = oh * oW + ow; q[oi] = best; qi[oi] = bi;
        }
    }
    return ACLNN_SUCCESS;
}

// ---- RoiAlignV2 fwd: in[N,C,H,W], rois[K,5]={batch,x1,y1,x2,y2}(input scale) -> out[K,C,ph,pw]. avg of ratio^2 bilinear samples. ----
aclnnStatus run_roialignv2(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *in = e->a, *rois = e->b; aclTensor *o = e->out;
    int64_t C = in->viewDims[1], H = in->viewDims[2], W = in->viewDims[3];
    int64_t K = o->viewDims[0], ph = o->viewDims[2], pw = o->viewDims[3];
    float scale = (float)e->alpha; int ratio = (int)e->reduceCount; if (ratio <= 0) ratio = 2;
    const float *xp = FP(in), *rp = FP(rois); float *op = FP(o);
    for (int64_t k = 0; k < K; ++k) for (int64_t c = 0; c < C; ++c) for (int64_t py = 0; py < ph; ++py) for (int64_t px = 0; px < pw; ++px) {
        const float *r = rp + k * 5; int b = (int)r[0];
        float x1 = r[1] * scale, y1 = r[2] * scale, x2 = r[3] * scale, y2 = r[4] * scale;
        float rw = std::max(x2 - x1, 1.f), rh = std::max(y2 - y1, 1.f), bw = rw / pw, bh = rh / ph;
        const float *p = xp + (b * C + c) * H * W; double sum = 0; int cnt = ratio * ratio;
        for (int iy = 0; iy < ratio; ++iy) for (int ix = 0; ix < ratio; ++ix) {
            float yy = y1 + py * bh + (iy + 0.5f) * bh / ratio, xx = x1 + px * bw + (ix + 0.5f) * bw / ratio;
            sum += bilin(p, (int)H, (int)W, yy, xx); }
        op[((k * C + c) * ph + py) * pw + px] = (float)(sum / cnt);
    }
    return ACLNN_SUCCESS;
}

// ---- RoiPoolingGradWithArgMax: scatter-add gradOut into gradInput at saved argmax (flat in H*W). ----
aclnnStatus run_roipoolgrad(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *go = e->a, *rois = e->b, *am = e->c; aclTensor *gi = e->out;
    int64_t C = gi->viewDims[1], H = gi->viewDims[2], W = gi->viewDims[3];
    int64_t K = go->viewDims[0], ph = go->viewDims[2], pw = go->viewDims[3];
    const float *gop = FP(go), *rp = FP(rois); const int64_t *amp = IP(am); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t k = 0; k < K; ++k) { const float *r = rp + k * 5; int b = (int)r[0];
        for (int64_t c = 0; c < C; ++c) for (int64_t py = 0; py < ph; ++py) for (int64_t px = 0; px < pw; ++px) {
            int64_t oi = ((k * C + c) * ph + py) * pw + px; int64_t fi = amp[oi];
            if (fi < 0) continue; gip[(b * C + c) * H * W + fi] += gop[oi];
        }
    }
    return ACLNN_SUCCESS;
}

// ---- GridSampler2D bilinear, zeros padding; in[N,C,H,W], grid[N,oH,oW,2] -> out[N,C,oH,oW]. ----
aclnnStatus run_gridsampler2d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *a = e->a, *g = e->b; aclTensor *o = e->out; bool align = (e->m != 0);
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3];
    const float *xp = FP(a), *gp = FP(g); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        const float *gg = gp + ((n * oH + oh) * oW + ow) * 2; float gx = gg[0], gy = gg[1]; float fx, fy;
        if (align) { fx = (gx + 1) * 0.5f * (W - 1); fy = (gy + 1) * 0.5f * (H - 1); }
        else { fx = ((gx + 1) * W - 1) * 0.5f; fy = ((gy + 1) * H - 1) * 0.5f; }
        int64_t x0 = (int64_t)std::floor(fx), y0 = (int64_t)std::floor(fy), x1 = x0 + 1, y1 = y0 + 1;
        float dx = fx - x0, dy = fy - y0; const float *p = xp + (n * C + c) * H * W;
        auto at = [&](int64_t y, int64_t x) -> float { return (y >= 0 && y < H && x >= 0 && x < W) ? p[y * W + x] : 0.f; };
        op[((n * C + c) * oH + oh) * oW + ow] = at(y0, x0) * (1 - dy) * (1 - dx) + at(y0, x1) * (1 - dy) * dx
                                              + at(y1, x0) * dy * (1 - dx) + at(y1, x1) * dy * dx;
    }
    return ACLNN_SUCCESS;
}

// ---- GridSampler3D trilinear, zeros padding; in[N,C,D,H,W], grid[N,oD,oH,oW,3] -> out[N,C,oD,oH,oW]. ----
aclnnStatus run_gridsampler3d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *a = e->a, *g = e->b; aclTensor *o = e->out; bool align = (e->m != 0);
    int64_t N = a->viewDims[0], C = a->viewDims[1], D = a->viewDims[2], H = a->viewDims[3], W = a->viewDims[4];
    int64_t oD = o->viewDims[2], oH = o->viewDims[3], oW = o->viewDims[4];
    const float *xp = FP(a), *gp = FP(g); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c)
      for (int64_t od = 0; od < oD; ++od) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        const float *gg = gp + (((n * oD + od) * oH + oh) * oW + ow) * 3; float gx = gg[0], gy = gg[1], gz = gg[2]; float fx, fy, fz;
        if (align) { fx = (gx + 1) * 0.5f * (W - 1); fy = (gy + 1) * 0.5f * (H - 1); fz = (gz + 1) * 0.5f * (D - 1); }
        else { fx = ((gx + 1) * W - 1) * 0.5f; fy = ((gy + 1) * H - 1) * 0.5f; fz = ((gz + 1) * D - 1) * 0.5f; }
        int64_t x0 = (int64_t)std::floor(fx), y0 = (int64_t)std::floor(fy), z0 = (int64_t)std::floor(fz);
        float dx = fx - x0, dy = fy - y0, dz = fz - z0; const float *p = xp + (n * C + c) * D * H * W;
        auto at = [&](int64_t z, int64_t y, int64_t x) -> float { return (z >= 0 && z < D && y >= 0 && y < H && x >= 0 && x < W) ? p[(z * H + y) * W + x] : 0.f; };
        double v = 0;
        for (int dz_ = 0; dz_ < 2; dz_++) for (int dy_ = 0; dy_ < 2; dy_++) for (int dx_ = 0; dx_ < 2; dx_++) {
            float wz = dz_ ? dz : 1 - dz, wy = dy_ ? dy : 1 - dy, wx = dx_ ? dx : 1 - dx;
            v += (double)wz * wy * wx * at(z0 + dz_, y0 + dy_, x0 + dx_); }
        op[(((n * C + c) * oD + od) * oH + oh) * oW + ow] = (float)v;
    }
    return ACLNN_SUCCESS;
}

// ---- ConvDepthwise2d: grouped conv with groups == Cin; weight[Cin,1,R,S], bias[Cin] (nullable). ----
aclnnStatus run_depthwise2d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *X = e->a, *Wt = e->b, *B = e->c; aclTensor *Y = e->out;
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], W = X->viewDims[3];
    int64_t R = Wt->viewDims[2], Sd = Wt->viewDims[3];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    int64_t sh = e->stride[0], sw = e->stride[1], ph = e->pad[0], pw = e->pad[1], dh = e->dil[0], dw = e->dil[1];
    const float *xp = FP(X), *wp = FP(Wt), *bp = B ? FP(B) : nullptr; float *op = FP(Y);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c)
      for (int64_t ho = 0; ho < Ho; ++ho) for (int64_t wo = 0; wo < Wo; ++wo) {
        double acc = bp ? bp[c] : 0.0; const float *p = xp + (n * C + c) * H * W; const float *kw_ = wp + c * R * Sd;
        for (int64_t kr = 0; kr < R; ++kr) for (int64_t ks = 0; ks < Sd; ++ks) {
            int64_t hi = ho * sh - ph + kr * dh, wi = wo * sw - pw + ks * dw;
            if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
            acc += (double)p[hi * W + wi] * kw_[kr * Sd + ks]; }
        op[((n * C + c) * Ho + ho) * Wo + wo] = (float)acc;
    }
    return ACLNN_SUCCESS;
}

// ---- FusedCausalConv1d (= CausalConv1d): x[B,C,L], weight[C,K], bias[C] (nullable); left-padded K-1; act 1 -> SiLU. ----
aclnnStatus run_causalconv1d(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *X = e->a, *Wt = e->b, *B = e->c; aclTensor *Y = e->out; int act = (int)e->dim;
    int64_t Bn = X->viewDims[0], C = X->viewDims[1], L = X->viewDims[2], K = Wt->viewDims[1];
    const float *xp = FP(X), *wp = FP(Wt), *bp = B ? FP(B) : nullptr; float *op = FP(Y);
    for (int64_t b = 0; b < Bn; ++b) for (int64_t c = 0; c < C; ++c) {
        const float *p = xp + (b * C + c) * L; const float *w = wp + c * K;
        for (int64_t t = 0; t < L; ++t) { double acc = bp ? bp[c] : 0.0;
            for (int64_t k = 0; k < K; ++k) { int64_t ti = t - (K - 1) + k; if (ti >= 0) acc += (double)w[k] * p[ti]; }
            op[(b * C + c) * L + t] = act == 1 ? silu((float)acc) : (float)acc; }
    }
    return ACLNN_SUCCESS;
}

// ---- MultiScaleDeformableAttnFunction. value[N,S,nH,hd]; shapes[L,2](H,W); lstart[L];
//      samp[N,Lq,nH,L,P,2] (normalized xy in [0,1]); attn[N,Lq,nH,L,P] -> out[N,Lq,nH,hd]. ----
aclnnStatus run_msda(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *value = e->a, *shapes = e->b, *lstart = e->c;
    const aclTensor *samp = e->inputs[0], *attn = e->inputs[1]; aclTensor *out = e->out;
    int N = (int)value->viewDims[0], S = (int)value->viewDims[1], nH = (int)value->viewDims[2], hd = (int)value->viewDims[3];
    int Lq = (int)samp->viewDims[1], L = (int)samp->viewDims[3], P = (int)samp->viewDims[4];
    const float *vp = FP(value), *sp = FP(samp), *ap = FP(attn); float *op = FP(out);
    const int64_t *shp = IP(shapes), *lsp = IP(lstart);
    for (int n = 0; n < N; ++n) for (int q = 0; q < Lq; ++q) for (int h = 0; h < nH; ++h) for (int d = 0; d < hd; ++d) {
        float acc = 0;
        for (int l = 0; l < L; ++l) { int H = (int)shp[l * 2], W = (int)shp[l * 2 + 1]; int64_t base = lsp[l];
            for (int p = 0; p < P; ++p) {
                int64_t si = ((((int64_t)n * Lq + q) * nH + h) * L + l) * P + p;
                float x = sp[si * 2] * W - 0.5f, y = sp[si * 2 + 1] * H - 0.5f, aw = ap[si];
                int x0 = (int)std::floor(x), y0 = (int)std::floor(y), x1 = x0 + 1, y1 = y0 + 1; float dx = x - x0, dy = y - y0;
                auto val = [&](int yy, int xx) -> float { if (yy < 0 || yy >= H || xx < 0 || xx >= W) return 0.f;
                    int64_t sidx = base + (int64_t)yy * W + xx; return vp[(((int64_t)n * S + sidx) * nH + h) * hd + d]; };
                float v = (1 - dy) * (1 - dx) * val(y0, x0) + (1 - dy) * dx * val(y0, x1)
                        + dy * (1 - dx) * val(y1, x0) + dy * dx * val(y1, x1);
                acc += aw * v; }
        }
        op[(((int64_t)n * Lq + q) * nH + h) * hd + d] = acc;
    }
    return ACLNN_SUCCESS;
}

// ---- TransConvolutionWeight: pure copy of the weight tensor into out (same dtype/numel). ----
aclnnStatus run_transweight(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    size_t bytes = (size_t)e->a->numel() * dtype_size(e->a->dtype);
    std::memcpy((char *)e->out->data + e->out->offset * (int64_t)dtype_size(e->out->dtype),
                (const char *)e->a->data + e->a->offset * (int64_t)dtype_size(e->a->dtype), bytes);
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace pcg

using namespace pcg;

extern "C" {

// ============================ upsample family (nearest / linear) ============================
// interp: 0 nearest, 1 nearest-exact, 2 linear. e->m carries exact|align flag. nsp in reduceCount.
aclnnStatus aclnnUpsampleNearest1dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 0; e->m = 0; e->reduceCount = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearest1dV2, run_upsample)

aclnnStatus aclnnUpsampleNearest2dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 0; e->m = 0; e->reduceCount = 2; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearest2dV2, run_up2d)

aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 0; e->m = 0; e->reduceCount = 3; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearest3d, run_upsample)

aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 1; e->m = 1; e->reduceCount = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearestExact1d, run_upsample)

aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 1; e->m = 1; e->reduceCount = 2; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearestExact2d, run_upsample)

aclnnStatus aclnnUpsampleNearestExact3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 1; e->m = 1; e->reduceCount = 3; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleNearestExact3d, run_upsample)

aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 2; e->m = alignCorners ? 1 : 0; e->reduceCount = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleLinear1d, run_upsample)

aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 2; e->m = alignCorners ? 1 : 0; e->reduceCount = 3; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleTrilinear3d, run_upsample)

// ============================ 2D upsample (bilinear / bicubic + AA aliases) ============================
// interp: 0 nearest, 2 bilinear, 3 bicubic. AA variants fall back to non-AA (matches CUDA reference).
aclnnStatus aclnnUpsampleBilinear2DGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 2; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleBilinear2D, run_up2d)

aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 2; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleBilinear2dAA, run_up2d)

aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 3; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleBicubic2dAA, run_up2d)

aclnnStatus aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradInput; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnUpsampleBilinear2dBackwardV2, run_upbilinear2d_bwd)

// ============================ pooling ============================
aclnnStatus aclnnGlobalMaxPoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || self->viewDims.size() < 2) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out;
    int64_t NC = self->viewDims[0] * self->viewDims[1]; e->m = NC; e->n = self->numel() / NC; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnGlobalMaxPool, run_globalmax)

// MaxPool generic: NCHW max pool; kernel/stride/pad packed into axes (drops ceilMode/dilation).
aclnnStatus aclnnMaxPoolGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dilation; (void)ceilMode;
    if (!self || !kernel || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out;
    int64_t kh = IA(kernel, 0, 1), kw = IA(kernel, 1, kh);
    int64_t sh = IA(stride, 0, kh), sw = IA(stride, 1, sh);
    int64_t ph = IA(padding, 0, 0), pw = IA(padding, 1, ph);
    e->axes = {kh, kw, sh, sw, ph, pw}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMaxPool, run_maxpool2d)

// MaxPool2dWithMask: NCHW max pool + flat indices (== MaxPool2dWithIndices).
aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    (void)dilation; (void)ceilMode;
    if (!self || !kernel || !out || !mask || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->out2 = mask;
    int64_t kh = IA(kernel, 0, 1), kw = IA(kernel, 1, kh);
    int64_t sh = IA(stride, 0, kh), sw = IA(stride, 1, sh);
    int64_t ph = IA(padding, 0, 0), pw = IA(padding, 1, ph);
    e->axes = {kh, kw, sh, sw, ph, pw}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMaxPool2dWithMask, run_maxpool2d_mask)

// ============================ RoI ============================
aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !rois || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = rois; e->out = out; e->alpha = spatialScale; e->reduceCount = samplingRatio > 0 ? samplingRatio : 2;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnRoiAlignV2, run_roialignv2)

aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, const aclTensor *argmax,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !rois || !argmax || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = rois; e->c = argmax; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnRoiPoolingGradWithArgMax, run_roipoolgrad)

// ============================ grid sampler ============================
aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode,
        bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)interpolationMode; (void)paddingMode;     // bilinear / zeros (matches CUDA reference)
    if (!self || !grid || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = grid; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnGridSampler2D, run_gridsampler2d)

aclnnStatus aclnnGridSampler3DGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode,
        bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)interpolationMode; (void)paddingMode;     // trilinear / zeros
    if (!self || !grid || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = grid; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnGridSampler3D, run_gridsampler3d)

// ============================ convolution family ============================
aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclIntArray *kernelSize,
        const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)kernelSize;
    if (!self || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = weight; e->c = bias; e->out = out;
    e->stride[0] = IA(stride, 0, 1); e->stride[1] = IA(stride, 1, e->stride[0]);
    e->pad[0] = IA(padding, 0, 0);   e->pad[1] = IA(padding, 1, e->pad[0]);
    e->dil[0] = IA(dilation, 0, 1);  e->dil[1] = IA(dilation, 1, e->dil[0]);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnConvDepthwise2d, run_depthwise2d)

aclnnStatus aclnnFusedCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = bias; e->out = out; e->dim = activation; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnFusedCausalConv1d, run_causalconv1d)

aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes,
        const aclTensor *levelStartIndex, const aclTensor *samplingLocations, const aclTensor *attnWeights,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!value || !spatialShapes || !levelStartIndex || !samplingLocations || !attnWeights || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = value; e->b = spatialShapes; e->c = levelStartIndex; e->out = out;
    e->inputs.push_back(samplingLocations); e->inputs.push_back(attnWeights); *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnMultiScaleDeformableAttnFunction, run_msda)

aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} RUN(aclnnTransConvolutionWeight, run_transweight)

} // extern "C"
