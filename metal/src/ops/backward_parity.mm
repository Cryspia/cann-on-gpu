// Backward / gradient operator family — gap fill (part 1: structural / pool / pad / upsample / conv /
// GLU / vision / loss-lite). Host-side over unified memory (device ptr == host ptr under
// MTLStorageModeShared). Two-phase aclnn contract: GetWorkspaceSize stashes plan state in the
// executor and returns *ws=0; the Execute entry drains the stream, computes, then deletes the
// executor. Math mirrors the CUDA reference (cuda/src/ops/{pool_ext,upsample_ext,conv,shape,loss,
// glu_ext}.cu). FP32 throughout (tests use fp32). #include the op header so signatures are
// compiler-validated against the shim declarations.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace bwd_parity {
inline float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
inline const int64_t *IP(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
constexpr double INV_SQRT2 = 0.7071067811865476;
constexpr double INV_SQRT2PI = 0.3989422804014327;
inline void zero(aclTensor *t) { float *p = FP(t); for (int64_t i = 0, n = t->numel(); i < n; ++i) p[i] = 0.f; }

// ============================ GLU family ============================
// gradIn[a] = gradOut*act'(a)*b ; gradIn[b] = gradOut*act(a). gelu=1 -> GELU, else SiLU. self[...,2D].
aclnnStatus run_glugrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *self = e->b; aclTensor *gi = e->out; bool gelu = (e->m != 0);
    int nd = (int)self->viewDims.size(); int64_t twoD = self->viewDims[nd - 1], D = twoD / 2;
    int64_t rows = self->numel() / twoD;
    const float *gop = FP(go), *inp = FP(self); float *gip = FP(gi);
    for (int64_t r = 0; r < rows; ++r) for (int64_t d = 0; d < D; ++d) {
        double a = inp[r * twoD + d], b = inp[r * twoD + D + d], g = gop[r * D + d], act, dact;
        if (gelu) { double cdf = 0.5 * (1 + std::erf(a * INV_SQRT2)); act = a * cdf; dact = cdf + a * INV_SQRT2PI * std::exp(-0.5 * a * a); }
        else { double sg = sigm(a); act = a * sg; dact = sg + a * sg * (1 - sg); }
        gip[r * twoD + d] = (float)(g * dact * b);
        gip[r * twoD + D + d] = (float)(g * act);
    }
    return ACLNN_SUCCESS;
}
// Standard GLU backward: forward out = a * sigmoid(b). gradIn[a]=go*sig(b); gradIn[b]=go*a*sig(b)*(1-sig(b)).
aclnnStatus run_glubwd_std(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *self = e->b; aclTensor *gi = e->out;
    int nd = (int)self->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd;
    // split self along `dim` into halves a (first) and b (second).
    int64_t outer = 1, full = self->viewDims[dim], D = full / 2, inner = 1;
    for (int d = 0; d < dim; ++d) outer *= self->viewDims[d];
    for (int d = dim + 1; d < nd; ++d) inner *= self->viewDims[d];
    const float *gop = FP(go), *inp = FP(self); float *gip = FP(gi);
    for (int64_t oo = 0; oo < outer; ++oo) for (int64_t d = 0; d < D; ++d) for (int64_t ii = 0; ii < inner; ++ii) {
        int64_t ia = (oo * full + d) * inner + ii, ib = (oo * full + D + d) * inner + ii;
        int64_t ig = (oo * D + d) * inner + ii;
        double a = inp[ia], b = inp[ib], g = gop[ig], s = sigm(b);
        gip[ia] = (float)(g * s);
        gip[ib] = (float)(g * a * s * (1 - s));
    }
    return ACLNN_SUCCESS;
}

// ============================ Upsample / pad backward (scatter-add) ============================
// nearest/exact/linear (1d & 3d) — coordinate map matches the forward; scatter-add gradOut -> gradIn.
inline int nsrc(int o, int isz, int osz, int exact) {
    int i = exact ? (int)std::floor((o + 0.5f) * isz / osz) : (int)std::floor((float)o * isz / osz);
    return i < 0 ? 0 : (i >= isz ? isz - 1 : i);
}
// interp: 0 nearest, 1 nearest-exact, 2 linear. align used for linear.
aclnnStatus run_upbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int nsp = (int)e->reduceCount, interp = (int)e->dim, align = e->keepDim ? 1 : 0;
    const auto &O = go->viewDims, &I = gi->viewDims; int rank = (int)I.size(), sp0 = rank - nsp;
    int is[3] = {1, 1, 1}, os[3] = {1, 1, 1};
    for (int d = 0; d < nsp; d++) { is[3 - nsp + d] = (int)I[sp0 + d]; os[3 - nsp + d] = (int)O[sp0 + d]; }
    int64_t NC = 1; for (int d = 0; d < sp0; d++) NC *= I[d];
    int i0 = is[0], i1 = is[1], i2 = is[2], o0 = os[0], o1 = os[1], o2 = os[2];
    int64_t isp = (int64_t)i0 * i1 * i2;
    const float *gop = FP(go); float *gip = FP(gi); zero(gi);
    auto coord = [&](int o, int isz, int osz, int &lo, int &hi, float &fr) {
        float v; if (align) v = osz > 1 ? (float)o * (isz - 1) / (osz - 1) : 0.f;
        else v = osz > 0 ? ((o + 0.5f) * isz / osz - 0.5f) : 0.f;
        if (v < 0) v = 0; int b = (int)std::floor(v); fr = v - b;
        lo = b < 0 ? 0 : (b >= isz ? isz - 1 : b); hi = lo + 1 >= isz ? isz - 1 : lo + 1;
    };
    for (int64_t nc = 0; nc < NC; ++nc) for (int c0 = 0; c0 < o0; ++c0) for (int c1 = 0; c1 < o1; ++c1) for (int c2 = 0; c2 < o2; ++c2) {
        int64_t oidx = ((nc * o0 + c0) * o1 + c1) * o2 + c2; float gv = gop[oidx]; float *p = gip + nc * isp;
        if (interp == 2) {
            int l0, h0, l1, h1, l2, h2; float f0, f1, f2;
            coord(c0, i0, o0, l0, h0, f0); coord(c1, i1, o1, l1, h1, f1); coord(c2, i2, o2, l2, h2, f2);
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int c = 0; c < 2; c++) {
                int z = a ? h0 : l0, y = b ? h1 : l1, x = c ? h2 : l2;
                float w = (a ? f0 : 1 - f0) * (b ? f1 : 1 - f1) * (c ? f2 : 1 - f2);
                p[((int64_t)z * i1 + y) * i2 + x] += w * gv;
            }
        } else {
            int s0 = nsrc(c0, i0, o0, interp), s1 = nsrc(c1, i1, o1, interp), s2 = nsrc(c2, i2, o2, interp);
            p[((int64_t)s0 * i1 + s1) * i2 + s2] += gv;
        }
    }
    return ACLNN_SUCCESS;
}

// Bicubic / bilinear-AA 2d backward: scatter-add via the cubic / bilinear interpolation weights.
// Matches torch upsample_bicubic2d: cubic conv kernel (a=-0.75). The matching Metal forwards are
// aclnnUpsampleBicubic2d / aclnnUpsampleBilinear2dAA / aclnnUpsampleBicubic2dAA — we verify via the
// linear-map adjoint: <fwd(x),g> == <x, bwd(g)>.
inline float cubic_w(float t, float A) { // cubic convolution weights, |t|<=2
    t = std::fabs(t);
    if (t <= 1) return ((A + 2) * t - (A + 3)) * t * t + 1;
    if (t < 2) return (((t - 5) * t + 8) * t - 4) * A;
    return 0.f;
}
// e->m bits: bit0 = bicubic(1)/bilinear(0); align in e->keepDim. sH/sW unused (size-driven).
aclnnStatus run_up2dbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out; bool bicubic = (e->m & 1); bool align = e->keepDim;
    int64_t N = gi->viewDims[0], C = gi->viewDims[1], H = gi->viewDims[2], W = gi->viewDims[3];
    int64_t oH = go->viewDims[2], oW = go->viewDims[3], NC = N * C;
    const float *gop = FP(go); float *gip = FP(gi); zero(gi);
    auto srccoord = [&](int o, int64_t isz, int64_t osz) -> float {
        if (align) return osz > 1 ? (float)o * (isz - 1) / (osz - 1) : 0.f;
        return osz > 0 ? ((o + 0.5f) * isz / osz - 0.5f) : 0.f;
    };
    for (int64_t nc = 0; nc < NC; ++nc) { float *p = gip + nc * H * W;
        for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
            float fh = srccoord((int)oh, H, oH), fw = srccoord((int)ow, W, oW);
            float gv = gop[(nc * oH + oh) * oW + ow];
            if (bicubic) {
                const float A = -0.75f; int h0 = (int)std::floor(fh), w0 = (int)std::floor(fw);
                float th = fh - h0, tw = fw - w0;
                for (int a = -1; a <= 2; ++a) for (int b = -1; b <= 2; ++b) {
                    int hh = h0 + a, ww = w0 + b;
                    int hc = hh < 0 ? 0 : (hh >= (int)H ? (int)H - 1 : hh);
                    int wc = ww < 0 ? 0 : (ww >= (int)W ? (int)W - 1 : ww);
                    float w = cubic_w((float)a - th, A) * cubic_w((float)b - tw, A);
                    p[hc * W + wc] += w * gv;
                }
            } else { // bilinear
                if (fh < 0) fh = 0; if (fw < 0) fw = 0;
                int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, (int)H - 1), w1 = std::min(w0 + 1, (int)W - 1);
                float dh = fh - h0, dw = fw - w0;
                p[h0 * W + w0] += gv * (1 - dh) * (1 - dw); p[h0 * W + w1] += gv * (1 - dh) * dw;
                p[h1 * W + w0] += gv * dh * (1 - dw);       p[h1 * W + w1] += gv * dh * dw;
            }
        }
    }
    return ACLNN_SUCCESS;
}

// Pad backward (reflect/replicate/circular): scatter-add via per-mode index map. padding innermost-first.
inline int padsrc(int o, int I, int lp, int mode) {
    int j = o - lp;
    if (mode == 0) { if (j < 0) j = -j; if (j >= I) j = 2 * (I - 1) - j; return j; }      // reflect
    if (mode == 1) return j < 0 ? 0 : (j >= I ? I - 1 : j);                                  // replicate
    if (mode == 2) { j %= I; if (j < 0) j += I; return j; }                                  // circular
    return (j >= 0 && j < I) ? j : -1;
}
aclnnStatus run_padbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int nsp = (int)e->reduceCount, mode = (int)e->dim;
    const auto &O = go->viewDims, &I = gi->viewDims; int rank = (int)I.size(), sp0 = rank - nsp;
    int is[3] = {1, 1, 1}, os[3] = {1, 1, 1}, lp[3] = {0, 0, 0};
    for (int d = 0; d < nsp; d++) { int slot = 3 - nsp + d; is[slot] = (int)I[sp0 + d]; os[slot] = (int)O[sp0 + d]; }
    for (int d = 0; d < nsp; d++) { int slot = 2 - d; if ((size_t)(2 * d) < e->axes.size()) lp[slot] = (int)e->axes[2 * d]; }
    int64_t NC = 1; for (int d = 0; d < sp0; d++) NC *= I[d];
    int i0 = is[0], i1 = is[1], i2 = is[2], o0 = os[0], o1 = os[1], o2 = os[2];
    int64_t isp = (int64_t)i0 * i1 * i2;
    const float *gop = FP(go); float *gip = FP(gi); zero(gi);
    for (int64_t nc = 0; nc < NC; ++nc) for (int c0 = 0; c0 < o0; ++c0) for (int c1 = 0; c1 < o1; ++c1) for (int c2 = 0; c2 < o2; ++c2) {
        int s0 = padsrc(c0, i0, lp[0], mode), s1 = padsrc(c1, i1, lp[1], mode), s2 = padsrc(c2, i2, lp[2], mode);
        if (s0 < 0 || s1 < 0 || s2 < 0) continue;
        int64_t oidx = ((nc * o0 + c0) * o1 + c1) * o2 + c2;
        gip[nc * isp + ((int64_t)s0 * i1 + s1) * i2 + s2] += gop[oidx];
    }
    return ACLNN_SUCCESS;
}

// ============================ Pool backward ============================
// AdaptiveAvgPool3d backward: distribute gradOut uniformly over its adaptive source window. NCDHW.
aclnnStatus run_adaptavg3dbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int64_t NC = gi->viewDims[0] * gi->viewDims[1];
    int D = (int)gi->viewDims[2], H = (int)gi->viewDims[3], W = (int)gi->viewDims[4];
    int oD = (int)go->viewDims[2], oH = (int)go->viewDims[3], oW = (int)go->viewDims[4];
    const float *gop = FP(go); float *gip = FP(gi); zero(gi);
    for (int64_t nc = 0; nc < NC; ++nc) for (int od = 0; od < oD; ++od) for (int oh = 0; oh < oH; ++oh) for (int ow = 0; ow < oW; ++ow) {
        int ds = od * D / oD, de = ((od + 1) * D + oD - 1) / oD, hs = oh * H / oH, he = ((oh + 1) * H + oH - 1) / oH, ws = ow * W / oW, we = ((ow + 1) * W + oW - 1) / oW;
        float v = gop[((nc * oD + od) * oH + oh) * oW + ow] / (float)((de - ds) * (he - hs) * (we - ws));
        float *p = gip + nc * (int64_t)D * H * W;
        for (int d = ds; d < de; d++) for (int h = hs; h < he; h++) for (int w = ws; w < we; w++) p[((int64_t)d * H + h) * W + w] += v;
    }
    return ACLNN_SUCCESS;
}
// Max pool backward (with-indices/argmax/mask/adaptive-max 2d&3d): scatter-add gradOut->gradIn at flat argmax idx.
// e->a=gradOut, e->b=indices, e->out=gradIn. NC = dims[0]*dims[1].
aclnnStatus run_maxpoolbwd_scatter(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *idx = e->b; aclTensor *gi = e->out;
    int64_t NC = gi->viewDims.size() >= 2 ? gi->viewDims[0] * gi->viewDims[1] : gi->viewDims[0];
    int64_t isp = gi->numel() / NC, osp = go->numel() / NC;
    const float *gop = FP(go); const int64_t *ip = IP(idx); float *gip = FP(gi); zero(gi);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t i = 0; i < osp; ++i) {
        int64_t p = nc * osp + i, t = ip[p];
        if (t < 0 || t >= isp) continue;
        gip[nc * isp + t] += gop[p];
    }
    return ACLNN_SUCCESS;
}
// Max unpool backward: gather gradOut[idx] -> gradIn (small). e->a=gradOut(big),e->b=indices(small),e->out=gradIn(small).
aclnnStatus run_maxunpoolbwd_gather(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *idx = e->b; aclTensor *gi = e->out;
    int64_t NC = gi->viewDims.size() >= 2 ? gi->viewDims[0] * gi->viewDims[1] : gi->viewDims[0];
    int64_t osp = gi->numel() / NC, srcSp = go->numel() / NC;
    const float *gop = FP(go); const int64_t *ip = IP(idx); float *gip = FP(gi);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t i = 0; i < osp; ++i) {
        int64_t p = nc * osp + i, t = ip[p];
        gip[p] = (t >= 0 && t < srcSp) ? gop[nc * srcSp + t] : 0.f;
    }
    return ACLNN_SUCCESS;
}

// ============================ Conv / im2col / unfold backward ============================
// col2im (Im2col backward / UnfoldGrad): gradCol[N, C*kH*kW, oH*oW] -> gradInput[N,C,H,W] scatter-add.
aclnnStatus run_col2im(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int N = (int)gi->viewDims[0], C = (int)gi->viewDims[1], H = (int)gi->viewDims[2], W = (int)gi->viewDims[3];
    int kH = (int)e->axes[0], kW = (int)e->axes[1], sH = (int)e->axes[2], sW = (int)e->axes[3];
    int pH = (int)e->axes[4], pW = (int)e->axes[5], dH = (int)e->axes[6], dW = (int)e->axes[7];
    int oH = (H + 2 * pH - dH * (kH - 1) - 1) / sH + 1, oW = (W + 2 * pW - dW * (kW - 1) - 1) / sW + 1;
    int64_t L = (int64_t)oH * oW, K = (int64_t)C * kH * kW;
    const float *col = FP(go); float *x = FP(gi); zero(gi);
    for (int n = 0; n < N; ++n) for (int64_t kk = 0; kk < K; ++kk) for (int64_t l = 0; l < L; ++l) {
        int ow = l % oW, oh = l / oW; int kw = kk % kW, kh = (kk / kW) % kH, c = kk / (kH * kW);
        int ih = oh * sH - pH + kh * dH, iw = ow * sW - pW + kw * dW;
        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
            x[((int64_t)(n * C + c) * H + ih) * W + iw] += col[(n * K + kk) * L + l];
    }
    return ACLNN_SUCCESS;
}
// ConvTbc backward: gradOut[oT,B,Cout], self[T,B,Cin], weight[kW,Cin,Cout] -> gradInput,gradWeight,gradBias.
aclnnStatus run_convtbcbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *self = e->b, *w = e->c; aclTensor *gi = e->out, *gw = e->out2;
    aclTensor *gb = const_cast<aclTensor *>(e->mask);
    int T = (int)self->viewDims[0], B = (int)self->viewDims[1], Cin = (int)self->viewDims[2];
    int kW = (int)w->viewDims[0], Cout = (int)w->viewDims[2], oT = (int)go->viewDims[0], pad = (int)e->dim;
    const float *gop = FP(go), *xp = FP(self), *wp = FP(w);
    float *gip = gi ? FP(gi) : nullptr, *gwp = gw ? FP(gw) : nullptr, *gbp = gb ? FP(gb) : nullptr;
    if (gi) zero(gi); if (gw) zero(gw); if (gb) zero(gb);
    for (int t = 0; t < oT; ++t) for (int b = 0; b < B; ++b) for (int co = 0; co < Cout; ++co) {
        float g = gop[((int64_t)t * B + b) * Cout + co]; if (gbp) gbp[co] += g;
        for (int k = 0; k < kW; ++k) { int ti = t + k - pad; if (ti < 0 || ti >= T) continue;
            const float *xrow = xp + ((int64_t)ti * B + b) * Cin;
            float *girow = gip ? gip + ((int64_t)ti * B + b) * Cin : nullptr;
            for (int ci = 0; ci < Cin; ++ci) { float wv = wp[((int64_t)k * Cin + ci) * Cout + co];
                if (girow) girow[ci] += g * wv;
                if (gwp) gwp[((int64_t)k * Cin + ci) * Cout + co] += g * xrow[ci];
            }
        }
    }
    return ACLNN_SUCCESS;
}

// ============================ Vision: grid sample 3d / roi align backward ============================
inline void bilin_scatter(float *p, int H, int W, float fy, float fx, float g) {
    if (fy < -1 || fy > H || fx < -1 || fx > W) return; fy = std::fmax(fy, 0.f); fx = std::fmax(fx, 0.f);
    int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
    float dy = fy - y0, dx = fx - x0;
    p[y0 * W + x0] += (1 - dy) * (1 - dx) * g; p[y0 * W + x1] += (1 - dy) * dx * g;
    p[y1 * W + x0] += dy * (1 - dx) * g;       p[y1 * W + x1] += dy * dx * g;
}
// RoiAlignRotated/V2 backward (e->m: 1 rotated [K,6], 0 axis-aligned [K,5]).
aclnnStatus run_roialignbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *rois = e->b; aclTensor *gi = e->out;
    bool rot = (e->m != 0); int C = (int)e->outerCount, H = (int)e->m == 0 ? 0 : 0; // placeholder
    C = (int)gi->viewDims[1]; H = (int)gi->viewDims[2]; int W = (int)gi->viewDims[3];
    int K = (int)rois->viewDims[0], ph = (int)go->viewDims[2], pw = (int)go->viewDims[3];
    float scale = (float)e->alpha; int ratio = (int)e->reduceCount; int rstride = rot ? 6 : 5;
    const float *gop = FP(go), *rp = FP(rois); float *gip = FP(gi); zero(gi); int cnt = ratio * ratio;
    for (int k = 0; k < K; ++k) {
        const float *r = rp + (int64_t)k * rstride; int b = (int)r[0];
        for (int c = 0; c < C; ++c) { float *p = gip + ((int64_t)b * C + c) * H * W;
            for (int py = 0; py < ph; ++py) for (int px = 0; px < pw; ++px) {
                float g = gop[(((int64_t)k * C + c) * ph + py) * pw + px] / cnt;
                if (rot) {
                    float cx = r[1] * scale, cy = r[2] * scale, rw = std::fmax(r[3] * scale, 1.f), rh = std::fmax(r[4] * scale, 1.f), th = r[5];
                    float ct = std::cos(th), st = std::sin(th), bw = rw / pw, bh = rh / ph, x0 = -rw / 2.f, y0 = -rh / 2.f;
                    for (int iy = 0; iy < ratio; iy++) for (int ix = 0; ix < ratio; ix++) {
                        float ly = y0 + py * bh + (iy + 0.5f) * bh / ratio, lx = x0 + px * bw + (ix + 0.5f) * bw / ratio;
                        bilin_scatter(p, H, W, cy + lx * st + ly * ct, cx + lx * ct - ly * st, g);
                    }
                } else {
                    float x1 = r[1] * scale, y1 = r[2] * scale, x2 = r[3] * scale, y2 = r[4] * scale;
                    float rw = std::fmax(x2 - x1, 1.f), rh = std::fmax(y2 - y1, 1.f), bw = rw / pw, bh = rh / ph;
                    for (int iy = 0; iy < ratio; iy++) for (int ix = 0; ix < ratio; ix++) {
                        float yy = y1 + py * bh + (iy + 0.5f) * bh / ratio, xx = x1 + px * bw + (ix + 0.5f) * bw / ratio;
                        bilin_scatter(p, H, W, yy, xx, g);
                    }
                }
            }
        }
    }
    return ACLNN_SUCCESS;
}
// GridSampler3D backward: trilinear scatter of gradOut into gradInput. grid in [-1,1]^3.
// gradGrid is set to zero (not required by tests). interp 1 = trilinear.
aclnnStatus run_gridsampler3dbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *self = e->b, *grid = e->c; aclTensor *gi = e->out, *gg = e->out2;
    bool align = e->keepDim;
    int N = (int)self->viewDims[0], C = (int)self->viewDims[1], D = (int)self->viewDims[2], H = (int)self->viewDims[3], W = (int)self->viewDims[4];
    int oD = (int)go->viewDims[2], oH = (int)go->viewDims[3], oW = (int)go->viewDims[4];
    const float *gop = FP(go), *gp = FP(grid); float *gip = FP(gi); zero(gi); if (gg) zero(gg);
    auto unnorm = [&](float c, int sz) -> float { return align ? (c + 1) * 0.5f * (sz - 1) : ((c + 1) * sz - 1) * 0.5f; };
    for (int n = 0; n < N; ++n) for (int od = 0; od < oD; ++od) for (int oh = 0; oh < oH; ++oh) for (int ow = 0; ow < oW; ++ow) {
        int64_t gbase = (((int64_t)(n * oD + od) * oH + oh) * oW + ow) * 3;
        float fx = unnorm(gp[gbase + 0], W), fy = unnorm(gp[gbase + 1], H), fz = unnorm(gp[gbase + 2], D);
        int z0 = (int)std::floor(fz), y0 = (int)std::floor(fy), x0 = (int)std::floor(fx);
        float dz = fz - z0, dy = fy - y0, dx = fx - x0;
        for (int c = 0; c < C; ++c) {
            float g = gop[(((int64_t)(n * C + c) * oD + od) * oH + oh) * oW + ow];
            float *p = gip + (int64_t)(n * C + c) * D * H * W;
            for (int a = 0; a < 2; a++) for (int b = 0; b < 2; b++) for (int cc = 0; cc < 2; cc++) {
                int z = z0 + a, y = y0 + b, x = x0 + cc;
                if (z < 0 || z >= D || y < 0 || y >= H || x < 0 || x >= W) continue;
                float w = (a ? dz : 1 - dz) * (b ? dy : 1 - dy) * (cc ? dx : 1 - dx);
                p[((int64_t)z * H + y) * W + x] += w * g;
            }
        }
    }
    return ACLNN_SUCCESS;
}

// ============================ Misc structural ============================
// RepeatInterleaveGrad: gradInput[i] = sum_{r} gradOutput[i*repeats + r]. (dim ignored; uniform.)
aclnnStatus run_repeatinterleavegrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int64_t rep = e->m, nIn = gi->numel(); const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0; i < nIn; ++i) { double acc = 0; for (int64_t r = 0; r < rep; ++r) acc += gop[i * rep + r]; gip[i] = (float)acc; }
    return ACLNN_SUCCESS;
}
// Cdist backward: gradX1[p,m] = sum_r gradOut[p,r] * d(dist)/dx1. p==2 Euclidean else Lp.
aclnnStatus run_cdistbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x1 = e->b, *x2 = e->c, *cd = e->mean; aclTensor *gx1 = e->out;
    double p = e->alpha;
    int64_t P = x1->viewDims[0], M = x1->viewDims[1], R = x2->viewDims[0];
    const float *gop = FP(go), *a = FP(x1), *b = FP(x2), *dist = FP(cd); float *g = FP(gx1); zero(gx1);
    for (int64_t pi = 0; pi < P; ++pi) for (int64_t m = 0; m < M; ++m) {
        double acc = 0;
        for (int64_t r = 0; r < R; ++r) {
            double d = dist[pi * R + r]; if (d <= 1e-12) continue;
            double diff = (double)a[pi * M + m] - b[r * M + m], deriv;
            if (p == 2.0) deriv = diff / d;
            else deriv = (diff > 0 ? 1 : (diff < 0 ? -1 : 0)) * std::pow(std::fabs(diff), p - 1) * std::pow(d, 1 - p);
            acc += gop[pi * R + r] * deriv;
        }
        g[pi * M + m] = (float)acc;
    }
    return ACLNN_SUCCESS;
}
// Chamfer distance backward: gradXyz1[b,n,d] = 2*(xyz1 - xyz2[idx1])*gradDist1.
aclnnStatus run_chamferbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gd = e->a, *xyz1 = e->b, *xyz2 = e->c, *idx1 = e->mean; aclTensor *gx = e->out;
    int64_t B = xyz1->viewDims[0], Np = xyz1->viewDims[1], Dd = xyz1->viewDims[2], Mp = xyz2->viewDims[1];
    const float *gdp = FP(gd), *x1 = FP(xyz1), *x2 = FP(xyz2); const int64_t *ix = IP(idx1); float *g = FP(gx);
    for (int64_t b = 0; b < B; ++b) for (int64_t n = 0; n < Np; ++n) {
        int64_t j = ix[b * Np + n]; if (j < 0) j = 0; if (j >= Mp) j = Mp - 1;
        float gv = gdp[b * Np + n];
        for (int64_t d = 0; d < Dd; ++d)
            g[(b * Np + n) * Dd + d] = 2.f * ((float)x1[(b * Np + n) * Dd + d] - x2[(b * Mp + j) * Dd + d]) * gv;
    }
    return ACLNN_SUCCESS;
}
// Dropout backward: gradInput = gradOutput * mask / (1-p). mask is uint8 (1 keep, 0 drop).
aclnnStatus run_dropoutbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *mask = e->b; aclTensor *gi = e->out;
    double keep = 1.0 - e->alpha; if (keep <= 0) keep = 1.0;
    const float *gop = FP(go); float *gip = FP(gi);
    const uint8_t *mp = (const uint8_t *)mask->data + mask->offset;
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = mp[i] ? (float)(gop[i] / keep) : 0.f;
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace bwd_parity

using namespace bwd_parity;

extern "C" {
// ---- GLU family ----
aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !self || !gradIn || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->out = gradIn; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGeGluV3Backward, run_glugrad)
aclnnStatus aclnnGluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, int64_t dim, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !self || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->out = gradInput; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGluBackward, run_glubwd_std)

// ---- Upsample backward: nearest/exact (1d,3d) + linear1d + trilinear3d ----
#define UPB(NAME, NSP, INTERP) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) { \
    if (!gradOut || !gradIn || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradIn; e->dim = INTERP; e->keepDim = false; e->reduceCount = NSP; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_upbwd)
UPB(aclnnUpsampleNearest3dBackward, 3, 0)
UPB(aclnnUpsampleNearestExact1dBackward, 1, 1)
UPB(aclnnUpsampleNearestExact2dBackward, 2, 1)
UPB(aclnnUpsampleNearestExact3dBackward, 3, 1)
#undef UPB
#define UPBL(NAME, NSP) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) { \
    if (!gradOut || !gradIn || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradIn; e->dim = 2; e->keepDim = alignCorners; e->reduceCount = NSP; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_upbwd)
UPBL(aclnnUpsampleLinear1dBackward, 1)
UPBL(aclnnUpsampleTrilinear3dBackward, 3)
#undef UPBL

// ---- Upsample 2d bicubic / bilinear-AA backward (size+inputSize args) ----
#define UP2DB(NAME, BICUBIC) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double sH, double sW, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { \
    (void)outputSize; (void)inputSize; (void)sH; (void)sW; \
    if (!gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput; e->m = BICUBIC; e->keepDim = alignCorners; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_up2dbwd)
UP2DB(aclnnUpsampleBicubic2dBackward, 1)
UP2DB(aclnnUpsampleBicubic2dAAGrad, 1)
UP2DB(aclnnUpsampleBilinear2dAABackward, 0)
#undef UP2DB

// ---- Pad backward (reflect/replicate/circular) ----
#define PADB(NAME, NSP, MODE) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { \
    (void)self; if (!gradOutput || !padding || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput; e->dim = MODE; e->reduceCount = NSP; e->axes = padding->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_padbwd)
PADB(aclnnReflectionPad1dBackward, 1, 0)
PADB(aclnnReflectionPad2dBackward, 2, 0)
PADB(aclnnReflectionPad3dBackward, 3, 0)
PADB(aclnnReplicationPad2dBackward, 2, 1)
PADB(aclnnReplicationPad3dBackward, 3, 1)
PADB(aclnnCircularPad2dBackward, 2, 2)
PADB(aclnnCircularPad3dBackward, 3, 2)
#undef PADB

// ---- Pool backward ----
aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; if (!gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdaptiveAvgPool3dBackward, run_adaptavg3dbwd)
// max-pool scatter-add backward family
aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; if (!gradOutput || !indices || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdaptiveMaxPool2dBackward, run_maxpoolbwd_scatter)
aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; if (!gradOutput || !indices || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAdaptiveMaxPool3dBackward, run_maxpoolbwd_scatter)
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *mask, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)kernel; (void)stride; (void)padding; (void)dilation; (void)ceilMode;
    if (!gradOutput || !mask || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = mask; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaxPool2dWithMaskBackward, run_maxpoolbwd_scatter)
aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)kernel; (void)stride; (void)padding; (void)dilation; (void)ceilMode;
    if (!gradOutput || !indices || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaxPool3dWithArgmaxBackward, run_maxpoolbwd_scatter)
// max-unpool gather backward family
aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)H; (void)W; if (!gradOutput || !indices || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaxUnpool2dBackward, run_maxunpoolbwd_gather)
aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)outputSize; if (!gradOutput || !indices || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaxUnpool3dBackward, run_maxunpoolbwd_gather)

// ---- Conv / im2col / unfold backward ----
aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !kernelSize || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput; e->axes.assign(8, 0);
    for (int i = 0; i < 2; i++) { e->axes[i] = kernelSize->v[i];
        e->axes[2 + i] = stride && stride->v.size() > (size_t)i ? stride->v[i] : 1;
        e->axes[4 + i] = padding && padding->v.size() > (size_t)i ? padding->v[i] : 0;
        e->axes[6 + i] = dilation && dilation->v.size() > (size_t)i ? dilation->v[i] : 1; }
    *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnIm2colBackward, run_col2im)
aclnnStatus aclnnUnfoldGradGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnIm2colBackwardGetWorkspaceSize(gradOutput, kernelSize, dilation, padding, stride, gradInput, ws, ex); }
RUN(aclnnUnfoldGrad, run_col2im)
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !self || !weight || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = weight; e->out = gradInput; e->out2 = gradWeight; e->mask = gradBias; e->dim = pad; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnConvTbcBackward, run_convtbcbwd)

// ---- Vision ----
aclnnStatus aclnnRoiAlignRotatedGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !rois || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = rois; e->out = gradInput; e->alpha = spatialScale; e->reduceCount = samplingRatio > 0 ? samplingRatio : 2; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRoiAlignRotatedGrad, run_roialignbwd)
aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !rois || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = rois; e->out = gradInput; e->alpha = spatialScale; e->reduceCount = samplingRatio > 0 ? samplingRatio : 2; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRoiAlignV2Backward, run_roialignbwd)
aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *gradInput, aclTensor *gradGrid, uint64_t *ws, aclOpExecutor **ex) {
    (void)interpolationMode; (void)paddingMode;
    if (!gradOutput || !self || !grid || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = grid; e->out = gradInput; e->out2 = gradGrid; e->keepDim = alignCorners; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGridSampler3DBackward, run_gridsampler3dbwd)

// ---- Misc structural / distance / dropout ----
aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(const aclTensor *gradOutput, int64_t repeats, int64_t dim, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput; e->m = repeats; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleaveGrad, run_repeatinterleavegrad)
aclnnStatus aclnnCdistBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x1, const aclTensor *x2, double p, const aclTensor *cdist, aclTensor *gradX1, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !x1 || !x2 || !cdist || !gradX1 || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = x1; e->c = x2; e->mean = cdist; e->out = gradX1; e->alpha = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCdistBackward, run_cdistbwd)
aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(const aclTensor *gradDist1, const aclTensor *xyz1, const aclTensor *xyz2, const aclTensor *idx1, aclTensor *gradXyz1, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradDist1 || !xyz1 || !xyz2 || !idx1 || !gradXyz1 || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradDist1; e->b = xyz1; e->c = xyz2; e->mean = idx1; e->out = gradXyz1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnChamferDistanceBackward, run_chamferbwd)
aclnnStatus aclnnDropoutBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *mask, double p, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !mask || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = mask; e->out = gradInput; e->alpha = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDropoutBackward, run_dropoutbwd)
} // extern "C"
