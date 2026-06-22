// Pooling / interpolate / vision FORWARD family (test_pool spec): upsample nearest/bilinear,
// adaptive max2d / avg3d, max/avg pool1d, grid_sample2d, affine_grid, NMS, max_pool3d_with_argmax,
// avg_pool3d_backward, roi_align_rotated, roi_pooling_with_argmax. Host-side over unified memory
// (device ptr == host ptr under MTLStorageModeShared). Standard two-phase aclnn contract:
// GetWorkspaceSize news the executor + stashes plan state + sets *ws=0; Execute drains the stream,
// computes, then deletes the executor. Math/semantics mirror tests/test_pool.cpp and the CUDA
// reference (cuda/src/ops/pool_ext.cu). The *Backward upsample/adaptive-avg-pool variants and
// MaxPool2dWithIndices live in backward.mm and are NOT redefined here.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *IP(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// bilinear sample of plane p[H,W] at fractional (fy,fx); zero padding outside [-1,H]/[-1,W].
inline float bilin(const float *p, int H, int W, float fy, float fx) {
    if (fy < -1 || fy > H || fx < -1 || fx > W) return 0.f;
    fy = std::max(fy, 0.f); fx = std::max(fx, 0.f);
    int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1);
    float dy = fy - y0, dx = fx - x0;
    return (1 - dy) * (1 - dx) * p[y0 * W + x0] + (1 - dy) * dx * p[y0 * W + x1]
         + dy * (1 - dx) * p[y1 * W + x0] + dy * dx * p[y1 * W + x1];
}

// ---- UpsampleNearest2d ([N,C,H,W] -> [N,C,oH,oW]); src ih=min(oh*H/oH,H-1), iw=min(ow*W/oW,W-1) ----
aclnnStatus run_upnearest(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out;
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        int64_t ih = std::min(oh * H / oH, H - 1), iw = std::min(ow * W / oW, W - 1);
        op[(nc * oH + oh) * oW + ow] = xp[(nc * H + ih) * W + iw];
    }
    return ACLNN_SUCCESS;
}
// ---- UpsampleBilinear2d; align_corners selects coord mapping; e->m carries align flag ----
aclnnStatus run_upbilinear(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; bool align = (e->m != 0);
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        float fh, fw;
        if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
        else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
        fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw;
        int h0 = (int)fh, w0 = (int)fw, h1 = std::min(h0 + 1, (int)H - 1), w1 = std::min(w0 + 1, (int)W - 1);
        float dh = fh - h0, dw = fw - w0; const float *p = &xp[nc * H * W];
        op[(nc * oH + oh) * oW + ow] = p[h0 * W + w0] * (1 - dh) * (1 - dw) + p[h0 * W + w1] * (1 - dh) * dw
                                     + p[h1 * W + w0] * dh * (1 - dw) + p[h1 * W + w1] * dh * dw;
    }
    return ACLNN_SUCCESS;
}
// ---- AdaptiveMaxPool2d ([N,C,H,W] -> [N,C,oH,oW]); optional indices in out2 (flat ih*W+iw) ----
aclnnStatus run_adaptmax2d(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out, *id = e->out2;
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3], NC = N * C;
    const float *xp = FP(a); float *op = FP(o); int64_t *ip = id ? IP(id) : nullptr;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        int64_t hs = oh * H / oH, he = (oh + 1) * H / oH + ((oh + 1) * H % oH ? 1 : 0);
        int64_t ws = ow * W / oW, we = (ow + 1) * W / oW + ((ow + 1) * W % oW ? 1 : 0);
        float best = -INFINITY; int64_t bi = 0;
        for (int64_t h = hs; h < he; ++h) for (int64_t w = ws; w < we; ++w) {
            float v = xp[(nc * H + h) * W + w]; if (v > best) { best = v; bi = h * W + w; } }
        op[(nc * oH + oh) * oW + ow] = best; if (ip) ip[(nc * oH + oh) * oW + ow] = bi;
    }
    return ACLNN_SUCCESS;
}
// ---- AdaptiveAvgPool3d ([N,C,D,H,W] -> [N,C,oD,oH,oW]) ----
aclnnStatus run_adaptavg3d(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out;
    int64_t N = a->viewDims[0], C = a->viewDims[1], D = a->viewDims[2], H = a->viewDims[3], W = a->viewDims[4];
    int64_t oD = o->viewDims[2], oH = o->viewDims[3], oW = o->viewDims[4], NC = N * C;
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t od = 0; od < oD; ++od) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        int64_t ds = od * D / oD, de = (od + 1) * D / oD + ((od + 1) * D % oD ? 1 : 0);
        int64_t hs = oh * H / oH, he = (oh + 1) * H / oH + ((oh + 1) * H % oH ? 1 : 0);
        int64_t ws = ow * W / oW, we = (ow + 1) * W / oW + ((ow + 1) * W % oW ? 1 : 0);
        double sum = 0; int64_t cnt = 0;
        for (int64_t d = ds; d < de; ++d) for (int64_t h = hs; h < he; ++h) for (int64_t w = ws; w < we; ++w) { sum += xp[((nc * D + d) * H + h) * W + w]; cnt++; }
        op[((nc * oD + od) * oH + oh) * oW + ow] = (float)(sum / cnt);
    }
    return ACLNN_SUCCESS;
}
// ---- MaxPool1d / AvgPool1d ([N,C,L] -> [N,C,oL]); kernel/stride/pad in axes[0]/stride[0]/pad[0]; e->m: 1=max ----
aclnnStatus run_pool1d(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; bool mx = (e->m != 0);
    int64_t N = a->viewDims[0], C = a->viewDims[1], L = a->viewDims[2], oL = o->viewDims[2], NC = N * C;
    int k = (int)e->axes[0], st = (int)e->stride[0], pad = (int)e->pad[0];
    const float *xp = FP(a); float *op = FP(o);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t ol = 0; ol < oL; ++ol) {
        double acc = mx ? -INFINITY : 0; int cnt = 0;
        for (int j = 0; j < k; ++j) { int64_t l = ol * st - pad + j; if (l < 0 || l >= L) continue;
            double v = xp[nc * L + l]; if (mx) acc = std::max(acc, v); else acc += v; cnt++; }
        op[nc * oL + ol] = (float)(mx ? acc : acc / (cnt ? cnt : 1));
    }
    return ACLNN_SUCCESS;
}
// ---- AffineGrid (theta[N,2,3] -> grid[N,H,W,2]); base coords normalized to [-1,1]; e->m=H, e->n=W, align in keepDim ----
aclnnStatus run_affinegrid(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *t = e->a; aclTensor *o = e->out; bool align = e->keepDim;
    int64_t N = t->viewDims[0], H = e->m, W = e->n;
    const float *tp = FP(t); float *gp = FP(o);
    for (int64_t n = 0; n < N; ++n) { const float *th = tp + n * 6;
        for (int64_t h = 0; h < H; ++h) for (int64_t w = 0; w < W; ++w) {
            float xn = align ? (W > 1 ? (float)w / (W - 1) * 2 - 1 : 0.f) : ((w + 0.5f) / W) * 2 - 1;
            float yn = align ? (H > 1 ? (float)h / (H - 1) * 2 - 1 : 0.f) : ((h + 0.5f) / H) * 2 - 1;
            int64_t base = ((n * H + h) * W + w) * 2;
            gp[base + 0] = th[0] * xn + th[1] * yn + th[2];
            gp[base + 1] = th[3] * xn + th[4] * yn + th[5];
        }
    }
    return ACLNN_SUCCESS;
}
// ---- GridSample2d (bilinear, zeros padding); in[N,C,H,W], grid[N,oH,oW,2] -> out[N,C,oH,oW]; align in e->m ----
aclnnStatus run_gridsample2d(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a, *g = e->b; aclTensor *o = e->out; bool align = (e->m != 0);
    int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t oH = o->viewDims[2], oW = o->viewDims[3];
    const float *xp = FP(a), *gp = FP(g); float *op = FP(o);
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        const float *gg = gp + ((n * oH + oh) * oW + ow) * 2; float gx = gg[0], gy = gg[1];
        float fx, fy;
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
// ---- NMS greedy; boxes[M,4]=x1y1x2y2, scores[M] -> keepOut[M], countOut[1]; iou in e->alpha ----
aclnnStatus run_nms(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *b = e->a, *sc = e->b; aclTensor *keep = e->out, *cnt = e->out2;
    int64_t M = b->viewDims[0]; double iou = e->alpha;
    const float *bx = FP(b), *scp = FP(sc); int64_t *kp = IP(keep); int64_t *cp = IP(cnt);
    // order = indices sorted by score descending (stable selection).
    std::vector<int64_t> order(M); for (int64_t i = 0; i < M; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int64_t i, int64_t j) { return scp[i] > scp[j]; });
    std::vector<char> removed(M, 0); int64_t kc = 0;
    for (int64_t a = 0; a < M; ++a) { int64_t i = order[a]; if (removed[i]) continue; kp[kc++] = i;
        float ax1 = bx[i * 4], ay1 = bx[i * 4 + 1], ax2 = bx[i * 4 + 2], ay2 = bx[i * 4 + 3];
        float aarea = (ax2 - ax1) * (ay2 - ay1);
        for (int64_t bb = a + 1; bb < M; ++bb) { int64_t j = order[bb]; if (removed[j]) continue;
            float bx1 = bx[j * 4], by1 = bx[j * 4 + 1], bx2 = bx[j * 4 + 2], by2 = bx[j * 4 + 3];
            float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1), ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
            float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1), inter = iw * ih;
            float barea = (bx2 - bx1) * (by2 - by1), u = aarea + barea - inter;
            if (u > 0 && inter / u > iou) removed[j] = 1; }
    }
    cp[0] = kc;
    return ACLNN_SUCCESS;
}
// ---- MaxPool3dWithArgmax ([N,C,D,H,W] -> [N,C,oD,oH,oW] + int64 indices=flat (dd*H+hh)*W+ww) ----
aclnnStatus run_maxpool3d(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out, *id = e->out2;
    int64_t N = a->viewDims[0], C = a->viewDims[1], D = a->viewDims[2], H = a->viewDims[3], W = a->viewDims[4];
    int64_t oD = o->viewDims[2], oH = o->viewDims[3], oW = o->viewDims[4], NC = N * C;
    int kd = (int)e->axes[0], kh = (int)e->axes[1], kw = (int)e->axes[2];
    int sd = (int)e->dscalars[0], sh = (int)e->dscalars[1], sw = (int)e->dscalars[2];
    int pd = (int)e->dscalars[3], ph = (int)e->dscalars[4], pw = (int)e->dscalars[5];
    const float *xp = FP(a); float *op = FP(o); int64_t *ip = IP(id);
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t od = 0; od < oD; ++od) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        const float *p = xp + nc * D * H * W; float best = -INFINITY; int64_t bi = 0;
        for (int a3 = 0; a3 < kd; ++a3) for (int b3 = 0; b3 < kh; ++b3) for (int c3 = 0; c3 < kw; ++c3) {
            int64_t dd = od * sd - pd + a3, hh = oh * sh - ph + b3, ww = ow * sw - pw + c3;
            if (dd < 0 || dd >= D || hh < 0 || hh >= H || ww < 0 || ww >= W) continue;
            int64_t fi = (dd * H + hh) * W + ww; float v = p[fi]; if (v > best) { best = v; bi = fi; } }
        int64_t oidx = ((nc * oD + od) * oH + oh) * oW + ow; op[oidx] = best; ip[oidx] = bi;
    }
    return ACLNN_SUCCESS;
}
// ---- AvgPool3dBackward: distribute gradOut/(kd*kh*kw) over window into gradInput ----
aclnnStatus run_avgpool3dbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a; aclTensor *gi = e->out;
    int64_t N = gi->viewDims[0], C = gi->viewDims[1], D = gi->viewDims[2], H = gi->viewDims[3], W = gi->viewDims[4];
    int64_t oD = go->viewDims[2], oH = go->viewDims[3], oW = go->viewDims[4], NC = N * C;
    int kd = (int)e->axes[0], kh = (int)e->axes[1], kw = (int)e->axes[2];
    int sd = (int)e->dscalars[0], sh = (int)e->dscalars[1], sw = (int)e->dscalars[2];
    int pd = (int)e->dscalars[3], ph = (int)e->dscalars[4], pw = (int)e->dscalars[5];
    const float *gop = FP(go); float *gip = FP(gi);
    for (int64_t i = 0, n = gi->numel(); i < n; ++i) gip[i] = 0.f;
    for (int64_t nc = 0; nc < NC; ++nc) for (int64_t od = 0; od < oD; ++od) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        float v = gop[((nc * oD + od) * oH + oh) * oW + ow] / (float)(kd * kh * kw); float *p = gip + nc * D * H * W;
        for (int a3 = 0; a3 < kd; ++a3) for (int b3 = 0; b3 < kh; ++b3) for (int c3 = 0; c3 < kw; ++c3) {
            int64_t dd = od * sd - pd + a3, hh = oh * sh - ph + b3, ww = ow * sw - pw + c3;
            if (dd < 0 || dd >= D || hh < 0 || hh >= H || ww < 0 || ww >= W) continue;
            p[(dd * H + hh) * W + ww] += v; }
    }
    return ACLNN_SUCCESS;
}
// ---- RoiAlignRotated fwd: in[N,C,H,W], rois[K,6]={batch,cx,cy,w,h,theta(rad)} -> out[K,C,ph,pw] ----
aclnnStatus run_roialignrot(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *in = e->a, *rois = e->b; aclTensor *o = e->out;
    int64_t C = in->viewDims[1], H = in->viewDims[2], W = in->viewDims[3];
    int64_t K = o->viewDims[0], ph = o->viewDims[2], pw = o->viewDims[3];
    float scale = (float)e->alpha; int ratio = (int)e->reduceCount; if (ratio <= 0) ratio = 2;
    const float *xp = FP(in), *rp = FP(rois); float *op = FP(o);
    for (int64_t k = 0; k < K; ++k) for (int64_t c = 0; c < C; ++c) for (int64_t py = 0; py < ph; ++py) for (int64_t px = 0; px < pw; ++px) {
        const float *r = rp + k * 6; int b = (int)r[0];
        float cx = r[1] * scale, cy = r[2] * scale, rw = std::max(r[3] * scale, 1.f), rh = std::max(r[4] * scale, 1.f), th = r[5];
        float ct = std::cos(th), st = std::sin(th), bw = rw / pw, bh = rh / ph;
        const float *p = xp + (b * C + c) * H * W; float x0 = -rw / 2.f, y0 = -rh / 2.f;
        double sum = 0; int cnt = ratio * ratio;
        for (int iy = 0; iy < ratio; ++iy) for (int ix = 0; ix < ratio; ++ix) {
            float ly = y0 + py * bh + (iy + 0.5f) * bh / ratio, lx = x0 + px * bw + (ix + 0.5f) * bw / ratio;
            float gx = cx + lx * ct - ly * st, gy = cy + lx * st + ly * ct;
            sum += bilin(p, (int)H, (int)W, gy, gx); }
        op[((k * C + c) * ph + py) * pw + px] = (float)(sum / cnt);
    }
    return ACLNN_SUCCESS;
}
// ---- RoiPoolingWithArgMax: in[N,C,H,W], rois[K,5]={batch,x1,y1,x2,y2} -> out[K,C,ph,pw] + argmax (flat h*W+w) ----
aclnnStatus run_roipool(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *in = e->a, *rois = e->b; aclTensor *o = e->out, *am = e->out2;
    int64_t C = in->viewDims[1], H = in->viewDims[2], W = in->viewDims[3];
    int64_t K = o->viewDims[0], ph = o->viewDims[2], pw = o->viewDims[3];
    float scale = (float)e->alpha;
    const float *xp = FP(in), *rp = FP(rois); float *op = FP(o); int64_t *amp = IP(am);
    for (int64_t k = 0; k < K; ++k) for (int64_t c = 0; c < C; ++c) for (int64_t py = 0; py < ph; ++py) for (int64_t px = 0; px < pw; ++px) {
        const float *r = rp + k * 5; int b = (int)r[0];
        int x1 = (int)std::round(r[1] * scale), y1 = (int)std::round(r[2] * scale), x2 = (int)std::round(r[3] * scale), y2 = (int)std::round(r[4] * scale);
        int rw = std::max(x2 - x1 + 1, 1), rh = std::max(y2 - y1 + 1, 1); float bw = (float)rw / pw, bh = (float)rh / ph;
        int hs = y1 + (int)std::floor(py * bh), he = y1 + (int)std::ceil((py + 1) * bh);
        int ws = x1 + (int)std::floor(px * bw), we = x1 + (int)std::ceil((px + 1) * bw);
        hs = std::min(std::max(hs, 0), (int)H); he = std::min(std::max(he, 0), (int)H);
        ws = std::min(std::max(ws, 0), (int)W); we = std::min(std::max(we, 0), (int)W);
        const float *p = xp + (b * C + c) * H * W; float best = -INFINITY; int64_t bi = -1;
        for (int h = hs; h < he; ++h) for (int w = ws; w < we; ++w) { float v = p[h * W + w]; if (v > best) { best = v; bi = (int64_t)h * W + w; } }
        if (bi < 0) { best = 0; bi = 0; }
        int64_t oidx = ((k * C + c) * ph + py) * pw + px; op[oidx] = best; amp[oidx] = bi;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnUpsampleNearest2d, run_upnearest)
aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnUpsampleBilinear2d, run_upbilinear)
aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->out2 = indices; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAdaptiveMaxPool2d, run_adaptmax2d)
aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAdaptiveAvgPool3d, run_adaptavg3d)
aclnnStatus aclnnMaxPool1dGetWorkspaceSize(const aclTensor *self, int64_t kernel, int64_t stride, int64_t padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = 1; e->axes = {kernel}; e->stride[0] = stride ? stride : kernel; e->pad[0] = padding; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnMaxPool1d, run_pool1d)
aclnnStatus aclnnAvgPool1dGetWorkspaceSize(const aclTensor *self, int64_t kernel, int64_t stride, int64_t padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = 0; e->axes = {kernel}; e->stride[0] = stride ? stride : kernel; e->pad[0] = padding; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAvgPool1d, run_pool1d)
aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor *theta, int64_t H, int64_t W, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = theta; e->out = out; e->m = H; e->n = W; e->keepDim = alignCorners; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAffineGrid, run_affinegrid)
aclnnStatus aclnnGridSample2dGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->b = grid; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnGridSample2d, run_gridsample2d)
aclnnStatus aclnnNmsGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = boxes; e->b = scores; e->out = keepOut; e->out2 = countOut; e->alpha = iouThreshold; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnNms, run_nms)
aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    (void)dilation; (void)ceilMode; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->out2 = indices;
    e->axes = {kernel->v[0], kernel->v.size() > 1 ? kernel->v[1] : kernel->v[0], kernel->v.size() > 2 ? kernel->v[2] : kernel->v[0]};
    int64_t s0 = stride && !stride->v.empty() ? stride->v[0] : kernel->v[0];
    int64_t s1 = stride && stride->v.size() > 1 ? stride->v[1] : s0, s2 = stride && stride->v.size() > 2 ? stride->v[2] : s0;
    int64_t p0 = padding && !padding->v.empty() ? padding->v[0] : 0;
    int64_t p1 = padding && padding->v.size() > 1 ? padding->v[1] : p0, p2 = padding && padding->v.size() > 2 ? padding->v[2] : p0;
    e->dscalars = {(double)s0, (double)s1, (double)s2, (double)p0, (double)p1, (double)p2};
    *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnMaxPool3dWithArgmax, run_maxpool3d)
aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->out = gradInput;
    e->axes = {kernel->v[0], kernel->v.size() > 1 ? kernel->v[1] : kernel->v[0], kernel->v.size() > 2 ? kernel->v[2] : kernel->v[0]};
    int64_t s0 = stride && !stride->v.empty() ? stride->v[0] : kernel->v[0];
    int64_t s1 = stride && stride->v.size() > 1 ? stride->v[1] : s0, s2 = stride && stride->v.size() > 2 ? stride->v[2] : s0;
    int64_t p0 = padding && !padding->v.empty() ? padding->v[0] : 0;
    int64_t p1 = padding && padding->v.size() > 1 ? padding->v[1] : p0, p2 = padding && padding->v.size() > 2 ? padding->v[2] : p0;
    e->dscalars = {(double)s0, (double)s1, (double)s2, (double)p0, (double)p1, (double)p2};
    *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnAvgPool3dBackward, run_avgpool3dbwd)
aclnnStatus aclnnRoiAlignRotatedGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->b = rois; e->out = out; e->alpha = spatialScale; e->reduceCount = samplingRatio > 0 ? samplingRatio : 2; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnRoiAlignRotated, run_roialignrot)
aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, aclTensor *out, aclTensor *argmax, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->a = self; e->b = rois; e->out = out; e->out2 = argmax; e->alpha = spatialScale; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(aclnnRoiPoolingWithArgMax, run_roipool)
} // extern "C"
