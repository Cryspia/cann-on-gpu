// test_remainder R5/R12: vision remainder (AdaptiveMaxPool3d, LpPool2d, UpsampleBicubic2d, GridSample3d,
// RoiAlign, Iou). Host-side over unified memory. Math matches tests/test_remainder.cpp references.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

enum { K_AMP3D, K_LPPOOL2D, K_UPBICUBIC, K_GRIDSAMPLE3D, K_ROIALIGN, K_IOU };

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    case K_AMP3D: {   // self [N,C,D,H,W] -> [N,C,oD,oH,oW] adaptive max
        const aclTensor *X = e->a; aclTensor *Z = e->out; int nd = (int)X->viewDims.size();
        int D = (int)X->viewDims[nd - 3], H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1];
        int znd = (int)Z->viewDims.size();
        int oD = (int)Z->viewDims[znd - 3], oH = (int)Z->viewDims[znd - 2], oW = (int)Z->viewDims[znd - 1];
        int64_t outer = 1; for (int d = 0; d < nd - 3; d++) outer *= X->viewDims[d];
        const float *x = FP(X); float *z = FP(Z);
        for (int64_t o = 0; o < outer; o++) { const float *xb = x + o * D * H * W; float *zb = z + o * oD * oH * oW;
            for (int od = 0; od < oD; od++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
                double best = -1e30;
                for (int d = od * D / oD; d < (od + 1) * D / oD; d++) for (int h = oh * H / oH; h < (oh + 1) * H / oH; h++) for (int w = ow * W / oW; w < (ow + 1) * W / oW; w++)
                    best = std::max(best, (double)xb[(d * H + h) * W + w]);
                zb[(od * oH + oh) * oW + ow] = (float)best;
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_LPPOOL2D: {   // self [N,C,H,W] -> [N,C,oH,oW]; p in alpha; kernel/stride in axes (kh,kw,sh,sw)
        const aclTensor *X = e->a; aclTensor *Z = e->out; double p = e->alpha; int nd = (int)X->viewDims.size();
        int H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1];
        int kh = (int)e->axes[0], kw = (int)e->axes[1], sh = (int)e->axes[2], sw = (int)e->axes[3];
        int znd = (int)Z->viewDims.size(); int oH = (int)Z->viewDims[znd - 2], oW = (int)Z->viewDims[znd - 1];
        int64_t outer = 1; for (int d = 0; d < nd - 2; d++) outer *= X->viewDims[d];
        const float *x = FP(X); float *z = FP(Z);
        for (int64_t o = 0; o < outer; o++) { const float *xb = x + o * H * W; float *zb = z + o * oH * oW;
            for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
                double s2 = 0; for (int a = 0; a < kh; a++) for (int b = 0; b < kw; b++) s2 += std::pow(std::fabs((double)xb[(oh * sh + a) * W + (ow * sw + b)]), p);
                zb[oh * oW + ow] = (float)std::pow(s2, 1.0 / p);
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_UPBICUBIC: {   // self [N,C,H,W] -> [N,C,oH,oW]; alignCorners in m; cubic a=-0.75
        const aclTensor *X = e->a; aclTensor *Z = e->out; bool align = e->m != 0; int nd = (int)X->viewDims.size();
        int H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1];
        int znd = (int)Z->viewDims.size(); int oH = (int)Z->viewDims[znd - 2], oW = (int)Z->viewDims[znd - 1];
        int64_t outer = 1; for (int d = 0; d < nd - 2; d++) outer *= X->viewDims[d];
        auto cub = [](float t) { t = std::fabs(t); float a = -0.75f; if (t <= 1) return ((a + 2) * t - (a + 3)) * t * t + 1; if (t < 2) return (((t - 5) * t + 8) * t - 4) * a; return 0.f; };
        const float *x = FP(X); float *z = FP(Z);
        for (int64_t o = 0; o < outer; o++) { const float *xb = x + o * H * W; float *zb = z + o * oH * oW;
            for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
                float fh, fw;
                if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0; }
                else { fh = (oh + 0.5f) * H / oH - 0.5f; fw = (ow + 0.5f) * W / oW - 0.5f; }
                int y0 = (int)std::floor(fh), x0 = (int)std::floor(fw); float dy = fh - y0, dx = fw - x0;
                double acc = 0;
                for (int m = -1; m <= 2; m++) { float wy = cub(dy - m); int yy = std::min(std::max(y0 + m, 0), H - 1);
                    for (int nn = -1; nn <= 2; nn++) { float wx = cub(dx - nn); int xx = std::min(std::max(x0 + nn, 0), W - 1); acc += (double)wy * wx * xb[yy * W + xx]; } }
                zb[oh * oW + ow] = (float)acc;
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_GRIDSAMPLE3D: {   // self [N,C,D,H,W], grid [N,oD,oH,oW,3] (x,y,z normalized), bilinear, alignCorners in m
        const aclTensor *X = e->a, *G = e->b; aclTensor *Z = e->out; bool align = e->m != 0;
        int nd = (int)X->viewDims.size();
        int D = (int)X->viewDims[nd - 3], H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1];
        int C = (int)X->viewDims[1];
        int gnd = (int)G->viewDims.size(); int oD = (int)G->viewDims[gnd - 4], oH = (int)G->viewDims[gnd - 3], oW = (int)G->viewDims[gnd - 2];
        const float *x = FP(X), *g = FP(G); float *z = FP(Z);
        auto unnorm = [&](float c, int sz) { return align ? (c + 1) * 0.5f * (sz - 1) : ((c + 1) * sz - 1) * 0.5f; };
        for (int c = 0; c < C; c++) { const float *xb = x + c * D * H * W; float *zb = z + c * oD * oH * oW;
            for (int od = 0; od < oD; od++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
                int go = ((od * oH + oh) * oW + ow) * 3; float gx = g[go], gy = g[go + 1], gz = g[go + 2];
                float fx = unnorm(gx, W), fy = unnorm(gy, H), fz = unnorm(gz, D);
                int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), z0 = (int)std::floor(fz);
                int x1 = x0 + 1, y1 = y0 + 1, z1 = z0 + 1;
                float wx1 = fx - x0, wy1 = fy - y0, wz1 = fz - z0, wx0 = 1 - wx1, wy0 = 1 - wy1, wz0 = 1 - wz1;
                auto smp = [&](int zz, int yy, int xx) -> double { if (xx < 0 || xx >= W || yy < 0 || yy >= H || zz < 0 || zz >= D) return 0.0; return xb[(zz * H + yy) * W + xx]; };
                double v = wz0 * (wy0 * (wx0 * smp(z0, y0, x0) + wx1 * smp(z0, y0, x1)) + wy1 * (wx0 * smp(z0, y1, x0) + wx1 * smp(z0, y1, x1)))
                         + wz1 * (wy0 * (wx0 * smp(z1, y0, x0) + wx1 * smp(z1, y0, x1)) + wy1 * (wx0 * smp(z1, y1, x0) + wx1 * smp(z1, y1, x1)));
                zb[(od * oH + oh) * oW + ow] = (float)v;
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_ROIALIGN: {   // self [1,C,H,W], rois [K,5]=(batch,x1,y1,x2,y2); spatialScale in alpha; ratio in dim
        const aclTensor *X = e->a, *R = e->b; aclTensor *Z = e->out; double scale = e->alpha; int ratio = (int)e->dim;
        int nd = (int)X->viewDims.size(); int H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1]; int C = (int)X->viewDims[1];
        int znd = (int)Z->viewDims.size(); int ph = (int)Z->viewDims[znd - 2], pw = (int)Z->viewDims[znd - 1];
        const float *x = FP(X), *rois = FP(R); float *z = FP(Z); int K = (int)R->viewDims[0];
        auto bil = [&](const float *xb, float fy, float fx) -> float {
            if (fy < -1 || fy > H || fx < -1 || fx > W) return 0.f; fy = std::max(fy, 0.f); fx = std::max(fx, 0.f);
            int y0 = (int)fy, x0 = (int)fx, y1 = std::min(y0 + 1, H - 1), x1 = std::min(x0 + 1, W - 1); float dy = fy - y0, dx = fx - x0;
            return (1 - dy) * (1 - dx) * xb[y0 * W + x0] + (1 - dy) * dx * xb[y0 * W + x1] + dy * (1 - dx) * xb[y1 * W + x0] + dy * dx * xb[y1 * W + x1];
        };
        for (int k = 0; k < K; k++) { const float *roi = rois + k * 5;
            float rx1 = roi[1] * scale, ry1 = roi[2] * scale, rx2 = roi[3] * scale, ry2 = roi[4] * scale;
            float rw = rx2 - rx1, rh = ry2 - ry1, bw = rw / pw, bh = rh / ph;
            for (int c = 0; c < C; c++) { const float *xb = x + c * H * W; float *zb = z + (k * C + c) * ph * pw;
                for (int py = 0; py < ph; py++) for (int px = 0; px < pw; px++) {
                    double s2 = 0; for (int iy = 0; iy < ratio; iy++) for (int ix = 0; ix < ratio; ix++) {
                        float yy = ry1 + py * bh + (iy + 0.5f) * bh / ratio, xx = rx1 + px * bw + (ix + 0.5f) * bw / ratio; s2 += bil(xb, yy, xx); }
                    zb[py * pw + px] = (float)(s2 / (ratio * ratio));
                }
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_IOU: {   // boxes1/boxes2 [N,4]=(x1,y1,x2,y2) -> out[N]
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int N = (int)A->viewDims[0];
        const float *a = FP(A), *b = FP(B); float *o = FP(O);
        for (int i = 0; i < N; i++) { const float *Ab = a + i * 4, *Bb = b + i * 4;
            double ix1 = std::max(Ab[0], Bb[0]), iy1 = std::max(Ab[1], Bb[1]), ix2 = std::min(Ab[2], Bb[2]), iy2 = std::min(Ab[3], Bb[3]);
            double iw = std::max(ix2 - ix1, 0.0), ih = std::max(iy2 - iy1, 0.0), inter = iw * ih;
            double ua = (Ab[2] - Ab[0]) * (Ab[3] - Ab[1]) + (Bb[2] - Bb[0]) * (Bb[3] - Bb[1]) - inter;
            o[i] = (float)(ua > 0 ? inter / ua : 0);
        }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUNFN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUNFN(rv_run)
} // namespace

extern "C" {
aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_AMP3D; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnAdaptiveMaxPool3d)
aclnnStatus aclnnLpPool2dGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *kernel, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LPPOOL2D; e->a = self; e->out = out; e->alpha = p;
    int kh = kernel->v[0], kw = kernel->v.size() > 1 ? (int)kernel->v[1] : kh;
    int sh = stride && !stride->v.empty() ? (int)stride->v[0] : kh, sw = stride && stride->v.size() > 1 ? (int)stride->v[1] : sh;
    e->axes = {kh, kw, sh, sw}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLpPool2d)
aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_UPBICUBIC; e->a = self; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnUpsampleBicubic2d)
aclnnStatus aclnnGridSample3dGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_GRIDSAMPLE3D; e->a = self; e->b = grid; e->out = out; e->m = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnGridSample3d)
aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_ROIALIGN; e->a = self; e->b = rois; e->out = out; e->alpha = spatialScale; e->dim = samplingRatio; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnRoiAlign)
aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_IOU; e->a = boxes1; e->b = boxes2; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnIou)
} // extern "C"
