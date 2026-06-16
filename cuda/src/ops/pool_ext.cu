// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <vector>

namespace _pool_ext {
// Pooling / interpolate / vision extensions (P6): UpsampleNearest2d/Bilinear2d, AdaptiveMaxPool2d,
// AdaptiveAvgPool3d, MaxPool1d/AvgPool1d, GridSample2d (bilinear), AffineGrid, NMS.
// fp32-centric (vision ops); self-contained executors. out contiguous.

namespace {

constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__device__ inline int64_t imin(int64_t a, int64_t b) { return a < b ? a : b; }

template <typename T> __global__ void k_upsample_nearest(const T *in, T *o, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= NC * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, nc = i / (oH * oW);
    int64_t ih = imin((int64_t)(oh * H / oH), H - 1), iw = imin((int64_t)(ow * W / oW), W - 1);
    o[i] = in[(nc * H + ih) * W + iw];
}
template <typename T> __global__ void k_upsample_bilinear(const T *in, T *o, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW, bool align) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= NC * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, nc = i / (oH * oW);
    float fh, fw;
    if (align) { fh = oH > 1 ? (float)oh * (H - 1) / (oH - 1) : 0.f; fw = oW > 1 ? (float)ow * (W - 1) / (oW - 1) : 0.f; }
    else { fh = ((oh + 0.5f) * H / oH) - 0.5f; fw = ((ow + 0.5f) * W / oW) - 0.5f; }
    fh = fh < 0 ? 0 : fh; fw = fw < 0 ? 0 : fw;
    int64_t h0 = (int64_t)fh, w0 = (int64_t)fw, h1 = imin(h0 + 1, H - 1), w1 = imin(w0 + 1, W - 1);
    float dh = fh - h0, dw = fw - w0; const T *p = in + nc * H * W;
    float v = (float)p[h0*W+w0]*(1-dh)*(1-dw) + (float)p[h0*W+w1]*(1-dh)*dw + (float)p[h1*W+w0]*dh*(1-dw) + (float)p[h1*W+w1]*dh*dw;
    o[i] = (T)v;
}
// AdaptiveMaxPool2d: pool over adaptive region; optional indices
template <typename T> __global__ void k_adaptive_max2d(const T *in, T *o, int64_t *idx, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= NC * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, nc = i / (oH * oW);
    int64_t hs = oh * H / oH, he = (oh + 1) * H / oH + ((oh + 1) * H % oH ? 1 : 0);
    int64_t ws = ow * W / oW, we = (ow + 1) * W / oW + ((ow + 1) * W % oW ? 1 : 0);
    const T *p = in + nc * H * W; float best = -1e30f; int64_t bidx = hs * W + ws;
    for (int64_t h = hs; h < he; ++h) for (int64_t w = ws; w < we; ++w) { float v = (float)p[h*W+w]; if (v > best) { best = v; bidx = h * W + w; } }
    o[i] = (T)best; if (idx) idx[i] = bidx;
}
template <typename T> __global__ void k_adaptive_avg3d(const T *in, T *o, int64_t NC, int64_t D, int64_t H, int64_t W, int64_t oD, int64_t oH, int64_t oW) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= NC * oD * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, od = (i / (oW * oH)) % oD, nc = i / (oW * oH * oD);
    int64_t ds = od*D/oD, de = (od+1)*D/oD + ((od+1)*D%oD?1:0);
    int64_t hs = oh*H/oH, he = (oh+1)*H/oH + ((oh+1)*H%oH?1:0);
    int64_t ws = ow*W/oW, we = (ow+1)*W/oW + ((ow+1)*W%oW?1:0);
    const T *p = in + nc * D * H * W; double s = 0; int64_t cnt = 0;
    for (int64_t d=ds; d<de; ++d) for (int64_t h=hs; h<he; ++h) for (int64_t w=ws; w<we; ++w) { s += (double)p[(d*H+h)*W+w]; cnt++; }
    o[i] = (T)(s / cnt);
}
// 1D pooling (max/avg) over [NC, L]
template <typename T, bool MAX> __global__ void k_pool1d(const T *in, T *o, int64_t NC, int64_t L, int64_t oL, int64_t k, int64_t st, int64_t pad) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= NC * oL) return;
    int64_t ol = i % oL, nc = i / oL; const T *p = in + nc * L;
    float acc = MAX ? -1e30f : 0.f; int64_t cnt = 0;
    for (int64_t j = 0; j < k; ++j) { int64_t l = ol * st - pad + j; if (l < 0 || l >= L) continue;
        float v = (float)p[l]; if (MAX) acc = v > acc ? v : acc; else acc += v; cnt++; }
    o[i] = (T)(MAX ? acc : acc / (cnt ? cnt : 1));
}
// GridSample2d bilinear, padding=zeros
template <typename T> __global__ void k_grid_sample2d(const T *in, const float *grid, T *o, int64_t N, int64_t C, int64_t H, int64_t W, int64_t oH, int64_t oW, bool align) {
    int64_t total = N * C * oH * oW; int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, c = (i / (oW * oH)) % C, n = i / (oW * oH * C);
    const float *g = grid + ((n * oH + oh) * oW + ow) * 2; float gx = g[0], gy = g[1];
    float fx, fy;
    if (align) { fx = (gx + 1) * 0.5f * (W - 1); fy = (gy + 1) * 0.5f * (H - 1); }
    else { fx = ((gx + 1) * W - 1) * 0.5f; fy = ((gy + 1) * H - 1) * 0.5f; }
    int64_t x0 = (int64_t)floorf(fx), y0 = (int64_t)floorf(fy), x1 = x0 + 1, y1 = y0 + 1;
    float dx = fx - x0, dy = fy - y0; const T *p = in + (n * C + c) * H * W;
    auto at = [&](int64_t y, int64_t x) -> float { return (y >= 0 && y < H && x >= 0 && x < W) ? (float)p[y*W+x] : 0.f; };
    o[i] = (T)(at(y0,x0)*(1-dy)*(1-dx) + at(y0,x1)*(1-dy)*dx + at(y1,x0)*dy*(1-dx) + at(y1,x1)*dy*dx);
}
// AffineGrid: theta[N,2,3] -> grid[N,H,W,2]; base coords normalized to [-1,1]
__global__ void k_affine_grid(const float *theta, float *grid, int64_t N, int64_t H, int64_t W, bool align) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= N * H * W) return;
    int64_t w = i % W, h = (i / W) % H, n = i / (H * W);
    float xn, yn;
    if (align) { xn = W > 1 ? (float)w / (W - 1) * 2 - 1 : 0.f; yn = H > 1 ? (float)h / (H - 1) * 2 - 1 : 0.f; }
    else { xn = ((w + 0.5f) / W) * 2 - 1; yn = ((h + 0.5f) / H) * 2 - 1; }
    const float *t = theta + n * 6;
    grid[i*2+0] = t[0]*xn + t[1]*yn + t[2];
    grid[i*2+1] = t[3]*xn + t[4]*yn + t[5];
}
// NMS greedy (single thread): boxes[M,4] x1y1x2y2, scores[M] (assumed already sorted desc by caller index order), iou thresh
__global__ void k_nms(const float *boxes, const int64_t *order, int64_t *keep, int64_t *kcount, int64_t M, float iou, bool *removed) {
    for (int64_t i = 0; i < M; ++i) removed[i] = false;
    int64_t kc = 0;
    for (int64_t a = 0; a < M; ++a) { int64_t i = order[a]; if (removed[i]) continue; keep[kc++] = i;
        float ax1=boxes[i*4],ay1=boxes[i*4+1],ax2=boxes[i*4+2],ay2=boxes[i*4+3]; float aarea=(ax2-ax1)*(ay2-ay1);
        for (int64_t b = a + 1; b < M; ++b) { int64_t j = order[b]; if (removed[j]) continue;
            float bx1=boxes[j*4],by1=boxes[j*4+1],bx2=boxes[j*4+2],by2=boxes[j*4+3];
            float ix1=fmaxf(ax1,bx1),iy1=fmaxf(ay1,by1),ix2=fminf(ax2,bx2),iy2=fminf(ay2,by2);
            float iw=fmaxf(0.f,ix2-ix1),ih=fmaxf(0.f,iy2-iy1),inter=iw*ih;
            float barea=(bx2-bx1)*(by2-by1); float u=aarea+barea-inter;
            if (u > 0 && inter / u > iou) removed[j] = true; } }
    *kcount = kc;
}

inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
#define DISP_FH(esz, L) do { switch (esz) { case 2: { L(__half); } break; case 4: { L(float); } break; default: return ACLNN_ERR_PARAM_INVALID; } } while (0)

} // namespace

extern "C" {

// ---- UpsampleNearest2d / UpsampleBilinear2d ([N,C,H,W] -> [N,C,oH,oW]) ----
static aclnnStatus up_ws(int mode, const aclTensor *self, bool align, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 4 || out->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = mode; e->a = self; e->out = out; e->keepDim = align;
    e->outerCount = self->viewDims[0] * self->viewDims[1]; e->m = self->viewDims[2]; e->n = self->viewDims[3];
    e->k = out->viewDims[2]; e->reduceCount = out->viewDims[3];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return up_ws(0, self, false, out, ws, ex); }
aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return up_ws(1, self, alignCorners, out, ws, ex); }
static aclnnStatus up_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t NC = e->outerCount, H = e->m, W = e->n, oH = e->k, oW = e->reduceCount; size_t esz = dtype_size(e->a->dtype); int64_t g = nb(NC * oH * oW);
    if (e->op == 0) {
        #define L(T) k_upsample_nearest<T><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,H,W,oH,oW)
        DISP_FH(esz, L);
        #undef L
    } else {
        #define L(T) k_upsample_bilinear<T><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,H,W,oH,oW,e->keepDim)
        DISP_FH(esz, L);
        #undef L
    }
    return done(e);
}
aclnnStatus aclnnUpsampleNearest2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return up_run(e, (cudaStream_t)s); }
aclnnStatus aclnnUpsampleBilinear2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return up_run(e, (cudaStream_t)s); }

// ---- AdaptiveMaxPool2d (values + optional indices) ----
aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (indices && indices->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->out2 = indices;
    e->outerCount = self->viewDims[0]*self->viewDims[1]; e->m = self->viewDims[2]; e->n = self->viewDims[3];
    e->k = out->viewDims[2]; e->reduceCount = out->viewDims[3];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveMaxPool2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount; size_t esz=dtype_size(e->a->dtype); int64_t g=nb(NC*oH*oW);
    int64_t *idx = e->out2 ? (int64_t*)e->out2->data : nullptr;
    #define L(T) k_adaptive_max2d<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,idx,NC,H,W,oH,oW)
    DISP_FH(esz, L);
    #undef L
    return done(e);
}
// ---- AdaptiveAvgPool3d ----
aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 5 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out;
    e->outerCount = self->viewDims[0]*self->viewDims[1];
    e->dscalars = { (double)self->viewDims[2], (double)self->viewDims[3], (double)self->viewDims[4], (double)out->viewDims[2], (double)out->viewDims[3], (double)out->viewDims[4] };
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveAvgPool3d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t NC=e->outerCount, D=(int64_t)e->dscalars[0],H=(int64_t)e->dscalars[1],W=(int64_t)e->dscalars[2],oD=(int64_t)e->dscalars[3],oH=(int64_t)e->dscalars[4],oW=(int64_t)e->dscalars[5];
    size_t esz=dtype_size(e->a->dtype); int64_t g=nb(NC*oD*oH*oW);
    #define L(T) k_adaptive_avg3d<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,NC,D,H,W,oD,oH,oW)
    DISP_FH(esz, L);
    #undef L
    return done(e);
}
// ---- MaxPool1d / AvgPool1d ([N,C,L]) ----
static aclnnStatus pool1d_ws(int mode, const aclTensor *self, int64_t k, int64_t stride, int64_t pad, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 3 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = mode; e->a = self; e->out = out;
    e->outerCount = self->viewDims[0]*self->viewDims[1]; e->m = self->viewDims[2]; e->n = out->viewDims[2];
    e->k = k; e->stride[0] = stride ? stride : k; e->pad[0] = pad;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxPool1dGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t stride, int64_t pad, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pool1d_ws(0, self, k, stride, pad, out, ws, ex); }
aclnnStatus aclnnAvgPool1dGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t stride, int64_t pad, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pool1d_ws(1, self, k, stride, pad, out, ws, ex); }
static aclnnStatus pool1d_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t NC=e->outerCount,L=e->m,oL=e->n,k=e->k,st=e->stride[0],pad=e->pad[0]; size_t esz=dtype_size(e->a->dtype); int64_t g=nb(NC*oL);
    #define L_(T) do { if (e->op==0) k_pool1d<T,true><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,L,oL,k,st,pad); \
                       else k_pool1d<T,false><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,L,oL,k,st,pad); } while(0)
    DISP_FH(esz, L_);
    #undef L_
    return done(e);
}
aclnnStatus aclnnMaxPool1d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool1d_run(e, (cudaStream_t)s); }
aclnnStatus aclnnAvgPool1d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool1d_run(e, (cudaStream_t)s); }

// ---- GridSample2d (bilinear, zeros padding) ----
aclnnStatus aclnnGridSample2dGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !grid || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 4 || grid->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->b = grid; e->out = out; e->keepDim = alignCorners;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGridSample2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a, *grid = e->b; int64_t N=a->viewDims[0],C=a->viewDims[1],H=a->viewDims[2],W=a->viewDims[3];
    int64_t oH=grid->viewDims[1],oW=grid->viewDims[2]; size_t esz=dtype_size(a->dtype); int64_t g=nb(N*C*oH*oW);
    #define L(T) k_grid_sample2d<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)a->data,(const float*)grid->data,(T*)e->out->data,N,C,H,W,oH,oW,e->keepDim)
    DISP_FH(esz, L);
    #undef L
    return done(e);
}
// ---- AffineGrid (theta[N,2,3], output grid[N,H,W,2]) ----
aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor *theta, int64_t H, int64_t W, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!theta || !out || !ex || theta->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = theta; e->out = out; e->keepDim = alignCorners;
    e->outerCount = theta->viewDims[0]; e->m = H; e->n = W;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAffineGrid(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N=e->outerCount,H=e->m,W=e->n; int64_t g=nb(N*H*W);
    k_affine_grid<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,N,H,W,e->keepDim);
    return done(e);
}
// ---- NMS (boxes[M,4], scores[M] -> keep indices[M] + count). order = scores descending (computed on host-side sort not needed: we sort by simple selection on device single-thread is O(M^2); instead caller passes pre-sorted? We sort here via a tiny order kernel). ----
__global__ void k_argsort_desc(const float *scores, int64_t *order, int64_t M) {
    for (int64_t i = 0; i < M; ++i) order[i] = i;
    for (int64_t i = 0; i < M; ++i) { int64_t best = i; for (int64_t j = i + 1; j < M; ++j) if (scores[order[j]] > scores[order[best]]) best = j;
        int64_t t = order[i]; order[i] = order[best]; order[best] = t; }
}
aclnnStatus aclnnNmsGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!boxes || !scores || !keepOut || !countOut || !ex || boxes->dtype != ACL_FLOAT || keepOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = boxes; e->b = scores; e->out = keepOut; e->out2 = countOut;
    e->m = boxes->viewDims[0]; e->alpha = iouThreshold;
    if (ws) *ws = (uint64_t)e->m * sizeof(int64_t) + (uint64_t)e->m + 64;   // order[M] (int64) + removed[M] (bool) scratch
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnNms(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s; int64_t M = e->m; int64_t *order = (int64_t *)ws; bool *removed = (bool *)(order + M);
    k_argsort_desc<<<1,1,0,st>>>((const float*)e->b->data, order, M);
    k_nms<<<1,1,0,st>>>((const float*)e->a->data, order, (int64_t*)e->out->data, (int64_t*)e->out2->data, M, (float)e->alpha, removed);
    return done(e);
}

} // extern "C"
} // namespace _pool_ext
#undef DISP_FH

namespace _pool2_ext {
// Pooling remainder (P13/P6): MaxPool2dWithIndices (forward + argmax indices) and MaxUnpool2d (scatter by indices,
// which is also the max-pool backward when fed gradients). fp32; indices are flat (ih*W+iw) within each (n,c) plane.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__global__ void k_maxpool_idx(const float *x, float *o, int64_t *idx, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW,
        int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return;
    int64_t ow=i%oW, oh=(i/oW)%oH, nc=i/(oH*oW); const float *p=x+nc*H*W;
    float best=-1e30f; int64_t bidx=0;
    for(int64_t a=0;a<kh;a++)for(int64_t b=0;b<kw;b++){ int64_t ih=oh*sh-ph+a, iw=ow*sw-pw+b; if(ih<0||ih>=H||iw<0||iw>=W) continue;
        float v=p[ih*W+iw]; if(v>best){best=v;bidx=ih*W+iw;} }
    o[i]=best; idx[i]=bidx;
}
__global__ void k_maxunpool(const float *x, const int64_t *idx, float *o, int64_t NC, int64_t oH, int64_t oW, int64_t H, int64_t W) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return;
    int64_t nc=i/(oH*oW); o[nc*H*W + idx[i]] = x[i];   // input is the pooled map [NC,oH,oW]; place into [NC,H,W] at stored index
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// MaxPool2dWithIndices: self[N,C,H,W] -> out[N,C,oH,oW] + indices (int64)
aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    if (!self||!kernel||!stride||!padding||!out||!indices||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4||indices->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->out2=indices;
    e->outerCount=self->viewDims[0]*self->viewDims[1]; e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->k=out->viewDims[2]; e->reduceCount=out->viewDims[3];
    e->axes={kernel->v[0],kernel->v[1],stride->v[0],stride->v[1],padding->v[0],padding->v[1]};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxPool2dWithIndices(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    k_maxpool_idx<<<nb(NC*oH*oW),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,(int64_t*)e->out2->data,NC,H,W,oH,oW,e->axes[0],e->axes[1],e->axes[2],e->axes[3],e->axes[4],e->axes[5]);
    return done(e);
}
// MaxUnpool2d: self[N,C,oH,oW] + indices -> out[N,C,H,W] (zeros elsewhere). Also = MaxPool2dWithIndices backward (feed grads as self).
aclnnStatus aclnnMaxUnpool2dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self||!indices||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4||indices->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->b=indices; e->out=out;
    e->outerCount=self->viewDims[0]*self->viewDims[1]; e->k=self->viewDims[2]; e->reduceCount=self->viewDims[3]; e->m=H; e->n=W;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxUnpool2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t NC=e->outerCount,oH=e->k,oW=e->reduceCount,H=e->m,W=e->n;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_maxunpool<<<nb(NC*oH*oW),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,NC,oH,oW,H,W);
    return done(e);
}

} // extern "C"
} // namespace _pool2_ext

namespace _pool3_ext {
// Vision remainder (R5): AdaptiveMaxPool3d, LpPool2d, UpsampleBicubic2d, GridSampler3d (trilinear), RoiAlign. fp32.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__device__ inline int64_t imn(int64_t a,int64_t b){return a<b?a:b;}
__device__ inline int64_t imx(int64_t a,int64_t b){return a>b?a:b;}
__device__ inline float cubicw(float t){ t=fabsf(t); float a=-0.75f; if(t<=1) return ((a+2)*t-(a+3))*t*t+1; if(t<2) return (((t-5)*t+8)*t-4)*a; return 0; }

__global__ void k_adaptive_max3d(const float*in,float*o,int64_t NC,int64_t D,int64_t H,int64_t W,int64_t oD,int64_t oH,int64_t oW){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oD*oH*oW) return;
    int64_t ow=i%oW,oh=(i/oW)%oH,od=(i/(oW*oH))%oD,nc=i/(oW*oH*oD);
    int64_t ds=od*D/oD,de=(od+1)*D/oD+((od+1)*D%oD?1:0),hs=oh*H/oH,he=(oh+1)*H/oH+((oh+1)*H%oH?1:0),ws=ow*W/oW,we=(ow+1)*W/oW+((ow+1)*W%oW?1:0);
    const float*p=in+nc*D*H*W; float best=-1e30f; for(int64_t d=ds;d<de;++d)for(int64_t h=hs;h<he;++h)for(int64_t w=ws;w<we;++w){float v=p[(d*H+h)*W+w]; if(v>best)best=v;} o[i]=best;
}
__global__ void k_lppool2d(const float*in,float*o,int64_t NC,int64_t H,int64_t W,int64_t oH,int64_t oW,int64_t kh,int64_t kw,int64_t sh,int64_t sw,float p){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return; int64_t ow=i%oW,oh=(i/oW)%oH,nc=i/(oW*oH); const float*pp=in+nc*H*W;
    double s=0; for(int64_t a=0;a<kh;a++)for(int64_t b=0;b<kw;b++){ int64_t ih=oh*sh+a,iw=ow*sw+b; if(ih<H&&iw<W) s+=pow(fabs((double)pp[ih*W+iw]),(double)p); }
    o[i]=(float)pow(s,1.0/p);
}
__global__ void k_bicubic(const float*in,float*o,int64_t NC,int64_t H,int64_t W,int64_t oH,int64_t oW,bool align){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return; int64_t ow=i%oW,oh=(i/oW)%oH,nc=i/(oW*oH);
    float fh,fw; if(align){fh=oH>1?(float)oh*(H-1)/(oH-1):0;fw=oW>1?(float)ow*(W-1)/(oW-1):0;} else {fh=(oh+0.5f)*H/oH-0.5f;fw=(ow+0.5f)*W/oW-0.5f;}
    int64_t y0=(int64_t)floorf(fh),x0=(int64_t)floorf(fw); float dy=fh-y0,dx=fw-x0; const float*pp=in+nc*H*W; double acc=0;
    for(int m=-1;m<=2;m++){ float wy=cubicw(dy-m); int64_t yy=imn(imx(y0+m,0),H-1); for(int nn=-1;nn<=2;nn++){ float wx=cubicw(dx-nn); int64_t xx=imn(imx(x0+nn,0),W-1); acc+=(double)wy*wx*pp[yy*W+xx]; } }
    o[i]=(float)acc;
}
__global__ void k_grid3d(const float*in,const float*grid,float*o,int64_t N,int64_t C,int64_t D,int64_t H,int64_t W,int64_t oD,int64_t oH,int64_t oW,bool align){
    int64_t tot=N*C*oD*oH*oW; int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot)return;
    int64_t ow=i%oW,oh=(i/oW)%oH,od=(i/(oW*oH))%oD,c=(i/(oW*oH*oD))%C,n=i/(oW*oH*oD*C);
    const float*g=grid+(((n*oD+od)*oH+oh)*oW+ow)*3; float gx=g[0],gy=g[1],gz=g[2];
    float fx,fy,fz; if(align){fx=(gx+1)*0.5f*(W-1);fy=(gy+1)*0.5f*(H-1);fz=(gz+1)*0.5f*(D-1);} else {fx=((gx+1)*W-1)*0.5f;fy=((gy+1)*H-1)*0.5f;fz=((gz+1)*D-1)*0.5f;}
    int64_t x0=(int64_t)floorf(fx),y0=(int64_t)floorf(fy),z0=(int64_t)floorf(fz); float dx=fx-x0,dy=fy-y0,dz=fz-z0; const float*p=in+(n*C+c)*D*H*W;
    auto at=[&](int64_t z,int64_t y,int64_t x)->float{ return (z>=0&&z<D&&y>=0&&y<H&&x>=0&&x<W)?p[(z*H+y)*W+x]:0.f; };
    double v=0; for(int dz_=0;dz_<2;dz_++)for(int dy_=0;dy_<2;dy_++)for(int dx_=0;dx_<2;dx_++){ float wz=dz_?dz:1-dz, wy=dy_?dy:1-dy, wx=dx_?dx:1-dx; v+=(double)wz*wy*wx*at(z0+dz_,y0+dy_,x0+dx_); }
    o[i]=(float)v;
}
// RoiAlign: in[N,C,H,W], rois[K,5]={batch,x1,y1,x2,y2}(input scale) -> out[K,C,ph,pw]; sampling_ratio ratio, avg.
__global__ void k_roialign(const float*in,const float*rois,float*o,int64_t C,int64_t H,int64_t W,int64_t K,int64_t ph,int64_t pw,float scale,int ratio){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=K*C*ph*pw)return; int64_t px=i%pw,py=(i/pw)%ph,c=(i/(pw*ph))%C,k=i/(pw*ph*C);
    const float*r=rois+k*5; int64_t nb_=(int64_t)r[0]; float x1=r[1]*scale,y1=r[2]*scale,x2=r[3]*scale,y2=r[4]*scale;
    float rw=fmaxf(x2-x1,1.f),rh=fmaxf(y2-y1,1.f),bw=rw/pw,bh=rh/ph; const float*p=in+(nb_*C+c)*H*W;
    auto bil=[&](float fy,float fx)->float{ if(fy<-1||fy>H||fx<-1||fx>W)return 0; fy=fmaxf(fy,0);fx=fmaxf(fx,0); int y0=(int)fy,x0=(int)fx,y1i=min(y0+1,(int)H-1),x1i=min(x0+1,(int)W-1); float dy=fy-y0,dx=fx-x0;
        return (1-dy)*(1-dx)*p[y0*W+x0]+(1-dy)*dx*p[y0*W+x1i]+dy*(1-dx)*p[y1i*W+x0]+dy*dx*p[y1i*W+x1i]; };
    double s=0; int cnt=ratio*ratio; for(int iy=0;iy<ratio;iy++)for(int ix=0;ix<ratio;ix++){ float yy=y1+py*bh+(iy+0.5f)*bh/ratio; float xx=x1+px*bw+(ix+0.5f)*bw/ratio; s+=bil(yy,xx); }
    o[i]=(float)(s/cnt);
}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor*self,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=5) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->outerCount=self->viewDims[0]*self->viewDims[1];
    e->dscalars={(double)self->viewDims[2],(double)self->viewDims[3],(double)self->viewDims[4],(double)out->viewDims[2],(double)out->viewDims[3],(double)out->viewDims[4]};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveMaxPool3d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ auto&d=e->dscalars; int64_t NC=e->outerCount,D=d[0],H=d[1],W=d[2],oD=d[3],oH=d[4],oW=d[5];
    k_adaptive_max3d<<<nb(NC*oD*oH*oW),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,NC,D,H,W,oD,oH,oW); return done(e); }

aclnnStatus aclnnLpPool2dGetWorkspaceSize(const aclTensor*self,double p,const aclIntArray*kernel,const aclIntArray*stride,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!kernel||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->alpha=p; e->outerCount=self->viewDims[0]*self->viewDims[1]; e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->k=out->viewDims[2]; e->reduceCount=out->viewDims[3];
    int64_t sh=stride&&stride->v.size()>0?stride->v[0]:kernel->v[0], sw=stride&&stride->v.size()>1?stride->v[1]:kernel->v[1];
    e->axes={kernel->v[0],kernel->v[1],sh,sw}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLpPool2d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    k_lppool2d<<<nb(NC*oH*oW),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,NC,H,W,oH,oW,e->axes[0],e->axes[1],e->axes[2],e->axes[3],(float)e->alpha); return done(e); }

aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor*self,bool alignCorners,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->keepDim=alignCorners; e->outerCount=self->viewDims[0]*self->viewDims[1]; e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->k=out->viewDims[2]; e->reduceCount=out->viewDims[3];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUpsampleBicubic2d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    k_bicubic<<<nb(NC*oH*oW),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,NC,H,W,oH,oW,e->keepDim); return done(e); }

aclnnStatus aclnnGridSample3dGetWorkspaceSize(const aclTensor*self,const aclTensor*grid,bool alignCorners,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!grid||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=5||grid->viewDims.size()!=5) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=grid; e->out=out; e->keepDim=alignCorners; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGridSample3d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ const aclTensor*a=e->a,*g=e->b;
    int64_t N=a->viewDims[0],C=a->viewDims[1],D=a->viewDims[2],H=a->viewDims[3],W=a->viewDims[4],oD=g->viewDims[1],oH=g->viewDims[2],oW=g->viewDims[3];
    k_grid3d<<<nb(N*C*oD*oH*oW),TH,0,(cudaStream_t)s>>>((const float*)a->data,(const float*)g->data,(float*)e->out->data,N,C,D,H,W,oD,oH,oW,e->keepDim); return done(e); }

aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor*self,const aclTensor*rois,double spatialScale,int64_t samplingRatio,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!rois||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4||out->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=rois; e->out=out; e->alpha=spatialScale; e->reduceCount=samplingRatio;
    e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->outerCount=self->viewDims[1]; e->k=rois->viewDims[0];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiAlign(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->out->viewDims[2],pw=e->out->viewDims[3];
    k_roialign<<<nb(K*C*ph*pw),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,C,H,W,K,ph,pw,(float)e->alpha,(int)e->reduceCount); return done(e); }

} // extern "C"
} // namespace _pool3_ext

namespace _pool4_ext {
// Pool backward (avg 3d / adaptive-avg 3d), MaxPool3dWithArgmax forward, MaxPool generic alias,
// and RoI (align-rotated + grad, align v2 backward, pooling-with-argmax + grad). fp32.
//   Avg backward distributes grad uniformly over the window; max/argmax scatter-add by saved index;
//   RoI backward scatters bilinear weights of gradOut into gradInput.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

// MaxPool3dWithArgmax: NCDHW window max + flat argmax (index within this NC plane's D*H*W).
__global__ void k_maxpool3d_argmax(const float*x,float*o,int64_t*idx,int64_t NC,int D,int H,int W,int oD,int oH,int oW,
        int kd,int kh,int kw,int sd,int sh,int sw,int pd,int ph,int pw){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oD*oH*oW) return;
    int ow=i%oW,oh=(i/oW)%oH,od=(i/(oW*oH))%oD; int64_t nc=i/((int64_t)oW*oH*oD);
    const float*p=x+nc*D*H*W; float best=-1e30f; int64_t bidx=0;
    for(int a=0;a<kd;a++)for(int b=0;b<kh;b++)for(int c=0;c<kw;c++){ int dd=od*sd-pd+a, hh=oh*sh-ph+b, ww=ow*sw-pw+c;
        if(dd<0||dd>=D||hh<0||hh>=H||ww<0||ww>=W) continue; int64_t fi=((int64_t)dd*H+hh)*W+ww; float v=p[fi]; if(v>best){best=v;bidx=fi;} }
    o[i]=best; idx[i]=bidx;
}
// AvgPool3d backward: distribute gradOut/(kd*kh*kw) over window into gradIn.
__global__ void k_avgpool3d_bwd(const float*go,float*gi,int64_t NC,int D,int H,int W,int oD,int oH,int oW,
        int kd,int kh,int kw,int sd,int sh,int sw,int pd,int ph,int pw){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oD*oH*oW) return;
    int ow=i%oW,oh=(i/oW)%oH,od=(i/(oW*oH))%oD; int64_t nc=i/((int64_t)oW*oH*oD);
    float v=go[i]/(float)(kd*kh*kw); float*p=gi+nc*D*H*W;
    for(int a=0;a<kd;a++)for(int b=0;b<kh;b++)for(int c=0;c<kw;c++){ int dd=od*sd-pd+a, hh=oh*sh-ph+b, ww=ow*sw-pw+c;
        if(dd<0||dd>=D||hh<0||hh>=H||ww<0||ww>=W) continue; atomicAdd(&p[((int64_t)dd*H+hh)*W+ww], v); }
}
// AdaptiveAvgPool3d backward: distribute gradOut over adaptive region.
__global__ void k_adaptive_avgpool3d_bwd(const float*go,float*gi,int64_t NC,int D,int H,int W,int oD,int oH,int oW){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oD*oH*oW) return;
    int ow=i%oW,oh=(i/oW)%oH,od=(i/(oW*oH))%oD; int64_t nc=i/((int64_t)oW*oH*oD);
    int ds=od*D/oD,de=((od+1)*D+oD-1)/oD,hs=oh*H/oH,he=((oh+1)*H+oH-1)/oH,ws=ow*W/oW,we=((ow+1)*W+oW-1)/oW;
    float v=go[i]/(float)((de-ds)*(he-hs)*(we-ws)); float*p=gi+nc*D*H*W;
    for(int d=ds;d<de;d++)for(int h=hs;h<he;h++)for(int w=ws;w<we;w++) atomicAdd(&p[((int64_t)d*H+h)*W+w], v);
}
// bilinear sample with accumulate weights into a callback
__device__ inline float bilin(const float*p,int H,int W,float fy,float fx){
    if(fy<-1||fy>H||fx<-1||fx>W) return 0; fy=fmaxf(fy,0);fx=fmaxf(fx,0);
    int y0=(int)fy,x0=(int)fx,y1=min(y0+1,H-1),x1=min(x0+1,W-1); float dy=fy-y0,dx=fx-x0;
    return (1-dy)*(1-dx)*p[y0*W+x0]+(1-dy)*dx*p[y0*W+x1]+dy*(1-dx)*p[y1*W+x0]+dy*dx*p[y1*W+x1];
}
__device__ inline void bilin_scatter(float*p,int H,int W,float fy,float fx,float g){
    if(fy<-1||fy>H||fx<-1||fx>W) return; fy=fmaxf(fy,0);fx=fmaxf(fx,0);
    int y0=(int)fy,x0=(int)fx,y1=min(y0+1,H-1),x1=min(x0+1,W-1); float dy=fy-y0,dx=fx-x0;
    atomicAdd(&p[y0*W+x0],(1-dy)*(1-dx)*g); atomicAdd(&p[y0*W+x1],(1-dy)*dx*g);
    atomicAdd(&p[y1*W+x0],dy*(1-dx)*g);     atomicAdd(&p[y1*W+x1],dy*dx*g);
}
// RoiAlignRotated fwd: rois[K,6]={batch,cx,cy,w,h,theta(rad)}
__global__ void k_roialign_rot(const float*in,const float*rois,float*o,int C,int H,int W,int K,int ph,int pw,float scale,int ratio){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)K*C*ph*pw) return;
    int px=i%pw,py=(i/pw)%ph,c=(i/(pw*ph))%C; int64_t k=i/((int64_t)pw*ph*C);
    const float*r=rois+k*6; int b=(int)r[0]; float cx=r[1]*scale,cy=r[2]*scale,rw=fmaxf(r[3]*scale,1.f),rh=fmaxf(r[4]*scale,1.f),th=r[5];
    float ct=cosf(th),st=sinf(th); float bw=rw/pw,bh=rh/ph; const float*p=in+((int64_t)b*C+c)*H*W;
    float x0=-rw/2.f, y0=-rh/2.f; double s=0; int cnt=ratio*ratio;
    for(int iy=0;iy<ratio;iy++)for(int ix=0;ix<ratio;ix++){
        float ly=y0+py*bh+(iy+0.5f)*bh/ratio, lx=x0+px*bw+(ix+0.5f)*bw/ratio;
        float gx=cx+lx*ct-ly*st, gy=cy+lx*st+ly*ct; s+=bilin(p,H,W,gy,gx); }
    o[i]=(float)(s/cnt);
}
__global__ void k_roialign_rot_bwd(const float*go,const float*rois,float*gi,int C,int H,int W,int K,int ph,int pw,float scale,int ratio){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)K*C*ph*pw) return;
    int px=i%pw,py=(i/pw)%ph,c=(i/(pw*ph))%C; int64_t k=i/((int64_t)pw*ph*C);
    const float*r=rois+k*6; int b=(int)r[0]; float cx=r[1]*scale,cy=r[2]*scale,rw=fmaxf(r[3]*scale,1.f),rh=fmaxf(r[4]*scale,1.f),th=r[5];
    float ct=cosf(th),st=sinf(th); float bw=rw/pw,bh=rh/ph; float*p=gi+((int64_t)b*C+c)*H*W;
    float x0=-rw/2.f, y0=-rh/2.f; int cnt=ratio*ratio; float g=go[i]/cnt;
    for(int iy=0;iy<ratio;iy++)for(int ix=0;ix<ratio;ix++){
        float ly=y0+py*bh+(iy+0.5f)*bh/ratio, lx=x0+px*bw+(ix+0.5f)*bw/ratio;
        float gx=cx+lx*ct-ly*st, gy=cy+lx*st+ly*ct; bilin_scatter(p,H,W,gy,gx,g); }
}
// RoiAlign (axis-aligned) backward: rois[K,5]={batch,x1,y1,x2,y2}
__global__ void k_roialign_bwd(const float*go,const float*rois,float*gi,int C,int H,int W,int K,int ph,int pw,float scale,int ratio){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)K*C*ph*pw) return;
    int px=i%pw,py=(i/pw)%ph,c=(i/(pw*ph))%C; int64_t k=i/((int64_t)pw*ph*C);
    const float*r=rois+k*5; int b=(int)r[0]; float x1=r[1]*scale,y1=r[2]*scale,x2=r[3]*scale,y2=r[4]*scale;
    float rw=fmaxf(x2-x1,1.f),rh=fmaxf(y2-y1,1.f),bw=rw/pw,bh=rh/ph; float*p=gi+((int64_t)b*C+c)*H*W;
    int cnt=ratio*ratio; float g=go[i]/cnt;
    for(int iy=0;iy<ratio;iy++)for(int ix=0;ix<ratio;ix++){ float yy=y1+py*bh+(iy+0.5f)*bh/ratio, xx=x1+px*bw+(ix+0.5f)*bw/ratio; bilin_scatter(p,H,W,yy,xx,g); }
}
// RoiPooling (max) fwd: rois[K,5]={batch,x1,y1,x2,y2}(scaled), integer bins → max + argmax(flat in H*W)
__global__ void k_roipool(const float*in,const float*rois,float*o,int64_t*argmax,int C,int H,int W,int K,int ph,int pw,float scale){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)K*C*ph*pw) return;
    int px=i%pw,py=(i/pw)%ph,c=(i/(pw*ph))%C; int64_t k=i/((int64_t)pw*ph*C);
    const float*r=rois+k*5; int b=(int)r[0]; int x1=(int)roundf(r[1]*scale),y1=(int)roundf(r[2]*scale),x2=(int)roundf(r[3]*scale),y2=(int)roundf(r[4]*scale);
    int rw=max(x2-x1+1,1),rh=max(y2-y1+1,1); float bw=(float)rw/pw,bh=(float)rh/ph;
    int hs=y1+(int)floorf(py*bh),he=y1+(int)ceilf((py+1)*bh),ws=x1+(int)floorf(px*bw),we=x1+(int)ceilf((px+1)*bw);
    hs=min(max(hs,0),H);he=min(max(he,0),H);ws=min(max(ws,0),W);we=min(max(we,0),W);
    const float*p=in+((int64_t)b*C+c)*H*W; float best=-1e30f; int64_t bidx=-1;
    for(int h=hs;h<he;h++)for(int w=ws;w<we;w++){ float v=p[h*W+w]; if(v>best){best=v;bidx=(int64_t)h*W+w;} }
    if(bidx<0){best=0;bidx=0;} o[i]=best; argmax[i]=bidx;
}
__global__ void k_roipool_bwd(const float*go,const float*rois,const int64_t*argmax,float*gi,int C,int H,int W,int K,int ph,int pw){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)K*C*ph*pw) return;
    int c=(i/(pw*ph))%C; int64_t k=i/((int64_t)pw*ph*C); const float*r=rois+k*5; int b=(int)r[0];
    int64_t fi=argmax[i]; if(fi<0) return; atomicAdd(&gi[(((int64_t)b*C+c)*H*W)+fi], go[i]);
}
} // namespace

extern "C" {

// ---- MaxPool generic: forward to MaxPool2d (drops ceilMode/dilation) ----
aclnnStatus aclnnMaxPoolGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)dilation;(void)ceilMode; return aclnnMaxPool2dGetWorkspaceSize(self, kernel, stride, padding, out, ws, ex);
}
aclnnStatus aclnnMaxPool(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMaxPool2d(w,wz,e,s); }

// ---- MaxPool3dWithArgmax (forward) ----
aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex){
    (void)dilation;(void)ceilMode;
    if(!self||!kernel||!out||!indices||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=5||indices->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->out2=indices;
    e->axes.assign(9,0); for(int i=0;i<3;i++){ e->axes[i]=kernel->v[i]; e->axes[3+i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:kernel->v[i]; e->axes[6+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxPool3dWithArgmax(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&xi=e->a->viewDims,&yi=e->out->viewDims; int64_t NC=xi[0]*xi[1]; auto st=(cudaStream_t)s;
    k_maxpool3d_argmax<<<nb(NC*yi[2]*yi[3]*yi[4]),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,(int64_t*)e->out2->data,NC,
        (int)xi[2],(int)xi[3],(int)xi[4],(int)yi[2],(int)yi[3],(int)yi[4],
        (int)e->axes[0],(int)e->axes[1],(int)e->axes[2],(int)e->axes[3],(int)e->axes[4],(int)e->axes[5],(int)e->axes[6],(int)e->axes[7],(int)e->axes[8]);
    return done(e);
}

// ---- AvgPool3dBackward ----
aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!kernel||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=5) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradOutput; e->out=gradInput;
    e->axes.assign(9,0); for(int i=0;i<3;i++){ e->axes[i]=kernel->v[i]; e->axes[3+i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:kernel->v[i]; e->axes[6+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAvgPool3dBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&gi=e->out->viewDims,&go=e->a->viewDims; int64_t NC=gi[0]*gi[1]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_avgpool3d_bwd<<<nb(NC*go[2]*go[3]*go[4]),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,NC,
        (int)gi[2],(int)gi[3],(int)gi[4],(int)go[2],(int)go[3],(int)go[4],
        (int)e->axes[0],(int)e->axes[1],(int)e->axes[2],(int)e->axes[3],(int)e->axes[4],(int)e->axes[5],(int)e->axes[6],(int)e->axes[7],(int)e->axes[8]);
    return done(e);
}
// ---- AdaptiveAvgPool3dBackward ----
aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    (void)self; if(!gradOutput||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=5) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradOutput; e->out=gradInput; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveAvgPool3dBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&gi=e->out->viewDims,&go=e->a->viewDims; int64_t NC=gi[0]*gi[1]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_adaptive_avgpool3d_bwd<<<nb(NC*go[2]*go[3]*go[4]),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,NC,
        (int)gi[2],(int)gi[3],(int)gi[4],(int)go[2],(int)go[3],(int)go[4]);
    return done(e);
}

// ---- RoiAlignRotated (fwd) + Grad ----
aclnnStatus aclnnRoiAlignRotatedGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!rois||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4||out->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=rois; e->out=out; e->alpha=spatialScale; e->reduceCount=samplingRatio>0?samplingRatio:2;
    e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->outerCount=self->viewDims[1]; e->k=rois->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiAlignRotated(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->out->viewDims[2],pw=e->out->viewDims[3]; auto st=(cudaStream_t)s;
    k_roialign_rot<<<nb((int64_t)K*C*ph*pw),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,C,H,W,K,ph,pw,(float)e->alpha,(int)e->reduceCount);
    return done(e);
}
aclnnStatus aclnnRoiAlignRotatedGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!rois||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradOutput; e->b=rois; e->out=gradInput; e->alpha=spatialScale; e->reduceCount=samplingRatio>0?samplingRatio:2;
    e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->outerCount=gradInput->viewDims[1]; e->k=rois->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiAlignRotatedGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->a->viewDims[2],pw=e->a->viewDims[3]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_roialign_rot_bwd<<<nb((int64_t)K*C*ph*pw),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,C,H,W,K,ph,pw,(float)e->alpha,(int)e->reduceCount);
    return done(e);
}
// ---- RoiAlignV2Backward (axis-aligned RoiAlign backward) ----
aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!rois||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradOutput; e->b=rois; e->out=gradInput; e->alpha=spatialScale; e->reduceCount=samplingRatio>0?samplingRatio:2;
    e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->outerCount=gradInput->viewDims[1]; e->k=rois->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiAlignV2Backward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->a->viewDims[2],pw=e->a->viewDims[3]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_roialign_bwd<<<nb((int64_t)K*C*ph*pw),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,C,H,W,K,ph,pw,(float)e->alpha,(int)e->reduceCount);
    return done(e);
}
// ---- RoiPoolingWithArgMax (fwd) + Grad ----
aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, aclTensor *out, aclTensor *argmax, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!rois||!out||!argmax||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4||argmax->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=rois; e->out=out; e->out2=argmax; e->alpha=spatialScale;
    e->m=self->viewDims[2]; e->n=self->viewDims[3]; e->outerCount=self->viewDims[1]; e->k=rois->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiPoolingWithArgMax(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->out->viewDims[2],pw=e->out->viewDims[3]; auto st=(cudaStream_t)s;
    k_roipool<<<nb((int64_t)K*C*ph*pw),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,(int64_t*)e->out2->data,C,H,W,K,ph,pw,(float)e->alpha);
    return done(e);
}
aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, const aclTensor *argmax, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!rois||!argmax||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradOutput; e->b=rois; e->c=argmax; e->out=gradInput;
    e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->outerCount=gradInput->viewDims[1]; e->k=rois->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoiPoolingGradWithArgMax(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int C=e->outerCount,H=e->m,W=e->n,K=e->k,ph=e->a->viewDims[2],pw=e->a->viewDims[3]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_roipool_bwd<<<nb((int64_t)K*C*ph*pw),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const int64_t*)e->c->data,(float*)e->out->data,C,H,W,K,ph,pw);
    return done(e);
}

} // extern "C"
} // namespace _pool4_ext

namespace _pool_bwd_ext {
// Pool/unpool backward + a couple forwards via generic index scatter/gather (fp32, int64 indices).
//   MaxPool*Backward: scatter-add gradOut into gradIn at argmax flat indices.
//   MaxUnpool*: forward scatters self→out at indices; backward gathers gradOut[idx]→gradIn.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
// scatter: out[nc*outSp + idx[nc*srcSp+i]] (+)= src[nc*srcSp+i]  (add=1 backward, add=0 unpool forward)
__global__ void k_scatter(const float *src, const int64_t *idx, float *out, int64_t NC, int64_t srcSp, int64_t outSp, int add) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (p>=NC*srcSp) return; int64_t nc=p/srcSp; int64_t t=idx[p];
    if (t<0||t>=outSp) return; float *o=&out[nc*outSp+t];
    if (add) atomicAdd(o, src[p]); else *o = src[p];
}
// gather: out[nc*outSp+i] = src[nc*srcSp + idx[nc*outSp+i]]
__global__ void k_gather(const float *src, const int64_t *idx, float *out, int64_t NC, int64_t outSp, int64_t srcSp) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (p>=NC*outSp) return; int64_t nc=p/outSp; int64_t t=idx[p];
    out[p] = (t>=0&&t<srcSp) ? src[nc*srcSp+t] : 0.f;
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
inline int64_t NCof(const aclTensor *t){ return t->viewDims.size()>=2 ? t->viewDims[0]*t->viewDims[1] : t->viewDims[0]; }

// Maxpool backward: gradOut + indices (pooled shape) → gradIn (input shape) scatter-add. e->a=gradOut, e->b=indices, e->out=gradIn
static aclnnStatus mpb_ws(const aclTensor *gradOut, const aclTensor *indices, aclTensor *gradIn, aclOpExecutor **ex) {
    if (!gradOut || !indices || !gradIn || !ex || gradOut->dtype != ACL_FLOAT || gradIn->dtype != ACL_FLOAT || indices->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=indices; e->out=gradIn; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus mpb_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t NC=NCof(e->out), isp=e->out->numel()/NC, osp=e->a->numel()/NC;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),s);
    k_scatter<<<nb(NC*osp),TH,0,s>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,NC,osp,isp,1);
    return fin(e);
}
// Unpool backward: gather. e->a=gradOut(big), e->b=indices(small), e->out=gradIn(small)
static aclnnStatus upb_ws(const aclTensor *gradOut, const aclTensor *indices, aclTensor *gradIn, aclOpExecutor **ex) {
    if (!gradOut || !indices || !gradIn || !ex || gradOut->dtype != ACL_FLOAT || indices->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=indices; e->out=gradIn; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus upb_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t NC=NCof(e->out), osp=e->out->numel()/NC, srcSp=e->a->numel()/NC;
    k_gather<<<nb(NC*osp),TH,0,s>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,NC,osp,srcSp);
    return fin(e);
}
} // namespace

extern "C" {

// MaxPool2dWithIndicesBackward / MaxPool3dWithArgmaxBackward / MaxPool2dWithMaskBackward: scatter-add by indices
aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self;(void)kernel;(void)stride;(void)padding;(void)dilation;(void)ceilMode; if(ws)*ws=0; return mpb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnMaxPool2dWithIndicesBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return mpb_run(e,(cudaStream_t)s); }
aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self;(void)kernel;(void)stride;(void)padding;(void)dilation;(void)ceilMode; if(ws)*ws=0; return mpb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnMaxPool3dWithArgmaxBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return mpb_run(e,(cudaStream_t)s); }
aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *mask,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self;(void)kernel;(void)stride;(void)padding;(void)dilation;(void)ceilMode; if(ws)*ws=0; return mpb_ws(gradOutput, mask, gradInput, ex);
}
aclnnStatus aclnnMaxPool2dWithMaskBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return mpb_run(e,(cudaStream_t)s); }
// AdaptiveMaxPool2d/3dBackward: same scatter (gradOutput, self, indices, gradInput)
aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; if(ws)*ws=0; return mpb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnAdaptiveMaxPool2dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return mpb_run(e,(cudaStream_t)s); }
aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; if(ws)*ws=0; return mpb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnAdaptiveMaxPool3dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return mpb_run(e,(cudaStream_t)s); }

// MaxUnpool2d/3dBackward: gather gradOut[idx] → gradIn
aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self;(void)H;(void)W; if(ws)*ws=0; return upb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnMaxUnpool2dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return upb_run(e,(cudaStream_t)s); }
aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self;(void)outputSize; if(ws)*ws=0; return upb_ws(gradOutput, indices, gradInput, ex);
}
aclnnStatus aclnnMaxUnpool3dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s){ return upb_run(e,(cudaStream_t)s); }

// MaxUnpool3d (forward): scatter self → out (zeros elsewhere) at flat indices. out shape from outputSize / out tensor.
aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)outputSize;
    if (!self || !indices || !out || !ex || self->dtype != ACL_FLOAT || indices->dtype != ACL_INT64 || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->b=indices; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxUnpool3d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t NC=NCof(e->out), srcSp=e->a->numel()/NC, outSp=e->out->numel()/NC;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_scatter<<<nb(NC*srcSp),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,NC,srcSp,outSp,0);
    return fin(e);
}

// MaxPool2dWithMask (forward) → MaxPool2dWithIndices (mask == indices)
aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
        const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    (void)dilation;(void)ceilMode; return aclnnMaxPool2dWithIndicesGetWorkspaceSize(self, kernel, stride, padding, out, mask, ws, ex);
}
aclnnStatus aclnnMaxPool2dWithMask(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s){ return aclnnMaxPool2dWithIndices(w, wz, e, s); }

} // extern "C"
} // namespace _pool_bwd_ext

