// Convolution / pooling / quant-&-grouped-matmul family for test_matmul_conv.
// All host-side over unified memory (device ptr == host ptr; tensor data is readable/writable as C++
// arrays). Naive direct loops (test sizes are small) for conv/pool; Accelerate cblas for the fp32 GEMM
// building blocks inside the quant/grouped matmuls. Generic dtype read/write helpers cover the many
// element types the test exercises (fp32/fp16/bf16/int8/int4/E8M0 + subfp fp4/fp6/fp8 via subfp.h).
#import "../internal.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>   // native fp32 GEMM for conv2d forward (im2col)
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {

void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ---- host half / bf16 codecs (bit-identical to the test's f2h/h2f, f2bf/bf2f) ----
inline float h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; std::memcpy(&x, &f, 4); x |= sign; }
    else if (e == 31) x = sign | 0x7F800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
inline uint16_t f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t r = m & 0x1FFF, h = sign | (e << 10) | (m >> 13);
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}
inline float bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }
inline uint16_t f2bf(float f) { uint32_t b; std::memcpy(&b, &f, 4); b += 0x7FFF + ((b >> 16) & 1); return (uint16_t)(b >> 16); }

// Read element i of tensor t (honoring t->offset) as float, covering every dtype the test uses.
inline float rd(const aclTensor *t, int64_t i) {
    const void *base = t->data;
    int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:    return ((const float *)base)[off + i];
        case ACL_FLOAT16:  return h2f(((const uint16_t *)base)[off + i]);
        case ACL_BF16:     return bf2f(((const uint16_t *)base)[off + i]);
        case ACL_INT8:     return (float)((const int8_t *)base)[off + i];
        case ACL_INT32:    return (float)((const int32_t *)base)[off + i];
        case ACL_INT64:    return (float)((const int64_t *)base)[off + i];
        default: break;
    }
    // sub-byte float family (fp4 / fp6 / fp8 / hif8). fp4 is nibble-packed.
    if (subfp::is_low(t->dtype))
        return subfp::load(t->dtype, (const uint8_t *)base + off, i);
    return 0.f;
}
// Write float v into element i of tensor t (covers output dtypes the test reads back).
inline void wr(aclTensor *t, int64_t i, float v) {
    void *base = t->data; int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:    ((float *)base)[off + i] = v; return;
        case ACL_FLOAT16:  ((uint16_t *)base)[off + i] = f2h(v); return;
        case ACL_BF16:     ((uint16_t *)base)[off + i] = f2bf(v); return;
        case ACL_INT8:     ((int8_t *)base)[off + i] = (int8_t)std::lrint(v); return;
        case ACL_INT32:    ((int32_t *)base)[off + i] = (int32_t)std::lrint(v); return;
        case ACL_INT64:    ((int64_t *)base)[off + i] = (int64_t)std::llrint(v); return;
        default: return;
    }
}
// Read signed int4 nibble n (range [-8,7]) from a packed buffer (lo nibble first).
inline int rd_i4(const uint8_t *base, int64_t n) {
    uint8_t b = base[n / 2]; uint8_t nib = (n & 1) ? (b >> 4) : (b & 0xF);
    return (nib & 0x8) ? (int)nib - 16 : (int)nib;
}
// E8M0 scale byte -> 2^(byte-127)
inline double e8m0(uint8_t b) { return std::ldexp(1.0, (int)b - 127); }

inline int64_t IA(const aclIntArray *a, int i, int64_t def) { return (a && (int)a->v.size() > i) ? a->v[i] : def; }

// ===================================================================================================
// Op dispatch. Each GetWorkspaceSize stashes inputs/params on the executor; Execute runs the host loop.
// ===================================================================================================
enum CKind {
    K_CONV = 1, K_CONV3D, K_CONVT2D, K_CONV_BWD_DATA, K_CONV_BWD_WEIGHT, K_CONV_BWD, K_CONVTBC,
    K_DEFORMCONV, K_IM2COL, K_CALC_WSIZE, K_TOINT4PACK,
    K_AVGPOOL2D, K_MAXPOOL2D, K_AVGPOOL3D, K_MAXPOOL3D, K_ADAPTAVG2D, K_AVGPOOL2D_BWD, K_MAXPOOL2D_BWD,
    K_QMM, K_WQMM, K_BMMQUANT, K_QMM_IADD, K_GMM, K_TBMM, K_TQBMM, K_MXFP, // K_MXFP shared fp8/fp4 MX
};

// conv params packed into stride[2]/pad[2]/dil[2]; bias in c; groups in k; transposed flag in n;
// scale tensor (quant) in b's sibling stored via inputs.

// ---------- forward 2d convolution (NCHW, groups) ----------
void conv2d_fwd(const aclTensor *X, const aclTensor *W, const aclTensor *B, aclTensor *Y,
                int64_t sh, int64_t sw, int64_t ph, int64_t pw, int64_t dh, int64_t dw, int64_t groups) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Co = W->viewDims[0], Cpg = W->viewDims[1], R = W->viewDims[2], S = W->viewDims[3];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    int64_t Cog = Co / groups;
    for (int64_t n = 0; n < N; ++n)
      for (int64_t co = 0; co < Co; ++co) {
        int64_t g = co / Cog;
        for (int64_t ho = 0; ho < Ho; ++ho)
          for (int64_t wo = 0; wo < Wo; ++wo) {
            double acc = B ? rd(B, co) : 0.0;
            for (int64_t cc = 0; cc < Cpg; ++cc) {
              int64_t ci = g * Cpg + cc;
              for (int64_t kr = 0; kr < R; ++kr)
                for (int64_t ks = 0; ks < S; ++ks) {
                  int64_t hi = ho * sh - ph + kr * dh, wi = wo * sw - pw + ks * dw;
                  if (hi < 0 || hi >= H || wi < 0 || wi >= Wd) continue;
                  acc += (double)rd(X, ((n * C + ci) * H + hi) * Wd + wi) *
                         rd(W, ((co * Cpg + cc) * R + kr) * S + ks);
                }
            }
            wr(Y, ((n * Co + co) * Ho + ho) * Wo + wo, (float)acc);
          }
      }
}

// forward 2d convolution via im2col + native MPS fp32 GEMM. Per group g: Y_g[Cog,Ho*Wo] = Wg[Cog,Cpg*R*S] @
// col[Cpg*R*S,Ho*Wo], where col is the unfolded input patches (zero-padded). Inputs are read through rd()
// (any dtype) into fp32 temps; the output is written back through wr() (any dtype). Same math as conv2d_fwd
// (cross-correlation, zero padding). Returns false on any setup failure so the caller uses the host loop.
bool conv2d_mps(const aclTensor *X, const aclTensor *W, const aclTensor *B, aclTensor *Y,
                int64_t sh, int64_t sw, int64_t ph, int64_t pw, int64_t dh, int64_t dw, int64_t groups) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Co = W->viewDims[0], Cpg = W->viewDims[1], R = W->viewDims[2], S = W->viewDims[3];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    int64_t Cog = Co / groups, Kd = Cpg * R * S, HW = Ho * Wo;
    if (Kd <= 0 || HW <= 0 || Cog <= 0) return false;
    float *col = (float *)mtl::alloc((size_t)Kd * HW * 4);
    float *wf = (float *)mtl::alloc((size_t)Cog * Kd * 4);
    float *yt = (float *)mtl::alloc((size_t)Cog * HW * 4);
    if (!col || !wf || !yt) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
    size_t ocol, owf, oyt;
    id<MTLBuffer> bcol = mtl::bufferFor(col, &ocol), bwf = mtl::bufferFor(wf, &owf), byt = mtl::bufferFor(yt, &oyt);
    id<MTLDevice> dev = mtl::device();
    if (!bcol || !bwf || !byt || !dev) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
    MPSMatrixDescriptor *dWd = [MPSMatrixDescriptor matrixDescriptorWithRows:Cog columns:Kd rowBytes:Kd * 4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dCd = [MPSMatrixDescriptor matrixDescriptorWithRows:Kd columns:HW rowBytes:HW * 4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dOd = [MPSMatrixDescriptor matrixDescriptorWithRows:Cog columns:HW rowBytes:HW * 4 dataType:MPSDataTypeFloat32];
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO
                                       transposeRight:NO resultRows:Cog resultColumns:HW interiorColumns:Kd alpha:1.0 beta:0.0];
    MPSMatrix *mWf = [[MPSMatrix alloc] initWithBuffer:bwf offset:owf descriptor:dWd];
    MPSMatrix *mCol = [[MPSMatrix alloc] initWithBuffer:bcol offset:ocol descriptor:dCd];
    MPSMatrix *mYt = [[MPSMatrix alloc] initWithBuffer:byt offset:oyt descriptor:dOd];
    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t co = 0; co < Cog; ++co) for (int64_t kk = 0; kk < Kd; ++kk) wf[co * Kd + kk] = rd(W, (g * Cog + co) * Kd + kk);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cc = 0; cc < Cpg; ++cc) { int64_t ci = g * Cpg + cc;
                for (int64_t kr = 0; kr < R; ++kr) for (int64_t ks = 0; ks < S; ++ks) { int64_t row = (cc * R + kr) * S + ks;
                    for (int64_t ho = 0; ho < Ho; ++ho) for (int64_t wo = 0; wo < Wo; ++wo) {
                        int64_t hi = ho * sh - ph + kr * dh, wi = wo * sw - pw + ks * dw;
                        col[row * HW + ho * Wo + wo] = (hi < 0 || hi >= H || wi < 0 || wi >= Wd) ? 0.f : rd(X, ((n * C + ci) * H + hi) * Wd + wi);
                    }
                }
            }
            id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:mWf rightMatrix:mCol resultMatrix:mYt];
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
            for (int64_t co = 0; co < Cog; ++co) {
                double bias = B ? rd(B, g * Cog + co) : 0.0;
                int64_t ybase = ((n * Co + g * Cog + co) * Ho) * Wo;
                for (int64_t hw = 0; hw < HW; ++hw) wr(Y, ybase + hw, (float)((double)yt[co * HW + hw] + bias));
            }
        }
    }
    mtl::free_(col); mtl::free_(wf); mtl::free_(yt);
    return true;
}

void conv3d_fwd(const aclTensor *X, const aclTensor *W, aclTensor *Y,
                int64_t s, int64_t p, int64_t d, int64_t groups) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], D = X->viewDims[2], H = X->viewDims[3], Wd = X->viewDims[4];
    int64_t Co = W->viewDims[0], Cpg = W->viewDims[1], K = W->viewDims[2];
    int64_t Do = Y->viewDims[2], Ho = Y->viewDims[3], Wo = Y->viewDims[4];
    int64_t Cog = Co / groups;
    for (int64_t n = 0; n < N; ++n)
      for (int64_t co = 0; co < Co; ++co) {
        int64_t g = co / Cog;
        for (int64_t od = 0; od < Do; ++od)
         for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            double acc = 0;
            for (int64_t cc = 0; cc < Cpg; ++cc) {
              int64_t ci = g * Cpg + cc;
              for (int64_t kd = 0; kd < K; ++kd)
               for (int64_t kh = 0; kh < K; ++kh)
                for (int64_t kw = 0; kw < K; ++kw) {
                  int64_t id = od * s - p + kd * d, ih = oh * s - p + kh * d, iw = ow * s - p + kw * d;
                  if (id < 0 || id >= D || ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
                  acc += (double)rd(X, (((n * C + ci) * D + id) * H + ih) * Wd + iw) *
                         rd(W, (((co * Cpg + cc) * K + kd) * K + kh) * K + kw);
                }
            }
            wr(Y, (((n * Co + co) * Do + od) * Ho + oh) * Wo + ow, (float)acc);
          }
      }
}

// forward 3d convolution via im2col + native MPS fp32 GEMM (same structure as conv2d_mps, cubic K window).
bool conv3d_mps(const aclTensor *X, const aclTensor *W, aclTensor *Y, int64_t s, int64_t p, int64_t dl, int64_t groups) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], Dp = X->viewDims[2], H = X->viewDims[3], Wd = X->viewDims[4];
    int64_t Co = W->viewDims[0], Cpg = W->viewDims[1], K = W->viewDims[2];
    int64_t Do = Y->viewDims[2], Ho = Y->viewDims[3], Wo = Y->viewDims[4];
    int64_t Cog = Co / groups, Kd = Cpg * K * K * K, HW = Do * Ho * Wo;
    if (Kd <= 0 || HW <= 0 || Cog <= 0) return false;
    float *col = (float *)mtl::alloc((size_t)Kd * HW * 4), *wf = (float *)mtl::alloc((size_t)Cog * Kd * 4), *yt = (float *)mtl::alloc((size_t)Cog * HW * 4);
    if (!col || !wf || !yt) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
    size_t ocol, owf, oyt; id<MTLBuffer> bcol = mtl::bufferFor(col, &ocol), bwf = mtl::bufferFor(wf, &owf), byt = mtl::bufferFor(yt, &oyt);
    id<MTLDevice> dev = mtl::device();
    if (!bcol || !bwf || !byt || !dev) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
    MPSMatrixDescriptor *dWd = [MPSMatrixDescriptor matrixDescriptorWithRows:Cog columns:Kd rowBytes:Kd*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dCd = [MPSMatrixDescriptor matrixDescriptorWithRows:Kd columns:HW rowBytes:HW*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dOd = [MPSMatrixDescriptor matrixDescriptorWithRows:Cog columns:HW rowBytes:HW*4 dataType:MPSDataTypeFloat32];
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:Cog resultColumns:HW interiorColumns:Kd alpha:1.0 beta:0.0];
    MPSMatrix *mWf = [[MPSMatrix alloc] initWithBuffer:bwf offset:owf descriptor:dWd], *mCol = [[MPSMatrix alloc] initWithBuffer:bcol offset:ocol descriptor:dCd], *mYt = [[MPSMatrix alloc] initWithBuffer:byt offset:oyt descriptor:dOd];
    for (int64_t g = 0; g < groups; ++g) {
        for (int64_t co = 0; co < Cog; ++co) for (int64_t kk = 0; kk < Kd; ++kk) wf[co * Kd + kk] = rd(W, (g * Cog + co) * Kd + kk);
        for (int64_t n = 0; n < N; ++n) {
            for (int64_t cc = 0; cc < Cpg; ++cc) { int64_t ci = g * Cpg + cc;
                for (int64_t kd = 0; kd < K; ++kd) for (int64_t kh = 0; kh < K; ++kh) for (int64_t kw = 0; kw < K; ++kw) {
                    int64_t row = ((cc * K + kd) * K + kh) * K + kw;
                    for (int64_t od = 0; od < Do; ++od) for (int64_t oh = 0; oh < Ho; ++oh) for (int64_t ow = 0; ow < Wo; ++ow) {
                        int64_t id = od * s - p + kd * dl, ih = oh * s - p + kh * dl, iw = ow * s - p + kw * dl;
                        int64_t oidx = (od * Ho + oh) * Wo + ow;
                        col[row * HW + oidx] = (id < 0 || id >= Dp || ih < 0 || ih >= H || iw < 0 || iw >= Wd) ? 0.f : rd(X, (((n * C + ci) * Dp + id) * H + ih) * Wd + iw);
                    }
                }
            }
            id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:mWf rightMatrix:mCol resultMatrix:mYt];
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { mtl::free_(col); mtl::free_(wf); mtl::free_(yt); return false; }
            for (int64_t co = 0; co < Cog; ++co) { int64_t ybase = ((n * Co + g * Cog + co) * Do) * Ho * Wo; for (int64_t hw = 0; hw < HW; ++hw) wr(Y, ybase + hw, yt[co * HW + hw]); }
        }
    }
    mtl::free_(col); mtl::free_(wf); mtl::free_(yt);
    return true;
}

// weight layout [Ci,Co,K,K] (transpose conv). out = scatter-add.
void convt2d(const aclTensor *X, const aclTensor *W, aclTensor *Y,
             int64_t s, int64_t p, int64_t /*d*/, int64_t /*groups*/) {
    int64_t N = X->viewDims[0], Ci = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Co = W->viewDims[1], K = W->viewDims[2];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    int64_t yn = Y->numel();
    for (int64_t i = 0; i < yn; ++i) wr(Y, i, 0.f);
    std::vector<double> out(yn, 0.0);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t ci = 0; ci < Ci; ++ci)
        for (int64_t h = 0; h < H; ++h)
          for (int64_t w = 0; w < Wd; ++w) {
            double v = rd(X, ((n * Ci + ci) * H + h) * Wd + w);
            for (int64_t co = 0; co < Co; ++co)
              for (int64_t kh = 0; kh < K; ++kh)
                for (int64_t kw = 0; kw < K; ++kw) {
                  int64_t oh = h * s - p + kh, ow = w * s - p + kw;
                  if (oh < 0 || oh >= Ho || ow < 0 || ow >= Wo) continue;
                  out[((n * Co + co) * Ho + oh) * Wo + ow] +=
                      v * rd(W, ((ci * Co + co) * K + kh) * K + kw);
                }
          }
    for (int64_t i = 0; i < yn; ++i) wr(Y, i, (float)out[i]);
}

// dgrad: gradInput from gradOutput & weight (NCHW, weight [Co,C,KH,KW])
void conv_bwd_data(const aclTensor *dY, const aclTensor *W, aclTensor *dX,
                   int64_t s, int64_t p, int64_t d) {
    int64_t N = dX->viewDims[0], C = dX->viewDims[1], H = dX->viewDims[2], Wd = dX->viewDims[3];
    int64_t Co = W->viewDims[0], KH = W->viewDims[2], KW = W->viewDims[3];
    int64_t Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    int64_t xn = dX->numel();
    std::vector<double> g(xn, 0.0);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t co = 0; co < Co; ++co)
        for (int64_t ho = 0; ho < Ho; ++ho)
          for (int64_t wo = 0; wo < Wo; ++wo) {
            double go = rd(dY, ((n * Co + co) * Ho + ho) * Wo + wo);
            for (int64_t ci = 0; ci < C; ++ci)
              for (int64_t r = 0; r < KH; ++r)
                for (int64_t ss = 0; ss < KW; ++ss) {
                  int64_t h = ho * s - p + r * d, w = wo * s - p + ss * d;
                  if (h < 0 || h >= H || w < 0 || w >= Wd) continue;
                  g[((n * C + ci) * H + h) * Wd + w] += go * rd(W, ((co * C + ci) * KH + r) * KW + ss);
                }
          }
    for (int64_t i = 0; i < xn; ++i) wr(dX, i, (float)g[i]);
}

// wgrad: gradWeight from input & gradOutput
void conv_bwd_weight(const aclTensor *X, const aclTensor *dY, aclTensor *dW,
                     int64_t s, int64_t p, int64_t d) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Co = dW->viewDims[0], KH = dW->viewDims[2], KW = dW->viewDims[3];
    int64_t Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    for (int64_t co = 0; co < Co; ++co)
      for (int64_t ci = 0; ci < C; ++ci)
        for (int64_t r = 0; r < KH; ++r)
          for (int64_t ss = 0; ss < KW; ++ss) {
            double acc = 0;
            for (int64_t n = 0; n < N; ++n)
              for (int64_t ho = 0; ho < Ho; ++ho)
                for (int64_t wo = 0; wo < Wo; ++wo) {
                  int64_t h = ho * s - p + r * d, w = wo * s - p + ss * d;
                  if (h < 0 || h >= H || w < 0 || w >= Wd) continue;
                  acc += rd(dY, ((n * Co + co) * Ho + ho) * Wo + wo) *
                         rd(X, ((n * C + ci) * H + h) * Wd + w);
                }
            wr(dW, ((co * C + ci) * KH + r) * KW + ss, (float)acc);
          }
}

// ---------- pooling 2d ----------
void pool2d(const aclTensor *X, aclTensor *Y, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
            int64_t ph, int64_t pw, bool avg) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t ho = 0; ho < Ho; ++ho)
          for (int64_t wo = 0; wo < Wo; ++wo) {
            double mx = -1e30, sum = 0; int cnt = 0;
            for (int64_t r = 0; r < kh; ++r)
              for (int64_t ss = 0; ss < kw; ++ss) {
                int64_t h = ho * sh - ph + r, w = wo * sw - pw + ss;
                if (h < 0 || h >= H || w < 0 || w >= Wd) continue;
                double v = rd(X, ((n * C + c) * H + h) * Wd + w); mx = std::max(mx, v); sum += v; cnt++;
              }
            wr(Y, ((n * C + c) * Ho + ho) * Wo + wo, (float)(avg ? sum / cnt : mx));
          }
}
// 2d pooling on the GPU via the pool2d_k MSL kernel (fp32/fp16/bf16). Returns false (caller uses host pool2d)
// for other dtypes or setup failure. Matches pool2d (count_exclude_pad average, zero-free max over in-bounds).
struct PoolMetaH { uint32_t N, C, H, Wd, Ho, Wo; int32_t kh, kw, sh, sw, ph, pw, avg; };
bool pool2d_gpu(const aclTensor *X, aclTensor *Y, int64_t kh, int64_t kw, int64_t sh, int64_t sw,
                int64_t ph, int64_t pw, bool avg) {
    const char *suf = X->dtype == ACL_FLOAT ? "f32" : X->dtype == ACL_FLOAT16 ? "f16" : X->dtype == ACL_BF16 ? "bf16" : nullptr;
    if (!suf || X->dtype != Y->dtype) return false;
    id<MTLComputePipelineState> pso = mtl::pipeline([NSString stringWithFormat:@"pool2d_%s", suf]);
    if (!pso) return false;
    size_t ox, oy;
    id<MTLBuffer> bx = mtl::bufferFor(X->data, &ox), by = mtl::bufferFor(Y->data, &oy);
    if (!bx || !by) return false;
    ox += (size_t)X->offset * dtype_size(X->dtype); oy += (size_t)Y->offset * dtype_size(Y->dtype);
    if ((ox | oy) & 3u) return false;
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3], Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    PoolMetaH m{ (uint32_t)N, (uint32_t)C, (uint32_t)H, (uint32_t)Wd, (uint32_t)Ho, (uint32_t)Wo,
                 (int32_t)kh, (int32_t)kw, (int32_t)sh, (int32_t)sw, (int32_t)ph, (int32_t)pw, avg ? 1 : 0 };
    id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:ox atIndex:0]; [enc setBuffer:by offset:oy atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    NSUInteger n = (NSUInteger)(N * C * Ho * Wo), tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.error == nil;
}
// pooling 3d
void pool3d(const aclTensor *X, aclTensor *Y, int64_t k, int64_t s, int64_t p, bool avg) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], D = X->viewDims[2], H = X->viewDims[3], Wd = X->viewDims[4];
    int64_t Do = Y->viewDims[2], Ho = Y->viewDims[3], Wo = Y->viewDims[4];
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t od = 0; od < Do; ++od)
         for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            double mx = -1e30, sum = 0; int cnt = 0;
            for (int64_t kd = 0; kd < k; ++kd)
             for (int64_t kh = 0; kh < k; ++kh)
              for (int64_t kw = 0; kw < k; ++kw) {
                int64_t d = od * s - p + kd, h = oh * s - p + kh, w = ow * s - p + kw;
                if (d < 0 || d >= D || h < 0 || h >= H || w < 0 || w >= Wd) continue;
                double v = rd(X, (((n * C + c) * D + d) * H + h) * Wd + w); mx = std::max(mx, v); sum += v; cnt++;
              }
            wr(Y, (((n * C + c) * Do + od) * Ho + oh) * Wo + ow, (float)(avg ? sum / cnt : mx));
          }
}
void adaptive_avgpool2d(const aclTensor *X, aclTensor *Y) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            int64_t hs = oh * H / Ho, he = ((oh + 1) * H + Ho - 1) / Ho;
            int64_t ws = ow * Wd / Wo, we = ((ow + 1) * Wd + Wo - 1) / Wo;
            double s = 0;
            for (int64_t h = hs; h < he; ++h) for (int64_t w = ws; w < we; ++w) s += rd(X, ((n * C + c) * H + h) * Wd + w);
            wr(Y, ((n * C + c) * Ho + oh) * Wo + ow, (float)(s / ((he - hs) * (we - ws))));
          }
}
// 3D pool / adaptive-avg-pool on the GPU (fp32/fp16/bf16); host fallback otherwise.
struct Pool3MetaH { uint32_t N, C, D, H, Wd, Do, Ho, Wo; int32_t k, s, p, avg; };
struct AAMetaH { uint32_t N, C, H, Wd, Ho, Wo; };
static const char *poolsuf(aclDataType d) { return d == ACL_FLOAT ? "f32" : d == ACL_FLOAT16 ? "f16" : d == ACL_BF16 ? "bf16" : nullptr; }
template <typename Meta>
static bool pool_dispatch(NSString *kn, id<MTLBuffer> bx, size_t ox, id<MTLBuffer> by, size_t oy, const Meta &m, NSUInteger n) {
    id<MTLComputePipelineState> pso = mtl::pipeline(kn); if (!pso) return false;
    id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:ox atIndex:0]; [enc setBuffer:by offset:oy atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.error == nil;
}
static bool pool_bufs(const aclTensor *X, aclTensor *Y, id<MTLBuffer> *bx, size_t *ox, id<MTLBuffer> *by, size_t *oy) {
    if (X->dtype != Y->dtype || !poolsuf(X->dtype)) return false;
    *bx = mtl::bufferFor(X->data, ox); *by = mtl::bufferFor(Y->data, oy);
    if (!*bx || !*by) return false;
    *ox += (size_t)X->offset * dtype_size(X->dtype); *oy += (size_t)Y->offset * dtype_size(Y->dtype);
    return !((*ox | *oy) & 3u);
}
bool pool3d_gpu(const aclTensor *X, aclTensor *Y, int64_t k, int64_t s, int64_t p, bool avg) {
    id<MTLBuffer> bx, by; size_t ox, oy; if (!pool_bufs(X, Y, &bx, &ox, &by, &oy)) return false;
    int64_t N = X->viewDims[0], C = X->viewDims[1], D = X->viewDims[2], H = X->viewDims[3], Wd = X->viewDims[4], Do = Y->viewDims[2], Ho = Y->viewDims[3], Wo = Y->viewDims[4];
    Pool3MetaH m{ (uint32_t)N,(uint32_t)C,(uint32_t)D,(uint32_t)H,(uint32_t)Wd,(uint32_t)Do,(uint32_t)Ho,(uint32_t)Wo,(int32_t)k,(int32_t)s,(int32_t)p,avg?1:0 };
    return pool_dispatch([NSString stringWithFormat:@"pool3d_%s", poolsuf(X->dtype)], bx, ox, by, oy, m, (NSUInteger)(N*C*Do*Ho*Wo));
}
bool adaptive_avg2d_gpu(const aclTensor *X, aclTensor *Y) {
    id<MTLBuffer> bx, by; size_t ox, oy; if (!pool_bufs(X, Y, &bx, &ox, &by, &oy)) return false;
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3], Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    AAMetaH m{ (uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)Wd,(uint32_t)Ho,(uint32_t)Wo };
    return pool_dispatch([NSString stringWithFormat:@"adaptive_avg2d_%s", poolsuf(X->dtype)], bx, ox, by, oy, m, (NSUInteger)(N*C*Ho*Wo));
}
// pooling backward 2d (avg excludes padding; max routes grad to first argmax)
void pool2d_bwd(const aclTensor *X, const aclTensor *dY, aclTensor *dX,
                int64_t k, int64_t s, int64_t p, bool avg) {
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    int64_t xn = dX->numel();
    std::vector<double> g(xn, 0.0);
    for (int64_t n = 0; n < N; ++n)
      for (int64_t c = 0; c < C; ++c)
        for (int64_t oh = 0; oh < Ho; ++oh)
          for (int64_t ow = 0; ow < Wo; ++ow) {
            double go = rd(dY, ((n * C + c) * Ho + oh) * Wo + ow);
            if (avg) {
              int cnt = 0;
              for (int64_t r = 0; r < k; ++r) for (int64_t ss = 0; ss < k; ++ss) { int64_t h = oh * s - p + r, w = ow * s - p + ss; if (h >= 0 && h < H && w >= 0 && w < Wd) cnt++; }
              for (int64_t r = 0; r < k; ++r) for (int64_t ss = 0; ss < k; ++ss) { int64_t h = oh * s - p + r, w = ow * s - p + ss; if (h >= 0 && h < H && w >= 0 && w < Wd) g[((n * C + c) * H + h) * Wd + w] += go / cnt; }
            } else {
              double mx = -1e30; int64_t mh = -1, mw = -1;
              for (int64_t r = 0; r < k; ++r) for (int64_t ss = 0; ss < k; ++ss) { int64_t h = oh * s - p + r, w = ow * s - p + ss; if (h < 0 || h >= H || w < 0 || w >= Wd) continue; double v = rd(X, ((n * C + c) * H + h) * Wd + w); if (v > mx) { mx = v; mh = h; mw = w; } }
              if (mh >= 0) g[((n * C + c) * H + mh) * Wd + mw] += go;
            }
          }
    for (int64_t i = 0; i < xn; ++i) wr(dX, i, (float)g[i]);
}

// ---------- fp32 row-major GEMM helper (used by quant/grouped) : C[M,N] = A[M,K]@B[K,N] ----------
void gemm_f32(const float *A, const float *Bm, float *C, int M, int K, int N) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, Bm, N, 0.0f, C, N);
}

// GPU gather kernels for the backward / transposed convs + pool backward (fp32/fp16; bf16 -> host fallback).
struct CTMetaH { uint32_t N, Ci, H, Wd, Co, K, Ho, Wo; int32_t s, p; };
struct CGMetaH { uint32_t N, C, H, Wd, Co, KH, KW, Ho, Wo; int32_t s, p, d; };
struct PBMetaH { uint32_t N, C, H, Wd, Ho, Wo; int32_t k, s, p, avg; };
static const char *bwsuf(aclDataType d) { return d == ACL_FLOAT ? "f32" : d == ACL_FLOAT16 ? "f16" : nullptr; }
template <typename Meta>
static bool conv3t_dispatch(NSString *kn, const aclTensor *A, const aclTensor *Bt, aclTensor *Ct, const Meta &m, NSUInteger n) {
    if (A->dtype != Ct->dtype || Bt->dtype != Ct->dtype) return false;
    id<MTLComputePipelineState> pso = mtl::pipeline(kn); if (!pso) return false;
    size_t oa, ob, oc; id<MTLBuffer> ba = mtl::bufferFor(A->data, &oa), bb = mtl::bufferFor(Bt->data, &ob), bc = mtl::bufferFor(Ct->data, &oc);
    if (!ba || !bb || !bc) return false;
    oa += (size_t)A->offset * dtype_size(A->dtype); ob += (size_t)Bt->offset * dtype_size(Bt->dtype); oc += (size_t)Ct->offset * dtype_size(Ct->dtype);
    if ((oa | ob | oc) & 3u) return false;
    id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bb offset:ob atIndex:1]; [enc setBuffer:bc offset:oc atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted]; return cb.error == nil;
}
bool convt2d_gpu(const aclTensor *X, const aclTensor *W, aclTensor *Y, int64_t s, int64_t p) {
    if (!bwsuf(Y->dtype)) return false;
    int64_t N = X->viewDims[0], Ci = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3], Co = W->viewDims[1], K = W->viewDims[2], Ho = Y->viewDims[2], Wo = Y->viewDims[3];
    CTMetaH m{ (uint32_t)N,(uint32_t)Ci,(uint32_t)H,(uint32_t)Wd,(uint32_t)Co,(uint32_t)K,(uint32_t)Ho,(uint32_t)Wo,(int32_t)s,(int32_t)p };
    return conv3t_dispatch([NSString stringWithFormat:@"convt2d_%s", bwsuf(Y->dtype)], X, W, Y, m, (NSUInteger)(N*Co*Ho*Wo));
}
bool conv_dgrad_gpu(const aclTensor *dY, const aclTensor *W, aclTensor *dX, int64_t s, int64_t p, int64_t d) {
    if (!bwsuf(dX->dtype)) return false;
    int64_t N = dX->viewDims[0], C = dX->viewDims[1], H = dX->viewDims[2], Wd = dX->viewDims[3], Co = W->viewDims[0], KH = W->viewDims[2], KW = W->viewDims[3], Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    CGMetaH m{ (uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)Wd,(uint32_t)Co,(uint32_t)KH,(uint32_t)KW,(uint32_t)Ho,(uint32_t)Wo,(int32_t)s,(int32_t)p,(int32_t)d };
    return conv3t_dispatch([NSString stringWithFormat:@"conv_dgrad_%s", bwsuf(dX->dtype)], dY, W, dX, m, (NSUInteger)(N*C*H*Wd));
}
bool conv_wgrad_gpu(const aclTensor *X, const aclTensor *dY, aclTensor *dW, int64_t s, int64_t p, int64_t d) {
    if (!bwsuf(dW->dtype)) return false;
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3], Co = dW->viewDims[0], KH = dW->viewDims[2], KW = dW->viewDims[3], Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    CGMetaH m{ (uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)Wd,(uint32_t)Co,(uint32_t)KH,(uint32_t)KW,(uint32_t)Ho,(uint32_t)Wo,(int32_t)s,(int32_t)p,(int32_t)d };
    return conv3t_dispatch([NSString stringWithFormat:@"conv_wgrad_%s", bwsuf(dW->dtype)], X, dY, dW, m, (NSUInteger)(Co*C*KH*KW));
}
bool pool2d_bwd_gpu(const aclTensor *X, const aclTensor *dY, aclTensor *dX, int64_t k, int64_t s, int64_t p, bool avg) {
    if (!bwsuf(dX->dtype)) return false;
    int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3], Ho = dY->viewDims[2], Wo = dY->viewDims[3];
    PBMetaH m{ (uint32_t)N,(uint32_t)C,(uint32_t)H,(uint32_t)Wd,(uint32_t)Ho,(uint32_t)Wo,(int32_t)k,(int32_t)s,(int32_t)p,avg?1:0 };
    return conv3t_dispatch([NSString stringWithFormat:@"pool2d_bwd_%s", bwsuf(dX->dtype)], X, dY, dX, m, (NSUInteger)(N*C*H*Wd));
}

// ===================================================================================================
aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    int64_t s0 = e->stride[0], s1 = e->stride[1];
    int64_t p0 = e->pad[0], p1 = e->pad[1];
    int64_t d0 = e->dil[0], d1 = e->dil[1];
    int64_t groups = e->k ? e->k : 1;

    switch (e->op) {
    case K_CONV: {   // large convs: im2col + native MPS fp32 GEMM; tiny convs / setup failure: host loop
        const aclTensor *W = e->b; aclTensor *Y = e->out;
        int64_t work = (W->viewDims[0] / groups) * (W->viewDims[1] * W->viewDims[2] * W->viewDims[3]) * (Y->viewDims[2] * Y->viewDims[3]);
        if (work < (1 << 15) || !conv2d_mps(e->a, e->b, e->c, e->out, s0, s1, p0, p1, d0, d1, groups))
            conv2d_fwd(e->a, e->b, e->c, e->out, s0, s1, p0, p1, d0, d1, groups);
        break;
    }
    case K_CONV3D: {   // im2col + MPS GEMM for substantial 3D convs; host loop otherwise
        const aclTensor *W = e->b; aclTensor *Y = e->out;
        int64_t K = W->viewDims[2], work = (W->viewDims[0] / groups) * (W->viewDims[1] * K * K * K) * (Y->viewDims[2] * Y->viewDims[3] * Y->viewDims[4]);
        if (work < (1 << 15) || !conv3d_mps(e->a, e->b, e->out, s0, p0, d0, groups))
            conv3d_fwd(e->a, e->b, e->out, s0, p0, d0, groups);
        break;
    }
    case K_CONVT2D: if (groups != 1 || !convt2d_gpu(e->a, e->b, e->out, s0, p0)) convt2d(e->a, e->b, e->out, s0, p0, d0, groups); break;
    case K_CONV_BWD_DATA: if (!conv_dgrad_gpu(e->a, e->b, e->out, s0, p0, d0)) conv_bwd_data(e->a, e->b, e->out, s0, p0, d0); break;
    case K_CONV_BWD_WEIGHT: if (!conv_wgrad_gpu(e->a, e->b, e->out, s0, p0, d0)) conv_bwd_weight(e->a, e->b, e->out, s0, p0, d0); break;
    case K_CONV_BWD: {
        // gradOutput=a, input=b, weight=c; gradInput=out, gradWeight=out2, gradBias=stored in inputs[0]
        if (e->out) conv_bwd_data(e->a, e->c, e->out, s0, p0, d0);
        if (e->out2) conv_bwd_weight(e->b, e->a, e->out2, s0, p0, d0);
        if (!e->inputs.empty() && e->inputs[0]) {            // gradBias[co] = sum gradOut over N,H,W
            aclTensor *gb = (aclTensor *)e->inputs[0]; const aclTensor *dY = e->a;
            int64_t N = dY->viewDims[0], Co = dY->viewDims[1], Ho = dY->viewDims[2], Wo = dY->viewDims[3];
            for (int64_t co = 0; co < Co; ++co) { double acc = 0; for (int64_t n = 0; n < N; ++n) for (int64_t pix = 0; pix < Ho * Wo; ++pix) acc += rd(dY, (n * Co + co) * Ho * Wo + pix); wr(gb, co, (float)acc); }
        }
        break;
    }
    case K_CONVTBC: {
        // self[T,B,Cin], weight[kW,Cin,Cout], bias[Cout], pad -> out[oT,B,Cout]
        const aclTensor *X = e->a, *W = e->b, *B = e->c; aclTensor *Y = e->out;
        int64_t T = X->viewDims[0], Bb = X->viewDims[1], Cin = X->viewDims[2];
        int64_t kW = W->viewDims[0], Cout = W->viewDims[2], oT = Y->viewDims[0]; int64_t pad = p0;
        for (int64_t t = 0; t < oT; ++t)
          for (int64_t b = 0; b < Bb; ++b)
            for (int64_t co = 0; co < Cout; ++co) {
              double acc = B ? rd(B, co) : 0.0;
              for (int64_t k = 0; k < kW; ++k) { int64_t ti = t + k - pad; if (ti < 0 || ti >= T) continue;
                for (int64_t ci = 0; ci < Cin; ++ci) acc += rd(X, (ti * Bb + b) * Cin + ci) * rd(W, (k * Cin + ci) * Cout + co); }
              wr(Y, (t * Bb + b) * Cout + co, (float)acc);
            }
        break;
    }
    case K_DEFORMCONV: {
        // offset==0 reduces to a regular conv (pad/stride/dil already on executor). weight [Cout,Cin,K,K].
        conv2d_fwd(e->a, e->b, nullptr, e->out, s0, s1, p0, p1, d0, d1, 1);
        break;
    }
    case K_IM2COL: {
        // self[N,C,H,W] -> out[N, C*kH*kW, oH*oW]
        const aclTensor *X = e->a; aclTensor *O = e->out;
        int64_t N = X->viewDims[0], C = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
        int64_t kH = e->dscalars[0], kW = e->dscalars[1];
        int64_t oH = (H + 2 * p0 - (d0 * (kH - 1) + 1)) / s0 + 1;
        int64_t oW = (Wd + 2 * p1 - (d1 * (kW - 1) + 1)) / s1 + 1;
        int64_t Kk = C * kH * kW, L = oH * oW;
        for (int64_t n = 0; n < N; ++n)
          for (int64_t c = 0; c < C; ++c)
            for (int64_t kh = 0; kh < kH; ++kh)
              for (int64_t kw = 0; kw < kW; ++kw)
                for (int64_t oh = 0; oh < oH; ++oh)
                  for (int64_t ow = 0; ow < oW; ++ow) {
                    int64_t h = oh * s0 - p0 + kh * d0, w = ow * s1 - p1 + kw * d1;
                    int64_t kk = (c * kH + kh) * kW + kw, l = oh * oW + ow;
                    float v = (h < 0 || h >= H || w < 0 || w >= Wd) ? 0.f : rd(X, ((n * C + c) * H + h) * Wd + w);
                    wr(O, (n * Kk + kk) * L + l, v);
                  }
        break;
    }
    case K_CALC_WSIZE: {
        int64_t prod = 1; for (double v : e->dscalars) prod *= (int64_t)v;
        ((int64_t *)e->out->data + e->out->offset)[0] = prod;
        break;
    }
    case K_TOINT4PACK: {
        // pack pairs of int8 (each [-8,7]) into one byte: hi<<4 | lo
        const aclTensor *W = e->a; aclTensor *O = e->out; int64_t nOut = O->numel();
        const int8_t *in = (const int8_t *)W->data + W->offset; int8_t *out = (int8_t *)O->data + O->offset;
        for (int64_t i = 0; i < nOut; ++i) out[i] = (int8_t)(((in[2 * i + 1] & 0xF) << 4) | (in[2 * i] & 0xF));
        break;
    }
    case K_AVGPOOL2D: case K_MAXPOOL2D: {   // GPU pool2d kernel (fp32/fp16/bf16); host pool2d otherwise
        bool avg = (e->op == K_AVGPOOL2D);
        int64_t kh = IA((const aclIntArray*)e->inputs[0], 0, 1), kw = IA((const aclIntArray*)e->inputs[0], 1, kh);
        if (!pool2d_gpu(e->a, e->out, kh, kw, s0, s1, p0, p1, avg)) pool2d(e->a, e->out, kh, kw, s0, s1, p0, p1, avg);
        break;
    }
    case K_AVGPOOL3D: case K_MAXPOOL3D: {
        bool avg = (e->op == K_AVGPOOL3D); int64_t k = e->dscalars[0];
        if (!pool3d_gpu(e->a, e->out, k, s0, p0, avg)) pool3d(e->a, e->out, k, s0, p0, avg);
        break;
    }
    case K_ADAPTAVG2D: if (!adaptive_avg2d_gpu(e->a, e->out)) adaptive_avgpool2d(e->a, e->out); break;
    case K_AVGPOOL2D_BWD: if (!pool2d_bwd_gpu(e->a, e->b, e->out, (int64_t)e->dscalars[0], s0, p0, true)) pool2d_bwd(e->a, e->b, e->out, (int64_t)e->dscalars[0], s0, p0, true); break;   // a=x, b=gradOut
    case K_MAXPOOL2D_BWD: if (!pool2d_bwd_gpu(e->a, e->b, e->out, (int64_t)e->dscalars[0], s0, p0, false)) pool2d_bwd(e->a, e->b, e->out, (int64_t)e->dscalars[0], s0, p0, false); break;

    // ----- quant / grouped matmuls -----
    case K_QMM: {
        // x int8 [M,K], weight int8 [K,N], scale[N] -> out = (int sum) * scale
        const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out;
        int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[1];
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            long long acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (long long)rd(X, m * K + k) * (long long)rd(W, k * N + n);
            wr(O, m * N + n, (float)((double)acc * rd(Sc, n)));
          }
        break;
    }
    case K_TQBMM: {
        // x int8 [M,K], weight int8 [N,K] (transposed), scale[N] -> out[M,N] = (x @ wᵀ)*scale
        const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out;
        int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[0];
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            long long acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (long long)rd(X, m * K + k) * (long long)rd(W, n * K + k);
            wr(O, m * N + n, (float)((double)acc * rd(Sc, n)));
          }
        break;
    }
    case K_QMM_IADD: {
        // out += dequant(int8 x @ int8 w, scale[N])
        const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out;
        int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[1];
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            long long acc = 0;
            for (int64_t k = 0; k < K; ++k) acc += (long long)rd(X, m * K + k) * (long long)rd(W, k * N + n);
            float prev = rd(O, m * N + n);
            wr(O, m * N + n, (float)(prev + (double)acc * rd(Sc, n)));
          }
        break;
    }
    case K_WQMM: {
        // x fp16 [M,K], weight int8/int4 [K,N], antiquant scale[N], offset[N] (fp16)
        // wf = (w - offset[n]) * scale[n];  out = x @ wf
        const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out;
        const aclTensor *Of = e->out2;   // antiquantOffset stashed here
        int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[1];
        bool int4 = (W->dtype == ACL_INT4);
        const uint8_t *wbase = (const uint8_t *)W->data + W->offset;
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            double acc = 0; double sc = rd(Sc, n), of = Of ? rd(Of, n) : 0.0;
            for (int64_t k = 0; k < K; ++k) {
              int wv = int4 ? rd_i4(wbase, k * N + n) : (int)(int8_t)wbase[k * N + n];
              double wf = ((double)wv - of) * sc;
              acc += (double)rd(X, m * K + k) * wf;
            }
            wr(O, m * N + n, (float)acc);
          }
        break;
    }
    case K_BMMQUANT: {
        // a int8 [B,M,K], b int8 [B,K,N], scale[N] -> out[B,M,N] = (a@b)*scale[n]
        const aclTensor *A = e->a, *Bt = e->b, *Sc = e->c; aclTensor *O = e->out;
        int64_t B = A->viewDims[0], M = A->viewDims[1], K = A->viewDims[2], N = Bt->viewDims[2];
        for (int64_t b = 0; b < B; ++b)
          for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
              long long acc = 0;
              for (int64_t k = 0; k < K; ++k) acc += (long long)rd(A, (b * M + m) * K + k) * (long long)rd(Bt, (b * K + k) * N + n);
              wr(O, (b * M + m) * N + n, (float)((double)acc * rd(Sc, n)));
            }
        break;
    }
    case K_TBMM: {
        // a[B,M,K] @ b[B,N,K]^T -> out[B,M,N]
        const aclTensor *A = e->a, *Bt = e->b; aclTensor *O = e->out;
        int64_t B = A->viewDims[0], M = A->viewDims[1], K = A->viewDims[2], N = Bt->viewDims[1];
        for (int64_t b = 0; b < B; ++b)
          for (int64_t m = 0; m < M; ++m)
            for (int64_t n = 0; n < N; ++n) {
              double acc = 0;
              for (int64_t k = 0; k < K; ++k) acc += (double)rd(A, (b * M + m) * K + k) * rd(Bt, (b * N + n) * K + k);
              wr(O, (b * M + m) * N + n, (float)acc);
            }
        break;
    }
    case K_GMM: {
        // x[M,K] grouped by groupList @ weight[E,K,N] -> out[M,N]
        const aclTensor *X = e->a, *W = e->b; aclTensor *O = e->out;
        int64_t K = X->viewDims[1], N = W->viewDims[2];
        int64_t off = 0; int E = (int)e->axes.size();
        std::vector<float> Af, Wf, Cf;
        for (int ee = 0; ee < E; ++ee) {
            int64_t cnt = e->axes[ee]; if (cnt <= 0) continue;
            Af.assign(cnt * K, 0.f); Wf.assign(K * N, 0.f); Cf.assign(cnt * N, 0.f);
            for (int64_t i = 0; i < cnt; ++i) for (int64_t k = 0; k < K; ++k) Af[i * K + k] = rd(X, (off + i) * K + k);
            for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < N; ++n) Wf[k * N + n] = rd(W, (ee * K + k) * N + n);
            gemm_f32(Af.data(), Wf.data(), Cf.data(), (int)cnt, (int)K, (int)N);
            for (int64_t i = 0; i < cnt; ++i) for (int64_t n = 0; n < N; ++n) wr(O, (off + i) * N + n, Cf[i * N + n]);
            off += cnt;
        }
        break;
    }
    case K_MXFP: {
        // self,other low-precision (fp8 e4m3 or fp4 e2m1); selfScale[M,nbk], otherScale[nbk,N] E8M0; block=32 along K.
        const aclTensor *A = e->a, *Bt = e->b, *SA = e->c, *SB = e->out2; aclTensor *O = e->out;
        int64_t M = A->viewDims[0], K = A->viewDims[1], N = Bt->viewDims[1];
        const int BLK = 32; int64_t nbk = K / BLK;
        const uint8_t *sa = (const uint8_t *)SA->data + SA->offset;
        const uint8_t *sb = (const uint8_t *)SB->data + SB->offset;
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            double acc = 0;
            for (int64_t b = 0; b < nbk; ++b) {
              double scA = e8m0(sa[m * nbk + b]), scB = e8m0(sb[b * N + n]);
              for (int t = 0; t < BLK; ++t) {
                int64_t k = b * BLK + t;
                acc += (double)rd(A, m * K + k) * scA * rd(Bt, k * N + n) * scB;
              }
            }
            wr(O, m * N + n, (float)acc);
          }
        break;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }

} // namespace

extern "C" {

// ---------- convolutions ----------
aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
        bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *output, int8_t cubeMathType,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)outputPadding; (void)cubeMathType;
    if (!input || !weight || !output || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = transposed ? K_CONVT2D : K_CONV;
    e->a = input; e->b = weight; e->c = bias; e->out = output; e->k = groups ? groups : 1;
    e->stride[0] = IA(stride, 0, 1); e->stride[1] = IA(stride, 1, IA(stride, 0, 1));
    e->pad[0] = IA(padding, 0, 0); e->pad[1] = IA(padding, 1, IA(padding, 0, 0));
    e->dil[0] = IA(dilation, 0, 1); e->dil[1] = IA(dilation, 1, IA(dilation, 0, 1));
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolution)

aclnnStatus aclnnConvolution3dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *output, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !weight || !output || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONV3D; e->a = input; e->b = weight; e->out = output; e->k = groups ? groups : 1;
    e->stride[0] = IA(stride, 0, 1); e->pad[0] = IA(padding, 0, 0); e->dil[0] = IA(dilation, 0, 1);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolution3d)

aclnnStatus aclnnConvolutionTranspose2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *output, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !weight || !output || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONVT2D; e->a = input; e->b = weight; e->out = output; e->k = groups ? groups : 1;
    e->stride[0] = IA(stride, 0, 1); e->pad[0] = IA(padding, 0, 0); e->dil[0] = IA(dilation, 0, 1);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolutionTranspose2d)

aclnnStatus aclnnConvolutionBackwardDataGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)groups;
    if (!gradOutput || !weight || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONV_BWD_DATA; e->a = gradOutput; e->b = weight; e->out = gradInput;
    e->stride[0] = IA(stride, 0, 1); e->pad[0] = IA(padding, 0, 0); e->dil[0] = IA(dilation, 0, 1);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolutionBackwardData)

aclnnStatus aclnnConvolutionBackwardWeightGetWorkspaceSize(const aclTensor *input, const aclTensor *gradOutput,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    (void)groups;
    if (!input || !gradOutput || !gradWeight || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONV_BWD_WEIGHT; e->a = input; e->b = gradOutput; e->out = gradWeight;
    e->stride[0] = IA(stride, 0, 1); e->pad[0] = IA(padding, 0, 0); e->dil[0] = IA(dilation, 0, 1);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolutionBackwardWeight)

aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input,
        const aclTensor *weight, const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding,
        const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups,
        aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, int8_t cubeMathType,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)biasSizes; (void)transposed; (void)outputPadding; (void)groups; (void)cubeMathType;
    if (!gradOutput || !input || !weight || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONV_BWD;
    e->a = gradOutput; e->b = input; e->c = weight; e->out = gradInput; e->out2 = gradWeight;
    if (gradBias) e->inputs.push_back(gradBias);
    e->stride[0] = IA(stride, 0, 1); e->pad[0] = IA(padding, 0, 0); e->dil[0] = IA(dilation, 0, 1);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvolutionBackward)

aclnnStatus aclnnConvTbcGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias,
        int64_t pad, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CONVTBC; e->a = self; e->b = weight; e->c = bias; e->out = out;
    e->pad[0] = pad; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvTbc)

aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclTensor *offset, const aclTensor *mask, const aclTensor *bias, const aclIntArray *stride,
        const aclIntArray *padding, const aclIntArray *dilation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)offset; (void)mask; (void)bias;
    if (!input || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_DEFORMCONV; e->a = input; e->b = weight; e->out = out;
    e->stride[0] = IA(stride, 0, 1); e->stride[1] = IA(stride, 1, IA(stride, 0, 1));
    e->pad[0] = IA(padding, 0, 0); e->pad[1] = IA(padding, 1, IA(padding, 0, 0));
    e->dil[0] = IA(dilation, 0, 1); e->dil[1] = IA(dilation, 1, IA(dilation, 0, 1));
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnDeformableConv2d)

aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *dilation,
        const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_IM2COL; e->a = self; e->out = out;
    e->dscalars = { (double)IA(kernelSize, 0, 1), (double)IA(kernelSize, 1, IA(kernelSize, 0, 1)) };
    e->dil[0] = IA(dilation, 0, 1); e->dil[1] = IA(dilation, 1, IA(dilation, 0, 1));
    e->pad[0] = IA(padding, 0, 0); e->pad[1] = IA(padding, 1, IA(padding, 0, 0));
    e->stride[0] = IA(stride, 0, 1); e->stride[1] = IA(stride, 1, IA(stride, 0, 1));
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnIm2col)

aclnnStatus aclnnCalculateConvolutionWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!weightShape || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_CALC_WSIZE; e->out = out;
    for (int64_t v : weightShape->v) e->dscalars.push_back((double)v);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnCalculateConvolutionWeightSize)

aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_TOINT4PACK; e->a = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnConvertWeightToINT4Pack)

// ---------- pooling ----------
aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
        const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_AVGPOOL2D; e->a = self; e->out = out; e->inputs.push_back((const aclTensor *)kernel);
    e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->stride[1] = IA(stride, 1, e->stride[0]);
    e->pad[0] = IA(padding, 0, 0); e->pad[1] = IA(padding, 1, IA(padding, 0, 0));
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAvgPool2d)
aclnnStatus aclnnMaxPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
        const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_MAXPOOL2D; e->a = self; e->out = out; e->inputs.push_back((const aclTensor *)kernel);
    e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->stride[1] = IA(stride, 1, e->stride[0]);
    e->pad[0] = IA(padding, 0, 0); e->pad[1] = IA(padding, 1, IA(padding, 0, 0));
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMaxPool2d)
aclnnStatus aclnnAvgPool3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
        const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_AVGPOOL3D; e->a = self; e->out = out;
    e->dscalars = { (double)IA(kernel, 0, 1) }; e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->pad[0] = IA(padding, 0, 0);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAvgPool3d)
aclnnStatus aclnnMaxPool3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
        const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_MAXPOOL3D; e->a = self; e->out = out;
    e->dscalars = { (double)IA(kernel, 0, 1) }; e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->pad[0] = IA(padding, 0, 0);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMaxPool3d)
aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_ADAPTAVG2D; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAdaptiveAvgPool2d)
aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)y;
    if (!x || !gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_AVGPOOL2D_BWD; e->a = x; e->b = gradOutput; e->out = gradInput;
    e->dscalars = { (double)IA(kernel, 0, 1) }; e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->pad[0] = IA(padding, 0, 0);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAvgPool2dBackward)
aclnnStatus aclnnMaxPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)y;
    if (!x || !gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_MAXPOOL2D_BWD; e->a = x; e->b = gradOutput; e->out = gradInput;
    e->dscalars = { (double)IA(kernel, 0, 1) }; e->stride[0] = IA(stride, 0, IA(kernel, 0, 1)); e->pad[0] = IA(padding, 0, 0);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMaxPool2dBackward)

// ---------- quant / grouped matmuls ----------
aclnnStatus aclnnQuantMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_QMM; e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantMatmul)
aclnnStatus aclnnQuantMatmulV3GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_QMM; e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantMatmulV3)
aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_QMM_IADD; e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnQuantBatchMatmulInplaceAdd)
aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !antiquantScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_WQMM; e->a = x; e->b = weight; e->c = antiquantScale; e->out2 = (aclTensor *)antiquantOffset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnWeightQuantBatchMatmul)
aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_BMMQUANT; e->a = self; e->b = other; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnBatchMatmulQuant)
aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType;
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_TBMM; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnTransposeBatchMatMul)
aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_TQBMM; e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnTransposeQuantBatchMatMul)
aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !groupList || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = K_GMM; e->a = x; e->b = weight; e->out = out; e->axes = groupList->v; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnGroupedMatmul)

// MX fp8 / fp4 (block-scaled). selfScale/otherScale are E8M0 per-32-block.
#define MXDEF(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale, const aclTensor *other, \
        const aclTensor *otherScale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = K_MXFP; e->a = self; e->b = other; e->c = selfScale; \
    e->out2 = (aclTensor *)otherScale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
MXDEF(aclnnMatmulMxFp8)
MXDEF(aclnnMatmulMxFp8Hw)
MXDEF(aclnnMatmulMxFp4)
MXDEF(aclnnMatmulMxFp4Hw)

} // extern "C"
