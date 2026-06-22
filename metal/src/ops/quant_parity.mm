// Quant/dequant/antiquant/MX-fp gap operators — CUDA-parity set not previously implemented for Metal.
// Host-side over unified memory (MTLStorageModeShared => device ptr == host ptr; drain the stream first).
// Semantics mirror the CUDA reference (cuda/src/ops/{quant,matmul,norm_ext,glu_ext,conv}.cu) and the
// existing Metal quant.mm / quant_ext.mm / norm_ext.mm. Only ops with no existing symbol are defined here
// (see `nm -gU metal/lib/libascendcl.dylib | grep -iE quant`) so there are no duplicate definitions.
//
// Scale conventions (shared with the rest of the backend):
//   per-row / per-block / per-group dynamic int8 : scale = absmax/127, q = clamp(round(x/scale), lo, hi)
//   MX power-of-2                                 : scale = exp2(ceil(log2(absmax/127)))
//   dual-level MX                                 : s1 = MX(absmax), s2 = (absmax/127)/s1, q rounds vs s1*s2
//   per-channel affine                            : q = clamp(round(x*scale[c] + offset[c]), -128, 127)
//   W8A8 matmul                                   : out[m,n] = (sum_k x_i8 * w_i8) * scale[n]
//   W8A16/W4A16 (weight-only)                     : out = x @ ((w - offset[n]) * scale[n])
#import "../internal.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>   // native int8 GEMM for W8A8 (qmm_core)
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_mc2.h"   // canonical quant-MC2 signatures (carry HcclComm; single-rank == local QuantMatmul)
#include "subfp.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace {
const float *FP(const aclTensor *t) { return (const float *)t->data + t->offset; }
float *FPW(const aclTensor *t) { return (float *)t->data + t->offset; }
int8_t *I8(const aclTensor *t) { return (int8_t *)t->data + t->offset; }
const int8_t *CI8(const aclTensor *t) { return (const int8_t *)t->data + t->offset; }
const int32_t *CI32(const aclTensor *t) { return (const int32_t *)t->data + t->offset; }
const int64_t *CI64(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
int64_t *I64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
uint8_t *U8W(const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
const uint8_t *CU8(const aclTensor *t) { return (const uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

inline int clampi(long v, int lo, int hi) { return (int)(v < lo ? lo : (v > hi ? hi : v)); }
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
inline float gelu(float v) { return 0.5f * v * std::erfc(-v * (float)M_SQRT1_2); }
inline double mx_scale(double amax) { return amax > 0 ? std::exp2(std::ceil(std::log2(amax / 127.0))) : 1.0; }
// int4 read (signed nibble) from a packed buffer.
inline int rd_i4(const uint8_t *b, int64_t i) { uint8_t n = (i & 1) ? (b[i / 2] >> 4) : (b[i / 2] & 0xf); return (n & 0x8) ? (int)n - 16 : (int)n; }

// ---- shared kernels ----
// Per-row symmetric int8: q = round(v/scale), scale = absmax/127 (clamp [-127,127] like CUDA dynamic path).
void quant_rows_dyn(const std::vector<double> &v, int64_t R, int64_t D, int8_t *q, float *scl) {
    for (int64_t r = 0; r < R; ++r) {
        double amax = 0; for (int64_t d = 0; d < D; ++d) amax = std::max(amax, std::fabs(v[r * D + d]));
        double sc = amax > 0 ? amax / 127.0 : 1.0; if (scl) scl[r] = (float)sc; double qi = 1.0 / sc;
        for (int64_t d = 0; d < D; ++d) q[r * D + d] = (int8_t)clampi(std::lrint(v[r * D + d] * qi), -127, 127);
    }
}
// Per-row MX (power-of-2) int8: clamp [-128,127] like the MX paths in norm_ext.mm.
void quant_rows_mx(const std::vector<double> &v, int64_t R, int64_t D, int8_t *q, float *scl) {
    for (int64_t r = 0; r < R; ++r) {
        double amax = 0; for (int64_t d = 0; d < D; ++d) amax = std::max(amax, std::fabs(v[r * D + d]));
        double sc = mx_scale(amax); if (scl) scl[r] = (float)sc; double qi = 1.0 / sc;
        for (int64_t d = 0; d < D; ++d) q[r * D + d] = (int8_t)clampi(std::lrint(v[r * D + d] * qi), -128, 127);
    }
}

// =================================================================================================
// ---- pure quant/dequant ----

// AscendQuantV3: per-channel affine int8 (last-dim channel).  q = clamp(round(x*sc[c] + of[c]), -128, 127)
aclnnStatus run_ascend_quant_v3(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
    int64_t C = Sc->numel(), n = X->numel(); const float *x = FP(X), *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; int8_t *q = I8(O);
    for (int64_t i = 0; i < n; ++i) { int64_t c = i % C; double v = (double)x[i] * sc[c] + (of ? of[c] : 0.0);
        q[i] = (int8_t)clampi(std::lrint(v), -128, 127); }
    return ACLNN_SUCCESS;
}
// AscendAntiQuant: per-channel dequant.  y = (q - offset[c]) * scale[c]
aclnnStatus run_ascend_antiquant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
    int64_t C = Sc->numel(), n = Q->numel(); const int8_t *q = CI8(Q); const float *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; float *y = FPW(O);
    for (int64_t i = 0; i < n; ++i) { int64_t c = i % C; y[i] = (float)(((double)q[i] - (of ? of[c] : 0.0)) * sc[c]); }
    return ACLNN_SUCCESS;
}
// GroupQuant: per contiguous group of groupSize elements (over the flattened tensor), affine int8.
aclnnStatus run_group_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out; int64_t G = e->k;
    int64_t n = X->numel(); const float *x = FP(X), *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; int8_t *q = I8(O);
    for (int64_t i = 0; i < n; ++i) { int64_t g = i / G; double v = (double)x[i] * sc[g] + (of ? of[g] : 0.0);
        q[i] = (int8_t)clampi(std::lrint(v), -128, 127); }
    return ACLNN_SUCCESS;
}
// GeluQuant: y = gelu(x) per row (last dim), per-row absmax int8.
aclnnStatus run_gelu_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out, *Sc = e->out2;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D; const float *x = FP(X); int8_t *q = I8(O); float *sc = FPW(Sc);
    std::vector<double> g(R * D);
    for (int64_t r = 0; r < R; ++r) for (int64_t d = 0; d < D; ++d) g[r * D + d] = gelu(x[r * D + d]);
    quant_rows_dyn(g, R, D, q, sc);
    return ACLNN_SUCCESS;
}
// GroupNormSiluQuant: x[N,C,S...]; per (n,group) normalize (affine per-channel), SiLU, then per-(n,group) absmax int8.
aclnnStatus run_gnsilu_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *g = e->b, *b = e->c; aclTensor *O = e->out, *Sc = e->out2; double eps = e->eps; int G = (int)e->dim;
    int nd = (int)X->viewDims.size(); int64_t N = X->viewDims[0], C = X->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= X->viewDims[d];
    int64_t Cg = C / G, cnt = Cg * S; const float *xp = FP(X), *gp = g ? FP(g) : nullptr, *bp = b ? FP(b) : nullptr; int8_t *q = I8(O); float *scp = FPW(Sc);
    std::vector<double> y(cnt);
    for (int64_t n = 0; n < N; ++n) for (int grp = 0; grp < G; ++grp) {
        double m = 0, qsum = 0; for (int64_t cc = 0; cc < Cg; ++cc) for (int64_t sp = 0; sp < S; ++sp) { double v = xp[(n * C + grp * Cg + cc) * S + sp]; m += v; qsum += v * v; }
        m /= cnt; double var = qsum / cnt - m * m; double inv = 1.0 / std::sqrt(var + eps);
        double amax = 0; int64_t idx = 0;
        for (int64_t cc = 0; cc < Cg; ++cc) { int64_t c = grp * Cg + cc; double gv = gp ? gp[c] : 1.0, bv = bp ? bp[c] : 0.0;
            for (int64_t sp = 0; sp < S; ++sp) { double yy = silu((float)((xp[(n * C + c) * S + sp] - m) * inv * gv + bv)); y[idx++] = yy; amax = std::max(amax, std::fabs(yy)); } }
        double sc = amax > 0 ? amax / 127.0 : 1.0; scp[n * G + grp] = (float)sc; double qi = 1.0 / sc;
        idx = 0; for (int64_t cc = 0; cc < Cg; ++cc) { int64_t c = grp * Cg + cc; for (int64_t sp = 0; sp < S; ++sp) q[(n * C + c) * S + sp] = (int8_t)clampi(std::lrint(y[idx++] * qi), -127, 127); }
    }
    return ACLNN_SUCCESS;
}
// FakeQuantPerChannelAffineCachemask: per-channel q = clamp(round(x/scale[c]+zp[c]), qmin,qmax);
//   out = (q-zp[c])*scale[c]; mask[i] = (qmin <= q <= qmax before clamp). axis = channel dim.
aclnnStatus run_fakeq_perchannel(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Sc = e->b, *Zp = e->c; aclTensor *O = e->out, *Mask = e->out2;
    int axis = (int)e->dim; int qmin = (int)e->dscalars[0], qmax = (int)e->dscalars[1];
    int nd = (int)X->viewDims.size(); if (axis < 0) axis += nd;
    int64_t C = X->viewDims[axis], n = X->numel();
    int64_t inner = 1; for (int d = axis + 1; d < nd; ++d) inner *= X->viewDims[d];
    const float *x = FP(X), *sc = FP(Sc); const float *zp = Zp ? FP(Zp) : nullptr; float *y = FPW(O); uint8_t *mk = Mask ? U8W(Mask) : nullptr;
    for (int64_t i = 0; i < n; ++i) { int64_t c = (i / inner) % C; double z = zp ? zp[c] : 0.0;
        long qr = std::lrint((double)x[i] / sc[c] + z); int q = clampi(qr, qmin, qmax);
        y[i] = (float)(((double)q - z) * sc[c]); if (mk) mk[i] = (uint8_t)(qr >= qmin && qr <= qmax); }
    return ACLNN_SUCCESS;
}
// QuantizedBatchNorm: bn(x) = (x-mean[c])*invstd[c]*weight[c]+bias[c]; q = clamp(round(bn/scale)+zp, -128,127).
aclnnStatus run_qbn(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *W = e->b, *B = e->c, *Mn = e->mean, *Iv = e->rstd; aclTensor *O = e->out;
    double scale = e->alpha; int zp = (int)e->dscalars[0];
    int nd = (int)X->viewDims.size(); int64_t N = X->viewDims[0], C = X->viewDims[1], S = 1; for (int d = 2; d < nd; ++d) S *= X->viewDims[d];
    const float *xp = FP(X), *wp = W ? FP(W) : nullptr, *bp = B ? FP(B) : nullptr, *mp = FP(Mn), *ip = FP(Iv); int8_t *q = I8(O);
    double qi = 1.0 / scale;
    for (int64_t n = 0; n < N; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t sp = 0; sp < S; ++sp) {
        int64_t idx = (n * C + c) * S + sp; double bn = (xp[idx] - mp[c]) * ip[c] * (wp ? wp[c] : 1.0) + (bp ? bp[c] : 0.0);
        q[idx] = (int8_t)clampi(std::lrint(bn * qi) + zp, -128, 127); }
    return ACLNN_SUCCESS;
}

// DynamicQuant per-token int8 (V2/V3/V4): scale = absmax/127.
aclnnStatus run_dynamic_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out, *Sco = e->out2;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D; const float *x = FP(X); int8_t *q = I8(O); float *sc = FPW(Sco);
    std::vector<double> v(R * D); for (int64_t i = 0; i < R * D; ++i) v[i] = x[i];
    quant_rows_dyn(v, R, D, q, sc);
    return ACLNN_SUCCESS;
}
// DynamicBlockQuantV2: per contiguous block (last dim of size blk) absmax/127 int8.
aclnnStatus run_block_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out, *Sco = e->out2; int64_t blk = e->k;
    int64_t nblk = X->numel() / blk; const float *x = FP(X); int8_t *q = I8(O); float *sc = FPW(Sco);
    std::vector<double> v(X->numel()); for (int64_t i = 0; i < X->numel(); ++i) v[i] = x[i];
    quant_rows_dyn(v, nblk, blk, q, sc);
    return ACLNN_SUCCESS;
}
// DynamicMxQuant / V2 / GroupedDynamicMxQuant: per-block MX power-of-2 int8.
aclnnStatus run_mx_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out, *Sco = e->out2; int64_t blk = e->k;
    int64_t nblk = X->numel() / blk; const float *x = FP(X); int8_t *q = I8(O); float *sc = FPW(Sco);
    std::vector<double> v(X->numel()); for (int64_t i = 0; i < X->numel(); ++i) v[i] = x[i];
    quant_rows_mx(v, nblk, blk, q, sc);
    return ACLNN_SUCCESS;
}
// DynamicMxQuantWithDualAxis: per-block MX over last dim (scaleOut) AND per-block over leading axis (scaleOut2).
// Both scales describe the SAME q (q = round(x/scaleLast)); scaleOut2 is the leading-axis MX scale of x.
aclnnStatus run_mx_dualaxis(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out, *S1 = e->out2; aclTensor *S2 = const_cast<aclTensor *>(e->mean); int64_t blk = e->k;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D; (void)blk;
    const float *x = FP(X); int8_t *q = I8(O); float *s1 = FPW(S1); float *s2 = S2 ? FPW(S2) : nullptr;
    // last-dim MX (one scale per row), used to quantize.
    for (int64_t r = 0; r < R; ++r) {
        double amax = 0; for (int64_t d = 0; d < D; ++d) amax = std::max(amax, std::fabs((double)x[r * D + d]));
        double sc = mx_scale(amax); s1[r] = (float)sc; double qi = 1.0 / sc;
        for (int64_t d = 0; d < D; ++d) q[r * D + d] = (int8_t)clampi(std::lrint((double)x[r * D + d] * qi), -128, 127);
    }
    // leading-axis MX (one scale per column).
    if (s2) for (int64_t d = 0; d < D; ++d) { double amax = 0; for (int64_t r = 0; r < R; ++r) amax = std::max(amax, std::fabs((double)x[r * D + d])); s2[d] = (float)mx_scale(amax); }
    return ACLNN_SUCCESS;
}
// GroupedDynamicBlockQuant: per contiguous block absmax/127 int8 (alias of block quant).

// SwiGluQuantV2: swiglu(in[...,2D]) -> [...,D], per-row absmax int8.
aclnnStatus run_swiglu_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *In = e->a; aclTensor *O = e->out, *Sc = e->out2;
    int64_t D = O->viewDims.back(); int64_t R = In->numel() / (2 * D); const float *x = FP(In); int8_t *q = I8(O); float *sc = FPW(Sc);
    std::vector<double> g(R * D);
    for (int64_t r = 0; r < R; ++r) for (int64_t d = 0; d < D; ++d) { double a = x[r * 2 * D + d], b = x[r * 2 * D + D + d]; g[r * D + d] = silu((float)a) * b; }
    quant_rows_dyn(g, R, D, q, sc);
    return ACLNN_SUCCESS;
}
// DequantSwigluQuantV2: optional per-row dequant scale, swiglu, per-row absmax int8.
aclnnStatus run_dequant_swiglu_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *In = e->a, *Dq = e->b; aclTensor *O = e->out, *Sc = e->out2;
    int64_t D = O->viewDims.back(); int64_t R = In->numel() / (2 * D); const float *x = FP(In); const float *dq = Dq ? FP(Dq) : nullptr; int8_t *q = I8(O); float *sc = FPW(Sc);
    std::vector<double> g(R * D);
    for (int64_t r = 0; r < R; ++r) { double scl = dq ? dq[r] : 1.0; for (int64_t d = 0; d < D; ++d) { double a = x[r * 2 * D + d] * scl, b = x[r * 2 * D + D + d] * scl; g[r * D + d] = silu((float)a) * b; } }
    quant_rows_dyn(g, R, D, q, sc);
    return ACLNN_SUCCESS;
}

// ---- norm+quant fusions ----
// AddRmsNormQuant (V2): s = x+res; yn = rms(s)*gamma; q = clamp(round(yn*scale + offset), -128,127); residualSum=s.
aclnnStatus run_addrms_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Res = e->b, *G = e->c; aclTensor *Yq = e->out, *Rs = const_cast<aclTensor *>(e->mean);
    double eps = e->eps, scale = e->alpha, off = e->dscalars[0];
    int64_t D = G->numel(); int64_t R = X->numel() / D; const float *xp = FP(X), *rp = FP(Res), *gp = FP(G); int8_t *q = I8(Yq); float *rsum = Rs ? FPW(Rs) : nullptr;
    double qi = 1.0 / scale;
    for (int64_t r = 0; r < R; ++r) { double ss = 0; std::vector<double> sv(D);
        for (int64_t d = 0; d < D; ++d) { double v = (double)xp[r * D + d] + rp[r * D + d]; sv[d] = v; ss += v * v; if (rsum) rsum[r * D + d] = (float)v; }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int64_t d = 0; d < D; ++d) { double yn = sv[d] * inv * gp[d]; q[r * D + d] = (int8_t)clampi(std::lrint(yn * qi + off), -128, 127); } }
    return ACLNN_SUCCESS;
}
// AddRmsNormDynamicQuantV2: s=x+res; yn=rms(s)*gamma; per-row absmax int8 (scaleOut), residualSum=s.
aclnnStatus run_addrms_dyn_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Res = e->b, *G = e->c; aclTensor *Yq = e->out, *Sc = e->out2, *Rs = const_cast<aclTensor *>(e->mean);
    double eps = e->eps; int64_t D = G->numel(); int64_t R = X->numel() / D;
    const float *xp = FP(X), *rp = FP(Res), *gp = FP(G); int8_t *q = I8(Yq); float *scp = FPW(Sc); float *rsum = Rs ? FPW(Rs) : nullptr;
    std::vector<double> yv(R * D);
    for (int64_t r = 0; r < R; ++r) { double ss = 0; std::vector<double> sv(D);
        for (int64_t d = 0; d < D; ++d) { double v = (double)xp[r * D + d] + rp[r * D + d]; sv[d] = v; ss += v * v; if (rsum) rsum[r * D + d] = (float)v; }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int64_t d = 0; d < D; ++d) yv[r * D + d] = sv[d] * inv * gp[d]; }
    quant_rows_dyn(yv, R, D, q, scp);
    return ACLNN_SUCCESS;
}
// AddRmsNormDynamicMxQuant: s=x+res; yn=rms(s)*gamma; per-row MX int8 (scaleOut), residualSum=s.
aclnnStatus run_addrms_mx_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Res = e->b, *G = e->c; aclTensor *Yq = e->out, *Sc = e->out2, *Rs = const_cast<aclTensor *>(e->mean);
    double eps = e->eps; int64_t D = G->numel(); int64_t R = X->numel() / D;
    const float *xp = FP(X), *rp = FP(Res), *gp = FP(G); int8_t *q = I8(Yq); float *scp = FPW(Sc); float *rsum = Rs ? FPW(Rs) : nullptr;
    std::vector<double> yv(R * D);
    for (int64_t r = 0; r < R; ++r) { double ss = 0; std::vector<double> sv(D);
        for (int64_t d = 0; d < D; ++d) { double v = (double)xp[r * D + d] + rp[r * D + d]; sv[d] = v; ss += v * v; if (rsum) rsum[r * D + d] = (float)v; }
        double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int64_t d = 0; d < D; ++d) yv[r * D + d] = sv[d] * inv * gp[d]; }
    quant_rows_mx(yv, R, D, q, scp);
    return ACLNN_SUCCESS;
}
// AdaLayerNormQuant: y = norm(x)*(1+scale)+shift over last dim; per-row MX int8 (scaleOut).
aclnnStatus run_adaln_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Sc = e->b, *Sh = e->c; aclTensor *O = e->out, *Sco = e->out2; double eps = e->eps;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D; const float *xp = FP(X), *scp = FP(Sc), *shp = FP(Sh); int8_t *q = I8(O); float *sco = FPW(Sco);
    std::vector<double> yv(R * D);
    for (int64_t r = 0; r < R; ++r) { double m = 0; for (int64_t d = 0; d < D; ++d) m += xp[r * D + d]; m /= D;
        double v = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - m; v += u * u; } v /= D; double inv = 1.0 / std::sqrt(v + eps);
        for (int64_t d = 0; d < D; ++d) yv[r * D + d] = (xp[r * D + d] - m) * inv * (1.0 + scp[r * D + d]) + shp[r * D + d]; }
    quant_rows_mx(yv, R, D, q, sco);
    return ACLNN_SUCCESS;
}
// AddLayerNormQuant: s=x+res; y = LN(s)*gamma+beta over last dim; per-row absmax int8 (scaleOut), residualSum=s.
aclnnStatus run_addln_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Res = e->b, *G = e->c, *B = e->mask; aclTensor *O = e->out, *Sc = e->out2, *Rs = const_cast<aclTensor *>(e->mean);
    double eps = e->eps; int64_t D = G->numel(); int64_t R = X->numel() / D;
    const float *xp = FP(X), *rp = FP(Res), *gp = FP(G), *bp = B ? FP(B) : nullptr; int8_t *q = I8(O); float *scp = FPW(Sc); float *rsum = Rs ? FPW(Rs) : nullptr;
    std::vector<double> yv(R * D);
    for (int64_t r = 0; r < R; ++r) { std::vector<double> sv(D); double m = 0;
        for (int64_t d = 0; d < D; ++d) { double v = (double)xp[r * D + d] + rp[r * D + d]; sv[d] = v; m += v; if (rsum) rsum[r * D + d] = (float)v; }
        m /= D; double var = 0; for (int64_t d = 0; d < D; ++d) { double u = sv[d] - m; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        for (int64_t d = 0; d < D; ++d) yv[r * D + d] = (sv[d] - m) * inv * gp[d] + (bp ? bp[d] : 0.0); }
    quant_rows_dyn(yv, R, D, q, scp);
    return ACLNN_SUCCESS;
}
// SwinTransformerLnQkvQuant: y = LN(x)*gamma+beta over last dim; per-row absmax int8 (scaleOut).
aclnnStatus run_swin_lnqkv_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *G = e->b, *B = e->c; aclTensor *O = e->out, *Sc = e->out2; double eps = e->eps;
    int64_t D = G->numel(); int64_t R = X->numel() / D; const float *xp = FP(X), *gp = FP(G), *bp = B ? FP(B) : nullptr; int8_t *q = I8(O); float *scp = FPW(Sc);
    std::vector<double> yv(R * D);
    for (int64_t r = 0; r < R; ++r) { double m = 0; for (int64_t d = 0; d < D; ++d) m += xp[r * D + d]; m /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double u = xp[r * D + d] - m; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        for (int64_t d = 0; d < D; ++d) yv[r * D + d] = (xp[r * D + d] - m) * inv * gp[d] + (bp ? bp[d] : 0.0); }
    quant_rows_dyn(yv, R, D, q, scp);
    return ACLNN_SUCCESS;
}

// ---- TransQuantParam V2/V3: pack fp32 scale bits (low32) + optional offset (high32) into int64 ----
aclnnStatus run_trans_quant_param(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Sc = e->a, *Of = e->b; aclTensor *O = e->out;
    int64_t n = Sc->numel(); const float *sc = FP(Sc); const float *of = Of ? FP(Of) : nullptr; int64_t *o = I64(O);
    for (int64_t i = 0; i < n; ++i) { uint32_t lo; float f = sc[i]; std::memcpy(&lo, &f, 4);
        uint64_t hi = 0; if (of) { uint32_t ob; float fo = of[i]; std::memcpy(&ob, &fo, 4); hi = (uint64_t)ob; }
        o[i] = (int64_t)((hi << 32) | (uint64_t)lo); }
    return ACLNN_SUCCESS;
}
// ApplyAdamWQuant: simplified equivalence — full-precision AdamW update (param/m/v in place).
aclnnStatus run_apply_adamw_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *P = const_cast<aclTensor *>(e->a); aclTensor *M = const_cast<aclTensor *>(e->b); aclTensor *V = const_cast<aclTensor *>(e->c); const aclTensor *Gr = e->mean;
    double lr = e->dscalars[0], b1 = e->dscalars[1], b2 = e->dscalars[2], eps = e->dscalars[3], wd = e->dscalars[4]; int64_t step = (int64_t)e->dscalars[5];
    int64_t n = P->numel(); float *p = FPW(P), *m = FPW(M), *v = FPW(V); const float *g = FP(Gr);
    double bc1 = 1.0 - std::pow(b1, (double)step), bc2 = 1.0 - std::pow(b2, (double)step);
    for (int64_t i = 0; i < n; ++i) { double pw = p[i] - lr * wd * p[i];
        double mi = b1 * m[i] + (1 - b1) * g[i]; double vi = b2 * v[i] + (1 - b2) * (double)g[i] * g[i]; m[i] = (float)mi; v[i] = (float)vi;
        double mh = mi / bc1, vh = vi / bc2; p[i] = (float)(pw - lr * mh / (std::sqrt(vh) + eps)); }
    return ACLNN_SUCCESS;
}

// int8·int8 [M,K]@[K,N] -> fp32 raw integer sums via MPS. MPSMatrixMultiplication accepts Int8 inputs but
// requires an fp16/fp32 result; the sum is exact while it stays in fp32's 2^24 integer range (int8·int8 ⇒
// K up to ~1000), larger K rounds (within W8A8 quant tolerance). Returns false (→ host fallback) on any error.
static bool qmm_mps_int8(const aclTensor *X, const aclTensor *W, float *raw, int64_t M, int64_t K, int64_t N) {
    size_t ox = 0, ow = 0, oo = 0;
    id<MTLBuffer> bx = mtl::bufferFor(X->data, &ox), bw = mtl::bufferFor(W->data, &ow), bo = mtl::bufferFor(raw, &oo);
    if (!bx || !bw || !bo) return false;
    ox += (size_t)X->offset; ow += (size_t)W->offset;   // int8 ⇒ 1 byte/element
    MPSMatrixDescriptor *dX = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K dataType:MPSDataTypeInt8];
    MPSMatrixDescriptor *dW = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N rowBytes:N dataType:MPSDataTypeInt8];
    MPSMatrixDescriptor *dO = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N rowBytes:N * 4 dataType:MPSDataTypeFloat32];
    MPSMatrix *mX = [[MPSMatrix alloc] initWithBuffer:bx offset:ox descriptor:dX];
    MPSMatrix *mW = [[MPSMatrix alloc] initWithBuffer:bw offset:ow descriptor:dW];
    MPSMatrix *mO = [[MPSMatrix alloc] initWithBuffer:bo offset:oo descriptor:dO];
    MPSMatrixMultiplication *mm = [[MPSMatrixMultiplication alloc] initWithDevice:mtl::device() transposeLeft:NO
                                       transposeRight:NO resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];
    id<MTLCommandBuffer> cb = [mtl::defaultQueue() commandBuffer];
    [mm encodeToCommandBuffer:cb leftMatrix:mX rightMatrix:mW resultMatrix:mO];
    [cb commit]; [cb waitUntilCompleted];
    return cb.error == nil;
}

// GPU per-channel dequant epilogue: o[m,n] = raw[m,n] * scale[n] (+ accumulate), via the qscale kernel.
// Returns false (caller does the host loop) if the kernel/buffers are unavailable or offsets misaligned.
struct QScaleMeta { uint32_t M, N; int32_t accumulate; };
static bool qmm_scale_gpu(aclTensor *O, const float *raw, const aclTensor *Sc, int64_t M, int64_t N, bool accumulate) {
    bool sc16 = (Sc->dtype == ACL_FLOAT16);
    id<MTLComputePipelineState> pso = mtl::pipeline(sc16 ? @"qscale_f16" : @"qscale_f32");
    if (!pso) return false;
    size_t oo, oraw, osc;
    id<MTLBuffer> bo = mtl::bufferFor(O->data, &oo), braw = mtl::bufferFor((void *)raw, &oraw), bsc = mtl::bufferFor(Sc->data, &osc);
    if (!bo || !braw || !bsc) return false;
    oo += (size_t)O->offset * 4; osc += (size_t)Sc->offset * dtype_size(Sc->dtype);
    if ((oo | oraw | osc) & 3u) return false;   // Metal compute buffer offsets must be 4-byte aligned
    QScaleMeta qs{ (uint32_t)M, (uint32_t)N, accumulate ? 1 : 0 };
    id<MTLCommandQueue> q = mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bo offset:oo atIndex:0]; [enc setBuffer:braw offset:oraw atIndex:1]; [enc setBuffer:bsc offset:osc atIndex:2];
    [enc setBytes:&qs length:sizeof(qs) atIndex:3];
    NSUInteger n = (NSUInteger)(M * N), tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
    return cb.error == nil;
}

// ---- W8A8 quant matmul core: out[m,n] = (sum_k x_i8 * w_i8) * scale[n] ----
// Native MPS Int8 GEMM (int8·int8 → fp32 raw) + per-channel scale epilogue (GPU qscale, host loop fallback);
// exact host int-accumulation fallback on MPS error.
void qmm_core(const aclTensor *X, const aclTensor *W, const aclTensor *Sc, aclTensor *O, bool accumulate) {
    int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[1];
    float *o = FPW(O);
    bool sc16 = (Sc->dtype == ACL_FLOAT16);   // CANN deqScale is fp16 (CUDA-canonical); also accept fp32
    const float *scf = sc16 ? nullptr : FP(Sc);
    const _Float16 *sch = sc16 ? (const _Float16 *)Sc->data + Sc->offset : nullptr;
    auto scale = [&](int64_t n) -> double { return sc16 ? (double)sch[n] : (double)scf[n]; };
    float *raw = (float *)mtl::alloc((size_t)M * N * 4);
    if (raw && qmm_mps_int8(X, W, raw, M, K, N)) {
        if (!qmm_scale_gpu(O, raw, Sc, M, N, accumulate)) {
            for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
                double val = (double)raw[m * N + n] * scale(n);
                o[m * N + n] = (float)(accumulate ? (double)o[m * N + n] + val : val);
            }
        }
        mtl::free_(raw); return;
    }
    if (raw) mtl::free_(raw);
    const int8_t *x = CI8(X), *w = CI8(W);   // host fallback (exact int accumulation)
    for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        long long acc = 0; for (int64_t k = 0; k < K; ++k) acc += (long long)x[m * K + k] * (long long)w[k * N + n];
        double val = (double)acc * scale(n); o[m * N + n] = (float)(accumulate ? (double)o[m * N + n] + val : val);
    }
}
aclnnStatus run_qmm(aclOpExecutor *e, aclrtStream s) { drain(s); qmm_core(e->a, e->b, e->c, e->out, false); return ACLNN_SUCCESS; }
aclnnStatus run_qmm_iadd_g(aclOpExecutor *e, aclrtStream s) { drain(s); qmm_core(e->a, e->b, e->c, e->out, true); return ACLNN_SUCCESS; }

// ---- W8A16/W4A16 weight-only matmul: out = x @ ((w - offset[n]) * scale[n]); x fp32, w int8/int4 [K,N] ----
aclnnStatus run_wqmm(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out; const aclTensor *Of = e->out2;
    int64_t M = X->viewDims[0], K = X->viewDims[1], N = W->viewDims[1]; bool int4 = (W->dtype == ACL_INT4);
    const float *x = FP(X), *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; const uint8_t *wb = CU8(W); float *o = FPW(O);
    for (int64_t m = 0; m < M; ++m) for (int64_t n = 0; n < N; ++n) {
        double acc = 0, scn = sc[n], ofn = of ? of[n] : 0.0;
        for (int64_t k = 0; k < K; ++k) { int wv = int4 ? rd_i4(wb, k * N + n) : (int)(int8_t)wb[k * N + n]; acc += (double)x[m * K + k] * (((double)wv - ofn) * scn); }
        o[m * N + n] = (float)acc;
    }
    return ACLNN_SUCCESS;
}

// ---- grouped W8A8 matmul: x[totalM,K] int8, weight[E,K,N] int8, scale[E,N], groupList prefix-sums over rows ----
// inplaceAdd: out += grouped result.
void gqmm_core(aclOpExecutor *e, bool accumulate) {
    const aclTensor *X = e->a, *W = e->b, *Sc = e->c; aclTensor *O = e->out; const std::vector<int64_t> &gl = e->axes;
    int64_t K = X->viewDims[1], N = W->viewDims.back();
    const int8_t *x = CI8(X), *w = CI8(W); const float *sc = FP(Sc); float *o = FPW(O);
    int64_t row0 = 0;
    for (size_t g = 0; g < gl.size(); ++g) {
        int64_t row1 = gl[g]; const int8_t *wg = w + (int64_t)g * K * N; const float *scg = sc + (int64_t)g * N;
        for (int64_t m = row0; m < row1; ++m) for (int64_t n = 0; n < N; ++n) {
            long long acc = 0; for (int64_t k = 0; k < K; ++k) acc += (long long)x[m * K + k] * (long long)wg[k * N + n];
            double val = (double)acc * scg[n]; o[m * N + n] = (float)(accumulate ? (double)o[m * N + n] + val : val);
        }
        row0 = row1;
    }
}
aclnnStatus run_gqmm(aclOpExecutor *e, aclrtStream s) { drain(s); gqmm_core(e, false); return ACLNN_SUCCESS; }
aclnnStatus run_gqmm_iadd(aclOpExecutor *e, aclrtStream s) { drain(s); gqmm_core(e, true); return ACLNN_SUCCESS; }

// ---- grouped matmul + swiglu + per-row int8 quant: x[totalM,K] int8, weight[E,K,2N] int8 ----
// per expert group: tmp[m,2N] = x@w (int accum); swiglu -> g[m,N]; per-row absmax int8 -> out[m,N], scaleOut[m].
// (signature carries no separate column scale; the per-row output scale is the only scale.)
aclnnStatus run_gmm_swiglu_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *W = e->b; aclTensor *O = e->out, *Sco = e->out2; const std::vector<int64_t> &gl = e->axes;
    int64_t K = X->viewDims[1], N2 = W->viewDims.back(), N = N2 / 2, M = X->viewDims[0];
    const int8_t *x = CI8(X), *w = CI8(W); int8_t *o = I8(O); float *sco = FPW(Sco);
    std::vector<double> g(M * N);
    int64_t row0 = 0;
    for (size_t gi = 0; gi < gl.size(); ++gi) {
        int64_t row1 = gl[gi]; const int8_t *wg = w + (int64_t)gi * K * N2;
        for (int64_t m = row0; m < row1; ++m) {
            std::vector<double> t(N2);
            for (int64_t n = 0; n < N2; ++n) { long long acc = 0; for (int64_t k = 0; k < K; ++k) acc += (long long)x[m * K + k] * (long long)wg[k * N2 + n]; t[n] = (double)acc; }
            for (int64_t n = 0; n < N; ++n) g[m * N + n] = silu((float)t[n]) * t[N + n];
        }
        row0 = row1;
    }
    quant_rows_dyn(g, M, N, o, sco);
    return ACLNN_SUCCESS;
}

// ---- QuantConvolution: x[N,Cin,H,W] int8, weight[Cout,Cin,kH,kW] int8, bias[Cout] fp32 ----
// out[N,Cout,oH,oW] = scale * (int32 conv accum) + bias.
aclnnStatus run_quant_conv(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *W = e->b, *B = e->c; aclTensor *O = e->out; double scale = e->alpha;
    int64_t s0 = e->stride[0], s1 = e->stride[1], p0 = e->pad[0], p1 = e->pad[1], d0 = e->dil[0], d1 = e->dil[1];
    int64_t N = X->viewDims[0], Cin = X->viewDims[1], H = X->viewDims[2], Wd = X->viewDims[3];
    int64_t Cout = W->viewDims[0], kH = W->viewDims[2], kW = W->viewDims[3];
    int64_t oH = O->viewDims[2], oW = O->viewDims[3];
    const int8_t *x = CI8(X), *w = CI8(W); const float *b = B ? FP(B) : nullptr; float *o = FPW(O);
    for (int64_t n = 0; n < N; ++n) for (int64_t co = 0; co < Cout; ++co) for (int64_t oh = 0; oh < oH; ++oh) for (int64_t ow = 0; ow < oW; ++ow) {
        long long acc = 0;
        for (int64_t ci = 0; ci < Cin; ++ci) for (int64_t kh = 0; kh < kH; ++kh) for (int64_t kw = 0; kw < kW; ++kw) {
            int64_t hh = oh * s0 - p0 + kh * d0, ww = ow * s1 - p1 + kw * d1; if (hh < 0 || hh >= H || ww < 0 || ww >= Wd) continue;
            acc += (long long)x[((n * Cin + ci) * H + hh) * Wd + ww] * (long long)w[((co * Cin + ci) * kH + kh) * kW + kw];
        }
        o[((n * Cout + co) * oH + oh) * oW + ow] = (float)((double)acc * scale + (b ? b[co] : 0.0));
    }
    return ACLNN_SUCCESS;
}

// ---- QuantFlashAttentionScore: q/k/v fp32 [B,Nh,S,D]; out = softmax(q@kᵀ * scaleValue [+causal mask])@v ----
aclnnStatus run_quant_fa(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *K = e->b, *V = e->c; aclTensor *O = e->out; double scale = e->alpha; bool causal = e->causal;
    int64_t B = Q->viewDims[0], Nh = Q->viewDims[1], Sq = Q->viewDims[2], D = Q->viewDims[3], Sk = K->viewDims[2];
    const float *q = FP(Q), *k = FP(K), *v = FP(V); float *o = FPW(O);
    std::vector<double> sc(Sk);
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Nh; ++h) {
        int64_t base = (b * Nh + h) * Sq * D, basek = (b * Nh + h) * Sk * D;
        for (int64_t i = 0; i < Sq; ++i) {
            double mx = -1e30; int64_t lim = causal ? std::min(Sk, i + 1 + (Sk - Sq)) : Sk;
            for (int64_t j = 0; j < Sk; ++j) { if (causal && j >= lim) { sc[j] = -1e30; continue; }
                double dot = 0; for (int64_t d = 0; d < D; ++d) dot += (double)q[base + i * D + d] * k[basek + j * D + d]; sc[j] = dot * scale; mx = std::max(mx, sc[j]); }
            double sum = 0; for (int64_t j = 0; j < Sk; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t j = 0; j < Sk; ++j) acc += sc[j] * v[basek + j * D + d]; o[base + i * D + d] = (float)(acc / sum); }
        }
    }
    return ACLNN_SUCCESS;
}

// ---- DequantRopeQuantKvcache (ROPE_DECL signature: x, cos, sin, mode, out) ----
// x fp32 [R,D] (already-dequantized q), cos/sin [R,D]; out fp32 [R,D] = rotate-half RoPE(x).
aclnnStatus run_dequant_rope_quant_kv(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Cos = e->b, *Sin = e->c; aclTensor *O = e->out;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D;
    const float *x = FP(X); const float *cs = FP(Cos), *sn = FP(Sin); float *o = FPW(O);
    int64_t half = D / 2;
    for (int64_t r = 0; r < R; ++r)
        for (int64_t d = 0; d < D; ++d) { double rot = (d < half) ? -(double)x[r * D + d + half] : (double)x[r * D + d - half];
            o[r * D + d] = (float)((double)x[r * D + d] * cs[r * D + d] + rot * sn[r * D + d]); }
    return ACLNN_SUCCESS;
}

// ---- SwinAttentionScoreQuant: q/k/v fp32 [B,Nh,S,D]; attention then per-row int8 quant ----
aclnnStatus run_swin_attn_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *K = e->b, *V = e->c; aclTensor *O = e->out, *Sco = e->out2; double scale = e->alpha;
    int64_t B = Q->viewDims[0], Nh = Q->viewDims[1], S = Q->viewDims[2], D = Q->viewDims[3];
    const float *q = FP(Q), *k = FP(K), *v = FP(V); int8_t *o = I8(O); float *sco = Sco ? FPW(Sco) : nullptr;
    int64_t R = B * Nh * S; std::vector<double> out(R * D); std::vector<double> sc(S);
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Nh; ++h) { int64_t base = (b * Nh + h) * S * D;
        for (int64_t i = 0; i < S; ++i) { double mx = -1e30;
            for (int64_t j = 0; j < S; ++j) { double dot = 0; for (int64_t d = 0; d < D; ++d) dot += (double)q[base + i * D + d] * k[base + j * D + d]; sc[j] = dot * scale; mx = std::max(mx, sc[j]); }
            double sum = 0; for (int64_t j = 0; j < S; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            int64_t orow = ((b * Nh + h) * S + i);
            for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t j = 0; j < S; ++j) acc += sc[j] * v[base + j * D + d]; out[orow * D + d] = acc / sum; } } }
    quant_rows_dyn(out, R, D, o, sco);
    return ACLNN_SUCCESS;
}

// ---- MoeInitRoutingQuant (MOEIR_DECL: x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx) ----
// Stable-sort rows by expert; gather into expandedX with per-row dynamic int8 quant (scale folded into the value
// via quant_rows_dyn — scale discarded as there is no scale output slot). expandedRowIdx = orig row;
// expandedExpertIdx = expert id of each expanded row.
aclnnStatus run_moe_init_routing_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Eidx = e->b; aclTensor *Ex = e->out, *Rowi = e->out2, *Eei = const_cast<aclTensor *>(e->mean);
    int64_t numExperts = e->k; int64_t T = X->viewDims[0], D = X->viewDims[1];
    const float *x = FP(X); const int32_t *eid = CI32(Eidx); int8_t *ex = I8(Ex);
    int32_t *ri = (int32_t *)Rowi->data + Rowi->offset; int32_t *ei = (int32_t *)Eei->data + Eei->offset;
    std::vector<double> gathered(T * D); std::vector<float> tmpscale(T); int64_t w = 0;
    for (int64_t exp = 0; exp < numExperts; ++exp) for (int64_t t = 0; t < T; ++t) if (eid[t] == (int32_t)exp) {
        for (int64_t d = 0; d < D; ++d) gathered[w * D + d] = x[t * D + d]; ri[w] = (int32_t)t; ei[w] = (int32_t)exp; ++w; }
    for (int64_t t = 0; t < T; ++t) if (eid[t] < 0 || eid[t] >= numExperts) { for (int64_t d = 0; d < D; ++d) gathered[w * D + d] = x[t * D + d]; ri[w] = (int32_t)t; ei[w] = eid[t]; ++w; }
    quant_rows_dyn(gathered, w, D, ex, tmpscale.data());
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
RUN(ex_ascend_quant_v3, run_ascend_quant_v3)
RUN(ex_ascend_antiquant, run_ascend_antiquant)
RUN(ex_group_quant, run_group_quant)
RUN(ex_gelu_quant, run_gelu_quant)
RUN(ex_gnsilu_quant, run_gnsilu_quant)
RUN(ex_fakeq_perchannel, run_fakeq_perchannel)
RUN(ex_qbn, run_qbn)
RUN(ex_dynamic_quant, run_dynamic_quant)
RUN(ex_block_quant, run_block_quant)
RUN(ex_mx_quant, run_mx_quant)
RUN(ex_mx_dualaxis, run_mx_dualaxis)
RUN(ex_swiglu_quant, run_swiglu_quant)
RUN(ex_dequant_swiglu_quant, run_dequant_swiglu_quant)
RUN(ex_addrms_quant, run_addrms_quant)
RUN(ex_addrms_dyn_quant, run_addrms_dyn_quant)
RUN(ex_addrms_mx_quant, run_addrms_mx_quant)
RUN(ex_adaln_quant, run_adaln_quant)
RUN(ex_addln_quant, run_addln_quant)
RUN(ex_swin_lnqkv_quant, run_swin_lnqkv_quant)
RUN(ex_trans_quant_param, run_trans_quant_param)
RUN(ex_apply_adamw_quant, run_apply_adamw_quant)
RUN(ex_qmm, run_qmm)
RUN(ex_qmm_iadd_g, run_qmm_iadd_g)
RUN(ex_wqmm, run_wqmm)
RUN(ex_gqmm, run_gqmm)
RUN(ex_gqmm_iadd, run_gqmm_iadd)
RUN(ex_gmm_swiglu_quant, run_gmm_swiglu_quant)
RUN(ex_quant_conv, run_quant_conv)
RUN(ex_quant_fa, run_quant_fa)
RUN(ex_dequant_rope_quant_kv, run_dequant_rope_quant_kv)
RUN(ex_swin_attn_quant, run_swin_attn_quant)
RUN(ex_moe_init_routing_quant, run_moe_init_routing_quant)
} // namespace

// =================================================================================================
extern "C" {

// ---- pure quant/dequant ----
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAscendQuantV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_ascend_quant_v3(w, wss, e, s); }

aclnnStatus aclnnAscendAntiQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAscendAntiQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_ascend_antiquant(w, wss, e, s); }

aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, int64_t groupSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->k = groupSize; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGroupQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_group_quant(w, wss, e, s); }

aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGeluQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gelu_quant(w, wss, e, s); }

aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->dim = group; e->eps = eps; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGroupNormSiluQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gnsilu_quant(w, wss, e, s); }

aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclTensor *scale, const aclTensor *zeroPoint, int64_t axis, int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = scale; e->c = zeroPoint; e->dim = axis; e->dscalars = {(double)quantMin, (double)quantMax}; e->out = out; e->out2 = mask; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_fakeq_perchannel(w, wss, e, s); }

aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd, double scale, int64_t zeroPoint, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mean || !invstd || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = weight; e->c = bias; e->mean = mean; e->rstd = invstd; e->alpha = scale; e->dscalars = {(double)zeroPoint}; e->eps = eps; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnQuantizedBatchNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_qbn(w, wss, e, s); }

// DynamicQuant V2/V3/V4 (per-token int8)
#define DQ_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_dynamic_quant(w, wss, e, s); }
DQ_DEF(aclnnDynamicQuantV2)
DQ_DEF(aclnnDynamicQuantV3)
DQ_DEF(aclnnDynamicQuantV4)
#undef DQ_DEF

aclnnStatus aclnnDynamicBlockQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDynamicBlockQuantV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_block_quant(w, wss, e, s); }

aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGroupedDynamicBlockQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_block_quant(w, wss, e, s); }

aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDynamicMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mx_quant(w, wss, e, s); }

aclnnStatus aclnnDynamicMxQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDynamicMxQuantV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mx_quant(w, wss, e, s); }

aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGroupedDynamicMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mx_quant(w, wss, e, s); }

aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, aclTensor *scaleOut2, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; e->mean = scaleOut2; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDynamicMxQuantWithDualAxis(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_mx_dualaxis(w, wss, e, s); }

aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSwiGluQuantV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swiglu_quant(w, wss, e, s); }

aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = dequantScale; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDequantSwigluQuantV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_dequant_swiglu_quant(w, wss, e, s); }

// ---- norm+quant fusions ----
#define ADDRMSQ_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double scale, double offset, double eps, aclTensor *yQuant, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !residual || !gamma || !yQuant || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->alpha = scale; e->dscalars = {offset}; e->eps = eps; e->out = yQuant; e->mean = residualSum; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms_quant(w, wss, e, s); }
ADDRMSQ_DEF(aclnnAddRmsNormQuant)
ADDRMSQ_DEF(aclnnAddRmsNormQuantV2)
#undef ADDRMSQ_DEF

aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *yQuant, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !yQuant || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->eps = eps; e->out = yQuant; e->out2 = scaleOut; e->mean = residualSum; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddRmsNormDynamicQuantV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms_dyn_quant(w, wss, e, s); }

aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->eps = eps; e->out = out; e->out2 = scaleOut; e->mean = residualSum; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddRmsNormDynamicMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addrms_mx_quant(w, wss, e, s); }

aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !shift || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = shift; e->eps = eps; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAdaLayerNormQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_adaln_quant(w, wss, e, s); }

aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !gamma || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->mask = beta; e->eps = eps; e->out = out; e->out2 = scaleOut; e->mean = residualSum; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddLayerNormQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_addln_quant(w, wss, e, s); }

aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = beta; e->eps = eps; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSwinTransformerLnQkvQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swin_lnqkv_quant(w, wss, e, s); }

// ---- TransQuantParam V2/V3 + ApplyAdamWQuant ----
aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = scale; e->b = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnTransQuantParamV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_trans_quant_param(w, wss, e, s); }

aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = scale; e->b = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnTransQuantParamV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_trans_quant_param(w, wss, e, s); }

aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = param; e->b = m; e->c = v; e->mean = grad; e->dscalars = {lr, beta1, beta2, eps, weightDecay, (double)step}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnApplyAdamWQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_apply_adamw_quant(w, wss, e, s); }

// ---- W8A8 quant matmul: V2/V4/V5 + WeightNz + Dequant + Fused + FusedWeightNz + MatmulCompressDequant ----
#define QMM_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_qmm(w, wss, e, s); }
QMM_DEF(aclnnQuantMatmulV2)
QMM_DEF(aclnnQuantMatmulV4)
QMM_DEF(aclnnQuantMatmulV5)
QMM_DEF(aclnnQuantMatmulWeightNz)
QMM_DEF(aclnnQuantMatmulDequant)
QMM_DEF(aclnnFusedQuantMatmul)
QMM_DEF(aclnnFusedQuantMatmulWeightNz)
QMM_DEF(aclnnMatmulCompressDequant)
QMM_DEF(aclnnSparse4to2QuantMatmulWeightNz)
QMM_DEF(aclnnDualLevelQuantMatmulWeightNz)
#undef QMM_DEF

// quant collective-matmul (canonical mc2.h signature carries an HcclComm). On a single rank the collective
// collapses to identity, so these are the local W8A8 QuantMatmul (matches the CUDA backend's QMM_FWD).
#define QMM_COMM_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        (void)comm; if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = scale; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_qmm(w, wss, e, s); }
QMM_COMM_DEF(aclnnAlltoAllQuantMatmul)
QMM_COMM_DEF(aclnnQuantMatmulAlltoAll)
QMM_COMM_DEF(aclnnQuantMatmulReduceSumWeightNz)
QMM_COMM_DEF(aclnnAlltoAllvQuantGroupedMatMul)
QMM_COMM_DEF(aclnnQuantGroupedMatMulAlltoAllv)
#undef QMM_COMM_DEF

// ---- W8A16/W4A16 weight-only matmul: Nz / V2 / V3 (antiquant scale + offset) ----
#define WQMM_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !weight || !antiquantScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = antiquantScale; e->out2 = const_cast<aclTensor *>(antiquantOffset); e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_wqmm(w, wss, e, s); }
WQMM_DEF(aclnnWeightQuantBatchMatmulNz)
WQMM_DEF(aclnnWeightQuantBatchMatmulV2)
WQMM_DEF(aclnnWeightQuantBatchMatmulV3)
#undef WQMM_DEF

// ---- grouped W8A8 quant matmul (Dequant / DequantWeightNZ / InplaceAdd / collective-collapsed) ----
#define GQMM_DEF(NAME, FN) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !weight || !scale || !groupList || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = scale; e->axes = groupList->v; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return FN(w, wss, e, s); }
GQMM_DEF(aclnnQuantGroupedMatmulDequant, ex_gqmm)
GQMM_DEF(aclnnQuantGroupedMatmulDequantWeightNZ, ex_gqmm)
GQMM_DEF(aclnnQuantGroupedMatmulInplaceAdd, ex_gqmm_iadd)
#undef GQMM_DEF

// ---- grouped matmul + swiglu + per-row int8 quant (+ V2 / WeightNZ / WeightNzV2) ----
#define GMM_SWQ_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !weight || !groupList || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->axes = groupList->v; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_gmm_swiglu_quant(w, wss, e, s); }
GMM_SWQ_DEF(aclnnGroupedMatmulSwigluQuant)
GMM_SWQ_DEF(aclnnGroupedMatmulSwigluQuantV2)
GMM_SWQ_DEF(aclnnGroupedMatmulSwigluQuantWeightNZ)
GMM_SWQ_DEF(aclnnGroupedMatmulSwigluQuantWeightNzV2)
#undef GMM_SWQ_DEF

// ---- QuantConvolution (+ WeightNz) ----
#define QCONV_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, double scale, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        (void)groups; if (!input || !weight || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = input; e->b = weight; e->c = bias; e->alpha = scale; e->out = out; \
        e->stride[0] = stride && stride->v.size() > 0 ? stride->v[0] : 1; e->stride[1] = stride && stride->v.size() > 1 ? stride->v[1] : e->stride[0]; \
        e->pad[0] = padding && padding->v.size() > 0 ? padding->v[0] : 0; e->pad[1] = padding && padding->v.size() > 1 ? padding->v[1] : e->pad[0]; \
        e->dil[0] = dilation && dilation->v.size() > 0 ? dilation->v[0] : 1; e->dil[1] = dilation && dilation->v.size() > 1 ? dilation->v[1] : e->dil[0]; \
        *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_quant_conv(w, wss, e, s); }
QCONV_DEF(aclnnQuantConvolution)
QCONV_DEF(aclnnQuantConvolutionWeightNz)
#undef QCONV_DEF

// ---- QuantFlashAttentionScore ----
aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)attenMask; (void)headNum; if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->alpha = scaleValue; e->causal = causal; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnQuantFlashAttentionScore(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_quant_fa(w, wss, e, s); }

// ---- DequantRopeQuantKvcache (ROPE_DECL: x, cos, sin, mode, out) ----
aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)mode; if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = cos; e->c = sin; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDequantRopeQuantKvcache(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_dequant_rope_quant_kv(w, wss, e, s); }

// ---- SwinAttentionScoreQuant (ATT_DECL: q, k, v, attenMask, scaleValue, headNum, causal, out) ----
aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)attenMask; (void)headNum; (void)causal; if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->alpha = scaleValue; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSwinAttentionScoreQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swin_attn_quant(w, wss, e, s); }

// ---- MoeInitRoutingQuant (+V2) (MOEIR_DECL: x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx) ----
#define MOEQ_DEF(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !expertIdx || !expandedX || !expandedRowIdx || !expandedExpertIdx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = expertIdx; e->k = numExperts; e->out = expandedX; e->out2 = expandedRowIdx; e->mean = expandedExpertIdx; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_moe_init_routing_quant(w, wss, e, s); }
MOEQ_DEF(aclnnMoeInitRoutingQuant)
MOEQ_DEF(aclnnMoeInitRoutingQuantV2)
#undef MOEQ_DEF

} // extern "C"
