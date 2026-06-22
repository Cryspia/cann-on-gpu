// Quantization extensions (P14) — host-side over unified memory (MTLStorageModeShared => device ptr == host ptr).
// Covers the test_quant_ext spec: AscendQuant/AscendDequant (per-channel int8 affine), DequantBias (int32->fp32),
// FakeQuant (scalar affine, fake-quantized fp32), DynamicBlockMxQuant / DynamicDualLevelMxQuant (per-block MX
// power-of-2 int8), FlatQuant (per-row absmax int8), SwigluMxQuant (swiglu then per-row MX int8),
// InplaceQuantScatter (scatter-quantize rows), TransQuantParam (pack fp32 scale bits into int64 low word).
// Scale math mirrors norm_ext.mm: dynamic uses per-row absmax/127; MX uses exp2(ceil(log2(amax/127))).
// Only ops currently undefined for test_quant_ext are defined here (no overlap with quant.mm / blas.mm / norm_ext.mm).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
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
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

inline int clamp_i(long v, int lo, int hi) { return (int)(v < lo ? lo : (v > hi ? hi : v)); }
inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
// MX power-of-2 scale from a block absmax (matches RmsNormDynamicMxQuant in norm_ext.mm).
inline double mx_scale(double amax) { return amax > 0 ? std::exp2(std::ceil(std::log2(amax / 127.0))) : 1.0; }

// ---------- AscendQuant: q = clamp(round(x*scale[c] + offset[c]), -128, 127), per last-dim channel ----------
aclnnStatus run_ascend_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
    int64_t C = Sc->numel(), n = X->numel(); const float *x = FP(X), *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; int8_t *q = I8(O);
    for (int64_t i = 0; i < n; ++i) { int64_t c = i % C; double v = (double)x[i] * sc[c] + (of ? of[c] : 0.0);
        q[i] = (int8_t)clamp_i(std::lrint(v), -128, 127); }
    return ACLNN_SUCCESS;
}
// ---------- AscendDequant: y = (q - offset[c]) * scale[c], per last-dim channel ----------
aclnnStatus run_ascend_dequant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
    int64_t C = Sc->numel(), n = Q->numel(); const int8_t *q = CI8(Q); const float *sc = FP(Sc), *of = Of ? FP(Of) : nullptr; float *y = FPW(O);
    for (int64_t i = 0; i < n; ++i) { int64_t c = i % C; y[i] = (float)(((double)q[i] - (of ? of[c] : 0.0)) * sc[c]); }
    return ACLNN_SUCCESS;
}
// ---------- DequantBias: y = q(int32) * scale[c] + bias[c], per last-dim channel ----------
aclnnStatus run_dequant_bias(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *Sc = e->b, *B = e->c; aclTensor *O = e->out;
    int64_t C = Sc->numel(), n = Q->numel(); const int32_t *q = CI32(Q); const float *sc = FP(Sc), *b = B ? FP(B) : nullptr; float *y = FPW(O);
    for (int64_t i = 0; i < n; ++i) { int64_t c = i % C; y[i] = (float)((double)q[i] * sc[c] + (b ? b[c] : 0.0)); }
    return ACLNN_SUCCESS;
}
// ---------- FakeQuant: q = clamp(round(x/scale + zp), qmin, qmax); y = (q - zp)*scale (scalar params) ----------
aclnnStatus run_fake_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *O = e->out;
    double scale = e->dscalars[0], zp = e->dscalars[1]; int qmin = (int)e->dscalars[2], qmax = (int)e->dscalars[3];
    int64_t n = X->numel(); const float *x = FP(X); float *y = FPW(O);
    for (int64_t i = 0; i < n; ++i) { int q = clamp_i(std::lrint((double)x[i] / scale + zp), qmin, qmax); y[i] = (float)(((double)q - zp) * scale); }
    return ACLNN_SUCCESS;
}
// ---------- DynamicBlockMxQuant / DynamicDualLevelMxQuant: per-block (last-dim of size blk) MX power-of-2 int8 ----------
// m: 0 single-level (scaleOut = mx scale), 1 dual-level (scaleL1 = mx scale, scaleL2 = 1 so q*s1*s2 == q*s1).
aclnnStatus run_block_mx(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *Q = e->out, *S1 = e->out2; aclTensor *S2 = e->mean ? const_cast<aclTensor *>(e->mean) : nullptr;
    int64_t blk = e->k; int64_t nblk = X->numel() / blk; const float *x = FP(X); int8_t *q = I8(Q); float *s1 = FPW(S1); float *s2 = S2 ? FPW(S2) : nullptr;
    for (int64_t b = 0; b < nblk; ++b) {
        double amax = 0; for (int64_t i = 0; i < blk; ++i) amax = std::max(amax, std::fabs((double)x[b * blk + i]));
        double sc = mx_scale(amax); s1[b] = (float)sc; if (s2) s2[b] = 1.0f; double qi = 1.0 / sc;
        for (int64_t i = 0; i < blk; ++i) q[b * blk + i] = (int8_t)clamp_i(std::lrint((double)x[b * blk + i] * qi), -128, 127);
    }
    return ACLNN_SUCCESS;
}
// ---------- FlatQuant: per-row (last dim) symmetric int8, scale = amax/127 ----------
aclnnStatus run_flat_quant(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *X = e->a; aclTensor *Q = e->out, *Sc = e->out2;
    int64_t D = X->viewDims.back(); int64_t R = X->numel() / D; const float *x = FP(X); int8_t *q = I8(Q); float *sc = FPW(Sc);
    for (int64_t r = 0; r < R; ++r) {
        double amax = 0; for (int64_t d = 0; d < D; ++d) amax = std::max(amax, std::fabs((double)x[r * D + d]));
        double s = amax > 0 ? amax / 127.0 : 1.0; sc[r] = (float)s; double qi = 1.0 / s;
        for (int64_t d = 0; d < D; ++d) q[r * D + d] = (int8_t)clamp_i(std::lrint((double)x[r * D + d] * qi), -128, 127);
    }
    return ACLNN_SUCCESS;
}
// ---------- SwigluMxQuant: in[R,2D] -> g = silu(a)*b over [R,D]; per-row MX power-of-2 int8 ----------
aclnnStatus run_swiglu_mx(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *In = e->a; aclTensor *Q = e->out, *Sc = e->out2;
    int64_t D = Q->viewDims.back(); int64_t R = In->numel() / (2 * D); const float *x = FP(In); int8_t *q = I8(Q); float *sc = FPW(Sc);
    std::vector<double> g(D);
    for (int64_t r = 0; r < R; ++r) {
        double amax = 0; for (int64_t d = 0; d < D; ++d) { double a = x[r * 2 * D + d], b = x[r * 2 * D + D + d]; double gv = silu((float)a) * b; g[d] = gv; amax = std::max(amax, std::fabs(gv)); }
        double s = mx_scale(amax); sc[r] = (float)s; double qi = 1.0 / s;
        for (int64_t d = 0; d < D; ++d) q[r * D + d] = (int8_t)clamp_i(std::lrint(g[d] * qi), -128, 127);
    }
    return ACLNN_SUCCESS;
}
// ---------- InplaceQuantScatter: self[idx[k]][d] = clamp(round(upd[k][d]/scale), -127, 127) ----------
aclnnStatus run_quant_scatter(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *Self = const_cast<aclTensor *>(e->a); const aclTensor *Idx = e->b, *Upd = e->c; double scale = e->alpha;
    int64_t D = Upd->viewDims.back(); int64_t K = Idx->numel(); int8_t *self = I8(Self); const int64_t *idx = CI64(Idx); const float *upd = FP(Upd);
    double qi = 1.0 / scale;
    for (int64_t k = 0; k < K; ++k) { int64_t row = idx[k];
        for (int64_t d = 0; d < D; ++d) self[row * D + d] = (int8_t)clamp_i(std::lrint((double)upd[k * D + d] * qi), -127, 127); }
    return ACLNN_SUCCESS;
}
// ---------- TransQuantParam: out(int64) low32 = float bits of scale[i] (offset folded into high bits if present) ----------
aclnnStatus run_trans_quant_param(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Sc = e->a, *Of = e->b; aclTensor *O = e->out;
    int64_t n = Sc->numel(); const float *sc = FP(Sc); const float *of = Of ? FP(Of) : nullptr; int64_t *o = I64(O);
    for (int64_t i = 0; i < n; ++i) { uint32_t lo; float f = sc[i]; std::memcpy(&lo, &f, 4);
        uint64_t hi = 0; if (of) { uint32_t ob; float fo = of[i]; std::memcpy(&ob, &fo, 4); hi = (uint64_t)ob; }
        o[i] = (int64_t)((hi << 32) | (uint64_t)lo); }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
RUN(ex_ascend_quant, run_ascend_quant)
RUN(ex_ascend_dequant, run_ascend_dequant)
RUN(ex_dequant_bias, run_dequant_bias)
RUN(ex_fake_quant, run_fake_quant)
RUN(ex_block_mx, run_block_mx)
RUN(ex_flat_quant, run_flat_quant)
RUN(ex_swiglu_mx, run_swiglu_mx)
RUN(ex_quant_scatter, run_quant_scatter)
RUN(ex_trans_quant_param, run_trans_quant_param)
} // namespace

extern "C" {
aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAscendQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_ascend_quant(w, wss, e, s); }

aclnnStatus aclnnAscendDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAscendDequant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_ascend_dequant(w, wss, e, s); }

aclnnStatus aclnnDequantBiasGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDequantBias(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_dequant_bias(w, wss, e, s); }

aclnnStatus aclnnFakeQuantGetWorkspaceSize(const aclTensor *x, double scale, double zeroPoint, int64_t qmin, int64_t qmax, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->dscalars = {scale, zeroPoint, (double)qmin, (double)qmax}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFakeQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_fake_quant(w, wss, e, s); }

aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleOut; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicBlockMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_block_mx(w, wss, e, s); }

aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleL1, aclTensor *scaleL2, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleL1 || !scaleL2 || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->k = blockSize; e->out = out; e->out2 = scaleL1; e->mean = scaleL2; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicDualLevelMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_block_mx(w, wss, e, s); }

aclnnStatus aclnnFlatQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlatQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_flat_quant(w, wss, e, s); }

aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSwigluMxQuant(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_swiglu_mx(w, wss, e, s); }

aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !indices || !updates || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = selfRef; e->b = indices; e->c = updates; e->alpha = scale; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceQuantScatter(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_quant_scatter(w, wss, e, s); }

aclnnStatus aclnnTransQuantParamGetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = scale; e->b = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransQuantParam(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return ex_trans_quant_param(w, wss, e, s); }
} // extern "C"
