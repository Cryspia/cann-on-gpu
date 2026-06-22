// Elementwise / misc operator gap fill: parity with the CUDA backend for the elementwise/misc/activation/
// generator families that had not yet landed on Metal. All ops execute host-side over unified memory
// (device ptr == host ptr under MTLStorageModeShared) — these are bandwidth-light pointwise/small ops, so a
// plain CPU loop matches the existing remainder_math.mm / ops_ext.mm style and keeps semantics exact.
//
// Signatures are validated against the canonical macro/explicit declarations in aclnnop/aclnn_ops.h
// (and aclnn_add.h). Pure-alias ops forward to already-exported base symbols.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float  *FP (const aclTensor *t) { return (float *)t->data + t->offset; }
int32_t *IP32(const aclTensor *t) { return (int32_t *)t->data + t->offset; }
int64_t *IP64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
uint8_t *UP8 (const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// load any of {f32,f16,bf16,i32} as float for mixed-friendly reads
inline float ld(const aclTensor *t, int64_t i) {
    switch (t->dtype) {
        case ACL_FLOAT16: return (float)(((const __fp16 *)t->data + t->offset)[i]);
        case ACL_INT32:   return (float)(((const int32_t *)t->data + t->offset)[i]);
        default:          return ((const float *)t->data + t->offset)[i];
    }
}
inline void st(const aclTensor *t, int64_t i, float v) {
    switch (t->dtype) {
        case ACL_FLOAT16: ((__fp16 *)t->data + t->offset)[i] = (__fp16)v; break;
        case ACL_INT32:   ((int32_t *)t->data + t->offset)[i] = (int32_t)v; break;
        default:          ((float *)t->data + t->offset)[i] = v; break;
    }
}
// PCG-style hash matching ops_ext.mm RNG for statistically-checkable Bernoulli.
inline uint32_t pcg(uint32_t v) { uint32_t s = v * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
inline float u01(uint32_t &s) { s = s * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; w = (w >> 22u) ^ w; return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }

// ---- dispatch kinds ----
enum {
    // unary float (self -> out)
    G_SOFTSIGN, G_REAL, G_ANGLE,
    // unary float with one scalar param
    G_SCALE,
    // binary float (a,b -> out)
    G_FMOD, G_XLOGY, G_SILUMUL, G_GELUMUL, G_FLOORDIV, G_REALDIV, G_DIVV3,
    G_CLAMPMAX_T, G_CLAMPMIN_T, G_LOGICALXOR, G_COMPLEX, G_POLAR, G_MAXN, G_MINN,
    // binary float + scalar weight
    G_LERPS, G_ISCLOSE,
    // scalar (op) tensor / tensor (op) scalar -> out
    G_POW_ST, G_REM_ST, G_XLOGY_SO, G_XLOGY_SS, G_ISIN_TS, G_ISIN_ST,
    G_FLOORDIVIDES, G_RSUBS,
    // int32 binary / scalar-shift
    G_LSHIFT, G_RSHIFT, G_LSHIFTS,
    // bool out predicate
    G_SIGNBIT,
    // generators / misc
    G_GLU, G_MODULATE, G_MINDIM, G_VIEWCOPY, G_BERNOULLI_T,
};

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {

    // -------- unary float --------
    case G_SOFTSIGN: {
        int64_t n = e->out->numel();
        for (int64_t i = 0; i < n; i++) { double v = ld(e->a, i); st(e->out, i, (float)(v / (1.0 + std::fabs(v)))); }
        return ACLNN_SUCCESS;
    }
    case G_REAL: {   // interleaved complex [...,2] -> real component
        int64_t n = e->out->numel(); const float *c = FP(e->a); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = c[2 * i];
        return ACLNN_SUCCESS;
    }
    case G_ANGLE: {  // interleaved complex (re,im) pairs -> atan2(im,re)
        int64_t n = e->out->numel(); const float *c = FP(e->a); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = std::atan2(c[2 * i + 1], c[2 * i]);
        return ACLNN_SUCCESS;
    }
    case G_SIGNBIT: {  // out bool/uint8: sign bit set (negative, incl. -0.0)
        int64_t n = e->out->numel(); uint8_t *o = UP8(e->out);
        for (int64_t i = 0; i < n; i++) { float v = ld(e->a, i); o[i] = (uint8_t)(std::signbit(v) ? 1 : 0); }
        return ACLNN_SUCCESS;
    }

    // -------- unary float + scalars --------
    case G_SCALE: {  // out = self*scale + bias
        int64_t n = e->out->numel(); double sc = e->alpha, bi = e->dscalars[0];
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)(ld(e->a, i) * sc + bi));
        return ACLNN_SUCCESS;
    }

    // -------- binary float (no broadcast; equal-shape) --------
    case G_FMOD: case G_XLOGY: case G_SILUMUL: case G_GELUMUL: case G_FLOORDIV:
    case G_REALDIV: case G_DIVV3: case G_CLAMPMAX_T: case G_CLAMPMIN_T:
    case G_MAXN: case G_MINN: {
        int64_t n = e->out->numel();
        for (int64_t i = 0; i < n; i++) {
            double x = ld(e->a, i), y = ld(e->b, i), r;
            switch (e->op) {
            case G_FMOD:       r = std::fmod(x, y); break;
            case G_XLOGY:      r = (x == 0.0) ? 0.0 : x * std::log(y); break;
            case G_SILUMUL:    r = (x / (1.0 + std::exp(-x))) * y; break;
            case G_GELUMUL:    r = (0.5 * x * std::erfc(-x * 0.70710678118654752440)) * y; break;
            case G_FLOORDIV:   r = std::floor(x / y); break;
            case G_REALDIV: case G_DIVV3: r = x / y; break;
            case G_CLAMPMAX_T: r = x < y ? x : y; break;   // clamp to max bound -> min(self,other)
            case G_CLAMPMIN_T: r = x > y ? x : y; break;   // clamp to min bound -> max(self,other)
            case G_MAXN:       r = x > y ? x : y; break;
            default:           r = x < y ? x : y; break;   // G_MINN
            }
            st(e->out, i, (float)r);
        }
        return ACLNN_SUCCESS;
    }
    case G_LOGICALXOR: {  // bool in/out
        int64_t n = e->out->numel(); const uint8_t *a = UP8(e->a), *b = UP8(e->b); uint8_t *o = UP8(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = (uint8_t)((a[i] != 0) ^ (b[i] != 0));
        return ACLNN_SUCCESS;
    }
    case G_ISCLOSE: {  // bool out: |a-b| <= atol + rtol*|b|
        int64_t n = e->out->numel(); double rt = e->dscalars[0], at = e->dscalars[1]; uint8_t *o = UP8(e->out);
        for (int64_t i = 0; i < n; i++) { double x = ld(e->a, i), y = ld(e->b, i); o[i] = (uint8_t)(std::fabs(x - y) <= at + rt * std::fabs(y)); }
        return ACLNN_SUCCESS;
    }
    case G_COMPLEX: {  // (re,im) -> interleaved out[...,2]
        int64_t n = e->a->numel(); const float *re = FP(e->a), *im = FP(e->b); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) { o[2 * i] = re[i]; o[2 * i + 1] = im[i]; }
        return ACLNN_SUCCESS;
    }
    case G_POLAR: {  // (abs,angle) -> interleaved (abs*cos, abs*sin)
        int64_t n = e->a->numel(); const float *ab = FP(e->a), *an = FP(e->b); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) { o[2 * i] = ab[i] * std::cos(an[i]); o[2 * i + 1] = ab[i] * std::sin(an[i]); }
        return ACLNN_SUCCESS;
    }
    case G_LERPS: {  // out = a + w*(b-a)
        int64_t n = e->out->numel(); double w = e->alpha;
        for (int64_t i = 0; i < n; i++) { double x = ld(e->a, i), y = ld(e->b, i); st(e->out, i, (float)(x + w * (y - x))); }
        return ACLNN_SUCCESS;
    }

    // -------- scalar / tensor mixes --------
    case G_POW_ST: {  // pow(scalarSelf, tensorExp)
        int64_t n = e->out->numel(); double sv = e->alpha;
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)std::pow(sv, (double)ld(e->b, i)));
        return ACLNN_SUCCESS;
    }
    case G_REM_ST: {  // remainder(scalarSelf, tensorOther) = s - floor(s/t)*t
        int64_t n = e->out->numel(); double sv = e->alpha;
        for (int64_t i = 0; i < n; i++) { double y = ld(e->b, i); st(e->out, i, (float)(sv - std::floor(sv / y) * y)); }
        return ACLNN_SUCCESS;
    }
    case G_XLOGY_SO: {  // self_tensor * log(scalar_other)
        int64_t n = e->out->numel(); double c = std::log(e->alpha);
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)(ld(e->a, i) * c));
        return ACLNN_SUCCESS;
    }
    case G_XLOGY_SS: {  // scalar_self * log(other_tensor)
        int64_t n = e->out->numel(); double sv = e->alpha;
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)(sv * std::log((double)ld(e->b, i))));
        return ACLNN_SUCCESS;
    }
    case G_FLOORDIVIDES: {  // floor(self / scalar)
        int64_t n = e->out->numel(); double c = e->alpha;
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)std::floor(ld(e->a, i) / c));
        return ACLNN_SUCCESS;
    }
    case G_RSUBS: {  // scalar - self
        int64_t n = e->out->numel(); double c = e->alpha;
        for (int64_t i = 0; i < n; i++) st(e->out, i, (float)(c - ld(e->a, i)));
        return ACLNN_SUCCESS;
    }
    case G_ISIN_TS: {  // out[i] = (tensor[i] == scalar)
        int64_t n = e->out->numel(); float v = (float)e->alpha; uint8_t *o = UP8(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = (uint8_t)(ld(e->a, i) == v);
        return ACLNN_SUCCESS;
    }
    case G_ISIN_ST: {  // out[0] = scalar present in tensor
        int64_t n = e->a->numel(); float v = (float)e->alpha; uint8_t f = 0;
        for (int64_t i = 0; i < n; i++) if (ld(e->a, i) == v) { f = 1; break; }
        UP8(e->out)[0] = f;
        return ACLNN_SUCCESS;
    }

    // -------- int32 shifts --------
    case G_LSHIFT: case G_RSHIFT: {
        int64_t n = e->out->numel(); const int32_t *a = IP32(e->a), *b = IP32(e->b); int32_t *o = IP32(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = (e->op == G_LSHIFT) ? (a[i] << b[i]) : (a[i] >> b[i]);
        return ACLNN_SUCCESS;
    }
    case G_LSHIFTS: {  // self << scalar
        int64_t n = e->out->numel(); const int32_t *a = IP32(e->a); int32_t *o = IP32(e->out); int sh = (int)e->dim;
        for (int64_t i = 0; i < n; i++) o[i] = a[i] << sh;
        return ACLNN_SUCCESS;
    }

    // -------- generators / misc --------
    case G_GLU: {  // in[...,2D] -> out[...,D] ; out = first[d] * sigmoid(second[d])
        int64_t rows = e->outerCount, D = e->reduceCount; const float *x = FP(e->a); float *o = FP(e->out);
        for (int64_t r = 0; r < rows; r++) { const float *p = x + r * 2 * D;
            for (int64_t d = 0; d < D; d++) o[r * D + d] = p[d] * (1.f / (1.f + std::exp(-p[D + d]))); }
        return ACLNN_SUCCESS;
    }
    case G_MODULATE: {  // out = x*(1+scale) + shift
        int64_t n = e->out->numel(); const float *x = FP(e->a), *sc = FP(e->b), *sh = FP(e->c); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) o[i] = x[i] * (1.f + sc[i]) + sh[i];
        return ACLNN_SUCCESS;
    }
    case G_MINDIM: {  // (values,indices) of min over dim
        int dim = (int)e->dim; int nd = (int)e->a->viewDims.size(); if (dim < 0) dim += nd;
        int64_t outer = 1, D = e->a->viewDims[dim], inner = 1;
        for (int d = 0; d < dim; d++) outer *= e->a->viewDims[d];
        for (int d = dim + 1; d < nd; d++) inner *= e->a->viewDims[d];
        const float *x = FP(e->a); float *v = FP(e->out); int64_t *idx = IP64(e->out2);
        for (int64_t o = 0; o < outer; o++) for (int64_t ii = 0; ii < inner; ii++) {
            float best = x[(o * D + 0) * inner + ii]; int64_t bi = 0;
            for (int64_t d = 1; d < D; d++) { float c = x[(o * D + d) * inner + ii]; if (c < best) { best = c; bi = d; } }
            v[o * inner + ii] = best; idx[o * inner + ii] = bi;
        }
        return ACLNN_SUCCESS;
    }
    case G_VIEWCOPY: {  // squeeze/unsqueeze: contiguous reinterpret -> byte copy
        size_t bytes = (size_t)e->a->numel() * dtype_size(e->a->dtype);
        std::memmove(e->out->data, e->a->data, bytes);
        return ACLNN_SUCCESS;
    }
    case G_BERNOULLI_T: {  // out[i] ~ Bernoulli(prob[i]); seed in e->dim
        int64_t n = e->out->numel(); uint32_t seed = (uint32_t)(int64_t)e->dim; const float *p = FP(e->a); float *o = FP(e->out);
        for (int64_t i = 0; i < n; i++) { uint32_t s2 = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u)); o[i] = u01(s2) < p[i] ? 1.f : 0.f; }
        return ACLNN_SUCCESS;
    }

    default: return ACLNN_ERR_PARAM_INVALID;
    }
}

#define RUNG(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus r; @autoreleasepool { r = run(e, s); } delete e; return r; }
RUNG(g_run)
} // namespace

extern "C" {

// ============================= unary float =============================
#define UN(NAME, KIND) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->op = KIND; e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(NAME)
UN(aclnnSoftsign, G_SOFTSIGN)
UN(aclnnReal,     G_REAL)
UN(aclnnAngleV2,  G_ANGLE)

aclnnStatus aclnnSignbitGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || (out->dtype != ACL_BOOL && out->dtype != ACL_UINT8)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = G_SIGNBIT; e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnSignbit)

// ============================= unary + scalars =============================
aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *bias, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !scale || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_SCALE; e->a = self; e->out = out; e->alpha = scale->v; e->dscalars = {bias ? bias->v : 0.0}; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnScale)

// ============================= binary float (equal-shape) =============================
#define BIN(NAME, KIND) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->op = KIND; e->a = self; e->b = other; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(NAME)
BIN(aclnnFmodTensor,     G_FMOD)
BIN(aclnnXLogYTensor,    G_XLOGY)
BIN(aclnnSiluMul,        G_SILUMUL)
BIN(aclnnGeluMul,        G_GELUMUL)
BIN(aclnnFloorDiv,       G_FLOORDIV)
BIN(aclnnRealDiv,        G_REALDIV)
BIN(aclnnDivV3,          G_DIVV3)
BIN(aclnnClampMaxTensor, G_CLAMPMAX_T)
BIN(aclnnClampMinTensor, G_CLAMPMIN_T)
BIN(aclnnLogicalXor,     G_LOGICALXOR)
BIN(aclnnComplex,        G_COMPLEX)
BIN(aclnnPolar,          G_POLAR)

// int32 shifts
BIN(aclnnLeftShift,  G_LSHIFT)
BIN(aclnnRightShift, G_RSHIFT)

aclnnStatus aclnnLeftShiftsGetWorkspaceSize(const aclTensor *self, const aclScalar *shift, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !shift || !out || !ex || out->dtype != ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = G_LSHIFTS; e->a = self; e->out = out; e->dim = (int64_t)shift->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnLeftShifts)

// IsClose
aclnnStatus aclnnIsCloseGetWorkspaceSize(const aclTensor *self, const aclTensor *other, double rtol, double atol, bool equalNan, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)equalNan; if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_ISCLOSE; e->a = self; e->b = other; e->out = out; e->dscalars = {rtol, atol}; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnIsClose)

// Lerps (scalar weight)
aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor *self, const aclTensor *end, const aclScalar *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !end || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_LERPS; e->a = self; e->b = end; e->out = out; e->alpha = weight->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnLerps)

// ============================= scalar/tensor mixes =============================
aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *exponent, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !exponent || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_POW_ST; e->b = exponent; e->out = out; e->alpha = self->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnPowScalarTensor)

aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_REM_ST; e->b = other; e->out = out; e->alpha = self->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnRemainderScalarTensor)

aclnnStatus aclnnXLogYScalarOtherGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_XLOGY_SO; e->a = self; e->out = out; e->alpha = other->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnXLogYScalarOther)

aclnnStatus aclnnXLogYScalarSelfGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_XLOGY_SS; e->b = other; e->out = out; e->alpha = self->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnXLogYScalarSelf)

aclnnStatus aclnnFloorDividesGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_FLOORDIVIDES; e->a = self; e->out = out; e->alpha = other->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnFloorDivides)

aclnnStatus aclnnRsubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_RSUBS; e->a = self; e->out = out; e->alpha = other->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnRsubs)

aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *element, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !element || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_ISIN_TS; e->a = self; e->out = out; e->alpha = element->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnIsInTensorScalar)

aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(const aclScalar *element, const aclTensor *testElements, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!element || !testElements || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_ISIN_ST; e->a = testElements; e->out = out; e->alpha = element->v; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnIsInScalarTensor)

// ============================= n-ary max/min over tensor list =============================
static aclnnStatus naryN(const aclTensorList *tl, aclTensor *out, int kind, uint64_t *ws, aclOpExecutor **ex) {
    if (!tl || !out || !ex || tl->v.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = kind; e->out = out; for (auto t : tl->v) e->inputs.push_back(t); if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus naryRun(aclOpExecutor *e, aclrtStream s) {
    drain(s); int K = (int)e->inputs.size(); int64_t n = e->out->numel(); bool mx = (e->op == G_MAXN); float *o = FP(e->out);
    for (int64_t i = 0; i < n; i++) {
        double acc = ld(e->inputs[0], i);
        for (int k = 1; k < K; k++) { double v = ld(e->inputs[k], i); acc = mx ? std::max(acc, v) : std::min(acc, v); }
        o[i] = (float)acc;
    }
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaxNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return naryN(tensors, out, G_MAXN, ws, ex); }
aclnnStatus aclnnMaxN(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus r; @autoreleasepool { r = naryRun(e, s); } delete e; return r; }
aclnnStatus aclnnMinNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return naryN(tensors, out, G_MINN, ws, ex); }
aclnnStatus aclnnMinN(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus r; @autoreleasepool { r = naryRun(e, s); } delete e; return r; }

// ============================= generators / misc =============================
aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !out || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = out->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = G_GLU; e->a = self; e->out = out; e->reduceCount = D; e->outerCount = out->numel() / D; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnGlu)

aclnnStatus aclnnModulateGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !shift || !out || !ex) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = G_MODULATE; e->a = x; e->b = scale; e->c = shift; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnModulate)

aclnnStatus aclnnMinDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; if (!self || !valuesOut || !indicesOut || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_MINDIM; e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnMinDim)

// Squeeze / Unsqueeze: pure view reshape over contiguous data -> byte copy
aclnnStatus aclnnSqueezeGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_VIEWCOPY; e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnSqueeze)
aclnnStatus aclnnUnsqueezeGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_VIEWCOPY; e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnUnsqueeze)

// BernoulliTensor: out ~ Bernoulli(prob) with per-element prob tensor
aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(aclTensor *out, const aclTensor *prob, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !prob || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_BERNOULLI_T; e->a = prob; e->out = out; e->dim = seed; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNG(aclnnBernoulliTensor)

// ============================= pure-alias forwards =============================
// AddV3 == Add(self, other, alpha)
aclnnStatus aclnnAddV3GetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddGetWorkspaceSize(self, other, alpha, out, ws, ex); }
aclnnStatus aclnnAddV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAdd(w, wss, e, s); }

// FastGeluV2 == FastGelu (tanh-approx GELU)
aclnnStatus aclnnFastGeluV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnFastGeluGetWorkspaceSize(self, out, ws, ex); }
aclnnStatus aclnnFastGeluV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnFastGelu(w, wss, e, s); }

// GeGluV3 == GeGlu (gelu-gated GLU on [...,2D])
aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGeGluGetWorkspaceSize(self, out, ws, ex); }
aclnnStatus aclnnGeGluV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnGeGlu(w, wss, e, s); }

// Hstack/Dstack == Cat along dim 1 / dim 2
aclnnStatus aclnnHstackGetWorkspaceSize(const aclTensor *const *t, uint64_t num, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnCatGetWorkspaceSize(t, num, 1, out, ws, ex); }
aclnnStatus aclnnHstack(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnCat(w, wss, e, s); }
aclnnStatus aclnnDstackGetWorkspaceSize(const aclTensor *const *t, uint64_t num, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnCatGetWorkspaceSize(t, num, 2, out, ws, ex); }
aclnnStatus aclnnDstack(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnCat(w, wss, e, s); }

// Vdot/Inner (1-D) == Dot
aclnnStatus aclnnVdotGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDotGetWorkspaceSize(self, mat2, out, ws, ex); }
aclnnStatus aclnnVdot(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnDot(w, wss, e, s); }
aclnnStatus aclnnInnerGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDotGetWorkspaceSize(self, mat2, out, ws, ex); }
aclnnStatus aclnnInner(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnDot(w, wss, e, s); }

// MaxV2 == Amax (full reduce), Min == Amin (full reduce)
aclnnStatus aclnnMaxV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAmaxGetWorkspaceSize(self, nullptr, false, out, ws, ex); }
aclnnStatus aclnnMaxV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAmax(w, wss, e, s); }
aclnnStatus aclnnMinGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAminGetWorkspaceSize(self, nullptr, false, out, ws, ex); }
aclnnStatus aclnnMin(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAmin(w, wss, e, s); }

// MultinomialTensor: numSamples carried in a tensor -> read scalar at plan time, forward to Multinomial
aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor *probs, const aclTensor *numSamples, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t ns = 1;
    if (numSamples && numSamples->data) {
        if (numSamples->dtype == ACL_INT64) ns = *((const int64_t *)numSamples->data + numSamples->offset);
        else if (numSamples->dtype == ACL_INT32) ns = *((const int32_t *)numSamples->data + numSamples->offset);
        else ns = (int64_t)*((const float *)numSamples->data + numSamples->offset);
    }
    return aclnnMultinomialGetWorkspaceSize(probs, ns, seed, out, ws, ex);
}
aclnnStatus aclnnMultinomialTensor(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnMultinomial(w, wss, e, s); }

} // extern "C"
