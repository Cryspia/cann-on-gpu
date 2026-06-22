// Elementwise op family host side: generic unary (+ tensor-scalar as unary-with-constant) and generic
// broadcast binary. Two-phase aclnn contract; one-shot executor freed on Execute.
#import "../internal.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"
#include "ew_ops.h"
#include "subfp.h"
#include <cstring>

namespace {

struct UnMeta { uint32_t n; uint32_t ndim; int32_t op; float p0; float p1; uint32_t odims[8]; uint32_t astr[8]; };
struct BcMeta { uint32_t n; uint32_t ndim; int32_t op; float alpha; uint32_t odims[8]; uint32_t astr[8]; uint32_t bstr[8]; };

const char *fdt(aclDataType dt) { return dt == ACL_FLOAT ? "f32" : dt == ACL_FLOAT16 ? "f16" : nullptr; }
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
bool build_bcmeta(BcMeta &m, const aclTensor *out, const aclTensor *a, const aclTensor *b, int op, float alpha) {
    uint32_t nd = (uint32_t)out->viewDims.size(); if (nd > 8) return false;
    m.n = (uint32_t)out->numel(); m.ndim = nd; m.op = op; m.alpha = alpha;
    for (uint32_t d = 0; d < 8; ++d) { m.odims[d] = 1; m.astr[d] = 0; m.bstr[d] = 0; }
    for (uint32_t d = 0; d < nd; ++d) m.odims[d] = (uint32_t)out->viewDims[d];
    auto fill = [&](const aclTensor *t, uint32_t *str) {
        uint32_t tnd = (uint32_t)t->viewDims.size();
        for (uint32_t d = 0; d < nd; ++d) {
            int td = (int)d - (int)(nd - tnd);
            str[d] = (td < 0 || t->viewDims[td] == 1) ? 0u : (uint32_t)t->strides[td];
        }
    };
    fill(a, m.astr); fill(b, m.bstr);
    return true;
}

typedef void (^BindBlock)(id<MTLComputeCommandEncoder>);
aclnnStatus dispatch1d(NSString *kname, uint64_t n, aclrtStream stream, BindBlock bind) {
    if (n == 0) return ACLNN_SUCCESS;
    id<MTLComputePipelineState> pso = mtl::pipeline(kname);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    bind(enc);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}

aclnnStatus run_unary(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = fdt(o->dtype); if (!suf) return ACLNN_ERR_PARAM_INVALID;
    uint32_t nd = (uint32_t)o->viewDims.size(); if (nd > 8) return ACLNN_ERR_PARAM_INVALID;
    UnMeta m{ (uint32_t)o->numel(), nd, (int32_t)e->m,
              (float)(e->dscalars.size() > 0 ? e->dscalars[0] : 0.0),
              (float)(e->dscalars.size() > 1 ? e->dscalars[1] : 0.0), {0}, {0} };
    for (uint32_t d = 0; d < nd; ++d) { m.odims[d] = (uint32_t)o->viewDims[d]; m.astr[d] = (uint32_t)a->strides[d]; }
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    return dispatch1d([NSString stringWithFormat:@"ew_un_%s", suf], o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    });
}

aclnnStatus run_binary(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a, *b = e->b; aclTensor *o = e->out;
    if (!a || !b || !o || !a->data || !b->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    BcMeta m; if (!build_bcmeta(m, o, a, b, (int)e->m, (float)e->alpha)) return ACLNN_ERR_PARAM_INVALID;
    // fp8 / fp4 / fp6 inputs/outputs: decode -> compute(float) -> encode, host-side over unified memory.
    if (subfp::is_low(o->dtype)) {
        auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted];
        const uint8_t *ab = (const uint8_t *)a->data, *bb = (const uint8_t *)b->data; uint8_t *ob = (uint8_t *)o->data;
        int64_t n = o->numel();
        if (subfp::is_fp4(o->dtype)) memset(ob, 0, (size_t)((n + 1) / 2));
        for (int64_t g = 0; g < n; ++g) {
            int64_t rem = g, ai = 0, bi = 0;
            for (int d = (int)m.ndim - 1; d >= 0; --d) { int64_t od = m.odims[d]; int64_t k = rem % od; rem /= od; ai += k * (int64_t)m.astr[d]; bi += k * (int64_t)m.bstr[d]; }
            float av = subfp::load(a->dtype, ab, ai), bv = subfp::load(b->dtype, bb, bi);
            float r = (e->m == B_SUB) ? av - (float)e->alpha * bv : (e->m == B_MUL) ? av * bv : av + (float)e->alpha * bv;
            subfp::store(o->dtype, ob, g, r);
        }
        return ACLNN_SUCCESS;
    }
    // mixed float dtypes (e.g. fp16 * fp32 -> fp32): no single-T kernel covers it — compute host-side in fp32.
    if ((a->dtype != o->dtype || b->dtype != o->dtype) && (o->dtype == ACL_FLOAT || o->dtype == ACL_FLOAT16)) {
        auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted];
        auto ld = [](aclDataType dt, const void *p, int64_t i) -> float {
            if (dt == ACL_FLOAT16) return (float)((const __fp16 *)p)[i];
            if (dt == ACL_INT32) return (float)((const int32_t *)p)[i];
            return ((const float *)p)[i]; };
        int64_t n = o->numel();
        for (int64_t g = 0; g < n; ++g) {
            int64_t rem = g, ai = 0, bi = 0;
            for (int d = (int)m.ndim - 1; d >= 0; --d) { int64_t od = m.odims[d]; int64_t kk = rem % od; rem /= od; ai += kk * (int64_t)m.astr[d]; bi += kk * (int64_t)m.bstr[d]; }
            float av = ld(a->dtype, a->data, ai), bv = ld(b->dtype, b->data, bi), al = (float)e->alpha, r;
            switch (e->m) { case B_SUB: r = av - al * bv; break; case B_MUL: r = av * bv; break; case B_DIV: r = av / bv; break;
                            case B_MAX: r = av > bv ? av : bv; break; case B_MIN: r = av < bv ? av : bv; break; default: r = av + al * bv; }
            if (o->dtype == ACL_FLOAT16) ((__fp16 *)o->data)[g] = (__fp16)r; else ((float *)o->data)[g] = r;
        }
        return ACLNN_SUCCESS;
    }
    NSString *k;
    if (o->dtype == ACL_INT32) k = @"ew_bin_i32";
    else { const char *suf = fdt(o->dtype); if (!suf) return ACLNN_ERR_PARAM_INVALID; k = [NSString stringWithFormat:@"ew_bin_%s", suf]; }
    size_t oa, ob, oo; id<MTLBuffer> ba = buf_of(a, &oa), bb = buf_of(b, &ob), bo = buf_of(o, &oo);
    if (!ba || !bb || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    return dispatch1d(k, o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    });
}

} // namespace

// ---- entry macros ----
#define RUN_DECL(NAME, RUN) \
extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; \
    if (aclCaptureRecord(s, &NAME, e, w, wss)) return ACLNN_SUCCESS; \
    aclnnStatus st; @autoreleasepool { st = RUN(e, s); } \
    for (auto *t : e->owned) delete t; delete e; return st; }

#define DEF_UN(NAME, OPC) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = OPC; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN_DECL(NAME, run_unary)

#define DEF_UN_S(NAME, OPC) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = OPC; e->a = self; e->out = out; e->dscalars = {other->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN_DECL(NAME, run_unary)

#define DEF_BIN(NAME, OPC) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = OPC; e->a = self; e->b = other; e->out = out; e->alpha = 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN_DECL(NAME, run_binary)

#define DEF_BIN_A(NAME, OPC) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = OPC; e->a = self; e->b = other; e->out = out; e->alpha = alpha ? alpha->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN_DECL(NAME, run_binary)

// unary (ACLNN_UN)
DEF_UN(aclnnExp, U_EXP) DEF_UN(aclnnLog, U_LOG) DEF_UN(aclnnAbs, U_ABS) DEF_UN(aclnnSqrt, U_SQRT)
DEF_UN(aclnnRsqrt, U_RSQRT) DEF_UN(aclnnReciprocal, U_RECIP) DEF_UN(aclnnRelu, U_RELU) DEF_UN(aclnnNeg, U_NEG)
DEF_UN(aclnnSigmoid, U_SIGMOID) DEF_UN(aclnnTanh, U_TANH) DEF_UN(aclnnErf, U_ERF) DEF_UN(aclnnGelu, U_GELU)
DEF_UN(aclnnSilu, U_SILU) DEF_UN(aclnnSoftplus, U_SOFTPLUS) DEF_UN(aclnnSin, U_SIN) DEF_UN(aclnnCos, U_COS)
DEF_UN(aclnnTan, U_TAN) DEF_UN(aclnnAtan, U_ATAN) DEF_UN(aclnnSign, U_SIGN) DEF_UN(aclnnFloor, U_FLOOR)
DEF_UN(aclnnCeil, U_CEIL) DEF_UN(aclnnRound, U_ROUND) DEF_UN(aclnnTrunc, U_TRUNC) DEF_UN(aclnnSquare, U_SQUARE)
DEF_UN(aclnnSinh, U_SINH) DEF_UN(aclnnCosh, U_COSH) DEF_UN(aclnnAsin, U_ASIN) DEF_UN(aclnnAcos, U_ACOS)
DEF_UN(aclnnErfc, U_ERFC) DEF_UN(aclnnFrac, U_FRAC) DEF_UN(aclnnLgamma, U_LGAMMA) DEF_UN(aclnnExpm1, U_EXPM1)
DEF_UN(aclnnLog1p, U_LOG1P) DEF_UN(aclnnLog2, U_LOG2) DEF_UN(aclnnLog10, U_LOG10) DEF_UN(aclnnExp2, U_EXP2)
DEF_UN(aclnnErfinv, U_ERFINV) DEF_UN(aclnnMish, U_MISH) DEF_UN(aclnnHardswish, U_HARDSWISH)
DEF_UN(aclnnHardsigmoid, U_HARDSIGMOID) DEF_UN(aclnnLogSigmoid, U_LOGSIGMOID) DEF_UN(aclnnSelu, U_SELU)
DEF_UN(aclnnTanhshrink, U_TANHSHRINK) DEF_UN(aclnnRelu6, U_RELU6)

// binary (ACLNN_BIN)
DEF_BIN(aclnnMul, B_MUL) DEF_BIN(aclnnDiv, B_DIV) DEF_BIN(aclnnMaximum, B_MAX) DEF_BIN(aclnnMinimum, B_MIN)
DEF_BIN(aclnnFmod, B_FMOD) DEF_BIN(aclnnHypot, B_HYPOT) DEF_BIN(aclnnPowTensorTensor, B_POW)
DEF_BIN(aclnnAtan2, B_ATAN2) DEF_BIN(aclnnRemainderTensorTensor, B_REMAINDER) DEF_BIN(aclnnXLogYTensorTensor, B_XLOGY)
DEF_BIN(aclnnLogAddExp, B_LOGADDEXP) DEF_BIN(aclnnCopysign, B_COPYSIGN) DEF_BIN(aclnnHeaviside, B_HEAVISIDE)
DEF_BIN(aclnnPrelu, B_PRELU)

// binary with alpha (ACLNN_BIN_ALPHA) + Add (aclnn_add.h)
DEF_BIN_A(aclnnAdd, B_ADD) DEF_BIN_A(aclnnSub, B_SUB) DEF_BIN_A(aclnnLerp, B_LERP)

// tensor-scalar (ACLNN_SCALAR) mapped to unary-with-constant
DEF_UN_S(aclnnMuls, U_MULC) DEF_UN_S(aclnnClampMin, U_CLAMP_LO) DEF_UN_S(aclnnClampMax, U_CLAMP_HI)
DEF_UN_S(aclnnPowTensorScalar, U_POWS) DEF_UN_S(aclnnLeakyRelu, U_LEAKYRELU) DEF_UN_S(aclnnElu, U_ELU)
DEF_UN_S(aclnnCelu, U_CELU) DEF_UN_S(aclnnHardshrink, U_HARDSHRINK) DEF_UN_S(aclnnSoftshrink, U_SOFTSHRINK)

// Divs (x/scalar -> mul by reciprocal), Adds/Subs (x +/- alpha*scalar -> add constant): explicit
extern "C" aclnnStatus aclnnDivsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = U_MULC; e->a = self; e->out = out; e->dscalars = {1.0 / other->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN_DECL(aclnnDivs, run_unary)
extern "C" aclnnStatus aclnnAddsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = U_ADDC; e->a = self; e->out = out; e->dscalars = {(alpha ? alpha->v : 1.0) * other->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN_DECL(aclnnAdds, run_unary)
extern "C" aclnnStatus aclnnSubsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = U_ADDC; e->a = self; e->out = out; e->dscalars = {-(alpha ? alpha->v : 1.0) * other->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN_DECL(aclnnSubs, run_unary)
