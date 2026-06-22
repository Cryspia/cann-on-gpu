// Comparison / logical / predicate / masked-fill host side (bool output). Two-phase aclnn contract.
// Tensor compares and logical-and/or support general broadcasting (built into EwMeta strides).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {

struct EwMeta  { uint32_t n; uint32_t ndim; int32_t op; uint32_t odims[8]; uint32_t astr[8]; uint32_t bstr[8]; };
struct EwMetaS { uint32_t n; int32_t op; float s; };

const char *dt_suffix(aclDataType dt) {
    switch (dt) { case ACL_FLOAT: return "f32"; case ACL_FLOAT16: return "f16"; case ACL_INT32: return "i32"; default: return nullptr; }
}
int cmp_code(int opk) {
    switch (opk) { case OP_GT: return 0; case OP_LT: return 1; case OP_EQ: return 2;
                   case OP_NE: return 3; case OP_GE: return 4; default: return 5; }   // OP_LE
}
int pred_code(int opk) {
    switch (opk) { case OP_ISNAN: return 0; case OP_ISFINITE: return 1; case OP_ISINF: return 2;
                   case OP_ISPOSINF: return 3; default: return 4; }                   // OP_ISNEGINF
}

// Broadcast strides of inputs relative to the output shape (right-aligned; size-1 dim -> stride 0).
bool build_meta(EwMeta &m, const aclTensor *out, const aclTensor *a, const aclTensor *b) {
    uint32_t nd = (uint32_t)out->viewDims.size();
    if (nd > 8) return false;
    m.n = (uint32_t)out->numel(); m.ndim = nd; m.op = 0;
    for (uint32_t d = 0; d < 8; ++d) { m.odims[d] = 1; m.astr[d] = 0; m.bstr[d] = 0; }
    for (uint32_t d = 0; d < nd; ++d) m.odims[d] = (uint32_t)out->viewDims[d];
    auto fill = [&](const aclTensor *t, uint32_t *str) {
        if (!t) return;
        uint32_t tnd = (uint32_t)t->viewDims.size();
        for (uint32_t d = 0; d < nd; ++d) {
            int td = (int)d - (int)(nd - tnd);
            if (td < 0) { str[d] = 0; continue; }
            str[d] = (t->viewDims[td] == 1) ? 0u : (uint32_t)t->strides[td];
        }
    };
    fill(a, m.astr); fill(b, m.bstr);
    return true;
}

// device buffer + byte offset for a tensor's data pointer (incl. element offset)
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
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

aclnnStatus run_cmp_tensor(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a, *b = e->b; aclTensor *o = e->out;
    if (!a || !b || !o || !a->data || !b->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = dt_suffix(a->dtype); if (!suf) return ACLNN_ERR_PARAM_INVALID;
    EwMeta m; if (!build_meta(m, o, a, b)) return ACLNN_ERR_PARAM_INVALID; m.op = cmp_code(e->op);
    size_t oa, ob, oo; id<MTLBuffer> ba = buf_of(a, &oa), bb = buf_of(b, &ob), bo = buf_of(o, &oo);
    if (!ba || !bb || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    NSString *k = [NSString stringWithFormat:@"cmp_%s", suf];
    return dispatch1d(k, o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    });
}

aclnnStatus run_cmp_scalar(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = dt_suffix(a->dtype); if (!suf) return ACLNN_ERR_PARAM_INVALID;
    EwMetaS m{ (uint32_t)a->numel(), cmp_code(e->op), (float)e->alpha };
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    NSString *k = [NSString stringWithFormat:@"cmps_%s", suf];
    return dispatch1d(k, a->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1];
        [enc setBytes:&m length:sizeof(m) atIndex:2];
    });
}

aclnnStatus run_logic(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (e->op == OP_LNOT) {
        EwMetaS m{ (uint32_t)a->numel(), 0, 0.f };
        size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
        if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
        return dispatch1d(@"lnot_b", a->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
            [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1];
            [enc setBytes:&m length:sizeof(m) atIndex:2];
        });
    }
    const aclTensor *b = e->b; if (!b || !b->data) return ACLNN_ERR_PARAM_NULLPTR;
    EwMeta m; if (!build_meta(m, o, a, b)) return ACLNN_ERR_PARAM_INVALID; m.op = (e->op == OP_LAND) ? 0 : 1;
    size_t oa, ob, oo; id<MTLBuffer> ba = buf_of(a, &oa), bb = buf_of(b, &ob), bo = buf_of(o, &oo);
    if (!ba || !bb || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    return dispatch1d(@"logic_b", o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bb offset:ob atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    });
}

aclnnStatus run_pred(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = (a->dtype == ACL_FLOAT) ? "f32" : (a->dtype == ACL_FLOAT16) ? "f16" : nullptr;
    if (!suf) return ACLNN_ERR_PARAM_INVALID;
    EwMetaS m{ (uint32_t)a->numel(), pred_code(e->op), 0.f };
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    NSString *k = [NSString stringWithFormat:@"pred_%s", suf];
    return dispatch1d(k, a->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1];
        [enc setBytes:&m length:sizeof(m) atIndex:2];
    });
}

aclnnStatus run_maskfill(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a, *mask = e->b; aclTensor *o = e->out;
    if (!a || !mask || !o || !a->data || !mask->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = dt_suffix(a->dtype); if (!suf) return ACLNN_ERR_PARAM_INVALID;
    EwMetaS m{ (uint32_t)a->numel(), 0, (float)e->alpha };
    size_t oa, om, oo; id<MTLBuffer> ba = buf_of(a, &oa), bm = buf_of(mask, &om), bo = buf_of(o, &oo);
    if (!ba || !bm || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    NSString *k = [NSString stringWithFormat:@"maskfill_%s", suf];
    return dispatch1d(k, a->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bm offset:om atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    });
}

} // namespace

// ---- aclnn entry points (signatures from include/aclnnop/aclnn_ops.h: ACLNN_CMP / ACLNN_CMP_S / ACLNN_PRED) ----
#define DEF_BIN(NAME, OPK, RUN) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, \
        uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = OPK; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = RUN(e, s); } delete e; return st; }

#define DEF_SCALAR(NAME, OPK) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, \
        uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = OPK; e->a = self; e->out = out; e->alpha = other->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_cmp_scalar(e, s); } delete e; return st; }

#define DEF_UN(NAME, OPK, RUN) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = OPK; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = RUN(e, s); } delete e; return st; }

DEF_BIN(aclnnGtTensor, OP_GT, run_cmp_tensor)
DEF_BIN(aclnnLtTensor, OP_LT, run_cmp_tensor)
DEF_BIN(aclnnEqTensor, OP_EQ, run_cmp_tensor)
DEF_BIN(aclnnNeTensor, OP_NE, run_cmp_tensor)
DEF_BIN(aclnnGeTensor, OP_GE, run_cmp_tensor)
DEF_BIN(aclnnLeTensor, OP_LE, run_cmp_tensor)
DEF_BIN(aclnnLogicalAnd, OP_LAND, run_logic)
DEF_BIN(aclnnLogicalOr,  OP_LOR,  run_logic)

DEF_SCALAR(aclnnGtScalar, OP_GT)
DEF_SCALAR(aclnnLtScalar, OP_LT)
DEF_SCALAR(aclnnEqScalar, OP_EQ)
DEF_SCALAR(aclnnNeScalar, OP_NE)
DEF_SCALAR(aclnnGeScalar, OP_GE)
DEF_SCALAR(aclnnLeScalar, OP_LE)

DEF_UN(aclnnLogicalNot, OP_LNOT,    run_logic)
DEF_UN(aclnnIsNan,      OP_ISNAN,   run_pred)
DEF_UN(aclnnIsFinite,   OP_ISFINITE,run_pred)
DEF_UN(aclnnIsInf,      OP_ISINF,   run_pred)
DEF_UN(aclnnIsPosInf,   OP_ISPOSINF,run_pred)
DEF_UN(aclnnIsNegInf,   OP_ISNEGINF,run_pred)

extern "C" aclnnStatus aclnnMaskedFillScalarGetWorkspaceSize(const aclTensor *self, const aclTensor *mask,
        const aclScalar *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mask || !value || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_MASKEDFILL; e->a = self; e->b = mask; e->out = out; e->alpha = value->v;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
extern "C" aclnnStatus aclnnMaskedFillScalar(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_maskfill(e, s); } delete e; return st;
}
