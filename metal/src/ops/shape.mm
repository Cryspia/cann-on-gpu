// Shape / gather host side: aclnnPermute, aclnnEmbedding.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
struct PermMeta { uint32_t n; uint32_t ndim; uint32_t odims[8]; uint32_t istr[8]; };
struct EmbMeta  { uint32_t n; uint32_t H; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus dispatch(NSString *kname, uint64_t n, aclrtStream stream,
                     void (^bind)(id<MTLComputeCommandEncoder>)) {
    if (n == 0) return ACLNN_SUCCESS;
    id<MTLComputePipelineState> pso = mtl::pipeline(kname);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso]; bind(enc);
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (NSUInteger)n;
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
NSString *perm_kernel(size_t esz) { return esz == 2 ? @"perm_b2" : esz == 8 ? @"perm_b8" : @"perm_b4"; }

aclnnStatus run_permute(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    uint32_t nd = (uint32_t)a->viewDims.size();
    if (nd != e->axes.size() || nd > 8) return ACLNN_ERR_PARAM_INVALID;
    size_t esz = dtype_size(a->dtype); if (esz != 2 && esz != 4 && esz != 8) return ACLNN_ERR_PARAM_INVALID;
    PermMeta m{}; m.n = (uint32_t)o->numel(); m.ndim = nd;
    for (uint32_t i = 0; i < nd; ++i) {
        int64_t ax = e->axes[i]; if (ax < 0) ax += nd; if (ax < 0 || ax >= nd) return ACLNN_ERR_PARAM_INVALID;
        m.odims[i] = (uint32_t)a->viewDims[ax];      // output axis i = input axis ax
        m.istr[i]  = (uint32_t)a->strides[ax];
    }
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    return dispatch(perm_kernel(esz), o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    });
}

aclnnStatus run_embedding(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *w = e->a, *ids = e->b; aclTensor *o = e->out;
    if (!w || !ids || !o || !w->data || !ids->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    size_t esz = dtype_size(w->dtype); if (esz != 2 && esz != 4) return ACLNN_ERR_PARAM_INVALID;
    int64_t H = w->viewDims.back();
    EmbMeta m{ (uint32_t)o->numel(), (uint32_t)H };
    const char *wt = esz == 2 ? "b2" : "b4";
    const char *it = (ids->dtype == ACL_INT64) ? "i64" : (ids->dtype == ACL_INT32) ? "i32" : nullptr;
    if (!it) return ACLNN_ERR_PARAM_INVALID;
    size_t ow, oi, oo; id<MTLBuffer> bw = buf_of(w, &ow), bi = buf_of(ids, &oi), bo = buf_of(o, &oo);
    if (!bw || !bi || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    return dispatch([NSString stringWithFormat:@"embed_%s_%s", wt, it], o->numel(), s, ^(id<MTLComputeCommandEncoder> enc) {
        [enc setBuffer:bw offset:ow atIndex:0]; [enc setBuffer:bi offset:oi atIndex:1];
        [enc setBuffer:bo offset:oo atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    });
}
} // namespace

extern "C" {
aclnnStatus aclnnPermuteGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !dims || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_PERMUTE; e->a = self; e->out = out; e->axes = dims->v; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPermute(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_permute(e, s); } delete e; return st;
}
aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor *weight, const aclTensor *ids, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !ids || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = weight; e->b = ids; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEmbedding(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_embedding(e, s); } delete e; return st;
}
} // extern "C"
