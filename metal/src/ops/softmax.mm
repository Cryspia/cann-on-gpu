// Softmax / LogSoftmax host side (over an arbitrary dim).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
struct SoftMeta { uint32_t groups; uint32_t D; uint32_t inner; int32_t isLog; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus run_softmax(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *x = e->a; aclTensor *o = e->out;
    if (!x || !o || !x->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = x->dtype == ACL_FLOAT ? "f32" : x->dtype == ACL_FLOAT16 ? "f16" : nullptr;
    if (!suf) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)x->viewDims.size();
    int dim = (int)e->dim; if (dim < 0) dim += nd; if (dim < 0 || dim >= nd) return ACLNN_ERR_PARAM_INVALID;
    uint32_t D = (uint32_t)x->viewDims[dim], inner = 1, outer = 1;
    for (int d = dim + 1; d < nd; ++d) inner *= (uint32_t)x->viewDims[d];
    for (int d = 0; d < dim; ++d) outer *= (uint32_t)x->viewDims[d];
    SoftMeta m{ outer * inner, D, inner, (int32_t)e->keepDim };   // keepDim repurposed as isLog flag
    id<MTLComputePipelineState> pso = mtl::pipeline([NSString stringWithFormat:@"softmax_%s", suf]);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    size_t ox, oo; id<MTLBuffer> bx = buf_of(x, &ox), bo = buf_of(o, &oo);
    if (!bx || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:ox atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    NSUInteger n = m.groups, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_SOFTMAX; e->a = self; e->out = out; e->dim = dim; e->keepDim = false; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSoftmax(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_softmax(e, s); } delete e; return st;
}
aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_SOFTMAX; e->a = self; e->out = out; e->dim = dim; e->keepDim = true; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLogSoftmax(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_softmax(e, s); } delete e; return st;
}
} // extern "C"
