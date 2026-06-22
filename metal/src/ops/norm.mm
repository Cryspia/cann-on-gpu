// Normalization host side: aclnnRmsNorm (normalize over the last gamma.numel() dims).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
struct RmsMeta { uint32_t rows; uint32_t cols; float eps; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus run_rmsnorm(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *x = e->a, *g = e->b; aclTensor *y = e->out;
    if (!x || !g || !y || !x->data || !g->data || !y->data) return ACLNN_ERR_PARAM_NULLPTR;
    const char *suf = x->dtype == ACL_FLOAT ? "f32" : x->dtype == ACL_FLOAT16 ? "f16" : nullptr;
    if (!suf) return ACLNN_ERR_PARAM_INVALID;
    int64_t cols = g->numel(); if (cols <= 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = x->numel() / cols;
    RmsMeta m{ (uint32_t)rows, (uint32_t)cols, (float)e->eps };
    id<MTLComputePipelineState> pso = mtl::pipeline([NSString stringWithFormat:@"rmsnorm_%s", suf]);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    size_t ox, og, oy; id<MTLBuffer> bx = buf_of(x, &ox), bg = buf_of(g, &og), by = buf_of(y, &oy);
    if (!bx || !bg || !by) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:ox atIndex:0]; [enc setBuffer:bg offset:og atIndex:1];
    [enc setBuffer:by offset:oy atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3];
    NSUInteger n = (NSUInteger)rows, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, double eps,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !gamma || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_RMSNORM; e->a = self; e->b = gamma; e->out = out; e->eps = eps;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    if (aclCaptureRecord(s, &aclnnRmsNorm, e, w, wss)) return ACLNN_SUCCESS;   // graph capture: record, don't run
    aclnnStatus st; @autoreleasepool { st = run_rmsnorm(e, s); } delete e; return st;
}
} // extern "C"
