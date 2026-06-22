// Reduction host side. Reduce over an arbitrary axis set; fp32. ops: sum/mean/max/min/prod.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

#define R_SUM 0
#define R_MEAN 1
#define R_MAX 2
#define R_MIN 3
#define R_PROD 4

namespace {
struct RedMeta {
    uint32_t n_out, n_red, ndim_out, ndim_red; int32_t op;
    uint32_t out_dims[8], out_istr[8];
    uint32_t red_dims[8], red_istr[8];
};
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus run_reduce(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (a->dtype != ACL_FLOAT || o->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)a->viewDims.size(); if (nd > 8) return ACLNN_ERR_PARAM_INVALID;
    bool red[8] = {false};
    if (e->axes.empty()) { for (int d = 0; d < nd; ++d) red[d] = true; }
    else for (int64_t ax : e->axes) { int d = (int)(ax < 0 ? ax + nd : ax); if (d < 0 || d >= nd) return ACLNN_ERR_PARAM_INVALID; red[d] = true; }
    RedMeta m{}; m.op = (int32_t)e->m; m.n_out = 1; m.n_red = 1;
    for (int d = 0; d < nd; ++d) {
        if (red[d]) { m.red_dims[m.ndim_red] = (uint32_t)a->viewDims[d]; m.red_istr[m.ndim_red] = (uint32_t)a->strides[d]; m.ndim_red++; m.n_red *= (uint32_t)a->viewDims[d]; }
        else        { m.out_dims[m.ndim_out] = (uint32_t)a->viewDims[d]; m.out_istr[m.ndim_out] = (uint32_t)a->strides[d]; m.ndim_out++; m.n_out *= (uint32_t)a->viewDims[d]; }
    }
    id<MTLComputePipelineState> pso = mtl::pipeline(@"reduce_f32");
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
    NSUInteger n = m.n_out, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
} // namespace

// dim+keepDim+dtype variants (Sum/Mean/Prod) and dim+keepDim variants (Amax/Amin)
#define RUN(NAME) extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_reduce(e, s); } delete e; return st; }
#define DEF_RED_DT(NAME, OP) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, \
        aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)dtype; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->m = OP; e->a = self; e->out = out; e->keepDim = keepDim; \
    if (dim) e->axes = dim->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
#define DEF_RED(NAME, OP) \
extern "C" aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, \
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->m = OP; e->a = self; e->out = out; e->keepDim = keepDim; \
    if (dim) e->axes = dim->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)

DEF_RED_DT(aclnnReduceSum, R_SUM)
DEF_RED_DT(aclnnMean, R_MEAN)
DEF_RED_DT(aclnnProd, R_PROD)
DEF_RED(aclnnAmax, R_MAX)
DEF_RED(aclnnAmin, R_MIN)
