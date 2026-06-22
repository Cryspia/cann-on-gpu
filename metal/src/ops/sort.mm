// Sort / scan host side. Scans (cumsum/cumprod/cummax/cummin/logcumsumexp) run as GPU kernels;
// Sort/Topk run on the host over unified memory (exact ties/stability, any length) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <algorithm>
#include <vector>

namespace {
struct ScanMeta { uint32_t groups; uint32_t D; uint32_t inner; int32_t op; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
void group_shape(const aclTensor *t, int dim, uint32_t &outer, uint32_t &D, uint32_t &inner) {
    int nd = (int)t->viewDims.size(); outer = 1; inner = 1; D = (uint32_t)t->viewDims[dim];
    for (int d = 0; d < dim; ++d) outer *= (uint32_t)t->viewDims[d];
    for (int d = dim + 1; d < nd; ++d) inner *= (uint32_t)t->viewDims[d];
}

aclnnStatus run_scan(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *x = e->a; aclTensor *o = e->out, *oi = e->out2;
    if (!x || !o || !x->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)x->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd; if (dim < 0 || dim >= nd) return ACLNN_ERR_PARAM_INVALID;
    uint32_t outer, D, inner; group_shape(x, dim, outer, D, inner);
    ScanMeta m{ outer * inner, D, inner, (int32_t)e->m };
    bool arg = (e->m == 2 || e->m == 3);
    id<MTLComputePipelineState> pso = mtl::pipeline(arg ? @"scan_arg" : @"scan_val");
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    size_t ox, oo, ooi = 0; id<MTLBuffer> bx = buf_of(x, &ox), bo = buf_of(o, &oo), boi = (arg && oi) ? buf_of(oi, &ooi) : nil;
    if (!bx || !bo || (arg && !boi)) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bx offset:ox atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1];
    if (arg) { [enc setBuffer:boi offset:ooi atIndex:2]; [enc setBytes:&m length:sizeof(m) atIndex:3]; }
    else [enc setBytes:&m length:sizeof(m) atIndex:2];
    NSUInteger n = m.groups, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}

// host sort/topk over unified memory. e->m = k (>0 topk) or -1 (full sort); e->causal = descending.
aclnnStatus run_sort(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *x = e->a; aclTensor *ov = e->out, *oi = e->out2;
    if (!x || !ov || !oi || !x->data || !ov->data || !oi->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)x->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd; if (dim < 0 || dim >= nd) return ACLNN_ERR_PARAM_INVALID;
    uint32_t outer, D, inner; group_shape(x, dim, outer, D, inner);
    int K = (e->m > 0) ? (int)e->m : (int)D;
    bool desc = e->causal;
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];   // drain pending GPU writes
    const float *in = (const float *)x->data + x->offset;
    float *vout = (float *)ov->data + ov->offset;
    int64_t *iout = (int64_t *)oi->data + oi->offset;
    std::vector<int> idx(D);
    for (uint32_t o = 0; o < outer; ++o) for (uint32_t i = 0; i < inner; ++i) {
        uint32_t base = o * D * inner + i;
        for (int j = 0; j < (int)D; ++j) idx[j] = j;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) {
            float va = in[base + (uint32_t)a * inner], vb = in[base + (uint32_t)b * inner];
            if (va == vb) return a < b;
            return desc ? (va > vb) : (va < vb);
        });
        uint32_t obase = o * (uint32_t)K * inner + i;
        for (int kk = 0; kk < K; ++kk) {
            vout[obase + (uint32_t)kk * inner] = in[base + (uint32_t)idx[kk] * inner];
            iout[obase + (uint32_t)kk * inner] = idx[kk];
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }

aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCumsum, run_scan)
aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCumprod, run_scan)
aclnnStatus aclnnLogcumsumexpGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = 4; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLogcumsumexp, run_scan)
aclnnStatus aclnnCummaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !valuesOut || !indicesOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->m = 2; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCummax, run_scan)
aclnnStatus aclnnCumminGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !valuesOut || !indicesOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->m = 3; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCummin, run_scan)
aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool stable, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)stable; if (!self || !valuesOut || !indicesOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->causal = descending; e->m = -1; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSort, run_sort)
aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)sorted; if (!self || !valuesOut || !indicesOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->causal = largest; e->m = k; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTopk, run_sort)
} // extern "C"
