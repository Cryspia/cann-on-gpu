// Indexing/gather host ops: Slice (along a dim) and ScatterUpdate (replace rows by index along axis 0).
// Host compute over unified memory (exact, dtype-agnostic byte copy) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <vector>

namespace {
char *bp(const aclTensor *t) { return (char *)t->data + (size_t)t->offset * dtype_size(t->dtype); }

aclnnStatus run_slice(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    int nd = (int)a->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd; if (dim < 0 || dim >= nd) return ACLNN_ERR_PARAM_INVALID;
    int64_t start = e->m, step = e->reduceCount ? e->reduceCount : 1;
    size_t esz = dtype_size(a->dtype);
    // contiguous input strides (elements)
    std::vector<int64_t> istr(nd); { int64_t s2 = 1; for (int d = nd - 1; d >= 0; --d) { istr[d] = s2; s2 *= a->viewDims[d]; } }
    const char *in = bp(a); char *out = bp(o);
    int64_t total = o->numel();
    for (int64_t g = 0; g < total; ++g) {
        int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t ik = (d == dim) ? (start + k * step) : k; ioff += ik * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz);
    }
    return ACLNN_SUCCESS;
}

aclnnStatus run_scatter_update(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    const aclTensor *self = e->a, *index = e->b, *src = e->c; aclTensor *o = e->out;
    if (!self || !index || !src || !o) return ACLNN_ERR_PARAM_NULLPTR;
    size_t esz = dtype_size(self->dtype);
    int64_t R = self->viewDims[0], rest = self->numel() / R, K = index->numel();
    char *outp = bp(o); const char *selfp = bp(self), *srcp = bp(src);
    if (outp != selfp) memcpy(outp, selfp, (size_t)self->numel() * esz);   // out = self, then overwrite rows
    const int64_t *idx = (const int64_t *)bp(index);
    for (int64_t i = 0; i < K; ++i) {
        int64_t row = idx[i]; if (row < 0 || row >= R) continue;
        memcpy(outp + (size_t)row * rest * esz, srcp + (size_t)i * rest * esz, (size_t)rest * esz);
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnSliceGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end, int64_t step,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)end; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = start; e->reduceCount = step; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnSlice, run_slice)
aclnnStatus aclnnScatterUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->c = src; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnScatterUpdate, run_scatter_update)
} // extern "C"
