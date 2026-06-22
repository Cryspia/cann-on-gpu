// Format / data-movement family: Contiguous, AsStrided, ViewCopy, Copy (+ Identity).
// Host strided-gather over unified memory (exact, dtype-agnostic byte copy) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <vector>

namespace {
char *bp0(const aclTensor *t) { return (char *)t->data; }   // base (offset applied explicitly per op)

aclnnStatus run_format(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    size_t esz = dtype_size(o->dtype);
    char *out = (char *)o->data + (size_t)o->offset * esz;
    if (e->m == 0) {   // Contiguous: materialize strided `a` (its viewDims/strides) into contiguous out
        int nd = (int)a->viewDims.size(); int64_t total = a->numel();
        const char *in = bp0(a);
        for (int64_t g = 0; g < total; ++g) {
            int64_t rem = g, ioff = a->offset;
            for (int d = nd - 1; d >= 0; --d) { int64_t vd = a->viewDims[d]; int64_t k = rem % vd; rem /= vd; ioff += k * a->strides[d]; }
            memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz);
        }
    } else if (e->m == 1) {   // AsStrided: out[idx] = a.data[storageOffset + sum idx*stride]
        int nd = (int)o->viewDims.size(); int64_t total = o->numel(); const char *in = bp0(a);
        for (int64_t g = 0; g < total; ++g) {
            int64_t rem = g, ioff = e->dim;   // dim holds storageOffset
            for (int d = nd - 1; d >= 0; --d) { int64_t vd = o->viewDims[d]; int64_t k = rem % vd; rem /= vd; ioff += k * e->axes[d]; }
            memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz);
        }
    } else {   // ViewCopy / Copy / Identity: flat element copy (numel-equal), respecting source contiguity
        int64_t total = o->numel();
        if (a->contiguous()) memcpy(out, (char *)a->data + (size_t)a->offset * esz, (size_t)total * esz);
        else { int nd=(int)a->viewDims.size(); const char *in=bp0(a);
            for (int64_t g=0; g<total; ++g){ int64_t rem=g, ioff=a->offset; for(int d=nd-1;d>=0;--d){int64_t vd=a->viewDims[d];int64_t k=rem%vd;rem/=vd;ioff+=k*a->strides[d];} memcpy(out+(size_t)g*esz, in+(size_t)ioff*esz, esz); } }
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_format(e, s); } delete e; return st; }
} // namespace

extern "C" {
#define TRANS(NAME, K) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = K; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(NAME)
TRANS(aclnnContiguous, 0) TRANS(aclnnViewCopy, 2) TRANS(aclnnCopy, 2) TRANS(aclnnIdentity, 2)
aclnnStatus aclnnAsStridedGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, const aclIntArray *stride, int64_t storageOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !size || !stride || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 1; e->a = self; e->out = out; e->axes = stride->v; e->dim = storageOffset; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnAsStrided)
} // extern "C"
