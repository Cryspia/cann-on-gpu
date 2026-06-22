// test_remainder R1/R4: shape composites (pads, meshgrid, vstack) and Unique/UniqueConsecutive.
// Host-side over unified memory.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *IP64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// pad index maps (matching the test): reflect / replicate / circular
int padidx(int mode, int x, int n) {
    if (mode == 0) { if (n == 1) return 0; while (x < 0 || x >= n) { if (x < 0) x = -x; if (x >= n) x = 2 * (n - 1) - x; } return x; }  // reflect
    if (mode == 1) return x < 0 ? 0 : (x >= n ? n - 1 : x);                                                                          // replicate
    return ((x % n) + n) % n;                                                                                                        // circular
}

enum { K_PAD1D, K_PAD3D, K_MESHGRID, K_VSTACK, K_UNIQUE, K_UNIQCONS };

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    case K_PAD1D: {   // self [outer..., W], padding {left,right}, mode in m
        const aclTensor *X = e->a; aclTensor *Z = e->out; int mode = (int)e->m;
        int nd = (int)X->viewDims.size(); int W = (int)X->viewDims[nd - 1];
        int64_t outer = 1; for (int d = 0; d < nd - 1; d++) outer *= X->viewDims[d];
        int pl = (int)e->axes[0]; int oW = (int)Z->viewDims[Z->viewDims.size() - 1];
        const float *x = FP(X); float *z = FP(Z);
        for (int64_t o = 0; o < outer; o++) for (int ow = 0; ow < oW; ow++) { int iw = padidx(mode, ow - pl, W); z[o * oW + ow] = x[o * W + iw]; }
        return ACLNN_SUCCESS;
    }
    case K_PAD3D: {   // self [...,D,H,W], padding {wl,wr,hl,hr,dl,dr}, mode in m
        const aclTensor *X = e->a; aclTensor *Z = e->out; int mode = (int)e->m;
        int nd = (int)X->viewDims.size();
        int D = (int)X->viewDims[nd - 3], H = (int)X->viewDims[nd - 2], W = (int)X->viewDims[nd - 1];
        int64_t outer = 1; for (int d = 0; d < nd - 3; d++) outer *= X->viewDims[d];
        int wl = (int)e->axes[0], hl = (int)e->axes[2], dl = (int)e->axes[4];
        int znd = (int)Z->viewDims.size();
        int oD = (int)Z->viewDims[znd - 3], oH = (int)Z->viewDims[znd - 2], oW = (int)Z->viewDims[znd - 1];
        const float *x = FP(X); float *z = FP(Z);
        for (int64_t o = 0; o < outer; o++) for (int od = 0; od < oD; od++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            int id = padidx(mode, od - dl, D), ih = padidx(mode, oh - hl, H), iw = padidx(mode, ow - wl, W);
            z[((o * oD + od) * oH + oh) * oW + ow] = x[((o * D + id) * H + ih) * W + iw];
        }
        return ACLNN_SUCCESS;
    }
    case K_MESHGRID: {   // ij indexing for N tensors; out[k] broadcasts input[k] over all dims
        const aclTensorList *ins = e->tl[0], *outs = e->tl[1]; int N = (int)ins->v.size();
        std::vector<int> sz(N); int64_t total = 1; for (int k = 0; k < N; k++) { sz[k] = (int)ins->v[k]->numel(); total *= sz[k]; }
        for (int k = 0; k < N; k++) {
            const float *src = FP(ins->v[k]); float *dst = FP(const_cast<aclTensor *>(outs->v[k]));
            for (int64_t g = 0; g < total; g++) {
                int64_t rem = g; int idxk = 0;
                for (int d = N - 1; d >= 0; d--) { int coord = (int)(rem % sz[d]); rem /= sz[d]; if (d == k) idxk = coord; }
                dst[g] = src[idxk];
            }
        }
        return ACLNN_SUCCESS;
    }
    case K_VSTACK: {   // concat along dim0
        aclTensor *Z = e->out; float *z = FP(Z); int64_t off = 0;
        for (const aclTensor *t : e->inputs) { const float *src = FP(t); int64_t n = t->numel(); for (int64_t i = 0; i < n; i++) z[off + i] = src[i]; off += n; }
        return ACLNN_SUCCESS;
    }
    case K_UNIQUE: {   // sorted unique values; count; inverse map; counts
        const aclTensor *X = e->a; const float *x = FP(X); int64_t n = X->numel();
        std::vector<float> vals(x, x + n); std::sort(vals.begin(), vals.end());
        std::vector<float> uniq; for (int64_t i = 0; i < n; i++) if (uniq.empty() || vals[i] != uniq.back()) uniq.push_back(vals[i]);
        int U = (int)uniq.size();
        float *vo = FP(e->out); for (int i = 0; i < U; i++) vo[i] = uniq[i];
        if (e->out2) IP64(e->out2)[0] = U;                                  // count
        if (e->c) { int64_t *inv = IP64(const_cast<aclTensor *>(e->c));     // inverse: orig -> uidx
            for (int64_t i = 0; i < n; i++) { int u = (int)(std::lower_bound(uniq.begin(), uniq.end(), x[i]) - uniq.begin()); inv[i] = u; } }
        if (e->mean) { int64_t *cnt = IP64(const_cast<aclTensor *>(e->mean));  // counts
            for (int u = 0; u < U; u++) { int64_t c = 0; for (int64_t i = 0; i < n; i++) if (x[i] == uniq[u]) c++; cnt[u] = c; } }
        return ACLNN_SUCCESS;
    }
    case K_UNIQCONS: {   // collapse runs of equal consecutive values
        const aclTensor *X = e->a; const float *x = FP(X); int64_t n = X->numel();
        std::vector<float> vals; std::vector<int64_t> counts;
        for (int64_t i = 0; i < n; i++) { if (vals.empty() || x[i] != vals.back()) { vals.push_back(x[i]); counts.push_back(1); } else counts.back()++; }
        int U = (int)vals.size();
        float *vo = FP(e->out); for (int i = 0; i < U; i++) vo[i] = vals[i];
        if (e->out2) IP64(e->out2)[0] = U;
        if (e->mean) { int64_t *cnt = IP64(const_cast<aclTensor *>(e->mean)); for (int u = 0; u < U; u++) cnt[u] = counts[u]; }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUNFN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUNFN(rs_run)
} // namespace

extern "C" {
// pads: PADX signature (self, padding, out). mode encoded per op.
#define PAD1D(NAME, MODE) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->op = K_PAD1D; e->m = MODE; e->a = self; e->out = out; if (padding) e->axes = padding->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(NAME)
#define PAD3D(NAME, MODE) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->op = K_PAD3D; e->m = MODE; e->a = self; e->out = out; if (padding) e->axes = padding->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(NAME)
PAD1D(aclnnReflectionPad1d, 0) PAD1D(aclnnReplicationPad1d, 1) PAD1D(aclnnCircularPad1d, 2)
PAD3D(aclnnReflectionPad3d, 0) PAD3D(aclnnReplicationPad3d, 1) PAD3D(aclnnCircularPad3d, 2)

aclnnStatus aclnnMeshgridGetWorkspaceSize(const aclTensorList *tensors, const aclTensorList *outs, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MESHGRID; e->tl[0] = tensors; e->tl[1] = outs; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMeshgrid)
aclnnStatus aclnnVstackGetWorkspaceSize(const aclTensor *const *tensors, uint64_t num, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_VSTACK; e->out = out; for (uint64_t i = 0; i < num; i++) e->inputs.push_back(tensors[i]); *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnVstack)

// Unique/UniqueConsecutive: (self, valuesOut, countOut, inverseOut, countsOut)
aclnnStatus aclnnUniqueGetWorkspaceSize(const aclTensor *self, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_UNIQUE; e->a = self; e->out = valuesOut; e->out2 = countOut; e->c = inverseOut; e->mean = countsOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnUnique)
aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(const aclTensor *self, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)inverseOut; auto *e = new aclOpExecutor(); e->op = K_UNIQCONS; e->a = self; e->out = valuesOut; e->out2 = countOut; e->mean = countsOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnUniqueConsecutive)
} // extern "C"
