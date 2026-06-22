// Shape / index / layout ops (host-side over unified memory; dtype-agnostic byte copy where possible).
// Covers test_shape's set beyond the GPU permute/embedding in shape.mm: strided-slice, tile, pad, gather(V2),
// cat(list)/stack/split, grouped-matmul-weightlist, arange, onehot, flip, roll, repeat-interleave, expand,
// scatter-add, masked-select, nonzero, embedding-dense-backward, and TransData NZ/FZ/5HD layout round-trips.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <vector>
#include <cmath>

namespace {
char *BP(const aclTensor *t) { return (char *)t->data + (size_t)t->offset * dtype_size(t->dtype); }
size_t ES(const aclTensor *t) { return dtype_size(t->dtype); }
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
void contig(const aclTensor *t, std::vector<int64_t> &str) {
    int nd = (int)t->viewDims.size(); str.resize(nd); int64_t s = 1;
    for (int d = nd - 1; d >= 0; --d) { str[d] = s; s *= t->viewDims[d]; }
}

// ---- index-mapping ops: out (contiguous) element g -> source byte offset ----
aclnnStatus run_strided_slice(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)a->viewDims.size();
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    const auto &v = e->axes;   // [begin(nd), end(nd), strides(nd)]
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; ioff += (v[d] + k * v[2 * nd + d]) * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_tile(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size();
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; ioff += (k % a->viewDims[d]) * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_pad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size();
    std::vector<int64_t> istr; contig(a, istr); float val = (float)e->alpha; float *in = FP(a), *out = FP(o);
    const auto &pad = e->axes;   // [lo0,hi0,lo1,hi1,...]
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0; bool inside = true;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t ii = k - pad[2 * d]; if (ii < 0 || ii >= a->viewDims[d]) { inside = false; } ioff += ii * istr[d]; }
        out[g] = inside ? in[ioff] : val; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_flip(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size(); int dim = (int)e->dim;
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t ik = (d == dim) ? (a->viewDims[d] - 1 - k) : k; ioff += ik * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_roll(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size(); int dim = (int)e->dim; int64_t sh = e->m;
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t D = a->viewDims[d]; int64_t ik = (d == dim) ? ((k - sh % D + D) % D) : k; ioff += ik * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_expand(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size(), and_ = (int)a->viewDims.size();
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int ad = d - (nd - and_); if (ad >= 0 && a->viewDims[ad] != 1) ioff += k * istr[ad]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_gather(aclOpExecutor *e, aclrtStream s) {   // index_select along dim, 1D index
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int nd = (int)o->viewDims.size(); int dim = (int)e->dim;
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o); const int64_t *ix = (const int64_t *)BP(idx);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t ik = (d == dim) ? ix[k] : k; ioff += ik * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_gather_v2(aclOpExecutor *e, aclrtStream s) {   // batched (batchDims=1, axis): out[b,i,post]=self[b,idx[b,i],post]
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int axis = (int)e->dim;
    int64_t B = a->viewDims[0], A = a->viewDims[axis], post = 1; for (int d = axis + 1; d < (int)a->viewDims.size(); ++d) post *= a->viewDims[d];
    int64_t I = idx->numel() / B; size_t esz = ES(a); char *in = BP(a), *out = BP(o); const int64_t *ix = (const int64_t *)BP(idx);
    for (int64_t b = 0; b < B; ++b) for (int64_t i = 0; i < I; ++i) { int64_t g = ix[b * I + i];
        memcpy(out + (size_t)((b * I + i) * post) * esz, in + (size_t)((b * A + g) * post) * esz, (size_t)post * esz); }
    return ACLNN_SUCCESS;
}
// ---- list ops ----
aclnnStatus run_cat(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *o = e->out; int dim = (int)e->dim; int nd = (int)o->viewDims.size();
    int64_t outer = 1, inner = 1; for (int d = 0; d < dim; ++d) outer *= o->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= o->viewDims[d];
    int64_t dimTotal = o->viewDims[dim]; size_t esz = ES(o); char *dst = BP(o); int64_t catoff = 0;
    for (const aclTensor *t : e->inputs) { int64_t dk = t->viewDims[dim]; char *src = BP(t);
        for (int64_t ot = 0; ot < outer; ++ot) for (int64_t k = 0; k < dk; ++k)
            memcpy(dst + (size_t)((ot * dimTotal + catoff + k) * inner) * esz, src + (size_t)((ot * dk + k) * inner) * esz, (size_t)inner * esz);
        catoff += dk; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_stack(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *o = e->out; int dim = (int)e->dim; int64_t K = (int64_t)e->inputs.size();
    int64_t outer = 1, inner = 1; for (int d = 0; d < dim; ++d) outer *= o->viewDims[d]; for (int d = dim + 1; d < (int)o->viewDims.size(); ++d) inner *= o->viewDims[d];
    size_t esz = ES(o); char *dst = BP(o);
    for (int64_t k = 0; k < K; ++k) { char *src = BP(e->inputs[k]);
        for (int64_t ot = 0; ot < outer; ++ot) memcpy(dst + (size_t)((ot * K + k) * inner) * esz, src + (size_t)(ot * inner) * esz, (size_t)inner * esz); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_split(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; int dim = (int)e->dim; int nd = (int)a->viewDims.size();
    int64_t outer = 1, inner = 1; for (int d = 0; d < dim; ++d) outer *= a->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= a->viewDims[d];
    int64_t dimTotal = a->viewDims[dim]; size_t esz = ES(a); char *src = BP(a); int64_t catoff = 0;
    for (const aclTensor *t : e->inputs) { aclTensor *to = (aclTensor *)t; int64_t dk = to->viewDims[dim]; char *dst = BP(to);
        for (int64_t ot = 0; ot < outer; ++ot) for (int64_t k = 0; k < dk; ++k)
            memcpy(dst + (size_t)((ot * dk + k) * inner) * esz, src + (size_t)((ot * dimTotal + catoff + k) * inner) * esz, (size_t)inner * esz);
        catoff += dk; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_gmm(aclOpExecutor *e, aclrtStream s) {   // x[M,K], weights[E]=[K,N], groupList[E] -> out[M,N]
    drain(s); const aclTensor *x = e->a; aclTensor *o = e->out; const float *xp = FP(x); float *op = FP(o);
    int64_t K = x->viewDims[1], N = o->viewDims[1]; int64_t off = 0;
    for (size_t e2 = 0; e2 < e->inputs.size(); ++e2) { const float *w = FP(e->inputs[e2]); int64_t rows = e->axes[e2];
        for (int64_t r = 0; r < rows; ++r) { int64_t row = off + r;
            for (int64_t n = 0; n < N; ++n) { double acc = 0; for (int64_t k = 0; k < K; ++k) acc += (double)xp[row * K + k] * w[k * N + n]; op[row * N + n] = (float)acc; } }
        off += rows; }
    return ACLNN_SUCCESS;
}
// ---- generators / scatter / select ----
aclnnStatus run_arange(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *o = e->out; float *op = FP(o); double start = e->dscalars[0], step = e->dscalars[2];
    for (int64_t i = 0; i < o->numel(); ++i) op[i] = (float)(start + i * step);
    return ACLNN_SUCCESS;
}
aclnnStatus run_onehot(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *ids = e->a; aclTensor *o = e->out; int64_t L = ids->numel(), C = e->m; const int64_t *ix = (const int64_t *)BP(ids); float *op = FP(o);
    float on = (float)e->dscalars[0], off = (float)e->dscalars[1];
    for (int64_t l = 0; l < L; ++l) for (int64_t c = 0; c < C; ++c) op[l * C + c] = (c == ix[l]) ? on : off;
    return ACLNN_SUCCESS;
}
aclnnStatus run_repeat_interleave(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t r = e->m; size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t i = 0; i < o->numel(); ++i) memcpy(out + (size_t)i * esz, in + (size_t)(i / r) * esz, esz);
    return ACLNN_SUCCESS;
}
aclnnStatus run_scatter_add(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *self = e->a, *idx = e->b, *src = e->c; aclTensor *o = e->out;
    int64_t R = self->viewDims[0], D = self->numel() / R, L = idx->numel(); const int64_t *ix = (const int64_t *)BP(idx);
    const float *sp = FP(self), *srp = FP(src); float *op = FP(o);
    for (int64_t i = 0; i < self->numel(); ++i) op[i] = sp[i];
    for (int64_t l = 0; l < L; ++l) { int64_t row = ix[l]; for (int64_t d = 0; d < D; ++d) op[row * D + d] += srp[l * D + d]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_masked_select(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a, *m = e->b; aclTensor *o = e->out, *cnt = e->out2;
    int64_t n = a->numel(); const float *ap = FP(a); const uint8_t *mp = (const uint8_t *)BP(m); float *op = FP(o); int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) if (mp[i]) op[k++] = ap[i];
    *(int64_t *)BP(cnt) = k; return ACLNN_SUCCESS;
}
aclnnStatus run_nonzero(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out, *cnt = e->out2;
    int64_t n = a->numel(); const float *ap = FP(a); int64_t *op = (int64_t *)BP(o); int64_t k = 0;
    for (int64_t i = 0; i < n; ++i) if (ap[i] != 0.f) op[k++] = i;
    *(int64_t *)BP(cnt) = k; return ACLNN_SUCCESS;
}
aclnnStatus run_embed_bwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *grad = e->a, *ids = e->b; aclTensor *gw = e->out;
    int64_t L = ids->numel(), D = grad->numel() / L, V = gw->numel() / D; const int64_t *ix = (const int64_t *)BP(ids); const float *g = FP(grad); float *w = FP(gw);
    for (int64_t i = 0; i < V * D; ++i) w[i] = 0.f;
    for (int64_t l = 0; l < L; ++l) for (int64_t d = 0; d < D; ++d) w[ix[l] * D + d] += g[l * D + d];
    return ACLNN_SUCCESS;
}
// ---- TransData layout round-trips (F=16; exact formulas matching the test) ----
aclnnStatus run_transdata(aclOpExecutor *e, aclrtStream s) {
    drain(s); const int F = 16; int kind = (int)e->m; const aclTensor *a = e->a; aclTensor *o = e->out;
    if (kind == 0 || kind == 1) {   // ND2NZ (0) / NZ2ND (1): ND[M,N], NZ[N1,M1,F,F]
        const aclTensor *nd = (kind == 0) ? a : o; int64_t M = nd->viewDims[0], N = nd->viewDims[1], M1 = (M + F - 1) / F, M1n = M1;
        (void)M1n; float *X = (kind == 0) ? FP((aclTensor *)a) : FP(o), *Z = (kind == 0) ? FP(o) : FP((aclTensor *)a);
        if (kind == 0) memset(Z, 0, (size_t)((N + F - 1) / F) * M1 * F * F * 4);
        for (int64_t i = 0; i < M; ++i) for (int64_t j = 0; j < N; ++j) { int64_t i1 = i / F, i0 = i % F, j1 = j / F, j0 = j % F, off = ((j1 * M1 + i1) * F + i0) * F + j0;
            if (kind == 0) Z[off] = X[i * N + j]; else X[i * N + j] = Z[off]; }
    } else if (kind == 2 || kind == 3) {   // ND2FZ (2) / FZ2ND (3): ND[K,N], FZ[K1,N1,F,F]
        const aclTensor *nd = (kind == 2) ? a : o; int64_t K = nd->viewDims[0], N = nd->viewDims[1], N1 = (N + F - 1) / F;
        float *X = (kind == 2) ? FP((aclTensor *)a) : FP(o), *Z = (kind == 2) ? FP(o) : FP((aclTensor *)a);
        if (kind == 2) memset(Z, 0, (size_t)((K + F - 1) / F) * N1 * F * F * 4);
        for (int64_t k = 0; k < K; ++k) for (int64_t n = 0; n < N; ++n) { int64_t k1 = k / F, k0 = k % F, n1 = n / F, n0 = n % F, off = ((k1 * N1 + n1) * F + k0) * F + n0;
            if (kind == 2) Z[off] = X[k * N + n]; else X[k * N + n] = Z[off]; }
    } else {   // NCHW2NC1HWC0 (4) / NC1HWC0toNCHW (5): NCHW[N,C,H,W], 5HD[N,C1,H,W,F]
        const aclTensor *nchw = (kind == 4) ? a : o; int64_t Nn = nchw->viewDims[0], C = nchw->viewDims[1], H = nchw->viewDims[2], W = nchw->viewDims[3], C1 = (C + F - 1) / F;
        float *X = (kind == 4) ? FP((aclTensor *)a) : FP(o), *Z = (kind == 4) ? FP(o) : FP((aclTensor *)a);
        if (kind == 4) memset(Z, 0, (size_t)Nn * C1 * H * W * F * 4);
        for (int64_t n = 0; n < Nn; ++n) for (int64_t c = 0; c < C; ++c) for (int64_t h = 0; h < H; ++h) for (int64_t w = 0; w < W; ++w) {
            int64_t c1 = c / F, c0 = c % F, off = ((((n * C1 + c1) * H + h) * W + w) * F) + c0, ndi = ((n * C + c) * H + h) * W + w;
            if (kind == 4) Z[off] = X[ndi]; else X[ndi] = Z[off]; }
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnStridedSliceGetWorkspaceSize(const aclTensor *self, const aclIntArray *begin, const aclIntArray *end, const aclIntArray *strides, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out;
    e->axes = begin->v; e->axes.insert(e->axes.end(), end->v.begin(), end->v.end()); e->axes.insert(e->axes.end(), strides->v.begin(), strides->v.end()); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnStridedSlice, run_strided_slice)
aclnnStatus aclnnTileGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)repeats; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTile, run_tile)
aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, const aclScalar *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !padding || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->axes = padding->v; e->alpha = value ? value->v : 0.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnConstantPadNd, run_pad)
aclnnStatus aclnnGatherGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->out = out; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGather, run_gather)
aclnnStatus aclnnGatherV2GetWorkspaceSize(const aclTensor *self, int64_t batchDims, int64_t axis, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)batchDims; if (!self || !index || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->out = out; e->dim = axis; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGatherV2, run_gather_v2)
aclnnStatus aclnnCatGetWorkspaceSize(const aclTensor *const *tensors, uint64_t num, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->out = out; e->dim = dim; for (uint64_t i = 0; i < num; ++i) e->inputs.push_back(tensors[i]); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCat, run_cat)
aclnnStatus aclnnCatListGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->out = out; e->dim = dim; e->inputs = tensors->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCatList, run_cat)
aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList *tensors, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->out = out; e->dim = dim; e->inputs = tensors->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnStack, run_stack)
aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *const *outputs, uint64_t num, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !outputs || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->dim = dim; for (uint64_t i = 0; i < num; ++i) e->inputs.push_back(outputs[i]); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSplitWithSize, run_split)
aclnnStatus aclnnGroupedMatmulWeightListGetWorkspaceSize(const aclTensor *x, const aclTensorList *weights, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weights || !groupList || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->inputs = weights->v; e->axes = groupList->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupedMatmulWeightList, run_gmm)
aclnnStatus aclnnArangeGetWorkspaceSize(double start, double end, double step, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->out = out; e->dscalars = {start, end, step}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnArange, run_arange)
aclnnStatus aclnnOneHotGetWorkspaceSize(const aclTensor *ids, int64_t numClasses, double onValue, double offValue, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ids || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = ids; e->out = out; e->m = numClasses; e->dscalars = {onValue, offValue}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnOneHot, run_onehot)
aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFlip, run_flip)
aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t shift, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = shift; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRoll, run_roll)
aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(const aclTensor *self, int64_t repeats, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = repeats; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleave, run_repeat_interleave)
aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnExpand, run_expand)
aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->c = src; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScatterAdd, run_scatter_add)
aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mask || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = mask; e->out = out; e->out2 = countOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaskedSelect, run_masked_select)
aclnnStatus aclnnNonzeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->out2 = countOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNonzero, run_nonzero)
aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *ids, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!grad || !ids || !gradWeight || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = grad; e->b = ids; e->out = gradWeight; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnEmbeddingDenseBackward, run_embed_bwd)
#define TD(NAME, K) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = K; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(NAME, run_transdata)
TD(aclnnTransDataND2NZ, 0) TD(aclnnTransDataNZ2ND, 1) TD(aclnnTransDataND2FZ, 2) TD(aclnnTransDataFZ2ND, 3)
TD(aclnnTransDataNCHW2NC1HWC0, 4) TD(aclnnTransDataNC1HWC0toNCHW, 5)
} // extern "C"
