// Indexing extensions (test_index): triangular/trace/diagonal, bincount/searchsorted, index_add/fill/copy,
// scatter max/min/mul/nd, take/take_along_dim, masked_scatter, narrow, reshape/movedim/rot90, 2d pads,
// im2col, pixel(un)shuffle, channel_shuffle, gather_nd, tf_scatter_add, embedding_renorm, topk_topp,
// pa_kv_cache, lightning_indexer. Host over unified memory (exact). dtype: fp32 + int32/int64 where noted.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
const int64_t *I64(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

aclnnStatus run_tri(aclOpExecutor *e, aclrtStream s) {   // m=1 lower(tril), 0 upper(triu); dim=k
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int M = (int)a->viewDims[0], N = (int)a->viewDims[1]; int64_t k = e->dim; bool lower = e->m;
    const float *x = FP(a); float *y = FP(o);
    for (int r = 0; r < M; ++r) for (int c = 0; c < N; ++c) { bool keep = lower ? (c <= r + k) : (c >= r + k); y[r * N + c] = keep ? x[r * N + c] : 0.f; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_trace(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; int M = (int)a->viewDims[0], N = (int)a->viewDims[1]; const float *x = FP(a); double acc = 0;
    for (int i = 0; i < std::min(M, N); ++i) acc += x[i * N + i]; FP(e->out)[0] = (float)acc; return ACLNN_SUCCESS;
}
aclnnStatus run_diagonal(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int M = (int)a->viewDims[0], N = (int)a->viewDims[1]; int off = (int)e->dim;
    int roff = off >= 0 ? 0 : -off, coff = off >= 0 ? off : 0, len = std::min(M - roff, N - coff); const float *x = FP(a); float *y = FP(o);
    for (int i = 0; i < len; ++i) y[i] = x[(roff + i) * N + (coff + i)]; return ACLNN_SUCCESS;
}
aclnnStatus run_diagflat(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int L = (int)a->numel(), off = (int)e->dim, S = (int)o->viewDims[0]; const float *v = FP(a); float *y = FP(o);
    for (int r = 0; r < S; ++r) for (int c = 0; c < S; ++c) y[r * S + c] = (c == r + off && r < L) ? v[r] : 0.f; return ACLNN_SUCCESS;
}
aclnnStatus run_bincount(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int n = (int)a->numel(), C = (int)e->m; const int32_t *x = (const int32_t *)a->data + a->offset; int64_t *c = (int64_t *)o->data + o->offset;
    for (int i = 0; i < C; ++i) c[i] = 0; for (int i = 0; i < n; ++i) if (x[i] >= 0 && x[i] < C) c[x[i]]++; return ACLNN_SUCCESS;
}
aclnnStatus run_searchsorted(aclOpExecutor *e, aclrtStream s) {   // boundaries=a, values=b, right=causal, out int64
    drain(s); const aclTensor *bnd = e->a, *val = e->b; aclTensor *o = e->out; int B = (int)bnd->numel(), nv = (int)val->numel(); bool right = e->causal;
    const float *bp = FP(bnd), *vp = FP(val); int64_t *op = (int64_t *)o->data + o->offset;
    for (int i = 0; i < nv; ++i) { int64_t r = 0; for (int j = 0; j < B; ++j) if (right ? (bp[j] <= vp[i]) : (bp[j] < vp[i])) r++; op[i] = r; } return ACLNN_SUCCESS;
}
aclnnStatus run_searchsorteds(aclOpExecutor *e, aclrtStream s) {   // seq=a, scalar=alpha, right=causal
    drain(s); const aclTensor *seq = e->a; int B = (int)seq->numel(); bool right = e->causal; const float *sp = FP(seq); float v = (float)e->alpha; int64_t r = 0;
    for (int j = 0; j < B; ++j) if (right ? (sp[j] <= v) : (sp[j] < v)) r++; *((int64_t *)e->out->data + e->out->offset) = r; return ACLNN_SUCCESS;
}
// index_add/fill/copy/inplace-copy along dim0: m=0 add(alpha), 1 fill(value), 2 copy
aclnnStatus run_index_rows(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *self = e->a, *idx = e->b, *src = e->c; aclTensor *o = e->out ? e->out : (aclTensor *)self;
    int V = (int)self->viewDims[0], row = (int)(self->numel() / V), L = (int)idx->numel(); const int64_t *ix = I64(idx);
    const float *sp = FP(self), *srp = src ? FP(src) : nullptr; float *op = FP(o); double param = e->alpha;
    if (op != sp) for (int64_t i = 0; i < self->numel(); ++i) op[i] = sp[i];
    for (int l = 0; l < L; ++l) { int r = (int)ix[l]; for (int c = 0; c < row; ++c) {
        if (e->m == 0) op[r * row + c] += param * srp[l * row + c]; else if (e->m == 1) op[r * row + c] = (float)param; else op[r * row + c] = srp[l * row + c]; } }
    return ACLNN_SUCCESS;
}
// scatter max/min/mul/add along dim0: m=0 max,1 min,2 mul,3 add
aclnnStatus run_scatter_reduce(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *self = e->a, *idx = e->b, *src = e->c; aclTensor *o = e->out;
    int V = (int)self->viewDims[0], row = (int)(self->numel() / V), L = (int)idx->numel(); const int64_t *ix = I64(idx);
    const float *sp = FP(self), *srp = FP(src); float *op = FP(o);
    for (int64_t i = 0; i < self->numel(); ++i) op[i] = sp[i];
    for (int l = 0; l < L; ++l) { int r = (int)ix[l]; for (int c = 0; c < row; ++c) { float v = srp[l * row + c]; float &d = op[r * row + c];
        if (e->m == 0) d = std::max(d, v); else if (e->m == 1) d = std::min(d, v); else if (e->m == 2) d *= v; else if (e->m == 4) d = v; else d += v; } }   // m4 = replace (scatter reduce=none)
    return ACLNN_SUCCESS;
}
aclnnStatus run_take(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int k = (int)idx->numel(); const float *x = FP(a); const int64_t *ix = I64(idx); float *y = FP(o);
    for (int i = 0; i < k; ++i) y[i] = x[ix[i]]; return ACLNN_SUCCESS;
}
aclnnStatus run_take_along(aclOpExecutor *e, aclrtStream s) {   // dim=last; self[M,N], idx[M,N]
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int M = (int)a->viewDims[0], N = (int)a->viewDims[1]; const float *x = FP(a); const int64_t *ix = I64(idx); float *y = FP(o);
    for (int r = 0; r < M; ++r) for (int c = 0; c < N; ++c) y[r * N + c] = x[r * N + ix[r * N + c]]; return ACLNN_SUCCESS;
}
aclnnStatus run_masked_scatter(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *self = e->a, *m = e->b, *src = e->c; aclTensor *o = e->out; int n = (int)self->numel();
    const float *sp = FP(self), *srp = FP(src); const uint8_t *mp = (const uint8_t *)m->data + m->offset; float *op = FP(o); int pos = 0;
    for (int i = 0; i < n; ++i) op[i] = mp[i] ? srp[pos++] : sp[i]; return ACLNN_SUCCESS;
}
aclnnStatus run_narrow(aclOpExecutor *e, aclrtStream s) {   // dim, start, length in dim/m/(reduceCount)
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)a->viewDims.size(); int dim = (int)e->dim; int64_t start = e->m;
    int64_t outer = 1, inner = 1; for (int d = 0; d < dim; ++d) outer *= a->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= a->viewDims[d];
    int64_t Din = a->viewDims[dim], Dout = o->viewDims[dim]; const float *x = FP(a); float *y = FP(o);
    for (int64_t ot = 0; ot < outer; ++ot) for (int64_t k = 0; k < Dout; ++k) for (int64_t in = 0; in < inner; ++in)
        y[(ot * Dout + k) * inner + in] = x[(ot * Din + start + k) * inner + in];
    return ACLNN_SUCCESS;
}
aclnnStatus run_reshape(aclOpExecutor *e, aclrtStream s) {
    drain(s); memcpy(FP(e->out), FP(e->a), (size_t)e->a->numel() * 4); return ACLNN_SUCCESS;
}
aclnnStatus run_movedim(aclOpExecutor *e, aclrtStream s) {   // src=m, dst=dim
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)a->viewDims.size(); int src = (int)e->m, dst = (int)e->dim;
    std::vector<int> perm; for (int d = 0; d < nd; ++d) if (d != src) perm.push_back(d); perm.insert(perm.begin() + dst, src);
    std::vector<int64_t> istr(nd); { int64_t s2 = 1; for (int d = nd - 1; d >= 0; --d) { istr[d] = s2; s2 *= a->viewDims[d]; } }
    const float *x = FP(a); float *y = FP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0; for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; ioff += k * istr[perm[d]]; } y[g] = x[ioff]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_rot90(aclOpExecutor *e, aclrtStream s) {   // k=m
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int H = (int)a->viewDims[0], W = (int)a->viewDims[1], k = ((int)e->m) % 4; int oH = (int)o->viewDims[0], oW = (int)o->viewDims[1];
    const float *x = FP(a); float *y = FP(o);
    for (int r = 0; r < oH; ++r) for (int c = 0; c < oW; ++c) { int ih, iw; if (k == 1) { ih = c; iw = W - 1 - r; } else if (k == 2) { ih = H - 1 - r; iw = W - 1 - c; } else { ih = H - 1 - c; iw = r; } y[r * oW + c] = x[ih * W + iw]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_pad2d(aclOpExecutor *e, aclrtStream s) {   // m=0 reflect,1 replicate,2 circular; axes=[left,right,top,bottom]
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int H = (int)a->viewDims[2], W = (int)a->viewDims[3], oH = (int)o->viewDims[2], oW = (int)o->viewDims[3];
    int NC = (int)(a->numel() / (H * W)); int pl = (int)e->axes[0], pt = (int)e->axes[2]; const float *x = FP(a); float *y = FP(o); int mode = (int)e->m;
    auto refl = [](int v, int n) { if (n == 1) return 0; while (v < 0 || v >= n) { if (v < 0) v = -v; if (v >= n) v = 2 * (n - 1) - v; } return v; };
    auto repl = [](int v, int n) { return v < 0 ? 0 : (v >= n ? n - 1 : v); };
    auto circ = [](int v, int n) { return ((v % n) + n) % n; };
    for (int z = 0; z < NC; ++z) for (int oh = 0; oh < oH; ++oh) for (int ow = 0; ow < oW; ++ow) { int ih = oh - pt, iw = ow - pl;
        if (mode == 0) { ih = refl(ih, H); iw = refl(iw, W); } else if (mode == 1) { ih = repl(ih, H); iw = repl(iw, W); } else { ih = circ(ih, H); iw = circ(iw, W); }
        y[(z * oH + oh) * oW + ow] = x[(z * H + ih) * W + iw]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_im2col(aclOpExecutor *e, aclrtStream s) {   // axes packs [kh,kw,dh,dw,ph,pw,sh,sw]
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int C = (int)a->viewDims[1], H = (int)a->viewDims[2], W = (int)a->viewDims[3];
    const auto &p = e->axes; int kh = p[0], kw = p[1], dh = p[2], dw = p[3], ph = p[4], pw = p[5], sh = p[6], sw = p[7];
    int oH = (H + 2 * ph - dh * (kh - 1) - 1) / sh + 1, oW = (W + 2 * pw - dw * (kw - 1) - 1) / sw + 1, L = oH * oW, K = C * kh * kw;
    const float *x = FP(a); float *y = FP(o);
    for (int kr = 0; kr < K; ++kr) for (int l = 0; l < L; ++l) { int ow = l % oW, oh = l / oW, kj = kr % kw, ki = (kr / kw) % kh, c = kr / (kh * kw);
        int ih = oh * sh - ph + ki * dh, iw = ow * sw - pw + kj * dw; y[kr * L + l] = (ih >= 0 && ih < H && iw >= 0 && iw < W) ? x[(c * H + ih) * W + iw] : 0.f; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_pixelshuffle(aclOpExecutor *e, aclrtStream s) {   // m=r, dim: 0 shuffle, 1 unshuffle
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int r = (int)e->m;
    if (e->dim == 0) { int C = (int)o->viewDims[1], H = (int)a->viewDims[2], W = (int)a->viewDims[3]; const float *x = FP(a); float *y = FP(o);
        for (int c = 0; c < C; ++c) for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) for (int i = 0; i < r; ++i) for (int j = 0; j < r; ++j)
            y[(c * (H * r) + h * r + i) * (W * r) + w * r + j] = x[((c * r * r + i * r + j) * H + h) * W + w]; }
    else { int C = (int)o->viewDims[1] / (r * r), H = (int)o->viewDims[2], W = (int)o->viewDims[3]; const float *x = FP(a); float *y = FP(o);
        for (int c = 0; c < C; ++c) for (int h = 0; h < H; ++h) for (int w = 0; w < W; ++w) for (int i = 0; i < r; ++i) for (int j = 0; j < r; ++j)
            y[((c * r * r + i * r + j) * H + h) * W + w] = x[(c * (H * r) + h * r + i) * (W * r) + w * r + j]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_channelshuffle(aclOpExecutor *e, aclrtStream s) {   // m=groups
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int C = (int)a->viewDims[1], HW = (int)(a->numel() / C), g = (int)e->m, cpg = C / g; const float *x = FP(a); float *y = FP(o);
    for (int c = 0; c < C; ++c) { int src = (c % g) * cpg + c / g; for (int h = 0; h < HW; ++h) y[c * HW + h] = x[src * HW + h]; } return ACLNN_SUCCESS;
}
aclnnStatus run_gathernd(aclOpExecutor *e, aclrtStream s) {   // indices last-dim K; gather self[idx]
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int K = (int)idx->viewDims.back(); int rows = (int)(idx->numel() / K);
    int sliceNd = (int)a->viewDims.size() - K; int64_t slice = 1; for (int d = K; d < (int)a->viewDims.size(); ++d) slice *= a->viewDims[d]; (void)sliceNd;
    std::vector<int64_t> astr; { int64_t s2 = 1; astr.resize(a->viewDims.size()); for (int d = (int)a->viewDims.size() - 1; d >= 0; --d) { astr[d] = s2; s2 *= a->viewDims[d]; } }
    const float *x = FP(a); const int64_t *ix = I64(idx); float *y = FP(o);
    for (int r = 0; r < rows; ++r) { int64_t off = 0; for (int kk = 0; kk < K; ++kk) off += ix[r * K + kk] * astr[kk]; memcpy(y + (size_t)r * slice, x + off, (size_t)slice * 4); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_scatternd(aclOpExecutor *e, aclrtStream s) {   // self + indices[rows,K] + updates -> out
    drain(s); const aclTensor *self = e->a, *idx = e->b, *upd = e->c; aclTensor *o = e->out; int K = (int)idx->viewDims.back(); int rows = (int)(idx->numel() / K);
    int64_t slice = 1; for (int d = K; d < (int)self->viewDims.size(); ++d) slice *= self->viewDims[d];
    std::vector<int64_t> astr; { int64_t s2 = 1; astr.resize(self->viewDims.size()); for (int d = (int)self->viewDims.size() - 1; d >= 0; --d) { astr[d] = s2; s2 *= self->viewDims[d]; } }
    const float *sp = FP(self), *up = FP(upd); const int64_t *ix = I64(idx); float *op = FP(o);
    for (int64_t i = 0; i < self->numel(); ++i) op[i] = sp[i];
    for (int r = 0; r < rows; ++r) { int64_t off = 0; for (int kk = 0; kk < K; ++kk) off += ix[r * K + kk] * astr[kk]; memcpy(op + off, up + (size_t)r * slice, (size_t)slice * 4); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_embed_renorm(aclOpExecutor *e, aclrtStream s) {   // self in place; indices=b; maxNorm=alpha
    drain(s); aclTensor *self = (aclTensor *)e->a; const aclTensor *idx = e->b; int D = (int)self->viewDims[1], L = (int)idx->numel(); double maxn = e->alpha; const int64_t *ix = I64(idx); float *w = FP(self);
    for (int l = 0; l < L; ++l) { int r = (int)ix[l]; double n0 = 0; for (int d = 0; d < D; ++d) n0 += (double)w[r * D + d] * w[r * D + d]; n0 = std::sqrt(n0);
        if (n0 > maxn) { double sc = maxn / (n0 + 1e-7); for (int d = 0; d < D; ++d) w[r * D + d] = (float)(w[r * D + d] * sc); } }
    return ACLNN_SUCCESS;
}
aclnnStatus run_topk_topp(aclOpExecutor *e, aclrtStream s) {   // logits[R,V]; topk=m; topp=alpha
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int R = (int)a->viewDims[0], V = (int)a->viewDims[1]; int K = (int)e->m; double topp = e->alpha; const float *x = FP(a); float *y = FP(o);
    for (int r = 0; r < R; ++r) { std::vector<int> id(V); for (int v = 0; v < V; ++v) id[v] = v;
        std::sort(id.begin(), id.end(), [&](int p, int q) { return x[r * V + p] > x[r * V + q]; });
        int keepN = std::min(K, V);
        // nucleus: softmax over the top-K, keep while cumulative prob < topp (always keep first)
        double mx = x[r * V + id[0]], sum = 0; for (int i = 0; i < keepN; ++i) sum += std::exp(x[r * V + id[i]] - mx);
        double cum = 0; int kept = 0; for (int i = 0; i < keepN; ++i) { cum += std::exp(x[r * V + id[i]] - mx) / sum; kept = i + 1; if (cum >= topp) break; }
        std::vector<bool> keep(V, false); for (int i = 0; i < kept; ++i) keep[id[i]] = true;
        for (int v = 0; v < V; ++v) y[r * V + v] = keep[v] ? x[r * V + v] : -1e30f; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_pa_kv(aclOpExecutor *e, aclrtStream s) {   // keyCache=a, valueCache=b, slot=c, keyOut=out, valueOut=out2
    drain(s); const aclTensor *kc = e->a, *vc = e->b, *slot = e->c; aclTensor *ko = e->out, *vo = e->out2; int D = (int)kc->viewDims[1], K = (int)slot->numel(); const int64_t *sl = I64(slot);
    const float *kcp = FP(kc), *vcp = FP(vc); float *kop = FP(ko), *vop = FP(vo);
    for (int k = 0; k < K; ++k) for (int d = 0; d < D; ++d) { kop[k * D + d] = kcp[sl[k] * D + d]; vop[k * D + d] = vcp[sl[k] * D + d]; } return ACLNN_SUCCESS;
}
aclnnStatus run_lightning(aclOpExecutor *e, aclrtStream s) {   // q[Q,H,D], k[K,H,D], wq[Q,H] -> score[Q,K]
    drain(s); const aclTensor *q = e->a, *k = e->b, *wq = e->c; aclTensor *o = e->out; int Q = (int)q->viewDims[0], H = (int)q->viewDims[1], D = (int)q->viewDims[2], Kk = (int)k->viewDims[0];
    const float *qp = FP(q), *kp = FP(k), *wp = FP(wq); float *sc = FP(o);
    for (int qi = 0; qi < Q; ++qi) for (int ki = 0; ki < Kk; ++ki) { double acc = 0; for (int h = 0; h < H; ++h) { double dot = 0; for (int d = 0; d < D; ++d) dot += (double)qp[(qi * H + h) * D + d] * kp[(ki * H + h) * D + d]; acc += wp[qi * H + h] * std::max(dot, 0.0); } sc[qi * Kk + ki] = (float)acc; }
    return ACLNN_SUCCESS;
}
#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnTrilGetWorkspaceSize(const aclTensor *self, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=k; e->m=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTril, run_tri)
aclnnStatus aclnnTriuGetWorkspaceSize(const aclTensor *self, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=k; e->m=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTriu, run_tri)
aclnnStatus aclnnTraceGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTrace, run_trace)
aclnnStatus aclnnDiagonalGetWorkspaceSize(const aclTensor *self, int64_t off, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=off; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnDiagonal, run_diagonal)
aclnnStatus aclnnDiagFlatGetWorkspaceSize(const aclTensor *self, int64_t off, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=off; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnDiagFlat, run_diagflat)
aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor *self, int64_t numClasses, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=numClasses; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnBincount, run_bincount)
aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor *sorted, const aclTensor *values, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=sorted; e->b=values; e->out=out; e->causal=right; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSearchSorted, run_searchsorted)
aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor *seq, const aclScalar *value, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=seq; e->out=out; e->alpha=value->v; e->causal=right; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSearchSorteds, run_searchsorteds)
aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->alpha=alpha; e->m=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnIndexAdd, run_index_rows)
aclnnStatus aclnnIndexFillGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, double value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->out=out; e->alpha=value; e->m=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnIndexFill, run_index_rows)
aclnnStatus aclnnIndexCopyGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnIndexCopy, run_index_rows)
aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=nullptr; e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceIndexCopy, run_index_rows)
aclnnStatus aclnnScatterMaxGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->m=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnScatterMax, run_scatter_reduce)
aclnnStatus aclnnScatterMinGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->m=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnScatterMin, run_scatter_reduce)
aclnnStatus aclnnScatterMulGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnScatterMul, run_scatter_reduce)
aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->c=src; e->out=out; e->m = (reduce==1)?3:(reduce==2)?2:4; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnScatter, run_scatter_reduce)
aclnnStatus aclnnTfScatterAddGetWorkspaceSize(const aclTensor *ref, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=ref; e->b=indices; e->c=updates; e->out=out; e->m=3; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTfScatterAdd, run_scatter_reduce)
aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor *self, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTake, run_take)
aclnnStatus aclnnTakeAlongDimGetWorkspaceSize(const aclTensor *self, const aclTensor *index, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; auto *e=new aclOpExecutor(); e->a=self; e->b=index; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTakeAlongDim, run_take_along)
aclnnStatus aclnnMaskedScatterGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=mask; e->c=src; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMaskedScatter, run_masked_scatter)
aclnnStatus aclnnNarrowGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t length, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)length; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->m=start; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnNarrow, run_narrow)
aclnnStatus aclnnReshapeGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnReshape, run_reshape)
aclnnStatus aclnnMovedimGetWorkspaceSize(const aclTensor *self, int64_t src, int64_t dst, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=src; e->dim=dst; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMovedim, run_movedim)
aclnnStatus aclnnRot90GetWorkspaceSize(const aclTensor *self, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=k; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnRot90, run_rot90)
#define PAD2D(NAME, MODE) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->axes=padding->v; e->m=MODE; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_pad2d)
PAD2D(aclnnReflectionPad2d, 0) PAD2D(aclnnReplicationPad2d, 1) PAD2D(aclnnCircularPad2d, 2)
aclnnStatus aclnnIm2ColGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->axes={kernel->v[0],kernel->v[1],dilation->v[0],dilation->v[1],padding->v[0],padding->v[1],stride->v[0],stride->v[1]}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnIm2Col, run_im2col)
aclnnStatus aclnnPixelShuffleGetWorkspaceSize(const aclTensor *self, int64_t r, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=r; e->dim=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnPixelShuffle, run_pixelshuffle)
aclnnStatus aclnnPixelUnshuffleGetWorkspaceSize(const aclTensor *self, int64_t r, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=r; e->dim=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnPixelUnshuffle, run_pixelshuffle)
aclnnStatus aclnnChannelShuffleGetWorkspaceSize(const aclTensor *self, int64_t groups, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=groups; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnChannelShuffle, run_channelshuffle)
aclnnStatus aclnnGatherNdGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=indices; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnGatherNd, run_gathernd)
aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=indices; e->c=updates; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnScatterNdUpdate, run_scatternd)
aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *self, const aclTensor *indices, double maxNorm, double normType, uint64_t *ws, aclOpExecutor **ex) { (void)normType; auto *e=new aclOpExecutor(); e->a=self; e->b=indices; e->alpha=maxNorm; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnEmbeddingRenorm, run_embed_renorm)
aclnnStatus aclnnApplyTopKTopPGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=logits; e->out=out; e->m=topk; e->alpha=topp; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnApplyTopKTopP, run_topk_topp)
aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache, const aclTensor *slot, aclTensor *keyOut, aclTensor *valueOut, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=keyCache; e->b=valueCache; e->c=slot; e->out=keyOut; e->out2=valueOut; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnGatherPaKvCache, run_pa_kv)
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *weights, aclTensor *indexScore, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=query; e->b=key; e->c=weights; e->out=indexScore; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnLightningIndexer, run_lightning)
} // extern "C"
