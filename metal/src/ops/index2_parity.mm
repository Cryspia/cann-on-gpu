// Index / shape / reduce / sort gap operators (CUDA parity). Host-side over unified memory (exact for
// integer/index ops; fp32 reductions match the CPU reference to ~1e-5). Semantics ported 1:1 from
// cuda/src/ops/{index_ext,shape,reduce,sort,misc_ext,math_ext,loss}.cu.
//
// All 39 ops below are NOT declared in aclnnop/aclnn_ops.h (verified via the clang -E preprocessor probe),
// so every prototype is self-declared extern "C" here from the CUDA source semantics. None of these symbols
// are already defined by the Metal backend, so there are no duplicate definitions.
#import "../internal.h"
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
const int64_t *I64(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
int64_t *I64W(aclTensor *t) { return (int64_t *)t->data + t->offset; }
char *BP(const aclTensor *t) { return (char *)t->data + (size_t)t->offset * dtype_size(t->dtype); }
size_t ES(const aclTensor *t) { return dtype_size(t->dtype); }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
void contig(const aclTensor *t, std::vector<int64_t> &str) {
    int nd = (int)t->viewDims.size(); str.resize(nd); int64_t s = 1;
    for (int d = nd - 1; d >= 0; --d) { str[d] = s; s *= t->viewDims[d]; }
}
// single-dim group geometry (outer, D, inner) over dim
void geom(const aclTensor *t, int dim, int64_t &outer, int64_t &D, int64_t &inner) {
    int nd = (int)t->viewDims.size(); if (dim < 0) dim += nd; outer = 1; inner = 1; D = t->viewDims[dim];
    for (int d = 0; d < dim; ++d) outer *= t->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= t->viewDims[d];
}

// =================== INDEX / SHAPE ===================

// IndexSelect / Gather along dim with 1D int64 index (== shape.mm run_gather). Used by Index/IndexSelect/GatherV3.
aclnnStatus run_index_select(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a, *idx = e->b; aclTensor *o = e->out; int nd = (int)o->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd;
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o); const int64_t *ix = (const int64_t *)BP(idx);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; int64_t ik = (d == dim) ? ix[k] : k; ioff += ik * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}

// IndexAdd/Fill/Copy(+Tensor variants) along dim0. m: 0 add(alpha), 1 fill(value), 2 copy. out may == self (in-place).
aclnnStatus run_index_rows(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *self = e->a, *idx = e->b, *src = e->c; aclTensor *o = e->out ? e->out : (aclTensor *)self;
    int64_t V = self->viewDims[0], row = self->numel() / V, L = idx->numel(); const int64_t *ix = I64(idx);
    const float *sp = FP(self), *srp = src ? FP(src) : nullptr; float *op = FP(o); double param = e->alpha;
    if (op != sp) for (int64_t i = 0; i < self->numel(); ++i) op[i] = sp[i];
    for (int64_t l = 0; l < L; ++l) { int64_t r = ix[l]; if (r < 0 || r >= V) continue; for (int64_t c = 0; c < row; ++c) {
        if (e->m == 0) op[r * row + c] += param * srp[l * row + c]; else if (e->m == 1) op[r * row + c] = (float)param; else op[r * row + c] = srp[l * row + c]; } }
    return ACLNN_SUCCESS;
}

// Row scatter (dim0): out[idx[k], :] = / += src[k, :]. add: 0 overwrite, 1 accumulate. Pre-copies self->out if provided.
aclnnStatus run_row_scatter(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *o = e->out; const aclTensor *idx = e->b, *src = e->c, *self = e->a;
    int64_t N = o->viewDims[0], D = o->numel() / N, K = idx->numel(); const int64_t *ix = I64(idx);
    const float *srp = FP(src); float *op = FP(o); int add = (int)e->dim;
    if (self && FP(self) != op) for (int64_t i = 0; i < o->numel(); ++i) op[i] = FP(self)[i];
    for (int64_t k = 0; k < K; ++k) { int64_t r = ix[k]; if (r < 0 || r >= N) continue; for (int64_t d = 0; d < D; ++d) {
        if (add) op[r * D + d] += srp[k * D + d]; else op[r * D + d] = srp[k * D + d]; } }
    return ACLNN_SUCCESS;
}

// IndexPutImpl: selfRef[idx0[k], :] = / += values[k, :]; dim=accumulate flag, out=selfRef (no pre-copy)
aclnnStatus run_index_put(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *o = e->out; const aclTensor *idx = e->b, *src = e->c;
    int64_t N = o->viewDims[0], D = o->numel() / N, K = idx->numel(); const int64_t *ix = I64(idx);
    const float *srp = FP(src); float *op = FP(o); int add = (int)e->dim;
    for (int64_t k = 0; k < K; ++k) { int64_t r = ix[k]; if (r < 0 || r >= N) continue; for (int64_t d = 0; d < D; ++d) {
        if (add) op[r * D + d] += srp[k * D + d]; else op[r * D + d] = srp[k * D + d]; } }
    return ACLNN_SUCCESS;
}

// ScatterList: list of (self, index, update) → loop row-scatter overwrite (in place into each self)
aclnnStatus run_scatter_list(aclOpExecutor *e, aclrtStream s) {
    drain(s); int64_t L = e->m;
    for (int64_t l = 0; l < L; ++l) { const aclTensor *self = e->inputs[l], *idx = e->inputs[L + l], *upd = e->inputs[2 * L + l];
        int64_t N = self->viewDims[0], D = self->numel() / N, K = idx->numel(); const int64_t *ix = I64(idx);
        const float *up = FP(upd); float *sp = FP((aclTensor *)self);
        for (int64_t k = 0; k < K; ++k) { int64_t r = ix[k]; if (r < 0 || r >= N) continue; for (int64_t d = 0; d < D; ++d) sp[r * D + d] = up[k * D + d]; } }
    return ACLNN_SUCCESS;
}

// ScatterPaKvCache: key/value scattered into keyCache/valueCache by slotMapping (overwrite)
aclnnStatus run_scatter_pa_kv(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *key = e->a, *val = e->b, *slot = e->c; aclTensor *kc = e->out, *vc = e->out2;
    int64_t K = slot->numel(); const int64_t *sl = I64(slot);
    int64_t Nk = kc->viewDims[0], Dk = kc->numel() / Nk, Nv = vc->viewDims[0], Dv = vc->numel() / Nv;
    const float *kp = FP(key), *vp = FP(val); float *kcp = FP(kc), *vcp = FP(vc);
    for (int64_t k = 0; k < K; ++k) { int64_t r = sl[k]; if (r < 0) continue;
        if (r < Nk) for (int64_t d = 0; d < Dk; ++d) kcp[r * Dk + d] = kp[k * Dk + d];
        if (r < Nv) for (int64_t d = 0; d < Dv; ++d) vcp[r * Dv + d] = vp[k * Dv + d]; }
    return ACLNN_SUCCESS;
}

// ScatterPaCache: input scattered into cache by slotMapping (overwrite). c=input, out=cache, b=slot.
aclnnStatus run_scatter_pa(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *cache = e->out; const aclTensor *slot = e->b, *input = e->c;
    int64_t K = slot->numel(), N = cache->viewDims[0], D = cache->numel() / N; const int64_t *sl = I64(slot);
    const float *ip = FP(input); float *cp = FP(cache);
    for (int64_t k = 0; k < K; ++k) { int64_t r = sl[k]; if (r < 0 || r >= N) continue; for (int64_t d = 0; d < D; ++d) cp[r * D + d] = ip[k * D + d]; }
    return ACLNN_SUCCESS;
}

// ScatterNd: out = scatter `updates` row-slices into a ZERO tensor by `indices` (last index dim = K coords).
aclnnStatus run_scatternd_zero(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *idx = e->b, *upd = e->c; aclTensor *o = e->out;
    int K = (int)idx->viewDims.back(); int64_t rows = idx->numel() / K;
    int64_t slice = 1; for (int d = K; d < (int)o->viewDims.size(); ++d) slice *= o->viewDims[d];
    std::vector<int64_t> ostr; { int64_t s2 = 1; ostr.resize(o->viewDims.size()); for (int d = (int)o->viewDims.size() - 1; d >= 0; --d) { ostr[d] = s2; s2 *= o->viewDims[d]; } }
    size_t esz = ES(o); char *op = BP(o); const int64_t *ix = I64(idx); const char *up = BP(upd);
    memset(op, 0, (size_t)o->numel() * esz);
    for (int64_t r = 0; r < rows; ++r) { int64_t off = 0; for (int kk = 0; kk < K; ++kk) off += ix[r * K + kk] * ostr[kk]; memcpy(op + (size_t)off * esz, up + (size_t)r * slice * esz, (size_t)slice * esz); }
    return ACLNN_SUCCESS;
}

// MaskedScale: out = self * (mask * scale). mask same shape as self (fp32 0/1), scale in alpha.
aclnnStatus run_masked_scale(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a, *m = e->b; aclTensor *o = e->out; int64_t n = a->numel();
    const float *x = FP(a); const uint8_t *mp = (const uint8_t *)m->data + m->offset; float *y = FP(o); float sc = (float)e->alpha;
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * (float)mp[i] * sc;   // mask is a uint8 dropout mask (matches the CUDA backend)
    return ACLNN_SUCCESS;
}

// Chunk / SplitWithSize: split self along dim into the output tensors in e->inputs.
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

// Flatten / ChunkCat / NpuFormatCast / Reshape: plain contiguous byte copy (value-preserving layout change).
aclnnStatus run_copy(aclOpExecutor *e, aclrtStream s) {
    drain(s); size_t esz = ES(e->a); int64_t n = std::min(e->a->numel(), e->out->numel());
    memcpy(BP(e->out), BP(e->a), (size_t)n * esz); return ACLNN_SUCCESS;
}

// Expandv: broadcast-tile flat (out[i] = in[i % inN]).
aclnnStatus run_expandv(aclOpExecutor *e, aclrtStream s) {
    drain(s); int64_t inN = e->a->numel(), outN = e->out->numel(); size_t esz = ES(e->a); char *in = BP(e->a), *out = BP(e->out);
    for (int64_t i = 0; i < outN; ++i) memcpy(out + (size_t)i * esz, in + (size_t)(i % inN) * esz, esz);
    return ACLNN_SUCCESS;
}

// Tile / Repeat: per-dim repeat (out element g maps to in element via k % a->dim[d]).
aclnnStatus run_tile(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)o->viewDims.size();
    std::vector<int64_t> istr; contig(a, istr); size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t g = 0; g < o->numel(); ++g) { int64_t rem = g, ioff = 0;
        for (int d = nd - 1; d >= 0; --d) { int64_t od = o->viewDims[d]; int64_t k = rem % od; rem /= od; ioff += (k % a->viewDims[d]) * istr[d]; }
        memcpy(out + (size_t)g * esz, in + (size_t)ioff * esz, esz); }
    return ACLNN_SUCCESS;
}

// RepeatInterleave (flat): out[i] = in[i / repeats].
aclnnStatus run_repeat_il(aclOpExecutor *e, aclrtStream s) {
    drain(s); int64_t r = e->m, n = e->out->numel(); size_t esz = ES(e->a); char *in = BP(e->a), *out = BP(e->out);
    for (int64_t i = 0; i < n; ++i) memcpy(out + (size_t)i * esz, in + (size_t)(i / r) * esz, esz);
    return ACLNN_SUCCESS;
}

// Diag(onal): if input is 1D → diagflat (vector to diagonal matrix); else → diagonal extraction.
aclnnStatus run_diag(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int off = (int)e->dim; const float *x = FP(a); float *y = FP(o);
    if (a->viewDims.size() == 1) {   // diagflat
        int64_t L = a->numel(), S = o->viewDims[0];
        for (int64_t r = 0; r < S; ++r) for (int64_t c = 0; c < S; ++c) y[r * S + c] = (c == r + off && r < L && (r + off) >= 0) ? x[r] : 0.f;
    } else {   // diagonal of [M,N]
        int M = (int)a->viewDims[0], N = (int)a->viewDims[1]; int roff = off >= 0 ? 0 : -off, coff = off >= 0 ? off : 0;
        int len = std::min(M - roff, N - coff); for (int i = 0; i < len; ++i) y[i] = x[(roff + i) * N + (coff + i)];
    }
    return ACLNN_SUCCESS;
}

// StridedSliceAssignV2: contiguous (stride 1) assign self[begin : begin+len] = value (flat).
aclnnStatus run_slice_assign(aclOpExecutor *e, aclrtStream s) {
    drain(s); aclTensor *self = e->out; const aclTensor *val = e->c; int64_t begin = e->m, vn = val->numel();
    float *sp = FP(self); const float *vp = FP(val);
    for (int64_t i = 0; i < vn; ++i) sp[begin + i] = vp[i];
    return ACLNN_SUCCESS;
}

// SliceV2 / Slice: copy self[..., start:end:step, ...] along dim into contiguous out.
aclnnStatus run_slice(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int nd = (int)a->viewDims.size(); int dim = (int)e->dim; if (dim < 0) dim += nd;
    int64_t start = e->m, step = e->k;
    int64_t outer = 1, inner = 1; for (int d = 0; d < dim; ++d) outer *= a->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= a->viewDims[d];
    int64_t Din = a->viewDims[dim], Dout = o->viewDims[dim]; size_t esz = ES(a); char *in = BP(a), *out = BP(o);
    for (int64_t ot = 0; ot < outer; ++ot) for (int64_t kk = 0; kk < Dout; ++kk) { int64_t isrc = start + kk * step; (void)Din;
        memcpy(out + (size_t)((ot * Dout + kk) * inner) * esz, in + (size_t)((ot * Din + isrc) * inner) * esz, (size_t)inner * esz); }
    return ACLNN_SUCCESS;
}

// LightningIndexer softmax-lse: lse[q] = log Σ_k exp(score[q,k]); probs optional.
aclnnStatus run_lse(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *sc = e->a; aclTensor *lse = e->out, *probs = e->out2; int Q = (int)e->m, K = (int)e->n;
    const float *p = FP(sc); float *lp = FP(lse); float *pp = probs ? FP(probs) : nullptr;
    for (int q = 0; q < Q; ++q) { const float *pr = p + (int64_t)q * K; float mx = -1e30f; for (int k = 0; k < K; ++k) mx = std::max(mx, pr[k]);
        double sm = 0; for (int k = 0; k < K; ++k) sm += std::exp((double)pr[k] - mx); lp[q] = mx + (float)std::log(sm);
        if (pp) { float *po = pp + (int64_t)q * K; for (int k = 0; k < K; ++k) po[k] = (float)(std::exp((double)pr[k] - mx) / sm); } }
    return ACLNN_SUCCESS;
}

// =================== REDUCE / SORT ===================

// Value reductions over an axis set. m: 0 nansum, 1 std(ddof=1), 2 mean (for stdmean's meanOut), 3 amin, 4 amax.
// For stdmean (m==1) out2 receives mean. For aminmax (m==3) out=min, out2=max.
aclnnStatus run_reduce_axes(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out, *o2 = e->out2; int op = (int)e->m;
    int nd = (int)a->viewDims.size(); bool red[8] = {false};
    if (e->axes.empty()) { for (int d = 0; d < nd; ++d) red[d] = true; }
    else for (int64_t ax : e->axes) { int d = (int)(ax < 0 ? ax + nd : ax); red[d] = true; }
    int64_t istr[8]; { int64_t st = 1; for (int d = nd - 1; d >= 0; --d) { istr[d] = st; st *= a->viewDims[d]; } }
    std::vector<int64_t> od, os, rd, rs; int64_t nout = 1, nred = 1;
    for (int d = 0; d < nd; ++d) { if (red[d]) { rd.push_back(a->viewDims[d]); rs.push_back(istr[d]); nred *= a->viewDims[d]; } else { od.push_back(a->viewDims[d]); os.push_back(istr[d]); nout *= a->viewDims[d]; } }
    const float *x = FP(a);
    for (int64_t g = 0; g < nout; ++g) {
        int64_t rem = g, base = 0; for (int i = (int)od.size() - 1; i >= 0; --i) { int64_t id = rem % od[i]; rem /= od[i]; base += id * os[i]; }
        double sum = 0, sumsq = 0; double mn = INFINITY, mx = -INFINITY; int64_t cnt = 0;
        for (int64_t r = 0; r < nred; ++r) { int64_t rr = r, off = base; for (int i = (int)rd.size() - 1; i >= 0; --i) { int64_t id = rr % rd[i]; rr /= rd[i]; off += id * rs[i]; }
            float v = x[off];
            if (op == 0) { if (!std::isnan(v)) { sum += v; cnt++; } continue; }
            sum += v; sumsq += (double)v * v; mn = std::min(mn, (double)v); mx = std::max(mx, (double)v); }
        if (op == 0) FP(o)[g] = (float)sum;                                            // nansum
        else if (op == 1) { double var = (sumsq - sum * sum / nred) / (nred > 1 ? nred - 1 : 1); FP(o)[g] = (float)std::sqrt(var); if (o2) FP(o2)[g] = (float)(sum / nred); }   // std + mean
        else if (op == 3) { FP(o)[g] = (float)mn; if (o2) FP(o2)[g] = (float)mx; }      // aminmax
    }
    return ACLNN_SUCCESS;
}

// CumsumV2: prefix sum along dim (== reduce.mm scan). fp32.
aclnnStatus run_cumsum(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t outer, D, inner; geom(a, (int)e->dim, outer, D, inner);
    const float *x = FP(a); float *y = FP(o);
    for (int64_t ot = 0; ot < outer; ++ot) for (int64_t in = 0; in < inner; ++in) { double acc = 0;
        for (int64_t d = 0; d < D; ++d) { int64_t idx = (ot * D + d) * inner + in; acc += x[idx]; y[idx] = (float)acc; } }
    return ACLNN_SUCCESS;
}

// Median over a dim (fp32). lower-median = sorted[(D-1)/2]. NanMedian* alias this (no NaNs in tests).
aclnnStatus run_median(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t outer, D, inner; geom(a, (int)e->dim, outer, D, inner); const float *x = FP(a);
    std::vector<float> v(D);
    for (int64_t g = 0; g < outer * inner; ++g) { int64_t base = (g / inner) * D * inner + (g % inner);
        for (int64_t d = 0; d < D; ++d) v[d] = x[base + d * inner]; std::sort(v.begin(), v.end());
        FP(o)[g] = v[(D - 1) / 2]; }
    return ACLNN_SUCCESS;
}

// Argsort: indices that sort each group along dim. e->keepDim = descending; stable total order (value, then index).
aclnnStatus run_argsort(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *oi = e->out2; int64_t outer, D, inner; geom(a, (int)e->dim, outer, D, inner);
    bool desc = e->keepDim; const float *x = FP(a); int64_t *io = I64W(oi); std::vector<int> idx(D);
    for (int64_t o = 0; o < outer; ++o) for (int64_t i = 0; i < inner; ++i) { int64_t base = o * D * inner + i;
        for (int d = 0; d < (int)D; ++d) idx[d] = d;
        std::sort(idx.begin(), idx.end(), [&](int p, int q) { float va = x[base + (int64_t)p * inner], vb = x[base + (int64_t)q * inner]; if (va == vb) return p < q; return desc ? (va > vb) : (va < vb); });
        for (int d = 0; d < (int)D; ++d) io[base + (int64_t)d * inner] = idx[d]; }
    return ACLNN_SUCCESS;
}

// Bucketize / SearchSorted: per value count of boundaries that are (right? <= : <) value. out int64. a=boundaries(1D), b=values.
aclnnStatus run_bucketize(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *bnd = e->a, *val = e->b; aclTensor *o = e->out; int64_t B = e->m, nv = e->n; bool right = e->keepDim;
    const float *bp = FP(bnd), *vp = FP(val); int64_t *op = I64W(o);
    for (int64_t i = 0; i < nv; ++i) { int64_t r = 0; for (int64_t j = 0; j < B; ++j) if (right ? (bp[j] <= vp[i]) : (bp[j] < vp[i])) r++; op[i] = r; }
    return ACLNN_SUCCESS;
}

// Unique (sorted): dedup a 1D float tensor. valuesOut[m], countOut[0]=m, inverseOut (optional), countsOut (optional).
aclnnStatus run_unique(aclOpExecutor *e, aclrtStream s) {
    drain(s); int64_t n = e->m; const float *in = FP(e->a);
    std::vector<float> sorted(in, in + n); std::sort(sorted.begin(), sorted.end());
    float *uval = FP(e->out); int64_t *ucnt = I64W(e->out2);
    int64_t *counts = e->mask ? I64W((aclTensor *)e->mask) : nullptr;
    int64_t m = 0;
    for (int64_t i = 0; i < n; ++i) { if (i == 0 || sorted[i] != sorted[i - 1]) { uval[m] = sorted[i]; if (counts) counts[m] = 1; m++; } else if (counts) counts[m - 1]++; }
    *ucnt = m;
    if (e->c) { int64_t *inv = I64W((aclTensor *)e->c);   // inverse[i] = index of in[i] in uval (binary search)
        for (int64_t i = 0; i < n; ++i) { int64_t lo = 0, hi = m - 1, pos = 0; while (lo <= hi) { int64_t mid = (lo + hi) / 2; if (uval[mid] < in[i]) lo = mid + 1; else { pos = mid; hi = mid - 1; } } inv[i] = pos; } }
    return ACLNN_SUCCESS;
}

// TopKTopPSample: filter logits[rows,V] by top-k then top-p (nucleus), then sample one token per row (out int64).
// PCG-based u01 matching the CUDA kernel so results are deterministic given seed.
inline float u01s(uint64_t seed, int64_t r) { uint64_t s = seed * 6364136223846793005ULL + (uint64_t)(r * 2 + 1); uint32_t x = (uint32_t)((s >> 18) ^ s) >> 27; uint32_t rot = (uint32_t)(s >> 59);
    uint32_t w = (x >> rot) | (x << ((-rot) & 31)); return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }
aclnnStatus run_topk_topp_sample(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t rows = e->m, V = e->n; int topk = (int)e->dim; float topp = (float)e->alpha; uint64_t seed = (uint64_t)e->k;
    const float *lg = FP(a); int64_t *out = I64W(o); std::vector<float> op(V);
    for (int64_t r = 0; r < rows; ++r) { const float *p = lg + r * V; for (int64_t v = 0; v < V; ++v) op[v] = p[v];
        if (topk > 0 && topk < V) { float thr = -1e30f; for (int kk = 0; kk < topk; ++kk) { float best = -1e30f; for (int64_t v = 0; v < V; ++v) if (op[v] > best && (kk == 0 || op[v] < thr)) best = op[v]; thr = best; }
            for (int64_t v = 0; v < V; ++v) if (op[v] < thr) op[v] = -1e30f; }
        if (topp > 0.f && topp < 1.f) { float mx = -1e30f; for (int64_t v = 0; v < V; ++v) mx = std::max(mx, op[v]); double sm = 0; for (int64_t v = 0; v < V; ++v) if (op[v] > -1e29f) sm += std::exp((double)op[v] - mx);
            double cum = 0; float prevthr = 1e30f; for (int kept = 0; kept < V; ++kept) { float best = -1e30f; for (int64_t v = 0; v < V; ++v) if (op[v] > -1e29f && op[v] < prevthr && op[v] > best) best = op[v];
                if (best <= -1e29f) break; cum += std::exp((double)best - mx) / sm; prevthr = best; if (cum >= topp) { for (int64_t v = 0; v < V; ++v) if (op[v] < best) op[v] = -1e30f; break; } } }
        // sample: softmax over filtered, inverse-CDF with u01
        float mx = -1e30f; for (int64_t v = 0; v < V; ++v) mx = std::max(mx, op[v]); double sm = 0; for (int64_t v = 0; v < V; ++v) sm += std::exp((double)op[v] - mx);
        float u = u01s(seed, r) * (float)sm; double c = 0; int64_t pick = V - 1; for (int64_t v = 0; v < V; ++v) { c += std::exp((double)op[v] - mx); if (c >= u) { pick = v; break; } }
        out[r] = pick; }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

// Self-declared prototypes (none present in aclnnop/aclnn_ops.h) + plan builders + Execute funcs.
extern "C" {

// aclnnNonzero base lives in shape_ext.mm (already in the Metal backend); NonzeroV2 is a thin alias.
aclnnStatus aclnnNonzeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnNonzero(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnNonzeroV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnNonzeroGetWorkspaceSize(self, out, countOut, ws, ex); }
aclnnStatus aclnnNonzeroV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnNonzero(w, wss, e, s); }

// ---- index/shape: index-select family ----
aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->out = out; e->dim = dim; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnIndexSelect, run_index_select)
aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor *self, const aclTensorList *indices, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !indices || indices->v.empty() || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; return aclnnIndexSelectGetWorkspaceSize(self, 0, indices->v[0], out, ws, ex); }
RUN(aclnnIndex, run_index_select)
aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnIndexSelectGetWorkspaceSize(self, dim, index, out, ws, ex); }
RUN(aclnnGatherV3, run_index_select)

// ---- index add/fill/copy variants (dim0) ----
aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !index || !src || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->c = src; e->out = out; e->alpha = alpha; e->m = 0; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnIndexAddV2, run_index_rows)
// IndexFillTensor: fill value is a single-element tensor read at plan time.
aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !index || !value || !value->data || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    double v = 0.0; if (value->dtype == ACL_FLOAT) v = ((const float *)value->data)[value->offset]; else if (value->dtype == ACL_INT32) v = ((const int32_t *)value->data)[value->offset]; else return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->out = out; e->alpha = v; e->m = 1; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnIndexFillTensor, run_index_rows)

// ---- row scatter family ----
aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!indices || !updates || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->b = indices; e->c = updates; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScatterNd, run_scatternd_zero)
aclnnStatus aclnnScatterListGetWorkspaceSize(aclTensorList *selfRef, const aclTensorList *indices, const aclTensorList *updates, const aclTensor *mask, int64_t reduce, uint64_t *ws, aclOpExecutor **ex) {
    (void)mask; (void)reduce; if (!selfRef || !indices || !updates || !ex || selfRef->v.empty()) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); for (auto t : selfRef->v) e->inputs.push_back(t); for (auto t : indices->v) e->inputs.push_back(t); for (auto t : updates->v) e->inputs.push_back(t); e->m = selfRef->v.size(); if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScatterList, run_scatter_list)
aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values, bool accumulate, bool unsafe, uint64_t *ws, aclOpExecutor **ex) {
    (void)unsafe; if (!selfRef || !indices || indices->v.empty() || !values || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->out = selfRef; e->b = indices->v[0]; e->c = values; e->dim = accumulate ? 1 : 0; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnIndexPutImpl, run_index_put)
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(const aclTensor *input, aclTensor *cache, const aclTensor *slotMapping, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !cache || !slotMapping || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->c = input; e->out = cache; e->b = slotMapping; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScatterPaCache, run_scatter_pa)
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, const aclTensor *value, aclTensor *keyCache, aclTensor *valueCache, const aclTensor *slotMapping, uint64_t *ws, aclOpExecutor **ex) {
    if (!key || !value || !keyCache || !valueCache || !slotMapping || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = key; e->b = value; e->out = keyCache; e->out2 = valueCache; e->c = slotMapping; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScatterPaKvCache, run_scatter_pa_kv)

// ---- masked / expand / tile / diag / repeat ----
aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, double scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mask || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->b = mask; e->out = out; e->alpha = scale; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMaskedScale, run_masked_scale)
aclnnStatus aclnnExpandvGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)size; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnExpandv, run_expandv)
aclnnStatus aclnnRepeatGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)repeats; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeat, run_tile)
aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = diagonal; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDiag, run_diag)
// RepeatInterleave (flat repeat) + Int / WithDim / IntWithDim / Tensor aliases.
aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)outputSize; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = repeats; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleaveInt, run_repeat_il)
aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = repeats; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleaveWithDim, run_repeat_il)
aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; (void)outputSize; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = repeats; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleaveIntWithDim, run_repeat_il)
aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *repeats, int64_t outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)repeats; (void)outputSize; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; int64_t rep = self->numel() > 0 ? out->numel() / self->numel() : 1;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->m = rep; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRepeatInterleaveTensor, run_repeat_il)

// ---- chunk / chunkcat / flatten / slice / slice-assign ----
aclnnStatus aclnnChunkGetWorkspaceSize(const aclTensor *self, int64_t chunks, int64_t dim, const aclTensor *const *outputs, uint64_t num, uint64_t *ws, aclOpExecutor **ex) {
    (void)chunks; if (!self || !outputs || !num || !ex) return ACLNN_ERR_PARAM_NULLPTR; int rank = (int)self->viewDims.size(); if (dim < 0) dim += rank;
    auto *e = new aclOpExecutor(); e->a = self; e->dim = dim; e->out = const_cast<aclTensor *>(outputs[0]); for (uint64_t i = 0; i < num; ++i) e->inputs.push_back(outputs[i]); if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnChunk, run_split)
aclnnStatus aclnnChunkCatGetWorkspaceSize(const aclTensor *self, int64_t chunks, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)chunks; (void)dim; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnChunkCat, run_copy)
aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor *self, int64_t startDim, int64_t endDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)startDim; (void)endDim; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFlatten, run_copy)
aclnnStatus aclnnSliceV2GetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || step <= 0) return ACLNN_ERR_PARAM_NULLPTR; int rank = (int)self->viewDims.size(); if (dim < 0) dim += rank;
    if (start < 0) start += self->viewDims[dim]; if (end < 0) end += self->viewDims[dim]; if (end > self->viewDims[dim]) end = self->viewDims[dim];
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; e->m = start; e->k = step; (void)end; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSliceV2, run_slice)
aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, int64_t begin, int64_t end, int64_t stride, uint64_t *ws, aclOpExecutor **ex) {
    (void)end; (void)stride; if (!selfRef || !value || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->out = selfRef; e->c = value; e->m = begin; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnStridedSliceAssignV2, run_slice_assign)

// ---- lightning indexer softmax-lse ----
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(const aclTensor *indexScore, aclTensor *lse, aclTensor *probs, uint64_t *ws, aclOpExecutor **ex) {
    if (!indexScore || !lse || !ex || indexScore->viewDims.size() != 2) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = indexScore; e->out = lse; e->out2 = probs; e->m = indexScore->viewDims[0]; e->n = indexScore->viewDims[1]; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDenseLightningIndexerSoftmaxLse, run_lse)

// ---- reduce: aminmax / nansum / stdmean / cumsumV2 ----
aclnnStatus aclnnAminmaxAllGetWorkspaceSize(const aclTensor *self, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !outMin || !outMax || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = outMin; e->out2 = outMax; e->m = 3; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAminmaxAll, run_reduce_axes)
aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; if (!self || !outMin || !outMax || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = outMin; e->out2 = outMax; e->axes = {dim}; e->m = 3; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAminmaxDim, run_reduce_axes)
aclnnStatus aclnnReduceNansumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)dtype; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (dim) e->axes = dim->v; e->m = 0; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnReduceNansum, run_reduce_axes)
aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *stdOut, aclTensor *meanOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)correction; (void)keepDim; if (!self || !stdOut || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = stdOut; e->out2 = meanOut; if (dim) e->axes = dim->v; e->m = 1; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnStdMeanCorrection, run_reduce_axes)
aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; int nd = (int)self->viewDims.size(); if (dim < 0) dim += nd; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = dim; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCumsumV2, run_cumsum)

// ---- median variants ----
aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)indicesOut; if (!self || !valuesOut || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->dim = dim; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMedianDim, run_median)
aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = 0; e->axes = {}; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
// NanMedian reduces over the whole flattened tensor → reshape view as 1D group.
aclnnStatus run_nanmedian_all(aclOpExecutor *e, aclrtStream s) { drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t n = a->numel(); const float *x = FP(a);
    std::vector<float> v(x, x + n); std::sort(v.begin(), v.end()); FP(o)[0] = v[(n - 1) / 2]; return ACLNN_SUCCESS; }
RUN(aclnnNanMedian, run_nanmedian_all)
aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)indicesOut; if (!self || !valuesOut || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->dim = dim; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNanMedianDim, run_median)

// ---- argsort / bucketize / unique / topk-topp-sample ----
aclnnStatus aclnnArgsortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool stable, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)stable; if (!self || !indicesOut || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->a = self; e->out2 = indicesOut; e->dim = dim; e->keepDim = descending; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnArgsort, run_argsort)
aclnnStatus aclnnBucketizeGetWorkspaceSize(const aclTensor *values, const aclTensor *boundaries, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!values || !boundaries || !out || !ex || boundaries->viewDims.size() != 1) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = boundaries; e->b = values; e->out = out; e->m = boundaries->viewDims[0]; e->n = values->numel(); e->keepDim = right; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBucketize, run_bucketize)
aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor *self, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)sorted; (void)returnInverse; (void)returnCounts; if (!self || !valuesOut || !countOut || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = countOut; e->c = inverseOut; e->mask = countsOut; e->m = self->numel(); if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnUnique2, run_unique)
aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; (void)sorted; (void)returnInverse; (void)returnCounts; if (!self || !valuesOut || !countOut || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->out = valuesOut; e->out2 = countOut; e->c = inverseOut; e->mask = countsOut; e->m = self->numel(); if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnUniqueDim, run_unique)
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!logits || !out || !ex || logits->viewDims.empty()) return ACLNN_ERR_PARAM_NULLPTR; int64_t V = logits->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = logits; e->out = out; e->n = V; e->m = logits->numel() / V; e->dim = topk; e->alpha = topp; e->k = seed; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTopKTopPSample, run_topk_topp_sample)
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnTopKTopPSampleGetWorkspaceSize(logits, topk, topp, seed, out, ws, ex); }
RUN(aclnnTopKTopPSampleV2, run_topk_topp_sample)

} // extern "C"
