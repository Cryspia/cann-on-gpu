// Sort/scan op family (along single dim, contiguous assumed): Cumsum/Cumprod, Sort/ArgSort, Topk.
// Dims flattened into [outer, L, inner]; one thread per (outer,inner) segment processes sequentially (PoC, moderate L).
// Sort uses (value, original index) total-order comparison to break ties; CPU reference uses the same comparator.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math_constants.h>

namespace {

constexpr int TH = 128;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// Returns true if (va,ia) should come before (vb,ib). desc=true: larger value first; ties broken by original index ascending.
__device__ inline bool before(float va, int64_t ia, float vb, int64_t ib, bool desc) {
    if (va == vb) return ia < ib;
    return desc ? va > vb : va < vb;
}

template <typename T, typename A, bool PROD>
__global__ void k_cumscan(const T *x, T *o, int64_t outer, int64_t L, int64_t inner) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t base = (seg / inner) * L * inner + (seg % inner);
    A acc = PROD ? (A)1 : (A)0;
    for (int64_t l = 0; l < L; ++l) { A v = (A)x[base + l * inner]; acc = PROD ? acc * v : acc + v; o[base + l * inner] = (T)acc; }
}

// Cummax/Cummin: running max/min along dim, plus index where the running extremum was last achieved
template <typename T, bool MAX>
__global__ void k_cummaxmin(const T *x, T *o, int64_t *oi, int64_t outer, int64_t L, int64_t inner) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t base = (seg / inner) * L * inner + (seg % inner);
    float best = 0; int64_t bidx = 0;
    for (int64_t l = 0; l < L; ++l) { float v = (float)x[base + l * inner];
        if (l == 0 || (MAX ? v > best : v < best)) { best = v; bidx = l; }
        o[base + l * inner] = (T)best; if (oi) oi[base + l * inner] = bidx; }
}
// Logcumsumexp: running log(sum(exp)) along dim, numerically stable (tracks running max)
template <typename T>
__global__ void k_logcumsumexp(const T *x, T *o, int64_t outer, int64_t L, int64_t inner) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t base = (seg / inner) * L * inner + (seg % inner);
    double m = -1e300, se = 0;
    for (int64_t l = 0; l < L; ++l) { double v = (double)x[base + l * inner];
        if (l == 0) { m = v; se = 1.0; }
        else if (v > m) { se = se * exp(m - v) + 1.0; m = v; }
        else se += exp(v - m);
        o[base + l * inner] = (T)(m + log(se)); }
}

template <typename T>
__global__ void k_sort(const T *x, T *ov, int64_t *oi, int64_t outer, int64_t L, int64_t inner, bool desc) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t base = (seg / inner) * L * inner + (seg % inner);
    for (int64_t p = 0; p < L; ++p) oi[base + p * inner] = p;            // sorted position → original index
    for (int64_t p = 0; p < L; ++p) {
        int64_t best = p;
        for (int64_t q = p + 1; q < L; ++q) {
            int64_t lq = oi[base + q * inner], lb = oi[base + best * inner];
            if (before((float)x[base + lq * inner], lq, (float)x[base + lb * inner], lb, desc)) best = q;
        }
        if (best != p) { int64_t t = oi[base + p * inner]; oi[base + p * inner] = oi[base + best * inner]; oi[base + best * inner] = t; }
    }
    for (int64_t p = 0; p < L; ++p) { int64_t l = oi[base + p * inner]; ov[base + p * inner] = x[base + l * inner]; }
}

template <typename T>
__global__ void k_topk(const T *x, T *ov, int64_t *oi, int64_t outer, int64_t L, int64_t inner, int64_t k, bool largest) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t o = seg / inner, i = seg % inner;
    int64_t base = o * L * inner + i, obase = o * k * inner + i;
    for (int64_t l = 0; l < L; ++l) {
        float vl = (float)x[base + l * inner]; int64_t rank = 0;
        for (int64_t m = 0; m < L; ++m) { if (m == l) continue; if (before((float)x[base + m * inner], m, vl, l, largest)) ++rank; }
        if (rank < k) { ov[obase + rank * inner] = x[base + l * inner]; oi[obase + rank * inner] = l; }
    }
}

// ===== Large-L parallel sort: bitonic, inner==1 (along last dim; primary use: logits/vocabulary argsort) =====
// O(L·log²L); one kernel launch per (k,j) phase; replaces O(L²) selection sort. Value comparisons always in float (consistent with before).
inline int64_t np2(int64_t n){ int64_t p=1; while(p<n) p<<=1; return p; }
// Initialize padded buffer: real elements get (value, original index); padding gets sentinels (desc: -inf, asc: +inf; indices >= L always sort after real elements)
template <typename T>
__global__ void bit_init(const T *x, float *sv, int64_t *si, int64_t outer, int64_t L, int64_t P, bool desc) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= outer * P) return;
    int64_t seg = idx / P, p = idx % P;
    if (p < L) { sv[idx] = (float)x[seg * L + p]; si[idx] = p; }
    else { sv[idx] = desc ? -CUDART_INF_F : CUDART_INF_F; si[idx] = p; }
}
// Single bitonic compare-and-swap phase (k=subsequence length, j=step size)
__global__ void bit_step(float *sv, int64_t *si, int64_t outer, int64_t P, int64_t k, int64_t j, bool desc) {
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (t >= outer * (P / 2)) return;
    int64_t seg = t / (P / 2), w = t % (P / 2);
    int64_t i = (w / j) * (2 * j) + (w % j), ixj = i + j;
    int64_t base = seg * P;
    float va = sv[base + i], vb = sv[base + ixj]; int64_t ia = si[base + i], ib = si[base + ixj];
    bool up = ((i & k) == 0);                       // ascending sub-sequence
    bool aBefore = before(va, ia, vb, ib, desc);    // a should precede b in the final order
    bool needSwap = up ? !aBefore : aBefore;
    if (needSwap) { sv[base + i] = vb; sv[base + ixj] = va; si[base + i] = ib; si[base + ixj] = ia; }
}
// Retrieve the top cnt sorted elements: original values fetched from x by index (preserves original dtype precision; consistent semantics with the O(L²) path)
template <typename T>
__global__ void bit_gather(const T *x, const int64_t *si, T *ov, int64_t *oi, int64_t outer, int64_t L, int64_t P, int64_t cnt) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= outer * cnt) return;
    int64_t seg = idx / cnt, p = idx % cnt; int64_t l = si[seg * P + p];
    ov[seg * cnt + p] = x[seg * L + l]; oi[seg * cnt + p] = l;
}
template <typename T>
void bitonic_sort(const T *x, T *ov, int64_t *oi, int64_t outer, int64_t L, int64_t cnt, bool desc, void *ws, cudaStream_t s) {
    int64_t P = np2(L);
    float *sv = (float *)ws; int64_t *si = (int64_t *)(sv + outer * P);
    int64_t nInit = outer * P, gInit = (nInit + TH - 1) / TH;
    bit_init<T><<<gInit, TH, 0, s>>>(x, sv, si, outer, L, P, desc);
    int64_t nStep = outer * (P / 2), gStep = (nStep + TH - 1) / TH;
    for (int64_t k = 2; k <= P; k <<= 1)
        for (int64_t j = k >> 1; j > 0; j >>= 1)
            bit_step<<<gStep, TH, 0, s>>>(sv, si, outer, P, k, j, desc);
    int64_t nG = outer * cnt, gG = (nG + TH - 1) / TH;
    bit_gather<T><<<gG, TH, 0, s>>>(x, si, ov, oi, outer, L, P, cnt);
}
// bitonic workspace (bytes): padded values (float) + indices (int64)
inline uint64_t bitonic_ws(int64_t outer, int64_t L) { int64_t P = np2(L); return (uint64_t)outer * P * (4 + 8); }

void seg_layout(const aclTensor *t, int dim, int64_t &outer, int64_t &L, int64_t &inner) {
    int rank = (int)t->viewDims.size();
    outer = 1; for (int i = 0; i < dim; ++i) outer *= t->viewDims[i];
    L = t->viewDims[dim];
    inner = 1; for (int i = dim + 1; i < rank; ++i) inner *= t->viewDims[i];
}

aclnnStatus run_sort(aclOpExecutor *e, void *ws, cudaStream_t s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *a = e->a; aclTensor *o = e->out; int dim = (int)e->dim;
    int64_t outer, L, inner; seg_layout(a, dim, outer, L, inner);
    int64_t segs = outer * inner, g = nb(segs);
    aclnnStatus st = ACLNN_SUCCESS;
    // Large-L parallel bitonic path (inner==1, along last dim; ws pre-allocated by bitonic_ws)
    bool bit = ws && inner == 1 && L >= 2 && (e->op == OP_SORT || e->op == OP_TOPK);
    if (bit) {
        int64_t *oi = (int64_t *)e->out2->data;
        int64_t cnt = (e->op == OP_TOPK) ? e->m : L;
        bool desc = e->keepDim;
        switch (a->dtype) {
            case ACL_FLOAT:   bitonic_sort<float>((const float*)a->data,(float*)o->data,oi,outer,L,cnt,desc,ws,s); break;
            case ACL_FLOAT16: bitonic_sort<__half>((const __half*)a->data,(__half*)o->data,oi,outer,L,cnt,desc,ws,s); break;
            case ACL_INT32:   bitonic_sort<int32_t>((const int32_t*)a->data,(int32_t*)o->data,oi,outer,L,cnt,desc,ws,s); break;
            default: st = ACLNN_ERR_PARAM_INVALID;
        }
        if (st == ACLNN_SUCCESS && cudaGetLastError() != cudaSuccess) st = ACLNN_ERR_RUNTIME_ERROR;
        delete e; return st;
    }
    switch (e->op) {
        case OP_CUMSUM: case OP_CUMPROD: {
            bool prod = e->op == OP_CUMPROD;
            switch (a->dtype) {
                case ACL_FLOAT:   prod ? k_cumscan<float,double,true><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,outer,L,inner)
                                       : k_cumscan<float,double,false><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,outer,L,inner); break;
                case ACL_FLOAT16: prod ? k_cumscan<__half,float,true><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,outer,L,inner)
                                       : k_cumscan<__half,float,false><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,outer,L,inner); break;
                case ACL_INT32:   prod ? k_cumscan<int32_t,int64_t,true><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,outer,L,inner)
                                       : k_cumscan<int32_t,int64_t,false><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,outer,L,inner); break;
                default: st = ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case OP_CUMMAX: case OP_CUMMIN: {
            bool mx = e->op == OP_CUMMAX; int64_t *oi = e->out2 ? (int64_t *)e->out2->data : nullptr;
            switch (a->dtype) {
                case ACL_FLOAT:   mx ? k_cummaxmin<float,true><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,oi,outer,L,inner)
                                     : k_cummaxmin<float,false><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,oi,outer,L,inner); break;
                case ACL_FLOAT16: mx ? k_cummaxmin<__half,true><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,oi,outer,L,inner)
                                     : k_cummaxmin<__half,false><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,oi,outer,L,inner); break;
                case ACL_INT32:   mx ? k_cummaxmin<int32_t,true><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,oi,outer,L,inner)
                                     : k_cummaxmin<int32_t,false><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,oi,outer,L,inner); break;
                default: st = ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case OP_LCUMSUMEXP: {
            switch (a->dtype) {
                case ACL_FLOAT:   k_logcumsumexp<float><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,outer,L,inner); break;
                case ACL_FLOAT16: k_logcumsumexp<__half><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,outer,L,inner); break;
                default: st = ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case OP_SORT: {
            int64_t *oi = (int64_t *)e->out2->data; bool desc = e->keepDim;
            switch (a->dtype) {
                case ACL_FLOAT:   k_sort<float><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,oi,outer,L,inner,desc); break;
                case ACL_FLOAT16: k_sort<__half><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,oi,outer,L,inner,desc); break;
                case ACL_INT32:   k_sort<int32_t><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,oi,outer,L,inner,desc); break;
                default: st = ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case OP_TOPK: {
            int64_t *oi = (int64_t *)e->out2->data; bool largest = e->keepDim; int64_t k = e->m;
            switch (a->dtype) {
                case ACL_FLOAT:   k_topk<float><<<g,TH,0,s>>>((const float*)a->data,(float*)o->data,oi,outer,L,inner,k,largest); break;
                case ACL_FLOAT16: k_topk<__half><<<g,TH,0,s>>>((const __half*)a->data,(__half*)o->data,oi,outer,L,inner,k,largest); break;
                case ACL_INT32:   k_topk<int32_t><<<g,TH,0,s>>>((const int32_t*)a->data,(int32_t*)o->data,oi,outer,L,inner,k,largest); break;
                default: st = ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        default: st = ACLNN_ERR_PARAM_INVALID;
    }
    if (st == ACLNN_SUCCESS && cudaGetLastError() != cudaSuccess) st = ACLNN_ERR_RUNTIME_ERROR;
    delete e;
    return st;
}

aclnnStatus make_scan(int op, const aclTensor *self, int64_t dim, aclTensor *out, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->contiguous() || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = op; e->a = self; e->out = out; e->dim = dim;
    *ex = e; return ACLNN_SUCCESS;
}

} // namespace

extern "C" {

aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_scan(OP_CUMSUM, self, dim, out, ex);
}
aclnnStatus aclnnCumsum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, nullptr, (cudaStream_t)s); }

aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_scan(OP_CUMPROD, self, dim, out, ex);
}
aclnnStatus aclnnCumprod(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, nullptr, (cudaStream_t)s); }

// Cummax/Cummin: valuesOut (running extremum) + indicesOut (int64, where it was achieved); indicesOut may be null
aclnnStatus aclnnCummaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (indicesOut && indicesOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (ws) *ws = 0;
    aclnnStatus st = make_scan(OP_CUMMAX, self, dim, valuesOut, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = indicesOut;
    return st;
}
aclnnStatus aclnnCummax(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, nullptr, (cudaStream_t)s); }
aclnnStatus aclnnCumminGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (indicesOut && indicesOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (ws) *ws = 0;
    aclnnStatus st = make_scan(OP_CUMMIN, self, dim, valuesOut, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = indicesOut;
    return st;
}
aclnnStatus aclnnCummin(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, nullptr, (cudaStream_t)s); }
aclnnStatus aclnnLogcumsumexpGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_scan(OP_LCUMSUMEXP, self, dim, out, ex);
}
aclnnStatus aclnnLogcumsumexp(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, nullptr, (cudaStream_t)s); }

// Sort: valuesOut same shape as self; indicesOut (int64) same shape, holds original indices along dim
aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool /*stable*/,
                                      aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !valuesOut || !indicesOut || !ex || !self->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->contiguous() || self->dtype != valuesOut->dtype || indicesOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SORT; e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->keepDim = descending;
    int64_t outer, L, inner; seg_layout(self, dim, outer, L, inner);
    if (ws) *ws = (inner == 1 && L >= 2) ? bitonic_ws(outer, L) : 0;   // large-L parallel bitonic needs padded buffer
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSort(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, ws, (cudaStream_t)s); }

// Topk: valuesOut/indicesOut have length k along dim
aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool /*sorted*/,
                                      aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !valuesOut || !indicesOut || !ex || !self->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->contiguous() || self->dtype != valuesOut->dtype || indicesOut->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || k <= 0 || k > self->viewDims[dim]) return ACLNN_ERR_PARAM_INVALID;
    if (valuesOut->viewDims[dim] != k || indicesOut->viewDims[dim] != k) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_TOPK; e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; e->m = k; e->keepDim = largest;
    int64_t outer, L, inner; seg_layout(self, dim, outer, L, inner);
    if (ws) *ws = (inner == 1 && L >= 2) ? bitonic_ws(outer, L) : 0;   // large-L parallel bitonic needs padded buffer
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTopk(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sort(e, ws, (cudaStream_t)s); }

} // extern "C"
