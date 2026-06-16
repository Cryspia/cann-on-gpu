// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace _index_ext {
// Indexing extensions (P4): triangular (Tril/Triu), Trace, Diagonal/DiagFlat, Bincount, Searchsorted/Bucketize,
// IndexAdd/IndexFill/IndexCopy, ScatterMax/Min/Mul (along dim0), Take/TakeAlongDim, MaskedScatter, Narrow.
// Self-contained: each op builds a minimal executor (params stashed in generic fields) and runs a dedicated kernel.

namespace {

constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// ---- atomic float max/min/mul via CAS (CUDA has no native float atomicMax/Min/Mul) ----
__device__ inline float atomicMaxf(float *addr, float v) {
    int *ai = (int *)addr, old = *ai, assumed;
    do { assumed = old; float cur = __int_as_float(assumed); if (cur >= v) break; old = atomicCAS(ai, assumed, __float_as_int(v)); } while (assumed != old);
    return __int_as_float(old);
}
__device__ inline float atomicMinf(float *addr, float v) {
    int *ai = (int *)addr, old = *ai, assumed;
    do { assumed = old; float cur = __int_as_float(assumed); if (cur <= v) break; old = atomicCAS(ai, assumed, __float_as_int(v)); } while (assumed != old);
    return __int_as_float(old);
}
__device__ inline float atomicMulf(float *addr, float v) {
    int *ai = (int *)addr, old = *ai, assumed;
    do { assumed = old; old = atomicCAS(ai, assumed, __float_as_int(__int_as_float(assumed) * v)); } while (assumed != old);
    return __int_as_float(old);
}

// ---- triangular: keep elements on one side of the (offset) diagonal, zero the rest ----
template <typename T, bool LOWER>
__global__ void k_tri(const T *x, T *o, int64_t outer, int64_t M, int64_t N, int64_t k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= outer * M * N) return;
    int64_t col = i % N, row = (i / N) % M;
    bool keep = LOWER ? (col <= row + k) : (col >= row + k);
    o[i] = keep ? x[i] : (T)0;
}
// ---- trace: sum of the main diagonal of a 2D [M,N] ----
template <typename T>
__global__ void k_trace(const T *x, T *o, int64_t M, int64_t N) {
    double s = 0; int64_t d = M < N ? M : N; for (int64_t i = 0; i < d; ++i) s += (double)x[i * N + i]; *o = (T)s;
}
// ---- diagonal extract (2D [M,N], offset) -> 1D ----
template <typename T>
__global__ void k_diagonal(const T *x, T *o, int64_t N, int64_t len, int64_t roff, int64_t coff) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= len) return;
    o[i] = x[(roff + i) * N + (coff + i)];
}
// ---- diagflat (1D [L], offset) -> 2D [S,S] with diagonal set (out pre-zeroed) ----
template <typename T>
__global__ void k_diagflat_set(const T *x, T *o, int64_t L, int64_t S, int64_t roff, int64_t coff) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L) return;
    o[(roff + i) * S + (coff + i)] = x[i];
}
// ---- bincount: count occurrences of each integer value into out[numClasses] (int64) ----
template <typename T>
__global__ void k_bincount(const T *x, int64_t *o, int64_t n, int64_t C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t v = (int64_t)x[i]; if (v >= 0 && v < C) atomicAdd((unsigned long long *)&o[v], 1ULL);
}
// ---- searchsorted/bucketize: for each value, count boundaries strictly-less (right=false) or <= (right=true) ----
template <typename T>
__global__ void k_searchsorted(const T *bnd, const T *val, int64_t *o, int64_t B, int64_t n, bool right) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    double v = (double)val[i]; int64_t lo = 0, hi = B;
    while (lo < hi) { int64_t mid = (lo + hi) >> 1; double bm = (double)bnd[mid];
        if (right ? (bm <= v) : (bm < v)) lo = mid + 1; else hi = mid; }
    o[i] = lo;
}
// ---- index_* along dim0: out = self (pre-copied); update rows selected by index ----
template <typename T> __global__ void k_index_add0(const T *src, const int64_t *idx, T *o, int64_t L, int64_t row, float alpha) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; atomicAdd(&o[idx[l] * row + c], (T)(alpha * (float)src[i]));
}
template <typename T> __global__ void k_index_fill0(const int64_t *idx, T *o, int64_t L, int64_t row, float val) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; o[idx[l] * row + c] = (T)val;
}
template <typename T> __global__ void k_index_copy0(const T *src, const int64_t *idx, T *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; o[idx[l] * row + c] = src[i];
}
// ScatterMax/Min/Mul along dim0 (float only; uses CAS atomics)
__global__ void k_scatter_max0(const float *src, const int64_t *idx, float *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; atomicMaxf(&o[idx[l] * row + c], src[i]);
}
__global__ void k_scatter_min0(const float *src, const int64_t *idx, float *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; atomicMinf(&o[idx[l] * row + c], src[i]);
}
__global__ void k_scatter_mul0(const float *src, const int64_t *idx, float *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; atomicMulf(&o[idx[l] * row + c], src[i]);
}
// Take: flat gather out[i] = flat(self)[index[i]]
template <typename T> __global__ void k_take(const T *x, const int64_t *idx, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = x[idx[i]];
}
// TakeAlongDim: out shape == index shape; replace coord at `dim` by index value (self addressed by contiguous strides)
struct TAD { int rank, gd; int64_t od[8], istr[8]; };
template <typename T> __global__ void k_take_along(const T *x, const int64_t *idx, T *o, TAD d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t rem = i, off = 0;
    for (int kk = d.rank - 1; kk >= 0; --kk) { int64_t c = rem % d.od[kk]; rem /= d.od[kk];
        off += ((kk == d.gd) ? idx[i] : c) * d.istr[kk]; }
    o[i] = x[off];
}
// MaskedScatter: out = self (pre-copied); fill masked positions from src in order (single-thread O(n))
__global__ void k_masked_scatter(const uint8_t *mask, const float *src, float *o, int64_t n) {
    int64_t pos = 0; for (int64_t i = 0; i < n; ++i) if (mask[i]) o[i] = src[pos++];
}

#define DISP_SZ(esz, LAUNCH) do { switch (esz) { \
    case 1: { LAUNCH(uint8_t);  } break; case 2: { LAUNCH(uint16_t); } break; \
    case 4: { LAUNCH(uint32_t); } break; case 8: { LAUNCH(uint64_t); } break; \
    default: return ACLNN_ERR_PARAM_INVALID; } } while (0)

inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// ---- Tril / Triu (batched, last two dims are [M,N]; diagonal offset k) ----
static aclnnStatus tri_ws(int lower, const aclTensor *self, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size(); if (rank < 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = lower; e->a = self; e->out = out; e->k = k;
    e->m = self->viewDims[rank-2]; e->n = self->viewDims[rank-1];
    e->outerCount = self->numel() / (e->m * e->n);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTrilGetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return tri_ws(1, self, diagonal, out, ws, ex); }
aclnnStatus aclnnTriuGetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return tri_ws(0, self, diagonal, out, ws, ex); }
static aclnnStatus tri_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t outer = e->outerCount, M = e->m, N = e->n, k = e->k; size_t esz = dtype_size(e->a->dtype); int64_t g = nb(outer * M * N);
    bool lower = e->op == 1;
    #define L(T) (lower ? k_tri<T,true><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,M,N,k) \
                        : k_tri<T,false><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,M,N,k))
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
aclnnStatus aclnnTril(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return tri_run(e, (cudaStream_t)s); }
aclnnStatus aclnnTriu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return tri_run(e, (cudaStream_t)s); }
// in-place tril/triu (result written back into selfRef; element-wise so in-place safe)
aclnnStatus aclnnInplaceTrilGetWorkspaceSize(aclTensor *self, int64_t diagonal, uint64_t *ws, aclOpExecutor **ex) { return tri_ws(1, self, diagonal, self, ws, ex); }
aclnnStatus aclnnInplaceTril(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return tri_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceTriuGetWorkspaceSize(aclTensor *self, int64_t diagonal, uint64_t *ws, aclOpExecutor **ex) { return tri_ws(0, self, diagonal, self, ws, ex); }
aclnnStatus aclnnInplaceTriu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return tri_run(e, (cudaStream_t)s); }

// ---- Trace (2D) ----
aclnnStatus aclnnTraceGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->m = self->viewDims[0]; e->n = self->viewDims[1];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTrace(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_trace<float><<<1,1,0,st>>>((const float*)e->a->data,(float*)e->out->data,e->m,e->n); break;
        case ACL_FLOAT16: k_trace<__half><<<1,1,0,st>>>((const __half*)e->a->data,(__half*)e->out->data,e->m,e->n); break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    return done(e);
}

// ---- Diagonal (2D extract, offset) ----
aclnnStatus aclnnDiagonalGetWorkspaceSize(const aclTensor *self, int64_t offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 2 || out->viewDims.size() != 1) return ACLNN_ERR_PARAM_INVALID;
    int64_t M = self->viewDims[0], N = self->viewDims[1];
    int64_t roff = offset >= 0 ? 0 : -offset, coff = offset >= 0 ? offset : 0;
    int64_t len = std::min(M - roff, N - coff); if (len < 0) len = 0;
    if (out->viewDims[0] != len) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->n = N; e->m = len; e->k = roff; e->reduceCount = coff;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDiagonal(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N = e->n, len = e->m, roff = e->k, coff = e->reduceCount; size_t esz = dtype_size(e->a->dtype); int64_t g = nb(len);
    #define L(T) k_diagonal<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,N,len,roff,coff)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
// ---- DiagFlat (1D -> 2D, offset) ----
aclnnStatus aclnnDiagFlatGetWorkspaceSize(const aclTensor *self, int64_t offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 1 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t L = self->viewDims[0], S = out->viewDims[0];
    if (out->viewDims[1] != S || S != L + std::abs(offset)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->m = L; e->n = S;
    e->k = offset >= 0 ? 0 : -offset; e->reduceCount = offset >= 0 ? offset : 0;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDiagFlat(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s; int64_t L = e->m, S = e->n, roff = e->k, coff = e->reduceCount; size_t esz = dtype_size(e->a->dtype);
    cudaMemsetAsync(e->out->data, 0, (size_t)S * S * esz, st);
    #define L_(T) k_diagflat_set<T><<<nb(L),TH,0,st>>>((const T*)e->a->data,(T*)e->out->data,L,S,roff,coff)
    DISP_SZ(esz, L_);
    #undef L_
    return done(e);
}

// ---- Bincount (int input -> int64[C]) ----
aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor *self, int64_t numClasses, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_INT32 && self->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->m = self->numel(); e->n = numClasses;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBincount(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s; int64_t n = e->m, C = e->n;
    cudaMemsetAsync(e->out->data, 0, (size_t)C * sizeof(int64_t), st);
    if (e->a->dtype == ACL_INT32) k_bincount<int32_t><<<nb(n),TH,0,st>>>((const int32_t*)e->a->data,(int64_t*)e->out->data,n,C);
    else k_bincount<int64_t><<<nb(n),TH,0,st>>>((const int64_t*)e->a->data,(int64_t*)e->out->data,n,C);
    return done(e);
}

// ---- Searchsorted / Bucketize (boundaries 1D sorted, values any shape -> int64 same shape) ----
static aclnnStatus ss_ws(const aclTensor *bnd, const aclTensor *val, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!bnd || !val || !out || !ex || out->dtype != ACL_INT64 || bnd->dtype != val->dtype || bnd->viewDims.size() != 1) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = bnd; e->b = val; e->out = out; e->m = bnd->viewDims[0]; e->n = val->numel(); e->keepDim = right;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor *sorted, const aclTensor *values, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return ss_ws(sorted, values, right, out, ws, ex); }
aclnnStatus aclnnBucketizeGetWorkspaceSize(const aclTensor *values, const aclTensor *boundaries, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return ss_ws(boundaries, values, right, out, ws, ex); }
static aclnnStatus ss_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t B = e->m, n = e->n; bool right = e->keepDim; int64_t g = nb(n);
    switch (e->a->dtype) {
        case ACL_FLOAT: k_searchsorted<float><<<g,TH,0,s>>>((const float*)e->a->data,(const float*)e->b->data,(int64_t*)e->out->data,B,n,right); break;
        case ACL_INT32: k_searchsorted<int32_t><<<g,TH,0,s>>>((const int32_t*)e->a->data,(const int32_t*)e->b->data,(int64_t*)e->out->data,B,n,right); break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    return done(e);
}
aclnnStatus aclnnSearchSorted(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return ss_run(e, (cudaStream_t)s); }
aclnnStatus aclnnBucketize(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return ss_run(e, (cudaStream_t)s); }

// ---- IndexAdd / IndexFill / IndexCopy (along dim0) ----
static aclnnStatus idx0_ws(int kind, const aclTensor *self, const aclTensor *index, const aclTensor *src, double val, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ex || index->dtype != ACL_INT64 || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (kind != 1 && (!src || src->dtype != out->dtype)) return ACLNN_ERR_PARAM_INVALID;   // fill (kind=1) has no src
    auto *e = new aclOpExecutor(); e->op = kind; e->a = self; e->b = index; e->c = src; e->out = out;
    e->m = index->numel(); e->n = self->numel() / self->viewDims[0]; e->alpha = (kind == 1) ? val : alpha;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor *self, int64_t /*dim*/, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return idx0_ws(0, self, index, src, 0, alpha, out, ws, ex); }
aclnnStatus aclnnIndexFillGetWorkspaceSize(const aclTensor *self, int64_t /*dim*/, const aclTensor *index, double value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return idx0_ws(1, self, index, nullptr, value, 0, out, ws, ex); }
aclnnStatus aclnnIndexCopyGetWorkspaceSize(const aclTensor *self, int64_t /*dim*/, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return idx0_ws(2, self, index, src, 0, 0, out, ws, ex); }
static aclnnStatus idx0_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t L = e->m, row = e->n; size_t esz = dtype_size(e->a->dtype);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel() * esz, cudaMemcpyDeviceToDevice, s);
    int64_t g = nb(L * row); const int64_t *idx = (const int64_t *)e->b->data;
    if (e->op == 0) {   // index_add
        switch (e->a->dtype) {
            case ACL_FLOAT:   k_index_add0<float><<<g,TH,0,s>>>((const float*)e->c->data,idx,(float*)e->out->data,L,row,(float)e->alpha); break;
            case ACL_FLOAT16: k_index_add0<__half><<<g,TH,0,s>>>((const __half*)e->c->data,idx,(__half*)e->out->data,L,row,(float)e->alpha); break;
            default: delete e; return ACLNN_ERR_PARAM_INVALID;
        }
    } else if (e->op == 1) {   // index_fill
        #define LF(T) k_index_fill0<T><<<g,TH,0,s>>>(idx,(T*)e->out->data,L,row,(float)e->alpha)
        switch (e->a->dtype) { case ACL_FLOAT: LF(float); break; case ACL_FLOAT16: LF(__half); break; case ACL_INT32: LF(int32_t); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
        #undef LF
    } else {   // index_copy
        #define LC(T) k_index_copy0<T><<<g,TH,0,s>>>((const T*)e->c->data,idx,(T*)e->out->data,L,row)
        switch (esz) { case 1: LC(uint8_t); break; case 2: LC(uint16_t); break; case 4: LC(uint32_t); break; case 8: LC(uint64_t); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
        #undef LC
    }
    return done(e);
}
aclnnStatus aclnnIndexAdd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
aclnnStatus aclnnIndexFill(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
aclnnStatus aclnnIndexCopy(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
// IndexAddV2: alias of IndexAdd (same dim0 semantics)
aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return idx0_ws(0, self, index, src, 0, alpha, out, ws, ex); }
aclnnStatus aclnnIndexAddV2(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
// IndexFillTensor: fill value comes from a single-element tensor (read at plan time)
static aclnnStatus idxfilltensor_ws(const aclTensor *self, const aclTensor *index, const aclTensor *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!value || !value->data) return ACLNN_ERR_PARAM_NULLPTR;
    double v = 0.0;
    if (value->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, value->data, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return ACLNN_ERR_RUNTIME_ERROR; v = h; }
    else if (value->dtype == ACL_INT32) { int32_t h; if (cudaMemcpy(&h, value->data, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return ACLNN_ERR_RUNTIME_ERROR; v = h; }
    else return ACLNN_ERR_PARAM_INVALID;
    return idx0_ws(1, self, index, nullptr, v, 0, out, ws, ex);
}
aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return idxfilltensor_ws(self, index, value, out, ws, ex); }
aclnnStatus aclnnIndexFillTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
// in-place index variants (result written back into selfRef; ops pre-copy self→out so out=self is safe)
aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return idx0_ws(2, self, index, src, 0, 0, self, ws, ex); }
aclnnStatus aclnnInplaceIndexCopy(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(aclTensor *self, int64_t dim, const aclTensor *index, double value, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return idx0_ws(1, self, index, nullptr, value, 0, self, ws, ex); }
aclnnStatus aclnnInplaceIndexFill(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceIndexFillTensorGetWorkspaceSize(aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return idxfilltensor_ws(self, index, value, self, ws, ex); }
aclnnStatus aclnnInplaceIndexFillTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return idx0_run(e, (cudaStream_t)s); }

// ---- ScatterMax / ScatterMin / ScatterMul (along dim0, float) ----
static aclnnStatus scat_ws(int kind, const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ex || index->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT || src->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = kind; e->a = self; e->b = index; e->c = src; e->out = out;
    e->m = index->numel(); e->n = self->numel() / self->viewDims[0];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterMaxGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return scat_ws(0, self, index, src, out, ws, ex); }
aclnnStatus aclnnScatterMinGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return scat_ws(1, self, index, src, out, ws, ex); }
aclnnStatus aclnnScatterMulGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return scat_ws(2, self, index, src, out, ws, ex); }
static aclnnStatus scat_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t L = e->m, row = e->n;
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel() * sizeof(float), cudaMemcpyDeviceToDevice, s);
    int64_t g = nb(L * row); const int64_t *idx = (const int64_t *)e->b->data;
    const float *src = (const float *)e->c->data; float *o = (float *)e->out->data;
    if (e->op == 0) k_scatter_max0<<<g,TH,0,s>>>(src,idx,o,L,row);
    else if (e->op == 1) k_scatter_min0<<<g,TH,0,s>>>(src,idx,o,L,row);
    else k_scatter_mul0<<<g,TH,0,s>>>(src,idx,o,L,row);
    return done(e);
}
aclnnStatus aclnnScatterMax(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scat_run(e, (cudaStream_t)s); }
aclnnStatus aclnnScatterMin(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scat_run(e, (cudaStream_t)s); }
aclnnStatus aclnnScatterMul(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scat_run(e, (cudaStream_t)s); }

// ---- Take (flat gather) ----
aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor *self, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ex || index->dtype != ACL_INT64 || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (out->numel() != index->numel()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->b = index; e->out = out; e->m = index->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTake(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->m; size_t esz = dtype_size(e->a->dtype); int64_t g = nb(n); const int64_t *idx = (const int64_t *)e->b->data;
    #define L(T) k_take<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,idx,(T*)e->out->data,n)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
// ---- TakeAlongDim (gather along dim with full-shape index, self contiguous) ----
aclnnStatus aclnnTakeAlongDimGetWorkspaceSize(const aclTensor *self, const aclTensor *index, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ex || index->dtype != ACL_INT64 || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size(); if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || (int)index->viewDims.size() != rank || index->viewDims != out->viewDims || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->b = index; e->out = out; e->dim = dim;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTakeAlongDim(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int rank = (int)e->a->viewDims.size(); TAD d{}; d.rank = rank; d.gd = (int)e->dim;
    int64_t acc = 1; int64_t istr[8]; for (int i = rank-1; i >= 0; --i) { istr[i] = acc; acc *= e->a->viewDims[i]; }
    for (int i = 0; i < rank; ++i) { d.od[i] = e->out->viewDims[i]; d.istr[i] = istr[i]; }
    int64_t n = e->out->numel(); size_t esz = dtype_size(e->a->dtype); int64_t g = nb(n); const int64_t *idx = (const int64_t *)e->b->data;
    #define L(T) k_take_along<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,idx,(T*)e->out->data,d,n)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
// ---- MaskedScatter (float) ----
aclnnStatus aclnnMaskedScatterGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mask || !src || !out || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT || src->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->b = mask; e->c = src; e->out = out; e->m = self->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaskedScatter(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s;
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->m * sizeof(float), cudaMemcpyDeviceToDevice, st);
    k_masked_scatter<<<1,1,0,st>>>((const uint8_t*)e->b->data,(const float*)e->c->data,(float*)e->out->data,e->m);
    return done(e);
}
// ---- Narrow = Slice(dim, start, start+length, step=1) ----
aclnnStatus aclnnNarrowGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t length, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnSliceGetWorkspaceSize(self, dim, start, start + length, 1, out, ws, ex);
}
aclnnStatus aclnnNarrow(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnSlice(ws, wsz, e, s); }

} // extern "C"
} // namespace _index_ext
#undef DISP_SZ

namespace _index2_ext {
// Indexing remainder (R4): Unique (sorted unique values + count + optional counts/inverse) and UniqueConsecutive.
// 1D fp32 input. Dynamic output size reported via countOut[0]; valuesOut must be allocated to >= n.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__global__ void k_sort1(float*a,int64_t n){ for(int64_t i=0;i<n;i++){ int64_t b=i; for(int64_t j=i+1;j<n;j++) if(a[j]<a[b])b=j; if(b!=i){float t=a[i];a[i]=a[b];a[b]=t;} } }
// dedup (single thread): if consecutive, dedup adjacent on src as-is; else src is pre-sorted
__global__ void k_dedup(const float*src,float*uval,int64_t*ucnt,int64_t*counts,int64_t n){
    int64_t m=0; for(int64_t i=0;i<n;i++){ if(i==0||src[i]!=src[i-1]){ uval[m]=src[i]; if(counts)counts[m]=1; m++; } else if(counts) counts[m-1]++; } *ucnt=m;
}
__global__ void k_inverse(const float*in,const float*uval,const int64_t*ucnt,int64_t*inv,int64_t n){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; int64_t m=*ucnt; float v=in[i];
    for(int64_t j=0;j<m;j++) if(uval[j]==v){ inv[i]=j; return; } inv[i]=0;
}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// op: 0 = Unique (sorts first), 1 = UniqueConsecutive (no sort). countsOut / inverseOut may be null.
static aclnnStatus uniq_ws(int op,const aclTensor*self,aclTensor*valuesOut,aclTensor*countOut,aclTensor*inverseOut,aclTensor*countsOut,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!valuesOut||!countOut||!ex||self->dtype!=ACL_FLOAT||valuesOut->dtype!=ACL_FLOAT||countOut->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->op=op; e->a=self; e->out=valuesOut; e->out2=countOut; e->c=inverseOut; e->mask=countsOut; e->m=self->numel();
    if(ws)*ws=(uint64_t)self->numel()*sizeof(float); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUniqueGetWorkspaceSize(const aclTensor*self,aclTensor*valuesOut,aclTensor*countOut,aclTensor*inverseOut,aclTensor*countsOut,uint64_t*ws,aclOpExecutor**ex){ return uniq_ws(0,self,valuesOut,countOut,inverseOut,countsOut,ws,ex); }
aclnnStatus aclnnUniqueConsecutiveGetWorkspaceSize(const aclTensor*self,aclTensor*valuesOut,aclTensor*countOut,aclTensor*inverseOut,aclTensor*countsOut,uint64_t*ws,aclOpExecutor**ex){ return uniq_ws(1,self,valuesOut,countOut,inverseOut,countsOut,ws,ex); }
static aclnnStatus uniq_run(void*ws,aclOpExecutor*e,cudaStream_t s){
    int64_t n=e->m; const float*in=(const float*)e->a->data; const float*src=in;
    if(e->op==0){ float*sorted=(float*)ws; cudaMemcpyAsync(sorted,in,(size_t)n*4,cudaMemcpyDeviceToDevice,s); k_sort1<<<1,1,0,s>>>(sorted,n); src=sorted; }
    int64_t*counts=e->mask?(int64_t*)e->mask->data:nullptr;
    k_dedup<<<1,1,0,s>>>(src,(float*)e->out->data,(int64_t*)e->out2->data,counts,n);
    if(e->c) k_inverse<<<nb(n),TH,0,s>>>(in,(const float*)e->out->data,(const int64_t*)e->out2->data,(int64_t*)e->c->data,n);
    return done(e);
}
aclnnStatus aclnnUnique(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ return uniq_run(ws,e,(cudaStream_t)s); }
aclnnStatus aclnnUniqueConsecutive(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ return uniq_run(ws,e,(cudaStream_t)s); }

} // extern "C"
} // namespace _index2_ext

namespace _index3_ext {
// Index / scatter / gather / sort / sampling remainder (fp32 + int64 indices). Advanced indexing,
// paged-KV-cache scatter/gather, TF scatter-add, top-k/top-p logit filtering + sampling, embedding renorm,
// search-sorted scalar, unique/argsort naming forwards, and the DeepSeek-style lightning indexer cluster.

// aclnnUnique is defined later in this file (extern "C", not in the public header).
extern "C" {
aclnnStatus aclnnUniqueGetWorkspaceSize(const aclTensor *self, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnUnique(void *, uint64_t, aclOpExecutor *, aclrtStream);
}

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
__device__ inline uint32_t hash32(uint32_t x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x; }
__device__ inline float u01(uint64_t seed,int64_t i){ uint32_t h=hash32((uint32_t)seed ^ hash32((uint32_t)i ^ (uint32_t)(i>>32))); return (h>>8)*(1.f/16777216.f); }

// row gather/scatter (dim0): out[i,:] = self[idx[i],:] / dst[idx[i],:] (+=) src[i,:]
__global__ void k_row_gather(const float*src,const int64_t*idx,float*o,int64_t K,int64_t D,int64_t N){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=K*D) return; int64_t k=p/D,d=p%D; int64_t r=idx[k]; o[p]=(r>=0&&r<N)?src[r*D+d]:0.f;
}
__global__ void k_row_scatter(float*dst,const int64_t*idx,const float*src,int64_t K,int64_t D,int64_t N,int add){
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=K*D) return; int64_t k=p/D,d=p%D; int64_t r=idx[k]; if(r<0||r>=N)return;
    if(add) atomicAdd(&dst[r*D+d], src[p]); else dst[r*D+d]=src[p];
}
// embedding renorm: for unique rows in idx, scale row if ||row||_p > maxnorm
__global__ void k_emb_renorm(float*w,const int64_t*idx,int64_t K,int64_t D,float maxnorm,float ntype){
    int64_t k=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(k>=K) return; int64_t r=idx[k]; float*row=w+r*D;
    double nrm=0; for(int64_t d=0;d<D;d++){ float v=row[d]; nrm += (ntype==1.f)? fabsf(v) : (double)v*v; }
    nrm = (ntype==1.f)? nrm : sqrt(nrm);
    if(nrm>maxnorm){ float s=(float)(maxnorm/(nrm+1e-7)); for(int64_t d=0;d<D;d++) row[d]*=s; }
}
// apply top-k/top-p filter to logits per row → set filtered to -inf (single-thread per row)
__global__ void k_topk_topp(const float*logits,float*o,int64_t rows,int64_t V,int topk,float topp){
    int64_t r=blockIdx.x; if(r>=rows||threadIdx.x!=0) return; const float*p=logits+r*V; float*op=o+r*V;
    // copy
    for(int64_t v=0;v<V;v++) op[v]=p[v];
    // top-k: find kth largest threshold via partial selection (V small in tests)
    if(topk>0 && topk<V){ float thr=-1e30f; for(int kk=0;kk<topk;kk++){ float best=-1e30f; for(int64_t v=0;v<V;v++){ if(op[v]>best && (kk==0||op[v]<thr)) best=op[v]; } thr=best; } for(int64_t v=0;v<V;v++) if(op[v]<thr) op[v]=-1e30f; }
    // top-p (nucleus): softmax over remaining, cumulative from largest until >= topp
    if(topp>0.f && topp<1.f){ float mx=-1e30f; for(int64_t v=0;v<V;v++) mx=fmaxf(mx,op[v]); double sm=0; for(int64_t v=0;v<V;v++) if(op[v]>-1e29f) sm+=exp(op[v]-mx);
        // iteratively keep largest until cumulative prob >= topp
        double cum=0; float prevthr=1e30f; for(int kept=0;kept<V;kept++){ float best=-1e30f; for(int64_t v=0;v<V;v++){ if(op[v]>-1e29f && op[v]<prevthr && op[v]>best) best=op[v]; }
            if(best<=-1e29f) break; cum += exp(best-mx)/sm; prevthr=best; if(cum>=topp) { // drop everything strictly below best
                for(int64_t v=0;v<V;v++) if(op[v]<best) op[v]=-1e30f; break; } }
    }
}
// sample one token per row from filtered logits (softmax → inverse-CDF with u01)
__global__ void k_sample(const float*logits,int64_t*out,int64_t rows,int64_t V,uint64_t seed){
    int64_t r=blockIdx.x; if(r>=rows||threadIdx.x!=0) return; const float*p=logits+r*V; float mx=-1e30f; for(int64_t v=0;v<V;v++)mx=fmaxf(mx,p[v]);
    double sm=0; for(int64_t v=0;v<V;v++) sm+=exp(p[v]-mx); float u=u01(seed,r)*(float)sm; double c=0; int64_t pick=V-1;
    for(int64_t v=0;v<V;v++){ c+=exp(p[v]-mx); if(c>=u){ pick=v; break; } } out[r]=pick;
}
// TF scatter-add: ref[idx[k]] += updates[k] (dim0 rows)
// lightning indexer scoring: score[q,k] = Σ_h wq[q,h]*relu(Σ_d q[q,h,d]*key[k,h,d])
__global__ void k_lightning(const float*q,const float*key,const float*wq,float*score,int Q,int K,int Hh,int Dd){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)Q*K) return; int qi=i/K,ki=i%K; float acc=0;
    for(int h=0;h<Hh;h++){ float dot=0; const float*qp=q+((int64_t)qi*Hh+h)*Dd; const float*kp=key+((int64_t)ki*Hh+h)*Dd; for(int d=0;d<Dd;d++)dot+=qp[d]*kp[d];
        acc += wq[(int64_t)qi*Hh+h]*fmaxf(dot,0.f); }
    score[i]=acc;
}
} // namespace

extern "C" {

// ---- Index (advanced, single index tensor on dim 0) → IndexSelect(dim=0) ----
aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor *self, const aclTensorList *indices, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!indices||indices->v.empty()||!out||!ex) return ACLNN_ERR_PARAM_INVALID;
    return aclnnIndexSelectGetWorkspaceSize(self, 0, indices->v[0], out, ws, ex);
}
aclnnStatus aclnnIndex(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnIndexSelect(w,wz,e,s); }

// ---- IndexPutImpl: selfRef[idx0[k], :] = / += values[k, :] (single index tensor, dim0) ----
aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values, bool accumulate, bool unsafe, uint64_t *ws, aclOpExecutor **ex){
    (void)unsafe; if(!selfRef||!indices||indices->v.empty()||!values||!ex||selfRef->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->out=selfRef; e->b=indices->v[0]; e->c=values; e->dim=accumulate?1:0; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnIndexPutImpl(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->b->numel(), N=e->out->viewDims[0], D=e->out->numel()/N; auto st=(cudaStream_t)s;
    k_row_scatter<<<nb(K*D),TH,0,st>>>((float*)e->out->data,(const int64_t*)e->b->data,(const float*)e->c->data,K,D,N,(int)e->dim);
    return done(e);
}
// ---- TfScatterAdd: ref[idx[k]] += updates[k] (rows) → out is the updated ref ----
aclnnStatus aclnnTfScatterAddGetWorkspaceSize(const aclTensor *ref, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!ref||!indices||!updates||!out||!ex||out->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=ref; e->b=indices; e->c=updates; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTfScatterAdd(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->out->viewDims[0], D=e->out->numel()/N, K=e->b->numel(); auto st=(cudaStream_t)s;
    if(e->a->data!=e->out->data) cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->out->numel()*4,cudaMemcpyDeviceToDevice,st);
    k_row_scatter<<<nb(K*D),TH,0,st>>>((float*)e->out->data,(const int64_t*)e->b->data,(const float*)e->c->data,K,D,N,1);
    return done(e);
}
// ---- ScatterList: list of (self, index, update) → loop row-scatter (update) ----
aclnnStatus aclnnScatterListGetWorkspaceSize(aclTensorList *selfRef, const aclTensorList *indices, const aclTensorList *updates, const aclTensor *mask, int64_t reduce, uint64_t *ws, aclOpExecutor **ex){
    (void)mask;(void)reduce; if(!selfRef||!indices||!updates||!ex||selfRef->v.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); for(auto t:selfRef->v) e->inputs.push_back(t); for(auto t:indices->v) e->inputs.push_back(t); for(auto t:updates->v) e->inputs.push_back(t);
    e->m=selfRef->v.size(); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterList(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t L=e->m; auto st=(cudaStream_t)s;
    for(int64_t l=0;l<L;l++){ const aclTensor*self=e->inputs[l],*idx=e->inputs[L+l],*upd=e->inputs[2*L+l];
        int64_t N=self->viewDims[0],D=self->numel()/N,K=idx->numel();
        k_row_scatter<<<nb(K*D),TH,0,st>>>((float*)const_cast<aclTensor*>(self)->data,(const int64_t*)idx->data,(const float*)upd->data,K,D,N,0); }
    return done(e);
}
// ---- Paged-KV-cache scatter/gather ----
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(const aclTensor *input, aclTensor *cache, const aclTensor *slotMapping, uint64_t *ws, aclOpExecutor **ex){
    if(!input||!cache||!slotMapping||!ex||cache->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->c=input; e->out=cache; e->b=slotMapping; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterPaCache(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->b->numel(), N=e->out->viewDims[0], D=e->out->numel()/N; auto st=(cudaStream_t)s;
    k_row_scatter<<<nb(K*D),TH,0,st>>>((float*)e->out->data,(const int64_t*)e->b->data,(const float*)e->c->data,K,D,N,0); return done(e);
}
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, const aclTensor *value, aclTensor *keyCache, aclTensor *valueCache, const aclTensor *slotMapping, uint64_t *ws, aclOpExecutor **ex){
    if(!key||!value||!keyCache||!valueCache||!slotMapping||!ex) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=key; e->b=value; e->out=keyCache; e->out2=valueCache; e->c=slotMapping; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterPaKvCache(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->c->numel(), Nk=e->out->viewDims[0], Dk=e->out->numel()/Nk, Nv=e->out2->viewDims[0], Dv=e->out2->numel()/Nv; auto st=(cudaStream_t)s;
    k_row_scatter<<<nb(K*Dk),TH,0,st>>>((float*)e->out->data,(const int64_t*)e->c->data,(const float*)e->a->data,K,Dk,Nk,0);
    k_row_scatter<<<nb(K*Dv),TH,0,st>>>((float*)e->out2->data,(const int64_t*)e->c->data,(const float*)e->b->data,K,Dv,Nv,0);
    return done(e);
}
aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache, const aclTensor *slotMapping, aclTensor *keyOut, aclTensor *valueOut, uint64_t *ws, aclOpExecutor **ex){
    if(!keyCache||!valueCache||!slotMapping||!keyOut||!valueOut||!ex) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=keyCache; e->b=valueCache; e->c=slotMapping; e->out=keyOut; e->out2=valueOut; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGatherPaKvCache(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->c->numel(), Nk=e->a->viewDims[0], Dk=e->a->numel()/Nk, Nv=e->b->viewDims[0], Dv=e->b->numel()/Nv; auto st=(cudaStream_t)s;
    k_row_gather<<<nb(K*Dk),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->c->data,(float*)e->out->data,K,Dk,Nk);
    k_row_gather<<<nb(K*Dv),TH,0,st>>>((const float*)e->b->data,(const int64_t*)e->c->data,(float*)e->out2->data,K,Dv,Nv);
    return done(e);
}
// ---- SearchSorteds (scalar values): forward to SearchSorted with a 1-element values tensor is not possible here;
//       implement directly: out = count of sorted <= value (or < for right) ----
aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor *sortedSequence, const aclScalar *value, bool right, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!sortedSequence||!value||!out||!ex||out->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=sortedSequence; e->out=out; e->alpha=value->v; e->dim=right?1:0; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
} // extern "C"
namespace { __global__ void k_searchsorteds(const float*seq,int64_t n,float v,int right,int64_t*o){
    if(threadIdx.x!=0||blockIdx.x!=0) return; int64_t c=0; for(int64_t i=0;i<n;i++){ if(right? (seq[i]<=v):(seq[i]<v)) c++; } o[0]=c; } }
extern "C" {
aclnnStatus aclnnSearchSorteds(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t n=e->a->numel(); k_searchsorteds<<<1,1,0,(cudaStream_t)s>>>((const float*)e->a->data,n,(float)e->alpha,(int)e->dim,(int64_t*)e->out->data); return done(e); }

// ---- Unique2 / UniqueDim → forward to Unique ----
aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor *self, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex){
    (void)sorted;(void)returnInverse;(void)returnCounts; return aclnnUniqueGetWorkspaceSize(self, valuesOut, countOut, inverseOut, countsOut, ws, ex);
}
aclnnStatus aclnnUnique2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnUnique(w,wz,e,s); }
aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *ws, aclOpExecutor **ex){
    (void)dim;(void)sorted;(void)returnInverse;(void)returnCounts; return aclnnUniqueGetWorkspaceSize(self, valuesOut, countOut, inverseOut, countsOut, ws, ex);
}
aclnnStatus aclnnUniqueDim(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnUnique(w,wz,e,s); }

// ---- Argsort → Sort (indices only; values to a scratch buffer) ----
aclnnStatus aclnnArgsortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool stable, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!indicesOut||!ex) return ACLNN_ERR_PARAM_INVALID;
    aclTensor scratch = *self;  // value scratch shares shape/dtype; data set in run
    (void)scratch;
    auto*e=new aclOpExecutor(); e->a=self; e->out2=indicesOut; e->dim=dim; e->keepDim=descending; e->m=stable?1:0; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnArgsort(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    void*vbuf=nullptr; size_t bytes=(size_t)e->a->numel()*dtype_size(e->a->dtype); cudaMallocAsync(&vbuf,bytes,(cudaStream_t)s);
    aclTensor vals=*e->a; vals.data=vbuf;
    uint64_t w2=0; aclOpExecutor*e2=nullptr;
    aclnnStatus st=aclnnSortGetWorkspaceSize(e->a,e->dim,e->keepDim,e->m!=0,&vals,e->out2,&w2,&e2);
    if(st==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); st=aclnnSort(wb,w2,e2,s); if(wb)cudaFree(wb); }
    cudaFreeAsync(vbuf,(cudaStream_t)s); delete e; return st;
}

// ---- EmbeddingRenorm (in place) ----
aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, double maxNorm, double normType, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!indices||!ex||selfRef->dtype!=ACL_FLOAT||selfRef->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->out=selfRef; e->b=indices; e->alpha=maxNorm; e->eps=normType; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEmbeddingRenorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->b->numel(), D=e->out->viewDims[1]; k_emb_renorm<<<nb(K),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(const int64_t*)e->b->data,K,D,(float)e->alpha,(float)e->eps); return done(e); }

// ---- ApplyTopKTopP: filter logits[rows,V] → out ----
aclnnStatus aclnnApplyTopKTopPGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!logits||!out||!ex||logits->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t V=logits->viewDims.back(); auto*e=new aclOpExecutor(); e->a=logits; e->out=out; e->n=V; e->m=logits->numel()/V; e->dim=topk; e->alpha=topp; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyTopKTopP(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->m,V=e->n; k_topk_topp<<<(unsigned)rows,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,rows,V,(int)e->dim,(float)e->alpha); return done(e); }

// ---- TopKTopPSample (+V2): filter then sample one token per row ----
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!logits||!out||!ex||out->dtype!=ACL_INT64||logits->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t V=logits->viewDims.back(); auto*e=new aclOpExecutor(); e->a=logits; e->out=out; e->n=V; e->m=logits->numel()/V; e->dim=topk; e->alpha=topp; e->k=seed; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTopKTopPSample(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->m,V=e->n; auto st=(cudaStream_t)s; void*fbuf=nullptr; cudaMallocAsync(&fbuf,(size_t)rows*V*4,st);
    k_topk_topp<<<(unsigned)rows,1,0,st>>>((const float*)e->a->data,(float*)fbuf,rows,V,(int)e->dim,(float)e->alpha);
    k_sample<<<(unsigned)rows,1,0,st>>>((const float*)fbuf,(int64_t*)e->out->data,rows,V,(uint64_t)e->k);
    cudaFreeAsync(fbuf,st); return done(e); }
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnTopKTopPSampleGetWorkspaceSize(logits,topk,topp,seed,out,ws,ex);
}
aclnnStatus aclnnTopKTopPSampleV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnTopKTopPSample(w,wz,e,s); }

// ---- AllGatherAdd: single-rank degenerate = elementwise add of x + bias-broadcast ----
} // extern "C"
namespace { __global__ void k_bcast_add(const float*x,const float*b,float*o,int64_t n,int64_t bn){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=x[i]+(b?b[i%bn]:0.f); } }
extern "C" {
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *x, const aclTensor *bias, const char *group, int64_t rankSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)group;(void)rankSize; if(!x||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto*e=new aclOpExecutor(); e->a=x; e->b=bias; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAllGatherAdd(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(),bn=e->b?e->b->numel():1;
    k_bcast_add<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,e->b?(const float*)e->b->data:nullptr,(float*)e->out->data,n,bn); return done(e); }

// ---- LightningIndexer cluster (DeepSeek-style sparse attention indexer) ----
aclnnStatus aclnnLightningIndexerGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *weights, aclTensor *indexScore, uint64_t *ws, aclOpExecutor **ex){
    if(!query||!key||!weights||!indexScore||!ex||query->viewDims.size()!=3||key->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=query; e->b=key; e->c=weights; e->out=indexScore;
    e->m=query->viewDims[0]; e->k=key->viewDims[0]; e->n=query->viewDims[1]; e->reduceCount=query->viewDims[2]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLightningIndexer(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int Q=e->m,K=e->k,Hh=e->n,Dd=e->reduceCount; k_lightning<<<nb((int64_t)Q*K),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,Q,K,Hh,Dd); return done(e); }
// softmax-lse over the index scores (per query row): lse[q] = log Σ_k exp(score[q,k]); probs optional
} // extern "C"
namespace { __global__ void k_softmax_lse(const float*score,float*lse,float*probs,int Q,int K){
    int q=blockIdx.x; if(q>=Q||threadIdx.x!=0) return; const float*p=score+(int64_t)q*K; float mx=-1e30f; for(int k=0;k<K;k++)mx=fmaxf(mx,p[k]);
    double s=0; for(int k=0;k<K;k++)s+=exp(p[k]-mx); lse[q]=mx+(float)log(s); if(probs){ float*pp=probs+(int64_t)q*K; for(int k=0;k<K;k++)pp[k]=(float)(exp(p[k]-mx)/s); } } }
extern "C" {
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(const aclTensor *indexScore, aclTensor *lse, aclTensor *probs, uint64_t *ws, aclOpExecutor **ex){
    if(!indexScore||!lse||!ex||indexScore->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=indexScore; e->out=lse; e->out2=probs; e->m=indexScore->viewDims[0]; e->n=indexScore->viewDims[1]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int Q=e->m,K=e->n; k_softmax_lse<<<(unsigned)Q,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->out2?(float*)e->out2->data:nullptr,Q,K); return done(e); }
// GradKLLoss (dense/sparse): grad of KL(target||softmax(score)) wrt score = softmax(score) - target  (per query row)
} // extern "C"
namespace { __global__ void k_indexer_klgrad(const float*score,const float*target,float*grad,int Q,int K){
    int q=blockIdx.x; if(q>=Q||threadIdx.x!=0) return; const float*p=score+(int64_t)q*K,*tp=target+(int64_t)q*K; float mx=-1e30f; for(int k=0;k<K;k++)mx=fmaxf(mx,p[k]);
    double s=0; for(int k=0;k<K;k++)s+=exp(p[k]-mx); float*gp=grad+(int64_t)q*K; for(int k=0;k<K;k++) gp[k]=(float)(exp(p[k]-mx)/s)-tp[k]; } }
extern "C" {
aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *ws, aclOpExecutor **ex){
    if(!indexScore||!target||!gradScore||!ex||indexScore->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=indexScore; e->b=target; e->out=gradScore; e->m=indexScore->viewDims[0]; e->n=indexScore->viewDims[1]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDenseLightningIndexerGradKLLoss(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int Q=e->m,K=e->n; k_indexer_klgrad<<<(unsigned)Q,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,Q,K); return done(e); }
aclnnStatus aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *ws, aclOpExecutor **ex){
    return aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(indexScore,target,gradScore,ws,ex);
}
aclnnStatus aclnnSparseLightningIndexerGradKLLoss(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDenseLightningIndexerGradKLLoss(w,wz,e,s); }
// LightningIndexerGrad: grad wrt weights = Σ_k gradScore[q,k]*relu(dot_h); here gradWeights[q,h] = Σ_k gradScore[q,k]*relu(q_h·k_h)
} // extern "C"
namespace { __global__ void k_lightning_grad(const float*gscore,const float*q,const float*key,float*gw,int Q,int K,int Hh,int Dd){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)Q*Hh) return; int qi=i/Hh,h=i%Hh; float acc=0;
    const float*qp=q+((int64_t)qi*Hh+h)*Dd;
    for(int k=0;k<K;k++){ const float*kp=key+((int64_t)k*Hh+h)*Dd; float dot=0; for(int d=0;d<Dd;d++)dot+=qp[d]*kp[d]; acc+=gscore[(int64_t)qi*K+k]*fmaxf(dot,0.f); }
    gw[i]=acc; } }
extern "C" {
aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(const aclTensor *gradIndexScore, const aclTensor *query, const aclTensor *key, aclTensor *gradWeights, uint64_t *ws, aclOpExecutor **ex){
    if(!gradIndexScore||!query||!key||!gradWeights||!ex||query->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=gradIndexScore; e->b=query; e->c=key; e->out=gradWeights;
    e->m=query->viewDims[0]; e->k=key->viewDims[0]; e->n=query->viewDims[1]; e->reduceCount=query->viewDims[2]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLightningIndexerGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int Q=e->m,K=e->k,Hh=e->n,Dd=e->reduceCount; k_lightning_grad<<<nb((int64_t)Q*Hh),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,Q,K,Hh,Dd); return done(e); }

} // extern "C"
} // namespace _index3_ext

namespace _ndindex_ext {
// N-dimensional gather/scatter (ops-nn/index): GatherNd, ScatterNdUpdate.
// indices last dim K selects the first K dims of data; the trailing data dims form the copied slice.

namespace {
constexpr int NT = 256, NMAXK = 8;
inline int64_t nb(int64_t n) { return (n + NT - 1) / NT; }
struct NdDesc { int K; int64_t stride[NMAXK]; int64_t numRows, sliceSize; };

template <typename T>
__global__ void k_gather_nd(const T *data, const int64_t *idx, T *out, NdDesc d) {
    int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (e >= d.numRows * d.sliceSize) return;
    int64_t row = e / d.sliceSize, s = e % d.sliceSize;
    int64_t base = 0; const int64_t *ip = idx + row * d.K;
    for (int k = 0; k < d.K; ++k) base += ip[k] * d.stride[k];
    out[e] = data[base + s];
}
template <typename T>
__global__ void k_scatter_nd_upd(const int64_t *idx, const T *upd, T *out, NdDesc d) {
    int64_t e = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (e >= d.numRows * d.sliceSize) return;
    int64_t row = e / d.sliceSize, s = e % d.sliceSize;
    int64_t base = 0; const int64_t *ip = idx + row * d.K;
    for (int k = 0; k < d.K; ++k) base += ip[k] * d.stride[k];
    out[base + s] = upd[e];
}

// build NdDesc from data shape + indices shape
bool nd_build(const aclTensor *data, const aclTensor *indices, NdDesc &d) {
    int r = (int)data->viewDims.size();
    int K = (int)indices->viewDims.back();
    if (K < 1 || K > r || K > NMAXK) return false;
    d.K = K;
    // stride[k] = product of data dims after k (contiguous)
    int64_t acc = 1;
    for (int j = r - 1; j >= 0; --j) { if (j < K) d.stride[j] = acc; acc *= data->viewDims[j]; }
    // sliceSize = product of data dims [K..r-1]
    int64_t slice = 1; for (int j = K; j < r; ++j) slice *= data->viewDims[j];
    d.sliceSize = slice;
    d.numRows = indices->numel() / K;
    return true;
}
inline aclnnStatus fin(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnGatherNdGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !indices || !out || !ex || indices->dtype != ACL_INT64 || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !indices->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    NdDesc d; if (!nd_build(self, indices, d)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->b = indices; e->out = out;
    e->k = d.K; e->m = d.numRows; e->n = d.sliceSize; e->axes.assign(d.stride, d.stride + d.K);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGatherNd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    NdDesc d; d.K = (int)e->k; d.numRows = e->m; d.sliceSize = e->n;
    for (int k = 0; k < d.K; ++k) d.stride[k] = e->axes[k];
    int64_t n = d.numRows * d.sliceSize, g = nb(n); size_t esz = dtype_size(e->a->dtype); auto st = (cudaStream_t)s;
    #define LG(T) k_gather_nd<T><<<g,NT,0,st>>>((const T*)e->a->data,(const int64_t*)e->b->data,(T*)e->out->data,d)
    switch (esz) { case 1: LG(uint8_t); break; case 2: LG(uint16_t); break; case 4: LG(uint32_t); break; case 8: LG(uint64_t); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
    #undef LG
    return fin(e);
}

// ScatterNdUpdate: out = self, then out[indices[row]] = updates[row] (slice-wise overwrite)
aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !indices || !updates || !out || !ex || indices->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != out->dtype || updates->dtype != out->dtype || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    NdDesc d; if (!nd_build(self, indices, d)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 1; e->a = self; e->b = indices; e->c = updates; e->out = out;
    e->k = d.K; e->m = d.numRows; e->n = d.sliceSize; e->axes.assign(d.stride, d.stride + d.K);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterNdUpdate(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    NdDesc d; d.K = (int)e->k; d.numRows = e->m; d.sliceSize = e->n;
    for (int k = 0; k < d.K; ++k) d.stride[k] = e->axes[k];
    auto st = (cudaStream_t)s; size_t esz = dtype_size(e->a->dtype);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel() * esz, cudaMemcpyDeviceToDevice, st);  // out = self
    int64_t n = d.numRows * d.sliceSize, g = nb(n);
    #define LS(T) k_scatter_nd_upd<T><<<g,NT,0,st>>>((const int64_t*)e->b->data,(const T*)e->c->data,(T*)e->out->data,d)
    switch (esz) { case 1: LS(uint8_t); break; case 2: LS(uint16_t); break; case 4: LS(uint32_t); break; case 8: LS(uint64_t); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
    #undef LS
    return fin(e);
}
// ScatterNd: out (caller-allocated, pre-zeroed not required) = scatter updates into a zero tensor of out's shape
aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!indices || !updates || !out || !ex || indices->dtype != ACL_INT64 || updates->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (!out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    NdDesc d; if (!nd_build(out, indices, d)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 2; e->b = indices; e->c = updates; e->out = out;
    e->k = d.K; e->m = d.numRows; e->n = d.sliceSize; e->axes.assign(d.stride, d.stride + d.K);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterNd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    NdDesc d; d.K = (int)e->k; d.numRows = e->m; d.sliceSize = e->n;
    for (int k = 0; k < d.K; ++k) d.stride[k] = e->axes[k];
    auto st = (cudaStream_t)s; size_t esz = dtype_size(e->out->dtype);
    cudaMemsetAsync(e->out->data, 0, (size_t)e->out->numel() * esz, st);   // zero, then overwrite scattered slices
    int64_t n = d.numRows * d.sliceSize, g = nb(n);
    #define LN(T) k_scatter_nd_upd<T><<<g,NT,0,st>>>((const int64_t*)e->b->data,(const T*)e->c->data,(T*)e->out->data,d)
    switch (esz) { case 1: LN(uint8_t); break; case 2: LN(uint16_t); break; case 4: LN(uint32_t); break; case 8: LN(uint64_t); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
    #undef LN
    return fin(e);
}

} // extern "C"
} // namespace _ndindex_ext

