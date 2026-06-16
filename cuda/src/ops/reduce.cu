// Reduction family: full + arbitrary dim subset + keepDim.
//   - Full reduction to scalar (out numel=1): fast path k_part/k_final (shared reduction within block; Sum uses double accumulation).
//   - Arbitrary dim subset reduction: segmented kernel k_seg (one thread per output element, loops over the reduction region).
// Covers Sum/Max/Min/Mean/Prod (values) + ArgMax/ArgMin (along single dim, returns int64 indices).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>

namespace {

constexpr int RT = 256;       // threads per block
constexpr int MAXB = 1024;    // max partial blocks (sufficient for 2^28 elements)
constexpr int MAXD = 8;

enum RKind { R_SUM, R_MAX, R_MIN, R_PROD };

template <typename A, int K>
__device__ inline A rop(A a, A b) {
    return K == R_SUM ? a + b : K == R_PROD ? a * b : K == R_MAX ? (a > b ? a : b) : (a < b ? a : b);
}

// ---- Full reduction fast path (scalar output) ----
template <typename T, typename A, int K>
__global__ void k_part(const T *x, A *part, int64_t n) {
    __shared__ A sm[RT];
    A v = K == R_SUM ? (A)0 : (A)x[0];
    for (int64_t i = (int64_t)blockIdx.x * RT + threadIdx.x; i < n; i += (int64_t)gridDim.x * RT)
        v = rop<A, K>(v, (A)x[i]);
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int s = RT / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = rop<A, K>(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) part[blockIdx.x] = sm[0];
}

template <typename T, typename A, int K>
__global__ void k_final(const A *part, int m, T *out) {
    __shared__ A sm[RT];
    A v = K == R_SUM ? (A)0 : part[0];
    for (int i = threadIdx.x; i < m; i += RT) v = rop<A, K>(v, part[i]);
    sm[threadIdx.x] = v;
    __syncthreads();
    for (int s = RT / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] = rop<A, K>(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = (T)sm[0];
}

template <typename T, typename A, int K>
aclnnStatus run_full(const aclTensor *a, aclTensor *o, void *ws, cudaStream_t s) {
    const int64_t n = a->numel();
    int blocks = (int)std::min<int64_t>((n + RT - 1) / RT, MAXB);
    k_part<T, A, K><<<blocks, RT, 0, s>>>((const T *)a->data, (A *)ws, n);
    k_final<T, A, K><<<1, RT, 0, s>>>((const A *)ws, blocks, (T *)o->data);
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- Arbitrary dim subset reduction ----
struct RDesc {
    int nk, nr;                       // number of kept dims / number of reduced dims
    int64_t ks[MAXD], kstr[MAXD];     // kept dims: size, input stride
    int64_t rs[MAXD], rstr[MAXD];     // reduced dims: size, input stride
    int64_t nout, rcount;
};

// Value reduction: one thread per output element, loops over reduction region. Divides by rcount when mean=1.
template <typename T, typename A, int K>
__global__ void k_seg(const T *x, T *out, RDesc d, int mean) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    A acc = (A)0;
    for (int64_t j = 0; j < d.rcount; ++j) {
        int64_t rr = j, off = base;
        for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; off += c * d.rstr[i]; }
        A v = (A)x[off];
        acc = (j == 0) ? v : rop<A, K>(acc, v);   // j==0: direct assignment; SUM/PROD/MAX/MIN all use this unified start
    }
    if (mean) acc = acc / (A)d.rcount;
    out[oi] = (T)acc;
}

// Index reduction (argmax/argmin): returns the flat index of the best element in the reduction region (for single-dim reduction this is the index along that dim).
template <typename T, typename A, bool MAX>
__global__ void k_seg_arg(const T *x, int64_t *out, RDesc d) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    A best = 0; int64_t bestj = 0;
    for (int64_t j = 0; j < d.rcount; ++j) {
        int64_t rr = j, off = base;
        for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; off += c * d.rstr[i]; }
        A v = (A)x[off];
        if (j == 0 || (MAX ? v > best : v < best)) { best = v; bestj = j; }
    }
    out[oi] = bestj;
}

// Statistical segmented reduction: one thread per output, loops over reduction region. out has same type as input (All/Any use a dedicated bool kernel).
//   MODE: 0=Norm(param=p) 1=Var 2=Std 3=LogSumExp; Var/Std use unbiased (/(cnt-1)).
template <typename T, int MODE>
__global__ void k_seg_stat(const T *x, T *out, RDesc d, double param) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    auto off = [&](int64_t j) { int64_t rr = j, o = base; for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; o += c * d.rstr[i]; } return o; };
    if (MODE == 0) { double s = 0; for (int64_t j = 0; j < d.rcount; ++j) s += pow(fabs((double)x[off(j)]), param); out[oi] = (T)pow(s, 1.0 / param); }
    else if (MODE == 3) { double mx = -1e300; for (int64_t j = 0; j < d.rcount; ++j) mx = fmax(mx, (double)x[off(j)]);
        double se = 0; for (int64_t j = 0; j < d.rcount; ++j) se += exp((double)x[off(j)] - mx); out[oi] = (T)(mx + log(se)); }
    else if (MODE == 4 || MODE == 5) {   // 4=nansum 5=nanmean: skip NaN
        double s = 0; int64_t cnt = 0; for (int64_t j = 0; j < d.rcount; ++j) { double v = (double)x[off(j)]; if (!isnan(v)) { s += v; cnt++; } }
        out[oi] = (T)(MODE == 5 ? (cnt ? s / cnt : 0.0) : s); }
    else if (MODE == 6) {   // 6=reduce-logsum: log(sum(x))
        double s = 0; for (int64_t j = 0; j < d.rcount; ++j) s += (double)x[off(j)]; out[oi] = (T)log(s); }
    else { double m = 0; for (int64_t j = 0; j < d.rcount; ++j) m += (double)x[off(j)]; m /= d.rcount;
        double v = 0; for (int64_t j = 0; j < d.rcount; ++j) { double t = (double)x[off(j)] - m; v += t * t; }
        v /= (d.rcount > 1 ? d.rcount - 1 : 1); out[oi] = (T)(MODE == 2 ? sqrt(v) : v); }
}
// CountNonzero: int64 output, counts nonzero elements in the reduction region
template <typename T>
__global__ void k_seg_count(const T *x, int64_t *out, RDesc d) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    int64_t cnt = 0;
    for (int64_t j = 0; j < d.rcount; ++j) { int64_t rr = j, o = base; for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; o += c * d.rstr[i]; }
        if ((double)x[o] != 0.0) cnt++; }
    out[oi] = cnt;
}
// k-th smallest by rank (0-indexed k) via counting selection (O(rcount^2), functional). Median = k=(rcount-1)/2.
template <typename T, typename A>
__device__ A kth_select(const T *x, const RDesc &d, int64_t base, int64_t k) {
    auto val = [&](int64_t j) { int64_t rr = j, o = base; for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; o += c * d.rstr[i]; } return (A)x[o]; };
    for (int64_t j = 0; j < d.rcount; ++j) {
        A vj = val(j); int64_t less = 0, eqbefore = 0;
        for (int64_t m = 0; m < d.rcount; ++m) { A vm = val(m); if (vm < vj) less++; else if (vm == vj && m < j) eqbefore++; }
        if (less <= k && k < less + eqbefore + 1) return vj;   // rank of element j (stable on ties) equals k
    }
    return val(0);
}
template <typename T, typename A>
__global__ void k_seg_kth(const T *x, T *out, RDesc d, int64_t k) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    out[oi] = (T)kth_select<T, A>(x, d, base, k);
}
// Quantile with linear interpolation: pos=q*(n-1); blends two order statistics
template <typename T, typename A>
__global__ void k_seg_quantile(const T *x, T *out, RDesc d, double q) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    double pos = q * (double)(d.rcount - 1); int64_t lo = (int64_t)floor(pos), hi = (int64_t)ceil(pos);
    double frac = pos - (double)lo;
    A vlo = kth_select<T, A>(x, d, base, lo), vhi = (hi == lo) ? vlo : kth_select<T, A>(x, d, base, hi);
    out[oi] = (T)((double)vlo * (1.0 - frac) + (double)vhi * frac);
}
// Mode: most frequent value; ties broken by smallest value (PyTorch returns the smallest among modes)
template <typename T, typename A>
__global__ void k_seg_mode(const T *x, T *out, RDesc d) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    auto val = [&](int64_t j) { int64_t rr = j, o = base; for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; o += c * d.rstr[i]; } return (A)x[o]; };
    A bestv = val(0); int64_t bestc = 0;
    for (int64_t j = 0; j < d.rcount; ++j) { A vj = val(j); int64_t c = 0; for (int64_t m = 0; m < d.rcount; ++m) if (val(m) == vj) c++;
        if (c > bestc || (c == bestc && vj < bestv)) { bestc = c; bestv = vj; } }
    out[oi] = (T)bestv;
}
template <typename T, bool ANY>
__global__ void k_seg_bool(const T *x, uint8_t *out, RDesc d) {
    int64_t oi = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (oi >= d.nout) return;
    int64_t rem = oi, base = 0;
    for (int i = d.nk - 1; i >= 0; --i) { int64_t c = rem % d.ks[i]; rem /= d.ks[i]; base += c * d.kstr[i]; }
    bool a = ANY ? false : true;
    for (int64_t j = 0; j < d.rcount; ++j) { int64_t rr = j, o = base; for (int i = d.nr - 1; i >= 0; --i) { int64_t c = rr % d.rs[i]; rr /= d.rs[i]; o += c * d.rstr[i]; }
        bool nz = (double)x[o] != 0.0; a = ANY ? (a || nz) : (a && nz); }
    out[oi] = a ? 1 : 0;
}

// Build RDesc on the host: axes is the sorted, deduplicated set of reduction dims
bool build_rdesc(const aclTensor *a, const std::vector<int64_t> &axes, RDesc &d) {
    int rank = (int)a->viewDims.size();
    if (rank > MAXD) return false;
    int64_t istr[MAXD], acc = 1;
    for (int i = rank - 1; i >= 0; --i) { istr[i] = acc; acc *= a->viewDims[i]; }
    bool isred[MAXD] = {false};
    for (auto ax : axes) { if (ax < 0 || ax >= rank) return false; isred[ax] = true; }
    d.nk = d.nr = 0; d.nout = 1; d.rcount = 1;
    for (int i = 0; i < rank; ++i) {
        if (isred[i]) { d.rs[d.nr] = a->viewDims[i]; d.rstr[d.nr] = istr[i]; d.rcount *= a->viewDims[i]; d.nr++; }
        else          { d.ks[d.nk] = a->viewDims[i]; d.kstr[d.nk] = istr[i]; d.nout *= a->viewDims[i]; d.nk++; }
    }
    return true;
}

template <typename T, typename A, int K>
aclnnStatus launch_seg(const aclTensor *a, aclTensor *o, const RDesc &d, int mean, cudaStream_t s) {
    int64_t g = (d.nout + RT - 1) / RT;
    k_seg<T, A, K><<<g, RT, 0, s>>>((const T *)a->data, (T *)o->data, d, mean);
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

template <typename T, typename A, bool MAX>
aclnnStatus launch_arg(const aclTensor *a, aclTensor *o, const RDesc &d, cudaStream_t s) {
    int64_t g = (d.nout + RT - 1) / RT;
    k_seg_arg<T, A, MAX><<<g, RT, 0, s>>>((const T *)a->data, (int64_t *)o->data, d);
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// dtype dispatch for value reduction (mean flag indicates whether to compute mean)
template <int K>
aclnnStatus seg_value(const aclTensor *a, aclTensor *o, const RDesc &d, int mean, cudaStream_t s) {
    switch (a->dtype) {
        case ACL_FLOAT:   return launch_seg<float, double, K>(a, o, d, mean, s);
        case ACL_FLOAT16: return launch_seg<__half, float, K>(a, o, d, mean, s);
        case ACL_BF16:    return launch_seg<__nv_bfloat16, float, K>(a, o, d, mean, s);
        case ACL_INT32:   return launch_seg<int32_t, int64_t, K>(a, o, d, mean, s);
        default: return ACLNN_ERR_PARAM_INVALID;
    }
}

aclnnStatus seg_arg(const aclTensor *a, aclTensor *o, const RDesc &d, bool ismax, cudaStream_t s) {
    if (o->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;  // index output is always int64
    switch (a->dtype) {
        case ACL_FLOAT:   return ismax ? launch_arg<float, float, true>(a, o, d, s)   : launch_arg<float, float, false>(a, o, d, s);
        case ACL_FLOAT16: return ismax ? launch_arg<__half, float, true>(a, o, d, s)  : launch_arg<__half, float, false>(a, o, d, s);
        case ACL_INT32:   return ismax ? launch_arg<int32_t, int32_t, true>(a, o, d, s): launch_arg<int32_t, int32_t, false>(a, o, d, s);
        default: return ACLNN_ERR_PARAM_INVALID;
    }
}

template <int MODE>
aclnnStatus seg_statf(const aclTensor *a, aclTensor *o, const RDesc &d, double param, cudaStream_t s) {
    int64_t g = (d.nout + RT - 1) / RT;
    switch (a->dtype) {
        case ACL_FLOAT:   k_seg_stat<float, MODE><<<g, RT, 0, s>>>((const float *)a->data, (float *)o->data, d, param); break;
        case ACL_FLOAT16: k_seg_stat<__half, MODE><<<g, RT, 0, s>>>((const __half *)a->data, (__half *)o->data, d, param); break;
        case ACL_BF16:    k_seg_stat<__nv_bfloat16, MODE><<<g, RT, 0, s>>>((const __nv_bfloat16 *)a->data, (__nv_bfloat16 *)o->data, d, param); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
aclnnStatus seg_bool_d(const aclTensor *a, aclTensor *o, const RDesc &d, bool any, cudaStream_t s) {
    int64_t g = (d.nout + RT - 1) / RT; uint8_t *out = (uint8_t *)o->data;
    #define BL(T) (any ? k_seg_bool<T,true><<<g,RT,0,s>>>((const T*)a->data,out,d) : k_seg_bool<T,false><<<g,RT,0,s>>>((const T*)a->data,out,d))
    switch (a->dtype) {
        case ACL_FLOAT: BL(float); break; case ACL_FLOAT16: BL(__half); break;
        case ACL_INT32: BL(int32_t); break; case ACL_BOOL: case ACL_UINT8: BL(uint8_t); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    #undef BL
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

aclnnStatus seg_count_d(const aclTensor *a, aclTensor *o, const RDesc &d, cudaStream_t s) {
    if (o->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int64_t g = (d.nout + RT - 1) / RT; int64_t *out = (int64_t *)o->data;
    switch (a->dtype) {
        case ACL_FLOAT:   k_seg_count<float><<<g,RT,0,s>>>((const float*)a->data,out,d); break;
        case ACL_FLOAT16: k_seg_count<__half><<<g,RT,0,s>>>((const __half*)a->data,out,d); break;
        case ACL_INT32:   k_seg_count<int32_t><<<g,RT,0,s>>>((const int32_t*)a->data,out,d); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
// selection ops (kthvalue/median/quantile/mode); SEL: 0=kth(param=k as int) 1=quantile(param=q) 2=mode
template <int SEL>
aclnnStatus seg_select_d(const aclTensor *a, aclTensor *o, const RDesc &d, double param, cudaStream_t s) {
    int64_t g = (d.nout + RT - 1) / RT;
    #define SEL_CALL(T, A) do { \
        if (SEL == 0) k_seg_kth<T,A><<<g,RT,0,s>>>((const T*)a->data,(T*)o->data,d,(int64_t)param); \
        else if (SEL == 1) k_seg_quantile<T,A><<<g,RT,0,s>>>((const T*)a->data,(T*)o->data,d,param); \
        else k_seg_mode<T,A><<<g,RT,0,s>>>((const T*)a->data,(T*)o->data,d); } while (0)
    switch (a->dtype) {
        case ACL_FLOAT:   SEL_CALL(float, double); break;
        case ACL_FLOAT16: SEL_CALL(__half, float); break;
        case ACL_BF16:    SEL_CALL(__nv_bfloat16, float); break;
        case ACL_INT32:   SEL_CALL(int32_t, int64_t); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    #undef SEL_CALL
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
// Renorm scale: out = x * min(1, maxnorm/(norm[idx along dim]+eps))
template <typename T>
__global__ void k_renorm_scale(const T *x, const T *norm, T *out, int64_t n, int64_t strideDim, int64_t dimSize, double maxnorm) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t idx = (i / strideDim) % dimSize;
    double nv = (double)norm[idx], sc = nv > maxnorm ? maxnorm / (nv + 1e-7) : 1.0;
    out[i] = (T)((double)x[i] * sc);
}

// ---- General: normalize reduce dims, validate out shape ----
// Normalize IntArray/single dim to a sorted, deduplicated list of non-negative indices; empty means full reduction
bool normalize_axes(const aclIntArray *dim, int rank, std::vector<int64_t> &out) {
    out.clear();
    if (!dim || dim->v.empty()) { for (int i = 0; i < rank; ++i) out.push_back(i); return true; }
    for (auto ax : dim->v) { int64_t a = ax < 0 ? ax + rank : ax; if (a < 0 || a >= rank) return false; out.push_back(a); }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return true;
}

// Expected output shape: remove reduced dims (keepDim keeps them as 1). Validate out matches (numel comparison as fallback for scalar equivalence)
bool check_out_shape(const aclTensor *a, const std::vector<int64_t> &axes, bool keepDim, const aclTensor *out) {
    int rank = (int)a->viewDims.size();
    bool isred[MAXD] = {false};
    for (auto ax : axes) isred[ax] = true;
    std::vector<int64_t> exp;
    for (int i = 0; i < rank; ++i) {
        if (isred[i]) { if (keepDim) exp.push_back(1); }
        else exp.push_back(a->viewDims[i]);
    }
    int64_t en = 1; for (auto d : exp) en *= d;
    return out->numel() == en;   // PoC: validate by numel (tolerates scalar {} vs {1})
}

aclnnStatus make_reduce(int op, const aclTensor *self, const aclIntArray *dim, bool keepDim,
                        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    std::vector<int64_t> axes;
    if (!normalize_axes(dim, rank, axes)) return ACLNN_ERR_PARAM_INVALID;
    if (!check_out_shape(self, axes, keepDim, out)) return ACLNN_ERR_PARAM_INVALID;
    bool isArg = (op == OP_ARGMAX || op == OP_ARGMIN);
    bool isBool = (op == OP_ALL || op == OP_ANY);
    bool isCount = (op == OP_COUNTNZ);
    if (isCount && out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (!isArg && !isBool && !isCount && out->dtype != self->dtype) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = op; e->a = self; e->out = out; e->axes = axes; e->keepDim = keepDim;
    *ws = MAXB * sizeof(double);    // used by the full fast path; unused by segmented but harmless to allocate
    *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus run_reduce(aclOpExecutor *e, void *ws, cudaStream_t s) {
    if (!e || !ws) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *a = e->a; aclTensor *o = e->out;
    int rank = (int)a->viewDims.size();
    bool full = ((int)e->axes.size() == rank);   // reduce all dims → scalar
    aclnnStatus st = ACLNN_ERR_PARAM_INVALID;

    // Full Sum/Max/Min use the original fast path (preserves deterministic double accumulation and performance)
    if (full && (e->op == OP_REDUCE_SUM || e->op == OP_AMAX || e->op == OP_AMIN)) {
        if (e->op == OP_REDUCE_SUM) switch (a->dtype) {
            case ACL_FLOAT:   st = run_full<float, double, R_SUM>(a, o, ws, s); break;
            case ACL_FLOAT16: st = run_full<__half, float, R_SUM>(a, o, ws, s); break;
            case ACL_INT32:   st = run_full<int32_t, int64_t, R_SUM>(a, o, ws, s); break;
            default: break;
        } else if (e->op == OP_AMAX) switch (a->dtype) {
            case ACL_FLOAT:   st = run_full<float, float, R_MAX>(a, o, ws, s); break;
            case ACL_FLOAT16: st = run_full<__half, float, R_MAX>(a, o, ws, s); break;
            case ACL_INT32:   st = run_full<int32_t, int32_t, R_MAX>(a, o, ws, s); break;
            default: break;
        } else switch (a->dtype) {
            case ACL_FLOAT:   st = run_full<float, float, R_MIN>(a, o, ws, s); break;
            case ACL_FLOAT16: st = run_full<__half, float, R_MIN>(a, o, ws, s); break;
            case ACL_INT32:   st = run_full<int32_t, int32_t, R_MIN>(a, o, ws, s); break;
            default: break;
        }
        delete e;
        return st;
    }

    // Renorm: norm each sub-tensor (over all dims except `dim`), then scale so its p-norm <= maxnorm
    if (e->op == OP_RENORM) {
        RDesc d;
        if (!build_rdesc(a, e->axes, d)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        aclTensor tmp; tmp.viewDims = {d.nout}; tmp.strides = {1}; tmp.dtype = a->dtype; tmp.format = ACL_FORMAT_ND; tmp.data = ws;
        st = seg_statf<0>(a, &tmp, d, e->eps, s);   // per-slice p-norm into ws
        if (st != ACLNN_SUCCESS) { delete e; return st; }
        int64_t dim = e->dim, strideDim = 1; for (int i = (int)dim + 1; i < rank; ++i) strideDim *= a->viewDims[i];
        int64_t dimSize = a->viewDims[dim], n = a->numel(), g = (n + RT - 1) / RT;
        switch (a->dtype) {
            case ACL_FLOAT:   k_renorm_scale<float><<<g,RT,0,s>>>((const float*)a->data,(const float*)ws,(float*)o->data,n,strideDim,dimSize,e->alpha); break;
            case ACL_FLOAT16: k_renorm_scale<__half><<<g,RT,0,s>>>((const __half*)a->data,(const __half*)ws,(__half*)o->data,n,strideDim,dimSize,e->alpha); break;
            case ACL_BF16:    k_renorm_scale<__nv_bfloat16><<<g,RT,0,s>>>((const __nv_bfloat16*)a->data,(const __nv_bfloat16*)ws,(__nv_bfloat16*)o->data,n,strideDim,dimSize,e->alpha); break;
            default: delete e; return ACLNN_ERR_PARAM_INVALID;
        }
        delete e;
        return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }

    // General segmented path
    RDesc d;
    if (!build_rdesc(a, e->axes, d)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    switch (e->op) {
        case OP_REDUCE_SUM: st = seg_value<R_SUM>(a, o, d, 0, s); break;
        case OP_MEAN:       st = seg_value<R_SUM>(a, o, d, 1, s); break;
        case OP_PROD:       st = seg_value<R_PROD>(a, o, d, 0, s); break;
        case OP_AMAX:       st = seg_value<R_MAX>(a, o, d, 0, s); break;
        case OP_AMIN:       st = seg_value<R_MIN>(a, o, d, 0, s); break;
        case OP_ARGMAX:     st = seg_arg(a, o, d, true, s); break;
        case OP_ARGMIN:     st = seg_arg(a, o, d, false, s); break;
        case OP_NORM:       st = seg_statf<0>(a, o, d, e->eps, s); break;   // eps field reused to store p
        case OP_VAR:        st = seg_statf<1>(a, o, d, 0, s); break;
        case OP_STD:        st = seg_statf<2>(a, o, d, 0, s); break;
        case OP_LSE:        st = seg_statf<3>(a, o, d, 0, s); break;
        case OP_ALL:        st = seg_bool_d(a, o, d, false, s); break;
        case OP_ANY:        st = seg_bool_d(a, o, d, true, s); break;
        case OP_NANSUM:     st = seg_statf<4>(a, o, d, 0, s); break;
        case OP_NANMEAN:    st = seg_statf<5>(a, o, d, 0, s); break;
        case OP_COUNTNZ:    st = seg_count_d(a, o, d, s); break;
        case OP_MEDIAN:     st = seg_select_d<0>(a, o, d, (double)((d.rcount - 1) / 2), s); break;
        case OP_KTHVALUE:   st = seg_select_d<0>(a, o, d, (double)e->reduceCount, s); break;
        case OP_QUANTILE:   st = seg_select_d<1>(a, o, d, e->eps, s); break;
        case OP_MODE:       st = seg_select_d<2>(a, o, d, 0, s); break;
        case OP_AMINMAX:    st = seg_value<R_MIN>(a, o, d, 0, s); if (st == ACLNN_SUCCESS && e->out2) st = seg_value<R_MAX>(a, e->out2, d, 0, s); break;
        case OP_REDUCE_LOGSUM: st = seg_statf<6>(a, o, d, 0, s); break;
        case OP_VARMEAN:    st = seg_statf<1>(a, o, d, 0, s); if (st == ACLNN_SUCCESS && e->out2) st = seg_value<R_SUM>(a, e->out2, d, 1, s); break;
        case OP_STDMEAN:    st = seg_statf<2>(a, o, d, 0, s); if (st == ACLNN_SUCCESS && e->out2) st = seg_value<R_SUM>(a, e->out2, d, 1, s); break;
        case OP_MAXDIM:     st = seg_value<R_MAX>(a, o, d, 0, s); if (st == ACLNN_SUCCESS && e->out2) st = seg_arg(a, e->out2, d, true, s); break;
        case OP_MINDIM:     st = seg_value<R_MIN>(a, o, d, 0, s); if (st == ACLNN_SUCCESS && e->out2) st = seg_arg(a, e->out2, d, false, s); break;
        default: break;
    }
    delete e;
    return st;
}

} // namespace

extern "C" {

aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype,
                                           aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (dtype != ACL_DT_UNDEFINED && out && dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_reduce(OP_REDUCE_SUM, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnReduceSum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnAmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim,
                                      aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_AMAX, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnAmax(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnAminGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim,
                                      aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_AMIN, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnAmin(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype,
                                      aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (dtype != ACL_DT_UNDEFINED && out && dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_reduce(OP_MEAN, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnMean(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnProdGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype,
                                      aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (dtype != ACL_DT_UNDEFINED && out && dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_reduce(OP_PROD, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnProd(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim,
                                        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    return make_reduce(OP_ARGMAX, self, &ia, keepDim, out, ws, ex);
}
aclnnStatus aclnnArgMax(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

aclnnStatus aclnnArgMinGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim,
                                        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    return make_reduce(OP_ARGMIN, self, &ia, keepDim, out, ws, ex);
}
aclnnStatus aclnnArgMin(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// Statistical reductions
aclnnStatus aclnnAllGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return make_reduce(OP_ALL, self, dim, keepDim, out, ws, ex); }
aclnnStatus aclnnAll(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnAnyGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return make_reduce(OP_ANY, self, dim, keepDim, out, ws, ex); }
aclnnStatus aclnnAny(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnVarGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return make_reduce(OP_VAR, self, dim, keepDim, out, ws, ex); }
aclnnStatus aclnnVar(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnStdGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return make_reduce(OP_STD, self, dim, keepDim, out, ws, ex); }
aclnnStatus aclnnStd(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnLogSumExpGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return make_reduce(OP_LSE, self, dim, keepDim, out, ws, ex); }
aclnnStatus aclnnLogSumExp(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_reduce(OP_NORM, self, dim, keepDim, out, ws, ex); if (st == ACLNN_SUCCESS) (*ex)->eps = p; return st;
}
aclnnStatus aclnnNorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
// LinalgVectorNorm: p-norm over given dims (same as Norm)
aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; aclnnStatus st = make_reduce(OP_NORM, self, dim, keepDim, out, ws, ex); if (st == ACLNN_SUCCESS) (*ex)->eps = p; return st;
}
aclnnStatus aclnnLinalgVectorNorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
// InplaceRenorm: renormalize in place (out = self)
aclnnStatus aclnnInplaceRenormGetWorkspaceSize(aclTensor *self, double p, int64_t dim, double maxnorm, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRenormGetWorkspaceSize(self, p, dim, maxnorm, self, ws, ex);
}
aclnnStatus aclnnInplaceRenorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// ---- P3 reduce & scan extensions ----
aclnnStatus aclnnNansumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (dtype != ACL_DT_UNDEFINED && out && dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_reduce(OP_NANSUM, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnNansum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnNanmeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (dtype != ACL_DT_UNDEFINED && out && dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_reduce(OP_NANMEAN, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnNanmean(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnCountNonzeroGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_COUNTNZ, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnCountNonzero(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnAminmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_reduce(OP_AMINMAX, self, dim, keepDim, outMin, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = outMax;
    return st;
}
aclnnStatus aclnnAminmax(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    return make_reduce(OP_MEDIAN, self, &ia, keepDim, out, ws, ex);
}
aclnnStatus aclnnMedian(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnKthvalueGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    aclnnStatus st = make_reduce(OP_KTHVALUE, self, &ia, keepDim, out, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->reduceCount = (k > 0 ? k - 1 : 0);   // k is 1-indexed (PyTorch); store 0-indexed rank
    return st;
}
aclnnStatus aclnnKthvalue(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnQuantileGetWorkspaceSize(const aclTensor *self, double q, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    aclnnStatus st = make_reduce(OP_QUANTILE, self, &ia, keepDim, out, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->eps = q;
    return st;
}
aclnnStatus aclnnQuantile(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnModeGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    int64_t d[1] = {dim}; aclIntArray ia{std::vector<int64_t>(d, d + 1)};
    return make_reduce(OP_MODE, self, &ia, keepDim, out, ws, ex);
}
aclnnStatus aclnnMode(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
// Renorm: renormalize each sub-tensor along `dim` so its p-norm <= maxnorm. Output keeps full input shape.
aclnnStatus aclnnRenormGetWorkspaceSize(const aclTensor *self, double p, int64_t dim, double maxnorm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->contiguous() || self->viewDims != out->viewDims || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    int64_t nd = dim < 0 ? dim + rank : dim;
    if (nd < 0 || nd >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_RENORM; e->a = self; e->out = out; e->eps = p; e->alpha = maxnorm; e->dim = nd;
    for (int i = 0; i < rank; ++i) if (i != nd) e->axes.push_back(i);   // reduce all dims except `dim`
    *ws = self->viewDims[nd] * (int64_t)sizeof(double) + 256;
    *ex = e;
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnRenorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// ================= additional reduction variants =================
// helpers: single-dim IntArray
#define ONE_DIM(name) int64_t _d[1] = {name}; aclIntArray _ia{std::vector<int64_t>(_d, _d + 1)}

// full reductions (reduce over all dims → scalar)
aclnnStatus aclnnSumGetWorkspaceSize(const aclTensor *self, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; return make_reduce(OP_REDUCE_SUM, self, nullptr, false, out, ws, ex);
}
aclnnStatus aclnnSum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnMaxGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_AMAX, self, nullptr, false, out, ws, ex);
}
aclnnStatus aclnnMax(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnMaxV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_AMAX, self, nullptr, false, out, ws, ex);
}
aclnnStatus aclnnMaxV2(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnMinGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_AMIN, self, nullptr, false, out, ws, ex);
}
aclnnStatus aclnnMin(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// dim variants / aliases
aclnnStatus aclnnMeanV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; return make_reduce(OP_MEAN, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnMeanV2(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnProdDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; ONE_DIM(dim); return make_reduce(OP_PROD, self, &_ia, keepDim, out, ws, ex);
}
aclnnStatus aclnnProdDim(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnReduceNansumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; return make_reduce(OP_NANSUM, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnReduceNansum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_reduce(OP_REDUCE_LOGSUM, self, dim, keepDim, out, ws, ex);
}
aclnnStatus aclnnReduceLogSum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// aminmax variants
aclnnStatus aclnnAminmaxAllGetWorkspaceSize(const aclTensor *self, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_reduce(OP_AMINMAX, self, nullptr, false, outMin, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = outMax;
    return st;
}
aclnnStatus aclnnAminmaxAll(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    ONE_DIM(dim); aclnnStatus st = make_reduce(OP_AMINMAX, self, &_ia, keepDim, outMin, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = outMax;
    return st;
}
aclnnStatus aclnnAminmaxDim(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// dual-output: (values, indices) along a dim
aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    ONE_DIM(dim); aclnnStatus st = make_reduce(OP_MAXDIM, self, &_ia, keepDim, valuesOut, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = indicesOut;
    return st;
}
aclnnStatus aclnnMaxDim(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnMinDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    ONE_DIM(dim); aclnnStatus st = make_reduce(OP_MINDIM, self, &_ia, keepDim, valuesOut, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = indicesOut;
    return st;
}
aclnnStatus aclnnMinDim(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }

// dual-output: (var/std, mean)
aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *varOut, aclTensor *meanOut, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_reduce(OP_VARMEAN, self, dim, keepDim, varOut, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = meanOut;
    return st;
}
aclnnStatus aclnnVarMean(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *stdOut, aclTensor *meanOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)correction; aclnnStatus st = make_reduce(OP_STDMEAN, self, dim, keepDim, stdOut, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out2 = meanOut;
    return st;
}
aclnnStatus aclnnStdMeanCorrection(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_reduce(e, ws, (cudaStream_t)s); }
#undef ONE_DIM

} // extern "C"
