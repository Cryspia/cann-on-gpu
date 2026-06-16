// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace _shape {
// Shape/index op family: Permute (transpose), Slice, Tile/BroadcastTo, Pad (constant), Gather (along dim), Concat, Reshape.
// Copy family (except special semantics of Pad/Gather/Concat) dispatches by dtype byte width (1/2/4/8) via uintN generics, dtype-agnostic.
// out is always contiguous (freshly allocated tensor); inputs may be non-contiguous views (real stride+offset used).

namespace {

constexpr int MAXD = 8;
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// ---- General index copy: Permute/Slice/Tile/BroadcastTo ----
// off = base + Σ_k ( (imod[k] ? coord_k % imod[k] : coord_k) * istr[k] )
struct SCopy { int rank; int64_t od[MAXD], istr[MAXD], imod[MAXD], base; };
template <typename T>
__global__ void k_scopy(const T *in, T *out, SCopy d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, off = d.base;
    for (int k = d.rank - 1; k >= 0; --k) {
        int64_t c = rem % d.od[k]; rem /= d.od[k];
        int64_t ic = d.imod[k] ? (c % d.imod[k]) : c;
        off += ic * d.istr[k];
    }
    out[i] = in[off];
}

// ---- Constant Pad ----
struct SPad { int rank; int64_t od[MAXD], idim[MAXD], istr[MAXD], lo[MAXD], base; };
template <typename T>
__global__ void k_pad(const T *in, T *out, SPad d, T val, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, off = d.base; bool inside = true;
    for (int k = d.rank - 1; k >= 0; --k) {
        int64_t c = rem % d.od[k]; rem /= d.od[k];
        int64_t ic = c - d.lo[k];
        if (ic < 0 || ic >= d.idim[k]) inside = false; else off += ic * d.istr[k];
    }
    out[i] = inside ? in[off] : val;
}

// ---- Gather along a single dim (index 1D int64) ----
struct SGather { int rank, gd; int64_t od[MAXD], istr[MAXD], base; };
template <typename T>
__global__ void k_gather(const T *in, const int64_t *idx, T *out, SGather d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, off = d.base, g = 0;
    for (int k = d.rank - 1; k >= 0; --k) {
        int64_t c = rem % d.od[k]; rem /= d.od[k];
        if (k == d.gd) g = c; else off += c * d.istr[k];
    }
    off += idx[g] * d.istr[d.gd];
    out[i] = in[off];
}

// ---- Concat: single input (contiguous) written into the offset region of out along cd ----
struct SCat { int rank, cd; int64_t id[MAXD], ostr[MAXD], coff; };
template <typename T>
__global__ void k_cat(const T *in, T *out, SCat d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, off = 0;
    for (int k = d.rank - 1; k >= 0; --k) {
        int64_t c = rem % d.id[k]; rem /= d.id[k];
        off += ((k == d.cd) ? c + d.coff : c) * d.ostr[k];
    }
    out[off] = in[i];
}

// GatherV2 batch_dims: self[batchN, A, tail], index[batchN, I] → out[batchN, I, tail]
// (common case: batch_dims=b and axis=b — each batch gathers along the axis by its own indices, e.g. beam/KV reorder, per-batch select)
template <typename T> __global__ void k_gather_batched(const T *in, const int64_t *idx, T *o,
        int64_t batchN, int64_t A, int64_t I, int64_t tail) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= batchN * I * tail) return;
    int64_t tc = i % tail, tmp = i / tail, ii = tmp % I, bb = tmp / I;
    int64_t g = idx[bb * I + ii];
    o[i] = in[(bb * A + g) * tail + tc];
}

// Embedding: out[l,:] = weight[ids[l],:] (row-level gather, dtype-agnostic by byte width)
template <typename T> __global__ void k_embed(const T *w, const int64_t *ids, T *o, int64_t L, int64_t D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * D) return;
    int64_t l = i / D, d = i % D; o[i] = w[ids[l] * D + d];
}
__global__ void k_embed_bwd(const float *grad, const int64_t *ids, float *gw, int64_t L, int64_t D) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * D) return;
    int64_t l = i / D, d = i % D; atomicAdd(&gw[ids[l] * D + d], grad[i]);
}

// ---- Index op completions ----
__global__ void k_arange(float *o, int64_t n, float start, float step) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = start + i * step;
}
__global__ void k_onehot(const int64_t *idx, float *o, int64_t L, int64_t C, float on, float off) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * C) return;
    int64_t l = i / C, c = i % C; o[i] = (idx[l] == c) ? on : off;
}
// Flip along single dim fd (size FD, inner stride): out same shape as in, reversed along fd
template <typename T> __global__ void k_flip(const T *in, T *o, int64_t outer, int64_t FD, int64_t inner) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= outer * FD * inner) return;
    int64_t in_ = i % inner, f = (i / inner) % FD, ou = i / (inner * FD);
    o[i] = in[(ou * FD + (FD - 1 - f)) * inner + in_];
}
// Roll along single dim fd by shift sh (wrap-around)
template <typename T> __global__ void k_roll(const T *in, T *o, int64_t outer, int64_t FD, int64_t inner, int64_t sh) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= outer * FD * inner) return;
    int64_t in_ = i % inner, f = (i / inner) % FD, ou = i / (inner * FD);
    int64_t src = ((f - sh) % FD + FD) % FD;
    o[i] = in[(ou * FD + src) * inner + in_];
}
// RepeatInterleave (1D): out[i] = in[i / repeats]
template <typename T> __global__ void k_repeat_il(const T *in, T *o, int64_t n, int64_t repeats) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = in[i / repeats];
}
// ScatterUpdate along dim0: out first = self, then out[index[l],:] = src[l,:]
// Duplicate indices: last-write-wins semantics (unordered among duplicates, consistent with PyTorch scatter_ — result not guaranteed)
template <typename T> __global__ void k_scatter0(const T *src, const int64_t *idx, T *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; o[idx[l] * row + c] = src[i];
}
// ScatterAdd / index_put(accumulate) along dim0: out first = self, then out[index[l],:] += src[l,:]
// Duplicate indices: atomic accumulate (deterministic sum); relied upon by KV-cache multi-write and index_add
template <typename T> __global__ void k_scatter_add0(const T *src, const int64_t *idx, T *o, int64_t L, int64_t row) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= L * row) return;
    int64_t l = i / row, c = i % row; atomicAdd(&o[idx[l] * row + c], src[i]);
}
// MaskedSelect / Nonzero: order-preserving compaction (single-thread O(n), PoC; a prefix-sum version can be used for large n)
__global__ void k_masked_select(const float *x, const uint8_t *mask, float *o, int64_t *cnt, int64_t n) {
    int64_t pos = 0; for (int64_t i = 0; i < n; ++i) if (mask[i]) o[pos++] = x[i]; *cnt = pos;
}
__global__ void k_nonzero(const float *x, int64_t *o, int64_t *cnt, int64_t n) {
    int64_t pos = 0; for (int64_t i = 0; i < n; ++i) if (x[i] != 0.f) o[pos++] = i; *cnt = pos;
}

inline bool is_subfp(aclDataType t) {
    return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2 || t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2;
}

// Dispatch copy-family kernels by byte width (dtype-agnostic)
#define DISP_SZ(esz, LAUNCH) do {                              \
    switch (esz) {                                             \
        case 1: { LAUNCH(uint8_t);  } break;                   \
        case 2: { LAUNCH(uint16_t); } break;                   \
        case 4: { LAUNCH(uint32_t); } break;                   \
        case 8: { LAUNCH(uint64_t); } break;                   \
        default: return ACLNN_ERR_PARAM_INVALID;               \
    } } while (0)

aclnnStatus run_shape(aclOpExecutor *e, cudaStream_t s) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (is_subfp(a->dtype)) { for (auto *t : e->owned) delete t; delete e; return ACLNN_ERR_PARAM_INVALID; }
    size_t esz = dtype_size(a->dtype);
    int rank = (int)a->viewDims.size();
    int64_t n = o->numel();
    int64_t g = nb(n);
    const void *pin = a->data; void *pout = o->data;

    switch (e->op) {
        case OP_PERMUTE: {
            SCopy d{}; d.rank = rank; d.base = a->offset;
            for (int i = 0; i < rank; ++i) { int p = (int)e->axes[i]; d.od[i] = a->viewDims[p]; d.istr[i] = a->strides[p]; d.imod[i] = 0; }
            #define L(T) k_scopy<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_SLICE: {
            // along e->dim: start=e->m, step=e->k; other dims full extent
            int sd = (int)e->dim;
            SCopy d{}; d.rank = rank; d.base = a->offset + e->m * a->strides[sd];
            for (int i = 0; i < rank; ++i) { d.od[i] = o->viewDims[i]; d.istr[i] = a->strides[i] * (i == sd ? e->k : 1); d.imod[i] = 0; }
            #define L(T) k_scopy<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_STRIDED_SLICE: {
            // multi-dim strided slice: e->axes interleaved as [start0,step0,start1,step1,...] (length 2*rank)
            SCopy d{}; d.rank = rank; d.base = a->offset;
            for (int i = 0; i < rank; ++i) {
                int64_t st = e->axes[2*i], sp = e->axes[2*i+1];
                d.base += st * a->strides[i];
                d.od[i] = o->viewDims[i]; d.istr[i] = a->strides[i] * sp; d.imod[i] = 0;
            }
            #define L(T) k_scopy<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_TILE: {
            // e->axes = repeats (length rank); out dim = in dim * repeat
            SCopy d{}; d.rank = rank; d.base = a->offset;
            for (int i = 0; i < rank; ++i) { d.od[i] = o->viewDims[i]; d.istr[i] = a->strides[i]; d.imod[i] = a->viewDims[i]; }
            #define L(T) k_scopy<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_PAD: {
            // e->axes = [lo0,hi0,lo1,hi1,...]; pad value = e->alpha
            SPad d{}; d.rank = rank; d.base = a->offset;
            for (int i = 0; i < rank; ++i) { d.od[i] = o->viewDims[i]; d.idim[i] = a->viewDims[i]; d.istr[i] = a->strides[i]; d.lo[i] = e->axes[2*i]; }
            // encode the pad value into bytes for the corresponding dtype
            union { uint64_t u; } v; v.u = 0;
            switch (a->dtype) {
                case ACL_FLOAT:   { float  f = (float)e->alpha;  std::memcpy(&v.u, &f, 4); } break;
                case ACL_INT32:   { int32_t f = (int32_t)e->alpha; std::memcpy(&v.u, &f, 4); } break;
                case ACL_FLOAT16: { float f=(float)e->alpha; uint32_t x; std::memcpy(&x,&f,4);
                                    uint32_t sign=(x>>16)&0x8000; int32_t ex=((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffff; uint16_t h;
                                    if (ex<=0) h=(uint16_t)sign; else if (ex>=31) h=(uint16_t)(sign|0x7c00); else h=(uint16_t)(sign|(ex<<10)|(m>>13));
                                    v.u = h; } break;
                default: { int32_t f=(int32_t)e->alpha; std::memcpy(&v.u,&f,esz<4?esz:4); } break;
            }
            #define L(T) k_pad<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,(T)v.u,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_GATHER: {
            // e->dim = gather dim; e->b = index (1D int64)
            SGather d{}; d.rank = rank; d.gd = (int)e->dim; d.base = a->offset;
            for (int i = 0; i < rank; ++i) { d.od[i] = o->viewDims[i]; d.istr[i] = a->strides[i]; }
            const int64_t *idx = (const int64_t *)e->b->data;
            #define L(T) k_gather<T><<<g,TH,0,s>>>((const T*)pin,idx,(T*)pout,d,n)
            DISP_SZ(esz, L);
            #undef L
            break;
        }
        case OP_CONCAT: {
            // e->inputs = input list; e->dim = concat dim. Each input written into its offset region of out.
            int cd = (int)e->dim;
            int64_t ostr[MAXD]; { int64_t acc = 1; for (int i = rank-1; i>=0; --i){ ostr[i]=acc; acc*=o->viewDims[i]; } }
            int64_t coff = 0;
            for (auto t : e->inputs) {
                SCat d{}; d.rank = rank; d.cd = cd; d.coff = coff;
                for (int i = 0; i < rank; ++i) { d.id[i] = t->viewDims[i]; d.ostr[i] = ostr[i]; }
                int64_t tn = t->numel(); int64_t tg = nb(tn);
                #define L(T) k_cat<T><<<tg,TH,0,s>>>((const T*)t->data,(T*)pout,d,tn)
                DISP_SZ(esz, L);
                #undef L
                coff += t->viewDims[cd];
            }
            break;
        }
        case OP_SPLIT: {
            // e->inputs = output list; e->dim = split dim. Each output is a contiguous slice along cd.
            int cd = (int)e->dim; int64_t coff = 0;
            for (auto t : e->inputs) {
                aclTensor *to = const_cast<aclTensor *>(t);
                SCopy d{}; d.rank = rank; d.base = a->offset + coff * a->strides[cd];
                for (int i = 0; i < rank; ++i) { d.od[i] = to->viewDims[i]; d.istr[i] = a->strides[i]; d.imod[i] = 0; }
                int64_t tn = to->numel(); int64_t tg = nb(tn);
                #define L(T) k_scopy<T><<<tg,TH,0,s>>>((const T*)pin,(T*)to->data,d,tn)
                DISP_SZ(esz, L);
                #undef L
                coff += to->viewDims[cd];
            }
            break;
        }
        default: for (auto *t : e->owned) delete t; delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    cudaError_t err = cudaGetLastError();
    for (auto *t : e->owned) delete t;
    delete e;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

extern "C" {

// Permute (transpose): out[i0..] = self[dims-permuted]
aclnnStatus aclnnPermuteGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims, aclTensor *out,
                                         uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !dims || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if ((int)dims->v.size() != rank || rank > MAXD || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    for (int i = 0; i < rank; ++i) if (dims->v[i] < 0 || dims->v[i] >= rank || self->viewDims[dims->v[i]] != out->viewDims[i]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_PERMUTE; e->a = self; e->out = out; e->axes = dims->v;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPermute(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// Slice: take [start, end) along dim with step
aclnnStatus aclnnSliceGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end, int64_t step,
                                       aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || rank > MAXD || step <= 0 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (start < 0) start += self->viewDims[dim];
    if (end < 0) end += self->viewDims[dim];
    if (end > self->viewDims[dim]) end = self->viewDims[dim];
    if (start < 0 || start >= end) return ACLNN_ERR_PARAM_INVALID;
    int64_t len = (end - start + step - 1) / step;
    if (out->viewDims[dim] != len) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SLICE; e->a = self; e->out = out; e->dim = dim; e->m = start; e->n = end; e->k = step;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSlice(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// StridedSlice: multi-dim begin/end/strides (each of length rank, strides>0). Negative indices normalized by dim size; end clamped to dim size.
aclnnStatus aclnnStridedSliceGetWorkspaceSize(const aclTensor *self, const aclIntArray *begin, const aclIntArray *end,
                                              const aclIntArray *strides, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !begin || !end || !strides || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if ((int)begin->v.size() != rank || (int)end->v.size() != rank || (int)strides->v.size() != rank) return ACLNN_ERR_PARAM_INVALID;
    if (rank > MAXD || !out->contiguous() || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_STRIDED_SLICE; e->a = self; e->out = out; e->axes.resize(2*rank);
    for (int i = 0; i < rank; ++i) {
        int64_t D = self->viewDims[i], b = begin->v[i], en = end->v[i], sp = strides->v[i];
        if (sp <= 0) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        if (b < 0) b += D; if (en < 0) en += D; if (en > D) en = D; if (b < 0) b = 0;
        if (b >= en) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        int64_t len = (en - b + sp - 1) / sp;
        if (out->viewDims[i] != len) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        e->axes[2*i] = b; e->axes[2*i+1] = sp;
    }
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnStridedSlice(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// Tile: repeat each dim repeats[i] times (out dim = in dim * repeats[i])
aclnnStatus aclnnTileGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out,
                                      uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !repeats || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if ((int)repeats->v.size() != rank || rank > MAXD || !out->contiguous() || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    for (int i = 0; i < rank; ++i) if (out->viewDims[i] != self->viewDims[i] * repeats->v[i]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_TILE; e->a = self; e->out = out; e->axes = repeats->v;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTile(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// ConstantPadNd: padding = [lo0,hi0,lo1,hi1,...] (length 2*rank), filled with value
aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, const aclScalar *value,
                                               aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !padding || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if ((int)padding->v.size() != 2 * rank || rank > MAXD || !out->contiguous() || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    for (int i = 0; i < rank; ++i) if (out->viewDims[i] != self->viewDims[i] + padding->v[2*i] + padding->v[2*i+1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_PAD; e->a = self; e->out = out; e->axes = padding->v; e->alpha = value ? value->v : 0.0;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConstantPadNd(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// Gather: select rows along dim by index (1D int64)
aclnnStatus aclnnGatherGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
                                        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || rank > MAXD || !out->contiguous() || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (index->dtype != ACL_INT64 || index->viewDims.size() != 1) return ACLNN_ERR_PARAM_INVALID;
    if (out->viewDims[dim] != index->viewDims[0]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = index; e->out = out; e->dim = dim;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGather(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// GatherV2 (batch_dims): self[B...,A,T...], index[B...,I] → out[B...,I,T...], gathering along axis=batchDims.
// Constraint: axis immediately follows batch dims (axis == batchDims, covering beam/KV reorder and per-batch select).
aclnnStatus aclnnGatherV2GetWorkspaceSize(const aclTensor *self, int64_t batchDims, int64_t axis,
        const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !out || !ws || !ex || !self->data || !index->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || index->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int sr = (int)self->viewDims.size(), ir = (int)index->viewDims.size();
    if (axis < 0) axis += sr;
    if (batchDims < 0 || axis != batchDims || batchDims > sr || batchDims > ir) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !index->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t batchN = 1; for (int i = 0; i < batchDims; ++i) { if (self->viewDims[i] != index->viewDims[i]) return ACLNN_ERR_PARAM_INVALID; batchN *= self->viewDims[i]; }
    int64_t A = self->viewDims[axis];
    int64_t I = 1; for (int i = batchDims; i < ir; ++i) I *= index->viewDims[i];
    int64_t tail = 1; for (int i = axis+1; i < sr; ++i) tail *= self->viewDims[i];
    if (out->numel() != batchN * I * tail) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = index; e->out = out;
    e->outerCount = batchN; e->k = A; e->m = I; e->n = tail; e->dim = -2;   // dim=-2 marks batched gather
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGatherV2(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t batchN=e->outerCount, A=e->k, I=e->m, tail=e->n; size_t esz=dtype_size(e->a->dtype);
    int64_t g=nb(batchN*I*tail); const int64_t *idx=(const int64_t*)e->b->data;
    #define L_(T) k_gather_batched<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,idx,(T*)e->out->data,batchN,A,I,tail)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

// Cat: tensors[0..num) concatenated along dim (PoC: takes raw array, not aclTensorList)
aclnnStatus aclnnCatGetWorkspaceSize(const aclTensor *const *tensors, uint64_t num, int64_t dim,
                                     aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || !num || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)out->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || rank > MAXD || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t sum = 0;
    for (uint64_t i = 0; i < num; ++i) {
        const aclTensor *t = tensors[i];
        if (!t || (int)t->viewDims.size() != rank || t->dtype != out->dtype || !t->contiguous()) return ACLNN_ERR_PARAM_INVALID;
        for (int k = 0; k < rank; ++k) if (k != dim && t->viewDims[k] != out->viewDims[k]) return ACLNN_ERR_PARAM_INVALID;
        sum += t->viewDims[dim];
    }
    if (sum != out->viewDims[dim]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONCAT; e->out = out; e->dim = dim;
    e->a = tensors[0];   // used by run_shape only to get dtype/rank
    for (uint64_t i = 0; i < num; ++i) e->inputs.push_back(tensors[i]);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCat(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// Native aclTensorList interface (aligns with standard CANN aclnnCat signature): unpacks the list and reuses the array path
aclnnStatus aclnnCatListGetWorkspaceSize(const aclTensorList *tensors, int64_t dim,
                                         aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || tensors->v.empty()) return ACLNN_ERR_PARAM_NULLPTR;
    return aclnnCatGetWorkspaceSize(tensors->v.data(), tensors->v.size(), dim, out, ws, ex);
}
aclnnStatus aclnnCatList(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnCat(ws, wsz, e, s); }

// Stack (native aclTensorList): N same-shape tensors stacked along new dim → out has one extra dim of size N.
// Implemented as "insert a unit dim into each input, then Cat along dim": the unit dim copies no data;
// view tensors sharing device data are constructed and attached to e->owned for cleanup on execution.
aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList *tensors, int64_t dim,
                                       aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!tensors || tensors->v.empty() || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    uint64_t num = tensors->v.size();
    int outRank = (int)out->viewDims.size();
    if (dim < 0) dim += outRank;
    if (dim < 0 || dim >= outRank || outRank > MAXD || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (out->viewDims[dim] != (int64_t)num) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *t0 = tensors->v[0];
    int inRank = (int)t0->viewDims.size();
    if (inRank + 1 != outRank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONCAT; e->out = out; e->dim = dim;
    for (uint64_t i = 0; i < num; ++i) {
        const aclTensor *t = tensors->v[i];
        if (!t || (int)t->viewDims.size() != inRank || t->dtype != out->dtype || !t->contiguous())
            { for (auto *o : e->owned) delete o; delete e; return ACLNN_ERR_PARAM_INVALID; }
        // verify t shape matches out with dim removed
        for (int k = 0, ok = 0; k < outRank; ++k) { if (k == dim) continue; if (t->viewDims[ok++] != out->viewDims[k])
            { for (auto *o : e->owned) delete o; delete e; return ACLNN_ERR_PARAM_INVALID; } }
        // construct a view tensor with the unit dim inserted (shares device data)
        auto *v = new aclTensor(*t);
        v->viewDims.insert(v->viewDims.begin() + dim, 1);
        v->strides.assign(outRank, 1);
        { int64_t acc = 1; for (int k = outRank - 1; k >= 0; --k) { v->strides[k] = acc; acc *= v->viewDims[k]; } }
        e->owned.push_back(v); e->inputs.push_back(v);
    }
    e->a = e->inputs[0];
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnStack(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// SplitWithSize: split self along dim according to outputs[i]'s size along that dim
aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, int64_t dim,
                                               const aclTensor *const *outputs, uint64_t num,
                                               uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !outputs || !num || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size();
    if (dim < 0) dim += rank;
    if (dim < 0 || dim >= rank || rank > MAXD) return ACLNN_ERR_PARAM_INVALID;
    int64_t sum = 0;
    for (uint64_t i = 0; i < num; ++i) {
        const aclTensor *t = outputs[i];
        if (!t || (int)t->viewDims.size() != rank || t->dtype != self->dtype || !t->contiguous()) return ACLNN_ERR_PARAM_INVALID;
        for (int k = 0; k < rank; ++k) if (k != dim && t->viewDims[k] != self->viewDims[k]) return ACLNN_ERR_PARAM_INVALID;
        sum += t->viewDims[dim];
    }
    if (sum != self->viewDims[dim]) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SPLIT; e->a = self; e->dim = dim;
    e->out = const_cast<aclTensor *>(outputs[0]);   // placeholder only to avoid null-pointer in shared preamble logic
    for (uint64_t i = 0; i < num; ++i) e->inputs.push_back(outputs[i]);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSplitWithSize(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_shape(e, (cudaStream_t)s); }

// Embedding: weight[V,D] + ids[L] int64 → out[L,D] (dtype-agnostic)
aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor *weight, const aclTensor *ids, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !ids || !out || !ex || !weight->data || !ids->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (ids->dtype != ACL_INT64 || weight->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (weight->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = weight; e->b = ids; e->out = out;
    e->n = weight->viewDims[1]; e->m = ids->numel(); e->dim = -1;   // dim=-1 marks embedding
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEmbedding(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t L = e->m, D = e->n, g = nb(L * D); size_t esz = dtype_size(e->a->dtype);
    const int64_t *ids = (const int64_t *)e->b->data;
    #define L_(T) k_embed<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,ids,(T*)e->out->data,L,D)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// EmbeddingDenseBackward：grad[L,D] fp32 + ids → gradWeight[V,D] fp32（scatter-add）
aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *ids, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!grad || !ids || !gradWeight || !ex || !grad->data || !ids->data || !gradWeight->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (grad->dtype != ACL_FLOAT || gradWeight->dtype != ACL_FLOAT || ids->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (gradWeight->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = grad; e->b = ids; e->out = gradWeight;
    e->n = gradWeight->viewDims[1]; e->m = ids->numel(); e->reduceCount = gradWeight->viewDims[0];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEmbeddingDenseBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t L = e->m, D = e->n, V = e->reduceCount; auto st_=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data, 0, (size_t)V * D * sizeof(float), st_);
    k_embed_bwd<<<nb(L*D),TH,0,st_>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,L,D);
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// Arange: out[i] = start + i*step (out fp32, n = out numel)
aclnnStatus aclnnArangeGetWorkspaceSize(double start, double end, double step, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || !out->data || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->out = out; e->eps = start; e->alpha = step;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnArange(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_arange<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data, n, (float)e->eps, (float)e->alpha);
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// OneHot: ids[L] int64 → out[L,C] fp32 (on/off values)
aclnnStatus aclnnOneHotGetWorkspaceSize(const aclTensor *ids, int64_t numClasses, double onValue, double offValue, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ids || !out || !ex || !ids->data || !out->data || ids->dtype != ACL_INT64 || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = ids; e->out = out; e->n = numClasses; e->m = ids->numel(); e->eps = onValue; e->alpha = offValue;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnOneHot(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t L = e->m, C = e->n; k_onehot<<<nb(L*C),TH,0,(cudaStream_t)s>>>((const int64_t*)e->a->data,(float*)e->out->data,L,C,(float)e->eps,(float)e->alpha);
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// Flip along single dim
aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data || self->dtype != out->dtype) return ACLNN_ERR_PARAM_NULLPTR;
    int rank = (int)self->viewDims.size(); if (dim < 0) dim += rank; if (dim < 0 || dim >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SLICE; e->a = self; e->out = out; e->dim = dim;
    int64_t outer = 1; for (int i = 0; i < dim; ++i) outer *= self->viewDims[i]; int64_t inner = 1; for (int i = dim+1; i < rank; ++i) inner *= self->viewDims[i];
    e->m = outer; e->n = inner; e->k = self->viewDims[dim];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlip(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t outer=e->m,inner=e->n,FD=e->k; size_t esz=dtype_size(e->a->dtype); int64_t g=nb(outer*FD*inner);
    #define L_(T) k_flip<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,outer,FD,inner)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// Roll along single dim by shift
aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t shift, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype) return ACLNN_ERR_PARAM_NULLPTR;
    int rank=(int)self->viewDims.size(); if(dim<0)dim+=rank; if(dim<0||dim>=rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SLICE; e->a = self; e->out = out;
    int64_t outer=1; for(int i=0;i<dim;++i)outer*=self->viewDims[i]; int64_t inner=1; for(int i=dim+1;i<rank;++i)inner*=self->viewDims[i];
    e->m=outer; e->n=inner; e->k=self->viewDims[dim]; e->dim=((shift % e->k)+e->k)%e->k;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRoll(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t outer=e->m,inner=e->n,FD=e->k,sh=e->dim; size_t esz=dtype_size(e->a->dtype); int64_t g=nb(outer*FD*inner);
    #define L_(T) k_roll<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,outer,FD,inner,sh)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// RepeatInterleave (1D, each element repeated repeats times)
aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(const aclTensor *self, int64_t repeats, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->numel() != self->numel() * repeats) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_SLICE; e->a = self; e->out = out; e->k = repeats;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRepeatInterleave(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); size_t esz=dtype_size(e->a->dtype); int64_t g=nb(n);
    #define L_(T) k_repeat_il<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,n,e->k)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// RepeatInterleaveInt: flatten-and-repeat by an integer (same as RepeatInterleave; outputSize ignored)
aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t outputSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)outputSize; return aclnnRepeatInterleaveGetWorkspaceSize(self, repeats, out, ws, ex); }
aclnnStatus aclnnRepeatInterleaveInt(void *w, uint64_t ws, aclOpExecutor *e, aclrtStream s) { return aclnnRepeatInterleave(w, ws, e, s); }
// Repeat: tile by per-dim repeat counts (alias of Tile when repeats length == rank)
aclnnStatus aclnnRepeatGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnTileGetWorkspaceSize(self, repeats, out, ws, ex); }
aclnnStatus aclnnRepeat(void *w, uint64_t ws, aclOpExecutor *e, aclrtStream s) { return aclnnTile(w, ws, e, s); }
// Expand / BroadcastTo: broadcast to out shape (size-1 dims get stride 0)
aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype) return ACLNN_ERR_PARAM_NULLPTR;
    int r = (int)out->viewDims.size(), rs = (int)self->viewDims.size();
    if (rs > r || r > MAXD) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_PERMUTE; e->a = self; e->out = out; e->axes.assign(r, 0);  // axes reused to store istr
    int64_t sstr[MAXD]; { int64_t acc=1; for(int i=rs-1;i>=0;--i){sstr[i]=acc; acc*=self->viewDims[i];} }
    for (int i = 0; i < r; ++i) { int si = i - (r - rs);
        e->axes[i] = (si >= 0 && self->viewDims[si] == out->viewDims[i]) ? sstr[si] : 0; }  // broadcast dims get stride 0
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnExpand(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int rank=(int)e->out->viewDims.size(); int64_t n=e->out->numel(); size_t esz=dtype_size(e->a->dtype);
    SCopy d{}; d.rank=rank; d.base=0; for(int i=0;i<rank;++i){d.od[i]=e->out->viewDims[i]; d.istr[i]=e->axes[i]; d.imod[i]=0;}
    int64_t g=nb(n);
    #define L_(T) k_scopy<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,d,n)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// ScatterUpdate along dim0: out = self (caller pre-copies), out[index[l],:] = src[l,:]
aclnnStatus aclnnScatterUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (index->dtype != ACL_INT64 || self->dtype != out->dtype || src->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = index; e->c = src; e->out = out;
    e->m = index->numel(); e->n = src->numel() / index->numel();   // row length
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterUpdate(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st_=(cudaStream_t)s; int64_t L=e->m, row=e->n; size_t esz=dtype_size(e->a->dtype);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*esz, cudaMemcpyDeviceToDevice, st_);  // out = self
    int64_t g=nb(L*row);
    #define L_(T) k_scatter0<T><<<g,TH,0,st_>>>((const T*)e->c->data,(const int64_t*)e->b->data,(T*)e->out->data,L,row)
    DISP_SZ(esz, L_);
    #undef L_
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// ScatterAdd along dim0: out = self (caller pre-copies), out[index[l],:] += src[l,:] (duplicate indices: atomic accumulate)
aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (index->dtype != ACL_INT64 || self->dtype != out->dtype || src->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    aclDataType dt = self->dtype;
    if (dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16 && dt != ACL_INT32) return ACLNN_ERR_PARAM_INVALID;  // only dtypes that support atomic accumulation
    if (index->numel() == 0 || src->numel() % index->numel() != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = index; e->c = src; e->out = out;
    e->m = index->numel(); e->n = src->numel() / index->numel();   // row length
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterAdd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st_=(cudaStream_t)s; int64_t L=e->m, row=e->n; size_t esz=dtype_size(e->a->dtype);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*esz, cudaMemcpyDeviceToDevice, st_);  // out = self
    int64_t g=nb(L*row); const int64_t *idx=(const int64_t*)e->b->data;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_scatter_add0<float><<<g,TH,0,st_>>>((const float*)e->c->data,idx,(float*)e->out->data,L,row); break;
        case ACL_FLOAT16: k_scatter_add0<__half><<<g,TH,0,st_>>>((const __half*)e->c->data,idx,(__half*)e->out->data,L,row); break;
        case ACL_BF16:    k_scatter_add0<__nv_bfloat16><<<g,TH,0,st_>>>((const __nv_bfloat16*)e->c->data,idx,(__nv_bfloat16*)e->out->data,L,row); break;
        case ACL_INT32:   k_scatter_add0<int32_t><<<g,TH,0,st_>>>((const int32_t*)e->c->data,idx,(int32_t*)e->out->data,L,row); break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// Scatter (dim0) with reduce mode: 0=replace, 1=add. self pre-copied to out.
static aclnnStatus scatter_build(const aclTensor *self, const aclTensor *index, const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !src || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (index->dtype != ACL_INT64 || self->dtype != out->dtype || src->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (index->numel() == 0 || src->numel() % index->numel() != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = index; e->c = src; e->out = out;
    e->m = index->numel(); e->n = src->numel() / index->numel(); e->reduceCount = reduce;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus scatter_apply(aclOpExecutor *e, cudaStream_t st_) {
    int64_t L=e->m, row=e->n; size_t esz=dtype_size(e->a->dtype);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*esz, cudaMemcpyDeviceToDevice, st_);
    int64_t g=nb(L*row); const int64_t *idx=(const int64_t*)e->b->data;
    if (e->reduceCount == 1) {   // add
        switch (e->a->dtype) {
            case ACL_FLOAT:   k_scatter_add0<float><<<g,TH,0,st_>>>((const float*)e->c->data,idx,(float*)e->out->data,L,row); break;
            case ACL_FLOAT16: k_scatter_add0<__half><<<g,TH,0,st_>>>((const __half*)e->c->data,idx,(__half*)e->out->data,L,row); break;
            case ACL_BF16:    k_scatter_add0<__nv_bfloat16><<<g,TH,0,st_>>>((const __nv_bfloat16*)e->c->data,idx,(__nv_bfloat16*)e->out->data,L,row); break;
            case ACL_INT32:   k_scatter_add0<int32_t><<<g,TH,0,st_>>>((const int32_t*)e->c->data,idx,(int32_t*)e->out->data,L,row); break;
            default: delete e; return ACLNN_ERR_PARAM_INVALID;
        }
    } else {   // replace
        #define LR(T) k_scatter0<T><<<g,TH,0,st_>>>((const T*)e->c->data,idx,(T*)e->out->data,L,row)
        DISP_SZ(esz, LR);
        #undef LR
    }
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return scatter_build(self, index, src, reduce, out, ws, ex); }
aclnnStatus aclnnScatter(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scatter_apply(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, uint64_t *ws, aclOpExecutor **ex) { (void)dim; return scatter_build(self, index, src, reduce, self, ws, ex); }
aclnnStatus aclnnInplaceScatter(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scatter_apply(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(aclTensor *self, const aclTensor *index, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex) { return scatter_build(self, index, src, 0, self, ws, ex); }
aclnnStatus aclnnInplaceScatterUpdate(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return scatter_apply(e, (cudaStream_t)s); }

// MaskedSelect: mask-selected elements compacted in order into out (fp32); count written to countOut (int64[1])
aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mask || !out || !countOut || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->b = mask; e->out = out; e->out2 = countOut; e->m = self->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaskedSelect(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_masked_select<<<1,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(const uint8_t*)e->b->data,(float*)e->out->data,(int64_t*)e->out2->data,e->m);
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// Nonzero: flat indices of non-zero elements written in order to out (int64); count written to countOut (int64[1])
aclnnStatus aclnnNonzeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !countOut || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_GATHER; e->a = self; e->out = out; e->out2 = countOut; e->m = self->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnNonzero(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_nonzero<<<1,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(int64_t*)e->out->data,(int64_t*)e->out2->data,e->m);
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

// IndexSelect: select along dim by int64 index — semantics identical to aclnnGather; directly routed
aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGatherGetWorkspaceSize(self, dim, index, out, ws, ex);
}
aclnnStatus aclnnIndexSelect(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream s) { return aclnnGather(ws, wsSize, e, s); }

} // extern "C"
} // namespace _shape
#undef DISP_SZ

namespace _shape_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _shape_ext {
// Shape extensions (P5): Reshape/Squeeze/Unsqueeze (view copies), Movedim, Rot90,
// Reflection/Replication/Circular pad (2D), Im2Col/Col2Im (Unfold/Fold), PixelShuffle/Unshuffle, ChannelShuffle.
// Self-contained: each op builds a minimal executor and runs a dedicated kernel. out is contiguous.

namespace {

constexpr int TH = 256;
constexpr int MAXD = 8;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// General permute/strided gather: out contiguous, in addressed by per-out-dim stride (movedim/rot90 helper)
struct PG { int rank; int64_t od[MAXD], istr[MAXD], base; };
template <typename T> __global__ void k_pg(const T *in, T *o, PG d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t rem = i, off = d.base;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t c = rem % d.od[k]; rem /= d.od[k]; off += c * d.istr[k]; }
    o[i] = in[off];
}
// 2D pad with boundary mode: 0=reflect 1=replicate 2=circular. Layout outer*[H,W], pads l/r/t/b.
template <typename T, int MODE> __global__ void k_pad2d(const T *in, T *o, int64_t outer, int64_t H, int64_t W,
        int64_t oH, int64_t oW, int64_t padT, int64_t padL) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= outer * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, ou = i / (oH * oW);
    int64_t ih = oh - padT, iw = ow - padL;
    auto fix = [](int64_t x, int64_t n) -> int64_t {
        if (MODE == 1) { return x < 0 ? 0 : (x >= n ? n - 1 : x); }            // replicate
        if (MODE == 2) { return ((x % n) + n) % n; }                            // circular
        // reflect (no repeat of border): mirror across edges
        if (n == 1) return 0;
        while (x < 0 || x >= n) { if (x < 0) x = -x; if (x >= n) x = 2 * (n - 1) - x; }
        return x; };
    ih = fix(ih, H); iw = fix(iw, W);
    o[i] = in[(ou * H + ih) * W + iw];
}
// Im2Col: in[N,C,H,W] -> out[N, C*kh*kw, L], L=oH*oW
template <typename T> __global__ void k_im2col(const T *in, T *o, int64_t N, int64_t C, int64_t H, int64_t W,
        int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw, int64_t dh, int64_t dw, int64_t oH, int64_t oW) {
    int64_t L = oH * oW, K = C * kh * kw; int64_t total = N * K * L;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t l = i % L, krow = (i / L) % K, n = i / (L * K);
    int64_t ow = l % oW, oh = l / oW;
    int64_t kj = krow % kw, ki = (krow / kw) % kh, c = krow / (kh * kw);
    int64_t ih = oh * sh - ph + ki * dh, iw = ow * sw - pw + kj * dw;
    T v = (T)0; if (ih >= 0 && ih < H && iw >= 0 && iw < W) v = in[((n * C + c) * H + ih) * W + iw];
    o[i] = v;
}
// Col2Im: out[N,C,H,W] (pre-zeroed) accumulate from cols[N, C*kh*kw, L]
__global__ void k_col2im(const float *cols, float *o, int64_t N, int64_t C, int64_t H, int64_t W,
        int64_t kh, int64_t kw, int64_t sh, int64_t sw, int64_t ph, int64_t pw, int64_t dh, int64_t dw, int64_t oH, int64_t oW) {
    int64_t L = oH * oW, K = C * kh * kw; int64_t total = N * K * L;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t l = i % L, krow = (i / L) % K, n = i / (L * K);
    int64_t ow = l % oW, oh = l / oW;
    int64_t kj = krow % kw, ki = (krow / kw) % kh, c = krow / (kh * kw);
    int64_t ih = oh * sh - ph + ki * dh, iw = ow * sw - pw + kj * dw;
    if (ih >= 0 && ih < H && iw >= 0 && iw < W) atomicAdd(&o[((n * C + c) * H + ih) * W + iw], cols[i]);
}
// PixelShuffle r: in[N, C*r*r, H, W] -> out[N, C, H*r, W*r]
template <typename T> __global__ void k_pixel_shuffle(const T *in, T *o, int64_t N, int64_t C, int64_t H, int64_t W, int64_t r) {
    int64_t oH = H * r, oW = W * r, total = N * C * oH * oW;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, c = (i / (oW * oH)) % C, n = i / (oW * oH * C);
    int64_t h = oh / r, w = ow / r, sh = oh % r, sw = ow % r;
    int64_t ic = c * r * r + sh * r + sw;
    o[i] = in[((n * (C * r * r) + ic) * H + h) * W + w];
}
// PixelUnshuffle r: in[N, C, H*r, W*r] -> out[N, C*r*r, H, W]
template <typename T> __global__ void k_pixel_unshuffle(const T *in, T *o, int64_t N, int64_t C, int64_t H, int64_t W, int64_t r) {
    int64_t iH = H * r, iW = W * r, total = N * C * r * r * H * W;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t w = i % W, h = (i / W) % H, oc = (i / (W * H)) % (C * r * r), n = i / (W * H * C * r * r);
    int64_t c = oc / (r * r), sh = (oc / r) % r, sw = oc % r;
    int64_t ih = h * r + sh, iw = w * r + sw;
    o[i] = in[((n * C + c) * iH + ih) * iW + iw];
}
// Rot90 (last two dims), k times counter-clockwise
template <typename T> __global__ void k_rot90(const T *in, T *o, int64_t outer, int64_t H, int64_t W, int64_t oH, int64_t oW, int kk) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= outer * oH * oW) return;
    int64_t ow = i % oW, oh = (i / oW) % oH, ou = i / (oH * oW);
    int64_t ih, iw;
    if (kk == 1) { ih = ow; iw = W - 1 - oh; }            // CCW
    else if (kk == 2) { ih = H - 1 - oh; iw = W - 1 - ow; }
    else { ih = H - 1 - ow; iw = oh; }                    // kk==3
    o[i] = in[(ou * H + ih) * W + iw];
}
// ChannelShuffle groups g: in[N,C,HW] -> out, channel c maps from (c%g)*(C/g) + c/g
template <typename T> __global__ void k_channel_shuffle(const T *in, T *o, int64_t N, int64_t C, int64_t HW, int64_t g) {
    int64_t total = N * C * HW; int64_t cpg = C / g;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t hw = i % HW, c = (i / HW) % C, n = i / (HW * C);
    int64_t src_c = (c % g) * cpg + c / g;
    o[i] = in[(n * C + src_c) * HW + hw];
}

#define DISP_SZ(esz, LAUNCH) do { switch (esz) { \
    case 1: { LAUNCH(uint8_t);  } break; case 2: { LAUNCH(uint16_t); } break; \
    case 4: { LAUNCH(uint32_t); } break; case 8: { LAUNCH(uint64_t); } break; \
    default: return ACLNN_ERR_PARAM_INVALID; } } while (0)

inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// ---- Reshape / Squeeze / Unsqueeze: contiguous data copy (same numel) ----
static aclnnStatus view_ws(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->numel() != out->numel() || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->m = self->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus view_run(aclOpExecutor *e, cudaStream_t s) {
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->m * dtype_size(e->a->dtype), cudaMemcpyDeviceToDevice, s);
    return done(e);
}
aclnnStatus aclnnReshapeGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return view_ws(self, out, ws, ex); }
aclnnStatus aclnnReshape(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return view_run(e, (cudaStream_t)s); }
aclnnStatus aclnnSqueezeGetWorkspaceSize(const aclTensor *self, int64_t /*dim*/, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return view_ws(self, out, ws, ex); }
aclnnStatus aclnnSqueeze(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return view_run(e, (cudaStream_t)s); }
aclnnStatus aclnnUnsqueezeGetWorkspaceSize(const aclTensor *self, int64_t /*dim*/, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return view_ws(self, out, ws, ex); }
aclnnStatus aclnnUnsqueeze(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return view_run(e, (cudaStream_t)s); }

// ---- Movedim: move dim `src` to position `dst` (permute) ----
aclnnStatus aclnnMovedimGetWorkspaceSize(const aclTensor *self, int64_t src, int64_t dst, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size(); if (rank > MAXD) return ACLNN_ERR_PARAM_INVALID;
    if (src < 0) src += rank; if (dst < 0) dst += rank;
    if (src < 0 || src >= rank || dst < 0 || dst >= rank) return ACLNN_ERR_PARAM_INVALID;
    std::vector<int> perm; for (int i = 0; i < rank; ++i) if (i != src) perm.push_back(i);
    perm.insert(perm.begin() + dst, (int)src);
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out;
    for (int i = 0; i < rank; ++i) e->axes.push_back(perm[i]);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMovedim(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int rank = (int)e->a->viewDims.size(); int64_t istr[MAXD], acc = 1;
    for (int i = rank - 1; i >= 0; --i) { istr[i] = acc; acc *= e->a->viewDims[i]; }
    PG d{}; d.rank = rank; d.base = 0;
    for (int i = 0; i < rank; ++i) { int p = (int)e->axes[i]; d.od[i] = e->a->viewDims[p]; d.istr[i] = istr[p]; }
    int64_t n = e->out->numel(); size_t esz = dtype_size(e->a->dtype); int64_t g = nb(n);
    #define L(T) k_pg<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,d,n)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}

// ---- Rot90 (last two dims), k times counter-clockwise ----
aclnnStatus aclnnRot90GetWorkspaceSize(const aclTensor *self, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() < 2 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->k = ((k % 4) + 4) % 4;
    int rank = (int)self->viewDims.size(); e->m = self->viewDims[rank-2]; e->n = self->viewDims[rank-1];
    e->outerCount = self->numel() / (e->m * e->n);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRot90(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t outer = e->outerCount, H = e->m, W = e->n; int kk = (int)e->k;
    int64_t oH = (kk % 2) ? W : H, oW = (kk % 2) ? H : W;
    if (kk == 0) return view_run(e, (cudaStream_t)s);
    size_t esz = dtype_size(e->a->dtype); int64_t g = nb(outer * oH * oW);
    #define L(T) k_rot90<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,outer,H,W,oH,oW,kk)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}

// ---- 2D boundary pad (Reflection/Replication/Circular), padding=[left,right,top,bottom] ----
static aclnnStatus pad2d_ws(int mode, const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !padding || !out || !ex || self->dtype != out->dtype || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (padding->v.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size(); if (rank < 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = mode; e->a = self; e->out = out;
    e->m = self->viewDims[rank-2]; e->n = self->viewDims[rank-1];
    e->outerCount = self->numel() / (e->m * e->n);
    e->dscalars = { (double)padding->v[0], (double)padding->v[1], (double)padding->v[2], (double)padding->v[3] };  // l,r,t,b
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnReflectionPad2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pad2d_ws(0, self, padding, out, ws, ex); }
aclnnStatus aclnnReplicationPad2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pad2d_ws(1, self, padding, out, ws, ex); }
aclnnStatus aclnnCircularPad2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pad2d_ws(2, self, padding, out, ws, ex); }
static aclnnStatus pad2d_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t outer = e->outerCount, H = e->m, W = e->n;
    int64_t padL = (int64_t)e->dscalars[0], padR = (int64_t)e->dscalars[1], padT = (int64_t)e->dscalars[2], padB = (int64_t)e->dscalars[3];
    int64_t oH = H + padT + padB, oW = W + padL + padR; size_t esz = dtype_size(e->a->dtype); int64_t g = nb(outer * oH * oW);
    #define L(T) do { if (e->op==0) k_pad2d<T,0><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,H,W,oH,oW,padT,padL); \
                      else if (e->op==1) k_pad2d<T,1><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,H,W,oH,oW,padT,padL); \
                      else k_pad2d<T,2><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,H,W,oH,oW,padT,padL); } while(0)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
aclnnStatus aclnnReflectionPad2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pad2d_run(e, (cudaStream_t)s); }
aclnnStatus aclnnReplicationPad2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pad2d_run(e, (cudaStream_t)s); }
aclnnStatus aclnnCircularPad2d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pad2d_run(e, (cudaStream_t)s); }

// ---- Im2Col / Col2Im ----
aclnnStatus aclnnIm2ColGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *dilation,
        const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !kernel || !stride || !padding || !dilation || !out || !ex || self->viewDims.size() != 4 || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out;
    e->axes = { kernel->v[0], kernel->v[1], stride->v[0], stride->v[1], padding->v[0], padding->v[1], dilation->v[0], dilation->v[1] };
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnIm2Col(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; int64_t N = a->viewDims[0], C = a->viewDims[1], H = a->viewDims[2], W = a->viewDims[3];
    int64_t kh=e->axes[0],kw=e->axes[1],sh=e->axes[2],sw=e->axes[3],ph=e->axes[4],pw=e->axes[5],dh=e->axes[6],dw=e->axes[7];
    int64_t oH=(H+2*ph-dh*(kh-1)-1)/sh+1, oW=(W+2*pw-dw*(kw-1)-1)/sw+1;
    int64_t total = N * C * kh * kw * oH * oW; size_t esz = dtype_size(a->dtype); int64_t g = nb(total);
    #define L(T) k_im2col<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)a->data,(T*)e->out->data,N,C,H,W,kh,kw,sh,sw,ph,pw,dh,dw,oH,oW)
    switch (esz) { case 2: L(__half); break; case 4: L(float); break; default: delete e; return ACLNN_ERR_PARAM_INVALID; }
    #undef L
    return done(e);
}
aclnnStatus aclnnCol2ImGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, const aclIntArray *kernel,
        const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !outputSize || !kernel || !stride || !padding || !dilation || !out || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out;
    e->axes = { kernel->v[0], kernel->v[1], stride->v[0], stride->v[1], padding->v[0], padding->v[1], dilation->v[0], dilation->v[1] };
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCol2Im(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st = (cudaStream_t)s; aclTensor *o = e->out; int64_t N=o->viewDims[0],C=o->viewDims[1],H=o->viewDims[2],W=o->viewDims[3];
    int64_t kh=e->axes[0],kw=e->axes[1],sh=e->axes[2],sw=e->axes[3],ph=e->axes[4],pw=e->axes[5],dh=e->axes[6],dw=e->axes[7];
    int64_t oH=(H+2*ph-dh*(kh-1)-1)/sh+1, oW=(W+2*pw-dw*(kw-1)-1)/sw+1;
    cudaMemsetAsync(o->data, 0, (size_t)o->numel()*sizeof(float), st);
    int64_t total = N * C * kh * kw * oH * oW;
    k_col2im<<<nb(total),TH,0,st>>>((const float*)e->a->data,(float*)o->data,N,C,H,W,kh,kw,sh,sw,ph,pw,dh,dw,oH,oW);
    return done(e);
}

// ---- PixelShuffle / PixelUnshuffle / ChannelShuffle ----
aclnnStatus aclnnPixelShuffleGetWorkspaceSize(const aclTensor *self, int64_t upscale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->k = upscale;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPixelShuffle(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; int64_t r = e->k, N=a->viewDims[0], Crr=a->viewDims[1], H=a->viewDims[2], W=a->viewDims[3], C=Crr/(r*r);
    int64_t total = N*C*H*r*W*r; size_t esz = dtype_size(a->dtype); int64_t g = nb(total);
    #define L(T) k_pixel_shuffle<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)a->data,(T*)e->out->data,N,C,H,W,r)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
aclnnStatus aclnnPixelUnshuffleGetWorkspaceSize(const aclTensor *self, int64_t downscale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->k = downscale;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPixelUnshuffle(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    const aclTensor *a = e->a; int64_t r = e->k, N=a->viewDims[0], C=a->viewDims[1], iH=a->viewDims[2], iW=a->viewDims[3], H=iH/r, W=iW/r;
    int64_t total = N*C*r*r*H*W; size_t esz = dtype_size(a->dtype); int64_t g = nb(total);
    #define L(T) k_pixel_unshuffle<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)a->data,(T*)e->out->data,N,C,H,W,r)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
aclnnStatus aclnnChannelShuffleGetWorkspaceSize(const aclTensor *self, int64_t groups, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims.size() < 2 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->k = groups;
    e->m = self->viewDims[1]; e->n = self->numel() / (self->viewDims[0] * self->viewDims[1]); e->outerCount = self->viewDims[0];
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnChannelShuffle(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N = e->outerCount, C = e->m, HW = e->n, g = e->k; size_t esz = dtype_size(e->a->dtype); int64_t total = N*C*HW, gg = nb(total);
    #define L(T) k_channel_shuffle<T><<<gg,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,N,C,HW,g)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}

} // extern "C"
} // namespace _shape_ext
#undef DISP_SZ

namespace _shape2_ext {
// Shape composites remainder (R1): Reflection/Replication/Circular pad 1d & 3d, Meshgrid, Chunk, Hstack/Vstack/Dstack.
// Boundary pads reuse the reflect/replicate/circular index fix; stack/chunk wrap the existing Cat/Split paths.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__device__ inline int64_t bfix(int64_t x, int64_t n, int mode){          // 0=reflect 1=replicate 2=circular
    if (mode==1) return x<0?0:(x>=n?n-1:x);
    if (mode==2) return ((x%n)+n)%n;
    if (n==1) return 0; while (x<0||x>=n){ if(x<0)x=-x; if(x>=n)x=2*(n-1)-x; } return x;
}
template <typename T,int MODE> __global__ void k_pad1d(const T*in,T*o,int64_t outer,int64_t W,int64_t oW,int64_t padL){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=outer*oW) return; int64_t ow=i%oW, ou=i/oW;
    o[i]=in[ou*W + bfix(ow-padL,W,MODE)];
}
template <typename T,int MODE> __global__ void k_pad3d(const T*in,T*o,int64_t outer,int64_t Dd,int64_t H,int64_t W,
        int64_t oD,int64_t oH,int64_t oW,int64_t pD,int64_t pH,int64_t pW){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=outer*oD*oH*oW) return;
    int64_t ow=i%oW, oh=(i/oW)%oH, od=(i/(oW*oH))%oD, ou=i/(oW*oH*oD);
    int64_t id=bfix(od-pD,Dd,MODE), ih=bfix(oh-pH,H,MODE), iw=bfix(ow-pW,W,MODE);
    o[i]=in[((ou*Dd+id)*H+ih)*W+iw];
}
// Meshgrid (ij indexing): inputs t_k length n_k -> out_k[n_0,...,n_{N-1}] = t_k broadcast on dim k
__global__ void k_meshgrid_dim(const float*src,float*o,int64_t total,int64_t inner,int64_t len){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=total) return; o[i]=src[(i/inner)%len];
}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
#define DISP_SZ(esz,L) do{switch(esz){case 1:{L(uint8_t);}break;case 2:{L(uint16_t);}break;case 4:{L(uint32_t);}break;case 8:{L(uint64_t);}break;default:return ACLNN_ERR_PARAM_INVALID;}}while(0)
} // namespace

extern "C" {

// ---- 1D boundary pad: self[...,W], padding={left,right} ----
static aclnnStatus pad1d_ws(int mode,const aclTensor*self,const aclIntArray*padding,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!padding||!out||!ex||self->dtype!=out->dtype||padding->v.size()!=2||self->viewDims.size()<1) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)self->viewDims.size(); auto*e=new aclOpExecutor(); e->op=mode; e->a=self; e->out=out;
    e->n=self->viewDims[rank-1]; e->outerCount=self->numel()/e->n; e->dscalars={(double)padding->v[0],(double)padding->v[1]};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnReflectionPad1dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad1d_ws(0,self,p,out,ws,ex);}
aclnnStatus aclnnReplicationPad1dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad1d_ws(1,self,p,out,ws,ex);}
aclnnStatus aclnnCircularPad1dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad1d_ws(2,self,p,out,ws,ex);}
static aclnnStatus pad1d_run(aclOpExecutor*e,cudaStream_t s){ int64_t outer=e->outerCount,W=e->n,padL=(int64_t)e->dscalars[0],padR=(int64_t)e->dscalars[1],oW=W+padL+padR; int64_t g=nb(outer*oW);
    #define L(T) do{ if(e->op==0)k_pad1d<T,0><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,W,oW,padL); else if(e->op==1)k_pad1d<T,1><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,W,oW,padL); else k_pad1d<T,2><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,W,oW,padL);}while(0)
    DISP_SZ(dtype_size(e->a->dtype),L);
    #undef L
    return done(e); }
aclnnStatus aclnnReflectionPad1d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad1d_run(e,(cudaStream_t)s);}
aclnnStatus aclnnReplicationPad1d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad1d_run(e,(cudaStream_t)s);}
aclnnStatus aclnnCircularPad1d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad1d_run(e,(cudaStream_t)s);}

// ---- 3D boundary pad: self[...,D,H,W], padding={l,r,t,b,front,back} (W,H,D order) ----
static aclnnStatus pad3d_ws(int mode,const aclTensor*self,const aclIntArray*padding,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!self||!padding||!out||!ex||self->dtype!=out->dtype||padding->v.size()!=6||self->viewDims.size()<3) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)self->viewDims.size(); auto*e=new aclOpExecutor(); e->op=mode; e->a=self; e->out=out;
    e->k=self->viewDims[rank-3]; e->m=self->viewDims[rank-2]; e->n=self->viewDims[rank-1]; e->outerCount=self->numel()/(e->k*e->m*e->n);
    e->dscalars={(double)padding->v[0],(double)padding->v[1],(double)padding->v[2],(double)padding->v[3],(double)padding->v[4],(double)padding->v[5]};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnReflectionPad3dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad3d_ws(0,self,p,out,ws,ex);}
aclnnStatus aclnnReplicationPad3dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad3d_ws(1,self,p,out,ws,ex);}
aclnnStatus aclnnCircularPad3dGetWorkspaceSize(const aclTensor*self,const aclIntArray*p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){return pad3d_ws(2,self,p,out,ws,ex);}
static aclnnStatus pad3d_run(aclOpExecutor*e,cudaStream_t s){ int64_t outer=e->outerCount,Dd=e->k,H=e->m,W=e->n;
    int64_t pW=(int64_t)e->dscalars[0],pH=(int64_t)e->dscalars[2],pD=(int64_t)e->dscalars[4];
    int64_t oD=Dd+pD+(int64_t)e->dscalars[5], oH=H+pH+(int64_t)e->dscalars[3], oW=W+pW+(int64_t)e->dscalars[1]; int64_t g=nb(outer*oD*oH*oW);
    #define L(T) do{ if(e->op==0)k_pad3d<T,0><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,Dd,H,W,oD,oH,oW,pD,pH,pW); else if(e->op==1)k_pad3d<T,1><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,Dd,H,W,oD,oH,oW,pD,pH,pW); else k_pad3d<T,2><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,outer,Dd,H,W,oD,oH,oW,pD,pH,pW);}while(0)
    DISP_SZ(dtype_size(e->a->dtype),L);
    #undef L
    return done(e); }
aclnnStatus aclnnReflectionPad3d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad3d_run(e,(cudaStream_t)s);}
aclnnStatus aclnnReplicationPad3d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad3d_run(e,(cudaStream_t)s);}
aclnnStatus aclnnCircularPad3d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){return pad3d_run(e,(cudaStream_t)s);}

// ---- Meshgrid (ij): tensors (1D, lengths n_k) -> outs (same N-D shape), out_k broadcasts t_k on axis k ----
aclnnStatus aclnnMeshgridGetWorkspaceSize(const aclTensorList*tensors,const aclTensorList*outs,uint64_t*ws,aclOpExecutor**ex){
    if(!tensors||!outs||!ex||tensors->v.size()!=outs->v.size()||tensors->v.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->op=0; for(auto*t:tensors->v) e->inputs.push_back(t); for(auto*t:outs->v) e->owned.push_back(const_cast<aclTensor*>(t));
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMeshgrid(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int N=(int)e->inputs.size(); int64_t total=1; std::vector<int64_t> lens(N); for(int k=0;k<N;k++){ lens[k]=e->inputs[k]->numel(); total*=lens[k]; }
    for(int k=0;k<N;k++){ int64_t inner=1; for(int j=k+1;j<N;j++) inner*=lens[j];
        k_meshgrid_dim<<<nb(total),TH,0,(cudaStream_t)s>>>((const float*)e->inputs[k]->data,(float*)e->owned[k]->data,total,inner,lens[k]); }
    aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; e->owned.clear(); delete e; return st;
}

// ---- Chunk (equal split along dim) / Hstack(dim1) / Vstack(dim0) / Dstack(dim2): wrap existing Split/Cat ----
aclnnStatus aclnnChunkGetWorkspaceSize(const aclTensor*self,int64_t /*chunks*/,int64_t dim,const aclTensor*const*outputs,uint64_t num,uint64_t*ws,aclOpExecutor**ex){
    return aclnnSplitWithSizeGetWorkspaceSize(self,dim,outputs,num,ws,ex);
}
aclnnStatus aclnnChunk(void*ws,uint64_t wsz,aclOpExecutor*e,aclrtStream s){ return aclnnSplitWithSize(ws,wsz,e,s); }
aclnnStatus aclnnVstackGetWorkspaceSize(const aclTensor*const*t,uint64_t num,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){ return aclnnCatGetWorkspaceSize(t,num,0,out,ws,ex); }
aclnnStatus aclnnVstack(void*ws,uint64_t wsz,aclOpExecutor*e,aclrtStream s){ return aclnnCat(ws,wsz,e,s); }
aclnnStatus aclnnHstackGetWorkspaceSize(const aclTensor*const*t,uint64_t num,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){ return aclnnCatGetWorkspaceSize(t,num,1,out,ws,ex); }
aclnnStatus aclnnHstack(void*ws,uint64_t wsz,aclOpExecutor*e,aclrtStream s){ return aclnnCat(ws,wsz,e,s); }
aclnnStatus aclnnDstackGetWorkspaceSize(const aclTensor*const*t,uint64_t num,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){ return aclnnCatGetWorkspaceSize(t,num,2,out,ws,ex); }
aclnnStatus aclnnDstack(void*ws,uint64_t wsz,aclOpExecutor*e,aclrtStream s){ return aclnnCat(ws,wsz,e,s); }

} // extern "C"
} // namespace _shape2_ext
#undef DISP_SZ

namespace _pad_bwd_ext {
// Pad backward (reflection / replication / circular, 1d/2d/3d), fp32. gradOut (padded) → gradIn (original),
// scatter-add via the per-mode index mapping. padding = [W_l,W_r, H_l,H_r, D_l,D_r] (innermost dim first).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
// map padded output coord o (input size I, left pad lp) to source coord per mode; returns -1 to drop (constant interior only)
__device__ inline int padsrc(int o, int I, int lp, int mode) {
    int j = o - lp;
    if (mode==0){ if(j<0) j=-j; if(j>=I) j=2*(I-1)-j; return j; }          // reflect (101)
    if (mode==1){ return j<0?0:(j>=I?I-1:j); }                              // replicate
    if (mode==2){ j%=I; if(j<0) j+=I; return j; }                          // circular
    return (j>=0&&j<I)?j:-1;                                               // constant (interior only)
}
__global__ void k_pad_bwd(const float *gOut, float *gIn, int64_t NC, int i0,int i1,int i2, int o0,int o1,int o2,
                          int l0,int l1,int l2, int mode) {
    int64_t osp=(int64_t)o0*o1*o2, isp=(int64_t)i0*i1*i2;
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx>=NC*osp) return;
    int64_t nc=idx/osp, r=idx%osp; int c2=r%o2,c1=(r/o2)%o1,c0=r/(o2*o1);
    int s0=padsrc(c0,i0,l0,mode), s1=padsrc(c1,i1,l1,mode), s2=padsrc(c2,i2,l2,mode);
    if (s0<0||s1<0||s2<0) return;
    atomicAdd(&gIn[nc*isp + ((int64_t)s0*i1+s1)*i2 + s2], gOut[idx]);
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

static aclnnStatus padb_ws(const aclTensor *gradOut, const aclIntArray *padding, int nsp, int mode, aclTensor *gradIn, aclOpExecutor **ex) {
    if (!gradOut || !padding || !gradIn || !ex || gradOut->dtype != ACL_FLOAT || gradIn->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->out=gradIn; e->dim=mode; e->reduceCount=nsp; e->axes=padding->v; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus padb_run(aclOpExecutor *e, cudaStream_t s) {
    int nsp=(int)e->reduceCount; const auto &O=e->a->viewDims,&I=e->out->viewDims; int rank=(int)I.size(), sp0=rank-nsp;
    int is[3]={1,1,1},os[3]={1,1,1},lp[3]={0,0,0};
    for (int d=0;d<nsp;d++){ int slot=3-nsp+d; is[slot]=(int)I[sp0+d]; os[slot]=(int)O[sp0+d]; }
    // padding [W_l,W_r,H_l,H_r,D_l,D_r] → innermost(slot2)=pad[0], slot1=pad[2], slot0=pad[4]
    for (int d=0;d<nsp;d++){ int slot=2-d; if ((size_t)(2*d) < e->axes.size()) lp[slot]=(int)e->axes[2*d]; }
    int64_t NC=1; for (int d=0;d<sp0;d++) NC*=I[d];
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),s);
    int64_t g=nb(NC*(int64_t)os[0]*os[1]*os[2]);
    k_pad_bwd<<<g,TH,0,s>>>((const float*)e->a->data,(float*)e->out->data,NC,is[0],is[1],is[2],os[0],os[1],os[2],lp[0],lp[1],lp[2],e->dim);
    return fin(e);
}
} // namespace

extern "C" {
#define PADB(NAME, NSP, MODE) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { (void)self; if(ws)*ws=0; return padb_ws(gradOutput, padding, NSP, MODE, gradInput, ex); } \
aclnnStatus NAME(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return padb_run(e, (cudaStream_t)s); }
PADB(aclnnReflectionPad1dBackward,1,0)
PADB(aclnnReflectionPad2dBackward,2,0)
PADB(aclnnReflectionPad3dBackward,3,0)
PADB(aclnnReplicationPad1dBackward,1,1)
PADB(aclnnReplicationPad2dBackward,2,1)
PADB(aclnnReplicationPad3dBackward,3,1)
PADB(aclnnCircularPad2dBackward,2,2)
PADB(aclnnCircularPad3dBackward,3,2)
} // extern "C"
} // namespace _pad_bwd_ext
#undef PADB

} // namespace _shape_ext

