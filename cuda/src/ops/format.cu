// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cstring>
#include <vector>

namespace _format {
// Ascend FRACTAL_NZ fractal format ↔ ND conversion layer (2D, 16×16 fractal tiles).
//   ND[M,N] → NZ[N1,M1,16,16] (N1=ceil(N/16), M1=ceil(M/16), zero-padded as needed):
//     element (i,j) → NZ offset ((j1*M1+i1)*16+i0)*16+j0, i1=i/16, i0=i%16, j1=j/16, j0=j%16.
//   Moved generically as uintN by dtype byte width (1/2/4/8); layout is independent of numeric value.
// GPUs have no device-specific layout concept: when receiving an NZ-format tensor, convert to ND
// before the op and back afterward (this file provides the conversion primitives).

namespace {
constexpr int F = 16, TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

template <typename T>
__global__ void k_nd2nz(const T *nd, T *nz, int64_t M, int64_t N, int64_t M1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int64_t i = idx / N, j = idx % N;
    int64_t i1 = i / F, i0 = i % F, j1 = j / F, j0 = j % F;
    nz[((j1 * M1 + i1) * F + i0) * F + j0] = nd[idx];
}
template <typename T>
__global__ void k_nz2nd(const T *nz, T *nd, int64_t M, int64_t N, int64_t M1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    int64_t i = idx / N, j = idx % N;
    int64_t i1 = i / F, i0 = i % F, j1 = j / F, j0 = j % F;
    nd[idx] = nz[((j1 * M1 + i1) * F + i0) * F + j0];
}

// NCHW[N,C,H,W] ↔ NC1HWC0[N,C1,H,W,16] (C1=ceil(C/16), C0=16, padded channels zeroed)
template <typename T>
__global__ void k_nchw2_5hd(const T *in, T *out, int64_t N, int64_t C, int64_t H, int64_t W, int64_t C1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N*C*H*W) return;
    int64_t w = idx % W, h = (idx/W)%H, c = (idx/(W*H))%C, n = idx/(W*H*C);
    int64_t c1 = c/F, c0 = c%F;
    out[(((((n*C1+c1)*H+h)*W+w))*F)+c0] = in[idx];
}
template <typename T>
__global__ void k_5hd2_nchw(const T *in, T *out, int64_t N, int64_t C, int64_t H, int64_t W, int64_t C1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= N*C*H*W) return;
    int64_t w = idx % W, h = (idx/W)%H, c = (idx/(W*H))%C, n = idx/(W*H*C);
    int64_t c1 = c/F, c0 = c%F;
    out[idx] = in[(((((n*C1+c1)*H+h)*W+w))*F)+c0];
}
// ND[K,N] ↔ FRACTAL_Z[K1,N1,16,16] (K tiles outer, N tiles next, inner (k0,n0); zero-padded)
template <typename T>
__global__ void k_nd2fz(const T *nd, T *fz, int64_t K, int64_t N, int64_t N1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= K*N) return;
    int64_t k = idx / N, n = idx % N, k1 = k/F, k0 = k%F, n1 = n/F, n0 = n%F;
    fz[((k1*N1+n1)*F+k0)*F+n0] = nd[idx];
}
template <typename T>
__global__ void k_fz2nd(const T *fz, T *nd, int64_t K, int64_t N, int64_t N1) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (idx >= K*N) return;
    int64_t k = idx / N, n = idx % N, k1 = k/F, k0 = k%F, n1 = n/F, n0 = n%F;
    nd[idx] = fz[((k1*N1+n1)*F+k0)*F+n0];
}

#define NZ_DISP(esz, LAUNCH) do { switch (esz) {        \
    case 1: { LAUNCH(uint8_t);  } break;                \
    case 2: { LAUNCH(uint16_t); } break;                \
    case 4: { LAUNCH(uint32_t); } break;                \
    case 8: { LAUNCH(uint64_t); } break;                \
    default: delete e; return ACLNN_ERR_PARAM_INVALID; } } while (0)
} // namespace

extern "C" {

// ND[M,N] → NZ[N1,M1,16,16] (out must be allocated for this shape/capacity; padding zeroed)
aclnnStatus aclnnTransDataND2NZGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims.size() != 2 || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->m = self->viewDims[0]; e->n = self->viewDims[1]; e->dim = 0;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataND2NZ(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t M = e->m, N = e->n, M1 = (M + F - 1) / F, N1 = (N + F - 1) / F;
    size_t esz = dtype_size(e->a->dtype);
    cudaMemsetAsync(e->out->data, 0, (size_t)N1 * M1 * F * F * esz, (cudaStream_t)s);   // zero padding
    int64_t g = nb(M * N);
    #define L(T) k_nd2nz<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,M,N,M1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

// NZ[N1,M1,16,16] → ND[M,N] (out shape [M,N]; M and N taken from out)
aclnnStatus aclnnTransDataNZ2NDGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->viewDims.size() != 2 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->m = out->viewDims[0]; e->n = out->viewDims[1]; e->dim = 1;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataNZ2ND(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t M = e->m, N = e->n, M1 = (M + F - 1) / F;
    size_t esz = dtype_size(e->out->dtype);
    int64_t g = nb(M * N);
    #define L(T) k_nz2nd<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,M,N,M1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

// NCHW[N,C,H,W] → NC1HWC0[N,C1,H,W,16] (out must be allocated for 5HD capacity; padding zeroed)
aclnnStatus aclnnTransDataNCHW2NC1HWC0GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims.size() != 4 || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->axes.assign(self->viewDims.begin(), self->viewDims.end());   // [N,C,H,W]
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataNCHW2NC1HWC0(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N=e->axes[0],C=e->axes[1],H=e->axes[2],W=e->axes[3],C1=(C+F-1)/F;
    size_t esz = dtype_size(e->a->dtype);
    cudaMemsetAsync(e->out->data, 0, (size_t)N*C1*H*W*F*esz, (cudaStream_t)s);   // zero padding
    int64_t g = nb(N*C*H*W);
    #define L(T) k_nchw2_5hd<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,N,C,H,W,C1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// NC1HWC0 → NCHW (out shape [N,C,H,W])
aclnnStatus aclnnTransDataNC1HWC0toNCHWGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->viewDims.size() != 4 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->axes.assign(out->viewDims.begin(), out->viewDims.end());
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataNC1HWC0toNCHW(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N=e->axes[0],C=e->axes[1],H=e->axes[2],W=e->axes[3],C1=(C+F-1)/F;
    size_t esz = dtype_size(e->out->dtype);
    int64_t g = nb(N*C*H*W);
    #define L(T) k_5hd2_nchw<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,N,C,H,W,C1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

// ND[K,N] → FRACTAL_Z[K1,N1,16,16] (out must be allocated for this capacity; padding zeroed)
aclnnStatus aclnnTransDataND2FZGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims.size() != 2 || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->m = self->viewDims[0]; e->n = self->viewDims[1];
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataND2FZ(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t K=e->m, N=e->n, K1=(K+F-1)/F, N1=(N+F-1)/F;
    size_t esz = dtype_size(e->a->dtype);
    cudaMemsetAsync(e->out->data, 0, (size_t)K1*N1*F*F*esz, (cudaStream_t)s);
    int64_t g = nb(K*N);
    #define L(T) k_nd2fz<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,K,N,N1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// FRACTAL_Z → ND[K,N] (out shape [K,N])
aclnnStatus aclnnTransDataFZ2NDGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->viewDims.size() != 2 || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out;
    e->m = out->viewDims[0]; e->n = out->viewDims[1];
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransDataFZ2ND(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t K=e->m, N=e->n, N1=(N+F-1)/F;
    size_t esz = dtype_size(e->out->dtype);
    int64_t g = nb(K*N);
    #define L(T) k_fz2nd<T><<<g,TH,0,(cudaStream_t)s>>>((const T*)e->a->data,(T*)e->out->data,K,N,N1)
    NZ_DISP(esz, L);
    #undef L
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

} // extern "C"
} // namespace _format
#undef NZ_DISP

namespace _format_ext {
// Format / data-movement extensions (P19): Contiguous, AsStrided, ViewCopy, Copy, Identity.
// Materialize strided/aliased views into a contiguous output; dtype-agnostic by byte width.

namespace {
constexpr int TH = 256;
constexpr int MAXD = 8;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// Gather out[i] (contiguous) from in addressed by per-out-dim stride + base offset
struct SG { int rank; int64_t od[MAXD], istr[MAXD], base; };
template <typename T> __global__ void k_gather_strided(const T *in, T *o, SG d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t rem = i, off = d.base;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t c = rem % d.od[k]; rem /= d.od[k]; off += c * d.istr[k]; }
    o[i] = in[off];
}
inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; for (auto *t : e->owned) delete t; delete e; return st; }
#define DISP_SZ(esz, LAUNCH) do { switch (esz) { \
    case 1: { LAUNCH(uint8_t);  } break; case 2: { LAUNCH(uint16_t); } break; \
    case 4: { LAUNCH(uint32_t); } break; case 8: { LAUNCH(uint64_t); } break; \
    default: return ACLNN_ERR_PARAM_INVALID; } } while (0)

// Run a strided gather using the input tensor's own viewDims/strides/offset (Contiguous / Copy / ViewCopy / Identity)
static aclnnStatus run_view(aclOpExecutor *e, cudaStream_t s) {
    const aclTensor *a = e->a; aclTensor *o = e->out; size_t esz = dtype_size(a->dtype);
    int rank = (int)a->viewDims.size(); int64_t n = o->numel();
    SG d{}; d.rank = rank; d.base = a->offset;
    for (int i = 0; i < rank; ++i) { d.od[i] = a->viewDims[i]; d.istr[i] = a->strides[i]; }
    const void *pin = a->data; void *pout = o->data; int64_t g = nb(n);
    #define L(T) k_gather_strided<T><<<g,TH,0,s>>>((const T*)pin,(T*)pout,d,n)
    DISP_SZ(esz, L);
    #undef L
    return done(e);
}
} // namespace

extern "C" {

// Contiguous: materialize a (possibly non-contiguous) view into a contiguous output of identical shape.
aclnnStatus aclnnContiguousGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != out->dtype || self->viewDims != out->viewDims || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnContiguous(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_view(e, (cudaStream_t)s); }

// Copy / Identity: out = self (same shape & dtype); supports non-contiguous source.
aclnnStatus aclnnCopyGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnContiguousGetWorkspaceSize(self, out, ws, ex); }
aclnnStatus aclnnCopy(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnContiguous(ws, wsz, e, s); }
aclnnStatus aclnnIdentityGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnContiguousGetWorkspaceSize(self, out, ws, ex); }
aclnnStatus aclnnIdentity(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnContiguous(ws, wsz, e, s); }

// ViewCopy: copy src (possibly strided) into a contiguous dst with the same element count (logical reshape copy).
aclnnStatus aclnnViewCopyGetWorkspaceSize(const aclTensor *src, aclTensor *dst, uint64_t *ws, aclOpExecutor **ex) {
    if (!src || !dst || !ex || src->dtype != dst->dtype || src->numel() != dst->numel() || !dst->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = src; e->out = dst; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnViewCopy(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    // iterate over src in its own logical order (row-major over src viewDims), write linearly to dst
    return run_view(e, (cudaStream_t)s);
}

// AsStrided: build a view (sizes, strides, storageOffset) over self's storage and materialize into out.
aclnnStatus aclnnAsStridedGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, const aclIntArray *stride, int64_t storageOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !size || !stride || !out || !ex || self->dtype != out->dtype || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)size->v.size(); if (rank != (int)stride->v.size() || rank > MAXD) return ACLNN_ERR_PARAM_INVALID;
    int64_t en = 1; for (int i = 0; i < rank; ++i) en *= size->v[i];
    if (out->numel() != en) return ACLNN_ERR_PARAM_INVALID;
    // build a temporary view tensor describing (size, stride, storageOffset) over self's data, owned for cleanup
    auto *v = new aclTensor(*self); v->viewDims.assign(size->v.begin(), size->v.end()); v->strides.assign(stride->v.begin(), stride->v.end()); v->offset = storageOffset;
    auto *e = new aclOpExecutor(); e->a = v; e->out = out; e->owned.push_back(v);
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAsStrided(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    aclnnStatus st = run_view(e, (cudaStream_t)s);   // run_view deletes e but not e->owned; free the view here
    return st;
}

} // extern "C"
} // namespace _format_ext
#undef DISP_SZ

