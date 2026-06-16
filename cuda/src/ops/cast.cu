// aclnnCast: dtype conversion, supporting all pairwise combinations of
// {fp32, fp16, bf16, int32, int8, uint8, fp8e4m3, fp8e5m2} via float intermediary.
// fp8 is the canonical path for quantization/dequantization (fp32→fp8 quantize, fp8→fp32 dequantize).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "hif8.cuh"
#include "subfp.cuh"

namespace {

inline int subfp_kind(aclDataType t) {
    switch (t) {
        case ACL_FLOAT4_E2M1: return SF_FP4E2M1;
        case ACL_FLOAT4_E1M2: return SF_FP4E1M2;
        case ACL_FLOAT6_E2M3: return SF_FP6E2M3;
        case ACL_FLOAT6_E3M2: return SF_FP6E3M2;
        default: return -1;
    }
}
inline bool is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool is_fp6(aclDataType t) { return t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline bool is_subfp(aclDataType t) { return is_fp4(t) || is_fp6(t); }

// fp4: 2 elements per byte (even element = low nibble, odd element = high nibble). One thread per output byte, handling 2 elements.
template <typename S> __global__ void k_to_fp4(const S *a, uint8_t *o, int64_t n, int k) {
    int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;     // byte index
    int64_t nb = (n + 1) / 2;
    if (b >= nb) return;
    uint8_t lo = subfp_encode(k, (float)a[2 * b]) & 0xf;
    uint8_t hi = (2 * b + 1 < n) ? (subfp_encode(k, (float)a[2 * b + 1]) & 0xf) : 0;
    o[b] = (uint8_t)(lo | (hi << 4));
}
template <typename D> __global__ void k_from_fp4(const uint8_t *a, D *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t byte = a[i / 2];
    uint8_t code = (i & 1) ? (byte >> 4) : (byte & 0xf);
    o[i] = (D)subfp_decode(k, code);
}
// fp6: 1 byte per element
template <typename S> __global__ void k_to_fp6(const S *a, uint8_t *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = subfp_encode(k, (float)a[i]);
}
template <typename D> __global__ void k_from_fp6(const uint8_t *a, D *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (D)subfp_decode(k, a[i] & 0x3f);
}

template <typename S, typename D>
__global__ void k_cast(const S *a, D *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (D)(float)a[i];
}

// HiF8 dedicated path: CUDA has no such type; routes through float with hand-written encode/decode.
template <typename S> __global__ void k_to_hif8(const S *a, uint8_t *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = hif8_encode((float)a[i]);
}
template <typename D> __global__ void k_from_hif8(const uint8_t *a, D *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (D)hif8_decode(a[i]);
}

template <typename S>
aclnnStatus launch_dst(const void *pa, void *po, aclDataType dst, int64_t n, cudaStream_t s) {
    const int64_t g = (n + 255) / 256;
    switch (dst) {
        case ACL_FLOAT:   k_cast<S, float><<<g, 256, 0, s>>>((const S *)pa, (float *)po, n); break;
        case ACL_FLOAT16: k_cast<S, __half><<<g, 256, 0, s>>>((const S *)pa, (__half *)po, n); break;
        case ACL_BF16:    k_cast<S, __nv_bfloat16><<<g, 256, 0, s>>>((const S *)pa, (__nv_bfloat16 *)po, n); break;
        case ACL_INT32:   k_cast<S, int32_t><<<g, 256, 0, s>>>((const S *)pa, (int32_t *)po, n); break;
        case ACL_INT8:    k_cast<S, int8_t><<<g, 256, 0, s>>>((const S *)pa, (int8_t *)po, n); break;
        case ACL_UINT8:   k_cast<S, uint8_t><<<g, 256, 0, s>>>((const S *)pa, (uint8_t *)po, n); break;
        case ACL_FLOAT8_E4M3FN: k_cast<S, __nv_fp8_e4m3><<<g, 256, 0, s>>>((const S *)pa, (__nv_fp8_e4m3 *)po, n); break;
        case ACL_FLOAT8_E5M2:   k_cast<S, __nv_fp8_e5m2><<<g, 256, 0, s>>>((const S *)pa, (__nv_fp8_e5m2 *)po, n); break;
        case ACL_HIFLOAT8:      k_to_hif8<S><<<g, 256, 0, s>>>((const S *)pa, (uint8_t *)po, n); break;
        case ACL_FLOAT4_E2M1: case ACL_FLOAT4_E1M2:
            k_to_fp4<S><<<((n + 1) / 2 + 255) / 256, 256, 0, s>>>((const S *)pa, (uint8_t *)po, n, subfp_kind(dst)); break;
        case ACL_FLOAT6_E2M3: case ACL_FLOAT6_E3M2:
            k_to_fp6<S><<<g, 256, 0, s>>>((const S *)pa, (uint8_t *)po, n, subfp_kind(dst)); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// HiF8 as source: decode to destination type.
static aclnnStatus launch_from_hif8(const void *pa, void *po, aclDataType dst, int64_t n, cudaStream_t s) {
    const int64_t g = (n + 255) / 256;
    switch (dst) {
        case ACL_FLOAT:   k_from_hif8<float><<<g, 256, 0, s>>>((const uint8_t *)pa, (float *)po, n); break;
        case ACL_FLOAT16: k_from_hif8<__half><<<g, 256, 0, s>>>((const uint8_t *)pa, (__half *)po, n); break;
        case ACL_BF16:    k_from_hif8<__nv_bfloat16><<<g, 256, 0, s>>>((const uint8_t *)pa, (__nv_bfloat16 *)po, n); break;
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// fp4/fp6 as source: decode to destination type (PoC: destination is fp32 only).
static aclnnStatus launch_from_subfp(const void *pa, void *po, aclDataType dst, int srcKind, int64_t n, cudaStream_t s) {
    if (dst != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    const int64_t g = (n + 255) / 256;
    if (srcKind == SF_FP4E2M1 || srcKind == SF_FP4E1M2)
        k_from_fp4<float><<<g, 256, 0, s>>>((const uint8_t *)pa, (float *)po, n, srcKind);
    else
        k_from_fp6<float><<<g, 256, 0, s>>>((const uint8_t *)pa, (float *)po, n, srcKind);
    return ACLNN_SUCCESS;
}

// fp6 Ascend-style 4-in-3 byte packing: 4 × 6-bit codes → 3 bytes (24 bits, little-endian bit order).
__global__ void k_fp6_pack(const uint8_t *in, uint8_t *out, int64_t n) {
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, i = g * 4;
    if (i >= n) return;
    uint8_t c0 = in[i] & 0x3f, c1 = (i+1<n)?in[i+1]&0x3f:0, c2 = (i+2<n)?in[i+2]&0x3f:0, c3 = (i+3<n)?in[i+3]&0x3f:0;
    out[g*3] = c0 | (c1 << 6); out[g*3+1] = (c1 >> 2) | (c2 << 4); out[g*3+2] = (c2 >> 4) | (c3 << 2);
}
__global__ void k_fp6_unpack(const uint8_t *in, uint8_t *out, int64_t n) {
    int64_t g = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, i = g * 4;
    if (i >= n) return;
    uint8_t b0 = in[g*3], b1 = in[g*3+1], b2 = in[g*3+2];
    out[i] = b0 & 0x3f;
    if (i+1<n) out[i+1] = ((b0 >> 6) | (b1 << 2)) & 0x3f;
    if (i+2<n) out[i+2] = ((b1 >> 4) | (b2 << 4)) & 0x3f;
    if (i+3<n) out[i+3] = (b2 >> 2) & 0x3f;
}

} // namespace

extern "C" {

// fp6 1-byte-per-element (low 6 bits) → true 4-in-3 packing (optional Ascend memory layout); out must be ceil(n/4)*3 bytes.
aclnnStatus aclnnFp6PackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out; e->castTo = ACL_DT_UNDEFINED; e->dim = 1;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFp6Pack(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    int64_t n = e->a->numel(); int64_t g = ((n + 3) / 4 + 255) / 256;
    k_fp6_pack<<<g, 256, 0, (cudaStream_t)stream>>>((const uint8_t *)e->a->data, (uint8_t *)e->out->data, n);
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
aclnnStatus aclnnFp6UnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out; e->castTo = ACL_DT_UNDEFINED; e->dim = 2;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFp6Unpack(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    int64_t n = e->out->numel(); int64_t g = ((n + 3) / 4 + 255) / 256;
    k_fp6_unpack<<<g, 256, 0, (cudaStream_t)stream>>>((const uint8_t *)e->a->data, (uint8_t *)e->out->data, n);
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, aclDataType dtype, aclTensor *out,
                                      uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != dtype || self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_CAST; e->a = self; e->out = out; e->castTo = dtype;
    *ws = 0; *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnCast(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_CAST) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t n = e->a->numel();
    aclnnStatus st;
    if (e->a->dtype == ACL_HIFLOAT8) {     // HiF8 source: use dedicated decode path
        st = launch_from_hif8(e->a->data, e->out->data, e->castTo, n, s);
        cudaError_t err = cudaGetLastError();
        delete e;
        if (st != ACLNN_SUCCESS) return st;
        return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    if (is_subfp(e->a->dtype)) {            // fp4/fp6 source: use dedicated decode path
        st = launch_from_subfp(e->a->data, e->out->data, e->castTo, subfp_kind(e->a->dtype), n, s);
        cudaError_t err = cudaGetLastError();
        delete e;
        if (st != ACLNN_SUCCESS) return st;
        return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    switch (e->a->dtype) {
        case ACL_FLOAT:   st = launch_dst<float>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_FLOAT16: st = launch_dst<__half>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_BF16:    st = launch_dst<__nv_bfloat16>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_INT32:   st = launch_dst<int32_t>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_INT8:    st = launch_dst<int8_t>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_UINT8:   st = launch_dst<uint8_t>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_FLOAT8_E4M3FN: st = launch_dst<__nv_fp8_e4m3>(e->a->data, e->out->data, e->castTo, n, s); break;
        case ACL_FLOAT8_E5M2:   st = launch_dst<__nv_fp8_e5m2>(e->a->data, e->out->data, e->castTo, n, s); break;
        default: st = ACLNN_ERR_PARAM_INVALID;
    }
    cudaError_t err = cudaGetLastError();
    delete e;
    if (st != ACLNN_SUCCESS) return st;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // extern "C"
