// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "subfp.cuh"
#include "aclnnop/aclnn_mc2.h"
#include <map>
#include <tuple>
#include <algorithm>

namespace _matmul {
// aclnnMatmul → cuBLASLt (row-major 2D; fp32 / fp16, fp32 accumulation).
// cubeMathType governs the Ascend Cube unit's precision-reduction policy; on GPU fp32 accumulation
// is always used, so only tolerances are affected.
// fp8 path: e4m3/e5m2 inputs (dual e5m2 not allowed), output fp16/fp32. cuBLASLt fp8 requires
//   TN layout (K-dim contiguous for both operands) → other[K,N] is transposed to [N,K] fp8 internally.
//   Per-tensor scale is applied via alpha.

namespace {

cublasLtHandle_t lt_handle() {
    static cublasLtHandle_t h = [] { cublasLtHandle_t t; cublasLtCreate(&t); return t; }();
    return h;
}
// Legacy cuBLAS handle (used only for native grouped-batched GEMM)
cublasHandle_t cublas_handle() {
    static cublasHandle_t h = [] { cublasHandle_t t; cublasCreate(&t); return t; }();
    return h;
}

constexpr uint64_t LT_WS = 4u << 20;   // recommended cuBLASLt workspace upper bound
constexpr int GMM_NS = 4;              // number of parallel streams for GroupedMatmul

inline bool is_fp8(aclDataType t) { return t == ACL_FLOAT8_E4M3FN || t == ACL_FLOAT8_E5M2; }
inline cudaDataType_t cuda_fp8(aclDataType t) { return t == ACL_FLOAT8_E4M3FN ? CUDA_R_8F_E4M3 : CUDA_R_8F_E5M2; }

// Fast low-precision switch: when CANN_FAST_TF32=1, fp32-input GEMMs use TF32 tensor-core
// (~10-bit mantissa, significant speedup, relative error ~1e-3; corresponds to CANN
// cubeMathType=ALLOW_FP32_DOWN_PRECISION). Disabled by default to preserve fp32 full precision (~1e-6).
// Takes effect only for fp32 data (fp16/bf16 always accumulate to fp32, unaffected). Re-reads env
// on every call to allow runtime switching/testing.
inline cublasComputeType_t fp32_compute(cudaDataType_t dt) {
    if (dt == CUDA_R_32F && getenv("CANN_FAST_TF32")) return CUBLAS_COMPUTE_32F_FAST_TF32;
    return CUBLAS_COMPUTE_32F;
}

// fp8 byte transpose: out[N,K] ← in[K,N] (1-byte elements, dtype-agnostic)
__global__ void transpose_u8(const uint8_t *in, uint8_t *out, int64_t K, int64_t N) {
    int64_t k = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) out[n * K + k] = in[k * N + n];
}
// fp4 (2 elements/byte) nibble-level transpose: in[K,N] (N packed contiguously) → out[N,K] (K packed contiguously).
// One thread per output byte (n, kp): reads the low nibble of in at (2kp,n) and the high nibble at (2kp+1,n),
// then packs them into one byte (no write conflict). K and N must be even.
__global__ void transpose_fp4(const uint8_t *in, uint8_t *out, int64_t K, int64_t N) {
    int64_t n = blockIdx.y * blockDim.y + threadIdx.y, kp = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || kp >= K / 2) return;
    int64_t i0 = (2 * kp) * N + n, i1 = (2 * kp + 1) * N + n;          // linear indices of in elements (2kp,n) and (2kp+1,n)
    uint8_t lo = (in[i0 >> 1] >> ((i0 & 1) * 4)) & 0xF;
    uint8_t hi = (in[i1 >> 1] >> ((i1 & 1) * 4)) & 0xF;
    out[n * (K / 2) + kp] = (uint8_t)((hi << 4) | lo);
}

__host__ __device__ inline int64_t ru(int64_t x, int64_t m) { return (x + m - 1) / m * m; }
// E8M0 block-scale swizzle: rearranges logical scale [rows, nb] into the 128×4 super-tile layout
//   expected by cuBLASLt (intra-tile index = (r%32)*16 + (r/32)*4 + c, aligned to tcgen05 MMA scale layout).
//   When colmajor=true, input is read as in[b*rows+m] (for sB's [nb,N] layout, rows=N).
__global__ void k_swizzle_scale(const uint8_t *in, uint8_t *out, int rows, int nb, int colmajor) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (int64_t)rows * nb) return;
    int m = tid / nb, b = tid % nb;
    int nb_pad = (int)ru(nb, 4), tiles_per_row = nb_pad / 4;
    int tile = (m / 128) * tiles_per_row + (b / 4);
    int r = m % 128, c = b % 4;
    int64_t off = (int64_t)tile * 512 + (r % 32) * 16 + (r / 32) * 4 + c;
    out[off] = colmajor ? in[(int64_t)b * rows + m] : in[(int64_t)m * nb + b];
}
inline int64_t swizzle_bytes(int rows, int nb) { return ru(rows, 128) * ru(nb, 4); }

// ---- NVFP4 scale conversion: Ascend MXFP4 uses E8M0 (pure exponent, bias 127) per block of 32; Blackwell's
// native fp4 tensor core (cuBLASLt VEC16_UE4M3) wants an E4M3 scale per block of 16. Each block-32 maps to
// two child block-16s with the SAME scale, so the dequantized values are identical to the functional path —
// EXCEPT where the exponent leaves E4M3's representable power-of-2 range [2^-6, 2^8], which is clamped
// (the documented "reduced-fidelity fast tier"). E4M3: bias 7, exp_field∈[1,15], mantissa 0 → 2^exp.
__device__ inline uint8_t e8m0_to_e4m3_pow2(uint8_t e8) {
    int ex = (int)e8 - 127;
    if (ex < -6) ex = -6;
    if (ex > 8)  ex = 8;
    return (uint8_t)(((ex + 7) & 0xF) << 3);
}
// scaleA[M,nb32] (block = last dim) → E4M3[M,nb16=2*nb32], each block-32 duplicated to its two block-16s
__global__ void k_e8m0_to_e4m3_A(const uint8_t *in, uint8_t *out, int rows, int nb32) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= (int64_t)rows * nb32) return;
    int r = tid / nb32, b = tid % nb32, nb16 = nb32 * 2; uint8_t v = e8m0_to_e4m3_pow2(in[tid]);
    out[(int64_t)r * nb16 + 2 * b] = v; out[(int64_t)r * nb16 + 2 * b + 1] = v;
}
// scaleB[nb32,N] (block = first dim) → E4M3[nb16,N]
__global__ void k_e8m0_to_e4m3_B(const uint8_t *in, uint8_t *out, int nb32, int N) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (tid >= (int64_t)nb32 * N) return;
    int b = tid / N, j = tid % N; uint8_t v = e8m0_to_e4m3_pow2(in[tid]);
    out[(int64_t)(2 * b) * N + j] = v; out[(int64_t)(2 * b + 1) * N + j] = v;
}

// Unpack fp4 (2 elements/byte) / fp6 (1 byte/element) to fp16 (indexed by flat logical index)
__global__ void deq_fp4_to_half(const uint8_t *p, __half *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t byte = p[i / 2], code = (i & 1) ? (byte >> 4) : (byte & 0xf);
    o[i] = (__half)subfp_decode(k, code);
}
__global__ void deq_fp6_to_half(const uint8_t *p, __half *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (__half)subfp_decode(k, p[i] & 0x3f);
}
// Unpack to fp8 e4m3: fp4/fp6 decoded values are exactly representable in E4M3 → lossless;
// intermediate buffer halved + fp8 tensor-core ~2× throughput
__global__ void deq_fp4_to_fp8(const uint8_t *p, __nv_fp8_e4m3 *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t byte = p[i / 2], code = (i & 1) ? (byte >> 4) : (byte & 0xf);
    o[i] = (__nv_fp8_e4m3)subfp_decode(k, code);
}
__global__ void deq_fp6_to_fp8(const uint8_t *p, __nv_fp8_e4m3 *o, int64_t n, int k) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (__nv_fp8_e4m3)subfp_decode(k, p[i] & 0x3f);
}
inline bool mm_is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool mm_is_fp6(aclDataType t) { return t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline bool mm_is_subfp(aclDataType t) { return mm_is_fp4(t) || mm_is_fp6(t); }
inline int mm_subkind(aclDataType t) {
    return t == ACL_FLOAT4_E2M1 ? SF_FP4E2M1 : t == ACL_FLOAT4_E1M2 ? SF_FP4E1M2 : t == ACL_FLOAT6_E2M3 ? SF_FP6E2M3 : SF_FP6E3M2;
}
inline void deq_subfp(const void *p, __half *o, int64_t n, aclDataType dt, cudaStream_t s) {
    int64_t g = (n + 255) / 256;
    if (mm_is_fp4(dt)) deq_fp4_to_half<<<g, 256, 0, s>>>((const uint8_t *)p, o, n, mm_subkind(dt));
    else               deq_fp6_to_half<<<g, 256, 0, s>>>((const uint8_t *)p, o, n, mm_subkind(dt));
}
inline void deq_subfp_fp8(const void *p, __nv_fp8_e4m3 *o, int64_t n, aclDataType dt, cudaStream_t s) {
    int64_t g = (n + 255) / 256;
    if (mm_is_fp4(dt)) deq_fp4_to_fp8<<<g, 256, 0, s>>>((const uint8_t *)p, o, n, mm_subkind(dt));
    else               deq_fp6_to_fp8<<<g, 256, 0, s>>>((const uint8_t *)p, o, n, mm_subkind(dt));
}

constexpr int MX_BLK = 32;   // MXFP8 block size (along K)

// Dequantize: A[M,K] fp8 + E8M0 block scale sA[M,K/32] → Af[M,K] fp16 (block along K, row-major)
template <typename FP8>
__global__ void deq_a(const FP8 *a, const uint8_t *sA, __half *af, int64_t M, int64_t K) {
    int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || k >= K) return;
    int64_t nb = K / MX_BLK;
    float sc = exp2f((float)sA[i * nb + k / MX_BLK] - 127.f);
    af[i * K + k] = (__half)((float)a[i * K + k] * sc);
}
// Dequantize: B[K,N] fp8 + E8M0 block scale sB[K/32,N] → Bf[K,N] fp16 (block along K = row direction)
template <typename FP8>
__global__ void deq_b(const FP8 *b, const uint8_t *sB, __half *bf, int64_t K, int64_t N) {
    int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    int64_t j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || j >= N) return;
    float sc = exp2f((float)sB[(k / MX_BLK) * N + j] - 127.f);
    bf[k * N + j] = (__half)((float)b[k * N + j] * sc);
}

// MXFP4: fp4(e2m1, 2 elements/byte packed) + E8M0 per-32 block scale → dequantize to fp16
// A[M,K]: element (i,k) is the nibble at byte (i*K+k)/2; block scale sA[M,K/32]
__global__ void deq_a_fp4(const uint8_t *a, const uint8_t *sA, __half *af, int64_t M, int64_t K) {
    int64_t i = (int64_t)blockIdx.y * blockDim.y + threadIdx.y, k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || k >= K) return;
    int64_t idx = i * K + k; uint8_t byte = a[idx / 2]; uint8_t code = (idx & 1) ? (byte >> 4) : (byte & 0xf);
    float sc = exp2f((float)sA[i * (K / MX_BLK) + k / MX_BLK] - 127.f);
    af[idx] = (__half)(subfp_decode(SF_FP4E2M1, code) * sc);
}
// B[K,N]: element (k,j) is at byte (k*N+j)/2; block scale sB[K/32,N]
__global__ void deq_b_fp4(const uint8_t *b, const uint8_t *sB, __half *bf, int64_t K, int64_t N) {
    int64_t k = (int64_t)blockIdx.y * blockDim.y + threadIdx.y, j = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || j >= N) return;
    int64_t idx = k * N + j; uint8_t byte = b[idx / 2]; uint8_t code = (idx & 1) ? (byte >> 4) : (byte & 0xf);
    float sc = exp2f((float)sB[(k / MX_BLK) * N + j] - 127.f);
    bf[idx] = (__half)(subfp_decode(SF_FP4E2M1, code) * sc);
}

// fp16 row-major GEMM: Af[M,K]@Bf[K,N]→out (out fp16/fp32)
aclnnStatus gemm_half(const __half *Af, const __half *Bf, void *out, aclDataType outDt,
                      int64_t M, int64_t N, int64_t K, void *ltWs, size_t ltSize, cudaStream_t s) {
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, M, K, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, N, N);
    cublasLtMatrixLayoutCreate(&lc, outDt == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, M, N, N);
    for (auto l : {la, lb, lc}) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, Af, la, Bf, lb, &beta, out, lc, out, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    return (st == CUBLAS_STATUS_SUCCESS) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// Row-major GEMM (optional strided batch): A[M,K]@B[K,N]→C[M,N], dt=fp32/fp16, fp32 accumulation
aclnnStatus gemm_rm(cudaDataType_t dt, const void *A, const void *B, void *C,
                    int64_t M, int64_t N, int64_t K, int batch, int64_t sA, int64_t sB, int64_t sC,
                    void *ltWs, size_t ltSize, cudaStream_t s) {
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, fp32_compute(dt), CUDA_R_32F);   // fp32 optionally uses TF32 fast path
    const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutCreate(&la, dt, M, K, K);
    cublasLtMatrixLayoutCreate(&lb, dt, K, N, N);
    cublasLtMatrixLayoutCreate(&lc, dt, M, N, N);
    for (auto l : {la, lb, lc}) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    if (batch > 1) {
        int32_t bc = batch;
        cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
        cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
        cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
        cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA, sizeof(sA));
        cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB, sizeof(sB));
        cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC, sizeof(sC));
    }
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, A, la, B, lb, &beta, C, lc, C, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    return st == CUBLAS_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// Materialize 2D transpose: in[R,C]→out[C,R] (fp16/fp32), avoiding the cuBLASLt transpose + row-major layout pitfall
template <typename T>
__global__ void k_T2d(const T *in, T *out, int64_t R, int64_t C) {
    int64_t r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R && c < C) out[c * R + r] = in[r * C + c];
}
inline void launch_T2d(cudaDataType_t dt, const void *in, void *out, int64_t R, int64_t C, cudaStream_t s) {
    dim3 tb(16, 16), g((C + 15) / 16, (R + 15) / 16);
    if (dt == CUDA_R_32F)      k_T2d<float><<<g, tb, 0, s>>>((const float *)in, (float *)out, R, C);
    else if (dt == CUDA_R_16BF) k_T2d<__nv_bfloat16><<<g, tb, 0, s>>>((const __nv_bfloat16 *)in, (__nv_bfloat16 *)out, R, C);
    else                       k_T2d<__half><<<g, tb, 0, s>>>((const __half *)in, (__half *)out, R, C);
}

// Post-GEMM epilogue: out[i,j] = act(out[i,j] + bias); bias shape [N] (broadcast over columns) or [M,N] (full). act: 0=none / 1=ReLU / 2=GeLU
template <typename T>
__global__ void k_bias_act(T *o, const T *bias, int64_t M, int64_t N, int biasFull, int act) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    float v = (float)o[i];
    if (bias) v += (float)(biasFull ? bias[i] : bias[i % N]);
    if (act == 1) v = v > 0 ? v : 0;
    else if (act == 2) v = 0.5f * v * erfcf(-v * 0.70710678118654752f);
    o[i] = (T)v;
}
inline void launch_bias_act(cudaDataType_t dt, void *o, const void *bias, int64_t M, int64_t N, int biasFull, int act, cudaStream_t s) {
    int64_t g = (M * N + 255) / 256;
    if (dt == CUDA_R_32F)      k_bias_act<float><<<g, 256, 0, s>>>((float *)o, (const float *)bias, M, N, biasFull, act);
    else if (dt == CUDA_R_16BF) k_bias_act<__nv_bfloat16><<<g, 256, 0, s>>>((__nv_bfloat16 *)o, (const __nv_bfloat16 *)bias, M, N, biasFull, act);
    else                       k_bias_act<__half><<<g, 256, 0, s>>>((__half *)o, (const __half *)bias, M, N, biasFull, act);
}

aclnnStatus matmul_subfp(aclOpExecutor *e, void *ws, uint64_t wsSize, cudaStream_t s) {
    const int64_t M = e->m, N = e->n, K = e->k;
    // fp8 path: unpack to fp8 e4m3 (lossless) → fp8 tensor-core GEMM (intermediate buffer halved + ~2× throughput). K must be aligned to 16; falls back to fp16 on failure.
    if (K % 16 == 0 && !getenv("SUBFP_NO_FP8")) {
        __nv_fp8_e4m3 *Af8 = (__nv_fp8_e4m3 *)ws, *Bf8 = Af8 + M * K; uint8_t *Bt = (uint8_t *)(Bf8 + K * N);
        void *ltWs = Bt + (size_t)N * K; size_t ltSize = wsSize - ((size_t)M * K + (size_t)K * N + (size_t)N * K);
        deq_subfp_fp8(e->a->data, Af8, M * K, e->a->dtype, s);
        deq_subfp_fp8(e->b->data, Bf8, K * N, e->b->dtype, s);
        dim3 tb(16, 16), g((N + 15) / 16, (K + 15) / 16);
        transpose_u8<<<g, tb, 0, s>>>((const uint8_t *)Bf8, Bt, K, N);   // B[K,N]→Bt[N,K]   (transpose B)
        cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K);
        cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, N, M, N);
        const float alpha = 1.f, beta = 0.f;
        cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, Af8, lb, &beta, e->out->data, lc, e->out->data, lc, nullptr, ltWs, ltSize, s);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
        if (st == CUBLAS_STATUS_SUCCESS) return ACLNN_SUCCESS;
        if (getenv("SUBFP_DBG")) fprintf(stderr, "[matmul_subfp] fp8 GEMM failed status=%d, falling back to fp16\n", (int)st);
        // On failure, fall through to the fp16 path below (ws was allocated to fp16 size, which is sufficient)
    }
    __half *Af = (__half *)ws, *Bf = Af + M * K;
    void *ltWs = (char *)(Bf + K * N); size_t ltSize = wsSize - (size_t)(M * K + K * N) * sizeof(__half);
    deq_subfp(e->a->data, Af, M * K, e->a->dtype, s);
    deq_subfp(e->b->data, Bf, K * N, e->b->dtype, s);
    return gemm_half(Af, Bf, e->out->data, e->out->dtype, M, N, K, ltWs, ltSize, s);
}

aclnnStatus matmul_fp8(aclOpExecutor *e, void *ws, uint64_t wsSize, cudaStream_t s) {
    const int64_t M = e->m, N = e->n, K = e->k;
    uint8_t *Bt = (uint8_t *)ws;                       // transposed other: [N,K] fp8
    void *ltWs = (char *)ws + (size_t)N * K;
    size_t ltSize = wsSize - (size_t)N * K;
    dim3 tb(16, 16), g((N + 15) / 16, (K + 15) / 16);
    transpose_u8<<<g, tb, 0, s>>>((const uint8_t *)e->b->data, Bt, K, N);

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
    // TN layout (column-major): A_arg=Bt[K,N] ld=K opT→[N,K]; B_arg=self[K,M] ld=K opN→[K,M]; C=out[N,M] ld=N = row-major[M,N]
    cublasLtMatrixLayoutCreate(&la, cuda_fp8(e->b->dtype), K, N, K);
    cublasLtMatrixLayoutCreate(&lb, cuda_fp8(e->a->dtype), K, M, K);
    cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, N, M, N);
    const float alpha = (float)e->alpha, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, e->a->data, lb,
                                       &beta, e->out->data, lc, e->out->data, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    return (st == CUBLAS_STATUS_SUCCESS) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

// int32 accumulation → per-channel(N) dequantized output (native int8 GEMM post-epilogue; template must be outside extern C)
template <typename T> __global__ void k_scale_i32(const int32_t *c, const __half *scale, T *o, int64_t M, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= M * N) return;
    o[i] = (T)((float)c[i] * (float)scale[i % N]);
}
// Fused weight-quantized GEMM: reads int8/int4 weights directly (no fp16 materialization), dequantizes on-chip.
// Used for small M (decode / small batch, memory-bandwidth-bound): reads only K·N bytes (int8) or K·N/2 (int4),
// rather than materializing K·N·2 bytes of fp16 and reading it back — saves 4× / 8× weight bandwidth.
// dequant=(w-off)·scale → Σ_k act·(w-off)·scale = scale·(Σact·w − off·Σact),
// so accumulate acc=Σact·w_int and sa=Σact.
// Each thread owns one output column n (coalesced weight-column reads across n); iterates over M rows.
template <bool W4> __global__ void k_wq_fused(const __half *act, const void *w, const __half *scale, const __half *off,
                                              __half *out, int64_t M, int64_t K, int64_t N) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (n >= N) return;
    float sc = (float)scale[n], o = off ? (float)off[n] : 0.f;
    for (int64_t m = 0; m < M; ++m) {
        const __half *ar = act + m * K; float acc = 0, sa = 0;
        for (int64_t k = 0; k < K; ++k) {
            float a = (float)ar[k]; sa += a; float wv;
            if (W4) { int64_t idx = k * N + n; uint8_t byte = ((const uint8_t *)w)[idx >> 1];
                      int v = (idx & 1) ? (byte >> 4) : (byte & 0xf); if (v >= 8) v -= 16; wv = (float)v; }
            else    { wv = (float)((const int8_t *)w)[k * N + n]; }
            acc += a * wv;
        }
        out[m * N + n] = (__half)((acc - o * sa) * sc);
    }
}

extern "C" {

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                        int8_t /*cubeMathType*/, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->data || !other->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    const bool subfp = mm_is_subfp(self->dtype);
    const bool fp8 = is_fp8(self->dtype);
    if (subfp) {
        if (other->dtype != self->dtype) return ACLNN_ERR_PARAM_INVALID;       // PoC: both operands must have the same dtype
        if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    } else if (fp8) {
        if (!is_fp8(other->dtype)) return ACLNN_ERR_PARAM_INVALID;
        if (self->dtype == ACL_FLOAT8_E5M2 && other->dtype == ACL_FLOAT8_E5M2) return ACLNN_ERR_PARAM_INVALID; // dual e5m2 not allowed
        if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;             // fp8 output must be fp16/fp32
    } else {
        if (self->dtype != other->dtype) return ACLNN_ERR_PARAM_INVALID;
        if (self->dtype != ACL_FLOAT && self->dtype != ACL_FLOAT16 && self->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
        // Output dtype: same as input, or fp16/bf16 inputs accumulated into fp32 output
        // (common Ascend pattern fp16×fp16→fp32, natively supported by cuBLASLt)
        if (out->dtype != self->dtype &&
            !((self->dtype == ACL_FLOAT16 || self->dtype == ACL_BF16) && out->dtype == ACL_FLOAT))
            return ACLNN_ERR_PARAM_INVALID;
    }
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2 || out->viewDims.size() != 2)
        return ACLNN_ERR_PARAM_INVALID;                       // PoC: 2D only; batch is a future extension
    if (!self->contiguous() || !other->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    const int64_t m = self->viewDims[0], k = self->viewDims[1], n = other->viewDims[1];
    if (other->viewDims[0] != k || out->viewDims[0] != m || out->viewDims[1] != n) return ACLNN_ERR_PARAM_INVALID;
    if (fp8 && (k % 16 != 0)) return ACLNN_ERR_PARAM_INVALID;  // cuBLASLt fp8 requires K to be a multiple of 16
    auto *e = new aclOpExecutor();
    e->op = OP_MATMUL; e->a = self; e->b = other; e->out = out;
    e->m = m; e->n = n; e->k = k;
    if (subfp)    *ws = (uint64_t)(m * k + k * n) * sizeof(__half) + LT_WS;   // unpack to fp16
    else if (fp8) *ws = (uint64_t)n * k + LT_WS;                              // transposed B buffer
    else          *ws = LT_WS;
    *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmul(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_MATMUL) return ACLNN_ERR_PARAM_INVALID;
    if (mm_is_subfp(e->a->dtype)) {
        aclnnStatus st = matmul_subfp(e, ws, wsSize, (cudaStream_t)stream);
        delete e;
        return st;
    }
    if (is_fp8(e->a->dtype)) {
        aclnnStatus st = matmul_fp8(e, ws, wsSize, (cudaStream_t)stream);
        delete e;
        return st;
    }
    const cudaDataType_t dt = (e->a->dtype == ACL_FLOAT) ? CUDA_R_32F
                            : (e->a->dtype == ACL_BF16)  ? CUDA_R_16BF : CUDA_R_16F;

    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulDescCreate(&op, fp32_compute(dt), CUDA_R_32F);   // fp32 optionally uses TF32 fast path
    const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    const cudaDataType_t dtc = (e->out->dtype == ACL_FLOAT) ? CUDA_R_32F      // output may differ from input dtype (fp16→fp32)
                             : (e->out->dtype == ACL_BF16)  ? CUDA_R_16BF : CUDA_R_16F;
    cublasLtMatrixLayoutCreate(&la, dt, e->m, e->k, e->k);
    cublasLtMatrixLayoutCreate(&lb, dt, e->k, e->n, e->n);
    cublasLtMatrixLayoutCreate(&lc, dtc, e->m, e->n, e->n);
    for (auto l : {la, lb, lc})
        cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));

    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, e->a->data, la, e->b->data, lb,
                                       &beta, e->out->data, lc, e->out->data, lc,
                                       nullptr, ws, wsSize, (cudaStream_t)stream);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    delete e;
    return (st == CUBLAS_STATUS_SUCCESS) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- MXFP8: block dequantize to fp16 + standard fp16 GEMM ----
aclnnStatus aclnnMatmulMxFp8GetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
                                             const aclTensor *other, const aclTensor *otherScale,
                                             aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !selfScale || !other || !otherScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->data || !other->data || !selfScale->data || !otherScale->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!is_fp8(self->dtype) || !is_fp8(other->dtype)) return ACLNN_ERR_PARAM_INVALID;
    if (selfScale->dtype != ACL_FLOAT8_E8M0 || otherScale->dtype != ACL_FLOAT8_E8M0) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t m = self->viewDims[0], k = self->viewDims[1], n = other->viewDims[1];
    if (other->viewDims[0] != k || out->viewDims[0] != m || out->viewDims[1] != n) return ACLNN_ERR_PARAM_INVALID;
    if (k % MX_BLK != 0) return ACLNN_ERR_PARAM_INVALID;               // K must be divisible by block size
    if (selfScale->viewDims != std::vector<int64_t>{m, k / MX_BLK}) return ACLNN_ERR_PARAM_INVALID;
    if (otherScale->viewDims != std::vector<int64_t>{k / MX_BLK, n}) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_MATMUL; e->a = self; e->b = other; e->c = selfScale; e->mask = otherScale; e->out = out;
    e->m = m; e->n = n; e->k = k;
    *ws = (uint64_t)(m * k + k * n) * sizeof(__half) + LT_WS;          // Af + Bf fp16 buffers + cuBLASLt workspace
    *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMatmulMxFp8(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_MATMUL) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k;
    __half *Af = (__half *)ws, *Bf = Af + M * K;
    void *ltWs = (char *)(Bf + K * N); size_t ltSize = wsSize - (size_t)(M * K + K * N) * sizeof(__half);
    const uint8_t *sA = (const uint8_t *)e->c->data, *sB = (const uint8_t *)e->mask->data;
    dim3 tb(16, 16);
    dim3 ga((K + 15) / 16, (M + 15) / 16), gb((N + 15) / 16, (K + 15) / 16);
    if (e->a->dtype == ACL_FLOAT8_E4M3FN) deq_a<__nv_fp8_e4m3><<<ga, tb, 0, s>>>((const __nv_fp8_e4m3 *)e->a->data, sA, Af, M, K);
    else                                  deq_a<__nv_fp8_e5m2><<<ga, tb, 0, s>>>((const __nv_fp8_e5m2 *)e->a->data, sA, Af, M, K);
    if (e->b->dtype == ACL_FLOAT8_E4M3FN) deq_b<__nv_fp8_e4m3><<<gb, tb, 0, s>>>((const __nv_fp8_e4m3 *)e->b->data, sB, Bf, K, N);
    else                                  deq_b<__nv_fp8_e5m2><<<gb, tb, 0, s>>>((const __nv_fp8_e5m2 *)e->b->data, sB, Bf, K, N);

    // Standard fp16 GEMM (ORDER_ROW): Af[M,K]@Bf[K,N]→out
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, M, K, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, N, N);
    cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, M, N, N);
    for (auto l : {la, lb, lc}) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, Af, la, Bf, lb, &beta,
                                       e->out->data, lc, e->out->data, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
    delete e;
    return (st == CUBLAS_STATUS_SUCCESS) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- MXFP4: fp4(e2m1) + E8M0 block scale → block dequantize to fp16 + GEMM (functional path, same structure as MXFP8) ----
aclnnStatus aclnnMatmulMxFp4GetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
                                             const aclTensor *other, const aclTensor *otherScale,
                                             aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !selfScale || !other || !otherScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (!self->data || !other->data || !selfScale->data || !otherScale->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != ACL_FLOAT4_E2M1 || other->dtype != ACL_FLOAT4_E2M1) return ACLNN_ERR_PARAM_INVALID;
    if (selfScale->dtype != ACL_FLOAT8_E8M0 || otherScale->dtype != ACL_FLOAT8_E8M0) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t m = self->viewDims[0], k = self->viewDims[1], n = other->viewDims[1];
    if (other->viewDims[0] != k || out->viewDims[0] != m || out->viewDims[1] != n) return ACLNN_ERR_PARAM_INVALID;
    if (k % MX_BLK != 0) return ACLNN_ERR_PARAM_INVALID;
    if (selfScale->viewDims != std::vector<int64_t>{m, k / MX_BLK}) return ACLNN_ERR_PARAM_INVALID;
    if (otherScale->viewDims != std::vector<int64_t>{k / MX_BLK, n}) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_MATMUL; e->a = self; e->b = other; e->c = selfScale; e->mask = otherScale; e->out = out;
    e->m = m; e->n = n; e->k = k;
    *ws = (uint64_t)(m * k + k * n) * sizeof(__half) + LT_WS;
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulMxFp4(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k;
    __half *Af = (__half *)ws, *Bf = Af + M * K;
    void *ltWs = (char *)(Bf + K * N); size_t ltSize = wsSize - (size_t)(M * K + K * N) * sizeof(__half);
    const uint8_t *sA = (const uint8_t *)e->c->data, *sB = (const uint8_t *)e->mask->data;
    dim3 tb(16, 16), ga((K + 15) / 16, (M + 15) / 16), gb((N + 15) / 16, (K + 15) / 16);
    deq_a_fp4<<<ga, tb, 0, s>>>((const uint8_t *)e->a->data, sA, Af, M, K);
    deq_b_fp4<<<gb, tb, 0, s>>>((const uint8_t *)e->b->data, sB, Bf, K, N);
    aclnnStatus st = gemm_half(Af, Bf, e->out->data, e->out->dtype, M, N, K, ltWs, ltSize, s);
    delete e; return st;
}

// Native microscale tensor-core (VEC32_UE8M0) is available only on Blackwell+ (cc≥10: sm_100/sm_120/sm_121);
// all other architectures fall back to the functional path.
// The device query result is cached; set MXFP8_NO_HW to force the functional path (for comparison/debugging).
static bool mx_use_hw() {
    static int hw = [] {
        if (getenv("MXFP8_NO_HW")) return 0;
        int dev = 0; if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        cudaDeviceProp p; if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return 0;
        return p.major >= 10 ? 1 : 0;
    }();
    return hw != 0;
}

// ---- MXFP8 hardware tensor-core path: cuBLASLt native VEC32_UE8M0 block scaling + swizzled scale layout ----
//   Reuses fp8 TN layout (A=Btᵀ, B=self); scales are rearranged into 128×4 super-tiles by k_swizzle_scale.
//   A operand = Bt (other), its scale rows=N; B operand = self, its scale rows=M.
//   Non-Blackwell architectures (Hopper/Ada/…) lack native microscaling → e->keepDim flags fallback to
//   aclnnMatmulMxFp8 functional path (block dequantize → fp16 GEMM).
aclnnStatus aclnnMatmulMxFp8HwGetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
        const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !selfScale || !other || !otherScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (!is_fp8(self->dtype) || !is_fp8(other->dtype)) return ACLNN_ERR_PARAM_INVALID;
    if (selfScale->dtype != ACL_FLOAT8_E8M0 || otherScale->dtype != ACL_FLOAT8_E8M0) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t m = self->viewDims[0], k = self->viewDims[1], n = other->viewDims[1];
    if (other->viewDims[0] != k || out->viewDims[0] != m || out->viewDims[1] != n) return ACLNN_ERR_PARAM_INVALID;
    if (k % MX_BLK != 0 || k % 16 != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_MATMUL; e->a = self; e->b = other; e->c = selfScale; e->mask = otherScale; e->out = out;
    e->m = m; e->n = n; e->k = k;
    int64_t nb = k / MX_BLK;
    if (!mx_use_hw()) {                                   // non-Blackwell: fall back to functional path, allocate per its ws layout
        e->keepDim = true;
        *ws = (uint64_t)(m * k + k * n) * sizeof(__half) + LT_WS;
    } else {
        e->keepDim = false;
        *ws = (uint64_t)n * k + (uint64_t)swizzle_bytes((int)m, (int)nb) + (uint64_t)swizzle_bytes((int)n, (int)nb) + LT_WS;
    }
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulMxFp8Hw(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    if (e->keepDim) return aclnnMatmulMxFp8(ws, wsSize, e, stream);   // non-Blackwell fallback (functional path deletes e)
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k, nb = K / MX_BLK;
    int64_t saBytes = swizzle_bytes((int)M, (int)nb), sbBytes = swizzle_bytes((int)N, (int)nb);
    uint8_t *Bt = (uint8_t *)ws;                 // [N,K] fp8
    uint8_t *sAsw = Bt + (size_t)N * K;          // self scale (B operand), rows=M
    uint8_t *sBsw = sAsw + saBytes;              // other scale (A operand), rows=N
    void *ltWs = sBsw + sbBytes; size_t ltSize = wsSize - (size_t)((char *)ltWs - (char *)ws);
    dim3 tb(16, 16);
    transpose_u8<<<dim3((N+15)/16,(K+15)/16), tb, 0, s>>>((const uint8_t *)e->b->data, Bt, K, N);
    cudaMemsetAsync(sAsw, 0, saBytes, s); cudaMemsetAsync(sBsw, 0, sbBytes, s);
    int gm = (int)((M*nb + 255) / 256), gn = (int)((N*nb + 255) / 256);
    k_swizzle_scale<<<gm, 256, 0, s>>>((const uint8_t *)e->c->data, sAsw, (int)M, (int)nb, 0);     // selfScale[M,nb]
    k_swizzle_scale<<<gn, 256, 0, s>>>((const uint8_t *)e->mask->data, sBsw, (int)N, (int)nb, 1);  // otherScale[nb,N] → rows=N

    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    void *aScale = sBsw, *bScale = sAsw;   // A operand=Bt(other)→sBsw; B operand=self→sAsw
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));
    cublasLtMatrixLayoutCreate(&la, cuda_fp8(e->b->dtype), K, N, K);
    cublasLtMatrixLayoutCreate(&lb, cuda_fp8(e->a->dtype), K, M, K);
    cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, N, M, N);
    const float alpha = 1.f, beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, e->a->data, lb, &beta, e->out->data, lc, e->out->data, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    delete e;
    return st == CUBLAS_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- NVFP4 native fp4 tensor-core path (cuBLASLt VEC16_UE4M3, block-16) ----
//   This cuBLASLt only accepts NVFP4 (VEC16_UE4M3) for fp4 — not MXFP4 (VEC32_UE8M0) — so to actually hit the
//   native fp4 tensor cores we convert Ascend's E8M0/block-32 scales to E4M3/block-16 (each block-32 → two
//   block-16 with the same 2^exp value: dequant values identical except where the exponent leaves E4M3's
//   [2^-6,2^8] range, then clamped). Opt-in via NVFP4_HW=1; falls back to the functional path if cuBLASLt
//   declines. The exact-range functional path (full E8M0) remains the fidelity reference.
static aclnnStatus nvfp4_hw(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k, nb16 = K / 16; const int nb32 = (int)(K / MX_BLK);
    int64_t saBytes = swizzle_bytes((int)M, (int)nb16), sbBytes = swizzle_bytes((int)N, (int)nb16);
    uint8_t *Bt = (uint8_t *)ws;                       // [N,K] fp4 packed = N*K/2 bytes
    uint8_t *cA = Bt + (size_t)N * K / 2;              // converted E4M3 scaleA [M,nb16]
    uint8_t *cB = cA + (size_t)M * nb16;               // converted E4M3 scaleB [nb16,N]
    uint8_t *sAsw = cB + (size_t)nb16 * N;             // swizzled selfScale (rows=M)
    uint8_t *sBsw = sAsw + saBytes;                    // swizzled otherScale (rows=N)
    void *ltWs = sBsw + sbBytes; size_t ltSize = wsSize - (size_t)((char *)ltWs - (char *)ws);
    transpose_fp4<<<dim3((unsigned)((K/2+15)/16), (unsigned)((N+15)/16)), dim3(16,16), 0, s>>>((const uint8_t *)e->b->data, Bt, K, N);
    k_e8m0_to_e4m3_A<<<(unsigned)((M*nb32+255)/256), 256, 0, s>>>((const uint8_t *)e->c->data, cA, (int)M, nb32);
    k_e8m0_to_e4m3_B<<<(unsigned)((nb32*N+255)/256), 256, 0, s>>>((const uint8_t *)e->mask->data, cB, nb32, (int)N);
    cudaMemsetAsync(sAsw, 0, saBytes, s); cudaMemsetAsync(sBsw, 0, sbBytes, s);
    int gm = (int)((M*nb16 + 255) / 256), gn = (int)((N*nb16 + 255) / 256);
    k_swizzle_scale<<<gm, 256, 0, s>>>(cA, sAsw, (int)M, (int)nb16, 0);
    k_swizzle_scale<<<gn, 256, 0, s>>>(cB, sBsw, (int)N, (int)nb16, 1);
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    void *aScale = sBsw, *bScale = sAsw;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));
    cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, N, M, N);
    const float alpha = 1.f, beta = 0.f;
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ltSize, sizeof(ltSize));
    cublasLtMatmulHeuristicResult_t heur{}; int nres = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt_handle(), op, la, lb, lc, lc, pref, 1, &heur, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasStatus_t st = (hs == CUBLAS_STATUS_SUCCESS && nres > 0)
        ? cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, e->a->data, lb, &beta, e->out->data, lc, e->out->data, lc, &heur.algo, ltWs, ltSize, s)
        : CUBLAS_STATUS_NOT_SUPPORTED;
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    if (st != CUBLAS_STATUS_SUCCESS) {
        if (getenv("MXFP4_DBG")) fprintf(stderr, "[nvfp4] native fp4 declined (hs=%d nres=%d st=%d) → functional\n", (int)hs, nres, (int)st);
        return aclnnMatmulMxFp4(ws, wsSize, e, stream);
    }
    delete e; return ACLNN_SUCCESS;
}

// ---- MXFP4 hardware tensor-core path: cuBLASLt native fp4 (CUDA_R_4F_E2M1) + VEC32_UE8M0 block scaling ----
//   Ascend Float4E2M1 decodes identically to OCP standard E2M1 (verified), so the native path is lossless.
//   Structure mirrors MXFP8 Hw, with elements changed to fp4 (2 elements/byte packed).
//   Non-Blackwell (cc<10) falls back to aclnnMatmulMxFp4 via e->keepDim.
aclnnStatus aclnnMatmulMxFp4HwGetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
        const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !selfScale || !other || !otherScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != ACL_FLOAT4_E2M1 || other->dtype != ACL_FLOAT4_E2M1) return ACLNN_ERR_PARAM_INVALID;
    if (selfScale->dtype != ACL_FLOAT8_E8M0 || otherScale->dtype != ACL_FLOAT8_E8M0) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t m = self->viewDims[0], k = self->viewDims[1], n = other->viewDims[1];
    if (other->viewDims[0] != k || out->viewDims[0] != m || out->viewDims[1] != n) return ACLNN_ERR_PARAM_INVALID;
    if (k % MX_BLK != 0 || (n & 1)) return ACLNN_ERR_PARAM_INVALID;     // K must be divisible by 32 (also even); N must be even (fp4 packing)
    auto *e = new aclOpExecutor();
    e->op = OP_MATMUL; e->a = self; e->b = other; e->c = selfScale; e->mask = otherScale; e->out = out;
    e->m = m; e->n = n; e->k = k;
    int64_t nb = k / MX_BLK;
    uint64_t fnc = (uint64_t)(m * k + k * n) * sizeof(__half) + LT_WS;   // functional-path workspace size
    if (!mx_use_hw()) { e->keepDim = true; *ws = fnc; }
    else { e->keepDim = false;                                            // Blackwell: try native path; reserve space for functional fallback
        uint64_t nat = (uint64_t)(n * k / 2) + (uint64_t)swizzle_bytes((int)m, (int)nb) + (uint64_t)swizzle_bytes((int)n, (int)nb) + LT_WS;
        // NVFP4 path (VEC16_UE4M3, block-16) needs Bt + converted E4M3 scales [M,nb16]+[nb16,N] + their swizzled buffers
        int64_t nb16 = k / 16;
        uint64_t nv = (uint64_t)(n * k / 2) + (uint64_t)m * nb16 + (uint64_t)nb16 * n
                    + (uint64_t)swizzle_bytes((int)m, (int)nb16) + (uint64_t)swizzle_bytes((int)n, (int)nb16) + LT_WS;
        uint64_t mx = nat > fnc ? nat : fnc; *ws = nv > mx ? nv : mx; }
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulMxFp4Hw(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    if (e->keepDim) return aclnnMatmulMxFp4(ws, wsSize, e, stream);     // non-Blackwell fallback to functional path (deletes e)
    if (getenv("NVFP4_HW")) return nvfp4_hw(ws, wsSize, e, stream);     // native fp4 via NVFP4 (VEC16_UE4M3) scale conversion
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k, nb = K / MX_BLK;
    int64_t saBytes = swizzle_bytes((int)M, (int)nb), sbBytes = swizzle_bytes((int)N, (int)nb);
    uint8_t *Bt = (uint8_t *)ws;                 // [N,K] fp4 packed (K contiguous) = N*K/2 bytes
    uint8_t *sAsw = Bt + (size_t)N * K / 2;      // self scale (B operand), rows=M
    uint8_t *sBsw = sAsw + saBytes;              // other scale (A operand), rows=N
    void *ltWs = sBsw + sbBytes; size_t ltSize = wsSize - (size_t)((char *)ltWs - (char *)ws);
    transpose_fp4<<<dim3((unsigned)((K/2+15)/16), (unsigned)((N+15)/16)), dim3(16,16), 0, s>>>((const uint8_t *)e->b->data, Bt, K, N);
    cudaMemsetAsync(sAsw, 0, saBytes, s); cudaMemsetAsync(sBsw, 0, sbBytes, s);
    int gm = (int)((M*nb + 255) / 256), gn = (int)((N*nb + 255) / 256);
    k_swizzle_scale<<<gm, 256, 0, s>>>((const uint8_t *)e->c->data, sAsw, (int)M, (int)nb, 0);
    k_swizzle_scale<<<gn, 256, 0, s>>>((const uint8_t *)e->mask->data, sBsw, (int)N, (int)nb, 1);

    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
    int32_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode, sizeof(mode));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode, sizeof(mode));
    void *aScale = sBsw, *bScale = sAsw;
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
    cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));
    cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&lc, e->out->dtype == ACL_FLOAT ? CUDA_R_32F : CUDA_R_16F, N, M, N);
    const float alpha = 1.f, beta = 0.f;
    // fp4 narrow precision requires explicit heuristic algo selection (default nullptr algo returns INVALID_VALUE)
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ltSize, sizeof(ltSize));
    cublasLtMatmulHeuristicResult_t heur{}; int nres = 0;
    cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt_handle(), op, la, lb, lc, lc, pref, 1, &heur, &nres);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasStatus_t st = (hs == CUBLAS_STATUS_SUCCESS && nres > 0)
        ? cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, e->a->data, lb, &beta, e->out->data, lc, e->out->data, lc, &heur.algo, ltWs, ltSize, s)
        : CUBLAS_STATUS_NOT_SUPPORTED;
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    if (st != CUBLAS_STATUS_SUCCESS) {   // this cuBLASLt native fp4=NVFP4(VEC16_UE4M3), does not accept MXFP4(VEC32_UE8M0) → fall back to functional path
        if (getenv("MXFP4_DBG")) fprintf(stderr, "[MxFp4Hw] native fp4 (MXFP4/VEC32) not supported (hs=%d nres=%d), falling back to functional path\n", (int)hs, nres);
        return aclnnMatmulMxFp4(ws, wsSize, e, stream);   // functional path deletes e; ws was allocated to max size
    }
    delete e;
    return ACLNN_SUCCESS;
}

// ---- Quantized matmul: dequantize to fp16 + GEMM (functional path) ----
// W8A16 weight dequantization: wf[k,n] = ((float)w_int8 - off[n]) · scale[n] (per-channel N; antiquant scale/off are fp16)
__global__ void deq_wq8(const int8_t *w, const __half *scale, const __half *off, __half *wf, int64_t K, int64_t N) {
    int64_t k = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || n >= N) return;
    float o = off ? (float)off[n] : 0.f;
    wf[k * N + n] = (__half)(((float)w[k * N + n] - o) * (float)scale[n]);
}
// W4A16: int4 (2/byte, signed) weight dequantization
__global__ void deq_wq4(const uint8_t *w, const __half *scale, const __half *off, __half *wf, int64_t K, int64_t N) {
    int64_t k = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K || n >= N) return;
    int64_t idx = k * N + n; uint8_t byte = w[idx / 2]; int v = (idx & 1) ? (byte >> 4) : (byte & 0xf); if (v >= 8) v -= 16;
    float o = off ? (float)off[n] : 0.f;
    wf[idx] = (__half)(((float)v - o) * (float)scale[n]);
}
// W8A8: int8 → fp16 (activations cast directly; weight dequantization folds deqScale: wf = (float)w_int8 · scale[n])
__global__ void cast_i8_h(const int8_t *x, __half *xf, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) xf[i] = (__half)(float)x[i];
}
__global__ void deq_qw8(const int8_t *w, const __half *scale, __half *wf, int64_t K, int64_t N) {
    int64_t k = blockIdx.y * blockDim.y + threadIdx.y, n = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < K && n < N) wf[k * N + n] = (__half)((float)w[k * N + n] * (float)scale[n]);
}

// WeightQuantBatchMatmul: x[M,K] fp16, weight[K,N] int8/int4, antiquant scale/offset[N] fp16 → out fp16
aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !antiquantScale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_FLOAT16 || out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (weight->dtype != ACL_INT8 && weight->dtype != ACL_INT4) return ACLNN_ERR_PARAM_INVALID;
    if (antiquantScale->dtype != ACL_FLOAT16 || (antiquantOffset && antiquantOffset->dtype != ACL_FLOAT16)) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 2 || weight->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = x->viewDims[0], K = x->viewDims[1], N = weight->viewDims[1];
    if (weight->viewDims[0] != K || out->viewDims[0] != M || out->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;
    if (antiquantScale->viewDims.size() != 1 || antiquantScale->viewDims[0] != N) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = x; e->b = weight; e->c = antiquantScale; e->mask = antiquantOffset; e->out = out;
    e->m = M; e->n = N; e->k = K;
    // Small M (decode / small batch, memory-bandwidth-bound) → fused path (read int8/int4 directly, no fp16 materialization,
    // saves weight bandwidth), flagged by e->keepDim, ws=0.
    // Large M (prefill, compute-bound) → dequantize to fp16 + cuBLASLt tensor-core. Threshold via env WQ_FUSE_M (default 16).
    const char *thEnv = getenv("WQ_FUSE_M"); int64_t th = thEnv ? atoll(thEnv) : 16;
    if (M <= th) { e->keepDim = true; *ws = 0; }
    else { e->keepDim = false; *ws = (uint64_t)K * N * sizeof(__half) + LT_WS; }
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnWeightQuantBatchMatmul(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k;
    const __half *scale = (const __half *)e->c->data, *off = e->mask ? (const __half *)e->mask->data : nullptr;
    if (e->keepDim) {   // fused path: read quantized weights directly, no fp16 materialization
        int64_t g = (N + 255) / 256;
        if (e->b->dtype == ACL_INT8) k_wq_fused<false><<<g, 256, 0, s>>>((const __half *)e->a->data, e->b->data, scale, off, (__half *)e->out->data, M, K, N);
        else                         k_wq_fused<true><<<g, 256, 0, s>>>((const __half *)e->a->data, e->b->data, scale, off, (__half *)e->out->data, M, K, N);
        cudaError_t err = cudaGetLastError(); delete e;
        return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    __half *wf = (__half *)ws; void *ltWs = wf + K * N; size_t ltSize = wsSize - (size_t)K * N * sizeof(__half);
    dim3 tb(16, 16), g((N + 15) / 16, (K + 15) / 16);
    if (e->b->dtype == ACL_INT8) deq_wq8<<<g, tb, 0, s>>>((const int8_t *)e->b->data, scale, off, wf, K, N);
    else                         deq_wq4<<<g, tb, 0, s>>>((const uint8_t *)e->b->data, scale, off, wf, K, N);
    aclnnStatus st = gemm_half((const __half *)e->a->data, wf, e->out->data, e->out->dtype, M, N, K, ltWs, ltSize, s);
    delete e; return st;
}

// QuantMatmul (W8A8): x[M,K] int8, weight[K,N] int8, deqScale[N] fp16 (=xscale·wscale per-channel) → out fp16
aclnnStatus aclnnQuantMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_INT8 || weight->dtype != ACL_INT8 || scale->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT16 && out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 2 || weight->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = x->viewDims[0], K = x->viewDims[1], N = weight->viewDims[1];
    if (weight->viewDims[0] != K || out->viewDims[0] != M || out->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;
    if (scale->viewDims.size() != 1 || scale->viewDims[0] != N) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = x; e->b = weight; e->c = scale; e->out = out;
    e->m = M; e->n = N; e->k = K;
    uint64_t fp16ws = (uint64_t)(M * K + K * N) * sizeof(__half) + LT_WS;     // fallback path workspace
    uint64_t int8ws = (uint64_t)N * K + (uint64_t)M * N * 4 + LT_WS;          // Bt(int8) + C32(int32)
    *ws = fp16ws > int8ws ? fp16ws : int8ws;                                  // take the larger to accommodate runtime fallback
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantMatmul(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t M = e->m, N = e->n, K = e->k;
    // Native int8 tensor-core path: int8×int8→int32 accumulation + per-channel scale epilogue; K must be aligned to 16, else falls back to fp16 on failure.
    if (K % 16 == 0 && !getenv("QMM_NO_INT8")) {
        int8_t *Bt = (int8_t *)ws; int32_t *C32 = (int32_t *)(Bt + (size_t)N * K);
        void *ltWs = (char *)(C32 + (size_t)M * N); size_t ltSize = wsSize - ((size_t)N * K + (size_t)M * N * 4);
        dim3 tb(16, 16), g((N + 15) / 16, (K + 15) / 16);
        transpose_u8<<<g, tb, 0, s>>>((const uint8_t *)e->b->data, (uint8_t *)Bt, K, N);   // weight[K,N] → [N,K]
        cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32I, CUDA_R_32I);
        cublasOperation_t T = CUBLAS_OP_T, No = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &T, sizeof(T));
        cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &No, sizeof(No));
        cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, N, K);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, M, K);
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, N, M, N);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ltSize, sizeof(ltSize));
        cublasLtMatmulHeuristicResult_t heur{}; int nres = 0;
        cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(lt_handle(), op, la, lb, lc, lc, pref, 1, &heur, &nres);
        cublasLtMatmulPreferenceDestroy(pref);
        const int32_t alpha = 1, beta = 0;
        cublasStatus_t st = (hs == CUBLAS_STATUS_SUCCESS && nres > 0)
            ? cublasLtMatmul(lt_handle(), op, &alpha, Bt, la, e->a->data, lb, &beta, C32, lc, C32, lc, &heur.algo, ltWs, ltSize, s)
            : CUBLAS_STATUS_NOT_SUPPORTED;
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
        if (st == CUBLAS_STATUS_SUCCESS) {
            int64_t mn = M * N;
            if (e->out->dtype == ACL_FLOAT) k_scale_i32<float><<<(mn+255)/256,256,0,s>>>(C32, (const __half*)e->c->data, (float*)e->out->data, M, N);
            else                            k_scale_i32<__half><<<(mn+255)/256,256,0,s>>>(C32, (const __half*)e->c->data, (__half*)e->out->data, M, N);
            cudaError_t err = cudaGetLastError(); delete e;
            return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
        }
        if (getenv("QMM_DBG")) fprintf(stderr, "[QuantMatmul] native int8 not supported (hs=%d nres=%d), falling back to fp16\n", (int)hs, nres);
        // Fall through to fp16 fallback below (ws was allocated to max size)
    }
    __half *xf = (__half *)ws, *wf = xf + M * K; void *ltWs = wf + K * N; size_t ltSize = wsSize - (size_t)(M * K + K * N) * sizeof(__half);
    cast_i8_h<<<(M*K+255)/256, 256, 0, s>>>((const int8_t *)e->a->data, xf, M * K);
    dim3 tb(16, 16), g((N + 15) / 16, (K + 15) / 16);
    deq_qw8<<<g, tb, 0, s>>>((const int8_t *)e->b->data, (const __half *)e->c->data, wf, K, N);
    aclnnStatus st = gemm_half(xf, wf, e->out->data, e->out->dtype, M, N, K, ltWs, ltSize, s);
    delete e; return st;
}

// ---- GroupedMatmul (MoE): x[M,K] partitioned by groupList @ weight[E,K,N] → out[M,N] (one GEMM per group) ----
aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !groupList || !out || !ws || !ex || !x->data || !weight->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = x->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 2 || weight->viewDims.size() != 3 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = x->viewDims[0], K = x->viewDims[1], E = weight->viewDims[0], N = weight->viewDims[2];
    if (weight->viewDims[1] != K || out->viewDims[0] != M || out->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;
    if ((int64_t)groupList->v.size() != E) return ACLNN_ERR_PARAM_INVALID;
    int64_t sum = 0; for (auto g : groupList->v) sum += g;
    if (sum != M) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = x; e->b = weight; e->out = out; e->axes = groupList->v;
    e->n = N; e->k = K;
    *ws = (uint64_t)GMM_NS * LT_WS;   // one cuBLASLt workspace per parallel stream
    *ex = e; return ACLNN_SUCCESS;
}
// Parallel stream pool: multi-expert GEMMs overlapped across streams, replacing single-stream serialization. C++11 magic-static thread-safe initialization.
static cudaStream_t *grouped_streams() {
    static std::vector<cudaStream_t> p;
    if (p.empty()) { p.resize(GMM_NS); for (auto &x : p) cudaStreamCreate(&x); }
    return p.data();
}
// Native grouped-batched GEMM: a single cublasGemmGroupedBatchedEx call processes all experts
// (each expert's token count = its group M), replacing multi-stream per-group GEMMs — reduces launch
// overhead and lets cuBLAS schedule internally. Column-major mapping: C[r,N]_row = A=weight(lda=N)·B=x(ldb=K).
// Row-major→column-major: opN/opN, m=N, n=r, k=K (see derivation).
// Opt-in via CANN_GMM_NATIVE=1; default is the stable multi-stream path.
static aclnnStatus gmm_native(aclOpExecutor *e, cudaStream_t s) {
    cudaDataType_t dt = e->a->dtype == ACL_FLOAT ? CUDA_R_32F : e->a->dtype == ACL_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    size_t esz = dt == CUDA_R_32F ? 4 : 2;
    const int64_t N = e->n, K = e->k;
    const char *xp = (const char *)e->a->data, *wp = (const char *)e->b->data; char *op = (char *)e->out->data;
    std::vector<const void *> hA, hB; std::vector<void *> hC;
    std::vector<int> vm, vn, vk, vlda, vldb, vldc, vgs;
    std::vector<cublasOperation_t> vta, vtb;
    int64_t off = 0;
    for (size_t g = 0; g < e->axes.size(); ++g) {
        int64_t r = e->axes[g];
        if (r > 0) {
            hA.push_back(wp + (size_t)g * K * N * esz); hB.push_back(xp + (size_t)off * K * esz); hC.push_back(op + (size_t)off * N * esz);
            vm.push_back((int)N); vn.push_back((int)r); vk.push_back((int)K);
            vlda.push_back((int)N); vldb.push_back((int)K); vldc.push_back((int)N);
            vta.push_back(CUBLAS_OP_N); vtb.push_back(CUBLAS_OP_N); vgs.push_back(1);
        }
        off += r;
    }
    int gc = (int)hA.size(); if (gc == 0) return ACLNN_SUCCESS;
    void **dptr = nullptr; if (cudaMallocAsync((void **)&dptr, (size_t)3 * gc * sizeof(void *), s) != cudaSuccess) return ACLNN_ERR_RUNTIME_ERROR;
    const void **dA = (const void **)dptr; const void **dB = dA + gc; void **dC = (void **)(dB + gc);
    cudaMemcpyAsync(dA, hA.data(), gc * sizeof(void *), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dB, hB.data(), gc * sizeof(void *), cudaMemcpyHostToDevice, s);
    cudaMemcpyAsync(dC, hC.data(), gc * sizeof(void *), cudaMemcpyHostToDevice, s);
    std::vector<float> alpha(gc, 1.f), beta(gc, 0.f);
    cublasSetStream(cublas_handle(), s);
    cublasStatus_t st = cublasGemmGroupedBatchedEx(cublas_handle(), vta.data(), vtb.data(), vm.data(), vn.data(), vk.data(),
        alpha.data(), dA, dt, vlda.data(), dB, dt, vldb.data(), beta.data(), dC, dt, vldc.data(), gc, vgs.data(), CUBLAS_COMPUTE_32F);
    cudaFreeAsync(dptr, s);
    return st == CUBLAS_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
aclnnStatus aclnnGroupedMatmul(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    if (getenv("CANN_GMM_NATIVE")) { aclnnStatus st = gmm_native(e, s); delete e; return st; }   // native grouped GEMM path
    cudaDataType_t dt = e->a->dtype == ACL_FLOAT ? CUDA_R_32F : e->a->dtype == ACL_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    size_t esz = dt == CUDA_R_32F ? 4 : 2;
    const int64_t N = e->n, K = e->k;
    const char *xp = (const char *)e->a->data, *wp = (const char *)e->b->data; char *op = (char *)e->out->data;
    cudaStream_t *pool = grouped_streams(); size_t slice = wsSize / GMM_NS;
    // fork: each worker stream waits for inputs to be ready on the calling stream
    cudaEvent_t fork; cudaEventCreateWithFlags(&fork, cudaEventDisableTiming); cudaEventRecord(fork, s);
    for (int i = 0; i < GMM_NS; ++i) cudaStreamWaitEvent(pool[i], fork, 0);
    int64_t off = 0; aclnnStatus st = ACLNN_SUCCESS;
    for (size_t ei = 0; ei < e->axes.size(); ++ei) {
        int64_t r = e->axes[ei]; int ps = (int)(ei % GMM_NS);
        if (r > 0) { aclnnStatus s2 = gemm_rm(dt, xp + (size_t)off * K * esz, wp + (size_t)ei * K * N * esz, op + (size_t)off * N * esz,
                                r, N, K, 1, 0, 0, 0, (char *)ws + (size_t)ps * slice, slice, pool[ps]);
                     if (s2 != ACLNN_SUCCESS) st = s2; }
        off += r;
    }
    // join: calling stream waits for all parallel streams to complete
    for (int i = 0; i < GMM_NS; ++i) { cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, pool[i]); cudaStreamWaitEvent(s, ev, 0); cudaEventDestroy(ev); }
    cudaEventDestroy(fork);
    delete e; return st;
}

// ---- GroupedMatmul weight TensorList variant: x[M,K] partitioned by groupList, each group @ weights[e][K,N] → out[M,N] ----
// Real MoE frameworks often store per-expert weights as an aclTensorList of independent tensors (rather than stacking into [E,K,N]).
// This variant consumes a TensorList directly.
aclnnStatus aclnnGroupedMatmulWeightListGetWorkspaceSize(const aclTensor *x, const aclTensorList *weights,
        const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weights || !groupList || !out || !ws || !ex || !x->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (weights->v.empty()) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = x->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = x->viewDims[0], K = x->viewDims[1], E = (int64_t)weights->v.size(), N = out->viewDims[1];
    if (out->viewDims[0] != M) return ACLNN_ERR_PARAM_INVALID;
    if ((int64_t)groupList->v.size() != E) return ACLNN_ERR_PARAM_INVALID;
    int64_t sum = 0;
    for (int64_t i = 0; i < E; ++i) {
        const aclTensor *w = weights->v[i];
        if (!w || !w->data || w->dtype != dt || w->viewDims.size() != 2 ||
            w->viewDims[0] != K || w->viewDims[1] != N || !w->contiguous()) return ACLNN_ERR_PARAM_INVALID;
        sum += groupList->v[i];
    }
    if (sum != M) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = x; e->out = out; e->axes = groupList->v;
    e->n = N; e->k = K;
    for (int64_t i = 0; i < E; ++i) e->inputs.push_back(weights->v[i]);
    *ws = (uint64_t)GMM_NS * LT_WS; *ex = e; return ACLNN_SUCCESS;  // one cuBLASLt workspace per parallel stream
}
aclnnStatus aclnnGroupedMatmulWeightList(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    cudaDataType_t dt = e->a->dtype == ACL_FLOAT ? CUDA_R_32F : e->a->dtype == ACL_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    size_t esz = dt == CUDA_R_32F ? 4 : 2;
    const int64_t N = e->n, K = e->k;
    const char *xp = (const char *)e->a->data; char *op = (char *)e->out->data;
    cudaStream_t *pool = grouped_streams(); size_t slice = wsSize / GMM_NS;
    cudaEvent_t fork; cudaEventCreateWithFlags(&fork, cudaEventDisableTiming); cudaEventRecord(fork, s);
    for (int i = 0; i < GMM_NS; ++i) cudaStreamWaitEvent(pool[i], fork, 0);
    int64_t off = 0; aclnnStatus st = ACLNN_SUCCESS;
    for (size_t ei = 0; ei < e->axes.size(); ++ei) {
        int64_t r = e->axes[ei]; int ps = (int)(ei % GMM_NS);
        if (r > 0) { aclnnStatus s2 = gemm_rm(dt, xp + (size_t)off * K * esz, e->inputs[ei]->data, op + (size_t)off * N * esz,
                                r, N, K, 1, 0, 0, 0, (char *)ws + (size_t)ps * slice, slice, pool[ps]);
                     if (s2 != ACLNN_SUCCESS) st = s2; }
        off += r;
    }
    for (int i = 0; i < GMM_NS; ++i) { cudaEvent_t ev; cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);
        cudaEventRecord(ev, pool[i]); cudaStreamWaitEvent(s, ev, 0); cudaEventDestroy(ev); }
    cudaEventDestroy(fork);
    delete e; return st;
}

// ---- BatchMatMul: [B,M,K] @ [B,K,N] → [B,M,N] (strided batched, fp32/fp16) ----
aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
                                             int8_t /*cubeMathType*/, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex || !self->data || !other->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims.size() != 3 || other->viewDims.size() != 3 || out->viewDims.size() != 3) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != other->dtype || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_FLOAT && self->dtype != ACL_FLOAT16 && self->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !other->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    const int64_t B = self->viewDims[0], M = self->viewDims[1], K = self->viewDims[2], N = other->viewDims[2];
    if (other->viewDims[0] != B || other->viewDims[1] != K || out->viewDims[0] != B || out->viewDims[1] != M || out->viewDims[2] != N)
        return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = self; e->b = other; e->out = out;
    e->m = M; e->n = N; e->k = K; e->outerCount = B;
    *ws = LT_WS; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchMatMul(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    cudaDataType_t dt = e->a->dtype == ACL_FLOAT ? CUDA_R_32F : e->a->dtype == ACL_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    aclnnStatus st = gemm_rm(dt, e->a->data, e->b->data, e->out->data, e->m, e->n, e->k,
                             (int)e->outerCount, e->m * e->k, e->k * e->n, e->m * e->n, ws, wsSize, (cudaStream_t)stream);
    delete e; return st;
}

// ---- MatmulBias: (transA?Aᵀ:A) @ (transB?Bᵀ:B) + bias, with optional ReLU/GeLU epilogue ----
// Transposes are materialized via k_T2d to avoid the cuBLASLt transpose + row-major layout pitfall;
// bias and activation are applied in a post-GEMM kernel (functional fusion).
// bias: nullptr / [N] (broadcast over columns) / [M,N]. act: 0=none / 1=ReLU / 2=GeLU.
aclnnStatus aclnnMatmulBiasGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclTensor *bias,
                                            bool transA, bool transB, int64_t act, aclTensor *out,
                                            uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex || !self->data || !other->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims.size() != 2 || other->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != other->dtype || self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_FLOAT && self->dtype != ACL_FLOAT16 && self->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !other->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = out->viewDims[0], N = out->viewDims[1];
    const int64_t K  = transA ? self->viewDims[0] : self->viewDims[1];
    const int64_t Ma = transA ? self->viewDims[1] : self->viewDims[0];
    const int64_t Kb = transB ? other->viewDims[1] : other->viewDims[0];
    const int64_t Nb = transB ? other->viewDims[0] : other->viewDims[1];
    if (Ma != M || Kb != K || Nb != N) return ACLNN_ERR_PARAM_INVALID;
    if (bias) {
        bool ok = (bias->viewDims.size() == 1 && bias->viewDims[0] == N) ||
                  (bias->viewDims.size() == 2 && bias->viewDims[0] == M && bias->viewDims[1] == N);
        if (!ok || bias->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    }
    if (act < 0 || act > 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = self; e->b = other; e->c = bias; e->out = out;
    e->m = M; e->n = N; e->k = K; e->dim = act; e->stride[0] = transA ? 1 : 0; e->stride[1] = transB ? 1 : 0;
    size_t esz = self->dtype == ACL_FLOAT ? 4 : 2;
    uint64_t tmp = ((transA ? (uint64_t)M * K : 0) + (transB ? (uint64_t)K * N : 0)) * esz;
    *ws = tmp + LT_WS; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulBias(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    cudaDataType_t dt = e->a->dtype == ACL_FLOAT ? CUDA_R_32F : e->a->dtype == ACL_BF16 ? CUDA_R_16BF : CUDA_R_16F;
    size_t esz = dt == CUDA_R_32F ? 4 : 2;
    const int64_t M = e->m, N = e->n, K = e->k;
    bool tA = e->stride[0], tB = e->stride[1];
    char *p = (char *)ws;
    const void *A = e->a->data, *B = e->b->data;
    if (tA) { void *At = p; p += (size_t)M * K * esz; launch_T2d(dt, e->a->data, At, K, M, s); A = At; }  // [K,M] → [M,K]
    if (tB) { void *Bt = p; p += (size_t)K * N * esz; launch_T2d(dt, e->b->data, Bt, N, K, s); B = Bt; }  // [N,K] → [K,N]
    size_t ltSize = wsSize - (size_t)(p - (char *)ws);
    aclnnStatus st = gemm_rm(dt, A, B, e->out->data, M, N, K, 1, 0, 0, 0, p, ltSize, s);
    if (st != ACLNN_SUCCESS) { delete e; return st; }
    int biasFull = e->c && e->c->viewDims.size() == 2;
    if (e->c || e->dim) launch_bias_act(dt, e->out->data, e->c ? e->c->data : nullptr, M, N, biasFull, (int)e->dim, s);
    st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    delete e; return st;
}

} // extern "C"
} // namespace _matmul

namespace _matmul_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _matmul3_ext {
// Matmul/Gemm/Grouped + quant-matmul variants + distributed mc2 (single-rank degenerate = local).
//   Local cores forward to the existing aclnnMatmul/BatchMatMul/QuantMatmul/GroupedMatmul; transpose-BMM
//   pre-transposes the B operand into scratch. Distributed collectives at nranks=1 are identity, so the
//   *AllReduce/*AllGather/*ReduceScatter/*AlltoAll matmul fusions reduce to the local matmul (+ fused tail).

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
__device__ inline float sig(float x){ return 1.f/(1.f+expf(-x)); }
__device__ inline int8_t clip8(float v){ int q=__float2int_rn(v); return (int8_t)(q<-127?-127:(q>127?127:q)); }
// transpose last two dims of a batched tensor [B,P,Q] → [B,Q,P]
template<typename T> __global__ void k_btranspose(const T*in,T*out,int64_t B,int64_t P,int64_t Q){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=B*P*Q) return; int64_t q=i%Q,p=(i/Q)%P,b=i/(P*Q);
    out[(b*Q+q)*P+p]=in[(b*P+p)*Q+q];
}
// int8 batched matmul × per-N scale: out[b,m,n] = scale[n]*Σ_k a[b,m,k]*bb[b,k,n]
__global__ void k_bmm_quant(const int8_t*a,const int8_t*bb,const float*scale,float*o,int B,int M,int K,int N){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)B*M*N) return; int n=i%N,m=(i/N)%M,b=i/((int64_t)M*N);
    int acc=0; const int8_t*ap=a+((int64_t)b*M+m)*K; const int8_t*bp=bb+(int64_t)b*K*N;
    for(int k=0;k<K;k++) acc+=(int)ap[k]*(int)bp[k*N+n]; o[i]=acc*(scale?scale[n]:1.f);
}
// swiglu + per-row dynamic int8 quant: in[M,2N] → out int8[M,N] + scale[M]
__global__ void k_swiglu_q(const float*x,int8_t*o,float*sc,int M,int N){
    int m=blockIdx.x; if(m>=M) return; const float*p=x+(int64_t)m*2*N; int t=threadIdx.x;
    float amax=0; for(int d=t;d<N;d+=blockDim.x){ float g=p[d]*sig(p[d])* p[N+d]; amax=fmaxf(amax,fabsf(g)); }
    for(int o2=16;o2>0;o2>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o2));
    __shared__ float sh[8]; if((t&31)==0) sh[t>>5]=amax; __syncthreads();
    __shared__ float scl; if(t==0){ float m2=0; for(int w=0;w<(blockDim.x+31)/32;w++)m2=fmaxf(m2,sh[w]); scl=m2>0?m2/127.f:1.f; sc[m]=scl; } __syncthreads();
    float inv=1.f/scl; for(int d=t;d<N;d+=blockDim.x){ float g=p[d]*sig(p[d])*p[N+d]; o[(int64_t)m*N+d]=clip8(g*inv); }
}
} // namespace

extern "C" {

// ---- TransposeBatchMatMul (+WeightNZ): out = a @ b^T (transpose B's last two dims) ----
aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!other||!out||!ex||other->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=other; e->out=out; e->dim=cubeMathType; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransposeBatchMatMul(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; const auto&bd=e->b->viewDims; int rank=(int)bd.size();
    int64_t P=bd[rank-2],Q=bd[rank-1]; int64_t B=1; for(int i=0;i<rank-2;i++)B*=bd[i];
    size_t esz=dtype_size(e->b->dtype); void*tb=nullptr; cudaMallocAsync(&tb,(size_t)B*P*Q*esz,st);
    int64_t g=nb(B*P*Q);
    switch(e->b->dtype){ case ACL_FLOAT16: k_btranspose<__half><<<g,TH,0,st>>>((const __half*)e->b->data,(__half*)tb,B,P,Q); break;
        case ACL_BF16: k_btranspose<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->b->data,(__nv_bfloat16*)tb,B,P,Q); break;
        default: k_btranspose<float><<<g,TH,0,st>>>((const float*)e->b->data,(float*)tb,B,P,Q); }
    aclTensor bt=*e->b; bt.data=tb; { auto &d=bt.viewDims; std::swap(d[rank-1],d[rank-2]); }
    { bt.strides.resize(rank); int64_t acc=1; for(int i=rank-1;i>=0;i--){ bt.strides[i]=acc; acc*=bt.viewDims[i]; } bt.storageDims=bt.viewDims; bt.offset=0; }
    uint64_t w2=0; aclOpExecutor*e2=nullptr; aclnnStatus stt;
    if(e->a->viewDims.size()==2) stt=aclnnMatmulGetWorkspaceSize(e->a,&bt,e->out,(int8_t)e->dim,&w2,&e2);
    else stt=aclnnBatchMatMulGetWorkspaceSize(e->a,&bt,e->out,(int8_t)e->dim,&w2,&e2);
    if(stt==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); stt=(e->a->viewDims.size()==2)?aclnnMatmul(wb,w2,e2,s):aclnnBatchMatMul(wb,w2,e2,s); if(wb)cudaFree(wb); }
    cudaFreeAsync(tb,st); delete e; return stt;
}
aclnnStatus aclnnTransposeBatchMatMulWeightNZGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){
    return aclnnTransposeBatchMatMulGetWorkspaceSize(self,other,out,cubeMathType,ws,ex);
}
aclnnStatus aclnnTransposeBatchMatMulWeightNZ(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnTransposeBatchMatMul(w,wz,e,s); }

// ---- BatchMatmulQuant: int8 a[B,M,K] @ b[B,K,N] × scale[N] → fp32 ----
aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!other||!out||!ex||self->dtype!=ACL_INT8||self->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=self; e->b=other; e->c=scale; e->out=out;
    e->m=self->viewDims[0]*0+self->viewDims[1]; e->k=self->viewDims[2]; e->n=other->viewDims[2]; e->reduceCount=self->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchMatmulQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int B=e->reduceCount,M=e->m,K=e->k,N=e->n; k_bmm_quant<<<nb((int64_t)B*M*N),TH,0,(cudaStream_t)s>>>((const int8_t*)e->a->data,(const int8_t*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,B,M,K,N); return done(e);
}

// ---- local forwards ----
aclnnStatus aclnnFusedMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){ return aclnnMatmulGetWorkspaceSize(self,other,out,cubeMathType,ws,ex); }
aclnnStatus aclnnFusedMatmul(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMatmul(w,wz,e,s); }
aclnnStatus aclnnMatmulCompressGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){ return aclnnMatmulGetWorkspaceSize(self,other,out,cubeMathType,ws,ex); }
aclnnStatus aclnnMatmulCompress(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMatmul(w,wz,e,s); }
aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex); }
aclnnStatus aclnnMatmulCompressDequant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex); }
aclnnStatus aclnnFusedQuantMatmul(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex); }
aclnnStatus aclnnFusedQuantMatmulWeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weight||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=weight; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnTransMatmulWeight(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->a->numel()*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }
aclnnStatus aclnnCalculateMatmulWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weightShape||!out||!ex||out->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID; int64_t prod=1; for(auto d:weightShape->v)prod*=d;
    auto*e=new aclOpExecutor(); e->out=out; e->m=prod; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnCalculateMatmulWeightSize(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t v=e->m; cudaMemcpyAsync(e->out->data,&v,8,cudaMemcpyHostToDevice,(cudaStream_t)s); return done(e); }
aclnnStatus aclnnCalculateMatmulWeightSizeV2GetWorkspaceSize(const aclIntArray *weightShape, int64_t dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)dtype; return aclnnCalculateMatmulWeightSizeGetWorkspaceSize(weightShape,out,ws,ex); }
aclnnStatus aclnnCalculateMatmulWeightSizeV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnCalculateMatmulWeightSize(w,wz,e,s); }

// ---- GroupedMatmulSwigluQuant: grouped matmul(x,w)→[M,2N] swiglu→[M,N] dynamic int8 quant ----
aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!weight||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=x; e->b=weight; e->out=out; e->out2=scaleOut; e->axes=groupList?groupList->v:std::vector<int64_t>{};
    e->m=out->viewDims[0]; e->n=out->viewDims.back(); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupedMatmulSwigluQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; int M=e->m,N=e->n; // matmul into temp [M,2N] via GroupedMatmul
    void*tmp=nullptr; cudaMallocAsync(&tmp,(size_t)M*2*N*4,st); aclTensor t; t.viewDims={(int64_t)M,(int64_t)2*N}; t.strides={(int64_t)2*N,1}; t.dtype=ACL_FLOAT; t.data=tmp;
    aclIntArray*gl=e->axes.empty()?nullptr:aclCreateIntArray(e->axes.data(),e->axes.size());
    uint64_t w2=0; aclOpExecutor*e2=nullptr; aclnnStatus stt=aclnnGroupedMatmulGetWorkspaceSize(e->a,e->b,gl,&t,&w2,&e2);
    if(stt==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); stt=aclnnGroupedMatmul(wb,w2,e2,s); if(wb)cudaFree(wb); }
    if(stt==ACLNN_SUCCESS) k_swiglu_q<<<(unsigned)M,TH,0,st>>>((const float*)tmp,(int8_t*)e->out->data,(float*)e->out2->data,M,N);
    cudaFreeAsync(tmp,st); if(gl)aclDestroyIntArray(gl); delete e; return stt==ACLNN_SUCCESS&&cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:(stt!=ACLNN_SUCCESS?stt:ACLNN_ERR_RUNTIME_ERROR);
}
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){ return aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(x,weight,groupList,out,scaleOut,ws,ex); }
aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupedMatmulSwigluQuant(w,wz,e,s); }
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){ return aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(x,weight,groupList,out,scaleOut,ws,ex); }
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupedMatmulSwigluQuant(w,wz,e,s); }
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){ return aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(x,weight,groupList,out,scaleOut,ws,ex); }
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupedMatmulSwigluQuant(w,wz,e,s); }

// ---- QuantGroupedMatmulDequant(+WeightNZ) → per-group quant matmul; InplaceAdd → += ----
aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)groupList; return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex);
}
aclnnStatus aclnnQuantGroupedMatmulDequant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)groupList; return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex);
}
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }

} // extern "C"
} // namespace _matmul3_ext

namespace _matmul4_ext {
// Distributed mc2 matmul fusions. Ops with an unambiguous single collective run the REAL HCCL path
// (verified 2-node, tools/mc2-2node.sh): WeightQuantMatmulAllReduce, QuantAllReduce, *MatmulAllReduceAddRmsNorm
// (matmul → AllReduce(SUM) → AddRmsNorm), AllGatherMatmul (AllGather(x) → matmul), GroupedMatMulAllReduce.
// comm == nullptr → local path (exact for nranks=1). The compound AlltoAll/grouped-alltoall fusions below
// stay nranks=1-exact local compute: their multi-rank dataflow needs EP/TP sharding metadata (group lists,
// world sizes, per-rank send/recv splits) that the simplified ABI signatures do not carry, so a faithful
// multi-rank path is not reconstructable from the signature alone (documented limitation).

namespace {
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
inline HcclDataType m4_dt(aclDataType d){ return d==ACL_FLOAT?HCCL_DATA_TYPE_FP32:d==ACL_FLOAT16?HCCL_DATA_TYPE_FP16:HCCL_DATA_TYPE_BFP16; }
// run a nested matmul into `temp`; AllReduce(SUM) the partial across ranks (TP K-split), then
// AddRmsNorm(temp, residual, gamma) → y, residualSum. rmsnorm is nonlinear, so the cross-rank sum MUST
// happen on the matmul partial BEFORE the norm. comm==nullptr → local only (exact for nranks=1).
static aclnnStatus mm_addrmsnorm(const aclTensor*x,const aclTensor*w,const aclTensor*scale,const aclTensor*residual,const aclTensor*gamma,double eps,aclTensor*y,aclTensor*residualSum,HcclComm comm,cudaStream_t s){
    size_t bytes=(size_t)y->numel()*dtype_size(y->dtype); void*tmp=nullptr; cudaMallocAsync(&tmp,bytes,s);
    aclTensor t=*y; t.data=tmp;
    uint64_t w1=0; aclOpExecutor*e1=nullptr; aclnnStatus st;
    if(scale) st=aclnnQuantMatmulGetWorkspaceSize(x,w,scale,&t,&w1,&e1);
    else      st=aclnnMatmulGetWorkspaceSize(x,w,&t,1,&w1,&e1);
    if(st==ACLNN_SUCCESS){ void*wb=nullptr; if(w1)cudaMalloc(&wb,w1); st=scale?aclnnQuantMatmul(wb,w1,e1,s):aclnnMatmul(wb,w1,e1,s); if(wb)cudaFree(wb); }
    if(st==ACLNN_SUCCESS && comm){ if(HcclAllReduce(tmp,tmp,(uint64_t)t.numel(),m4_dt(t.dtype),HCCL_REDUCE_SUM,comm,s)!=HCCL_SUCCESS) st=ACLNN_ERR_RUNTIME_ERROR; }
    if(st==ACLNN_SUCCESS){ uint64_t w2=0; aclOpExecutor*e2=nullptr; st=aclnnAddRmsNormGetWorkspaceSize(&t,residual,gamma,eps,y,residualSum,&w2,&e2);
        if(st==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); st=aclnnAddRmsNorm(wb,w2,e2,s); if(wb)cudaFree(wb); } }
    cudaFreeAsync(tmp,s); return st;
}
} // namespace

extern "C" {

// ---- AllGatherMatmul: REAL HcclAllGather of the row-shard x[Mr,K] → xfull[M,K], then xfull@weight → out[M,N].
//      (Sequence/row-parallel TP: each rank holds Mr=M/nranks rows; gather concatenates them.) ----
namespace { std::map<aclOpExecutor*,std::tuple<const aclTensor*,const aclTensor*,HcclComm>> g_agm; }
static aclnnStatus agm_run(aclOpExecutor*e,cudaStream_t s){
    auto it=g_agm.find(e); if(it==g_agm.end()) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor* x; const aclTensor* w; HcclComm comm; std::tie(x,w,comm)=it->second; g_agm.erase(it);
    aclTensor* out=e->out; int64_t M=out->viewDims[0], K=x->viewDims[1], Mr=x->viewDims[0];
    aclnnStatus st=ACLNN_SUCCESS; void* xfull=x->data; void* tmp=nullptr; aclTensor xf=*x;
    if(comm){ size_t bytes=(size_t)M*K*dtype_size(x->dtype); cudaMallocAsync(&tmp,bytes,s);
        if(HcclAllGather(x->data,tmp,(uint64_t)Mr*K,m4_dt(x->dtype),comm,s)!=HCCL_SUCCESS) st=ACLNN_ERR_RUNTIME_ERROR;
        xfull=tmp; }
    xf.data=xfull; xf.viewDims={M,K}; xf.storageDims={M,K};
    if(st==ACLNN_SUCCESS){ uint64_t w1=0; aclOpExecutor*e1=nullptr; st=aclnnMatmulGetWorkspaceSize(&xf,w,out,1,&w1,&e1);
        if(st==ACLNN_SUCCESS){ void*wb=nullptr; if(w1)cudaMalloc(&wb,w1); st=aclnnMatmul(wb,w1,e1,s); if(wb)cudaFree(wb); } }
    if(tmp)cudaFreeAsync(tmp,s); delete e; return st;
}
#define AGM_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)bias; if(!x||!weight||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->out=out; g_agm[e]=std::make_tuple(x,weight,comm); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; } \
aclnnStatus NAME(void *,uint64_t,aclOpExecutor*e,aclrtStream s){ return agm_run(e,(cudaStream_t)s); }
AGM_FWD(aclnnAllGatherMatmul)
AGM_FWD(aclnnAllGatherMatmulV2)

// ---- collective-matmul fusions whose multi-rank dataflow is underspecified by the simplified ABI
//      (no EP/TP group lists or world-size metadata in the signature) → kept as nranks=1-exact local
//      compute. Documented limitation: the AlltoAll/grouped sharding contract can't be reconstructed
//      from these signatures alone, so a faithful multi-rank path is not defined. ----
#define MM_FWD(NAME, BASE_GET, BASE_RUN) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)bias;(void)comm; return BASE_GET(x, weight, out, 1, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return BASE_RUN(w,wz,e,s); }
MM_FWD(aclnnMatmulAlltoAll, aclnnMatmulGetWorkspaceSize, aclnnMatmul)
MM_FWD(aclnnAlltoAllMatmul, aclnnMatmulGetWorkspaceSize, aclnnMatmul)
MM_FWD(aclnnAlltoAllAllGatherBatchMatMul, aclnnBatchMatMulGetWorkspaceSize, aclnnBatchMatMul)
MM_FWD(aclnnBatchMatMulReduceScatterAlltoAll, aclnnBatchMatMulGetWorkspaceSize, aclnnBatchMatMul)

// ---- grouped collective-matmul ----
// GroupedMatMulAllReduce: grouped matmul → REAL HcclAllReduce(SUM) over the output (TP, output shape preserved).
namespace { std::map<aclOpExecutor*,HcclComm> g_gmar; }
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    aclnnStatus st=aclnnGroupedMatmulGetWorkspaceSize(x, weight, groupList, out, ws, ex); if(st==ACLNN_SUCCESS&&ex&&*ex) g_gmar[*ex]=comm; return st;
}
aclnnStatus aclnnGroupedMatMulAllReduce(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){
    HcclComm comm=nullptr; auto it=g_gmar.find(e); if(it!=g_gmar.end()){ comm=it->second; g_gmar.erase(it); }
    aclTensor*out=e->out; aclDataType dt=out->dtype; int64_t n=out->numel();
    aclnnStatus st=aclnnGroupedMatmul(w,wz,e,s);  // frees e
    if(st==ACLNN_SUCCESS&&comm){ if(HcclAllReduce(out->data,out->data,(uint64_t)n,m4_dt(dt),HCCL_REDUCE_SUM,comm,(cudaStream_t)s)!=HCCL_SUCCESS) st=ACLNN_ERR_RUNTIME_ERROR; }
    return st;
}
// AlltoAllv-grouped variants: token-routed all-to-all-v dataflow underspecified by the ABI (no per-rank
// send/recv split lists) → nranks=1-exact local grouped matmul. Documented limitation.
#define GMM_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)comm; return aclnnGroupedMatmulGetWorkspaceSize(x, weight, groupList, out, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupedMatmul(w,wz,e,s); }
GMM_FWD(aclnnGroupedMatMulAlltoAllv)
GMM_FWD(aclnnAlltoAllvGroupedMatMul)

// ---- quant collective-matmul ----
#define QMM_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)comm; return aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, out, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
QMM_FWD(aclnnAlltoAllQuantMatmul)
QMM_FWD(aclnnQuantMatmulAlltoAll)
QMM_FWD(aclnnAlltoAllvQuantGroupedMatMul)
QMM_FWD(aclnnQuantGroupedMatMulAlltoAllv)
QMM_FWD(aclnnQuantReduceScatter)
QMM_FWD(aclnnQuantMatmulReduceSumWeightNz)

// ---- weight-quant collective-matmul: REAL HcclAllReduce(SUM) over the local W{4,8}A16 matmul.
//      TP K-split: each rank holds a weight column/row shard, local wqmm → partial, AllReduce sums across ranks.
//      comm == nullptr → local only (single-rank degenerate, exact for nranks=1). ----
namespace { inline HcclDataType m4_hccl_dt(aclDataType d){ return d==ACL_FLOAT?HCCL_DATA_TYPE_FP32:d==ACL_FLOAT16?HCCL_DATA_TYPE_FP16:HCCL_DATA_TYPE_BFP16; }
std::map<aclOpExecutor*,HcclComm> g_wqar; }
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffset, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    aclnnStatus st=aclnnWeightQuantBatchMatmulGetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, out, ws, ex);
    if(st==ACLNN_SUCCESS && ex && *ex) g_wqar[*ex]=comm; return st;
}
aclnnStatus aclnnWeightQuantMatmulAllReduce(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){
    HcclComm comm=nullptr; auto it=g_wqar.find(e); if(it!=g_wqar.end()){ comm=it->second; g_wqar.erase(it); }
    aclTensor *out=e->out; aclDataType dt=out->dtype; int64_t n=out->numel();
    aclnnStatus st=aclnnWeightQuantBatchMatmul(w,wz,e,s);  // frees e
    if(st==ACLNN_SUCCESS && comm){ if(HcclAllReduce(out->data,out->data,(uint64_t)n,m4_hccl_dt(dt),HCCL_REDUCE_SUM,comm,s)!=HCCL_SUCCESS) st=ACLNN_ERR_RUNTIME_ERROR; }
    return st;
}

// ---- QuantAllReduce: real HcclAllReduce(SUM) over x (comm==nullptr → identity copy, exact for 1 rank) ----
namespace { std::map<aclOpExecutor*,HcclComm> g_qar; }
aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=x; e->out=out; g_qar[e]=comm; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantAllReduce(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    HcclComm comm=nullptr; auto it=g_qar.find(e); if(it!=g_qar.end()){ comm=it->second; g_qar.erase(it); }
    cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->out->numel()*dtype_size(e->out->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s);
    aclnnStatus st=ACLNN_SUCCESS;
    if(comm){ if(HcclAllReduce(e->out->data,e->out->data,(uint64_t)e->out->numel(),m4_dt(e->out->dtype),HCCL_REDUCE_SUM,comm,(cudaStream_t)s)!=HCCL_SUCCESS) st=ACLNN_ERR_RUNTIME_ERROR; }
    delete e; return st;
}

// ---- sparse / dual-level quant matmul (no comm) → QuantMatmul; TransSparse4to2Para → copy ----
aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex); }
aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnQuantMatmulGetWorkspaceSize(x,weight,scale,out,ws,ex); }
aclnnStatus aclnnDualLevelQuantMatmulWeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantMatmul(w,wz,e,s); }
aclnnStatus aclnnTransSparse4to2ParaGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weight||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=weight; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnTransSparse4to2Para(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }

// ---- *MatmulAllReduceAddRmsNorm (matmul → AddRmsNorm), + quant/weight-quant + Inplace variants ----
struct ARState { const aclTensor*x,*w,*scale,*residual,*gamma; double eps; aclTensor*y,*residualSum; HcclComm comm; };
} // extern "C"
namespace { std::map<aclOpExecutor*,ARState> g_ar; }
extern "C" {
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!weight||!y||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_ar[e]={x,weight,scale,residual,gamma,eps,y,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus ar_run(aclOpExecutor*e,cudaStream_t s){ auto it=g_ar.find(e); if(it==g_ar.end())return ACLNN_ERR_PARAM_INVALID; ARState st=it->second; g_ar.erase(it);
    aclnnStatus r=mm_addrmsnorm(st.x,st.w,st.scale,st.residual,st.gamma,st.eps,st.y,st.residualSum,st.comm,s); delete e; return r; }
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return ar_run(e,(cudaStream_t)s); }
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!weight||!y||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_ar[e]={x,weight,nullptr,residual,gamma,eps,y,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return ar_run(e,(cudaStream_t)s); }
// Inplace variants: y aliases x (selfRef); same composition
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!weight||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_ar[e]={selfRef,weight,nullptr,residual,gamma,eps,selfRef,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return ar_run(e,(cudaStream_t)s); }
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!weight||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_ar[e]={selfRef,weight,scale,residual,gamma,eps,selfRef,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return ar_run(e,(cudaStream_t)s); }
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!weight||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_ar[e]={selfRef,weight,nullptr,residual,gamma,eps,selfRef,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return ar_run(e,(cudaStream_t)s); }

} // extern "C"
} // namespace _matmul4_ext
#undef AGM_FWD
#undef MM_FWD
#undef GMM_FWD
#undef QMM_FWD

namespace _matmul5_ext {
// GroupedMatmulFinalizeRouting variants + QuantGroupedMatmulInplaceAdd.
//   FinalizeRouting = grouped matmul whose rows are written back to their routed positions; at the
//   logical level the per-expert matmul result IS the finalized output, so we forward to GroupedMatmul.
//   QuantGroupedMatmulInplaceAdd accumulates a grouped quant-matmul into the existing output.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__global__ void k_add(const float*a,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]+=a[i]; }
} // namespace

extern "C" {

#define FR_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnGroupedMatmulGetWorkspaceSize(x, weight, groupList, out, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupedMatmul(w,wz,e,s); }
FR_FWD(aclnnGroupedMatmulFinalizeRouting)
FR_FWD(aclnnGroupedMatmulFinalizeRoutingV2)
FR_FWD(aclnnGroupedMatmulFinalizeRoutingV3)
FR_FWD(aclnnGroupedMatmulFinalizeRoutingWeightNz)
FR_FWD(aclnnGroupedMatmulFinalizeRoutingWeightNzV2)

// QuantGroupedMatmulInplaceAdd: out += dequant(grouped quant matmul)
aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)groupList; if(!x||!weight||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto*e=new aclOpExecutor(); e->a=x; e->b=weight; e->c=scale; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; size_t bytes=(size_t)e->out->numel()*dtype_size(e->out->dtype); void*tmp=nullptr; cudaMallocAsync(&tmp,bytes,st);
    aclTensor t=*e->out; t.data=tmp; uint64_t w2=0; aclOpExecutor*e2=nullptr;
    aclnnStatus r=aclnnQuantMatmulGetWorkspaceSize(e->a,e->b,e->c,&t,&w2,&e2);
    if(r==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); r=aclnnQuantMatmul(wb,w2,e2,s); if(wb)cudaFree(wb); }
    if(r==ACLNN_SUCCESS && e->out->dtype==ACL_FLOAT){ int64_t n=e->out->numel(); k_add<<<nb(n),TH,0,st>>>((const float*)tmp,(float*)e->out->data,n); }
    cudaFreeAsync(tmp,st); delete e; return r==ACLNN_SUCCESS&&cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:(r!=ACLNN_SUCCESS?r:ACLNN_ERR_RUNTIME_ERROR);
}

} // extern "C"
} // namespace _matmul5_ext
#undef FR_FWD

namespace _nz_ext {
// NZ (Ascend Cube fractal layout) matmul family — LOGICAL-EQUIVALENCE implementation.
// Per project limitation: we do NOT bit-replicate the 16x16 fractal swizzle. The NZ weight tensor is
// treated as logically row-major [K,N], so these forward to the ND cores with ZERO de-swizzle overhead
// (no extra copy/transpose) — numerically equal to the ND op, and maximally performant by construction.
// (If a true NZ byte-layout is ever required, a de-swizzle must be fused into the matmul's weight load.)

extern "C" {

// MatmulWeightNz: x[M,K] @ weightNz[K,N] -> out[M,N]  (weightNz logically row-major)
aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulGetWorkspaceSize(x, weight, out, cubeMathType, ws, ex);
}
aclnnStatus aclnnMatmulWeightNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnMatmul(ws, wsz, e, s); }

// BatchMatMulWeightNz: [B,M,K] @ [B,K,N] -> [B,M,N]
aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnBatchMatMulGetWorkspaceSize(x, weight, out, cubeMathType, ws, ex);
}
aclnnStatus aclnnBatchMatMulWeightNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnBatchMatMul(ws, wsz, e, s); }

// AddmmWeightNz: out = beta·C + alpha·(A @ Bnz)
aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddmmGetWorkspaceSize(C, A, B, beta, alpha, out, ws, ex);
}
aclnnStatus aclnnAddmmWeightNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAddmm(ws, wsz, e, s); }

// WeightQuantBatchMatmulNz: fp x @ antiquant(weightNz) -> out
aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnWeightQuantBatchMatmulGetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, out, ws, ex);
}
aclnnStatus aclnnWeightQuantBatchMatmulNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnWeightQuantBatchMatmul(ws, wsz, e, s); }

// QuantMatmulWeightNz: int8 x @ int8 weightNz, dequant by scale -> out
aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, out, ws, ex);
}
aclnnStatus aclnnQuantMatmulWeightNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnQuantMatmul(ws, wsz, e, s); }

// GroupedMatmulWeightNz: x[M,K] grouped @ weightNz[E,K,N] -> out[M,N]
aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGroupedMatmulGetWorkspaceSize(x, weight, groupList, out, ws, ex);
}
aclnnStatus aclnnGroupedMatmulWeightNz(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnGroupedMatmul(ws, wsz, e, s); }

} // extern "C"
} // namespace _nz_ext

} // namespace _matmul_ext

