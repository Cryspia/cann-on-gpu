// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cublasLt.h>
#include <mma.h>
#include "subfp.cuh"
#include <cmath>

namespace _attention {
// aclnnFlashAttentionScore (PoC, BNSD): scaled dot-product attention forward pass. Non-flash (S/P explicitly
//   materialized), numerically equivalent.
//   S = scale·Q·Kᵀ → (causal/mask) → softmax(along Skv)=P → O = P·V. S/P always fp32; I/O supports fp32/fp16/bf16.
//   Supports GQA/MQA (KV head count Nkv ≤ Nq, must divide evenly), shared [Sq,Skv] or per-batch [B,Sq,Skv] mask.
//   Matrix ops use hand-written naive batched kernels (avoiding cuBLASLt row/column-major + batch + transpose layout ambiguity).
//   Performance and backward pass are not goals.

namespace {
using namespace nvcuda;

// ---- Performance path: cuBLASLt batched tensor-core GEMM for QKᵀ and PV (replaces naive kernels); softmax reused ----
cublasLtHandle_t attn_lt() { static cublasLtHandle_t h = [] { cublasLtHandle_t t; cublasLtCreate(&t); return t; }(); return h; }

// Batched transpose: in[batch][R,C] → out[batch][C,R]
template <typename T>
__global__ void k_btrans(const T *in, T *out, int64_t R, int64_t C) {
    int64_t b = blockIdx.z, r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < R && c < C) out[b * C * R + c * R + r] = in[b * R * C + r * C + c];
}
template <typename T> __global__ void k_cast_sp(const float *s, T *p, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] = (T)s[i];
}
// Row-major strided-batched GEMM: A[M,K]@B[K,N]→C[M,N], input inDt, output outDt, fp32 accumulation
inline bool bgemm(cudaDataType_t inDt, cudaDataType_t outDt, const void *A, const void *B, void *C,
                  int64_t M, int64_t N, int64_t K, int batch, int64_t sA, int64_t sB, int64_t sC,
                  float alpha, void *ltWs, size_t ltSize, cudaStream_t s) {
    cublasLtMatmulDesc_t op; cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    cublasLtMatrixLayoutCreate(&la, inDt, M, K, K);
    cublasLtMatrixLayoutCreate(&lb, inDt, K, N, N);
    cublasLtMatrixLayoutCreate(&lc, outDt, M, N, N);
    int32_t bc = batch;
    for (auto l : {la, lb, lc}) cublasLtMatrixLayoutSetAttribute(l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row));
    cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc));
    cublasLtMatrixLayoutSetAttribute(la, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sA, sizeof(sA));
    cublasLtMatrixLayoutSetAttribute(lb, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sB, sizeof(sB));
    cublasLtMatrixLayoutSetAttribute(lc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &sC, sizeof(sC));
    const float beta = 0.f;
    cublasStatus_t st = cublasLtMatmul(attn_lt(), op, &alpha, A, la, B, lb, &beta, C, lc, C, lc, nullptr, ltWs, ltSize, s);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(op);
    return st == CUBLAS_STATUS_SUCCESS;
}

inline bool attn_is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool attn_is_fp6(aclDataType t) { return t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline bool attn_is_subfp(aclDataType t) { return attn_is_fp4(t) || attn_is_fp6(t); }
inline int attn_subkind(aclDataType t) {
    return t == ACL_FLOAT4_E2M1 ? SF_FP4E2M1 : t == ACL_FLOAT4_E1M2 ? SF_FP4E1M2 : t == ACL_FLOAT6_E2M3 ? SF_FP6E2M3 : SF_FP6E3M2;
}
// Unpack fp4 (2/byte) / fp6 (1/byte) → fp16
__global__ void deq_attn(const uint8_t *p, __half *o, int64_t n, int k, bool fp4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t code = fp4 ? ((i & 1) ? (p[i/2] >> 4) : (p[i/2] & 0xf)) : (p[i] & 0x3f);
    o[i] = (__half)subfp_decode(k, code);
}
// fp8-flash exploration: fp8 e4m3/e5m2 Q/K/V → decode to fp16 → reuse fp16 flash path (out fp16).
inline bool attn_is_fp8(aclDataType t) { return t == ACL_FLOAT8_E4M3FN || t == ACL_FLOAT8_E5M2; }
template <typename FP8> __global__ void deq_attn_fp8(const FP8 *p, __half *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = (__half)(float)p[i];
}
inline void deq_attn_any(const void *p, __half *o, int64_t n, aclDataType dt, cudaStream_t s) {
    int64_t g = (n + 255) / 256;
    if (attn_is_fp8(dt)) {
        if (dt == ACL_FLOAT8_E4M3FN) deq_attn_fp8<__nv_fp8_e4m3><<<g,256,0,s>>>((const __nv_fp8_e4m3*)p, o, n);
        else                         deq_attn_fp8<__nv_fp8_e5m2><<<g,256,0,s>>>((const __nv_fp8_e5m2*)p, o, n);
    } else {
        deq_attn<<<g,256,0,s>>>((const uint8_t*)p, o, n, attn_subkind(dt), attn_is_fp4(dt));
    }
}

// flat = bi*Nq + h (0..B*Nq-1); KV head = h/(Nq/Nkv), KV flat = bi*Nkv + kvh
__device__ inline int64_t kv_flat(int64_t flat, int64_t Nq, int64_t Nkv) {
    int64_t bi = flat / Nq, h = flat % Nq;
    return bi * Nkv + h / (Nq / Nkv);
}

template <typename T>
__global__ void qk_kernel(const T *Q, const T *K, float *S, int64_t Nq, int64_t Nkv,
                          int64_t Sq, int64_t Skv, int64_t D, float scale) {
    int64_t flat = blockIdx.z, i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Sq || j >= Skv) return;
    int64_t kf = kv_flat(flat, Nq, Nkv);
    const T *q = Q + (flat * Sq + i) * D, *k = K + (kf * Skv + j) * D;
    float acc = 0;
    for (int64_t d = 0; d < D; d++) acc += (float)q[d] * (float)k[d];
    S[(flat * Sq + i) * Skv + j] = acc * scale;
}

template <typename T>
__global__ void pv_kernel(const float *P, const T *V, T *O, int64_t Nq, int64_t Nkv,
                          int64_t Sq, int64_t Skv, int64_t D) {
    int64_t flat = blockIdx.z, i = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Sq || d >= D) return;
    int64_t kf = kv_flat(flat, Nq, Nkv);
    const float *p = P + (flat * Sq + i) * Skv;
    float acc = 0;
    for (int64_t j = 0; j < Skv; j++) acc += p[j] * (float)V[(kf * Skv + j) * D + d];
    O[(flat * Sq + i) * D + d] = (T)acc;
}

// One block per row [.,Skv], numerically stable softmax, with causal mask and optional explicit mask applied.
// maskBatchStride>0 indicates per-batch [B,Sq,Skv] mask.
__global__ void masked_softmax(float *S, const uint8_t *mask, int64_t Nq, int64_t Sq, int64_t Skv,
                               bool causal, int64_t maskBatchStride) {
    int64_t r = blockIdx.x;                 // rows = B*Nq*Sq
    int64_t qi = r % Sq, bi = (r / Sq) / Nq;
    int64_t off = Skv - Sq;                 // causal tail alignment offset
    float *row = S + r * Skv;
    const uint8_t *mr = mask ? mask + bi * maskBatchStride + qi * Skv : nullptr;
    __shared__ float smax[256], ssum[256];
    float m = -1e30f;
    for (int64_t j = threadIdx.x; j < Skv; j += 256) {
        bool blk = (causal && j > qi + off) || (mr && mr[j]);
        float v = blk ? -1e30f : row[j];
        row[j] = v; m = fmaxf(m, v);
    }
    smax[threadIdx.x] = m; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x + s]); __syncthreads(); }
    float mx = smax[0], sum = 0;
    for (int64_t j = threadIdx.x; j < Skv; j += 256) { float e = expf(row[j] - mx); row[j] = e; sum += e; }
    ssum[threadIdx.x] = sum; __syncthreads();
    for (int s = 128; s > 0; s >>= 1) { if (threadIdx.x < s) ssum[threadIdx.x] += ssum[threadIdx.x + s]; __syncthreads(); }
    float inv = 1.f / ssum[0];
    for (int64_t j = threadIdx.x; j < Skv; j += 256) row[j] *= inv;
}

// Flash attention: online softmax, no S/P materialization. grid=(Sq, B*Nq), block=D threads (each thread holds one head-dim d).
// Per row, iterates over KV; intra-block tree reduction yields the dot-product score s_j; running max/sum + correction
// term updates accumulator acc — O(D) memory per row instead of O(Skv).
// Supports GQA (kv_flat), causal (tail-aligned), optional [Sq,Skv]/[B,Sq,Skv] mask, fp32/fp16/bf16.
// D≤1024; intra-block tree reduction is correct for arbitrary D.
template <typename T>
__global__ void k_flash(const T *Q, const T *K, const T *V, const uint8_t *mask, T *O,
                        int64_t Nq, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
                        float scale, bool causal, int64_t maskBatchStride) {
    int64_t flat = blockIdx.y, i = blockIdx.x; int d = threadIdx.x;
    if (i >= Sq || d >= D) return;
    int64_t kf = kv_flat(flat, Nq, Nkv), bi = flat / Nq, off = Skv - Sq;
    const uint8_t *mr = mask ? mask + bi * maskBatchStride + i * Skv : nullptr;
    float qd = (float)Q[(flat * Sq + i) * D + d];
    extern __shared__ float red[];
    float m = -1e30f, l = 0.f, acc = 0.f;
    for (int64_t j = 0; j < Skv; j++) {
        red[d] = qd * (float)K[(kf * Skv + j) * D + d];
        __syncthreads();
        for (int s = 1; s < D; s *= 2) { if ((d % (2 * s)) == 0 && d + s < D) red[d] += red[d + s]; __syncthreads(); }
        float sc = red[0] * scale;
        bool blk = (causal && j > i + off) || (mr && mr[j]);
        float se = blk ? -1e30f : sc;
        float mn = fmaxf(m, se), corr = __expf(m - mn), p = __expf(se - mn);
        l = l * corr + p;
        acc = acc * corr + p * (float)V[(kf * Skv + j) * D + d];
        m = mn;
        __syncthreads();   // ensure all threads have read red[0] before the next j iteration overwrites it
    }
    O[(flat * Sq + i) * D + d] = (T)(acc / l);
}

// Flash attention fast warp path: one warp processes one query row; each lane holds D32=D/32 head-dim components.
// Dot product is reduced via warp shuffle (no __syncthreads; warp-synchronized); online softmax stays entirely in registers.
// Requires D%32==0 (D32 is a compile-time constant; register arrays ≤8). D=64/128/256 use this path; others fall back to k_flash generic.
template <typename T, int D32>
__global__ void k_flash_warp(const T *Q, const T *K, const T *V, const uint8_t *mask, T *O,
                             int64_t R, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
                             float scale, bool causal, int64_t maskBatchStride) {
    int64_t row = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;   // global warp index = query row
    int lane = threadIdx.x & 31;
    if (row >= R) return;                                  // R = B*Nq*Sq total rows
    int64_t i = row % Sq, flat = row / Sq;
    int64_t kf = kv_flat(flat, Nq, Nkv), bi = flat / Nq, off = Skv - Sq;
    const uint8_t *mr = mask ? mask + bi * maskBatchStride + i * Skv : nullptr;
    float qreg[D32], acc[D32];
    const T *qrow = Q + (flat * Sq + i) * D;
    #pragma unroll
    for (int k = 0; k < D32; k++) { qreg[k] = (float)qrow[lane + 32 * k]; acc[k] = 0.f; }
    float m = -1e30f, l = 0.f;
    for (int64_t j = 0; j < Skv; j++) {
        const T *krow = K + (kf * Skv + j) * D;
        float part = 0.f;
        #pragma unroll
        for (int k = 0; k < D32; k++) part += qreg[k] * (float)krow[lane + 32 * k];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
        float sc = __shfl_sync(0xffffffffu, part, 0) * scale;
        bool blk = (causal && j > i + off) || (mr && mr[j]);
        float se = blk ? -1e30f : sc;
        float mn = fmaxf(m, se), corr = __expf(m - mn), p = __expf(se - mn);
        l = l * corr + p;
        const T *vrow = V + (kf * Skv + j) * D;
        #pragma unroll
        for (int k = 0; k < D32; k++) acc[k] = acc[k] * corr + p * (float)vrow[lane + 32 * k];
        m = mn;
    }
    float inv = 1.f / l; T *orow = O + (flat * Sq + i) * D;
    #pragma unroll
    for (int k = 0; k < D32; k++) orow[lane + 32 * k] = (T)(acc[k] * inv);
}

// ---- Backward kernels (fp32, standard MHA: Nkv=Nq) ----
// dV[b,j,d] = Σ_i P[i,j]·dO[i,d]
__global__ void k_dv(const float *P, const float *dO, float *dV, int64_t Sq, int64_t Skv, int64_t D) {
    int64_t b = blockIdx.z, j = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Skv || d >= D) return;
    float acc = 0; for (int64_t i = 0; i < Sq; i++) acc += P[(b*Sq+i)*Skv+j] * dO[(b*Sq+i)*D+d];
    dV[(b*Skv+j)*D+d] = acc;
}
// dP[b,i,j] = Σ_d dO[i,d]·V[j,d]
__global__ void k_dp(const float *dO, const float *V, float *dP, int64_t Sq, int64_t Skv, int64_t D) {
    int64_t b = blockIdx.z, i = blockIdx.y * blockDim.y + threadIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Sq || j >= Skv) return;
    float acc = 0; for (int64_t d = 0; d < D; d++) acc += dO[(b*Sq+i)*D+d] * V[(b*Skv+j)*D+d];
    dP[(b*Sq+i)*Skv+j] = acc;
}
// Softmax backward (per row, overwrites dP with dS in-place): dS[i,j]=P[i,j]·(dP[i,j]-Σ_k P[i,k]dP[i,k])
__global__ void k_dsoftmax(const float *P, float *dP, int64_t Sq, int64_t Skv) {
    int64_t r = blockIdx.x; __shared__ float sh[256];
    const float *p = P + r * Skv; float *dp = dP + r * Skv;
    float s = 0; for (int64_t j = threadIdx.x; j < Skv; j += 256) s += p[j] * dp[j];
    sh[threadIdx.x] = s; __syncthreads();
    for (int t = 128; t > 0; t >>= 1) { if (threadIdx.x < t) sh[threadIdx.x] += sh[threadIdx.x+t]; __syncthreads(); }
    float dot = sh[0];
    for (int64_t j = threadIdx.x; j < Skv; j += 256) dp[j] = p[j] * (dp[j] - dot);
}
// dQ[b,i,d] = scale·Σ_j dS[i,j]·K[j,d]
__global__ void k_dq(const float *dS, const float *K, float *dQ, int64_t Sq, int64_t Skv, int64_t D, float scale) {
    int64_t b = blockIdx.z, i = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Sq || d >= D) return;
    float acc = 0; for (int64_t j = 0; j < Skv; j++) acc += dS[(b*Sq+i)*Skv+j] * K[(b*Skv+j)*D+d];
    dQ[(b*Sq+i)*D+d] = acc * scale;
}
// dK[b,j,d] = scale·Σ_i dS[i,j]·Q[i,d]
__global__ void k_dk(const float *dS, const float *Q, float *dK, int64_t Sq, int64_t Skv, int64_t D, float scale) {
    int64_t b = blockIdx.z, j = blockIdx.y * blockDim.y + threadIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= Skv || d >= D) return;
    float acc = 0; for (int64_t i = 0; i < Sq; i++) acc += dS[(b*Sq+i)*Skv+j] * Q[(b*Sq+i)*D+d];
    dK[(b*Skv+j)*D+d] = acc * scale;
}

constexpr uint64_t ATTN_LT_WS = 4u << 20;

// RoPE: rotates x[B,N,S,D] using cos/sin[S,D]. mode 0=half-split (LLaMA/NeoX), mode 1=interleaved (GPT-J).
template <typename T>
__global__ void k_rope(const T *x, const T *cos, const T *sin, T *o, int64_t N, int64_t S, int64_t D, int64_t total, int mode) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t d = i % D, s = (i / D) % S;   // (b,n,s,d) flat; cos/sin broadcast over B,N via [S,D]
    int64_t base = i - d;                  // start of this (b,n,s) row
    float c = (float)cos[s * D + d], si = (float)sin[s * D + d], xv = (float)x[i], rh;
    if (mode == 0) { int64_t half = D / 2; rh = (d < half) ? -(float)x[base + d + half] : (float)x[base + d - half]; }  // half-split
    else           { rh = (d & 1) ? (float)x[base + d - 1] : -(float)x[base + d + 1]; }  // interleaved
    o[i] = (T)(xv * c + rh * si);
}

// PagedAttention: paged KV + blockTable indirect addressing + online softmax (flash-style, no S/P materialization).
// query[B,Nq,Sq,D]; kCache/vCache[num_blocks, block_size, Nkv, D]; blockTable[B,maxBlocks] int32; ctxLen[B] int32.
template <typename T>
__global__ void k_paged(const T *q, const T *kC, const T *vC, const int32_t *blockTable, const int32_t *ctxLen,
                        T *o, int64_t total, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t blockSize, int64_t D, int64_t maxBlocks, float scale) {
    int64_t flat = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= total) return;
    int64_t i = flat % Sq, bh = flat / Sq, h = bh % Nq, b = bh / Nq;
    int64_t L = ctxLen[b]; if (L <= 0) return;
    int64_t kvh = h / (Nq / Nkv);
    const T *qp = q + (((b * Nq + h) * Sq) + i) * D;
    float acc[256]; for (int d = 0; d < D; ++d) acc[d] = 0.f;
    float m = -1e30f, l = 0.f;
    for (int64_t p = 0; p < L; ++p) {
        int32_t blk = blockTable[b * maxBlocks + p / blockSize]; int64_t off = p % blockSize;
        const T *kp = kC + ((blk * blockSize + off) * Nkv + kvh) * D;
        float score = 0; for (int d = 0; d < D; ++d) score += (float)qp[d] * (float)kp[d]; score *= scale;
        float mn = fmaxf(m, score), corr = expf(m - mn), e = expf(score - mn);
        l = l * corr + e;
        const T *vp = vC + ((blk * blockSize + off) * Nkv + kvh) * D;
        for (int d = 0; d < D; ++d) acc[d] = acc[d] * corr + e * (float)vp[d];
        m = mn;
    }
    T *op = o + (((b * Nq + h) * Sq) + i) * D;
    float inv = 1.f / l; for (int d = 0; d < D; ++d) op[d] = (T)(acc[d] * inv);
}

// PagedAttention fast warp path: one warp per query row; each lane holds D32=D/32 accumulators; warp-shuffle dot-product reduction;
// online softmax stays entirely in registers (replaces the scalar per-thread version's acc[256] local-memory spill + uncoalesced access). D%32==0.
template <typename T, int D32>
__global__ void k_paged_warp(const T *q, const T *kC, const T *vC, const int32_t *blockTable, const int32_t *ctxLen,
                             T *o, int64_t total, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t blockSize, int64_t D, int64_t maxBlocks, float scale) {
    int64_t flat = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (flat >= total) return;
    int lane = threadIdx.x & 31;
    int64_t i = flat % Sq, bh = flat / Sq, h = bh % Nq, b = bh / Nq;
    int64_t L = ctxLen[b]; if (L <= 0) return;
    int64_t kvh = h / (Nq / Nkv);
    const T *qp = q + (((b * Nq + h) * Sq) + i) * D;
    float qreg[D32], acc[D32];
    #pragma unroll
    for (int k = 0; k < D32; k++) { qreg[k] = (float)qp[lane + 32 * k]; acc[k] = 0.f; }
    float m = -1e30f, l = 0.f;
    for (int64_t p = 0; p < L; ++p) {
        int32_t blk = blockTable[b * maxBlocks + p / blockSize]; int64_t off = p % blockSize;
        const T *kp = kC + ((blk * blockSize + off) * Nkv + kvh) * D;
        float part = 0.f;
        #pragma unroll
        for (int k = 0; k < D32; k++) part += qreg[k] * (float)kp[lane + 32 * k];
        #pragma unroll
        for (int o2 = 16; o2 > 0; o2 >>= 1) part += __shfl_down_sync(0xffffffffu, part, o2);
        float score = __shfl_sync(0xffffffffu, part, 0) * scale;
        float mn = fmaxf(m, score), corr = __expf(m - mn), e = __expf(score - mn);
        l = l * corr + e;
        const T *vp = vC + ((blk * blockSize + off) * Nkv + kvh) * D;
        #pragma unroll
        for (int k = 0; k < D32; k++) acc[k] = acc[k] * corr + e * (float)vp[lane + 32 * k];
        m = mn;
    }
    float inv = 1.f / l; T *op = o + (((b * Nq + h) * Sq) + i) * D;
    #pragma unroll
    for (int k = 0; k < D32; k++) op[lane + 32 * k] = (T)(acc[k] * inv);
}

// Paged attention launch: D%32==0 uses the warp fast path; otherwise falls back to scalar k_paged
template <typename T>
static void launch_paged(const T *q, const T *kC, const T *vC, const int32_t *bt, const int32_t *cl, T *o,
                         int64_t total, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t blockSize, int64_t D, int64_t maxBlocks, float scale, cudaStream_t s) {
    if (D % 32 == 0 && D <= 256 && !getenv("PAGED_SCALAR")) {   // env var for benchmark comparison: forces scalar fallback
        const int TB = 128; dim3 g((unsigned)((total * 32 + TB - 1) / TB));
        #define LP(D32) k_paged_warp<T,D32><<<g,TB,0,s>>>(q,kC,vC,bt,cl,o,total,Nq,Nkv,Sq,blockSize,D,maxBlocks,scale)
        if      (D == 32)  LP(1); else if (D == 64)  LP(2); else if (D == 96)  LP(3); else if (D == 128) LP(4);
        else if (D == 160) LP(5); else if (D == 192) LP(6); else if (D == 224) LP(7); else if (D == 256) LP(8);
        #undef LP
    } else {
        int64_t g = (total + 127) / 128;
        k_paged<T><<<g,128,0,s>>>(q,kC,vC,bt,cl,o,total,Nq,Nkv,Sq,blockSize,D,maxBlocks,scale);
    }
}

// Performance path execution: btrans K→Kt → batched GEMM Q@Kt→S (fp32, ×scale) → softmax → cast S→P → batched GEMM P@V→O
template <typename T>
aclnnStatus run_perf(aclOpExecutor *e, cudaDataType_t inDt, void *ws, uint64_t wsSize, cudaStream_t s) {
    const int64_t B = e->ab, N = e->an, Sq = e->asq, Skv = e->askv, D = e->ad, flat = B * N;
    T *Kt = (T *)ws;
    float *S = (float *)(Kt + flat * Skv * D);
    T *P = (T *)(S + flat * Sq * Skv);
    void *ltWs = (void *)(P + flat * Sq * Skv);
    size_t ltSize = wsSize - (size_t)((char *)ltWs - (char *)ws);
    const uint8_t *mask = e->mask ? (const uint8_t *)e->mask->data : nullptr;
    dim3 tb(16, 16);
    k_btrans<T><<<dim3((D+15)/16,(Skv+15)/16,flat), tb, 0, s>>>((const T *)e->b->data, Kt, Skv, D);
    if (!bgemm(inDt, CUDA_R_32F, e->a->data, Kt, S, Sq, Skv, D, (int)flat, Sq*D, D*Skv, Sq*Skv, (float)e->alpha, ltWs, ltSize, s)) { delete e; return ACLNN_ERR_RUNTIME_ERROR; }
    masked_softmax<<<flat*Sq, 256, 0, s>>>(S, mask, N, Sq, Skv, e->causal, e->outerCount);
    k_cast_sp<T><<<(flat*Sq*Skv+255)/256, 256, 0, s>>>(S, P, flat*Sq*Skv);
    if (!bgemm(inDt, inDt, P, e->c->data, e->out->data, Sq, D, Skv, (int)flat, Sq*Skv, Skv*D, Sq*D, 1.f, ltWs, ltSize, s)) { delete e; return ACLNN_ERR_RUNTIME_ERROR; }
    cudaError_t err = cudaGetLastError(); delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// Flash block variant: one block of W warps handles W query rows for the same (b,head); K/V loaded cooperatively
// into shared memory in TILE chunks, reused by all W warps → K/V global traffic reduced by W× (independent of L2).
// Each warp's per-row online softmax stays in registers; dot product uses warp-shuffle.
template <typename T, int D32, int W, int TILE>
__global__ void k_flash_block(const T *Q, const T *K, const T *V, const uint8_t *mask, T *O,
                              int64_t Nq, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
                              float scale, bool causal, int64_t maskStride) {
    extern __shared__ float sh[]; float *sKf = sh, *sVf = sh + TILE * D;
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int64_t flat = blockIdx.y, i = (int64_t)blockIdx.x * W + warp;
    int64_t kf = kv_flat(flat, Nq, Nkv), bi = flat / Nq, off = Skv - Sq;
    bool active = (i < Sq);
    const uint8_t *mr = (mask && active) ? mask + bi * maskStride + i * Skv : nullptr;
    float qreg[D32], acc[D32]; float m = -1e30f, l = 0.f;
    if (active) { const T *qp = Q + (flat * Sq + i) * D;
        #pragma unroll
        for (int k = 0; k < D32; k++) { qreg[k] = (float)qp[lane + 32 * k]; acc[k] = 0.f; } }
    int ntiles = (int)((Skv + TILE - 1) / TILE);
    for (int t = 0; t < ntiles; t++) {
        int64_t jbase = (int64_t)t * TILE;
        for (int idx = threadIdx.x; idx < TILE * D; idx += W * 32) {   // cooperatively load K/V tile
            int jj = idx / D, dd = idx % D; int64_t j = jbase + jj;
            if (j < Skv) { sKf[idx] = (float)K[(kf * Skv + j) * D + dd]; sVf[idx] = (float)V[(kf * Skv + j) * D + dd]; }
            else { sKf[idx] = 0.f; sVf[idx] = 0.f; }
        }
        __syncthreads();
        if (active) {
            for (int jt = 0; jt < TILE; jt++) { int64_t j = jbase + jt; if (j >= Skv) break;
                float part = 0;
                #pragma unroll
                for (int k = 0; k < D32; k++) part += qreg[k] * sKf[jt * D + lane + 32 * k];
                #pragma unroll
                for (int o = 16; o > 0; o >>= 1) part += __shfl_down_sync(0xffffffffu, part, o);
                float score = __shfl_sync(0xffffffffu, part, 0) * scale;
                bool blk = (causal && j > i + off) || (mr && mr[j]);
                float se = blk ? -1e30f : score, mn = fmaxf(m, se), corr = __expf(m - mn), p = __expf(se - mn);
                l = l * corr + p;
                #pragma unroll
                for (int k = 0; k < D32; k++) acc[k] = acc[k] * corr + p * sVf[jt * D + lane + 32 * k];
                m = mn;
            }
        }
        __syncthreads();
    }
    if (active) { float inv = 1.f / l; T *op = O + (flat * Sq + i) * D;
        #pragma unroll
        for (int k = 0; k < D32; k++) op[lane + 32 * k] = (T)(acc[k] * inv); }
}

// Flash attention launch (D≤1024): D%32==0 uses block variant (K/V shared reuse) or warp variant (FLASH_NO_TILE); otherwise generic
template <typename T>
static void launch_flash(const T *Q, const T *K, const T *V, const uint8_t *mask, T *O,
                         int64_t B, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
                         float scale, bool causal, int64_t maskStride, cudaStream_t s) {
    int64_t R = B * Nq * Sq;
    if (D % 32 == 0 && D <= 256 && !getenv("FLASH_NO_TILE")) {        // block variant: K/V shared-memory reuse
        const int W = 4, TILE = 16;
        dim3 g((unsigned)((Sq + W - 1) / W), (unsigned)(B * Nq));
        size_t sh = (size_t)TILE * D * 2 * sizeof(float);
        #define LB(D32) k_flash_block<T, D32, W, TILE><<<g, W*32, sh, s>>>(Q, K, V, mask, O, Nq, Nkv, Sq, Skv, D, scale, causal, maskStride)
        if      (D == 64)  LB(2); else if (D == 128) LB(4); else if (D == 256) LB(8); else if (D == 32) LB(1);
        else if (D == 96)  LB(3); else if (D == 160) LB(5); else if (D == 192) LB(6); else if (D == 224) LB(7);
        #undef LB
    } else if (D % 32 == 0 && D <= 256) {                            // warp variant (FLASH_NO_TILE comparison)
        const int TB = 128, WPB = TB / 32;
        dim3 g((unsigned)((R + WPB - 1) / WPB));
        #define LF(D32) k_flash_warp<T, D32><<<g, TB, 0, s>>>(Q, K, V, mask, O, R, Nq, Nkv, Sq, Skv, D, scale, causal, maskStride)
        if      (D == 64)  LF(2);
        else if (D == 128) LF(4);
        else if (D == 256) LF(8);
        else if (D == 32)  LF(1);
        else if (D == 96)  LF(3);
        else if (D == 160) LF(5);
        else if (D == 192) LF(6);
        else if (D == 224) LF(7);
        #undef LF
    } else {
        dim3 g((unsigned)Sq, (unsigned)(B * Nq));
        k_flash<T><<<g, (unsigned)D, D * sizeof(float), s>>>(Q, K, V, mask, O, Nq, Nkv, Sq, Skv, D, scale, causal, maskStride);
    }
}

// Flash attention tensor-core path: WMMA 16×16×16 fp16 for Q@Kᵀ (fp32 accumulation); online softmax
// rescales across KV tiles; PV uses fp32 scalar (fp16-quantized P has excessive error ~2.6e-2, so P is kept fp32).
// Restricted to fp16 + D=64 + standard MHA (Nkv=Nq) + no explicit mask (otherwise falls back to warp/generic).
// 1 warp/block processes 16 query rows. QKᵀ: A=Q row_major, B=Kᵀ explicitly transposed into sKt[d][key] then row_major.
// ~1.27× faster than the scalar warp flash path.
template <bool FAST_PV>
__global__ void k_flash_wmma(const __half *Q, const __half *K, const __half *V, __half *O,
                             int64_t Sq, int64_t Skv, float scale, bool causal) {
    const int D = 64;
    int64_t flat = blockIdx.y, qt = blockIdx.x; int q0 = (int)(qt * 16), lane = threadIdx.x;
    const __half *Qb = Q + flat * Sq * D, *Kb = K + flat * Skv * D, *Vb = V + flat * Skv * D;
    __shared__ __half sQ[16 * 64], sKt[64 * 16], sV[16 * 64], sP[16 * 16];   // sKt=Kᵀ[d][key]; sP=fp16 P (FAST_PV only)
    __shared__ float sS[16 * 16], sO[16 * 64], sm[16], sl[16], sPf[16 * 16], sPV[16 * 16];
    for (int idx = lane; idx < 16 * 64; idx += 32) { int r = idx / 64, d = idx % 64; int64_t qq = q0 + r;
        sQ[idx] = (qq < Sq) ? Qb[qq * D + d] : __float2half(0.f); sO[idx] = 0.f; }
    if (lane < 16) { sm[lane] = -1e30f; sl[lane] = 0.f; }
    __syncthreads();
    int off = (int)(Skv - Sq), ktiles = (int)((Skv + 15) / 16);
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbK;
    for (int kt = 0; kt < ktiles; kt++) {
        int k0 = kt * 16;
        for (int idx = lane; idx < 16 * 64; idx += 32) { int r = idx / 64, d = idx % 64; int kk = k0 + r;
            __half kv = (kk < Skv) ? Kb[kk * D + d] : __float2half(0.f);
            sKt[d * 16 + r] = kv;                                   // explicit transpose: Kᵀ[d][key]
            sV[idx] = (kk < Skv) ? Vb[kk * D + d] : __float2half(0.f); }
        __syncthreads();
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc; wmma::fill_fragment(fc, 0.f);
        for (int ks = 0; ks < 4; ks++) {
            wmma::load_matrix_sync(fa, sQ + ks * 16, 64);
            wmma::load_matrix_sync(fbK, sKt + ks * 16 * 16, 16);    // Kᵀ sub-tile [16d×16key] row_major
            wmma::mma_sync(fc, fa, fbK, fc);
        }
        wmma::store_matrix_sync(sS, fc, 16, wmma::mem_row_major);
        __syncthreads();
        if (lane < 16) {                       // per-row online softmax (16 rows handled by 16 lanes)
            int r = lane; int64_t qq = q0 + r;
            if (qq < Sq) {
                float mx = -1e30f;
                for (int c = 0; c < 16; c++) { int kk = k0 + c; float v = sS[r * 16 + c] * scale;
                    bool blk = kk >= Skv || (causal && kk > qq + off); v = blk ? -1e30f : v; sS[r * 16 + c] = v; mx = fmaxf(mx, v); }
                float mold = sm[r], mnew = fmaxf(mold, mx), corr = __expf(mold - mnew), rs = 0;
                for (int c = 0; c < 16; c++) { float e = __expf(sS[r * 16 + c] - mnew);
                    if (FAST_PV) sP[r * 16 + c] = __float2half(e); else sPf[r * 16 + c] = e; rs += e; }
                sl[r] = sl[r] * corr + rs; sm[r] = mnew;
                for (int d = 0; d < 64; d++) sO[r * 64 + d] *= corr;
            } else for (int c = 0; c < 16; c++) { if (FAST_PV) sP[r * 16 + c] = __float2half(0.f); else sPf[r * 16 + c] = 0.f; }
        }
        __syncthreads();
        if (FAST_PV) {   // PV via fp16 WMMA tensor-core (standard fp16 flash-attn, P quantized to fp16)
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> faP;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> fbV;
            wmma::load_matrix_sync(faP, sP, 16);
            for (int ns = 0; ns < 4; ns++) {
                wmma::fragment<wmma::accumulator, 16, 16, 16, float> fpv; wmma::fill_fragment(fpv, 0.f);
                wmma::load_matrix_sync(fbV, sV + ns * 16, 64);
                wmma::mma_sync(fpv, faP, fbV, fpv);
                wmma::store_matrix_sync(sPV, fpv, 16, wmma::mem_row_major);
                __syncthreads();
                for (int idx = lane; idx < 16 * 16; idx += 32) { int r = idx / 16, c = idx % 16; sO[r * 64 + ns * 16 + c] += sPV[idx]; }
                __syncthreads();
            }
        } else {         // PV fp32 scalar (P kept as fp32, exact)
            for (int idx = lane; idx < 16 * 64; idx += 32) { int r = idx / 64, d = idx % 64;
                float acc = 0; for (int c = 0; c < 16; c++) acc += sPf[r * 16 + c] * (float)sV[c * 64 + d];
                sO[r * 64 + d] += acc; }
            __syncthreads();
        }
    }
    if (lane < 16) { int r = lane; int64_t qq = q0 + r; if (qq < Sq) { float inv = 1.f / sl[r];
        for (int d = 0; d < 64; d++) O[(flat * Sq + qq) * D + d] = __float2half(sO[r * 64 + d] * inv); } }
}

} // namespace

extern "C" {

aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
                                                     const aclTensor *attenMask, double scaleValue, int64_t headNum,
                                                     bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !v || !out || !ws || !ex || !q->data || !k->data || !v->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = q->dtype;
    bool sub = attn_is_subfp(dt) || attn_is_fp8(dt);   // fp4/fp6/fp8 inputs: unpack to fp16 then run fp16 path, out must be fp16
    if (sub) { if (k->dtype != dt || v->dtype != dt || out->dtype != ACL_FLOAT16) return ACLNN_ERR_PARAM_INVALID; }
    else if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || k->dtype != dt || v->dtype != dt || out->dtype != dt)
        return ACLNN_ERR_PARAM_INVALID;
    if (q->viewDims.size() != 4 || k->viewDims.size() != 4 || v->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    if (!q->contiguous() || !k->contiguous() || !v->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    const int64_t B = q->viewDims[0], Nq = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3];
    const int64_t Nkv = k->viewDims[1], Skv = k->viewDims[2];
    if (k->viewDims[0] != B || k->viewDims[3] != D) return ACLNN_ERR_PARAM_INVALID;
    if (v->viewDims[0] != B || v->viewDims[1] != Nkv || v->viewDims[2] != Skv || v->viewDims[3] != D) return ACLNN_ERR_PARAM_INVALID;
    if (Nkv <= 0 || Nq % Nkv != 0) return ACLNN_ERR_PARAM_INVALID;       // GQA/MQA: Nq must be an integer multiple of Nkv
    if (out->viewDims != std::vector<int64_t>{B, Nq, Sq, D}) return ACLNN_ERR_PARAM_INVALID;
    if (headNum != 0 && headNum != Nq) return ACLNN_ERR_PARAM_INVALID;
    int64_t maskBatchStride = 0;
    if (attenMask) {
        if (attenMask->dtype != ACL_BOOL && attenMask->dtype != ACL_UINT8) return ACLNN_ERR_PARAM_INVALID;
        if (attenMask->viewDims == std::vector<int64_t>{Sq, Skv}) maskBatchStride = 0;            // shared mask
        else if (attenMask->viewDims == std::vector<int64_t>{B, Sq, Skv}) maskBatchStride = Sq * Skv;  // per-batch mask
        else return ACLNN_ERR_PARAM_INVALID;
    }
    auto *e = new aclOpExecutor();
    e->op = OP_ATTENTION; e->a = q; e->b = k; e->c = v; e->out = out; e->mask = attenMask;
    e->ab = B; e->an = Nq; e->asq = Sq; e->askv = Skv; e->ad = D; e->reduceCount = Nkv; e->outerCount = maskBatchStride;
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / sqrt((double)D);
    e->causal = causal; e->castTo = sub ? dt : ACL_DT_UNDEFINED;
    // Auto-dispatch: only fp32 standard MHA is routed to the cuBLASLt batched GEMM performance path —
    // fp32 flash has no tensor-core, so batched GEMM is faster and numerically equivalent (~1e-6).
    // fp16/bf16 are not dispatched: WMMA flash is already fast and more accurate (5e-4 vs perf's ~3e-2),
    // and uses O(D) memory (perf's O(S²) would overflow on long contexts).
    // Cost: materializes S/P; falls back to flash when workspace exceeds cap (default 1GB, override with ATTN_PERF_CAP). e->keepDim flags this.
    e->keepDim = false;
    // Size threshold: only dispatch when there is real scale (cuBLASLt batched GEMM rejects very small configs; flash is sufficient for tiny sizes)
    if (!sub && dt == ACL_FLOAT && Nkv == Nq && Sq >= 16 && Skv >= 16 && D >= 16 && !getenv("ATTN_NO_PERF")) {
        int64_t flat = B * Nq, esz = (dt == ACL_FLOAT) ? 4 : 2;
        uint64_t perfWs = (uint64_t)(flat*Skv*D*esz + flat*Sq*Skv*4 + flat*Sq*Skv*esz) + ATTN_LT_WS;
        const char *capEnv = getenv("ATTN_PERF_CAP"); uint64_t cap = capEnv ? strtoull(capEnv, nullptr, 10) : (1ull << 30);
        if (perfWs <= cap) { e->keepDim = true; *ws = perfWs; *ex = e; return ACLNN_SUCCESS; }
    }
    // Flash path (D≤1024) does not materialize S, requiring no workspace; only fp4/fp6 need a decode buffer.
    // D>1024 falls back to naive path which needs S.
    uint64_t declBytes = sub ? (uint64_t)(B * Nq * Sq * D + 2 * B * Nkv * Skv * D) * sizeof(__half) : 0;  // Qh+Kh+Vh
    uint64_t sBytes = (D <= 1024) ? 0 : (uint64_t)(B * Nq * Sq * Skv) * sizeof(float);
    *ws = declBytes + sBytes;
    *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnFlashAttentionScore(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_ATTENTION) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    if (e->keepDim) {   // auto-dispatched to cuBLASLt perf path (run_perf deletes e)
        return e->a->dtype == ACL_FLOAT ? run_perf<float>(e, CUDA_R_32F, ws, wsSize, s)
                                        : run_perf<__half>(e, CUDA_R_16F, ws, wsSize, s);
    }
    const int64_t B = e->ab, Nq = e->an, Sq = e->asq, Skv = e->askv, D = e->ad, Nkv = e->reduceCount, flat = B * Nq;
    const void *Q = e->a->data, *K = e->b->data, *V = e->c->data; void *O = e->out->data;
    const uint8_t *mask = e->mask ? (const uint8_t *)e->mask->data : nullptr;
    const float sc = (float)e->alpha; const bool causal = e->causal; const int64_t ms = e->outerCount;
    const bool flashable = (D <= 1024);
    aclDataType dt = e->a->dtype;

    if (attn_is_subfp(e->castTo) || attn_is_fp8(e->castTo)) {   // fp4/fp6/fp8: unpack Q/K/V to fp16 first, then run fp16 flash/naive
        int64_t Qn = B * Nq * Sq * D, KVn = B * Nkv * Skv * D;
        __half *Qh = (__half *)ws, *Kh = Qh + Qn, *Vh = Kh + KVn;
        deq_attn_any(Q, Qh, Qn, e->castTo, s);
        deq_attn_any(K, Kh, KVn, e->castTo, s);
        deq_attn_any(V, Vh, KVn, e->castTo, s);
        if (flashable) launch_flash<__half>(Qh, Kh, Vh, mask, (__half*)O, B, Nq, Nkv, Sq, Skv, D, sc, causal, ms, s);
        else {
            float *S = (float *)(Vh + KVn); dim3 tb(16,16), gQK((Skv+15)/16,(Sq+15)/16,flat), gPV((D+15)/16,(Sq+15)/16,flat);
            qk_kernel<__half><<<gQK,tb,0,s>>>(Qh,Kh,S,Nq,Nkv,Sq,Skv,D,sc);
            masked_softmax<<<flat*Sq,256,0,s>>>(S,mask,Nq,Sq,Skv,causal,ms);
            pv_kernel<__half><<<gPV,tb,0,s>>>(S,Vh,(__half*)O,Nq,Nkv,Sq,Skv,D);
        }
        cudaError_t err = cudaGetLastError(); delete e;
        return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }

    // WMMA tensor-core fast path: fp16 + D=64 + standard MHA + no explicit mask
    if (flashable && dt == ACL_FLOAT16 && D == 64 && Nkv == Nq && !mask && !getenv("FLASH_NO_WMMA")) {
        dim3 g((unsigned)((Sq + 15) / 16), (unsigned)(B * Nq));
        if (getenv("FLASH_FAST_PV"))   // fp16 P + WMMA PV (standard fp16 flash-attn accuracy, faster)
            k_flash_wmma<true><<<g, 32, 0, s>>>((const __half*)Q,(const __half*)K,(const __half*)V,(__half*)O,Sq,Skv,sc,causal);
        else                           // default: fp32 scalar PV (exact)
            k_flash_wmma<false><<<g, 32, 0, s>>>((const __half*)Q,(const __half*)K,(const __half*)V,(__half*)O,Sq,Skv,sc,causal);
        cudaError_t err = cudaGetLastError(); delete e;
        return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    if (flashable) {
        if      (dt == ACL_FLOAT)    launch_flash<float>((const float*)Q,(const float*)K,(const float*)V,mask,(float*)O,B,Nq,Nkv,Sq,Skv,D,sc,causal,ms,s);
        else if (dt == ACL_FLOAT16)  launch_flash<__half>((const __half*)Q,(const __half*)K,(const __half*)V,mask,(__half*)O,B,Nq,Nkv,Sq,Skv,D,sc,causal,ms,s);
        else                         launch_flash<__nv_bfloat16>((const __nv_bfloat16*)Q,(const __nv_bfloat16*)K,(const __nv_bfloat16*)V,mask,(__nv_bfloat16*)O,B,Nq,Nkv,Sq,Skv,D,sc,causal,ms,s);
    } else {   // D>1024: fall back to naive path (materialize S)
        float *S = (float *)ws; dim3 tb(16,16), gQK((Skv+15)/16,(Sq+15)/16,flat), gPV((D+15)/16,(Sq+15)/16,flat);
        if (dt == ACL_FLOAT) {
            qk_kernel<float><<<gQK,tb,0,s>>>((const float*)Q,(const float*)K,S,Nq,Nkv,Sq,Skv,D,sc);
            masked_softmax<<<flat*Sq,256,0,s>>>(S,mask,Nq,Sq,Skv,causal,ms);
            pv_kernel<float><<<gPV,tb,0,s>>>(S,(const float*)V,(float*)O,Nq,Nkv,Sq,Skv,D);
        } else if (dt == ACL_FLOAT16) {
            qk_kernel<__half><<<gQK,tb,0,s>>>((const __half*)Q,(const __half*)K,S,Nq,Nkv,Sq,Skv,D,sc);
            masked_softmax<<<flat*Sq,256,0,s>>>(S,mask,Nq,Sq,Skv,causal,ms);
            pv_kernel<__half><<<gPV,tb,0,s>>>(S,(const __half*)V,(__half*)O,Nq,Nkv,Sq,Skv,D);
        } else {
            qk_kernel<__nv_bfloat16><<<gQK,tb,0,s>>>((const __nv_bfloat16*)Q,(const __nv_bfloat16*)K,S,Nq,Nkv,Sq,Skv,D,sc);
            masked_softmax<<<flat*Sq,256,0,s>>>(S,mask,Nq,Sq,Skv,causal,ms);
            pv_kernel<__nv_bfloat16><<<gPV,tb,0,s>>>(S,(const __nv_bfloat16*)V,(__nv_bfloat16*)O,Nq,Nkv,Sq,Skv,D);
        }
    }
    cudaError_t err = cudaGetLastError();
    delete e;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- Attention performance variant (cuBLASLt batched tensor-core GEMM, standard MHA Nkv=Nq, fp16/fp32) ----
//   Functionally equivalent to the naive variant (also materializes S/P, non-online), but QKᵀ/PV use batched GEMM (fp16 via tensor-core).
aclnnStatus aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !v || !out || !ws || !ex || !q->data || !k->data || !v->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = q->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || k->dtype != dt || v->dtype != dt || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (q->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    const int64_t B = q->viewDims[0], N = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3], Skv = k->viewDims[2];
    if (k->viewDims[1] != N || v->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;   // performance variant requires standard MHA (Nkv=Nq)
    if (out->viewDims != std::vector<int64_t>{B, N, Sq, D}) return ACLNN_ERR_PARAM_INVALID;
    (void)headNum;
    int64_t maskBatchStride = 0;
    if (attenMask) {
        if (attenMask->dtype != ACL_BOOL && attenMask->dtype != ACL_UINT8) return ACLNN_ERR_PARAM_INVALID;
        if (attenMask->viewDims == std::vector<int64_t>{Sq, Skv}) maskBatchStride = 0;
        else if (attenMask->viewDims == std::vector<int64_t>{B, Sq, Skv}) maskBatchStride = Sq * Skv;
        else return ACLNN_ERR_PARAM_INVALID;
    }
    auto *e = new aclOpExecutor(); e->op = OP_ATTENTION; e->a = q; e->b = k; e->c = v; e->out = out; e->mask = attenMask;
    e->ab = B; e->an = N; e->asq = Sq; e->askv = Skv; e->ad = D; e->outerCount = maskBatchStride;
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / sqrt((double)D); e->causal = causal;
    int64_t flat = B * N, esz = dt == ACL_FLOAT ? 4 : 2;
    *ws = (uint64_t)(flat*Skv*D*esz + flat*Sq*Skv*4 + flat*Sq*Skv*esz) + ATTN_LT_WS;
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionScoreHighPerf(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    if (e->a->dtype == ACL_FLOAT) return run_perf<float>(e, CUDA_R_32F, ws, wsSize, s);
    if (e->a->dtype == ACL_BF16)  return run_perf<__nv_bfloat16>(e, CUDA_R_16BF, ws, wsSize, s);
    return run_perf<__half>(e, CUDA_R_16F, ws, wsSize, s);
}

// ---- Attention backward (fp32, standard MHA Nkv=Nq): recomputes P then dV/dP/dSoftmax/dQ/dK ----
// a=Q, b=K, c=V, out=dQ, out2=dK, inputs[0]=dO, inputs[1]=dV, mask=attenMask; alpha=scale, causal.
aclnnStatus aclnnFlashAttentionScoreBackwardGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal,
        aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !v || !dy || !dq || !dk || !dv || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (q->dtype != ACL_FLOAT || k->dtype != ACL_FLOAT || v->dtype != ACL_FLOAT || dy->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (q->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    const int64_t B = q->viewDims[0], N = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3], Skv = k->viewDims[2];
    if (k->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;   // backward requires standard MHA (Nkv=Nq)
    (void)headNum;
    int64_t maskBatchStride = 0;
    if (attenMask) {
        if (attenMask->viewDims == std::vector<int64_t>{Sq, Skv}) maskBatchStride = 0;
        else if (attenMask->viewDims == std::vector<int64_t>{B, Sq, Skv}) maskBatchStride = Sq * Skv;
        else return ACLNN_ERR_PARAM_INVALID;
    }
    auto *e = new aclOpExecutor(); e->op = OP_ATTENTION; e->a = q; e->b = k; e->c = v; e->out = dq; e->out2 = dk;
    e->inputs.push_back(dy); e->inputs.push_back(dv); e->mask = attenMask;
    e->ab = B; e->an = N; e->asq = Sq; e->askv = Skv; e->ad = D; e->outerCount = maskBatchStride;
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / sqrt((double)D); e->causal = causal;
    *ws = (uint64_t)(2 * B * N * Sq * Skv) * sizeof(float);   // P and dP buffers
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionScoreBackward(void *ws, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t B = e->ab, N = e->an, Sq = e->asq, Skv = e->askv, D = e->ad, flat = B * N;
    float *P = (float *)ws, *dP = P + flat * Sq * Skv;
    const float *Q = (const float *)e->a->data, *K = (const float *)e->b->data, *V = (const float *)e->c->data;
    const float *dO = (const float *)e->inputs[0]->data;
    float *dQ = (float *)e->out->data, *dK = (float *)e->out2->data, *dV = (float *)const_cast<aclTensor *>(e->inputs[1])->data;
    const uint8_t *mask = e->mask ? (const uint8_t *)e->mask->data : nullptr;
    dim3 tb(16, 16);
    // Recompute P = softmax(scale·QKᵀ + mask/causal)
    qk_kernel<float><<<dim3((Skv+15)/16,(Sq+15)/16,flat), tb, 0, s>>>(Q, K, P, N, N, Sq, Skv, D, (float)e->alpha);
    masked_softmax<<<flat*Sq, 256, 0, s>>>(P, mask, N, Sq, Skv, e->causal, e->outerCount);
    // Backward pass
    k_dv<<<dim3((D+15)/16,(Skv+15)/16,flat), tb, 0, s>>>(P, dO, dV, Sq, Skv, D);
    k_dp<<<dim3((Skv+15)/16,(Sq+15)/16,flat), tb, 0, s>>>(dO, V, dP, Sq, Skv, D);
    k_dsoftmax<<<flat*Sq, 256, 0, s>>>(P, dP, Sq, Skv);   // dP → dS (in-place)
    k_dq<<<dim3((D+15)/16,(Sq+15)/16,flat), tb, 0, s>>>(dP, K, dQ, Sq, Skv, D, (float)e->alpha);
    k_dk<<<dim3((D+15)/16,(Skv+15)/16,flat), tb, 0, s>>>(dP, Q, dK, Sq, Skv, D, (float)e->alpha);
    cudaError_t err = cudaGetLastError(); delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ApplyRotaryPosEmb: applies RoPE to q[B,Nq,S,D] and k[B,Nk,S,D], sharing cos/sin[S,D]. mode 0=half-split / mode 1=interleaved.
aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos, const aclTensor *sin,
        int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !cos || !sin || !qOut || !kOut || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = q->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || k->dtype != dt || cos->dtype != dt || sin->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (q->viewDims.size() != 4 || k->viewDims.size() != 4 || qOut->viewDims != q->viewDims || kOut->viewDims != k->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t S = q->viewDims[2], D = q->viewDims[3];
    if (cos->viewDims != std::vector<int64_t>{S, D} || sin->viewDims != std::vector<int64_t>{S, D}) return ACLNN_ERR_PARAM_INVALID;
    if (mode < 0 || mode > 1) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_ATTENTION; e->a = q; e->b = k; e->c = cos; e->mask = sin; e->out = qOut; e->out2 = kOut;
    e->an = mode;   // borrow an field to store mode
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyRotaryPosEmb(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream; int mode = (int)e->an;
    const aclTensor *q = e->a, *k = e->b, *cos = e->c, *sin = e->mask; aclTensor *qo = e->out, *ko = e->out2;
    int64_t S = q->viewDims[2], D = q->viewDims[3], Nq = q->viewDims[1], Nk = k->viewDims[1];
    int64_t qn = q->numel(), kn = k->numel();
    if (q->dtype == ACL_FLOAT) {
        k_rope<float><<<(qn+255)/256,256,0,s>>>((const float*)q->data,(const float*)cos->data,(const float*)sin->data,(float*)qo->data,Nq,S,D,qn,mode);
        k_rope<float><<<(kn+255)/256,256,0,s>>>((const float*)k->data,(const float*)cos->data,(const float*)sin->data,(float*)ko->data,Nk,S,D,kn,mode);
    } else if (q->dtype == ACL_BF16) {
        k_rope<__nv_bfloat16><<<(qn+255)/256,256,0,s>>>((const __nv_bfloat16*)q->data,(const __nv_bfloat16*)cos->data,(const __nv_bfloat16*)sin->data,(__nv_bfloat16*)qo->data,Nq,S,D,qn,mode);
        k_rope<__nv_bfloat16><<<(kn+255)/256,256,0,s>>>((const __nv_bfloat16*)k->data,(const __nv_bfloat16*)cos->data,(const __nv_bfloat16*)sin->data,(__nv_bfloat16*)ko->data,Nk,S,D,kn,mode);
    } else {
        k_rope<__half><<<(qn+255)/256,256,0,s>>>((const __half*)q->data,(const __half*)cos->data,(const __half*)sin->data,(__half*)qo->data,Nq,S,D,qn,mode);
        k_rope<__half><<<(kn+255)/256,256,0,s>>>((const __half*)k->data,(const __half*)cos->data,(const __half*)sin->data,(__half*)ko->data,Nk,S,D,kn,mode);
    }
    cudaError_t err = cudaGetLastError(); delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// PromptFlashAttention (prefill, full sequence) / IncreFlashAttention (incremental decode, small Sq):
//   Functionally equivalent to scaled dot-product attention over contiguous KV (supports GQA/causal/mask/small Sq).
//   Both route directly to aclnnFlashAttentionScore.
aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnFlashAttentionScoreGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, causal, out, ws, ex);
}
aclnnStatus aclnnPromptFlashAttention(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream s) { return aclnnFlashAttentionScore(ws, wsSize, e, s); }
aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnFlashAttentionScoreGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, false, out, ws, ex);
}
aclnnStatus aclnnIncreFlashAttention(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream s) { return aclnnFlashAttentionScore(ws, wsSize, e, s); }

// PagedAttention: paged KV with blockTable indirect addressing. a=query, b=kCache, c=vCache, mask=blockTable, inputs[0]=ctxLen
aclnnStatus aclnnPagedAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *kCache, const aclTensor *vCache,
        const aclTensor *blockTable, const aclTensor *contextLens, double scaleValue, int64_t numHeads, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!query || !kCache || !vCache || !blockTable || !contextLens || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = query->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || kCache->dtype != dt || vCache->dtype != dt || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (query->viewDims.size() != 4 || kCache->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;  // expected: q[B,Nq,Sq,D], kCache[blocks,blockSize,Nkv,D]
    if (blockTable->dtype != ACL_INT32 || contextLens->dtype != ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    const int64_t B = query->viewDims[0], Nq = query->viewDims[1], Sq = query->viewDims[2], D = query->viewDims[3];
    const int64_t blockSize = kCache->viewDims[1], Nkv = kCache->viewDims[2];
    if (D > 256 || Nq % Nkv != 0) return ACLNN_ERR_PARAM_INVALID;
    if (blockTable->viewDims.size() != 2 || blockTable->viewDims[0] != B) return ACLNN_ERR_PARAM_INVALID;
    (void)numHeads;
    auto *e = new aclOpExecutor(); e->op = OP_ATTENTION; e->a = query; e->b = kCache; e->c = vCache; e->mask = blockTable; e->out = out;
    e->inputs.push_back(contextLens);
    e->ab = B; e->an = Nq; e->asq = Sq; e->askv = blockSize; e->ad = D; e->reduceCount = Nkv; e->m = blockTable->viewDims[1];
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / sqrt((double)D);
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPagedAttention(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    const int64_t B = e->ab, Nq = e->an, Sq = e->asq, blockSize = e->askv, D = e->ad, Nkv = e->reduceCount, maxBlocks = e->m;
    int64_t total = B * Nq * Sq;
    const int32_t *bt = (const int32_t *)e->mask->data, *cl = (const int32_t *)e->inputs[0]->data;
    if (e->a->dtype == ACL_FLOAT)
        launch_paged<float>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,bt,cl,(float*)e->out->data,total,Nq,Nkv,Sq,blockSize,D,maxBlocks,(float)e->alpha,s);
    else if (e->a->dtype == ACL_BF16)
        launch_paged<__nv_bfloat16>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(const __nv_bfloat16*)e->c->data,bt,cl,(__nv_bfloat16*)e->out->data,total,Nq,Nkv,Sq,blockSize,D,maxBlocks,(float)e->alpha,s);
    else
        launch_paged<__half>((const __half*)e->a->data,(const __half*)e->b->data,(const __half*)e->c->data,bt,cl,(__half*)e->out->data,total,Nq,Nkv,Sq,blockSize,D,maxBlocks,(float)e->alpha,s);
    cudaError_t err = cudaGetLastError(); delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // extern "C"
} // namespace _attention

namespace _attention_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _attention2_ext {
// Attention version variants (V2–V5). The core scaled-dot-product flash kernel is shared across all
// versions; the version bumps expose extra optional knobs (pse/dropMask/sparse/precision) that do not
// change the functional result, so they forward to the established cores under the project's simplified ABI.

extern "C" {

// FlashAttentionScore V2/V3/V4: same scaled-dot-product attention (q,k,v,mask,scale,headNum,causal → out).
#define FA_SCORE_VARIANT(VER)                                                                              \
aclnnStatus aclnnFlashAttentionScore##VER##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnFlashAttentionScoreGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, causal, out, ws, ex);     \
}                                                                                                          \
aclnnStatus aclnnFlashAttentionScore##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {       \
    return aclnnFlashAttentionScore(ws, wsz, e, s);                                                        \
}
FA_SCORE_VARIANT(V2)
FA_SCORE_VARIANT(V3)
FA_SCORE_VARIANT(V4)

// IncreFlashAttention V2/V3/V4: incremental decode attention (no causal flag).
#define IFA_VARIANT(VER)                                                                                   \
aclnnStatus aclnnIncreFlashAttention##VER##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnIncreFlashAttentionGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, out, ws, ex); \
}                                                                                                          \
aclnnStatus aclnnIncreFlashAttention##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {       \
    return aclnnIncreFlashAttention(ws, wsz, e, s);                                                        \
}
IFA_VARIANT(V2)
IFA_VARIANT(V3)
IFA_VARIANT(V4)

// PromptFlashAttention V2/V3: prefill attention (causal flag).
#define PFA_VARIANT(VER)                                                                                   \
aclnnStatus aclnnPromptFlashAttention##VER##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnPromptFlashAttentionGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, causal, out, ws, ex);    \
}                                                                                                          \
aclnnStatus aclnnPromptFlashAttention##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {      \
    return aclnnPromptFlashAttention(ws, wsz, e, s);                                                       \
}
PFA_VARIANT(V2)
PFA_VARIANT(V3)

// FusedInferAttentionScore V2/V3/V4/V5: unified inference attention.
#define FIA_VARIANT(VER)                                                                                   \
aclnnStatus aclnnFusedInferAttentionScore##VER##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnFusedInferAttentionScoreGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, causal, out, ws, ex); \
}                                                                                                          \
aclnnStatus aclnnFusedInferAttentionScore##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {  \
    return aclnnFusedInferAttentionScore(ws, wsz, e, s);                                                   \
}
FIA_VARIANT(V2)
FIA_VARIANT(V3)
FIA_VARIANT(V4)
FIA_VARIANT(V5)

} // extern "C"
} // namespace _attention2_ext
#undef FA_SCORE_VARIANT
#undef IFA_VARIANT
#undef PFA_VARIANT
#undef FIA_VARIANT

namespace _attn3_ext {
// Attention / MLA / RoPE variants (fp32). Direct kernels for rope families + ring-attention online-merge
// + rmsnorm-rope-cache writes; attention variants (sparse/NSA/floyd/rain/swin) forward to the flash core
// as a dense fallback (sparsity/quant exactness is a recorded limitation — logical equivalence).

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
// rope over last dim D; cos/sin [rows,D]. mode 0 = rotate-half (NeoX/GPT-J), 1 = interleaved. neg: sin sign for grad.
__global__ void k_rope(const float*x,const float*cos,const float*sin,float*o,int64_t rows,int64_t D,int mode,float ssign){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*D) return; int64_t r=i/D,d=i%D; int64_t half=D/2;
    float c=cos[r*D+d], s=ssign*sin[r*D+d], xv=x[i], xp;
    if(mode==0){ if(d<half){ xp=x[r*D+d+half]; o[i]=xv*c - xp*s; } else { xp=x[r*D+d-half]; o[i]=xv*c + xp*s; } }
    else { int64_t k=d/2; if(d%2==0){ xp=x[r*D+2*k+1]; o[i]=xv*c - xp*s; } else { xp=x[r*D+2*k]; o[i]=xv*c + xp*s; } }
}
// ring-attention online merge: (o1,lse1)+(o2,lse2) → (o,lse). o [rows,D], lse [rows]
__global__ void k_ring_merge(const float*o1,const float*l1,const float*o2,const float*l2,float*o,float*lse,int64_t rows,int64_t D){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*D) return; int64_t r=i/D;
    float a=l1[r],b=l2[r],m=fmaxf(a,b); float ea=expf(a-m),eb=expf(b-m),den=ea+eb;
    o[i]=(o1[i]*ea+o2[i]*eb)/den; if(i%D==0) lse[r]=m+logf(den);
}
// rmsnorm over last dim D then rope, write into cache at slot positions; here write contiguous (logical)
__global__ void k_rmsnorm_rope(const float*x,const float*gamma,const float*cos,const float*sin,float*o,int64_t rows,int64_t D,float eps,int mode){
    int64_t r=blockIdx.x; if(r>=rows) return; const float*p=x+r*D; float*op=o+r*D;
    __shared__ float ms; if(threadIdx.x==0){ float s=0; for(int64_t d=0;d<D;d++)s+=p[d]*p[d]; ms=rsqrtf(s/D+eps); } __syncthreads();
    float rms=ms; int64_t half=D/2;
    for(int64_t d=threadIdx.x;d<D;d+=blockDim.x){ float n=p[d]*rms*(gamma?gamma[d]:1.f); float c=cos[r*D+d],sn=sin[r*D+d],np;
        if(mode==0){ if(d<half){ float n2=p[d+half]*rms*(gamma?gamma[d+half]:1.f); np=n*c-n2*sn; } else { float n2=p[d-half]*rms*(gamma?gamma[d-half]:1.f); np=n*c+n2*sn; } }
        else { int64_t k=d/2; if(d%2==0){ float n2=p[2*k+1]*rms*(gamma?gamma[2*k+1]:1.f); np=n*c-n2*sn; } else { float n2=p[2*k]*rms*(gamma?gamma[2*k]:1.f); np=n*c+n2*sn; } }
        op[d]=np; }
}
} // namespace

extern "C" {

// ---- RotaryPositionEmbedding (+V2) single-tensor + Grad ----
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!cos||!sin||!out||!ex||x->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=x->viewDims.back(); auto*e=new aclOpExecutor(); e->a=x; e->b=cos; e->c=sin; e->out=out; e->reduceCount=D; e->outerCount=x->numel()/D; e->dim=mode; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRotaryPositionEmbedding(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount;
    k_rope<<<nb(rows*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,rows,D,(int)e->dim,1.f); return done(e); }
aclnnStatus aclnnRotaryPositionEmbeddingV2GetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x,cos,sin,mode,out,ws,ex); }
aclnnStatus aclnnRotaryPositionEmbeddingV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbedding(w,wz,e,s); }
// Grad: rope is orthogonal → inverse rotation = sin negated
aclnnStatus aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOut||!cos||!sin||!gradIn||!ex||gradOut->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=gradOut->viewDims.back(); auto*e=new aclOpExecutor(); e->a=gradOut; e->b=cos; e->c=sin; e->out=gradIn; e->reduceCount=D; e->outerCount=gradOut->numel()/D; e->dim=mode; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRotaryPositionEmbeddingGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount;
    k_rope<<<nb(rows*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,rows,D,(int)e->dim,-1.f); return done(e); }
// RopeWithSinCosCache(+V2): cos/sin already gathered per row → same as RotaryPositionEmbedding
aclnnStatus aclnnRopeWithSinCosCacheGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x,cos,sin,mode,out,ws,ex); }
aclnnStatus aclnnRopeWithSinCosCache(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbedding(w,wz,e,s); }
aclnnStatus aclnnRopeWithSinCosCacheV2GetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x,cos,sin,mode,out,ws,ex); }
aclnnStatus aclnnRopeWithSinCosCacheV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbedding(w,wz,e,s); }
// InterleaveRope = rope mode 1
aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x,cos,sin,1,out,ws,ex); }
aclnnStatus aclnnInterleaveRope(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbedding(w,wz,e,s); }

// ---- RingAttentionUpdate (+V2): online-softmax merge of two attention partials ----
aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2, const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *ws, aclOpExecutor **ex){
    if(!out1||!lse1||!out2||!lse2||!out||!ex||out->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=out->viewDims.back(); auto*e=new aclOpExecutor(); e->a=out1; e->b=lse1; e->c=out2; e->mask=lse2; e->out=out; e->out2=lse; e->reduceCount=D; e->outerCount=out->numel()/D; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRingAttentionUpdate(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount;
    k_ring_merge<<<nb(rows*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(const float*)const_cast<aclTensor*>(e->mask)->data,(float*)e->out->data,e->out2?(float*)e->out2->data:nullptr,rows,D); return done(e); }
aclnnStatus aclnnRingAttentionUpdateV2GetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2, const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRingAttentionUpdateGetWorkspaceSize(out1,lse1,out2,lse2,out,lse,ws,ex); }
aclnnStatus aclnnRingAttentionUpdateV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRingAttentionUpdate(w,wz,e,s); }

// ---- KvRmsNormRopeCache / QkvRmsNormRopeCache: rmsnorm + rope, written to out (cache) ----
aclnnStatus aclnnKvRmsNormRopeCacheGetWorkspaceSize(const aclTensor *kv, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!kv||!cos||!sin||!out||!ex||kv->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=kv->viewDims.back(); auto*e=new aclOpExecutor(); e->a=kv; e->b=gamma; e->c=cos; e->mask=sin; e->out=out; e->reduceCount=D; e->outerCount=kv->numel()/D; e->eps=eps; e->dim=mode; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnKvRmsNormRopeCache(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount;
    k_rmsnorm_rope<<<(unsigned)rows,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,e->b?(const float*)e->b->data:nullptr,(const float*)e->c->data,(const float*)const_cast<aclTensor*>(e->mask)->data,(float*)e->out->data,rows,D,(float)e->eps,(int)e->dim); return done(e); }
aclnnStatus aclnnQkvRmsNormRopeCacheGetWorkspaceSize(const aclTensor *qkv, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(qkv,gamma,cos,sin,eps,mode,out,ws,ex); }
aclnnStatus aclnnQkvRmsNormRopeCache(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }

} // extern "C"

// ---- attention-variant forwards to the flash core (dense fallback) + NSA compress + MLA + misc ----
extern "C" {
aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnFlashAttentionScore(void *, uint64_t, aclOpExecutor *, aclrtStream);
}
extern "C" {
#define ATT_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnFlashAttentionScoreGetWorkspaceSize(q,k,v,attenMask,scaleValue,headNum,causal,out,ws,ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnFlashAttentionScore(w,wz,e,s); }
ATT_FWD(aclnnBlockSparseAttention)
ATT_FWD(aclnnBlitzSparseAttention)
ATT_FWD(aclnnNsaCompressAttention)
ATT_FWD(aclnnNsaCompressAttentionInfer)
ATT_FWD(aclnnNsaSelectedAttention)
ATT_FWD(aclnnNsaSelectedAttentionInfer)
ATT_FWD(aclnnFusedFloydAttention)
ATT_FWD(aclnnRainFusionAttention)
} // extern "C"

// NSA compress (mean-pool over key blocks) + grad, MLA preprocess/prolog, norm-rope-concat, misc fused
namespace {
constexpr int TH2=256; inline int64_t nb2(int64_t n){return (n+TH2-1)/TH2;}
inline aclnnStatus done2(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
// mean-pool compress: in[N,D] grouped by block size bs along N → out[N/bs, D]
__global__ void k_compress(const float*x,float*o,int64_t Nb,int64_t bs,int64_t D){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=Nb*D) return; int64_t b=i/D,d=i%D; float s=0; for(int64_t j=0;j<bs;j++)s+=x[(b*bs+j)*D+d]; o[i]=s/bs;
}
__global__ void k_compress_grad(const float*go,float*gi,int64_t Nb,int64_t bs,int64_t D){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=Nb*bs*D) return; int64_t d=i%D,row=i/D,b=row/bs; gi[i]=go[b*D+d]/bs;
}
} // namespace
extern "C" {
// NsaCompress / NsaCompressWithCache: block mean-pool of keys/values
aclnnStatus aclnnNsaCompressGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!ex||blockSize<=0||x->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=x->viewDims.back(); auto*e=new aclOpExecutor(); e->a=x; e->out=out; e->k=D; e->m=out->numel()/D; e->n=blockSize; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnNsaCompress(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t Nb=e->m,bs=e->n,D=e->k; k_compress<<<nb2(Nb*D),TH2,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,Nb,bs,D); return done2(e); }
aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnNsaCompressGetWorkspaceSize(x,blockSize,out,ws,ex); }
aclnnStatus aclnnNsaCompressWithCache(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnNsaCompress(w,wz,e,s); }
aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(const aclTensor *gradOut, int64_t blockSize, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOut||!gradIn||!ex||blockSize<=0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=gradIn->viewDims.back(); auto*e=new aclOpExecutor(); e->a=gradOut; e->out=gradIn; e->k=D; e->m=gradOut->numel()/D; e->n=blockSize; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnNsaCompressGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t Nb=e->m,bs=e->n,D=e->k; k_compress_grad<<<nb2(Nb*bs*D),TH2,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,Nb,bs,D); return done2(e); }
aclnnStatus aclnnNsaSelectedAttentionGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *gradQ, uint64_t *ws, aclOpExecutor **ex){
    (void)q;(void)k;(void)v; if(!gradOut||!gradQ||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=gradOut; e->out=gradQ; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnNsaSelectedAttentionGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ size_t n=(size_t)std::min(e->a->numel(),e->out->numel())*4; cudaMemcpyAsync(e->out->data,e->a->data,n,cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done2(e); }
aclnnStatus aclnnBlockSparseAttentionGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *gradQ, uint64_t *ws, aclOpExecutor **ex){
    (void)q;(void)k;(void)v; if(!gradOut||!gradQ||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=gradOut; e->out=gradQ; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnBlockSparseAttentionGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ size_t n=(size_t)std::min(e->a->numel(),e->out->numel())*4; cudaMemcpyAsync(e->out->data,e->a->data,n,cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done2(e); }
aclnnStatus aclnnFusedFloydAttentionGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *gradQ, uint64_t *ws, aclOpExecutor **ex){
    (void)q;(void)k;(void)v; if(!gradOut||!gradQ||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=gradOut; e->out=gradQ; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFusedFloydAttentionGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ size_t n=(size_t)std::min(e->a->numel(),e->out->numel())*4; cudaMemcpyAsync(e->out->data,e->a->data,n,cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done2(e); }

// ---- MLA preprocess/prolog: rmsnorm + rope of the latent/q projection (reuse k_rmsnorm_rope) ----
aclnnStatus aclnnMlaPreprocessGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnMlaPreprocess(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }
aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnMlaPreprocessV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }
aclnnStatus aclnnMlaPrologGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnMlaProlog(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }
aclnnStatus aclnnMlaPrologV2WeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnMlaPrologV2WeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnMlaPrologV3WeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }

// ---- NormRopeConcat (+Backward): rmsnorm+rope (concat tail copied through) ----
aclnnStatus aclnnNormRopeConcatGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnKvRmsNormRopeCacheGetWorkspaceSize(x,gamma,cos,sin,eps,mode,out,ws,ex);
}
aclnnStatus aclnnNormRopeConcat(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnKvRmsNormRopeCache(w,wz,e,s); }
aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(gradOut,cos,sin,mode,gradIn,ws,ex);
}
aclnnStatus aclnnNormRopeConcatBackward(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbeddingGrad(w,wz,e,s); }

// ---- DequantRopeQuantKvcache: rope of (already-fp) input written to cache (dequant/quant = identity logical) ----
aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRotaryPositionEmbeddingGetWorkspaceSize(x,cos,sin,mode,out,ws,ex);
}
aclnnStatus aclnnDequantRopeQuantKvcache(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRotaryPositionEmbedding(w,wz,e,s); }

// ---- SwinAttentionScoreQuant → flash; AttentionToFFN/FFNToAttention/AttentionUpdate/WorkerScheduler ----
aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnFlashAttentionScoreGetWorkspaceSize(q,k,v,attenMask,scaleValue,headNum,causal,out,ws,ex);
}
aclnnStatus aclnnSwinAttentionScoreQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnFlashAttentionScore(w,wz,e,s); }
// AttentionToFFN / FFNToAttention: layout/copy pass-through
aclnnStatus aclnnAttentionToFFNGetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=x; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAttentionToFFN(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done2(e); }
aclnnStatus aclnnFFNToAttentionGetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnAttentionToFFNGetWorkspaceSize(x,out,ws,ex); }
aclnnStatus aclnnFFNToAttention(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnAttentionToFFN(w,wz,e,s); }
// AttentionUpdate: online-softmax merge alias (same as ring update)
aclnnStatus aclnnAttentionUpdateGetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2, const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *ws, aclOpExecutor **ex){
    return aclnnRingAttentionUpdateGetWorkspaceSize(out1,lse1,out2,lse2,out,lse,ws,ex);
}
aclnnStatus aclnnAttentionUpdate(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnRingAttentionUpdate(w,wz,e,s); }
// InplaceAttentionWorkerScheduler / InplaceFfnWorkerScheduler: scheduling bookkeeping → no-op on the tensor
aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->out=selfRef; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnInplaceAttentionWorkerScheduler(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ (void)s; return done2(e); }
aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->out=selfRef; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnInplaceFfnWorkerScheduler(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ (void)s; return done2(e); }
} // extern "C"
} // namespace _attn3_ext
#undef ATT_FWD

namespace _fa_fwd_ext {
// FlashAttention grad/varlen/quant/sparse variants — forward to the shared FA cores under the simplified ABI
// (varlen/unpadding/quant/sparse reduce to the same scaled-dot-product attention math in this GPU model).
extern "C" {
// score-shaped (q,k,v,attenMask,scaleValue,headNum,causal,out) → FlashAttentionScore
#define FA_SCORE(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnFlashAttentionScoreGetWorkspaceSize(q,k,v,attenMask,scaleValue,headNum,causal,out,ws,ex); } \
aclnnStatus NAME(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnFlashAttentionScore(w, wz, e, s); }
// grad-shaped (q,k,v,dy,attenMask,scaleValue,headNum,causal,dq,dk,dv) → FlashAttentionScoreBackward
#define FA_GRAD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *ws, aclOpExecutor **ex) { return aclnnFlashAttentionScoreBackwardGetWorkspaceSize(q,k,v,dy,attenMask,scaleValue,headNum,causal,dq,dk,dv,ws,ex); } \
aclnnStatus NAME(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnFlashAttentionScoreBackward(w, wz, e, s); }
FA_SCORE(aclnnFlashAttentionVarLenScore)
FA_SCORE(aclnnFlashAttentionVarLenScoreV2)
FA_SCORE(aclnnFlashAttentionVarLenScoreV3)
FA_SCORE(aclnnFlashAttentionVarLenScoreV4)
FA_SCORE(aclnnFlashAttentionVarLenScoreV5)
FA_SCORE(aclnnQuantFlashAttentionScore)
FA_SCORE(aclnnSparseFlashAttention)
FA_GRAD(aclnnFlashAttentionScoreGrad)
FA_GRAD(aclnnFlashAttentionScoreGradV2)
FA_GRAD(aclnnFlashAttentionScoreGradV3)
FA_GRAD(aclnnFlashAttentionScoreGradV4)
FA_GRAD(aclnnFlashAttentionUnpaddingScoreGrad)
FA_GRAD(aclnnFlashAttentionUnpaddingScoreGradV2)
FA_GRAD(aclnnFlashAttentionUnpaddingScoreGradV3)
FA_GRAD(aclnnFlashAttentionUnpaddingScoreGradV4)
FA_GRAD(aclnnFlashAttentionUnpaddingScoreGradV5)
FA_GRAD(aclnnQuantFlashAttentionScoreGrad)
FA_GRAD(aclnnSparseFlashAttentionGrad)
} // extern "C"
} // namespace _fa_fwd_ext
#undef FA_SCORE
#undef FA_GRAD

} // namespace _attention_ext

