// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace _norm_ext {
// Extended norm ops (fp32/fp16/bf16, compute in fp32): RMSNorm (last dim), GroupNorm (NCHW), BatchNorm (inference, per-channel).
// InstanceNorm = GroupNorm(G=C). LayerNorm: see layernorm.cu; backward (training) not implemented.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }


// RMSNorm fast path: one block per row; threads stride over D (coalesced access); warp-shuffle reduction for Σx².
// Replaces the per-row single-thread version (which strides D across warps — almost no coalescing → ~1/32 bandwidth).
template <typename T, int TB>
__global__ void k_rms_fast(const T *x, const T *g, T *o, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x;
    float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float v = (float)x[base + d]; ss += v * v; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o2);
    __shared__ float sh[TB / 32];
    if ((t & 31) == 0) sh[t >> 5] = ss;
    __syncthreads();
    if (t == 0) { float tot = 0; for (int w = 0; w < TB / 32; w++) tot += sh[w]; sh[0] = tot; }
    __syncthreads();
    float inv = rsqrtf(sh[0] / D + eps);
    for (int64_t d = t; d < D; d += TB) o[base + d] = (T)((float)x[base + d] * inv * (g ? (float)g[d] : 1.f));
}

// GroupNorm fast path: one block per (n,grp); Cg*HW elements within a group are contiguous in memory; threads coalesce access + warp/block reduction of sum/sq.
template <typename T, int TB>
__global__ void k_gn_fast(const T *x, const T *g, const T *b, T *o, int64_t C, int64_t HW, int64_t G, float eps) {
    int64_t blk = blockIdx.x, Cg = C / G, n = blk / G, grp = blk % G, cnt = Cg * HW, base = (n * C + grp * Cg) * HW;
    int t = threadIdx.x; float sum = 0, sq = 0;
    for (int64_t i = t; i < cnt; i += TB) { float v = (float)x[base + i]; sum += v; sq += v * v; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, o2); sq += __shfl_down_sync(0xffffffffu, sq, o2); }
    __shared__ float ssum[TB / 32], ssq[TB / 32];
    if ((t & 31) == 0) { ssum[t >> 5] = sum; ssq[t >> 5] = sq; } __syncthreads();
    __shared__ float mean, inv;
    if (t == 0) { float a = 0, q = 0; for (int w = 0; w < TB / 32; w++) { a += ssum[w]; q += ssq[w]; }
        mean = a / cnt; float var = q / cnt - mean * mean; inv = rsqrtf(var + eps); }
    __syncthreads();
    for (int64_t i = t; i < cnt; i += TB) { int64_t c = grp * Cg + i / HW; float gg = g ? (float)g[c] : 1.f, bb = b ? (float)b[c] : 0.f;
        o[base + i] = (T)(((float)x[base + i] - mean) * inv * gg + bb); }
}


template <typename T>
__global__ void k_bn(const T *x, const T *g, const T *b, const T *mean, const T *var, T *o,
                     int64_t total, int64_t C, int64_t HW, float eps) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t c = (i / HW) % C;
    float m = (float)mean[c], v = (float)var[c], gg = g ? (float)g[c] : 1.f, bb = b ? (float)b[c] : 0.f;
    o[i] = (T)(((float)x[i] - m) * rsqrtf(v + eps) * gg + bb);
}


// Backward fast path: one block per row + coalesced access + warp/block reduction. dgamma/dbeta still use atomicAdd (accumulated across rows).
template <int TB> __device__ inline float blk_sum(float v, float *red) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v; __syncthreads();
    if (threadIdx.x == 0) { float s = 0; for (int w = 0; w < TB / 32; w++) s += red[w]; red[0] = s; }
    __syncthreads(); float out = red[0]; __syncthreads(); return out;
}
template <typename T, int TB>
__global__ void k_rms_bwd_fast(const T *dy, const T *x, const T *g, T *dx, float *dgamma, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int t = threadIdx.x; int64_t base = r * D;
    __shared__ float red[TB/32]; float ss = 0, dot = 0;
    for (int64_t d = t; d < D; d += TB) { float xv = (float)x[base+d], gj = (float)dy[base+d]*(g?(float)g[d]:1.f); ss += xv*xv; dot += gj*xv; }
    float A = blk_sum<TB>(ss, red), B = blk_sum<TB>(dot, red);
    float rcp = rsqrtf(A/D + eps), r3 = rcp*rcp*rcp;
    for (int64_t d = t; d < D; d += TB) { float gj = (float)dy[base+d]*(g?(float)g[d]:1.f);
        dx[base+d] = (T)(rcp*gj - (r3/D)*(float)x[base+d]*B);
        if (dgamma) atomicAdd(&dgamma[d], (float)dy[base+d]*(float)x[base+d]*rcp); }
}
template <typename T, int TB>
__global__ void k_ln_bwd_fast(const T *dy, const T *x, const T *g, T *dx, float *dgamma, float *dbeta, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int t = threadIdx.x; int64_t base = r * D;
    __shared__ float red[TB/32]; float sx = 0;
    for (int64_t d = t; d < D; d += TB) sx += (float)x[base+d];
    float mean = blk_sum<TB>(sx, red) / D;
    float sv = 0; for (int64_t d = t; d < D; d += TB) { float tt = (float)x[base+d]-mean; sv += tt*tt; }
    float rstd = rsqrtf(blk_sum<TB>(sv, red)/D + eps);
    float smg = 0, smgx = 0;
    for (int64_t d = t; d < D; d += TB) { float xhat=((float)x[base+d]-mean)*rstd, gi=(float)dy[base+d]*(g?(float)g[d]:1.f); smg += gi; smgx += gi*xhat; }
    float mg = blk_sum<TB>(smg, red)/D, mgx = blk_sum<TB>(smgx, red)/D;
    for (int64_t d = t; d < D; d += TB) { float xhat=((float)x[base+d]-mean)*rstd, gi=(float)dy[base+d]*(g?(float)g[d]:1.f);
        dx[base+d] = (T)(rstd*(gi - mg - xhat*mgx));
        if (dgamma) atomicAdd(&dgamma[d], (float)dy[base+d]*xhat);
        if (dbeta) atomicAdd(&dbeta[d], (float)dy[base+d]); }
}

// DeepNorm (last dim): y = LayerNorm(alpha·x + gx) · gamma + beta (LayerNorm after DeepNet residual scaling)
// DeepNorm = LayerNorm(alpha·x + gx)·γ + β. One block per row, coalesced access + warp/block reduction
// (single-pass Σ and Σ² → mean/var, same scheme as k_gn_fast / k_add_layernorm).
template <typename T, int TB>
__global__ void k_deepnorm(const T *x, const T *gx, const T *g, const T *b, T *o, int64_t rows, int64_t D, float alpha, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x;
    float sum = 0, sq = 0;
    for (int64_t d = t; d < D; d += TB) { float tmp = alpha * (float)x[base+d] + (float)gx[base+d]; sum += tmp; sq += tmp * tmp; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, o2); sq += __shfl_down_sync(0xffffffffu, sq, o2); }
    __shared__ float ssum[TB / 32], ssq[TB / 32];
    if ((t & 31) == 0) { ssum[t >> 5] = sum; ssq[t >> 5] = sq; } __syncthreads();
    __shared__ float mean, rstd;
    if (t == 0) { float a = 0, q = 0; for (int w = 0; w < TB / 32; w++) { a += ssum[w]; q += ssq[w]; }
        mean = a / D; float var = q / D - mean * mean; rstd = rsqrtf(var + eps); } __syncthreads();
    for (int64_t d = t; d < D; d += TB) { float tmp = alpha * (float)x[base+d] + (float)gx[base+d];
        o[base+d] = (T)(((tmp - mean) * rstd) * (g ? (float)g[d] : 1.f) + (b ? (float)b[d] : 0.f)); }
}

// Fused AddRmsNorm = RmsNorm(x+res)·γ, outputs the residual sum. One block per row, coalesced access +
// warp/block reduction (same scheme as k_rms_fast). First pass writes ysum=x+res and accumulates Σ(x+res)²;
// second pass reads ysum back (one array, hot in L2) — avoiding the unfused Add→DRAM→RmsNorm round-trip.
template <typename T, int TB>
__global__ void k_add_rmsnorm(const T *x, const T *res, const T *g, T *ysum, T *y, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x;
    float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float v = (float)x[base+d] + (float)res[base+d]; ysum[base+d] = (T)v; ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    __shared__ float sh[TB / 32];
    if ((t & 31) == 0) sh[t >> 5] = ss; __syncthreads();
    if (t == 0) { float tot = 0; for (int w = 0; w < TB / 32; w++) tot += sh[w]; sh[0] = tot; } __syncthreads();
    float inv = rsqrtf(sh[0] / D + eps);
    for (int64_t d = t; d < D; d += TB) y[base+d] = (T)((float)ysum[base+d] * inv * (g ? (float)g[d] : 1.f));
}
// Fused AddLayerNorm = LayerNorm(x+res)·γ+β, outputs the residual sum. One block per row, coalesced, single-pass
// Σ and Σ² → mean/var (same scheme as k_gn_fast).
template <typename T, int TB>
__global__ void k_add_layernorm(const T *x, const T *res, const T *g, const T *b, T *ysum, T *y, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x;
    float sum = 0, sq = 0;
    for (int64_t d = t; d < D; d += TB) { float v = (float)x[base+d] + (float)res[base+d]; ysum[base+d] = (T)v; sum += v; sq += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, o); sq += __shfl_down_sync(0xffffffffu, sq, o); }
    __shared__ float ssum[TB / 32], ssq[TB / 32];
    if ((t & 31) == 0) { ssum[t >> 5] = sum; ssq[t >> 5] = sq; } __syncthreads();
    __shared__ float mean, inv;
    if (t == 0) { float a = 0, q = 0; for (int w = 0; w < TB / 32; w++) { a += ssum[w]; q += ssq[w]; }
        mean = a / D; float var = q / D - mean * mean; inv = rsqrtf(var + eps); } __syncthreads();
    for (int64_t d = t; d < D; d += TB)
        y[base+d] = (T)(((float)ysum[base+d] - mean) * inv * (g ? (float)g[d] : 1.f) + (b ? (float)b[d] : 0.f));
}
// SwiGlu/GeGlu: in[...,2D] split in half at the last dim → out[...,D] = act(a)·b (gelu=1 uses GELU, otherwise SiLU)
template <typename T>
__global__ void k_glu(const T *in, T *o, int64_t rows, int64_t D, int gelu) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * D) return;
    int64_t r = i / D, d = i % D; float a = (float)in[r * 2 * D + d], b = (float)in[r * 2 * D + D + d];
    float act = gelu ? 0.5f * a * erfcf(-a * 0.70710678118654752f) : a / (1.f + expf(-a));
    o[i] = (T)(act * b);
}

#define DISP3(KCALL) do { switch (e->a->dtype) {                       \
    case ACL_FLOAT:   { using T=float;        KCALL; } break;          \
    case ACL_FLOAT16: { using T=__half;       KCALL; } break;          \
    case ACL_BF16:    { using T=__nv_bfloat16; KCALL; } break;         \
    default: delete e; return ACLNN_ERR_PARAM_INVALID; } } while (0)

inline const void *D(const aclTensor *t) { return t ? t->data : nullptr; }

aclnnStatus finish(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// RMSNorm: along the last dim (D = last dim size), y = x / rms(x) * gamma, rms=sqrt(mean(x²)+eps)
aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, double eps,
                                         aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || self->viewDims != out->viewDims || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = self->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != D)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_RMSNORM; e->a = self; e->b = gamma; e->out = out;
    e->reduceCount = D; e->outerCount = self->numel() / D; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st=(cudaStream_t)s;
    // one block per row, coalesced access + warp reduction (block=256)
    DISP3(( k_rms_fast<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(T*)e->out->data,rows,Dd,(float)e->eps) ));
    return finish(e);
}

// GroupNorm (NCHW, self.dims=[N,C,*]): normalize each (n,group) over (C/G channels × spatial); gamma/beta per-channel [C]
aclnnStatus aclnnGroupNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
                                           int64_t numGroups, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || self->viewDims != out->viewDims || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t C = self->viewDims[1];
    if (numGroups <= 0 || C % numGroups != 0) return ACLNN_ERR_PARAM_INVALID;
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != C)) return ACLNN_ERR_PARAM_INVALID;
    if (beta && (beta->viewDims.size() != 1 || beta->viewDims[0] != C)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GROUPNORM; e->a = self; e->b = gamma; e->c = beta; e->out = out;
    e->reduceCount = numGroups; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t N = e->a->viewDims[0], C = e->a->viewDims[1], G = e->reduceCount, HW = e->a->numel() / (N * C);
    auto st = (cudaStream_t)s;
    DISP3(( k_gn_fast<T,256><<<(unsigned)(N*G),256,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(const T*)D(e->c),(T*)e->out->data,C,HW,G,(float)e->eps) ));
    return finish(e);
}

// BatchNorm inference (NCHW, per-channel mean/var/gamma/beta): y = gamma*(x-mean)/sqrt(var+eps)+beta
aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
                                           const aclTensor *mean, const aclTensor *var, double eps,
                                           aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !mean || !var || !out || !ex || !self->data || !mean->data || !var->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || self->viewDims != out->viewDims || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t C = self->viewDims[1];
    if (mean->viewDims.size()!=1||mean->viewDims[0]!=C||var->viewDims.size()!=1||var->viewDims[0]!=C) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_BATCHNORM; e->a = self; e->b = gamma; e->c = beta; e->mean = mean; e->rstd = var; e->out = out; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
// AddRmsNorm: a=x, b=residual, c=gamma, out=y, out2=residualSum; eps
aclnnStatus aclnnAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !y || !residualSum || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != y->dtype || x->viewDims != residual->viewDims || y->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = OP_RMSNORM; e->a = x; e->b = residual; e->c = gamma; e->out = y; e->out2 = residualSum;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps; e->keepDim = false;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    DISP3(( k_add_rmsnorm<T,TH><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D(e->c),(T*)e->out2->data,(T*)e->out->data,rows,Dd,(float)e->eps) ));
    return finish(e);
}
// AddLayerNorm: a=x, b=residual, c=gamma, mask=beta, out=y, out2=residualSum
aclnnStatus aclnnAddLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, const aclTensor *beta,
        double eps, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !y || !residualSum || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != y->dtype || x->viewDims != residual->viewDims || y->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = OP_GROUPNORM; e->a = x; e->b = residual; e->c = gamma; e->mask = beta; e->out = y; e->out2 = residualSum;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps; e->keepDim = true;  // keepDim=true marks add-layernorm
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddLayerNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    DISP3(( k_add_layernorm<T,TH><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D(e->c),(const T*)D(e->mask),(T*)e->out2->data,(T*)e->out->data,rows,Dd,(float)e->eps) ));
    return finish(e);
}
// SwiGlu/GeGlu: in[...,2D] → out[...,D]
static aclnnStatus glu_ws(const aclTensor *self, aclTensor *out, int gelu, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = out->viewDims.back();
    if (self->viewDims.back() != 2 * D) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GROUPNORM; e->a = self; e->out = out; e->reduceCount = D; e->outerCount = out->numel() / D; e->dim = gelu;
    *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus glu_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t rows = e->outerCount, Dd = e->reduceCount, g = nb(rows * Dd);
    DISP3(( k_glu<T><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,rows,Dd,(int)e->dim) ));
    return finish(e);
}
aclnnStatus aclnnSwiGluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { if(ws)*ws=0; return glu_ws(self,out,0,ex); }
aclnnStatus aclnnSwiGlu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return glu_run(e,(cudaStream_t)s); }
aclnnStatus aclnnGeGluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { if(ws)*ws=0; return glu_ws(self,out,1,ex); }
aclnnStatus aclnnGeGlu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return glu_run(e,(cudaStream_t)s); }

// DeepNorm: a=x, b=gx, c=gamma, mask=beta, out; alpha=residual scale factor, eps
aclnnStatus aclnnDeepNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gx, const aclTensor *gamma, const aclTensor *beta,
        double alpha, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gx || !out || !ex || !x->data || !gx->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != out->dtype || x->viewDims != out->viewDims || gx->viewDims != x->viewDims || !x->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != D)) return ACLNN_ERR_PARAM_INVALID;
    if (beta && (beta->viewDims.size() != 1 || beta->viewDims[0] != D)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_GROUPNORM; e->a = x; e->b = gx; e->c = gamma; e->mask = beta; e->out = out;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->alpha = alpha; e->eps = eps; e->keepDim = true;  // keepDim=true marks deepnorm
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDeepNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    DISP3(( k_deepnorm<T,TH><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D(e->c),(const T*)D(e->mask),(T*)e->out->data,rows,Dd,(float)e->alpha,(float)e->eps) ));
    return finish(e);
}

// RMSNorm backward: a=gradY, b=x, c=gamma, out=gradX, out2=gradGamma (fp32)
aclnnStatus aclnnRmsNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gradX || !ex || !gradY->data || !x->data || !gradX->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradY->viewDims != x->viewDims || gradX->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (gradGamma && gradGamma->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = OP_RMSNORM; e->a = gradY; e->b = x; e->c = gamma; e->out = gradX; e->out2 = gradGamma;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; int64_t g = nb(rows); auto st = (cudaStream_t)s;
    float *dg = e->out2 ? (float *)e->out2->data : nullptr;
    if (dg) cudaMemsetAsync(dg, 0, Dd * sizeof(float), st);
    DISP3(( k_rms_bwd_fast<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D(e->c),(T*)e->out->data,dg,rows,Dd,(float)e->eps) ));
    return finish(e);
}

// LayerNorm backward: a=gradY, b=x, c=gamma, out=gradX, out2=gradGamma, inputs[0]=gradBeta (all fp32 accumulated)
aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        const aclIntArray *normalizedShape, double eps, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gradX || !ex || !gradY->data || !x->data || !gradX->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradY->viewDims != x->viewDims || gradX->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (gradGamma && gradGamma->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (gradBeta && gradBeta->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = 1; if (normalizedShape) for (auto d : normalizedShape->v) D *= d; else D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = OP_GROUPNORM; e->a = gradY; e->b = x; e->c = gamma; e->out = gradX; e->out2 = gradGamma;
    if (gradBeta) e->inputs.push_back(gradBeta);
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLayerNormBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; int64_t g = nb(rows); auto st = (cudaStream_t)s;
    float *dg = e->out2 ? (float *)e->out2->data : nullptr;
    float *db = e->inputs.empty() ? nullptr : (float *)const_cast<aclTensor *>(e->inputs[0])->data;
    if (dg) cudaMemsetAsync(dg, 0, Dd * sizeof(float), st);
    if (db) cudaMemsetAsync(db, 0, Dd * sizeof(float), st);
    DISP3(( k_ln_bwd_fast<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D(e->c),(T*)e->out->data,dg,db,rows,Dd,(float)e->eps) ));
    return finish(e);
}

aclnnStatus aclnnBatchNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t total = e->a->numel(), C = e->a->viewDims[1], HW = total / (e->a->viewDims[0] * C);
    int64_t g = nb(total); auto st = (cudaStream_t)s;
    DISP3(( k_bn<T><<<g,TH,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(const T*)D(e->c),(const T*)e->mean->data,(const T*)e->rstd->data,(T*)e->out->data,total,C,HW,(float)e->eps) ));
    return finish(e);
}

} // extern "C"
} // namespace _norm_ext
#undef DISP3

namespace _norm2_ext {
// Normalization extensions (P7): InstanceNorm, LpNormalize (L2/Lp), LocalResponseNorm, RmsNormGated, BatchNormBackward.
// fp32-centric; self-contained executors. out contiguous.

namespace {

constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// InstanceNorm: per (N,C) normalize over spatial S; affine gamma/beta[C] optional
__global__ void k_instance_norm(const float *x, const float *gamma, const float *beta, float *o, int64_t NC, int64_t S, int64_t C, double eps) {
    int64_t nc = blockIdx.x; if (nc >= NC) return;          // one block per (n,c) group
    const float *p = x + nc * S; float *op = o + nc * S; int64_t c = nc % C;
    __shared__ double sm, sv;
    double ls = 0; for (int64_t i = threadIdx.x; i < S; i += blockDim.x) ls += p[i];
    __shared__ double red[TH]; red[threadIdx.x] = ls; __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if (threadIdx.x==0) sm = red[0]/S; __syncthreads();
    double m = sm, lv = 0; for (int64_t i = threadIdx.x; i < S; i += blockDim.x) { double d = p[i]-m; lv += d*d; }
    red[threadIdx.x] = lv; __syncthreads();
    for (int s = blockDim.x/2; s>0; s>>=1){ if(threadIdx.x<s) red[threadIdx.x]+=red[threadIdx.x+s]; __syncthreads(); }
    if (threadIdx.x==0) sv = red[0]/S; __syncthreads();
    double inv = rsqrt(sv + eps), g = gamma ? gamma[c] : 1.0, b = beta ? beta[c] : 0.0;
    for (int64_t i = threadIdx.x; i < S; i += blockDim.x) op[i] = (float)(((p[i]-m)*inv)*g + b);
}
// LpNormalize over last `dim` segments of length D (outer rows). out = x / max(norm_p, eps)
__global__ void k_lp_normalize(const float *x, float *o, int64_t rows, int64_t D, double p, double eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; const float *xp = x + r*D; float *op = o + r*D;
    __shared__ double red[TH]; double s = 0;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) s += pow(fabs((double)xp[i]), p);
    red[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x/2; k>0; k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); }
    __shared__ double nrm; if(threadIdx.x==0) nrm = pow(red[0], 1.0/p); __syncthreads();
    double denom = nrm < eps ? eps : nrm;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) op[i] = (float)(xp[i] / denom);
}
// LocalResponseNorm across channels: input [N,C,S]; denom=(k+alpha/size*sum window x^2)^beta
__global__ void k_lrn(const float *x, float *o, int64_t N, int64_t C, int64_t S, int64_t size, double alpha, double beta, double kk) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= N*C*S) return;
    int64_t s = i % S, c = (i / S) % C, n = i / (S*C);
    int64_t half = size/2; double acc = 0;
    for (int64_t j = c-half; j <= c+half; ++j) { if (j<0||j>=C) continue; double v = x[(n*C+j)*S+s]; acc += v*v; }
    double denom = pow(kk + alpha/size*acc, beta);
    o[i] = (float)(x[i] / denom);
}
// RmsNormGated: h = x*silu(gate); out = h*rsqrt(mean(h^2)+eps)*weight (over last dim D)
__global__ void k_rmsnorm_gated(const float *x, const float *gate, const float *weight, float *o, int64_t rows, int64_t D, double eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; const float *xp = x+r*D, *gp = gate+r*D; float *op = o+r*D;
    __shared__ double red[TH]; double s = 0;
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) { double g = gp[i]/(1.0+exp(-(double)gp[i])); double h = xp[i]*g; s += h*h; }
    red[threadIdx.x] = s; __syncthreads();
    for (int k = blockDim.x/2; k>0; k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); }
    __shared__ double inv; if(threadIdx.x==0) inv = rsqrt(red[0]/D + eps); __syncthreads();
    for (int64_t i = threadIdx.x; i < D; i += blockDim.x) { double g = gp[i]/(1.0+exp(-(double)gp[i])); double h = xp[i]*g; op[i] = (float)(h*inv*weight[i]); }
}
// BatchNorm backward (training): per-channel reductions then elementwise gradX. Layout [N,C,S].
__global__ void k_bn_bwd_stats(const float *gy, const float *x, const float *mean, const float *invstd, float *gGamma, float *gBeta, int64_t N, int64_t C, int64_t S) {
    int64_t c = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (c >= C) return;
    double gg = 0, gb = 0, m = mean[c], inv = invstd[c];
    for (int64_t n=0;n<N;n++) for (int64_t s=0;s<S;s++){ int64_t idx=(n*C+c)*S+s; double xhat=((double)x[idx]-m)*inv; gg += (double)gy[idx]*xhat; gb += gy[idx]; }
    gGamma[c] = (float)gg; gBeta[c] = (float)gb;
}
__global__ void k_bn_bwd_dx(const float *gy, const float *x, const float *gamma, const float *mean, const float *invstd, const float *gGamma, const float *gBeta, float *gx, int64_t N, int64_t C, int64_t S) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= N*C*S) return;
    int64_t c = (i/S)%C; double M = (double)N*S; double inv = invstd[c], g = gamma?gamma[c]:1.0;
    double xhat = ((double)x[i]-mean[c])*inv;
    gx[i] = (float)(g*inv/M * (M*(double)gy[i] - gBeta[c] - xhat*gGamma[c]));
}

inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// InstanceNorm: [N,C,*], gamma/beta[C] optional
aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT || self->viewDims.size() < 3) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=gamma; e->c=beta; e->out=out; e->eps=eps;
    int64_t N=self->viewDims[0], C=self->viewDims[1]; e->m=N*C; e->n=self->numel()/(N*C); e->k=C;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInstanceNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t NC=e->m,S=e->n,C=e->k;
    k_instance_norm<<<NC,TH,0,(cudaStream_t)s>>>((const float*)e->a->data, e->b?(const float*)e->b->data:nullptr, e->c?(const float*)e->c->data:nullptr, (float*)e->out->data, NC, S, C, e->eps);
    return done(e);
}
// LpNormalize over last dim (p, eps); Normalize(L2) = p=2
aclnnStatus aclnnLpNormalizeGetWorkspaceSize(const aclTensor *self, double p, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->out=out; e->eps=eps; e->alpha=p;
    int rank=(int)self->viewDims.size(); e->n=self->viewDims[rank-1]; e->m=self->numel()/e->n;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLpNormalize(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_lp_normalize<<<e->m,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->m,e->n,e->alpha,e->eps);
    return done(e);
}
// LocalResponseNorm across channels [N,C,*]
aclnnStatus aclnnLocalResponseNormGetWorkspaceSize(const aclTensor *self, int64_t size, double alpha, double beta, double k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT || self->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->out=out; e->eps=k; e->alpha=alpha; e->dscalars={beta,(double)size};
    e->outerCount=self->viewDims[0]; e->m=self->viewDims[1]; e->n=self->numel()/(e->outerCount*e->m);
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLocalResponseNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N=e->outerCount,C=e->m,S=e->n,size=(int64_t)e->dscalars[1]; int64_t g=nb(N*C*S);
    k_lrn<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,N,C,S,size,e->alpha,e->dscalars[0],e->eps);
    return done(e);
}
// RmsNormGated: x, gate, weight[D], eps -> out (over last dim)
aclnnStatus aclnnRmsNormGatedGetWorkspaceSize(const aclTensor *self, const aclTensor *gate, const aclTensor *weight, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !gate || !weight || !out || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=gate; e->c=weight; e->out=out; e->eps=eps;
    int rank=(int)self->viewDims.size(); e->n=self->viewDims[rank-1]; e->m=self->numel()/e->n;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormGated(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_rmsnorm_gated<<<e->m,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,e->m,e->n,e->eps);
    return done(e);
}
// BatchNormBackward (training): gradY,x,gamma,savedMean,savedInvStd -> gradX,gradGamma,gradBeta
aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        const aclTensor *savedMean, const aclTensor *savedInvStd, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !savedMean || !savedInvStd || !gradX || !gradGamma || !gradBeta || !ex || x->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=gradY; e->b=x; e->c=gamma; e->out=gradX; e->out2=gradGamma;
    e->mean=savedMean; e->rstd=savedInvStd; e->inputs.push_back(gradBeta);
    int64_t N=x->viewDims[0], C=x->viewDims[1]; e->outerCount=N; e->m=C; e->n=x->numel()/(N*C);
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t N=e->outerCount,C=e->m,S=e->n;
    const float *gy=(const float*)e->a->data,*x=(const float*)e->b->data,*gamma=e->c?(const float*)e->c->data:nullptr;
    const float *mean=(const float*)e->mean->data,*inv=(const float*)e->rstd->data;
    float *gGamma=(float*)e->out2->data,*gBeta=(float*)e->inputs[0]->data,*gx=(float*)e->out->data;
    k_bn_bwd_stats<<<nb(C),TH,0,st>>>(gy,x,mean,inv,gGamma,gBeta,N,C,S);
    k_bn_bwd_dx<<<nb(N*C*S),TH,0,st>>>(gy,x,gamma,mean,inv,gGamma,gBeta,gx,N,C,S);
    return done(e);
}

} // extern "C"
} // namespace _norm2_ext

namespace _norm3_ext {
// Norm-family extensions (fused/variant normalizations): GemmaRmsNorm, GroupNormSilu/Swish,
// FastLayerNorm, LayerNormWithImplMode. fp32/fp16/bf16, compute in fp32. Self-contained executors.

namespace {

// GemmaRmsNorm: y = x * rsqrt(mean(x^2)+eps) * (1 + gamma); one block per row, warp-shuffle Σx².
template <typename T, int TB>
__global__ void k_gemma_rms(const T *x, const T *g, T *o, float *rstd, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x; float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float v = (float)x[base + d]; ss += v * v; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o2);
    __shared__ float sh[TB / 32];
    if ((t & 31) == 0) sh[t >> 5] = ss;
    __syncthreads();
    if (t == 0) { float tot = 0; for (int w = 0; w < TB / 32; w++) tot += sh[w]; sh[0] = tot; }
    __syncthreads();
    float inv = rsqrtf(sh[0] / D + eps);
    if (t == 0 && rstd) rstd[r] = inv;
    for (int64_t d = t; d < D; d += TB) o[base + d] = (T)((float)x[base + d] * inv * (1.f + (g ? (float)g[d] : 0.f)));
}

// GroupNorm (NCHW) followed by optional activation: act<0 none, act==1 silu(x)=x·σ(x), act>0 swish with scale.
template <typename T, int TB>
__global__ void k_gn_act(const T *x, const T *g, const T *b, T *o, float *meanO, float *rstdO,
                         int64_t C, int64_t HW, int64_t G, float eps, float act) {
    int64_t blk = blockIdx.x, Cg = C / G, n = blk / G, grp = blk % G, cnt = Cg * HW, base = (n * C + grp * Cg) * HW;
    int t = threadIdx.x; float sum = 0, sq = 0;
    for (int64_t i = t; i < cnt; i += TB) { float v = (float)x[base + i]; sum += v; sq += v * v; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) { sum += __shfl_down_sync(0xffffffffu, sum, o2); sq += __shfl_down_sync(0xffffffffu, sq, o2); }
    __shared__ float ssum[TB / 32], ssq[TB / 32];
    if ((t & 31) == 0) { ssum[t >> 5] = sum; ssq[t >> 5] = sq; } __syncthreads();
    __shared__ float mean, inv;
    if (t == 0) { float a = 0, q = 0; for (int w = 0; w < TB / 32; w++) { a += ssum[w]; q += ssq[w]; }
        mean = a / cnt; float var = q / cnt - mean * mean; inv = rsqrtf(var + eps);
        if (meanO) meanO[blk] = mean; if (rstdO) rstdO[blk] = inv; }
    __syncthreads();
    for (int64_t i = t; i < cnt; i += TB) { int64_t c = grp * Cg + i / HW; float gg = g ? (float)g[c] : 1.f, bb = b ? (float)b[c] : 0.f;
        float y = ((float)x[base + i] - mean) * inv * gg + bb;
        if (act >= 0.f) y = y / (1.f + expf(-act * y));  // swish: y·σ(scale·y); scale==1 ≡ SiLU
        o[base + i] = (T)y; }
}

// LayerNorm over the last D columns (self-contained, multi-dtype): y=(x-mean)·rstd·gamma+beta.
template <typename T, int TB>
__global__ void k_ln_last(const T *x, const T *g, const T *b, T *o, float *meanO, float *rstdO, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    float s = 0; for (int64_t d = t; d < D; d += TB) s += (float)x[base + d];
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) s += __shfl_down_sync(0xffffffffu, s, o2);
    __shared__ float sh[TB / 32]; if ((t & 31) == 0) sh[t >> 5] = s; __syncthreads();
    __shared__ float mean;
    if (t == 0) { float a = 0; for (int w = 0; w < TB / 32; w++) a += sh[w]; mean = a / D; } __syncthreads();
    float v = 0; for (int64_t d = t; d < D; d += TB) { float u = (float)x[base + d] - mean; v += u * u; }
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) v += __shfl_down_sync(0xffffffffu, v, o2);
    if ((t & 31) == 0) sh[t >> 5] = v; __syncthreads();
    __shared__ float inv;
    if (t == 0) { float a = 0; for (int w = 0; w < TB / 32; w++) a += sh[w]; inv = rsqrtf(a / D + eps);
        if (meanO) meanO[r] = mean; if (rstdO) rstdO[r] = inv; } __syncthreads();
    for (int64_t d = t; d < D; d += TB) o[base + d] = (T)(((float)x[base + d] - mean) * inv * (g ? (float)g[d] : 1.f) + (b ? (float)b[d] : 0.f));
}

#define DISP3(KCALL) do { switch (e->a->dtype) {                       \
    case ACL_FLOAT:   { using T=float;        KCALL; } break;          \
    case ACL_FLOAT16: { using T=__half;       KCALL; } break;          \
    case ACL_BF16:    { using T=__nv_bfloat16; KCALL; } break;         \
    default: delete e; return ACLNN_ERR_PARAM_INVALID; } } while (0)
inline const void *D(const aclTensor *t) { return t ? t->data : nullptr; }
inline float *FP(const aclTensor *t) { return t ? (float *)t->data : nullptr; }
inline aclnnStatus fin(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// GemmaRmsNorm: y = x/rms(x)·(1+gamma) over the last dim; optional rstdOut (fp32, per row).
aclnnStatus aclnnGemmaRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps,
                                              aclTensor *yOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !yOut || !ex || !x->data || !yOut->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != yOut->dtype || x->viewDims != yOut->viewDims || !x->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    if (rstdOut && rstdOut->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->out = yOut; e->rstd = rstdOut;
    e->reduceCount = Dd; e->outerCount = x->numel() / Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGemmaRmsNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    DISP3(( k_gemma_rms<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(T*)e->out->data,FP(e->rstd),rows,Dd,(float)e->eps) ));
    return fin(e);
}

// GroupNormSilu (NCHW): GroupNorm then SiLU; gamma/beta[C] optional; meanOut/rstdOut[N,G] optional.
static aclnnStatus gn_act_ws(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
                             int64_t group, double eps, float act, aclTensor *out, aclTensor *meanO, aclTensor *rstdO, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || self->viewDims != out->viewDims || !self->contiguous() || self->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t C = self->viewDims[1];
    if (group <= 0 || C % group != 0) return ACLNN_ERR_PARAM_INVALID;
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != C)) return ACLNN_ERR_PARAM_INVALID;
    if (beta && (beta->viewDims.size() != 1 || beta->viewDims[0] != C)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = gamma; e->c = beta; e->out = out;
    e->mean = meanO; e->rstd = rstdO; e->reduceCount = group; e->eps = eps; e->alpha = act;
    *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus gn_act_run(aclOpExecutor *e, cudaStream_t st) {
    int64_t N = e->a->viewDims[0], C = e->a->viewDims[1], G = e->reduceCount, HW = e->a->numel() / (N * C);
    DISP3(( k_gn_act<T,256><<<(unsigned)(N*G),256,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(const T*)D(e->c),(T*)e->out->data,
            e->mean?(float*)e->mean->data:nullptr, e->rstd?(float*)e->rstd->data:nullptr, C,HW,G,(float)e->eps,(float)e->alpha) ));
    return fin(e);
}
aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
        int64_t group, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return gn_act_ws(self, gamma, beta, group, eps, 1.f, out, meanOut, rstdOut, ex);
}
aclnnStatus aclnnGroupNormSilu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return gn_act_run(e, (cudaStream_t)s); }
// V2: explicit activateSilu flag (false → plain GroupNorm).
aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
        int64_t group, double eps, bool activateSilu, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return gn_act_ws(self, gamma, beta, group, eps, activateSilu ? 1.f : -1.f, out, meanOut, rstdOut, ex);
}
aclnnStatus aclnnGroupNormSiluV2(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return gn_act_run(e, (cudaStream_t)s); }
// Swish: y·σ(scale·y). swishScale==1 ≡ SiLU.
aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
        int64_t group, double eps, double swishScale, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return gn_act_ws(self, gamma, beta, group, eps, (float)swishScale, out, meanOut, rstdOut, ex);
}
aclnnStatus aclnnGroupNormSwish(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return gn_act_run(e, (cudaStream_t)s); }

// FastLayerNorm: LayerNorm over the last dim (x, gamma, beta) → out, mean, rstd (fp32 optional).
static aclnnStatus ln_last_ws(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps,
                              aclTensor *out, aclTensor *meanO, aclTensor *rstdO, aclOpExecutor **ex) {
    if (!x || !out || !ex || !x->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != out->dtype || x->viewDims != out->viewDims || !x->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && gamma->numel() != Dd) return ACLNN_ERR_PARAM_INVALID;
    if (beta && beta->numel() != Dd) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = beta; e->out = out; e->mean = meanO; e->rstd = rstdO;
    e->reduceCount = Dd; e->outerCount = x->numel() / Dd; e->eps = eps;
    *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus ln_last_run(aclOpExecutor *e, cudaStream_t st) {
    int64_t rows = e->outerCount, Dd = e->reduceCount;
    DISP3(( k_ln_last<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data,(const T*)D(e->b),(const T*)D(e->c),(T*)e->out->data,
            e->mean?(float*)e->mean->data:nullptr, e->rstd?(float*)e->rstd->data:nullptr, rows,Dd,(float)e->eps) ));
    return fin(e);
}
aclnnStatus aclnnFastLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps,
        aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return ln_last_ws(x, gamma, beta, eps, out, meanOut, rstdOut, ex);
}
aclnnStatus aclnnFastLayerNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return ln_last_run(e, (cudaStream_t)s); }

// LayerNormWithImplMode: same as LayerNorm; implMode selects compute precision (ignored — we always compute in fp32).
aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
        const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
        int64_t implMode, uint64_t *ws, aclOpExecutor **ex) {
    (void)implMode;
    if (!input || !normalizedShape || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t Dd = 1; for (auto d : normalizedShape->v) Dd *= d;
    if (Dd <= 0 || input->viewDims.back() != normalizedShape->v.back()) return ACLNN_ERR_PARAM_INVALID;
    if (ws) *ws = 0; return ln_last_ws(input, weight, bias, eps, out, meanOut, rstdOut, ex);
}
aclnnStatus aclnnLayerNormWithImplMode(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return ln_last_run(e, (cudaStream_t)s); }

} // extern "C"
} // namespace _norm3_ext
#undef DISP3

namespace _norm4_ext {
// AddRmsNorm fused variants: AddRmsNormCast (emit y + a second-dtype cast of y) and InplaceAddRmsNorm
// (write the normalized result back into x). Both compute residualSum = x+residual.
// The normalize pass reads from residualSum (not x), so y may alias x safely.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// y = rms(x+res)·gamma over last dim D; ysum = x+res; yCast = (TC)y (optional). Reads ysum in pass 2 → alias-safe.
template <typename T, typename TC, int TB>
__global__ void k_add_rmsnorm_cast(const T *x, const T *res, const T *g, T *ysum, T *y, TC *yCast, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int64_t base = r * D; int t = threadIdx.x;
    float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float v = (float)x[base+d] + (float)res[base+d]; ysum[base+d] = (T)v; ss += v * v; }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    __shared__ float sh[TB / 32];
    if ((t & 31) == 0) sh[t >> 5] = ss; __syncthreads();
    if (t == 0) { float tot = 0; for (int w = 0; w < TB / 32; w++) tot += sh[w]; sh[0] = tot; } __syncthreads();
    float inv = rsqrtf(sh[0] / D + eps);
    for (int64_t d = t; d < D; d += TB) { float yv = (float)ysum[base+d] * inv * (g ? (float)g[d] : 1.f);
        y[base+d] = (T)yv; if (yCast) yCast[base+d] = (TC)yv; }
}

inline const void *D_(const aclTensor *t){ return t ? t->data : nullptr; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

// AdaLayerNorm: y = LayerNorm(x over last dim D)·(1+scale) + shift; scale/shift per-row [rows,D]. One block per row.
template <typename T, int TB>
__global__ void k_adaln(const T *x, const T *scale, const T *shift, T *y, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    float s = 0; for (int64_t d = t; d < D; d += TB) s += (float)x[base+d];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    __shared__ float sh[TB/32]; if ((t&31)==0) sh[t>>5]=s; __syncthreads();
    __shared__ float mean; if (t==0){ float a=0; for(int w=0;w<TB/32;w++) a+=sh[w]; mean=a/D; } __syncthreads();
    float v=0; for (int64_t d=t; d<D; d+=TB){ float u=(float)x[base+d]-mean; v+=u*u; }
    #pragma unroll
    for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffffu, v, o);
    if ((t&31)==0) sh[t>>5]=v; __syncthreads();
    __shared__ float inv; if (t==0){ float a=0; for(int w=0;w<TB/32;w++) a+=sh[w]; inv=rsqrtf(a/D+eps); } __syncthreads();
    for (int64_t d=t; d<D; d+=TB){ float n=((float)x[base+d]-mean)*inv;
        y[base+d]=(T)(n*(1.f+(scale?(float)scale[base+d]:0.f)) + (shift?(float)shift[base+d]:0.f)); }
}

// dispatch on input dtype T then cast dtype TC
template <typename T>
static void launch_castT(aclOpExecutor *e, int64_t rows, int64_t D, cudaStream_t st) {
    const T *x=(const T*)e->a->data,*res=(const T*)e->b->data,*g=(const T*)D_(e->c);
    T *ysum=(T*)e->out2->data, *y=(T*)e->out->data; unsigned blk=(unsigned)rows; float eps=(float)e->eps;
    if (!e->mask) { k_add_rmsnorm_cast<T,T,TH><<<blk,TH,0,st>>>(x,res,g,ysum,y,(T*)nullptr,rows,D,eps); return; }
    switch (e->mask->dtype) {
        case ACL_FLOAT16: k_add_rmsnorm_cast<T,__half,TH><<<blk,TH,0,st>>>(x,res,g,ysum,y,(__half*)e->mask->data,rows,D,eps); break;
        case ACL_BF16:    k_add_rmsnorm_cast<T,__nv_bfloat16,TH><<<blk,TH,0,st>>>(x,res,g,ysum,y,(__nv_bfloat16*)e->mask->data,rows,D,eps); break;
        default:          k_add_rmsnorm_cast<T,float,TH><<<blk,TH,0,st>>>(x,res,g,ysum,y,(float*)e->mask->data,rows,D,eps); break;
    }
}

// AdaLayerNorm backward. y = n·(1+scale)+shift, n=(x-mean)/std. Given gy:
//   gShift=gy; gScale=gy·n; g=gy·(1+scale); gX = (1/std)·(g - mean(g) - n·mean(g·n)).
template <typename T, int TB>
__global__ void k_adaln_bwd(const T *gy, const T *x, const T *scale, T *gX, T *gScale, T *gShift, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    __shared__ float sh[TB/32]; __shared__ float bcast;
    auto rsum = [&](float v) -> float {
        #pragma unroll
        for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((t&31)==0) sh[t>>5]=v; __syncthreads();
        if (t==0){ float a=0; for(int w=0;w<TB/32;w++) a+=sh[w]; bcast=a; } __syncthreads();
        float res=bcast; __syncthreads(); return res; };
    float s=0; for (int64_t d=t;d<D;d+=TB) s+=(float)x[base+d];
    float mean = rsum(s)/D;
    float v=0; for (int64_t d=t;d<D;d+=TB){ float u=(float)x[base+d]-mean; v+=u*u; }
    float inv = rsqrtf(rsum(v)/D + eps);
    float sg=0, sgn=0;
    for (int64_t d=t;d<D;d+=TB){ float n=((float)x[base+d]-mean)*inv; float g=(float)gy[base+d]*(1.f+(scale?(float)scale[base+d]:0.f)); sg+=g; sgn+=g*n; }
    float mg = rsum(sg)/D, mgn = rsum(sgn)/D;
    for (int64_t d=t;d<D;d+=TB){ float n=((float)x[base+d]-mean)*inv; float g=(float)gy[base+d]*(1.f+(scale?(float)scale[base+d]:0.f));
        gX[base+d]=(T)(inv*(g - mg - n*mgn));
        if (gScale) gScale[base+d]=(T)((float)gy[base+d]*n);
        if (gShift) gShift[base+d]=gy[base+d]; }
}

} // namespace

extern "C" {

// AddRmsNormCast: x, residual, gamma, eps -> y (input dtype), yCast (second dtype), residualSum (= x+residual).
aclnnStatus aclnnAddRmsNormCastGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *y, aclTensor *yCast, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !y || !residualSum || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != y->dtype || x->viewDims != residual->viewDims || y->viewDims != x->viewDims || residualSum->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (yCast && yCast->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = y; e->out2 = residualSum; e->mask = yCast;
    e->reduceCount = Dd; e->outerCount = x->numel() / Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormCast(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   launch_castT<float>(e, rows, Dd, st); break;
        case ACL_FLOAT16: launch_castT<__half>(e, rows, Dd, st); break;
        case ACL_BF16:    launch_castT<__nv_bfloat16>(e, rows, Dd, st); break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    return fin(e);
}

// InplaceAddRmsNorm: x updated in place to rms(x+residual)·gamma; residualSum = x+residual (separate buffer).
aclnnStatus aclnnInplaceAddRmsNormGetWorkspaceSize(aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !residualSum || !ex || !x->data || !residualSum->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->viewDims != residual->viewDims || residualSum->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = x; e->out2 = residualSum; e->mask = nullptr;
    e->reduceCount = Dd; e->outerCount = x->numel() / Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceAddRmsNorm(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAddRmsNormCast(ws, wsz, e, s); }

// AdaLayerNorm: y = LayerNorm(x)·(1+scale)+shift over last dim; scale/shift per-row [rows,D].
aclnnStatus aclnnAdaLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex || !x->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != out->dtype || x->viewDims != out->viewDims || !x->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (scale && scale->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (shift && shift->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = x; e->b = scale; e->c = shift; e->out = out;
    e->reduceCount = Dd; e->outerCount = x->numel() / Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaLayerNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s; float eps = (float)e->eps;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_adaln<float,256><<<(unsigned)rows,256,0,st>>>((const float*)e->a->data,(const float*)D_(e->b),(const float*)D_(e->c),(float*)e->out->data,rows,Dd,eps); break;
        case ACL_FLOAT16: k_adaln<__half,256><<<(unsigned)rows,256,0,st>>>((const __half*)e->a->data,(const __half*)D_(e->b),(const __half*)D_(e->c),(__half*)e->out->data,rows,Dd,eps); break;
        default:          k_adaln<__nv_bfloat16,256><<<(unsigned)rows,256,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)D_(e->b),(const __nv_bfloat16*)D_(e->c),(__nv_bfloat16*)e->out->data,rows,Dd,eps); break;
    }
    return fin(e);
}
aclnnStatus aclnnAdaLayerNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnAdaLayerNormGetWorkspaceSize(x, scale, shift, eps, out, ws, ex); }
aclnnStatus aclnnAdaLayerNormV2(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAdaLayerNorm(ws, wsz, e, s); }

// AdaLayerNormBackward: gradY, x, scale, eps -> gradX, gradScale, gradShift (per-row [rows,D]).
aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *scale, double eps,
        aclTensor *gradX, aclTensor *gradScale, aclTensor *gradShift, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gradX || !ex || !gradY->data || !x->data || !gradX->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradY->viewDims != x->viewDims || gradX->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = scale; e->out = gradX; e->out2 = gradScale; e->mask = gradShift;
    e->reduceCount = Dd; e->outerCount = x->numel()/Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaLayerNormBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s; float eps=(float)e->eps;
    void *gs = e->out2 ? e->out2->data : nullptr, *gsh = e->mask ? const_cast<aclTensor*>(e->mask)->data : nullptr;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_adaln_bwd<float,256><<<(unsigned)rows,256,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)D_(e->c),(float*)e->out->data,(float*)gs,(float*)gsh,rows,Dd,eps); break;
        case ACL_FLOAT16: k_adaln_bwd<__half,256><<<(unsigned)rows,256,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(const __half*)D_(e->c),(__half*)e->out->data,(__half*)gs,(__half*)gsh,rows,Dd,eps); break;
        default:          k_adaln_bwd<__nv_bfloat16,256><<<(unsigned)rows,256,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(const __nv_bfloat16*)D_(e->c),(__nv_bfloat16*)e->out->data,(__nv_bfloat16*)gs,(__nv_bfloat16*)gsh,rows,Dd,eps); break;
    }
    return fin(e);
}

} // extern "C"
} // namespace _norm4_ext

namespace _norm5_ext {
// Fused RmsNorm + quant: AddRmsNormDynamicQuant (per-token int8), AddRmsNormQuant / RmsNormQuant (static scale+offset int8).
// All compute y = rms(s)·gamma over the last dim (s = x or x+residual), then quantize to int8. fp32/fp16/bf16 input.

namespace {
constexpr int TB = 256;
__device__ inline int8_t clamp_i8(float v){ int q = __float2int_rn(v); return (int8_t)(q<-127?-127:(q>127?127:q)); }

// Block-per-row warp+block reduction helper for a single float (sum or max).
template <bool MAX>
__device__ inline float blk_reduce(float v, float *sh, int t) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) { float o2 = __shfl_down_sync(0xffffffffu, v, o); v = MAX ? fmaxf(v, o2) : v + o2; }
    if ((t & 31) == 0) sh[t >> 5] = v; __syncthreads();
    if (t == 0) { float a = sh[0]; for (int w = 1; w < TB/32; w++) a = MAX ? fmaxf(a, sh[w]) : a + sh[w]; sh[0] = a; }
    __syncthreads(); return sh[0];
}

// AddRmsNormDynamicQuant: s=x(+res); y=rms(s)·g; per-row absmax int8 quant; scaleOut[r]=absmax/127. ysum=s (optional).
template <typename T, bool ADD>
__global__ void k_addrms_dynquant(const T *x, const T *res, const T *g, T *ysum, int8_t *yq, float *scaleOut, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    __shared__ float sh[TB/32];
    float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float s = (float)x[base+d] + (ADD ? (float)res[base+d] : 0.f); if (ysum) ysum[base+d] = (T)s; ss += s*s; }
    float inv = rsqrtf(blk_reduce<false>(ss, sh, t) / D + eps);
    float amax = 0;
    for (int64_t d = t; d < D; d += TB) { float s = (float)x[base+d] + (ADD ? (float)res[base+d] : 0.f); amax = fmaxf(amax, fabsf(s*inv*(g?(float)g[d]:1.f))); }
    amax = blk_reduce<true>(amax, sh, t);
    float sc = amax > 0 ? amax/127.f : 1.f; if (t == 0) scaleOut[r] = sc; float qi = 1.f/sc;
    for (int64_t d = t; d < D; d += TB) { float s = (float)x[base+d] + (ADD ? (float)res[base+d] : 0.f); yq[base+d] = clamp_i8(s*inv*(g?(float)g[d]:1.f)*qi); }
}

// AddRmsNormQuant / RmsNormQuant: static scale+offset. yq = round(y/scale + offset).
template <typename T, bool ADD>
__global__ void k_addrms_quant(const T *x, const T *res, const T *g, T *ysum, int8_t *yq, int64_t rows, int64_t D, float eps, float scale, float offset) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    __shared__ float sh[TB/32]; float ss = 0;
    for (int64_t d = t; d < D; d += TB) { float s = (float)x[base+d] + (ADD ? (float)res[base+d] : 0.f); if (ysum) ysum[base+d] = (T)s; ss += s*s; }
    float inv = rsqrtf(blk_reduce<false>(ss, sh, t) / D + eps); float qi = 1.f/scale;
    for (int64_t d = t; d < D; d += TB) { float s = (float)x[base+d] + (ADD ? (float)res[base+d] : 0.f); yq[base+d] = clamp_i8(s*inv*(g?(float)g[d]:1.f)*qi + offset); }
}

inline const void *D_(const aclTensor *t){ return t ? t->data : nullptr; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// AddRmsNormDynamicQuant: x, residual, gamma, eps -> yQuant[int8], scaleOut[fp32 per-token], residualSum.
aclnnStatus aclnnAddRmsNormDynamicQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *yQuant, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !yQuant || !scaleOut || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (yQuant->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != residual->viewDims || yQuant->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = yQuant; e->out2 = residualSum; e->rstd = scaleOut;
    e->reduceCount = Dd; e->outerCount = x->numel()/Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormDynamicQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s;
    int8_t *yq = (int8_t*)e->out->data; float *sc = (float*)e->rstd->data;
    #define DQ(T) k_addrms_dynquant<T,true><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D_(e->c), e->out2?(T*)e->out2->data:nullptr, yq, sc, rows, Dd, (float)e->eps)
    switch (e->a->dtype) { case ACL_FLOAT: DQ(float); break; case ACL_FLOAT16: DQ(__half); break; default: DQ(__nv_bfloat16); break; }
    #undef DQ
    return fin(e);
}
aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *yQuant, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddRmsNormDynamicQuantGetWorkspaceSize(x, residual, gamma, eps, yQuant, scaleOut, residualSum, ws, ex);
}
aclnnStatus aclnnAddRmsNormDynamicQuantV2(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAddRmsNormDynamicQuant(ws, wsz, e, s); }

// AddRmsNormQuant: static scale+offset. x, residual, gamma, scale, offset, eps -> yQuant[int8], residualSum.
aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
        double scale, double offset, double eps, aclTensor *yQuant, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !yQuant || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (yQuant->dtype != ACL_INT8 || x->viewDims != residual->viewDims || yQuant->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = residual; e->c = gamma; e->out = yQuant; e->out2 = residualSum;
    e->reduceCount = Dd; e->outerCount = x->numel()/Dd; e->eps = eps; e->dscalars = {scale, offset};
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s; float scale=(float)e->dscalars[0], off=(float)e->dscalars[1];
    #define SQ(T,A) k_addrms_quant<T,A><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)D_(e->b),(const T*)D_(e->c), e->out2?(T*)e->out2->data:nullptr, (int8_t*)e->out->data, rows, Dd, (float)e->eps, scale, off)
    switch (e->a->dtype) { case ACL_FLOAT: SQ(float,true); break; case ACL_FLOAT16: SQ(__half,true); break; default: SQ(__nv_bfloat16,true); break; }
    #undef SQ
    return fin(e);
}
aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
        double scale, double offset, double eps, aclTensor *yQuant, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddRmsNormQuantGetWorkspaceSize(x, residual, gamma, scale, offset, eps, yQuant, residualSum, ws, ex);
}
aclnnStatus aclnnAddRmsNormQuantV2(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAddRmsNormQuant(ws, wsz, e, s); }

// RmsNormQuant: static scale+offset (no residual). x, gamma, scale, offset, eps -> yQuant[int8].
aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double scale, double offset, double eps,
        aclTensor *yQuant, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !yQuant || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (yQuant->dtype != ACL_INT8 || yQuant->viewDims != x->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    if (gamma && (gamma->viewDims.size() != 1 || gamma->viewDims[0] != Dd)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->c = gamma; e->out = yQuant;
    e->reduceCount = Dd; e->outerCount = x->numel()/Dd; e->eps = eps; e->dscalars = {scale, offset};
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s; float scale=(float)e->dscalars[0], off=(float)e->dscalars[1];
    #define RQ(T) k_addrms_quant<T,false><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)nullptr,(const T*)D_(e->c), (T*)nullptr, (int8_t*)e->out->data, rows, Dd, (float)e->eps, scale, off)
    switch (e->a->dtype) { case ACL_FLOAT: RQ(float); break; case ACL_FLOAT16: RQ(__half); break; default: RQ(__nv_bfloat16); break; }
    #undef RQ
    return fin(e);
}

} // extern "C"
} // namespace _norm5_ext

namespace _norm6_ext {
// BatchNorm functional decomposition (stats / elemt / reduce / gather-with-counts) + norm backward
// (group-norm, deep-norm, group-norm-swish) + norm naming-forwards + norm-then-quant fusions.
//   BatchNorm layout: [N, C, *] with channel = dim 1; per-channel reduce count = N * spatial.
//   Backward formulas are the standard normalization gradients; the *DynamicMxQuant fusions compute the
//   norm into an fp32 scratch buffer then per-block power-of-2 (E8M0/MX-style) int8 quantize.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
template <typename T> __device__ inline float ld(const T *p, int64_t i){ return (float)p[i]; }

// ---- BatchNorm stats: one block per channel, reduce over N*HW ----
template <typename T> __global__ void k_bn_stats(const T *x, float *mean, float *invstd, int64_t N, int64_t C, int64_t HW, float eps) {
    int64_t c = blockIdx.x; if (c >= C) return; int64_t cnt = N * HW;
    __shared__ float ss, ss2; float s=0, s2=0;
    for (int64_t i = threadIdx.x; i < cnt; i += blockDim.x) { int64_t n=i/HW, h=i%HW; float v=ld(x, (n*C+c)*HW+h); s+=v; s2+=v*v; }
    for (int o=warpSize/2;o>0;o>>=1){ s+=__shfl_down_sync(0xffffffff,s,o); s2+=__shfl_down_sync(0xffffffff,s2,o); }
    if (threadIdx.x==0){ ss=0; ss2=0; } __syncthreads();
    if ((threadIdx.x&31)==0){ atomicAdd(&ss,s); atomicAdd(&ss2,s2); } __syncthreads();
    if (threadIdx.x==0){ float m=ss/cnt, var=ss2/cnt-m*m; mean[c]=m; invstd[c]=rsqrtf(var+eps); }
}
// ---- BatchNorm elemt apply: y = (x-mean)*invstd*w + b ----
template <typename T> __global__ void k_bn_elemt(const T *x, const float *mean, const float *invstd, const T *w, const T *b, T *o, int64_t total, int64_t C, int64_t HW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=total) return; int64_t c=(i/HW)%C;
    float wv = w?(float)w[c]:1.f, bv = b?(float)b[c]:0.f;
    o[i]=(T)(((float)x[i]-mean[c])*invstd[c]*wv+bv);
}
// ---- BatchNorm backward reduce: per channel sumDy and sumDyXmu, gradWeight/gradBias ----
template <typename T> __global__ void k_bn_reduce(const T *gradOut, const T *x, const float *mean, const float *invstd,
        float *sumDy, float *sumDyXmu, float *gradW, float *gradB, int64_t N, int64_t C, int64_t HW) {
    int64_t c = blockIdx.x; if (c >= C) return; int64_t cnt=N*HW;
    __shared__ float sa, sb; float a=0,b=0;
    for (int64_t i=threadIdx.x;i<cnt;i+=blockDim.x){ int64_t n=i/HW,h=i%HW; int64_t idx=(n*C+c)*HW+h; float dy=ld(gradOut,idx); a+=dy; b+=dy*((float)x[idx]-mean[c]); }
    for (int o=warpSize/2;o>0;o>>=1){ a+=__shfl_down_sync(0xffffffff,a,o); b+=__shfl_down_sync(0xffffffff,b,o); }
    if (threadIdx.x==0){ sa=0; sb=0; } __syncthreads();
    if ((threadIdx.x&31)==0){ atomicAdd(&sa,a); atomicAdd(&sb,b); } __syncthreads();
    if (threadIdx.x==0){ if(sumDy)sumDy[c]=sa; if(sumDyXmu)sumDyXmu[c]=sb; if(gradW)gradW[c]=sb*invstd[c]; if(gradB)gradB[c]=sa; }
}
// ---- BatchNorm elemt backward: gradInput from per-channel reductions ----
template <typename T> __global__ void k_bn_elemt_bwd(const T *gradOut, const T *x, const float *mean, const float *invstd, const T *w,
        const float *sumDy, const float *sumDyXmu, T *gradIn, int64_t total, int64_t C, int64_t HW, int64_t cnt) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=total) return; int64_t c=(i/HW)%C;
    float wv = w?(float)w[c]:1.f, is=invstd[c];
    float dy=(float)gradOut[i], xmu=(float)x[i]-mean[c];
    float gi = is*wv*(dy - sumDy[c]/cnt - xmu*is*is*sumDyXmu[c]/cnt);
    gradIn[i]=(T)gi;
}
// ---- gather stats with counts: combine per-group (mean,invstd,count) → global per channel ----
__global__ void k_gather_stats(const float *means, const float *invstds, const float *counts, float *mean, float *invstd, int64_t L, int64_t C, float eps) {
    int64_t c=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(c>=C) return;
    float tot=0, mAcc=0;
    for (int64_t l=0;l<L;l++){ float cnt=counts[l]; tot+=cnt; mAcc+=cnt*means[l*C+c]; }
    float m = tot>0? mAcc/tot : 0.f;
    float m2=0;
    for (int64_t l=0;l<L;l++){ float cnt=counts[l], mi=means[l*C+c], vi=1.f/(invstds[l*C+c]*invstds[l*C+c])-eps; m2+=cnt*(vi+(mi-m)*(mi-m)); }
    float var = tot>0? m2/tot : 0.f;
    mean[c]=m; invstd[c]=rsqrtf(var+eps);
}
// ---- quantized batchnorm: int8 out = round((bn(x))/scale)+zp ----
template <typename T> __global__ void k_qbn(const T *x, const float *mean, const float *invstd, const T *w, const T *b, int8_t *o,
        int64_t total, int64_t C, int64_t HW, float scale, int zp) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=total) return; int64_t c=(i/HW)%C;
    float wv=w?(float)w[c]:1.f, bv=b?(float)b[c]:0.f;
    float y=((float)x[i]-mean[c])*invstd[c]*wv+bv;
    int q=(int)lrintf(y/scale)+zp; q=q<-128?-128:(q>127?127:q); o[i]=(int8_t)q;
}
// ---- GroupNorm backward: block per (n,g); reduce over (C/G)*HW ----
template <typename T> __global__ void k_gn_bwd(const T *gradOut, const T *x, const float *mean, const float *rstd, const T *gamma,
        T *gradX, float *gradGamma, float *gradBeta, int64_t N, int64_t C, int64_t G, int64_t HW) {
    int64_t ng = blockIdx.x; if (ng >= N*G) return; int64_t n=ng/G, g=ng%G; int64_t cpg=C/G; int64_t cnt=cpg*HW;
    float m=mean[ng], rs=rstd[ng];
    __shared__ float s1, s2; float a=0, b=0; // a=Σ dyhat, b=Σ dyhat*xhat  (dyhat = dy*gamma)
    for (int64_t i=threadIdx.x;i<cnt;i+=blockDim.x){ int64_t cc=i/HW, h=i%HW; int64_t c=g*cpg+cc; int64_t idx=(n*C+c)*HW+h;
        float gv=gamma?(float)gamma[c]:1.f; float dy=(float)gradOut[idx]*gv; float xhat=((float)x[idx]-m)*rs; a+=dy; b+=dy*xhat; }
    for (int o=warpSize/2;o>0;o>>=1){ a+=__shfl_down_sync(0xffffffff,a,o); b+=__shfl_down_sync(0xffffffff,b,o); }
    if (threadIdx.x==0){ s1=0; s2=0; } __syncthreads();
    if ((threadIdx.x&31)==0){ atomicAdd(&s1,a); atomicAdd(&s2,b); } __syncthreads();
    __syncthreads(); float sa=s1, sb=s2;
    for (int64_t i=threadIdx.x;i<cnt;i+=blockDim.x){ int64_t cc=i/HW, h=i%HW; int64_t c=g*cpg+cc; int64_t idx=(n*C+c)*HW+h;
        float gv=gamma?(float)gamma[c]:1.f; float dy=(float)gradOut[idx]*gv; float xhat=((float)x[idx]-m)*rs;
        float gi = rs*(dy - sa/cnt - xhat*sb/cnt); gradX[idx]=(T)gi;
        if (gradGamma) atomicAdd(&gradGamma[c], (float)gradOut[idx]*xhat);
        if (gradBeta)  atomicAdd(&gradBeta[c], (float)gradOut[idx]); }
}
// ---- DeepNorm backward: y = LN(alpha*x + gx); given gradY, produce gradX (= alpha * gradInputToLN), gradGx, gradGamma, gradBeta ----
template <typename T> __global__ void k_deepnorm_bwd(const T *gradY, const T *x, const T *gx, const T *gamma, float alpha,
        T *gradX, T *gradGx, float *gradGamma, float *gradBeta, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r>=rows) return; const int64_t base=r*D;
    __shared__ float sm, sv, s1, s2;
    float m=0; for (int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float in=alpha*(float)x[base+i]+(float)gx[base+i]; m+=in; }
    for (int o=warpSize/2;o>0;o>>=1) m+=__shfl_down_sync(0xffffffff,m,o);
    if(threadIdx.x==0)sm=0; __syncthreads(); if((threadIdx.x&31)==0)atomicAdd(&sm,m); __syncthreads(); float mean=sm/D;
    float v=0; for (int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float in=alpha*(float)x[base+i]+(float)gx[base+i]; v+=(in-mean)*(in-mean); }
    for (int o=warpSize/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
    if(threadIdx.x==0)sv=0; __syncthreads(); if((threadIdx.x&31)==0)atomicAdd(&sv,v); __syncthreads();
    float var=sv/D, rs=rsqrtf(var+eps);
    float a=0,b=0; for (int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float in=alpha*(float)x[base+i]+(float)gx[base+i]; float xhat=(in-mean)*rs;
        float gv=gamma?(float)gamma[i]:1.f; float dy=(float)gradY[base+i]*gv; a+=dy; b+=dy*xhat; }
    for (int o=warpSize/2;o>0;o>>=1){ a+=__shfl_down_sync(0xffffffff,a,o); b+=__shfl_down_sync(0xffffffff,b,o); }
    if(threadIdx.x==0){s1=0;s2=0;} __syncthreads(); if((threadIdx.x&31)==0){atomicAdd(&s1,a);atomicAdd(&s2,b);} __syncthreads();
    float sa=s1, sb=s2;
    for (int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float in=alpha*(float)x[base+i]+(float)gx[base+i]; float xhat=(in-mean)*rs;
        float gv=gamma?(float)gamma[i]:1.f; float dy=(float)gradY[base+i]*gv;
        float dIn = rs*(dy - sa/D - xhat*sb/D);
        if(gradX) gradX[base+i]=(T)(alpha*dIn); if(gradGx) gradGx[base+i]=(T)dIn;
        if(gradGamma) atomicAdd(&gradGamma[i], (float)gradY[base+i]*xhat);
        if(gradBeta) atomicAdd(&gradBeta[i], (float)gradY[base+i]); }
}
#define DT3(KC) switch(dt){case ACL_FLOAT:{using T=float;KC;}break;case ACL_FLOAT16:{using T=__half;KC;}break;default:{using T=__nv_bfloat16;KC;}break;}
} // namespace

extern "C" {

// ===== BatchNorm functional decomposition =====
aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!mean||!invstd||!ex||self->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->mean=mean; e->rstd=invstd; e->eps=eps; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormStats(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->a->viewDims[0],C=e->a->viewDims[1],HW=e->a->numel()/(N*C); auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_bn_stats<T><<<(unsigned)C,TH,0,st>>>((const T*)e->a->data,(float*)e->mean->data,(float*)e->rstd->data,N,C,HW,(float)e->eps) ));
    return fin(e);
}
aclnnStatus aclnnBatchNormElemtGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!mean||!invstd||!out||!ex||self->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->b=weight; e->c=bias; e->mean=mean; e->rstd=invstd; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormElemt(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->a->viewDims[0],C=e->a->viewDims[1],total=e->a->numel(),HW=total/(N*C); auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_bn_elemt<T><<<nb(total),TH,0,st>>>((const T*)e->a->data,(const float*)e->mean->data,(const float*)e->rstd->data,(const T*)(e->b?e->b->data:nullptr),(const T*)(e->c?e->c->data:nullptr),(T*)e->out->data,total,C,HW) ));
    return fin(e);
}
// BatchNormReduce: backward reduce → sumDy, sumDyXmu, gradWeight, gradBias
aclnnStatus aclnnBatchNormReduceGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd,
        aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOut||!self||!mean||!invstd||!ex||self->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=self; e->mean=mean; e->rstd=invstd; e->out=sumDy; e->out2=sumDyXmu;
    if(gradWeight)e->inputs.push_back(gradWeight); if(gradBias)e->inputs.push_back(gradBias); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormReduce(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->b->viewDims[0],C=e->b->viewDims[1],HW=e->b->numel()/(N*C); auto st=(cudaStream_t)s; aclDataType dt=e->b->dtype;
    float *gw = e->inputs.size()>=1? (float*)const_cast<aclTensor*>(e->inputs[0])->data : nullptr;
    float *gb = e->inputs.size()>=2? (float*)const_cast<aclTensor*>(e->inputs[1])->data : nullptr;
    DT3(( k_bn_reduce<T><<<(unsigned)C,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const float*)e->mean->data,(const float*)e->rstd->data,
        e->out?(float*)e->out->data:nullptr, e->out2?(float*)e->out2->data:nullptr, gw, gb, N,C,HW) ));
    return fin(e);
}
// BatchNormReduceBackward: naming variant of BatchNormReduce
aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd,
        aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBatchNormReduceGetWorkspaceSize(gradOut, self, mean, invstd, sumDy, sumDyXmu, gradWeight, gradBias, ws, ex);
}
aclnnStatus aclnnBatchNormReduceBackward(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBatchNormReduce(w,wz,e,s); }
// BatchNormElemtBackward: gradInput from reductions
aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd,
        const aclTensor *weight, const aclTensor *sumDy, const aclTensor *sumDyXmu, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOut||!self||!mean||!invstd||!sumDy||!sumDyXmu||!gradInput||!ex) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=self; e->c=weight; e->mean=mean; e->rstd=invstd; e->out=gradInput;
    e->inputs.push_back(sumDy); e->inputs.push_back(sumDyXmu); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormElemtBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->b->viewDims[0],C=e->b->viewDims[1],total=e->b->numel(),HW=total/(N*C),cnt=N*HW; auto st=(cudaStream_t)s; aclDataType dt=e->b->dtype;
    DT3(( k_bn_elemt_bwd<T><<<nb(total),TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const float*)e->mean->data,(const float*)e->rstd->data,
        (const T*)(e->c?e->c->data:nullptr),(const float*)e->inputs[0]->data,(const float*)e->inputs[1]->data,(T*)e->out->data,total,C,HW,cnt) ));
    return fin(e);
}
// BatchNormGatherStatsWithCounts: combine per-group stats. meanAll/invstdAll [L,C], counts [L]
aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts,
        double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex){
    if(!meanAll||!invstdAll||!counts||!mean||!invstd||!ex||meanAll->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=meanAll; e->b=invstdAll; e->c=counts; e->mean=mean; e->rstd=invstd; e->eps=eps; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormGatherStatsWithCounts(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t L=e->a->viewDims[0], C=e->a->viewDims[1]; auto st=(cudaStream_t)s;
    k_gather_stats<<<nb(C),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(float*)e->mean->data,(float*)e->rstd->data,L,C,(float)e->eps);
    return fin(e);
}
// SyncBatchNormGatherStats: naming variant
aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts,
        double eps, aclTensor *mean, aclTensor *invstd, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(meanAll, invstdAll, counts, eps, mean, invstd, ws, ex);
}
aclnnStatus aclnnSyncBatchNormGatherStats(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBatchNormGatherStatsWithCounts(w,wz,e,s); }
// QuantizedBatchNorm: bn then int8 quant (scale, zeroPoint)
aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd,
        double scale, int64_t zeroPoint, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)eps; if(!self||!mean||!invstd||!out||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->b=weight; e->c=bias; e->mean=mean; e->rstd=invstd; e->out=out; e->alpha=scale; e->dim=zeroPoint; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantizedBatchNorm(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->a->viewDims[0],C=e->a->viewDims[1],total=e->a->numel(),HW=total/(N*C); auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_qbn<T><<<nb(total),TH,0,st>>>((const T*)e->a->data,(const float*)e->mean->data,(const float*)e->rstd->data,(const T*)(e->b?e->b->data:nullptr),(const T*)(e->c?e->c->data:nullptr),(int8_t*)e->out->data,total,C,HW,(float)e->alpha,(int)e->dim) ));
    return fin(e);
}
// FastBatchNormBackward: forward to BatchNormBackward (saved mean/invstd)
aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, const aclTensor *savedMean, const aclTensor *savedInvStd,
        aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBatchNormBackwardGetWorkspaceSize(gradY, x, gamma, savedMean, savedInvStd, gradX, gradGamma, gradBeta, ws, ex);
}
aclnnStatus aclnnFastBatchNormBackward(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBatchNormBackward(w,wz,e,s); }

// ===== GroupNorm backward =====
aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma,
        int64_t numGroups, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOut||!self||!mean||!rstd||!gradX||!ex||self->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=self; e->mean=mean; e->rstd=rstd; e->c=gamma; e->out=gradX; e->out2=gradGamma;
    if(gradBeta)e->inputs.push_back(gradBeta); e->dim=numGroups; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNormBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t N=e->b->viewDims[0],C=e->b->viewDims[1],G=e->dim,HW=e->b->numel()/(N*C); auto st=(cudaStream_t)s; aclDataType dt=e->b->dtype;
    float *gg=e->out2?(float*)e->out2->data:nullptr; float *gb=e->inputs.empty()?nullptr:(float*)const_cast<aclTensor*>(e->inputs[0])->data;
    if(gg)cudaMemsetAsync(gg,0,C*sizeof(float),st); if(gb)cudaMemsetAsync(gb,0,C*sizeof(float),st);
    DT3(( k_gn_bwd<T><<<(unsigned)(N*G),TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const float*)e->mean->data,(const float*)e->rstd->data,(const T*)(e->c?e->c->data:nullptr),(T*)e->out->data,gg,gb,N,C,G,HW) ));
    return fin(e);
}
// GroupNormSwishGrad: reuse group-norm backward gradient w.r.t. the normalized input (swish-applied gradOut handled by caller); same reduction
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma,
        int64_t numGroups, double swishScale, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex){
    (void)swishScale; return aclnnGroupNormBackwardGetWorkspaceSize(gradOut, self, mean, rstd, gamma, numGroups, gradX, gradGamma, gradBeta, ws, ex);
}
aclnnStatus aclnnGroupNormSwishGrad(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGroupNormBackward(w,wz,e,s); }

// ===== DeepNorm backward =====
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gx, const aclTensor *gamma,
        double alpha, double eps, aclTensor *gradX, aclTensor *gradGx, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex){
    if(!gradY||!x||!gx||!ex||x->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradY; e->b=x; e->c=gx; e->mask=gamma; e->out=gradX; e->out2=gradGx;
    if(gradGamma)e->inputs.push_back(gradGamma); if(gradBeta)e->inputs.push_back(gradBeta);
    e->alpha=alpha; e->eps=eps; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDeepNormGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->b->dtype;
    float *gg=e->inputs.size()>=1?(float*)const_cast<aclTensor*>(e->inputs[0])->data:nullptr;
    float *gb=e->inputs.size()>=2?(float*)const_cast<aclTensor*>(e->inputs[1])->data:nullptr;
    if(gg)cudaMemsetAsync(gg,0,D*sizeof(float),st); if(gb)cudaMemsetAsync(gb,0,D*sizeof(float),st);
    DT3(( k_deepnorm_bwd<T><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)e->c->data,(const T*)(e->mask?e->mask->data:nullptr),(float)e->alpha,(T*)e->out->data,e->out2?(T*)e->out2->data:nullptr,gg,gb,rows,D,(float)e->eps) ));
    return fin(e);
}

// ===== LayerNorm naming forwards =====
aclnnStatus aclnnLayerNorm_noFunctionalGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias,
        double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLayerNormGetWorkspaceSize(input, normalizedShape, weight, bias, eps, out, meanOut, rstdOut, ws, ex);
}
aclnnStatus aclnnLayerNorm_noFunctional(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLayerNorm(w,wz,e,s); }
aclnnStatus aclnnLayerNorm_withFunctionalGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias,
        double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLayerNormGetWorkspaceSize(input, normalizedShape, weight, bias, eps, out, meanOut, rstdOut, ws, ex);
}
aclnnStatus aclnnLayerNorm_withFunctional(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLayerNorm(w,wz,e,s); }

} // extern "C"

// ===== norm-then-MX-quant fusions: per-row norm, then per-row power-of-2 (E8M0/MX-style) int8 quant =====
namespace {
// blockDim threads cooperate on one row of D elements; warp-shuffle + shared reduction.
template <typename T> __device__ inline float row_sum(float v){
    for (int o=warpSize/2;o>0;o>>=1) v+=__shfl_down_sync(0xffffffff,v,o);
    __shared__ float acc; if(threadIdx.x==0)acc=0; __syncthreads();
    if((threadIdx.x&31)==0)atomicAdd(&acc,v); __syncthreads(); return acc;
}
template <typename T> __global__ void k_rmsnorm_mxq(const T *x, const T *res, const T *gamma, T *resSum, int8_t *o, float *scaleOut,
        int64_t rows, int64_t D, float eps){
    int64_t r=blockIdx.x; if(r>=rows) return; const int64_t base=r*D;
    float ss=0; for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float xi=(float)x[base+i]+(res?(float)res[base+i]:0.f); if(res&&resSum)resSum[base+i]=(T)xi; ss+=xi*xi; }
    float ms=row_sum<T>(ss)/D, rrms=rsqrtf(ms+eps);
    float amax=0; for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float xi=(float)x[base+i]+(res?(float)res[base+i]:0.f); float y=xi*rrms*(gamma?(float)gamma[i]:1.f); amax=fmaxf(amax,fabsf(y)); }
    for (int op=warpSize/2;op>0;op>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffff,amax,op));
    __shared__ float sAmax, sScale; if(threadIdx.x==0)sAmax=0; __syncthreads(); if((threadIdx.x&31)==0)atomicMax((int*)&sAmax,__float_as_int(amax)); __syncthreads();
    if(threadIdx.x==0){ float a=sAmax; sScale = a>0? exp2f(ceilf(log2f(a/127.f))) : 1.f; scaleOut[r]=sScale; } __syncthreads();
    float sc=sScale;
    for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float xi=(float)x[base+i]+(res?(float)res[base+i]:0.f); float y=xi*rrms*(gamma?(float)gamma[i]:1.f); int q=(int)lrintf(y/sc); q=q<-128?-128:(q>127?127:q); o[base+i]=(int8_t)q; }
}
template <typename T> __global__ void k_adaln_mxq(const T *x, const T *scaleT, const T *shiftT, int8_t *o, float *scaleOut,
        int64_t rows, int64_t D, float eps){
    int64_t r=blockIdx.x; if(r>=rows) return; const int64_t base=r*D;
    float s=0; for(int64_t i=threadIdx.x;i<D;i+=blockDim.x) s+=(float)x[base+i];
    float mean=row_sum<T>(s)/D;
    float v=0; for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float d=(float)x[base+i]-mean; v+=d*d; }
    float var=row_sum<T>(v)/D, rstd=rsqrtf(var+eps);
    float amax=0; for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float n=((float)x[base+i]-mean)*rstd; float y=n*(1.f+(float)scaleT[base+i])+(float)shiftT[base+i]; amax=fmaxf(amax,fabsf(y)); }
    for (int op=warpSize/2;op>0;op>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffff,amax,op));
    __shared__ float sAmax, sScale; if(threadIdx.x==0)sAmax=0; __syncthreads(); if((threadIdx.x&31)==0)atomicMax((int*)&sAmax,__float_as_int(amax)); __syncthreads();
    if(threadIdx.x==0){ float a=sAmax; sScale=a>0?exp2f(ceilf(log2f(a/127.f))):1.f; scaleOut[r]=sScale; } __syncthreads();
    float sc=sScale;
    for(int64_t i=threadIdx.x;i<D;i+=blockDim.x){ float n=((float)x[base+i]-mean)*rstd; float y=n*(1.f+(float)scaleT[base+i])+(float)shiftT[base+i]; int q=(int)lrintf(y/sc); q=q<-128?-128:(q>127?127:q); o[base+i]=(int8_t)q; }
}
} // namespace

extern "C" {
aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->c=gamma; e->out=out; e->out2=scaleOut; e->eps=eps; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRmsNormDynamicMxQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_rmsnorm_mxq<T><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,nullptr,(const T*)(e->c?e->c->data:nullptr),nullptr,(int8_t*)e->out->data,(float*)e->out2->data,rows,D,(float)e->eps) ));
    return fin(e);
}
aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps,
        aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!residual||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=residual; e->c=gamma; e->out=out; e->out2=scaleOut; e->mean=residualSum; e->eps=eps; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRmsNormDynamicMxQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_rmsnorm_mxq<T><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)(e->c?e->c->data:nullptr),e->mean?(T*)e->mean->data:nullptr,(int8_t*)e->out->data,(float*)e->out2->data,rows,D,(float)e->eps) ));
    return fin(e);
}
aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps,
        aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!scale||!shift||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=scale; e->c=shift; e->out=out; e->out2=scaleOut; e->eps=eps; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaLayerNormQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_adaln_mxq<T><<<(unsigned)rows,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)e->c->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D,(float)e->eps) ));
    return fin(e);
}
} // extern "C"
} // namespace _norm6_ext
#undef DT3

namespace _normquant_ext {
// Norm/activation + quant fusions: LayerNormQuant, AddLayerNormQuant, GroupNormSiluQuant, DequantSwigluQuant.
// All compute the float result then per-token (per-row / per-group) dynamic int8 quant (absmax/127). fp32/fp16/bf16 in.

namespace {
constexpr int TB = 256;
__device__ inline int8_t clip8(float v){ int q=__float2int_rn(v); return (int8_t)(q<-127?-127:(q>127?127:q)); }

// LayerNorm(+add) over last dim D → y=n·gamma+beta → per-row absmax int8. ADD: s=x+res, write residualSum.
template <typename T, bool ADD>
__global__ void k_ln_quant(const T *x, const T *res, const T *g, const T *b, T *ysum, int8_t *out, float *scaleOut, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r*D; int t = threadIdx.x;
    __shared__ float sh[TB/32]; __shared__ float bc;
    auto red=[&](float v,int mx)->float{
        #pragma unroll
        for(int o=16;o>0;o>>=1){ float o2=__shfl_down_sync(0xffffffffu,v,o); v = mx?fmaxf(v,o2):v+o2; }
        if((t&31)==0) sh[t>>5]=v; __syncthreads();
        if(t==0){ float a=sh[0]; for(int w=1;w<TB/32;w++) a = mx?fmaxf(a,sh[w]):a+sh[w]; bc=a; } __syncthreads();
        float rr=bc; __syncthreads(); return rr; };
    float sm=0; for(int64_t d=t;d<D;d+=TB){ float s=(float)x[base+d]+(ADD?(float)res[base+d]:0.f); if(ADD&&ysum) ysum[base+d]=(T)s; sm+=s; }
    float mean=red(sm,0)/D;
    float v=0; for(int64_t d=t;d<D;d+=TB){ float s=(float)x[base+d]+(ADD?(float)res[base+d]:0.f); float u=s-mean; v+=u*u; }
    float inv=rsqrtf(red(v,0)/D+eps);
    float amax=0; for(int64_t d=t;d<D;d+=TB){ float s=(float)x[base+d]+(ADD?(float)res[base+d]:0.f); float y=(s-mean)*inv*(g?(float)g[d]:1.f)+(b?(float)b[d]:0.f); amax=fmaxf(amax,fabsf(y)); }
    amax=red(amax,1); float sc=amax>0?amax/127.f:1.f; if(t==0) scaleOut[r]=sc; float qi=1.f/sc;
    for(int64_t d=t;d<D;d+=TB){ float s=(float)x[base+d]+(ADD?(float)res[base+d]:0.f); float y=(s-mean)*inv*(g?(float)g[d]:1.f)+(b?(float)b[d]:0.f); out[base+d]=clip8(y*qi); }
}
// GroupNorm+SiLU then per-(n,grp) absmax int8. One block per (n,grp).
template <typename T>
__global__ void k_gn_silu_quant(const T *x, const T *g, const T *b, int8_t *out, float *scaleOut, int64_t C, int64_t HW, int64_t G, float eps) {
    int64_t blk=blockIdx.x, Cg=C/G, n=blk/G, grp=blk%G, cnt=Cg*HW, base=(n*C+grp*Cg)*HW; int t=threadIdx.x;
    __shared__ float sh[TB/32]; __shared__ float bc;
    auto red=[&](float v,int mx)->float{
        #pragma unroll
        for(int o=16;o>0;o>>=1){ float o2=__shfl_down_sync(0xffffffffu,v,o); v=mx?fmaxf(v,o2):v+o2; }
        if((t&31)==0) sh[t>>5]=v; __syncthreads();
        if(t==0){ float a=sh[0]; for(int w=1;w<TB/32;w++) a=mx?fmaxf(a,sh[w]):a+sh[w]; bc=a;} __syncthreads();
        float rr=bc; __syncthreads(); return rr; };
    float sum=0; for(int64_t i=t;i<cnt;i+=TB) sum+=(float)x[base+i];
    float mean=red(sum,0)/cnt;
    float vv=0; for(int64_t i=t;i<cnt;i+=TB){ float u=(float)x[base+i]-mean; vv+=u*u; }
    float inv=rsqrtf(red(vv,0)/cnt+eps);
    float amax=0; for(int64_t i=t;i<cnt;i+=TB){ int64_t c=grp*Cg+i/HW; float y=((float)x[base+i]-mean)*inv*(g?(float)g[c]:1.f)+(b?(float)b[c]:0.f); y=y/(1.f+expf(-y)); amax=fmaxf(amax,fabsf(y)); }
    amax=red(amax,1); float sc=amax>0?amax/127.f:1.f; if(t==0) scaleOut[blk]=sc; float qi=1.f/sc;
    for(int64_t i=t;i<cnt;i+=TB){ int64_t c=grp*Cg+i/HW; float y=((float)x[base+i]-mean)*inv*(g?(float)g[c]:1.f)+(b?(float)b[c]:0.f); y=y/(1.f+expf(-y)); out[base+i]=clip8(y*qi); }
}
// DequantSwigluQuant: xd = x·dq (per-row dq, optional); swiglu over [...,2D]→[...,D]; per-row absmax int8.
template <typename T>
__global__ void k_dequant_swiglu_quant(const T *x, const float *dq, int8_t *out, float *scaleOut, int64_t rows, int64_t D) {
    int64_t r=blockIdx.x; if(r>=rows) return; int64_t ib=r*2*D, ob=r*D; int t=threadIdx.x; float scl = dq?dq[r]:1.f;
    __shared__ float sh[TB/32]; __shared__ float bc;
    auto red=[&](float v)->float{
        #pragma unroll
        for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_down_sync(0xffffffffu,v,o));
        if((t&31)==0) sh[t>>5]=v; __syncthreads();
        if(t==0){ float a=sh[0]; for(int w=1;w<TB/32;w++) a=fmaxf(a,sh[w]); bc=a;} __syncthreads();
        float rr=bc; __syncthreads(); return rr; };
    float amax=0; for(int64_t d=t;d<D;d+=TB){ float a=(float)x[ib+d]*scl, bb=(float)x[ib+D+d]*scl; float gg=a/(1.f+expf(-a))*bb; amax=fmaxf(amax,fabsf(gg)); }
    amax=red(amax); float sc=amax>0?amax/127.f:1.f; if(t==0) scaleOut[r]=sc; float qi=1.f/sc;
    for(int64_t d=t;d<D;d+=TB){ float a=(float)x[ib+d]*scl, bb=(float)x[ib+D+d]*scl; float gg=a/(1.f+expf(-a))*bb; out[ob+d]=clip8(gg*qi); }
}
inline const void *D_(const aclTensor *t){ return t?t->data:nullptr; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
#define DT3(KCALL) switch(e->a->dtype){ case ACL_FLOAT:{using T=float;KCALL;}break; case ACL_FLOAT16:{using T=__half;KCALL;}break; default:{using T=__nv_bfloat16;KCALL;}break; }
} // namespace

extern "C" {

// LayerNormQuant: x, gamma, beta, eps -> out int8[...,D] + scaleOut[...] (per-token)
aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps,
        aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=x->viewDims.back(); auto *e=new aclOpExecutor(); e->a=x; e->c=gamma; e->mask=beta; e->out=out; e->rstd=scaleOut;
    e->reduceCount=D; e->outerCount=x->numel()/D; e->eps=eps; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLayerNormQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; float eps=(float)e->eps;
    DT3(( k_ln_quant<T,false><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)nullptr,(const T*)D_(e->c),(const T*)D_(e->mask),(T*)nullptr,(int8_t*)e->out->data,(float*)e->rstd->data,rows,D,eps) ));
    return fin(e);
}
// AddLayerNormQuant: x, residual, gamma, beta, eps -> out int8 + scaleOut + residualSum
aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, const aclTensor *beta, double eps,
        aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !residual || !out || !scaleOut || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || x->viewDims != residual->viewDims || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=x->viewDims.back(); auto *e=new aclOpExecutor(); e->a=x; e->b=residual; e->c=gamma; e->mask=beta; e->out=out; e->rstd=scaleOut; e->out2=residualSum;
    e->reduceCount=D; e->outerCount=x->numel()/D; e->eps=eps; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddLayerNormQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; float eps=(float)e->eps;
    DT3(( k_ln_quant<T,true><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(const T*)D_(e->c),(const T*)D_(e->mask),e->out2?(T*)e->out2->data:nullptr,(int8_t*)e->out->data,(float*)e->rstd->data,rows,D,eps) ));
    return fin(e);
}
// GroupNormSiluQuant: self[N,C,*], gamma/beta[C], group, eps -> out int8 + scaleOut[N,G]
aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps,
        aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !scaleOut || !ex || !self->data || self->viewDims.size()<2) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t C=self->viewDims[1]; if (group<=0||C%group!=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->c=gamma; e->mask=beta; e->out=out; e->rstd=scaleOut; e->reduceCount=group; e->eps=eps;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupNormSiluQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t N=e->a->viewDims[0],C=e->a->viewDims[1],G=e->reduceCount,HW=e->a->numel()/(N*C); auto st=(cudaStream_t)s; float eps=(float)e->eps;
    DT3(( k_gn_silu_quant<T><<<(unsigned)(N*G),TB,0,st>>>((const T*)e->a->data,(const T*)D_(e->c),(const T*)D_(e->mask),(int8_t*)e->out->data,(float*)e->rstd->data,C,HW,G,eps) ));
    return fin(e);
}
// DequantSwigluQuant: x[...,2D] (optional per-row dequant scale) -> swiglu -> out int8[...,D] + scaleOut[...]
aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=out->viewDims.back(); if (x->viewDims.back()!=2*D) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=dequantScale; e->out=out; e->rstd=scaleOut; e->reduceCount=D; e->outerCount=out->numel()/D;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDequantSwigluQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; const float *dq=e->b?(const float*)e->b->data:nullptr;
    DT3(( k_dequant_swiglu_quant<T><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,dq,(int8_t*)e->out->data,(float*)e->rstd->data,rows,D) ));
    return fin(e);
}
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDequantSwigluQuantGetWorkspaceSize(x, dequantScale, out, scaleOut, ws, ex);
}
aclnnStatus aclnnDequantSwigluQuantV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnDequantSwigluQuant(w, wz, e, s); }

} // extern "C"
} // namespace _normquant_ext
#undef DT3

