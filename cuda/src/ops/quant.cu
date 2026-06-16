// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <map>
#include <vector>

namespace _quant {
// Explicit quantization/dequantization: Quantize/Dequantize (per-tensor or per-channel),
// DynamicQuant (per-token symmetric).
//   Quantize:    q = clamp(round(x·scale + offset), -128, 127)  (int8)
//   Dequantize:  y = ((float)q - offset)·scale
//   DynamicQuant: per row along the last dim, amax→scale=amax/127, q=round(x/scale);
//                 outputs q(int8) + scale(fp32 per row).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__device__ inline int8_t clamp_i8(float v) { int q = (int)rintf(v); return (int8_t)(q < -128 ? -128 : q > 127 ? 127 : q); }

template <typename T> __global__ void k_quant(const T *x, const float *sc, const float *of, int8_t *q, int64_t n, int64_t C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t c = C > 1 ? i % C : 0;
    q[i] = clamp_i8((float)x[i] * sc[c] + (of ? of[c] : 0.f));
}
template <typename T> __global__ void k_dequant(const int8_t *q, const float *sc, const float *of, T *y, int64_t n, int64_t C) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t c = C > 1 ? i % C : 0;
    y[i] = (T)(((float)q[i] - (of ? of[c] : 0.f)) * sc[c]);
}
// Fast path: one block per row, threads stride over D for coalesced access + warp/block amax reduction
// (replaces the non-coalesced one-thread-per-row approach).
template <typename T, int TB> __global__ void k_dynquant_fast(const T *x, int8_t *q, float *scaleOut, int64_t rows, int64_t D) {
    int64_t r = blockIdx.x; if (r >= rows) return; int t = threadIdx.x; int64_t base = r * D;
    float amax = 0;
    for (int64_t d = t; d < D; d += TB) amax = fmaxf(amax, fabsf((float)x[base + d]));
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    __shared__ float sh[TB / 32];
    if ((t & 31) == 0) sh[t >> 5] = amax; __syncthreads();
    if (t == 0) { float m = 0; for (int w = 0; w < TB / 32; w++) m = fmaxf(m, sh[w]); sh[0] = (m == 0 ? 1.f : m / 127.f); scaleOut[r] = sh[0]; }
    __syncthreads();
    float inv = 1.f / sh[0];
    for (int64_t d = t; d < D; d += TB) q[base + d] = clamp_i8((float)x[base + d] * inv);
}

#define QDISP(KCALL) do { switch (e->a->dtype) {                  \
    case ACL_FLOAT:   { using T=float;        KCALL; } break;     \
    case ACL_FLOAT16: { using T=__half;       KCALL; } break;     \
    case ACL_BF16:    { using T=__nv_bfloat16; KCALL; } break;    \
    default: delete e; return ACLNN_ERR_PARAM_INVALID; } } while (0)
inline aclnnStatus fin(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// Quantize: x(fp)→int8. scale/offset are fp32, length 1 (per-tensor) or last-dim C (per-channel); offset may be null.
aclnnStatus aclnnQuantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || !x->data || !scale->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scale->dtype != ACL_FLOAT || (offset && offset->dtype != ACL_FLOAT)) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = x; e->c = scale; e->mask = offset; e->out = out;
    e->reduceCount = scale->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantize(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->a->numel(), g = nb(n); auto st = (cudaStream_t)s;
    const float *sc = (const float *)e->c->data, *of = e->mask ? (const float *)e->mask->data : nullptr;
    QDISP(( k_quant<T><<<g,TH,0,st>>>((const T*)e->a->data, sc, of, (int8_t*)e->out->data, n, e->reduceCount) ));
    return fin(e);
}
aclnnStatus aclnnDequantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || !x->data || !scale->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_INT8 || scale->dtype != ACL_FLOAT || (offset && offset->dtype != ACL_FLOAT)) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = out; e->b = x; e->c = scale; e->mask = offset; e->out = out;  // a repurposed to carry out dtype
    e->reduceCount = scale->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDequantize(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(), g = nb(n); auto st = (cudaStream_t)s;
    const float *sc = (const float *)e->c->data, *of = e->mask ? (const float *)e->mask->data : nullptr;
    const int8_t *q = (const int8_t *)e->b->data;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_dequant<float><<<g,TH,0,st>>>(q, sc, of, (float*)e->out->data, n, e->reduceCount); break;
        case ACL_FLOAT16: k_dequant<__half><<<g,TH,0,st>>>(q, sc, of, (__half*)e->out->data, n, e->reduceCount); break;
        default:          k_dequant<__nv_bfloat16><<<g,TH,0,st>>>(q, sc, of, (__nv_bfloat16*)e->out->data, n, e->reduceCount); break;
    }
    return fin(e);
}
// DynamicQuant: x(fp)[...,D]→int8 + scaleOut(fp32)[...] per-token symmetric quantization.
aclnnStatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data || !out->data || !scaleOut->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = x; e->out = out; e->out2 = scaleOut;
    e->reduceCount = D; e->outerCount = x->numel() / D;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows = e->outerCount, D = e->reduceCount; auto st = (cudaStream_t)s;
    QDISP(( k_dynquant_fast<T,256><<<(unsigned)rows,256,0,st>>>((const T*)e->a->data, (int8_t*)e->out->data, (float*)e->out2->data, rows, D) ));
    return fin(e);
}

} // extern "C"
} // namespace _quant
#undef QDISP

namespace _quant_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _quant_ext {
// Quantization extensions (P14): AscendQuant, AscendDequant, AscendAntiQuant, FakeQuant, DequantBias.
// Per-channel scale/offset along the last dim (size C). int8 quantized storage; fp32 real.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__device__ inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

// AscendQuant: q = clamp(round(x*scale[c] + offset[c]), -128, 127)
__global__ void k_ascend_quant(const float *x, const float *sc, const float *of, int8_t *q, int64_t n, int64_t C) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int64_t c=i%C;
    int v=(int)lrintf(x[i]*sc[c] + (of?of[c]:0.f)); q[i]=(int8_t)clampi(v,-128,127);
}
// AscendDequant: y = (q - offset[c]) * scale[c]
__global__ void k_ascend_dequant(const int8_t *q, const float *sc, const float *of, float *y, int64_t n, int64_t C) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int64_t c=i%C; y[i]=((float)q[i]-(of?of[c]:0.f))*sc[c];
}
// DequantBias: int32 accum -> y = q*scale[c] + bias[c]
__global__ void k_dequant_bias(const int32_t *q, const float *sc, const float *bias, float *y, int64_t n, int64_t C) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int64_t c=i%C; y[i]=(float)q[i]*sc[c] + (bias?bias[c]:0.f);
}
// FakeQuant: y = (clamp(round(x/scale + zp), qmin, qmax) - zp) * scale  (straight-through quant-dequant)
__global__ void k_fake_quant(const float *x, float *y, int64_t n, float scale, float zp, int qmin, int qmax) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int q=clampi((int)lrintf(x[i]/scale + zp), qmin, qmax); y[i]=((float)q - zp)*scale;
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnAscendQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || x->dtype != ACL_FLOAT || out->dtype != ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)x->viewDims.size(); auto *e=new aclOpExecutor(); e->a=x; e->b=scale; e->c=offset; e->out=out; e->m=x->numel(); e->n=x->viewDims[rank-1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAscendQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_ascend_quant<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(int8_t*)e->out->data,e->m,e->n); return done(e);
}
aclnnStatus aclnnAscendDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || x->dtype != ACL_INT8 || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)x->viewDims.size(); auto *e=new aclOpExecutor(); e->a=x; e->b=scale; e->c=offset; e->out=out; e->m=x->numel(); e->n=x->viewDims[rank-1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAscendDequant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_ascend_dequant<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const int8_t*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,e->m,e->n); return done(e);
}
// AntiQuant (weight int8 -> float): same arithmetic as dequant (q - offset)*scale
aclnnStatus aclnnAscendAntiQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnAscendDequantGetWorkspaceSize(x, scale, offset, out, ws, ex); }
aclnnStatus aclnnAscendAntiQuant(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAscendDequant(ws, wsz, e, s); }
// DequantBias: int32 -> y = q*scale + bias
aclnnStatus aclnnDequantBiasGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *bias, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || x->dtype != ACL_INT32 || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)x->viewDims.size(); auto *e=new aclOpExecutor(); e->a=x; e->b=scale; e->c=bias; e->out=out; e->m=x->numel(); e->n=x->viewDims[rank-1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDequantBias(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_dequant_bias<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const int32_t*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,e->m,e->n); return done(e);
}
// FakeQuant: per-tensor scale/zeroPoint, qmin/qmax (e.g. -128/127)
aclnnStatus aclnnFakeQuantGetWorkspaceSize(const aclTensor *x, double scale, double zeroPoint, int64_t qmin, int64_t qmax, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex || x->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->m=x->numel(); e->dscalars={scale,zeroPoint,(double)qmin,(double)qmax};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFakeQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_fake_quant<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->m,(float)d[0],(float)d[1],(int)d[2],(int)d[3]); return done(e);
}

} // extern "C"
} // namespace _quant_ext

namespace _quant2_ext {
// Quant matmul version variants. The dequant/antiquant matmul math is identical across versions;
// the version bumps expose extra optional knobs (per-axis/group scales, offset, output dtype) that
// do not change the core result, so they forward to the established cores under the simplified ABI.

extern "C" {

// QuantMatmul V2/V3/V4/V5: int8 x @ int8 weight, dequantized by scale → out.
#define QMM_VARIANT(VER)                                                                                  \
aclnnStatus aclnnQuantMatmul##VER##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, \
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {                                               \
    return aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, out, ws, ex);                               \
}                                                                                                        \
aclnnStatus aclnnQuantMatmul##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnQuantMatmul(ws, wsz, e, s); }
QMM_VARIANT(V2)
QMM_VARIANT(V3)
QMM_VARIANT(V4)
QMM_VARIANT(V5)

// WeightQuantBatchMatmul V2/V3: float x @ antiquantized weight (scale + offset) → out.
#define WQBMM_VARIANT(VER)                                                                                \
aclnnStatus aclnnWeightQuantBatchMatmul##VER##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, \
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnWeightQuantBatchMatmulGetWorkspaceSize(x, weight, antiquantScale, antiquantOffset, out, ws, ex);              \
}                                                                                                        \
aclnnStatus aclnnWeightQuantBatchMatmul##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnWeightQuantBatchMatmul(ws, wsz, e, s); }
WQBMM_VARIANT(V2)
WQBMM_VARIANT(V3)

// QuantMatmulDequant: int8 x @ int8 weight dequantized by scale → float out (same as the QuantMatmul core, which dequants).
aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, out, ws, ex);
}
aclnnStatus aclnnQuantMatmulDequant(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnQuantMatmul(ws, wsz, e, s); }

// DynamicQuant V2/V3/V4: per-token symmetric int8 quant; version bumps add optional smooth-scale/asym knobs → core.
#define DQ_VARIANT(VER)                                                                                   \
aclnnStatus aclnnDynamicQuant##VER##GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnDynamicQuantGetWorkspaceSize(x, out, scaleOut, ws, ex);                                   \
}                                                                                                        \
aclnnStatus aclnnDynamicQuant##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnDynamicQuant(ws, wsz, e, s); }
DQ_VARIANT(V2)
DQ_VARIANT(V3)
DQ_VARIANT(V4)

} // extern "C"
} // namespace _quant2_ext
#undef QMM_VARIANT
#undef WQBMM_VARIANT
#undef DQ_VARIANT

namespace _quant3_ext {
// QuantBatchMatmulInplaceAdd: out += dequant(x_int8[M,K] @ weight_int8[K,N], scale[N]).
// Nested aclnnQuantMatmul writes the product into a workspace temp; an add kernel accumulates into the
// pre-loaded out (residual/bias accumulator). fp16/fp32 out.

namespace {
struct QState { aclOpExecutor *inner; aclTensor *temp; uint64_t mmWs; };
std::map<aclOpExecutor *, QState> g_q;
inline int dt_size(aclDataType d) { return d == ACL_FLOAT ? 4 : 2; }
template <typename T>
__global__ void k_acc(T *out, const T *add, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    out[i] = (T)((float)out[i] + (float)add[i]);
}
// int8 transpose: in[N,K] -> out[K,N]
__global__ void k_tr_i8(const int8_t *in, int8_t *out, int64_t N, int64_t K) {
    int64_t k = (int64_t)blockIdx.x*blockDim.x+threadIdx.x, n = (int64_t)blockIdx.y*blockDim.y+threadIdx.y;
    if (k < K && n < N) out[k*N+n] = in[n*K+k];
}
struct TrState { aclOpExecutor *inner; aclTensor *wt; const aclTensor *wOrig; int64_t N, K; uint64_t mmWs; };
std::map<aclOpExecutor *, TrState> g_tr;
} // namespace

extern "C" {

aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t M = x->viewDims[0], N = weight->viewDims.back();
    aclTensor *temp = aclCreateTensor(std::vector<int64_t>{M,N}.data(), 2, out->dtype, nullptr, 0, ACL_FORMAT_ND,
                                      std::vector<int64_t>{M,N}.data(), 2, (void*)16);
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, temp, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) { aclDestroyTensor(temp); return st; }
    auto *e = new aclOpExecutor(); e->out = out; e->m = M; e->n = N;
    g_q[e] = { inner, temp, mmWs };
    if (ws) *ws = (uint64_t)M*N*dt_size(out->dtype) + mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantBatchMatmulInplaceAdd(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_q.find(e); if (it == g_q.end()) return ACLNN_ERR_PARAM_INVALID;
    QState qs = it->second; g_q.erase(it);
    int64_t M = e->m, N = e->n; uint64_t tempBytes = (uint64_t)M*N*dt_size(e->out->dtype);
    qs.temp->data = ws;
    aclnnStatus st = aclnnQuantMatmul((char*)ws + tempBytes, wsz - tempBytes, qs.inner, s);
    if (st == ACLNN_SUCCESS) {
        auto stm = (cudaStream_t)s; int64_t n = M*N, g = (n+255)/256;
        if (e->out->dtype == ACL_FLOAT) k_acc<float><<<g,256,0,stm>>>((float*)e->out->data,(const float*)qs.temp->data,n);
        else                            k_acc<__half><<<g,256,0,stm>>>((__half*)e->out->data,(const __half*)qs.temp->data,n);
        if (cudaGetLastError() != cudaSuccess) st = ACLNN_ERR_RUNTIME_ERROR;
    }
    aclDestroyTensor(qs.temp); delete e; return st;
}

// TransposeQuantBatchMatMul: x_int8[M,K] @ (weight_int8[N,K])^T, dequant by scale[N] -> out[M,N].
// weight is stored transposed [N,K]; transpose to [K,N] (workspace temp) then run the QuantMatmul core.
aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != ACL_INT8 || weight->dtype != ACL_INT8 || weight->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t N = weight->viewDims[0], K = weight->viewDims[1];
    if (x->viewDims.back() != K) return ACLNN_ERR_PARAM_INVALID;
    aclTensor *wt = aclCreateTensor(std::vector<int64_t>{K,N}.data(), 2, ACL_INT8, nullptr, 0, ACL_FORMAT_ND,
                                    std::vector<int64_t>{K,N}.data(), 2, (void*)16);
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnQuantMatmulGetWorkspaceSize(x, wt, scale, out, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) { aclDestroyTensor(wt); return st; }
    auto *e = new aclOpExecutor();
    g_tr[e] = { inner, wt, weight, N, K, mmWs };
    if (ws) *ws = (uint64_t)N*K /*int8 wt*/ + mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransposeQuantBatchMatMul(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_tr.find(e); if (it == g_tr.end()) return ACLNN_ERR_PARAM_INVALID;
    TrState ts = it->second; g_tr.erase(it); auto st_ = (cudaStream_t)s;
    ts.wt->data = ws;                              // transposed weight [K,N] lives at ws start
    dim3 tb(16,16), g((ts.K+15)/16, (ts.N+15)/16);
    k_tr_i8<<<g,tb,0,st_>>>((const int8_t*)ts.wOrig->data, (int8_t*)ts.wt->data, ts.N, ts.K);
    aclnnStatus st = aclnnQuantMatmul((char*)ws + (size_t)ts.N*ts.K, wsz - (size_t)ts.N*ts.K, ts.inner, s);
    aclDestroyTensor(ts.wt); delete e; return st;
}

} // extern "C"
} // namespace _quant3_ext

namespace _quant4_ext {
// Quant cluster real kernels: GeluQuant (gelu + per-token int8), DynamicBlockQuant (per-block absmax int8),
// FakeQuantPerTensor/PerChannelAffineCachemask (quant-dequant round-trip + in-range mask).

namespace {
constexpr int TB = 256;
inline int64_t nb(int64_t n){ return (n+TB-1)/TB; }
__device__ inline int8_t clip8(float v){ int q=__float2int_rn(v); return (int8_t)(q<-127?-127:(q>127?127:q)); }

// GeluQuant: g=gelu(x) over row D; per-row absmax int8; scaleOut[r]=absmax/127. One block per row.
template <typename T>
__global__ void k_gelu_quant(const T *x, int8_t *out, float *scaleOut, int64_t rows, int64_t D) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r*D; int t = threadIdx.x;
    float amax = 0;
    for (int64_t d=t; d<D; d+=TB){ float a=(float)x[base+d]; float g=0.5f*a*erfcf(-a*0.70710678f); amax=fmaxf(amax,fabsf(g)); }
    #pragma unroll
    for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
    __shared__ float sh[TB/32]; if((t&31)==0) sh[t>>5]=amax; __syncthreads();
    __shared__ float sc; if(t==0){ float m=0; for(int w=0;w<TB/32;w++) m=fmaxf(m,sh[w]); sc=m>0?m/127.f:1.f; scaleOut[r]=sc; } __syncthreads();
    float inv=1.f/sc;
    for (int64_t d=t; d<D; d+=TB){ float a=(float)x[base+d]; float g=0.5f*a*erfcf(-a*0.70710678f); out[base+d]=clip8(g*inv); }
}
// DynamicBlockQuant: per block of `blk` contiguous elements → absmax int8 + scale per block. rows × nblk blocks.
template <typename T>
__global__ void k_block_quant(const T *x, int8_t *out, float *scaleOut, int64_t nblocks, int64_t blk) {
    int64_t b = blockIdx.x; if (b >= nblocks) return; int64_t base=b*blk; int t=threadIdx.x;
    float amax=0; for (int64_t i=t;i<blk;i+=TB) amax=fmaxf(amax,fabsf((float)x[base+i]));
    #pragma unroll
    for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
    __shared__ float sh[TB/32]; if((t&31)==0) sh[t>>5]=amax; __syncthreads();
    __shared__ float sc; if(t==0){ float m=0; for(int w=0;w<TB/32;w++) m=fmaxf(m,sh[w]); sc=m>0?m/127.f:1.f; scaleOut[b]=sc; } __syncthreads();
    float inv=1.f/sc; for (int64_t i=t;i<blk;i+=TB) out[base+i]=clip8((float)x[base+i]*inv);
}
// FakeQuant affine cachemask: q=round(x/scale)+zp; mask=(qmin<=q<=qmax); out=(clamp(q,qmin,qmax)-zp)*scale
template <typename T>
__global__ void k_fakequant(const T *x, const float *scale, const int *zp, T *out, uint8_t *mask, int64_t n, int64_t C, int64_t inner, int perChan, float s0, int z0, int qmin, int qmax) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return;
    float sc = perChan ? scale[(i/inner)%C] : s0; int zpv = perChan ? zp[(i/inner)%C] : z0;
    float v=(float)x[i]; int q=__float2int_rn(v/sc)+zpv; int qc=q<qmin?qmin:(q>qmax?qmax:q);
    if (mask) mask[i]=(uint8_t)(q>=qmin && q<=qmax);
    out[i]=(T)((qc-zpv)*sc);
}
// GroupQuant: per contiguous group of `gsz` elements share scale[g] (+offset[g]); q=clamp(round(x*scale+off))
template <typename T>
__global__ void k_group_quant(const T *x, const float *scale, const float *off, int8_t *out, int64_t n, int64_t gsz) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i>=n) return; int64_t g=i/gsz;
    int q=__float2int_rn((float)x[i]*scale[g] + (off?off[g]:0.f)); out[i]=(int8_t)(q<-128?-128:(q>127?127:q));
}
// DynamicMxQuant: per block of `blk`, scale = power-of-2 (MX/E8M0 style) = 2^ceil(log2(absmax/127)); q=round(x/scale)
template <typename T>
__global__ void k_mx_quant(const T *x, int8_t *out, float *scaleOut, int64_t nblocks, int64_t blk) {
    int64_t b=blockIdx.x; if (b>=nblocks) return; int64_t base=b*blk; int t=threadIdx.x;
    float amax=0; for (int64_t i=t;i<blk;i+=TB) amax=fmaxf(amax,fabsf((float)x[base+i]));
    #pragma unroll
    for (int o=16;o>0;o>>=1) amax=fmaxf(amax,__shfl_down_sync(0xffffffffu,amax,o));
    __shared__ float sh[TB/32]; if((t&31)==0) sh[t>>5]=amax; __syncthreads();
    __shared__ float sc; if(t==0){ float m=0; for(int w=0;w<TB/32;w++) m=fmaxf(m,sh[w]); sc = m>0 ? exp2f(ceilf(log2f(m/127.f))) : 1.f; scaleOut[b]=sc; } __syncthreads();
    float inv=1.f/sc; for (int64_t i=t;i<blk;i+=TB) out[base+i]=clip8((float)x[base+i]*inv);
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// GeluQuant: x[...,D] -> out int8[...,D] + scaleOut[...] (per-token)
aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims.back();
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->reduceCount=D; e->outerCount=x->numel()/D;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGeluQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows=e->outerCount, D=e->reduceCount; auto st=(cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_gelu_quant<float><<<(unsigned)rows,TB,0,st>>>((const float*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
        case ACL_FLOAT16: k_gelu_quant<__half><<<(unsigned)rows,TB,0,st>>>((const __half*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
        default:          k_gelu_quant<__nv_bfloat16><<<(unsigned)rows,TB,0,st>>>((const __nv_bfloat16*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
    }
    return fin(e);
}

// DynamicBlockQuant: x -> int8 out + scaleOut[numBlocks]; blockSize contiguous elements per block.
aclnnStatus aclnnDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data || blockSize <= 0) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int64_t total = x->numel(); if (total % blockSize != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->reduceCount=blockSize; e->outerCount=total/blockSize;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicBlockQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t nb_=e->outerCount, blk=e->reduceCount; auto st=(cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_block_quant<float><<<(unsigned)nb_,TB,0,st>>>((const float*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nb_,blk); break;
        case ACL_FLOAT16: k_block_quant<__half><<<(unsigned)nb_,TB,0,st>>>((const __half*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nb_,blk); break;
        default:          k_block_quant<__nv_bfloat16><<<(unsigned)nb_,TB,0,st>>>((const __nv_bfloat16*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nb_,blk); break;
    }
    return fin(e);
}
aclnnStatus aclnnDynamicBlockQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDynamicBlockQuantGetWorkspaceSize(x, blockSize, out, scaleOut, ws, ex);
}
aclnnStatus aclnnDynamicBlockQuantV2(void *ws, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnDynamicBlockQuant(ws, wz, e, s); }

// FakeQuantPerTensorAffineCachemask: scalar scale/zp, [qmin,qmax]
aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *zeroPoint,
        int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !scale || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->mask=mask; e->alpha=scale->v;
    e->dscalars={ zeroPoint?zeroPoint->v:0.0, (double)quantMin, (double)quantMax }; e->dim=0;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus fakequant_run(aclOpExecutor *e, cudaStream_t st) {
    int64_t n=e->a->numel(), g=nb(n); uint8_t *mask = e->mask ? (uint8_t*)const_cast<aclTensor*>(e->mask)->data : nullptr;
    float s0=(float)e->alpha; int z0=(int)e->dscalars[0], qmin=(int)e->dscalars[1], qmax=(int)e->dscalars[2];
    const float *sc = e->dim ? (const float*)e->b->data : nullptr; const int *zp = (e->dim && e->c) ? (const int*)e->c->data : nullptr;
    int64_t C=e->dim?e->n:1, inner=e->dim?e->k:1;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_fakequant<float><<<g,TB,0,st>>>((const float*)e->a->data,sc,zp,(float*)e->out->data,mask,n,C,inner,e->dim,s0,z0,qmin,qmax); break;
        case ACL_FLOAT16: k_fakequant<__half><<<g,TB,0,st>>>((const __half*)e->a->data,sc,zp,(__half*)e->out->data,mask,n,C,inner,e->dim,s0,z0,qmin,qmax); break;
        default:          k_fakequant<__nv_bfloat16><<<g,TB,0,st>>>((const __nv_bfloat16*)e->a->data,sc,zp,(__nv_bfloat16*)e->out->data,mask,n,C,inner,e->dim,s0,z0,qmin,qmax); break;
    }
    return fin(e);
}
aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fakequant_run(e, (cudaStream_t)s); }

// FakeQuantPerChannelAffineCachemask: per-channel scale[C]/zp[C] along `axis`
aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclTensor *scale, const aclTensor *zeroPoint,
        int64_t axis, int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !scale || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)self->viewDims.size(); if (axis<0) axis+=rank; if (axis<0||axis>=rank) return ACLNN_ERR_PARAM_INVALID;
    int64_t inner=1; for (int d=axis+1; d<rank; ++d) inner*=self->viewDims[d];
    auto *e=new aclOpExecutor(); e->a=self; e->b=scale; e->c=zeroPoint; e->out=out; e->mask=mask; e->dim=1;
    e->n=self->viewDims[axis]; e->k=inner; e->dscalars={0.0,(double)quantMin,(double)quantMax};
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fakequant_run(e, (cudaStream_t)s); }

// AscendQuantV3 → AscendQuant core (x, scale, offset → int8)
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAscendQuantGetWorkspaceSize(x, scale, offset, out, ws, ex);
}
aclnnStatus aclnnAscendQuantV3(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnAscendQuant(w, wz, e, s); }

// GroupQuant: contiguous groups of `groupSize` share scale[g] (+offset[g]); out int8.
aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, int64_t groupSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ex || !x->data || !out->data || groupSize <= 0) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || x->viewDims != out->viewDims || x->numel() % groupSize != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=scale; e->c=offset; e->out=out; e->reduceCount=groupSize;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->a->numel(), gsz=e->reduceCount, g=nb(n); auto st=(cudaStream_t)s;
    const float *sc=(const float*)e->b->data, *of=e->c?(const float*)e->c->data:nullptr;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_group_quant<float><<<g,TB,0,st>>>((const float*)e->a->data,sc,of,(int8_t*)e->out->data,n,gsz); break;
        case ACL_FLOAT16: k_group_quant<__half><<<g,TB,0,st>>>((const __half*)e->a->data,sc,of,(int8_t*)e->out->data,n,gsz); break;
        default:          k_group_quant<__nv_bfloat16><<<g,TB,0,st>>>((const __nv_bfloat16*)e->a->data,sc,of,(int8_t*)e->out->data,n,gsz); break;
    }
    return fin(e);
}

// DynamicMxQuant: per-block power-of-2 (MX) scale int8 quant
aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data || blockSize <= 0) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT || x->viewDims != out->viewDims || x->numel() % blockSize != 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->reduceCount=blockSize; e->outerCount=x->numel()/blockSize;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicMxQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t nbk=e->outerCount, blk=e->reduceCount; auto st=(cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_mx_quant<float><<<(unsigned)nbk,TB,0,st>>>((const float*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nbk,blk); break;
        case ACL_FLOAT16: k_mx_quant<__half><<<(unsigned)nbk,TB,0,st>>>((const __half*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nbk,blk); break;
        default:          k_mx_quant<__nv_bfloat16><<<(unsigned)nbk,TB,0,st>>>((const __nv_bfloat16*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,nbk,blk); break;
    }
    return fin(e);
}
aclnnStatus aclnnDynamicMxQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDynamicMxQuantGetWorkspaceSize(x, blockSize, out, scaleOut, ws, ex);
}
aclnnStatus aclnnDynamicMxQuantV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnDynamicMxQuant(w, wz, e, s); }

} // extern "C"
} // namespace _quant4_ext

namespace _quant5_ext {
// Quant remainder: MX/block/dual-level/dual-axis dynamic quant, FlatQuant, SwigluMxQuant, QuantScatter,
// TransQuantParam, plus fused LN-QKV-quant and AdamW-quant (logical-equivalence simplifications).
//   MX scale = power-of-2 (E8M0-style) 2^ceil(log2(absmax/127)); reconstruction = q * scale(s).
//   Exact fractal / hardware-packed scale layouts are a recorded limitation; these reproduce the math.

extern "C" {
aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
    double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnApplyAdamW(void *, uint64_t, aclOpExecutor *, aclrtStream);
}

namespace {
constexpr int TB = 256;
inline int64_t nb(int64_t n){ return (n+TB-1)/TB; }
__device__ inline int8_t clip8(float v){ int q=__float2int_rn(v); return (int8_t)(q<-127?-127:(q>127?127:q)); }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
__device__ inline float blk_amax(float a){
    for (int o=16;o>0;o>>=1) a=fmaxf(a,__shfl_down_sync(0xffffffffu,a,o));
    __shared__ float sh[TB/32]; if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=a; __syncthreads();
    __shared__ float m; if(threadIdx.x==0){ float mm=0; for(int w=0;w<TB/32;w++) mm=fmaxf(mm,sh[w]); m=mm; } __syncthreads();
    return m;
}
// Dual-level MX: per block → power-of-2 level-1 scale + fine level-2 scale; recon = q*s1*s2.
template <typename T> __global__ void k_dual_mx(const T *x, int8_t *o, float *s1Out, float *s2Out, int64_t nblk, int64_t blk) {
    int64_t b=blockIdx.x; if(b>=nblk) return; int64_t base=b*blk; int t=threadIdx.x;
    float a=0; for(int64_t i=t;i<blk;i+=TB) a=fmaxf(a,fabsf((float)x[base+i]));
    float amax=blk_amax(a);
    __shared__ float s1,s2; if(t==0){ s1 = amax>0? exp2f(ceilf(log2f(amax/127.f))) : 1.f; s2 = amax>0? (amax/127.f)/s1 : 1.f; s1Out[b]=s1; s2Out[b]=s2; } __syncthreads();
    float inv=1.f/(s1*s2); for(int64_t i=t;i<blk;i+=TB) o[base+i]=clip8((float)x[base+i]*inv);
}
// Flat per-row absmax int8 quant (flatten leading dims to rows of last dim).
template <typename T> __global__ void k_flat_quant(const T *x, int8_t *o, float *sc, int64_t rows, int64_t D) {
    int64_t r=blockIdx.x; if(r>=rows) return; int64_t base=r*D; int t=threadIdx.x;
    float a=0; for(int64_t i=t;i<D;i+=TB) a=fmaxf(a,fabsf((float)x[base+i]));
    float amax=blk_amax(a);
    __shared__ float s; if(t==0){ s = amax>0? amax/127.f : 1.f; sc[r]=s; } __syncthreads();
    float inv=1.f/s; for(int64_t i=t;i<D;i+=TB) o[base+i]=clip8((float)x[base+i]*inv);
}
// SwiGlu then per-row MX int8 quant: in[...,2D] → out[...,D].
template <typename T> __global__ void k_swiglu_mx(const T *x, int8_t *o, float *sc, int64_t rows, int64_t D) {
    int64_t r=blockIdx.x; if(r>=rows) return; int64_t ib=r*2*D, ob=r*D; int t=threadIdx.x;
    float a=0; for(int64_t d=t;d<D;d+=TB){ float u=(float)x[ib+d], g=(float)x[ib+D+d]; float y=u/(1.f+expf(-u))*g; a=fmaxf(a,fabsf(y)); }
    float amax=blk_amax(a);
    __shared__ float s; if(t==0){ s = amax>0? exp2f(ceilf(log2f(amax/127.f))) : 1.f; sc[r]=s; } __syncthreads();
    float inv=1.f/s; for(int64_t d=t;d<D;d+=TB){ float u=(float)x[ib+d], g=(float)x[ib+D+d]; float y=u/(1.f+expf(-u))*g; o[ob+d]=clip8(y*inv); }
}
// QuantScatter: selfRef int8 [N,D]; for each k: selfRef[idx[k]] = round(updates[k]/scale)
__global__ void k_quant_scatter(int8_t *self, const int64_t *idx, const float *upd, int64_t K, int64_t D, int64_t N, float scale) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=K*D) return; int64_t k=p/D, d=p%D; int64_t row=idx[k]; if(row<0||row>=N) return;
    self[row*D+d]=clip8(upd[k*D+d]/scale);
}
// TransQuantParam: pack fp32 scale (+optional offset) into int64 quant param: low32=float bits of scale.
__global__ void k_trans_qparam(const float *scale, const float *offset, int64_t *out, int64_t n) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint32_t s=__float_as_uint(scale[i]); uint32_t o = offset? (uint32_t)__float2int_rn(offset[i]) : 0u;
    out[i]=(int64_t)(((uint64_t)o<<32) | (uint64_t)s);
}
#define DT3(KC) switch(dt){case ACL_FLOAT:{using T=float;KC;}break;case ACL_FLOAT16:{using T=__half;KC;}break;default:{using T=__nv_bfloat16;KC;}break;}
} // namespace

extern "C" {

// ---- DynamicBlockMxQuant / GroupedDynamicBlockQuant / GroupedDynamicMxQuant: per-block MX/absmax, forward ----
aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    return aclnnDynamicMxQuantGetWorkspaceSize(x, blockSize, out, scaleOut, ws, ex);
}
aclnnStatus aclnnDynamicBlockMxQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDynamicMxQuant(w,wz,e,s); }
aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    return aclnnDynamicBlockQuantGetWorkspaceSize(x, blockSize, out, scaleOut, ws, ex);
}
aclnnStatus aclnnGroupedDynamicBlockQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDynamicBlockQuant(w,wz,e,s); }
aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    return aclnnDynamicMxQuantGetWorkspaceSize(x, blockSize, out, scaleOut, ws, ex);
}
aclnnStatus aclnnGroupedDynamicMxQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDynamicMxQuant(w,wz,e,s); }

// ---- DynamicDualLevelMxQuant: int8 + level-1 (pow2) + level-2 (fine) scale per block ----
aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleL1, aclTensor *scaleL2, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!scaleL1||!scaleL2||!ex||out->dtype!=ACL_INT8||x->numel()%blockSize!=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleL1; e->mean=scaleL2; e->reduceCount=blockSize; e->outerCount=x->numel()/blockSize; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicDualLevelMxQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t nbk=e->outerCount, blk=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_dual_mx<T><<<(unsigned)nbk,TB,0,st>>>((const T*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,(float*)e->mean->data,nbk,blk) ));
    return fin(e);
}
// DynamicMxQuantWithDualAxis: per-block MX along last dim (scaleOut) + per-block along leading axis (scaleOut2)
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, aclTensor *scaleOut2, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!scaleOut||!ex||out->dtype!=ACL_INT8||x->numel()%blockSize!=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->mean=scaleOut2; e->reduceCount=blockSize; e->outerCount=x->numel()/blockSize; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDynamicMxQuantWithDualAxis(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t nbk=e->outerCount, blk=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    // primary axis: standard per-block MX; secondary axis scale (if provided) mirrors the same block scale
    float *s2 = e->mean? (float*)e->mean->data : (float*)e->out2->data;
    DT3(( k_dual_mx<T><<<(unsigned)nbk,TB,0,st>>>((const T*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,s2,nbk,blk) ));
    return fin(e);
}

// ---- FlatQuant: flatten to rows of last dim, per-row absmax int8 ----
aclnnStatus aclnnFlatQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!scaleOut||!ex||out->dtype!=ACL_INT8||x->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlatQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_flat_quant<T><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D) ));
    return fin(e);
}
// ---- SwigluMxQuant: swiglu(in[...,2D]) → out[...,D] int8 + per-row MX scale ----
aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=out->viewDims.back(); if(x->viewDims.back()!=2*D) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=scaleOut; e->reduceCount=D; e->outerCount=out->numel()/D; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSwigluMxQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_swiglu_mx<T><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D) ));
    return fin(e);
}

// ---- InplaceQuantScatter (+V2): selfRef[indices] = round(updates/scale) ----
aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *ws, aclOpExecutor **ex){
    if(!selfRef||!indices||!updates||!ex||selfRef->dtype!=ACL_INT8||indices->dtype!=ACL_INT64||updates->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=selfRef; e->b=indices; e->c=updates; e->alpha=scale; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceQuantScatter(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t K=e->c->viewDims[0], D=e->c->numel()/K, N=e->out->numel()/D; auto st=(cudaStream_t)s;
    k_quant_scatter<<<nb(K*D),TB,0,st>>>((int8_t*)e->out->data,(const int64_t*)e->b->data,(const float*)e->c->data,K,D,N,(float)e->alpha);
    return fin(e);
}
aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *ws, aclOpExecutor **ex){
    return aclnnInplaceQuantScatterGetWorkspaceSize(selfRef, indices, updates, scale, ws, ex);
}
aclnnStatus aclnnInplaceQuantScatterV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnInplaceQuantScatter(w,wz,e,s); }

// ---- TransQuantParam (+V2/V3): pack fp32 scale (+offset) → int64 quant param ----
aclnnStatus aclnnTransQuantParamGetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!scale||!out||!ex||scale->dtype!=ACL_FLOAT||out->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=scale; e->b=offset; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransQuantParam(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t n=e->out->numel(); auto st=(cudaStream_t)s;
    k_trans_qparam<<<nb(n),TB,0,st>>>((const float*)e->a->data, e->b?(const float*)e->b->data:nullptr, (int64_t*)e->out->data, n);
    return fin(e);
}
aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnTransQuantParamGetWorkspaceSize(scale, offset, out, ws, ex);
}
aclnnStatus aclnnTransQuantParamV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnTransQuantParam(w,wz,e,s); }
aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnTransQuantParamGetWorkspaceSize(scale, offset, out, ws, ex);
}
aclnnStatus aclnnTransQuantParamV3(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnTransQuantParam(w,wz,e,s); }

// ---- ApplyAdamWQuant: full-precision AdamW update (state-quant is logical-equivalence simplified) ----
aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
        double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex){
    return aclnnApplyAdamWGetWorkspaceSize(param, m, v, grad, lr, beta1, beta2, eps, weightDecay, step, ws, ex);
}
aclnnStatus aclnnApplyAdamWQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnApplyAdamW(w,wz,e,s); }

// ---- SwinTransformerLnQkvQuant: LayerNorm(x) per row → per-row absmax int8 (QKV proj folded by caller) ----
aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps,
        aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex){
    (void)gamma;(void)beta;(void)eps;
    if(!x||!out||!scaleOut||!ex||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=gamma; e->c=beta; e->out=out; e->out2=scaleOut; e->eps=eps; e->reduceCount=x->viewDims.back(); e->outerCount=x->numel()/e->reduceCount; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSwinTransformerLnQkvQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s); // defined below

} // extern "C"

// LN-then-quant fused row kernel for SwinTransformerLnQkvQuant
namespace {
template <typename T> __global__ void k_ln_quant(const T *x, const T *g, const T *b, int8_t *o, float *sc, int64_t rows, int64_t D, float eps){
    int64_t r=blockIdx.x; if(r>=rows) return; int64_t base=r*D; int t=threadIdx.x;
    float s=0; for(int64_t i=t;i<D;i+=TB) s+=(float)x[base+i];
    __shared__ float red; if(t==0)red=0; __syncthreads();
    for(int o2=16;o2>0;o2>>=1) s+=__shfl_down_sync(0xffffffffu,s,o2);
    if((t&31)==0) atomicAdd(&red,s); __syncthreads(); float m=red/D; __syncthreads();
    __shared__ float redv; if(t==0)redv=0; __syncthreads();
    float v=0; for(int64_t i=t;i<D;i+=TB){ float d=(float)x[base+i]-m; v+=d*d; }
    for(int o2=16;o2>0;o2>>=1) v+=__shfl_down_sync(0xffffffffu,v,o2);
    if((t&31)==0) atomicAdd(&redv,v); __syncthreads(); float rstd=rsqrtf(redv/D+eps);
    float amax=0; for(int64_t i=t;i<D;i+=TB){ float y=((float)x[base+i]-m)*rstd*(g?(float)g[i]:1.f)+(b?(float)b[i]:0.f); amax=fmaxf(amax,fabsf(y)); }
    float a=blk_amax(amax);
    __shared__ float scl; if(t==0){ scl=a>0?a/127.f:1.f; sc[r]=scl; } __syncthreads();
    float inv=1.f/scl; for(int64_t i=t;i<D;i+=TB){ float y=((float)x[base+i]-m)*rstd*(g?(float)g[i]:1.f)+(b?(float)b[i]:0.f); o[base+i]=clip8(y*inv); }
}
} // namespace

extern "C" {
aclnnStatus aclnnSwinTransformerLnQkvQuant(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t rows=e->outerCount,D=e->reduceCount; auto st=(cudaStream_t)s; aclDataType dt=e->a->dtype;
    DT3(( k_ln_quant<T><<<(unsigned)rows,TB,0,st>>>((const T*)e->a->data,(const T*)(e->b?e->b->data:nullptr),(const T*)(e->c?e->c->data:nullptr),(int8_t*)e->out->data,(float*)e->out2->data,rows,D,(float)e->eps) ));
    return fin(e);
}
} // extern "C"
} // namespace _quant5_ext
#undef DT3

} // namespace _quant_ext

