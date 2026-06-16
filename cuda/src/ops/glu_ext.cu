// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace _glu_ext {
// GLU gradients (SwiGluGrad, GeGluBackward) + GeGluV3 forward. Layout: in[...,2D] split at last dim
// into act-half a=in[:D] and gate-half b=in[D:2D]; out = act(a)·b. fp32/fp16/bf16, fp32 math.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// gradIn[a] = gradOut·act'(a)·b ; gradIn[b] = gradOut·act(a). gelu=1 → GELU, else SiLU.
template <typename T>
__global__ void k_glu_grad(const T *gOut, const T *in, T *gIn, int64_t rows, int64_t D, int gelu) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * D) return;
    int64_t r = i / D, d = i % D; float a = (float)in[r*2*D + d], b = (float)in[r*2*D + D + d], go = (float)gOut[i];
    float act, dact;
    if (gelu) { float c = 0.70710678118654752f; float cdf = 0.5f*(1.f+erff(a*c));
                act = a*cdf; dact = cdf + a*0.39894228040143270f*expf(-0.5f*a*a); }   // 1/sqrt(2π)=0.3989...
    else      { float s = 1.f/(1.f+expf(-a)); act = a*s; dact = s + a*s*(1.f-s); }
    gIn[r*2*D + d]     = (T)(go * dact * b);
    gIn[r*2*D + D + d] = (T)(go * act);
}

inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

static aclnnStatus glugrad_ws(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, int gelu, aclOpExecutor **ex) {
    if (!gradOut || !self || !gradIn || !ex || !gradOut->data || !self->data || !gradIn->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != gradOut->dtype || self->dtype != gradIn->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_FLOAT && self->dtype != ACL_FLOAT16 && self->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = gradOut->viewDims.back();
    if (self->viewDims.back() != 2*D || self->viewDims != gradIn->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->out = gradIn;
    e->reduceCount = D; e->outerCount = gradOut->numel() / D; e->dim = gelu;
    *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus glugrad_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t rows = e->outerCount, D = e->reduceCount, g = nb(rows*D);
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_glu_grad<float><<<g,TH,0,s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,rows,D,(int)e->dim); break;
        case ACL_FLOAT16: k_glu_grad<__half><<<g,TH,0,s>>>((const __half*)e->a->data,(const __half*)e->b->data,(__half*)e->out->data,rows,D,(int)e->dim); break;
        default:          k_glu_grad<__nv_bfloat16><<<g,TH,0,s>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(__nv_bfloat16*)e->out->data,rows,D,(int)e->dim); break;
    }
    return fin(e);
}
} // namespace

extern "C" {

// SwiGluGrad: gradOut[...,D], self[...,2D] -> gradIn[...,2D]
aclnnStatus aclnnSwiGluGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return glugrad_ws(gradOut, self, gradIn, 0, ex);
}
aclnnStatus aclnnSwiGluGrad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return glugrad_run(e, (cudaStream_t)s); }

// GeGluBackward / GeGluV3Backward: GELU-gated GLU gradient
aclnnStatus aclnnGeGluBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return glugrad_ws(gradOut, self, gradIn, 1, ex);
}
aclnnStatus aclnnGeGluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return glugrad_run(e, (cudaStream_t)s); }
aclnnStatus aclnnGeGluV3BackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return glugrad_ws(gradOut, self, gradIn, 1, ex);
}
aclnnStatus aclnnGeGluV3Backward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return glugrad_run(e, (cudaStream_t)s); }

// GeGluV3 forward → GeGlu core (gelu-gated GLU).
aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGeGluGetWorkspaceSize(self, out, ws, ex);
}
aclnnStatus aclnnGeGluV3(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnGeGlu(ws, wsz, e, s); }

} // extern "C"
} // namespace _glu_ext

namespace _glu2_ext {
// SwiGlu fused with per-token int8 quant (SwiGluQuant) and clipped-gate SwiGlu (ClippedSwiglu).
// Layout: in[...,2D] split at last dim into act-half a=in[:D] and gate-half b=in[D:2D]; g = silu(a)·b.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// SwiGluQuant: g[d]=silu(a)·b over D; per-row absmax → int8 quant; scaleOut[r]=absmax/127. One block per row.
template <typename T, int TB>
__global__ void k_swiglu_quant(const T *in, int8_t *out, float *scaleOut, int64_t rows, int64_t D) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t ib = r * 2 * D, ob = r * D; int t = threadIdx.x;
    float amax = 0;
    for (int64_t d = t; d < D; d += TB) { float a = (float)in[ib+d], b = (float)in[ib+D+d]; float g = a/(1.f+expf(-a))*b; amax = fmaxf(amax, fabsf(g)); }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    __shared__ float sh[TB/32]; if ((t&31)==0) sh[t>>5]=amax; __syncthreads();
    __shared__ float sc; if (t==0){ float m=0; for(int w=0;w<TB/32;w++) m=fmaxf(m,sh[w]); sc = m>0?m/127.f:1.f; scaleOut[r]=sc; } __syncthreads();
    float inv = 1.f / sc;
    for (int64_t d = t; d < D; d += TB) { float a = (float)in[ib+d], b = (float)in[ib+D+d]; float g = a/(1.f+expf(-a))*b;
        int q = __float2int_rn(g*inv); q = q<-127?-127:(q>127?127:q); out[ob+d] = (int8_t)q; }
}

// ClippedSwiglu: g[d] = silu(clamp(a,-c,c))·b
template <typename T>
__global__ void k_clipped_swiglu(const T *in, T *out, int64_t rows, int64_t D, float clip) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= rows * D) return;
    int64_t r = i / D, d = i % D; float a = (float)in[r*2*D+d], b = (float)in[r*2*D+D+d];
    a = a < -clip ? -clip : (a > clip ? clip : a);
    out[i] = (T)(a/(1.f+expf(-a))*b);
}

inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// SwiGluQuant: x[...,2D] -> out[...,D] (int8) + scaleOut[...] (fp32 per-token).
aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ex || !x->data || !out->data || !scaleOut->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_INT8 || scaleOut->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = out->viewDims.back();
    if (x->viewDims.back() != 2*D) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->out2 = scaleOut; e->reduceCount = D; e->outerCount = out->numel()/D;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSwiGluQuant(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, D = e->reduceCount; auto st = (cudaStream_t)s;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_swiglu_quant<float,256><<<(unsigned)rows,256,0,st>>>((const float*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
        case ACL_FLOAT16: k_swiglu_quant<__half,256><<<(unsigned)rows,256,0,st>>>((const __half*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
        default:          k_swiglu_quant<__nv_bfloat16,256><<<(unsigned)rows,256,0,st>>>((const __nv_bfloat16*)e->a->data,(int8_t*)e->out->data,(float*)e->out2->data,rows,D); break;
    }
    return fin(e);
}
aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnSwiGluQuantGetWorkspaceSize(x, out, scaleOut, ws, ex);
}
aclnnStatus aclnnSwiGluQuantV2(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnSwiGluQuant(ws, wsz, e, s); }

// ClippedSwiglu: x[...,2D], clipValue -> out[...,D] = silu(clamp(a,-clip,clip))·b
aclnnStatus aclnnClippedSwigluGetWorkspaceSize(const aclTensor *x, double clipValue, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex || !x->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = out->viewDims.back();
    if (x->viewDims.back() != 2*D) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->reduceCount = D; e->outerCount = out->numel()/D; e->eps = clipValue;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnClippedSwiglu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, D = e->reduceCount, g = nb(rows*D); auto st = (cudaStream_t)s; float clip = (float)e->eps;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_clipped_swiglu<float><<<g,TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,rows,D,clip); break;
        case ACL_FLOAT16: k_clipped_swiglu<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(__half*)e->out->data,rows,D,clip); break;
        default:          k_clipped_swiglu<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(__nv_bfloat16*)e->out->data,rows,D,clip); break;
    }
    return fin(e);
}

} // extern "C"
} // namespace _glu2_ext

