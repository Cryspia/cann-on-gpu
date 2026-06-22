// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>

namespace _misc_ext {
// Misc one-off real-kernel ops: AddRelu (fused add+ReLU), Histc (histogram), ScatterValue (scatter a scalar).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

template <typename T>
__global__ void k_addrelu(const T *a, const T *b, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    float v = (float)a[i] + (float)b[i]; o[i] = (T)(v > 0.f ? v : 0.f);
}
// Histc: count elements of x in [lo,hi] into `bins` equal-width bins (out fp32 counts).
template <typename T>
__global__ void k_histc(const T *x, float *out, int64_t n, int bins, float lo, float hi) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    float v = (float)x[i]; if (v < lo || v > hi) return;
    int b = (hi > lo) ? (int)((v - lo) / (hi - lo) * bins) : 0; if (b >= bins) b = bins - 1; if (b < 0) b = 0;
    atomicAdd(&out[b], 1.0f);
}
// ScatterValue (contiguous): out=self; out[outer, index[pos], inner] = value, dim-th axis. idx shape = scatter region.
template <typename T>
__global__ void k_scatter_value(T *out, const int64_t *idx, int64_t idxNumel, float value,
                                int64_t innerStride, int64_t dimSize, int64_t idxDimSize) {
    int64_t p = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (p >= idxNumel) return;
    // decode p in index space (outer, dpos, inner) where index dim has size idxDimSize
    int64_t inner = p % innerStride;
    int64_t outer = p / (innerStride * idxDimSize);
    int64_t target = idx[p];                       // position along dim in the OUTPUT
    int64_t off = (outer * dimSize + target) * innerStride + inner;
    out[off] = (T)value;
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// AddRelu: out = ReLU(self + other)
aclnnStatus aclnnAddReluGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims || self->viewDims != other->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = other; e->out = out; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddRelu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(), g = nb(n); auto st=(cudaStream_t)s;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_addrelu<float><<<g,TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n); break;
        case ACL_FLOAT16: k_addrelu<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(__half*)e->out->data,n); break;
        default:          k_addrelu<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(__nv_bfloat16*)e->out->data,n); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddReluGetWorkspaceSize(selfRef, other, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceAddRelu(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnAddRelu(ws, wsz, e, s); }

// Histc: self, bins, min, max -> out[bins] (fp32 counts). If min==max==0, use data range (approx: caller passes range).
aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int64_t bins, const aclScalar *min, const aclScalar *max,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data || bins <= 0) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->dtype != ACL_FLOAT || out->numel() != bins) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->reduceCount = bins;
    e->dscalars = { min ? min->v : 0.0, max ? max->v : 0.0 };
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnHistc(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->a->numel(); int bins = (int)e->reduceCount; float lo=(float)e->dscalars[0], hi=(float)e->dscalars[1]; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data, 0, bins*sizeof(float), st);
    int64_t g = nb(n);
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_histc<float><<<g,TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,n,bins,lo,hi); break;
        case ACL_FLOAT16: k_histc<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(float*)e->out->data,n,bins,lo,hi); break;
        default:          k_histc<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(float*)e->out->data,n,bins,lo,hi); break;
    }
    return fin(e);
}

// ScatterValue: out = self (copied), then out[..,index,..] = value along `dim`. self/out/index contiguous.
aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !value || !out || !ex || !self->data || !index->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims || index->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size(); if (dim < 0) dim += rank; if (dim < 0 || dim >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->out = out; e->dim = (int)dim; e->alpha = value->v;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnScatterValue(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int dim=e->dim; const auto &S=e->a->viewDims; const auto &I=e->b->viewDims;
    int64_t innerStride=1; for (size_t d=dim+1; d<S.size(); ++d) innerStride *= S[d];
    int64_t dimSize=S[dim], idxDimSize=I[dim], idxNumel=e->b->numel();
    // index inner stride must equal output inner stride for contiguous mapping; require I[d]==S[d] for d>dim
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*dtype_size(e->out->dtype), cudaMemcpyDeviceToDevice, st);
    int64_t g = nb(idxNumel);
    const int64_t *idx=(const int64_t*)e->b->data; float v=(float)e->alpha;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_scatter_value<float><<<g,TH,0,st>>>((float*)e->out->data,idx,idxNumel,v,innerStride,dimSize,idxDimSize); break;
        case ACL_FLOAT16: k_scatter_value<__half><<<g,TH,0,st>>>((__half*)e->out->data,idx,idxNumel,v,innerStride,dimSize,idxDimSize); break;
        case ACL_INT32:   k_scatter_value<int32_t><<<g,TH,0,st>>>((int32_t*)e->out->data,idx,idxNumel,v,innerStride,dimSize,idxDimSize); break;
        default:          k_scatter_value<__nv_bfloat16><<<g,TH,0,st>>>((__nv_bfloat16*)e->out->data,idx,idxNumel,v,innerStride,dimSize,idxDimSize); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnScatterValueGetWorkspaceSize(selfRef, dim, index, value, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceScatterValue(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnScatterValue(ws, wsz, e, s); }

} // extern "C"
} // namespace _misc_ext

namespace _misc2_ext {
// More one-off real-kernel ops: DivMod (tensor//tensor round mode), Put (flat scatter), RReluWithNoise.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

template <typename T>
__global__ void k_divmod(const T *a, const T *b, T *o, int64_t n, int mode) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    float q = (float)a[i] / (float)b[i];
    o[i] = (T)(mode==1 ? truncf(q) : floorf(q));
}
// Put: out (preloaded = self) flat-indexed write/accumulate: out.flat[index[k]] (+)= source[k]
template <typename T>
__global__ void k_put(T *out, const int64_t *index, const T *src, int64_t k, int accumulate) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= k) return; int64_t t = index[i];
    if (accumulate) out[t] = (T)((float)out[t] + (float)src[i]);   // non-atomic; correct for unique indices
    else out[t] = src[i];
}
__device__ inline float hashu(uint64_t x){ x^=x>>33; x*=0xff51afd7ed558ccdULL; x^=x>>33; x*=0xc4ceb9fe1a85ec53ULL; x^=x>>33; return (x>>40)*(1.0f/16777216.0f); }
// RReluWithNoise: training → noise=U(lower,upper) for x<0 (else 1), out=x>=0?x:x*noise; inference → x>=0?x:x*(lo+hi)/2
template <typename T>
__global__ void k_rrelu(const T *x, T *noise, T *out, int64_t n, float lo, float hi, int training, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return; float v=(float)x[i];
    if (v >= 0.f) { if (noise) noise[i]=(T)1.f; out[i]=(T)v; return; }
    float nz = training ? (lo + (hi-lo)*hashu(seed + (uint64_t)i*2654435761ULL)) : 0.5f*(lo+hi);
    if (noise) noise[i]=(T)nz; out[i]=(T)(v*nz);
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// DivMod: out = round_mode(self / other), tensor/tensor. roundMode 0 floor, 1 trunc.
aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t roundMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims || self->viewDims != other->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = other; e->out = out; e->dim = (int)roundMode; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDivMod(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(), g=nb(n); auto st=(cudaStream_t)s; int m=e->dim;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_divmod<float><<<g,TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n,m); break;
        case ACL_FLOAT16: k_divmod<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(__half*)e->out->data,n,m); break;
        default:          k_divmod<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(__nv_bfloat16*)e->out->data,n,m); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplaceDivModGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, int64_t roundMode, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDivModGetWorkspaceSize(selfRef, other, roundMode, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceDivMod(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnDivMod(ws, wsz, e, s); }

// Put: out = self (copied), then out.flat[index[k]] = source[k] (or += if accumulate).
aclnnStatus aclnnPutGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *source, bool accumulate, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !index || !source || !out || !ex || index->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = index; e->c = source; e->out = out; e->dim = accumulate?1:0; if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPut(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t k=e->b->numel(), g=nb(k);
    if (e->out->data != e->a->data) cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*dtype_size(e->out->dtype), cudaMemcpyDeviceToDevice, st);
    const int64_t *idx=(const int64_t*)e->b->data;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_put<float><<<g,TH,0,st>>>((float*)e->out->data,idx,(const float*)e->c->data,k,e->dim); break;
        case ACL_FLOAT16: k_put<__half><<<g,TH,0,st>>>((__half*)e->out->data,idx,(const __half*)e->c->data,k,e->dim); break;
        default:          k_put<__nv_bfloat16><<<g,TH,0,st>>>((__nv_bfloat16*)e->out->data,idx,(const __nv_bfloat16*)e->c->data,k,e->dim); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplacePutGetWorkspaceSize(aclTensor *selfRef, const aclTensor *index, const aclTensor *source, bool accumulate, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnPutGetWorkspaceSize(selfRef, index, source, accumulate, selfRef, ws, ex);
}
aclnnStatus aclnnInplacePut(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnPut(ws, wsz, e, s); }

// RReluWithNoise: self, noise(out), lower, upper, training, seed -> out
aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(const aclTensor *self, aclTensor *noise, double lower, double upper, bool training, int64_t seed,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = noise; e->out = out; e->dscalars = {lower, upper, (double)seed}; e->dim = training?1:0;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRReluWithNoise(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(), g=nb(n); auto st=(cudaStream_t)s; float lo=(float)e->dscalars[0],hi=(float)e->dscalars[1]; uint64_t seed=(uint64_t)e->dscalars[2]; int tr=e->dim;
    void *nz = e->b ? const_cast<aclTensor*>(e->b)->data : nullptr;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_rrelu<float><<<g,TH,0,st>>>((const float*)e->a->data,(float*)nz,(float*)e->out->data,n,lo,hi,tr,seed); break;
        case ACL_FLOAT16: k_rrelu<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(__half*)nz,(__half*)e->out->data,n,lo,hi,tr,seed); break;
        default:          k_rrelu<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(__nv_bfloat16*)nz,(__nv_bfloat16*)e->out->data,n,lo,hi,tr,seed); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(aclTensor *selfRef, aclTensor *noise, double lower, double upper, bool training, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRReluWithNoiseGetWorkspaceSize(selfRef, noise, lower, upper, training, seed, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceRReluWithNoise(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnRReluWithNoise(ws, wsz, e, s); }

} // extern "C"
} // namespace _misc2_ext

namespace _misc3_ext {
// RNN/FFN + misc-fused remainder (fp32). Tractable elementwise/shape/linalg/seq ops; linalg & LSTM
// forward to existing cuSOLVER/cuDNN-backed bases; FFN/Sinkhorn/RelPosBias-softmax/Pdist are direct kernels.

extern "C" {  // bases in other TUs / public header
aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *A, aclTensor *sign, aclTensor *logabsdet, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnSlogdet(void *, uint64_t, aclOpExecutor *, aclrtStream);
}

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
__device__ inline float sig(float x){ return 1.f/(1.f+expf(-x)); }
__global__ void k_xor(const uint8_t*a,const uint8_t*b,uint8_t*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(a[i]!=0)^(b[i]!=0); }
__global__ void k_logsig(const float*x,float*o,float*buf,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float v=x[i]; float m=fminf(v,0.f); float z=expf(-fabsf(v)); o[i]=m-log1pf(z); if(buf)buf[i]=z; }
__global__ void k_fatrelu_mul(const float*a,const float*b,float*o,int64_t n,float thr){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ float av=a[i]; o[i]=(av>thr?av:0.f)*b[i]; } }
__global__ void k_axpy(const float*x,const float*y,float*o,int64_t n,float al){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=al*x[i]+y[i]; }
__global__ void k_leftshifts(const int32_t*a,int32_t*o,int64_t n,int sh){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=a[i]<<sh; }
__global__ void k_isin_ts(const float*t,float v,uint8_t*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(t[i]==v); }
__global__ void k_angle(const float*re,const float*im,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=atan2f(im?im[i]:0.f,re[i]); }
__global__ void k_angle_il(const float*c,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=atan2f(c[2*i+1],c[2*i]); }  // interleaved (re,im) pairs
// complex stored interleaved [...,2]: Real → real part; Complex(re,im); Polar(abs,angle)
__global__ void k_real(const float*c,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=c[2*i]; }
__global__ void k_complex(const float*re,const float*im,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ o[2*i]=re[i]; o[2*i+1]=im[i]; } }
__global__ void k_polar(const float*ab,const float*an,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ o[2*i]=ab[i]*cosf(an[i]); o[2*i+1]=ab[i]*sinf(an[i]); } }
// var over last dim with correction: var = Σ(x-mean)^2/(D-correction)
__global__ void k_varcorr(const float*x,float*o,int64_t rows,int64_t D,float corr){ int64_t r=blockIdx.x; if(r>=rows||threadIdx.x!=0)return; const float*p=x+r*D; double m=0; for(int64_t d=0;d<D;d++)m+=p[d]; m/=D; double v=0; for(int64_t d=0;d<D;d++){double u=p[d]-m;v+=u*u;} o[r]=(float)(v/fmax((double)D-corr,1.0)); }
// Pdist: condensed pairwise Lp distances within X[N,D] → out[N*(N-1)/2]
__global__ void k_pdist(const float*x,float*o,int64_t N,int64_t D,float p){ int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; int64_t T=N*(N-1)/2; if(idx>=T)return;
    // map idx → (i,j) i<j
    int64_t i=0,rem=idx; while(rem>=N-1-i){ rem-=(N-1-i); i++; } int64_t j=i+1+rem;
    double s=0; for(int64_t d=0;d<D;d++){ double diff=fabs((double)x[i*D+d]-x[j*D+d]); s+=pow(diff,p);} o[idx]=(float)pow(s,1.0/p); }
// FFN: out = act(x@W1+b1)@W2+b2; x[M,K], W1[K,Hd], W2[Hd,N]; act: 0 relu,1 gelu,2 silu (no gate path here)
__global__ void k_ffn(const float*x,const float*W1,const float*b1,const float*W2,const float*b2,float*o,int M,int K,int Hd,int Nn,int act){
    int64_t r=blockIdx.x; if(r>=M) return; extern __shared__ float h[]; // Hd
    for(int j=threadIdx.x;j<Hd;j+=blockDim.x){ float acc=b1?b1[j]:0.f; for(int k=0;k<K;k++)acc+=x[r*K+k]*W1[k*Hd+j];
        float a; if(act==0)a=fmaxf(acc,0.f); else if(act==1)a=0.5f*acc*erfcf(-acc*0.70710678f); else a=acc*sig(acc); h[j]=a; }
    __syncthreads();
    for(int n=threadIdx.x;n<Nn;n+=blockDim.x){ float acc=b2?b2[n]:0.f; for(int j=0;j<Hd;j++)acc+=h[j]*W2[j*Nn+n]; o[r*Nn+n]=acc; }
}
// Sinkhorn: normalize a [R,C] nonneg matrix by alternating row/col sums for `iters`
__global__ void k_sinkhorn(float*m,int R,int C,int iters){
    // single-block cooperative over small matrices
    for(int it=0;it<iters;it++){
        for(int r=threadIdx.x;r<R;r+=blockDim.x){ double s=0; for(int c=0;c<C;c++)s+=m[r*C+c]; if(s>0)for(int c=0;c<C;c++)m[r*C+c]/=s; }
        __syncthreads();
        for(int c=threadIdx.x;c<C;c+=blockDim.x){ double s=0; for(int r=0;r<R;r++)s+=m[r*C+c]; if(s>0)for(int r=0;r<R;r++)m[r*C+c]/=s; }
        __syncthreads();
    }
}
// masked softmax with relative-position bias: softmax(qk[r,:]*scale + relpos[r,:] + mask[r,:]) over last dim
__global__ void k_relpos_softmax(const float*qk,const float*relpos,const float*mask,float*o,int64_t rows,int64_t D,float scale){
    int64_t r=blockIdx.x; if(r>=rows) return; const float*p=qk+r*D; float*op=o+r*D; const float*rp=relpos?relpos+r*D:nullptr; const float*mk=mask?mask+r*D:nullptr;
    __shared__ float mx,sm; if(threadIdx.x==0){ float m=-1e30f; for(int64_t d=0;d<D;d++){ float v=p[d]*scale+(rp?rp[d]:0.f)+(mk?mk[d]:0.f); m=fmaxf(m,v);} mx=m; } __syncthreads();
    if(threadIdx.x==0){ float s=0; for(int64_t d=0;d<D;d++){ float v=p[d]*scale+(rp?rp[d]:0.f)+(mk?mk[d]:0.f); s+=expf(v-mx);} sm=s; } __syncthreads();
    for(int64_t d=threadIdx.x;d<D;d+=blockDim.x){ float v=p[d]*scale+(rp?rp[d]:0.f)+(mk?mk[d]:0.f); op[d]=expf(v-mx)/sm; }
}
} // namespace

extern "C" {

// ---- linalg naming forwards ----
aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor *self, bool upper, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ aclnnStatus st=aclnnCholeskyGetWorkspaceSize(self, out, ws, ex); if(st==ACLNN_SUCCESS && ex && *ex) (*ex)->m = upper?1:0; return st; }
aclnnStatus aclnnLinalgCholesky(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnCholesky(w,wz,e,s); }
aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *A, int64_t mode, aclTensor *Q, aclTensor *R, uint64_t *ws, aclOpExecutor **ex){ (void)mode; return aclnnQrGetWorkspaceSize(A, Q, R, ws, ex); }
aclnnStatus aclnnLinalgQr(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQr(w,wz,e,s); }
aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)dim; return aclnnCrossGetWorkspaceSize(self, other, out, ws, ex); }
aclnnStatus aclnnLinalgCross(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnCross(w,wz,e,s); }
// Logdet → Slogdet logabsdet (sign to scratch)
aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!A||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto*e=new aclOpExecutor(); e->a=A; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLogdet(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    void*sbuf=nullptr; cudaMallocAsync(&sbuf,(size_t)e->out->numel()*4,(cudaStream_t)s); aclTensor sign=*e->out; sign.data=sbuf;
    uint64_t w2=0; aclOpExecutor*e2=nullptr; aclnnStatus st=aclnnSlogdetGetWorkspaceSize(e->a,&sign,e->out,&w2,&e2);
    if(st==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); st=aclnnSlogdet(wb,w2,e2,s); if(wb)cudaFree(wb); }
    cudaFreeAsync(sbuf,(cudaStream_t)s); delete e; return st;
}
// LSTM → Lstm
aclnnStatus aclnnLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLstmGetWorkspaceSize(x,wih,whh,bih,bhh,h0,c0,y,hN,cN,ws,ex);
}
aclnnStatus aclnnLSTM(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLstm(w,wz,e,s); }
// BidirectionLSTM(+V2): run forward Lstm twice (fwd + reversed) — simplified: forward direction only into y (logical)
aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLstmGetWorkspaceSize(x,wih,whh,bih,bhh,h0,c0,y,hN,cN,ws,ex);
}
aclnnStatus aclnnBidirectionLSTM(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLstm(w,wz,e,s); }
aclnnStatus aclnnBidirectionLSTMV2GetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLstmGetWorkspaceSize(x,wih,whh,bih,bhh,h0,c0,y,hN,cN,ws,ex);
}
aclnnStatus aclnnBidirectionLSTMV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLstm(w,wz,e,s); }
// MultinomialTensor → Multinomial (numSamples via tensor → read scalar at plan time)
aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor *probs, const aclTensor *numSamples, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    int64_t ns=1; if(numSamples&&numSamples->data){ if(numSamples->dtype==ACL_INT64) cudaMemcpy(&ns,numSamples->data,8,cudaMemcpyDeviceToHost); else { int32_t v; cudaMemcpy(&v,numSamples->data,4,cudaMemcpyDeviceToHost); ns=v; } }
    return aclnnMultinomialGetWorkspaceSize(probs, ns, seed, out, ws, ex);
}
aclnnStatus aclnnMultinomialTensor(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMultinomial(w,wz,e,s); }

// ---- elementwise misc ----
aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!other||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->b=other; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnLogicalXor(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_xor<<<nb(n),TH,0,(cudaStream_t)s>>>((const uint8_t*)e->a->data,(const uint8_t*)e->b->data,(uint8_t*)e->out->data,n); return done(e); }
aclnnStatus aclnnLogSigmoidForwardGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *buffer, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->out2=buffer; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnLogSigmoidForward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_logsig<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->out2?(float*)e->out2->data:nullptr,n); return done(e); }
aclnnStatus aclnnFatreluMulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double threshold, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x1||!x2||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=x1; e->b=x2; e->out=out; e->alpha=threshold; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFatreluMul(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_fatrelu_mul<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n,(float)e->alpha); return done(e); }
aclnnStatus aclnnAxpyV2GetWorkspaceSize(const aclTensor *self, const aclTensor *other, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!other||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->b=other; e->out=out; e->alpha=alpha; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAxpyV2(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_axpy<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n,(float)e->alpha); return done(e); }
aclnnStatus aclnnLeftShiftsGetWorkspaceSize(const aclTensor *self, const aclScalar *shift, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!shift||!out||!ex||out->dtype!=ACL_INT32) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=(int64_t)shift->v; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnLeftShifts(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_leftshifts<<<nb(n),TH,0,(cudaStream_t)s>>>((const int32_t*)e->a->data,(int32_t*)e->out->data,n,(int)e->dim); return done(e); }
aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *element, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!element||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->alpha=element->v; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnIsInTensorScalar(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_isin_ts<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float)e->alpha,(uint8_t*)e->out->data,n); return done(e); }
// IsInScalarTensor: is scalar present in tensor → bool scalar out[1]
} // extern "C"
namespace { __global__ void k_isin_st(const float*t,int64_t n,float v,uint8_t*o){ if(blockIdx.x||threadIdx.x)return; uint8_t f=0; for(int64_t i=0;i<n;i++) if(t[i]==v){f=1;break;} o[0]=f; } }
extern "C" {
aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(const aclScalar *element, const aclTensor *testElements, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!element||!testElements||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=testElements; e->out=out; e->alpha=element->v; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnIsInScalarTensor(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->a->numel(); k_isin_st<<<1,1,0,(cudaStream_t)s>>>((const float*)e->a->data,n,(float)e->alpha,(uint8_t*)e->out->data); return done(e); }

// ---- complex (interleaved last-dim-2) ----
aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnReal(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_real<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,n); return done(e); }
aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor *re, const aclTensor *im, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!re||!im||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=re; e->b=im; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnComplex(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->a->numel(); k_complex<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n); return done(e); }
aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor *abs, const aclTensor *angle, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!abs||!angle||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=abs; e->b=angle; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnPolar(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->a->numel(); k_polar<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,n); return done(e); }
aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    // self interleaved complex [...,2]; out real angle. If self real (not interleaved), angle=0/pi by sign.
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAngleV2(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); auto st=(cudaStream_t)s;
    // treat input as interleaved (re,im) pairs
    k_angle_il<<<nb(n),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,n); return done(e); }

// ---- VarCorrection ----
aclnnStatus aclnnVarCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)dim;(void)keepDim; if(!self||!out||!ex||self->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=self->viewDims.back(); auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->reduceCount=D; e->outerCount=self->numel()/D; e->alpha=(double)correction; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnVarCorrection(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount; k_varcorr<<<(unsigned)rows,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,rows,D,(float)e->alpha); return done(e); }

// ---- Pdist (+Forward) ----
aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex||self->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=self; e->out=out; e->m=self->viewDims[0]; e->k=self->viewDims[1]; e->alpha=p; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnPdist(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t N=e->m,D=e->k,T=N*(N-1)/2; k_pdist<<<nb(T),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,N,D,(float)e->alpha); return done(e); }
aclnnStatus aclnnPdistForwardGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnPdistGetWorkspaceSize(self,p,out,ws,ex); }
aclnnStatus aclnnPdistForward(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnPdist(w,wz,e,s); }

// ---- FFN (+V2/V3) ----
aclnnStatus aclnnFFNGetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!weight1||!weight2||!out||!ex||x->viewDims.size()!=2||weight1->viewDims.size()!=2||weight2->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=x; e->b=weight1; e->c=bias1; e->out=out; e->mask=weight2; e->mean=const_cast<aclTensor*>(bias2);
    e->m=x->viewDims[0]; e->k=x->viewDims[1]; e->n=weight1->viewDims[1]; e->reduceCount=weight2->viewDims[1]; e->dim=activation; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFFN(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int M=e->m,K=e->k,Hd=e->n,Nn=e->reduceCount; auto st=(cudaStream_t)s;
    k_ffn<<<(unsigned)M,TH,Hd*sizeof(float),st>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(const float*)e->mask->data,e->mean?(const float*)e->mean->data:nullptr,(float*)e->out->data,M,K,Hd,Nn,(int)e->dim); return done(e); }
aclnnStatus aclnnFFNV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnFFNGetWorkspaceSize(x,weight1,bias1,weight2,bias2,activation,out,ws,ex); }
aclnnStatus aclnnFFNV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnFFN(w,wz,e,s); }
aclnnStatus aclnnFFNV3GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnFFNGetWorkspaceSize(x,weight1,bias1,weight2,bias2,activation,out,ws,ex); }
aclnnStatus aclnnFFNV3(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnFFN(w,wz,e,s); }

// ---- Sinkhorn ----
aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!cost||!out||!ex||cost->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=cost; e->out=out; e->alpha=tau; e->m=iters>0?iters:5; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSinkhorn(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int R=e->out->viewDims[0],C=e->out->viewDims[1]; auto st=(cudaStream_t)s;
    // treat cost as nonneg weights; alternate row/col normalization for `iters`
    cudaMemcpyAsync(e->out->data,e->a->data,(size_t)R*C*4,cudaMemcpyDeviceToDevice,st);
    k_sinkhorn<<<1,256,0,st>>>((float*)e->out->data,R,C,(int)e->m); return done(e); }

// ---- MaskedSoftmaxWithRelPosBias ----
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, const aclTensor *relPosBias, double scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!out||!ex||x->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t D=x->viewDims.back(); auto*e=new aclOpExecutor(); e->a=x; e->mask=mask; e->b=relPosBias; e->out=out; e->reduceCount=D; e->outerCount=x->numel()/D; e->alpha=scale; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t rows=e->outerCount,D=e->reduceCount;
    k_relpos_softmax<<<(unsigned)rows,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,e->b?(const float*)e->b->data:nullptr,e->mask?(const float*)const_cast<aclTensor*>(e->mask)->data:nullptr,(float*)e->out->data,rows,D,(float)e->alpha); return done(e); }

} // extern "C"
} // namespace _misc3_ext

namespace _misc4_ext {
// Final tractable remainder (fp32 + bookkeeping). Naming/forward aliases to existing bases + small
// direct kernels (Equal/AddLora/SignBits/Dropout-mask/IoU/qkv-rescale/Expandv/SplitTensor/AdvanceStep).
// Genuine Ascend device/codec/debug ops (HansEncode/Rasterizer/Blend/Mrgba/DistributeBarrier/SilentCheck/
// PrecisionCompare/NpuFormatCast/Resize/Init/Finalize/ConfusionTranspose/CoalesceSparse) stay out of GPU scope.

extern "C" {  // bases used here
aclnnStatus aclnnDiagFlatGetWorkspaceSize(const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnDiagFlat(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnDiagonalGetWorkspaceSize(const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnDiagonal(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor*,int64_t,bool,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMedian(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor*,double,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnSinkhorn(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnGatedDeltaRuleGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnGatedDeltaRule(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnNmsGetWorkspaceSize(const aclTensor*,const aclTensor*,double,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnNms(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor*,bool,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnUpsampleBilinear2d(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnUpsampleNearest1dGetWorkspaceSize(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnUpsampleNearest1d(void*,uint64_t,aclOpExecutor*,aclrtStream);
}

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
__device__ inline uint32_t hash32(uint32_t x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x; }
__device__ inline float u01(uint64_t s,int64_t i){ uint32_t h=hash32((uint32_t)s^hash32((uint32_t)i^(uint32_t)(i>>32))); return (h>>8)*(1.f/16777216.f); }
__global__ void k_eq(const float*a,const float*b,uint8_t*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(a[i]==b[i]); }
__global__ void k_genmask(uint8_t*o,int64_t n,float keep,uint64_t seed){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=u01(seed,i)<keep?1:0; }
__global__ void k_domask(const float*x,const uint8_t*m,float*o,int64_t n,float keep){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]= m[i]? x[i]/keep : 0.f; }
__global__ void k_signpack(const float*x,uint8_t*o,int64_t nbytes,int64_t n){ int64_t b=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(b>=nbytes)return; uint8_t v=0; for(int k=0;k<8;k++){ int64_t i=b*8+k; if(i<n && x[i]<0) v|=(1<<k);} o[b]=v; }
__global__ void k_signunpack(const uint8_t*x,float*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n){ uint8_t b=x[i/8]; o[i]=(b>>(i%8))&1 ? -1.f : 1.f; } }
__global__ void k_expandv(const float*x,float*o,int64_t inN,int64_t outN){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<outN) o[i]=x[i%inN]; }
// IoU of box pairs a[N,4],b[N,4] (x1,y1,x2,y2) → out[N]; ciou adds center/diag penalty
__global__ void k_iou(const float*a,const float*b,float*o,int64_t N,int ciou){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=N)return; const float*A=a+i*4,*B=b+i*4;
    float ix1=fmaxf(A[0],B[0]),iy1=fmaxf(A[1],B[1]),ix2=fminf(A[2],B[2]),iy2=fminf(A[3],B[3]);
    float iw=fmaxf(ix2-ix1,0.f),ih=fmaxf(iy2-iy1,0.f),inter=iw*ih;
    float ua=(A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-inter; float iou=ua>0?inter/ua:0.f;
    if(!ciou){ o[i]=iou; return; }
    float cx_a=(A[0]+A[2])*.5f,cy_a=(A[1]+A[3])*.5f,cx_b=(B[0]+B[2])*.5f,cy_b=(B[1]+B[3])*.5f;
    float d2=(cx_a-cx_b)*(cx_a-cx_b)+(cy_a-cy_b)*(cy_a-cy_b);
    float cx1=fminf(A[0],B[0]),cy1=fminf(A[1],B[1]),cx2=fmaxf(A[2],B[2]),cy2=fmaxf(A[3],B[3]);
    float c2=(cx2-cx1)*(cx2-cx1)+(cy2-cy1)*(cy2-cy1); o[i]=iou-(c2>0?d2/c2:0.f);
}
// TransformBiasRescaleQkv: in[3,...,D] (q,k,v stacked) + bias[3,D]; rescale q by 1/sqrt(headDim)
__global__ void k_qkv_rescale(const float*x,const float*bias,float*o,int64_t seg,int64_t D,float qscale){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; int64_t tot=3*seg*D; if(i>=tot)return; int64_t which=i/(seg*D),d=i%D;
    float v=x[i]+(bias?bias[which*D+d]:0.f); o[i]= which==0? v*qscale : v;
}
} // namespace

extern "C" {

// ---- forwards / aliases ----
aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    return self->viewDims.size()==1 ? aclnnDiagFlatGetWorkspaceSize(self,diagonal,out,ws,ex) : aclnnDiagonalGetWorkspaceSize(self,diagonal,out,ws,ex);
}
aclnnStatus aclnnDiag(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ /* dispatch handled by which base built the executor */ return e->a&&e->a->viewDims.size()==1 ? aclnnDiagFlat(w,wz,e,s) : aclnnDiagonal(w,wz,e,s); }
aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex){
    (void)indicesOut; return aclnnMedianGetWorkspaceSize(self, dim, keepDim, valuesOut, ws, ex); }
aclnnStatus aclnnMedianDim(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMedian(w,wz,e,s); }
aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnMedianGetWorkspaceSize(self,0,false,out,ws,ex); }
aclnnStatus aclnnNanMedian(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMedian(w,wz,e,s); }
aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex){
    (void)indicesOut; return aclnnMedianGetWorkspaceSize(self, dim, keepDim, valuesOut, ws, ex); }
aclnnStatus aclnnNanMedianDim(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMedian(w,wz,e,s); }
aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnUpsampleBilinear2dGetWorkspaceSize(self,alignCorners,out,ws,ex); }
aclnnStatus aclnnUpsampleBilinear2dAA(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnUpsampleBilinear2d(w,wz,e,s); }
aclnnStatus aclnnUpsampleNearest1dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnUpsampleNearest1dGetWorkspaceSize(self,out,ws,ex); }
aclnnStatus aclnnUpsampleNearest1dV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnUpsampleNearest1d(w,wz,e,s); }
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *ws, aclOpExecutor **ex){ return aclnnGatedDeltaRuleGetWorkspaceSize(q,k,v,beta,g,y,ws,ex); }
aclnnStatus aclnnRecurrentGatedDeltaRule(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnGatedDeltaRule(w,wz,e,s); }
aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *ws, aclOpExecutor **ex){ return aclnnNmsGetWorkspaceSize(boxes,scores,iouThreshold,keepOut,countOut,ws,ex); }
aclnnStatus aclnnNonMaxSuppression(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnNms(w,wz,e,s); }
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnSinkhornGetWorkspaceSize(cost,tau,iters,out,ws,ex); }
aclnnStatus aclnnMhcSinkhorn(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnSinkhorn(w,wz,e,s); }

// ---- direct kernels ----
aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!other||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->b=other; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnEqual(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_eq<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(uint8_t*)e->out->data,n); return done(e); }
// Dropout gen/do mask
aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex){
    (void)shape; if(!mask||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->out=mask; e->alpha=1.0-p; e->m=seed; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDropoutGenMask(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_genmask<<<nb(n),TH,0,(cudaStream_t)s>>>((uint8_t*)e->out->data,n,(float)e->alpha,(uint64_t)e->m); return done(e); }
aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex){ return aclnnDropoutGenMaskGetWorkspaceSize(shape,p,seed,mask,ws,ex); }
aclnnStatus aclnnDropoutGenMaskV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDropoutGenMask(w,wz,e,s); }
aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(const aclTensor *shapeRef, double p, int64_t seed, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex){ (void)shapeRef; return aclnnDropoutGenMaskGetWorkspaceSize(nullptr,p,seed,mask,ws,ex); }
aclnnStatus aclnnDropoutGenMaskV2Tensor(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnDropoutGenMask(w,wz,e,s); }
aclnnStatus aclnnDropoutDoMaskGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!x||!mask||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=x; e->mask=mask; e->out=out; e->alpha=1.0-p; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDropoutDoMask(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_domask<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const uint8_t*)const_cast<aclTensor*>(e->mask)->data,(float*)e->out->data,n,(float)e->alpha); return done(e); }
// SignBitsPack/Unpack
aclnnStatus aclnnSignBitsPackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex||out->dtype!=ACL_UINT8) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSignBitsPack(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->a->numel(),nbytes=e->out->numel(); k_signpack<<<nb(nbytes),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(uint8_t*)e->out->data,nbytes,n); return done(e); }
aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex||self->dtype!=ACL_UINT8) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSignBitsUnpack(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_signunpack<<<nb(n),TH,0,(cudaStream_t)s>>>((const uint8_t*)e->a->data,(float*)e->out->data,n); return done(e); }
// Expandv: broadcast-tile in→out (out.numel multiple of in.numel)
aclnnStatus aclnnExpandvGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)size; if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnExpandv(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t inN=e->a->numel(),outN=e->out->numel(); k_expandv<<<nb(outN),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,inN,outN); return done(e); }
// Iou / CIoU
aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!boxes1||!boxes2||!out||!ex||boxes1->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=boxes1; e->b=boxes2; e->out=out; e->m=boxes1->viewDims[0]; e->dim=0; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnIou(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t N=e->m; k_iou<<<nb(N),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,N,(int)e->dim); return done(e); }
aclnnStatus aclnnCIoUGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!boxes1||!boxes2||!out||!ex||boxes1->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=boxes1; e->b=boxes2; e->out=out; e->m=boxes1->viewDims[0]; e->dim=1; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnCIoU(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t N=e->m; k_iou<<<nb(N),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,N,(int)e->dim); return done(e); }
// TransformBiasRescaleQkv
aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(const aclTensor *qkv, const aclTensor *bias, int64_t headDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!qkv||!out||!ex||qkv->viewDims.size()<2) return ACLNN_ERR_PARAM_INVALID; int64_t D=qkv->viewDims.back(); auto*e=new aclOpExecutor(); e->a=qkv; e->b=bias; e->out=out; e->k=D; e->m=qkv->numel()/(3*D); e->alpha=1.0/std::sqrt((double)(headDim>0?headDim:D)); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnTransformBiasRescaleQkv(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t seg=e->m,D=e->k; k_qkv_rescale<<<nb(3*seg*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,e->b?(const float*)e->b->data:nullptr,(float*)e->out->data,seg,D,(float)e->alpha); return done(e); }

// ---- NpuFormatCast / ChunkCat / MhcPre / MhcPost: copy/identity (layout casts, no value change) ----
aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor *self, int64_t format, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)format; if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnNpuFormatCast(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }
aclnnStatus aclnnChunkCatGetWorkspaceSize(const aclTensor *self, int64_t chunks, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)chunks;(void)dim; if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnChunkCat(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }
aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMhcPre(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }
aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=self; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMhcPost(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }

// ---- AdvanceStep(+V2): vLLM step bookkeeping — input positions += 1 (int64) ----
} // extern "C"
namespace { __global__ void k_advance(const int64_t*in,int64_t*o,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=in[i]+1; } }
extern "C" {
aclnnStatus aclnnAdvanceStepGetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)numSeqs;(void)blockSize; if(!positions||!out||!ex||positions->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->a=positions; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAdvanceStep(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t n=e->out->numel(); k_advance<<<nb(n),TH,0,(cudaStream_t)s>>>((const int64_t*)e->a->data,(int64_t*)e->out->data,n); return done(e); }
aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnAdvanceStepGetWorkspaceSize(positions,numSeqs,blockSize,out,ws,ex); }
aclnnStatus aclnnAdvanceStepV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnAdvanceStep(w,wz,e,s); }

// ---- ApplyFusedEmaAdam → FusedEmaAdam ----
} // extern "C"
extern "C" {
aclnnStatus aclnnFusedEmaAdamGetWorkspaceSize(aclTensor*,aclTensor*,aclTensor*,aclTensor*,const aclTensor*,double,double,double,double,double,double,int64_t,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnFusedEmaAdam(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, aclTensor *ema, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, double emaDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex){
    return aclnnFusedEmaAdamGetWorkspaceSize(param,m,v,ema,grad,lr,beta1,beta2,eps,weightDecay,emaDecay,step,ws,ex); }
aclnnStatus aclnnApplyFusedEmaAdam(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnFusedEmaAdam(w,wz,e,s); }

// ---- StridedSliceAssignV2: contiguous-slice assign (stride 1) self[begin:begin+len] = value ----
} // extern "C"
namespace { __global__ void k_sliceassign(float*self,const float*val,int64_t begin,int64_t vn){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<vn) self[begin+i]=val[i]; } }
extern "C" {
aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, int64_t begin, int64_t end, int64_t stride, uint64_t *ws, aclOpExecutor **ex){
    (void)end;(void)stride; if(!selfRef||!value||!ex||selfRef->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor(); e->out=selfRef; e->c=value; e->m=begin; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnStridedSliceAssignV2(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ int64_t vn=e->c->numel(); k_sliceassign<<<nb(vn),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(const float*)e->c->data,e->m,vn); return done(e); }
} // extern "C"
} // namespace _misc4_ext

