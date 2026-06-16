// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "aclnnop/aclnn_add.h"

namespace _math_ext {
// Misc math ops needing dedicated kernels: Addr (rank-1 update), DivMods (tensor // scalar with round mode).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// Addr: out[i,j] = beta·self[i,j] + alpha·vec1[i]·vec2[j]
template <typename T>
__global__ void k_addr(const T *self, const T *v1, const T *v2, T *out, int64_t M, int64_t N, float beta, float alpha) {
    int64_t idx = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx >= M*N) return;
    int64_t i = idx / N, j = idx % N;
    float s = self ? (float)self[idx] : 0.f;
    out[idx] = (T)(beta*s + alpha*(float)v1[i]*(float)v2[j]);
}
// DivMods: out = round(self / other) ; mode 0 = floor, 1 = trunc-toward-zero
template <typename T>
__global__ void k_divs_mode(const T *self, float other, T *out, int64_t n, int mode) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return;
    float q = (float)self[i] / other;
    out[i] = (T)(mode==1 ? truncf(q) : floorf(q));
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
inline const void *D_(const aclTensor *t){ return t?t->data:nullptr; }

// tensor∘scalar elementwise math. mode: 0 fmod(trunc rem), 1 remainder(floor rem), 2 x·log(c), 3 c·log(x)
template <typename T>
__global__ void k_smath(const T *x, float c, T *out, int64_t n, int mode) {
    int64_t i = (int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (i >= n) return; float v=(float)x[i], r;
    switch (mode) { case 0: r = fmodf(v, c); break;
                    case 1: r = v - floorf(v/c)*c; break;
                    case 2: r = v * logf(c); break;
                    default: r = c * logf(v); break; }
    out[i] = (T)r;
}
static aclnnStatus smath_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t n = e->out->numel(), g = nb(n); float c=(float)e->alpha; int mode=e->dim;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_smath<float><<<g,TH,0,s>>>((const float*)e->a->data,c,(float*)e->out->data,n,mode); break;
        case ACL_FLOAT16: k_smath<__half><<<g,TH,0,s>>>((const __half*)e->a->data,c,(__half*)e->out->data,n,mode); break;
        default:          k_smath<__nv_bfloat16><<<g,TH,0,s>>>((const __nv_bfloat16*)e->a->data,c,(__nv_bfloat16*)e->out->data,n,mode); break;
    }
    return fin(e);
}
static aclnnStatus smath_ws(const aclTensor *x, double c, int mode, aclTensor *out, aclOpExecutor **ex) {
    if (!x || !out || !ex || !x->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->alpha = c; e->dim = mode; *ex = e; return ACLNN_SUCCESS;
}

static aclnnStatus addr_ws(const aclTensor *self, const aclTensor *v1, const aclTensor *v2, double beta, double alpha, aclTensor *out, aclOpExecutor **ex) {
    if (!v1 || !v2 || !out || !ex || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t M = out->viewDims[0], N = out->viewDims[1];
    if (v1->numel() != M || v2->numel() != N) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->b = v1; e->c = v2; e->out = out; e->m = M; e->n = N; e->dscalars = {beta, alpha};
    *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus addr_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t M = e->m, N = e->n, g = nb(M*N); float beta=(float)e->dscalars[0], alpha=(float)e->dscalars[1];
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_addr<float><<<g,TH,0,s>>>((const float*)D_(e->a),(const float*)e->b->data,(const float*)e->c->data,(float*)e->out->data,M,N,beta,alpha); break;
        case ACL_FLOAT16: k_addr<__half><<<g,TH,0,s>>>((const __half*)D_(e->a),(const __half*)e->b->data,(const __half*)e->c->data,(__half*)e->out->data,M,N,beta,alpha); break;
        default:          k_addr<__nv_bfloat16><<<g,TH,0,s>>>((const __nv_bfloat16*)D_(e->a),(const __nv_bfloat16*)e->b->data,(const __nv_bfloat16*)e->c->data,(__nv_bfloat16*)e->out->data,M,N,beta,alpha); break;
    }
    return fin(e);
}
} // namespace

extern "C" {

// Addr: out = beta·self + alpha·(vec1 ⊗ vec2)
aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor *self, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    double b = beta ? beta->v : 1.0, a = alpha ? alpha->v : 1.0;
    if (ws) *ws = 0; return addr_ws(self, vec1, vec2, b, a, out, ex);
}
aclnnStatus aclnnAddr(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return addr_run(e, (cudaStream_t)s); }
// InplaceAddr: self = beta·self + alpha·(vec1 ⊗ vec2)
aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha,
        uint64_t *ws, aclOpExecutor **ex) {
    double b = beta ? beta->v : 1.0, a = alpha ? alpha->v : 1.0;
    if (ws) *ws = 0; return addr_ws(selfRef, vec1, vec2, b, a, selfRef, ex);
}
aclnnStatus aclnnInplaceAddr(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return addr_run(e, (cudaStream_t)s); }

// DivMods: out = round_mode(self / scalar). roundMode: 0 floor, 1 trunc.
aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, int64_t roundMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (out->dtype != ACL_FLOAT && out->dtype != ACL_FLOAT16 && out->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = self; e->out = out; e->dim = (int)roundMode; e->alpha = other->v;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDivMods(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->a->numel(), g = nb(n); auto st = (cudaStream_t)s; float other=(float)e->alpha; int mode=e->dim;
    switch (e->out->dtype) {
        case ACL_FLOAT:   k_divs_mode<float><<<g,TH,0,st>>>((const float*)e->a->data,other,(float*)e->out->data,n,mode); break;
        case ACL_FLOAT16: k_divs_mode<__half><<<g,TH,0,st>>>((const __half*)e->a->data,other,(__half*)e->out->data,n,mode); break;
        default:          k_divs_mode<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,other,(__nv_bfloat16*)e->out->data,n,mode); break;
    }
    return fin(e);
}
aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, int64_t roundMode, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDivModsGetWorkspaceSize(selfRef, other, roundMode, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceDivMods(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnDivMods(ws, wsz, e, s); }

// FmodScalar: out = fmod(self, scalar) (truncated remainder, sign of dividend)
aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(self, other->v, 0, out, ex);
}
aclnnStatus aclnnFmodScalar(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(selfRef, other->v, 0, selfRef, ex);
}
aclnnStatus aclnnInplaceFmodScalar(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }

// RemainderTensorScalar: out = self - floor(self/scalar)·scalar (floor remainder, sign of divisor)
aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(self, other->v, 1, out, ex);
}
aclnnStatus aclnnRemainderTensorScalar(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(selfRef, other->v, 1, selfRef, ex);
}
aclnnStatus aclnnInplaceRemainderTensorScalar(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }

// XLogYScalarOther: out = self · log(scalar)
aclnnStatus aclnnXLogYScalarOtherGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(self, other->v, 2, out, ex);
}
aclnnStatus aclnnXLogYScalarOther(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceXLogYScalarOtherGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!other) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(selfRef, other->v, 2, selfRef, ex);
}
aclnnStatus aclnnInplaceXLogYScalarOther(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }

// XLogYScalarSelf: out = scalar · log(other_tensor)
aclnnStatus aclnnXLogYScalarSelfGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self) return ACLNN_ERR_PARAM_NULLPTR; if (ws) *ws = 0; return smath_ws(other, self->v, 3, out, ex);
}
aclnnStatus aclnnXLogYScalarSelf(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return smath_run(e, (cudaStream_t)s); }

} // extern "C"
} // namespace _math_ext

namespace _simplemath_ext {
// Simple math/elementwise one-offs (task #154). Self-contained kernels, CPU-cross-checkable.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

// unary float op: mode 0 logit, 1 softsign, 2 signbit(out bool/uint8)
template <typename T> __global__ void k_un(const T *x, T *o, int64_t n, int mode, float eps) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float v=(float)x[i],r;
    if (mode==0){ float c=v<eps?eps:(v>1-eps?1-eps:v); r=logf(c/(1.f-c)); }
    else r=v/(1.f+fabsf(v));   // softsign
    o[i]=(T)r;
}
template <typename T> __global__ void k_signbit(const T *x, uint8_t *o, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(uint8_t)(((float)x[i])<0.f || (1.f/(float)x[i])<0.f); }
// binary float op: 0 fmod,1 xlogy,2 silumul(silu(a)*b),3 gelumul,4 logit_grad(grad=a, x=b),5 softsign_grad
template <typename T> __global__ void k_bin(const T *a, const T *b, T *o, int64_t n, int mode) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float x=(float)a[i],y=(float)b[i],r;
    switch(mode){ case 0: r=fmodf(x,y); break; case 1: r=(x==0.f)?0.f:x*logf(y); break;
        case 2: r=(x/(1.f+expf(-x)))*y; break; case 3: r=(0.5f*x*erfcf(-x*0.70710678f))*y; break;
        case 4: r=x/(y*(1.f-y)); break; default: r=x/((1.f+fabsf(y))*(1.f+fabsf(y))); }
    o[i]=(T)r;
}
__global__ void k_gcd(const int32_t *a, const int32_t *b, int32_t *o, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; int x=abs(a[i]),y=abs(b[i]); while(y){int t=x%y;x=y;y=t;} o[i]=x; }
__global__ void k_shift(const int32_t *a, const int32_t *b, int32_t *o, int64_t n, int left){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; o[i]=left? (a[i]<<b[i]) : (a[i]>>b[i]); }
template <typename T> __global__ void k_isclose(const T *a, const T *b, uint8_t *o, int64_t n, float rtol, float atol){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float x=(float)a[i],y=(float)b[i]; o[i]=(uint8_t)(fabsf(x-y)<=atol+rtol*fabsf(y)); }
template <typename T> __global__ void k_scale(const T *x, T *o, int64_t n, float sc, float bias){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(T)((float)x[i]*sc+bias); }
template <typename T> __global__ void k_mscale(const T *x, const uint8_t *m, T *o, int64_t n, float sc){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) o[i]=(T)(m[i]?(float)x[i]*sc:0.f); }
#define DT3(KC) switch(e->out->dtype){case ACL_FLOAT:{using T=float;KC;}break;case ACL_FLOAT16:{using T=__half;KC;}break;default:{using T=__nv_bfloat16;KC;}break;}
} // namespace

extern "C" {
// unary
aclnnStatus aclnnLogitGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->out=out;e->dim=0;e->alpha=eps; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnLogit(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;DT3((k_un<T><<<g,TH,0,st>>>((const T*)e->a->data,(T*)e->out->data,n,0,(float)e->alpha)));return fin(e);}
aclnnStatus aclnnSoftsignGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->out=out;e->dim=1; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnSoftsign(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;DT3((k_un<T><<<g,TH,0,st>>>((const T*)e->a->data,(T*)e->out->data,n,1,0.f)));return fin(e);}
aclnnStatus aclnnSignbitGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!out||!ex||out->dtype!=ACL_BOOL&&out->dtype!=ACL_UINT8)return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->a=self;e->out=out; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnSignbit(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;switch(e->a->dtype){case ACL_FLOAT:k_signbit<float><<<g,TH,0,st>>>((const float*)e->a->data,(uint8_t*)e->out->data,n);break;case ACL_FLOAT16:k_signbit<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(uint8_t*)e->out->data,n);break;default:k_signbit<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(uint8_t*)e->out->data,n);break;}return fin(e);}
// binary float
#define BIN_OP(NAME,MODE) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *a, const aclTensor *b, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!a||!b||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=a;e->b=b;e->out=out;e->dim=MODE; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;} \
aclnnStatus NAME(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;DT3((k_bin<T><<<g,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(T*)e->out->data,n,e->dim)));return fin(e);}
BIN_OP(aclnnFmodTensor,0)
BIN_OP(aclnnXLogYTensor,1)
BIN_OP(aclnnSiluMul,2)
BIN_OP(aclnnGeluMul,3)
// gcd / shifts (int32)
aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor *a, const aclTensor *b, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!a||!b||!out||!ex||out->dtype!=ACL_INT32)return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->a=a;e->b=b;e->out=out; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnGcd(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel();k_gcd<<<nb(n),TH,0,(cudaStream_t)s>>>((const int32_t*)e->a->data,(const int32_t*)e->b->data,(int32_t*)e->out->data,n);return fin(e);}
aclnnStatus aclnnLeftShiftGetWorkspaceSize(const aclTensor *a, const aclTensor *b, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!a||!b||!out||!ex||out->dtype!=ACL_INT32)return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->a=a;e->b=b;e->out=out;e->dim=1; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnLeftShift(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel();k_shift<<<nb(n),TH,0,(cudaStream_t)s>>>((const int32_t*)e->a->data,(const int32_t*)e->b->data,(int32_t*)e->out->data,n,1);return fin(e);}
aclnnStatus aclnnRightShiftGetWorkspaceSize(const aclTensor *a, const aclTensor *b, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!a||!b||!out||!ex||out->dtype!=ACL_INT32)return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->a=a;e->b=b;e->out=out;e->dim=0; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnRightShift(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel();k_shift<<<nb(n),TH,0,(cudaStream_t)s>>>((const int32_t*)e->a->data,(const int32_t*)e->b->data,(int32_t*)e->out->data,n,0);return fin(e);}
// isclose
aclnnStatus aclnnIsCloseGetWorkspaceSize(const aclTensor *self, const aclTensor *other, double rtol, double atol, bool equalNan, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)equalNan; if(!self||!other||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->b=other;e->out=out;e->dscalars={rtol,atol}; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnIsClose(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->a->numel(),g=nb(n);auto st=(cudaStream_t)s;float rt=(float)e->dscalars[0],at=(float)e->dscalars[1];switch(e->a->dtype){case ACL_FLOAT:k_isclose<float><<<g,TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(uint8_t*)e->out->data,n,rt,at);break;case ACL_FLOAT16:k_isclose<__half><<<g,TH,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(uint8_t*)e->out->data,n,rt,at);break;default:k_isclose<__nv_bfloat16><<<g,TH,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(uint8_t*)e->out->data,n,rt,at);break;}return fin(e);}
// scale / maskedscale
aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *bias, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!scale||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->out=out;e->alpha=scale->v;e->dscalars={bias?bias->v:0.0}; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnScale(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float sc=(float)e->alpha,bi=(float)e->dscalars[0];DT3((k_scale<T><<<g,TH,0,st>>>((const T*)e->a->data,(T*)e->out->data,n,sc,bi)));return fin(e);}
aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, double scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!mask||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->b=mask;e->out=out;e->alpha=scale; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnMaskedScale(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float sc=(float)e->alpha;DT3((k_mscale<T><<<g,TH,0,st>>>((const T*)e->a->data,(const uint8_t*)e->b->data,(T*)e->out->data,n,sc)));return fin(e);}
} // extern "C"
namespace {
// --- more #154 ops ---
// Shrink: out = (x>lambd)? x-bias : (x<-lambd)? x+bias : 0
template <typename T> __global__ void k_shrink(const T *x, T *o, int64_t n, float lambd, float bias){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float v=(float)x[i],r=0.f; if(v>lambd) r=v-bias; else if(v<-lambd) r=v+bias; o[i]=(T)r; }
// scalar∘tensor: mode 0 remainder(s,t)=s-floor(s/t)*t, 1 pow(s,t)=s^t
template <typename T> __global__ void k_st(float s, const T *t, T *o, int64_t n, int mode){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float y=(float)t[i],r; r = mode? powf(s,y) : s-floorf(s/y)*y; o[i]=(T)r; }
// Lerps: out = a + w*(b-a), scalar w
template <typename T> __global__ void k_lerps(const T *a, const T *b, T *o, int64_t n, float w){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float x=(float)a[i],y=(float)b[i]; o[i]=(T)(x+w*(y-x)); }
// MaxN/MinN over K tensors (pointers in dptr): out[i] = max/min_k src_k[i]
template <typename T> __global__ void k_nary(const T *const *dptr, int K, T *o, int64_t n, int mx){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float acc=(float)dptr[0][i]; for(int k=1;k<K;k++){ float v=(float)dptr[k][i]; acc = mx?fmaxf(acc,v):fminf(acc,v);} o[i]=(T)acc; }
// Cdist: x1[P,M], x2[R,M] -> out[P,R] = (Σ|x1-x2|^p)^(1/p)
template <typename T> __global__ void k_cdist(const T *x1, const T *x2, T *o, int64_t P, int64_t R, int64_t M, float p){ int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=P*R) return; int64_t pi=idx/R, ri=idx%R; double s=0; for(int64_t k=0;k<M;k++){ double d=fabs((double)x1[pi*M+k]-(double)x2[ri*M+k]); s+=pow(d,p);} o[idx]=(T)pow(s,1.0/p); }
} // namespace

extern "C" {
aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor *self, int64_t startDim, int64_t endDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)startDim;(void)endDim; if(!self||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->out=out; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnFlatten(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->a->numel()*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return fin(e);}
aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor *self, const aclScalar *lambd, const aclScalar *bias, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->out=out;e->alpha=lambd?lambd->v:0.5;e->dscalars={bias?bias->v:0.0}; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnShrink(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float L=(float)e->alpha,B=(float)e->dscalars[0];DT3((k_shrink<T><<<g,TH,0,st>>>((const T*)e->a->data,(T*)e->out->data,n,L,B)));return fin(e);}
aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!other||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->b=other;e->out=out;e->alpha=self->v;e->dim=0; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnRemainderScalarTensor(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float sv=(float)e->alpha;DT3((k_st<T><<<g,TH,0,st>>>(sv,(const T*)e->b->data,(T*)e->out->data,n,0)));return fin(e);}
aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *exponent, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!exponent||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->b=exponent;e->out=out;e->alpha=self->v;e->dim=1; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnPowScalarTensor(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float sv=(float)e->alpha;DT3((k_st<T><<<g,TH,0,st>>>(sv,(const T*)e->b->data,(T*)e->out->data,n,1)));return fin(e);}
aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor *self, const aclTensor *end, const aclScalar *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!self||!end||!weight||!out||!ex)return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor();e->a=self;e->b=end;e->out=out;e->alpha=weight->v; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnLerps(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t n=e->out->numel(),g=nb(n);auto st=(cudaStream_t)s;float w=(float)e->alpha;DT3((k_lerps<T><<<g,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(T*)e->out->data,n,w)));return fin(e);}
static aclnnStatus naryN(const aclTensorList *tl, aclTensor *out, int mx, aclOpExecutor **ex){ if(!tl||!out||!ex||tl->v.empty())return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->out=out;e->dim=mx;for(auto t:tl->v)e->inputs.push_back(t);*ex=e;return ACLNN_SUCCESS;}
static aclnnStatus naryRun(aclOpExecutor*e,cudaStream_t s){ int K=(int)e->inputs.size(); int64_t n=e->out->numel(),g=nb(n);
  std::vector<const void*> hp(K); for(int k=0;k<K;k++) hp[k]=e->inputs[k]->data; void**dp=nullptr; cudaMallocAsync((void**)&dp,K*sizeof(void*),s); cudaMemcpyAsync(dp,hp.data(),K*sizeof(void*),cudaMemcpyHostToDevice,s);
  DT3((k_nary<T><<<g,TH,0,s>>>((const T*const*)dp,K,(T*)e->out->data,n,e->dim))); cudaFreeAsync(dp,s); return fin(e);}
aclnnStatus aclnnMaxNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return naryN(tensors,out,1,ex);}
aclnnStatus aclnnMaxN(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return naryRun(e,(cudaStream_t)s);}
aclnnStatus aclnnMinNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return naryN(tensors,out,0,ex);}
aclnnStatus aclnnMinN(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return naryRun(e,(cudaStream_t)s);}
aclnnStatus aclnnCdistGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(!x1||!x2||!out||!ex||x1->viewDims.size()!=2||x2->viewDims.size()!=2)return ACLNN_ERR_PARAM_INVALID; auto*e=new aclOpExecutor();e->a=x1;e->b=x2;e->out=out;e->alpha=p;e->m=x1->viewDims[0];e->n=x2->viewDims[0];e->k=x1->viewDims[1]; if(ws)*ws=0;*ex=e;return ACLNN_SUCCESS;}
aclnnStatus aclnnCdist(void*,uint64_t,aclOpExecutor*e,aclrtStream s){int64_t P=e->m,R=e->n,M=e->k,g=nb(P*R);auto st=(cudaStream_t)s;float p=(float)e->alpha;DT3((k_cdist<T><<<g,TH,0,st>>>((const T*)e->a->data,(const T*)e->b->data,(T*)e->out->data,P,R,M,p)));return fin(e);}

// Ger: out[M,N]=vec1[M]*vec2[N] → forward to Addr (beta=0, no self)
aclnnStatus aclnnGerGetWorkspaceSize(const aclTensor *self, const aclTensor *vec2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnAddrGetWorkspaceSize(nullptr, self, vec2, nullptr, nullptr, out, ws, ex); }
aclnnStatus aclnnGer(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s){ return aclnnAddr(w, wz, e, s); }
} // extern "C"
} // namespace _simplemath_ext
#undef DT3
#undef BIN_OP

namespace _var_fwd_ext {
// Auto-generated version-variant forwards (same simplified ABI as base op). See operator-coverage-goal memory.
extern "C" {
aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnApplyAdamWGetWorkspaceSize(param, m, v, grad, lr, beta1, beta2, eps, weightDecay, step, workspaceSize, executor); }
aclnnStatus aclnnApplyAdamWV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnApplyAdamW(w, wz, e, s); }
aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnApplyRotaryPosEmbGetWorkspaceSize(q, k, cos, sin, mode, qOut, kOut, workspaceSize, executor); }
aclnnStatus aclnnApplyRotaryPosEmbV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnApplyRotaryPosEmb(w, wz, e, s); }
aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnCumsumGetWorkspaceSize(self, dim, dtype, out, workspaceSize, executor); }
aclnnStatus aclnnCumsumV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnCumsum(w, wz, e, s); }
aclnnStatus aclnnDropoutV3GetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnDropoutGetWorkspaceSize(x, p, seed, out, mask, workspaceSize, executor); }
aclnnStatus aclnnDropoutV3(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnDropout(w, wz, e, s); }
aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnGatherGetWorkspaceSize(self, dim, index, out, workspaceSize, executor); }
aclnnStatus aclnnGatherV3(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnGather(w, wz, e, s); }
aclnnStatus aclnnNonzeroV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnNonzeroGetWorkspaceSize(self, out, countOut, workspaceSize, executor); }
aclnnStatus aclnnNonzeroV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnNonzero(w, wz, e, s); }
aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnRoiAlignGetWorkspaceSize(self, rois, spatialScale, samplingRatio, out, workspaceSize, executor); }
aclnnStatus aclnnRoiAlignV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnRoiAlign(w, wz, e, s); }
aclnnStatus aclnnSliceV2GetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnSliceGetWorkspaceSize(self, dim, start, end, step, out, workspaceSize, executor); }
aclnnStatus aclnnSliceV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnSlice(w, wz, e, s); }
aclnnStatus aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(gradOut, alignCorners, gradInput, workspaceSize, executor); }
aclnnStatus aclnnUpsampleBilinear2dBackwardV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnUpsampleBilinear2dBackward(w, wz, e, s); }
aclnnStatus aclnnUpsampleNearest2dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnUpsampleNearest2dGetWorkspaceSize(self, out, workspaceSize, executor); }
aclnnStatus aclnnUpsampleNearest2dV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnUpsampleNearest2d(w, wz, e, s); }aclnnStatus aclnnAddV3GetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnAddGetWorkspaceSize(self, other, alpha, out, ws, ex); }
aclnnStatus aclnnAddV3(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnAdd(w, wz, e, s); }
aclnnStatus aclnnFastGeluV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnFastGeluGetWorkspaceSize(self, out, ws, ex); }
aclnnStatus aclnnFastGeluV2(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnFastGelu(w, wz, e, s); }
aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *self, int64_t dim, aclDataType dtype, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnCumprodGetWorkspaceSize(self, dim, dtype, self, workspaceSize, executor); }
aclnnStatus aclnnInplaceCumprod(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnCumprod(w, wz, e, s); }
aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *self, const aclTensor *mask, const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnMaskedFillScalarGetWorkspaceSize(self, mask, value, self, workspaceSize, executor); }
aclnnStatus aclnnInplaceMaskedFillScalar(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnMaskedFillScalar(w, wz, e, s); }
aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(aclTensor *self, const aclTensor *mask, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnMaskedScatterGetWorkspaceSize(self, mask, src, self, workspaceSize, executor); }
aclnnStatus aclnnInplaceMaskedScatter(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnMaskedScatter(w, wz, e, s); }
aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(aclTensor *selfRef, double p, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnBernoulliGetWorkspaceSize(selfRef, p, seed, workspaceSize, executor); }
aclnnStatus aclnnInplaceBernoulli(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnBernoulli(w, wz, e, s); }
aclnnStatus aclnnInplaceUniformGetWorkspaceSize(aclTensor *selfRef, double from, double to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnUniformGetWorkspaceSize(selfRef, from, to, seed, workspaceSize, executor); }
aclnnStatus aclnnInplaceUniform(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnUniform(w, wz, e, s); }
} // extern "C"
} // namespace _var_fwd_ext

