// Upsample (nearest 1d/3d + nearest-exact 1d/2d/3d, linear1d, trilinear3d) and GlobalAverage/MaxPool.
// Output spatial size is taken from the out tensor (simplified ABI). fp32/fp16/bf16.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__device__ inline int nsrc(int o, int isz, int osz, int exact){ int i = exact ? (int)floorf((o+0.5f)*isz/osz) : (int)floorf((float)o*isz/osz); return i<0?0:(i>=isz?isz-1:i); }

// Nearest up to 3 spatial dims: out[NC, o0,o1,o2] = in[NC, s0,s1,s2]
template <typename T>
__global__ void k_nearest(const T *in, T *out, int64_t NC, int i0,int i1,int i2, int o0,int o1,int o2, int exact) {
    int64_t osp=(int64_t)o0*o1*o2, isp=(int64_t)i0*i1*i2;
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx>=NC*osp) return;
    int64_t nc=idx/osp, r=idx%osp; int c2=r%o2, c1=(r/o2)%o1, c0=r/(o2*o1);
    int s0=nsrc(c0,i0,o0,exact), s1=nsrc(c1,i1,o1,exact), s2=nsrc(c2,i2,o2,exact);
    out[idx]=in[nc*isp + ((int64_t)s0*i1 + s1)*i2 + s2];
}
// Linear (1d/3d separable): align-corners aware. nsp spatial dims (1 or 3).
template <typename T>
__global__ void k_linear(const T *in, T *out, int64_t NC, int i0,int i1,int i2, int o0,int o1,int o2, int align) {
    int64_t osp=(int64_t)o0*o1*o2, isp=(int64_t)i0*i1*i2;
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx>=NC*osp) return;
    int64_t nc=idx/osp, r=idx%osp; int c2=r%o2, c1=(r/o2)%o1, c0=r/(o2*o1);
    auto coord=[&](int o,int isz,int osz,int &lo,int &hi,float &fr){ float s; if(align) s=osz>1?(float)o*(isz-1)/(osz-1):0.f; else s=osz>0?((o+0.5f)*isz/osz-0.5f):0.f; if(s<0)s=0; int b=(int)floorf(s); fr=s-b; lo=b<0?0:(b>=isz?isz-1:b); hi=lo+1>=isz?isz-1:lo+1; };
    int l0,h0,l1,h1,l2,h2; float f0,f1,f2; coord(c0,i0,o0,l0,h0,f0); coord(c1,i1,o1,l1,h1,f1); coord(c2,i2,o2,l2,h2,f2);
    const T *p=in+nc*isp; float acc=0;
    for (int a=0;a<2;a++)for(int b=0;b<2;b++)for(int c=0;c<2;c++){ int z=a?h0:l0,y=b?h1:l1,x=c?h2:l2; float w=(a?f0:1-f0)*(b?f1:1-f1)*(c?f2:1-f2); acc+=w*(float)p[((int64_t)z*i1+y)*i2+x]; }
    out[idx]=(T)acc;
}
// Global pool: reduce all spatial → 1. avg=1, else max.
template <typename T>
__global__ void k_globalpool(const T *in, T *out, int64_t NC, int64_t sp, int avg) {
    int64_t nc=blockIdx.x; if (nc>=NC) return; const T *p=in+nc*sp; int t=threadIdx.x;
    float acc = avg?0.f:-1e30f;
    for (int64_t i=t;i<sp;i+=TH){ float v=(float)p[i]; acc = avg?acc+v:fmaxf(acc,v); }
    __shared__ float sh[TH]; sh[t]=acc; __syncthreads();
    for (int s=TH/2;s>0;s>>=1){ if(t<s) sh[t]= avg?sh[t]+sh[t+s]:fmaxf(sh[t],sh[t+s]); __syncthreads(); }
    if (t==0) out[nc]=(T)(avg?sh[0]/sp:sh[0]);
}
// Upsample backward (fp32, atomic scatter-add of gradOut into gradIn). interp 0/1 nearest(exact), 2 linear.
__global__ void k_nearest_bwd(const float *gOut, float *gIn, int64_t NC, int i0,int i1,int i2, int o0,int o1,int o2, int exact) {
    int64_t osp=(int64_t)o0*o1*o2, isp=(int64_t)i0*i1*i2;
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx>=NC*osp) return;
    int64_t nc=idx/osp, r=idx%osp; int c2=r%o2,c1=(r/o2)%o1,c0=r/(o2*o1);
    int s0=nsrc(c0,i0,o0,exact),s1=nsrc(c1,i1,o1,exact),s2=nsrc(c2,i2,o2,exact);
    atomicAdd(&gIn[nc*isp + ((int64_t)s0*i1+s1)*i2 + s2], gOut[idx]);
}
__global__ void k_linear_bwd(const float *gOut, float *gIn, int64_t NC, int i0,int i1,int i2, int o0,int o1,int o2, int align) {
    int64_t osp=(int64_t)o0*o1*o2, isp=(int64_t)i0*i1*i2;
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if (idx>=NC*osp) return;
    int64_t nc=idx/osp, r=idx%osp; int c2=r%o2,c1=(r/o2)%o1,c0=r/(o2*o1);
    auto coord=[&](int o,int isz,int osz,int &lo,int &hi,float &fr){ float s; if(align) s=osz>1?(float)o*(isz-1)/(osz-1):0.f; else s=osz>0?((o+0.5f)*isz/osz-0.5f):0.f; if(s<0)s=0; int b=(int)floorf(s); fr=s-b; lo=b<0?0:(b>=isz?isz-1:b); hi=lo+1>=isz?isz-1:lo+1; };
    int l0,h0,l1,h1,l2,h2; float f0,f1,f2; coord(c0,i0,o0,l0,h0,f0); coord(c1,i1,o1,l1,h1,f1); coord(c2,i2,o2,l2,h2,f2);
    float go=gOut[idx]; float *p=gIn+nc*isp;
    for (int a=0;a<2;a++)for(int b=0;b<2;b++)for(int c=0;c<2;c++){ int z=a?h0:l0,y=b?h1:l1,x=c?h2:l2; float w=(a?f0:1-f0)*(b?f1:1-f1)*(c?f2:1-f2); atomicAdd(&p[((int64_t)z*i1+y)*i2+x], w*go); }
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
#define DT3(KC) switch(e->a->dtype){case ACL_FLOAT:{using T=float;KC;}break;case ACL_FLOAT16:{using T=__half;KC;}break;default:{using T=__nv_bfloat16;KC;}break;}

// Backward shared: e->a = gradOut, e->out = gradIn; sizes from both. fp32 only.
static aclnnStatus upb_ws(const aclTensor *gradOut, int nsp, int interp, int align, aclTensor *gradIn, aclOpExecutor **ex) {
    if (!gradOut || !gradIn || !ex || gradOut->dtype != ACL_FLOAT || gradIn->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->out=gradIn; e->dim=interp; e->keepDim=align; e->reduceCount=nsp; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus upb_run(aclOpExecutor *e, cudaStream_t s) {
    int nsp=(int)e->reduceCount; const auto &O=e->a->viewDims,&I=e->out->viewDims; int rank=(int)I.size(), sp0=rank-nsp;
    int is[3]={1,1,1},os[3]={1,1,1}; for (int d=0;d<nsp;d++){ is[3-nsp+d]=(int)I[sp0+d]; os[3-nsp+d]=(int)O[sp0+d]; }
    int64_t NC=1; for (int d=0;d<sp0;d++) NC*=I[d];
    cudaMemsetAsync(e->out->data, 0, (size_t)e->out->numel()*sizeof(float), s);
    int64_t g=nb(NC*(int64_t)os[0]*os[1]*os[2]);
    if (e->dim==2) k_linear_bwd<<<g,TH,0,s>>>((const float*)e->a->data,(float*)e->out->data,NC,is[0],is[1],is[2],os[0],os[1],os[2],e->keepDim);
    else           k_nearest_bwd<<<g,TH,0,s>>>((const float*)e->a->data,(float*)e->out->data,NC,is[0],is[1],is[2],os[0],os[1],os[2],e->dim);
    return fin(e);
}

// shared setup for nearest/linear: extract N*C and spatial sizes (last `nsp` dims). interp=0 nearest,1 nearest-exact,2 linear
static aclnnStatus up_ws(const aclTensor *self, int nsp, int interp, int align, aclTensor *out, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    int rank=(int)self->viewDims.size(); if (rank < nsp+1) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=interp; e->keepDim=align; e->reduceCount=nsp;
    *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus up_run(aclOpExecutor *e, cudaStream_t s) {
    int nsp=(int)e->reduceCount, rank=(int)e->a->viewDims.size();
    int i0=1,i1=1,i2=1,o0=1,o1=1,o2=1; const auto &S=e->a->viewDims,&O=e->out->viewDims;
    int sp0=rank-nsp;  // first spatial dim index
    int is[3]={1,1,1},os[3]={1,1,1};
    for (int d=0;d<nsp;d++){ is[3-nsp+d]=(int)S[sp0+d]; os[3-nsp+d]=(int)O[sp0+d]; }
    i0=is[0];i1=is[1];i2=is[2]; o0=os[0];o1=os[1];o2=os[2];
    int64_t NC=1; for (int d=0;d<sp0;d++) NC*=S[d];
    int64_t g=nb(NC*(int64_t)o0*o1*o2);
    if (e->dim==2) { DT3(( k_linear<T><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,i0,i1,i2,o0,o1,o2,e->keepDim) )); }
    else           { DT3(( k_nearest<T><<<g,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,i0,i1,i2,o0,o1,o2,e->dim) )); }
    return fin(e);
}
} // namespace

extern "C" {

aclnnStatus aclnnUpsampleNearest1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,1,0,0,out,ex);}
aclnnStatus aclnnUpsampleNearest1d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,3,0,0,out,ex);}
aclnnStatus aclnnUpsampleNearest3d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,1,1,0,out,ex);}
aclnnStatus aclnnUpsampleNearestExact1d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,2,1,0,out,ex);}
aclnnStatus aclnnUpsampleNearestExact2d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,3,1,0,out,ex);}
aclnnStatus aclnnUpsampleNearestExact3d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,1,2,alignCorners,out,ex);}
aclnnStatus aclnnUpsampleLinear1d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return up_ws(self,3,2,alignCorners,out,ex);}
aclnnStatus aclnnUpsampleTrilinear3d(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return up_run(e,(cudaStream_t)s);}

// GlobalAveragePool / GlobalMaxPool: self[N,C,*] -> out[N,C,1,...] (reduce all spatial)
static aclnnStatus gp_ws(const aclTensor *self, int avg, aclTensor *out, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data || self->viewDims.size() < 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=avg;
    int64_t NC=self->viewDims[0]*self->viewDims[1]; e->m=NC; e->n=self->numel()/NC; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGlobalAveragePoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return gp_ws(self,1,out,ex);}
aclnnStatus aclnnGlobalMaxPoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return gp_ws(self,0,out,ex);}
static aclnnStatus gp_run(aclOpExecutor *e, cudaStream_t s){ int64_t NC=e->m,sp=e->n; DT3(( k_globalpool<T><<<(unsigned)NC,TH,0,s>>>((const T*)e->a->data,(T*)e->out->data,NC,sp,e->dim) )); return fin(e); }
aclnnStatus aclnnGlobalAveragePool(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return gp_run(e,(cudaStream_t)s);}
aclnnStatus aclnnGlobalMaxPool(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return gp_run(e,(cudaStream_t)s);}

// Upsample backward family (fp32). gradOut, [alignCorners], gradIn.
aclnnStatus aclnnUpsampleNearest1dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,1,0,0,gradIn,ex);}
aclnnStatus aclnnUpsampleNearest1dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearest3dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,3,0,0,gradIn,ex);}
aclnnStatus aclnnUpsampleNearest3dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact1dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,1,1,0,gradIn,ex);}
aclnnStatus aclnnUpsampleNearestExact1dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,2,1,0,gradIn,ex);}
aclnnStatus aclnnUpsampleNearestExact2dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,3,1,0,gradIn,ex);}
aclnnStatus aclnnUpsampleNearestExact3dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleLinear1dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,1,2,alignCorners,gradIn,ex);}
aclnnStatus aclnnUpsampleLinear1dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}
aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex){ if(ws)*ws=0; return upb_ws(gradOut,3,2,alignCorners,gradIn,ex);}
aclnnStatus aclnnUpsampleTrilinear3dBackward(void *,uint64_t,aclOpExecutor *e,aclrtStream s){ return upb_run(e,(cudaStream_t)s);}

} // extern "C"
