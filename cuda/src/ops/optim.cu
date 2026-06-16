// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>

namespace _optim {
// Optimizer (fp32): ApplyAdamW — fused AdamW in-place update of param/m/v with decoupled weight decay.
//   m=β1·m+(1-β1)·g;  v=β2·v+(1-β2)·g²;  m̂=m/(1-β1^t);  v̂=v/(1-β2^t);
//   param = param - lr·(m̂/(√v̂+eps) + wd·param).
//   Foreach* list-fusion primitives require a real aclTensorList type and are not implemented
//   (their semantics can be covered by per-tensor calls).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__global__ void k_adamw(float *p, float *m, float *v, const float *g, int64_t n,
                        float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float gi = g[i];
    float mi = b1 * m[i] + (1.f - b1) * gi; m[i] = mi;
    float vi = b2 * v[i] + (1.f - b2) * gi * gi; v[i] = vi;
    float mh = mi / bc1, vh = vi / bc2;
    p[i] = p[i] - lr * (mh / (sqrtf(vh) + eps) + wd * p[i]);
}
} // namespace

extern "C" {

// ApplyAdamW: param=out (in-place), m=out2, v=inputs[0], grad=b; step used for bias correction.
aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
        double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !grad || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (param->dtype != ACL_FLOAT || m->dtype != ACL_FLOAT || v->dtype != ACL_FLOAT || grad->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (m->viewDims != param->viewDims || v->viewDims != param->viewDims || grad->viewDims != param->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = param; e->out2 = m; e->b = grad; e->inputs.push_back(v);
    double bc1 = 1.0 - std::pow(beta1, (double)step), bc2 = 1.0 - std::pow(beta2, (double)step);
    e->dscalars = {lr, beta1, beta2, eps, weightDecay, bc1, bc2};
    e->m = param->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdamW(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->m; auto &d = e->dscalars;
    k_adamw<<<nb(n), TH, 0, (cudaStream_t)s>>>((float *)e->out->data, (float *)e->out2->data,
        (float *)const_cast<aclTensor *>(e->inputs[0])->data, (const float *)e->b->data, n,
        (float)d[0], (float)d[1], (float)d[2], (float)d[3], (float)d[4], (float)d[5], (float)d[6]);
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}

} // extern "C"
} // namespace _optim

namespace _optim_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _optim_ext {
// Optimizer extensions (P11): Adam, Adagrad, Rmsprop, Momentum/SGD, Adamax, Adadelta, ClipGradNorm. fp32 in-place.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

__global__ void k_adam(float *p, float *m, float *v, const float *g, int64_t n, float lr, float b1, float b2, float eps, float wd, float bc1, float bc2) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi = g[i] + wd*p[i];   // classic Adam: weight decay folded into grad
    float mi = b1*m[i]+(1-b1)*gi; m[i]=mi; float vi = b2*v[i]+(1-b2)*gi*gi; v[i]=vi;
    p[i] = p[i] - lr*(mi/bc1)/(sqrtf(vi/bc2)+eps);
}
__global__ void k_adagrad(float *p, float *ss, const float *g, int64_t n, float lr, float eps, float wd) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]+wd*p[i]; float s=ss[i]+gi*gi; ss[i]=s; p[i]=p[i]-lr*gi/(sqrtf(s)+eps);
}
__global__ void k_rmsprop(float *p, float *v, const float *g, int64_t n, float lr, float alpha, float eps, float wd) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]+wd*p[i]; float vi=alpha*v[i]+(1-alpha)*gi*gi; v[i]=vi; p[i]=p[i]-lr*gi/(sqrtf(vi)+eps);
}
__global__ void k_momentum(float *p, float *buf, const float *g, int64_t n, float lr, float mu, float wd, float damp, bool nesterov) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]+wd*p[i]; float b=mu*buf[i]+(1-damp)*gi; buf[i]=b;
    float step = nesterov ? gi+mu*b : b; p[i]=p[i]-lr*step;
}
__global__ void k_adamax(float *p, float *m, float *u, const float *g, int64_t n, float lr, float b1, float b2, float eps, float bc1) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]; float mi=b1*m[i]+(1-b1)*gi; m[i]=mi; float ui=fmaxf(b2*u[i],fabsf(gi)); u[i]=ui;
    p[i]=p[i]-(lr/bc1)*mi/(ui+eps);
}
__global__ void k_adadelta(float *p, float *sq, float *ad, const float *g, int64_t n, float lr, float rho, float eps, float wd) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float gi=g[i]+wd*p[i]; float s=rho*sq[i]+(1-rho)*gi*gi; sq[i]=s;
    float delta=sqrtf(ad[i]+eps)/sqrtf(s+eps)*gi; ad[i]=rho*ad[i]+(1-rho)*delta*delta; p[i]=p[i]-lr*delta;
}
__global__ void k_norm_reduce(const float *g, float *acc, int64_t n) {
    __shared__ double red[TH]; double s=0;
    for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x) s+=(double)g[i]*g[i];
    red[threadIdx.x]=s; __syncthreads();
    for(int k=blockDim.x/2;k>0;k>>=1){if(threadIdx.x<k)red[threadIdx.x]+=red[threadIdx.x+k];__syncthreads();}
    if(threadIdx.x==0) atomicAdd(acc,(float)red[0]);
}
__global__ void k_clip_scale(float *g, const float *acc, int64_t n, float maxNorm) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float total=sqrtf(acc[0]); float sc = total>maxNorm ? maxNorm/(total+1e-6f) : 1.f; g[i]*=sc;
}
__global__ void k_write_norm(const float *acc, float *o) { *o = sqrtf(acc[0]); }
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnApplyAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
        double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!m||!v||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=m; e->inputs.push_back(v); e->b=grad; e->m=param->numel();
    double bc1=1.0-std::pow(beta1,(double)step), bc2=1.0-std::pow(beta2,(double)step);
    e->dscalars={lr,beta1,beta2,eps,weightDecay,bc1,bc2};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdam(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_adam<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(float*)const_cast<aclTensor*>(e->inputs[0])->data,(const float*)e->b->data,e->m,d[0],d[1],d[2],d[3],d[4],d[5],d[6]);
    return done(e);
}
aclnnStatus aclnnApplyAdagradGetWorkspaceSize(aclTensor *param, aclTensor *stateSum, const aclTensor *grad, double lr, double eps, double weightDecay, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!stateSum||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=stateSum; e->b=grad; e->m=param->numel(); e->dscalars={lr,eps,weightDecay};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdagrad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_adagrad<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(const float*)e->b->data,e->m,d[0],d[1],d[2]);
    return done(e);
}
aclnnStatus aclnnApplyRmspropGetWorkspaceSize(aclTensor *param, aclTensor *v, const aclTensor *grad, double lr, double alpha, double eps, double weightDecay, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!v||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=v; e->b=grad; e->m=param->numel(); e->dscalars={lr,alpha,eps,weightDecay};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyRmsprop(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_rmsprop<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(const float*)e->b->data,e->m,d[0],d[1],d[2],d[3]);
    return done(e);
}
aclnnStatus aclnnApplyMomentumGetWorkspaceSize(aclTensor *param, aclTensor *buf, const aclTensor *grad, double lr, double momentum, double weightDecay, double dampening, bool nesterov, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!buf||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=buf; e->b=grad; e->m=param->numel(); e->keepDim=nesterov; e->dscalars={lr,momentum,weightDecay,dampening};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyMomentum(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_momentum<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(const float*)e->b->data,e->m,d[0],d[1],d[2],d[3],e->keepDim);
    return done(e);
}
aclnnStatus aclnnApplyAdamaxGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *u, const aclTensor *grad, double lr, double beta1, double beta2, double eps, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!m||!u||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=m; e->inputs.push_back(u); e->b=grad; e->m=param->numel();
    double bc1=1.0-std::pow(beta1,(double)step); e->dscalars={lr,beta1,beta2,eps,bc1};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdamax(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_adamax<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(float*)const_cast<aclTensor*>(e->inputs[0])->data,(const float*)e->b->data,e->m,d[0],d[1],d[2],d[3],d[4]);
    return done(e);
}
aclnnStatus aclnnApplyAdadeltaGetWorkspaceSize(aclTensor *param, aclTensor *squareAvg, aclTensor *accDelta, const aclTensor *grad, double lr, double rho, double eps, double weightDecay, uint64_t *ws, aclOpExecutor **ex) {
    if (!param||!squareAvg||!accDelta||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=param; e->out2=squareAvg; e->inputs.push_back(accDelta); e->b=grad; e->m=param->numel(); e->dscalars={lr,rho,eps,weightDecay};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdadelta(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto &d=e->dscalars; k_adadelta<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(float*)const_cast<aclTensor*>(e->inputs[0])->data,(const float*)e->b->data,e->m,d[0],d[1],d[2],d[3]);
    return done(e);
}
// ClipGradNorm: scale grad in-place so its L2 norm <= maxNorm; totalNorm output (scalar) optional
aclnnStatus aclnnClipGradNormGetWorkspaceSize(aclTensor *grad, double maxNorm, aclTensor *totalNormOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!grad||!ex||grad->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=grad; e->out2=totalNormOut; e->m=grad->numel(); e->alpha=maxNorm;
    if(ws)*ws=sizeof(float); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnClipGradNorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->m; float *acc=(float*)ws; cudaMemsetAsync(acc,0,sizeof(float),st);
    int64_t g=(n+TH-1)/TH; if(g>256)g=256;
    k_norm_reduce<<<g,TH,0,st>>>((const float*)e->out->data,acc,n);
    if (e->out2) k_write_norm<<<1,1,0,st>>>(acc,(float*)e->out2->data);
    k_clip_scale<<<nb(n),TH,0,st>>>((float*)e->out->data,acc,n,(float)e->alpha);
    return done(e);
}

} // extern "C"
} // namespace _optim_ext

namespace _optim2_ext {
// Optimizer remainder (R7): ApplyLamb, ApplyLars (layer-wise trust ratio), FusedEmaAdam (AdamW + param EMA). fp32 in-place.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__global__ void k_sumsq(const float*x,float*acc,int64_t n){ __shared__ double r[TH]; double s=0; for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x) s+=(double)x[i]*x[i]; r[threadIdx.x]=s; __syncthreads(); for(int k=blockDim.x/2;k>0;k>>=1){if(threadIdx.x<k)r[threadIdx.x]+=r[threadIdx.x+k];__syncthreads();} if(threadIdx.x==0) atomicAdd(acc,(float)r[0]); }
// Lamb stage1: update m,v in-place; r = mhat/(sqrt(vhat)+eps) + wd*p
__global__ void k_lamb_r(float*p,float*m,float*v,const float*g,float*r,int64_t n,float b1,float b2,float eps,float wd,float bc1,float bc2){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float gi=g[i]; float mi=b1*m[i]+(1-b1)*gi; m[i]=mi; float vi=b2*v[i]+(1-b2)*gi*gi; v[i]=vi; r[i]=(mi/bc1)/(sqrtf(vi/bc2)+eps)+wd*p[i];
}
__global__ void k_lamb_apply(float*p,const float*r,const float*pn,const float*rn,int64_t n,float lr){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float pnorm=sqrtf(pn[0]),rnorm=sqrtf(rn[0]); float trust=(pnorm>0&&rnorm>0)?pnorm/rnorm:1.f; p[i]-=lr*trust*r[i]; }
__global__ void k_lars(float*p,float*buf,const float*g,const float*pn,const float*gn,int64_t n,float lr,float mu,float wd,float trustc,float eps){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float pnorm=sqrtf(pn[0]),gnorm=sqrtf(gn[0]); float llr=(pnorm>0&&gnorm>0)? trustc*pnorm/(gnorm+wd*pnorm+eps):1.f; float b=mu*buf[i]+(g[i]+wd*p[i]); buf[i]=b; p[i]-=lr*llr*b;
}
__global__ void k_adamw_ema(float*p,float*m,float*v,const float*g,float*ema,int64_t n,float lr,float b1,float b2,float eps,float wd,float bc1,float bc2,float ed){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; float gi=g[i]; float mi=b1*m[i]+(1-b1)*gi; m[i]=mi; float vi=b2*v[i]+(1-b2)*gi*gi; v[i]=vi; float pp=p[i]-lr*((mi/bc1)/(sqrtf(vi/bc2)+eps)+wd*p[i]); p[i]=pp; ema[i]=ed*ema[i]+(1-ed)*pp;
}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnApplyLambGetWorkspaceSize(aclTensor*param,aclTensor*m,aclTensor*v,const aclTensor*grad,double lr,double beta1,double beta2,double eps,double weightDecay,int64_t step,uint64_t*ws,aclOpExecutor**ex){
    if(!param||!m||!v||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->out=param; e->out2=m; e->inputs.push_back(v); e->b=grad; e->m=param->numel();
    double bc1=1-std::pow(beta1,(double)step),bc2=1-std::pow(beta2,(double)step); e->dscalars={lr,beta1,beta2,eps,weightDecay,bc1,bc2};
    if(ws)*ws=(uint64_t)e->m*4 + 2*sizeof(float); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyLamb(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ auto st=(cudaStream_t)s; int64_t n=e->m; auto&d=e->dscalars;
    float*r=(float*)ws; float*pn=r+n; float*rn=pn+1; cudaMemsetAsync(pn,0,8,st);
    k_lamb_r<<<nb(n),TH,0,st>>>((float*)e->out->data,(float*)e->out2->data,(float*)const_cast<aclTensor*>(e->inputs[0])->data,(const float*)e->b->data,r,n,d[1],d[2],d[3],d[4],d[5],d[6]);
    int g=(n+TH-1)/TH; if(g>256)g=256; k_sumsq<<<g,TH,0,st>>>((float*)e->out->data,pn,n); k_sumsq<<<g,TH,0,st>>>(r,rn,n);
    k_lamb_apply<<<nb(n),TH,0,st>>>((float*)e->out->data,r,pn,rn,n,(float)d[0]); return done(e);
}
aclnnStatus aclnnApplyLarsGetWorkspaceSize(aclTensor*param,aclTensor*buf,const aclTensor*grad,double lr,double momentum,double weightDecay,double trustCoeff,double eps,uint64_t*ws,aclOpExecutor**ex){
    if(!param||!buf||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->out=param; e->out2=buf; e->b=grad; e->m=param->numel(); e->dscalars={lr,momentum,weightDecay,trustCoeff,eps};
    if(ws)*ws=2*sizeof(float); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyLars(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ auto st=(cudaStream_t)s; int64_t n=e->m; auto&d=e->dscalars; float*pn=(float*)ws,*gn=pn+1; cudaMemsetAsync(pn,0,8,st);
    int g=(n+TH-1)/TH; if(g>256)g=256; k_sumsq<<<g,TH,0,st>>>((float*)e->out->data,pn,n); k_sumsq<<<g,TH,0,st>>>((const float*)e->b->data,gn,n);
    k_lars<<<nb(n),TH,0,st>>>((float*)e->out->data,(float*)e->out2->data,(const float*)e->b->data,pn,gn,n,(float)d[0],(float)d[1],(float)d[2],(float)d[3],(float)d[4]); return done(e);
}
aclnnStatus aclnnFusedEmaAdamGetWorkspaceSize(aclTensor*param,aclTensor*m,aclTensor*v,aclTensor*ema,const aclTensor*grad,double lr,double beta1,double beta2,double eps,double weightDecay,double emaDecay,int64_t step,uint64_t*ws,aclOpExecutor**ex){
    if(!param||!m||!v||!ema||!grad||!ex||param->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->out=param; e->out2=m; e->inputs={v,ema}; e->b=grad; e->m=param->numel();
    double bc1=1-std::pow(beta1,(double)step),bc2=1-std::pow(beta2,(double)step); e->dscalars={lr,beta1,beta2,eps,weightDecay,bc1,bc2,emaDecay};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFusedEmaAdam(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ auto&d=e->dscalars;
    k_adamw_ema<<<nb(e->m),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(float*)e->out2->data,(float*)const_cast<aclTensor*>(e->inputs[0])->data,(const float*)e->b->data,(float*)const_cast<aclTensor*>(e->inputs[1])->data,e->m,d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]); return done(e);
}

} // extern "C"
} // namespace _optim2_ext

} // namespace _optim_ext

