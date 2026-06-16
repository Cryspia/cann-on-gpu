// Linear attention / SSM (P18): CausalConv1d (depthwise), Mamba SelectiveScan (S6), GatedDeltaRule (GatedDeltaNet).
// These unblock Qwen3.5-class hybrid models. fp32, sequential recurrence (functional reference, not chunked-parallel).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
constexpr int TH = 128;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__device__ inline float silu(float x){ return x/(1.f+expf(-x)); }

// Causal depthwise Conv1d: x[B,C,L], weight[C,K], bias[C] -> out[B,C,L]; causal left-pad (K-1). act:0=none 1=silu
__global__ void k_causal_conv1d(const float *x, const float *w, const float *bias, float *o, int64_t B, int64_t C, int64_t L, int64_t K, int act) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=B*C*L) return;
    int64_t t=i%L, c=(i/L)%C, b=i/(L*C); const float *xp=x+(b*C+c)*L; const float *wp=w+c*K;
    double acc=bias?bias[c]:0.0;
    for(int64_t k=0;k<K;k++){ int64_t ti=t-(K-1)+k; if(ti>=0) acc+=(double)wp[k]*xp[ti]; }
    o[i] = act==1 ? silu((float)acc) : (float)acc;
}
// Mamba selective scan (S6). One thread per (b,d). State h[N].
// u[B,L,D], delta[B,L,D], A[D,N], B_[B,L,N], C_[B,L,N], Dskip[D] -> y[B,L,D]
__global__ void k_selective_scan(const float *u, const float *delta, const float *A, const float *Bm, const float *Cm, const float *Dskip,
        float *y, int64_t B, int64_t L, int64_t D, int64_t N) {
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*D) return; int64_t b=idx/D, d=idx%D;
    float h[64]; for(int64_t n=0;n<N;n++) h[n]=0.f;
    for(int64_t t=0;t<L;t++){ float dt=delta[(b*L+t)*D+d], ut=u[(b*L+t)*D+d]; double yt=0;
        const float *Bt=Bm+(b*L+t)*N, *Ct=Cm+(b*L+t)*N;
        for(int64_t n=0;n<N;n++){ float dA=expf(dt*A[d*N+n]); float dBu=dt*Bt[n]*ut; h[n]=dA*h[n]+dBu; yt+=(double)Ct[n]*h[n]; }
        y[(b*L+t)*D+d]=(float)yt + (Dskip?Dskip[d]*ut:0.f);
    }
}
// Gated delta rule (GatedDeltaNet). One thread per (b,head). State S[Dk,Dv] in workspace.
// q,k[B,Hd,L,Dk], v[B,Hd,L,Dv], beta[B,Hd,L], g(decay)[B,Hd,L] -> y[B,Hd,L,Dv]
__global__ void k_gated_delta(const float *q, const float *k, const float *v, const float *beta, const float *g,
        float *y, float *Sbuf, int64_t B, int64_t Hd, int64_t L, int64_t Dk, int64_t Dv) {
    int64_t bh=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(bh>=B*Hd) return;
    float *S=Sbuf + bh*Dk*Dv; for(int64_t i=0;i<Dk*Dv;i++) S[i]=0.f;
    for(int64_t t=0;t<L;t++){ int64_t base=(bh*L+t); const float *kt=k+base*Dk,*qt=q+base*Dk,*vt=v+base*Dv;
        float gt=g[base], bt=beta[base];
        // S *= gt
        for(int64_t i=0;i<Dk*Dv;i++) S[i]*=gt;
        // kv_j = v_j - sum_i S[i,j]*k_i
        for(int64_t j=0;j<Dv;j++){ double sk=0; for(int64_t i=0;i<Dk;i++) sk+=(double)S[i*Dv+j]*kt[i]; double kv=(double)vt[j]-sk;
            // S[i,j] += bt*k_i*kv
            for(int64_t i=0;i<Dk;i++) S[i*Dv+j]+=bt*kt[i]*(float)kv; }
        // y_j = sum_i q_i*S[i,j]
        for(int64_t j=0;j<Dv;j++){ double yt=0; for(int64_t i=0;i<Dk;i++) yt+=(double)qt[i]*S[i*Dv+j]; y[base*Dv+j]=(float)yt; }
    }
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// CausalConv1d: x[B,C,L], weight[C,K], bias[C] (nullable), activation (0/1) -> out[B,C,L]
aclnnStatus aclnnCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x||!weight||!out||!ex||x->dtype!=ACL_FLOAT||x->viewDims.size()!=3||weight->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=weight; e->c=bias; e->out=out; e->dim=activation;
    e->ab=x->viewDims[0]; e->an=x->viewDims[1]; e->asq=x->viewDims[2]; e->ad=weight->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCausalConv1d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t B=e->ab,C=e->an,L=e->asq,K=e->ad;
    k_causal_conv1d<<<nb(B*C*L),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,B,C,L,K,(int)e->dim);
    return done(e);
}
// SelectiveScan (Mamba S6)
aclnnStatus aclnnSelectiveScanGetWorkspaceSize(const aclTensor *u, const aclTensor *delta, const aclTensor *A, const aclTensor *Bm, const aclTensor *Cm, const aclTensor *Dskip, aclTensor *y, uint64_t *ws, aclOpExecutor **ex) {
    if (!u||!delta||!A||!Bm||!Cm||!y||!ex||u->dtype!=ACL_FLOAT||u->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    if (A->viewDims.size()!=2 || A->viewDims[1] > 64) return ACLNN_ERR_PARAM_INVALID;   // state N capped (register array)
    auto *e=new aclOpExecutor(); e->a=u; e->b=delta; e->c=A; e->inputs={Bm,Cm,Dskip}; e->out=y;
    e->ab=u->viewDims[0]; e->asq=u->viewDims[1]; e->an=u->viewDims[2]; e->ad=A->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSelectiveScan(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t B=e->ab,L=e->asq,D=e->an,N=e->ad;
    const float *Dskip=e->inputs[2]?(const float*)e->inputs[2]->data:nullptr;
    k_selective_scan<<<nb(B*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,
        (const float*)e->inputs[0]->data,(const float*)e->inputs[1]->data,Dskip,(float*)e->out->data,B,L,D,N);
    return done(e);
}
// GatedDeltaRule (GatedDeltaNet)
aclnnStatus aclnnGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *ws, aclOpExecutor **ex) {
    if (!q||!k||!v||!beta||!g||!y||!ex||q->dtype!=ACL_FLOAT||q->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=q; e->b=k; e->c=v; e->inputs={beta,g}; e->out=y;
    e->ab=q->viewDims[0]; e->an=q->viewDims[1]; e->asq=q->viewDims[2]; e->ad=q->viewDims[3]; e->askv=v->viewDims[3];
    if(ws)*ws=(uint64_t)e->ab*e->an*e->ad*e->askv*sizeof(float);   // S[Dk,Dv] per (b,head)
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGatedDeltaRule(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t B=e->ab,Hd=e->an,L=e->asq,Dk=e->ad,Dv=e->askv;
    k_gated_delta<<<nb(B*Hd),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,
        (const float*)e->inputs[0]->data,(const float*)e->inputs[1]->data,(float*)e->out->data,(float*)ws,B,Hd,L,Dk,Dv);
    return done(e);
}

} // extern "C"
