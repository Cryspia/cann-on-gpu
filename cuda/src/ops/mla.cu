// Multi-latent attention (P15): MLA forward. Compressed latent KV is up-projected per head, then standard MHA.
// q[B,Nh,S,D], cKV[B,S,Lc] (compressed), wUK/wUV[Lc, Nh*D] -> out[B,Nh,S,D]. fp32. Functional reference (DeepSeek-MLA core math).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
constexpr int TH = 128;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
// up-project latent into K and V: out[b,h,s,d] = sum_l cKV[b,s,l]*W[l, h*D+d]
__global__ void k_mla_proj(const float *cKV, const float *wUK, const float *wUV, float *K, float *V,
        int64_t B, int64_t Nh, int64_t S, int64_t D, int64_t Lc) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=B*Nh*S*D) return;
    int64_t d=i%D, s=(i/D)%S, h=(i/(D*S))%Nh, b=i/(D*S*Nh);
    const float *c=cKV+(b*S+s)*Lc; double ak=0,av=0; int64_t wc=h*D+d, WN=Nh*D;
    for(int64_t l=0;l<Lc;l++){ float cl=c[l]; ak+=(double)cl*wUK[l*WN+wc]; av+=(double)cl*wUV[l*WN+wc]; }
    K[i]=(float)ak; V[i]=(float)av;
}
// MHA: one thread per (b,h,si); online softmax over sj (causal optional), accumulate V row [D]
__global__ void k_mla_attn(const float *q, const float *K, const float *V, float *o,
        int64_t B, int64_t Nh, int64_t S, int64_t D, float scale, bool causal) {
    int64_t bhi=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(bhi>=B*Nh*S) return;
    int64_t si=bhi%S, h=(bhi/S)%Nh, b=bhi/(S*Nh);
    const float *qr=q+((b*Nh+h)*S+si)*D; const float *Kb=K+((b*Nh+h)*S)*D; const float *Vb=V+((b*Nh+h)*S)*D;
    float *orow=o+((b*Nh+h)*S+si)*D;
    int64_t lim = causal ? si+1 : S; double m=-1e30,l=0; double acc[256]; for(int64_t d=0;d<D;d++) acc[d]=0;
    for(int64_t sj=0;sj<lim;sj++){ double sc=0; const float *kr=Kb+sj*D; for(int64_t d=0;d<D;d++) sc+=(double)qr[d]*kr[d]; sc*=scale;
        double mn=fmax(m,sc), corr=exp(m-mn), p=exp(sc-mn); l=l*corr+p; const float *vr=Vb+sj*D; for(int64_t d=0;d<D;d++) acc[d]=acc[d]*corr+p*vr[d]; m=mn; }
    for(int64_t d=0;d<D;d++) orow[d]=(float)(acc[d]/(l+1e-20));
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnMultiLatentAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *cKV, const aclTensor *wUK, const aclTensor *wUV,
        double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!q||!cKV||!wUK||!wUV||!out||!ex||q->dtype!=ACL_FLOAT||q->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=q; e->b=cKV; e->c=wUK; e->mask=wUV; e->out=out; e->alpha=scaleValue; e->causal=causal;
    e->ab=q->viewDims[0]; e->an=q->viewDims[1]; e->asq=q->viewDims[2]; e->ad=q->viewDims[3]; e->askv=cKV->viewDims[2];   // Lc
    if(ws)*ws=(uint64_t)2*e->ab*e->an*e->asq*e->ad*sizeof(float);   // K + V scratch
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMultiLatentAttention(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t B=e->ab,Nh=e->an,S=e->asq,D=e->ad,Lc=e->askv;
    float *K=(float*)ws, *V=K + B*Nh*S*D;
    k_mla_proj<<<nb(B*Nh*S*D),TH,0,st>>>((const float*)e->b->data,(const float*)e->c->data,(const float*)e->mask->data,K,V,B,Nh,S,D,Lc);
    k_mla_attn<<<nb(B*Nh*S),TH,0,st>>>((const float*)e->a->data,K,V,(float*)e->out->data,B,Nh,S,D,(float)e->alpha,e->causal);
    return done(e);
}

} // extern "C"
