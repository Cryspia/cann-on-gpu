// Transformer / LLM fused completion (P15): ApplyRotaryPosEmbGrad (RoPE backward), FusedInferAttentionScore (unified inference attention).
// RoPE is an orthogonal rotation, so its gradient is the same rotation with sin negated (J^T == rotate by -theta).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
template <typename T>
__global__ void k_rope_grad(const T *go, const T *cos, const T *sin, T *gi, int64_t S, int64_t D, int64_t total, int mode) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=total) return;
    int64_t d=i%D, s=(i/D)%S, base=i-d;
    float c=(float)cos[s*D+d], si=-(float)sin[s*D+d], xv=(float)go[i], rh;   // negated sin == transpose of the rotation
    if (mode==0){ int64_t half=D/2; rh=(d<half)? -(float)go[base+d+half] : (float)go[base+d-half]; }
    else        { rh=(d&1)? (float)go[base+d-1] : -(float)go[base+d+1]; }
    gi[i]=(T)(xv*c + rh*si);
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// ApplyRotaryPosEmbGrad: gradQ[B,Nq,S,D], gradK[B,Nk,S,D], cos/sin[S,D] -> gradQOut, gradKOut (un-rotate)
aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(const aclTensor *gradQ, const aclTensor *gradK, const aclTensor *cos, const aclTensor *sin,
        int64_t mode, aclTensor *gradQOut, aclTensor *gradKOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradQ||!cos||!sin||!gradQOut||!ex||gradQ->dtype!=ACL_FLOAT||gradQ->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    if (mode<0||mode>1) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradQ; e->b=gradK; e->c=cos; e->mask=sin; e->out=gradQOut; e->out2=gradKOut; e->an=mode;
    e->asq=gradQ->viewDims[2]; e->ad=gradQ->viewDims[3];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyRotaryPosEmbGrad(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    auto s=(cudaStream_t)stream; int mode=(int)e->an; int64_t S=e->asq,D=e->ad;
    const float *cos=(const float*)e->c->data,*sin=(const float*)e->mask->data;
    int64_t qn=e->a->numel(); k_rope_grad<float><<<nb(qn),TH,0,s>>>((const float*)e->a->data,cos,sin,(float*)e->out->data,S,D,qn,mode);
    if (e->b && e->out2){ int64_t kn=e->b->numel(); k_rope_grad<float><<<nb(kn),TH,0,s>>>((const float*)e->b->data,cos,sin,(float*)e->out2->data,S,D,kn,mode); }
    return done(e);
}

// FusedInferAttentionScore: unified inference attention (prompt + incremental). Routes to the existing prompt flash attention.
aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnPromptFlashAttentionGetWorkspaceSize(q, k, v, attenMask, scaleValue, headNum, causal, out, ws, ex);
}
aclnnStatus aclnnFusedInferAttentionScore(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    return aclnnPromptFlashAttention(ws, wsz, e, s);
}

} // extern "C"
