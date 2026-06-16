// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_mc2.h"
#include "hccl/hccl.h"
#include <algorithm>
#include <map>

namespace _moe {
// Transformer / LLM fused — MoE routing (P15): MoeGatingTopKSoftmax, MoeComputeExpertTokens, MoeTokenPermute, MoeTokenUnpermute.
// fp32. Together these implement the MoE dispatch/combine path: softmax-topk gate -> permute tokens by expert -> (expert FFN) -> weighted unpermute.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// Softmax over E then top-K (renormalized) per token. logits[T,E] -> weights[T,K], indices[T,K] (int32)
__global__ void k_gating_topk(const float *logits, float *weights, int32_t *indices, int64_t T, int64_t E, int64_t K) {
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=T) return; const float *lg=logits+t*E;
    double mx=-1e30; for(int64_t e=0;e<E;e++) mx=fmax(mx,(double)lg[e]); double se=0; for(int64_t e=0;e<E;e++) se+=exp((double)lg[e]-mx);
    // top-K by selection on softmax prob (== selection on logits); renormalize selected
    bool used[256]; for(int64_t e=0;e<E;e++) used[e]=false; double wsum=0;
    for(int64_t kk=0;kk<K;kk++){ int64_t best=-1; double bv=-1e30; for(int64_t e=0;e<E;e++){ if(used[e]) continue; if((double)lg[e]>bv){bv=lg[e];best=e;} }
        used[best]=true; double p=exp((double)lg[best]-mx)/se; indices[t*K+kk]=(int32_t)best; weights[t*K+kk]=(float)p; wsum+=p; }
    for(int64_t kk=0;kk<K;kk++) weights[t*K+kk]=(float)(weights[t*K+kk]/wsum);   // renormalize
}
// counts per expert (int32 indices flattened length M) -> tokensPerExpert[E] (int64) and cumulative offsets
__global__ void k_count_experts(const int32_t *idx, int64_t *counts, int64_t M, int64_t E) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return; int64_t e=idx[i]; if(e>=0&&e<E) atomicAdd((unsigned long long*)&counts[e],1ULL);
}
__global__ void k_excl_scan(const int64_t *counts, int64_t *offs, int64_t E) { int64_t a=0; for(int64_t e=0;e<E;e++){ offs[e]=a; a+=counts[e]; } }
// permute scatter: token t with expert e[t] -> slot = atomicAdd(running[e]); permX[slot]=x[t]; srcIdx[slot]=t
__global__ void k_permute_scatter(const float *x, const int32_t *expert, float *permX, int64_t *srcIdx, int64_t *running, int64_t T, int64_t H) {
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=T) return; int64_t e=expert[t];
    int64_t slot=atomicAdd((unsigned long long*)&running[e],1ULL);
    srcIdx[slot]=t; const float *xp=x+t*H; float *op=permX+slot*H; for(int64_t h=0;h<H;h++) op[h]=xp[h];
}
// unpermute: out[srcIdx[p]] = permY[p] * weight[p] (weight optional)
__global__ void k_unpermute_scatter(const float *permY, const int64_t *srcIdx, const float *weight, float *out, int64_t P, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return; int64_t t=srcIdx[p]; float wv=weight?weight[p]:1.f;
    const float *yp=permY+p*H; for(int64_t h=0;h<H;h++) atomicAdd(&out[t*H+h], yp[h]*wv);
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// MoeGatingTopKSoftmax: logits[T,E] -> weights[T,K], indices[T,K]
aclnnStatus aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) {
    if (!logits||!weights||!indices||!ex||logits->dtype!=ACL_FLOAT||indices->dtype!=ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)logits->viewDims.size(); int64_t E=logits->viewDims[rank-1]; if(E>256) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=logits; e->out=weights; e->out2=indices; e->n=E; e->m=logits->numel()/E; e->k=k;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeGatingTopKSoftmax(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_gating_topk<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,(int32_t*)e->out2->data,e->m,e->n,e->k); return done(e);
}
// MoeComputeExpertTokens: indices flattened -> tokensPerExpert[E] (int64) + offsets[E] (int64, exclusive prefix) in out2
aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(const aclTensor *indices, int64_t numExperts, aclTensor *tokensPerExpert, aclTensor *offsets, uint64_t *ws, aclOpExecutor **ex) {
    if (!indices||!tokensPerExpert||!ex||indices->dtype!=ACL_INT32||tokensPerExpert->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=indices; e->out=tokensPerExpert; e->out2=offsets; e->m=indices->numel(); e->n=numExperts;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeComputeExpertTokens(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t M=e->m,E=e->n;
    cudaMemsetAsync(e->out->data,0,(size_t)E*sizeof(int64_t),st);
    k_count_experts<<<nb(M),TH,0,st>>>((const int32_t*)e->a->data,(int64_t*)e->out->data,M,E);
    if (e->out2) k_excl_scan<<<1,1,0,st>>>((const int64_t*)e->out->data,(int64_t*)e->out2->data,E);
    return done(e);
}
// MoeTokenPermute: x[T,H], expertId[T] (int32), numExperts -> permX[T,H] grouped by expert, srcIdx[T] (int64: permuted->original)
aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex) {
    if (!x||!expertId||!permX||!srcIdx||!ex||x->dtype!=ACL_FLOAT||expertId->dtype!=ACL_INT32||srcIdx->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=expertId; e->out=permX; e->out2=srcIdx; e->m=x->viewDims[0]; e->n=x->numel()/e->m; e->k=numExperts;
    if(ws)*ws=(uint64_t)2*numExperts*sizeof(int64_t);   // counts[E] + running[E]
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeTokenPermute(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->m,H=e->n,E=e->k; int64_t *counts=(int64_t*)ws,*running=counts+E;
    cudaMemsetAsync(counts,0,(size_t)E*sizeof(int64_t),st);
    k_count_experts<<<nb(T),TH,0,st>>>((const int32_t*)e->b->data,counts,T,E);
    k_excl_scan<<<1,1,0,st>>>(counts,running,E);   // running = exclusive offsets (start slots)
    k_permute_scatter<<<nb(T),TH,0,st>>>((const float*)e->a->data,(const int32_t*)e->b->data,(float*)e->out->data,(int64_t*)e->out2->data,running,T,H);
    return done(e);
}
// MoeTokenUnpermute: permY[P,H], srcIdx[P] (int64), weight[P] (nullable) -> out[T,H] (pre-zeroed, scatter-add)
aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!permY||!srcIdx||!out||!ex||permY->dtype!=ACL_FLOAT||srcIdx->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=permY; e->b=srcIdx; e->c=weight; e->out=out; e->m=permY->viewDims[0]; e->n=permY->numel()/e->m;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeTokenUnpermute(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t P=e->m,H=e->n;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_unpermute_scatter<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,P,H);
    return done(e);
}

} // extern "C"
} // namespace _moe

namespace _moe2_ext {
// MoE routing extensions: InitRouting (expand+group tokens by top-K expert) and FinalizeRouting
// (weighted combine of expert outputs), plus gating / GroupedMatmul version-variant forwards.
// fp32. InitRouting is the inverse-mapping partner of FinalizeRouting (verified round-trip).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }

// histogram of expert ids over M flat entries
__global__ void k_hist(const int32_t *idx, int64_t *counts, int64_t M, int64_t E) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return; int64_t e=idx[i];
    if(e>=0&&e<E) atomicAdd((unsigned long long*)&counts[e],1ULL);
}
__global__ void k_scan(const int64_t *counts, int64_t *running, int64_t E) { int64_t a=0; for(int64_t e=0;e<E;e++){ running[e]=a; a+=counts[e]; } }
// InitRouting scatter: flat entry i=(t*K+k), expert e -> slot; expandedX[slot]=x[t]; rowIdx[slot]=i; expertIdx[slot]=e
__global__ void k_init_scatter(const float *x, const int32_t *expert, float *expX, int32_t *rowIdx, int32_t *expIdxOut,
                               int64_t *running, int64_t M, int64_t K, int64_t H) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=M) return; int64_t e=expert[i];
    int64_t slot=atomicAdd((unsigned long long*)&running[e],1ULL);
    rowIdx[slot]=(int32_t)i; expIdxOut[slot]=(int32_t)e;
    int64_t t=i/K; const float *xp=x+t*H; float *op=expX+slot*H; for(int64_t h=0;h<H;h++) op[h]=xp[h];
}
// FinalizeRouting: out[t] += scales[t,k]·expandedY[p] where flat=rowIdx[p], t=flat/K, k=flat%K
__global__ void k_finalize(const float *expY, const int32_t *rowIdx, const float *scales, float *out, int64_t P, int64_t K, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return;
    int64_t flat=rowIdx[p], t=flat/K, k=flat%K; float w = scales ? scales[t*K+k] : 1.f;
    const float *yp=expY+p*H; for(int64_t h=0;h<H;h++) atomicAdd(&out[t*H+h], w*yp[h]);
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// MoeInitRouting: x[T,H], expertIdx[T,K] (int32) -> expandedX[T*K,H], expandedRowIdx[T*K] (int32), expandedExpertIdx[T*K] (int32),
// rows grouped (sorted) by expert. expandedRowIdx[p] = original flat index t*K+k of expanded row p.
aclnnStatus aclnnMoeInitRoutingGetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts,
        aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex) {
    if (!x||!expertIdx||!expandedX||!expandedRowIdx||!expandedExpertIdx||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype!=ACL_FLOAT||expertIdx->dtype!=ACL_INT32||expandedRowIdx->dtype!=ACL_INT32||expandedExpertIdx->dtype!=ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size()!=2||expertIdx->viewDims.size()!=2||numExperts<=0) return ACLNN_ERR_PARAM_INVALID;
    int64_t T=x->viewDims[0], H=x->viewDims[1], K=expertIdx->viewDims[1];
    if (expertIdx->viewDims[0]!=T) return ACLNN_ERR_PARAM_INVALID;
    if (expandedX->viewDims!=std::vector<int64_t>{T*K,H}) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=x; e->b=expertIdx; e->out=expandedX; e->out2=expandedRowIdx; e->mask=expandedExpertIdx;
    e->m=T; e->n=H; e->k=K; e->reduceCount=numExperts;
    if(ws)*ws=(uint64_t)2*numExperts*sizeof(int64_t); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeInitRouting(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->m,H=e->n,K=e->k,E=e->reduceCount,M=T*K;
    int64_t *counts=(int64_t*)ws,*running=counts+E;
    cudaMemsetAsync(counts,0,(size_t)E*sizeof(int64_t),st);
    k_hist<<<nb(M),TH,0,st>>>((const int32_t*)e->b->data,counts,M,E);
    k_scan<<<1,1,0,st>>>(counts,running,E);
    k_init_scatter<<<nb(M),TH,0,st>>>((const float*)e->a->data,(const int32_t*)e->b->data,(float*)e->out->data,
        (int32_t*)e->out2->data,(int32_t*)const_cast<aclTensor*>(e->mask)->data,running,M,K,H);
    return done(e);
}

// MoeFinalizeRouting: expandedY[T*K,H], expandedRowIdx[T*K] (int32, from InitRouting), scales[T,K] -> out[T,H] (weighted combine).
aclnnStatus aclnnMoeFinalizeRoutingGetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx,
        const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!expandedY||!expandedRowIdx||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (expandedY->dtype!=ACL_FLOAT||expandedRowIdx->dtype!=ACL_INT32||out->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (scales&&scales->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (expandedY->viewDims.size()!=2||out->viewDims.size()!=2||k<=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=expandedY; e->b=expandedRowIdx; e->c=scales; e->out=out;
    e->m=out->viewDims[0]; e->n=out->viewDims[1]; e->k=k;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeFinalizeRouting(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->m,H=e->n,K=e->k,P=e->a->viewDims[0];
    cudaMemsetAsync(e->out->data,0,(size_t)T*H*sizeof(float),st);
    k_finalize<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const int32_t*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,P,K,H);
    return done(e);
}

// Version-variant forwards (same simplified contract as the base op).
#define INIT_ROUTING_VARIANT(VER)                                                                                       \
aclnnStatus aclnnMoeInitRouting##VER##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, \
        aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnMoeInitRoutingGetWorkspaceSize(x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx, ws, ex); \
}                                                                                                                       \
aclnnStatus aclnnMoeInitRouting##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnMoeInitRouting(ws, wsz, e, s); }
INIT_ROUTING_VARIANT(V2)
INIT_ROUTING_VARIANT(V3)

#define FINAL_ROUTING_VARIANT(VER)                                                                                      \
aclnnStatus aclnnMoeFinalizeRouting##VER##GetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, \
        const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {                         \
    return aclnnMoeFinalizeRoutingGetWorkspaceSize(expandedY, expandedRowIdx, scales, k, out, ws, ex);                  \
}                                                                                                                       \
aclnnStatus aclnnMoeFinalizeRouting##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnMoeFinalizeRouting(ws, wsz, e, s); }
FINAL_ROUTING_VARIANT(V2)
FINAL_ROUTING_VARIANT(V3)

// Gating variants → shared softmax-topk gate.
#define GATING_VARIANT(NAME)                                                                                            \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex) { \
    return aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(logits, k, weights, indices, ws, ex);                             \
}                                                                                                                       \
aclnnStatus NAME(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnMoeGatingTopKSoftmax(ws, wsz, e, s); }
GATING_VARIANT(aclnnMoeGatingTopKSoftmaxV2)
GATING_VARIANT(aclnnMoeGatingTopK)
GATING_VARIANT(aclnnMoeFusedTopk)

// GroupedMatmul version variants → shared grouped matmul.
#define GMM_VARIANT(VER)                                                                                                \
aclnnStatus aclnnGroupedMatmul##VER##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, \
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {                                                            \
    return aclnnGroupedMatmulGetWorkspaceSize(x, weight, groupList, out, ws, ex);                                       \
}                                                                                                                       \
aclnnStatus aclnnGroupedMatmul##VER(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnGroupedMatmul(ws, wsz, e, s); }
GMM_VARIANT(V2)
GMM_VARIANT(V3)
GMM_VARIANT(V4)
GMM_VARIANT(V5)

} // extern "C"

// ---- MoE routing gradients (inverse mappings of the forward ops above) ----
namespace {
// InitRouting backward: gradX[t] += gradExpandedX[p], t = rowIdx[p]/K
__global__ void k_init_grad(const float *gExpX, const int32_t *rowIdx, float *gX, int64_t P, int64_t K, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return; int64_t t=rowIdx[p]/K;
    const float *gp=gExpX+p*H; for(int64_t h=0;h<H;h++) atomicAdd(&gX[t*H+h], gp[h]);
}
// FinalizeRouting backward: gradExpY[p]=scales[t,k]*gradOut[t]; gradScales[t,k]=Σ_h gradOut[t,h]*expY[p,h]
__global__ void k_final_grad(const float *gOut, const float *expY, const int32_t *rowIdx, const float *scales,
                             float *gExpY, float *gScales, int64_t P, int64_t K, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return;
    int64_t flat=rowIdx[p], t=flat/K, k=flat%K; float w=scales?scales[t*K+k]:1.f;
    const float *go=gOut+t*H,*yp=expY+p*H; float *ge=gExpY+p*H; float dot=0;
    for(int64_t h=0;h<H;h++){ ge[h]=w*go[h]; dot+=go[h]*yp[h]; }
    if(gScales) gScales[t*K+k]=dot;
}
// TokenPermute backward: gradX[srcIdx[p]] += gradPermX[p]
__global__ void k_permute_grad(const float *gPermX, const int64_t *srcIdx, float *gX, int64_t P, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return; int64_t t=srcIdx[p];
    const float *gp=gPermX+p*H; for(int64_t h=0;h<H;h++) atomicAdd(&gX[t*H+h], gp[h]);
}
// TokenUnpermute backward: gradPermY[p]=gradOut[srcIdx[p]]*w[p]; gradWeight[p]=Σ_h gradOut[srcIdx[p],h]*permY[p,h]
__global__ void k_unpermute_grad(const float *gOut, const float *permY, const int64_t *srcIdx, const float *weight,
                                 float *gPermY, float *gWeight, int64_t P, int64_t H) {
    int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(p>=P) return; int64_t t=srcIdx[p]; float w=weight?weight[p]:1.f;
    const float *go=gOut+t*H,*yp=permY+p*H; float *gy=gPermY+p*H; float dot=0;
    for(int64_t h=0;h<H;h++){ gy[h]=go[h]*w; dot+=go[h]*yp[h]; }
    if(gWeight) gWeight[p]=dot;
}
} // namespace

extern "C" {

// MoeInitRoutingV2Grad: gradExpandedX[T*K,H], expandedRowIdx[T*K] (int32), K -> gradX[T,H]
aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(const aclTensor *gradExpandedX, const aclTensor *expandedRowIdx,
        int64_t k, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradExpandedX||!expandedRowIdx||!gradX||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradExpandedX->dtype!=ACL_FLOAT||expandedRowIdx->dtype!=ACL_INT32||gradX->dtype!=ACL_FLOAT||k<=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradExpandedX; e->b=expandedRowIdx; e->out=gradX; e->k=k;
    e->m=gradX->viewDims[0]; e->n=gradX->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeInitRoutingV2Grad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->m,H=e->n,K=e->k,P=e->a->viewDims[0];
    cudaMemsetAsync(e->out->data,0,(size_t)T*H*sizeof(float),st);
    k_init_grad<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const int32_t*)e->b->data,(float*)e->out->data,P,K,H);
    return done(e);
}

// MoeFinalizeRoutingV2Grad: gradOut[T,H], expandedY[T*K,H], expandedRowIdx[T*K], scales[T,K], K -> gradExpandedY[T*K,H], gradScales[T,K]
aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *expandedY,
        const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *gradExpandedY, aclTensor *gradScales,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!expandedY||!expandedRowIdx||!gradExpandedY||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradOut->dtype!=ACL_FLOAT||expandedY->dtype!=ACL_FLOAT||expandedRowIdx->dtype!=ACL_INT32||k<=0) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=expandedY; e->c=expandedRowIdx; e->mask=scales;
    e->out=gradExpandedY; e->out2=gradScales; e->m=gradOut->viewDims[0]; e->n=gradOut->viewDims[1]; e->k=k;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeFinalizeRoutingV2Grad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t H=e->n,K=e->k,P=e->b->viewDims[0];
    k_final_grad<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const int32_t*)e->c->data,
        e->mask?(const float*)e->mask->data:nullptr,(float*)e->out->data,e->out2?(float*)e->out2->data:nullptr,P,K,H);
    return done(e);
}

// MoeTokenPermuteGrad: gradPermX[P,H], srcIdx[P] (int64) -> gradX[T,H]
aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradPermX||!srcIdx||!gradX||!ex||gradPermX->dtype!=ACL_FLOAT||srcIdx->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradPermX; e->b=srcIdx; e->out=gradX; e->m=gradX->viewDims[0]; e->n=gradX->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeTokenPermuteGrad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->m,H=e->n,P=e->a->viewDims[0];
    cudaMemsetAsync(e->out->data,0,(size_t)T*H*sizeof(float),st);
    k_permute_grad<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,(float*)e->out->data,P,H);
    return done(e);
}

// MoeTokenUnpermuteGrad: gradOut[T,H], permY[P,H], srcIdx[P] (int64), weight[P] (nullable) -> gradPermY[P,H], gradWeight[P] (nullable)
aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx,
        const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!permY||!srcIdx||!gradPermY||!ex||gradOut->dtype!=ACL_FLOAT||permY->dtype!=ACL_FLOAT||srcIdx->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOut; e->b=permY; e->c=srcIdx; e->mask=weight; e->out=gradPermY; e->out2=gradWeight;
    e->m=permY->viewDims[0]; e->n=permY->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMoeTokenUnpermuteGrad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t P=e->m,H=e->n;
    k_unpermute_grad<<<nb(P),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const int64_t*)e->c->data,
        e->mask?(const float*)e->mask->data:nullptr,(float*)e->out->data,e->out2?(float*)e->out2->data:nullptr,P,H);
    return done(e);
}

} // extern "C"
} // namespace _moe2_ext
#undef INIT_ROUTING_VARIANT
#undef FINAL_ROUTING_VARIANT
#undef GATING_VARIANT
#undef GMM_VARIANT

namespace _moe3_ext {
// MoE expert-parallel (EP) / routing-map permute-unpermute + distribute-combine-rmsnorm + init-routing-quant.
// At nranks=1 the EP token exchange is the local permute, so WithEp/WithRoutingMap variants forward to the
// base MoeTokenPermute/Unpermute(+Grad); distribute setup/teardown are bookkeeping no-ops; combine+addrmsnorm
// composes the single-rank combine (identity) with AddRmsNorm. Multi-rank EP needs the HCCL 2-node path.

namespace { inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
inline HcclDataType moe_dt(aclDataType d){ return d==ACL_FLOAT?HCCL_DATA_TYPE_FP32:d==ACL_FLOAT16?HCCL_DATA_TYPE_FP16:HCCL_DATA_TYPE_BFP16; } }

extern "C" {

// ---- WithEp / WithRoutingMap permute (forward to base MoeTokenPermute) ----
#define PERM_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex){ return aclnnMoeTokenPermuteGetWorkspaceSize(x, expertId, numExperts, permX, srcIdx, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeTokenPermute(w,wz,e,s); }
PERM_FWD(aclnnMoeTokenPermuteWithEp)
PERM_FWD(aclnnMoeTokenPermuteWithRoutingMap)
#define PERMG_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex){ return aclnnMoeTokenPermuteGradGetWorkspaceSize(gradPermX, srcIdx, gradX, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeTokenPermuteGrad(w,wz,e,s); }
PERMG_FWD(aclnnMoeTokenPermuteWithEpGrad)
PERMG_FWD(aclnnMoeTokenPermuteWithRoutingMapGrad)
#define UNPERM_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ return aclnnMoeTokenUnpermuteGetWorkspaceSize(permY, srcIdx, weight, out, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeTokenUnpermute(w,wz,e,s); }
UNPERM_FWD(aclnnMoeTokenUnpermuteWithEp)
UNPERM_FWD(aclnnMoeTokenUnpermuteWithRoutingMap)
#define UNPERMG_FWD(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex){ return aclnnMoeTokenUnpermuteGradGetWorkspaceSize(gradOut, permY, srcIdx, weight, gradPermY, gradWeight, ws, ex); } \
aclnnStatus NAME(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeTokenUnpermuteGrad(w,wz,e,s); }
UNPERMG_FWD(aclnnMoeTokenUnpermuteWithEpGrad)
UNPERMG_FWD(aclnnMoeTokenUnpermuteWithRoutingMapGrad)

// ---- MoeUpdateExpert: copy/refresh expert assignment (identity at single rank) ----
aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(const aclTensor *expertIds, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!expertIds||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=expertIds; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMoeUpdateExpert(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->a->numel()*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }

// ---- MoeInitRoutingQuant(+V2): routing scatter (quant of expandedX = recorded limitation, omitted) ----
aclnnStatus aclnnMoeInitRoutingQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex){
    return aclnnMoeInitRoutingGetWorkspaceSize(x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx, ws, ex);
}
aclnnStatus aclnnMoeInitRoutingQuant(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeInitRouting(w,wz,e,s); }
aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex){
    return aclnnMoeInitRoutingGetWorkspaceSize(x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx, ws, ex);
}
aclnnStatus aclnnMoeInitRoutingQuantV2(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMoeInitRouting(w,wz,e,s); }

// ---- MoeDistributeCombineAddRmsNorm(+V2): real EP combine (HcclAlltoAll over x) → AddRmsNorm(combined+residual)·gamma.
//      Combine returns expert-processed tokens to their origin ranks (inverse of Dispatch); the residual-add +
//      rmsnorm tail then runs locally. comm==nullptr → combine is identity (exact for nranks=1). rmsnorm is
//      nonlinear so the AlltoAll must precede it. ----
extern aclnnStatus aclnnAddRmsNormGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,double,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
extern aclnnStatus aclnnAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
namespace {
struct CARState { const aclTensor*x,*residual,*gamma; double eps; aclTensor*y,*residualSum; HcclComm comm; };
std::map<aclOpExecutor*,CARState> g_car;
static aclnnStatus car_ws(const aclTensor*x,const aclTensor*residual,const aclTensor*gamma,double eps,HcclComm comm,aclTensor*y,aclTensor*residualSum,uint64_t*ws,aclOpExecutor**ex){
    if(!x||!y||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); g_car[e]={x,residual,gamma,eps,y,residualSum,comm}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus car_run(aclOpExecutor*e,cudaStream_t s){
    auto it=g_car.find(e); if(it==g_car.end()) return ACLNN_ERR_PARAM_INVALID; CARState st=it->second; g_car.erase(it);
    const aclTensor* combined=st.x; aclTensor t=*st.x; void* tmp=nullptr; aclnnStatus r=ACLNN_SUCCESS;
    if(st.comm){
        size_t bytes=(size_t)st.x->numel()*dtype_size(st.x->dtype); cudaMallocAsync(&tmp,bytes,s);
        uint32_t nr=0; HcclGetRankSize(st.comm,&nr); if(nr==0)nr=1; uint64_t per=(uint64_t)st.x->numel()/nr;
        if(HcclAlltoAll(st.x->data,per,moe_dt(st.x->dtype),tmp,per,moe_dt(st.x->dtype),st.comm,s)!=HCCL_SUCCESS) r=ACLNN_ERR_RUNTIME_ERROR;
        t.data=tmp; combined=&t;
    }
    if(r==ACLNN_SUCCESS){ uint64_t w2=0; aclOpExecutor*e2=nullptr; r=aclnnAddRmsNormGetWorkspaceSize(combined,st.residual,st.gamma,st.eps,st.y,st.residualSum,&w2,&e2);
        if(r==ACLNN_SUCCESS){ void*wb=nullptr; if(w2)cudaMalloc(&wb,w2); r=aclnnAddRmsNorm(wb,w2,e2,s); if(wb)cudaFree(wb); } }
    if(tmp)cudaFreeAsync(tmp,s); delete e; return r;
}
}
aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){ return car_ws(x,residual,gamma,eps,comm,y,residualSum,ws,ex); }
aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void *,uint64_t,aclOpExecutor*e,aclrtStream s){ return car_run(e,(cudaStream_t)s); }
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex){ return car_ws(x,residual,gamma,eps,comm,y,residualSum,ws,ex); }
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(void *,uint64_t,aclOpExecutor*e,aclrtStream s){ return car_run(e,(cudaStream_t)s); }

// ---- Dispatch/Combine Setup/Teardown: handshake bookkeeping → no-op (copy x→out if provided) ----
#define MOE_SETUP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){ (void)comm; if(!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->a=x; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS; } \
aclnnStatus NAME(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ if(e->a&&e->out&&e->a->data&&e->out->data) cudaMemcpyAsync(e->out->data,e->a->data,(size_t)std::min(e->a->numel(),e->out->numel())*dtype_size(e->out->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e); }
MOE_SETUP(aclnnMoeDistributeDispatchSetup)
MOE_SETUP(aclnnMoeDistributeDispatchTeardown)
MOE_SETUP(aclnnMoeDistributeCombineSetup)
MOE_SETUP(aclnnMoeDistributeCombineTeardown)

} // extern "C"
} // namespace _moe3_ext
#undef PERM_FWD
#undef PERMG_FWD
#undef UNPERM_FWD
#undef UNPERMG_FWD
#undef MOE_SETUP

