// Tail aliases + small backward kernels: RmsNormGrad (alias), AddLayerNormGrad.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
constexpr int TB = 256;
// AddLayerNorm backward over s=x+residual (LayerNorm of the sum, affine gamma/beta). One block per row.
//   n=(s-mean)/std; g=gy·gamma; gradS = (1/std)(g - mean(g) - n·mean(g·n)); gradX = gradResidual = gradS;
//   gradGamma += gy·n; gradBeta += gy (fp32 atomics over rows).
template <typename T>
__global__ void k_addln_grad(const T *gy, const T *x, const T *res, const T *gamma, T *gradX, T *gradRes,
                             float *gGamma, float *gBeta, int64_t rows, int64_t D, float eps) {
    int64_t r = blockIdx.x; if (r >= rows) return; int64_t base = r * D; int t = threadIdx.x;
    __shared__ float sh[TB/32]; __shared__ float bc;
    auto rsum = [&](float v) -> float {
        #pragma unroll
        for (int o=16;o>0;o>>=1) v += __shfl_down_sync(0xffffffffu, v, o);
        if ((t&31)==0) sh[t>>5]=v; __syncthreads();
        if (t==0){ float a=0; for(int w=0;w<TB/32;w++) a+=sh[w]; bc=a; } __syncthreads();
        float res2=bc; __syncthreads(); return res2; };
    float sm=0; for (int64_t d=t;d<D;d+=TB) sm += (float)x[base+d]+(float)res[base+d];
    float mean = rsum(sm)/D;
    float v=0; for (int64_t d=t;d<D;d+=TB){ float u=(float)x[base+d]+(float)res[base+d]-mean; v+=u*u; }
    float inv = rsqrtf(rsum(v)/D + eps);
    float sg=0,sgn=0;
    for (int64_t d=t;d<D;d+=TB){ float n=((float)x[base+d]+(float)res[base+d]-mean)*inv; float g=(float)gy[base+d]*(gamma?(float)gamma[d]:1.f); sg+=g; sgn+=g*n; }
    float mg=rsum(sg)/D, mgn=rsum(sgn)/D;
    for (int64_t d=t;d<D;d+=TB){ float n=((float)x[base+d]+(float)res[base+d]-mean)*inv; float g=(float)gy[base+d]*(gamma?(float)gamma[d]:1.f);
        float gs=inv*(g-mg-n*mgn); gradX[base+d]=(T)gs; if (gradRes) gradRes[base+d]=(T)gs;
        if (gGamma) atomicAdd(&gGamma[d], (float)gy[base+d]*n);
        if (gBeta)  atomicAdd(&gBeta[d], (float)gy[base+d]); }
}
inline const void *D_(const aclTensor *t){ return t ? t->data : nullptr; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// RmsNormGrad == RmsNormBackward (gradY, x, gamma, eps -> gradX, gradGamma).
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
        double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRmsNormBackwardGetWorkspaceSize(gradY, x, gamma, eps, gradX, gradGamma, ws, ex);
}
aclnnStatus aclnnRmsNormGrad(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnRmsNormBackward(ws, wsz, e, s); }

// AddLayerNormGrad: gradY, x, residual, gamma, eps -> gradX, gradResidual, gradGamma, gradBeta (gamma/beta fp32 grads).
aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
        double eps, aclTensor *gradX, aclTensor *gradResidual, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !residual || !gradX || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (gradY->viewDims != x->viewDims || x->viewDims != residual->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (x->dtype != ACL_FLOAT && x->dtype != ACL_FLOAT16 && x->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (gradGamma && gradGamma->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    if (gradBeta && gradBeta->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int64_t Dd = x->viewDims.back();
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = residual; e->mask = gamma; e->out = gradX; e->out2 = gradResidual;
    if (gradGamma) e->inputs.push_back(gradGamma); if (gradBeta) e->inputs.push_back(gradBeta);
    e->reduceCount = Dd; e->outerCount = x->numel()/Dd; e->eps = eps;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddLayerNormGrad(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t rows = e->outerCount, Dd = e->reduceCount; auto st = (cudaStream_t)s; float eps=(float)e->eps;
    float *gG = e->inputs.size()>0 ? (float*)const_cast<aclTensor*>(e->inputs[0])->data : nullptr;
    float *gB = e->inputs.size()>1 ? (float*)const_cast<aclTensor*>(e->inputs[1])->data : nullptr;
    if (gG) cudaMemsetAsync(gG, 0, Dd*sizeof(float), st);
    if (gB) cudaMemsetAsync(gB, 0, Dd*sizeof(float), st);
    void *gr = e->out2 ? e->out2->data : nullptr;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_addln_grad<float><<<(unsigned)rows,TB,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,(const float*)D_(e->mask),(float*)e->out->data,(float*)gr,gG,gB,rows,Dd,eps); break;
        case ACL_FLOAT16: k_addln_grad<__half><<<(unsigned)rows,TB,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(const __half*)e->c->data,(const __half*)D_(e->mask),(__half*)e->out->data,(__half*)gr,gG,gB,rows,Dd,eps); break;
        default:          k_addln_grad<__nv_bfloat16><<<(unsigned)rows,TB,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(const __nv_bfloat16*)e->c->data,(const __nv_bfloat16*)D_(e->mask),(__nv_bfloat16*)e->out->data,(__nv_bfloat16*)gr,gG,gB,rows,Dd,eps); break;
    }
    return fin(e);
}

} // extern "C"
