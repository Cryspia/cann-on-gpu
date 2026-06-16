// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace _act_bwd {
// Activation backward coverage (P13): gradInput = gradOutput * f'(val). val = input x (most) or output y (sigmoid/tanh).
// Plus Softmax/LogSoftmax backward (per-row reduction). fp32.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

enum AK { A_RELU, A_LEAKY, A_GELU, A_SILU, A_SOFTPLUS, A_ELU, A_HARDSWISH, A_SIGMOID_Y, A_TANH_Y,
          A_FASTGELU, A_HARDSHRINK, A_HARDSIGMOID, A_HARDTANH, A_LOGSIGMOID, A_MISH, A_SELU, A_SOFTSHRINK, A_THRESHOLD };
__device__ inline double deriv(int ak, double v, double param, double param2) {
    switch (ak) {
        case A_RELU:      return v > 0 ? 1.0 : 0.0;
        case A_LEAKY:     return v > 0 ? 1.0 : param;
        case A_GELU:      { return 0.5*(1.0+erf(v*0.7071067811865476)) + v*0.39894228040143267*exp(-0.5*v*v); }
        case A_SILU:      { double s = 1.0/(1.0+exp(-v)); return s*(1.0 + v*(1.0 - s)); }
        case A_SOFTPLUS:  return 1.0/(1.0+exp(-v));
        case A_ELU:       return v > 0 ? 1.0 : param*exp(v);
        case A_HARDSWISH: return v < -3 ? 0.0 : (v > 3 ? 1.0 : (2.0*v+3.0)/6.0);
        case A_SIGMOID_Y: return v*(1.0-v);     // v=output y
        case A_TANH_Y:    return 1.0 - v*v;     // v=output y
        case A_FASTGELU:  { double g=0.7978845608028654, u=g*(v+0.044715*v*v*v), t=tanh(u), du=g*(1.0+0.134145*v*v);
                            return 0.5*(1.0+t) + 0.5*v*(1.0-t*t)*du; }
        case A_HARDSHRINK: return (v > param || v < -param) ? 1.0 : 0.0;
        case A_HARDSIGMOID: return (v > -3.0 && v < 3.0) ? (1.0/6.0) : 0.0;
        case A_HARDTANH:  return (v > param && v < param2) ? 1.0 : 0.0;
        case A_LOGSIGMOID: return 1.0/(1.0+exp(v));   // d/dx log(sigmoid(x)) = sigmoid(-x)
        case A_MISH:      { double sp = v > 0 ? v + log1p(exp(-v)) : log1p(exp(v)); double w = tanh(sp), sig = 1.0/(1.0+exp(-v));
                            return w + v*(1.0 - w*w)*sig; }
        case A_SELU:      { const double sc=1.0507009873554805, al=1.6732632423543772; return sc * (v > 0 ? 1.0 : al*exp(v)); }
        case A_SOFTSHRINK: return (v > param || v < -param) ? 1.0 : 0.0;
        case A_THRESHOLD: return v > param ? 1.0 : 0.0;
        default: return 0;
    }
}
__global__ void k_act_bwd(const float *go, const float *val, float *gi, int ak, double param, double param2, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) gi[i] = (float)((double)go[i] * deriv(ak, val[i], param, param2));
}
// Softmax backward over segments of length D: gi = y*(go - sum_j go_j*y_j)
__global__ void k_softmax_bwd(const float *go, const float *y, float *gi, int64_t rows, int64_t D, bool logsm) {
    int64_t r = blockIdx.x; if (r >= rows) return; const float *gp=go+r*D,*yp=y+r*D; float *o=gi+r*D;
    __shared__ double red[TH]; double s=0;
    if (logsm) for (int64_t i=threadIdx.x;i<D;i+=blockDim.x) s += gp[i];                 // sum of grad
    else       for (int64_t i=threadIdx.x;i<D;i+=blockDim.x) s += (double)gp[i]*yp[i];   // sum(go*y)
    red[threadIdx.x]=s; __syncthreads();
    for (int k=blockDim.x/2;k>0;k>>=1){ if(threadIdx.x<k) red[threadIdx.x]+=red[threadIdx.x+k]; __syncthreads(); }
    __shared__ double tot; if(threadIdx.x==0) tot=red[0]; __syncthreads();
    for (int64_t i=threadIdx.x;i<D;i+=blockDim.x) {
        if (logsm) o[i] = (float)((double)gp[i] - exp((double)yp[i])*tot);   // y is logsoftmax output
        else       o[i] = (float)((double)yp[i]*((double)gp[i] - tot));      // y is softmax output
    }
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

static aclnnStatus actbwd_ws2(int ak, const aclTensor *go, const aclTensor *val, double param, double param2, aclTensor *gi, uint64_t *ws, aclOpExecutor **ex) {
    if (!go || !val || !gi || !ex || go->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = ak; e->a = go; e->b = val; e->out = gi; e->m = go->numel(); e->alpha = param; e->eps = param2;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus actbwd_ws(int ak, const aclTensor *go, const aclTensor *val, double param, aclTensor *gi, uint64_t *ws, aclOpExecutor **ex) {
    return actbwd_ws2(ak, go, val, param, 0.0, gi, ws, ex);
}
static aclnnStatus actbwd_run(aclOpExecutor *e, cudaStream_t s) {
    k_act_bwd<<<nb(e->m),TH,0,s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,e->op,e->alpha,e->eps,e->m);
    return done(e);
}
} // namespace

extern "C" {

aclnnStatus aclnnReluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_RELU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnReluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnGeluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_GELU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnGeluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnSiluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_SILU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnSiluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnSoftplusBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_SOFTPLUS, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnSoftplusBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnHardswishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_HARDSWISH, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnHardswishBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double negativeSlope, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_LEAKY, gradOutput, self, negativeSlope, gradInput, ws, ex); }
aclnnStatus aclnnLeakyReluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnEluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double alpha, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_ELU, gradOutput, self, alpha, gradInput, ws, ex); }
aclnnStatus aclnnEluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
// sigmoid/tanh backward take the OUTPUT y
aclnnStatus aclnnSigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_SIGMOID_Y, gradOutput, output, 0, gradInput, ws, ex); }
aclnnStatus aclnnSigmoidBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnTanhBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_TANH_Y, gradOutput, output, 0, gradInput, ws, ex); }
aclnnStatus aclnnTanhBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }

// ---- additional activation backward ----
aclnnStatus aclnnFastGeluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_FASTGELU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnFastGeluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnGeluBackwardV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, int64_t approximate, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(approximate == 1 ? A_FASTGELU : A_GELU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnGeluBackwardV2(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnHardshrinkBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *lambd, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_HARDSHRINK, gradOutput, self, lambd ? lambd->v : 0.5, gradInput, ws, ex); }
aclnnStatus aclnnHardshrinkBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnHardsigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_HARDSIGMOID, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnHardsigmoidBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnHardswishBackwardV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_HARDSWISH, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnHardswishBackwardV2(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *minVal, const aclScalar *maxVal, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws2(A_HARDTANH, gradOutput, self, minVal ? minVal->v : -1.0, maxVal ? maxVal->v : 1.0, gradInput, ws, ex); }
aclnnStatus aclnnHardtanhBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnLogSigmoidBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_LOGSIGMOID, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnLogSigmoidBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnMishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_MISH, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnMishBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnSeluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_SELU, gradOutput, self, 0, gradInput, ws, ex); }
aclnnStatus aclnnSeluBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *lambd, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_SOFTSHRINK, gradOutput, self, lambd ? lambd->v : 0.5, gradInput, ws, ex); }
aclnnStatus aclnnSoftshrinkBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnThresholdBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *threshold, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return actbwd_ws(A_THRESHOLD, gradOutput, self, threshold ? threshold->v : 0.0, gradInput, ws, ex); }
aclnnStatus aclnnThresholdBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return actbwd_run(e, (cudaStream_t)s); }

// Softmax / LogSoftmax backward (over last dim)
static aclnnStatus smbwd_ws(bool logsm, const aclTensor *gradOutput, const aclTensor *output, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !output || !gradInput || !ex || output->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)output->viewDims.size(); auto *e=new aclOpExecutor(); e->op=logsm?1:0; e->a=gradOutput; e->b=output; e->out=gradInput;
    e->n=output->viewDims[rank-1]; e->m=output->numel()/e->n;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t /*dim*/, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return smbwd_ws(false, gradOutput, output, gradInput, ws, ex); }
aclnnStatus aclnnSoftmaxBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_softmax_bwd<<<e->m,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,e->m,e->n,false); return done(e);
}
aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t /*dim*/, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) { return smbwd_ws(true, gradOutput, output, gradInput, ws, ex); }
aclnnStatus aclnnLogSoftmaxBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_softmax_bwd<<<e->m,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,e->m,e->n,true); return done(e);
}

} // extern "C"
} // namespace _act_bwd

namespace _grad_ext {
// Backward coverage completion (P13): GatherBackward, UpsampleNearest2dBackward, UpsampleBilinear2dBackward,
// AdaptiveAvgPool2dBackward. All scatter-add grad into a pre-zeroed gradInput. fp32.

namespace {
constexpr int TH = 256;
constexpr int MAXD = 8;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__device__ inline int64_t imin(int64_t a,int64_t b){return a<b?a:b;}

// Gather backward: gradOut has out shape (self with dim replaced by index length); scatter-add into gradIn (self shape)
struct SGB { int rank, gd; int64_t od[MAXD], istr[MAXD], base; };
__global__ void k_gather_bwd(const float *go, const int64_t *idx, float *gi, SGB d, int64_t n) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    int64_t rem=i, off=d.base, g=0;
    // index is 1-D along the gather dim (index_select convention, matching aclnnGather forward):
    // idx is read by the gather-dim output coordinate g and shared across all other dims.
    for(int k=d.rank-1;k>=0;--k){ int64_t c=rem%d.od[k]; rem/=d.od[k]; if(k==d.gd) g=c; else off+=c*d.istr[k]; }
    off += idx[g]*d.istr[d.gd]; atomicAdd(&gi[off], go[i]);
}
__global__ void k_up_nearest_bwd(const float *go, float *gi, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return;
    int64_t ow=i%oW, oh=(i/oW)%oH, nc=i/(oH*oW); int64_t ih=imin(oh*H/oH,H-1), iw=imin(ow*W/oW,W-1);
    atomicAdd(&gi[(nc*H+ih)*W+iw], go[i]);
}
__global__ void k_up_bilinear_bwd(const float *go, float *gi, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW, bool align) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return;
    int64_t ow=i%oW, oh=(i/oW)%oH, nc=i/(oH*oW); float fh,fw;
    if(align){fh=oH>1?(float)oh*(H-1)/(oH-1):0;fw=oW>1?(float)ow*(W-1)/(oW-1):0;} else {fh=(oh+0.5f)*H/oH-0.5f;fw=(ow+0.5f)*W/oW-0.5f;}
    fh=fh<0?0:fh;fw=fw<0?0:fw; int64_t h0=(int64_t)fh,w0=(int64_t)fw,h1=imin(h0+1,H-1),w1=imin(w0+1,W-1); float dh=fh-h0,dw=fw-w0; float g=go[i]; float *p=gi+nc*H*W;
    atomicAdd(&p[h0*W+w0], g*(1-dh)*(1-dw)); atomicAdd(&p[h0*W+w1], g*(1-dh)*dw); atomicAdd(&p[h1*W+w0], g*dh*(1-dw)); atomicAdd(&p[h1*W+w1], g*dh*dw);
}
__global__ void k_adaptive_avg2d_bwd(const float *go, float *gi, int64_t NC, int64_t H, int64_t W, int64_t oH, int64_t oW) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=NC*oH*oW) return;
    int64_t ow=i%oW, oh=(i/oW)%oH, nc=i/(oH*oW);
    int64_t hs=oh*H/oH, he=(oh+1)*H/oH+((oh+1)*H%oH?1:0), ws=ow*W/oW, we=(ow+1)*W/oW+((ow+1)*W%oW?1:0);
    int64_t area=(he-hs)*(we-ws); float share=go[i]/(float)area; float *p=gi+nc*H*W;
    for(int64_t h=hs;h<he;++h)for(int64_t w=ws;w<we;++w) atomicAdd(&p[h*W+w], share);
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// GatherBackward: gradOut[out shape], index[L] int64, dim, selfShape -> gradInput (self shape, scatter-add)
aclnnStatus aclnnGatherBackwardGetWorkspaceSize(const aclTensor *gradOut, int64_t dim, const aclTensor *index, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!index||!gradInput||!ex||gradOut->dtype!=ACL_FLOAT||index->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)gradInput->viewDims.size(); if(dim<0)dim+=rank; if(dim<0||dim>=rank||rank>MAXD) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=gradOut; e->b=index; e->out=gradInput; e->dim=dim; e->m=gradOut->numel();
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGatherBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; const aclTensor *go=e->a, *gi=e->out; int rank=(int)gi->viewDims.size();
    cudaMemsetAsync(gi->data,0,(size_t)gi->numel()*sizeof(float),st);
    SGB d{}; d.rank=rank; d.gd=(int)e->dim; d.base=0; int64_t acc=1, istr[MAXD];
    for(int i=rank-1;i>=0;--i){ istr[i]=acc; acc*=gi->viewDims[i]; }
    for(int i=0;i<rank;++i){ d.od[i]=go->viewDims[i]; d.istr[i]=istr[i]; }
    k_gather_bwd<<<nb(e->m),TH,0,st>>>((const float*)go->data,(const int64_t*)e->b->data,(float*)gi->data,d,e->m);
    return done(e);
}
// UpsampleNearest2dBackward: gradOut[N,C,oH,oW] -> gradInput[N,C,H,W]
aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!gradInput||!ex||gradOut->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=gradOut; e->out=gradInput;
    e->outerCount=gradInput->viewDims[0]*gradInput->viewDims[1]; e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->k=gradOut->viewDims[2]; e->reduceCount=gradOut->viewDims[3];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUpsampleNearest2dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_up_nearest_bwd<<<nb(NC*oH*oW),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,NC,H,W,oH,oW); return done(e);
}
// UpsampleBilinear2dBackward
aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!gradInput||!ex||gradOut->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=gradOut; e->out=gradInput; e->keepDim=alignCorners;
    e->outerCount=gradInput->viewDims[0]*gradInput->viewDims[1]; e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->k=gradOut->viewDims[2]; e->reduceCount=gradOut->viewDims[3];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUpsampleBilinear2dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_up_bilinear_bwd<<<nb(NC*oH*oW),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,NC,H,W,oH,oW,e->keepDim); return done(e);
}
// AdaptiveAvgPool2dBackward
aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut||!gradInput||!ex||gradOut->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=gradOut; e->out=gradInput;
    e->outerCount=gradInput->viewDims[0]*gradInput->viewDims[1]; e->m=gradInput->viewDims[2]; e->n=gradInput->viewDims[3]; e->k=gradOut->viewDims[2]; e->reduceCount=gradOut->viewDims[3];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveAvgPool2dBackward(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t NC=e->outerCount,H=e->m,W=e->n,oH=e->k,oW=e->reduceCount;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*sizeof(float),st);
    k_adaptive_avg2d_bwd<<<nb(NC*oH*oW),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,NC,H,W,oH,oW); return done(e);
}

} // extern "C"
} // namespace _grad_ext

