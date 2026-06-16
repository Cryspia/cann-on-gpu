// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cufft.h>

namespace _fft {
// FFT / signal (P17): Fft/Ifft (C2C), Rfft/Irfft (R2C/C2R) over the last FFT axis, batched. Backed by cuFFT.
// Complex tensors are stored as interleaved float pairs (real, imag). Inverse transforms are 1/n normalized (PyTorch "backward" norm).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__global__ void k_scale(float *x, int64_t n, float s){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n) x[i]*=s; }
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// op: 0=Fft 1=Ifft 2=Rfft 3=Irfft. n = transform length along last axis. batch inferred from numel.
static aclnnStatus fft_ws(int op, const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex || x->dtype != ACL_FLOAT || out->dtype != ACL_FLOAT || n <= 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = op; e->a = x; e->out = out; e->n = n; e->m = x->numel();
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFftGetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return fft_ws(0, x, n, out, ws, ex); }
aclnnStatus aclnnIfftGetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return fft_ws(1, x, n, out, ws, ex); }
aclnnStatus aclnnRfftGetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return fft_ws(2, x, n, out, ws, ex); }
aclnnStatus aclnnIrfftGetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return fft_ws(3, x, n, out, ws, ex); }

static aclnnStatus fft_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t n = e->n, total = e->m; cufftHandle plan; cufftResult r;
    void *in = e->a->data, *out = e->out->data;
    if (e->op == 0 || e->op == 1) {                 // C2C: input complex (2 floats/elem)
        int64_t batch = (total / 2) / n; if (batch < 1) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        r = cufftPlan1d(&plan, (int)n, CUFFT_C2C, (int)batch); if (r != CUFFT_SUCCESS) { delete e; return ACLNN_ERR_RUNTIME_ERROR; }
        cufftSetStream(plan, s);
        cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, e->op == 0 ? CUFFT_FORWARD : CUFFT_INVERSE);
        if (e->op == 1) k_scale<<<nb(2*n*batch),TH,0,s>>>((float*)out, 2*n*batch, 1.f/(float)n);
    } else if (e->op == 2) {                          // R2C: input real (n floats), output complex (n/2+1)
        int64_t batch = total / n; if (batch < 1) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        r = cufftPlan1d(&plan, (int)n, CUFFT_R2C, (int)batch); if (r != CUFFT_SUCCESS) { delete e; return ACLNN_ERR_RUNTIME_ERROR; }
        cufftSetStream(plan, s); cufftExecR2C(plan, (cufftReal*)in, (cufftComplex*)out);
    } else {                                          // C2R: input complex (n/2+1), output real (n)
        int64_t nc = n/2+1; int64_t batch = (total / 2) / nc; if (batch < 1) { delete e; return ACLNN_ERR_PARAM_INVALID; }
        r = cufftPlan1d(&plan, (int)n, CUFFT_C2R, (int)batch); if (r != CUFFT_SUCCESS) { delete e; return ACLNN_ERR_RUNTIME_ERROR; }
        cufftSetStream(plan, s); cufftExecC2R(plan, (cufftComplex*)in, (cufftReal*)out);
        k_scale<<<nb(n*batch),TH,0,s>>>((float*)out, n*batch, 1.f/(float)n);
    }
    cudaStreamSynchronize(s); cufftDestroy(plan);
    return done(e);
}
aclnnStatus aclnnFft(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fft_run(e, (cudaStream_t)s); }
aclnnStatus aclnnIfft(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fft_run(e, (cudaStream_t)s); }
aclnnStatus aclnnRfft(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fft_run(e, (cudaStream_t)s); }
aclnnStatus aclnnIrfft(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return fft_run(e, (cudaStream_t)s); }

} // extern "C"
} // namespace _fft

namespace _fft2_ext {
// FFT remainder (R8): Fft2 (batched 2D C2C), Stft (framed windowed R2C). Backed by cuFFT.
// Complex stored as interleaved float pairs (real, imag). Inverse 2D FFT is 1/(n0*n1) normalized.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__global__ void k_scale(float*x,int64_t n,float s){int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n)x[i]*=s;}
// Build [B*frames, nFft] windowed real frames from signal [B, L].
__global__ void k_frame(const float*sig,const float*win,float*fb,int B,int L,int nFft,int hop,int frames){
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(int64_t)B*frames*nFft)return;
    int j=t%nFft; int fr=(t/nFft)%frames; int b=t/((int64_t)frames*nFft);
    int pos=fr*hop+j; float v=(pos<L)?sig[(int64_t)b*L+pos]:0.f; if(win)v*=win[j]; fb[t]=v;
}
// Transpose complex [B*frames, nFreq] -> [B, nFreq, frames].
__global__ void k_tr(const float*tmp,float*out,int B,int frames,int nFreq){
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(int64_t)B*frames*nFreq)return;
    int f=t%nFreq; int fr=(t/nFreq)%frames; int b=t/((int64_t)frames*nFreq);
    int64_t dst=(((int64_t)b*nFreq+f)*frames+fr); out[2*dst]=tmp[2*t]; out[2*dst+1]=tmp[2*t+1];
}
inline aclnnStatus done(aclOpExecutor*e){aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR;delete e;return st;}
} // namespace

extern "C" {

// x complex [.., n0, n1], out complex same shape. inverse 1/(n0*n1) normalized.
aclnnStatus aclnnFft2GetWorkspaceSize(const aclTensor*x,int64_t n0,int64_t n1,bool inverse,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!x||!out||!ex||x->dtype!=ACL_FLOAT||n0<=0||n1<=0) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=x; e->out=out; e->m=x->numel(); e->op=inverse?1:0; e->dscalars={(double)n0,(double)n1}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFft2(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ auto st=(cudaStream_t)s;
    int n0=(int)e->dscalars[0],n1=(int)e->dscalars[1]; int64_t batch=(e->m/2)/((int64_t)n0*n1); if(batch<1){delete e;return ACLNN_ERR_PARAM_INVALID;}
    int dims[2]={n0,n1}; cufftHandle plan;
    if(cufftPlanMany(&plan,2,dims,nullptr,1,0,nullptr,1,0,CUFFT_C2C,(int)batch)!=CUFFT_SUCCESS){delete e;return ACLNN_ERR_RUNTIME_ERROR;}
    cufftSetStream(plan,st); cufftExecC2C(plan,(cufftComplex*)e->a->data,(cufftComplex*)e->out->data,e->op?CUFFT_INVERSE:CUFFT_FORWARD);
    if(e->op){ int64_t tot=2*(int64_t)n0*n1*batch; k_scale<<<nb(tot),TH,0,st>>>((float*)e->out->data,tot,1.f/((float)n0*n1)); }
    cudaStreamSynchronize(st); cufftDestroy(plan); return done(e);
}
// x real [B, L]; window real [nFft] (optional). out complex [B, nFreq, frames], nFreq=nFft/2+1, frames=1+(L-nFft)/hop.
aclnnStatus aclnnStftGetWorkspaceSize(const aclTensor*x,int64_t nFft,int64_t hop,const aclTensor*window,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!x||!out||!ex||x->dtype!=ACL_FLOAT||x->viewDims.size()!=2||nFft<=0||hop<=0) return ACLNN_ERR_PARAM_INVALID;
    int64_t B=x->viewDims[0],L=x->viewDims[1]; if(L<nFft) return ACLNN_ERR_PARAM_INVALID;
    int64_t frames=1+(L-nFft)/hop, nFreq=nFft/2+1;
    auto*e=new aclOpExecutor(); e->a=x; e->b=window; e->out=out; e->ab=B; e->an=L; e->asq=nFft; e->askv=hop; e->ad=frames; e->k=nFreq;
    if(ws)*ws=(uint64_t)(B*frames*nFft + 2*B*frames*nFreq)*sizeof(float); *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnStft(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ auto st=(cudaStream_t)s;
    int B=e->ab,L=e->an,nFft=e->asq,hop=e->askv,frames=e->ad,nFreq=e->k;
    float*fb=(float*)ws; float*tmp=fb+(int64_t)B*frames*nFft;
    k_frame<<<nb((int64_t)B*frames*nFft),TH,0,st>>>((const float*)e->a->data,e->b?(const float*)e->b->data:nullptr,fb,B,L,nFft,hop,frames);
    cufftHandle plan; if(cufftPlan1d(&plan,nFft,CUFFT_R2C,B*frames)!=CUFFT_SUCCESS){delete e;return ACLNN_ERR_RUNTIME_ERROR;}
    cufftSetStream(plan,st); cufftExecR2C(plan,(cufftReal*)fb,(cufftComplex*)tmp);
    k_tr<<<nb((int64_t)B*frames*nFreq),TH,0,st>>>(tmp,(float*)e->out->data,B,frames,nFreq);
    cudaStreamSynchronize(st); cufftDestroy(plan); return done(e);
}

} // extern "C"
} // namespace _fft2_ext

