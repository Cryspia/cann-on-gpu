// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace _rnn {
// Sequence / RNN (P16): single-layer LSTM and GRU forward (fp32, layout [T,B,*], one direction).
// Gate weights match PyTorch: LSTM rows [i;f;g;o] (4H), GRU rows [r;z;n] (3H).

namespace {
constexpr int TH = 128;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__device__ inline float sig(float x){ return 1.f/(1.f+expf(-x)); }

// LSTM step: one thread per (b,h). hprev/cprev -> h/c; writes y row for timestep t.
__global__ void k_lstm_step(const float *x, const float *wih, const float *whh, const float *bih, const float *bhh,
        const float *hprev, const float *cprev, float *h, float *c, float *yt, int64_t B, int64_t I, int64_t H) {
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*H) return; int64_t b=idx/H, hh=idx%H;
    float g[4];
    for(int gate=0;gate<4;gate++){ int64_t row=gate*H+hh; double acc=(bih?bih[row]:0.)+(bhh?bhh[row]:0.);
        for(int64_t i=0;i<I;i++) acc+=(double)wih[row*I+i]*x[b*I+i];
        for(int64_t k=0;k<H;k++) acc+=(double)whh[row*H+k]*hprev[b*H+k]; g[gate]=(float)acc; }
    float gi=sig(g[0]), gf=sig(g[1]), gg=tanhf(g[2]), go=sig(g[3]);
    float cc=gf*cprev[b*H+hh]+gi*gg; c[b*H+hh]=cc; float hc=go*tanhf(cc); h[b*H+hh]=hc; yt[b*H+hh]=hc;
}
// GRU step
__global__ void k_gru_step(const float *x, const float *wih, const float *whh, const float *bih, const float *bhh,
        const float *hprev, float *h, float *yt, int64_t B, int64_t I, int64_t H) {
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=B*H) return; int64_t b=idx/H, hh=idx%H;
    double xr=(bih?bih[0*H+hh]:0.),xz=(bih?bih[1*H+hh]:0.),xn=(bih?bih[2*H+hh]:0.);
    double hr=(bhh?bhh[0*H+hh]:0.),hz=(bhh?bhh[1*H+hh]:0.),hn=(bhh?bhh[2*H+hh]:0.);
    for(int64_t i=0;i<I;i++){ float xi=x[b*I+i]; xr+=(double)wih[(0*H+hh)*I+i]*xi; xz+=(double)wih[(1*H+hh)*I+i]*xi; xn+=(double)wih[(2*H+hh)*I+i]*xi; }
    for(int64_t k=0;k<H;k++){ float hk=hprev[b*H+k]; hr+=(double)whh[(0*H+hh)*H+k]*hk; hz+=(double)whh[(1*H+hh)*H+k]*hk; hn+=(double)whh[(2*H+hh)*H+k]*hk; }
    float r=sig((float)(xr+hr)), z=sig((float)(xz+hz)); float n=tanhf((float)(xn + r*hn));
    float hc=(1.f-z)*n + z*hprev[b*H+hh]; h[b*H+hh]=hc; yt[b*H+hh]=hc;
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// LSTM: x[T,B,I], wih[4H,I], whh[4H,H], bih/bhh[4H] (nullable), h0/c0[B,H] -> y[T,B,H], hN[B,H], cN[B,H]
aclnnStatus aclnnLstmGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh,
        const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex) {
    if (!x||!wih||!whh||!h0||!c0||!y||!hN||!cN||!ex||x->dtype!=ACL_FLOAT||x->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=x; e->b=wih; e->c=whh; e->out=y; e->out2=hN;
    e->inputs={whh,bih,bhh,h0,c0,cN};   // pack extras
    e->ab=x->viewDims[0]; e->an=x->viewDims[1]; e->asq=x->viewDims[2]; e->ad=wih->viewDims[0]/4;
    // ws: 2 ping-pong h buffers + 2 c buffers (B*H each)
    if (ws) *ws = (uint64_t)4 * e->an * e->ad * sizeof(float);
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLstm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->ab,B=e->an,I=e->asq,H=e->ad;
    const float *x=(const float*)e->a->data,*wih=(const float*)e->b->data,*whh=(const float*)e->c->data;
    const float *bih=e->inputs[1]?(const float*)e->inputs[1]->data:nullptr, *bhh=e->inputs[2]?(const float*)e->inputs[2]->data:nullptr;
    const float *h0=(const float*)e->inputs[3]->data,*c0=(const float*)e->inputs[4]->data;
    float *y=(float*)e->out->data,*hN=(float*)e->out2->data,*cN=(float*)const_cast<aclTensor*>(e->inputs[5])->data;
    float *hbuf=(float*)ws, *cbuf=hbuf+2*B*H;   // hbuf[2][B*H], cbuf[2][B*H]
    cudaMemcpyAsync(hbuf, h0, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(cbuf, c0, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    int cur=0;
    for(int64_t t=0;t<T;t++){ float *hp=hbuf+cur*B*H,*cp=cbuf+cur*B*H,*hn=hbuf+(1-cur)*B*H,*cn=cbuf+(1-cur)*B*H;
        k_lstm_step<<<nb(B*H),TH,0,st>>>(x+t*B*I,wih,whh,bih,bhh,hp,cp,hn,cn,y+t*B*H,B,I,H); cur=1-cur; }
    cudaMemcpyAsync(hN, hbuf+cur*B*H, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    cudaMemcpyAsync(cN, cbuf+cur*B*H, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    return done(e);
}
// GRU: x[T,B,I], wih[3H,I], whh[3H,H], bih/bhh[3H], h0[B,H] -> y[T,B,H], hN[B,H]
aclnnStatus aclnnGruGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh,
        const aclTensor *h0, aclTensor *y, aclTensor *hN, uint64_t *ws, aclOpExecutor **ex) {
    if (!x||!wih||!whh||!h0||!y||!hN||!ex||x->dtype!=ACL_FLOAT||x->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=1; e->a=x; e->b=wih; e->c=whh; e->out=y; e->out2=hN; e->inputs={whh,bih,bhh,h0};
    e->ab=x->viewDims[0]; e->an=x->viewDims[1]; e->asq=x->viewDims[2]; e->ad=wih->viewDims[0]/3;
    if (ws) *ws = (uint64_t)2 * e->an * e->ad * sizeof(float);
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGru(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t T=e->ab,B=e->an,I=e->asq,H=e->ad;
    const float *x=(const float*)e->a->data,*wih=(const float*)e->b->data,*whh=(const float*)e->c->data;
    const float *bih=e->inputs[1]?(const float*)e->inputs[1]->data:nullptr, *bhh=e->inputs[2]?(const float*)e->inputs[2]->data:nullptr;
    const float *h0=(const float*)e->inputs[3]->data;
    float *y=(float*)e->out->data,*hN=(float*)e->out2->data; float *hbuf=(float*)ws;
    cudaMemcpyAsync(hbuf, h0, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    int cur=0;
    for(int64_t t=0;t<T;t++){ float *hp=hbuf+cur*B*H,*hn=hbuf+(1-cur)*B*H;
        k_gru_step<<<nb(B*H),TH,0,st>>>(x+t*B*I,wih,whh,bih,bhh,hp,hn,y+t*B*H,B,I,H); cur=1-cur; }
    cudaMemcpyAsync(hN, hbuf+cur*B*H, (size_t)B*H*sizeof(float), cudaMemcpyDeviceToDevice, st);
    return done(e);
}

} // extern "C"
} // namespace _rnn

namespace _rnn2_ext {
// RNN remainder (R8): Rnn (vanilla Elman cell, tanh/relu), EmbeddingBag (sum/mean/max pooling over bags).

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
// One Elman step: hnext[b,j] = act( sum_i x[b,i]*wih[j,i] + sum_k hprev[b,k]*whh[j,k] + bih[j] + bhh[j] )
__global__ void k_elman(const float*x,const float*hprev,const float*wih,const float*whh,const float*bih,const float*bhh,float*hnext,int B,int I,int H,int nl){
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(int64_t)B*H)return; int b=t/H, j=t%H;
    float s=(bih?bih[j]:0.f)+(bhh?bhh[j]:0.f);
    for(int i=0;i<I;i++) s+=x[b*I+i]*wih[j*I+i];
    for(int k=0;k<H;k++) s+=hprev[b*H+k]*whh[j*H+k];
    hnext[b*H+j]= nl==1 ? (s>0.f?s:0.f) : tanhf(s);
}
__global__ void k_embbag(const float*w,const int64_t*idx,const int64_t*off,float*out,int B,int D,int64_t total,int V,int mode){
    int64_t t=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(t>=(int64_t)B*D)return; int b=t/D, d=t%D;
    int64_t s=off[b], e=(b+1<B)?off[b+1]:total; float acc=(mode==2)?-3.4e38f:0.f; int cnt=0;
    for(int64_t p=s;p<e;p++){ int64_t id=idx[p]; if(id<0||id>=V)continue; float v=w[id*D+d]; if(mode==2)acc=fmaxf(acc,v); else acc+=v; cnt++; }
    if(mode==1&&cnt>0)acc/=cnt; if(mode==2&&cnt==0)acc=0.f; out[b*D+d]=acc;
}
inline aclnnStatus done(aclOpExecutor*e){aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR;delete e;return st;}
} // namespace

extern "C" {

// input [T,B,I], h0 [B,H], wih [H,I], whh [H,H], bih/bhh [H] (optional), nonlinearity 0=tanh 1=relu. out [T,B,H], hn [B,H].
aclnnStatus aclnnRnnGetWorkspaceSize(const aclTensor*input,const aclTensor*h0,const aclTensor*wih,const aclTensor*whh,const aclTensor*bih,const aclTensor*bhh,int64_t nonlinearity,aclTensor*out,aclTensor*hn,uint64_t*ws,aclOpExecutor**ex){
    if(!input||!h0||!wih||!whh||!out||!hn||!ex||input->dtype!=ACL_FLOAT||input->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=input; e->b=wih; e->c=whh; e->inputs={h0,bih,bhh}; e->out=out; e->out2=hn;
    e->ab=input->viewDims[0]; e->an=input->viewDims[1]; e->ad=input->viewDims[2]; e->k=h0->viewDims[1]; e->op=(int)nonlinearity;
    if(ws)*ws=(uint64_t)e->an*e->k*sizeof(float)*2; *ex=e; return ACLNN_SUCCESS;   // two h buffers for ping-pong
}
aclnnStatus aclnnRnn(void*ws,uint64_t,aclOpExecutor*e,aclrtStream s){ auto st=(cudaStream_t)s;
    int T=e->ab,B=e->an,I=e->ad,H=e->k; const float*x=(const float*)e->a->data,*wih=(const float*)e->b->data,*whh=(const float*)e->c->data;
    const float*bih=e->inputs[1]?(const float*)e->inputs[1]->data:nullptr,*bhh=e->inputs[2]?(const float*)e->inputs[2]->data:nullptr;
    float*hbuf=(float*)ws; float*hcur=hbuf, *hnxt=hbuf+(int64_t)B*H; float*outp=(float*)e->out->data;
    cudaMemcpyAsync(hcur,e->inputs[0]->data,(size_t)B*H*4,cudaMemcpyDeviceToDevice,st);
    for(int t=0;t<T;t++){
        k_elman<<<nb((int64_t)B*H),TH,0,st>>>(x+(int64_t)t*B*I,hcur,wih,whh,bih,bhh,hnxt,B,I,H,e->op);
        cudaMemcpyAsync(outp+(int64_t)t*B*H,hnxt,(size_t)B*H*4,cudaMemcpyDeviceToDevice,st);
        float*tmp=hcur;hcur=hnxt;hnxt=tmp;
    }
    cudaMemcpyAsync(e->out2->data,hcur,(size_t)B*H*4,cudaMemcpyDeviceToDevice,st); return done(e);
}
// weight [V,D], indices [total] int64, offsets [B] int64 (bag start), mode 0=sum 1=mean 2=max. out [B,D].
aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor*weight,const aclTensor*indices,const aclTensor*offsets,int64_t mode,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!weight||!indices||!offsets||!out||!ex||weight->dtype!=ACL_FLOAT||weight->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=weight; e->b=indices; e->c=offsets; e->out=out; e->op=(int)mode;
    e->k=weight->viewDims[0]; e->n=weight->viewDims[1]; e->m=indices->numel(); e->ab=offsets->numel();
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEmbeddingBag(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int B=e->ab,D=e->n,V=e->k;
    k_embbag<<<nb((int64_t)B*D),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const int64_t*)e->b->data,(const int64_t*)e->c->data,(float*)e->out->data,B,D,e->m,V,e->op); return done(e);
}

} // extern "C"
} // namespace _rnn2_ext

