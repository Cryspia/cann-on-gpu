// Activation backward (P13) cross-check: gradInput = gradOutput*f'(val). + Softmax/LogSoftmax backward.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_act(const char *name, std::function<double(double)> dfn, bool useY, std::function<double(double)> yfn,
                  std::function<aclnnStatus(aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**)> getws, aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)) {
    const int n=4096; auto go=randv(n,-1,1), x=randv(n,-3,3);
    std::vector<float> val(n); for(int i=0;i<n;i++) val[i]=useY?(float)yfn(x[i]):x[i];
    std::vector<float> hz(n); DevBuf dg(n*4),dv(n*4),dz(n*4); dg.up(go.data()); dv.up(val.data());
    aclTensor *tg=mk({n},ACL_FLOAT,dg.p),*tv=mk({n},ACL_FLOAT,dv.p),*tz=mk({n},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return getws(tg,tv,tz,w,e);}, r);
    dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<n;i++){double ref=go[i]*dfn(val[i]); me=std::max(me,std::fabs(hz[i]-ref)); mr=std::max(mr,std::fabs(ref));}
    report(name, me/(mr+1e-9), 1e-5); aclDestroyTensor(tg);aclDestroyTensor(tv);aclDestroyTensor(tz);
}
static void t_softmax_bwd(bool logsm) {
    const int R=16,D=20; auto go=randv(R*D,-1,1), xx=randv(R*D,-2,2);
    std::vector<float> y(R*D);
    for(int r=0;r<R;r++){ double mx=-1e30; for(int i=0;i<D;i++) mx=std::max(mx,(double)xx[r*D+i]); double se=0; for(int i=0;i<D;i++) se+=std::exp(xx[r*D+i]-mx);
        for(int i=0;i<D;i++){ double sm=std::exp(xx[r*D+i]-mx)/se; y[r*D+i]=logsm?(float)std::log(sm):(float)sm; } }
    std::vector<float> hz(R*D); DevBuf dg(R*D*4),dy(R*D*4),dz(R*D*4); dg.up(go.data()); dy.up(y.data());
    aclTensor *tg=mk({R,D},ACL_FLOAT,dg.p),*ty=mk({R,D},ACL_FLOAT,dy.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
    if(logsm) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogSoftmaxBackwardGetWorkspaceSize(tg,ty,1,tz,w,e);}, aclnnLogSoftmaxBackward);
    else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSoftmaxBackwardGetWorkspaceSize(tg,ty,1,tz,w,e);}, aclnnSoftmaxBackward);
    dz.down(hz.data()); double me=0,mr=0;
    for(int r=0;r<R;r++){ double tot=0; if(logsm){for(int i=0;i<D;i++) tot+=go[r*D+i];} else {for(int i=0;i<D;i++) tot+=(double)go[r*D+i]*y[r*D+i];}
        for(int i=0;i<D;i++){ double ref= logsm? (double)go[r*D+i]-std::exp((double)y[r*D+i])*tot : (double)y[r*D+i]*((double)go[r*D+i]-tot);
            me=std::max(me,std::fabs(hz[r*D+i]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report(logsm?"LogSoftmaxBackward":"SoftmaxBackward", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tz);
}
int main() {
    init(); srand(41);
    auto sig=[](double x){return 1.0/(1.0+std::exp(-x));};
    t_act("ReluBackward",[](double x){return x>0?1.0:0.0;},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnReluBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnReluBackward);
    t_act("GeluBackward",[](double x){return 0.5*(1+std::erf(x*0.7071067811865476))+x*0.39894228040143267*std::exp(-0.5*x*x);},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnGeluBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnGeluBackward);
    t_act("SiluBackward",[&](double x){double s=sig(x);return s*(1+x*(1-s));},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSiluBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnSiluBackward);
    t_act("SoftplusBackward",[&](double x){return sig(x);},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSoftplusBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnSoftplusBackward);
    t_act("HardswishBackward",[](double x){return x<-3?0.0:(x>3?1.0:(2*x+3)/6);},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardswishBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnHardswishBackward);
    t_act("SigmoidBackward",[](double y){return y*(1-y);},true,[&](double x){return sig(x);},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSigmoidBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnSigmoidBackward);
    t_act("TanhBackward",[](double y){return 1-y*y;},true,[](double x){return std::tanh(x);},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnTanhBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnTanhBackward);
    // ---- additional activation backward ----
    auto fastgelu_d=[](double x){double g=0.7978845608028654,u=g*(x+0.044715*x*x*x),t=std::tanh(u),du=g*(1+0.134145*x*x);return 0.5*(1+t)+0.5*x*(1-t*t)*du;};
    t_act("FastGeluBackward",fastgelu_d,false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnFastGeluBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnFastGeluBackward);
    t_act("GeluBackwardV2(tanh)",fastgelu_d,false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnGeluBackwardV2GetWorkspaceSize(g,v,1,o,w,e);},aclnnGeluBackwardV2);
    t_act("HardsigmoidBackward",[](double x){return (x>-3&&x<3)?1.0/6:0.0;},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardsigmoidBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnHardsigmoidBackward);
    t_act("HardswishBackwardV2",[](double x){return x<-3?0.0:(x>3?1.0:(2*x+3)/6);},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardswishBackwardV2GetWorkspaceSize(g,v,o,w,e);},aclnnHardswishBackwardV2);
    t_act("LogSigmoidBackward",[&](double x){return 1.0/(1.0+std::exp(x));},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnLogSigmoidBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnLogSigmoidBackward);
    t_act("MishBackward",[](double x){double sp=x>0?x+std::log1p(std::exp(-x)):std::log1p(std::exp(x));double w=std::tanh(sp),s=1.0/(1.0+std::exp(-x));return w+x*(1-w*w)*s;},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnMishBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnMishBackward);
    t_act("SeluBackward",[](double x){const double sc=1.0507009873554805,al=1.6732632423543772;return sc*(x>0?1.0:al*std::exp(x));},false,{},[](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSeluBackwardGetWorkspaceSize(g,v,o,w,e);},aclnnSeluBackward);
    { const int n=4096; auto go=randv(n,-1,1),x=randv(n,-3,3);
      auto run_pb=[&](const char*nm, std::function<aclnnStatus(aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**)> gw, aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream), std::function<double(double)> d){
        std::vector<float> hz(n); DevBuf dg(n*4),dv(n*4),dz(n*4); dg.up(go.data()); dv.up(x.data());
        auto tg=mk({n},ACL_FLOAT,dg.p),tv=mk({n},ACL_FLOAT,dv.p),tz=mk({n},ACL_FLOAT,dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tg,tv,tz,w,e);}, r); dz.down(hz.data());
        double me=0,mr=0; for(int i=0;i<n;i++){double ref=go[i]*d(x[i]); me=std::max(me,std::fabs(hz[i]-ref)); mr=std::max(mr,std::fabs(ref));}
        report(nm, me/(mr+1e-9), 1e-5); aclDestroyTensor(tg);aclDestroyTensor(tv);aclDestroyTensor(tz); };
      float lam=0.5f; auto sl=aclCreateScalar(&lam,ACL_FLOAT);
      run_pb("HardshrinkBackward",[&](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardshrinkBackwardGetWorkspaceSize(g,v,sl,o,w,e);},aclnnHardshrinkBackward,[](double x){return (x>0.5||x<-0.5)?1.0:0.0;});
      run_pb("SoftshrinkBackward",[&](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSoftshrinkBackwardGetWorkspaceSize(g,v,sl,o,w,e);},aclnnSoftshrinkBackward,[](double x){return (x>0.5||x<-0.5)?1.0:0.0;});
      float th=0.f; auto st0=aclCreateScalar(&th,ACL_FLOAT);
      run_pb("ThresholdBackward",[&](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnThresholdBackwardGetWorkspaceSize(g,v,st0,o,w,e);},aclnnThresholdBackward,[](double x){return x>0?1.0:0.0;});
      float lo=-1.f,hi=1.f; auto slo=aclCreateScalar(&lo,ACL_FLOAT),shi=aclCreateScalar(&hi,ACL_FLOAT);
      run_pb("HardtanhBackward",[&](aclTensor*g,aclTensor*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardtanhBackwardGetWorkspaceSize(g,v,slo,shi,o,w,e);},aclnnHardtanhBackward,[](double x){return (x>-1&&x<1)?1.0:0.0;}); }
    // LeakyRelu / Elu via dedicated signatures
    { const int n=4096; auto go=randv(n,-1,1),x=randv(n,-3,3); std::vector<float> hz(n); DevBuf dg(n*4),dv(n*4),dz(n*4); dg.up(go.data()); dv.up(x.data());
      aclTensor *tg=mk({n},ACL_FLOAT,dg.p),*tv=mk({n},ACL_FLOAT,dv.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLeakyReluBackwardGetWorkspaceSize(tg,tv,0.1,tz,w,e);}, aclnnLeakyReluBackward);
      dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<n;i++){double ref=go[i]*(x[i]>0?1.0:0.1);me=std::max(me,std::fabs(hz[i]-ref));mr=std::max(mr,std::fabs(ref));}
      report("LeakyReluBackward",me/(mr+1e-9),1e-5); aclDestroyTensor(tg);aclDestroyTensor(tv);aclDestroyTensor(tz); }
    { const int n=4096; auto go=randv(n,-1,1),x=randv(n,-3,3); std::vector<float> hz(n); DevBuf dg(n*4),dv(n*4),dz(n*4); dg.up(go.data()); dv.up(x.data());
      aclTensor *tg=mk({n},ACL_FLOAT,dg.p),*tv=mk({n},ACL_FLOAT,dv.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEluBackwardGetWorkspaceSize(tg,tv,1.0,tz,w,e);}, aclnnEluBackward);
      dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<n;i++){double ref=go[i]*(x[i]>0?1.0:std::exp(x[i]));me=std::max(me,std::fabs(hz[i]-ref));mr=std::max(mr,std::fabs(ref));}
      report("EluBackward",me/(mr+1e-9),1e-5); aclDestroyTensor(tg);aclDestroyTensor(tv);aclDestroyTensor(tz); }
    t_softmax_bwd(false); t_softmax_bwd(true);
    // ---- P13 completion: pooling / interpolate / gather backward ----
    { const int A=6,Bc=4,L=5; auto go=randv(L*Bc,-1,1); std::vector<int64_t> idx(L); for(auto&v:idx) v=rand()%A;
      std::vector<float> hgi(A*Bc); DevBuf dgo(L*Bc*4),di(L*8),dgi(A*Bc*4); dgo.up(go.data()); di.up(idx.data());
      aclTensor *tgo=mk({L,Bc},ACL_FLOAT,dgo.p),*ti=mk({L},ACL_INT64,di.p),*tgi=mk({A,Bc},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherBackwardGetWorkspaceSize(tgo,0,ti,tgi,w,e);}, aclnnGatherBackward);
      dgi.down(hgi.data()); std::vector<double> ref(A*Bc,0); for(int l=0;l<L;l++)for(int b=0;b<Bc;b++) ref[idx[l]*Bc+b]+=go[l*Bc+b];
      double bad=0; for(int i=0;i<A*Bc;i++) bad=std::max(bad,(double)std::fabs(hgi[i]-ref[i]));
      report("GatherBackward",bad,1e-5); aclDestroyTensor(tgo);aclDestroyTensor(ti);aclDestroyTensor(tgi); }
    { const int NC=2,H=3,W=3,oH=6,oW=6; auto go=randv(NC*oH*oW,-1,1); std::vector<float> hgi(NC*H*W);
      DevBuf dgo(NC*oH*oW*4),dgi(NC*H*W*4); dgo.up(go.data());
      aclTensor *tgo=mk({NC,1,oH,oW},ACL_FLOAT,dgo.p),*tgi=mk({NC,1,H,W},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleNearest2dBackwardGetWorkspaceSize(tgo,tgi,w,e);}, aclnnUpsampleNearest2dBackward);
      dgi.down(hgi.data()); std::vector<double> ref(NC*H*W,0);
      for(int nc=0;nc<NC;nc++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){int ih=std::min(oh*H/oH,H-1),iw=std::min(ow*W/oW,W-1); ref[(nc*H+ih)*W+iw]+=go[(nc*oH+oh)*oW+ow];}
      double bad=0; for(int i=0;i<NC*H*W;i++) bad=std::max(bad,(double)std::fabs(hgi[i]-ref[i]));
      report("UpsampleNearest2dBackward",bad,1e-4); aclDestroyTensor(tgo);aclDestroyTensor(tgi); }
    { const int NC=2,H=3,W=3,oH=6,oW=6; bool align=false; auto go=randv(NC*oH*oW,-1,1); std::vector<float> hgi(NC*H*W);
      DevBuf dgo(NC*oH*oW*4),dgi(NC*H*W*4); dgo.up(go.data());
      aclTensor *tgo=mk({NC,1,oH,oW},ACL_FLOAT,dgo.p),*tgi=mk({NC,1,H,W},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(tgo,align,tgi,w,e);}, aclnnUpsampleBilinear2dBackward);
      dgi.down(hgi.data()); std::vector<double> ref(NC*H*W,0);
      for(int nc=0;nc<NC;nc++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ float fh=(oh+0.5f)*H/oH-0.5f,fw=(ow+0.5f)*W/oW-0.5f; fh=fh<0?0:fh;fw=fw<0?0:fw;
          int h0=(int)fh,w0=(int)fw,h1=std::min(h0+1,H-1),w1=std::min(w0+1,W-1); float dh=fh-h0,dw=fw-w0,g=go[(nc*oH+oh)*oW+ow]; double *p=&ref[nc*H*W];
          p[h0*W+w0]+=g*(1-dh)*(1-dw); p[h0*W+w1]+=g*(1-dh)*dw; p[h1*W+w0]+=g*dh*(1-dw); p[h1*W+w1]+=g*dh*dw; }
      double bad=0,mr=0; for(int i=0;i<NC*H*W;i++){bad=std::max(bad,(double)std::fabs(hgi[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));}
      report("UpsampleBilinear2dBackward",bad/(mr+1e-9),1e-4); aclDestroyTensor(tgo);aclDestroyTensor(tgi); }
    { const int NC=2,H=8,W=6,oH=3,oW=2; auto go=randv(NC*oH*oW,-1,1); std::vector<float> hgi(NC*H*W);
      DevBuf dgo(NC*oH*oW*4),dgi(NC*H*W*4); dgo.up(go.data());
      aclTensor *tgo=mk({NC,1,oH,oW},ACL_FLOAT,dgo.p),*tgi=mk({NC,1,H,W},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(tgo,tgi,w,e);}, aclnnAdaptiveAvgPool2dBackward);
      dgi.down(hgi.data()); std::vector<double> ref(NC*H*W,0);
      for(int nc=0;nc<NC;nc++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){int hs=oh*H/oH,he=(oh+1)*H/oH+((oh+1)*H%oH?1:0),ws=ow*W/oW,we=(ow+1)*W/oW+((ow+1)*W%oW?1:0);
          double share=go[(nc*oH+oh)*oW+ow]/((he-hs)*(we-ws)); for(int h=hs;h<he;h++)for(int w=ws;w<we;w++) ref[(nc*H+h)*W+w]+=share;}
      double bad=0,mr=0; for(int i=0;i<NC*H*W;i++){bad=std::max(bad,(double)std::fabs(hgi[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));}
      report("AdaptiveAvgPool2dBackward",bad/(mr+1e-9),1e-5); aclDestroyTensor(tgo);aclDestroyTensor(tgi); }
    // MaxPool2dWithIndices + MaxUnpool2d round-trip on [1,1,4,4], k=2 s=2
    { const int H=4,W=4,oH=2,oW=2; auto x=randv(H*W,-2,2); std::vector<float> ho(oH*oW),hu(H*W); std::vector<int64_t> hidx(oH*oW);
      DevBuf dx(H*W*4),dop(oH*oW*4),didx(oH*oW*8),du(H*W*4); dx.up(x.data());
      int64_t kk[2]={2,2},stp[2]={2,2},pd[2]={0,0}; aclIntArray *ak=aclCreateIntArray(kk,2),*as=aclCreateIntArray(stp,2),*ap=aclCreateIntArray(pd,2);
      aclTensor *tx=mk({1,1,H,W},ACL_FLOAT,dx.p),*top=mk({1,1,oH,oW},ACL_FLOAT,dop.p),*tidx=mk({1,1,oH,oW},ACL_INT64,didx.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxPool2dWithIndicesGetWorkspaceSize(tx,ak,as,ap,top,tidx,w,e);}, aclnnMaxPool2dWithIndices);
      dop.down(ho.data()); didx.down(hidx.data());
      double bad=0; std::vector<int64_t> refidx(oH*oW);
      for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ float best=-1e30; int64_t bi=0; for(int a=0;a<2;a++)for(int b=0;b<2;b++){int ih=oh*2+a,iw=ow*2+b; if(x[ih*W+iw]>best){best=x[ih*W+iw];bi=ih*W+iw;}} bad=std::max(bad,(double)std::fabs(ho[oh*oW+ow]-best)); refidx[oh*oW+ow]=bi; if(hidx[oh*oW+ow]!=bi) bad+=1; }
      report("MaxPool2dWithIndices",bad,1e-5);
      aclTensor *top2=mk({1,1,oH,oW},ACL_FLOAT,dop.p),*tidx2=mk({1,1,oH,oW},ACL_INT64,didx.p),*tu=mk({1,1,H,W},ACL_FLOAT,du.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxUnpool2dGetWorkspaceSize(top2,tidx2,H,W,tu,w,e);}, aclnnMaxUnpool2d);
      du.down(hu.data()); std::vector<double> ref(H*W,0); for(int p=0;p<oH*oW;p++) ref[refidx[p]]=ho[p];
      double bad2=0; for(int i=0;i<H*W;i++) bad2=std::max(bad2,(double)std::fabs(hu[i]-ref[i]));
      report("MaxUnpool2d",bad2,1e-5);
      aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyTensor(tx);aclDestroyTensor(top);aclDestroyTensor(tidx);aclDestroyTensor(top2);aclDestroyTensor(tidx2);aclDestroyTensor(tu); }
    // ---- GLU gradients: gradIn[a]=gradOut·act'(a)·b, gradIn[b]=gradOut·act(a) ----
    { auto glu_grad=[&](bool gelu,const char*nm){ const int R=8,D=12; auto in=randv(R*2*D,-2,2),go=randv(R*D,-1,1); std::vector<float> hi(R*2*D);
        DevBuf din(R*2*D*4),dgo(R*D*4),dgi(R*2*D*4); din.up(in.data()); dgo.up(go.data());
        aclTensor*tin=mk({R,2*D},ACL_FLOAT,din.p),*tgo=mk({R,D},ACL_FLOAT,dgo.p),*tgi=mk({R,2*D},ACL_FLOAT,dgi.p);
        if(gelu) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGeGluBackwardGetWorkspaceSize(tgo,tin,tgi,w,e);},aclnnGeGluBackward);
        else     exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSwiGluGradGetWorkspaceSize(tgo,tin,tgi,w,e);},aclnnSwiGluGrad);
        dgi.down(hi.data()); double me=0,mr=0;
        for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double a=in[r*2*D+d],b=in[r*2*D+D+d],g=go[r*D+d],act,dact;
          if(gelu){ double cdf=0.5*(1+std::erf(a*0.7071067811865476)); act=a*cdf; dact=cdf+a*0.39894228040143267*std::exp(-0.5*a*a); }
          else    { double s=1.0/(1.0+std::exp(-a)); act=a*s; dact=s+a*s*(1-s); }
          double ra=g*dact*b, rb=g*act;
          me=std::max(me,std::fabs((double)hi[r*2*D+d]-ra)); me=std::max(me,std::fabs((double)hi[r*2*D+D+d]-rb)); mr=std::max({mr,std::fabs(ra),std::fabs(rb)}); }
        report(nm,me/(mr+1e-9),1e-5); aclDestroyTensor(tin);aclDestroyTensor(tgo);aclDestroyTensor(tgi); };
      glu_grad(false,"SwiGluGrad"); glu_grad(true,"GeGluBackward"); }
    return finish();
}
