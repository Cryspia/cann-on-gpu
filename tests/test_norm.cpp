// Normalization extensions (P7) cross-check: InstanceNorm, LpNormalize, LocalResponseNorm, RmsNormGated, BatchNormBackward.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_instance() {
    const int N=2,C=3,S=8; double eps=1e-5; auto x=randv(N*C*S,-2,2); auto g=randv(C,0.5,1.5), b=randv(C,-1,1);
    std::vector<float> hz(N*C*S); DevBuf dx(N*C*S*4),dg(C*4),db(C*4),dz(N*C*S*4); dx.up(x.data()); dg.up(g.data()); db.up(b.data());
    aclTensor *tx=mk({N,C,S},ACL_FLOAT,dx.p),*tg=mk({C},ACL_FLOAT,dg.p),*tb=mk({C},ACL_FLOAT,db.p),*tz=mk({N,C,S},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInstanceNormGetWorkspaceSize(tx,tg,tb,eps,tz,w,e);}, aclnnInstanceNorm);
    dz.down(hz.data()); double me=0,mr=0;
    for(int n=0;n<N;n++)for(int c=0;c<C;c++){ double m=0; for(int s=0;s<S;s++) m+=x[(n*C+c)*S+s]; m/=S; double v=0; for(int s=0;s<S;s++){double d=x[(n*C+c)*S+s]-m; v+=d*d;} v/=S;
        double inv=1.0/std::sqrt(v+eps); for(int s=0;s<S;s++){ double ref=((x[(n*C+c)*S+s]-m)*inv)*g[c]+b[c]; me=std::max(me,std::fabs(hz[(n*C+c)*S+s]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("InstanceNorm", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_lp_normalize() {
    const int R=8,D=16; double eps=1e-12; auto x=randv(R*D,-2,2); std::vector<float> hz(R*D);
    DevBuf dx(R*D*4),dz(R*D*4); dx.up(x.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLpNormalizeGetWorkspaceSize(tx,2.0,eps,tz,w,e);}, aclnnLpNormalize);
    dz.down(hz.data()); double me=0,mr=0;
    for(int r=0;r<R;r++){ double n=0; for(int i=0;i<D;i++) n+=(double)x[r*D+i]*x[r*D+i]; n=std::sqrt(n); double den=n<eps?eps:n;
        for(int i=0;i<D;i++){ double ref=x[r*D+i]/den; me=std::max(me,std::fabs(hz[r*D+i]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("LpNormalize p=2", me/(mr+1e-9), 1e-6);
    aclDestroyTensor(tx);aclDestroyTensor(tz);
}
static void t_lrn() {
    const int N=1,C=5,S=4; int size=3; double alpha=1e-4,beta=0.75,k=1.0; auto x=randv(N*C*S,-2,2); std::vector<float> hz(N*C*S);
    DevBuf dx(N*C*S*4),dz(N*C*S*4); dx.up(x.data());
    aclTensor *tx=mk({N,C,S},ACL_FLOAT,dx.p),*tz=mk({N,C,S},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLocalResponseNormGetWorkspaceSize(tx,size,alpha,beta,k,tz,w,e);}, aclnnLocalResponseNorm);
    dz.down(hz.data()); double me=0,mr=0; int half=size/2;
    for(int c=0;c<C;c++)for(int s=0;s<S;s++){ double acc=0; for(int j=c-half;j<=c+half;j++){ if(j<0||j>=C)continue; double v=x[j*S+s]; acc+=v*v; }
        double den=std::pow(k+alpha/size*acc,beta); double ref=x[c*S+s]/den; me=std::max(me,std::fabs(hz[c*S+s]-ref)); mr=std::max(mr,std::fabs(ref)); }
    report("LocalResponseNorm", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tz);
}
static void t_rmsnorm_gated() {
    const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),gate=randv(R*D,-2,2),wt=randv(D,0.5,1.5); std::vector<float> hz(R*D);
    DevBuf dx(R*D*4),dg(R*D*4),dw(D*4),dz(R*D*4); dx.up(x.data()); dg.up(gate.data()); dw.up(wt.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({R,D},ACL_FLOAT,dg.p),*tw=mk({D},ACL_FLOAT,dw.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGatedGetWorkspaceSize(tx,tg,tw,eps,tz,w,e);}, aclnnRmsNormGated);
    dz.down(hz.data()); double me=0,mr=0;
    for(int r=0;r<R;r++){ double ss=0; std::vector<double> h(D); for(int i=0;i<D;i++){ double si=gate[r*D+i]/(1.0+std::exp(-(double)gate[r*D+i])); h[i]=x[r*D+i]*si; ss+=h[i]*h[i]; }
        double inv=1.0/std::sqrt(ss/D+eps); for(int i=0;i<D;i++){ double ref=h[i]*inv*wt[i]; me=std::max(me,std::fabs(hz[r*D+i]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("RmsNormGated", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tw);aclDestroyTensor(tz);
}
static void t_bn_bwd() {
    const int N=4,C=3,S=2; double eps=1e-5; auto x=randv(N*C*S,-2,2),gy=randv(N*C*S,-1,1),gamma=randv(C,0.5,1.5);
    std::vector<double> mean(C),inv(C);
    for(int c=0;c<C;c++){ double m=0; for(int n=0;n<N;n++)for(int s=0;s<S;s++) m+=x[(n*C+c)*S+s]; m/=N*S; double v=0; for(int n=0;n<N;n++)for(int s=0;s<S;s++){double d=x[(n*C+c)*S+s]-m;v+=d*d;} v/=N*S; mean[c]=m; inv[c]=1.0/std::sqrt(v+eps); }
    std::vector<float> hmean(C),hinv(C); for(int c=0;c<C;c++){hmean[c]=mean[c];hinv[c]=inv[c];}
    std::vector<float> hgx(N*C*S),hgg(C),hgb(C);
    DevBuf dx(N*C*S*4),dgy(N*C*S*4),dgamma(C*4),dm(C*4),di(C*4),dgx(N*C*S*4),dgg(C*4),dgb(C*4);
    dx.up(x.data()); dgy.up(gy.data()); dgamma.up(gamma.data()); dm.up(hmean.data()); di.up(hinv.data());
    aclTensor *tgy=mk({N,C,S},ACL_FLOAT,dgy.p),*tx=mk({N,C,S},ACL_FLOAT,dx.p),*tgamma=mk({C},ACL_FLOAT,dgamma.p),
              *tm=mk({C},ACL_FLOAT,dm.p),*ti=mk({C},ACL_FLOAT,di.p),*tgx=mk({N,C,S},ACL_FLOAT,dgx.p),*tgg=mk({C},ACL_FLOAT,dgg.p),*tgb=mk({C},ACL_FLOAT,dgb.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBatchNormBackwardGetWorkspaceSize(tgy,tx,tgamma,tm,ti,tgx,tgg,tgb,w,e);}, aclnnBatchNormBackward);
    dgx.down(hgx.data()); dgg.down(hgg.data()); dgb.down(hgb.data());
    double me=0,mr=0;
    for(int c=0;c<C;c++){ double gg=0,gb=0; for(int n=0;n<N;n++)for(int s=0;s<S;s++){ double xhat=(x[(n*C+c)*S+s]-mean[c])*inv[c]; gg+=gy[(n*C+c)*S+s]*xhat; gb+=gy[(n*C+c)*S+s]; }
        me=std::max(me,std::fabs(hgg[c]-gg)); me=std::max(me,std::fabs(hgb[c]-gb)); mr=std::max({mr,std::fabs(gg),std::fabs(gb)});
        double M=N*S; for(int n=0;n<N;n++)for(int s=0;s<S;s++){ double xhat=(x[(n*C+c)*S+s]-mean[c])*inv[c]; double ref=gamma[c]*inv[c]/M*(M*gy[(n*C+c)*S+s]-gb-xhat*gg);
            me=std::max(me,std::fabs(hgx[(n*C+c)*S+s]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("BatchNormBackward", me/(mr+1e-9), 1e-4);
    aclDestroyTensor(tgy);aclDestroyTensor(tx);aclDestroyTensor(tgamma);aclDestroyTensor(tm);aclDestroyTensor(ti);aclDestroyTensor(tgx);aclDestroyTensor(tgg);aclDestroyTensor(tgb);
}

static void t_gemma_rms() {
    const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),g=randv(D,-0.5,0.5); std::vector<float> hz(R*D),hr(R);
    DevBuf dx(R*D*4),dg(D*4),dz(R*D*4),dr(R*4); dx.up(x.data()); dg.up(g.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tz=mk({R,D},ACL_FLOAT,dz.p),*tr=mk({R},ACL_FLOAT,dr.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGemmaRmsNormGetWorkspaceSize(tx,tg,eps,tz,tr,w,e);}, aclnnGemmaRmsNorm);
    dz.down(hz.data()); dr.down(hr.data()); double me=0,mr=0,mer=0;
    for(int r=0;r<R;r++){ double ss=0; for(int i=0;i<D;i++) ss+=(double)x[r*D+i]*x[r*D+i]; double inv=1.0/std::sqrt(ss/D+eps);
        mer=std::max(mer,std::fabs(hr[r]-inv)/(std::fabs(inv)+1e-9));
        for(int i=0;i<D;i++){ double ref=x[r*D+i]*inv*(1.0+g[i]); me=std::max(me,std::fabs(hz[r*D+i]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("GemmaRmsNorm", me/(mr+1e-9), 1e-5); report("GemmaRmsNorm rstd", mer, 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tz);aclDestroyTensor(tr);
}
static void t_gn_silu() {
    const int N=2,C=4,S=6; int G=2; double eps=1e-5; auto x=randv(N*C*S,-2,2),g=randv(C,0.5,1.5),b=randv(C,-1,1);
    std::vector<float> hz(N*C*S); DevBuf dx(N*C*S*4),dg(C*4),db(C*4),dz(N*C*S*4); dx.up(x.data()); dg.up(g.data()); db.up(b.data());
    aclTensor *tx=mk({N,C,S},ACL_FLOAT,dx.p),*tg=mk({C},ACL_FLOAT,dg.p),*tb=mk({C},ACL_FLOAT,db.p),*tz=mk({N,C,S},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupNormSiluGetWorkspaceSize(tx,tg,tb,G,eps,tz,nullptr,nullptr,w,e);}, aclnnGroupNormSilu);
    dz.down(hz.data()); double me=0,mr=0; int Cg=C/G;
    for(int n=0;n<N;n++)for(int grp=0;grp<G;grp++){ int cnt=Cg*S; double m=0,q=0;
        for(int cc=0;cc<Cg;cc++)for(int s=0;s<S;s++){ double v=x[(n*C+grp*Cg+cc)*S+s]; m+=v; q+=v*v; } m/=cnt; double var=q/cnt-m*m; double inv=1.0/std::sqrt(var+eps);
        for(int cc=0;cc<Cg;cc++){ int c=grp*Cg+cc; for(int s=0;s<S;s++){ double y=(x[(n*C+c)*S+s]-m)*inv*g[c]+b[c]; double ref=y/(1.0+std::exp(-y));
            me=std::max(me,std::fabs(hz[(n*C+c)*S+s]-ref)); mr=std::max(mr,std::fabs(ref)); } } }
    report("GroupNormSilu", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_fast_ln() {
    const int R=8,D=16; double eps=1e-5; auto x=randv(R*D,-2,2),g=randv(D,0.5,1.5),b=randv(D,-1,1); std::vector<float> hz(R*D);
    DevBuf dx(R*D*4),dg(D*4),db(D*4),dz(R*D*4); dx.up(x.data()); dg.up(g.data()); db.up(b.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tb=mk({D},ACL_FLOAT,db.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFastLayerNormGetWorkspaceSize(tx,tg,tb,eps,tz,nullptr,nullptr,w,e);}, aclnnFastLayerNorm);
    dz.down(hz.data()); double me=0,mr=0;
    for(int r=0;r<R;r++){ double m=0; for(int i=0;i<D;i++) m+=x[r*D+i]; m/=D; double v=0; for(int i=0;i<D;i++){double d=x[r*D+i]-m;v+=d*d;} v/=D; double inv=1.0/std::sqrt(v+eps);
        for(int i=0;i<D;i++){ double ref=(x[r*D+i]-m)*inv*g[i]+b[i]; me=std::max(me,std::fabs(hz[r*D+i]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("FastLayerNorm", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void addrms_ref(const std::vector<float>&x,const std::vector<float>&r,const std::vector<float>&g,int R,int D,double eps,
                       std::vector<double>&y,std::vector<double>&sum){
    y.assign(R*D,0); sum.assign(R*D,0);
    for(int i=0;i<R;i++){ double ss=0; for(int d=0;d<D;d++){ double t=x[i*D+d]+r[i*D+d]; sum[i*D+d]=t; ss+=t*t; }
        double inv=1.0/std::sqrt(ss/D+eps); for(int d=0;d<D;d++) y[i*D+d]=sum[i*D+d]*inv*g[d]; }
}
static void t_addrms_cast() {
    const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),r=randv(R*D,-2,2),g=randv(D,0.5,1.5);
    std::vector<float> hy(R*D),hsum(R*D),hyc(R*D);
    DevBuf dx(R*D*4),dr(R*D*4),dg(D*4),dy(R*D*4),dsum(R*D*4),dyc(R*D*2); dx.up(x.data()); dr.up(r.data()); dg.up(g.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tr=mk({R,D},ACL_FLOAT,dr.p),*tg=mk({D},ACL_FLOAT,dg.p),
              *ty=mk({R,D},ACL_FLOAT,dy.p),*tsum=mk({R,D},ACL_FLOAT,dsum.p),*tyc=mk({R,D},ACL_FLOAT16,dyc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddRmsNormCastGetWorkspaceSize(tx,tr,tg,eps,ty,tyc,tsum,w,e);}, aclnnAddRmsNormCast);
    dy.down(hy.data()); dsum.down(hsum.data());
    std::vector<double> ry,rsum; addrms_ref(x,r,g,R,D,eps,ry,rsum);
    double me=0,mr=0,se=0,sr=0; for(int i=0;i<R*D;i++){ me=std::max(me,std::fabs((double)hy[i]-ry[i])); mr=std::max(mr,std::fabs(ry[i])); se=std::max(se,std::fabs((double)hsum[i]-rsum[i])); sr=std::max(sr,std::fabs(rsum[i])); }
    report("AddRmsNormCast y", me/(mr+1e-9), 1e-5); report("AddRmsNormCast residualSum", se/(sr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);aclDestroyTensor(tyc);
}
static void t_inplace_addrms() {
    const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),r=randv(R*D,-2,2),g=randv(D,0.5,1.5);
    std::vector<float> hy(R*D),hsum(R*D);
    DevBuf dx(R*D*4),dr(R*D*4),dg(D*4),dsum(R*D*4); dx.up(x.data()); dr.up(r.data()); dg.up(g.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tr=mk({R,D},ACL_FLOAT,dr.p),*tg=mk({D},ACL_FLOAT,dg.p),*tsum=mk({R,D},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceAddRmsNormGetWorkspaceSize(tx,tr,tg,eps,tsum,w,e);}, aclnnInplaceAddRmsNorm);
    dx.down(hy.data()); dsum.down(hsum.data());   // x now holds y (in place)
    std::vector<double> ry,rsum; addrms_ref(x,r,g,R,D,eps,ry,rsum);   // x already overwritten? no: host copy x unchanged
    double me=0,mr=0; for(int i=0;i<R*D;i++){ me=std::max(me,std::fabs((double)hy[i]-ry[i])); mr=std::max(mr,std::fabs(ry[i])); }
    report("InplaceAddRmsNorm", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tsum);
}
static void t_adaln() {
    const int R=8,D=16; double eps=1e-5; auto x=randv(R*D,-2,2),sc=randv(R*D,-0.5,0.5),sh=randv(R*D,-1,1);
    std::vector<float> hy(R*D); DevBuf dx(R*D*4),ds(R*D*4),dh(R*D*4),dy(R*D*4); dx.up(x.data()); ds.up(sc.data()); dh.up(sh.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*ts=mk({R,D},ACL_FLOAT,ds.p),*th=mk({R,D},ACL_FLOAT,dh.p),*ty=mk({R,D},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaLayerNormGetWorkspaceSize(tx,ts,th,eps,ty,w,e);}, aclnnAdaLayerNorm);
    dy.down(hy.data()); double me=0,mr=0;
    for(int r=0;r<R;r++){ double m=0; for(int d=0;d<D;d++) m+=x[r*D+d]; m/=D; double v=0; for(int d=0;d<D;d++){double u=x[r*D+d]-m;v+=u*u;} v/=D; double inv=1.0/std::sqrt(v+eps);
        for(int d=0;d<D;d++){ double ref=((x[r*D+d]-m)*inv)*(1.0+sc[r*D+d])+sh[r*D+d]; me=std::max(me,std::fabs(hy[r*D+d]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("AdaLayerNorm", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(ts);aclDestroyTensor(th);aclDestroyTensor(ty);
}
static void t_swiglu_quant() {
    const int R=8,D=16; auto in=randv(R*2*D,-2,2); std::vector<int8_t> hq(R*D); std::vector<float> hsc(R);
    DevBuf din(R*2*D*4),dq(R*D),dsc(R*4); din.up(in.data());
    aclTensor *tin=mk({R,2*D},ACL_FLOAT,din.p),*tq=mk({R,D},ACL_INT8,dq.p),*tsc=mk({R},ACL_FLOAT,dsc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSwiGluQuantGetWorkspaceSize(tin,tq,tsc,w,e);}, aclnnSwiGluQuant);
    dq.down(hq.data()); dsc.down(hsc.data()); double me=0,mr=0;
    for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double a=in[r*2*D+d],b=in[r*2*D+D+d]; double g=a/(1.0+std::exp(-a))*b;
        double deq=(double)hq[r*D+d]*hsc[r]; me=std::max(me,std::fabs(deq-g)); mr=std::max(mr,std::fabs(g)); }
    report("SwiGluQuant (dequant)", me/(mr+1e-9), 1e-2);   // int8 quant: ~1/127 relative
    aclDestroyTensor(tin);aclDestroyTensor(tq);aclDestroyTensor(tsc);
}
static void t_clipped_swiglu() {
    const int R=8,D=16; double clip=0.8; auto in=randv(R*2*D,-2,2); std::vector<float> ho(R*D);
    DevBuf din(R*2*D*4),dout(R*D*4); din.up(in.data());
    aclTensor *tin=mk({R,2*D},ACL_FLOAT,din.p),*tout=mk({R,D},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnClippedSwigluGetWorkspaceSize(tin,clip,tout,w,e);}, aclnnClippedSwiglu);
    dout.down(ho.data()); double me=0,mr=0;
    for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double a=in[r*2*D+d],b=in[r*2*D+D+d]; a=a<-clip?-clip:(a>clip?clip:a); double g=a/(1.0+std::exp(-a))*b;
        me=std::max(me,std::fabs(ho[r*D+d]-g)); mr=std::max(mr,std::fabs(g)); }
    report("ClippedSwiglu", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tin);aclDestroyTensor(tout);
}
static void t_addrms_dynquant() {
    const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),r=randv(R*D,-2,2),g=randv(D,0.5,1.5);
    std::vector<int8_t> hq(R*D); std::vector<float> hsc(R),hsum(R*D);
    DevBuf dx(R*D*4),dr(R*D*4),dg(D*4),dq(R*D),dsc(R*4),dsum(R*D*4); dx.up(x.data()); dr.up(r.data()); dg.up(g.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tr=mk({R,D},ACL_FLOAT,dr.p),*tg=mk({D},ACL_FLOAT,dg.p),*tq=mk({R,D},ACL_INT8,dq.p),*tsc=mk({R},ACL_FLOAT,dsc.p),*tsum=mk({R,D},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddRmsNormDynamicQuantGetWorkspaceSize(tx,tr,tg,eps,tq,tsc,tsum,w,e);}, aclnnAddRmsNormDynamicQuant);
    dq.down(hq.data()); dsc.down(hsc.data()); dsum.down(hsum.data()); double me=0,mr=0,se=0;
    for(int i=0;i<R;i++){ double ss=0; std::vector<double> s(D); for(int d=0;d<D;d++){ s[d]=x[i*D+d]+r[i*D+d]; ss+=s[d]*s[d]; se=std::max(se,std::fabs(hsum[i*D+d]-s[d])); }
        double inv=1.0/std::sqrt(ss/D+eps); for(int d=0;d<D;d++){ double y=s[d]*inv*g[d]; double deq=(double)hq[i*D+d]*hsc[i]; me=std::max(me,std::fabs(deq-y)); mr=std::max(mr,std::fabs(y)); } }
    report("AddRmsNormDynamicQuant residualSum", se, 1e-5); report("AddRmsNormDynamicQuant (dequant)", me/(mr+1e-9), 1e-2);
    aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tq);aclDestroyTensor(tsc);aclDestroyTensor(tsum);
}
static void t_rmsnorm_quant() {
    const int R=8,D=16; double eps=1e-6,scale=0.05,off=0; auto x=randv(R*D,-2,2),g=randv(D,0.5,1.5);
    std::vector<int8_t> hq(R*D); DevBuf dx(R*D*4),dg(D*4),dq(R*D); dx.up(x.data()); dg.up(g.data());
    aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tq=mk({R,D},ACL_INT8,dq.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormQuantGetWorkspaceSize(tx,tg,scale,off,eps,tq,w,e);}, aclnnRmsNormQuant);
    dq.down(hq.data()); int bad=0;
    for(int i=0;i<R;i++){ double ss=0; for(int d=0;d<D;d++) ss+=(double)x[i*D+d]*x[i*D+d]; double inv=1.0/std::sqrt(ss/D+eps);
        for(int d=0;d<D;d++){ double y=x[i*D+d]*inv*g[d]; int q=(int)std::lround(y/scale+off); q=q<-127?-127:(q>127?127:q); if(std::abs(q-(int)hq[i*D+d])>1) bad++; } }
    report("RmsNormQuant", bad, 0.0);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tq);
}
static void t_adaln_bwd() {
    const int R=8,D=16; double eps=1e-6; auto gy=randv(R*D,-1,1),x=randv(R*D,-2,2),sc=randv(R*D,-0.5,0.5);
    std::vector<float> hgx(R*D),hgs(R*D),hgsh(R*D);
    DevBuf dgy(R*D*4),dx(R*D*4),ds(R*D*4),dgx(R*D*4),dgs(R*D*4),dgsh(R*D*4); dgy.up(gy.data()); dx.up(x.data()); ds.up(sc.data());
    aclTensor *tgy=mk({R,D},ACL_FLOAT,dgy.p),*tx=mk({R,D},ACL_FLOAT,dx.p),*ts=mk({R,D},ACL_FLOAT,ds.p),*tgx=mk({R,D},ACL_FLOAT,dgx.p),*tgs=mk({R,D},ACL_FLOAT,dgs.p),*tgsh=mk({R,D},ACL_FLOAT,dgsh.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaLayerNormBackwardGetWorkspaceSize(tgy,tx,ts,eps,tgx,tgs,tgsh,w,e);}, aclnnAdaLayerNormBackward);
    dgx.down(hgx.data()); dgs.down(hgs.data()); dgsh.down(hgsh.data());
    double ex=0,mx=0, es=0,ms=0, esh=0;
    for(int r=0;r<R;r++){ double mean=0; for(int d=0;d<D;d++) mean+=x[r*D+d]; mean/=D; double var=0; for(int d=0;d<D;d++){double u=x[r*D+d]-mean;var+=u*u;} var/=D; double inv=1.0/std::sqrt(var+eps);
        double sg=0,sgn=0; std::vector<double> n(D),g(D); for(int d=0;d<D;d++){ n[d]=(x[r*D+d]-mean)*inv; g[d]=gy[r*D+d]*(1.0+sc[r*D+d]); sg+=g[d]; sgn+=g[d]*n[d]; }
        double mg=sg/D,mgn=sgn/D;
        for(int d=0;d<D;d++){ double rgx=inv*(g[d]-mg-n[d]*mgn), rgs=gy[r*D+d]*n[d], rgsh=gy[r*D+d];
            ex=std::max(ex,std::fabs(hgx[r*D+d]-rgx)); mx=std::max(mx,std::fabs(rgx));
            es=std::max(es,std::fabs(hgs[r*D+d]-rgs)); ms=std::max(ms,std::fabs(rgs));
            esh=std::max(esh,std::fabs(hgsh[r*D+d]-rgsh)); } }
    report("AdaLayerNormBackward gradX", ex/(mx+1e-9), 1e-5); report("AdaLayerNormBackward gradScale", es/(ms+1e-9), 1e-5); report("AdaLayerNormBackward gradShift", esh, 1e-6);
    aclDestroyTensor(tgy);aclDestroyTensor(tx);aclDestroyTensor(ts);aclDestroyTensor(tgx);aclDestroyTensor(tgs);aclDestroyTensor(tgsh);
}
static void t_addln_grad() {
    const int R=8,D=16; double eps=1e-6; auto gy=randv(R*D,-1,1),x=randv(R*D,-2,2),res=randv(R*D,-2,2),g=randv(D,0.5,1.5);
    std::vector<float> hgx(R*D),hgr(R*D),hgg(D),hgb(D);
    DevBuf dgy(R*D*4),dx(R*D*4),dres(R*D*4),dg(D*4),dgx(R*D*4),dgr(R*D*4),dgg(D*4),dgb(D*4);
    dgy.up(gy.data()); dx.up(x.data()); dres.up(res.data()); dg.up(g.data());
    aclTensor *tgy=mk({R,D},ACL_FLOAT,dgy.p),*tx=mk({R,D},ACL_FLOAT,dx.p),*tres=mk({R,D},ACL_FLOAT,dres.p),*tg=mk({D},ACL_FLOAT,dg.p),
              *tgx=mk({R,D},ACL_FLOAT,dgx.p),*tgr=mk({R,D},ACL_FLOAT,dgr.p),*tgg=mk({D},ACL_FLOAT,dgg.p),*tgb=mk({D},ACL_FLOAT,dgb.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddLayerNormGradGetWorkspaceSize(tgy,tx,tres,tg,eps,tgx,tgr,tgg,tgb,w,e);}, aclnnAddLayerNormGrad);
    dgx.down(hgx.data()); dgr.down(hgr.data()); dgg.down(hgg.data()); dgb.down(hgb.data());
    std::vector<double> rgg(D,0),rgb(D,0); double ex=0,mx=0,er=0;
    for(int r=0;r<R;r++){ double mean=0; for(int d=0;d<D;d++) mean+=x[r*D+d]+res[r*D+d]; mean/=D; double var=0; for(int d=0;d<D;d++){double u=x[r*D+d]+res[r*D+d]-mean;var+=u*u;} var/=D; double inv=1.0/std::sqrt(var+eps);
        double sg=0,sgn=0; std::vector<double> n(D),gg(D); for(int d=0;d<D;d++){ n[d]=(x[r*D+d]+res[r*D+d]-mean)*inv; gg[d]=gy[r*D+d]*g[d]; sg+=gg[d]; sgn+=gg[d]*n[d]; }
        double mg=sg/D,mgn=sgn/D;
        for(int d=0;d<D;d++){ double gs=inv*(gg[d]-mg-n[d]*mgn); ex=std::max(ex,std::fabs(hgx[r*D+d]-gs)); er=std::max(er,std::fabs(hgr[r*D+d]-gs)); mx=std::max(mx,std::fabs(gs)); rgg[d]+=gy[r*D+d]*n[d]; rgb[d]+=gy[r*D+d]; } }
    double eg=0,mg2=0,eb=0,mb=0; for(int d=0;d<D;d++){ eg=std::max(eg,std::fabs(hgg[d]-rgg[d])); mg2=std::max(mg2,std::fabs(rgg[d])); eb=std::max(eb,std::fabs(hgb[d]-rgb[d])); mb=std::max(mb,std::fabs(rgb[d])); }
    report("AddLayerNormGrad gradX", ex/(mx+1e-9), 1e-5); report("AddLayerNormGrad gradResidual", er/(mx+1e-9), 1e-5);
    report("AddLayerNormGrad gradGamma", eg/(mg2+1e-9), 1e-5); report("AddLayerNormGrad gradBeta", eb/(mb+1e-9), 1e-5);
    aclDestroyTensor(tgy);aclDestroyTensor(tx);aclDestroyTensor(tres);aclDestroyTensor(tg);aclDestroyTensor(tgx);aclDestroyTensor(tgr);aclDestroyTensor(tgg);aclDestroyTensor(tgb);
}
int main() {
    init(); srand(17);
    t_instance(); t_lp_normalize(); t_lrn(); t_rmsnorm_gated(); t_bn_bwd();
    t_gemma_rms(); t_gn_silu(); t_fast_ln(); t_addrms_cast(); t_inplace_addrms(); t_adaln();
    t_swiglu_quant(); t_clipped_swiglu(); t_addrms_dynquant(); t_rmsnorm_quant(); t_adaln_bwd(); t_addln_grad();
    { // LayerNormQuant: dequant(out·scale) ≈ layernorm(x)·g+b
      const int R=8,D=16; double eps=1e-5; auto x=randv(R*D,-2,2),g=randv(D,0.5,1.5),b=randv(D,-1,1);
      std::vector<int8_t> q(R*D); std::vector<float> sc(R);
      DevBuf dx(R*D*4),dg(D*4),db(D*4),dq(R*D),ds(R*4); dx.up(x.data()); dg.up(g.data()); db.up(b.data());
      aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tb=mk({D},ACL_FLOAT,db.p),*tq=mk({R,D},ACL_INT8,dq.p),*ts=mk({R},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLayerNormQuantGetWorkspaceSize(tx,tg,tb,eps,tq,ts,w,e);}, aclnnLayerNormQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int r=0;r<R;r++){ double m=0; for(int d=0;d<D;d++) m+=x[r*D+d]; m/=D; double v=0; for(int d=0;d<D;d++){double u=x[r*D+d]-m;v+=u*u;} v/=D; double inv=1.0/std::sqrt(v+eps);
        for(int d=0;d<D;d++){ double y=(x[r*D+d]-m)*inv*g[d]+b[d]; double deq=(double)q[r*D+d]*sc[r]; me=std::max(me,std::fabs(deq-y)); mr=std::max(mr,std::fabs(y)); } }
      report("LayerNormQuant", me/(mr+1e-9), 1e-2); aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // DequantSwigluQuant: dequant(out·scale) ≈ swiglu(x) (dq=null)
      const int R=8,D=12; auto in=randv(R*2*D,-2,2); std::vector<int8_t> q(R*D); std::vector<float> sc(R);
      DevBuf din(R*2*D*4),dq(R*D),ds(R*4); din.up(in.data());
      aclTensor *tin=mk({R,2*D},ACL_FLOAT,din.p),*tq=mk({R,D},ACL_INT8,dq.p),*ts=mk({R},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDequantSwigluQuantGetWorkspaceSize(tin,nullptr,tq,ts,w,e);}, aclnnDequantSwigluQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double a=in[r*2*D+d],bb=in[r*2*D+D+d]; double gg=a/(1.0+std::exp(-a))*bb; double deq=(double)q[r*D+d]*sc[r]; me=std::max(me,std::fabs(deq-gg)); mr=std::max(mr,std::fabs(gg)); }
      report("DequantSwigluQuant", me/(mr+1e-9), 1e-2); aclDestroyTensor(tin);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // BatchNormStats + BatchNormElemt == inference batchnorm formula
      const int N=4,C=3,S=5; double eps=1e-5; auto x=randv(N*C*S,-2,2),g=randv(C,0.5,1.5),b=randv(C,-1,1);
      DevBuf dx(N*C*S*4),dg(C*4),db(C*4),dm(C*4),di(C*4),dz(N*C*S*4); dx.up(x.data()); dg.up(g.data()); db.up(b.data());
      aclTensor *tx=mk({N,C,S},ACL_FLOAT,dx.p),*tg=mk({C},ACL_FLOAT,dg.p),*tb=mk({C},ACL_FLOAT,db.p),
                *tm=mk({C},ACL_FLOAT,dm.p),*ti=mk({C},ACL_FLOAT,di.p),*tz=mk({N,C,S},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBatchNormStatsGetWorkspaceSize(tx,eps,tm,ti,w,e);}, aclnnBatchNormStats);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBatchNormElemtGetWorkspaceSize(tx,tg,tb,tm,ti,tz,w,e);}, aclnnBatchNormElemt);
      std::vector<float> got(N*C*S); dz.down(got.data()); double me=0,mr=0;
      for(int c=0;c<C;c++){ double m=0; for(int n=0;n<N;n++)for(int s=0;s<S;s++)m+=x[(n*C+c)*S+s]; m/=N*S;
        double v=0; for(int n=0;n<N;n++)for(int s=0;s<S;s++){double u=x[(n*C+c)*S+s]-m;v+=u*u;} v/=N*S; double inv=1.0/std::sqrt(v+eps);
        for(int n=0;n<N;n++)for(int s=0;s<S;s++){ int idx=(n*C+c)*S+s; double y=(x[idx]-m)*inv*g[c]+b[c]; me=std::max(me,std::fabs(got[idx]-y)); mr=std::max(mr,std::fabs(y)); } }
      report("BatchNormStats+Elemt", me/(mr+1e-9), 1e-4);
      aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tm);aclDestroyTensor(ti);aclDestroyTensor(tz); }
    { // BatchNormGatherStatsWithCounts: combine 2 partitions == full-batch stats
      const int C=3; double eps=1e-5;
      // partition counts/means/invstds from two synthetic groups
      std::vector<float> means={1.0f,2.0f,-1.0f, 3.0f,0.0f,1.0f}, invs={1.0f,0.5f,2.0f, 0.5f,1.0f,1.0f}, counts={10.f,30.f};
      DevBuf dmean(6*4),dinv(6*4),dcnt(2*4),dom(C*4),doi(C*4); dmean.up(means.data()); dinv.up(invs.data()); dcnt.up(counts.data());
      aclTensor *tm=mk({2,C},ACL_FLOAT,dmean.p),*ti=mk({2,C},ACL_FLOAT,dinv.p),*tc=mk({2},ACL_FLOAT,dcnt.p),
                *tom=mk({C},ACL_FLOAT,dom.p),*toi=mk({C},ACL_FLOAT,doi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(tm,ti,tc,eps,tom,toi,w,e);}, aclnnBatchNormGatherStatsWithCounts);
      std::vector<float> gm(C),gi(C); dom.down(gm.data()); doi.down(gi.data()); double me=0,mr=0;
      for(int c=0;c<C;c++){ double tot=counts[0]+counts[1]; double m=(counts[0]*means[c]+counts[1]*means[C+c])/tot;
        double v0=1.0/(invs[c]*invs[c])-eps, v1=1.0/(invs[C+c]*invs[C+c])-eps;
        double var=(counts[0]*(v0+(means[c]-m)*(means[c]-m))+counts[1]*(v1+(means[C+c]-m)*(means[C+c]-m)))/tot;
        double inv=1.0/std::sqrt(var+eps);
        me=std::max(me,std::fabs(gm[c]-m)); me=std::max(me,std::fabs(gi[c]-inv)); mr=std::max(mr,std::fabs(m)); mr=std::max(mr,std::fabs(inv)); }
      report("BatchNormGatherStatsWithCounts", me/(mr+1e-9), 1e-5);
      aclDestroyTensor(tm);aclDestroyTensor(ti);aclDestroyTensor(tc);aclDestroyTensor(tom);aclDestroyTensor(toi); }
    { // GroupNormBackward (G=1): gradX matches the analytic group-norm gradient
      const int N=2,C=4,S=3,G=1; int Dn=C*S; auto gy=randv(N*C*S,-1,1),x=randv(N*C*S,-2,2);
      // CPU forward stats per (n,g): here g=1 → per sample over C*S
      std::vector<float> mean(N),rstd(N); double eps=1e-5;
      for(int n=0;n<N;n++){ double m=0; for(int i=0;i<Dn;i++)m+=x[n*Dn+i]; m/=Dn; double v=0; for(int i=0;i<Dn;i++){double u=x[n*Dn+i]-m;v+=u*u;} v/=Dn; mean[n]=(float)m; rstd[n]=(float)(1.0/std::sqrt(v+eps)); }
      DevBuf dgy(N*C*S*4),dx(N*C*S*4),dm(N*4),dr(N*4),dgx(N*C*S*4); dgy.up(gy.data()); dx.up(x.data()); dm.up(mean.data()); dr.up(rstd.data());
      aclTensor *tgy=mk({N,C,S},ACL_FLOAT,dgy.p),*tx=mk({N,C,S},ACL_FLOAT,dx.p),*tm=mk({N},ACL_FLOAT,dm.p),*tr=mk({N},ACL_FLOAT,dr.p),*tgx=mk({N,C,S},ACL_FLOAT,dgx.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupNormBackwardGetWorkspaceSize(tgy,tx,tm,tr,nullptr,(int64_t)G,tgx,nullptr,nullptr,w,e);}, aclnnGroupNormBackward);
      std::vector<float> got(N*C*S); dgx.down(got.data()); double me=0,mr=0;
      for(int n=0;n<N;n++){ double m=mean[n],rs=rstd[n],sa=0,sb=0;
        for(int i=0;i<Dn;i++){ double dy=gy[n*Dn+i]; double xhat=(x[n*Dn+i]-m)*rs; sa+=dy; sb+=dy*xhat; }
        for(int i=0;i<Dn;i++){ double dy=gy[n*Dn+i]; double xhat=(x[n*Dn+i]-m)*rs; double gi=rs*(dy-sa/Dn-xhat*sb/Dn); me=std::max(me,std::fabs(got[n*Dn+i]-gi)); mr=std::max(mr,std::fabs(gi)); } }
      report("GroupNormBackward", me/(mr+1e-9), 1e-4);
      aclDestroyTensor(tgy);aclDestroyTensor(tx);aclDestroyTensor(tm);aclDestroyTensor(tr);aclDestroyTensor(tgx); }
    { // RmsNormDynamicMxQuant: dequant(q·scale) ≈ rmsnorm(x)·g
      const int R=8,D=16; double eps=1e-6; auto x=randv(R*D,-2,2),g=randv(D,0.5,1.5);
      std::vector<int8_t> q(R*D); std::vector<float> sc(R);
      DevBuf dx(R*D*4),dg(D*4),dq(R*D),ds(R*4); dx.up(x.data()); dg.up(g.data());
      aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tq=mk({R,D},ACL_INT8,dq.p),*ts=mk({R},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormDynamicMxQuantGetWorkspaceSize(tx,tg,eps,tq,ts,w,e);}, aclnnRmsNormDynamicMxQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int r=0;r<R;r++){ double ms=0; for(int d=0;d<D;d++)ms+=x[r*D+d]*x[r*D+d]; ms/=D; double rr=1.0/std::sqrt(ms+eps);
        for(int d=0;d<D;d++){ double y=x[r*D+d]*rr*g[d]; double deq=(double)q[r*D+d]*sc[r]; me=std::max(me,std::fabs(deq-y)); mr=std::max(mr,std::fabs(y)); } }
      report("RmsNormDynamicMxQuant", me/(mr+1e-9), 2e-2);
      aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    return finish();
}
