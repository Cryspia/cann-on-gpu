// Loss extensions (P10) cross-check: L1/SmoothL1/Huber/KLDiv/BCE/SoftMargin/MarginRanking/HingeEmbedding + L1/SmoothL1 backward.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_basic_losses() {
    const int n = 1000; auto p = randv(n,-2,2), t = randv(n,-2,2);
    auto run = [&](const char *name, int lk, double param,
                   std::function<aclnnStatus(aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**)> getws,
                   aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),
                   std::function<double(double,double)> ref, std::vector<float> pp, std::vector<float> tt, double tol) {
        std::vector<float> hz(1); DevBuf dp(n*4),dt(n*4),dz(4); dp.up(pp.data()); dt.up(tt.data());
        aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tg=mk({n},ACL_FLOAT,dt.p),*tz=mk({1},ACL_FLOAT,dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return getws(tp,tg,tz,w,e);}, r);
        dz.down(hz.data()); double s=0; for(int i=0;i<n;i++) s+=ref(pp[i],tt[i]); s/=n;
        report(name, std::fabs(hz[0]-s)/(std::fabs(s)+1e-9), tol);
        aclDestroyTensor(tp);aclDestroyTensor(tg);aclDestroyTensor(tz);
    };
    run("L1Loss mean", 0, 0, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnL1LossGetWorkspaceSize(a,b,1,o,w,e);}, aclnnL1Loss,
        [](double p,double t){return std::fabs(p-t);}, p, t, 1e-5);
    run("SmoothL1 b=1 mean", 0, 1, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSmoothL1LossGetWorkspaceSize(a,b,1,1.0,o,w,e);}, aclnnSmoothL1Loss,
        [](double p,double t){double d=std::fabs(p-t);return d<1.0?0.5*d*d:d-0.5;}, p, t, 1e-5);
    run("Huber d=1 mean", 0, 1, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHuberLossGetWorkspaceSize(a,b,1,1.0,o,w,e);}, aclnnHuberLoss,
        [](double p,double t){double d=std::fabs(p-t);return d<1.0?0.5*d*d:1.0*(d-0.5);}, p, t, 1e-5);
    run("SoftMargin mean", 0, 0, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSoftMarginLossGetWorkspaceSize(a,b,1,o,w,e);}, aclnnSoftMarginLoss,
        [](double p,double t){return std::log1p(std::exp(-t*p));}, p, std::vector<float>([&]{std::vector<float> y(n);for(int i=0;i<n;i++)y[i]=(t[i]>0?1.f:-1.f);return y;}()), 1e-5);
    // KLDiv: input is log-prob, target prob in (0,1); use positive target
    auto tpos = randv(n,0.05f,1.0f); auto plog = randv(n,-3,-0.1f);
    run("KLDiv mean", 0, 0, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnKlDivGetWorkspaceSize(a,b,1,o,w,e);}, aclnnKlDiv,
        [](double p,double t){return t*(std::log(t)-p);}, plog, tpos, 1e-4);
    // BCE: input prob (0,1), target prob
    auto pp = randv(n,0.05f,0.95f); auto tt = randv(n,0.05f,0.95f);
    run("BCE mean", 0, 0, [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnBinaryCrossEntropyGetWorkspaceSize(a,b,1,o,w,e);}, aclnnBinaryCrossEntropy,
        [](double p,double t){return -(t*std::log(p)+(1-t)*std::log(1-p));}, pp, tt, 1e-4);
}
static void t_margin() {
    const int n=500; auto x1=randv(n,-2,2),x2=randv(n,-2,2); std::vector<float> y(n); for(int i=0;i<n;i++) y[i]=(rand()&1)?1.f:-1.f;
    double margin=0.5;
    { std::vector<float> hz(1); DevBuf d1(n*4),d2(n*4),dy(n*4),dz(4); d1.up(x1.data()); d2.up(x2.data()); dy.up(y.data());
      aclTensor *t1=mk({n},ACL_FLOAT,d1.p),*t2=mk({n},ACL_FLOAT,d2.p),*ty=mk({n},ACL_FLOAT,dy.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMarginRankingLossGetWorkspaceSize(t1,t2,ty,margin,1,tz,w,e);}, aclnnMarginRankingLoss);
      dz.down(hz.data()); double s=0; for(int i=0;i<n;i++) s+=std::max(0.0,-(double)y[i]*(x1[i]-x2[i])+margin); s/=n;
      report("MarginRankingLoss", std::fabs(hz[0]-s)/(std::fabs(s)+1e-9), 1e-5); aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(ty);aclDestroyTensor(tz); }
    { auto x=randv(n,-2,2); std::vector<float> hz(1); DevBuf dx(n*4),dy(n*4),dz(4); dx.up(x.data()); dy.up(y.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*ty=mk({n},ACL_FLOAT,dy.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnHingeEmbeddingLossGetWorkspaceSize(tx,ty,margin,1,tz,w,e);}, aclnnHingeEmbeddingLoss);
      dz.down(hz.data()); double s=0; for(int i=0;i<n;i++) s+=(y[i]>0)?x[i]:std::max(0.0,margin-x[i]); s/=n;
      report("HingeEmbeddingLoss", std::fabs(hz[0]-s)/(std::fabs(s)+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(tz); }
}
static void t_bwd() {
    const int n=200; auto p=randv(n,-2,2),t=randv(n,-2,2); float go=1.5f;
    auto run=[&](const char *name,int sl,double beta,std::function<aclnnStatus(aclTensor*,aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**)> getws,aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
        std::vector<float> hz(n); DevBuf dgo(4),dp(n*4),dt(n*4),dz(n*4); dgo.up(&go); dp.up(p.data()); dt.up(t.data());
        aclTensor *tgo=mk({1},ACL_FLOAT,dgo.p),*tp=mk({n},ACL_FLOAT,dp.p),*tg=mk({n},ACL_FLOAT,dt.p),*tz=mk({n},ACL_FLOAT,dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return getws(tgo,tp,tg,tz,w,e);}, r);
        dz.down(hz.data()); double me=0,mr=0;
        for(int i=0;i<n;i++){ double d=p[i]-t[i],g; if(sl==0) g=d>0?1:(d<0?-1:0); else {double ad=std::fabs(d); g= ad<beta? d/beta : (d>0?1:-1);} double ref=go*g/n;
            me=std::max(me,std::fabs(hz[i]-ref)); mr=std::max(mr,std::fabs(ref)); }
        report(name, me/(mr+1e-9), 1e-5); aclDestroyTensor(tgo);aclDestroyTensor(tp);aclDestroyTensor(tg);aclDestroyTensor(tz);
    };
    run("L1LossBackward",0,0,[](aclTensor*go,aclTensor*p,aclTensor*t,aclTensor*gi,uint64_t*w,aclOpExecutor**e){return aclnnL1LossBackwardGetWorkspaceSize(go,p,t,1,gi,w,e);},aclnnL1LossBackward);
    run("SmoothL1Backward",1,1.0,[](aclTensor*go,aclTensor*p,aclTensor*t,aclTensor*gi,uint64_t*w,aclOpExecutor**e){return aclnnSmoothL1LossBackwardGetWorkspaceSize(go,p,t,1,1.0,gi,w,e);},aclnnSmoothL1LossBackward);
}
static void t_grad_ext() {
    const int n=64; auto x=randv(n,-2,2), go=randv(n,-1,1);
    auto run1=[&](const char*nm,std::function<aclnnStatus(aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**)> gw,
                  aclnnStatus (*rn)(void*,uint64_t,aclOpExecutor*,aclrtStream), std::function<double(double,double)> ref, double tol){
        DevBuf dgo(n*4),dx(n*4),dgi(n*4); dgo.up(go.data()); dx.up(x.data());
        auto tgo=mk({n},ACL_FLOAT,dgo.p),tx=mk({n},ACL_FLOAT,dx.p),tgi=mk({n},ACL_FLOAT,dgi.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tgo,tx,tgi,w,e);}, rn);
        std::vector<float> gi(n); dgi.down(gi.data()); double me=0,mr=0;
        for(int i=0;i<n;i++){ double r=ref(go[i],x[i]); me=std::max(me,std::fabs(gi[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report(nm, me/(mr+1e-9), tol); aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tgi);
    };
    run1("SwishBackward",[](aclTensor*go,aclTensor*x,aclTensor*gi,uint64_t*w,aclOpExecutor**e){return aclnnSwishBackwardGetWorkspaceSize(go,x,1.0,gi,w,e);},aclnnSwishBackward,
         [](double g,double xv){ double s=1.0/(1.0+std::exp(-xv)); return g*(s+xv*s*(1-s)); },1e-5);
    run1("SoftsignBackward",[](aclTensor*go,aclTensor*x,aclTensor*gi,uint64_t*w,aclOpExecutor**e){return aclnnSoftsignBackwardGetWorkspaceSize(go,x,gi,w,e);},aclnnSoftsignBackward,
         [](double g,double xv){ double a=1+std::fabs(xv); return g/(a*a); },1e-5);
    { auto xp=randv(n,0.05,0.95); // logit grad needs x in (0,1)
      DevBuf dgo(n*4),dx(n*4),dgi(n*4); dgo.up(go.data()); dx.up(xp.data());
      auto tgo=mk({n},ACL_FLOAT,dgo.p),tx=mk({n},ACL_FLOAT,dx.p),tgi=mk({n},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogitGradGetWorkspaceSize(tgo,tx,1e-6,tgi,w,e);},aclnnLogitGrad);
      std::vector<float> gi(n); dgi.down(gi.data()); double me=0,mr=0;
      for(int i=0;i<n;i++){ double r=go[i]/(xp[i]*(1-xp[i])); me=std::max(me,std::fabs(gi[i]-r)); mr=std::max(mr,std::fabs(r)); }
      report("LogitGrad", me/(mr+1e-9), 1e-4); aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tgi); }
    { // PreluBackward: gradInput per-element + gradWeight (shared weight)
      const int N=4,C=3,S=2,M=N*C*S; auto xv=randv(M,-2,2),gov=randv(M,-1,1); float wv=0.25f;
      DevBuf dgo(M*4),dx(M*4),dw(4),dgi(M*4),dgw(4); dgo.up(gov.data()); dx.up(xv.data()); dw.up(&wv);
      auto tgo=mk({N,C,S},ACL_FLOAT,dgo.p),tx=mk({N,C,S},ACL_FLOAT,dx.p),tw=mk({1},ACL_FLOAT,dw.p),tgi=mk({N,C,S},ACL_FLOAT,dgi.p),tgw=mk({1},ACL_FLOAT,dgw.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPreluBackwardGetWorkspaceSize(tgo,tx,tw,tgi,tgw,w,e);},aclnnPreluBackward);
      std::vector<float> gi(M); float gw; dgi.down(gi.data()); dgw.down(&gw); double me=0,mr=0,gwref=0;
      for(int i=0;i<M;i++){ double r=xv[i]>0?gov[i]:gov[i]*wv; me=std::max(me,std::fabs(gi[i]-r)); mr=std::max(mr,std::fabs(r)); if(xv[i]<=0) gwref+=gov[i]*xv[i]; }
      double bad=std::max(me/(mr+1e-9), std::fabs(gw-gwref)/(std::fabs(gwref)+1e-9));
      report("PreluBackward", bad, 1e-4); aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tgi);aclDestroyTensor(tgw); }
    { // BCEWithLogitsBackward (reduction none): gi = go*(sigmoid(x)-t)
      auto xv=randv(n,-2,2),tv=randv(n,0,1);
      DevBuf dgo(n*4),dx(n*4),dt(n*4),dgi(n*4); dgo.up(go.data()); dx.up(xv.data()); dt.up(tv.data());
      auto tgo=mk({n},ACL_FLOAT,dgo.p),tx=mk({n},ACL_FLOAT,dx.p),tt=mk({n},ACL_FLOAT,dt.p),tgi=mk({n},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(tgo,tx,tt,nullptr,nullptr,0,tgi,w,e);},aclnnBinaryCrossEntropyWithLogitsBackward);
      std::vector<float> gi(n); dgi.down(gi.data()); double me=0,mr=0;
      for(int i=0;i<n;i++){ double s=1.0/(1.0+std::exp(-xv[i])); double r=go[i]*(s-tv[i]); me=std::max(me,std::fabs(gi[i]-r)); mr=std::max(mr,std::fabs(r)); }
      report("BCEWithLogitsBackward", me/(mr+1e-9), 1e-5); aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tgi); }
    { // CrossEntropyLossGrad: gi[n,c]=(softmax(x)-onehot)*go/N
      const int N=4,C=5; auto xv=randv(N*C,-2,2); std::vector<int64_t> tgt={0,2,4,1}; float go=1.0f;
      DevBuf dx(N*C*4),dtg(N*8),dgi(N*C*4); dx.up(xv.data()); dtg.up(tgt.data());
      auto tx=mk({N,C},ACL_FLOAT,dx.p),tt=mk({N},ACL_INT64,dtg.p),tgi=mk({N,C},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCrossEntropyLossGradGetWorkspaceSize(tx,tt,go,1,tgi,w,e);},aclnnCrossEntropyLossGrad);
      std::vector<float> gi(N*C); dgi.down(gi.data()); double me=0,mr=0;
      for(int nn=0;nn<N;nn++){ double mx=-1e30; for(int c=0;c<C;c++)mx=std::max(mx,(double)xv[nn*C+c]); double sm=0; for(int c=0;c<C;c++)sm+=std::exp(xv[nn*C+c]-mx);
        for(int c=0;c<C;c++){ double so=std::exp(xv[nn*C+c]-mx)/sm; double r=(so-(c==tgt[nn]?1.0:0.0))*go/N; me=std::max(me,std::fabs(gi[nn*C+c]-r)); mr=std::max(mr,std::fabs(r)); } }
      report("CrossEntropyLossGrad", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tgi); }
    { // ScaledMaskedSoftmax fwd: rows of softmax(x*scale)
      const int R=4,D=6; auto xv=randv(R*D,-2,2); float scale=0.5f;
      DevBuf dx(R*D*4),dz(R*D*4); dx.up(xv.data());
      auto tx=mk({R,D},ACL_FLOAT,dx.p),tz=mk({R,D},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScaledMaskedSoftmaxGetWorkspaceSize(tx,nullptr,scale,false,tz,w,e);},aclnnScaledMaskedSoftmax);
      std::vector<float> z(R*D); dz.down(z.data()); double me=0,mr=0;
      for(int r=0;r<R;r++){ double mx=-1e30; for(int d=0;d<D;d++)mx=std::max(mx,(double)xv[r*D+d]*scale); double sm=0; for(int d=0;d<D;d++)sm+=std::exp(xv[r*D+d]*scale-mx);
        for(int d=0;d<D;d++){ double rf=std::exp(xv[r*D+d]*scale-mx)/sm; me=std::max(me,std::fabs(z[r*D+d]-rf)); mr=std::max(mr,std::fabs(rf)); } }
      report("ScaledMaskedSoftmax", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { // GridSampler2DBackward: constant gradOut → gradInput sums to total scattered weight (==#out since bilinear weights sum to 1)
      const int N=1,C=1,H=4,W=4,oH=3,oW=3; std::vector<float> gov(N*C*oH*oW,1.f);
      std::vector<float> grid(N*oH*oW*2); for(int i=0;i<oH*oW;i++){ grid[i*2]= ((i%oW)+0.5f)/oW*2-1; grid[i*2+1]=((i/oW)+0.5f)/oH*2-1; }
      DevBuf dgo(gov.size()*4),dx(N*C*H*W*4),dgr(grid.size()*4),dgi(N*C*H*W*4); dgo.up(gov.data()); dgr.up(grid.data());
      auto tgo=mk({N,C,oH,oW},ACL_FLOAT,dgo.p),tx=mk({N,C,H,W},ACL_FLOAT,dx.p),tgr=mk({N,oH,oW,2},ACL_FLOAT,dgr.p),tgi=mk({N,C,H,W},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGridSampler2DBackwardGetWorkspaceSize(tgo,tx,tgr,0,0,false,tgi,nullptr,w,e);},aclnnGridSampler2DBackward);
      std::vector<float> gi(N*C*H*W); dgi.down(gi.data()); double sum=0; for(float v:gi) sum+=v;
      report("GridSampler2DBackward(sum)", std::fabs(sum-(double)(oH*oW))/(oH*oW), 1e-4); aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tgr);aclDestroyTensor(tgi); }
    { // ThreeInterpolateBackward: gradFeat[b,c,m] = Σ go*weight at idx
      const int B=1,C=2,Nn=3,M=4; auto gov=randv(B*C*Nn,-1,1);
      std::vector<int64_t> idx={0,1,2, 1,2,3, 0,3,1}; std::vector<float> wt={0.5f,0.3f,0.2f, 0.1f,0.6f,0.3f, 0.4f,0.4f,0.2f};
      DevBuf dgo(gov.size()*4),didx(idx.size()*8),dwt(wt.size()*4),dgf(B*C*M*4); dgo.up(gov.data()); didx.up(idx.data()); dwt.up(wt.data());
      auto tgo=mk({B,C,Nn},ACL_FLOAT,dgo.p),ti=mk({B,Nn,3},ACL_INT64,didx.p),tw=mk({B,Nn,3},ACL_FLOAT,dwt.p),tgf=mk({B,C,M},ACL_FLOAT,dgf.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnThreeInterpolateBackwardGetWorkspaceSize(tgo,ti,tw,(int64_t)M,tgf,w,e);},aclnnThreeInterpolateBackward);
      std::vector<float> gf(B*C*M); dgf.down(gf.data()); std::vector<double> ref(B*C*M,0.0);
      for(int c=0;c<C;c++)for(int nn=0;nn<Nn;nn++)for(int k=0;k<3;k++){ int64_t m=idx[nn*3+k]; ref[c*M+m]+=gov[c*Nn+nn]*wt[nn*3+k]; }
      double me=0,mr=0; for(int i=0;i<B*C*M;i++){ me=std::max(me,std::fabs(gf[i]-ref[i])); mr=std::max(mr,std::fabs(ref[i])); }
      report("ThreeInterpolateBackward", me/(mr+1e-9), 1e-5); aclDestroyTensor(tgo);aclDestroyTensor(ti);aclDestroyTensor(tw);aclDestroyTensor(tgf); }
    { // ThnnFusedLstmCell fwd: h=o*tanh(c), c=f*cprev+i*g
      const int Bb=2,Hd=3; auto gates=randv(Bb*4*Hd,-1,1),cp=randv(Bb*Hd,-1,1);
      DevBuf dg(gates.size()*4),dcp(cp.size()*4),dh(Bb*Hd*4),dc(Bb*Hd*4); dg.up(gates.data()); dcp.up(cp.data());
      auto tg=mk({Bb,4*Hd},ACL_FLOAT,dg.p),tcp=mk({Bb,Hd},ACL_FLOAT,dcp.p),th=mk({Bb,Hd},ACL_FLOAT,dh.p),tc=mk({Bb,Hd},ACL_FLOAT,dc.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnThnnFusedLstmCellGetWorkspaceSize(tg,tcp,th,tc,w,e);},aclnnThnnFusedLstmCell);
      std::vector<float> h(Bb*Hd),c(Bb*Hd); dh.down(h.data()); dc.down(c.data()); double me=0,mr=0;
      auto sg=[](double v){return 1.0/(1.0+std::exp(-v));};
      for(int b=0;b<Bb;b++)for(int hh=0;hh<Hd;hh++){ const float*g=&gates[b*4*Hd]; double ig=sg(g[hh]),fg=sg(g[Hd+hh]),gg=std::tanh(g[2*Hd+hh]),og=sg(g[3*Hd+hh]);
        double cn=fg*cp[b*Hd+hh]+ig*gg, hn=og*std::tanh(cn); me=std::max(me,std::fabs(c[b*Hd+hh]-cn)); me=std::max(me,std::fabs(h[b*Hd+hh]-hn)); mr=std::max(mr,std::fabs(hn)); }
      report("ThnnFusedLstmCell", me/(mr+1e-9), 1e-5); aclDestroyTensor(tg);aclDestroyTensor(tcp);aclDestroyTensor(th);aclDestroyTensor(tc); }
    { // ExpSegsum: out[i,j]=exp(Σ_{j<k<=i} x[k]) lower-tri
      const int L=4; auto xv=randv(L,-0.5,0.5); DevBuf dx(L*4),do_(L*L*4); dx.up(xv.data());
      auto tx=mk({L},ACL_FLOAT,dx.p),to=mk({L,L},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnExpSegsumGetWorkspaceSize(tx,to,w,e);},aclnnExpSegsum);
      std::vector<float> o(L*L); do_.down(o.data()); double me=0,mr=0;
      for(int i=0;i<L;i++)for(int j=0;j<L;j++){ double r=0; if(i>=j){ double s=0; for(int k=j+1;k<=i;k++)s+=xv[k]; r=std::exp(s); } me=std::max(me,std::fabs(o[i*L+j]-r)); mr=std::max(mr,std::fabs(r)); }
      report("ExpSegsum", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(to); }
}
int main() {
    init(); srand(23);
    t_basic_losses(); t_margin(); t_bwd(); t_grad_ext();
    return finish();
}
