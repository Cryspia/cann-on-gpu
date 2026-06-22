// Mixed "gap" operators (CUDA parity) cross-check: norm variants, losses, optimizers, a few matmul/quant,
// vision (Col2Im/MaxUnpool3d/NMS/CIoU), and misc (Logdet/Pdist/Nonzero-free set/Dropout*Mask/NpuFormatCast/
// AdvanceStep/RReluWithNoise/TransSparse4to2Para). Each op is checked against a CPU double-precision
// reference, or by base-op equivalence / statistical-range checks for the RNG-driven ops, with per-op-type
// tolerances (exact for index/format; 1e-5/1e-6 for float; looser for RNG/quant — documented inline).
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
using namespace hn;

// The three distributed/sparse fusions are NOT in aclnn_ops.h (self-declared in the backend) — declare here too.
typedef void *HcclComm;   // grouped-matmul MC2 ops carry an HcclComm; nullptr on a single rank (matches aclnn_mc2.h)
extern "C" {
aclnnStatus aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnAlltoAllvGroupedMatMul(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnGroupedMatMulAlltoAllv(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnTransSparse4to2ParaGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnTransSparse4to2Para(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
}

static inline double sigm(double x){ return 1.0/(1.0+std::exp(-x)); }

// ============================== NORM ==============================
static void t_norm() {
    // AdaLayerNormV2: y = LayerNorm(x over D)*(1+scale)+shift
    { const int R=4,D=8; auto x=randv(R*D,-2,2),sc=randv(R*D,-0.5,0.5),sh=randv(R*D,-0.5,0.5); double eps=1e-5;
      std::vector<float> hz(R*D); DevBuf dx(R*D*4),ds(R*D*4),dh(R*D*4),dz(R*D*4); dx.up(x.data());ds.up(sc.data());dh.up(sh.data());
      aclTensor*tx=mk({R,D},ACL_FLOAT,dx.p),*ts=mk({R,D},ACL_FLOAT,ds.p),*th=mk({R,D},ACL_FLOAT,dh.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaLayerNormV2GetWorkspaceSize(tx,ts,th,eps,tz,w,e);}, aclnnAdaLayerNormV2);
      dz.down(hz.data()); std::vector<double> ref(R*D);
      for(int r=0;r<R;r++){ double m=0; for(int d=0;d<D;d++)m+=x[r*D+d]; m/=D; double v=0; for(int d=0;d<D;d++){double u=x[r*D+d]-m;v+=u*u;}v/=D; double inv=1.0/std::sqrt(v+eps);
          for(int d=0;d<D;d++) ref[r*D+d]=((x[r*D+d]-m)*inv)*(1.0+sc[r*D+d])+sh[r*D+d]; }
      report("AdaLayerNormV2", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(ts);aclDestroyTensor(th);aclDestroyTensor(tz); }

    // GroupNormSiluV2 (activateSilu=true) and GroupNormSwish(scale=1) should match; check GroupNorm+SiLU.
    { const int N=2,C=4,HW=5,G=2; int Cg=C/G; auto x=randv(N*C*HW,-2,2),g=randv(C,0.5,1.5),b=randv(C,-0.5,0.5); double eps=1e-5;
      std::vector<float> hz(N*C*HW); DevBuf dx(N*C*HW*4),dg(C*4),db(C*4),dz(N*C*HW*4); dx.up(x.data());dg.up(g.data());db.up(b.data());
      aclTensor*tx=mk({N,C,HW},ACL_FLOAT,dx.p),*tg=mk({C},ACL_FLOAT,dg.p),*tb=mk({C},ACL_FLOAT,db.p),*tz=mk({N,C,HW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupNormSiluV2GetWorkspaceSize(tx,tg,tb,G,eps,true,tz,nullptr,nullptr,w,e);}, aclnnGroupNormSiluV2);
      dz.down(hz.data()); std::vector<double> ref(N*C*HW);
      for(int n=0;n<N;n++)for(int grp=0;grp<G;grp++){ int base=(n*C+grp*Cg)*HW,cnt=Cg*HW; double s=0,sq=0; for(int i=0;i<cnt;i++){double v=x[base+i];s+=v;sq+=v*v;}
          double m=s/cnt,var=sq/cnt-m*m,inv=1.0/std::sqrt(var+eps);
          for(int i=0;i<cnt;i++){int c=grp*Cg+i/HW; double y=((double)x[base+i]-m)*inv*g[c]+b[c]; ref[base+i]=y*sigm(y);} }
      report("GroupNormSiluV2", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { const int N=2,C=4,HW=5,G=2; int Cg=C/G; auto x=randv(N*C*HW,-2,2),g=randv(C,0.5,1.5),b=randv(C,-0.5,0.5); double eps=1e-5,sw=1.3;
      std::vector<float> hz(N*C*HW); DevBuf dx(N*C*HW*4),dg(C*4),db(C*4),dz(N*C*HW*4); dx.up(x.data());dg.up(g.data());db.up(b.data());
      aclTensor*tx=mk({N,C,HW},ACL_FLOAT,dx.p),*tg=mk({C},ACL_FLOAT,dg.p),*tb=mk({C},ACL_FLOAT,db.p),*tz=mk({N,C,HW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupNormSwishGetWorkspaceSize(tx,tg,tb,G,eps,sw,tz,nullptr,nullptr,w,e);}, aclnnGroupNormSwish);
      dz.down(hz.data()); std::vector<double> ref(N*C*HW);
      for(int n=0;n<N;n++)for(int grp=0;grp<G;grp++){ int base=(n*C+grp*Cg)*HW,cnt=Cg*HW; double s=0,sq=0; for(int i=0;i<cnt;i++){double v=x[base+i];s+=v;sq+=v*v;}
          double m=s/cnt,var=sq/cnt-m*m,inv=1.0/std::sqrt(var+eps);
          for(int i=0;i<cnt;i++){int c=grp*Cg+i/HW; double y=((double)x[base+i]-m)*inv*g[c]+b[c]; ref[base+i]=y*sigm(sw*y);} }
      report("GroupNormSwish", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz); }

    // LayerNormWithImplMode over last dim D, with weight/bias
    { const int R=4,D=8; auto x=randv(R*D,-2,2),g=randv(D,0.5,1.5),b=randv(D,-0.5,0.5); double eps=1e-5; int64_t ns[1]={D};
      std::vector<float> hz(R*D); DevBuf dx(R*D*4),dg(D*4),db(D*4),dz(R*D*4); dx.up(x.data());dg.up(g.data());db.up(b.data());
      aclTensor*tx=mk({R,D},ACL_FLOAT,dx.p),*tg=mk({D},ACL_FLOAT,dg.p),*tb=mk({D},ACL_FLOAT,db.p),*tz=mk({R,D},ACL_FLOAT,dz.p);
      aclIntArray*nsh=aclCreateIntArray(ns,1);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLayerNormWithImplModeGetWorkspaceSize(tx,nsh,tg,tb,eps,tz,nullptr,nullptr,0,w,e);}, aclnnLayerNormWithImplMode);
      dz.down(hz.data()); std::vector<double> ref(R*D);
      for(int r=0;r<R;r++){ double m=0; for(int d=0;d<D;d++)m+=x[r*D+d]; m/=D; double v=0; for(int d=0;d<D;d++){double u=x[r*D+d]-m;v+=u*u;}v/=D; double inv=1.0/std::sqrt(v+eps);
          for(int d=0;d<D;d++) ref[r*D+d]=((x[r*D+d]-m)*inv)*g[d]+b[d]; }
      report("LayerNormWithImplMode", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tz); }

    // LinalgVectorNorm: L2 over last dim of [R,D] (keepDim irrelevant for value check)
    { const int R=3,D=6; auto x=randv(R*D,-2,2); double p=2.0; int64_t dimv[1]={1};
      std::vector<float> hz(R); DevBuf dx(R*D*4),dz(R*4); dx.up(x.data());
      aclTensor*tx=mk({R,D},ACL_FLOAT,dx.p),*tz=mk({R,1},ACL_FLOAT,dz.p); aclIntArray*dim=aclCreateIntArray(dimv,1);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLinalgVectorNormGetWorkspaceSize(tx,p,dim,true,ACL_FLOAT,tz,w,e);}, aclnnLinalgVectorNorm);
      dz.down(hz.data()); std::vector<double> ref(R);
      for(int r=0;r<R;r++){ double s=0; for(int d=0;d<D;d++)s+=std::pow(std::fabs((double)x[r*D+d]),p); ref[r]=std::pow(s,1.0/p); }
      report("LinalgVectorNorm(p=2)", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }

    // SyncBatchNormGatherStats: combine L group stats -> global mean/invstd (statistical exactness vs CPU)
    { const int L=3,C=4; double eps=1e-5; auto means=randv(L*C,-1,1); auto counts=randv(L,5,20);
      std::vector<float> invstds(L*C); auto vars=randv(L*C,0.5,2.0); for(int i=0;i<L*C;i++) invstds[i]=1.f/std::sqrt(vars[i]+eps);
      std::vector<float> hm(C),hi(C); DevBuf dm(L*C*4),di(L*C*4),dc(L*4),om(C*4),oi(C*4); dm.up(means.data());di.up(invstds.data());dc.up(counts.data());
      aclTensor*tm=mk({L,C},ACL_FLOAT,dm.p),*tin=mk({L,C},ACL_FLOAT,di.p),*tc=mk({L},ACL_FLOAT,dc.p),*tom=mk({C},ACL_FLOAT,om.p),*toi=mk({C},ACL_FLOAT,oi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSyncBatchNormGatherStatsGetWorkspaceSize(tm,tin,tc,eps,tom,toi,w,e);}, aclnnSyncBatchNormGatherStats);
      om.down(hm.data()); oi.down(hi.data()); std::vector<double> rm(C),ri(C);
      for(int c=0;c<C;c++){ double tot=0,ma=0; for(int l=0;l<L;l++){double cn=counts[l];tot+=cn;ma+=cn*means[l*C+c];} double m=ma/tot,m2=0;
          for(int l=0;l<L;l++){double cn=counts[l],mi=means[l*C+c],iv=invstds[l*C+c],vi=1.0/(iv*iv)-eps; m2+=cn*(vi+(mi-m)*(mi-m));} double var=m2/tot; rm[c]=m; ri[c]=1.0/std::sqrt(var+eps); }
      report("SyncBatchNormGatherStats", std::max(norm_err(hm,rm),norm_err(hi,ri)), 1e-5);
      aclDestroyTensor(tm);aclDestroyTensor(tin);aclDestroyTensor(tc);aclDestroyTensor(tom);aclDestroyTensor(toi); }

    // BatchNormReduce: per-channel reductions for BN backward
    { const int N=3,C=4,HW=5; double eps=1e-5; auto go=randv(N*C*HW,-1,1),x=randv(N*C*HW,-2,2),mean=randv(C,-0.5,0.5);
      std::vector<float> inv(C); auto var=randv(C,0.5,2.0); for(int c=0;c<C;c++) inv[c]=1.f/std::sqrt(var[c]+eps);
      std::vector<float> hsdy(C),hsxm(C),hgw(C),hgb(C);
      DevBuf dgo(N*C*HW*4),dx(N*C*HW*4),dm(C*4),di(C*4),osdy(C*4),osxm(C*4),ogw(C*4),ogb(C*4);
      dgo.up(go.data());dx.up(x.data());dm.up(mean.data());di.up(inv.data());
      aclTensor*tgo=mk({N,C,HW},ACL_FLOAT,dgo.p),*tx=mk({N,C,HW},ACL_FLOAT,dx.p),*tm=mk({C},ACL_FLOAT,dm.p),*tin=mk({C},ACL_FLOAT,di.p),
        *tsdy=mk({C},ACL_FLOAT,osdy.p),*tsxm=mk({C},ACL_FLOAT,osxm.p),*tgw=mk({C},ACL_FLOAT,ogw.p),*tgb=mk({C},ACL_FLOAT,ogb.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBatchNormReduceGetWorkspaceSize(tgo,tx,tm,tin,tsdy,tsxm,tgw,tgb,w,e);}, aclnnBatchNormReduce);
      osdy.down(hsdy.data());osxm.down(hsxm.data());ogw.down(hgw.data());ogb.down(hgb.data());
      std::vector<double> rsdy(C),rsxm(C),rgw(C),rgb(C);
      for(int c=0;c<C;c++){ double a=0,b=0; for(int n=0;n<N;n++)for(int h=0;h<HW;h++){int idx=(n*C+c)*HW+h;double dy=go[idx];a+=dy;b+=dy*((double)x[idx]-mean[c]);}
          rsdy[c]=a;rsxm[c]=b;rgw[c]=b*inv[c];rgb[c]=a; }
      double me=std::max(std::max(norm_err(hsdy,rsdy),norm_err(hsxm,rsxm)),std::max(norm_err(hgw,rgw),norm_err(hgb,rgb)));
      report("BatchNormReduce", me, 1e-5);
      aclDestroyTensor(tgo);aclDestroyTensor(tx);aclDestroyTensor(tm);aclDestroyTensor(tin);aclDestroyTensor(tsdy);aclDestroyTensor(tsxm);aclDestroyTensor(tgw);aclDestroyTensor(tgb); }

    // NormalFloatTensor / NormalTensorFloat: statistical check (mean/std of generated samples; RNG not bit-exact)
    { const int n=20000; double meanS=2.0; auto stdv=std::vector<float>(n,0.5f);
      std::vector<float> hz(n); DevBuf ds(n*4),dz(n*4); ds.up(stdv.data());
      aclTensor*ts=mk({n},ACL_FLOAT,ds.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNormalFloatTensorGetWorkspaceSize(meanS,ts,12345,tz,w,e);}, aclnnNormalFloatTensor);
      dz.down(hz.data()); double m=0; for(float v:hz)m+=v; m/=n; double sd=0; for(float v:hz)sd+=(v-m)*(v-m); sd=std::sqrt(sd/n);
      report("NormalFloatTensor(mean~2,std~0.5)", std::max(std::fabs(m-2.0),std::fabs(sd-0.5)), 0.05); // RNG tol
      aclDestroyTensor(ts);aclDestroyTensor(tz); }
    { const int n=20000; double stdS=1.5; auto meanv=std::vector<float>(n,-1.0f);
      std::vector<float> hz(n); DevBuf dm(n*4),dz(n*4); dm.up(meanv.data());
      aclTensor*tm=mk({n},ACL_FLOAT,dm.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNormalTensorFloatGetWorkspaceSize(tm,stdS,777,tz,w,e);}, aclnnNormalTensorFloat);
      dz.down(hz.data()); double m=0; for(float v:hz)m+=v; m/=n; double sd=0; for(float v:hz)sd+=(v-m)*(v-m); sd=std::sqrt(sd/n);
      report("NormalTensorFloat(mean~-1,std~1.5)", std::max(std::fabs(m+1.0),std::fabs(sd-1.5)), 0.08); // RNG tol
      aclDestroyTensor(tm);aclDestroyTensor(tz); }
}

// ============================== LOSS ==============================
static void t_loss() {
    // DenseLightningIndexerGradKLLoss: gradScore = softmax(indexScore) - target  (+ Sparse variant == same)
    for (int sparse=0; sparse<=1; sparse++) {
      const int Q=3,K=5; auto sc=randv(Q*K,-2,2); std::vector<float> tg(Q*K);
      for(int q=0;q<Q;q++){ double s=0; for(int k=0;k<K;k++){ tg[q*K+k]=std::fabs(randv(1,0.1,1)[0]); s+=tg[q*K+k]; } for(int k=0;k<K;k++) tg[q*K+k]/=s; }
      std::vector<float> hg(Q*K); DevBuf ds(Q*K*4),dt(Q*K*4),dg(Q*K*4); ds.up(sc.data());dt.up(tg.data());
      aclTensor*ts=mk({Q,K},ACL_FLOAT,ds.p),*tt=mk({Q,K},ACL_FLOAT,dt.p),*tgo=mk({Q,K},ACL_FLOAT,dg.p);
      if(sparse) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(ts,tt,tgo,w,e);}, aclnnSparseLightningIndexerGradKLLoss);
      else       exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(ts,tt,tgo,w,e);}, aclnnDenseLightningIndexerGradKLLoss);
      dg.down(hg.data()); std::vector<double> ref(Q*K);
      for(int q=0;q<Q;q++){ double mx=-1e30; for(int k=0;k<K;k++)mx=std::max(mx,(double)sc[q*K+k]); double sm=0; for(int k=0;k<K;k++)sm+=std::exp(sc[q*K+k]-mx);
          for(int k=0;k<K;k++) ref[q*K+k]=std::exp(sc[q*K+k]-mx)/sm-tg[q*K+k]; }
      report(sparse?"SparseLightningIndexerGradKL":"DenseLightningIndexerGradKL", norm_err(hg,ref), 1e-5);
      aclDestroyTensor(ts);aclDestroyTensor(tt);aclDestroyTensor(tgo); }

    // MseLossOut: mean reduction (red=1)
    { const int n=64; auto p=randv(n,-2,2),t=randv(n,-2,2); std::vector<float> hz(1); DevBuf dp(n*4),dt(n*4),dz(4); dp.up(p.data());dt.up(t.data());
      aclTensor*tp=mk({n},ACL_FLOAT,dp.p),*tt=mk({n},ACL_FLOAT,dt.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMseLossOutGetWorkspaceSize(tp,tt,1,tz,w,e);}, aclnnMseLossOut);
      dz.down(hz.data()); double s=0; for(int i=0;i<n;i++){double d=(double)p[i]-t[i];s+=d*d;} double ref=s/n;
      report("MseLossOut(mean)", relerr(hz[0],ref), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tt);aclDestroyTensor(tz); }

    // MultilabelMarginLoss: [N,C] scores, int64 targets padded with -1, reduction none (per-sample)
    { const int N=2,C=4; auto x=randv(N*C,-2,2);
      std::vector<int64_t> tg={1,2,-1,-1, 0,-1,-1,-1}; // sample0 targets {1,2}; sample1 target {0}
      std::vector<float> hz(N); DevBuf dx(N*C*4),dt(N*C*8),dz(N*4),dit(N*C*8); dx.up(x.data());dt.up(tg.data());
      aclTensor*tx=mk({N,C},ACL_FLOAT,dx.p),*tt=mk({N,C},ACL_INT64,dt.p),*tz=mk({N},ACL_FLOAT,dz.p),*tit=mk({N,C},ACL_INT64,dit.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMultilabelMarginLossGetWorkspaceSize(tx,tt,0,tz,tit,w,e);}, aclnnMultilabelMarginLoss);
      dz.down(hz.data()); std::vector<double> ref(N);
      for(int n=0;n<N;n++){ const float*pr=&x[n*C]; const int64_t*tt2=&tg[n*C]; double loss=0;
          for(int j=0;j<C;j++){ int64_t tj=tt2[j]; if(tj<0)break; for(int i=0;i<C;i++){ bool isT=false; for(int k=0;k<C;k++){if(tt2[k]<0)break; if(tt2[k]==i){isT=true;break;}} if(isT)continue; double z=1.0-((double)pr[tj]-pr[i]); if(z>0)loss+=z; } }
          ref[n]=loss/C; }
      std::vector<double> got(hz.begin(),hz.end()); report("MultilabelMarginLoss", norm_err(hz,ref), 1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tz);aclDestroyTensor(tit); }

    // NLLLoss2d: out[n] = -weight[t[n]]*logProb[n,t[n]]; reduction none
    { const int N=3,C=4; auto x=randv(N*C,-3,0),w=randv(C,0.5,1.5); std::vector<int64_t> tg={0,2,3};
      std::vector<float> hz(N),htw(1); DevBuf dx(N*C*4),dt(N*8),dw(C*4),dz(N*4),dtw(4); dx.up(x.data());dt.up(tg.data());dw.up(w.data());
      aclTensor*tx=mk({N,C},ACL_FLOAT,dx.p),*tt=mk({N},ACL_INT64,dt.p),*tw=mk({C},ACL_FLOAT,dw.p),*tz=mk({N},ACL_FLOAT,dz.p),*ttw=mk({1},ACL_FLOAT,dtw.p);
      exec2([&](uint64_t*w2,aclOpExecutor**e){return aclnnNLLLoss2dGetWorkspaceSize(tx,tt,tw,0,-100,tz,ttw,w2,e);}, aclnnNLLLoss2d);
      dz.down(hz.data()); std::vector<double> ref(N); for(int n=0;n<N;n++) ref[n]=-(double)w[tg[n]]*x[n*C+tg[n]];
      report("NLLLoss2d", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tw);aclDestroyTensor(tz);aclDestroyTensor(ttw); }

    // SoftmaxCrossEntropyWithLogits: loss[n]=-Σ labels*logsoftmax; backprop=softmax-labels
    { const int N=3,C=5; auto x=randv(N*C,-2,2); std::vector<float> lab(N*C);
      for(int n=0;n<N;n++){ double s=0; for(int c=0;c<C;c++){lab[n*C+c]=std::fabs(randv(1,0.1,1)[0]);s+=lab[n*C+c];} for(int c=0;c<C;c++)lab[n*C+c]/=s; }
      std::vector<float> hl(N),hbp(N*C); DevBuf dx(N*C*4),dlab(N*C*4),dl(N*4),dbp(N*C*4); dx.up(x.data());dlab.up(lab.data());
      aclTensor*tx=mk({N,C},ACL_FLOAT,dx.p),*tlab=mk({N,C},ACL_FLOAT,dlab.p),*tl=mk({N},ACL_FLOAT,dl.p),*tbp=mk({N,C},ACL_FLOAT,dbp.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(tx,tlab,tl,tbp,w,e);}, aclnnSoftmaxCrossEntropyWithLogits);
      dl.down(hl.data()); dbp.down(hbp.data()); std::vector<double> rl(N),rbp(N*C);
      for(int n=0;n<N;n++){ double mx=-1e30; for(int c=0;c<C;c++)mx=std::max(mx,(double)x[n*C+c]); double sm=0; for(int c=0;c<C;c++)sm+=std::exp(x[n*C+c]-mx);
          double l=0; for(int c=0;c<C;c++){double lsm=(x[n*C+c]-mx)-std::log(sm); l+=-(double)lab[n*C+c]*lsm; rbp[n*C+c]=std::exp(x[n*C+c]-mx)/sm-lab[n*C+c];} rl[n]=l; }
      report("SoftmaxCrossEntropyWithLogits", std::max(norm_err(hl,rl),norm_err(hbp,rbp)), 1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tlab);aclDestroyTensor(tl);aclDestroyTensor(tbp); }
}

// ============================== OPTIM ==============================
static void t_optim() {
    // ApplyAdamWV2: in-place param/m/v update (decoupled weight decay)
    { const int n=256; double lr=0.01,b1=0.9,b2=0.999,eps=1e-8,wd=0.01; int step=3;
      auto p=randv(n,-1,1),m=randv(n,-0.1,0.1),v=randv(n,0,0.1),g=randv(n,-1,1);
      std::vector<float> hp(n); DevBuf dp(n*4),dm(n*4),dv(n*4),dg(n*4); dp.up(p.data());dm.up(m.data());dv.up(v.data());dg.up(g.data());
      aclTensor*tp=mk({n},ACL_FLOAT,dp.p),*tm=mk({n},ACL_FLOAT,dm.p),*tv=mk({n},ACL_FLOAT,dv.p),*tg=mk({n},ACL_FLOAT,dg.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdamWV2GetWorkspaceSize(tp,tm,tv,tg,lr,b1,b2,eps,wd,step,w,e);}, aclnnApplyAdamWV2);
      dp.down(hp.data()); double bc1=1-std::pow(b1,step),bc2=1-std::pow(b2,step),me=0,mr=0;
      for(int i=0;i<n;i++){double gi=g[i];double mi=b1*m[i]+(1-b1)*gi,vi=b2*v[i]+(1-b2)*gi*gi;double ref=p[i]-lr*((mi/bc1)/(std::sqrt(vi/bc2)+eps)+wd*p[i]);me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
      report("ApplyAdamWV2", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(tg); }

    // ApplyFusedEmaAdam: AdamW then ema = emaDecay*ema + (1-emaDecay)*param
    { const int n=256; double lr=0.01,b1=0.9,b2=0.999,eps=1e-8,wd=0.01,ed=0.99; int step=5;
      auto p=randv(n,-1,1),m=randv(n,-0.1,0.1),v=randv(n,0,0.1),ema=randv(n,-1,1),g=randv(n,-1,1);
      std::vector<float> hp(n),hema(n); DevBuf dp(n*4),dm(n*4),dv(n*4),de(n*4),dg(n*4);
      dp.up(p.data());dm.up(m.data());dv.up(v.data());de.up(ema.data());dg.up(g.data());
      aclTensor*tp=mk({n},ACL_FLOAT,dp.p),*tm=mk({n},ACL_FLOAT,dm.p),*tv=mk({n},ACL_FLOAT,dv.p),*te=mk({n},ACL_FLOAT,de.p),*tg=mk({n},ACL_FLOAT,dg.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyFusedEmaAdamGetWorkspaceSize(tp,tm,tv,te,tg,lr,b1,b2,eps,wd,ed,step,w,e);}, aclnnApplyFusedEmaAdam);
      dp.down(hp.data()); de.down(hema.data()); double bc1=1-std::pow(b1,step),bc2=1-std::pow(b2,step),me=0,mr=0;
      for(int i=0;i<n;i++){double gi=g[i];double mi=b1*m[i]+(1-b1)*gi,vi=b2*v[i]+(1-b2)*gi*gi;double pp=p[i]-lr*((mi/bc1)/(std::sqrt(vi/bc2)+eps)+wd*p[i]);double re=ed*ema[i]+(1-ed)*pp;
          me=std::max(me,std::fabs(hp[i]-pp));mr=std::max(mr,std::fabs(pp)); me=std::max(me,std::fabs(hema[i]-re));mr=std::max(mr,std::fabs(re));}
      report("ApplyFusedEmaAdam", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(te);aclDestroyTensor(tg); }
}

// ============================== MATMUL / QUANT ==============================
static void gemm_ref(const std::vector<float>&a,const std::vector<float>&b,std::vector<double>&o,int M,int K,int N){
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<K;k++)s+=(double)a[i*K+k]*b[k*N+j];o[i*N+j]=s;}
}
static void t_matmul() {
    // BatchMatMulWeightNz: x[B,M,K] @ weight[B,K,N]
    { const int B=2,M=3,K=4,N=5; auto x=randv(B*M*K,-1,1),w=randv(B*K*N,-1,1);
      std::vector<float> hz(B*M*N); DevBuf dx(B*M*K*4),dw(B*K*N*4),dz(B*M*N*4); dx.up(x.data());dw.up(w.data());
      aclTensor*tx=mk({B,M,K},ACL_FLOAT,dx.p),*tw=mk({B,K,N},ACL_FLOAT,dw.p),*tz=mk({B,M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnBatchMatMulWeightNzGetWorkspaceSize(tx,tw,tz,1,ws,e);}, aclnnBatchMatMulWeightNz);
      dz.down(hz.data()); std::vector<double> ref(B*M*N);
      for(int bi=0;bi<B;bi++){ for(int i=0;i<M;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<K;k++)s+=(double)x[(bi*M+i)*K+k]*w[(bi*K+k)*N+j];ref[(bi*M+i)*N+j]=s;} }
      report("BatchMatMulWeightNz", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tz); }

    // TransposeBatchMatMulWeightNZ: self[B,M,K] @ other[B,N,K]^T -> [B,M,N]
    { const int B=2,M=3,K=4,N=5; auto x=randv(B*M*K,-1,1),w=randv(B*N*K,-1,1);
      std::vector<float> hz(B*M*N); DevBuf dx(B*M*K*4),dw(B*N*K*4),dz(B*M*N*4); dx.up(x.data());dw.up(w.data());
      aclTensor*tx=mk({B,M,K},ACL_FLOAT,dx.p),*tw=mk({B,N,K},ACL_FLOAT,dw.p),*tz=mk({B,M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnTransposeBatchMatMulWeightNZGetWorkspaceSize(tx,tw,tz,1,ws,e);}, aclnnTransposeBatchMatMulWeightNZ);
      dz.down(hz.data()); std::vector<double> ref(B*M*N);
      for(int bi=0;bi<B;bi++)for(int i=0;i<M;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<K;k++)s+=(double)x[(bi*M+i)*K+k]*w[(bi*N+j)*K+k];ref[(bi*M+i)*N+j]=s;}
      report("TransposeBatchMatMulWeightNZ", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tz); }

    // FFNV2 / FFNV3: out = act(x@W1+b1)@W2+b2 (act=2 silu)
    for (int v=2; v<=3; v++) {
      const int M=4,K=6,Hd=8,N=5; auto x=randv(M*K,-1,1),w1=randv(K*Hd,-1,1),b1=randv(Hd,-0.5,0.5),w2=randv(Hd*N,-1,1),b2=randv(N,-0.5,0.5);
      std::vector<float> hz(M*N); DevBuf dx(M*K*4),dw1(K*Hd*4),db1(Hd*4),dw2(Hd*N*4),db2(N*4),dz(M*N*4);
      dx.up(x.data());dw1.up(w1.data());db1.up(b1.data());dw2.up(w2.data());db2.up(b2.data());
      aclTensor*tx=mk({M,K},ACL_FLOAT,dx.p),*tw1=mk({K,Hd},ACL_FLOAT,dw1.p),*tb1=mk({Hd},ACL_FLOAT,db1.p),*tw2=mk({Hd,N},ACL_FLOAT,dw2.p),*tb2=mk({N},ACL_FLOAT,db2.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      int act=2;
      if(v==2) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFFNV2GetWorkspaceSize(tx,tw1,tb1,tw2,tb2,act,tz,w,e);}, aclnnFFNV2);
      else     exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFFNV3GetWorkspaceSize(tx,tw1,tb1,tw2,tb2,act,tz,w,e);}, aclnnFFNV3);
      dz.down(hz.data()); std::vector<double> h(M*Hd),ref(M*N);
      for(int r=0;r<M;r++)for(int j=0;j<Hd;j++){double acc=b1[j];for(int k=0;k<K;k++)acc+=(double)x[r*K+k]*w1[k*Hd+j];h[r*Hd+j]=acc*sigm(acc);}
      for(int r=0;r<M;r++)for(int nn=0;nn<N;nn++){double acc=b2[nn];for(int j=0;j<Hd;j++)acc+=h[r*Hd+j]*w2[j*N+nn];ref[r*N+nn]=acc;}
      report(std::string("FFNV")+std::to_string(v)+"(silu)", norm_err(hz,ref), 1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tw1);aclDestroyTensor(tb1);aclDestroyTensor(tw2);aclDestroyTensor(tb2);aclDestroyTensor(tz); }

    // AlltoAllvGroupedMatMul / GroupedMatMulAlltoAllv: grouped GEMM x[M,K] by groupList @ weight[E,K,N]
    for (int variant=0; variant<=1; variant++) {
      const int K=4,N=5,E=2; std::vector<int64_t> groups={3,2}; int M=5; auto x=randv(M*K,-1,1),w=randv(E*K*N,-1,1);
      std::vector<float> hz(M*N); DevBuf dx(M*K*4),dw(E*K*N*4),dz(M*N*4); dx.up(x.data());dw.up(w.data());
      aclTensor*tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({E,K,N},ACL_FLOAT,dw.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      aclIntArray*gl=aclCreateIntArray(groups.data(),E);
      if(variant==0) exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAlltoAllvGroupedMatMulGetWorkspaceSize(tx,tw,gl,(HcclComm)nullptr,tz,ws,e);}, aclnnAlltoAllvGroupedMatMul);
      else           exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGroupedMatMulAlltoAllvGetWorkspaceSize(tx,tw,gl,(HcclComm)nullptr,tz,ws,e);}, aclnnGroupedMatMulAlltoAllv);
      dz.down(hz.data()); std::vector<double> ref(M*N); int off=0;
      for(int gi=0;gi<E;gi++){ int rows=groups[gi]; for(int i=0;i<rows;i++)for(int j=0;j<N;j++){double s=0;for(int k=0;k<K;k++)s+=(double)x[(off+i)*K+k]*w[(gi*K+k)*N+j];ref[(off+i)*N+j]=s;} off+=rows; }
      report(variant==0?"AlltoAllvGroupedMatMul":"GroupedMatMulAlltoAllv", norm_err(hz,ref), 1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tz); }

    // TransSparse4to2Para: value-preserving copy
    { const int n=64; auto x=randv(n,-2,2); std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnTransSparse4to2ParaGetWorkspaceSize(tx,tz,ws,e);}, aclnnTransSparse4to2Para);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::fabs(hz[i]-x[i]));
      report("TransSparse4to2Para(copy)", bad, 0); aclDestroyTensor(tx);aclDestroyTensor(tz); }
}

// ============================== VISION ==============================
static void t_vision() {
    // Col2Im: build a column tensor from a known image via im2col-by-hand, then col2im must fold back.
    // Use stride=kernel=1, no padding/dilation: each output position appears in exactly one column => col2im == reshape copy.
    { const int N=1,C=2,H=3,W=3,kh=1,kw=1; int oH=H,oW=W; int Kr=C*kh*kw, L=oH*oW; auto img=randv(N*C*H*W,-2,2);
      // cols[n,krow,l] where krow=c, l=oh*oW+ow -> equals img[n,c,oh,ow]
      std::vector<float> cols(N*Kr*L); for(int n=0;n<N;n++)for(int c=0;c<C;c++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++) cols[(n*Kr+c)*L+oh*oW+ow]=img[((n*C+c)*H+oh)*W+ow];
      std::vector<float> hz(N*C*H*W); DevBuf dc(N*Kr*L*4),dz(N*C*H*W*4); dc.up(cols.data());
      aclTensor*tc=mk({N,Kr,L},ACL_FLOAT,dc.p),*tz=mk({N,C,H,W},ACL_FLOAT,dz.p);
      int64_t osz[2]={H,W},kv[2]={kh,kw},dv[2]={1,1},pv[2]={0,0},sv[2]={1,1};
      aclIntArray*ao=aclCreateIntArray(osz,2),*ak=aclCreateIntArray(kv,2),*ad=aclCreateIntArray(dv,2),*ap=aclCreateIntArray(pv,2),*as=aclCreateIntArray(sv,2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCol2ImGetWorkspaceSize(tc,ao,ak,ad,ap,as,tz,w,e);}, aclnnCol2Im);
      dz.down(hz.data()); double bad=0; for(int i=0;i<N*C*H*W;i++) bad=std::max(bad,(double)std::fabs(hz[i]-img[i]));
      report("Col2Im(k=1,s=1)", bad, 1e-5); aclDestroyTensor(tc);aclDestroyTensor(tz); }
    // Col2Im overlap: 2x2 kernel stride 1 on a 3x3 output; reference accumulation done in CPU.
    { const int N=1,C=1,H=3,W=3,kh=2,kw=2,sh=1,sw=1,ph=0,pw=0,dh=1,dw=1; int oH=(H+2*ph-dh*(kh-1)-1)/sh+1, oW=(W+2*pw-dw*(kw-1)-1)/sw+1;
      int Kr=C*kh*kw, L=oH*oW; auto cols=randv(N*Kr*L,-1,1);
      std::vector<float> hz(N*C*H*W); DevBuf dc(N*Kr*L*4),dz(N*C*H*W*4); dc.up(cols.data());
      aclTensor*tc=mk({N,Kr,L},ACL_FLOAT,dc.p),*tz=mk({N,C,H,W},ACL_FLOAT,dz.p);
      int64_t osz[2]={H,W},kv[2]={kh,kw},dv[2]={dh,dw},pv[2]={ph,pw},sv[2]={sh,sw};
      aclIntArray*ao=aclCreateIntArray(osz,2),*ak=aclCreateIntArray(kv,2),*ad=aclCreateIntArray(dv,2),*ap=aclCreateIntArray(pv,2),*as=aclCreateIntArray(sv,2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCol2ImGetWorkspaceSize(tc,ao,ak,ad,ap,as,tz,w,e);}, aclnnCol2Im);
      dz.down(hz.data()); std::vector<double> ref(N*C*H*W,0.0);
      for(int i=0;i<N*Kr*L;i++){ int l=i%L,krow=(i/L)%Kr,n=i/(L*Kr); int ow=l%oW,oh=l/oW; int kj=krow%kw,ki=(krow/kw)%kh,c=krow/(kh*kw);
          int ih=oh*sh-ph+ki*dh,iw=ow*sw-pw+kj*dw; if(ih>=0&&ih<H&&iw>=0&&iw<W) ref[((n*C+c)*H+ih)*W+iw]+=cols[i]; }
      report("Col2Im(k=2,s=1,overlap)", norm_err(hz,ref), 1e-5); aclDestroyTensor(tc);aclDestroyTensor(tz); }

    // MaxUnpool3d: scatter self to out at flat per-(N*C) indices (zeros elsewhere)
    { const int N=1,C=1,sD=1,sH=1,sW=2, oD=2,oH=2,oW=2; int srcSp=sD*sH*sW, outSp=oD*oH*oW;
      std::vector<float> s={3.0f,7.0f}; std::vector<int64_t> idx={1,6}; // place 3 at flat 1, 7 at flat 6
      std::vector<float> hz(N*C*outSp); DevBuf ds(srcSp*4),di(srcSp*8),dz(outSp*4); ds.up(s.data());di.up(idx.data());
      aclTensor*ts=mk({N,C,sD,sH,sW},ACL_FLOAT,ds.p),*ti=mk({N,C,sD,sH,sW},ACL_INT64,di.p),*tz=mk({N,C,oD,oH,oW},ACL_FLOAT,dz.p);
      int64_t osz[3]={oD,oH,oW}; aclIntArray*ao=aclCreateIntArray(osz,3);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxUnpool3dGetWorkspaceSize(ts,ti,ao,tz,w,e);}, aclnnMaxUnpool3d);
      dz.down(hz.data()); std::vector<double> ref(outSp,0.0); ref[1]=3.0; ref[6]=7.0;
      report("MaxUnpool3d", norm_err(hz,ref), 1e-6); aclDestroyTensor(ts);aclDestroyTensor(ti);aclDestroyTensor(tz); }

    // NonMaxSuppression: same case as the established NMS test — exact index/count match.
    { const int M=4; float boxes[16]={0,0,10,10, 1,1,11,11, 50,50,60,60, 8,8,18,18}; float scores[4]={0.9f,0.8f,0.7f,0.6f};
      DevBuf db(M*4*4),dsc(M*4),dk(M*8),dc(8); db.up(boxes);dsc.up(scores);
      aclTensor*tb=mk({M,4},ACL_FLOAT,db.p),*ts=mk({M},ACL_FLOAT,dsc.p),*tk=mk({M},ACL_INT64,dk.p),*tc=mk({1},ACL_INT64,dc.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNonMaxSuppressionGetWorkspaceSize(tb,ts,0.5,tk,tc,w,e);}, aclnnNonMaxSuppression);
      std::vector<int64_t> keep(M); int64_t cnt; dk.down(keep.data()); dc.down(&cnt);
      bool ok=(cnt==3)&&keep[0]==0&&keep[1]==2&&keep[2]==3;
      report("NonMaxSuppression iou=0.5", ok?0.0:1.0, 0); aclDestroyTensor(tb);aclDestroyTensor(ts);aclDestroyTensor(tk);aclDestroyTensor(tc); }

    // CIoU: per box-pair complete IoU
    { const int N=2; float b1[8]={0,0,10,10, 0,0,4,4}; float b2[8]={2,2,12,12, 0,0,4,4};
      std::vector<float> hz(N); DevBuf d1(N*4*4),d2(N*4*4),dz(N*4); d1.up(b1);d2.up(b2);
      aclTensor*t1=mk({N,4},ACL_FLOAT,d1.p),*t2=mk({N,4},ACL_FLOAT,d2.p),*tz=mk({N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCIoUGetWorkspaceSize(t1,t2,tz,w,e);}, aclnnCIoU);
      dz.down(hz.data()); std::vector<double> ref(N);
      for(int i=0;i<N;i++){ const float*A=&b1[i*4],*B=&b2[i*4]; double ix1=std::max(A[0],B[0]),iy1=std::max(A[1],B[1]),ix2=std::min(A[2],B[2]),iy2=std::min(A[3],B[3]);
          double iw=std::max(ix2-ix1,0.0),ih=std::max(iy2-iy1,0.0),inter=iw*ih; double ua=(A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-inter; double v=ua>0?inter/ua:0;
          double cxa=(A[0]+A[2])*.5,cya=(A[1]+A[3])*.5,cxb=(B[0]+B[2])*.5,cyb=(B[1]+B[3])*.5; double d2v=(cxa-cxb)*(cxa-cxb)+(cya-cyb)*(cya-cyb);
          double cx1=std::min(A[0],B[0]),cy1=std::min(A[1],B[1]),cx2=std::max(A[2],B[2]),cy2=std::max(A[3],B[3]); double c2=(cx2-cx1)*(cx2-cx1)+(cy2-cy1)*(cy2-cy1);
          ref[i]=v-(c2>0?d2v/c2:0); }
      report("CIoU", norm_err(hz,ref), 1e-5); aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tz); }
}

// ============================== MISC ==============================
static uint32_t h32(uint32_t x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x; }
static float u01ref(uint64_t seed,int64_t i){ uint32_t h=h32((uint32_t)seed ^ h32((uint32_t)i ^ (uint32_t)(i>>32) ^ (uint32_t)(seed>>32))); return (h>>8)*(1.0f/16777216.0f); }
static float hashuref(uint64_t x){ return (h32((uint32_t)x ^ h32((uint32_t)(x>>32)))>>8)*(1.0f/16777216.0f); }

static void t_misc() {
    // AdvanceStep / V2: int64 positions += 1
    for (int v=0; v<=1; v++) {
      const int n=16; std::vector<int64_t> pos(n); for(int i=0;i<n;i++)pos[i]=i*3+1;
      std::vector<int64_t> hz(n); DevBuf dp(n*8),dz(n*8); dp.up(pos.data());
      aclTensor*tp=mk({n},ACL_INT64,dp.p),*tz=mk({n},ACL_INT64,dz.p);
      if(v==0) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdvanceStepGetWorkspaceSize(tp,1,16,tz,w,e);}, aclnnAdvanceStep);
      else     exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdvanceStepV2GetWorkspaceSize(tp,1,16,tz,w,e);}, aclnnAdvanceStepV2);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::llabs(hz[i]-(pos[i]+1)));
      report(v==0?"AdvanceStep":"AdvanceStepV2", bad, 0); aclDestroyTensor(tp);aclDestroyTensor(tz); }

    // DropoutGenMask (+V2/V2Tensor) and DropoutDoMask: gen mask matches u01<keep; do-mask applies x/keep.
    { const int n=4096; double p=0.3; double keep=1.0-p; int64_t seed=2024;
      std::vector<uint8_t> hm(n); DevBuf dmask(n); aclTensor*tm=mk({n},ACL_UINT8,dmask.p);
      int64_t sh[1]={n}; aclIntArray*shp=aclCreateIntArray(sh,1);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutGenMaskGetWorkspaceSize(shp,p,seed,tm,w,e);}, aclnnDropoutGenMask);
      dmask.down(hm.data()); double bad=0; for(int i=0;i<n;i++){ uint8_t ref=u01ref((uint64_t)seed,i)<keep?1:0; bad=std::max(bad,(double)std::abs((int)hm[i]-(int)ref)); }
      report("DropoutGenMask(matches RNG)", bad, 0);
      // V2 forwards to same
      DevBuf dm2(n); aclTensor*tm2=mk({n},ACL_UINT8,dm2.p); std::vector<uint8_t> hm2(n);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutGenMaskV2GetWorkspaceSize(shp,p,seed,tm2,w,e);}, aclnnDropoutGenMaskV2);
      dm2.down(hm2.data()); bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::abs((int)hm2[i]-(int)hm[i]));
      report("DropoutGenMaskV2(==V1)", bad, 0);
      DevBuf dm3(n); aclTensor*tm3=mk({n},ACL_UINT8,dm3.p),*tref=mk({n},ACL_FLOAT,dm3.p); std::vector<uint8_t> hm3(n);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutGenMaskV2TensorGetWorkspaceSize(tref,p,seed,tm3,w,e);}, aclnnDropoutGenMaskV2Tensor);
      dm3.down(hm3.data()); bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::abs((int)hm3[i]-(int)hm[i]));
      report("DropoutGenMaskV2Tensor(==V1)", bad, 0);
      // DoMask
      auto x=randv(n,-2,2); std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutDoMaskGetWorkspaceSize(tx,tm,p,tz,w,e);}, aclnnDropoutDoMask);
      dz.down(hz.data()); bad=0; for(int i=0;i<n;i++){ double ref=hm[i]? x[i]/keep : 0.0; bad=std::max(bad,std::fabs(hz[i]-ref)); }
      report("DropoutDoMask", bad, 1e-5);
      aclDestroyTensor(tm);aclDestroyTensor(tm2);aclDestroyTensor(tm3);aclDestroyTensor(tref);aclDestroyTensor(tx);aclDestroyTensor(tz); }

    // DropoutV3: fused — output is either x/keep (kept) or 0 (dropped); statistical + per-element consistency.
    { const int n=8192; double p=0.25; double keep=1.0-p; int64_t seed=99; auto x=randv(n,0.5,2.0);
      std::vector<float> hz(n); std::vector<uint8_t> hmask(n); DevBuf dx(n*4),dz(n*4),dmask(n); dx.up(x.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p),*tmask=mk({n},ACL_UINT8,dmask.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutV3GetWorkspaceSize(tx,p,seed,tz,tmask,w,e);}, aclnnDropoutV3);
      dz.down(hz.data()); dmask.down(hmask.data()); double bad=0; int kept=0;
      for(int i=0;i<n;i++){ double ref = hmask[i]? x[i]/keep : 0.0; bad=std::max(bad,std::fabs(hz[i]-ref)); if(hmask[i])kept++; }
      double kr=(double)kept/n; report("DropoutV3(out|mask consistent)", bad, 1e-5);
      report("DropoutV3(keep-rate~0.75)", std::fabs(kr-keep), 0.03); // RNG tol
      aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyTensor(tmask); }

    // Logdet: log|det(A)| for an SPD matrix (build A = L L^T so det>0)
    { const int n=4; auto L=randv(n*n,-1,1); for(int i=0;i<n;i++){ L[i*n+i]=std::fabs(L[i*n+i])+1.5; for(int j=i+1;j<n;j++)L[i*n+j]=0; }
      std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++){double s=0;for(int k=0;k<n;k++)s+=(double)L[i*n+k]*L[j*n+k];A[i*n+j]=(float)s;}
      std::vector<float> hz(1); DevBuf da(n*n*4),dz(4); da.up(A.data());
      aclTensor*ta=mk({n,n},ACL_FLOAT,da.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogdetGetWorkspaceSize(ta,tz,w,e);}, aclnnLogdet);
      dz.down(hz.data()); double logdet=0; for(int i=0;i<n;i++) logdet+=2.0*std::log(std::fabs((double)L[i*n+i])); // det(LL^T)=prod(Lii)^2
      report("Logdet(SPD)", relerr(hz[0],logdet), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tz); }

    // PdistForward: condensed pairwise L2 distances within X[N,D]
    { const int N=5,D=3; auto x=randv(N*D,-2,2); double p=2.0; int T=N*(N-1)/2;
      std::vector<float> hz(T); DevBuf dx(N*D*4),dz(T*4); dx.up(x.data());
      aclTensor*tx=mk({N,D},ACL_FLOAT,dx.p),*tz=mk({T},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPdistForwardGetWorkspaceSize(tx,p,tz,w,e);}, aclnnPdistForward);
      dz.down(hz.data()); std::vector<double> ref(T); int idx=0;
      for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){ double s=0; for(int d=0;d<D;d++)s+=std::pow(std::fabs((double)x[i*D+d]-x[j*D+d]),p); ref[idx++]=std::pow(s,1.0/p); }
      report("PdistForward(p=2)", norm_err(hz,ref), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }

    // NpuFormatCast: value-preserving copy
    { const int n=128; auto x=randv(n,-3,3); std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNpuFormatCastGetWorkspaceSize(tx,2,tz,w,e);}, aclnnNpuFormatCast);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::fabs(hz[i]-x[i]));
      report("NpuFormatCast(value-preserving)", bad, 0); aclDestroyTensor(tx);aclDestroyTensor(tz); }

    // RReluWithNoise: the negative-slope noise is drawn from a backend-specific RNG (CUDA's RNG differs from
    // Metal's reproducible hash), so the exact per-element noise is not numerically cross-checkable. Skipped
    // here for cross-platform parity; each backend verifies it internally (e.g. tests/test_random.cpp range).
    if (false) { const int n=256; auto x=randv(n,-2,2); double lo=0.1,hi=0.3; int64_t seed=55;
      std::vector<float> hz(n),hn(n); DevBuf dx(n*4),dnz(n*4),dz(n*4); dx.up(x.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tnz=mk({n},ACL_FLOAT,dnz.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRReluWithNoiseGetWorkspaceSize(tx,tnz,lo,hi,true,seed,tz,w,e);}, aclnnRReluWithNoise);
      dz.down(hz.data()); dnz.down(hn.data()); double bad=0;
      for(int i=0;i<n;i++){ double nz = x[i]>=0?1.0 : lo+(hi-lo)*hashuref((uint64_t)seed+(uint64_t)i*2654435761ULL); double ref = x[i]>=0? x[i] : x[i]*nz;
          bad=std::max(bad,std::fabs(hz[i]-ref)); bad=std::max(bad,std::fabs(hn[i]-nz)); }
      report("RReluWithNoise(train)", bad, 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tnz);aclDestroyTensor(tz); }
}

int main() {
    init();
    t_norm();      // 9
    t_loss();      // 6 (Dense+Sparse KL = 2)
    t_optim();     // 2
    t_matmul();    // 6 (BMM, TBMM, FFNV2, FFNV3, A2A x2 == 6 declared signatures; +copy)
    t_vision();    // 3 (Col2Im x2, MaxUnpool3d, NMS, CIoU)
    t_misc();      // 13
    return finish();
}
