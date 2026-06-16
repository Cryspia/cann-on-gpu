// MoE routing (P15) cross-check: GatingTopKSoftmax, ComputeExpertTokens, TokenPermute+Unpermute round-trip.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_gating() {
    const int T=32,E=8,K=2; auto lg=randv(T*E,-3,3);
    std::vector<float> hw(T*K); std::vector<int32_t> hi(T*K);
    DevBuf dl(T*E*4),dw(T*K*4),di(T*K*4); dl.up(lg.data());
    aclTensor *tl=mk({T,E},ACL_FLOAT,dl.p),*tw=mk({T,K},ACL_FLOAT,dw.p),*ti=mk({T,K},ACL_INT32,di.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(tl,K,tw,ti,w,e);}, aclnnMoeGatingTopKSoftmax);
    dw.down(hw.data()); di.down(hi.data());
    double me=0,mr=0; int64_t ibad=0;
    for(int t=0;t<T;t++){ double mx=-1e30; for(int e=0;e<E;e++) mx=std::max(mx,(double)lg[t*E+e]); double se=0; for(int e=0;e<E;e++) se+=std::exp(lg[t*E+e]-mx);
        std::vector<std::pair<double,int>> v; for(int e=0;e<E;e++) v.push_back({(double)lg[t*E+e],e});
        std::sort(v.begin(),v.end(),[](auto&a,auto&b){return a.first>b.first;});
        double wsum=0; std::vector<double> p(K); for(int k=0;k<K;k++){ p[k]=std::exp(v[k].first-mx)/se; wsum+=p[k]; if(hi[t*K+k]!=v[k].second) ibad++; }
        for(int k=0;k<K;k++){ double ref=p[k]/wsum; me=std::max(me,std::fabs(hw[t*K+k]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("MoeGatingTopKSoftmax w", me/(mr+1e-9), 1e-5);
    report("MoeGatingTopKSoftmax idx", ibad?1.0:0.0, 0);
    aclDestroyTensor(tl);aclDestroyTensor(tw);aclDestroyTensor(ti);
}
static void t_compute_tokens() {
    const int M=40,E=6; std::vector<int32_t> idx(M); for(auto&v:idx) v=rand()%E;
    std::vector<int64_t> hc(E),ho(E); DevBuf di(M*4),dc(E*8),do_(E*8); di.up(idx.data());
    aclTensor *ti=mk({M},ACL_INT32,di.p),*tc=mk({E},ACL_INT64,dc.p),*to=mk({E},ACL_INT64,do_.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeComputeExpertTokensGetWorkspaceSize(ti,E,tc,to,w,e);}, aclnnMoeComputeExpertTokens);
    dc.down(hc.data()); do_.down(ho.data());
    std::vector<int64_t> cnt(E,0); for(int v:idx) cnt[v]++; int64_t bad=0,acc=0;
    for(int e=0;e<E;e++){ if(hc[e]!=cnt[e]) bad++; if(ho[e]!=acc) bad++; acc+=cnt[e]; }
    report("MoeComputeExpertTokens", bad?1.0:0.0, 0); aclDestroyTensor(ti);aclDestroyTensor(tc);aclDestroyTensor(to);
}
static void t_permute_roundtrip() {
    const int T=24,H=5,E=4; auto x=randv(T*H,-2,2); std::vector<int32_t> ex(T); for(auto&v:ex) v=rand()%E;
    std::vector<float> hperm(T*H),hout(T*H); std::vector<int64_t> hsrc(T);
    DevBuf dx(T*H*4),dex(T*4),dperm(T*H*4),dsrc(T*8),dout(T*H*4); dx.up(x.data()); dex.up(ex.data());
    aclTensor *tx=mk({T,H},ACL_FLOAT,dx.p),*tex=mk({T},ACL_INT32,dex.p),*tperm=mk({T,H},ACL_FLOAT,dperm.p),*tsrc=mk({T},ACL_INT64,dsrc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeTokenPermuteGetWorkspaceSize(tx,tex,E,tperm,tsrc,w,e);}, aclnnMoeTokenPermute);
    dperm.down(hperm.data()); dsrc.down(hsrc.data());
    // verify grouping: experts along permuted order are non-decreasing
    bool grouped=true; int prev=-1; for(int p=0;p<T;p++){ int e=ex[hsrc[p]]; if(e<prev) grouped=false; prev=e; }
    report("MoeTokenPermute grouped", grouped?0.0:1.0, 0);
    // unpermute round-trip (weight=null) reconstructs x
    aclTensor *tperm2=mk({T,H},ACL_FLOAT,dperm.p),*tsrc2=mk({T},ACL_INT64,dsrc.p),*tout=mk({T,H},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeTokenUnpermuteGetWorkspaceSize(tperm2,tsrc2,nullptr,tout,w,e);}, aclnnMoeTokenUnpermute);
    dout.down(hout.data()); double bad=0; for(int i=0;i<T*H;i++) bad=std::max(bad,(double)std::fabs(hout[i]-x[i]));
    report("MoeTokenUnpermute roundtrip", bad, 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tex);aclDestroyTensor(tperm);aclDestroyTensor(tsrc);aclDestroyTensor(tperm2);aclDestroyTensor(tsrc2);aclDestroyTensor(tout);
}
static void t_rope_grad() {
    // RoPE is orthogonal: forward then backward (un-rotate) recovers the input. cos/sin equal across the half-pair (valid rotation).
    const int B=1,N=2,S=4,D=8,half=D/2; auto q=randv(B*N*S*D,-1,1);
    std::vector<float> cos(S*D),sin(S*D);
    for(int s=0;s<S;s++)for(int dh=0;dh<half;dh++){ double ang=(s+1)*0.1*(dh+1); float c=std::cos(ang),si=std::sin(ang); cos[s*D+dh]=c; cos[s*D+dh+half]=c; sin[s*D+dh]=si; sin[s*D+dh+half]=si; }
    std::vector<float> qb(B*N*S*D);
    DevBuf dq(B*N*S*D*4),dk(B*N*S*D*4),dc(S*D*4),ds(S*D*4),dqr(B*N*S*D*4),dkr(B*N*S*D*4),dqb(B*N*S*D*4),dkb(B*N*S*D*4);
    dq.up(q.data()); dk.up(q.data()); dc.up(cos.data()); ds.up(sin.data());
    aclTensor *tq=mk({B,N,S,D},ACL_FLOAT,dq.p),*tk=mk({B,N,S,D},ACL_FLOAT,dk.p),*tc=mk({S,D},ACL_FLOAT,dc.p),*ts=mk({S,D},ACL_FLOAT,ds.p),*tqr=mk({B,N,S,D},ACL_FLOAT,dqr.p),*tkr=mk({B,N,S,D},ACL_FLOAT,dkr.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,0,tqr,tkr,w,e);}, aclnnApplyRotaryPosEmb);
    aclTensor *tqr2=mk({B,N,S,D},ACL_FLOAT,dqr.p),*tkr2=mk({B,N,S,D},ACL_FLOAT,dkr.p),*tqb=mk({B,N,S,D},ACL_FLOAT,dqb.p),*tkb=mk({B,N,S,D},ACL_FLOAT,dkb.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGradGetWorkspaceSize(tqr2,tkr2,tc,ts,0,tqb,tkb,w,e);}, aclnnApplyRotaryPosEmbGrad);
    dqb.down(qb.data()); double me=0,mr=0; for(int i=0;i<B*N*S*D;i++){me=std::max(me,(double)std::fabs(qb[i]-q[i]));mr=std::max(mr,std::fabs((double)q[i]));}
    report("ApplyRotaryPosEmbGrad (un-rotate)", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tc);aclDestroyTensor(ts);aclDestroyTensor(tqr);aclDestroyTensor(tkr);aclDestroyTensor(tqr2);aclDestroyTensor(tkr2);aclDestroyTensor(tqb);aclDestroyTensor(tkb);
}
static void t_fia() {
    // FusedInferAttentionScore routes to prompt flash attention; verify it equals aclnnPromptFlashAttention.
    const int B=1,Nh=2,S=4,D=8; auto q=randv(B*Nh*S*D,-1,1),k=randv(B*Nh*S*D,-1,1),v=randv(B*Nh*S*D,-1,1);
    double scale=1.0/std::sqrt((double)D);
    std::vector<float> a(B*Nh*S*D),b(B*Nh*S*D);
    DevBuf dq(B*Nh*S*D*4),dk(B*Nh*S*D*4),dv(B*Nh*S*D*4),da(B*Nh*S*D*4),db(B*Nh*S*D*4); dq.up(q.data());dk.up(k.data());dv.up(v.data());
    aclTensor *tq=mk({B,Nh,S,D},ACL_FLOAT,dq.p),*tk=mk({B,Nh,S,D},ACL_FLOAT,dk.p),*tv=mk({B,Nh,S,D},ACL_FLOAT,dv.p);
    aclTensor *toa=mk({B,Nh,S,D},ACL_FLOAT,da.p),*tob=mk({B,Nh,S,D},ACL_FLOAT,db.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPromptFlashAttentionGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,true,toa,w,e);}, aclnnPromptFlashAttention);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFusedInferAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,true,tob,w,e);}, aclnnFusedInferAttentionScore);
    da.down(a.data()); db.down(b.data()); double bad=0; for(int i=0;i<B*Nh*S*D;i++) bad=std::max(bad,(double)std::fabs(a[i]-b[i]));
    report("FusedInferAttentionScore==Prompt", bad, 0);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(toa);aclDestroyTensor(tob);
}
static void t_mla() {
    const int B=1,Nh=2,S=4,D=8,Lc=6; double scale=1.0/std::sqrt((double)D);
    auto q=randv(B*Nh*S*D,-1,1),cKV=randv(B*S*Lc,-1,1),wUK=randv(Lc*Nh*D,-0.5,0.5),wUV=randv(Lc*Nh*D,-0.5,0.5);
    std::vector<float> ho(B*Nh*S*D);
    DevBuf dq(q.size()*4),dc(cKV.size()*4),dk(wUK.size()*4),dv(wUV.size()*4),doo(ho.size()*4);
    dq.up(q.data());dc.up(cKV.data());dk.up(wUK.data());dv.up(wUV.data());
    aclTensor *tq=mk({B,Nh,S,D},ACL_FLOAT,dq.p),*tc=mk({B,S,Lc},ACL_FLOAT,dc.p),*tk=mk({Lc,Nh*D},ACL_FLOAT,dk.p),*tv=mk({Lc,Nh*D},ACL_FLOAT,dv.p),*to=mk({B,Nh,S,D},ACL_FLOAT,doo.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMultiLatentAttentionGetWorkspaceSize(tq,tc,tk,tv,scale,Nh,true,to,w,e);}, aclnnMultiLatentAttention);
    doo.down(ho.data());
    // CPU reference
    auto K=[&](int h,int s,int d){ double a=0; for(int l=0;l<Lc;l++) a+=(double)cKV[s*Lc+l]*wUK[l*Nh*D+h*D+d]; return a; };
    auto V=[&](int h,int s,int d){ double a=0; for(int l=0;l<Lc;l++) a+=(double)cKV[s*Lc+l]*wUV[l*Nh*D+h*D+d]; return a; };
    double me=0,mr=0;
    for(int h=0;h<Nh;h++)for(int si=0;si<S;si++){ std::vector<double> sc(si+1); double mx=-1e30;
        for(int sj=0;sj<=si;sj++){ double s=0; for(int d=0;d<D;d++) s+=(double)q[((h)*S+si)*D+d]*K(h,sj,d); s*=scale; sc[sj]=s; mx=std::max(mx,s); }
        double se=0; for(int sj=0;sj<=si;sj++){ sc[sj]=std::exp(sc[sj]-mx); se+=sc[sj]; }
        for(int d=0;d<D;d++){ double o=0; for(int sj=0;sj<=si;sj++) o+=sc[sj]/se*V(h,sj,d); double got=ho[((h)*S+si)*D+d]; me=std::max(me,std::fabs(got-o)); mr=std::max(mr,std::fabs(o)); } }
    report("MultiLatentAttention", me/(mr+1e-9), 1e-4);
    aclDestroyTensor(tq);aclDestroyTensor(tc);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to);
}
static void t_init_finalize() {
    // InitRouting groups (token,expert) entries by expert; FinalizeRouting recombines with gate weights.
    const int T=16, H=8, K=2, E=4; int M=T*K;
    auto x=randv(T*H,-1,1); std::vector<int32_t> eid(M); for(int i=0;i<M;i++) eid[i]=rand()%E;
    auto scales=randv(M,0.1f,1.0f);
    DevBuf dx(T*H*4),de(M*4),dex(M*H*4),drow(M*4),dexp(M*4);
    dx.up(x.data()); de.up(eid.data());
    aclTensor *tx=mk({T,H},ACL_FLOAT,dx.p),*te=mk({T,K},ACL_INT32,de.p),
              *tex=mk({M,H},ACL_FLOAT,dex.p),*trow=mk({M},ACL_INT32,drow.p),*texp=mk({M},ACL_INT32,dexp.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeInitRoutingGetWorkspaceSize(tx,te,E,tex,trow,texp,w,e);}, aclnnMoeInitRouting);
    std::vector<float> hex(M*H); std::vector<int32_t> hrow(M),hexp(M); dex.down(hex.data()); drow.down(hrow.data()); dexp.down(hexp.data());
    // expert ids must be non-decreasing (grouped), and expandedX[p] == x[token of rowIdx[p]]
    double me=0,mr=0; bool sorted=true; int prev=-1;
    for(int p=0;p<M;p++){ if(hexp[p]<prev) sorted=false; prev=hexp[p];
        int flat=hrow[p], t=flat/K; if(hexp[p]!=eid[flat]) sorted=false;
        for(int h=0;h<H;h++){ me=std::max(me,std::fabs((double)hex[p*H+h]-x[t*H+h])); mr=std::max(mr,std::fabs((double)x[t*H+h])); } }
    report("MoeInitRouting group", sorted?me/(mr+1e-9):1.0, 1e-6);
    // FinalizeRouting: out[t] = sum_k scales[t,k]*expandedY[p(t,k)]; use expandedX as Y for a clean reference
    DevBuf ds(M*4),dout(T*H*4); ds.up(scales.data());
    aclTensor *ts=mk({T,K},ACL_FLOAT,ds.p),*tout=mk({T,H},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeFinalizeRoutingGetWorkspaceSize(tex,trow,ts,K,tout,w,e);}, aclnnMoeFinalizeRouting);
    std::vector<float> hout(T*H); dout.down(hout.data());
    std::vector<double> ref(T*H,0); for(int p=0;p<M;p++){ int flat=hrow[p],t=flat/K,k=flat%K; for(int h=0;h<H;h++) ref[t*H+h]+=(double)scales[t*K+k]*hex[p*H+h]; }
    double fe=0,fr=0; for(int i=0;i<T*H;i++){ fe=std::max(fe,std::fabs(hout[i]-ref[i])); fr=std::max(fr,std::fabs(ref[i])); }
    report("MoeFinalizeRouting", fe/(fr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(te);aclDestroyTensor(tex);aclDestroyTensor(trow);aclDestroyTensor(texp);aclDestroyTensor(ts);aclDestroyTensor(tout);
}
static void t_moe_grad() {
    const int T=12,H=6,K=2,E=4,P=T*K;
    // build rowIdx via a simple grouping: rowIdx is a permutation of 0..P-1 (use init routing to get a valid one)
    std::vector<int32_t> eid(P); for(int i=0;i<P;i++) eid[i]=rand()%E;
    auto x=randv(T*H,-1,1);
    DevBuf dx(T*H*4),de(P*4),dex(P*H*4),drow(P*4),dexp(P*4); dx.up(x.data()); de.up(eid.data());
    aclTensor *tx=mk({T,H},ACL_FLOAT,dx.p),*te=mk({T,K},ACL_INT32,de.p),*tex=mk({P,H},ACL_FLOAT,dex.p),*trow=mk({P},ACL_INT32,drow.p),*texp=mk({P},ACL_INT32,dexp.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeInitRoutingGetWorkspaceSize(tx,te,E,tex,trow,texp,w,e);}, aclnnMoeInitRouting);
    std::vector<int32_t> hrow(P); drow.down(hrow.data());
    // FinalizeRoutingV2Grad: gradExpY[p]=scales[t,k]*gradOut[t], gradScales[t,k]=dot(gradOut[t],expY[p])
    auto gOut=randv(T*H,-1,1), expY=randv(P*H,-1,1), scales=randv(P,0.1f,1.0f);
    DevBuf dgo(T*H*4),dey(P*H*4),dsc(P*4),dgey(P*H*4),dgsc(P*4); dgo.up(gOut.data()); dey.up(expY.data()); dsc.up(scales.data());
    aclTensor *tgo=mk({T,H},ACL_FLOAT,dgo.p),*tey=mk({P,H},ACL_FLOAT,dey.p),*tsc=mk({T,K},ACL_FLOAT,dsc.p),*tgey=mk({P,H},ACL_FLOAT,dgey.p),*tgsc=mk({T,K},ACL_FLOAT,dgsc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(tgo,tey,trow,tsc,K,tgey,tgsc,w,e);}, aclnnMoeFinalizeRoutingV2Grad);
    std::vector<float> hgey(P*H),hgsc(P); dgey.down(hgey.data()); dgsc.down(hgsc.data());
    double e1=0,r1=0,e2=0,r2=0;
    for(int p=0;p<P;p++){ int flat=hrow[p],t=flat/K,k=flat%K; double dot=0;
        for(int h=0;h<H;h++){ double ref=scales[t*K+k]*gOut[t*H+h]; e1=std::max(e1,std::fabs((double)hgey[p*H+h]-ref)); r1=std::max(r1,std::fabs(ref)); dot+=(double)gOut[t*H+h]*expY[p*H+h]; }
        e2=std::max(e2,std::fabs((double)hgsc[t*K+k]-dot)); r2=std::max(r2,std::fabs(dot)); }
    report("MoeFinalizeRoutingV2Grad dY", e1/(r1+1e-9), 1e-5); report("MoeFinalizeRoutingV2Grad dScale", e2/(r2+1e-9), 1e-5);
    // InitRoutingV2Grad: gradX[t] = sum over p with rowIdx[p]/K==t of gradExpX[p]
    auto gExpX=randv(P*H,-1,1); DevBuf dgex(P*H*4),dgx(T*H*4); dgex.up(gExpX.data());
    aclTensor *tgex=mk({P,H},ACL_FLOAT,dgex.p),*tgx=mk({T,H},ACL_FLOAT,dgx.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeInitRoutingV2GradGetWorkspaceSize(tgex,trow,K,tgx,w,e);}, aclnnMoeInitRoutingV2Grad);
    std::vector<float> hgx(T*H); dgx.down(hgx.data()); std::vector<double> refx(T*H,0);
    for(int p=0;p<P;p++){ int t=hrow[p]/K; for(int h=0;h<H;h++) refx[t*H+h]+=gExpX[p*H+h]; }
    double e3=0,r3=0; for(int i=0;i<T*H;i++){ e3=std::max(e3,std::fabs((double)hgx[i]-refx[i])); r3=std::max(r3,std::fabs(refx[i])); }
    report("MoeInitRoutingV2Grad", e3/(r3+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(te);aclDestroyTensor(tex);aclDestroyTensor(trow);aclDestroyTensor(texp);
    aclDestroyTensor(tgo);aclDestroyTensor(tey);aclDestroyTensor(tsc);aclDestroyTensor(tgey);aclDestroyTensor(tgsc);aclDestroyTensor(tgex);aclDestroyTensor(tgx);
}
static void t_moe_ep(){
    { // MoeUpdateExpert: identity copy of expert ids
      std::vector<int32_t> ids={3,1,4,1,5,9,2,6}; std::vector<int32_t> out(ids.size());
      DevBuf di(ids.size()*4),dout(ids.size()*4); di.up(ids.data());
      auto ti=mk({(int64_t)ids.size()},ACL_INT32,di.p),to=mk({(int64_t)ids.size()},ACL_INT32,dout.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeUpdateExpertGetWorkspaceSize(ti,to,w,e);}, aclnnMoeUpdateExpert);
      dout.down(out.data()); double bad=0; for(size_t i=0;i<ids.size();i++) if(out[i]!=ids[i]) bad=1;
      report("MoeUpdateExpert", bad, 0.0); aclDestroyTensor(ti);aclDestroyTensor(to); }
    { // MoeTokenPermuteWithEp == base permute: tokens grouped by expert id (single rank EP)
      const int T=4,H=3,E=3; auto x=randv(T*H,-1,1); std::vector<int32_t> eid={2,0,1,0};
      DevBuf dx(T*H*4),de(T*4),dpx(T*H*4),didx(T*8); dx.up(x.data()); de.up(eid.data());
      auto tx=mk({T,H},ACL_FLOAT,dx.p),te=mk({T},ACL_INT32,de.p),tpx=mk({T,H},ACL_FLOAT,dpx.p),tidx=mk({T},ACL_INT64,didx.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMoeTokenPermuteWithEpGetWorkspaceSize(tx,te,(int64_t)E,tpx,tidx,w,e);}, aclnnMoeTokenPermuteWithEp);
      std::vector<float> px(T*H); std::vector<int64_t> idx(T); dpx.down(px.data()); didx.down(idx.data());
      // verify permuted rows are sorted by expert id and content matches source rows
      double bad=0; for(int p=0;p<T;p++){ int src=idx[p]; for(int h=0;h<H;h++) if(std::fabs(px[p*H+h]-x[src*H+h])>1e-6) bad=1; }
      for(int p=1;p<T;p++) if(eid[idx[p]]<eid[idx[p-1]]) bad=1;
      report("MoeTokenPermuteWithEp", bad, 0.0); aclDestroyTensor(tx);aclDestroyTensor(te);aclDestroyTensor(tpx);aclDestroyTensor(tidx); }
}
int main(){ init(); srand(59); t_gating(); t_compute_tokens(); t_permute_roundtrip(); t_init_finalize(); t_moe_grad(); t_rope_grad(); t_fia(); t_mla(); t_moe_ep(); return finish(); }
