// Linear attention / SSM (P18) cross-check: CausalConv1d, SelectiveScan (Mamba), GatedDeltaRule. CPU double reference.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_causal_conv1d() {
    const int B=2,C=3,L=8,K=4; auto x=randv(B*C*L,-1,1),w=randv(C*K,-1,1),bias=randv(C,-0.5,0.5);
    std::vector<float> hz(B*C*L); DevBuf dx(B*C*L*4),dw(C*K*4),db(C*4),dz(B*C*L*4); dx.up(x.data());dw.up(w.data());db.up(bias.data());
    aclTensor *tx=mk({B,C,L},ACL_FLOAT,dx.p),*tw=mk({C,K},ACL_FLOAT,dw.p),*tb=mk({C},ACL_FLOAT,db.p),*tz=mk({B,C,L},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w_,aclOpExecutor**e){return aclnnCausalConv1dGetWorkspaceSize(tx,tw,tb,0,tz,w_,e);}, aclnnCausalConv1d);
    dz.down(hz.data()); double me=0,mr=0;
    for(int b=0;b<B;b++)for(int c=0;c<C;c++)for(int t=0;t<L;t++){ double acc=bias[c]; for(int k=0;k<K;k++){int ti=t-(K-1)+k; if(ti>=0) acc+=(double)w[c*K+k]*x[(b*C+c)*L+ti];}
        me=std::max(me,std::fabs(hz[(b*C+c)*L+t]-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("CausalConv1d", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_selective_scan() {
    const int B=2,L=6,D=4,N=8; auto u=randv(B*L*D,-1,1),delta=randv(B*L*D,0.01,0.2),A=randv(D*N,-1,-0.1),Bm=randv(B*L*N,-1,1),Cm=randv(B*L*N,-1,1),Ds=randv(D,-0.5,0.5);
    std::vector<float> hy(B*L*D); DevBuf du(B*L*D*4),dd(B*L*D*4),dA(D*N*4),dB(B*L*N*4),dC(B*L*N*4),dDs(D*4),dy(B*L*D*4);
    du.up(u.data());dd.up(delta.data());dA.up(A.data());dB.up(Bm.data());dC.up(Cm.data());dDs.up(Ds.data());
    aclTensor *tu=mk({B,L,D},ACL_FLOAT,du.p),*td=mk({B,L,D},ACL_FLOAT,dd.p),*tA=mk({D,N},ACL_FLOAT,dA.p),*tB=mk({B,L,N},ACL_FLOAT,dB.p),*tC=mk({B,L,N},ACL_FLOAT,dC.p),*tDs=mk({D},ACL_FLOAT,dDs.p),*ty=mk({B,L,D},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSelectiveScanGetWorkspaceSize(tu,td,tA,tB,tC,tDs,ty,w,e);}, aclnnSelectiveScan);
    dy.down(hy.data()); double me=0,mr=0;
    for(int b=0;b<B;b++)for(int d=0;d<D;d++){ std::vector<double> h(N,0);
        for(int t=0;t<L;t++){ double dt=delta[(b*L+t)*D+d],ut=u[(b*L+t)*D+d],yt=0;
            for(int n=0;n<N;n++){ double dA_=std::exp(dt*A[d*N+n]); double dBu=dt*Bm[(b*L+t)*N+n]*ut; h[n]=dA_*h[n]+dBu; yt+=Cm[(b*L+t)*N+n]*h[n]; }
            double ref=yt+Ds[d]*ut; me=std::max(me,std::fabs(hy[(b*L+t)*D+d]-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("SelectiveScan (Mamba)", me/(mr+1e-9), 1e-4); aclDestroyTensor(tu);aclDestroyTensor(td);aclDestroyTensor(tA);aclDestroyTensor(tB);aclDestroyTensor(tC);aclDestroyTensor(tDs);aclDestroyTensor(ty);
}
static void t_gated_delta() {
    const int B=2,Hd=2,L=5,Dk=4,Dv=4;
    auto q=randv(B*Hd*L*Dk,-1,1),k=randv(B*Hd*L*Dk,-1,1),v=randv(B*Hd*L*Dv,-1,1),beta=randv(B*Hd*L,0,1),g=randv(B*Hd*L,0.8,1.0);
    std::vector<float> hy(B*Hd*L*Dv);
    DevBuf dq(q.size()*4),dk(k.size()*4),dv(v.size()*4),dbeta(beta.size()*4),dg(g.size()*4),dy(hy.size()*4);
    dq.up(q.data());dk.up(k.data());dv.up(v.data());dbeta.up(beta.data());dg.up(g.data());
    aclTensor *tq=mk({B,Hd,L,Dk},ACL_FLOAT,dq.p),*tk=mk({B,Hd,L,Dk},ACL_FLOAT,dk.p),*tv=mk({B,Hd,L,Dv},ACL_FLOAT,dv.p),*tbeta=mk({B,Hd,L},ACL_FLOAT,dbeta.p),*tg=mk({B,Hd,L},ACL_FLOAT,dg.p),*ty=mk({B,Hd,L,Dv},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatedDeltaRuleGetWorkspaceSize(tq,tk,tv,tbeta,tg,ty,w,e);}, aclnnGatedDeltaRule);
    dy.down(hy.data()); double me=0,mr=0;
    for(int bh=0;bh<B*Hd;bh++){ std::vector<double> S(Dk*Dv,0);
        for(int t=0;t<L;t++){ int64_t base=bh*L+t; const float *kt=&k[base*Dk],*qt=&q[base*Dk],*vt=&v[base*Dv]; double gt=g[base],bt=beta[base];
            for(int i=0;i<Dk*Dv;i++) S[i]*=gt;
            for(int j=0;j<Dv;j++){ double sk=0; for(int i=0;i<Dk;i++) sk+=S[i*Dv+j]*kt[i]; double kv=vt[j]-sk; for(int i=0;i<Dk;i++) S[i*Dv+j]+=bt*kt[i]*kv; }
            for(int j=0;j<Dv;j++){ double yt=0; for(int i=0;i<Dk;i++) yt+=qt[i]*S[i*Dv+j]; me=std::max(me,std::fabs(hy[base*Dv+j]-yt)); mr=std::max(mr,std::fabs(yt)); } } }
    report("GatedDeltaRule", me/(mr+1e-9), 1e-4);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(tbeta);aclDestroyTensor(tg);aclDestroyTensor(ty);
}
int main(){ init(); srand(53); t_causal_conv1d(); t_selective_scan(); t_gated_delta(); return finish(); }
