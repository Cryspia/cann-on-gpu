// Quantization extensions (P14) cross-check: AscendQuant/Dequant/AntiQuant/DequantBias/FakeQuant.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
using namespace hn;

static int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

static void t_quant() {
    const int N=8,C=4; auto x=randv(N*C,-3,3),sc=randv(C,0.5,2.0),of=randv(C,-1,1);
    std::vector<int8_t> hq(N*C); DevBuf dx(N*C*4),ds(C*4),do_(C*4),dq(N*C); dx.up(x.data());ds.up(sc.data());do_.up(of.data());
    aclTensor *tx=mk({N,C},ACL_FLOAT,dx.p),*tsc=mk({C},ACL_FLOAT,ds.p),*tof=mk({C},ACL_FLOAT,do_.p),*tq=mk({N,C},ACL_INT8,dq.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAscendQuantGetWorkspaceSize(tx,tsc,tof,tq,w,e);}, aclnnAscendQuant);
    dq.down(hq.data()); int64_t bad=0;
    for(int i=0;i<N*C;i++){int c=i%C; int ref=clampi((int)std::lrint(x[i]*sc[c]+of[c]),-128,127); if((int)hq[i]!=ref)bad++;}
    report("AscendQuant", bad?1.0:0.0, 0); aclDestroyTensor(tx);aclDestroyTensor(tsc);aclDestroyTensor(tof);aclDestroyTensor(tq);
}
static void t_dequant() {
    const int N=8,C=4; std::vector<int8_t> q(N*C); for(auto&v:q) v=(int8_t)(rand()%255-127); auto sc=randv(C,0.5,2.0),of=randv(C,-1,1);
    std::vector<float> hy(N*C); DevBuf dq(N*C),ds(C*4),do_(C*4),dy(N*C*4); dq.up(q.data());ds.up(sc.data());do_.up(of.data());
    aclTensor *tq=mk({N,C},ACL_INT8,dq.p),*tsc=mk({C},ACL_FLOAT,ds.p),*tof=mk({C},ACL_FLOAT,do_.p),*ty=mk({N,C},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAscendDequantGetWorkspaceSize(tq,tsc,tof,ty,w,e);}, aclnnAscendDequant);
    dy.down(hy.data()); double me=0,mr=0; for(int i=0;i<N*C;i++){int c=i%C; double ref=((double)q[i]-of[c])*sc[c]; me=std::max(me,std::fabs(hy[i]-ref)); mr=std::max(mr,std::fabs(ref));}
    report("AscendDequant", me/(mr+1e-9), 1e-5); aclDestroyTensor(tq);aclDestroyTensor(tsc);aclDestroyTensor(tof);aclDestroyTensor(ty);
}
static void t_dequant_bias() {
    const int N=8,C=4; std::vector<int32_t> q(N*C); for(auto&v:q) v=rand()%2000-1000; auto sc=randv(C,0.01,0.05),bias=randv(C,-1,1);
    std::vector<float> hy(N*C); DevBuf dq(N*C*4),ds(C*4),db(C*4),dy(N*C*4); dq.up(q.data());ds.up(sc.data());db.up(bias.data());
    aclTensor *tq=mk({N,C},ACL_INT32,dq.p),*tsc=mk({C},ACL_FLOAT,ds.p),*tb=mk({C},ACL_FLOAT,db.p),*ty=mk({N,C},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDequantBiasGetWorkspaceSize(tq,tsc,tb,ty,w,e);}, aclnnDequantBias);
    dy.down(hy.data()); double me=0,mr=0; for(int i=0;i<N*C;i++){int c=i%C; double ref=(double)q[i]*sc[c]+bias[c]; me=std::max(me,std::fabs(hy[i]-ref)); mr=std::max(mr,std::fabs(ref));}
    report("DequantBias", me/(mr+1e-9), 1e-5); aclDestroyTensor(tq);aclDestroyTensor(tsc);aclDestroyTensor(tb);aclDestroyTensor(ty);
}
static void t_fake_quant() {
    const int n=4096; double scale=0.05,zp=0; int qmin=-128,qmax=127; auto x=randv(n,-3,3); std::vector<float> hy(n);
    DevBuf dx(n*4),dy(n*4); dx.up(x.data());
    aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*ty=mk({n},ACL_FLOAT,dy.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFakeQuantGetWorkspaceSize(tx,scale,zp,qmin,qmax,ty,w,e);}, aclnnFakeQuant);
    dy.down(hy.data()); double me=0,mr=0; for(int i=0;i<n;i++){int q=clampi((int)std::lrint(x[i]/scale+zp),qmin,qmax); double ref=((double)q-zp)*scale; me=std::max(me,std::fabs(hy[i]-ref)); mr=std::max(mr,std::fabs(ref));}
    report("FakeQuant", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(ty);
}
int main() {
    init(); srand(43);
    t_quant(); t_dequant(); t_dequant_bias(); t_fake_quant();
    { // DynamicBlockMxQuant: dequant(q·scale) ≈ x per block (scale power-of-2)
      const int nblk=6, blk=16, n=nblk*blk; auto x=randv(n,-4,4); std::vector<int8_t> q(n); std::vector<float> sc(nblk);
      DevBuf dx(n*4),dq(n),ds(nblk*4); dx.up(x.data());
      aclTensor *tx=mk({nblk,blk},ACL_FLOAT,dx.p),*tq=mk({nblk,blk},ACL_INT8,dq.p),*ts=mk({nblk},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDynamicBlockMxQuantGetWorkspaceSize(tx,(int64_t)blk,tq,ts,w,e);}, aclnnDynamicBlockMxQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int b=0;b<nblk;b++)for(int i=0;i<blk;i++){ double deq=(double)q[b*blk+i]*sc[b]; me=std::max(me,std::fabs(deq-x[b*blk+i])); mr=std::max(mr,std::fabs((double)x[b*blk+i])); }
      report("DynamicBlockMxQuant", me/(mr+1e-9), 2e-2); aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // DynamicDualLevelMxQuant: dequant(q·s1·s2) ≈ x
      const int nblk=5, blk=16, n=nblk*blk; auto x=randv(n,-4,4); std::vector<int8_t> q(n); std::vector<float> s1(nblk),s2(nblk);
      DevBuf dx(n*4),dq(n),d1(nblk*4),d2(nblk*4); dx.up(x.data());
      aclTensor *tx=mk({nblk,blk},ACL_FLOAT,dx.p),*tq=mk({nblk,blk},ACL_INT8,dq.p),*t1=mk({nblk},ACL_FLOAT,d1.p),*t2=mk({nblk},ACL_FLOAT,d2.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDynamicDualLevelMxQuantGetWorkspaceSize(tx,(int64_t)blk,tq,t1,t2,w,e);}, aclnnDynamicDualLevelMxQuant);
      dq.down(q.data()); d1.down(s1.data()); d2.down(s2.data()); double me=0,mr=0;
      for(int b=0;b<nblk;b++)for(int i=0;i<blk;i++){ double deq=(double)q[b*blk+i]*s1[b]*s2[b]; me=std::max(me,std::fabs(deq-x[b*blk+i])); mr=std::max(mr,std::fabs((double)x[b*blk+i])); }
      report("DynamicDualLevelMxQuant", me/(mr+1e-9), 2e-2); aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(t1);aclDestroyTensor(t2); }
    { // FlatQuant: dequant(q·scale) ≈ x per row (absmax)
      const int R=6,D=20,n=R*D; auto x=randv(n,-3,3); std::vector<int8_t> q(n); std::vector<float> sc(R);
      DevBuf dx(n*4),dq(n),ds(R*4); dx.up(x.data());
      aclTensor *tx=mk({R,D},ACL_FLOAT,dx.p),*tq=mk({R,D},ACL_INT8,dq.p),*ts=mk({R},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFlatQuantGetWorkspaceSize(tx,tq,ts,w,e);}, aclnnFlatQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double deq=(double)q[r*D+d]*sc[r]; me=std::max(me,std::fabs(deq-x[r*D+d])); mr=std::max(mr,std::fabs((double)x[r*D+d])); }
      report("FlatQuant", me/(mr+1e-9), 1e-2); aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // SwigluMxQuant: dequant(q·scale) ≈ swiglu(in)
      const int R=6,D=12,n=R*2*D; auto in=randv(n,-2,2); std::vector<int8_t> q(R*D); std::vector<float> sc(R);
      DevBuf din(n*4),dq(R*D),ds(R*4); din.up(in.data());
      aclTensor *tin=mk({R,2*D},ACL_FLOAT,din.p),*tq=mk({R,D},ACL_INT8,dq.p),*ts=mk({R},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSwigluMxQuantGetWorkspaceSize(tin,tq,ts,w,e);}, aclnnSwigluMxQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0;
      for(int r=0;r<R;r++)for(int d=0;d<D;d++){ double a=in[r*2*D+d],b=in[r*2*D+D+d]; double g=a/(1.0+std::exp(-a))*b; double deq=(double)q[r*D+d]*sc[r]; me=std::max(me,std::fabs(deq-g)); mr=std::max(mr,std::fabs(g)); }
      report("SwigluMxQuant", me/(mr+1e-9), 2e-2); aclDestroyTensor(tin);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // InplaceQuantScatter: self[idx[k]] = round(upd[k]/scale)
      const int N=5,D=4,K=3; double scale=0.5; std::vector<int8_t> self(N*D,0); std::vector<int64_t> idx={1,3,4}; auto upd=randv(K*D,-10,10);
      DevBuf dself(N*D),didx(K*8),dupd(K*D*4); dself.up(self.data()); didx.up(idx.data()); dupd.up(upd.data());
      aclTensor *ts=mk({N,D},ACL_INT8,dself.p),*ti=mk({K},ACL_INT64,didx.p),*tu=mk({K,D},ACL_FLOAT,dupd.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceQuantScatterGetWorkspaceSize(ts,ti,tu,scale,w,e);}, aclnnInplaceQuantScatter);
      std::vector<int8_t> got(N*D); dself.down(got.data()); double bad=0;
      for(int k=0;k<K;k++)for(int d=0;d<D;d++){ int ref=(int)std::lrint(upd[k*D+d]/scale); ref=ref<-127?-127:(ref>127?127:ref); if(got[idx[k]*D+d]!=ref) bad=1; }
      for(int d=0;d<D;d++){ if(got[0*D+d]!=0||got[2*D+d]!=0) bad=1; } // untouched rows stay 0
      report("InplaceQuantScatter", bad, 0.0); aclDestroyTensor(ts);aclDestroyTensor(ti);aclDestroyTensor(tu); }
    { // TransQuantParam: low32 of int64 out == float bits of scale
      const int n=8; auto scv=randv(n,0.01,2.0); DevBuf ds(n*4),do_(n*8); ds.up(scv.data());
      aclTensor *ts=mk({n},ACL_FLOAT,ds.p),*to=mk({n},ACL_INT64,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransQuantParamGetWorkspaceSize(ts,nullptr,to,w,e);}, aclnnTransQuantParam);
      std::vector<int64_t> got(n); do_.down(got.data()); double bad=0;
      for(int i=0;i<n;i++){ uint32_t lo=(uint32_t)((uint64_t)got[i]&0xffffffffu); float f; std::memcpy(&f,&lo,4); if(std::fabs(f-scv[i])>1e-6) bad=1; }
      report("TransQuantParam", bad, 0.0); aclDestroyTensor(ts);aclDestroyTensor(to); }
    return finish();
}
