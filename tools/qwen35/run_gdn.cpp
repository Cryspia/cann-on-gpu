// Cross-check the cann-on-gpu shim forward of one Qwen3-Next GatedDeltaNet (linear-attention) layer against the HF
// reference dumped by export_gdn.py. Pipeline: in-proj -> causal conv1d+silu -> gated delta rule -> gated RMSNorm -> out-proj.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <string>
using namespace hn;

static std::string DIR;
static std::vector<float> rd(const std::string &name, int64_t n) {
    std::vector<float> v(n); FILE *f = fopen((DIR + "/" + name).c_str(), "rb");
    if (!f || (int64_t)fread(v.data(), 4, n, f) != n) { printf("[FATAL] read %s\n", name.c_str()); exit(1); } fclose(f); return v;
}
static DevBuf *up(const std::vector<float> &h) { auto *d = new DevBuf(h.size()*4); d->up(h.data()); return d; }
static double nerr(const std::vector<float>&g, const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<r.size();i++){me=std::max(me,(double)std::fabs(g[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc, char **argv) {
    DIR = argc > 1 ? argv[1] : "data_gdn";
    init();
    FILE *mf = fopen((DIR + "/meta.txt").c_str(), "r");
    int B,S,D,nk,nv,hkd,hvd,ck,key_dim,value_dim; double eps;
    fscanf(mf, "%d %d %d %d %d %d %d %d %d %d %lf", &B,&S,&D,&nk,&nv,&hkd,&hvd,&ck,&key_dim,&value_dim,&eps); fclose(mf);
    const int conv_ch = key_dim + key_dim + value_dim;   // 128

    auto *dh = up(rd("hin.bin", (int64_t)S*D));
    auto *dWq=up(rd("Wq.bin",(int64_t)key_dim*D)), *dWk=up(rd("Wk.bin",(int64_t)key_dim*D)),
         *dWv=up(rd("Wv.bin",(int64_t)value_dim*D)), *dWz=up(rd("Wz.bin",(int64_t)value_dim*D)),
         *dWb=up(rd("Wb.bin",(int64_t)nv*D)), *dWa=up(rd("Wa.bin",(int64_t)nv*D));
    auto *dconv=up(rd("conv1d.bin",(int64_t)conv_ch*ck));
    auto Alog=rd("A_log.bin",nv), DtB=rd("dt_bias.bin",nv);
    auto *dAlog=up(Alog), *dDtB=up(DtB), *dnorm=up(rd("norm.bin",hvd)), *dout=up(rd("out_proj.bin",(int64_t)D*value_dim));

    // ---- helpers ----
    auto mmT = [&](aclTensor *a, aclTensor *w, int64_t M, int64_t K, int64_t N) {   // a[M,K] @ w[N,K]^T -> [M,N]
        auto *o = new DevBuf(M*N*4); aclTensor *to = mk({M,N}, ACL_FLOAT, o->p);
        exec2([&](uint64_t *ws, aclOpExecutor **e){ return aclnnMatmulBiasGetWorkspaceSize(a, w, nullptr, false, true, 0, to, ws, e); }, aclnnMatmulBias);
        return std::make_pair(o, to);
    };
    auto perm = [&](aclTensor *a, std::vector<int64_t> in, std::vector<int64_t> dims) {   // permute, returns new buf+tensor
        std::vector<int64_t> od; for (auto d : dims) od.push_back(in[d]); int64_t n=1; for(auto x:od)n*=x;
        auto *o=new DevBuf(n*4); aclTensor *to=mk(od, ACL_FLOAT, o->p); aclIntArray *dd=aclCreateIntArray(dims.data(),dims.size());
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,dd,to,ws,e);}, aclnnPermute);
        return std::make_pair(o,to);
    };
    auto down=[&](DevBuf*d,int64_t n){ std::vector<float> h(n); d->down(h.data()); return h; };

    // ---- in-projections ----
    auto q = mmT(mk({S,D},ACL_FLOAT,dh->p), mk({key_dim,D},ACL_FLOAT,dWq->p), S,D,key_dim);
    auto k = mmT(mk({S,D},ACL_FLOAT,dh->p), mk({key_dim,D},ACL_FLOAT,dWk->p), S,D,key_dim);
    auto v = mmT(mk({S,D},ACL_FLOAT,dh->p), mk({value_dim,D},ACL_FLOAT,dWv->p), S,D,value_dim);
    auto z = mmT(mk({S,D},ACL_FLOAT,dh->p), mk({value_dim,D},ACL_FLOAT,dWz->p), S,D,value_dim);
    auto bb= mmT(mk({S,D},ACL_FLOAT,dh->p), mk({nv,D},ACL_FLOAT,dWb->p), S,D,nv);
    auto aa= mmT(mk({S,D},ACL_FLOAT,dh->p), mk({nv,D},ACL_FLOAT,dWa->p), S,D,nv);

    // ---- mixed = cat(q,k,v) [S,128] -> [128,S] -> causal conv1d+silu -> [S,128] ----
    auto *dmix=new DevBuf((int64_t)S*conv_ch*4); aclTensor *tmix=mk({S,conv_ch},ACL_FLOAT,dmix->p);
    { const aclTensor *ins[3]={q.second,k.second,v.second};
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCatGetWorkspaceSize(ins,3,1,tmix,ws,e);}, aclnnCat); }
    auto mt = perm(tmix, {S,conv_ch}, {1,0});   // [128,S]
    auto *dco=new DevBuf((int64_t)conv_ch*S*4); aclTensor *tco=mk({1,conv_ch,S},ACL_FLOAT,dco->p);
    aclTensor *tmt3=mk({1,conv_ch,S},ACL_FLOAT,mt.first->p), *tcw=mk({conv_ch,ck},ACL_FLOAT,dconv->p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCausalConv1dGetWorkspaceSize(tmt3,tcw,nullptr,1,tco,ws,e);}, aclnnCausalConv1d);
    auto co = perm(mk({conv_ch,S},ACL_FLOAT,dco->p), {conv_ch,S}, {1,0});   // [S,128]
    report("conv_out", nerr(down(co.first,(int64_t)S*conv_ch), rd("conv_out.bin",(int64_t)S*conv_ch)), 1e-4);

    // ---- split conv output into q2[S,kd] k2[S,kd] v2[S,vd] ----
    auto slice=[&](aclTensor*a,int64_t cols,int64_t s0,int64_t len){ auto*o=new DevBuf((int64_t)S*len*4); aclTensor*to=mk({S,len},ACL_FLOAT,o->p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnSliceGetWorkspaceSize(a,1,s0,s0+len,1,to,ws,e);}, aclnnSlice); return std::make_pair(o,to); };
    aclTensor *tcoflat=mk({S,conv_ch},ACL_FLOAT,co.first->p);
    auto q2=slice(tcoflat,conv_ch,0,key_dim), k2=slice(tcoflat,conv_ch,key_dim,key_dim), v2=slice(tcoflat,conv_ch,2*key_dim,value_dim);

    // ---- beta = sigmoid(b); g = -exp(A_log)*softplus(a+dt_bias); gate=exp(g) ----
    auto un=[&](aclTensor*a,int64_t n,aclnnStatus(*gw)(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**),aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
        auto*o=new DevBuf(n*4); aclTensor*to=mk({n},ACL_FLOAT,o->p); exec2([&](uint64_t*ws,aclOpExecutor**e){return gw(a,to,ws,e);},r); return std::make_pair(o,to); };
    auto beta = un(mk({(int64_t)S*nv},ACL_FLOAT,bb.first->p),(int64_t)S*nv,aclnnSigmoidGetWorkspaceSize,aclnnSigmoid);
    // a + dt_bias  (broadcast [S,nv]+[nv])
    auto *dapb=new DevBuf((int64_t)S*nv*4); aclTensor *tapb=mk({S,nv},ACL_FLOAT,dapb->p);
    { aclTensor *taa=mk({S,nv},ACL_FLOAT,aa.first->p), *tdt=mk({nv},ACL_FLOAT,dDtB->p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(taa,tdt,nullptr,tapb,ws,e);}, aclnnAdd); }
    auto sp = un(mk({(int64_t)S*nv},ACL_FLOAT,dapb->p),(int64_t)S*nv,aclnnSoftplusGetWorkspaceSize,aclnnSoftplus);  // softplus(a+dt)
    auto ealog = un(mk({nv},ACL_FLOAT,dAlog->p),nv,aclnnExpGetWorkspaceSize,aclnnExp);            // exp(A_log)
    // nealog = -ealog
    auto *dne=new DevBuf(nv*4); aclTensor*tne=mk({nv},ACL_FLOAT,dne->p);
    { float m1=-1; aclScalar*s=aclCreateScalar(&m1,ACL_FLOAT); exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulsGetWorkspaceSize(ealog.second,s,tne,ws,e);}, aclnnMuls); }
    // g = sp * nealog  (broadcast)
    auto *dg=new DevBuf((int64_t)S*nv*4); aclTensor*tg=mk({S,nv},ACL_FLOAT,dg->p);
    { aclTensor *sp2=mk({S,nv},ACL_FLOAT,sp.first->p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulGetWorkspaceSize(sp2,tne,tg,ws,e);}, aclnnMul); }
    auto gate = un(mk({(int64_t)S*nv},ACL_FLOAT,dg->p),(int64_t)S*nv,aclnnExpGetWorkspaceSize,aclnnExp);   // gate = exp(g)

    // ---- repeat q2,k2 heads (nk->nv) via gather dim=1 with idx[0,0,1,1...] ; build [1,nv,S,hkd] ----
    std::vector<int64_t> idxh(nv); for(int i=0;i<nv;i++) idxh[i]=i/(nv/nk);
    DevBuf didx(nv*8); { std::vector<int64_t> t=idxh; didx.up(t.data()); }
    auto headrep=[&](DevBuf*src){ aclTensor*t3=mk({S,nk,hkd},ACL_FLOAT,src->p);   // gather along dim1
        auto*o=new DevBuf((int64_t)S*nv*hkd*4); aclTensor*to=mk({S,nv,hkd},ACL_FLOAT,o->p); aclTensor*ti=mk({nv},ACL_INT64,didx.p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGatherGetWorkspaceSize(t3,1,ti,to,ws,e);}, aclnnGather);
        return perm(to,{S,nv,hkd},{1,0,2}); };   // -> [nv,S,hkd]
    auto qr=headrep(q2.first), kr=headrep(k2.first);
    auto vr=perm(mk({S,nv,hvd},ACL_FLOAT,v2.first->p),{S,nv,hvd},{1,0,2});   // [nv,S,hvd]

    // l2norm qr,kr over last dim; scale qr by 1/sqrt(hkd)
    auto l2=[&](DevBuf*b){ aclTensor*flat=mk({(int64_t)nv*S,hkd},ACL_FLOAT,b->p); auto*o=new DevBuf((int64_t)nv*S*hkd*4); aclTensor*to=mk({(int64_t)nv*S,hkd},ACL_FLOAT,o->p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnLpNormalizeGetWorkspaceSize(flat,2.0,1e-6,to,ws,e);}, aclnnLpNormalize); return o; };
    DevBuf *qn=l2(qr.first), *kn=l2(kr.first);
    { float sc=1.f/std::sqrt((float)hkd); aclScalar*s=aclCreateScalar(&sc,ACL_FLOAT); aclTensor*flat=mk({(int64_t)nv*S,hkd},ACL_FLOAT,qn->p);
      auto*o=new DevBuf((int64_t)nv*S*hkd*4); aclTensor*to=mk({(int64_t)nv*S,hkd},ACL_FLOAT,o->p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulsGetWorkspaceSize(flat,s,to,ws,e);}, aclnnMuls); qn=o; }

    // beta,gate -> [nv,S] -> [1,nv,S]
    auto betaT=perm(mk({S,nv},ACL_FLOAT,beta.first->p),{S,nv},{1,0});
    auto gateT=perm(mk({S,nv},ACL_FLOAT,gate.first->p),{S,nv},{1,0});

    // ---- gated delta rule ----
    auto *dcore=new DevBuf((int64_t)nv*S*hvd*4); aclTensor*tcore=mk({1,nv,S,hvd},ACL_FLOAT,dcore->p);
    aclTensor *tq=mk({1,nv,S,hkd},ACL_FLOAT,qn->p), *tk=mk({1,nv,S,hkd},ACL_FLOAT,kn->p), *tv=mk({1,nv,S,hvd},ACL_FLOAT,vr.first->p),
              *tbeta=mk({1,nv,S},ACL_FLOAT,betaT.first->p), *tgate=mk({1,nv,S},ACL_FLOAT,gateT.first->p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGatedDeltaRuleGetWorkspaceSize(tq,tk,tv,tbeta,tgate,tcore,ws,e);}, aclnnGatedDeltaRule);
    // core [1,nv,S,hvd] -> [S,nv,hvd]
    auto coreS=perm(mk({nv,S,hvd},ACL_FLOAT,dcore->p),{nv,S,hvd},{1,0,2});
    report("core(pre-norm)", nerr(down(coreS.first,(int64_t)S*nv*hvd), rd("core.bin",(int64_t)S*nv*hvd)), 2e-4);

    // ---- gated RMSNorm: (norm.weight * rmsnorm(core)) * silu(z) ----
    auto *drms=new DevBuf((int64_t)S*nv*hvd*4); aclTensor*trms=mk({(int64_t)S*nv,hvd},ACL_FLOAT,drms->p);
    { aclTensor*cf=mk({(int64_t)S*nv,hvd},ACL_FLOAT,coreS.first->p), *gw=mk({hvd},ACL_FLOAT,dnorm->p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(cf,gw,eps,trms,ws,e);}, aclnnRmsNorm); }
    auto sz=un(mk({(int64_t)S*nv*hvd},ACL_FLOAT,z.first->p),(int64_t)S*nv*hvd,aclnnSiluGetWorkspaceSize,aclnnSilu);  // silu(z)
    auto *dng=new DevBuf((int64_t)S*nv*hvd*4); aclTensor*tng=mk({(int64_t)S*nv*hvd},ACL_FLOAT,dng->p);
    { aclTensor*rf=mk({(int64_t)S*nv*hvd},ACL_FLOAT,drms->p);
      exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulGetWorkspaceSize(rf,sz.second,tng,ws,e);}, aclnnMul); }

    // ---- out_proj: [S,value_dim] @ out_proj[D,value_dim]^T -> [S,D] ----
    auto outp = mmT(mk({S,value_dim},ACL_FLOAT,dng->p), mk({D,value_dim},ACL_FLOAT,dout->p), S,value_dim,D);
    report("GDN layer output", nerr(down(outp.first,(int64_t)S*D), rd("ref_out.bin",(int64_t)S*D)), 2e-4);
    return finish();
}
