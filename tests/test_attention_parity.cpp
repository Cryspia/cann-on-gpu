// Self-contained verification for the ~30 attention / FlashAttention / RoPE gap operators.
// Each op is checked against a CPU reference (scaled-dot-product attention, rotate-half RoPE,
// online-softmax merge, RMSNorm+RoPE, block mean-pool, rel-pos masked softmax).
// extern "C" prototypes below match the canonical aclnnop/aclnn_ops.h signatures EXACTLY.
// Shapes are small BNSD with GQA + causal + a bool-mask case. Tolerance ~1e-5 fp32.
#include "harness.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

// ---- canonical signatures (exactly as declared in aclnnop/aclnn_ops.h) ----
extern "C" {
// ATT_DECL / FA_DECL: (q,k,v,attenMask,scaleValue,headNum,causal,out)
#define DECL_ATT(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,int64_t,bool,aclTensor*,uint64_t*,aclOpExecutor**); \
    aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
DECL_ATT(aclnnBlitzSparseAttention)
DECL_ATT(aclnnNsaCompressAttention)
DECL_ATT(aclnnNsaCompressAttentionInfer)
DECL_ATT(aclnnNsaSelectedAttention)
DECL_ATT(aclnnNsaSelectedAttentionInfer)
DECL_ATT(aclnnFusedFloydAttention)
DECL_ATT(aclnnRainFusionAttention)
DECL_ATT(aclnnSparseFlashAttention)
DECL_ATT(aclnnFlashAttentionVarLenScore)
DECL_ATT(aclnnFlashAttentionVarLenScoreV2)
DECL_ATT(aclnnFlashAttentionVarLenScoreV3)
DECL_ATT(aclnnFlashAttentionVarLenScoreV4)
DECL_ATT(aclnnFlashAttentionVarLenScoreV5)
DECL_ATT(aclnnFusedInferAttentionScoreV3)
DECL_ATT(aclnnFusedInferAttentionScoreV4)
#undef DECL_ATT
// IncreFlashAttention: (q,k,v,attenMask,scaleValue,headNum,out) -- no causal
aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnIncreFlashAttention(void*,uint64_t,aclOpExecutor*,aclrtStream);
// ROPE_DECL: (x,cos,sin,mode,out)
#define DECL_ROPE(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**); \
    aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
DECL_ROPE(aclnnRotaryPositionEmbeddingV2)
DECL_ROPE(aclnnRopeWithSinCosCache)
DECL_ROPE(aclnnRopeWithSinCosCacheV2)
#undef DECL_ROPE
// InterleaveRope: (x,cos,sin,out)
aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnInterleaveRope(void*,uint64_t,aclOpExecutor*,aclrtStream);
// ApplyRotaryPosEmbV2: (q,k,cos,sin,mode,qOut,kOut)
aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,int64_t,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnApplyRotaryPosEmbV2(void*,uint64_t,aclOpExecutor*,aclrtStream);
// RING_DECL: (out1,lse1,out2,lse2,out,lse)
#define DECL_RING(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**); \
    aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
DECL_RING(aclnnRingAttentionUpdateV2)
DECL_RING(aclnnAttentionUpdate)
#undef DECL_RING
// NRC_DECL: (x,gamma,cos,sin,eps,mode,out)
#define DECL_NRC(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,int64_t,aclTensor*,uint64_t*,aclOpExecutor**); \
    aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
DECL_NRC(aclnnKvRmsNormRopeCache)
DECL_NRC(aclnnQkvRmsNormRopeCache)
DECL_NRC(aclnnNormRopeConcat)
#undef DECL_NRC
// NSAC_DECL: (x,blockSize,out)
aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnNsaCompressWithCache(void*,uint64_t,aclOpExecutor*,aclrtStream);
// A2F_DECL: (x,out)
#define DECL_A2F(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**); \
    aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
DECL_A2F(aclnnAttentionToFFN)
DECL_A2F(aclnnFFNToAttention)
#undef DECL_A2F
// MaskedSoftmaxWithRelPosBias: (x,mask,relPosBias,scale,out)
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,double,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(void*,uint64_t,aclOpExecutor*,aclrtStream);
} // extern "C"

// ---- attention test config ----
static const int B=2, Nq=4, Nkv=2, Sq=6, Skv=8, D=16;

// CPU reference scaled-dot-product attention: BNSD, GQA, optional causal (bottom-right) + bool mask[B,Sq,Skv].
static std::vector<double> cpu_sdpa(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,
                                    double scale,bool causal,const std::vector<uint8_t>*mask){
    int off=Skv-Sq; std::vector<double> o((size_t)B*Nq*Sq*D);
    for(int bi=0;bi<B;bi++)for(int h=0;h<Nq;h++){ int kvh=h/(Nq/Nkv),qf=bi*Nq+h,kf=bi*Nkv+kvh;
        for(int i=0;i<Sq;i++){ std::vector<double> sc(Skv); double mx=-1e300;
            for(int j=0;j<Skv;j++){ double p=0; for(int t=0;t<D;t++)p+=(double)q[(qf*Sq+i)*D+t]*(double)k[(kf*Skv+j)*D+t];
                double s=p*scale; bool blk=(causal&&j>i+off)||(mask&&(*mask)[(bi*Sq+i)*Skv+j]); if(blk)s=-1e300; sc[j]=s; mx=std::max(mx,s); }
            double den=0; for(int j=0;j<Skv;j++){ sc[j]=std::exp(sc[j]-mx); den+=sc[j]; }
            for(int t=0;t<D;t++){ double a=0; for(int j=0;j<Skv;j++)a+=sc[j]*(double)v[(kf*Skv+j)*D+t]; o[(qf*Sq+i)*D+t]=a/den; } } }
    return o;
}

// Run an ATT/FA-style op (q,k,v,mask,scale,headNum,causal,out) and return host result.
template<typename G,typename R>
static std::vector<float> run_att(const std::vector<float>&q,const std::vector<float>&k,const std::vector<float>&v,
                                  const std::vector<uint8_t>*mask,double scale,bool causal,G gw,R run){
    int64_t qn=(int64_t)B*Nq*Sq*D, kn=(int64_t)B*Nkv*Skv*D, on=qn, mn=(int64_t)B*Sq*Skv;
    DevBuf dq(qn*4),dk(kn*4),dv(kn*4),dout(on*4); dq.up(q.data()); dk.up(k.data()); dv.up(v.data());
    DevBuf *dm=nullptr; aclTensor *tm=nullptr;
    if(mask){ dm=new DevBuf(mn); dm->up(mask->data()); tm=mk({B,Sq,Skv},ACL_BOOL,dm->p); }
    auto tq=mk({B,Nq,Sq,D},ACL_FLOAT,dq.p),tk=mk({B,Nkv,Skv,D},ACL_FLOAT,dk.p),tv=mk({B,Nkv,Skv,D},ACL_FLOAT,dv.p),to=mk({B,Nq,Sq,D},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tq,tk,tv,tm,scale,(int64_t)Nq,causal,to,w,e);},run);
    std::vector<float> h(on); dout.down(h.data());
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to);
    if(tm){aclDestroyTensor(tm); delete dm;}
    return h;
}

int main(){
    init(); srand(7);
    double scale=1.0/std::sqrt((double)D);
    auto q=randv((int64_t)B*Nq*Sq*D,-1,1), k=randv((int64_t)B*Nkv*Skv*D,-1,1), v=randv((int64_t)B*Nkv*Skv*D,-1,1);
    auto ref_plain  = cpu_sdpa(q,k,v,scale,false,nullptr);
    auto ref_causal = cpu_sdpa(q,k,v,scale,true,nullptr);
    // bool mask [B,Sq,Skv]: block a scattered set of positions
    std::vector<uint8_t> mask((size_t)B*Sq*Skv,0);
    for(size_t i=0;i<mask.size();i++) mask[i]=(i*7+3)%5==0?1:0;
    auto ref_masked = cpu_sdpa(q,k,v,scale,false,&mask);

    // ---- ATT/FA family: plain (non-causal), to compare against CPU SDPA ----
    #define CHK_ATT(NAME) report(#NAME, norm_err(run_att(q,k,v,nullptr,scale,false,NAME##GetWorkspaceSize,NAME),ref_plain),1e-5)
    CHK_ATT(aclnnBlitzSparseAttention);
    CHK_ATT(aclnnNsaCompressAttention);
    CHK_ATT(aclnnNsaCompressAttentionInfer);
    CHK_ATT(aclnnNsaSelectedAttention);
    CHK_ATT(aclnnNsaSelectedAttentionInfer);
    CHK_ATT(aclnnFusedFloydAttention);
    CHK_ATT(aclnnRainFusionAttention);
    CHK_ATT(aclnnSparseFlashAttention);
    CHK_ATT(aclnnFlashAttentionVarLenScore);
    CHK_ATT(aclnnFlashAttentionVarLenScoreV2);
    CHK_ATT(aclnnFlashAttentionVarLenScoreV3);
    CHK_ATT(aclnnFlashAttentionVarLenScoreV4);
    CHK_ATT(aclnnFlashAttentionVarLenScoreV5);
    CHK_ATT(aclnnFusedInferAttentionScoreV3);
    CHK_ATT(aclnnFusedInferAttentionScoreV4);
    #undef CHK_ATT
    // causal + mask coverage on representative ops
    report("FlashAttentionVarLenScore[causal]", norm_err(run_att(q,k,v,nullptr,scale,true,aclnnFlashAttentionVarLenScoreGetWorkspaceSize,aclnnFlashAttentionVarLenScore),ref_causal),1e-5);
    report("SparseFlashAttention[mask]", norm_err(run_att(q,k,v,&mask,scale,false,aclnnSparseFlashAttentionGetWorkspaceSize,aclnnSparseFlashAttention),ref_masked),1e-5);

    // IncreFlashAttention (no causal arg -> plain)
    {
        int64_t qn=(int64_t)B*Nq*Sq*D, kn=(int64_t)B*Nkv*Skv*D;
        DevBuf dq(qn*4),dk(kn*4),dv(kn*4),dout(qn*4); dq.up(q.data()); dk.up(k.data()); dv.up(v.data());
        auto tq=mk({B,Nq,Sq,D},ACL_FLOAT,dq.p),tk=mk({B,Nkv,Skv,D},ACL_FLOAT,dk.p),tv=mk({B,Nkv,Skv,D},ACL_FLOAT,dv.p),to=mk({B,Nq,Sq,D},ACL_FLOAT,dout.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIncreFlashAttentionGetWorkspaceSize(tq,tk,tv,nullptr,scale,(int64_t)Nq,to,w,e);},aclnnIncreFlashAttention);
        std::vector<float> h(qn); dout.down(h.data());
        report("IncreFlashAttention", norm_err(h,ref_plain),1e-5);
        aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to);
    }

    // ---- RoPE family (rows,D); cos/sin per-row. mode 0 = rotate-half, mode 1 = interleaved ----
    {
        const int R=5,Dd=8,h=Dd/2; auto x=randv(R*Dd,-1,1),cs=randv(R*Dd,-1,1),sn=randv(R*Dd,-1,1);
        // mode 0 reference
        std::vector<double> r0(R*Dd);
        for(int r=0;r<R;r++)for(int d=0;d<Dd;d++){ double xp=(d<h)?x[r*Dd+d+h]:x[r*Dd+d-h];
            r0[r*Dd+d]=(d<h)? (double)x[r*Dd+d]*cs[r*Dd+d]-xp*sn[r*Dd+d] : (double)x[r*Dd+d]*cs[r*Dd+d]+xp*sn[r*Dd+d]; }
        // mode 1 (interleaved) reference
        std::vector<double> r1(R*Dd);
        for(int r=0;r<R;r++)for(int d=0;d<Dd;d++){ int kk=d/2; double xp=(d%2==0)?x[r*Dd+2*kk+1]:x[r*Dd+2*kk];
            r1[r*Dd+d]=(d%2==0)? (double)x[r*Dd+d]*cs[r*Dd+d]-xp*sn[r*Dd+d] : (double)x[r*Dd+d]*cs[r*Dd+d]+xp*sn[r*Dd+d]; }
        auto rope_run=[&](const char*nm,int64_t mode,const std::vector<double>&ref,
                          aclnnStatus(*gw)(const aclTensor*,const aclTensor*,const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**),
                          aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
            DevBuf dx(R*Dd*4),dc(R*Dd*4),ds(R*Dd*4),doo(R*Dd*4); dx.up(x.data()); dc.up(cs.data()); ds.up(sn.data());
            auto tx=mk({R,Dd},ACL_FLOAT,dx.p),tc=mk({R,Dd},ACL_FLOAT,dc.p),tsn=mk({R,Dd},ACL_FLOAT,ds.p),to=mk({R,Dd},ACL_FLOAT,doo.p);
            exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tx,tc,tsn,mode,to,w,e);},run);
            std::vector<float> o(R*Dd); doo.down(o.data()); report(nm,norm_err(o,ref),1e-5);
            aclDestroyTensor(tx);aclDestroyTensor(tc);aclDestroyTensor(tsn);aclDestroyTensor(to);
        };
        rope_run("RotaryPositionEmbeddingV2",0,r0,aclnnRotaryPositionEmbeddingV2GetWorkspaceSize,aclnnRotaryPositionEmbeddingV2);
        rope_run("RopeWithSinCosCache",0,r0,aclnnRopeWithSinCosCacheGetWorkspaceSize,aclnnRopeWithSinCosCache);
        rope_run("RopeWithSinCosCacheV2",1,r1,aclnnRopeWithSinCosCacheV2GetWorkspaceSize,aclnnRopeWithSinCosCacheV2);
        // InterleaveRope (mode 1)
        {
            DevBuf dx(R*Dd*4),dc(R*Dd*4),ds(R*Dd*4),doo(R*Dd*4); dx.up(x.data()); dc.up(cs.data()); ds.up(sn.data());
            auto tx=mk({R,Dd},ACL_FLOAT,dx.p),tc=mk({R,Dd},ACL_FLOAT,dc.p),tsn=mk({R,Dd},ACL_FLOAT,ds.p),to=mk({R,Dd},ACL_FLOAT,doo.p);
            exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInterleaveRopeGetWorkspaceSize(tx,tc,tsn,to,w,e);},aclnnInterleaveRope);
            std::vector<float> o(R*Dd); doo.down(o.data()); report("InterleaveRope",norm_err(o,r1),1e-5);
            aclDestroyTensor(tx);aclDestroyTensor(tc);aclDestroyTensor(tsn);aclDestroyTensor(to);
        }
    }

    // ---- ApplyRotaryPosEmbV2: q & k rope, BNSD, cos/sin broadcast over [S,D] (mode 0 half-split) ----
    {
        const int b=1,n=2,s=4,d=8,half=d/2;
        auto qx=randv(b*n*s*d,-1,1),kx=randv(b*n*s*d,-1,1),cs=randv(s*d,-1,1),sn=randv(s*d,-1,1);
        auto ref_bnsd=[&](const std::vector<float>&x){ std::vector<double> o(b*n*s*d);
            for(int i=0;i<b*n*s*d;i++){ int dd=i%d, si=(i/d)%s, base=i-dd; double c=cs[si*d+dd],ss=sn[si*d+dd];
                double rh=(dd<half)? -(double)x[base+dd+half] : (double)x[base+dd-half]; o[i]=(double)x[i]*c+rh*ss; } return o; };
        auto rq=ref_bnsd(qx), rk=ref_bnsd(kx);
        DevBuf dq(b*n*s*d*4),dk(b*n*s*d*4),dc(s*d*4),ds(s*d*4),dqo(b*n*s*d*4),dko(b*n*s*d*4);
        dq.up(qx.data()); dk.up(kx.data()); dc.up(cs.data()); ds.up(sn.data());
        auto tq=mk({b,n,s,d},ACL_FLOAT,dq.p),tk=mk({b,n,s,d},ACL_FLOAT,dk.p),tc=mk({s,d},ACL_FLOAT,dc.p),tsn=mk({s,d},ACL_FLOAT,ds.p),tqo=mk({b,n,s,d},ACL_FLOAT,dqo.p),tko=mk({b,n,s,d},ACL_FLOAT,dko.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyRotaryPosEmbV2GetWorkspaceSize(tq,tk,tc,tsn,0,tqo,tko,w,e);},aclnnApplyRotaryPosEmbV2);
        std::vector<float> oq(b*n*s*d),ok(b*n*s*d); dqo.down(oq.data()); dko.down(ok.data());
        double eq=norm_err(oq,rq), ek=norm_err(ok,rk); report("ApplyRotaryPosEmbV2",std::max(eq,ek),1e-5);
        aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tc);aclDestroyTensor(tsn);aclDestroyTensor(tqo);aclDestroyTensor(tko);
    }

    // ---- Ring family: online-softmax merge of two partials ----
    {
        const int R=4,Dd=5; auto o1=randv(R*Dd,-1,1),o2=randv(R*Dd,-1,1),l1=randv(R,-2,2),l2=randv(R,-2,2);
        std::vector<double> ro(R*Dd),rlse(R);
        for(int r=0;r<R;r++){ double m=std::max(l1[r],l2[r]),ea=std::exp(l1[r]-m),eb=std::exp(l2[r]-m),den=ea+eb;
            for(int d=0;d<Dd;d++) ro[r*Dd+d]=((double)o1[r*Dd+d]*ea+(double)o2[r*Dd+d]*eb)/den; rlse[r]=m+std::log(den); }
        auto ring_run=[&](const char*nm,aclnnStatus(*gw)(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**),
                          aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
            DevBuf d1(R*Dd*4),d2(R*Dd*4),dl1(R*4),dl2(R*4),doo(R*Dd*4),dlse(R*4);
            d1.up(o1.data()); d2.up(o2.data()); dl1.up(l1.data()); dl2.up(l2.data());
            auto t1=mk({R,Dd},ACL_FLOAT,d1.p),t2=mk({R,Dd},ACL_FLOAT,d2.p),tl1=mk({R},ACL_FLOAT,dl1.p),tl2=mk({R},ACL_FLOAT,dl2.p),to=mk({R,Dd},ACL_FLOAT,doo.p),tlse=mk({R},ACL_FLOAT,dlse.p);
            exec2([&](uint64_t*w,aclOpExecutor**e){return gw(t1,tl1,t2,tl2,to,tlse,w,e);},run);
            std::vector<float> oo(R*Dd),ol(R); doo.down(oo.data()); dlse.down(ol.data());
            report(nm,std::max(norm_err(oo,ro),norm_err(ol,rlse)),1e-5);
            aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tl1);aclDestroyTensor(tl2);aclDestroyTensor(to);aclDestroyTensor(tlse);
        };
        ring_run("RingAttentionUpdateV2",aclnnRingAttentionUpdateV2GetWorkspaceSize,aclnnRingAttentionUpdateV2);
        ring_run("AttentionUpdate",aclnnAttentionUpdateGetWorkspaceSize,aclnnAttentionUpdate);
    }

    // ---- NRC family: RMSNorm(x)*gamma then rotate-half RoPE ----
    {
        const int R=4,Dd=8,half=Dd/2; double eps=1e-5;
        auto x=randv(R*Dd,-1,1),g=randv(Dd,0.5,1.5),cs=randv(R*Dd,-1,1),sn=randv(R*Dd,-1,1);
        std::vector<double> ref(R*Dd);
        for(int r=0;r<R;r++){ double ss=0; for(int d=0;d<Dd;d++) ss+=(double)x[r*Dd+d]*x[r*Dd+d]; double rms=1.0/std::sqrt(ss/Dd+eps);
            std::vector<double> nrm(Dd); for(int d=0;d<Dd;d++) nrm[d]=(double)x[r*Dd+d]*rms*g[d];
            for(int d=0;d<Dd;d++){ double c=cs[r*Dd+d],s2=sn[r*Dd+d];
                ref[r*Dd+d]=(d<half)? nrm[d]*c-nrm[d+half]*s2 : nrm[d]*c+nrm[d-half]*s2; } }
        auto nrc_run=[&](const char*nm,aclnnStatus(*gw)(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,int64_t,aclTensor*,uint64_t*,aclOpExecutor**),
                         aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
            DevBuf dx(R*Dd*4),dg(Dd*4),dc(R*Dd*4),ds(R*Dd*4),doo(R*Dd*4);
            dx.up(x.data()); dg.up(g.data()); dc.up(cs.data()); ds.up(sn.data());
            auto tx=mk({R,Dd},ACL_FLOAT,dx.p),tg=mk({Dd},ACL_FLOAT,dg.p),tc=mk({R,Dd},ACL_FLOAT,dc.p),tsn=mk({R,Dd},ACL_FLOAT,ds.p),to=mk({R,Dd},ACL_FLOAT,doo.p);
            exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tx,tg,tc,tsn,eps,0,to,w,e);},run);
            std::vector<float> o(R*Dd); doo.down(o.data()); report(nm,norm_err(o,ref),1e-5);
            aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tc);aclDestroyTensor(tsn);aclDestroyTensor(to);
        };
        nrc_run("KvRmsNormRopeCache",aclnnKvRmsNormRopeCacheGetWorkspaceSize,aclnnKvRmsNormRopeCache);
        nrc_run("QkvRmsNormRopeCache",aclnnQkvRmsNormRopeCacheGetWorkspaceSize,aclnnQkvRmsNormRopeCache);
        nrc_run("NormRopeConcat",aclnnNormRopeConcatGetWorkspaceSize,aclnnNormRopeConcat);
    }

    // ---- NsaCompressWithCache: block mean-pool ----
    {
        const int Nb=3,bs=4,Dd=6; auto x=randv(Nb*bs*Dd,-2,2);
        std::vector<double> ref(Nb*Dd);
        for(int b=0;b<Nb;b++)for(int d=0;d<Dd;d++){ double s=0; for(int j=0;j<bs;j++)s+=x[(b*bs+j)*Dd+d]; ref[b*Dd+d]=s/bs; }
        DevBuf dx(Nb*bs*Dd*4),doo(Nb*Dd*4); dx.up(x.data());
        auto tx=mk({Nb*bs,Dd},ACL_FLOAT,dx.p),to=mk({Nb,Dd},ACL_FLOAT,doo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNsaCompressWithCacheGetWorkspaceSize(tx,(int64_t)bs,to,w,e);},aclnnNsaCompressWithCache);
        std::vector<float> o(Nb*Dd); doo.down(o.data()); report("NsaCompressWithCache",norm_err(o,ref),1e-5);
        aclDestroyTensor(tx);aclDestroyTensor(to);
    }

    // ---- A2F family: layout pass-through (copy) ----
    {
        const int n=4,d=7; auto x=randv(n*d,-1,1); std::vector<double> ref(x.begin(),x.end());
        auto a2f_run=[&](const char*nm,aclnnStatus(*gw)(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**),
                         aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
            DevBuf dx(n*d*4),doo(n*d*4); dx.up(x.data());
            auto tx=mk({n,d},ACL_FLOAT,dx.p),to=mk({n,d},ACL_FLOAT,doo.p);
            exec2([&](uint64_t*w,aclOpExecutor**e){return gw(tx,to,w,e);},run);
            std::vector<float> o(n*d); doo.down(o.data()); report(nm,norm_err(o,ref),1e-6);
            aclDestroyTensor(tx);aclDestroyTensor(to);
        };
        a2f_run("AttentionToFFN",aclnnAttentionToFFNGetWorkspaceSize,aclnnAttentionToFFN);
        a2f_run("FFNToAttention",aclnnFFNToAttentionGetWorkspaceSize,aclnnFFNToAttention);
    }

    // ---- MaskedSoftmaxWithRelPosBias: softmax(x*scale + relPosBias + mask) over last dim ----
    {
        const int R=4,Dd=8; double sc=1.5;
        auto x=randv(R*Dd,-1,1),rp=randv(R*Dd,-1,1),mk_=randv(R*Dd,-3,0);
        std::vector<double> ref(R*Dd);
        for(int r=0;r<R;r++){ double mx=-1e300; for(int d=0;d<Dd;d++){ double vv=(double)x[r*Dd+d]*sc+rp[r*Dd+d]+mk_[r*Dd+d]; mx=std::max(mx,vv);}
            double sm=0; for(int d=0;d<Dd;d++){ double vv=(double)x[r*Dd+d]*sc+rp[r*Dd+d]+mk_[r*Dd+d]; sm+=std::exp(vv-mx);}
            for(int d=0;d<Dd;d++){ double vv=(double)x[r*Dd+d]*sc+rp[r*Dd+d]+mk_[r*Dd+d]; ref[r*Dd+d]=std::exp(vv-mx)/sm; } }
        DevBuf dx(R*Dd*4),drp(R*Dd*4),dmk(R*Dd*4),doo(R*Dd*4); dx.up(x.data()); drp.up(rp.data()); dmk.up(mk_.data());
        auto tx=mk({R,Dd},ACL_FLOAT,dx.p),trp=mk({R,Dd},ACL_FLOAT,drp.p),tmk=mk({R,Dd},ACL_FLOAT,dmk.p),to=mk({R,Dd},ACL_FLOAT,doo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(tx,tmk,trp,sc,to,w,e);},aclnnMaskedSoftmaxWithRelPosBias);
        std::vector<float> o(R*Dd); doo.down(o.data()); report("MaskedSoftmaxWithRelPosBias",norm_err(o,ref),1e-5);
        aclDestroyTensor(tx);aclDestroyTensor(trp);aclDestroyTensor(tmk);aclDestroyTensor(to);
    }

    return finish();
}
