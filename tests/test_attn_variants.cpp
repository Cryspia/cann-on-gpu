// Attention version variants (V2–V5): each must match its V1 core bit-for-bit (shared kernel).
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static const int B=2, Nq=4, Nkv=2, Sq=8, Skv=8, D=16;

// Run an attention op into a fresh output buffer and return host result.
template <typename G, typename R>
static std::vector<float> run_attn(const std::vector<float> &q, const std::vector<float> &k, const std::vector<float> &v, G getws, R run) {
    int64_t qn=B*Nq*Sq*D, kn=B*Nkv*Skv*D, on=B*Nq*Sq*D;
    DevBuf dq(qn*4),dk(kn*4),dv(kn*4),dout(on*4); dq.up(q.data()); dk.up(k.data()); dv.up(v.data());
    aclTensor *tq=mk({B,Nq,Sq,D},ACL_FLOAT,dq.p),*tk=mk({B,Nkv,Skv,D},ACL_FLOAT,dk.p),*tv=mk({B,Nkv,Skv,D},ACL_FLOAT,dv.p),*to=mk({B,Nq,Sq,D},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return getws(tq,tk,tv,to,w,e);}, run);
    std::vector<float> h(on); dout.down(h.data());
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to);
    return h;
}
static double diff(const std::vector<float>&a,const std::vector<float>&b){ double me=0,mr=0; for(size_t i=0;i<a.size();i++){me=std::max(me,std::fabs((double)a[i]-b[i]));mr=std::max(mr,std::fabs((double)b[i]));} return me/(mr+1e-9); }

int main() {
    init(); srand(23);
    double scale = 1.0/std::sqrt((double)D);
    auto q=randv(B*Nq*Sq*D,-1,1), k=randv(B*Nkv*Skv*D,-1,1), v=randv(B*Nkv*Skv*D,-1,1);
    auto ref = run_attn(q,k,v,
        [&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},
        aclnnFlashAttentionScore);
    // FA score variants
    report("FlashAttentionScoreV2", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreV2GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnFlashAttentionScoreV2),ref),1e-6);
    report("FlashAttentionScoreV3", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreV3GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnFlashAttentionScoreV3),ref),1e-6);
    report("FlashAttentionScoreV4", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreV4GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnFlashAttentionScoreV4),ref),1e-6);
    // Prompt variants (causal=false to match ref)
    report("PromptFlashAttentionV2", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnPromptFlashAttentionV2GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnPromptFlashAttentionV2),ref),1e-6);
    report("PromptFlashAttentionV3", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnPromptFlashAttentionV3GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnPromptFlashAttentionV3),ref),1e-6);
    // Incre variants (no causal arg)
    report("IncreFlashAttentionV2", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnIncreFlashAttentionV2GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,to,w,e);},aclnnIncreFlashAttentionV2),ref),1e-6);
    report("IncreFlashAttentionV3", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnIncreFlashAttentionV3GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,to,w,e);},aclnnIncreFlashAttentionV3),ref),1e-6);
    report("IncreFlashAttentionV4", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnIncreFlashAttentionV4GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,to,w,e);},aclnnIncreFlashAttentionV4),ref),1e-6);
    // FusedInfer variants
    report("FusedInferAttentionScoreV2", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFusedInferAttentionScoreV2GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnFusedInferAttentionScoreV2),ref),1e-6);
    report("FusedInferAttentionScoreV5", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnFusedInferAttentionScoreV5GetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnFusedInferAttentionScoreV5),ref),1e-6);
    // BlockSparseAttention dense fallback == FlashAttentionScore ref
    report("BlockSparseAttention", diff(run_attn(q,k,v,[&](aclTensor*tq,aclTensor*tk,aclTensor*tv,aclTensor*to,uint64_t*w,aclOpExecutor**e){return aclnnBlockSparseAttentionGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nq,false,to,w,e);},aclnnBlockSparseAttention),ref),1e-6);
    { // RotaryPositionEmbedding rotate-half: out[:, :h]=x1*c-x2*s; out[:,h:]=x2*c+x1*s
      const int R=4,Dd=8,h=Dd/2; auto x=randv(R*Dd,-1,1),cs=randv(R*Dd,-1,1),sn=randv(R*Dd,-1,1);
      DevBuf dx(R*Dd*4),dc(R*Dd*4),ds(R*Dd*4),do_(R*Dd*4); dx.up(x.data()); dc.up(cs.data()); ds.up(sn.data());
      auto tx=mk({R,Dd},ACL_FLOAT,dx.p),tc=mk({R,Dd},ACL_FLOAT,dc.p),tsn=mk({R,Dd},ACL_FLOAT,ds.p),to=mk({R,Dd},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRotaryPositionEmbeddingGetWorkspaceSize(tx,tc,tsn,0,to,w,e);}, aclnnRotaryPositionEmbedding);
      std::vector<float> o(R*Dd); do_.down(o.data()); double me=0,mr=0;
      for(int r=0;r<R;r++)for(int d=0;d<Dd;d++){ double rf; if(d<h) rf=x[r*Dd+d]*cs[r*Dd+d]-x[r*Dd+d+h]*sn[r*Dd+d]; else rf=x[r*Dd+d]*cs[r*Dd+d]+x[r*Dd+d-h]*sn[r*Dd+d];
        me=std::max(me,std::fabs(o[r*Dd+d]-rf)); mr=std::max(mr,std::fabs(rf)); }
      report("RotaryPositionEmbedding", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tc);aclDestroyTensor(tsn);aclDestroyTensor(to); }
    { // RingAttentionUpdate online merge
      const int R=3,Dd=4; auto o1=randv(R*Dd,-1,1),o2=randv(R*Dd,-1,1),l1=randv(R,-2,2),l2=randv(R,-2,2);
      DevBuf d1(R*Dd*4),d2(R*Dd*4),dl1(R*4),dl2(R*4),doo(R*Dd*4),dlse(R*4); d1.up(o1.data()); d2.up(o2.data()); dl1.up(l1.data()); dl2.up(l2.data());
      auto t1=mk({R,Dd},ACL_FLOAT,d1.p),t2=mk({R,Dd},ACL_FLOAT,d2.p),tl1=mk({R},ACL_FLOAT,dl1.p),tl2=mk({R},ACL_FLOAT,dl2.p),to=mk({R,Dd},ACL_FLOAT,doo.p),tlse=mk({R},ACL_FLOAT,dlse.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRingAttentionUpdateGetWorkspaceSize(t1,tl1,t2,tl2,to,tlse,w,e);}, aclnnRingAttentionUpdate);
      std::vector<float> o(R*Dd); doo.down(o.data()); double me=0,mr=0;
      for(int r=0;r<R;r++){ double m=std::max(l1[r],l2[r]),ea=std::exp(l1[r]-m),eb=std::exp(l2[r]-m),den=ea+eb;
        for(int d=0;d<Dd;d++){ double rf=(o1[r*Dd+d]*ea+o2[r*Dd+d]*eb)/den; me=std::max(me,std::fabs(o[r*Dd+d]-rf)); mr=std::max(mr,std::fabs(rf)); } }
      report("RingAttentionUpdate", me/(mr+1e-9), 1e-5); aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tl1);aclDestroyTensor(tl2);aclDestroyTensor(to);aclDestroyTensor(tlse); }
    { // NsaCompress block mean-pool
      const int Nb=3,bs=4,Dd=5; auto x=randv(Nb*bs*Dd,-2,2);
      DevBuf dx(Nb*bs*Dd*4),do_(Nb*Dd*4); dx.up(x.data());
      auto tx=mk({Nb*bs,Dd},ACL_FLOAT,dx.p),to=mk({Nb,Dd},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNsaCompressGetWorkspaceSize(tx,(int64_t)bs,to,w,e);}, aclnnNsaCompress);
      std::vector<float> o(Nb*Dd); do_.down(o.data()); double me=0,mr=0;
      for(int b=0;b<Nb;b++)for(int d=0;d<Dd;d++){ double s=0; for(int j=0;j<bs;j++)s+=x[(b*bs+j)*Dd+d]; s/=bs; me=std::max(me,std::fabs(o[b*Dd+d]-s)); mr=std::max(mr,std::fabs(s)); }
      report("NsaCompress", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(to); }
    return finish();
}
