// Full tiny Qwen3-Next ("Qwen3.5") forward on the cann-on-gpu shim, cross-checked against the HF reference
// dumped by export_qwen35.py. Hybrid stack: GatedDeltaNet linear-attention layers + gated full-attention layer +
// per-layer sparse MoE. Validates per-layer hidden states and the final logits / greedy tokens.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cmath>
#include <map>
using namespace hn;

static std::string DIR;
struct Tn { DevBuf *d; std::vector<int64_t> s; int64_t numel() const { int64_t n=1; for(auto x:s)n*=x; return n; } };
static std::map<std::string,DevBuf*> WCACHE;
static long fsize(const std::string&p){ FILE*f=fopen(p.c_str(),"rb"); if(!f)return -1; fseek(f,0,SEEK_END); long n=ftell(f); fclose(f); return n; }
static DevBuf* loadw(const std::string&name){ auto it=WCACHE.find(name); if(it!=WCACHE.end())return it->second;
    long b=fsize(DIR+"/"+name); if(b<0){printf("[FATAL] missing %s\n",name.c_str());exit(1);} std::vector<float> h(b/4);
    FILE*f=fopen((DIR+"/"+name).c_str(),"rb"); fread(h.data(),4,h.size(),f); fclose(f); auto*d=new DevBuf(b); d->up(h.data()); WCACHE[name]=d; return d; }
static aclTensor* T(const Tn&t){ return mk(t.s, ACL_FLOAT, t.d->p); }
static aclTensor* Tv(DevBuf*d, std::vector<int64_t> s){ return mk(s, ACL_FLOAT, d->p); }
static Tn alloc(std::vector<int64_t> s){ int64_t n=1; for(auto x:s)n*=x; return {new DevBuf(n*4), s}; }
static std::vector<float> dn(const Tn&t){ std::vector<float> h(t.numel()); t.d->down(h.data()); return h; }

int main(int argc, char**argv){
    DIR = argc>1?argv[1]:"data_full"; init();
    // ---- meta ----
    FILE*mf=fopen((DIR+"/meta.txt").c_str(),"r"); char buf[512];
    int Dh,L,NH,NKV,HD,V,S,rd, GNK,GHKD,GNV,GHVD,key_dim,value_dim,ck, E,topk,moe,shared; double eps;
    fgets(buf,512,mf); sscanf(buf,"D=%d L=%d NH=%d NKV=%d HD=%d V=%d S=%d eps=%lf rotary_dim=%d",&Dh,&L,&NH,&NKV,&HD,&V,&S,&eps,&rd);
    fgets(buf,512,mf); sscanf(buf,"GNK=%d GHKD=%d GNV=%d GHVD=%d key_dim=%d value_dim=%d conv_k=%d",&GNK,&GHKD,&GNV,&GHVD,&key_dim,&value_dim,&ck);
    fgets(buf,512,mf); sscanf(buf,"E=%d topk=%d moe_inter=%d shared_inter=%d",&E,&topk,&moe,&shared);
    fgets(buf,512,mf); std::string ltline=buf; std::vector<int> isLin(L);
    { std::string s=ltline.substr(ltline.find('=')+1); size_t p=0; for(int i=0;i<L;i++){ size_t c=s.find(',',p); std::string tok=s.substr(p, c==std::string::npos?std::string::npos:c-p); isLin[i]= tok.find("linear")!=std::string::npos; p=c+1; } }
    fclose(mf);
    const int D=Dh;

    // ---- shim op helpers ----
    auto mmT=[&](aclTensor*a,int64_t M,int64_t K, DevBuf*wb,int64_t N){ Tn o=alloc({M,N}); aclTensor*w=Tv(wb,{N,K});
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulBiasGetWorkspaceSize(a,w,nullptr,false,true,0,T(o),ws,e);},aclnnMatmulBias); return o; };
    auto perm=[&](aclTensor*a,std::vector<int64_t> in,std::vector<int64_t> dims){ std::vector<int64_t> od; for(auto d:dims)od.push_back(in[d]);
        Tn o=alloc(od); aclIntArray*dd=aclCreateIntArray(dims.data(),dims.size());
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,dd,T(o),ws,e);},aclnnPermute); return o; };
    auto un1=[&](aclTensor*a,int64_t n,aclnnStatus(*gw)(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**),aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
        Tn o=alloc({n}); exec2([&](uint64_t*ws,aclOpExecutor**e){return gw(a,T(o),ws,e);},r); return o; };
    auto mul=[&](aclTensor*a,aclTensor*b,std::vector<int64_t> os){ Tn o=alloc(os);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulGetWorkspaceSize(a,b,T(o),ws,e);},aclnnMul); return o; };
    auto add=[&](aclTensor*a,aclTensor*b,std::vector<int64_t> os){ Tn o=alloc(os);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(a,b,nullptr,T(o),ws,e);},aclnnAdd); return o; };
    auto sigmoid=[&](aclTensor*a,int64_t n){ return un1(a,n,aclnnSigmoidGetWorkspaceSize,aclnnSigmoid); };
    auto silu=[&](aclTensor*a,int64_t n){ return un1(a,n,aclnnSiluGetWorkspaceSize,aclnnSilu); };
    // RMSNorm over last `cols`; zero-centered uses (1+weight)
    auto rms=[&](aclTensor*x,int64_t rows,int64_t cols,DevBuf*wb,bool zc){ DevBuf*wuse=wb;
        if(zc){ Tn w1=alloc({cols}); float one=1; aclScalar*s=aclCreateScalar(&one,ACL_FLOAT);
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAddsGetWorkspaceSize(Tv(wb,{cols}),s,nullptr,T(w1),ws,e);},aclnnAdds); wuse=w1.d; }
        Tn o=alloc({rows,cols}); exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,Tv(wuse,{cols}),eps,T(o),ws,e);},aclnnRmsNorm); return o; };
    auto slicelast=[&](aclTensor*a,std::vector<int64_t> sh,int64_t s0,int64_t len){ int dimL=sh.size()-1; std::vector<int64_t> os=sh; os[dimL]=len; Tn o=alloc(os);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnSliceGetWorkspaceSize(a,dimL,s0,s0+len,1,T(o),ws,e);},aclnnSlice); return o; };

    // gather heads along dim0 (GQA / GDN repeat): [nin,...]->[nout,...] by idx[g]=g/(nout/nin)
    auto headrep0=[&](aclTensor*a,std::vector<int64_t> sh,int nin,int nout){ static std::map<int,DevBuf*> idxc; DevBuf*ib;
        auto it=idxc.find(nout*100+nin); if(it!=idxc.end()) ib=it->second; else { std::vector<int64_t> ix(nout); for(int i=0;i<nout;i++) ix[i]=i/(nout/nin); ib=new DevBuf(nout*8); ib->up(ix.data()); idxc[nout*100+nin]=ib; }
        std::vector<int64_t> os=sh; os[0]=nout; Tn o=alloc(os); aclTensor*ti=mk({(int64_t)nout},ACL_INT64,ib->p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGatherGetWorkspaceSize(a,0,ti,T(o),ws,e);},aclnnGather); return o; };

    // ---- RoPE cos/sin (partial, dim=rd) ----
    int half=rd/2; std::vector<float> invf(half); { FILE*f=fopen((DIR+"/inv_freq.bin").c_str(),"rb"); fread(invf.data(),4,half,f); fclose(f); }
    std::vector<float> cosv(S*rd), sinv(S*rd);
    for(int s=0;s<S;s++)for(int j=0;j<rd;j++){ double ang=s*(double)invf[j%half]; cosv[s*rd+j]=std::cos(ang); sinv[s*rd+j]=std::sin(ang); }
    DevBuf dcos(S*rd*4),dsin(S*rd*4); dcos.up(cosv.data()); dsin.up(sinv.data());

    // ---- embeddings: h = embed[ids] ----
    std::vector<float> idf(S); { FILE*f=fopen((DIR+"/ids.bin").c_str(),"rb"); fread(idf.data(),4,S,f); fclose(f); }
    std::vector<int64_t> ids(S); for(int i=0;i<S;i++) ids[i]=(int64_t)idf[i];
    DevBuf dids(S*8); dids.up(ids.data());
    Tn h=alloc({S,D});
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(Tv(loadw("embed.bin"),{V,D}),mk({S},ACL_INT64,dids.p),T(h),ws,e);},aclnnEmbedding);
    auto cmp=[&](const char*lbl,const Tn&t,const std::string&ref){ long b=fsize(DIR+"/"+ref); std::vector<float> r(b/4);
        FILE*f=fopen((DIR+"/"+ref).c_str(),"rb"); fread(r.data(),4,r.size(),f); fclose(f); auto g=dn(t);
        double me=0,mr=0; for(size_t i=0;i<r.size();i++){me=std::max(me,(double)std::fabs(g[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));}
        report(lbl, me/(mr+1e-9), 2e-3); };
    cmp("embed (hs0)", h, "hs0.bin");

    // ================= per-layer forward =================
    for(int li=0; li<L; li++){
        std::string P="l"+std::to_string(li)+".";
        Tn resid=h;
        Tn hn=rms(T(h),S,D,loadw(P+"input_ln.bin"),true);
        if(li==0) cmp("  ln0", hn, "dbg_ln0.bin");
        Tn mix;
        if(isLin[li]){
            // ---------- GatedDeltaNet ----------
            Tn q=mmT(T(hn),S,D,loadw(P+"Wq.bin"),key_dim), k=mmT(T(hn),S,D,loadw(P+"Wk.bin"),key_dim);
            Tn v=mmT(T(hn),S,D,loadw(P+"Wv.bin"),value_dim), z=mmT(T(hn),S,D,loadw(P+"Wz.bin"),value_dim);
            Tn bb=mmT(T(hn),S,D,loadw(P+"Wb.bin"),GNV), aa=mmT(T(hn),S,D,loadw(P+"Wa.bin"),GNV);
            int conv_ch=key_dim+key_dim+value_dim;
            Tn mixc=alloc({S,conv_ch}); { const aclTensor*ins[3]={T(q),T(k),T(v)};
                exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCatGetWorkspaceSize(ins,3,1,T(mixc),ws,e);},aclnnCat); }
            Tn mt=perm(T(mixc),{S,conv_ch},{1,0});
            Tn co=alloc({1,conv_ch,S});
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCausalConv1dGetWorkspaceSize(Tv(mt.d,{1,conv_ch,S}),Tv(loadw(P+"conv.bin"),{conv_ch,ck}),nullptr,1,T(co),ws,e);},aclnnCausalConv1d);
            Tn cot=perm(Tv(co.d,{conv_ch,S}),{conv_ch,S},{1,0});   // [S,128]
            Tn q2=slicelast(T(cot),{S,conv_ch},0,key_dim), k2=slicelast(T(cot),{S,conv_ch},key_dim,key_dim), v2=slicelast(T(cot),{S,conv_ch},2*key_dim,value_dim);
            Tn beta=sigmoid(Tv(bb.d,{(int64_t)S*GNV}),(int64_t)S*GNV);
            Tn apb=add(Tv(aa.d,{S,GNV}),Tv(loadw(P+"dt_bias.bin"),{GNV}),{S,GNV});
            Tn sp=un1(Tv(apb.d,{(int64_t)S*GNV}),(int64_t)S*GNV,aclnnSoftplusGetWorkspaceSize,aclnnSoftplus);
            Tn ealog=un1(Tv(loadw(P+"A_log.bin"),{GNV}),GNV,aclnnExpGetWorkspaceSize,aclnnExp);
            Tn neal=alloc({GNV}); { float m1=-1; aclScalar*s=aclCreateScalar(&m1,ACL_FLOAT); exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulsGetWorkspaceSize(T(ealog),s,T(neal),ws,e);},aclnnMuls); }
            Tn g=mul(Tv(sp.d,{S,GNV}),T(neal),{S,GNV});
            Tn gate=un1(Tv(g.d,{(int64_t)S*GNV}),(int64_t)S*GNV,aclnnExpGetWorkspaceSize,aclnnExp);
            // q2 [S,nk,hkd] -> permute [nk,S,hkd] -> headrep0 dim0 -> [nv,S,hkd]
            Tn qp=perm(Tv(q2.d,{S,GNK,GHKD}),{S,GNK,GHKD},{1,0,2});  // [nk,S,hkd]
            Tn kp=perm(Tv(k2.d,{S,GNK,GHKD}),{S,GNK,GHKD},{1,0,2});
            Tn qH=headrep0(T(qp),{GNK,S,GHKD},GNK,GNV);  // [nv,S,hkd]
            Tn kH=headrep0(T(kp),{GNK,S,GHKD},GNK,GNV);
            Tn vH=perm(Tv(v2.d,{S,GNV,GHVD}),{S,GNV,GHVD},{1,0,2});  // [nv,S,hvd]
            // l2norm q,k over last dim; scale q by 1/sqrt(hkd)
            Tn qn2=alloc({(int64_t)GNV*S,GHKD}); exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnLpNormalizeGetWorkspaceSize(Tv(qH.d,{(int64_t)GNV*S,GHKD}),2.0,1e-6,T(qn2),ws,e);},aclnnLpNormalize);
            Tn kn2=alloc({(int64_t)GNV*S,GHKD}); exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnLpNormalizeGetWorkspaceSize(Tv(kH.d,{(int64_t)GNV*S,GHKD}),2.0,1e-6,T(kn2),ws,e);},aclnnLpNormalize);
            { float sc=1.f/std::sqrt((float)GHKD); aclScalar*s=aclCreateScalar(&sc,ACL_FLOAT); Tn t2=alloc({(int64_t)GNV*S,GHKD});
              exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMulsGetWorkspaceSize(Tv(qn2.d,{(int64_t)GNV*S,GHKD}),s,T(t2),ws,e);},aclnnMuls); qn2=t2; }
            Tn betaT=perm(Tv(beta.d,{S,GNV}),{S,GNV},{1,0}), gateT=perm(Tv(gate.d,{S,GNV}),{S,GNV},{1,0});
            Tn core=alloc({1,GNV,S,GHVD});
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGatedDeltaRuleGetWorkspaceSize(Tv(qn2.d,{1,GNV,S,GHKD}),Tv(kn2.d,{1,GNV,S,GHKD}),Tv(vH.d,{1,GNV,S,GHVD}),Tv(betaT.d,{1,GNV,S}),Tv(gateT.d,{1,GNV,S}),T(core),ws,e);},aclnnGatedDeltaRule);
            Tn coreS=perm(Tv(core.d,{GNV,S,GHVD}),{GNV,S,GHVD},{1,0,2});  // [S,nv,hvd]
            // gated RMSNorm: (norm.w * rmsnorm(core)) * silu(z)
            Tn rn=rms(Tv(coreS.d,{(int64_t)S*GNV,GHVD}),(int64_t)S*GNV,GHVD,loadw(P+"gdn_norm.bin"),false);
            Tn sz=silu(Tv(z.d,{(int64_t)S*GNV*GHVD}),(int64_t)S*GNV*GHVD);
            Tn ng=mul(Tv(rn.d,{(int64_t)S*GNV*GHVD}),T(sz),{(int64_t)S*GNV*GHVD});
            mix=mmT(Tv(ng.d,{S,value_dim}),S,value_dim,loadw(P+"gdn_out.bin"),D);
        } else {
            // ---------- gated full attention ----------
            Tn qq=mmT(T(hn),S,D,loadw(P+"Wqq.bin"),NH*HD), gt=mmT(T(hn),S,D,loadw(P+"Wqg.bin"),NH*HD);
            Tn kp=mmT(T(hn),S,D,loadw(P+"k_proj.bin"),NKV*HD), vp=mmT(T(hn),S,D,loadw(P+"v_proj.bin"),NKV*HD);
            Tn qn=rms(Tv(qq.d,{(int64_t)S*NH,HD}),(int64_t)S*NH,HD,loadw(P+"q_norm.bin"),true);
            Tn kn=rms(Tv(kp.d,{(int64_t)S*NKV,HD}),(int64_t)S*NKV,HD,loadw(P+"k_norm.bin"),true);
            Tn qh=perm(Tv(qn.d,{S,NH,HD}),{S,NH,HD},{1,0,2}), kh=perm(Tv(kn.d,{S,NKV,HD}),{S,NKV,HD},{1,0,2}), vh=perm(Tv(vp.d,{S,NKV,HD}),{S,NKV,HD},{1,0,2});
            // partial RoPE on first rd dims
            Tn qrot=slicelast(Tv(qh.d,{NH,S,HD}),{NH,S,HD},0,rd), qpass=slicelast(Tv(qh.d,{NH,S,HD}),{NH,S,HD},rd,HD-rd);
            Tn krot=slicelast(Tv(kh.d,{NKV,S,HD}),{NKV,S,HD},0,rd), kpass=slicelast(Tv(kh.d,{NKV,S,HD}),{NKV,S,HD},rd,HD-rd);
            Tn qro=alloc({1,NH,S,rd}), kro=alloc({1,NKV,S,rd});
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(Tv(qrot.d,{1,NH,S,rd}),Tv(krot.d,{1,NKV,S,rd}),Tv(&dcos,{S,rd}),Tv(&dsin,{S,rd}),0,T(qro),T(kro),ws,e);},aclnnApplyRotaryPosEmb);
            // cat rotated + pass -> [N,S,HD]
            Tn qf=alloc({NH,S,HD}); { const aclTensor*ins[2]={Tv(qro.d,{NH,S,rd}),T(qpass)}; exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCatGetWorkspaceSize(ins,2,2,T(qf),ws,e);},aclnnCat); }
            Tn kf=alloc({NKV,S,HD}); { const aclTensor*ins[2]={Tv(kro.d,{NKV,S,rd}),T(kpass)}; exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnCatGetWorkspaceSize(ins,2,2,T(kf),ws,e);},aclnnCat); }
            Tn kE=headrep0(T(kf),{NKV,S,HD},NKV,NH), vE=headrep0(T(vh),{NKV,S,HD},NKV,NH);
            Tn ao=alloc({1,NH,S,HD}); float scale=1.f/std::sqrt((float)HD);
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(Tv(qf.d,{1,NH,S,HD}),Tv(kE.d,{1,NH,S,HD}),Tv(vE.d,{1,NH,S,HD}),nullptr,scale,NH,true,T(ao),ws,e);},aclnnFlashAttentionScore);
            Tn aoS=perm(Tv(ao.d,{NH,S,HD}),{NH,S,HD},{1,0,2});  // [S,NH,HD]
            Tn sg=sigmoid(Tv(gt.d,{(int64_t)S*NH*HD}),(int64_t)S*NH*HD);
            Tn gated=mul(Tv(aoS.d,{(int64_t)S*NH*HD}),T(sg),{(int64_t)S*NH*HD});
            mix=mmT(Tv(gated.d,{S,NH*HD}),S,NH*HD,loadw(P+"o_proj.bin"),D);
        }
        if(li==0) cmp("  la0(mix)", mix, "dbg_la0.bin");
        h=add(T(resid),T(mix),{S,D});

        // ---------- MoE FFN ----------
        Tn resid2=h;
        Tn hn2=rms(T(h),S,D,loadw(P+"post_ln.bin"),true);
        if(li==0) cmp("  post0", hn2, "dbg_post0.bin");
        // router
        Tn rl=mmT(T(hn2),S,D,loadw(P+"router.bin"),E);
        Tn rw=alloc({S,topk}); DevBuf didx(S*topk*4); aclTensor*tidx=mk({S,topk},ACL_INT32,didx.p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(T(rl),topk,T(rw),tidx,ws,e);},aclnnMoeGatingTopKSoftmax);
        std::vector<float> rwh=dn(rw); std::vector<int32_t> idxh(S*topk); didx.down(idxh.data());
        Tn moeout=alloc({S,D}); aclrtMemset(moeout.d->p,(size_t)S*D*4,0,(size_t)S*D*4);
        DevBuf*egu=loadw(P+"egu.bin"), *edn=loadw(P+"edn.bin");
        for(int e=0;e<E;e++){
            // per-token weight for expert e (0 if not selected)
            std::vector<float> we(S,0); for(int s=0;s<S;s++)for(int kk=0;kk<topk;kk++) if(idxh[s*topk+kk]==e) we[s]+=rwh[s*topk+kk];
            DevBuf dwe(S*4); dwe.up(we.data());
            aclTensor*tgu=mk({2*moe,D},ACL_FLOAT,(char*)egu->p+(size_t)e*2*moe*D*4);
            aclTensor*tdn=mk({D,moe},ACL_FLOAT,(char*)edn->p+(size_t)e*D*moe*4);
            // gate_up = hn2 @ gu^T -> [S,2*moe]
            Tn gup=alloc({S,2*moe}); exec2([&](uint64_t*ws,aclOpExecutor**ex){return aclnnMatmulBiasGetWorkspaceSize(T(hn2),tgu,nullptr,false,true,0,T(gup),ws,ex);},aclnnMatmulBias);
            Tn gpart=slicelast(T(gup),{S,2*moe},0,moe), upart=slicelast(T(gup),{S,2*moe},moe,moe);
            Tn sgg=silu(Tv(gpart.d,{(int64_t)S*moe}),(int64_t)S*moe);
            Tn act=mul(Tv(sgg.d,{(int64_t)S*moe}),Tv(upart.d,{(int64_t)S*moe}),{(int64_t)S*moe});
            Tn de=alloc({S,D}); exec2([&](uint64_t*ws,aclOpExecutor**ex){return aclnnMatmulBiasGetWorkspaceSize(Tv(act.d,{S,moe}),tdn,nullptr,false,true,0,T(de),ws,ex);},aclnnMatmulBias);
            Tn des=mul(T(de),Tv(&dwe,{S,1}),{S,D});      // scale by per-token weight
            moeout=add(T(moeout),T(des),{S,D});
        }
        Tn sg1=mmT(T(hn2),S,D,loadw(P+"sgate.bin"),shared), su1=mmT(T(hn2),S,D,loadw(P+"sup.bin"),shared);
        Tn ssil=silu(Tv(sg1.d,{(int64_t)S*shared}),(int64_t)S*shared);
        Tn sact=mul(Tv(ssil.d,{(int64_t)S*shared}),Tv(su1.d,{(int64_t)S*shared}),{(int64_t)S*shared});
        Tn sdn=mmT(Tv(sact.d,{S,shared}),S,shared,loadw(P+"sdown.bin"),D);
        // shared_expert_gate logit = per-token dot(hn2, sgate_g[0]); broadcast-mul + reduce-sum (avoids degenerate N=1 GEMM)
        Tn sgm=mul(T(hn2),Tv(loadw(P+"sgate_g.bin"),{1,D}),{S,D});
        Tn sgl=alloc({S}); { int64_t rdim[1]={1}; aclIntArray*rd1=aclCreateIntArray(rdim,1);
            exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnReduceSumGetWorkspaceSize(Tv(sgm.d,{S,D}),rd1,false,ACL_FLOAT,T(sgl),ws,e);},aclnnReduceSum); }
        Tn sgsig=sigmoid(Tv(sgl.d,{(int64_t)S}),(int64_t)S);
        Tn shr=mul(T(sdn),Tv(sgsig.d,{S,1}),{S,D});
        moeout=add(T(moeout),T(shr),{S,D});
        if(li==0) cmp("  moe0", moeout, "dbg_moe0.bin");
        h=add(T(resid2),T(moeout),{S,D});
        // HF's hidden_states[L] is post-final-norm, so only layers < L-1 map directly to hs[li+1]; the last is checked via final-norm + logits.
        if(li<L-1) cmp(("layer "+std::to_string(li)).c_str(), h, "hs"+std::to_string(li+1)+".bin");
    }
    // ---- final norm + lm_head ----
    Tn hf=rms(T(h),S,D,loadw("fnorm.bin"),true);
    cmp("final-norm (hsL)", hf, "hs"+std::to_string(L)+".bin");
    Tn logits=mmT(T(hf),S,D,loadw("lm_head.bin"),V);
    cmp("logits", logits, "logits.bin");
    // greedy argmax of last row vs gen[0]
    { auto lg=dn(logits); int best=0; for(int v=1;v<V;v++) if(lg[(S-1)*V+v]>lg[(S-1)*V+best]) best=v;
      std::vector<int64_t> gen(1); FILE*f=fopen((DIR+"/gen.bin").c_str(),"rb"); fread(gen.data(),8,1,f); fclose(f);
      report("greedy next-token", best==(int)gen[0]?0.0:1.0, 0); }
    return finish();
}
