// End-to-end real open-source small-model demo: runs the Qwen3-0.6B forward pass via the shim aclnn operator chain, cross-checked against HF reference logits/tokens.
// Pure ACL client (links only libascendcl.so, same as an Ascend application). Weights and reference data exported by export_qwen3.py into data/.
// Architecture: Embedding -> [RMSNorm -> QKV(no bias) -> per-head QK-RMSNorm -> RoPE -> GQA causal attention -> O+residual -> RMSNorm -> SwiGLU MLP+residual]*L -> RMSNorm -> lm_head(tied).
// Validation: (1) prefill full-position logits vs HF normalized error + per-position top-1 token hit rate; (2) greedy decode of N_GEN tokens matched against HF token-by-token.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

static aclrtStream S_;
#define CHECK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

// ---- Configuration (read from config.txt) ----
static int64_t D,L,Hq,Hkv,hd,F,V,Sprompt,NGEN; static int tied; static double EPS,THETA;
static int64_t Dq, Dkv;
static std::string DIR;

// ---- Compute dtype (default fp32; set QWEN_BF16=1 / QWEN_FP16=1 to run the full stack at deployment precision) ----
static aclDataType DT=ACL_FLOAT; static int ES=4; static const char* DTNAME="fp32";
// Host-side f32->fp16/bf16 bit-cast conversion (no CUDA headers required)
static uint16_t f2h(float f){ uint32_t x; std::memcpy(&x,&f,4);
    uint32_t s=(x>>16)&0x8000u; int32_t e=(int32_t)((x>>23)&0xff)-127+15; uint32_t m=x&0x7fffffu;
    if(e<=0){ if(e<-10) return (uint16_t)s; m|=0x800000u; int sh=14-e; uint16_t h=(uint16_t)(s|(m>>sh)); if((m>>(sh-1))&1)h++; return h; }
    if(e>=31) return (uint16_t)(s|0x7c00u);
    uint16_t h=(uint16_t)(s|((uint32_t)e<<10)|(m>>13)); if(m&0x1000u)h++; return h; }
static uint16_t f2bf(float f){ uint32_t x; std::memcpy(&x,&f,4); uint32_t r=(x>>16)+((x>>15)&1); return (uint16_t)r; }
static float h2f(uint16_t h){ uint32_t s=(h&0x8000u)<<16; uint32_t e=(h>>10)&0x1f, m=h&0x3ff; uint32_t o;
    if(e==0){ if(m==0)o=s; else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; o=s|(e<<23)|(m<<13);} }
    else if(e==31) o=s|0x7f800000u|(m<<13); else o=s|((e-15+127)<<23)|(m<<13);
    float f; std::memcpy(&f,&o,4); return f; }
static float bf2f(uint16_t b){ uint32_t o=((uint32_t)b)<<16; float f; std::memcpy(&f,&o,4); return f; }
// Encode fp32 host data into dst byte buffer according to DT
static void enc(const std::vector<float>&v, void*dst){
    if(DT==ACL_FLOAT) std::memcpy(dst,v.data(),v.size()*4);
    else { uint16_t*o=(uint16_t*)dst; for(size_t i=0;i<v.size();i++) o[i]=(DT==ACL_BF16)?f2bf(v[i]):f2h(v[i]); } }
static float dec1(const void*p,int64_t i){ if(DT==ACL_FLOAT) return ((const float*)p)[i];
    return (DT==ACL_BF16)?bf2f(((const uint16_t*)p)[i]):h2f(((const uint16_t*)p)[i]); }

// ---- Device helpers ----
static std::vector<void*> g_frees;
static void* dalloc(size_t b){ void*p; CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); g_frees.push_back(p); return p; }
static aclTensor* T(std::vector<int64_t> d, void*p, aclDataType dt){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); }
static aclTensor* T(std::vector<int64_t> d, void*p){ return T(d,p,DT); }   // default compute dtype
template<typename G> static void run2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp,ws,ex,S_)); CHECK(aclrtSynchronizeStream(S_)); if(wsp)aclrtFree(wsp);
}
static std::vector<float> rfile(const std::string&name, int64_t n){
    std::string p=DIR+"/"+name; FILE*f=fopen(p.c_str(),"rb"); if(!f){printf("[FATAL] open %s\n",p.c_str());exit(1);}
    std::vector<float> v(n); if((int64_t)fread(v.data(),4,n,f)!=n){printf("[FATAL] short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<int64_t> rfile64(const std::string&name, int64_t n){
    std::string p=DIR+"/"+name; FILE*f=fopen(p.c_str(),"rb"); if(!f){printf("[FATAL] open %s\n",p.c_str());exit(1);}
    std::vector<int64_t> v(n); if((int64_t)fread(v.data(),8,n,f)!=n){printf("[FATAL] short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static void* up_f(const std::string&name, int64_t n){ auto v=rfile(name,n); void*p=dalloc(n*ES);
    std::vector<char> buf(n*ES); enc(v,buf.data()); CHECK(aclrtMemcpy(p,n*ES,buf.data(),n*ES,ACL_MEMCPY_HOST_TO_DEVICE)); return p; }

// ---- Thin operator wrappers ----
// MatmulBias: out[M,N] = self[M,K] @ other[N,K]^T (+bias[N]), transB=true uses HF native [out,in] weight layout directly (bias may be null)
static void* mmb(void*A,int64_t M,int64_t K,void*W,int64_t N,void*bias){ void*C=dalloc(M*N*ES);
    aclTensor*a=T({M,K},A),*w=T({N,K},W),*c=T({M,N},C),*b=bias?T({N},bias):nullptr;
    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulBiasGetWorkspaceSize(a,w,b,false,true,0,c,ws,e);},aclnnMatmulBias); return C; }
// RMSNorm over the last `cols` dims with gamma[cols]; used both for hidden-state norms (cols=D) and per-head QK-norm (cols=hd)
static void* rms_n(void*X,int64_t rows,int64_t cols,void*g){ void*Y=dalloc(rows*cols*ES);
    aclTensor*x=T({rows,cols},X),*gg=T({cols},g),*y=T({rows,cols},Y);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,gg,EPS,y,w,e);},aclnnRmsNorm); return Y; }
static void* rms(void*X,int64_t rows,void*g){ return rms_n(X,rows,D,g); }
static void* add(void*A,void*B,int64_t n){ void*C=dalloc(n*ES);
    aclTensor*a=T({n},A),*b=T({n},B),*c=T({n},C);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(a,b,nullptr,c,w,e);},aclnnAdd); return C; }
static void* mul(void*A,void*B,int64_t n){ void*C=dalloc(n*ES);
    aclTensor*a=T({n},A),*b=T({n},B),*c=T({n},C);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnMulGetWorkspaceSize(a,b,c,w,e);},aclnnMul); return C; }
static void* silu(void*A,int64_t n){ void*C=dalloc(n*ES);
    aclTensor*a=T({n},A),*c=T({n},C);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnSiluGetWorkspaceSize(a,c,w,e);},aclnnSilu); return C; }
// [s,H*hd] -> (view as [s,H,hd] then permute{1,0,2}) -> [H,s,hd]
static void* to_bnsd(void*src,int64_t s,int64_t H){ void*o=dalloc(H*s*hd*ES);
    aclTensor*a=T({s,H,hd},src),*b=T({H,s,hd},o); int64_t pd[3]={1,0,2}; aclIntArray*p=aclCreateIntArray(pd,3);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,p,b,w,e);},aclnnPermute); aclDestroyIntArray(p); return o; }
static void* from_bnsd(void*src,int64_t s,int64_t H){ void*o=dalloc(s*H*hd*ES);
    aclTensor*a=T({H,s,hd},src),*b=T({s,H,hd},o); int64_t pd[3]={1,0,2}; aclIntArray*p=aclCreateIntArray(pd,3);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,p,b,w,e);},aclnnPermute); aclDestroyIntArray(p); return o; }

// ---- Weights (uploaded once, permanently resident) ----
struct LW { void *g1,*qw,*kw,*vw,*qn,*kn,*ow,*g2,*gatew,*upw,*downw; };
static std::vector<LW> gL; static void *gEmb,*gNorm,*gCos,*gSin;

// Debug: snapshot of the residual stream at each layer entry and final norm output (same layout as HF output_hidden_states, for layer-by-layer bisect)
static bool g_dump=false; static std::vector<std::vector<float>> g_snaps; static int64_t g_dumpS=0;
static void snap(void*x,int64_t s){ if(!g_dump||s!=g_dumpS||DT!=ACL_FLOAT) return; std::vector<float> h(s*D);
    CHECK(aclrtMemcpy(h.data(),s*D*4,x,s*D*4,ACL_MEMCPY_DEVICE_TO_HOST)); g_snaps.push_back(std::move(h)); }

// Debug: dump layer-0 internal tensors
static bool g_dbgL0=false;
static void dumpT(const char*name,void*p,int64_t n){ if(!g_dbgL0||DT!=ACL_FLOAT) return;
    std::vector<float> h(n); CHECK(aclrtMemcpy(h.data(),n*4,p,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::string fp=DIR+"/l0_"+name+".bin"; FILE*f=fopen(fp.c_str(),"wb"); fwrite(h.data(),4,n,f); fclose(f); }

// Prefill forward pass: ids[s] -> logits[s,V] (all positions, causal attention)
static void* forward(const std::vector<int64_t>&ids){
    int64_t s=ids.size();
    void*dids=dalloc(s*8); CHECK(aclrtMemcpy(dids,s*8,ids.data(),s*8,ACL_MEMCPY_HOST_TO_DEVICE));
    void*x=dalloc(s*D*ES);
    { aclTensor*w=T({V,D},gEmb),*id=T({s},dids,ACL_INT64),*o=T({s,D},x);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(w,id,o,ws,e);},aclnnEmbedding); }
    double scale=1.0/std::sqrt((double)hd);
    for(int64_t l=0;l<L;l++){
        snap(x,s);                                    // layer-entry residual stream (HF hs[l])
        LW&w=gL[l];
        void*h=rms(x,s,w.g1);
        void*q=mmb(h,s,D,w.qw,Dq,nullptr), *k=mmb(h,s,D,w.kw,Dkv,nullptr), *v=mmb(h,s,D,w.vw,Dkv,nullptr);
        // Qwen3 per-head QK-RMSNorm: normalize each head's hd-vector with learned weight, before RoPE (view [s,H*hd] as [s*H,hd])
        void*qn=rms_n(q,s*Hq,hd,w.qn), *kn=rms_n(k,s*Hkv,hd,w.kn);
        void*qb=to_bnsd(qn,s,Hq), *kb=to_bnsd(kn,s,Hkv), *vb=to_bnsd(v,s,Hkv);
        void*qr=dalloc(Hq*s*hd*ES), *kr=dalloc(Hkv*s*hd*ES);
        { aclTensor*tq=T({1,Hq,s,hd},qb),*tk=T({1,Hkv,s,hd},kb),*tc=T({s,hd},gCos),*ts=T({s,hd},gSin),
            *tqo=T({1,Hq,s,hd},qr),*tko=T({1,Hkv,s,hd},kr);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,0,tqo,tko,ws,e);},aclnnApplyRotaryPosEmb); }
        void*attn=dalloc(Hq*s*hd*ES);
        { aclTensor*tq=T({1,Hq,s,hd},qr),*tk=T({1,Hkv,s,hd},kr),*tv=T({1,Hkv,s,hd},vb),*to=T({1,Hq,s,hd},attn);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Hq,true,to,ws,e);},aclnnFlashAttentionScore); }
        void*ao=from_bnsd(attn,s,Hq);                 // reshape to [s,Dq]
        void*op=mmb(ao,s,Dq,w.ow,D,nullptr);
        void*x2=add(x,op,s*D);
        void*h2=rms(x2,s,w.g2);
        void*gate=mmb(h2,s,D,w.gatew,F,nullptr);
        void*upp =mmb(h2,s,D,w.upw,F,nullptr);
        void*mlp =mul(silu(gate,s*F),upp,s*F);
        void*down=mmb(mlp,s,F,w.downw,D,nullptr);
        void*xn=add(x2,down,s*D);
        if(g_dbgL0 && l==0 && s==Sprompt){ dumpT("h",h,s*D); dumpT("q",q,s*Dq); dumpT("k",k,s*Dkv); dumpT("v",v,s*Dkv);   // prefill-length dump only (generation re-enters forward with growing s)
            dumpT("qn",qn,s*Dq); dumpT("kn",kn,s*Dkv); dumpT("qr",qr,Hq*s*hd); dumpT("kr",kr,Hkv*s*hd);
            dumpT("attn",ao,s*Dq); dumpT("op",op,s*D); dumpT("x2",x2,s*D); dumpT("h2",h2,s*D);
            dumpT("mlp",mlp,s*F); dumpT("down",down,s*D); dumpT("xn",xn,s*D); }
        x=xn;
    }
    void*fin=rms(x,s,gNorm);
    snap(fin,s);                                      // final norm output (HF hs[L])
    return mmb(fin,s,D,gEmb,V,nullptr);               // tied lm_head: fin @ embed^T -> [s,V]
}

int main(int argc,char**argv){
    DIR = (argc>1)? argv[1] : "data";
    { FILE*f=fopen((DIR+"/config.txt").c_str(),"r"); if(!f){printf("[FATAL] no %s/config.txt\n",DIR.c_str());return 1;}
      if(fscanf(f,"%ld %ld %ld %ld %ld %ld %ld %lf %lf %ld %ld %d",
                &D,&L,&Hq,&Hkv,&hd,&F,&V,&EPS,&THETA,&Sprompt,&NGEN,&tied)!=12){printf("[FATAL] config parse\n");return 1;} fclose(f); }
    Dq=Hq*hd; Dkv=Hkv*hd;
    if(getenv("QWEN_BF16")){ DT=ACL_BF16; ES=2; DTNAME="bf16"; }
    else if(getenv("QWEN_FP16")){ DT=ACL_FLOAT16; ES=2; DTNAME="fp16"; }
    printf("[cfg] D=%ld L=%ld Hq=%ld Hkv=%ld hd=%ld F=%ld V=%ld eps=%g S=%ld n_gen=%ld tied=%d dtype=%s\n",
           D,L,Hq,Hkv,hd,F,V,EPS,Sprompt,NGEN,tied,DTNAME);

    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&S_));

    // Upload weights (permanently resident)
    gEmb=up_f("embed.bin", V*D); gNorm=up_f("gnorm.bin", D);
    gL.resize(L);
    for(int64_t l=0;l<L;l++){ char b[32]; LW&w=gL[l];
        auto N=[&](const char*s){ snprintf(b,sizeof b,"l%ld.%s",l,s); return std::string(b); };
        w.g1=up_f(N("g1"),D); w.g2=up_f(N("g2"),D);
        w.qw=up_f(N("q.w"),Dq*D); w.kw=up_f(N("k.w"),Dkv*D); w.vw=up_f(N("v.w"),Dkv*D);
        w.qn=up_f(N("qn"),hd);   w.kn=up_f(N("kn"),hd);     // per-head QK-norm weights
        w.ow=up_f(N("o.w"),D*Dq);
        w.gatew=up_f(N("gate.w"),F*D); w.upw=up_f(N("up.w"),F*D); w.downw=up_f(N("down.w"),D*F);
    }
    // RoPE cos/sin (positions 0..S_max-1, half-split): use HF-exported real inv_freq[hd/2] to avoid theta parsing pitfalls
    int64_t Smax=Sprompt+NGEN, half=hd/2;
    auto inv=rfile("inv_freq.bin", half);
    std::vector<float> cs(Smax*hd), sn(Smax*hd);
    for(int64_t s=0;s<Smax;s++)for(int64_t d=0;d<hd;d++){
        double a=(double)s*(double)inv[d%half];
        cs[s*hd+d]=(float)std::cos(a); sn[s*hd+d]=(float)std::sin(a); }
    // Note: forward uses T({s,hd},gCos) to take the first s rows -- gCos must contain Smax rows
    { void*p=dalloc(Smax*hd*ES); std::vector<char> b(Smax*hd*ES); enc(cs,b.data()); CHECK(aclrtMemcpy(p,Smax*hd*ES,b.data(),Smax*hd*ES,ACL_MEMCPY_HOST_TO_DEVICE)); gCos=p; }
    { void*p=dalloc(Smax*hd*ES); std::vector<char> b(Smax*hd*ES); enc(sn,b.data()); CHECK(aclrtMemcpy(p,Smax*hd*ES,b.data(),Smax*hd*ES,ACL_MEMCPY_HOST_TO_DEVICE)); gSin=p; }

    auto ids = rfile64("ids.bin", Sprompt);
    // bf16 mode: compare against HF **bf16** reference (fair -- vanilla bf16 also diverges from fp32); fp32/fp16 compares against fp32 reference
    bool useBf16Ref = false;
    if(DT==ACL_BF16){ FILE*f=fopen((DIR+"/gen_bf16.bin").c_str(),"rb"); if(f){ fclose(f); useBf16Ref=true; } }
    auto refgen = rfile64(useBf16Ref?"gen_bf16.bin":"gen.bin", NGEN);
    auto reflog = rfile(useBf16Ref?"logits_bf16.bin":"logits.bin", Sprompt*V);
    if(useBf16Ref) printf("[ref] reference = HF bf16 (fair comparison for shim bf16)\n");

    // ===== (1) Prefill full-position logits cross-check =====
    bool DBG = getenv("QWEN_DUMP_HIDDEN")!=nullptr;
    if(getenv("QWEN_DUMP_L0")) g_dbgL0=true;
    if(DBG){ g_dump=true; g_dumpS=Sprompt; }
    void*dlog=forward(ids);
    if(DBG){ FILE*f=fopen((DIR+"/shim_hidden.bin").c_str(),"wb");
        for(auto&h:g_snaps) fwrite(h.data(),4,h.size(),f); fclose(f);
        printf("[dbg] dumped %zu hidden snapshots ([S,D] each) -> shim_hidden.bin\n", g_snaps.size());
        g_dump=false; }
    std::vector<char> lbuf(Sprompt*V*ES); CHECK(aclrtMemcpy(lbuf.data(),Sprompt*V*ES,dlog,Sprompt*V*ES,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> logits(Sprompt*V); for(int64_t i=0;i<Sprompt*V;i++) logits[i]=dec1(lbuf.data(),i);
    double me=0,mr=0; int top1=0;
    for(int64_t s=0;s<Sprompt;s++){
        int am_shim=0, am_ref=0; float mx_s=-1e30f, mx_r=-1e30f;
        for(int64_t j=0;j<V;j++){ float a=logits[s*V+j], b=reflog[s*V+j];
            me=std::max(me,(double)std::fabs(a-b)); mr=std::max(mr,(double)std::fabs(b));
            if(a>mx_s){mx_s=a;am_shim=(int)j;} if(b>mx_r){mx_r=b;am_ref=(int)j;} }
        if(am_shim==am_ref) top1++;
    }
    double lerr=me/(mr+1e-9);
    printf("[prefill] logits normalized error=%.3e  per-position top-1 hit %d/%ld\n", lerr, top1, Sprompt);

    // ===== (2) Greedy decode token-by-token cross-check (recompute, no KV cache) =====
    std::vector<int64_t> seq=ids; int genok=0;
    printf("[generate] HF ref: ["); for(int64_t i=0;i<NGEN;i++)printf("%ld%s",refgen[i],i+1<NGEN?", ":""); printf("]\n");
    printf("[generate] shim  : [");
    for(int64_t t=0;t<NGEN;t++){
        void*dl=forward(seq);
        int64_t n=seq.size(); std::vector<char> lb(V*ES);
        CHECK(aclrtMemcpy(lb.data(),V*ES,(char*)dl+(n-1)*V*ES,V*ES,ACL_MEMCPY_DEVICE_TO_HOST));
        int am=0; float mx=-1e30f; for(int64_t j=0;j<V;j++){ float lv=dec1(lb.data(),j); if(lv>mx){mx=lv;am=(int)j;} }
        printf("%d%s",am,t+1<NGEN?", ":""); fflush(stdout);
        if(am==(int)refgen[t]) genok++;
        seq.push_back(am);
    }
    printf("]\n");
    printf("[generate] token-by-token hit %d/%ld\n", genok, NGEN);

    // PASS criteria by dtype:
    //   fp32  -- strict (logits ~1e-5, top-1 all correct, greedy decode exact token match);
    //   fp16  -- 10-bit mantissa: top-1 all correct + majority of generated tokens match;
    //   bf16  -- 7-bit mantissa: pure-bf16 layers accumulate rounding and coarsen logit ranking, so greedy decode may diverge;
    //            vs HF bf16 reference (same precision): error should fall back to bf16 rounding level, top-1 all correct + majority of generated tokens match.
    bool ok = (DT==ACL_FLOAT)   ? (lerr<=2e-3 && top1==Sprompt && genok==NGEN)
            : (DT==ACL_FLOAT16) ? (top1==Sprompt && genok>=NGEN*3/4)
            : useBf16Ref        ? (top1==Sprompt && genok>=NGEN*3/4)
                                : (top1==Sprompt);
    printf("== Qwen3-0.6B end-to-end (%s%s) %s == (logits err=%.2e, top1=%d/%ld, gen=%d/%ld)\n",
           DTNAME, useBf16Ref?" vs HF-bf16":"", ok?"PASS":"FAIL", lerr, top1, Sprompt, genok, NGEN);

    for(void*p:g_frees) aclrtFree(p);
    CHECK(aclrtDestroyStream(S_)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    return ok?0:1;
}
