// Real-model end-to-end graph test: 2-layer GQA (Nq>Nkv) transformer with RoPE and KV-cache decode path.
// The golden invariant for KV-cache correctness: the last-token logits from "prefill S0 -> decode 1" with cache
// must equal the last-row logits from "full prefill S0+1" (same weights, same tokens, causal attention -> bit-identical).
// Chains Embedding/RMSNorm/Matmul/Permute/Slice/RoPE/FlashAttention(GQA)/SwiGlu/ScatterUpdate (cache write).
// All fp32; pure ACL client (no CUDA headers), simulating an Ascend application linked against libascendcl.so.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

static aclrtStream g_stream;
#define CHECK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

// ---- Dimensions (GQA: Nq query heads sharing Nkv kv heads) ----
static const int64_t nL=2, V=32, Nq=4, Nkv=2, hd=8;
static const int64_t Dm=Nq*hd, Dkv=Nkv*hd, F=48, S0=5, St=S0+1;
static const double EPS=1e-6; static const double SCALE=0.353553390593;  // 1/sqrt(hd)=1/sqrt(8)

static std::vector<void*> g_frees;
static void* dalloc(size_t b){ void*p; CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); g_frees.push_back(p); return p; }
static void* dup(const std::vector<float>&v){ void*p=dalloc(v.size()*4); CHECK(aclrtMemcpy(p,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return p; }
static aclTensor* T(std::vector<int64_t> d, void*p, aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); }
template<typename G> static void run2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
}
// ---- Thin op wrappers (return device output pointer) ----
static void* mm(void*dA,int64_t M,int64_t K,void*dB,int64_t N){ void*dC=dalloc(M*N*4);
    aclTensor*a=T({M,K},dA),*b=T({K,N},dB),*c=T({M,N},dC);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(a,b,c,1,w,e);},aclnnMatmul); return dC; }
static void* rms(void*dX,int64_t rows,void*dG){ void*dY=dalloc(rows*Dm*4);
    aclTensor*x=T({rows,Dm},dX),*g=T({Dm},dG),*y=T({rows,Dm},dY);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,g,EPS,y,w,e);},aclnnRmsNorm); return dY; }
static void* add(void*dA,void*dB,int64_t n){ void*dC=dalloc(n*4);
    aclTensor*a=T({n},dA),*b=T({n},dB),*c=T({n},dC);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(a,b,nullptr,c,w,e);},aclnnAdd); return dC; }
// [S, H*hd] -> (viewed as [S,H,hd] permute{1,0,2}) -> [H,S,hd] (BNSD layout: N,S,hd)
static void* to_bnsd(void*src,int64_t S,int64_t H){ void*o=dalloc(H*S*hd*4);
    aclTensor*a=T({S,H,hd},src),*b=T({H,S,hd},o); int64_t pd[3]={1,0,2}; aclIntArray*p=aclCreateIntArray(pd,3);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,p,b,w,e);},aclnnPermute); aclDestroyIntArray(p); return o; }
// [H,Sq,hd] -> (permute{1,0,2}) -> [Sq,H,hd]=[Sq,H*hd]
static void* from_bnsd(void*src,int64_t Sq,int64_t H){ void*o=dalloc(Sq*H*hd*4);
    aclTensor*a=T({H,Sq,hd},src),*b=T({Sq,H,hd},o); int64_t pd[3]={1,0,2}; aclIntArray*p=aclCreateIntArray(pd,3);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,p,b,w,e);},aclnnPermute); aclDestroyIntArray(p); return o; }

// Per-layer weights
struct LW { void *g1,*Wq,*Wk,*Wv,*Wo,*g2,*Wg,*Wd; };
static LW gL[nL]; static void *gWemb,*gWl,*gG3,*gCos,*gSin;

// One forward step: processes S tokens (hidden dX[S,Dm]) starting at position pos0.
// Writes KV for each layer into cache[writeStart:writeStart+S]; attention spans cache[0:writeStart+S].
// Kc/Vc: per-layer cache ([Nkv,St,hd]) device pointer arrays.
// Returns [S,V] logits after the final RMSNorm + logits projection.
static void* step(void*dX,int64_t S,int64_t pos0,int64_t writeStart,void**Kc,void**Vc){
    int64_t Skv=writeStart+S;
    void*x=dX;
    for(int64_t L=0;L<nL;L++){
        void*h=rms(x,S,gL[L].g1);
        void*q=mm(h,S,Dm,gL[L].Wq,Dm), *k=mm(h,S,Dm,gL[L].Wk,Dkv), *v=mm(h,S,Dm,gL[L].Wv,Dkv);
        void*qb=to_bnsd(q,S,Nq), *kb=to_bnsd(k,S,Nkv), *vb=to_bnsd(v,S,Nkv);   // [Nq,S,hd] / [Nkv,S,hd]
        // RoPE: take rows [pos0:pos0+S] of cos/sin ([S,hd])
        void*qr=dalloc(Nq*S*hd*4), *kr=dalloc(Nkv*S*hd*4);
        void*cosp=(char*)gCos+pos0*hd*4, *sinp=(char*)gSin+pos0*hd*4;
        { aclTensor*tq=T({1,Nq,S,hd},qb),*tk=T({1,Nkv,S,hd},kb),*tc=T({S,hd},cosp),*ts=T({S,hd},sinp),
            *tqo=T({1,Nq,S,hd},qr),*tko=T({1,Nkv,S,hd},kr);
          run2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,0,tqo,tko,w,e);},aclnnApplyRotaryPosEmb); }
        // Write KV cache: place this step's K/V (post-RoPE) into cache[writeStart:writeStart+S]
        if(S>1){ // prefill: copy S rows per head (DEVICE_TO_DEVICE)
            for(int64_t hd_h=0;hd_h<Nkv;hd_h++){
                CHECK(aclrtMemcpy((char*)Kc[L]+(hd_h*St+writeStart)*hd*4,S*hd*4,(char*)kr+hd_h*S*hd*4,S*hd*4,ACL_MEMCPY_DEVICE_TO_DEVICE));
                CHECK(aclrtMemcpy((char*)Vc[L]+(hd_h*St+writeStart)*hd*4,S*hd*4,(char*)vb+hd_h*S*hd*4,S*hd*4,ACL_MEMCPY_DEVICE_TO_DEVICE));
            }
        } else { // decode: single token written via ScatterUpdate (cache viewed as [Nkv*St,hd], index={h*St+writeStart})
            std::vector<int64_t> idx(Nkv); for(int64_t hh=0;hh<Nkv;hh++) idx[hh]=hh*St+writeStart;
            void*didx=dalloc(Nkv*8); CHECK(aclrtMemcpy(didx,Nkv*8,idx.data(),Nkv*8,ACL_MEMCPY_HOST_TO_DEVICE));
            { aclTensor*self=T({Nkv*St,hd},Kc[L]),*ti=T({Nkv},didx,ACL_INT64),*src=T({Nkv,hd},kr),*o=T({Nkv*St,hd},Kc[L]);
              run2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterUpdateGetWorkspaceSize(self,ti,src,o,w,e);},aclnnScatterUpdate); }
            { aclTensor*self=T({Nkv*St,hd},Vc[L]),*ti=T({Nkv},didx,ACL_INT64),*src=T({Nkv,hd},vb),*o=T({Nkv*St,hd},Vc[L]);
              run2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterUpdateGetWorkspaceSize(self,ti,src,o,w,e);},aclnnScatterUpdate); }
        }
        // Attention K/V = cache[0:Skv]. Slice along dim1 to get contiguous [Nkv,Skv,hd]
        void*kall,*vall;
        if(Skv==St){ kall=Kc[L]; vall=Vc[L]; }
        else {
            kall=dalloc(Nkv*Skv*hd*4); vall=dalloc(Nkv*Skv*hd*4);
            { aclTensor*a=T({Nkv,St,hd},Kc[L]),*b=T({Nkv,Skv,hd},kall);
              run2([&](uint64_t*w,aclOpExecutor**e){return aclnnSliceGetWorkspaceSize(a,1,0,Skv,1,b,w,e);},aclnnSlice); }
            { aclTensor*a=T({Nkv,St,hd},Vc[L]),*b=T({Nkv,Skv,hd},vall);
              run2([&](uint64_t*w,aclOpExecutor**e){return aclnnSliceGetWorkspaceSize(a,1,0,Skv,1,b,w,e);},aclnnSlice); }
        }
        void*attn=dalloc(Nq*S*hd*4);
        { aclTensor*tq=T({1,Nq,S,hd},qr),*tk=T({1,Nkv,Skv,hd},kall),*tv=T({1,Nkv,Skv,hd},vall),*to=T({1,Nq,S,hd},attn);
          run2([&](uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,SCALE,Nq,true,to,w,e);},aclnnFlashAttentionScore); }
        void*ao=from_bnsd(attn,S,Nq);                 // [S,Dm] after permute back
        void*op=mm(ao,S,Dm,gL[L].Wo,Dm);
        void*x2=add(x,op,S*Dm);
        void*h2=rms(x2,S,gL[L].g2);
        void*gate=mm(h2,S,Dm,gL[L].Wg,2*F);
        void*mlp=dalloc(S*F*4);
        { aclTensor*g=T({S,2*F},gate),*o=T({S,F},mlp);
          run2([&](uint64_t*w,aclOpExecutor**e){return aclnnSwiGluGetWorkspaceSize(g,o,w,e);},aclnnSwiGlu); }
        void*down=mm(mlp,S,F,gL[L].Wd,Dm);
        x=add(x2,down,S*Dm);
    }
    void*fin=rms(x,S,gG3);
    return mm(fin,S,Dm,gWl,V);                          // [S,V] logits
}

// Embedding: ids[S] -> [S,Dm]
static void* embed(const std::vector<int64_t>&ids){ int64_t S=ids.size();
    void*dids=dalloc(S*8); CHECK(aclrtMemcpy(dids,S*8,ids.data(),S*8,ACL_MEMCPY_HOST_TO_DEVICE));
    void*o=dalloc(S*Dm*4);
    aclTensor*w=T({V,Dm},gWemb),*id=T({S},dids,ACL_INT64),*ot=T({S,Dm},o);
    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(w,id,ot,ws,e);},aclnnEmbedding); return o; }

int main(){
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(123);
    auto rv=[&](int64_t n,double sc){ std::vector<float> v(n); for(auto&x:v)x=(float)((rand()/(double)RAND_MAX*2-1)*sc); return v; };
    auto ones=[&](int64_t n){ std::vector<float> v(n,0); for(auto&x:v)x=1.0f+(float)((rand()/(double)RAND_MAX*2-1)*0.1); return v; };

    gWemb=dup(rv(V*Dm,0.5)); gWl=dup(rv(Dm*V,0.3)); gG3=dup(ones(Dm));
    for(int64_t L=0;L<nL;L++){
        gL[L].g1=dup(ones(Dm)); gL[L].g2=dup(ones(Dm));
        gL[L].Wq=dup(rv(Dm*Dm,0.3)); gL[L].Wk=dup(rv(Dm*Dkv,0.3)); gL[L].Wv=dup(rv(Dm*Dkv,0.3)); gL[L].Wo=dup(rv(Dm*Dm,0.3));
        gL[L].Wg=dup(rv(Dm*2*F,0.3)); gL[L].Wd=dup(rv(F*Dm,0.3));
    }
    std::vector<float> cs(St*hd),sn(St*hd);
    for(int64_t s=0;s<St;s++)for(int64_t d=0;d<hd;d++){ double th=s*std::pow(10000.0,-2.0*(d%(hd/2))/hd); cs[s*hd+d]=std::cos(th); sn[s*hd+d]=std::sin(th); }
    gCos=dup(cs); gSin=dup(sn);

    std::vector<int64_t> ids(St); for(auto&x:ids)x=rand()%V;

    // ---- Reference path: full prefill of St tokens ----
    void *KcR[nL],*VcR[nL]; for(int64_t L=0;L<nL;L++){ KcR[L]=dalloc(Nkv*St*hd*4); VcR[L]=dalloc(Nkv*St*hd*4); }
    void*xR=embed(ids);
    void*logR=step(xR,St,0,0,KcR,VcR);                  // [St,V] full-prefill logits
    std::vector<float> lR(St*V); CHECK(aclrtMemcpy(lR.data(),St*V*4,logR,St*V*4,ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- Cache path: prefill S0 tokens, then decode token S0 ----
    void *Kc[nL],*Vc[nL]; for(int64_t L=0;L<nL;L++){ Kc[L]=dalloc(Nkv*St*hd*4); Vc[L]=dalloc(Nkv*St*hd*4); }
    std::vector<int64_t> ids0(ids.begin(),ids.begin()+S0);
    void*x0=embed(ids0);
    step(x0,S0,0,0,Kc,Vc);                              // prefill: fill cache[0:S0], logits discarded
    std::vector<int64_t> idD(1,ids[S0]);
    void*xD=embed(idD);
    void*logD=step(xD,1,S0,S0,Kc,Vc);                   // decode: write cache[S0], logits[1,V]
    std::vector<float> lD(V); CHECK(aclrtMemcpy(lD.data(),V*4,logD,V*4,ACL_MEMCPY_DEVICE_TO_HOST));

    // ---- Cross-check: decode logits == last row of reference ----
    double me=0,mr=0; for(int64_t j=0;j<V;j++){ double r=lR[(St-1)*V+j]; me=std::max(me,std::fabs(lD[j]-r)); mr=std::max(mr,std::fabs(r)); }
    double err=me/(mr+1e-9); bool ok=err<=1e-4;
    printf("GQA 2-layer KV-cache decode (Nq=%lld,Nkv=%lld,hd=%lld) decode vs full-prefill last-row normalized error=%.3e  %s\n",
           (long long)Nq,(long long)Nkv,(long long)hd, err, ok?"PASS":"FAIL");
    printf("== %d PASS, %d FAIL ==\n", ok?1:0, ok?0:1);

    for(void*p:g_frees) aclrtFree(p);
    CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    return ok?0:1;
}
