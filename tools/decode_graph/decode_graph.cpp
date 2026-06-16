// Decode-loop full-loop CUDA Graph capture + replay + on-machine per-token latency measurement.
// Captures a single decode step as a graph, then drives the full decode loop by
// "replaying the same graph + updating a small number of device buffers between steps".
// Decode is launch-bound (tens of small kernels per token; launch overhead dominates on GB10,
// not masked by UMA bandwidth) -- CUDA Graph submits the entire kernel chain in one call,
// eliminating per-kernel launch overhead. This setup can measure the true benefit on local hardware
// (unlike bandwidth-bound workloads masked by UMA).
//
// Design notes (enabling a single static graph to run variable-length decode):
//   - Fixed shape: S=1 single token; attention always covers the full pre-allocated KV cache
//     [Nkv,Smax,hd], using a [1,Smax] mask to block slots beyond the current position.
//   - Fixed pointers: all activations/workspace pre-allocated and reused (no cudaMalloc during capture);
//     weights/cache are resident.
//   - Between steps, only device buffer contents are updated (same pointers):
//     token id, RoPE cos/sin (one row for the current position), ScatterUpdate index for KV write, mask.
//     Replay reads the buffer contents at exec time -> same graph naturally handles different positions.
// Verification: token sequence produced by graph replay == sequence from direct step() execution (token-exact).
// Pure ACL client (links only libascendcl.so).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

static aclrtStream S_;
#define CK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

// ---- dimensions (GQA) ----
static const int64_t nL=2, V=32, Nq=4, Nkv=2, hd=8;
static const int64_t Dm=Nq*hd, Dkv=Nkv*hd, F=48, Smax=64;
static const double EPS=1e-6; static const double SCALE=0.353553390593;  // 1/sqrt(8)

// ---- device helpers ----
static std::vector<void*> g_frees;
static void* dalloc(size_t b){ void*p; CK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); g_frees.push_back(p); return p; }
static void* dup(const std::vector<float>&v){ void*p=dalloc(v.size()*4); CK(aclrtMemcpy(p,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return p; }
static aclTensor* T(std::vector<int64_t> d, void*p, aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); }

// Capture-safe run: uses pre-allocated global workspace; never calls cudaMalloc
static void* gWS=nullptr; static uint64_t gWScap=0;
template<typename G> static void run2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(getws(&ws,&ex));
    if(ws>gWScap){ printf("[FATAL] ws %llu > cap %llu\n",(unsigned long long)ws,(unsigned long long)gWScap); exit(1); }
    CK(run(ws?gWS:nullptr,ws,ex,S_));
}

// ---- weights + resident cache/buffers ----
struct LW { void *g1,*Wq,*Wk,*Wv,*Wo,*g2,*Wg,*Wd; };
static LW gL[nL]; static void *gWemb,*gWl,*gG3;
static void *gCosTab,*gSinTab;       // [Smax,hd] full table (computed on host)
static void *Kc[nL],*Vc[nL];         // [Nkv,Smax,hd] cache
// Device buffers updated between steps (graph replay reads their contents at exec time)
static void *dId;                    // [1] int64 current token
static void *dCosCur,*dSinCur;       // [1,hd] cos/sin for current position
static void *dScatterIdx;            // [Nkv] int64 cache write index for current position
static void *dMask;                  // [1,Smax] uint8 attention mask (1=blocked for positions >pos)
// Pre-allocated activations (per-layer, independent, to avoid aliasing during capture)
struct Act { void *h,*q,*k,*v,*qb,*kb,*vb,*qr,*kr,*attn,*ao,*op,*x2,*h2,*gate,*mlp,*down,*xres; };
static Act A[nL]; static void *gX0,*gFin,*gLogits;

static void* mm(void*dA,int64_t M,int64_t K,void*dB,int64_t N,void*out){
    aclTensor*a=T({M,K},dA),*b=T({K,N},dB),*c=T({M,N},out);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(a,b,c,1,w,e);},aclnnMatmul); return out; }
static void* rms(void*dX,int64_t rows,void*dG,void*out){
    aclTensor*x=T({rows,Dm},dX),*g=T({Dm},dG),*y=T({rows,Dm},out);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,g,EPS,y,w,e);},aclnnRmsNorm); return out; }
static void* add(void*dA,void*dB,int64_t n,void*out){
    aclTensor*a=T({n},dA),*b=T({n},dB),*c=T({n},out);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(a,b,nullptr,c,w,e);},aclnnAdd); return out; }
static void* permute(void*src,int64_t H,void*out){ // [1,H,hd]->[H,1,hd]
    aclTensor*a=T({1,H,hd},src),*b=T({H,1,hd},out); int64_t pd[3]={1,0,2}; aclIntArray*p=aclCreateIntArray(pd,3);
    run2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(a,p,b,w,e);},aclnnPermute); aclDestroyIntArray(p); return out; }

// Single-token decode step: reads dId/dCosCur/dSinCur/dScatterIdx/dMask, writes gLogits. All fixed pointers/shapes.
static void step_once(){
    // embed: dId[1] → gX0[1,Dm]
    { aclTensor*w=T({V,Dm},gWemb),*id=T({1},dId,ACL_INT64),*o=T({1,Dm},gX0);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(w,id,o,ws,e);},aclnnEmbedding); }
    void*x=gX0;
    for(int64_t L=0;L<nL;L++){ Act&a=A[L]; LW&w=gL[L];
        rms(x,1,w.g1,a.h);
        mm(a.h,1,Dm,w.Wq,Dm,a.q); mm(a.h,1,Dm,w.Wk,Dkv,a.k); mm(a.h,1,Dm,w.Wv,Dkv,a.v);
        permute(a.q,Nq,a.qb); permute(a.k,Nkv,a.kb); permute(a.v,Nkv,a.vb);
        // RoPE (one row of cos/sin for the current position)
        { aclTensor*tq=T({1,Nq,1,hd},a.qb),*tk=T({1,Nkv,1,hd},a.kb),*tc=T({1,hd},dCosCur),*ts=T({1,hd},dSinCur),
            *tqo=T({1,Nq,1,hd},a.qr),*tko=T({1,Nkv,1,hd},a.kr);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,0,tqo,tko,ws,e);},aclnnApplyRotaryPosEmb); }
        // write KV cache (ScatterUpdate: cache viewed as [Nkv*Smax,hd], index=dScatterIdx)
        { aclTensor*self=T({Nkv*Smax,hd},Kc[L]),*ti=T({Nkv},dScatterIdx,ACL_INT64),*src=T({Nkv,hd},a.kr),*o=T({Nkv*Smax,hd},Kc[L]);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnScatterUpdateGetWorkspaceSize(self,ti,src,o,ws,e);},aclnnScatterUpdate); }
        { aclTensor*self=T({Nkv*Smax,hd},Vc[L]),*ti=T({Nkv},dScatterIdx,ACL_INT64),*src=T({Nkv,hd},a.vb),*o=T({Nkv*Smax,hd},Vc[L]);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnScatterUpdateGetWorkspaceSize(self,ti,src,o,ws,e);},aclnnScatterUpdate); }
        // attention: q[1,Nq,1,hd] against full cache[1,Nkv,Smax,hd], mask[1,Smax] blocks positions >pos
        { aclTensor*tq=T({1,Nq,1,hd},a.qr),*tk=T({1,Nkv,Smax,hd},Kc[L]),*tv=T({1,Nkv,Smax,hd},Vc[L]),
            *tm=T({1,Smax},dMask,ACL_UINT8),*to=T({1,Nq,1,hd},a.attn);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,tm,SCALE,Nq,false,to,ws,e);},aclnnFlashAttentionScore); }
        // attn[1,Nq,1,hd] memory is [Nq*hd]=[Dm] (head-major); for S=1 directly treated as [1,Dm] for o_proj
        mm(a.attn,1,Dm,w.Wo,Dm,a.op);
        add(x,a.op,Dm,a.x2);
        rms(a.x2,1,w.g2,a.h2);
        mm(a.h2,1,Dm,w.Wg,2*F,a.gate);
        { aclTensor*g=T({1,2*F},a.gate),*o=T({1,F},a.mlp);
          run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnSwiGluGetWorkspaceSize(g,o,ws,e);},aclnnSwiGlu); }
        mm(a.mlp,1,F,w.Wd,Dm,a.down);
        add(a.x2,a.down,Dm,a.xres);
        x=a.xres;
    }
    rms(x,1,gG3,gFin);
    mm(gFin,1,Dm,gWl,V,gLogits);
}

// Between-step update: write current token id, cos/sin for position pos, cache write index, and mask into fixed device buffers
static std::vector<float> g_cos,g_sin;   // full table on host
static void set_step(int64_t tok,int64_t pos){
    int64_t id=tok; CK(aclrtMemcpy(dId,8,&id,8,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dCosCur,hd*4,&g_cos[pos*hd],hd*4,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dSinCur,hd*4,&g_sin[pos*hd],hd*4,ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<int64_t> idx(Nkv); for(int64_t h=0;h<Nkv;h++) idx[h]=h*Smax+pos;
    CK(aclrtMemcpy(dScatterIdx,Nkv*8,idx.data(),Nkv*8,ACL_MEMCPY_HOST_TO_DEVICE));
    std::vector<uint8_t> mask(Smax,1); for(int64_t j=0;j<=pos;j++) mask[j]=0;   // 0=visible, 1=blocked
    CK(aclrtMemcpy(dMask,Smax,mask.data(),Smax,ACL_MEMCPY_HOST_TO_DEVICE));
}
static int64_t argmax_logits(){
    std::vector<float> lg(V); CK(aclrtMemcpy(lg.data(),V*4,gLogits,V*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int am=0; float mx=lg[0]; for(int j=1;j<V;j++) if(lg[j]>mx){mx=lg[j];am=j;} return am;
}

int main(int argc,char**argv){
    int64_t NGEN = argc>1? atoll(argv[1]) : 32;
    if(NGEN>Smax-1) NGEN=Smax-1;
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&S_)); srand(123);
    auto rv=[&](int64_t n,double sc){ std::vector<float> v(n); for(auto&x:v)x=(float)((rand()/(double)RAND_MAX*2-1)*sc); return v; };
    auto ones=[&](int64_t n){ std::vector<float> v(n,0); for(auto&x:v)x=1.0f+(float)((rand()/(double)RAND_MAX*2-1)*0.1); return v; };

    gWemb=dup(rv(V*Dm,0.5)); gWl=dup(rv(Dm*V,0.3)); gG3=dup(ones(Dm));
    for(int64_t L=0;L<nL;L++){ gL[L].g1=dup(ones(Dm)); gL[L].g2=dup(ones(Dm));
        gL[L].Wq=dup(rv(Dm*Dm,0.3)); gL[L].Wk=dup(rv(Dm*Dkv,0.3)); gL[L].Wv=dup(rv(Dm*Dkv,0.3)); gL[L].Wo=dup(rv(Dm*Dm,0.3));
        gL[L].Wg=dup(rv(Dm*2*F,0.3)); gL[L].Wd=dup(rv(F*Dm,0.3)); }
    // RoPE full table
    g_cos.resize(Smax*hd); g_sin.resize(Smax*hd);
    for(int64_t s=0;s<Smax;s++)for(int64_t d=0;d<hd;d++){ double th=s*std::pow(10000.0,-2.0*(d%(hd/2))/hd); g_cos[s*hd+d]=std::cos(th); g_sin[s*hd+d]=std::sin(th); }

    // resident cache + between-step buffers + activations, all pre-allocated (before capture)
    for(int64_t L=0;L<nL;L++){ Kc[L]=dalloc(Nkv*Smax*hd*4); Vc[L]=dalloc(Nkv*Smax*hd*4); }
    dId=dalloc(8); dCosCur=dalloc(hd*4); dSinCur=dalloc(hd*4); dScatterIdx=dalloc(Nkv*8); dMask=dalloc(Smax);
    gX0=dalloc(Dm*4); gFin=dalloc(Dm*4); gLogits=dalloc(V*4);
    for(int64_t L=0;L<nL;L++){ Act&a=A[L];
        a.h=dalloc(Dm*4); a.q=dalloc(Dm*4); a.k=dalloc(Dkv*4); a.v=dalloc(Dkv*4);
        a.qb=dalloc(Dm*4); a.kb=dalloc(Dkv*4); a.vb=dalloc(Dkv*4); a.qr=dalloc(Dm*4); a.kr=dalloc(Dkv*4);
        a.attn=dalloc(Dm*4); a.ao=dalloc(Dm*4); a.op=dalloc(Dm*4); a.x2=dalloc(Dm*4); a.h2=dalloc(Dm*4);
        a.gate=dalloc(2*F*4); a.mlp=dalloc(F*4); a.down=dalloc(Dm*4); a.xres=dalloc(Dm*4); }
    gWScap=16u<<20; gWS=dalloc(gWScap);

    auto zero_cache=[&](){ for(int64_t L=0;L<nL;L++){ CK(aclrtMemset(Kc[L],Nkv*Smax*hd*4,0,Nkv*Smax*hd*4)); CK(aclrtMemset(Vc[L],Nkv*Smax*hd*4,0,Nkv*Smax*hd*4)); } };

    const int64_t START=7;   // starting token

    // ===== Path A: no graph, direct step_once (reference sequence + timing) =====
    zero_cache();
    std::vector<int64_t> seqDirect; int64_t tok=START;
    for(int64_t p=0;p<NGEN;p++){ set_step(tok,p); step_once(); CK(aclrtSynchronizeStream(S_)); tok=argmax_logits(); seqDirect.push_back(tok); }

    // ===== Path B: capture single step -> replay for full decode loop =====
    zero_cache();
    set_step(START,0); step_once(); CK(aclrtSynchronizeStream(S_));   // warmup (build cublas handle, JIT), position 0
    zero_cache();
    aclmdlRI model=nullptr;
    CK(aclmdlRICaptureBegin(S_, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    step_once();
    aclError ce=aclmdlRICaptureEnd(S_, &model);
    if(ce!=ACL_SUCCESS || !model){ printf("[FATAL] capture failed ret=%d\n",(int)ce); return 1; }
    std::vector<int64_t> seqGraph; tok=START;
    for(int64_t p=0;p<NGEN;p++){ set_step(tok,p); CK(aclmdlRIExecuteAsync(model,S_)); CK(aclrtSynchronizeStream(S_)); tok=argmax_logits(); seqGraph.push_back(tok); }

    // ===== verification: both sequences are token-exact ====
    int match=0; for(int64_t i=0;i<NGEN;i++) if(seqDirect[i]==seqGraph[i]) match++;
    bool ok = (match==NGEN);
    printf("decode-loop graph replay vs direct execution: per-token match %d/%lld  %s\n", match,(long long)NGEN, ok?"PASS":"FAIL");

    // ===== timing: per-token latency (graph replay vs no graph) =====
    const int WARM=20, ITERS=200;
    // no graph: direct step_once (between-step update fixed, position fixed at pos=NGEN-1 to preserve shape)
    set_step(seqDirect.back(), NGEN-1);
    for(int i=0;i<WARM;i++) step_once(); CK(aclrtSynchronizeStream(S_));
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<ITERS;i++) step_once();
    CK(aclrtSynchronizeStream(S_));
    double ms_direct=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/ITERS;
    // graph replay
    for(int i=0;i<WARM;i++) CK(aclmdlRIExecuteAsync(model,S_)); CK(aclrtSynchronizeStream(S_));
    auto t1=std::chrono::steady_clock::now();
    for(int i=0;i<ITERS;i++) CK(aclmdlRIExecuteAsync(model,S_));
    CK(aclrtSynchronizeStream(S_));
    double ms_graph=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t1).count()/ITERS;

    int nkern_layer = 1/*emb*/ ; (void)nkern_layer;
    printf("[perf] per-token latency: no graph = %.4f ms, graph replay = %.4f ms, speedup %.2fx (saved launch overhead)\n",
           ms_direct, ms_graph, ms_direct/ms_graph);
    printf("== decode-loop CUDA Graph %s == (match=%d/%lld, direct=%.4fms, graph=%.4fms, %.2fx)\n",
           ok?"PASS":"FAIL", match,(long long)NGEN, ms_direct, ms_graph, ms_direct/ms_graph);

    aclmdlRIDestroy(model);
    for(void*p:g_frees) aclrtFree(p);
    CK(aclrtDestroyStream(S_)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    return ok?0:1;
}
