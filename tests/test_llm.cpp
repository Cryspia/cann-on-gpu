// End-to-end tiny LLM forward integration test: chains LLM operators into a single-head transformer block + logits,
// verifying operator composition is correct (all ops green individually does not guarantee pipeline correctness).
// All fp32, single head (avoids Nh<->S transpose), CPU double reference for whole-pipeline cross-check.
// Pipeline: Embedding -> RMSNorm -> Q/K/V projection -> RoPE(q,k) -> causal attention -> O projection + residual
//        -> RMSNorm -> SwiGlu MLP(gate->down) + residual -> RMSNorm -> logits.
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

static aclrtStream g_stream;
#define CHECK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

// ---- dimensions ----
static const int64_t V=32, S=4, D=16, F=24;   // vocab/seq/model-dim/FFN-hidden; single head hd=D
static const double EPS=1e-6, SCALE=1.0/4.0;  // 1/sqrt(D)=1/4

// ---- device helpers ----
static std::vector<void*> g_frees;
static void* dalloc(size_t b){ void*p; CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); g_frees.push_back(p); return p; }
static void* dup(const std::vector<float>&v){ void*p=dalloc(v.size()*4); CHECK(aclrtMemcpy(p,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return p; }
static aclTensor* T(std::vector<int64_t> d, void*p, aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); }
template<typename G> static void exec2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
}
// Thin shim op wrappers (return device output pointer)
static void* mm(void*dA,int64_t M,int64_t K,void*dB,int64_t Ncol){ void*dC=dalloc(M*Ncol*4);
    aclTensor*a=T({M,K},dA),*b=T({K,Ncol},dB),*c=T({M,Ncol},dC);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(a,b,c,1,w,e);},aclnnMatmul); return dC; }
static void* rms(void*dX,int64_t rows,void*dG){ void*dY=dalloc(rows*D*4);
    aclTensor*x=T({rows,D},dX),*g=T({D},dG),*y=T({rows,D},dY);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,g,EPS,y,w,e);},aclnnRmsNorm); return dY; }
static void* add(void*dA,void*dB,int64_t n){ void*dC=dalloc(n*4);
    aclTensor*a=T({n},dA),*b=T({n},dB),*c=T({n},dC);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(a,b,nullptr,c,w,e);},aclnnAdd); return dC; }

// ---- CPU double reference ----
typedef std::vector<double> V_;
static V_ mmf(const V_&A,int64_t M,int64_t K,const V_&B,int64_t N){ V_ C(M*N,0);
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){double s=0;for(int64_t k=0;k<K;k++)s+=A[i*K+k]*B[k*N+j];C[i*N+j]=s;}return C; }
static V_ rmsf(const V_&X,int64_t rows,const V_&G){ V_ Y(rows*D);
    for(int64_t r=0;r<rows;r++){double ms=0;for(int64_t d=0;d<D;d++)ms+=X[r*D+d]*X[r*D+d];ms=1.0/std::sqrt(ms/D+EPS);
        for(int64_t d=0;d<D;d++)Y[r*D+d]=X[r*D+d]*ms*G[d];}return Y; }

int main(){
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(99);
    auto rv=[&](int64_t n,double sc){ std::vector<float> v(n); for(auto&x:v)x=(float)((rand()/(double)RAND_MAX*2-1)*sc); return v; };

    // Weights (small scale, avoid exp/softmax overflow)
    auto Wemb=rv(V*D,0.5), g1=rv(D,0), g2=rv(D,0), g3=rv(D,0);
    for(auto&x:g1)x+=1; for(auto&x:g2)x+=1; for(auto&x:g3)x+=1;     // gamma ~1
    auto Wq=rv(D*D,0.3),Wk=rv(D*D,0.3),Wv=rv(D*D,0.3),Wo=rv(D*D,0.3);
    auto Wg=rv(D*2*F,0.3),Wd=rv(F*D,0.3),Wl=rv(D*V,0.3);
    std::vector<int64_t> ids(S); for(auto&x:ids)x=rand()%V;
    std::vector<float> cs(S*D),sn(S*D); for(int64_t i=0;i<S*D;i++){cs[i]=std::cos(0.01*i);sn[i]=std::sin(0.013*i);}

    // ===== shim forward (weights/inputs uploaded once; forward wrapped as lambda for timing) =====
    void*dWemb=dup(Wemb),*dids=dalloc(S*8); { std::vector<int64_t> tmp=ids; CHECK(aclrtMemcpy(dids,S*8,tmp.data(),S*8,ACL_MEMCPY_HOST_TO_DEVICE)); }
    void*dg1=dup(g1),*dg2=dup(g2),*dg3=dup(g3),*dcs=dup(cs),*dsn=dup(sn);
    void*dWq=dup(Wq),*dWk=dup(Wk),*dWv=dup(Wv),*dWo=dup(Wo),*dWg=dup(Wg),*dWd=dup(Wd),*dWl=dup(Wl);
    auto fwd=[&]()->void*{
        void*dx=dalloc(S*D*4);
        { aclTensor*w=T({V,D},dWemb),*id=T({S},dids,ACL_INT64),*o=T({S,D},dx);
          exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(w,id,o,ws,e);},aclnnEmbedding); }
        void*dh=rms(dx,S,dg1);
        void*dq=mm(dh,S,D,dWq,D), *dk=mm(dh,S,D,dWk,D), *dv=mm(dh,S,D,dWv,D);
        void*dqo=dalloc(S*D*4),*dko=dalloc(S*D*4);
        { aclTensor*q=T({1,1,S,D},dq),*k=T({1,1,S,D},dk),*c=T({S,D},dcs),*s=T({S,D},dsn),*qo=T({1,1,S,D},dqo),*ko=T({1,1,S,D},dko);
          exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnApplyRotaryPosEmbGetWorkspaceSize(q,k,c,s,0,qo,ko,ws,e);},aclnnApplyRotaryPosEmb); }
        void*dattn=dalloc(S*D*4);
        { aclTensor*q=T({1,1,S,D},dqo),*k=T({1,1,S,D},dko),*v=T({1,1,S,D},dv),*o=T({1,1,S,D},dattn);
          exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(q,k,v,nullptr,SCALE,1,true,o,ws,e);},aclnnFlashAttentionScore); }
        void*dop=mm(dattn,S,D,dWo,D);
        void*dx2=add(dx,dop,S*D);
        void*dh2=rms(dx2,S,dg2);
        void*dgate=mm(dh2,S,D,dWg,2*F);
        void*dmlp=dalloc(S*F*4);
        { aclTensor*g=T({S,2*F},dgate),*o=T({S,F},dmlp);
          exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnSwiGluGetWorkspaceSize(g,o,ws,e);},aclnnSwiGlu); }
        void*ddown=mm(dmlp,S,F,dWd,D);
        void*dx3=add(dx2,ddown,S*D);
        void*dfin=rms(dx3,S,dg3);
        return mm(dfin,S,D,dWl,V);                  // [S,V] logits
    };
    void*dlog=fwd();                                // correctness run
    std::vector<float> logits(S*V); CHECK(aclrtMemcpy(logits.data(),S*V*4,dlog,S*V*4,ACL_MEMCPY_DEVICE_TO_HOST));

    // ===== CPU double reference =====
    V_ x(S*D); for(int64_t s=0;s<S;s++)for(int64_t d=0;d<D;d++)x[s*D+d]=Wemb[ids[s]*D+d];
    V_ G1(g1.begin(),g1.end()),G2(g2.begin(),g2.end()),G3(g3.begin(),g3.end());
    V_ h=rmsf(x,S,G1);
    V_ WQ(Wq.begin(),Wq.end()),WK(Wk.begin(),Wk.end()),WV(Wv.begin(),Wv.end()),WO(Wo.begin(),Wo.end());
    V_ q=mmf(h,S,D,WQ,D),k=mmf(h,S,D,WK,D),v=mmf(h,S,D,WV,D);
    // RoPE half-split
    auto rope=[&](V_&X){ V_ o(S*D); for(int64_t s=0;s<S;s++)for(int64_t d=0;d<D;d++){int64_t half=D/2;
        double rh=(d<half)?-X[s*D+d+half]:X[s*D+d-half]; o[s*D+d]=X[s*D+d]*cs[s*D+d]+rh*sn[s*D+d];} return o; };
    V_ qr=rope(q),kr=rope(k);
    // Causal single-head attention
    V_ attn(S*D,0);
    for(int64_t i=0;i<S;i++){ std::vector<double> sc(i+1); double mx=-1e30;
        for(int64_t j=0;j<=i;j++){double s=0;for(int64_t d=0;d<D;d++)s+=qr[i*D+d]*kr[j*D+d];sc[j]=s*SCALE;mx=std::max(mx,sc[j]);}
        double den=0; for(int64_t j=0;j<=i;j++){sc[j]=std::exp(sc[j]-mx);den+=sc[j];}
        for(int64_t d=0;d<D;d++){double s=0;for(int64_t j=0;j<=i;j++)s+=sc[j]/den*v[j*D+d];attn[i*D+d]=s;} }
    V_ op=mmf(attn,S,D,WO,D);
    V_ x2(S*D); for(int64_t i=0;i<S*D;i++)x2[i]=x[i]+op[i];
    V_ h2=rmsf(x2,S,G2);
    V_ WG(Wg.begin(),Wg.end()),WD(Wd.begin(),Wd.end()),WL(Wl.begin(),Wl.end());
    V_ gate=mmf(h2,S,D,WG,2*F);
    V_ mlp(S*F); for(int64_t s=0;s<S;s++)for(int64_t f=0;f<F;f++){double a=gate[s*2*F+f],b=gate[s*2*F+F+f];
        mlp[s*F+f]=a/(1+std::exp(-a))*b;}
    V_ down=mmf(mlp,S,F,WD,D);
    V_ x3(S*D); for(int64_t i=0;i<S*D;i++)x3[i]=x2[i]+down[i];
    V_ fin=rmsf(x3,S,G3);
    V_ ref=mmf(fin,S,D,WL,V);

    double me=0,mr=0; for(int64_t i=0;i<S*V;i++){me=std::max(me,std::fabs(logits[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));}
    double err=me/(mr+1e-9);
    bool ok=err<=1e-4;
    printf("tiny LLM forward (1-layer, single-head) logits normalized error=%.3e  %s\n", err, ok?"PASS":"FAIL");
    printf("== %d PASS, %d FAIL ==\n", ok?1:0, ok?0:1);

    // ===== Forward wall-clock timing: weights already uploaded, measure forward op chain only =====
    const int WARM=10, ITERS=200;
    for(int i=0;i<WARM;i++) fwd();
    CHECK(aclrtSynchronizeStream(g_stream));
    auto t0=std::chrono::steady_clock::now();
    for(int i=0;i<ITERS;i++) fwd();
    CHECK(aclrtSynchronizeStream(g_stream));
    double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/ITERS;
    printf("[perf] forward wall-clock = %.3f ms/iter (GPU) vs cannsim single-op validation ~85 s (CPU simulation without GPU)\n", ms);

    for(void*p:g_frees) aclrtFree(p);
    CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    return ok?0:1;
}
