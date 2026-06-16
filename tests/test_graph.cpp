// CUDA Graph capture/replay: captures a chain of 3 aclnnAdd ops t=a+b+b+b, replays after changing inputs to verify recomputation.
// Validates aclmdlRICaptureBegin/End + aclmdlRIExecuteAsync + aclmdlRIDestroy end-to-end.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

static aclrtStream S;
static int g_pass = 0, g_fail = 0;
#define CK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r); exit(1);} }while(0)
static void report(const char *n, bool ok){ (ok?g_pass:g_fail)++; printf("%-34s %s\n", n, ok?"PASS":"FAIL"); }
static void *up(const void *h, size_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,b,h,b,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static aclTensor *mk(int64_t n, void *p){ int64_t d[1]={n}; return aclCreateTensor(d,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d,1,p); }

static void add(aclTensor *a, aclTensor *b, aclTensor *o){
    uint64_t ws=0; aclOpExecutor *e=nullptr; CK(aclnnAddGetWorkspaceSize(a,b,nullptr,o,&ws,&e));
    void *wsp=nullptr; if(ws)CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclnnAdd(wsp,ws,e,S)); if(wsp)aclrtFree(wsp);
}

// Extend capture to a matmul-bearing decode step: captures one step of RMSNorm->Matmul->SiLU->Matmul->Add,
// replayed as a decode loop. Key: workspace pre-allocated and reused (no cudaMalloc during capture); GetWorkspaceSize only builds the host executor (legal).
static int64_t gD=64, gH=128, gS=4;
static void t_decode_capture(){
    auto rv=[&](int64_t n,float sc){ std::vector<float> v(n); for(auto&x:v)x=((rand()/(float)RAND_MAX)*2-1)*sc; return v; };
    auto X=rv(gS*gD,1.0f); std::vector<float> g(gD,1.0f); auto W1=rv(gD*gH,0.3f), W2=rv(gH*gD,0.3f);
    void*dX=up(X.data(),gS*gD*4),*dG=up(g.data(),gD*4),*dW1=up(W1.data(),gD*gH*4),*dW2=up(W2.data(),gH*gD*4);
    void*dH,*dA,*dR,*dO,*dXo;
    CK(aclrtMalloc(&dH,gS*gD*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dA,gS*gH*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dR,gS*gH*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dO,gS*gD*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&dXo,gS*gD*4,ACL_MEM_MALLOC_HUGE_FIRST));
    auto T2=[&](int64_t a,int64_t b,void*p){ int64_t d[2]={a,b}; return aclCreateTensor(d,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d,2,p); };
    auto T1=[&](int64_t a,void*p){ int64_t d[1]={a}; return aclCreateTensor(d,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d,1,p); };
    // Pre-allocate reusable workspace (sized for largest op; matmul needs cuBLASLt workspace)
    void*WS; CK(aclrtMalloc(&WS, 8u<<20, ACL_MEM_MALLOC_HUGE_FIRST));
    // One step: dXo = dX + W2·SiLU(W1·RMSNorm(dX)). GetWorkspaceSize only builds the host executor; Execute reuses WS (pre-allocated).
    auto step=[&](){
        { auto x=T2(gS,gD,dX),gg=T1(gD,dG),o=T2(gS,gD,dH); uint64_t w; aclOpExecutor*e;
          CK(aclnnRmsNormGetWorkspaceSize(x,gg,1e-6,o,&w,&e)); CK(aclnnRmsNorm(WS,w,e,S)); aclDestroyTensor(x);aclDestroyTensor(gg);aclDestroyTensor(o); }
        { auto a=T2(gS,gD,dH),b=T2(gD,gH,dW1),o=T2(gS,gH,dA); uint64_t w; aclOpExecutor*e;
          CK(aclnnMatmulGetWorkspaceSize(a,b,o,1,&w,&e)); CK(aclnnMatmul(WS,w,e,S)); aclDestroyTensor(a);aclDestroyTensor(b);aclDestroyTensor(o); }
        { auto a=T2(gS,gH,dA),o=T2(gS,gH,dR); uint64_t w; aclOpExecutor*e;
          CK(aclnnSiluGetWorkspaceSize(a,o,&w,&e)); CK(aclnnSilu(WS,w,e,S)); aclDestroyTensor(a);aclDestroyTensor(o); }
        { auto a=T2(gS,gH,dR),b=T2(gH,gD,dW2),o=T2(gS,gD,dO); uint64_t w; aclOpExecutor*e;
          CK(aclnnMatmulGetWorkspaceSize(a,b,o,1,&w,&e)); CK(aclnnMatmul(WS,w,e,S)); aclDestroyTensor(a);aclDestroyTensor(b);aclDestroyTensor(o); }
        { auto a=T2(gS,gD,dX),b=T2(gS,gD,dO),o=T2(gS,gD,dXo); uint64_t w; aclOpExecutor*e;
          CK(aclnnAddGetWorkspaceSize(a,b,nullptr,o,&w,&e)); CK(aclnnAdd(WS,w,e,S)); aclDestroyTensor(a);aclDestroyTensor(b);aclDestroyTensor(o); }
    };
    step(); CK(aclrtSynchronizeStream(S));   // warmup (cache allocation, cuBLAS handles)
    std::vector<float> ref(gS*gD); CK(aclrtMemcpy(ref.data(),gS*gD*4,dXo,gS*gD*4,ACL_MEMCPY_DEVICE_TO_HOST));
    // Capture this step
    aclmdlRI model=nullptr;
    CK(aclmdlRICaptureBegin(S, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    step();
    aclError ce = aclmdlRICaptureEnd(S, &model);
    bool captured = (ce==ACL_SUCCESS && model!=nullptr);
    report("Decode-step capture (matmul-bearing)", captured);
    if(!captured){ return; }
    // Replay as decode loop: 3 replays (same inputs, idempotent), each output must equal the direct-execution ref
    bool ok=true;
    for(int it=0; it<3 && ok; it++){
        CK(aclmdlRIExecuteAsync(model, S)); CK(aclrtSynchronizeStream(S));
        std::vector<float> got(gS*gD); CK(aclrtMemcpy(got.data(),gS*gD*4,dXo,gS*gD*4,ACL_MEMCPY_DEVICE_TO_HOST));
        for(int64_t i=0;i<gS*gD;i++) if(std::fabs(got[i]-ref[i])>1e-4){ ok=false; break; }
    }
    report("Decode-step graph replay x3 == direct", ok);
    CK(aclmdlRIDestroy(model));
    aclrtFree(dX);aclrtFree(dG);aclrtFree(dW1);aclrtFree(dW2);aclrtFree(dH);aclrtFree(dA);aclrtFree(dR);aclrtFree(dO);aclrtFree(dXo);aclrtFree(WS);
}

int main(){
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&S));
    const int64_t n = 4096;
    std::vector<float> a(n,1.f), b(n,2.f), out(n);
    void *da=up(a.data(),n*4), *db=up(b.data(),n*4), *t1, *t2, *t3;
    CK(aclrtMalloc(&t1,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&t2,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&t3,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *ta=mk(n,da),*tb=mk(n,db),*x1=mk(n,t1),*x2=mk(n,t2),*x3=mk(n,t3);
    // Chain: t1=a+b, t2=t1+b, t3=t2+b -> t3 = a + 3b
    auto chain=[&](){ add(ta,tb,x1); add(x1,tb,x2); add(x2,tb,x3); };

    chain(); CK(aclrtSynchronizeStream(S));   // warmup (prime the allocator to avoid cudaMalloc during capture)

    // Capture
    aclmdlRI model=nullptr; aclmdlRICaptureStatus st;
    CK(aclmdlRICaptureBegin(S, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL));
    chain();
    CK(aclmdlRICaptureGetInfo(S, &st, nullptr));
    bool capturing = (st == ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE);
    CK(aclmdlRICaptureEnd(S, &model));
    report("CaptureBegin/GetInfo(active)/End", capturing && model != nullptr);

    // Change input a=5 -> replay should give t3 = 5 + 3*2 = 11
    std::vector<float> a2(n,5.f); CK(aclrtMemcpy(da,n*4,a2.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclmdlRIExecuteAsync(model, S)); CK(aclrtSynchronizeStream(S));
    CK(aclrtMemcpy(out.data(),n*4,t3,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    bool ok1=true; for(auto v:out) if(std::fabs(v-11.f)>1e-4) ok1=false;
    report("Graph replay (a=5 → 5+3·2=11)", ok1);

    // Change a=10 -> replay t3 = 16, verifies multiple replays work
    std::vector<float> a3(n,10.f); CK(aclrtMemcpy(da,n*4,a3.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclmdlRIExecuteAsync(model, S)); CK(aclrtSynchronizeStream(S));
    CK(aclrtMemcpy(out.data(),n*4,t3,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    bool ok2=true; for(auto v:out) if(std::fabs(v-16.f)>1e-4) ok2=false;
    report("Graph re-replay (a=10 → 16)", ok2);

    CK(aclmdlRIDestroy(model));

    t_decode_capture();

    CK(aclrtDestroyStream(S)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
