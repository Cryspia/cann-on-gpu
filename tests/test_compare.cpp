// Comparison/logical/predicate operator cross-check: bool outputs, exact match. Includes broadcast and MaskedFill.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)
static aclrtStream g_stream; static int g_pass = 0, g_fail = 0;
struct DevBuf { void *p=nullptr; size_t bytes=0; DevBuf(size_t b):bytes(b){CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST));}
    ~DevBuf(){aclrtFree(p);} void up(const void*h){CHECK(aclrtMemcpy(p,bytes,h,bytes,ACL_MEMCPY_HOST_TO_DEVICE));}
    void down(void*h)const{CHECK(aclrtMemcpy(h,bytes,p,bytes,ACL_MEMCPY_DEVICE_TO_HOST));} };
static aclTensor *mk(const std::vector<int64_t>&d, aclDataType dt, void*data){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),data); }
static void report(const std::string&name,int64_t bad){ (bad==0?g_pass:g_fail)++; printf("%-28s mismatch=%lld %s\n",name.c_str(),(long long)bad,bad==0?"PASS":"FAIL"); }
template<typename G> static void exec2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex)); void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); }
static std::vector<float> randv(int64_t n,float lo,float hi){ std::vector<float> v(n); for(auto&x:v)x=lo+(hi-lo)*(rand()/(float)RAND_MAX); return v; }

typedef aclnnStatus(*CmpWs)(const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
typedef aclnnStatus(*CmpRun)(void*,uint64_t,aclOpExecutor*,aclrtStream);
static void t_cmp(const char*name, CmpWs ws, CmpRun run, bool(*ref)(float,float)){
    const int64_t n=1<<16; auto fa=randv(n,-1,1), fb=randv(n,-1,1);
    // Inject some equal pairs
    for(int64_t i=0;i<n;i+=37) fb[i]=fa[i];
    std::vector<uint8_t> hz(n);
    DevBuf da(n*4),db(n*4),dz(n); da.up(fa.data()); db.up(fb.data());
    aclTensor*ta=mk({n},ACL_FLOAT,da.p),*tb=mk({n},ACL_FLOAT,db.p),*tz=mk({n},ACL_BOOL,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return ws(ta,tb,tz,w,e);},run);
    dz.down(hz.data());
    int64_t bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=ref(fa[i],fb[i])) bad++;
    report(name,bad);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

static void t_cmp_bcast(){
    const int64_t M=64,N=48; auto fa=randv(M*N,-1,1), fb=randv(N,-1,1);
    std::vector<uint8_t> hz(M*N);
    DevBuf da(M*N*4),db(N*4),dz(M*N); da.up(fa.data()); db.up(fb.data());
    aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tb=mk({N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_BOOL,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGtTensorGetWorkspaceSize(ta,tb,tz,w,e);},aclnnGtTensor);
    dz.down(hz.data());
    int64_t bad=0; for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++) if((hz[i*N+j]!=0)!=(fa[i*N+j]>fb[j])) bad++;
    report("Greater bcast [M,N]>[N]",bad);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

static void t_logic(){
    const int64_t n=1<<16; std::vector<uint8_t> a(n),b(n),hz(n);
    for(int64_t i=0;i<n;i++){a[i]=rand()&1; b[i]=rand()&1;}
    DevBuf da(n),db(n),dz(n); da.up(a.data()); db.up(b.data());
    aclTensor*ta=mk({n},ACL_BOOL,da.p),*tb=mk({n},ACL_BOOL,db.p),*tz=mk({n},ACL_BOOL,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogicalAndGetWorkspaceSize(ta,tb,tz,w,e);},aclnnLogicalAnd);
    dz.down(hz.data()); int64_t bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=((a[i]&&b[i]))) bad++; report("LogicalAnd",bad);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogicalOrGetWorkspaceSize(ta,tb,tz,w,e);},aclnnLogicalOr);
    dz.down(hz.data()); bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=((a[i]||b[i]))) bad++; report("LogicalOr",bad);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogicalNotGetWorkspaceSize(ta,tz,w,e);},aclnnLogicalNot);
    dz.down(hz.data()); bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=(!a[i])) bad++; report("LogicalNot",bad);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

static void t_pred(){
    const int64_t n=8; float vals[n]={1.f, 0.f, INFINITY, -INFINITY, NAN, 3.5f, -2.f, NAN};
    std::vector<uint8_t> hz(n);
    DevBuf da(n*4),dz(n); da.up(vals);
    aclTensor*ta=mk({n},ACL_FLOAT,da.p),*tz=mk({n},ACL_BOOL,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIsNanGetWorkspaceSize(ta,tz,w,e);},aclnnIsNan);
    dz.down(hz.data()); int64_t bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=(bool)std::isnan(vals[i])) bad++; report("IsNan",bad);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIsFiniteGetWorkspaceSize(ta,tz,w,e);},aclnnIsFinite);
    dz.down(hz.data()); bad=0; for(int64_t i=0;i<n;i++) if((hz[i]!=0)!=(bool)std::isfinite(vals[i])) bad++; report("IsFinite",bad);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

static void t_maskfill(){
    const int64_t n=1<<16; auto fa=randv(n,-2,2); std::vector<uint8_t> mask(n); for(auto&m:mask)m=rand()&1;
    std::vector<float> hz(n); float val=-9.f;
    DevBuf da(n*4),dm(n),dz(n*4); da.up(fa.data()); dm.up(mask.data());
    aclTensor*ta=mk({n},ACL_FLOAT,da.p),*tm=mk({n},ACL_BOOL,dm.p),*tz=mk({n},ACL_FLOAT,dz.p);
    aclScalar*sv=aclCreateScalar(&val,ACL_FLOAT);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedFillScalarGetWorkspaceSize(ta,tm,sv,tz,w,e);},aclnnMaskedFillScalar);
    dz.down(hz.data()); int64_t bad=0; for(int64_t i=0;i<n;i++){ float ref=mask[i]?val:fa[i]; if(hz[i]!=ref) bad++; }
    report("MaskedFill v=-9",bad);
    aclDestroyScalar(sv); aclDestroyTensor(ta); aclDestroyTensor(tm); aclDestroyTensor(tz);
}

typedef aclnnStatus(*CmpSWs)(const aclTensor*,const aclScalar*,aclTensor*,uint64_t*,aclOpExecutor**);
static void t_cmp_s(const char*name, CmpSWs ws, CmpRun run, float thr, bool(*ref)(float,float)){
    const int64_t n=1<<16; auto fa=randv(n,-1,1);
    for(int64_t i=0;i<n;i+=53) fa[i]=thr;   // inject equal-to-threshold cases
    DevBuf ba(n*4); ba.up(fa.data()); DevBuf bo(n);
    auto ta=mk({n},ACL_FLOAT,ba.p), to=mk({n},ACL_BOOL,bo.p);
    float t=thr; auto sc=aclCreateScalar(&t,ACL_FLOAT);
    exec2([&](uint64_t*w,aclOpExecutor**e){return ws(ta,sc,to,w,e);}, run);
    std::vector<uint8_t> got(n); bo.down(got.data());
    int64_t bad=0; for(int64_t i=0;i<n;i++){ uint8_t r=ref(fa[i],thr)?1:0; if(got[i]!=r)bad++; }
    report(name, bad);
}
static void t_scalar_cmp(){
    t_cmp_s("GtScalar", aclnnGtScalarGetWorkspaceSize, aclnnGtScalar, 0.25f, [](float a,float b){return a>b;});
    t_cmp_s("LtScalar", aclnnLtScalarGetWorkspaceSize, aclnnLtScalar, 0.25f, [](float a,float b){return a<b;});
    t_cmp_s("EqScalar", aclnnEqScalarGetWorkspaceSize, aclnnEqScalar, 0.25f, [](float a,float b){return a==b;});
    t_cmp_s("NeScalar", aclnnNeScalarGetWorkspaceSize, aclnnNeScalar, 0.25f, [](float a,float b){return a!=b;});
    t_cmp_s("GeScalar", aclnnGeScalarGetWorkspaceSize, aclnnGeScalar, 0.25f, [](float a,float b){return a>=b;});
    t_cmp_s("LeScalar", aclnnLeScalarGetWorkspaceSize, aclnnLeScalar, 0.25f, [](float a,float b){return a<=b;});
}

int main(){
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(11);
    t_cmp("Greater", aclnnGtTensorGetWorkspaceSize, aclnnGtTensor, [](float a,float b){return a>b;});
    t_cmp("Less",    aclnnLtTensorGetWorkspaceSize, aclnnLtTensor, [](float a,float b){return a<b;});
    t_cmp("Equal",   aclnnEqTensorGetWorkspaceSize, aclnnEqTensor, [](float a,float b){return a==b;});
    t_cmp("NotEqual",aclnnNeTensorGetWorkspaceSize, aclnnNeTensor, [](float a,float b){return a!=b;});
    t_cmp("GreaterEqual",aclnnGeTensorGetWorkspaceSize, aclnnGeTensor, [](float a,float b){return a>=b;});
    t_cmp("LessEqual",   aclnnLeTensorGetWorkspaceSize, aclnnLeTensor, [](float a,float b){return a<=b;});
    t_cmp_bcast();
    t_scalar_cmp();
    t_logic();
    t_pred();
    t_maskfill();
    CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
