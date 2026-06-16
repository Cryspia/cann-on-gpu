// Performance benchmark: pure ACL client, warmup + multi-iteration average wall-clock (including
// GetWorkspaceSize executor rebuild per call, i.e. true per-call cost); workspace buffer pre-allocated
// and reused. Reports ms + derived throughput (GFLOP/s or GB/s).
// Covers gemm(fp16/bf16), FlashAttentionScore(naive) vs HighPerf(cuBLASLt batched), elementwise add,
// reduce, rmsnorm.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

#define CK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static aclrtStream S;
static double now_ms(){ return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static uint16_t f2h(float f){ uint32_t x;__builtin_memcpy(&x,&f,4);uint32_t s=(x>>16)&0x8000;int32_t e=((x>>23)&0xFF)-112;uint32_t m=x&0x7FFFFF;
    if(e<=0)return s; if(e>=31)return s|0x7C00; return s|(e<<10)|(m>>13); }
static uint16_t f2bf(float f){ uint32_t b;__builtin_memcpy(&b,&f,4); b+=0x7FFF+((b>>16)&1); return (uint16_t)(b>>16); }
static void* dmalloc(size_t b){ void*p; CK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); return p; }
static aclTensor* T(std::vector<int64_t> d, void*p, aclDataType dt){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); }

// Two-phase benchmark: getws rebuilds the executor each call (true cost); ws buffer is reused
template<class GetWs, class Run>
static double bench(int iters, GetWs getws, Run run){
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(getws(&ws,&ex));
    void*wsp=nullptr; if(ws) wsp=dmalloc(ws);
    CK(run(wsp,ws,ex,S));                                   // consume the first executor
    for(int i=0;i<5;i++){ CK(getws(&ws,&ex)); CK(run(wsp,ws,ex,S)); }   // warmup
    CK(aclrtSynchronizeStream(S));
    double t0=now_ms();
    for(int i=0;i<iters;i++){ CK(getws(&ws,&ex)); CK(run(wsp,ws,ex,S)); }
    CK(aclrtSynchronizeStream(S));
    double ms=(now_ms()-t0)/iters;
    if(wsp) aclrtFree(wsp);
    return ms;
}
static void row(const char*name, double ms, double metric, const char*unit){
    printf("%-34s %8.4f ms   %8.1f %s\n", name, ms, metric, unit);
}

// ---- GEMM ----
static void bench_gemm(aclDataType dt, const char*nm, int64_t M,int64_t K,int64_t N){
    size_t es = (dt==ACL_FLOAT)?4:2;
    void*da=dmalloc(M*K*es),*db=dmalloc(K*N*es),*dc=dmalloc(M*N*es);
    aclTensor*ta=T({M,K},da,dt),*tb=T({K,N},db,dt),*tc=T({M,N},dc,dt);
    double ms=bench(50,[&](uint64_t*w,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(ta,tb,tc,1,w,e);},aclnnMatmul);
    double gflops=2.0*M*N*K/(ms*1e-3)/1e9;
    char b[64]; snprintf(b,64,"GEMM %s %ldx%ldx%ld",nm,(long)M,(long)K,(long)N);
    row(b,ms,gflops,"GFLOP/s");
    aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
// ---- attention: default (naive) vs HighPerf (cuBLASLt batched) ----
static void bench_attn(bool perf, int64_t B,int64_t Nh,int64_t Sq,int64_t D){
    aclDataType dt=ACL_FLOAT16; size_t es=2;
    void*dq=dmalloc(B*Nh*Sq*D*es),*dk=dmalloc(B*Nh*Sq*D*es),*dv=dmalloc(B*Nh*Sq*D*es),*doO=dmalloc(B*Nh*Sq*D*es);
    aclTensor*tq=T({B,Nh,Sq,D},dq,dt),*tk=T({B,Nh,Sq,D},dk,dt),*tv=T({B,Nh,Sq,D},dv,dt),*to=T({B,Nh,Sq,D},doO,dt);
    double scale=1.0/std::sqrt((double)D);
    double ms = perf
      ? bench(30,[&](uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,false,to,w,e);},aclnnFlashAttentionScoreHighPerf)
      : bench(30,[&](uint64_t*w,aclOpExecutor**e){return aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,false,to,w,e);},aclnnFlashAttentionScore);
    double gflops=4.0*B*Nh*Sq*Sq*D/(ms*1e-3)/1e9;   // QK^T + PV ≈ 2*2*B*Nh*S*S*D
    char b[64]; snprintf(b,64,"Attn %s B%ldN%ldS%ldD%ld",perf?"perf ":"flash",(long)B,(long)Nh,(long)Sq,(long)D);
    row(b,ms,gflops,"GFLOP/s");
    aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(doO);
}
// ---- elementwise add ----
static void bench_add(int64_t n){
    void*da=dmalloc(n*4),*db=dmalloc(n*4),*dc=dmalloc(n*4);
    aclTensor*ta=T({n},da,ACL_FLOAT),*tb=T({n},db,ACL_FLOAT),*tc=T({n},dc,ACL_FLOAT);
    double ms=bench(100,[&](uint64_t*w,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(ta,tb,nullptr,tc,w,e);},aclnnAdd);
    double gbs=3.0*n*4/(ms*1e-3)/1e9;
    char b[48]; snprintf(b,48,"Add n=%ldM",(long)(n>>20)); row(b,ms,gbs,"GB/s");
    aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
// ---- reduce sum (full reduction) ----
static void bench_reduce(int64_t n){
    void*dx=dmalloc(n*4),*dz=dmalloc(4);
    aclTensor*tx=T({n},dx,ACL_FLOAT),*tz=T({1},dz,ACL_FLOAT);
    int64_t ax[1]={0}; aclIntArray*dim=aclCreateIntArray(ax,1);
    double ms=bench(100,[&](uint64_t*w,aclOpExecutor**e){return aclnnReduceSumGetWorkspaceSize(tx,dim,false,ACL_FLOAT,tz,w,e);},aclnnReduceSum);
    double gbs=(double)n*4/(ms*1e-3)/1e9;
    char b[48]; snprintf(b,48,"ReduceSum n=%ldM",(long)(n>>20)); row(b,ms,gbs,"GB/s");
    aclDestroyIntArray(dim); aclrtFree(dx);aclrtFree(dz);
}
// ---- rmsnorm ----
static void bench_rms(int64_t rows,int64_t D){
    void*dx=dmalloc(rows*D*4),*dg=dmalloc(D*4),*dy=dmalloc(rows*D*4);
    aclTensor*tx=T({rows,D},dx,ACL_FLOAT),*tg=T({D},dg,ACL_FLOAT),*ty=T({rows,D},dy,ACL_FLOAT);
    double ms=bench(100,[&](uint64_t*w,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(tx,tg,1e-6,ty,w,e);},aclnnRmsNorm);
    double gbs=2.0*rows*D*4/(ms*1e-3)/1e9;
    char b[48]; snprintf(b,48,"RMSNorm %ldx%ld",(long)rows,(long)D); row(b,ms,gbs,"GB/s");
    aclrtFree(dx);aclrtFree(dg);aclrtFree(dy);
}

// ---- PagedAttention (decode: large batch, Sq=1, long context) ----
static void bench_paged(bool scalar, int64_t B,int64_t Nh,int64_t D,int64_t L,int64_t blockSize){
    if (scalar) setenv("PAGED_SCALAR","1",1); else unsetenv("PAGED_SCALAR");
    aclDataType dt=ACL_FLOAT16; size_t es=2;
    int64_t Sq=1, Nkv=Nh, maxBlk=(L+blockSize-1)/blockSize, numBlk=B*maxBlk;
    void*dq=dmalloc(B*Nh*Sq*D*es),*dkc=dmalloc(numBlk*blockSize*Nkv*D*es),*dvc=dmalloc(numBlk*blockSize*Nkv*D*es),*doO=dmalloc(B*Nh*Sq*D*es);
    // blockTable[b*maxBlk+p]=b*maxBlk+p (each sequence owns contiguous blocks); ctxLen[b]=L
    std::vector<int32_t> bt(B*maxBlk), cl(B,(int32_t)L);
    for(int64_t b=0;b<B;b++)for(int64_t p=0;p<maxBlk;p++) bt[b*maxBlk+p]=(int32_t)(b*maxBlk+p);
    void*dbt=dmalloc(B*maxBlk*4),*dcl=dmalloc(B*4);
    CK(aclrtMemcpy(dbt,B*maxBlk*4,bt.data(),B*maxBlk*4,ACL_MEMCPY_HOST_TO_DEVICE));
    CK(aclrtMemcpy(dcl,B*4,cl.data(),B*4,ACL_MEMCPY_HOST_TO_DEVICE));
    aclTensor*tq=T({B,Nh,Sq,D},dq,dt),*tkc=T({numBlk,blockSize,Nkv,D},dkc,dt),*tvc=T({numBlk,blockSize,Nkv,D},dvc,dt);
    aclTensor*tbt=T({B,maxBlk},dbt,ACL_INT32),*tcl=T({B},dcl,ACL_INT32),*to=T({B,Nh,Sq,D},doO,dt);
    double scale=1.0/std::sqrt((double)D);
    double ms=bench(30,[&](uint64_t*w,aclOpExecutor**e){return aclnnPagedAttentionGetWorkspaceSize(tq,tkc,tvc,tbt,tcl,scale,Nh,to,w,e);},aclnnPagedAttention);
    double gflops=4.0*B*Nh*L*D/(ms*1e-3)/1e9;
    char b[64]; snprintf(b,64,"Paged %s B%ldN%ldL%ldD%ld",scalar?"scalar":"warp  ",(long)B,(long)Nh,(long)L,(long)D);
    row(b,ms,gflops,"GFLOP/s");
    unsetenv("PAGED_SCALAR");
    aclrtFree(dq);aclrtFree(dkc);aclrtFree(dvc);aclrtFree(doO);aclrtFree(dbt);aclrtFree(dcl);
}

// ---- softmax (last dimension) ----
static void bench_softmax(int64_t rows,int64_t L){
    void*dx=dmalloc(rows*L*4),*dy=dmalloc(rows*L*4);
    aclTensor*tx=T({rows,L},dx,ACL_FLOAT),*ty=T({rows,L},dy,ACL_FLOAT);
    double ms=bench(100,[&](uint64_t*w,aclOpExecutor**e){return aclnnSoftmaxGetWorkspaceSize(tx,-1,ty,w,e);},aclnnSoftmax);
    double gbs=2.0*rows*L*4/(ms*1e-3)/1e9;
    char b[48]; snprintf(b,48,"Softmax %ldx%ld",(long)rows,(long)L); row(b,ms,gbs,"GB/s");
    aclrtFree(dx);aclrtFree(dy);
}

// ---- allocator churn: repeated malloc+free, comparing cached vs passthrough ----
static void bench_alloc(int iters){
    size_t szs[4] = {1u<<16, 1u<<20, 4u<<20, 16u<<20};   // 64KB/1MB/4MB/16MB
    // warmup (prime the cache / driver initialization)
    for(int i=0;i<8;i++){ void*p; CK(aclrtMalloc(&p,szs[i&3],ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtFree(p)); }
    double t0=now_ms();
    for(int i=0;i<iters;i++){ void*p; CK(aclrtMalloc(&p,szs[i&3],ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtFree(p)); }
    double us=(now_ms()-t0)/iters*1000.0;
    char b[64]; snprintf(b,64,"malloc+free/pair (%s)", getenv("ACL_NO_CACHE_ALLOC")?"passthrough":"cached");
    printf("%-34s %8.3f us/pair\n", b, us);
}

int main(){
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&S));
    printf("---- device memory allocator ----\n");
    bench_alloc(2000);
    printf("==== cann-on-gpu performance benchmark (wall-clock/call, including executor build) ====\n");
    bench_gemm(ACL_FLOAT16,"fp16",1024,1024,1024); bench_gemm(ACL_FLOAT16,"fp16",4096,4096,4096);
    bench_gemm(ACL_BF16,"bf16",1024,1024,1024);     bench_gemm(ACL_BF16,"bf16",4096,4096,4096);
    printf("---- attention (fp16, flash online-softmax no-materialization vs perf cuBLASLt batched) ----\n");
    for(int64_t s : {128,512,1024,2048}){ bench_attn(false,4,8,s,64); bench_attn(true,4,8,s,64); }
    printf("---- PagedAttention decode (fp16, B64 N8 D64, scalar legacy vs warp optimized) ----\n");
    for(int64_t L : {1024,4096}){ bench_paged(true,64,8,64,L,128); bench_paged(false,64,8,64,L,128); }
    printf("---- memory-bandwidth-bound ----\n");
    bench_add(1<<22); bench_reduce(1<<22); bench_rms(4096,1024); bench_softmax(4096,1024);
    CK(aclrtDestroyStream(S)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    return 0;
}
