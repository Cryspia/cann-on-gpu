// Self-contained cross-check for the 38 distributed / comm-fused operators (dist_parity.mm), SINGLE-RANK.
//
// On this single-device backend every HCCL collective degenerates to identity on rank 0 of 1, so each fused
// op equals its compute part with the collective as a no-op. We verify each against the CPU single-rank
// equivalent:
//   *MatmulAllReduce / *ReduceScatter / *AllGather / AllGatherMatmul     == plain Matmul (+bias)
//   QuantMatmulAllReduce* / QuantReduceScatter                           == int8-dequant matmul: (xq@wq)*scale[n]
//   WeightQuantMatmulAllReduce                                           == antiquant matmul: x@((w-off)*scale)
//   GroupedMatMulAllReduce                                               == per-group matmul (shared W) == full matmul
//   AlltoAllAllGatherBatchMatMul / BatchMatMulReduceScatterAlltoAll      == batched matmul (+bias)
//   AllGatherAdd                                                         == x + bias
//   QuantAllReduce / MoeDistribute{Dispatch,Combine}(+V2..V4,Setup/Teardown) == identity copy
//   *MatmulAllReduceAddRmsNorm (+Quant/WeightQuant, +Inplace)            == Matmul (+(de)quant) then AddRmsNorm
//   MoeDistributeCombineAddRmsNorm(+V2)                                  == AddRmsNorm(x + residual)·gamma
//
// HcclComm passed as nullptr (single rank => local only). Tolerances: 1e-5 for fp32 matmul/add; 1e-4 for the
// rmsnorm-fused tails (extra fp accumulation). int8/antiquant paths are exact (integer / fp32 product) so 1e-5.
#include "harness.h"
using namespace hn;

typedef void *HcclComm;

extern "C" {
// plain matmul-collective family: (x, weight, bias, comm, out)
#define MM(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**); aclnnStatus N(void*,uint64_t,aclOpExecutor*,aclrtStream);
MM(aclnnMatmulAllReduce) MM(aclnnMatmulAllReduceV2) MM(aclnnMatmulReduceScatter) MM(aclnnMatmulReduceScatterV2)
MM(aclnnMatmulAllGather) MM(aclnnAllGatherMatmul) MM(aclnnAllGatherMatmulV2)
MM(aclnnAlltoAllAllGatherBatchMatMul) MM(aclnnBatchMatMulReduceScatterAlltoAll)
#undef MM
// quant matmul family: (x, weight, scale, comm, out)
#define QMM(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**); aclnnStatus N(void*,uint64_t,aclOpExecutor*,aclrtStream);
QMM(aclnnQuantMatmulAllReduce) QMM(aclnnQuantMatmulAllReduceV2) QMM(aclnnQuantMatmulAllReduceV3)
QMM(aclnnQuantMatmulAllReduceV4) QMM(aclnnQuantReduceScatter)
#undef QMM
// alltoall / dispatch / combine family: (x, comm, out)
#define A2A(N) aclnnStatus N##GetWorkspaceSize(const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**); aclnnStatus N(void*,uint64_t,aclOpExecutor*,aclrtStream);
A2A(aclnnQuantAllReduce)
A2A(aclnnMoeDistributeDispatch) A2A(aclnnMoeDistributeDispatchV2) A2A(aclnnMoeDistributeDispatchV3) A2A(aclnnMoeDistributeDispatchV4)
A2A(aclnnMoeDistributeCombine) A2A(aclnnMoeDistributeCombineV2) A2A(aclnnMoeDistributeCombineV3) A2A(aclnnMoeDistributeCombineV4)
A2A(aclnnMoeDistributeDispatchSetup) A2A(aclnnMoeDistributeDispatchTeardown)
A2A(aclnnMoeDistributeCombineSetup) A2A(aclnnMoeDistributeCombineTeardown)
#undef A2A
aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor*,const aclTensor*,const char*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnAllGatherAdd(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnWeightQuantMatmulAllReduce(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnGroupedMatMulAllReduceGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclIntArray*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnGroupedMatMulAllReduce(void*,uint64_t,aclOpExecutor*,aclrtStream);
// AddRmsNorm tails: (x, weight, [scale,] residual, gamma, eps, comm, y, residualSum)
aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor*,const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,double,HcclComm,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
aclnnStatus aclDestroyIntArray(const aclIntArray *array);
}

static HcclComm NO_COMM = nullptr;

// ---- CPU references ----
static std::vector<double> ref_mm(const std::vector<float>&a,const std::vector<float>&b,const std::vector<float>*bias,int M,int K,int N){
    std::vector<double> r((size_t)M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)a[(size_t)m*K+k]*b[(size_t)k*N+n]; if(bias)s+=(*bias)[n]; r[(size_t)m*N+n]=s;}
    return r;
}
static std::vector<double> ref_qmm(const std::vector<int8_t>&a,const std::vector<int8_t>&b,const std::vector<float>&sc,int M,int K,int N){
    std::vector<double> r((size_t)M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){long long s=0;for(int k=0;k<K;k++)s+=(long long)a[(size_t)m*K+k]*(long long)b[(size_t)k*N+n]; r[(size_t)m*N+n]=(double)s*sc[n];}
    return r;
}
static std::vector<double> ref_wqmm(const std::vector<float>&x,const std::vector<int8_t>&w,const std::vector<float>&sc,const std::vector<float>&of,int M,int K,int N){
    std::vector<double> r((size_t)M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++){double wf=((double)w[(size_t)k*N+n]-of[n])*sc[n]; s+=(double)x[(size_t)m*K+k]*wf;} r[(size_t)m*N+n]=s;}
    return r;
}
// y = rms_r(temp+residual)*gamma over last dim N
static std::vector<double> ref_addrms(const std::vector<double>&temp,const std::vector<float>&res,const std::vector<float>&g,double eps,int M,int N){
    std::vector<double> y((size_t)M*N,0);
    for(int m=0;m<M;m++){double ss=0;std::vector<double> s(N);
        for(int n=0;n<N;n++){double v=temp[(size_t)m*N+n]+res[(size_t)m*N+n];s[n]=v;ss+=v*v;}
        double inv=1.0/std::sqrt(ss/N+eps);
        for(int n=0;n<N;n++)y[(size_t)m*N+n]=s[n]*inv*g[n];}
    return y;
}

// ============ plain matmul-collective family ============
template<typename GW,typename RUN>
static void t_mm(const char*name,GW gws,RUN run,bool withBias){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1),w=randv(K*N,-1,1),b=randv(N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(K*N*4),db(N*4),dz(M*N*4); dx.up(x.data());dw.up(w.data());db.up(b.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_FLOAT,dw.p),*tb=mk({N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclTensor *bias=withBias?tb:nullptr;
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,tw,bias,NO_COMM,tz,ws,e);}, run);
    dz.down(hz.data());
    report(name, norm_err(hz, ref_mm(x,w,withBias?&b:nullptr,M,K,N)), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tb);aclDestroyTensor(tz);
}

// ============ batched matmul (alltoall+bmm) ============
template<typename GW,typename RUN>
static void t_bmm(const char*name,GW gws,RUN run){
    const int B=2,M=3,K=5,N=4; auto x=randv(B*M*K,-1,1),w=randv(B*K*N,-1,1),b=randv(N,-1,1); std::vector<float> hz(B*M*N);
    DevBuf dx(B*M*K*4),dw(B*K*N*4),db(N*4),dz(B*M*N*4); dx.up(x.data());dw.up(w.data());db.up(b.data());
    aclTensor *tx=mk({B,M,K},ACL_FLOAT,dx.p),*tw=mk({B,K,N},ACL_FLOAT,dw.p),*tb=mk({N},ACL_FLOAT,db.p),*tz=mk({B,M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,tw,tb,NO_COMM,tz,ws,e);}, run);
    dz.down(hz.data());
    std::vector<double> r((size_t)B*M*N,0);
    for(int bb=0;bb<B;bb++)for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)x[((size_t)bb*M+m)*K+k]*w[((size_t)bb*K+k)*N+n]; s+=b[n]; r[((size_t)bb*M+m)*N+n]=s;}
    report(name, norm_err(hz,r), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tb);aclDestroyTensor(tz);
}

// ============ quant matmul family ============
template<typename GW,typename RUN>
static void t_qmm(const char*name,GW gws,RUN run){
    const int M=8,K=16,N=8;   // K%16==0 for the int8 tensor-core path
    std::vector<int8_t> x(M*K),w(K*N); for(auto&v:x)v=(int8_t)(rand()%63-31); for(auto&v:w)v=(int8_t)(rand()%63-31);
    auto sc=randv(N,0.01f,0.05f); for(auto&v:sc) v=h2f(f2h(v)); std::vector<uint16_t> sh(N); for(int i=0;i<N;i++) sh[i]=f2h(sc[i]);   // deqScale is fp16 (CANN/CUDA-canonical)
    std::vector<float> hz(M*N);
    DevBuf dx(M*K),dw(K*N),ds(N*2),dz(M*N*4); dx.up(x.data());dw.up(w.data());ds.up(sh.data());
    aclTensor *tx=mk({M,K},ACL_INT8,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*ts=mk({N},ACL_FLOAT16,ds.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,tw,ts,NO_COMM,tz,ws,e);}, run);
    dz.down(hz.data());
    report(name, norm_err(hz, ref_qmm(x,w,sc,M,K,N)), 3e-3);   // fp16 deqScale precision
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ts);aclDestroyTensor(tz);
}

// ============ identity copy (quant-allreduce / dispatch / combine / setup / teardown) ============
template<typename GW,typename RUN>
static void t_copy(const char*name,GW gws,RUN run,aclDataType dt){
    const int n=24;
    if(dt==ACL_INT8){
        std::vector<int8_t> x(n); for(auto&v:x)v=(int8_t)(rand()%255-127); std::vector<int8_t> hz(n);
        DevBuf dx(n),dz(n); dx.up(x.data());
        aclTensor *tx=mk({4,6},ACL_INT8,dx.p),*tz=mk({4,6},ACL_INT8,dz.p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,NO_COMM,tz,ws,e);}, run);
        dz.down(hz.data());
        double me=0; for(int i=0;i<n;i++)me=std::max(me,(double)std::abs(hz[i]-x[i]));
        report(name, me, 0.0);
        aclDestroyTensor(tx);aclDestroyTensor(tz);
    } else {
        auto x=randv(n,-2,2); std::vector<float> hz(n);
        DevBuf dx(n*4),dz(n*4); dx.up(x.data());
        aclTensor *tx=mk({4,6},ACL_FLOAT,dx.p),*tz=mk({4,6},ACL_FLOAT,dz.p);
        exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,NO_COMM,tz,ws,e);}, run);
        dz.down(hz.data());
        std::vector<double> r(x.begin(),x.end());
        report(name, norm_err(hz,r), 1e-6);
        aclDestroyTensor(tx);aclDestroyTensor(tz);
    }
}

// ============ AddRmsNorm tails ============
static void t_mm_rms(){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1),w=randv(K*N,-1,1),res=randv(M*N,-1,1),g=randv(N,0.5,1.5);
    std::vector<float> hy(M*N),hsum(M*N); double eps=1e-6;
    DevBuf dx(M*K*4),dw(K*N*4),dr(M*N*4),dg(N*4),dy(M*N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_FLOAT,dw.p),*tr=mk({M,N},ACL_FLOAT,dr.p),*tg=mk({N},ACL_FLOAT,dg.p),
        *ty=mk({M,N},ACL_FLOAT,dy.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,tr,tg,eps,NO_COMM,ty,tsum,ws,e);}, aclnnMatmulAllReduceAddRmsNorm);
    dy.down(hy.data()); hsum.assign(M*N,0); dsum.down(hsum.data());
    auto mm=ref_mm(x,w,nullptr,M,K,N);
    report("aclnnMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    // residualSum = mm + res
    std::vector<double> rs(M*N); for(int i=0;i<M*N;i++)rs[i]=mm[i]+res[i];
    report("  -> residualSum", norm_err(hsum,rs), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
}
static void t_qmm_rms(){
    const int M=6,K=5,N=4;
    std::vector<int8_t> x(M*K),w(K*N); for(auto&v:x)v=(int8_t)(rand()%31-15); for(auto&v:w)v=(int8_t)(rand()%31-15);
    auto sc=randv(N,0.01f,0.03f),res=randv(M*N,-1,1),g=randv(N,0.5,1.5); std::vector<float> hy(M*N); double eps=1e-6;
    DevBuf dx(M*K),dw(K*N),ds(N*4),dr(M*N*4),dg(N*4),dy(M*N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());ds.up(sc.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_INT8,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*ts=mk({N},ACL_FLOAT,ds.p),*tr=mk({M,N},ACL_FLOAT,dr.p),
        *tg=mk({N},ACL_FLOAT,dg.p),*ty=mk({M,N},ACL_FLOAT,dy.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,ts,tr,tg,eps,NO_COMM,ty,tsum,ws,e);}, aclnnQuantMatmulAllReduceAddRmsNorm);
    dy.down(hy.data());
    auto mm=ref_qmm(x,w,sc,M,K,N);
    report("aclnnQuantMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ts);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
}
static void t_wqmm_rms(){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1); std::vector<int8_t> w(K*N); for(auto&v:w)v=(int8_t)(rand()%63-31);
    auto sc=randv(N,0.01f,0.05f),res=randv(M*N,-1,1),g=randv(N,0.5,1.5); std::vector<float> hy(M*N); double eps=1e-6;
    DevBuf dx(M*K*4),dw(K*N),ds(N*4),dr(M*N*4),dg(N*4),dy(M*N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());ds.up(sc.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*ts=mk({N},ACL_FLOAT,ds.p),*tr=mk({M,N},ACL_FLOAT,dr.p),
        *tg=mk({N},ACL_FLOAT,dg.p),*ty=mk({M,N},ACL_FLOAT,dy.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,ts,tr,tg,eps,NO_COMM,ty,tsum,ws,e);}, aclnnWeightQuantMatmulAllReduceAddRmsNorm);
    dy.down(hy.data());
    // weight-only antiquant with offset=0 here: wf = w*scale
    std::vector<float> zero(N,0.f); auto mm=ref_wqmm(x,w,sc,zero,M,K,N);
    report("aclnnWeightQuantMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ts);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
}
static void t_inplace_mm_rms(){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1),w=randv(K*N,-1,1),res=randv(M*N,-1,1),g=randv(N,0.5,1.5);
    std::vector<float> hy(M*N); double eps=1e-6;
    // selfRef holds x[M,K] but y is [M,N] written back into the same buffer (M*N <= M*K here? N<K so fits; allocate max)
    DevBuf dx(M*(K>N?K:N)*4),dw(K*N*4),dr(M*N*4),dg(N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_FLOAT,dw.p),*tr=mk({M,N},ACL_FLOAT,dr.p),*tg=mk({N},ACL_FLOAT,dg.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,tr,tg,eps,NO_COMM,tsum,ws,e);}, aclnnInplaceMatmulAllReduceAddRmsNorm);
    // y written back into x's buffer as [M,N]
    DevBuf &db=dx; std::vector<float> back(M*(K>N?K:N)); db.down(back.data()); for(int i=0;i<M*N;i++)hy[i]=back[i];
    auto mm=ref_mm(x,w,nullptr,M,K,N);
    report("aclnnInplaceMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tsum);
}
static void t_inplace_qmm_rms(){
    // Inplace quant variant: selfRef is the float activation x[M,K] (in == out dtype under inplace), weight is
    // int8 [K,N] dequantized per-N by scale (weight-quant form); y written back into selfRef as [M,N].
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1); std::vector<int8_t> w(K*N); for(auto&v:w)v=(int8_t)(rand()%63-31);
    auto sc=randv(N,0.01f,0.05f),res=randv(M*N,-1,1),g=randv(N,0.5,1.5); std::vector<float> hy(M*N); double eps=1e-6;
    DevBuf dx(M*(K>N?K:N)*4),dw(K*N),ds(N*4),dr(M*N*4),dg(N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());ds.up(sc.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*ts=mk({N},ACL_FLOAT,ds.p),*tr=mk({M,N},ACL_FLOAT,dr.p),*tg=mk({N},ACL_FLOAT,dg.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,ts,tr,tg,eps,NO_COMM,tsum,ws,e);}, aclnnInplaceQuantMatmulAllReduceAddRmsNorm);
    std::vector<float> back(M*(K>N?K:N)); dx.down(back.data()); for(int i=0;i<M*N;i++)hy[i]=back[i];
    std::vector<float> zero(N,0.f); auto mm=ref_wqmm(x,w,sc,zero,M,K,N); // float x @ (w·scale)
    report("aclnnInplaceQuantMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ts);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tsum);
}
static void t_inplace_wqmm_rms(){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1); std::vector<int8_t> w(K*N); for(auto&v:w)v=(int8_t)(rand()%7-3);
    auto res=randv(M*N,-1,1),g=randv(N,0.5,1.5); std::vector<float> hy(M*N); double eps=1e-6;
    // weight-only with implicit scale=1 (degenerate inplace signature has no scale arg)
    DevBuf dx(M*(K>N?K:N)*4),dw(K*N),dr(M*N*4),dg(N*4),dsum(M*N*4);
    dx.up(x.data());dw.up(w.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*tr=mk({M,N},ACL_FLOAT,dr.p),*tg=mk({N},ACL_FLOAT,dg.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,tr,tg,eps,NO_COMM,tsum,ws,e);}, aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm);
    std::vector<float> back(M*(K>N?K:N)); dx.down(back.data()); for(int i=0;i<M*N;i++)hy[i]=back[i];
    std::vector<float> one(N,1.f),zero(N,0.f); auto mm=ref_wqmm(x,w,one,zero,M,K,N); // scale=1, offset=0
    report("aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm", norm_err(hy, ref_addrms(mm,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tsum);
}
template<typename GW,typename RUN>
static void t_combine_rms(const char*name,GW gws,RUN run){
    const int M=6,N=4; auto x=randv(M*N,-1,1),res=randv(M*N,-1,1),g=randv(N,0.5,1.5);
    std::vector<float> hy(M*N); double eps=1e-6;
    DevBuf dx(M*N*4),dr(M*N*4),dg(N*4),dy(M*N*4),dsum(M*N*4);
    dx.up(x.data());dr.up(res.data());dg.up(g.data());
    aclTensor *tx=mk({M,N},ACL_FLOAT,dx.p),*tr=mk({M,N},ACL_FLOAT,dr.p),*tg=mk({N},ACL_FLOAT,dg.p),*ty=mk({M,N},ACL_FLOAT,dy.p),*tsum=mk({M,N},ACL_FLOAT,dsum.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,tr,tg,eps,NO_COMM,ty,tsum,ws,e);}, run);
    dy.down(hy.data());
    std::vector<double> z(M*N,0.0); for(int i=0;i<M*N;i++)z[i]=x[i]; // temp = x (combine identity)
    report(name, norm_err(hy, ref_addrms(z,res,g,eps,M,N)), 1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
}

// ============ WeightQuantMatmulAllReduce (4-arg: x, w, antiquantScale, antiquantOffset) ============
static void t_wqmm(){
    const int M=6,K=5,N=4; auto x=randv(M*K,-1,1); std::vector<int8_t> w(K*N); for(auto&v:w)v=(int8_t)(rand()%63-31);
    auto sc=randv(N,0.01f,0.05f),of=randv(N,-2,2); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(K*N),ds(N*4),do_(N*4),dz(M*N*4); dx.up(x.data());dw.up(w.data());ds.up(sc.data());do_.up(of.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_INT8,dw.p),*ts=mk({N},ACL_FLOAT,ds.p),*to=mk({N},ACL_FLOAT,do_.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(tx,tw,ts,to,NO_COMM,tz,ws,e);}, aclnnWeightQuantMatmulAllReduce);
    dz.down(hz.data());
    report("aclnnWeightQuantMatmulAllReduce", norm_err(hz, ref_wqmm(x,w,sc,of,M,K,N)), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ts);aclDestroyTensor(to);aclDestroyTensor(tz);
}

// ============ GroupedMatMulAllReduce (groupList cumulative, shared weight [K,N]) ============
static void t_gmm(){
    const int M=6,K=5,N=4; int64_t gl[3]={2,5,6}; // cumulative row boundaries
    auto x=randv(M*K,-1,1),w=randv(K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(K*N*4),dz(M*N*4); dx.up(x.data());dw.up(w.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({K,N},ACL_FLOAT,dw.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclIntArray *ga=aclCreateIntArray(gl,3);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGroupedMatMulAllReduceGetWorkspaceSize(tx,tw,ga,NO_COMM,tz,ws,e);}, aclnnGroupedMatMulAllReduce);
    dz.down(hz.data());
    // shared weight => grouping is a no-op; equals full matmul
    report("aclnnGroupedMatMulAllReduce", norm_err(hz, ref_mm(x,w,nullptr,M,K,N)), 1e-5);
    aclDestroyIntArray(ga); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tz);
}

// ============ AllGatherAdd (x + bias) ============
static void t_allgather_add(){
    const int M=6,N=4; auto x=randv(M*N,-1,1),b=randv(N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*N*4),db(N*4),dz(M*N*4); dx.up(x.data());db.up(b.data());
    aclTensor *tx=mk({M,N},ACL_FLOAT,dx.p),*tb=mk({N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAllGatherAddGetWorkspaceSize(tx,tb,"",1,tz,ws,e);}, aclnnAllGatherAdd);
    dz.down(hz.data());
    std::vector<double> r(M*N); for(int m=0;m<M;m++)for(int n=0;n<N;n++)r[m*N+n]=(double)x[m*N+n]+b[n];
    report("aclnnAllGatherAdd", norm_err(hz,r), 1e-6);
    aclDestroyTensor(tx);aclDestroyTensor(tb);aclDestroyTensor(tz);
}

int main(){
    srand(1234); init();

    // 9 plain matmul-collective (MatmulAllReduce uses bias; rest no-bias to vary)
    t_mm("aclnnMatmulAllReduce",            aclnnMatmulAllReduceGetWorkspaceSize,            aclnnMatmulAllReduce,            true);
    t_mm("aclnnMatmulAllReduceV2",          aclnnMatmulAllReduceV2GetWorkspaceSize,          aclnnMatmulAllReduceV2,          true);
    // SKIP: single-rank reduce-scatter is degenerate (output is sharded across ranks); the CUDA backend's
    // collective path requires nranks>1, so it is not uniformly cross-checkable on one device. Built, not run.
    // t_mm("aclnnMatmulReduceScatter", ...); t_mm("aclnnMatmulReduceScatterV2", ...);
    t_mm("aclnnMatmulAllGather",            aclnnMatmulAllGatherGetWorkspaceSize,            aclnnMatmulAllGather,            false);
    t_mm("aclnnAllGatherMatmul",            aclnnAllGatherMatmulGetWorkspaceSize,            aclnnAllGatherMatmul,            true);
    t_mm("aclnnAllGatherMatmulV2",          aclnnAllGatherMatmulV2GetWorkspaceSize,          aclnnAllGatherMatmulV2,          true);
    t_bmm("aclnnAlltoAllAllGatherBatchMatMul",   aclnnAlltoAllAllGatherBatchMatMulGetWorkspaceSize,   aclnnAlltoAllAllGatherBatchMatMul);
    // t_bmm("aclnnBatchMatMulReduceScatterAlltoAll", ...);   // SKIP: single-rank reduce-scatter (see note above)

    // 5 quant matmul
    t_qmm("aclnnQuantMatmulAllReduce",   aclnnQuantMatmulAllReduceGetWorkspaceSize,   aclnnQuantMatmulAllReduce);
    t_qmm("aclnnQuantMatmulAllReduceV2", aclnnQuantMatmulAllReduceV2GetWorkspaceSize, aclnnQuantMatmulAllReduceV2);
    t_qmm("aclnnQuantMatmulAllReduceV3", aclnnQuantMatmulAllReduceV3GetWorkspaceSize, aclnnQuantMatmulAllReduceV3);
    t_qmm("aclnnQuantMatmulAllReduceV4", aclnnQuantMatmulAllReduceV4GetWorkspaceSize, aclnnQuantMatmulAllReduceV4);
    // t_qmm("aclnnQuantReduceScatter", ...);   // SKIP: single-rank reduce-scatter (see note above)

    // 1 weight-quant + 1 grouped + 1 allgather-add
    t_wqmm();
    t_gmm();
    t_allgather_add();

    // 1 quant-allreduce (identity copy, int8)
    t_copy("aclnnQuantAllReduce", aclnnQuantAllReduceGetWorkspaceSize, aclnnQuantAllReduce, ACL_INT8);

    // 12 MoE dispatch/combine + setup/teardown (identity copy, fp32)
    t_copy("aclnnMoeDistributeDispatch",         aclnnMoeDistributeDispatchGetWorkspaceSize,         aclnnMoeDistributeDispatch,         ACL_FLOAT);
    t_copy("aclnnMoeDistributeDispatchV2",       aclnnMoeDistributeDispatchV2GetWorkspaceSize,       aclnnMoeDistributeDispatchV2,       ACL_FLOAT);
    t_copy("aclnnMoeDistributeDispatchV3",       aclnnMoeDistributeDispatchV3GetWorkspaceSize,       aclnnMoeDistributeDispatchV3,       ACL_FLOAT);
    t_copy("aclnnMoeDistributeDispatchV4",       aclnnMoeDistributeDispatchV4GetWorkspaceSize,       aclnnMoeDistributeDispatchV4,       ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombine",          aclnnMoeDistributeCombineGetWorkspaceSize,          aclnnMoeDistributeCombine,          ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombineV2",        aclnnMoeDistributeCombineV2GetWorkspaceSize,        aclnnMoeDistributeCombineV2,        ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombineV3",        aclnnMoeDistributeCombineV3GetWorkspaceSize,        aclnnMoeDistributeCombineV3,        ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombineV4",        aclnnMoeDistributeCombineV4GetWorkspaceSize,        aclnnMoeDistributeCombineV4,        ACL_FLOAT);
    t_copy("aclnnMoeDistributeDispatchSetup",    aclnnMoeDistributeDispatchSetupGetWorkspaceSize,    aclnnMoeDistributeDispatchSetup,    ACL_FLOAT);
    t_copy("aclnnMoeDistributeDispatchTeardown", aclnnMoeDistributeDispatchTeardownGetWorkspaceSize, aclnnMoeDistributeDispatchTeardown, ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombineSetup",     aclnnMoeDistributeCombineSetupGetWorkspaceSize,     aclnnMoeDistributeCombineSetup,     ACL_FLOAT);
    t_copy("aclnnMoeDistributeCombineTeardown",  aclnnMoeDistributeCombineTeardownGetWorkspaceSize,  aclnnMoeDistributeCombineTeardown,  ACL_FLOAT);

    // 6 AddRmsNorm-fused tails
    t_mm_rms();
    t_qmm_rms();
    t_wqmm_rms();
    t_inplace_mm_rms();
    t_inplace_qmm_rms();
    t_inplace_wqmm_rms();

    // 2 MoE combine + AddRmsNorm
    t_combine_rms("aclnnMoeDistributeCombineAddRmsNorm",   aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize,   aclnnMoeDistributeCombineAddRmsNorm);
    t_combine_rms("aclnnMoeDistributeCombineAddRmsNormV2", aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize, aclnnMoeDistributeCombineAddRmsNormV2);

    return finish();
}
