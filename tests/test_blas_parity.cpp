// Matmul/BLAS/linalg GAP-op cross-check (CUDA parity): the ~22 GEMM-family + linalg ops added in
// metal/src/ops/blas_parity.mm. Self-contained — extern "C" prototypes match the canonical aclnnop header
// EXACTLY (no header include for these, so the test also pins the ABI). CPU triple-loop / known-identity
// references; linalg uses reconstruction invariants (Q@R==A, L@L^T==A). Tolerances 1e-4..1e-5 (fp32 GEMM
// accumulation and LAPACK round-trips); documented per case via the printed tol.
#include "harness.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;
typedef void *HcclComm;   // MC2 ops carry an HcclComm; nullptr on a single rank (matches aclnn_mc2.h)

// ---- Canonical prototypes (must match aclnnop/aclnn_ops.h exactly) ----
extern "C" {
// GroupedMatmul V2..V5 / WeightNz: x[M,K] by groupList @ weight[E,K,N] -> out[M,N]
#define GMM_P(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclIntArray*,aclTensor*,uint64_t*,aclOpExecutor**); \
aclnnStatus NAME(void*,uint64_t,aclOpExecutor*,aclrtStream);
GMM_P(aclnnGroupedMatmulV2) GMM_P(aclnnGroupedMatmulV3) GMM_P(aclnnGroupedMatmulV4)
GMM_P(aclnnGroupedMatmulV5) GMM_P(aclnnGroupedMatmulWeightNz)
GMM_P(aclnnGroupedMatmulFinalizeRouting) GMM_P(aclnnGroupedMatmulFinalizeRoutingV2)
GMM_P(aclnnGroupedMatmulFinalizeRoutingV3) GMM_P(aclnnGroupedMatmulFinalizeRoutingWeightNz)
GMM_P(aclnnGroupedMatmulFinalizeRoutingWeightNzV2)
#undef GMM_P

aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,const aclIntArray*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnGroupedMatmulAddV2(void*,uint64_t,aclOpExecutor*,aclrtStream);

aclnnStatus aclnnFusedMatmulGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnFusedMatmul(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnMatmulCompressGetWorkspaceSize(const aclTensor*,const aclTensor*,aclTensor*,int8_t,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMatmulCompress(void*,uint64_t,aclOpExecutor*,aclrtStream);

aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,double,double,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnAddmmWeightNz(void*,uint64_t,aclOpExecutor*,aclrtStream);

aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnTransMatmulWeight(void*,uint64_t,aclOpExecutor*,aclrtStream);

aclnnStatus aclnnCalculateMatmulWeightSizeGetWorkspaceSize(const aclIntArray*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnCalculateMatmulWeightSize(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnCalculateMatmulWeightSizeV2GetWorkspaceSize(const aclIntArray*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnCalculateMatmulWeightSizeV2(void*,uint64_t,aclOpExecutor*,aclrtStream);

aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnAlltoAllMatmul(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor*,const aclTensor*,const aclTensor*,HcclComm,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnMatmulAlltoAll(void*,uint64_t,aclOpExecutor*,aclrtStream);

// Linalg
aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor*,bool,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnLinalgCholesky(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor*,int64_t,aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnLinalgQr(void*,uint64_t,aclOpExecutor*,aclrtStream);
aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor*,const aclTensor*,int64_t,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnLinalgCross(void*,uint64_t,aclOpExecutor*,aclrtStream);

// aclCreateIntArray (provided by the backend; declared here for the self-contained build)
aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size);
aclnnStatus aclDestroyIntArray(const aclIntArray *array);
}

static double nerr(const std::vector<float>&g, const std::vector<double>&r){
    double me=0,mr=0; for(size_t i=0;i<r.size();i++){me=std::max(me,std::fabs(g[i]-r[i]));mr=std::max(mr,std::fabs(r[i]));} return me/(mr+1e-9);
}

// ---- CPU helpers ----
// grouped: x[M,K] partitioned by groups @ weight[E,K,N] -> ref[M,N]
static std::vector<double> ref_grouped(const std::vector<float>&x,const std::vector<float>&w,
        const std::vector<int64_t>&groups,int K,int N){
    int M=0; for(auto g:groups) M+=(int)g; std::vector<double> r((size_t)M*N,0);
    int off=0; for(size_t g=0;g<groups.size();g++){ int rows=(int)groups[g];
        for(int i=0;i<rows;i++)for(int n=0;n<N;n++){ double s=0; for(int k=0;k<K;k++) s+=(double)x[(size_t)(off+i)*K+k]*w[((size_t)g*K+k)*N+n]; r[(size_t)(off+i)*N+n]=s; }
        off+=rows; }
    return r;
}
static std::vector<double> ref_mm(const std::vector<float>&a,const std::vector<float>&b,int M,int K,int N){
    std::vector<double> r((size_t)M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)a[(size_t)m*K+k]*b[(size_t)k*N+n];r[(size_t)m*N+n]=s;}
    return r;
}

// ---- generic grouped-matmul test driver (for all V2..V5 / WeightNz / FinalizeRouting*) ----
template<typename GW, typename RUN>
static void t_gmm(const char*name, GW gws, RUN run){
    const int E=3,K=5,N=4; int64_t gl[E]={2,3,1}; int M=0; for(int i=0;i<E;i++)M+=gl[i];
    auto x=randv(M*K,-1,1), w=randv(E*K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(E*K*N*4),dz(M*N*4); dx.up(x.data()); dw.up(w.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({E,K,N},ACL_FLOAT,dw.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclIntArray *ga=aclCreateIntArray(gl,E);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(tx,tw,ga,tz,ws,e);}, run);
    dz.down(hz.data());
    auto ref=ref_grouped(x,w,std::vector<int64_t>(gl,gl+E),K,N);
    report(name, nerr(hz,ref), 1e-4);
    aclDestroyIntArray(ga); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tz);
}

static void t_gmm_add_v2(){
    const int E=3,K=5,N=4; int64_t gl[E]={2,3,1}; int M=0; for(int i=0;i<E;i++)M+=gl[i];
    auto x=randv(M*K,-1,1), w=randv(E*K*N,-1,1), y=randv(M*N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(E*K*N*4),dy(M*N*4),dz(M*N*4); dx.up(x.data()); dw.up(w.data()); dy.up(y.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({E,K,N},ACL_FLOAT,dw.p),*ty=mk({M,N},ACL_FLOAT,dy.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclIntArray *ga=aclCreateIntArray(gl,E);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return aclnnGroupedMatmulAddV2GetWorkspaceSize(tx,tw,ty,ga,tz,ws,e);}, aclnnGroupedMatmulAddV2);
    dz.down(hz.data());
    auto ref=ref_grouped(x,w,std::vector<int64_t>(gl,gl+E),K,N); for(size_t i=0;i<ref.size();i++) ref[i]+=y[i];
    report("GroupedMatmulAddV2", nerr(hz,ref), 1e-4);
    aclDestroyIntArray(ga); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ty);aclDestroyTensor(tz);
}

// FusedMatmul / MatmulCompress / AlltoAll: plain A[M,K]@B[K,N]
template<typename GW, typename RUN>
static void t_mm_cube(const char*name, GW gws, RUN run){
    const int M=6,K=5,N=4; auto a=randv(M*K,-1,1),b=randv(K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf da(M*K*4),db(K*N*4),dz(M*N*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(ta,tb,tz,(int8_t)1,ws,e);}, run);
    dz.down(hz.data()); report(name, nerr(hz,ref_mm(a,b,M,K,N)), 1e-5);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
template<typename GW, typename RUN>
static void t_mm_plain(const char*name, GW gws, RUN run){
    const int M=6,K=5,N=4; auto a=randv(M*K,-1,1),b=randv(K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf da(M*K*4),db(K*N*4),dz(M*N*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*ws,aclOpExecutor**e){return gws(ta,tb,nullptr,(HcclComm)nullptr,tz,ws,e);}, run);
    dz.down(hz.data()); report(name, nerr(hz,ref_mm(a,b,M,K,N)), 1e-5);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}

static void t_addmm_nz(){
    const int M=5,K=4,N=6; double beta=0.5,alpha=2.0; auto A=randv(M*K,-1,1),B=randv(K*N,-1,1),C=randv(M*N,-1,1);
    std::vector<float> hz(M*N); DevBuf da(M*K*4),db(K*N*4),dc(M*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data()); dc.up(C.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tc=mk({M,N},ACL_FLOAT,dc.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddmmWeightNzGetWorkspaceSize(tc,ta,tb,beta,alpha,tz,w,e);}, aclnnAddmmWeightNz);
    dz.down(hz.data()); auto ref=ref_mm(A,B,M,K,N); for(int i=0;i<M*N;i++) ref[i]=beta*C[i]+alpha*ref[i];
    report("AddmmWeightNz", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc);aclDestroyTensor(tz);
}

static void t_trans_weight(){
    const int K=4,N=6; auto W=randv(K*N,-1,1); std::vector<float> hz(K*N);
    DevBuf dw(K*N*4),dz(K*N*4); dw.up(W.data());
    aclTensor *tw=mk({K,N},ACL_FLOAT,dw.p),*tz=mk({K,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransMatmulWeightGetWorkspaceSize(tw,tz,w,e);}, aclnnTransMatmulWeight);
    dz.down(hz.data()); std::vector<double> ref(W.begin(),W.end());   // value-preserving reformat
    report("TransMatmulWeight", nerr(hz,ref), 1e-6); aclDestroyTensor(tw);aclDestroyTensor(tz);
}

static void t_calc_wsize(){
    int64_t shp[3]={3,5,4}; int64_t expect=3*5*4;
    { std::vector<int64_t> hz(1); DevBuf dz(8);
      aclTensor *tz=mk({1},ACL_INT64,dz.p); aclIntArray *sa=aclCreateIntArray(shp,3);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCalculateMatmulWeightSizeGetWorkspaceSize(sa,tz,w,e);}, aclnnCalculateMatmulWeightSize);
      dz.down(hz.data()); report("CalculateMatmulWeightSize", relerr((double)hz[0],(double)expect), 1e-9);
      aclDestroyIntArray(sa); aclDestroyTensor(tz); }
    { std::vector<int64_t> hz(1); DevBuf dz(8);
      aclTensor *tz=mk({1},ACL_INT64,dz.p); aclIntArray *sa=aclCreateIntArray(shp,3);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCalculateMatmulWeightSizeV2GetWorkspaceSize(sa,(int64_t)ACL_FLOAT,tz,w,e);}, aclnnCalculateMatmulWeightSizeV2);
      dz.down(hz.data()); report("CalculateMatmulWeightSizeV2", relerr((double)hz[0],(double)expect), 1e-9);
      aclDestroyIntArray(sa); aclDestroyTensor(tz); }
}

// ---- Linalg ----
static void t_cholesky(){
    const int n=5; std::vector<float> A(n*n);   // build SPD: A = B B^T + n*I
    auto B=randv(n*n,-1,1);
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){ double s=0; for(int k=0;k<n;k++) s+=(double)B[i*n+k]*B[j*n+k]; A[i*n+j]=(float)(s+(i==j?n:0)); }
    for(int up=0;up<2;up++){
        std::vector<float> hz(n*n); DevBuf da(n*n*4),dz(n*n*4); da.up(A.data());
        aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tz=mk({n,n},ACL_FLOAT,dz.p);
        bool upper=(up==1);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLinalgCholeskyGetWorkspaceSize(ta,upper,tz,w,e);}, aclnnLinalgCholesky);
        dz.down(hz.data());
        // reconstruct: lower -> L L^T; upper -> R^T R ; compare to A
        std::vector<double> rec((size_t)n*n,0);
        for(int i=0;i<n;i++)for(int j=0;j<n;j++){ double s=0;
            for(int k=0;k<n;k++){ double f=upper ? (double)hz[k*n+i]*hz[k*n+j] : (double)hz[i*n+k]*hz[j*n+k]; s+=f; } rec[i*n+j]=s; }
        std::vector<double> aref(A.begin(),A.end());
        report(upper?"LinalgCholesky(upper)":"LinalgCholesky(lower)", nerr(std::vector<float>(rec.begin(),rec.end()),aref), 1e-4);
        aclDestroyTensor(ta);aclDestroyTensor(tz);
    }
}

static void t_qr(){
    const int m=6,n=4,k=(m<n?m:n);   // reduced QR
    auto A=randv(m*n,-1,1);
    std::vector<float> hQ(m*k),hR(k*n); DevBuf da(m*n*4),dq(m*k*4),dr(k*n*4); da.up(A.data());
    aclTensor *ta=mk({m,n},ACL_FLOAT,da.p),*tq=mk({m,k},ACL_FLOAT,dq.p),*tr=mk({k,n},ACL_FLOAT,dr.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLinalgQrGetWorkspaceSize(ta,(int64_t)0,tq,tr,w,e);}, aclnnLinalgQr);
    dq.down(hQ.data()); dr.down(hR.data());
    // reconstruct Q@R == A
    std::vector<double> rec((size_t)m*n,0);
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double s=0; for(int l=0;l<k;l++) s+=(double)hQ[i*k+l]*hR[l*n+j]; rec[i*n+j]=s; }
    std::vector<double> aref(A.begin(),A.end());
    report("LinalgQr (Q@R==A)", nerr(std::vector<float>(rec.begin(),rec.end()),aref), 1e-4);
    // Q^T Q == I (orthonormal columns)
    double oe=0; for(int i=0;i<k;i++)for(int j=0;j<k;j++){ double s=0; for(int r=0;r<m;r++) s+=(double)hQ[r*k+i]*hQ[r*k+j]; oe=std::max(oe,std::fabs(s-(i==j?1.0:0.0))); }
    report("LinalgQr (Q ortho)", oe, 1e-4);
    aclDestroyTensor(ta);aclDestroyTensor(tq);aclDestroyTensor(tr);
}

static void t_cross(){
    const int Bn=4;   // [Bn,3], cross along last dim (dim=-1)
    auto A=randv(Bn*3,-1,1),B=randv(Bn*3,-1,1); std::vector<float> hz(Bn*3);
    DevBuf da(Bn*3*4),db(Bn*3*4),dz(Bn*3*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({Bn,3},ACL_FLOAT,da.p),*tb=mk({Bn,3},ACL_FLOAT,db.p),*tz=mk({Bn,3},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLinalgCrossGetWorkspaceSize(ta,tb,(int64_t)-1,tz,w,e);}, aclnnLinalgCross);
    dz.down(hz.data()); std::vector<double> ref(Bn*3);
    for(int i=0;i<Bn;i++){ const float*x=A.data()+i*3,*y=B.data()+i*3;
        ref[i*3]=(double)x[1]*y[2]-(double)x[2]*y[1]; ref[i*3+1]=(double)x[2]*y[0]-(double)x[0]*y[2]; ref[i*3+2]=(double)x[0]*y[1]-(double)x[1]*y[0]; }
    report("LinalgCross", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}

int main(){
    init(); srand(20260617);
    // grouped matmul family (10): V2..V5, WeightNz, FinalizeRouting + V2/V3/WeightNz/WeightNzV2
    t_gmm("GroupedMatmulV2", aclnnGroupedMatmulV2GetWorkspaceSize, aclnnGroupedMatmulV2);
    t_gmm("GroupedMatmulV3", aclnnGroupedMatmulV3GetWorkspaceSize, aclnnGroupedMatmulV3);
    t_gmm("GroupedMatmulV4", aclnnGroupedMatmulV4GetWorkspaceSize, aclnnGroupedMatmulV4);
    t_gmm("GroupedMatmulV5", aclnnGroupedMatmulV5GetWorkspaceSize, aclnnGroupedMatmulV5);
    t_gmm("GroupedMatmulWeightNz", aclnnGroupedMatmulWeightNzGetWorkspaceSize, aclnnGroupedMatmulWeightNz);
    t_gmm("GroupedMatmulFinalizeRouting", aclnnGroupedMatmulFinalizeRoutingGetWorkspaceSize, aclnnGroupedMatmulFinalizeRouting);
    t_gmm("GroupedMatmulFinalizeRoutingV2", aclnnGroupedMatmulFinalizeRoutingV2GetWorkspaceSize, aclnnGroupedMatmulFinalizeRoutingV2);
    t_gmm("GroupedMatmulFinalizeRoutingV3", aclnnGroupedMatmulFinalizeRoutingV3GetWorkspaceSize, aclnnGroupedMatmulFinalizeRoutingV3);
    t_gmm("GroupedMatmulFinalizeRoutingNz", aclnnGroupedMatmulFinalizeRoutingWeightNzGetWorkspaceSize, aclnnGroupedMatmulFinalizeRoutingWeightNz);
    t_gmm("GroupedMatmulFinalizeRoutingNzV2", aclnnGroupedMatmulFinalizeRoutingWeightNzV2GetWorkspaceSize, aclnnGroupedMatmulFinalizeRoutingWeightNzV2);
    // grouped matmul + residual (1)
    t_gmm_add_v2();
    // plain matmul family (4): FusedMatmul, MatmulCompress (cube), AlltoAllMatmul, MatmulAlltoAll (plain)
    t_mm_cube("FusedMatmul", aclnnFusedMatmulGetWorkspaceSize, aclnnFusedMatmul);
    t_mm_cube("MatmulCompress", aclnnMatmulCompressGetWorkspaceSize, aclnnMatmulCompress);
    t_mm_plain("AlltoAllMatmul", aclnnAlltoAllMatmulGetWorkspaceSize, aclnnAlltoAllMatmul);
    t_mm_plain("MatmulAlltoAll", aclnnMatmulAlltoAllGetWorkspaceSize, aclnnMatmulAlltoAll);
    // addmm-nz, trans-weight, calc-wsize x2 (4)
    t_addmm_nz();
    t_trans_weight();
    t_calc_wsize();
    // linalg (3 ops -> cholesky x2 + qr x2 + cross)
    t_cholesky();
    t_qr();
    t_cross();
    return finish();
}
