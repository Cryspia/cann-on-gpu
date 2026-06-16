// BLAS extras (P8) cross-check: Mm/Bmm/Mv/Addmv/Addmm/Baddbmm/Addbmm/Dot/Outer/Kron. CPU double reference, normalized error.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static double nerr(const std::vector<float>&g, const std::vector<double>&r){ double me=0,mr=0; for(size_t i=0;i<r.size();i++){me=std::max(me,std::fabs(g[i]-r[i]));mr=std::max(mr,std::fabs(r[i]));} return me/(mr+1e-9); }

static void t_mm() {
    const int M=8,K=6,N=5; auto A=randv(M*K,-1,1),B=randv(K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf da(M*K*4),db(K*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMmGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnMm);
    dz.down(hz.data()); std::vector<double> ref(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*B[k*N+n];ref[m*N+n]=s;}
    report("Mm", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_bmm() {
    const int Bt=3,M=4,K=5,N=6; auto A=randv(Bt*M*K,-1,1),B=randv(Bt*K*N,-1,1); std::vector<float> hz(Bt*M*N);
    DevBuf da(Bt*M*K*4),db(Bt*K*N*4),dz(Bt*M*N*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({Bt,M,K},ACL_FLOAT,da.p),*tb=mk({Bt,K,N},ACL_FLOAT,db.p),*tz=mk({Bt,M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBmmGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnBmm);
    dz.down(hz.data()); std::vector<double> ref(Bt*M*N,0);
    for(int b=0;b<Bt;b++)for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[(b*M+m)*K+k]*B[(b*K+k)*N+n];ref[(b*M+m)*N+n]=s;}
    report("Bmm", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_mv_addmv() {
    const int M=6,K=5; auto A=randv(M*K,-1,1),x=randv(K,-1,1),y=randv(M,-1,1);
    { std::vector<float> hz(M); DevBuf da(M*K*4),dx(K*4),dz(M*4); da.up(A.data()); dx.up(x.data());
      aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tx=mk({K},ACL_FLOAT,dx.p),*tz=mk({M},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMvGetWorkspaceSize(ta,tx,tz,w,e);}, aclnnMv);
      dz.down(hz.data()); std::vector<double> ref(M,0); for(int m=0;m<M;m++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*x[k];ref[m]=s;}
      report("Mv", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { double beta=0.5,alpha=2.0; std::vector<float> hz(M); DevBuf da(M*K*4),dx(K*4),dy(M*4),dz(M*4); da.up(A.data()); dx.up(x.data()); dy.up(y.data());
      aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tx=mk({K},ACL_FLOAT,dx.p),*ty=mk({M},ACL_FLOAT,dy.p),*tz=mk({M},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddmvGetWorkspaceSize(ty,ta,tx,beta,alpha,tz,w,e);}, aclnnAddmv);
      dz.down(hz.data()); std::vector<double> ref(M,0); for(int m=0;m<M;m++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*x[k];ref[m]=beta*y[m]+alpha*s;}
      report("Addmv", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(tz); }
}
static void t_addmm() {
    const int M=5,K=4,N=6; double beta=0.5,alpha=2.0; auto A=randv(M*K,-1,1),B=randv(K*N,-1,1),C=randv(M*N,-1,1);
    std::vector<float> hz(M*N); DevBuf da(M*K*4),db(K*N*4),dc(M*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data()); dc.up(C.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tc=mk({M,N},ACL_FLOAT,dc.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddmmGetWorkspaceSize(tc,ta,tb,beta,alpha,tz,w,e);}, aclnnAddmm);
    dz.down(hz.data()); std::vector<double> ref(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*B[k*N+n];ref[m*N+n]=beta*C[m*N+n]+alpha*s;}
    report("Addmm", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc);aclDestroyTensor(tz);
}
static void t_baddbmm_addbmm() {
    const int Bt=2,M=3,K=4,N=5; double beta=0.5,alpha=1.5; auto A=randv(Bt*M*K,-1,1),B=randv(Bt*K*N,-1,1);
    { auto C=randv(Bt*M*N,-1,1); std::vector<float> hz(Bt*M*N); DevBuf da(Bt*M*K*4),db(Bt*K*N*4),dc(Bt*M*N*4),dz(Bt*M*N*4); da.up(A.data()); db.up(B.data()); dc.up(C.data());
      aclTensor *ta=mk({Bt,M,K},ACL_FLOAT,da.p),*tb=mk({Bt,K,N},ACL_FLOAT,db.p),*tc=mk({Bt,M,N},ACL_FLOAT,dc.p),*tz=mk({Bt,M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBaddbmmGetWorkspaceSize(tc,ta,tb,beta,alpha,tz,w,e);}, aclnnBaddbmm);
      dz.down(hz.data()); std::vector<double> ref(Bt*M*N,0);
      for(int b=0;b<Bt;b++)for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[(b*M+m)*K+k]*B[(b*K+k)*N+n];ref[(b*M+m)*N+n]=beta*C[(b*M+m)*N+n]+alpha*s;}
      report("Baddbmm", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc);aclDestroyTensor(tz); }
    { auto C=randv(M*N,-1,1); std::vector<float> hz(M*N); DevBuf da(Bt*M*K*4),db(Bt*K*N*4),dc(M*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data()); dc.up(C.data());
      aclTensor *ta=mk({Bt,M,K},ACL_FLOAT,da.p),*tb=mk({Bt,K,N},ACL_FLOAT,db.p),*tc=mk({M,N},ACL_FLOAT,dc.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddbmmGetWorkspaceSize(tc,ta,tb,beta,alpha,tz,w,e);}, aclnnAddbmm);
      dz.down(hz.data()); std::vector<double> ref(M*N,0);
      for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int b=0;b<Bt;b++)for(int k=0;k<K;k++)s+=(double)A[(b*M+m)*K+k]*B[(b*K+k)*N+n];ref[m*N+n]=beta*C[m*N+n]+alpha*s;}
      report("Addbmm", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc);aclDestroyTensor(tz); }
}
static void t_dot_outer_kron() {
    { const int n=64; auto a=randv(n,-1,1),b=randv(n,-1,1); std::vector<float> hz(1); DevBuf da(n*4),db(n*4),dz(4); da.up(a.data()); db.up(b.data());
      aclTensor *ta=mk({n},ACL_FLOAT,da.p),*tb=mk({n},ACL_FLOAT,db.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDotGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnDot);
      dz.down(hz.data()); double ref=0; for(int i=0;i<n;i++) ref+=(double)a[i]*b[i];
      report("Dot", std::fabs(hz[0]-ref)/(std::fabs(ref)+1e-9), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { const int M=6,N=7; auto a=randv(M,-1,1),b=randv(N,-1,1); std::vector<float> hz(M*N); DevBuf da(M*4),db(N*4),dz(M*N*4); da.up(a.data()); db.up(b.data());
      aclTensor *ta=mk({M},ACL_FLOAT,da.p),*tb=mk({N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnOuterGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnOuter);
      dz.down(hz.data()); std::vector<double> ref(M*N); for(int m=0;m<M;m++)for(int n=0;n<N;n++)ref[m*N+n]=(double)a[m]*b[n];
      report("Outer", nerr(hz,ref), 1e-6); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { const int M=2,N=3,P=2,Q=2; auto A=randv(M*N,-1,1),B=randv(P*Q,-1,1); std::vector<float> hz(M*P*N*Q);
      DevBuf da(M*N*4),db(P*Q*4),dz(M*P*N*Q*4); da.up(A.data()); db.up(B.data());
      aclTensor *ta=mk({M,N},ACL_FLOAT,da.p),*tb=mk({P,Q},ACL_FLOAT,db.p),*tz=mk({M*P,N*Q},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnKronGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnKron);
      dz.down(hz.data()); std::vector<double> ref(M*P*N*Q);
      for(int m=0;m<M;m++)for(int p=0;p<P;p++)for(int n=0;n<N;n++)for(int q=0;q<Q;q++) ref[(m*P+p)*(N*Q)+(n*Q+q)]=(double)A[m*N+n]*B[p*Q+q];
      report("Kron", nerr(hz,ref), 1e-6); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
}
static void t_gemm() {
    // out = alpha·Aᵀ·B + beta·C with transA=1: A[K,M], B[K,N], C/out[M,N]
    const int M=7,K=5,N=6; float alpha=1.5f, beta=0.5f;
    auto A=randv(K*M,-1,1),B=randv(K*N,-1,1),C=randv(M*N,-1,1); std::vector<float> hz(M*N);
    DevBuf da(K*M*4),db(K*N*4),dc(M*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data()); dc.up(C.data());
    aclTensor *ta=mk({K,M},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tc=mk({M,N},ACL_FLOAT,dc.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGemmGetWorkspaceSize(ta,tb,tc,alpha,beta,1,0,tz,1,w,e);}, aclnnGemm);
    dz.down(hz.data()); std::vector<double> ref(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[k*M+m]*B[k*N+n];ref[m*N+n]=alpha*s+beta*C[m*N+n];}
    report("Gemm(transA,alpha,beta)", nerr(hz,ref), 1e-5);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc);aclDestroyTensor(tz);
    // transB=1 path, beta=0 (no C)
    auto A2=randv(M*K,-1,1),B2=randv(N*K,-1,1); std::vector<float> hz2(M*N);
    DevBuf da2(M*K*4),db2(N*K*4),dz2(M*N*4); da2.up(A2.data()); db2.up(B2.data());
    aclTensor *ta2=mk({M,K},ACL_FLOAT,da2.p),*tb2=mk({N,K},ACL_FLOAT,db2.p),*tz2=mk({M,N},ACL_FLOAT,dz2.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGemmGetWorkspaceSize(ta2,tb2,nullptr,1.0f,0.0f,0,1,tz2,1,w,e);}, aclnnGemm);
    dz2.down(hz2.data()); std::vector<double> ref2(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A2[m*K+k]*B2[n*K+k];ref2[m*N+n]=s;}
    report("Gemm(transB)", nerr(hz2,ref2), 1e-5);
    aclDestroyTensor(ta2);aclDestroyTensor(tb2);aclDestroyTensor(tz2);
}
static void t_gmm_add() {
    // x[M,K] grouped by counts @ weight[E,K,N] + y[M,N]
    const int K=6,N=5; std::vector<int64_t> counts={3,2,4}; int E=counts.size(),M=0; for(auto c:counts)M+=c;
    auto X=randv(M*K,-1,1),W=randv(E*K*N,-1,1),Y=randv(M*N,-1,1); std::vector<float> hz(M*N);
    DevBuf dx(M*K*4),dw(E*K*N*4),dy(M*N*4),dz(M*N*4); dx.up(X.data()); dw.up(W.data()); dy.up(Y.data());
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p),*tw=mk({E,K,N},ACL_FLOAT,dw.p),*ty=mk({M,N},ACL_FLOAT,dy.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclIntArray *gl=aclCreateIntArray(counts.data(),E);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupedMatmulAddGetWorkspaceSize(tx,tw,ty,gl,tz,w,e);}, aclnnGroupedMatmulAdd);
    dz.down(hz.data()); std::vector<double> ref(M*N,0); int64_t off=0;
    for(int e=0;e<E;e++){ for(int i=0;i<counts[e];i++)for(int n=0;n<N;n++){ double acc=0; for(int k=0;k<K;k++)acc+=(double)X[(off+i)*K+k]*W[(e*K+k)*N+n];
        ref[(off+i)*N+n]=acc+Y[(off+i)*N+n]; } off+=counts[e]; }
    report("GroupedMatmulAdd", nerr(hz,ref), 1e-5);
    aclDestroyIntArray(gl); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ty);aclDestroyTensor(tz);
}
static void t_matmul_nz() {
    // NZ = logical-equivalence: MatmulWeightNz treats weight row-major, must equal x@W
    const int M=8,K=6,N=5; auto A=randv(M*K,-1,1),B=randv(K*N,-1,1); std::vector<float> hz(M*N);
    DevBuf da(M*K*4),db(K*N*4),dz(M*N*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({M,K},ACL_FLOAT,da.p),*tb=mk({K,N},ACL_FLOAT,db.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatmulWeightNzGetWorkspaceSize(ta,tb,tz,1,w,e);}, aclnnMatmulWeightNz);
    dz.down(hz.data()); std::vector<double> ref(M*N,0);
    for(int m=0;m<M;m++)for(int n=0;n<N;n++){double s=0;for(int k=0;k<K;k++)s+=(double)A[m*K+k]*B[k*N+n];ref[m*N+n]=s;}
    report("MatmulWeightNz", nerr(hz,ref), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_addr() {
    const int M=6,N=5; float beta=0.5f,alpha=2.0f; auto self=randv(M*N,-1,1),v1=randv(M,-1,1),v2=randv(N,-1,1); std::vector<float> hz(M*N);
    DevBuf ds(M*N*4),d1(M*4),d2(N*4),dz(M*N*4); ds.up(self.data()); d1.up(v1.data()); d2.up(v2.data());
    aclTensor *ts=mk({M,N},ACL_FLOAT,ds.p),*t1=mk({M},ACL_FLOAT,d1.p),*t2=mk({N},ACL_FLOAT,d2.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
    aclScalar *sb=aclCreateScalar(&beta,ACL_FLOAT),*sa=aclCreateScalar(&alpha,ACL_FLOAT);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddrGetWorkspaceSize(ts,t1,t2,sb,sa,tz,w,e);}, aclnnAddr);
    dz.down(hz.data()); std::vector<double> ref(M*N); for(int i=0;i<M;i++)for(int j=0;j<N;j++) ref[i*N+j]=beta*self[i*N+j]+alpha*(double)v1[i]*v2[j];
    report("Addr", nerr(hz,ref), 1e-5); aclDestroyTensor(ts);aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tz);aclDestroyScalar(sb);aclDestroyScalar(sa);
}
static void t_divmods() {
    const int n=64; auto x=randv(n,-10,10); float other=3.0f; std::vector<float> hz(n);
    DevBuf dx(n*4),dz(n*4); dx.up(x.data());
    aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p); aclScalar *so=aclCreateScalar(&other,ACL_FLOAT);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDivModsGetWorkspaceSize(tx,so,0,tz,w,e);}, aclnnDivMods);   // floor mode
    dz.down(hz.data()); double me=0; for(int i=0;i<n;i++) me=std::max(me,(double)std::fabs(hz[i]-std::floor(x[i]/other)));
    report("DivMods floor", me, 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyScalar(so);
}
int main() {
    init(); srand(19);
    t_mm(); t_bmm(); t_mv_addmv(); t_addmm(); t_baddbmm_addbmm(); t_dot_outer_kron(); t_gemm(); t_gmm_add(); t_matmul_nz(); t_addr(); t_divmods();
    { const int n=64; auto x=randv(n,-9,9); float c=2.5f; std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p); aclScalar *sc=aclCreateScalar(&c,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFmodScalarGetWorkspaceSize(tx,sc,tz,w,e);}, aclnnFmodScalar);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++) me=std::max(me,(double)std::fabs(hz[i]-std::fmod(x[i],c))); report("FmodScalar",me,1e-5);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRemainderTensorScalarGetWorkspaceSize(tx,sc,tz,w,e);}, aclnnRemainderTensorScalar);
      dz.down(hz.data()); double mr=0; for(int i=0;i<n;i++) mr=std::max(mr,(double)std::fabs(hz[i]-(x[i]-std::floor(x[i]/c)*c))); report("RemainderTensorScalar",mr,1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyScalar(sc); }
    { // AddRelu
      const int n=40; auto a=randv(n,-2,2),b=randv(n,-2,2); std::vector<float> hz(n); DevBuf da(n*4),db(n*4),dz(n*4); da.up(a.data()); db.up(b.data());
      aclTensor *ta=mk({n},ACL_FLOAT,da.p),*tb=mk({n},ACL_FLOAT,db.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAddReluGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnAddRelu);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++){double r=a[i]+b[i]; if(r<0)r=0; me=std::max(me,(double)std::fabs(hz[i]-r));} report("AddRelu",me,1e-5);
      aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { // Histc: 0..10 values, 5 bins over [0,10]
      const int n=1000; std::vector<float> x(n); for(int i=0;i<n;i++) x[i]=(float)(i%10); int bins=5; float lo=0,hi=10;
      std::vector<float> hz(bins); DevBuf dx(n*4),dz(bins*4); dx.up(x.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({bins},ACL_FLOAT,dz.p); aclScalar *sl=aclCreateScalar(&lo,ACL_FLOAT),*sh=aclCreateScalar(&hi,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnHistcGetWorkspaceSize(tx,bins,sl,sh,tz,w,e);}, aclnnHistc);
      dz.down(hz.data()); double tot=0; for(int b=0;b<bins;b++) tot+=hz[b]; report("Histc total", std::fabs(tot-n)/n, 1e-6);
      aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyScalar(sl);aclDestroyScalar(sh); }
    { // ScatterValue: [4,3], dim=1, index picks col per row, set to 9
      const int R=4,C=3; std::vector<float> self(R*C,1.0f); std::vector<int64_t> idx={0,2,1,0}; float val=9;
      std::vector<float> hz(R*C); DevBuf ds(R*C*4),di(R*8),dz(R*C*4); ds.up(self.data()); di.up(idx.data());
      aclTensor *ts=mk({R,C},ACL_FLOAT,ds.p),*ti=mk({R,1},ACL_INT64,di.p),*tz=mk({R,C},ACL_FLOAT,dz.p); aclScalar *sv=aclCreateScalar(&val,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterValueGetWorkspaceSize(ts,1,ti,sv,tz,w,e);}, aclnnScatterValue);
      dz.down(hz.data()); int bad=0; for(int r=0;r<R;r++)for(int c=0;c<C;c++){ float exp=(c==idx[r])?9.0f:1.0f; if(std::fabs(hz[r*C+c]-exp)>1e-6)bad++; }
      report("ScatterValue", bad, 0.0); aclDestroyTensor(ts);aclDestroyTensor(ti);aclDestroyTensor(tz);aclDestroyScalar(sv); }
    { // DivMod tensor/tensor floor
      const int n=50; auto a=randv(n,-8,8),b=randv(n,1,4); std::vector<float> hz(n); DevBuf da(n*4),db(n*4),dz(n*4); da.up(a.data()); db.up(b.data());
      aclTensor *ta=mk({n},ACL_FLOAT,da.p),*tb=mk({n},ACL_FLOAT,db.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDivModGetWorkspaceSize(ta,tb,0,tz,w,e);}, aclnnDivMod);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++) me=std::max(me,(double)std::fabs(hz[i]-std::floor(a[i]/b[i]))); report("DivMod floor",me,1e-5);
      aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { // Put: out=self, set flat indices to source
      const int n=12; std::vector<float> self(n,1.0f),src={5,6,7}; std::vector<int64_t> idx={2,7,11}; std::vector<float> hz(n);
      DevBuf ds(n*4),di(3*8),dsrc(3*4),dz(n*4); ds.up(self.data()); di.up(idx.data()); dsrc.up(src.data());
      aclTensor *ts=mk({n},ACL_FLOAT,ds.p),*ti=mk({3},ACL_INT64,di.p),*tsr=mk({3},ACL_FLOAT,dsrc.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPutGetWorkspaceSize(ts,ti,tsr,false,tz,w,e);}, aclnnPut);
      dz.down(hz.data()); int bad=0; for(int i=0;i<n;i++){ float exp=1.0f; for(int k=0;k<3;k++) if(idx[k]==i) exp=src[k]; if(std::fabs(hz[i]-exp)>1e-6)bad++; }
      report("Put",bad,0.0); aclDestroyTensor(ts);aclDestroyTensor(ti);aclDestroyTensor(tsr);aclDestroyTensor(tz); }
    { // FakeQuantPerTensorAffineCachemask: scale=0.1, zp=0, [-128,127]
      const int n=50; auto x=randv(n,-5,5); float scale=0.1f; int zp=0,qmin=-128,qmax=127; std::vector<float> hz(n);
      DevBuf dx(n*4),dz(n*4),dm(n); dx.up(x.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p),*tm=mk({n},ACL_UINT8,dm.p);
      float zpf=zp; aclScalar *ss=aclCreateScalar(&scale,ACL_FLOAT),*sz=aclCreateScalar(&zpf,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(tx,ss,sz,qmin,qmax,tz,tm,w,e);}, aclnnFakeQuantPerTensorAffineCachemask);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++){ int q=(int)std::lround(x[i]/scale)+zp; int qc=q<qmin?qmin:(q>qmax?qmax:q); double ref=(qc-zp)*scale; me=std::max(me,std::fabs(hz[i]-ref)); }
      report("FakeQuantPerTensorCachemask", me, 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyTensor(tm);aclDestroyScalar(ss);aclDestroyScalar(sz); }
    { // DynamicBlockQuant: 4 blocks of 16
      const int blk=16, nbk=4, n=blk*nbk; auto x=randv(n,-3,3); std::vector<int8_t> q(n); std::vector<float> sc(nbk);
      DevBuf dx(n*4),dq(n),ds(nbk*4); dx.up(x.data());
      aclTensor *tx=mk({nbk,blk},ACL_FLOAT,dx.p),*tq=mk({nbk,blk},ACL_INT8,dq.p),*ts=mk({nbk},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDynamicBlockQuantGetWorkspaceSize(tx,blk,tq,ts,w,e);}, aclnnDynamicBlockQuant);
      dq.down(q.data()); ds.down(sc.data()); double me=0,mr=0; for(int b=0;b<nbk;b++)for(int i=0;i<blk;i++){ double deq=(double)q[b*blk+i]*sc[b]; me=std::max(me,std::fabs(deq-x[b*blk+i])); mr=std::max(mr,std::fabs((double)x[b*blk+i])); }
      report("DynamicBlockQuant", me/(mr+1e-9), 1e-2); aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(ts); }
    { // UpsampleNearest1d: [1,2,4] -> [1,2,8]
      const int N=1,C=2,L=4,Lo=8; auto x=randv(N*C*L,-1,1); std::vector<float> hz(N*C*Lo);
      DevBuf dx(N*C*L*4),dz(N*C*Lo*4); dx.up(x.data());
      aclTensor *tx=mk({N,C,L},ACL_FLOAT,dx.p),*tz=mk({N,C,Lo},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleNearest1dGetWorkspaceSize(tx,tz,w,e);}, aclnnUpsampleNearest1d);
      dz.down(hz.data()); int bad=0; for(int nc=0;nc<N*C;nc++)for(int o=0;o<Lo;o++){ int sidx=(int)((float)o*L/Lo); if(std::fabs(hz[nc*Lo+o]-x[nc*L+sidx])>1e-6)bad++; }
      report("UpsampleNearest1d",bad,0.0); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { // GlobalAveragePool: [2,3,4,4] -> [2,3,1,1]
      const int N=2,C=3,H=4,W=4; auto x=randv(N*C*H*W,-2,2); std::vector<float> hz(N*C);
      DevBuf dx(N*C*H*W*4),dz(N*C*4); dx.up(x.data());
      aclTensor *tx=mk({N,C,H,W},ACL_FLOAT,dx.p),*tz=mk({N,C,1,1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGlobalAveragePoolGetWorkspaceSize(tx,tz,w,e);}, aclnnGlobalAveragePool);
      dz.down(hz.data()); double me=0; for(int nc=0;nc<N*C;nc++){ double m=0; for(int i=0;i<H*W;i++) m+=x[nc*H*W+i]; m/=H*W; me=std::max(me,std::fabs(hz[nc]-m)); }
      report("GlobalAveragePool",me,1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { // UpsampleNearest1dBackward: [1,2,8]->[1,2,4]; gradIn[i]=sum gradOut[o] where src(o)==i
      const int N=1,C=2,L=4,Lo=8; auto go=randv(N*C*Lo,-1,1); std::vector<float> hz(N*C*L);
      DevBuf dgo(N*C*Lo*4),dgi(N*C*L*4); dgo.up(go.data());
      aclTensor *tgo=mk({N,C,Lo},ACL_FLOAT,dgo.p),*tgi=mk({N,C,L},ACL_FLOAT,dgi.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleNearest1dBackwardGetWorkspaceSize(tgo,tgi,w,e);}, aclnnUpsampleNearest1dBackward);
      dgi.down(hz.data()); std::vector<double> ref(N*C*L,0); for(int nc=0;nc<N*C;nc++)for(int o=0;o<Lo;o++){ int sidx=(int)((float)o*L/Lo); ref[nc*L+sidx]+=go[nc*Lo+o]; }
      double me=0,mr=0; for(int i=0;i<N*C*L;i++){me=std::max(me,std::fabs(hz[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));} report("UpsampleNearest1dBackward",me/(mr+1e-9),1e-5);
      aclDestroyTensor(tgo);aclDestroyTensor(tgi); }
    { // ReplicationPad1dBackward: gradOut[1,1,8] pad [2,2] -> gradIn[1,1,4]; gradIn[clamp(o-2)] += gradOut[o]
      const int L=4,Lo=8,lp=2; auto go=randv(Lo,-1,1); std::vector<float> hz(L);
      DevBuf dgo(Lo*4),dgi(L*4); dgo.up(go.data());
      aclTensor *tgo=mk({1,1,Lo},ACL_FLOAT,dgo.p),*tself=mk({1,1,L},ACL_FLOAT,dgi.p),*tgi=mk({1,1,L},ACL_FLOAT,dgi.p);
      std::vector<int64_t> pad={lp,lp}; aclIntArray *pa=aclCreateIntArray(pad.data(),2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReplicationPad1dBackwardGetWorkspaceSize(tgo,tself,pa,tgi,w,e);}, aclnnReplicationPad1dBackward);
      dgi.down(hz.data()); std::vector<double> ref(L,0); for(int o=0;o<Lo;o++){ int j=o-lp; j=j<0?0:(j>=L?L-1:j); ref[j]+=go[o]; }
      double me=0,mr=0; for(int i=0;i<L;i++){me=std::max(me,std::fabs(hz[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));} report("ReplicationPad1dBackward",me/(mr+1e-9),1e-5);
      aclDestroyIntArray(pa);aclDestroyTensor(tgo);aclDestroyTensor(tself);aclDestroyTensor(tgi); }
    { // MaxPool2dWithIndicesBackward: gradOut[1,1,2,2], indices→flat into 4x4, scatter-add into gradIn[1,1,4,4]
      const int H=4,W=4,oH=2,oW=2; auto go=randv(oH*oW,-1,1); std::vector<int64_t> ind={0,2,8,10}; std::vector<float> hz(H*W);
      DevBuf dgo(oH*oW*4),di(oH*oW*8),dgi(H*W*4),dself(H*W*4); dgo.up(go.data()); di.up(ind.data());
      aclTensor *tgo=mk({1,1,oH,oW},ACL_FLOAT,dgo.p),*tself=mk({1,1,H,W},ACL_FLOAT,dself.p),*ti=mk({1,1,oH,oW},ACL_INT64,di.p),*tgi=mk({1,1,H,W},ACL_FLOAT,dgi.p);
      std::vector<int64_t> kk={2,2}; aclIntArray *ak=aclCreateIntArray(kk.data(),2),*as=aclCreateIntArray(kk.data(),2); std::vector<int64_t> pp={0,0}; aclIntArray *ap=aclCreateIntArray(pp.data(),2),*ad=aclCreateIntArray(kk.data(),2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(tgo,tself,ti,ak,as,ap,ad,false,tgi,w,e);}, aclnnMaxPool2dWithIndicesBackward);
      dgi.down(hz.data()); std::vector<double> ref(H*W,0); for(int o=0;o<oH*oW;o++) ref[ind[o]]+=go[o];
      double me=0,mr=0; for(int i=0;i<H*W;i++){me=std::max(me,std::fabs(hz[i]-ref[i]));mr=std::max(mr,std::fabs(ref[i]));} report("MaxPool2dWithIndicesBackward",me/(mr+1e-9),1e-5);
      aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);aclDestroyTensor(tgo);aclDestroyTensor(tself);aclDestroyTensor(ti);aclDestroyTensor(tgi); }
    { // simple math: Gcd, Logit, FmodTensor, Ger
      std::vector<int32_t> a={12,15,7,100},b={8,5,3,75}; std::vector<int32_t> hz(4); DevBuf da(16),db(16),dz(16); da.up(a.data()); db.up(b.data());
      aclTensor *ta=mk({4},ACL_INT32,da.p),*tb=mk({4},ACL_INT32,db.p),*tz=mk({4},ACL_INT32,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGcdGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnGcd);
      dz.down(hz.data()); int bad=(hz[0]!=4)+(hz[1]!=5)+(hz[2]!=1)+(hz[3]!=25); report("Gcd",bad,0.0);
      aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    { const int n=32; auto x=randv(n,0.1,0.9); std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogitGetWorkspaceSize(tx,1e-6,tz,w,e);}, aclnnLogit);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++) me=std::max(me,(double)std::fabs(hz[i]-std::log(x[i]/(1-x[i])))); report("Logit",me,1e-4);
      aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { const int M=5,N=4; auto v1=randv(M,-1,1),v2=randv(N,-1,1); std::vector<float> hz(M*N); DevBuf d1(M*4),d2(N*4),dz(M*N*4); d1.up(v1.data()); d2.up(v2.data());
      aclTensor *t1=mk({M},ACL_FLOAT,d1.p),*t2=mk({N},ACL_FLOAT,d2.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGerGetWorkspaceSize(t1,t2,tz,w,e);}, aclnnGer);
      dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<M;i++)for(int j=0;j<N;j++){double r=(double)v1[i]*v2[j];me=std::max(me,std::fabs(hz[i*N+j]-r));mr=std::max(mr,std::fabs(r));} report("Ger",me/(mr+1e-9),1e-5);
      aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tz); }
    { // Shrink (lambd=0.5,bias=0), Cdist (p=2)
      const int n=20; auto x=randv(n,-2,2); std::vector<float> hz(n); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      aclTensor *tx=mk({n},ACL_FLOAT,dx.p),*tz=mk({n},ACL_FLOAT,dz.p); float L=0.5f,B=0.f; aclScalar *sl=aclCreateScalar(&L,ACL_FLOAT),*sb=aclCreateScalar(&B,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnShrinkGetWorkspaceSize(tx,sl,sb,tz,w,e);}, aclnnShrink);
      dz.down(hz.data()); double me=0; for(int i=0;i<n;i++){double r=x[i]>0.5?x[i]:(x[i]<-0.5?x[i]:0); me=std::max(me,(double)std::fabs(hz[i]-r));} report("Shrink",me,1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tz);aclDestroyScalar(sl);aclDestroyScalar(sb); }
    { const int P=3,R=4,M=5; auto x1=randv(P*M,-1,1),x2=randv(R*M,-1,1); std::vector<float> hz(P*R); DevBuf d1(P*M*4),d2(R*M*4),dz(P*R*4); d1.up(x1.data()); d2.up(x2.data());
      aclTensor *t1=mk({P,M},ACL_FLOAT,d1.p),*t2=mk({R,M},ACL_FLOAT,d2.p),*tz=mk({P,R},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCdistGetWorkspaceSize(t1,t2,2.0,tz,w,e);}, aclnnCdist);
      dz.down(hz.data()); double me=0,mr=0; for(int p=0;p<P;p++)for(int r=0;r<R;r++){double s=0;for(int k=0;k<M;k++){double d=x1[p*M+k]-x2[r*M+k];s+=d*d;} double ref=std::sqrt(s); me=std::max(me,std::fabs(hz[p*R+r]-ref));mr=std::max(mr,ref);} report("Cdist p=2",me/(mr+1e-9),1e-5);
      aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(tz); }
    return finish();
}
