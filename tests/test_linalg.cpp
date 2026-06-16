// Linear algebra (P9) cross-check: Inverse (A@inv==I), Det (vs CPU cofactor n=3), Cholesky (L@L^T==A), Cross.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_inverse() {
    const int n=5; std::vector<float> A(n*n);
    for(int i=0;i<n*n;i++) A[i]=(rand()/(float)RAND_MAX)-0.5f;
    for(int i=0;i<n;i++) A[i*n+i]+=(float)n;   // diagonal dominance -> invertible
    std::vector<float> hinv(n*n); DevBuf da(n*n*4),di(n*n*4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*ti=mk({n,n},ACL_FLOAT,di.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInverseGetWorkspaceSize(ta,ti,w,e);}, aclnnInverse);
    di.down(hinv.data());
    double me=0; for(int i=0;i<n;i++)for(int j=0;j<n;j++){ double s=0; for(int k=0;k<n;k++) s+=(double)A[i*n+k]*hinv[k*n+j]; double ref=(i==j)?1.0:0.0; me=std::max(me,std::fabs(s-ref)); }
    report("Inverse (A@inv=I)", me, 1e-4); aclDestroyTensor(ta);aclDestroyTensor(ti);
}
static void t_det() {
    const int n=3; std::vector<float> A(n*n); for(auto&v:A) v=(rand()/(float)RAND_MAX)-0.5f;
    std::vector<float> hd(1); DevBuf da(n*n*4),dd(4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*td=mk({1},ACL_FLOAT,dd.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDetGetWorkspaceSize(ta,td,w,e);}, aclnnDet);
    dd.down(hd.data());
    double ref = (double)A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7]-A[4]*A[6]);
    report("Det 3x3", std::fabs(hd[0]-ref)/(std::fabs(ref)+1e-6), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(td);
}
static void t_cholesky() {
    const int n=4; std::vector<float> M(n*n); for(auto&v:M) v=(rand()/(float)RAND_MAX)-0.5f;
    std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++){ double s=0; for(int k=0;k<n;k++) s+=(double)M[i*n+k]*M[j*n+k]; A[i*n+j]=(float)s; }
    for(int i=0;i<n;i++) A[i*n+i]+=(float)n;   // ensure SPD
    std::vector<float> hL(n*n); DevBuf da(n*n*4),dl(n*n*4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tl=mk({n,n},ACL_FLOAT,dl.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCholeskyGetWorkspaceSize(ta,tl,w,e);}, aclnnCholesky);
    dl.down(hL.data());
    double me=0,mr=0; bool lower=true;
    for(int i=0;i<n;i++)for(int j=0;j<n;j++){ if(j>i && std::fabs(hL[i*n+j])>1e-5) lower=false;
        double s=0; for(int k=0;k<n;k++) s+=(double)hL[i*n+k]*hL[j*n+k]; me=std::max(me,std::fabs(s-A[i*n+j])); mr=std::max(mr,std::fabs((double)A[i*n+j])); }
    report("Cholesky L@L^T=A", me/(mr+1e-9), 1e-4);
    report("Cholesky lower-tri", lower?0.0:1.0, 0);
    aclDestroyTensor(ta);aclDestroyTensor(tl);
}
static void t_cross() {
    const int B=4; auto a=randv(B*3,-1,1),b=randv(B*3,-1,1); std::vector<float> hz(B*3);
    DevBuf da(B*3*4),db(B*3*4),dz(B*3*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({B,3},ACL_FLOAT,da.p),*tb=mk({B,3},ACL_FLOAT,db.p),*tz=mk({B,3},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCrossGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnCross);
    dz.down(hz.data()); double me=0,mr=0;
    for(int i=0;i<B;i++){ const float *x=&a[i*3],*y=&b[i*3]; double r0=x[1]*y[2]-x[2]*y[1],r1=x[2]*y[0]-x[0]*y[2],r2=x[0]*y[1]-x[1]*y[0];
        me=std::max({me,std::fabs(hz[i*3]-r0),std::fabs(hz[i*3+1]-r1),std::fabs(hz[i*3+2]-r2)}); mr=std::max({mr,std::fabs(r0),std::fabs(r1),std::fabs(r2)}); }
    report("Cross", me/(mr+1e-9), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
}
static void t_solve() {
    const int n=5,nrhs=2; std::vector<float> A(n*n); for(auto&v:A) v=(rand()/(float)RAND_MAX)-0.5f; for(int i=0;i<n;i++) A[i*n+i]+=(float)n;
    auto B=randv(n*nrhs,-1,1); std::vector<float> hX(n*nrhs);
    DevBuf da(n*n*4),db(n*nrhs*4),dx(n*nrhs*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tb=mk({n,nrhs},ACL_FLOAT,db.p),*tx=mk({n,nrhs},ACL_FLOAT,dx.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSolveGetWorkspaceSize(ta,tb,tx,w,e);}, aclnnSolve);
    dx.down(hX.data()); double me=0,mr=0;
    for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){ double s=0; for(int k=0;k<n;k++) s+=(double)A[i*n+k]*hX[k*nrhs+j]; me=std::max(me,std::fabs(s-B[i*nrhs+j])); mr=std::max(mr,std::fabs((double)B[i*nrhs+j])); }
    report("Solve (A@X=B)", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tx);
}
static void t_slogdet() {
    const int n=3; std::vector<float> A(n*n); for(auto&v:A) v=(rand()/(float)RAND_MAX)-0.5f;
    std::vector<float> hs(1),hl(1); DevBuf da(n*n*4),ds(4),dl(4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tsg=mk({1},ACL_FLOAT,ds.p),*tla=mk({1},ACL_FLOAT,dl.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSlogdetGetWorkspaceSize(ta,tsg,tla,w,e);}, aclnnSlogdet);
    ds.down(hs.data()); dl.down(hl.data());
    double det=(double)A[0]*(A[4]*A[8]-A[5]*A[7])-A[1]*(A[3]*A[8]-A[5]*A[6])+A[2]*(A[3]*A[7]-A[4]*A[6]);
    double refsign=det>0?1:(det<0?-1:0), reflog=std::log(std::fabs(det));
    report("Slogdet", std::fabs(hs[0]-refsign)+std::fabs(hl[0]-reflog)/(std::fabs(reflog)+1e-6), 1e-4);
    aclDestroyTensor(ta);aclDestroyTensor(tsg);aclDestroyTensor(tla);
}
static void t_qr() {
    const int m=6,n=4,k=4; auto A=randv(m*n,-1,1); std::vector<float> hQ(m*k),hR(k*n);
    DevBuf da(m*n*4),dq(m*k*4),dr(k*n*4); da.up(A.data());
    aclTensor *ta=mk({m,n},ACL_FLOAT,da.p),*tq=mk({m,k},ACL_FLOAT,dq.p),*tr=mk({k,n},ACL_FLOAT,dr.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnQrGetWorkspaceSize(ta,tq,tr,w,e);}, aclnnQr);
    dq.down(hQ.data()); dr.down(hR.data());
    double me=0,mr=0; for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double s=0; for(int l=0;l<k;l++) s+=(double)hQ[i*k+l]*hR[l*n+j]; me=std::max(me,std::fabs(s-A[i*n+j])); mr=std::max(mr,std::fabs((double)A[i*n+j])); }
    double qo=0; for(int a=0;a<k;a++)for(int b=0;b<k;b++){ double s=0; for(int i=0;i<m;i++) s+=(double)hQ[i*k+a]*hQ[i*k+b]; qo=std::max(qo,std::fabs(s-(a==b?1.0:0.0))); }
    report("Qr (Q@R=A)", me/(mr+1e-9), 1e-4); report("Qr (Q orthonormal)", qo, 1e-4);
    aclDestroyTensor(ta);aclDestroyTensor(tq);aclDestroyTensor(tr);
}
static void t_svd() {
    const int m=5,n=3; auto A=randv(m*n,-1,1); std::vector<float> hU(m*m),hS(n),hVT(n*n);
    DevBuf da(m*n*4),du(m*m*4),ds(n*4),dvt(n*n*4); da.up(A.data());
    aclTensor *ta=mk({m,n},ACL_FLOAT,da.p),*tu=mk({m,m},ACL_FLOAT,du.p),*ts=mk({n},ACL_FLOAT,ds.p),*tvt=mk({n,n},ACL_FLOAT,dvt.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSvdGetWorkspaceSize(ta,tu,ts,tvt,w,e);}, aclnnSvd);
    du.down(hU.data()); ds.down(hS.data()); dvt.down(hVT.data());
    double me=0,mr=0; for(int i=0;i<m;i++)for(int j=0;j<n;j++){ double s=0; for(int l=0;l<n;l++) s+=(double)hU[i*m+l]*hS[l]*hVT[l*n+j]; me=std::max(me,std::fabs(s-A[i*n+j])); mr=std::max(mr,std::fabs((double)A[i*n+j])); }
    report("Svd (U@S@VT=A)", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tu);aclDestroyTensor(ts);aclDestroyTensor(tvt);
}
static void t_eigh() {
    const int n=4; std::vector<float> M(n*n); for(auto&v:M) v=(rand()/(float)RAND_MAX)-0.5f;
    std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++) A[i*n+j]=0.5f*(M[i*n+j]+M[j*n+i]);   // symmetric
    std::vector<float> hW(n),hV(n*n); DevBuf da(n*n*4),dw(n*4),dv(n*n*4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tw=mk({n},ACL_FLOAT,dw.p),*tv=mk({n,n},ACL_FLOAT,dv.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEighGetWorkspaceSize(ta,tw,tv,w,e);}, aclnnEigh);
    dw.down(hW.data()); dv.down(hV.data());
    double me=0,mr=0;
    for(int j=0;j<n;j++)for(int i=0;i<n;i++){ double Av=0; for(int k=0;k<n;k++) Av+=(double)A[i*n+k]*hV[k*n+j]; double ref=(double)hW[j]*hV[i*n+j]; me=std::max(me,std::fabs(Av-ref)); mr=std::max(mr,std::fabs(ref)); }
    report("Eigh (A@v=w*v)", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tw);aclDestroyTensor(tv);
}
static void t_pinverse() {
    const int m=6,n=4; std::vector<float> A(m*n); for(auto&v:A) v=(rand()/(float)RAND_MAX)-0.5f;
    std::vector<float> hP(n*m); DevBuf da(m*n*4),dp(n*m*4); da.up(A.data());
    aclTensor *ta=mk({m,n},ACL_FLOAT,da.p),*tp=mk({n,m},ACL_FLOAT,dp.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPinverseGetWorkspaceSize(ta,tp,w,e);}, aclnnPinverse);
    dp.down(hP.data());
    // A @ pinv @ A == A
    std::vector<double> pa(n*n,0); for(int i=0;i<n;i++)for(int j=0;j<n;j++){double s=0;for(int k=0;k<m;k++)s+=(double)hP[i*m+k]*A[k*n+j];pa[i*n+j]=s;}
    double me=0,mr=0; for(int i=0;i<m;i++)for(int j=0;j<n;j++){double s=0;for(int k=0;k<n;k++)s+=(double)A[i*n+k]*pa[k*n+j];me=std::max(me,std::fabs(s-A[i*n+j]));mr=std::max(mr,std::fabs((double)A[i*n+j]));}
    report("Pinverse (A@pinv@A=A)", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tp);
}
static void t_lu_solve() {
    const int n=5,nrhs=2; std::vector<float> A(n*n); for(auto&v:A) v=(rand()/(float)RAND_MAX)-0.5f; for(int i=0;i<n;i++)A[i*n+i]+=(float)n;
    auto B=randv(n*nrhs,-1,1); std::vector<int32_t> piv(n); std::vector<float> lu(n*n),hX(n*nrhs);
    DevBuf da(n*n*4),dlu(n*n*4),dpiv(n*4),db(n*nrhs*4),dx(n*nrhs*4); da.up(A.data()); db.up(B.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tlu=mk({n,n},ACL_FLOAT,dlu.p),*tpiv=mk({n},ACL_INT32,dpiv.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLuGetWorkspaceSize(ta,tlu,tpiv,w,e);}, aclnnLu);
    aclTensor *tlu2=mk({n,n},ACL_FLOAT,dlu.p),*tpiv2=mk({n},ACL_INT32,dpiv.p),*tb=mk({n,nrhs},ACL_FLOAT,db.p),*tx=mk({n,nrhs},ACL_FLOAT,dx.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLuSolveGetWorkspaceSize(tlu2,tpiv2,tb,tx,w,e);}, aclnnLuSolve);
    dx.down(hX.data()); double me=0,mr=0;
    for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){double s=0;for(int k=0;k<n;k++)s+=(double)A[i*n+k]*hX[k*nrhs+j];me=std::max(me,std::fabs(s-B[i*nrhs+j]));mr=std::max(mr,std::fabs((double)B[i*nrhs+j]));}
    report("Lu+LuSolve (A@X=B)", me/(mr+1e-9), 1e-4);
    aclDestroyTensor(ta);aclDestroyTensor(tlu);aclDestroyTensor(tpiv);aclDestroyTensor(tlu2);aclDestroyTensor(tpiv2);aclDestroyTensor(tb);aclDestroyTensor(tx);
}
static void t_trisolve() {
    const int n=5,nrhs=2; std::vector<float> L(n*n,0); for(int i=0;i<n;i++)for(int j=0;j<=i;j++) L[i*n+j]=(rand()/(float)RAND_MAX)-0.5f+(i==j?(float)n:0.f);
    auto B=randv(n*nrhs,-1,1); std::vector<float> hX(n*nrhs);
    DevBuf da(n*n*4),db(n*nrhs*4),dx(n*nrhs*4); da.up(L.data()); db.up(B.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*tb=mk({n,nrhs},ACL_FLOAT,db.p),*tx=mk({n,nrhs},ACL_FLOAT,dx.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTriangularSolveGetWorkspaceSize(ta,tb,false,false,false,tx,w,e);}, aclnnTriangularSolve);
    dx.down(hX.data()); double me=0,mr=0;
    for(int i=0;i<n;i++)for(int j=0;j<nrhs;j++){double s=0;for(int k=0;k<n;k++)s+=(double)L[i*n+k]*hX[k*nrhs+j];me=std::max(me,std::fabs(s-B[i*nrhs+j]));mr=std::max(mr,std::fabs((double)B[i*nrhs+j]));}
    report("TriangularSolve (L@X=B)", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tx);
}
static void t_matrixexp() {
    const int n=4; std::vector<float> A(n*n); for(auto&v:A) v=((rand()/(float)RAND_MAX)-0.5f)*0.3f;   // small norm
    std::vector<float> hE(n*n); DevBuf da(n*n*4),de(n*n*4); da.up(A.data());
    aclTensor *ta=mk({n,n},ACL_FLOAT,da.p),*te=mk({n,n},ACL_FLOAT,de.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatrixExpGetWorkspaceSize(ta,te,w,e);}, aclnnMatrixExp);
    de.down(hE.data());
    // CPU reference: Taylor order 30 in double (small norm converges)
    std::vector<double> E(n*n,0),T(n*n,0); for(int i=0;i<n;i++){E[i*n+i]=1;T[i*n+i]=1;}
    for(int k=1;k<=30;k++){ std::vector<double> tmp(n*n,0); for(int i=0;i<n;i++)for(int j=0;j<n;j++){double s=0;for(int l=0;l<n;l++)s+=T[i*n+l]*A[l*n+j];tmp[i*n+j]=s/k;} T=tmp; for(int i=0;i<n*n;i++)E[i]+=T[i]; }
    double me=0,mr=0; for(int i=0;i<n*n;i++){me=std::max(me,std::fabs(hE[i]-E[i]));mr=std::max(mr,std::fabs(E[i]));}
    report("MatrixExp", me/(mr+1e-9), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(te);
}
int main(){ init(); srand(67); t_inverse(); t_det(); t_cholesky(); t_cross(); t_solve(); t_slogdet(); t_qr(); t_svd(); t_eigh(); t_pinverse(); t_lu_solve(); t_trisolve(); t_matrixexp(); return finish(); }
