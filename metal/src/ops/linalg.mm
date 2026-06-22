// Linear algebra family via Accelerate LAPACK (host, over unified memory). LAPACK is column-major; we
// transpose row-major tensors in/out. Verified by reconstruction invariants (tol 1e-4). MatrixExp = Taylor.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>

// Self-declared LAPACK (Accelerate provides these; avoids header deprecation churn).
extern "C" {
void sgetrf_(const int*,const int*,float*,const int*,int*,int*);
void sgetri_(const int*,float*,const int*,const int*,float*,const int*,int*);
void sgesv_(const int*,const int*,float*,const int*,int*,float*,const int*,int*);
void spotrf_(const char*,const int*,float*,const int*,int*);
void sgeqrf_(const int*,const int*,float*,const int*,float*,float*,const int*,int*);
void sorgqr_(const int*,const int*,const int*,float*,const int*,const float*,float*,const int*,int*);
void sgesvd_(const char*,const char*,const int*,const int*,float*,const int*,float*,float*,const int*,float*,const int*,float*,const int*,int*);
void ssyev_(const char*,const char*,const int*,float*,const int*,float*,float*,const int*,int*);
void sgetrs_(const char*,const int*,const int*,const float*,const int*,const int*,float*,const int*,int*);
void strtrs_(const char*,const char*,const char*,const int*,const int*,const float*,const int*,float*,const int*,int*);
}

namespace {
float *fp(const aclTensor *t) { return (float *)t->data + t->offset; }
int32_t *ip(const aclTensor *t) { return (int32_t *)t->data + t->offset; }
// row-major [r,c] <-> column-major
void r2c(const float *s, float *d, int r, int c) { for (int i=0;i<r;i++) for (int j=0;j<c;j++) d[i+j*r]=s[i*c+j]; }
void c2r(const float *s, float *d, int r, int c) { for (int i=0;i<r;i++) for (int j=0;j<c;j++) d[i*c+j]=s[i+j*r]; }

aclnnStatus run_linalg(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    int info = 0;
    switch (e->m) {
        case 0: { // Inverse [n,n]
            const aclTensor *A=e->a; aclTensor *O=e->out; int n=(int)A->viewDims[0];
            std::vector<float> ac(n*n); r2c(fp(A),ac.data(),n,n); std::vector<int> piv(n);
            sgetrf_(&n,&n,ac.data(),&n,piv.data(),&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            int lw=-1; float q; sgetri_(&n,ac.data(),&n,piv.data(),&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            sgetri_(&n,ac.data(),&n,piv.data(),wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(ac.data(),fp(O),n,n); break;
        }
        case 1: case 5: { // Det / Slogdet [n,n]
            const aclTensor *A=e->a; int n=(int)A->viewDims[0];
            std::vector<float> ac(n*n); r2c(fp(A),ac.data(),n,n); std::vector<int> piv(n);
            sgetrf_(&n,&n,ac.data(),&n,piv.data(),&info);
            double det=1; for(int i=0;i<n;i++){ det*=ac[i+i*n]; if(piv[i]!=i+1) det=-det; }
            if(e->m==1) fp(e->out)[0]=(float)det;
            else { fp(e->out)[0]=(float)(det>0?1:(det<0?-1:0)); fp(e->out2)[0]=(float)std::log(std::fabs(det)); }
            break;
        }
        case 2: { // Cholesky [n,n] -> lower L
            const aclTensor *A=e->a; aclTensor *O=e->out; int n=(int)A->viewDims[0];
            std::vector<float> ac(n*n); r2c(fp(A),ac.data(),n,n); const char L='L';
            spotrf_(&L,&n,ac.data(),&n,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *o=fp(O); for(int i=0;i<n;i++) for(int j=0;j<n;j++) o[i*n+j]=(i>=j)?ac[i+j*n]:0.f; break;
        }
        case 3: { // Cross [B,3]
            const aclTensor *A=e->a,*B=e->b; aclTensor *O=e->out; int Bn=(int)A->viewDims[0];
            const float *a=fp(A),*b=fp(B); float *o=fp(O);
            for(int i=0;i<Bn;i++){ const float *x=a+i*3,*y=b+i*3; o[i*3]=x[1]*y[2]-x[2]*y[1]; o[i*3+1]=x[2]*y[0]-x[0]*y[2]; o[i*3+2]=x[0]*y[1]-x[1]*y[0]; } break;
        }
        case 4: { // Solve A[n,n] X = B[n,nrhs]
            const aclTensor *A=e->a,*B=e->b; aclTensor *X=e->out; int n=(int)A->viewDims[0], nrhs=(int)B->viewDims[1];
            std::vector<float> ac(n*n), bc(n*nrhs); r2c(fp(A),ac.data(),n,n); r2c(fp(B),bc.data(),n,nrhs); std::vector<int> piv(n);
            sgesv_(&n,&nrhs,ac.data(),&n,piv.data(),bc.data(),&n,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(bc.data(),fp(X),n,nrhs); break;
        }
        case 6: { // Qr A[m,n] -> Q[m,k], R[k,n], k=min(m,n)
            const aclTensor *A=e->a; aclTensor *Q=e->out,*R=e->out2; int m=(int)A->viewDims[0], n=(int)A->viewDims[1], k=m<n?m:n;
            std::vector<float> ac(m*n); r2c(fp(A),ac.data(),m,n); std::vector<float> tau(k);
            int lw=-1; float q; sgeqrf_(&m,&n,ac.data(),&m,tau.data(),&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            sgeqrf_(&m,&n,ac.data(),&m,tau.data(),wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *r=fp(R); for(int i=0;i<k;i++) for(int j=0;j<n;j++) r[i*n+j]=(i<=j)?ac[i+j*m]:0.f;   // R upper k×n
            lw=-1; sorgqr_(&m,&k,&k,ac.data(),&m,tau.data(),&q,&lw,&info); lw=(int)q; std::vector<float> wk2(lw);
            sorgqr_(&m,&k,&k,ac.data(),&m,tau.data(),wk2.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *qo=fp(Q); for(int i=0;i<m;i++) for(int j=0;j<k;j++) qo[i*k+j]=ac[i+j*m]; break;   // Q m×k
        }
        case 7: { // Svd A[m,n] -> U[m,m], S[min], VT[n,n]
            const aclTensor *A=e->a; aclTensor *U=e->out,*S=e->out2,*VT=e->mean ? const_cast<aclTensor*>(e->mean) : nullptr;
            int m=(int)A->viewDims[0], n=(int)A->viewDims[1], mn=m<n?m:n;
            std::vector<float> ac(m*n), uc(m*m), vtc(n*n); r2c(fp(A),ac.data(),m,n); const char JA='A';
            int lw=-1; float q; sgesvd_(&JA,&JA,&m,&n,ac.data(),&m,fp(S),uc.data(),&m,vtc.data(),&n,&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            sgesvd_(&JA,&JA,&m,&n,ac.data(),&m,fp(S),uc.data(),&m,vtc.data(),&n,wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            (void)mn; c2r(uc.data(),fp(U),m,m); c2r(vtc.data(),fp(VT),n,n); break;
        }
        case 8: { // Eigh A[n,n] sym -> W[n], V[n,n]
            const aclTensor *A=e->a; aclTensor *W=e->out,*V=e->out2; int n=(int)A->viewDims[0];
            std::vector<float> ac(n*n); r2c(fp(A),ac.data(),n,n); const char JV='V',UP='U';
            int lw=-1; float q; ssyev_(&JV,&UP,&n,ac.data(),&n,fp(W),&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            ssyev_(&JV,&UP,&n,ac.data(),&n,fp(W),wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(ac.data(),fp(V),n,n); break;
        }
        case 9: { // Pinverse A[m,n] -> P[n,m] via SVD: P = V S^+ U^T
            const aclTensor *A=e->a; aclTensor *P=e->out; int m=(int)A->viewDims[0], n=(int)A->viewDims[1], mn=m<n?m:n;
            std::vector<float> ac(m*n), uc(m*m), vtc(n*n), S(mn); r2c(fp(A),ac.data(),m,n); const char JA='A';
            int lw=-1; float q; sgesvd_(&JA,&JA,&m,&n,ac.data(),&m,S.data(),uc.data(),&m,vtc.data(),&n,&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            sgesvd_(&JA,&JA,&m,&n,ac.data(),&m,S.data(),uc.data(),&m,vtc.data(),&n,wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *p=fp(P); double tol=1e-6;   // P[i][j] = sum_l V[i][l] (1/S[l]) U[j][l]; V=VT^T -> V[i][l]=vtc[l+i*n]; U[j][l]=uc[j+l*m]
            for(int i=0;i<n;i++) for(int j=0;j<m;j++){ double acc=0; for(int l=0;l<mn;l++){ if(S[l]>tol) acc += (double)vtc[l+i*n]*(1.0/S[l])*uc[j+l*m]; } p[i*m+j]=(float)acc; } break;
        }
        case 10: { // Lu A[n,n] -> LU[n,n], piv[n]
            const aclTensor *A=e->a; aclTensor *LU=e->out,*PV=e->out2; int n=(int)A->viewDims[0];
            std::vector<float> ac(n*n); r2c(fp(A),ac.data(),n,n); std::vector<int> piv(n);
            sgetrf_(&n,&n,ac.data(),&n,piv.data(),&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(ac.data(),fp(LU),n,n); int32_t *pv=ip(PV); for(int i=0;i<n;i++) pv[i]=piv[i]; break;
        }
        case 11: { // LuSolve LU[n,n], piv[n], B[n,nrhs] -> X
            const aclTensor *LU=e->a,*PV=e->b,*B=e->c; aclTensor *X=e->out; int n=(int)LU->viewDims[0], nrhs=(int)B->viewDims[1];
            std::vector<float> luc(n*n), bc(n*nrhs); r2c(fp(LU),luc.data(),n,n); r2c(fp(B),bc.data(),n,nrhs);
            std::vector<int> piv(n); const int32_t *pv=ip(PV); for(int i=0;i<n;i++) piv[i]=pv[i]; const char N='N';
            sgetrs_(&N,&n,&nrhs,luc.data(),&n,piv.data(),bc.data(),&n,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(bc.data(),fp(X),n,nrhs); break;
        }
        case 12: { // TriangularSolve A[n,n] X = B[n,nrhs]; flags upper/transpose/unit in dscalars
            const aclTensor *A=e->a,*B=e->b; aclTensor *X=e->out; int n=(int)A->viewDims[0], nrhs=(int)B->viewDims[1];
            std::vector<float> ac(n*n), bc(n*nrhs); r2c(fp(A),ac.data(),n,n); r2c(fp(B),bc.data(),n,nrhs);
            char uplo=e->dscalars[0]?'U':'L', trans=e->dscalars[1]?'T':'N', diag=e->dscalars[2]?'U':'N';
            strtrs_(&uplo,&trans,&diag,&n,&nrhs,ac.data(),&n,bc.data(),&n,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            c2r(bc.data(),fp(X),n,nrhs); break;
        }
        case 13: { // MatrixExp A[n,n] via Taylor (row-major matmuls)
            const aclTensor *A=e->a; aclTensor *O=e->out; int n=(int)A->viewDims[0]; const float *a=fp(A);
            std::vector<double> E(n*n,0), T(n*n,0), tmp(n*n); for(int i=0;i<n;i++){ E[i*n+i]=1; T[i*n+i]=1; }
            for(int k=1;k<=30;k++){ for(int i=0;i<n;i++) for(int j=0;j<n;j++){ double acc=0; for(int l=0;l<n;l++) acc+=T[i*n+l]*a[l*n+j]; tmp[i*n+j]=acc/k; } T=tmp; for(int i=0;i<n*n;i++) E[i]+=T[i]; }
            float *o=fp(O); for(int i=0;i<n*n;i++) o[i]=(float)E[i]; break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_linalg(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=0; e->a=self; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInverse)
aclnnStatus aclnnDetGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=1; e->a=self; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnDet)
aclnnStatus aclnnCholeskyGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=2; e->a=self; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnCholesky)
aclnnStatus aclnnCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=3; e->a=self; e->b=other; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnCross)
aclnnStatus aclnnSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=4; e->a=A; e->b=B; e->out=X; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSolve)
aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *A, aclTensor *sign, aclTensor *logabsdet, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=5; e->a=A; e->out=sign; e->out2=logabsdet; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSlogdet)
aclnnStatus aclnnQrGetWorkspaceSize(const aclTensor *A, aclTensor *Q, aclTensor *R, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=6; e->a=A; e->out=Q; e->out2=R; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnQr)
aclnnStatus aclnnSvdGetWorkspaceSize(const aclTensor *A, aclTensor *U, aclTensor *S, aclTensor *VT, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=7; e->a=A; e->out=U; e->out2=S; e->mean=VT; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSvd)
aclnnStatus aclnnEighGetWorkspaceSize(const aclTensor *A, aclTensor *W, aclTensor *V, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=8; e->a=A; e->out=W; e->out2=V; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnEigh)
aclnnStatus aclnnPinverseGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=9; e->a=A; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnPinverse)
aclnnStatus aclnnLuGetWorkspaceSize(const aclTensor *A, aclTensor *LU, aclTensor *pivots, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=10; e->a=A; e->out=LU; e->out2=pivots; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnLu)
aclnnStatus aclnnLuSolveGetWorkspaceSize(const aclTensor *LU, const aclTensor *pivots, const aclTensor *B, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=11; e->a=LU; e->b=pivots; e->c=B; e->out=X; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnLuSolve)
aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, bool upper, bool transpose, bool unitriangular, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=12; e->a=A; e->b=B; e->out=X; e->dscalars={(double)upper,(double)transpose,(double)unitriangular}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnTriangularSolve)
aclnnStatus aclnnMatrixExpGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto*e=new aclOpExecutor(); e->m=13; e->a=A; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMatrixExp)
} // extern "C"
