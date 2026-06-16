// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cusolverDn.h>

namespace _linalg {
// Linear algebra (P9): Inverse, Det, Cholesky (cuSOLVER), Cross (hand-written). fp32, 2D square (Cross: 3-vectors).
// Row-major note: cuSOLVER is column-major, so it sees our row-major A as A^T. For inverse this is exploited directly
// (the buffer read back row-major equals A^-1); for Cholesky we factor UPPER then zero the strict upper triangle.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__global__ void k_eye(float *a, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*n) return; a[i]=(i/n==i%n)?1.f:0.f; }
__global__ void k_det_from_lu(const float *lu, const int *ipiv, float *out, int64_t n){
    double d=1.0; int swaps=0; for(int64_t i=0;i<n;i++){ d*=lu[i*n+i]; if(ipiv[i]!=(int)(i+1)) swaps++; } *out=(float)(swaps&1 ? -d : d);
}
__global__ void k_zero_strict_upper(float *a, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*n) return; int64_t r=i/n,c=i%n; if(c>r) a[i]=0.f; }
__global__ void k_cross(const float *a, const float *b, float *o, int64_t batch){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=batch) return; const float *x=a+i*3,*y=b+i*3; float *z=o+i*3;
    z[0]=x[1]*y[2]-x[2]*y[1]; z[1]=x[2]*y[0]-x[0]*y[2]; z[2]=x[0]*y[1]-x[1]*y[0];
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// Inverse (n x n). out = self^-1.
aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || self->viewDims.size() != 2 || self->viewDims[0] != self->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    int64_t n = self->viewDims[0];
    auto *e = new aclOpExecutor(); e->op = 0; e->a = self; e->out = out; e->n = n;
    if (ws) *ws = (uint64_t)n*n*sizeof(float) + (uint64_t)512*1024 + n*sizeof(int) + sizeof(int);  // LU + cusolver work + ipiv + info
    *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInverse(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h, st);
    float *lu=(float*)ws; float *work=lu + n*n; int *ipiv=(int*)(work + 131072); int *info=ipiv + n;
    cudaMemcpyAsync(lu, e->a->data, (size_t)n*n*sizeof(float), cudaMemcpyDeviceToDevice, st);
    int lwork=0; cusolverDnSgetrf_bufferSize(h, n, n, lu, n, &lwork);
    k_eye<<<nb(n*n),TH,0,st>>>((float*)e->out->data, n);
    cusolverDnSgetrf(h, n, n, lu, n, work, ipiv, info);
    cusolverDnSgetrs(h, CUBLAS_OP_N, n, n, lu, n, ipiv, (float*)e->out->data, n, info);
    cudaStreamSynchronize(st); cusolverDnDestroy(h);
    return done(e);
}
// Det (n x n) -> scalar
aclnnStatus aclnnDetGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || self->viewDims.size() != 2 || self->viewDims[0] != self->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    int64_t n=self->viewDims[0]; auto *e=new aclOpExecutor(); e->op=1; e->a=self; e->out=out; e->n=n;
    if (ws) *ws = (uint64_t)n*n*sizeof(float) + 131072 + n*sizeof(int) + sizeof(int);
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDet(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h, st);
    float *lu=(float*)ws; float *work=lu + n*n; int *ipiv=(int*)(work + 32768); int *info=ipiv+n;
    cudaMemcpyAsync(lu, e->a->data, (size_t)n*n*sizeof(float), cudaMemcpyDeviceToDevice, st);
    int lwork=0; cusolverDnSgetrf_bufferSize(h, n, n, lu, n, &lwork);
    cusolverDnSgetrf(h, n, n, lu, n, work, ipiv, info);
    k_det_from_lu<<<1,1,0,st>>>(lu, ipiv, (float*)e->out->data, n);
    cudaStreamSynchronize(st); cusolverDnDestroy(h);
    return done(e);
}
// Cholesky: A SPD (n x n) -> lower-triangular L with A = L L^T (row-major)
aclnnStatus aclnnCholeskyGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || self->dtype != ACL_FLOAT || self->viewDims.size() != 2 || self->viewDims[0] != self->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    int64_t n=self->viewDims[0]; auto *e=new aclOpExecutor(); e->op=2; e->a=self; e->out=out; e->n=n;
    if (ws) *ws = 131072 + sizeof(int);
    *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCholesky(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h, st);
    int *info=(int*)ws; float *work=(float*)(info+1);
    cudaMemcpyAsync(e->out->data, e->a->data, (size_t)n*n*sizeof(float), cudaMemcpyDeviceToDevice, st);
    int lwork=0; cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_UPPER, n, (float*)e->out->data, n, &lwork);
    // UPPER in col-major == lower in row-major -> reading row-major gives L with A=LL^T
    cusolverDnSpotrf(h, CUBLAS_FILL_MODE_UPPER, n, (float*)e->out->data, n, work, lwork, info);
    k_zero_strict_upper<<<nb(n*n),TH,0,st>>>((float*)e->out->data, n);
    cudaStreamSynchronize(st); cusolverDnDestroy(h);
    return done(e);
}
// Cross product of 3-vectors (batched over leading dims; last dim = 3)
aclnnStatus aclnnCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)self->viewDims.size(); if (self->viewDims[rank-1] != 3) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=3; e->a=self; e->b=other; e->out=out; e->m=self->numel()/3;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCross(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_cross<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,e->m); return done(e);
}

} // extern "C"
} // namespace _linalg

namespace _linalg2 {
// Linear algebra completion (P9): Solve, Slogdet, Qr, Svd, Eigh (cuSOLVER). fp32, 2D.
// Row-major tensors are transposed to/from column-major around cuSOLVER calls (k_r2c / k_c2r). Scratch is cudaMalloc'd per call.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
// row-major [rows,cols] -> column-major buffer (col[i + j*rows] = row[i*cols + j])
__global__ void k_r2c(const float *r, float *c, int64_t rows, int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*cols) return; int64_t rr=i/cols,cc=i%cols; c[rr+cc*rows]=r[i]; }
// column-major [rows,cols] -> row-major
__global__ void k_c2r(const float *c, float *r, int64_t rows, int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*cols) return; int64_t rr=i/cols,cc=i%cols; r[i]=c[rr+cc*rows]; }
// extract upper-triangular R[k,n] (row-major) from col-major factored A[m,n]
__global__ void k_extract_R(const float *Acol, float *R, int64_t m, int64_t k, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=k*n) return; int64_t rr=i/n,cc=i%n; R[i]=(cc>=rr)?Acol[rr+cc*m]:0.f; }
__global__ void k_slogdet(const float *lu, const int *ipiv, float *sign, float *logabs, int64_t n){
    double la=0; int swaps=0; double sg=1.0; for(int64_t i=0;i<n;i++){ double d=lu[i+i*n]; if(d<0){sg=-sg;} la+=log(fabs(d)+1e-300); if(ipiv[i]!=(int)(i+1)) swaps++; }
    if(swaps&1) sg=-sg; *sign=(float)sg; *logabs=(float)la;
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// Solve A X = B; A[n,n], B[n,nrhs] -> X[n,nrhs]
aclnnStatus aclnnSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!B||!X||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]!=A->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=A; e->b=B; e->out=X; e->n=A->viewDims[0]; e->m=B->viewDims.size()==2?B->viewDims[1]:1;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSolve(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n,nrhs=e->m; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Acol,*Bcol; int *ipiv,*info; cudaMalloc(&Acol,n*n*4); cudaMalloc(&Bcol,n*nrhs*4); cudaMalloc(&ipiv,n*sizeof(int)); cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(n*n),TH,0,st>>>((const float*)e->a->data,Acol,n,n);
    k_r2c<<<nb(n*nrhs),TH,0,st>>>((const float*)e->b->data,Bcol,n,nrhs);
    int lwork=0; cusolverDnSgetrf_bufferSize(h,n,n,Acol,n,&lwork); float *work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSgetrf(h,n,n,Acol,n,work,ipiv,info);
    cusolverDnSgetrs(h,CUBLAS_OP_N,n,nrhs,Acol,n,ipiv,Bcol,n,info);
    k_c2r<<<nb(n*nrhs),TH,0,st>>>(Bcol,(float*)e->out->data,n,nrhs);
    cudaStreamSynchronize(st); cudaFree(Acol);cudaFree(Bcol);cudaFree(ipiv);cudaFree(info);cudaFree(work); cusolverDnDestroy(h);
    return done(e);
}
// Slogdet: A[n,n] -> sign (scalar), logabsdet (scalar)
aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *A, aclTensor *sign, aclTensor *logabsdet, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!sign||!logabsdet||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=A; e->out=sign; e->out2=logabsdet; e->n=A->viewDims[0];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSlogdet(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *lu; int *ipiv,*info; cudaMalloc(&lu,n*n*4); cudaMalloc(&ipiv,n*sizeof(int)); cudaMalloc(&info,sizeof(int));
    cudaMemcpyAsync(lu,e->a->data,(size_t)n*n*4,cudaMemcpyDeviceToDevice,st);   // LU diag is the same in row/col-major (det(A)=det(A^T))
    int lwork=0; cusolverDnSgetrf_bufferSize(h,n,n,lu,n,&lwork); float *work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSgetrf(h,n,n,lu,n,work,ipiv,info);
    k_slogdet<<<1,1,0,st>>>(lu,ipiv,(float*)e->out->data,(float*)e->out2->data,n);
    cudaStreamSynchronize(st); cudaFree(lu);cudaFree(ipiv);cudaFree(info);cudaFree(work); cusolverDnDestroy(h);
    return done(e);
}
// Qr: A[m,n] -> Q[m,k], R[k,n], k=min(m,n) (reduced)
aclnnStatus aclnnQrGetWorkspaceSize(const aclTensor *A, aclTensor *Q, aclTensor *R, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!Q||!R||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=A; e->out=Q; e->out2=R; e->m=A->viewDims[0]; e->n=A->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQr(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t m=e->m,n=e->n,k=m<n?m:n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Acol,*tau; int *info; cudaMalloc(&Acol,m*n*4); cudaMalloc(&tau,k*4); cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(m*n),TH,0,st>>>((const float*)e->a->data,Acol,m,n);
    int lwork=0; cusolverDnSgeqrf_bufferSize(h,m,n,Acol,m,&lwork); float *work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSgeqrf(h,m,n,Acol,m,tau,work,lwork,info);
    k_extract_R<<<nb(k*n),TH,0,st>>>(Acol,(float*)e->out2->data,m,k,n);   // R[k,n] row-major (before orgqr destroys upper)
    int lwork2=0; cusolverDnSorgqr_bufferSize(h,m,k,k,Acol,m,tau,&lwork2); float *work2; cudaMalloc(&work2,(size_t)lwork2*4);
    cusolverDnSorgqr(h,m,k,k,Acol,m,tau,work2,lwork2,info);              // Q[m,k] in first k cols of Acol
    k_c2r<<<nb(m*k),TH,0,st>>>(Acol,(float*)e->out->data,m,k);          // Q row-major [m,k]
    cudaStreamSynchronize(st); cudaFree(Acol);cudaFree(tau);cudaFree(info);cudaFree(work);cudaFree(work2); cusolverDnDestroy(h);
    return done(e);
}
// Svd (full): A[m,n] (require m>=n) -> U[m,m], S[min(m,n)], VT[n,n]
aclnnStatus aclnnSvdGetWorkspaceSize(const aclTensor *A, aclTensor *U, aclTensor *S, aclTensor *VT, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!U||!S||!VT||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]<A->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=A; e->out=U; e->out2=S; e->inputs={VT}; e->m=A->viewDims[0]; e->n=A->viewDims[1];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSvd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t m=e->m,n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Acol,*Ucol,*VTcol; int *info; cudaMalloc(&Acol,m*n*4); cudaMalloc(&Ucol,m*m*4); cudaMalloc(&VTcol,n*n*4); cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(m*n),TH,0,st>>>((const float*)e->a->data,Acol,m,n);
    int lwork=0; cusolverDnSgesvd_bufferSize(h,m,n,&lwork); float *work,*rwork; cudaMalloc(&work,(size_t)lwork*4); cudaMalloc(&rwork,(size_t)(n>1?n-1:1)*4);
    cusolverDnSgesvd(h,'A','A',m,n,Acol,m,(float*)e->out2->data,Ucol,m,VTcol,n,work,lwork,rwork,info);  // S written directly (1D, layout-agnostic)
    k_c2r<<<nb(m*m),TH,0,st>>>(Ucol,(float*)e->out->data,m,m);
    k_c2r<<<nb(n*n),TH,0,st>>>(VTcol,(float*)const_cast<aclTensor*>(e->inputs[0])->data,n,n);
    cudaStreamSynchronize(st); cudaFree(Acol);cudaFree(Ucol);cudaFree(VTcol);cudaFree(info);cudaFree(work);cudaFree(rwork); cusolverDnDestroy(h);
    return done(e);
}
// Eigh (symmetric): A[n,n] -> eigenvalues W[n] (ascending), eigenvectors V[n,n] (columns)
aclnnStatus aclnnEighGetWorkspaceSize(const aclTensor *A, aclTensor *W, aclTensor *V, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!W||!V||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]!=A->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->op=0; e->a=A; e->out=W; e->out2=V; e->n=A->viewDims[0];
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEigh(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Acol; int *info; cudaMalloc(&Acol,n*n*4); cudaMalloc(&info,sizeof(int));
    cudaMemcpyAsync(Acol,e->a->data,(size_t)n*n*4,cudaMemcpyDeviceToDevice,st);   // symmetric: row-major == col-major
    int lwork=0; cusolverDnSsyevd_bufferSize(h,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,n,Acol,n,(float*)e->out->data,&lwork); float *work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSsyevd(h,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_UPPER,n,Acol,n,(float*)e->out->data,work,lwork,info);
    k_c2r<<<nb(n*n),TH,0,st>>>(Acol,(float*)e->out2->data,n,n);   // eigenvectors are columns (col-major) -> row-major V[i,j]=col i of evec j... c2r maps col-major->row-major preserving (i,j)
    cudaStreamSynchronize(st); cudaFree(Acol);cudaFree(info);cudaFree(work); cusolverDnDestroy(h);
    return done(e);
}

} // extern "C"
} // namespace _linalg2

namespace _linalg3 {
// Linear algebra remainder (P9): Pinverse (SVD), Lu + LuSolve (getrf/getrs), TriangularSolve (substitution), MatrixExp (scaling & squaring).
// fp32. cuSOLVER for SVD/LU; hand-written substitution and scaling-squaring. Row<->col-major via k_r2c/k_c2r.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__global__ void k_r2c(const float *r, float *c, int64_t rows, int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*cols) return; c[i/cols + (i%cols)*rows]=r[i]; }
__global__ void k_c2r(const float *c, float *r, int64_t rows, int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=rows*cols) return; r[i]=c[i/cols + (i%cols)*rows]; }
// pinv[n,m] = sum_k (S_k>eps?1/S_k:0) * V[i,k] * U[j,k], with Ucol[m,m], VTcol[n,n] column-major (V[i,k]=VTcol[k+i*n], U[j,k]=Ucol[j+k*m])
__global__ void k_pinv_combine(const float *Ucol, const float *S, const float *VTcol, float *pinv, int64_t m, int64_t n, float eps){
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=n*m) return; int64_t i=idx/m, j=idx%m;
    double acc=0; for(int64_t k=0;k<n;k++){ float sk=S[k]; if(sk>eps) acc += (1.0/sk)*VTcol[k+i*n]*Ucol[j+k*m]; } pinv[idx]=(float)acc;
}
__global__ void k_eye(float *a, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*n) return; a[i]=(i/n==i%n)?1.f:0.f; }
__global__ void k_gemm(const float *A, const float *B, float *C, int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*n) return; int64_t r=i/n,c=i%n; double s=0; for(int64_t k=0;k<n;k++) s+=(double)A[r*n+k]*B[k*n+c]; C[i]=(float)s; }
__global__ void k_madd_scaled(float *E, const float *T, float sc, int64_t nn){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<nn) E[i]+=sc*T[i]; }
__global__ void k_scale(float *a, int64_t nn, float sc){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<nn) a[i]*=sc; }
__global__ void k_maxabs(const float *a, float *acc, int64_t nn){ __shared__ double r[TH]; double m=0; for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<nn;i+=(int64_t)gridDim.x*blockDim.x) m=fmax(m,fabs((double)a[i])); r[threadIdx.x]=m; __syncthreads(); for(int k=blockDim.x/2;k>0;k>>=1){if(threadIdx.x<k)r[threadIdx.x]=fmax(r[threadIdx.x],r[threadIdx.x+k]);__syncthreads();} if(threadIdx.x==0) atomicAdd(acc,(float)r[0]); }
// Triangular solve: op(A) X = B. A[n,n] row-major triangular. one thread per rhs column.
__global__ void k_trisolve(const float *A, const float *B, float *X, int64_t n, int64_t nrhs, bool upper, bool trans, bool unit){
    int64_t col=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(col>=nrhs) return;
    bool low_eff = !(upper ^ trans);   // effective lower-triangular (forward subst) when (lower & !trans) or (upper & trans)
    auto Aat=[&](int64_t r,int64_t c)->double{ return trans ? (double)A[c*n+r] : (double)A[r*n+c]; };
    if(low_eff){ for(int64_t i=0;i<n;i++){ double s=B[i*nrhs+col]; for(int64_t k=0;k<i;k++) s-=Aat(i,k)*X[k*nrhs+col]; X[i*nrhs+col]=(float)(unit?s:s/Aat(i,i)); } }
    else       { for(int64_t i=n-1;i>=0;i--){ double s=B[i*nrhs+col]; for(int64_t k=i+1;k<n;k++) s-=Aat(i,k)*X[k*nrhs+col]; X[i*nrhs+col]=(float)(unit?s:s/Aat(i,i)); } }
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// Pinverse: A[m,n] (m>=n) -> pinv[n,m]
aclnnStatus aclnnPinverseGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!out||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]<A->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=A; e->out=out; e->m=A->viewDims[0]; e->n=A->viewDims[1]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPinverse(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t m=e->m,n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Acol,*Ucol,*VTcol,*S; int *info; cudaMalloc(&Acol,m*n*4); cudaMalloc(&Ucol,m*m*4); cudaMalloc(&VTcol,n*n*4); cudaMalloc(&S,n*4); cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(m*n),TH,0,st>>>((const float*)e->a->data,Acol,m,n);
    int lwork=0; cusolverDnSgesvd_bufferSize(h,m,n,&lwork); float *work,*rwork; cudaMalloc(&work,(size_t)lwork*4); cudaMalloc(&rwork,(size_t)(n>1?n-1:1)*4);
    cusolverDnSgesvd(h,'A','A',m,n,Acol,m,S,Ucol,m,VTcol,n,work,lwork,rwork,info);
    k_pinv_combine<<<nb(n*m),TH,0,st>>>(Ucol,S,VTcol,(float*)e->out->data,m,n,1e-6f);
    cudaStreamSynchronize(st); cudaFree(Acol);cudaFree(Ucol);cudaFree(VTcol);cudaFree(S);cudaFree(info);cudaFree(work);cudaFree(rwork); cusolverDnDestroy(h);
    return done(e);
}
// Lu: A[n,n] -> LU (col-major, n x n) + pivots[n] int32
aclnnStatus aclnnLuGetWorkspaceSize(const aclTensor *A, aclTensor *LU, aclTensor *pivots, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!LU||!pivots||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]!=A->viewDims[1]||pivots->dtype!=ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=A; e->out=LU; e->out2=pivots; e->n=A->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLu(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *LU=(float*)e->out->data; int *info; cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(n*n),TH,0,st>>>((const float*)e->a->data,LU,n,n);   // col-major A in LU buffer
    int lwork=0; cusolverDnSgetrf_bufferSize(h,n,n,LU,n,&lwork); float *work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSgetrf(h,n,n,LU,n,work,(int*)e->out2->data,info);   // LU col-major + pivots in-place
    cudaStreamSynchronize(st); cudaFree(info);cudaFree(work); cusolverDnDestroy(h);
    return done(e);
}
// LuSolve: LU (col-major) + pivots + B[n,nrhs] -> X[n,nrhs]
aclnnStatus aclnnLuSolveGetWorkspaceSize(const aclTensor *LU, const aclTensor *pivots, const aclTensor *B, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    if (!LU||!pivots||!B||!X||!ex||LU->dtype!=ACL_FLOAT||pivots->dtype!=ACL_INT32) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=LU; e->b=pivots; e->c=B; e->out=X; e->n=LU->viewDims[0]; e->m=B->viewDims.size()==2?B->viewDims[1]:1; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLuSolve(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n,nrhs=e->m; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float *Bcol; int *info; cudaMalloc(&Bcol,n*nrhs*4); cudaMalloc(&info,sizeof(int));
    k_r2c<<<nb(n*nrhs),TH,0,st>>>((const float*)e->c->data,Bcol,n,nrhs);
    cusolverDnSgetrs(h,CUBLAS_OP_N,n,nrhs,(const float*)e->a->data,n,(const int*)e->b->data,Bcol,n,info);
    k_c2r<<<nb(n*nrhs),TH,0,st>>>(Bcol,(float*)e->out->data,n,nrhs);
    cudaStreamSynchronize(st); cudaFree(Bcol);cudaFree(info); cusolverDnDestroy(h);
    return done(e);
}
// TriangularSolve: op(A) X = B; A[n,n] row-major triangular -> X[n,nrhs]
aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, bool upper, bool transpose, bool unitriangular, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!B||!X||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=A; e->c=B; e->out=X; e->n=A->viewDims[0]; e->m=B->viewDims.size()==2?B->viewDims[1]:1;
    e->dscalars={(double)upper,(double)transpose,(double)unitriangular}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTriangularSolve(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->n,nrhs=e->m; bool up=e->dscalars[0],tr=e->dscalars[1],un=e->dscalars[2];
    k_trisolve<<<nb(nrhs),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->c->data,(float*)e->out->data,n,nrhs,up,tr,un); return done(e);
}
// MatrixExp: A[n,n] -> exp(A) via scaling-and-squaring + Taylor (order 12)
aclnnStatus aclnnMatrixExpGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A||!out||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]!=A->viewDims[1]) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=A; e->out=out; e->n=A->viewDims[0]; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatrixExp(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    auto st=(cudaStream_t)s; int64_t n=e->n, nn=n*n;
    float *As,*E,*term,*tmp,*nrm; cudaMalloc(&As,nn*4); cudaMalloc(&E,nn*4); cudaMalloc(&term,nn*4); cudaMalloc(&tmp,nn*4); cudaMalloc(&nrm,4);
    cudaMemsetAsync(nrm,0,4,st); cudaMemcpyAsync(As,e->a->data,(size_t)nn*4,cudaMemcpyDeviceToDevice,st);
    k_maxabs<<<32,TH,0,st>>>(As,nrm,nn); float hnrm; cudaMemcpyAsync(&hnrm,nrm,4,cudaMemcpyDeviceToHost,st); cudaStreamSynchronize(st);
    int sN=0; double bound=(double)hnrm*n; while(bound>0.5 && sN<30){ bound*=0.5; sN++; }
    k_scale<<<nb(nn),TH,0,st>>>(As,nn,1.f/(float)(1LL<<sN));   // As = A/2^sN
    k_eye<<<nb(nn),TH,0,st>>>(E,n); k_eye<<<nb(nn),TH,0,st>>>(term,n);   // E=I, term=I
    for(int k=1;k<=12;k++){ k_gemm<<<nb(nn),TH,0,st>>>(term,As,tmp,n); k_scale<<<nb(nn),TH,0,st>>>(tmp,nn,1.f/(float)k);
        cudaMemcpyAsync(term,tmp,(size_t)nn*4,cudaMemcpyDeviceToDevice,st); k_madd_scaled<<<nb(nn),TH,0,st>>>(E,term,1.f,nn); }
    for(int i=0;i<sN;i++){ k_gemm<<<nb(nn),TH,0,st>>>(E,E,tmp,n); cudaMemcpyAsync(E,tmp,(size_t)nn*4,cudaMemcpyDeviceToDevice,st); }
    cudaMemcpyAsync(e->out->data,E,(size_t)nn*4,cudaMemcpyDeviceToDevice,st);
    cudaStreamSynchronize(st); cudaFree(As);cudaFree(E);cudaFree(term);cudaFree(tmp);cudaFree(nrm);
    return done(e);
}

} // extern "C"
} // namespace _linalg3

namespace _linalg4 {
// Linalg remainder (R2): MatrixRank (SVD + threshold), Lstsq (normal equations via cuSOLVER), MatrixPower (binary exponentiation).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
__global__ void k_r2c(const float*r,float*c,int64_t rows,int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*cols) c[i/cols+(i%cols)*rows]=r[i]; }
__global__ void k_c2r(const float*c,float*r,int64_t rows,int64_t cols){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<rows*cols) r[i]=c[i/cols+(i%cols)*rows]; }
__global__ void k_eye(float*a,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i<n*n) a[i]=(i/n==i%n)?1.f:0.f; }
__global__ void k_gemm(const float*A,const float*B,float*C,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n*n)return; int64_t r=i/n,c=i%n; double s=0; for(int64_t k=0;k<n;k++) s+=(double)A[r*n+k]*B[k*n+c]; C[i]=(float)s; }
// out[K,N] = A[M,K]^T @ B[M,N]
__global__ void k_gemmTN(const float*A,const float*B,float*o,int64_t M,int64_t K,int64_t N){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=K*N)return; int64_t r=i/N,c=i%N; double s=0; for(int64_t m=0;m<M;m++) s+=(double)A[m*K+r]*B[m*N+c]; o[i]=(float)s; }
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// MatrixRank: A[m,n] -> rank (int64 scalar). rank = #{S_k > tol}, tol = max(m,n)*eps_machine*S_max (or given).
aclnnStatus aclnnMatrixRankGetWorkspaceSize(const aclTensor*A,double tol,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!A||!out||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||out->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=A; e->out=out; e->m=A->viewDims[0]; e->n=A->viewDims[1]; e->eps=tol; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatrixRank(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; int64_t m=e->m,n=e->n,mn=m<n?m:n,mx=m>n?m:n; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float*Acol,*U,*VT,*S; int*info; cudaMalloc(&Acol,m*n*4); cudaMalloc(&U,mx*mx*4); cudaMalloc(&VT,mx*mx*4); cudaMalloc(&S,mn*4); cudaMalloc(&info,4);
    bool tall=m>=n; int R=tall?m:n, C=tall?n:m;   // gesvd needs R>=C; singular values of A and A^T are identical
    if(tall) k_r2c<<<nb(m*n),TH,0,st>>>((const float*)e->a->data,Acol,m,n);     // A col-major [m,n]
    else     cudaMemcpyAsync(Acol,e->a->data,(size_t)m*n*4,cudaMemcpyDeviceToDevice,st);  // row-major A == A^T col-major [n,m]
    int lwork=0; cusolverDnSgesvd_bufferSize(h,R,C,&lwork); float*work,*rwork; cudaMalloc(&work,(size_t)lwork*4); cudaMalloc(&rwork,(size_t)(C>1?C-1:1)*4);
    cusolverDnSgesvd(h,'N','N',R,C,Acol,R,S,U,R,VT,C,work,lwork,rwork,info);
    std::vector<float> hs(mn); cudaMemcpyAsync(hs.data(),S,(size_t)mn*4,cudaMemcpyDeviceToHost,st); cudaStreamSynchronize(st);
    double smax=0; for(int i=0;i<mn;i++) smax=hs[i]>smax?hs[i]:smax;
    double thr = e->eps>0 ? e->eps : (double)(m>n?m:n)*1.1920929e-7*smax;
    int64_t rank=0; for(int i=0;i<mn;i++) if(hs[i]>thr) rank++;
    cudaMemcpyAsync(e->out->data,&rank,8,cudaMemcpyHostToDevice,st); cudaStreamSynchronize(st);
    cudaFree(Acol);cudaFree(U);cudaFree(VT);cudaFree(S);cudaFree(info);cudaFree(work);cudaFree(rwork); cusolverDnDestroy(h);
    return done(e);
}
// Lstsq: min_x ||A x - B||, A[m,n] (m>=n full rank), B[m,nrhs] -> X[n,nrhs] via normal equations (A^T A) x = A^T B
aclnnStatus aclnnLstsqGetWorkspaceSize(const aclTensor*A,const aclTensor*B,aclTensor*X,uint64_t*ws,aclOpExecutor**ex){
    if(!A||!B||!X||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=A; e->b=B; e->out=X; e->m=A->viewDims[0]; e->n=A->viewDims[1]; e->k=B->viewDims.size()==2?B->viewDims[1]:1; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLstsq(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; int64_t m=e->m,n=e->n,nrhs=e->k; cusolverDnHandle_t h; cusolverDnCreate(&h); cusolverDnSetStream(h,st);
    float*AtA,*Atb,*AtAcol,*Atbcol; int*ipiv,*info; cudaMalloc(&AtA,n*n*4); cudaMalloc(&Atb,n*nrhs*4); cudaMalloc(&AtAcol,n*n*4); cudaMalloc(&Atbcol,n*nrhs*4); cudaMalloc(&ipiv,n*sizeof(int)); cudaMalloc(&info,4);
    k_gemmTN<<<nb(n*n),TH,0,st>>>((const float*)e->a->data,(const float*)e->a->data,AtA,m,n,n);     // AtA[n,n]
    k_gemmTN<<<nb(n*nrhs),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,Atb,m,n,nrhs); // Atb[n,nrhs]
    k_r2c<<<nb(n*n),TH,0,st>>>(AtA,AtAcol,n,n); k_r2c<<<nb(n*nrhs),TH,0,st>>>(Atb,Atbcol,n,nrhs);
    int lwork=0; cusolverDnSgetrf_bufferSize(h,n,n,AtAcol,n,&lwork); float*work; cudaMalloc(&work,(size_t)lwork*4);
    cusolverDnSgetrf(h,n,n,AtAcol,n,work,ipiv,info); cusolverDnSgetrs(h,CUBLAS_OP_N,n,nrhs,AtAcol,n,ipiv,Atbcol,n,info);
    k_c2r<<<nb(n*nrhs),TH,0,st>>>(Atbcol,(float*)e->out->data,n,nrhs);
    cudaStreamSynchronize(st); cudaFree(AtA);cudaFree(Atb);cudaFree(AtAcol);cudaFree(Atbcol);cudaFree(ipiv);cudaFree(info);cudaFree(work); cusolverDnDestroy(h);
    return done(e);
}
// MatrixPower: A[n,n]^p (p>=0) via binary exponentiation
aclnnStatus aclnnMatrixPowerGetWorkspaceSize(const aclTensor*A,int64_t p,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!A||!out||!ex||A->dtype!=ACL_FLOAT||A->viewDims.size()!=2||A->viewDims[0]!=A->viewDims[1]||p<0) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=A; e->out=out; e->n=A->viewDims[0]; e->m=p; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatrixPower(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; int64_t n=e->n; int64_t p=e->m; int64_t nn=n*n;
    float*R,*base,*tmp; cudaMalloc(&R,nn*4); cudaMalloc(&base,nn*4); cudaMalloc(&tmp,nn*4);
    k_eye<<<nb(nn),TH,0,st>>>(R,n); cudaMemcpyAsync(base,e->a->data,(size_t)nn*4,cudaMemcpyDeviceToDevice,st);
    while(p>0){ if(p&1){ k_gemm<<<nb(nn),TH,0,st>>>(R,base,tmp,n); cudaMemcpyAsync(R,tmp,(size_t)nn*4,cudaMemcpyDeviceToDevice,st); }
        p>>=1; if(p){ k_gemm<<<nb(nn),TH,0,st>>>(base,base,tmp,n); cudaMemcpyAsync(base,tmp,(size_t)nn*4,cudaMemcpyDeviceToDevice,st); } }
    cudaMemcpyAsync(e->out->data,R,(size_t)nn*4,cudaMemcpyDeviceToDevice,st); cudaStreamSynchronize(st);
    cudaFree(R);cudaFree(base);cudaFree(tmp);
    return done(e);
}

} // extern "C"
} // namespace _linalg4

