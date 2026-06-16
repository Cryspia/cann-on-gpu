// blas_merged.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <vector>
#include <string>
#include <map>
#include <cuda_bf16.h>

namespace _blas_ext {
// BLAS extras (P8): Mm/Bmm (wrap matmul), Mv/Addmv, Addmm/Addbmm/Baddbmm, Dot/Vdot/Inner/Outer, Kron.
// Functional simple kernels (small "extra" ops); fp32-centric. out contiguous.

namespace {

constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

// out[m,n] = beta*C[m,n] + alpha*sum_k A[m,k]*B[k,n]; C may be null (beta ignored). Batched over `batch`.
__global__ void k_gemm(const float *A, const float *B, const float *C, float *o, int64_t batch, int64_t M, int64_t K, int64_t N,
        double alpha, double beta, bool addbmm) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t outN = addbmm ? M * N : batch * M * N; if (i >= outN) return;
    int64_t n = i % N, m = (i / N) % M, bb = addbmm ? 0 : i / (M * N);
    double acc = 0;
    if (addbmm) { for (int64_t b = 0; b < batch; ++b) { const float *a = A + b*M*K, *bb2 = B + b*K*N; for (int64_t k = 0; k < K; ++k) acc += (double)a[m*K+k]*bb2[k*N+n]; } }
    else { const float *a = A + bb*M*K, *b2 = B + bb*K*N; for (int64_t k = 0; k < K; ++k) acc += (double)a[m*K+k]*b2[k*N+n]; }
    double cval = C ? (double)C[(addbmm ? 0 : bb)*M*N + m*N + n] : 0.0;
    o[i] = (float)(beta * cval + alpha * acc);
}
// out[m] = beta*y[m] + alpha*sum_k A[m,k]*x[k]
__global__ void k_gemv(const float *A, const float *x, const float *y, float *o, int64_t M, int64_t K, double alpha, double beta) {
    int64_t m = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (m >= M) return;
    double acc = 0; for (int64_t k = 0; k < K; ++k) acc += (double)A[m*K+k]*x[k];
    o[m] = (float)(beta * (y ? (double)y[m] : 0.0) + alpha * acc);
}
__global__ void k_dot(const float *a, const float *b, float *o, int64_t n) {
    double s = 0; for (int64_t i = 0; i < n; ++i) s += (double)a[i]*b[i]; *o = (float)s;
}
__global__ void k_outer(const float *a, const float *b, float *o, int64_t M, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= M*N) return;
    o[i] = a[i/N] * b[i%N];
}
// Kron 2D: A[M,N] kron B[P,Q] -> out[M*P, N*Q]
__global__ void k_kron(const float *A, const float *B, float *o, int64_t M, int64_t N, int64_t P, int64_t Q) {
    int64_t total = M*P*N*Q; int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= total) return;
    int64_t oc = i % (N*Q), orr = i / (N*Q); int64_t m = orr / P, p = orr % P, n = oc / Q, q = oc % Q;
    o[i] = A[m*N+n] * B[p*Q+q];
}

inline aclnnStatus done(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// ---- Mm / Bmm: wrap existing matmul/batchmatmul ----
aclnnStatus aclnnMmGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMatmulGetWorkspaceSize(self, mat2, out, 1, ws, ex);
}
aclnnStatus aclnnMm(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnMatmul(ws, wsz, e, s); }
aclnnStatus aclnnBmmGetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnBatchMatMulGetWorkspaceSize(self, mat2, out, 1, ws, ex);
}
aclnnStatus aclnnBmm(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnBatchMatMul(ws, wsz, e, s); }

// ---- Mv: A[M,K] @ x[K] -> out[M] ----
aclnnStatus aclnnMvGetWorkspaceSize(const aclTensor *self, const aclTensor *vec, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !vec || !out || !ex || self->dtype != ACL_FLOAT || self->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=vec; e->out=out; e->m=self->viewDims[0]; e->k=self->viewDims[1]; e->alpha=1; e->eps=0;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMv(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_gemv<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,nullptr,(float*)e->out->data,e->m,e->k,1.0,0.0);
    return done(e);
}
// ---- Addmv: beta*y + alpha*(A@x) ----
aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *y, const aclTensor *mat, const aclTensor *vec, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!mat || !vec || !out || !ex || mat->dtype != ACL_FLOAT || mat->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=mat; e->b=vec; e->c=y; e->out=out; e->m=mat->viewDims[0]; e->k=mat->viewDims[1]; e->alpha=alpha; e->eps=beta;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddmv(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_gemv<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,e->m,e->k,e->alpha,e->eps);
    return done(e);
}
// ---- Addmm: beta*C + alpha*(A@B) ----
aclnnStatus aclnnAddmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A || !B || !out || !ex || A->dtype != ACL_FLOAT || A->viewDims.size() != 2 || B->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=A; e->b=B; e->c=C; e->out=out; e->m=A->viewDims[0]; e->k=A->viewDims[1]; e->n=B->viewDims[1]; e->alpha=alpha; e->eps=beta;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddmm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t M=e->m,K=e->k,N=e->n; int64_t g=nb(M*N);
    k_gemm<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,1,M,K,N,e->alpha,e->eps,false);
    return done(e);
}
// ---- Baddbmm: beta*C + alpha*(A@B) batched [B,M,K]@[B,K,N] ----
aclnnStatus aclnnBaddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A || !B || !out || !ex || A->dtype != ACL_FLOAT || A->viewDims.size() != 3) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=A; e->b=B; e->c=C; e->out=out; e->outerCount=A->viewDims[0]; e->m=A->viewDims[1]; e->k=A->viewDims[2]; e->n=B->viewDims[2]; e->alpha=alpha; e->eps=beta;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBaddbmm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t Bt=e->outerCount,M=e->m,K=e->k,N=e->n; int64_t g=nb(Bt*M*N);
    k_gemm<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,Bt,M,K,N,e->alpha,e->eps,false);
    return done(e);
}
// ---- Addbmm: beta*C + alpha*sum_b(A_b@B_b) -> [M,N] ----
aclnnStatus aclnnAddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!A || !B || !out || !ex || A->dtype != ACL_FLOAT || A->viewDims.size() != 3) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=A; e->b=B; e->c=C; e->out=out; e->outerCount=A->viewDims[0]; e->m=A->viewDims[1]; e->k=A->viewDims[2]; e->n=B->viewDims[2]; e->alpha=alpha; e->eps=beta;
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAddbmm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t Bt=e->outerCount,M=e->m,K=e->k,N=e->n; int64_t g=nb(M*N);
    k_gemm<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,Bt,M,K,N,e->alpha,e->eps,true);
    return done(e);
}
// ---- Dot / Vdot / Inner (1D real) ----
aclnnStatus aclnnDotGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=other; e->out=out; e->m=self->numel();
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDot(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_dot<<<1,1,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,e->m);
    return done(e);
}
aclnnStatus aclnnVdotGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnDotGetWorkspaceSize(self, other, out, ws, ex); }
aclnnStatus aclnnVdot(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnDot(ws, wsz, e, s); }
aclnnStatus aclnnInnerGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnDotGetWorkspaceSize(self, other, out, ws, ex); }
aclnnStatus aclnnInner(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnDot(ws, wsz, e, s); }
// ---- Outer: a[M] outer b[N] -> [M,N] ----
aclnnStatus aclnnOuterGetWorkspaceSize(const aclTensor *self, const aclTensor *vec2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !vec2 || !out || !ex || self->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=vec2; e->out=out; e->m=self->numel(); e->n=vec2->numel();
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnOuter(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t M=e->m,N=e->n; k_outer<<<nb(M*N),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,M,N);
    return done(e);
}
// ---- Kron (2D) ----
aclnnStatus aclnnKronGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex || self->dtype != ACL_FLOAT || self->viewDims.size() != 2 || other->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op=0; e->a=self; e->b=other; e->out=out;
    e->m=self->viewDims[0]; e->n=self->viewDims[1]; e->k=other->viewDims[0]; e->reduceCount=other->viewDims[1];
    if (ws) *ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnKron(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t M=e->m,N=e->n,P=e->k,Q=e->reduceCount; k_kron<<<nb(M*P*N*Q),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,M,N,P,Q);
    return done(e);
}

} // extern "C"
} // namespace _blas_ext

namespace _blas2_ext {
// BLAS remainder (R3): Tensordot (contract last-k of A with first-k of B) and Einsum (2-operand, general:
// permute each operand to [batch, free, contract] -> batched matmul -> permute to output label order). fp32.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
// generic permute (out contiguous): out[i] = in[ Σ coord_k * istr[perm-inverse...] ]; here we map out-dim d <- in-dim perm[d]
struct PG{ int rank; int64_t od[8],istr[8]; };
__global__ void k_perm(const float*in,float*o,PG d,int64_t n){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; int64_t rem=i,off=0; for(int k=d.rank-1;k>=0;--k){int64_t c=rem%d.od[k]; rem/=d.od[k]; off+=c*d.istr[k];} o[i]=in[off]; }
__global__ void k_bmm(const float*A,const float*B,float*C,int64_t Bt,int64_t M,int64_t K,int64_t N){ int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=Bt*M*N)return; int64_t n=i%N,m=(i/N)%M,b=i/(M*N); double s=0; const float*a=A+b*M*K,*bb=B+b*K*N; for(int64_t k=0;k<K;k++) s+=(double)a[m*K+k]*bb[k*N+n]; C[i]=(float)s; }
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
// permute helper: produce contiguous [in dims reordered by perm]; perm[d] = source axis of in for out axis d
static float* permute_dev(const float*in, const std::vector<int64_t>&indims, const std::vector<int>&perm, cudaStream_t s){
    int r=(int)indims.size(); int64_t istr[8],acc=1; for(int i=r-1;i>=0;--i){istr[i]=acc; acc*=indims[i];}
    PG d; d.rank=r; int64_t n=1; for(int i=0;i<r;i++){ d.od[i]=indims[perm[i]]; d.istr[i]=istr[perm[i]]; n*=d.od[i]; }
    float*o; cudaMalloc(&o,n*4); k_perm<<<nb(n),TH,0,s>>>(in,o,d,n); return o;
}
} // namespace

extern "C" {

// Tensordot: contract last `naxes` dims of A with first `naxes` dims of B -> [A free..., B free...]
aclnnStatus aclnnTensordotGetWorkspaceSize(const aclTensor*A,const aclTensor*B,int64_t naxes,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!A||!B||!out||!ex||A->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    int ra=(int)A->viewDims.size(), rb=(int)B->viewDims.size();
    int64_t C=1; for(int i=0;i<naxes;i++) C*= A->viewDims[ra-naxes+i];
    int64_t Fa=A->numel()/C, Fb=B->numel()/C;
    auto*e=new aclOpExecutor(); e->a=A; e->b=B; e->out=out; e->n=Fa; e->reduceCount=C; e->k=Fb; e->m=1; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTensordot(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t Fa=e->n,C=e->reduceCount,Fb=e->k;
    k_bmm<<<nb(Fa*Fb),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(const float*)e->b->data,(float*)e->out->data,1,Fa,C,Fb);
    return done(e);
}

// Einsum (2-operand): equation like "ij,jk->ik". General contraction via permute->bmm->permute.
static std::map<aclOpExecutor*,std::string> g_einsum_eq;   // equation stashed between GetWorkspaceSize and run
aclnnStatus aclnnEinsumGetWorkspaceSize(const char*equation,const aclTensor*A,const aclTensor*B,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!equation||!A||!B||!out||!ex||A->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=A; e->b=B; e->out=out; g_einsum_eq[e]=equation;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnEinsum(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    auto st=(cudaStream_t)s; const aclTensor*A=e->a,*B=e->b;
    std::string eq=g_einsum_eq[e]; g_einsum_eq.erase(e);
    // parse "la,lb->lo"
    size_t comma=eq.find(','), arrow=eq.find("->");
    std::string la=eq.substr(0,comma), lb=eq.substr(comma+1,arrow-comma-1), lo=eq.substr(arrow+2);
    auto trim=[](std::string&x){ std::string y; for(char c:x) if(c!=' ')y+=c; x=y; }; trim(la); trim(lb); trim(lo);
    int ra=la.size(), rb=lb.size(), ro=lo.size();
    auto inset=[](const std::string&s,char c){ return s.find(c)!=std::string::npos; };
    std::string batch,fa,fb,contract;
    for(char c:la){ if(inset(lb,c)&&inset(lo,c)) batch+= (inset(batch,c)?"":std::string(1,c)); else if(inset(lo,c)) fa+=c; else if(inset(lb,c)) contract+=c; }
    for(char c:lb){ if(inset(lo,c)&&!inset(la,c)) fb+=c; }
    // sizes per label
    std::map<char,int64_t> sz; for(int i=0;i<ra;i++) sz[la[i]]=A->viewDims[i]; for(int i=0;i<rb;i++) sz[lb[i]]=B->viewDims[i];
    auto prod=[&](const std::string&s){ int64_t p=1; for(char c:s)p*=sz[c]; return p; };
    int64_t Bt=prod(batch),Fa=prod(fa),Fb=prod(fb),C=prod(contract);
    // permute A to [batch,fa,contract]; perm[d]=index in la of that label
    auto mkperm=[&](const std::string&src,const std::string&order){ std::vector<int> p; for(char c:order){ p.push_back((int)src.find(c)); } return p; };
    std::vector<int64_t> Ad(A->viewDims), Bd(B->viewDims);
    float*Ap=permute_dev((const float*)A->data,Ad,mkperm(la,batch+fa+contract),st);
    float*Bp=permute_dev((const float*)B->data,Bd,mkperm(lb,batch+contract+fb),st);
    float*tmp; cudaMalloc(&tmp,Bt*Fa*Fb*4);
    k_bmm<<<nb(Bt*Fa*Fb),TH,0,st>>>(Ap,Bp,tmp,Bt,Fa,C,Fb);
    // tmp natural order labels = batch+fa+fb ; permute to lo
    std::string nat=batch+fa+fb; std::vector<int64_t> natd; for(char c:nat) natd.push_back(sz[c]);
    std::vector<int> fp=mkperm(nat,lo);
    if(nat==lo){ cudaMemcpyAsync(e->out->data,tmp,(size_t)Bt*Fa*Fb*4,cudaMemcpyDeviceToDevice,st); }
    else { float*op=permute_dev(tmp,natd,fp,st); cudaMemcpyAsync(e->out->data,op,(size_t)Bt*Fa*Fb*4,cudaMemcpyDeviceToDevice,st); cudaStreamSynchronize(st); cudaFree(op); }
    cudaStreamSynchronize(st); cudaFree(Ap);cudaFree(Bp);cudaFree(tmp);
    return done(e);
}

} // extern "C"
} // namespace _blas2_ext

namespace _blas3_ext {
// BLAS-3 GEMM (aclnnGemm): out = alpha·op(A)·op(B) + beta·C, op = transpose per transA/transB.
// fp32/fp16/bf16, fp32 accumulation. Correctness-first (perf deferred); handles transpose in-kernel.

namespace {

// out[M,N] = alpha·Σ_k opA(m,k)·opB(k,n) + beta·C[M,N]
// A is [M,K] (ta=0) or [K,M] (ta=1); B is [K,N] (tb=0) or [N,K] (tb=1). C/out row-major [M,N].
template <typename T>
__global__ void k_gemm(const T *A, const T *B, const T *C, T *O,
                       int64_t M, int64_t N, int64_t K, int ta, int tb, float alpha, float beta) {
    int64_t n = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t m = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) return;
    float acc = 0.f;
    for (int64_t k = 0; k < K; ++k) {
        float a = ta ? (float)A[k * M + m] : (float)A[m * K + k];
        float b = tb ? (float)B[n * K + k] : (float)B[k * N + n];
        acc += a * b;
    }
    float r = alpha * acc + (beta != 0.f && C ? beta * (float)C[m * N + n] : 0.f);
    O[m * N + n] = (T)r;
}

inline aclnnStatus fin(aclOpExecutor *e) { aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

} // namespace

extern "C" {

// aclnnGemm: a,b inputs; c optional bias matrix; alpha/beta scalars; transA/transB flags; out[M,N].
aclnnStatus aclnnGemmGetWorkspaceSize(const aclTensor *a, const aclTensor *b, const aclTensor *c,
                                      float alpha, float beta, int64_t transA, int64_t transB,
                                      aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType;
    if (!a || !b || !out || !ex || !a->data || !b->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (a->dtype != b->dtype || a->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (a->viewDims.size() != 2 || b->viewDims.size() != 2 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    int64_t M = transA ? a->viewDims[1] : a->viewDims[0];
    int64_t Ka = transA ? a->viewDims[0] : a->viewDims[1];
    int64_t Kb = transB ? b->viewDims[1] : b->viewDims[0];
    int64_t N = transB ? b->viewDims[0] : b->viewDims[1];
    if (Ka != Kb || out->viewDims[0] != M || out->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;
    if (c && beta != 0.f && (c->viewDims.size() != 2 || c->viewDims[0] != M || c->viewDims[1] != N)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = a; e->b = b; e->c = c; e->out = out;
    e->m = M; e->n = N; e->k = Ka; e->dim = (int)transA; e->reduceCount = transB;
    e->dscalars = {(double)alpha, (double)beta};
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGemm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    int64_t M = e->m, N = e->n, K = e->k; int ta = e->dim, tb = (int)e->reduceCount;
    float alpha = (float)e->dscalars[0], beta = (float)e->dscalars[1];
    auto st = (cudaStream_t)s; dim3 tb3(16, 16), g((N + 15) / 16, (M + 15) / 16);
    const void *C = (e->c && beta != 0.f) ? e->c->data : nullptr;
    switch (e->a->dtype) {
        case ACL_FLOAT:   k_gemm<float><<<g,tb3,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)C,(float*)e->out->data,M,N,K,ta,tb,alpha,beta); break;
        case ACL_FLOAT16: k_gemm<__half><<<g,tb3,0,st>>>((const __half*)e->a->data,(const __half*)e->b->data,(const __half*)C,(__half*)e->out->data,M,N,K,ta,tb,alpha,beta); break;
        case ACL_BF16:    k_gemm<__nv_bfloat16><<<g,tb3,0,st>>>((const __nv_bfloat16*)e->a->data,(const __nv_bfloat16*)e->b->data,(const __nv_bfloat16*)C,(__nv_bfloat16*)e->out->data,M,N,K,ta,tb,alpha,beta); break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    return fin(e);
}

} // extern "C"
} // namespace _blas3_ext

namespace _blas4_ext {
// GroupedMatmulAdd: per-expert GEMM (x[M,K] partitioned by groupList @ weight[E,K,N]) plus a residual y[M,N].
// Correctness-first naive per-group kernel (perf deferred); fp32/fp16/bf16, fp32 accumulation.

namespace {
// out[rowOff+rr, col] = Σ_k x[m,k]·w[eIdx,k,col] + (y ? y[m,col] : 0)
template <typename T>
__global__ void k_gmm_add(const T *x, const T *w, const T *y, T *out, int64_t rowOff, int64_t Mr, int64_t K, int64_t N, int64_t eIdx) {
    int64_t col = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t rr  = (int64_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (rr >= Mr || col >= N) return; int64_t m = rowOff + rr;
    float acc = 0.f; for (int64_t k = 0; k < K; ++k) acc += (float)x[m*K+k] * (float)w[(eIdx*K+k)*N+col];
    out[m*N+col] = (T)(acc + (y ? (float)y[m*N+col] : 0.f));
}
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

// GroupedMatmulAdd: x[M,K], weight[E,K,N], y[M,N] (residual, nullable), groupList[E] (row counts) -> out[M,N].
aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y,
        const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !groupList || !out || !ex || !x->data || !weight->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = x->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 2 || weight->viewDims.size() != 3 || out->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    const int64_t M = x->viewDims[0], K = x->viewDims[1], E = weight->viewDims[0], N = weight->viewDims[2];
    if (weight->viewDims[1] != K || out->viewDims[0] != M || out->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID;
    if ((int64_t)groupList->v.size() != E) return ACLNN_ERR_PARAM_INVALID;
    int64_t sum = 0; for (auto g : groupList->v) sum += g;
    if (sum != M) return ACLNN_ERR_PARAM_INVALID;
    if (y && (y->dtype != dt || y->viewDims != out->viewDims)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = y; e->out = out; e->axes = groupList->v; e->n = N; e->k = K;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGroupedMatmulAdd(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    auto st = (cudaStream_t)s; int64_t N = e->n, K = e->k;
    const void *xp = e->a->data, *wp = e->b->data, *yp = e->c ? e->c->data : nullptr; void *op = e->out->data;
    int64_t off = 0; dim3 tb(16, 16);
    for (size_t ei = 0; ei < e->axes.size(); ++ei) {
        int64_t Mr = e->axes[ei];
        if (Mr > 0) {
            dim3 g((N + 15) / 16, (Mr + 15) / 16);
            switch (e->a->dtype) {
                case ACL_FLOAT:   k_gmm_add<float><<<g,tb,0,st>>>((const float*)xp,(const float*)wp,(const float*)yp,(float*)op,off,Mr,K,N,ei); break;
                case ACL_FLOAT16: k_gmm_add<__half><<<g,tb,0,st>>>((const __half*)xp,(const __half*)wp,(const __half*)yp,(__half*)op,off,Mr,K,N,ei); break;
                default:          k_gmm_add<__nv_bfloat16><<<g,tb,0,st>>>((const __nv_bfloat16*)xp,(const __nv_bfloat16*)wp,(const __nv_bfloat16*)yp,(__nv_bfloat16*)op,off,Mr,K,N,ei); break;
            }
        }
        off += Mr;
    }
    return fin(e);
}
// V2: same simplified contract.
aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y,
        const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGroupedMatmulAddGetWorkspaceSize(x, weight, y, groupList, out, ws, ex);
}
aclnnStatus aclnnGroupedMatmulAddV2(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnGroupedMatmulAdd(ws, wsz, e, s); }

} // extern "C"
} // namespace _blas4_ext

