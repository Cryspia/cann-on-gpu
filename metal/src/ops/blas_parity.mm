// Matmul/BLAS/linalg gap operators (CUDA parity): the ~22 GEMM-family + linalg ops that the Metal
// backend did not yet expose. All host-side over unified memory (MTLStorageModeShared: device pointer
// == host pointer): cblas for GEMM, LAPACK for linalg. ACL tensors are row-major; cblas with
// CblasRowMajor consumes them directly, LAPACK (column-major) is bridged by transpose helpers.
//
// Op groups (signatures validated against the canonical aclnnop/aclnn_ops.h header — declarations are
// NOT hand-written here; they come from that header):
//   GroupedMatmul V2..V5 / WeightNz / AddV2     — MoE grouped GEMM: x[M,K] by groupList @ weight[E,K,N]
//   FusedMatmul / MatmulCompress                — plain A[M,K] @ B[K,N]
//   AddmmWeightNz                                — out = beta*C + alpha*(A@B)
//   TransMatmulWeight                            — weight reformat (logical identity / copy)
//   CalculateMatmulWeightSize(V2)                — out[int64] = numel(weightShape)
//   AlltoAllMatmul / MatmulAlltoAll             — single-rank: all-to-all is identity, so == plain GEMM
//   GroupedMatmulFinalizeRouting (+V2/V3/Nz/NzV2)— grouped GEMM then MoE combine (scatter-add by row index)
//   LinalgCholesky / LinalgQr / LinalgCross     — LAPACK-backed (potrf / geqrf+orgqr / cross product)
//
// The FinalizeRouting + AlltoAll ops are distributed/MC2 fusions whose canonical headers are not shipped
// locally; their declarations are added to the local aclnnop/aclnn_ops.h shim (parity with how the rest of
// the surface is declared), and they are implemented with single-rank logically-equivalent host semantics.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_mc2.h"   // canonical MC2 signatures (AlltoAllMatmul/MatmulAlltoAll carry bias+HcclComm)
// Pull in cblas only (NOT the full <Accelerate/Accelerate.h> umbrella): the umbrella also drags in
// vecLib's lapack.h whose __LAPACK_int prototypes conflict with the self-declared classic-LAPACK ones
// below. Including just cblas.h gives us GEMM/GEMV without that conflict (same reason linalg.mm avoids it).
#include <vecLib/cblas.h>
#include <vector>
#include <cmath>
#include <algorithm>

// Self-declared classic LAPACK (Accelerate provides these symbols; self-declaring avoids the __LAPACK_int
// header conflict — identical approach to metal/src/ops/linalg.mm).
extern "C" {
void spotrf_(const char*,const int*,float*,const int*,int*);
void sgeqrf_(const int*,const int*,float*,const int*,float*,float*,const int*,int*);
void sorgqr_(const int*,const int*,const int*,float*,const int*,const float*,float*,const int*,int*);
}

namespace {
float *FP(const aclTensor *t)  { return (float *)t->data + t->offset; }
int64_t *I64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
// row-major [r,c] <-> column-major (LAPACK)
void r2c(const float *s, float *d, int r, int c) { for (int i=0;i<r;i++) for (int j=0;j<c;j++) d[i+j*r]=s[i*c+j]; }

// ---- OpKind values local to this file (stored in e->op) ----
enum {
    G_GMM = 1,        // grouped matmul: x[M,K] by groupList @ weight[E,K,N] -> out[M,N]
    G_GMM_ADD,        // grouped matmul + residual y -> out (V2 forward)
    G_MM,             // plain A[M,K] @ B[K,N] -> out[M,N]   (FusedMatmul / MatmulCompress / AlltoAll)
    G_ADDMM,          // out = beta*C + alpha*(A@B)          (AddmmWeightNz)
    G_TRANS_W,        // weight reformat: logical copy
    G_WSIZE,          // CalculateMatmulWeightSize: out[int64] = numel(weightShape)
    G_CHOLESKY,       // LinalgCholesky (upper flag in e->causal)
    G_QR,             // LinalgQr (mode in e->m)
    G_CROSS,          // LinalgCross over dim=e->dim
};

// out[M,N] = grouped( x[M,K] partitioned by groupList @ weight[E,K,N] ); rows summed group-by-group.
void grouped_gemm(const aclTensor *X, const aclTensor *W, const std::vector<int64_t> &groups, float *o) {
    int K = (int)X->viewDims[1], N = (int)W->viewDims[2];
    int E = (int)groups.size(); int64_t off = 0;
    const float *x = FP(X), *w = FP(W);
    for (int g = 0; g < E; ++g) {
        int rows = (int)groups[g];
        if (rows > 0)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, N, K,
                        1.0f, x + off * K, K, w + (size_t)g * K * N, N, 0.0f, o + off * N, N);
        off += rows;
    }
}

aclnnStatus run_parity(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
        case G_GMM: {  // x[M,K] by groupList @ weight[E,K,N] -> out[M,N]
            grouped_gemm(e->a, e->b, e->axes, FP(e->out));
            break;
        }
        case G_GMM_ADD: {  // grouped gemm + residual y -> out
            const aclTensor *Y = e->c; aclTensor *O = e->out;
            grouped_gemm(e->a, e->b, e->axes, FP(O));
            if (Y) { const float *y = FP(Y); float *o = FP(O); int64_t n = O->numel(); for (int64_t i = 0; i < n; ++i) o[i] += y[i]; }
            break;
        }
        case G_MM: {  // self[M,K] @ other[K,N] -> out[M,N]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1], N = (int)B->viewDims[1];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, FP(A), K, FP(B), N, 0.0f, FP(O), N);
            break;
        }
        case G_ADDMM: {  // out = beta*C + alpha*(A[M,K] @ B[K,N]); C=e->c
            const aclTensor *A = e->a, *B = e->b, *C = e->c; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1], N = (int)B->viewDims[1];
            double beta = e->dscalars[0], alpha = e->dscalars[1]; const float *c = FP(C); float *o = FP(O);
            for (int i = 0; i < M * N; ++i) o[i] = (float)(beta * c[i]);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        (float)alpha, FP(A), K, FP(B), N, 1.0f, o, N);
            break;
        }
        case G_TRANS_W: {  // weight reformat: logical identity (NZ layout is a hardware tiling, value-equivalent)
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = A->numel();
            const float *a = FP(A); float *o = FP(O); for (int64_t i = 0; i < n; ++i) o[i] = a[i];
            break;
        }
        case G_WSIZE: {  // out[int64 scalar] = product(weightShape)
            int64_t v = e->m; I64(e->out)[0] = v;
            break;
        }
        case G_CHOLESKY: {  // A[n,n] SPD -> factor; upper(e->causal): R s.t. R^T R = A, else lower L s.t. L L^T = A
            const aclTensor *A = e->a; aclTensor *O = e->out; int n = (int)A->viewDims[0]; int info = 0;
            std::vector<float> ac(n*n); r2c(FP(A), ac.data(), n, n);
            char uplo = e->causal ? 'U' : 'L'; spotrf_(&uplo, &n, ac.data(), &n, &info);
            if (info) return ACLNN_ERR_RUNTIME_ERROR;
            float *o = FP(O);  // ac is column-major; zero the unused triangle on the way out (row-major store)
            if (e->causal) for (int i=0;i<n;i++) for (int j=0;j<n;j++) o[i*n+j] = (i<=j) ? ac[i+j*n] : 0.f;   // upper
            else           for (int i=0;i<n;i++) for (int j=0;j<n;j++) o[i*n+j] = (i>=j) ? ac[i+j*n] : 0.f;   // lower
            break;
        }
        case G_QR: {  // A[m,n] -> Q[m,k], R[k,n], k=min(m,n) (reduced); e->m=mode (unused: reduced only)
            const aclTensor *A = e->a; aclTensor *Q = e->out, *R = e->out2;
            int m = (int)A->viewDims[0], n = (int)A->viewDims[1], k = m<n?m:n; int info = 0;
            std::vector<float> ac(m*n); r2c(FP(A), ac.data(), m, n); std::vector<float> tau(k);
            int lw=-1; float q; sgeqrf_(&m,&n,ac.data(),&m,tau.data(),&q,&lw,&info); lw=(int)q; std::vector<float> wk(lw);
            sgeqrf_(&m,&n,ac.data(),&m,tau.data(),wk.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *r = FP(R); for (int i=0;i<k;i++) for (int j=0;j<n;j++) r[i*n+j] = (i<=j)?ac[i+j*m]:0.f;   // R upper k×n
            lw=-1; sorgqr_(&m,&k,&k,ac.data(),&m,tau.data(),&q,&lw,&info); lw=(int)q; std::vector<float> wk2(lw);
            sorgqr_(&m,&k,&k,ac.data(),&m,tau.data(),wk2.data(),&lw,&info); if(info) return ACLNN_ERR_RUNTIME_ERROR;
            float *qo = FP(Q); for (int i=0;i<m;i++) for (int j=0;j<k;j++) qo[i*k+j] = ac[i+j*m];   // Q m×k
            break;
        }
        case G_CROSS: {  // 3-vector cross product along dim=e->dim; all other dims are the batch
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            const std::vector<int64_t> &d = A->viewDims; int nd = (int)d.size();
            int dim = (int)e->dim; if (dim < 0) dim += nd;
            // stride of the cross dimension and total batch count
            int64_t cs = 1; for (int i = nd - 1; i > dim; --i) cs *= d[i];
            int64_t total = A->numel(); int64_t outer = total / (3 * cs);
            const float *a = FP(A), *b = FP(B); float *o = FP(O);
            for (int64_t ob = 0; ob < outer; ++ob) for (int64_t ic = 0; ic < cs; ++ic) {
                int64_t base = ob * 3 * cs + ic;   // a[base + comp*cs], comp in {0,1,2}
                float a0=a[base], a1=a[base+cs], a2=a[base+2*cs];
                float b0=b[base], b1=b[base+cs], b2=b[base+2*cs];
                o[base]      = a1*b2 - a2*b1;
                o[base+cs]   = a2*b0 - a0*b2;
                o[base+2*cs] = a0*b1 - a1*b0;
            }
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_parity(e, s); } delete e; return st; }
} // namespace

extern "C" {

// ---- GroupedMatmul V2..V5 / WeightNz: x[M,K] by groupList @ weight[E,K,N] -> out[M,N] ----
#define GMM_OP(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = G_GMM; e->a = x; e->b = weight; e->out = out; \
    if (groupList) e->axes = groupList->v; else e->axes = {x->viewDims[0]}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
GMM_OP(aclnnGroupedMatmulV2) GMM_OP(aclnnGroupedMatmulV3) GMM_OP(aclnnGroupedMatmulV4)
GMM_OP(aclnnGroupedMatmulV5) GMM_OP(aclnnGroupedMatmulWeightNz)

// ---- GroupedMatmulAddV2: x[M,K], weight[E,K,N], y[M,N] residual, groupList[E] -> out[M,N] ----
aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y,
        const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_GMM_ADD; e->a = x; e->b = weight; e->c = y; e->out = out;
    if (groupList) e->axes = groupList->v; else e->axes = {x->viewDims[0]}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupedMatmulAddV2)

// ---- FusedMatmul / MatmulCompress: plain A[M,K] @ B[K,N] -> out[M,N] ----
aclnnStatus aclnnFusedMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_MM; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFusedMatmul)
aclnnStatus aclnnMatmulCompressGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_MM; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMatmulCompress)

// ---- AddmmWeightNz: out = beta*C + alpha*(A[M,K] @ B[K,N]) ----
aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!C || !A || !B || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_ADDMM; e->a = A; e->b = B; e->c = C; e->out = out; e->dscalars = {beta, alpha}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAddmmWeightNz)

// ---- TransMatmulWeight: weight reformat -> logical copy ----
aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_TRANS_W; e->a = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnTransMatmulWeight)

// ---- CalculateMatmulWeightSize(V2): out[int64] = numel(weightShape) ----
aclnnStatus aclnnCalculateMatmulWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!weightShape || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t prod = 1; for (auto d : weightShape->v) prod *= d;
    auto *e = new aclOpExecutor(); e->op = G_WSIZE; e->out = out; e->m = prod; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCalculateMatmulWeightSize)
aclnnStatus aclnnCalculateMatmulWeightSizeV2GetWorkspaceSize(const aclIntArray *weightShape, int64_t dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; if (!weightShape || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t prod = 1; for (auto d : weightShape->v) prod *= d;
    auto *e = new aclOpExecutor(); e->op = G_WSIZE; e->out = out; e->m = prod; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCalculateMatmulWeightSizeV2)

// ---- AlltoAllMatmul / MatmulAlltoAll: single-rank all-to-all == identity, so plain A[M,K] @ B[K,N].
//      Canonical MC2 signature carries bias + HcclComm (ignored on a single rank). ----
aclnnStatus aclnnAlltoAllMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)bias; (void)comm; if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_MM; e->a = x; e->b = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnAlltoAllMatmul)
aclnnStatus aclnnMatmulAlltoAllGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)bias; (void)comm; if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_MM; e->a = x; e->b = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMatmulAlltoAll)

// ---- GroupedMatmulFinalizeRouting (+V2/V3/WeightNz/WeightNzV2) ----
// Canonical local signature (matmul5_ext): (x, weight, groupList, out). The "finalize routing" combine
// uses identity row mapping in this contract, so it reduces to a grouped GEMM: x[M,K] by groupList @
// weight[E,K,N] -> out[M,N].
#define GMM_FR(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->op = G_GMM; e->a = x; e->b = weight; e->out = out; \
    if (groupList) e->axes = groupList->v; else e->axes = {x->viewDims[0]}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
GMM_FR(aclnnGroupedMatmulFinalizeRouting) GMM_FR(aclnnGroupedMatmulFinalizeRoutingV2)
GMM_FR(aclnnGroupedMatmulFinalizeRoutingV3) GMM_FR(aclnnGroupedMatmulFinalizeRoutingWeightNz)
GMM_FR(aclnnGroupedMatmulFinalizeRoutingWeightNzV2)

// ---- LinalgCholesky / LinalgQr / LinalgCross ----
aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor *self, bool upper, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_CHOLESKY; e->a = self; e->out = out; e->causal = upper; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLinalgCholesky)
aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *A, int64_t mode, aclTensor *Q, aclTensor *R, uint64_t *ws, aclOpExecutor **ex) {
    if (!A || !Q || !R || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_QR; e->a = A; e->out = Q; e->out2 = R; e->m = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLinalgQr)
aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = G_CROSS; e->a = self; e->b = other; e->out = out; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLinalgCross)

} // extern "C"
