// test_remainder R2/R3: linalg remainder (MatrixRank, Lstsq, MatrixPower) via Accelerate LAPACK/BLAS,
// and BLAS remainder (Tensordot, Einsum). Host-side over unified memory; LAPACK is column-major.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <cstring>
#include <string>

// Self-declared LAPACK (Accelerate provides these; avoid including the Accelerate header which declares
// these with the legacy __LAPACK_int signature that conflicts with the const-int form used here).
extern "C" {
void sgesvd_(const char*,const char*,const int*,const int*,float*,const int*,float*,float*,const int*,float*,const int*,float*,const int*,int*);
void sgels_(const char*,const int*,const int*,const int*,float*,const int*,float*,const int*,float*,const int*,int*);
}

namespace {
float *fp(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *ip64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
void r2c(const float *s, float *d, int r, int c) { for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) d[i + j * r] = s[i * c + j]; }
void c2r(const float *s, float *d, int r, int c) { for (int i = 0; i < r; i++) for (int j = 0; j < c; j++) d[i * c + j] = s[i + j * r]; }

enum { K_MATRIXRANK, K_LSTSQ, K_MATRIXPOWER, K_TENSORDOT, K_EINSUM };

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s); int info = 0;
    switch (e->op) {
    case K_MATRIXRANK: {   // SVD singular values; count those above tol (or default tol)
        const aclTensor *A = e->a; int m = (int)A->viewDims[0], n = (int)A->viewDims[1], mn = m < n ? m : n;
        std::vector<float> ac(m * n), S(mn); r2c(fp(A), ac.data(), m, n); const char JN = 'N';
        int lw = -1; float q; sgesvd_(&JN, &JN, &m, &n, ac.data(), &m, S.data(), nullptr, &m, nullptr, &n, &q, &lw, &info); lw = (int)q;
        std::vector<float> wk(lw); sgesvd_(&JN, &JN, &m, &n, ac.data(), &m, S.data(), nullptr, &m, nullptr, &n, wk.data(), &lw, &info);
        if (info) return ACLNN_ERR_RUNTIME_ERROR;
        double smax = mn > 0 ? S[0] : 0; double tol = e->alpha > 0 ? e->alpha : smax * std::max(m, n) * 1.1920929e-7;
        int64_t rank = 0; for (int i = 0; i < mn; i++) if (S[i] > tol) rank++;
        ip64(e->out)[0] = rank; return ACLNN_SUCCESS;
    }
    case K_LSTSQ: {   // min ||A X - B|| ; A[m,n] B[m,nrhs] X[n,nrhs] via sgels
        const aclTensor *A = e->a, *B = e->b; aclTensor *X = e->out;
        int m = (int)A->viewDims[0], n = (int)A->viewDims[1], nrhs = (int)B->viewDims[1];
        std::vector<float> ac(m * n); r2c(fp(A), ac.data(), m, n);
        int ldb = m > n ? m : n; std::vector<float> bc(ldb * nrhs, 0); r2c(fp(B), bc.data(), m, nrhs);  // m rows used; col-major ldb
        // bc currently laid out with leading dim m from r2c; re-lay with leading dim ldb
        std::vector<float> bb(ldb * nrhs, 0);
        const float *Bp = fp(B); for (int i = 0; i < m; i++) for (int j = 0; j < nrhs; j++) bb[i + j * ldb] = Bp[i * nrhs + j];
        const char N = 'N'; int lw = -1; float q;
        sgels_(&N, &m, &n, &nrhs, ac.data(), &m, bb.data(), &ldb, &q, &lw, &info); lw = (int)q;
        std::vector<float> wk(lw); sgels_(&N, &m, &n, &nrhs, ac.data(), &m, bb.data(), &ldb, wk.data(), &lw, &info);
        if (info) return ACLNN_ERR_RUNTIME_ERROR;
        float *Xp = fp(X); for (int i = 0; i < n; i++) for (int j = 0; j < nrhs; j++) Xp[i * nrhs + j] = bb[i + j * ldb];
        return ACLNN_SUCCESS;
    }
    case K_MATRIXPOWER: {   // A^p, p>=1, row-major matmuls
        const aclTensor *A = e->a; aclTensor *O = e->out; int n = (int)A->viewDims[0]; int p = (int)e->dim;
        const float *a = fp(A); std::vector<double> M(n * n), T(n * n);
        for (int i = 0; i < n * n; i++) M[i] = a[i];
        for (int s2 = 1; s2 < p; s2++) {
            for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) { double acc = 0; for (int k = 0; k < n; k++) acc += M[i * n + k] * a[k * n + j]; T[i * n + j] = acc; }
            M = T;
        }
        float *o = fp(O); for (int i = 0; i < n * n; i++) o[i] = (float)M[i];
        return ACLNN_SUCCESS;
    }
    case K_TENSORDOT: {   // contract last `naxes` of A with first `naxes` of B
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int naxes = (int)e->dim;
        int andim = (int)A->viewDims.size(), bndim = (int)B->viewDims.size();
        int64_t M = 1; for (int d = 0; d < andim - naxes; d++) M *= A->viewDims[d];
        int64_t Kc = 1; for (int d = andim - naxes; d < andim; d++) Kc *= A->viewDims[d];
        int64_t Nn = 1; for (int d = naxes; d < bndim; d++) Nn *= B->viewDims[d];
        const float *a = fp(A), *b = fp(B); float *o = fp(O);
        for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < Nn; j++) { double acc = 0; for (int64_t k = 0; k < Kc; k++) acc += (double)a[i * Kc + k] * b[k * Nn + j]; o[i * Nn + j] = (float)acc; }
        return ACLNN_SUCCESS;
    }
    case K_EINSUM: {   // two-operand einsum "<la>,<lb>-><lo>" general; brute-force over all index labels
        const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
        std::string eq; for (double d : e->dscalars) eq.push_back((char)d);
        size_t comma = eq.find(','), arrow = eq.find("->");
        std::string la = eq.substr(0, comma), lb = eq.substr(comma + 1, arrow - comma - 1), lo = eq.substr(arrow + 2);
        // collect dimension sizes per label
        int dimsz[128] = {0};
        for (size_t i = 0; i < la.size(); i++) dimsz[(int)la[i]] = (int)A->viewDims[i];
        for (size_t i = 0; i < lb.size(); i++) dimsz[(int)lb[i]] = (int)B->viewDims[i];
        // free labels = those in lo; sum labels = labels in la|lb not in lo
        std::string allLab; for (char c : la) if (allLab.find(c) == std::string::npos) allLab.push_back(c);
        for (char c : lb) if (allLab.find(c) == std::string::npos) allLab.push_back(c);
        std::string sumLab; for (char c : allLab) if (lo.find(c) == std::string::npos) sumLab.push_back(c);
        const float *a = fp(A), *b = fp(B); float *o = fp(O);
        int64_t outN = O->numel(); for (int64_t i = 0; i < outN; i++) o[i] = 0.f;
        // iterate over output coordinates
        std::vector<int> labval(128, 0);
        std::vector<int> outDims(lo.size()); for (size_t i = 0; i < lo.size(); i++) outDims[i] = dimsz[(int)lo[i]];
        int64_t sumN = 1; for (char c : sumLab) sumN *= dimsz[(int)c];
        auto stridesOf = [](const aclTensor *T) { std::vector<int64_t> st(T->viewDims.size()); int64_t s = 1; for (int d = (int)T->viewDims.size() - 1; d >= 0; d--) { st[d] = s; s *= T->viewDims[d]; } return st; };
        auto sa = stridesOf(A), sb = stridesOf(B);
        for (int64_t oi = 0; oi < outN; oi++) {
            int64_t rem = oi; for (int d = (int)lo.size() - 1; d >= 0; d--) { labval[(int)lo[d]] = (int)(rem % outDims[d]); rem /= outDims[d]; }
            double acc = 0;
            for (int64_t si = 0; si < sumN; si++) {
                int64_t r2 = si; for (int d = (int)sumLab.size() - 1; d >= 0; d--) { labval[(int)sumLab[d]] = (int)(r2 % dimsz[(int)sumLab[d]]); r2 /= dimsz[(int)sumLab[d]]; }
                int64_t ia = 0; for (size_t i = 0; i < la.size(); i++) ia += (int64_t)labval[(int)la[i]] * sa[i];
                int64_t ib = 0; for (size_t i = 0; i < lb.size(); i++) ib += (int64_t)labval[(int)lb[i]] * sb[i];
                acc += (double)a[ia] * b[ib];
            }
            o[oi] = (float)acc;
        }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUNFN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUNFN(rl_run)
} // namespace

extern "C" {
aclnnStatus aclnnMatrixRankGetWorkspaceSize(const aclTensor *A, double tol, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MATRIXRANK; e->a = A; e->out = out; e->alpha = tol; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMatrixRank)
aclnnStatus aclnnLstsqGetWorkspaceSize(const aclTensor *A, const aclTensor *B, aclTensor *X, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LSTSQ; e->a = A; e->b = B; e->out = X; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLstsq)
aclnnStatus aclnnMatrixPowerGetWorkspaceSize(const aclTensor *A, int64_t p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MATRIXPOWER; e->a = A; e->out = out; e->dim = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMatrixPower)
aclnnStatus aclnnTensordotGetWorkspaceSize(const aclTensor *A, const aclTensor *B, int64_t naxes, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_TENSORDOT; e->a = A; e->b = B; e->out = out; e->dim = naxes; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnTensordot)
aclnnStatus aclnnEinsumGetWorkspaceSize(const char *equation, const aclTensor *A, const aclTensor *B, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EINSUM; e->a = A; e->b = B; e->out = out; for (const char *p = equation; *p; p++) e->dscalars.push_back((double)*p); *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnEinsum)
} // extern "C"
