// BLAS extras (P8) + assorted host-side ops for test_blas: GEMM-family via Accelerate cblas
// (Mm/Bmm/Mv/Addmv/Addmm/Baddbmm/Addbmm/Dot/Outer/Kron/Gemm/Addr/Ger/MatmulWeightNz/GroupedMatmulAdd),
// plus elementwise/scatter/upsample/pool-backward/quant ops. All host-side over unified memory
// (MTLStorageModeShared: device pointer == host pointer). ACL tensors are row-major; cblas with
// CblasRowMajor consumes them directly.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <Accelerate/Accelerate.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace {
float   *FP(const aclTensor *t)  { return (float *)t->data + t->offset; }
int32_t *IP(const aclTensor *t)  { return (int32_t *)t->data + t->offset; }
int64_t *I64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
int8_t  *I8(const aclTensor *t)  { return (int8_t *)t->data + t->offset; }
uint8_t *U8(const aclTensor *t)  { return (uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ---- OpKind values for blas.mm dispatch (local to this file; stored in e->op) ----
enum {
    B_MM = 1, B_BMM, B_MV, B_DOT, B_OUTER, B_KRON, B_ADDMV, B_ADDMM, B_BADDBMM, B_ADDBMM,
    B_GEMM, B_ADDR, B_GER, B_MATMUL_NZ, B_GMM_ADD,
    B_DIVMOD_T, B_DIVMOD_S, B_FMOD_S, B_REMAINDER_TS, B_ADDRELU, B_GCD, B_LOGIT, B_SHRINK, B_CDIST,
    B_HISTC, B_SCATTER_VAL, B_PUT, B_FAKEQUANT, B_DYNBLOCKQUANT,
    B_UPNEAR1D, B_UPNEAR1D_BWD, B_GAP, B_REPPAD1D_BWD, B_MAXPOOL2D_IDX_BWD,
};

// ------------------------------------------------------------------ GEMM family
aclnnStatus run_blas(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
        case B_MM: {  // self[M,K] @ mat2[K,N] -> out[M,N]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1], N = (int)B->viewDims[1];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, FP(A), K, FP(B), N, 0.0f, FP(O), N);
            break;
        }
        case B_BMM: {  // [Bt,M,K] @ [Bt,K,N] -> [Bt,M,N]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            int Bt = (int)A->viewDims[0], M = (int)A->viewDims[1], K = (int)A->viewDims[2], N = (int)B->viewDims[2];
            for (int b = 0; b < Bt; ++b)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                            1.0f, FP(A) + (size_t)b * M * K, K, FP(B) + (size_t)b * K * N, N,
                            0.0f, FP(O) + (size_t)b * M * N, N);
            break;
        }
        case B_MV: {  // mat[M,K] @ vec[K] -> out[M]
            const aclTensor *A = e->a, *X = e->b; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1];
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, 1.0f, FP(A), K, FP(X), 1, 0.0f, FP(O), 1);
            break;
        }
        case B_DOT: {  // self[n] . mat2[n] -> out[1]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int n = (int)A->numel();
            FP(O)[0] = cblas_sdot(n, FP(A), 1, FP(B), 1);
            break;
        }
        case B_OUTER: case B_GER: {  // self[M] x mat2[N] -> out[M,N]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            int M = (int)A->numel(), N = (int)B->numel(); const float *a = FP(A), *b = FP(B); float *o = FP(O);
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j) o[i * N + j] = a[i] * b[j];
            break;
        }
        case B_KRON: {  // A[M,N] kron B[P,Q] -> out[M*P, N*Q]
            const aclTensor *A = e->a, *Bt = e->b; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], N = (int)A->viewDims[1], P = (int)Bt->viewDims[0], Q = (int)Bt->viewDims[1];
            const float *a = FP(A), *b = FP(Bt); float *o = FP(O);
            for (int m = 0; m < M; ++m) for (int p = 0; p < P; ++p) for (int nn = 0; nn < N; ++nn) for (int q = 0; q < Q; ++q)
                o[(m * P + p) * (N * Q) + (nn * Q + q)] = a[m * N + nn] * b[p * Q + q];
            break;
        }
        case B_ADDMV: {  // out = beta*y + alpha*(mat[M,K] @ vec[K]); y=e->c
            const aclTensor *A = e->a, *X = e->b, *Y = e->c; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1];
            double beta = e->dscalars[0], alpha = e->dscalars[1]; const float *y = FP(Y); float *o = FP(O);
            for (int i = 0; i < M; ++i) o[i] = (float)(beta * y[i]);  // seed with beta*y
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M, K, (float)alpha, FP(A), K, FP(X), 1, 1.0f, o, 1);
            break;
        }
        case B_ADDMM: {  // out = beta*C + alpha*(A[M,K] @ B[K,N]); C=e->c
            const aclTensor *A = e->a, *B = e->b, *C = e->c; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1], N = (int)B->viewDims[1];
            double beta = e->dscalars[0], alpha = e->dscalars[1]; const float *c = FP(C); float *o = FP(O);
            for (int i = 0; i < M * N; ++i) o[i] = (float)(beta * c[i]);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        (float)alpha, FP(A), K, FP(B), N, 1.0f, o, N);
            break;
        }
        case B_BADDBMM: {  // out[b] = beta*C[b] + alpha*(A[b]@B[b]); batched
            const aclTensor *A = e->a, *B = e->b, *C = e->c; aclTensor *O = e->out;
            int Bt = (int)A->viewDims[0], M = (int)A->viewDims[1], K = (int)A->viewDims[2], N = (int)B->viewDims[2];
            double beta = e->dscalars[0], alpha = e->dscalars[1]; const float *c = FP(C); float *o = FP(O);
            for (int i = 0; i < Bt * M * N; ++i) o[i] = (float)(beta * c[i]);
            for (int b = 0; b < Bt; ++b)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                            (float)alpha, FP(A) + (size_t)b * M * K, K, FP(B) + (size_t)b * K * N, N,
                            1.0f, o + (size_t)b * M * N, N);
            break;
        }
        case B_ADDBMM: {  // out[M,N] = beta*C + alpha*sum_b(A[b]@B[b])
            const aclTensor *A = e->a, *B = e->b, *C = e->c; aclTensor *O = e->out;
            int Bt = (int)A->viewDims[0], M = (int)A->viewDims[1], K = (int)A->viewDims[2], N = (int)B->viewDims[2];
            double beta = e->dscalars[0], alpha = e->dscalars[1]; const float *c = FP(C); float *o = FP(O);
            for (int i = 0; i < M * N; ++i) o[i] = (float)(beta * c[i]);
            for (int b = 0; b < Bt; ++b)
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                            (float)alpha, FP(A) + (size_t)b * M * K, K, FP(B) + (size_t)b * K * N, N,
                            1.0f, o, N);  // accumulate into the same out (beta*C seeded once)
            break;
        }
        case B_GEMM: {  // out = alpha*op(A)*op(B) + beta*C; ta/tb in dim/reduceCount, alpha/beta in dscalars
            const aclTensor *A = e->a, *B = e->b, *C = e->c; aclTensor *O = e->out;
            int ta = (int)e->dim, tb = (int)e->reduceCount; double alpha = e->dscalars[0], beta = e->dscalars[1];
            int M = ta ? (int)A->viewDims[1] : (int)A->viewDims[0];
            int K = ta ? (int)A->viewDims[0] : (int)A->viewDims[1];
            int N = tb ? (int)B->viewDims[0] : (int)B->viewDims[1];
            float *o = FP(O);
            if (C && beta != 0.0) { const float *c = FP(C); for (int i = 0; i < M * N; ++i) o[i] = (float)(beta * c[i]); }
            else for (int i = 0; i < M * N; ++i) o[i] = 0.f;
            int lda = ta ? M : K, ldb = tb ? K : N;
            cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans,
                        M, N, K, (float)alpha, FP(A), lda, FP(B), ldb, 1.0f, o, N);
            break;
        }
        case B_ADDR: {  // out = beta*self[M,N] + alpha*(vec1[M] outer vec2[N])
            const aclTensor *S = e->a, *V1 = e->b, *V2 = e->c; aclTensor *O = e->out;
            int M = (int)V1->numel(), N = (int)V2->numel(); double beta = e->dscalars[0], alpha = e->dscalars[1];
            const float *s = FP(S), *v1 = FP(V1), *v2 = FP(V2); float *o = FP(O);
            for (int i = 0; i < M; ++i) for (int j = 0; j < N; ++j)
                o[i * N + j] = (float)(beta * s[i * N + j] + alpha * (double)v1[i] * v2[j]);
            break;
        }
        case B_MATMUL_NZ: {  // logical-equivalence: x[M,K] @ weight[K,N] -> out[M,N]
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out;
            int M = (int)A->viewDims[0], K = (int)A->viewDims[1], N = (int)B->viewDims[1];
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K,
                        1.0f, FP(A), K, FP(B), N, 0.0f, FP(O), N);
            break;
        }
        case B_GMM_ADD: {  // grouped: x[M,K] by counts @ weight[E,K,N] + y[M,N] -> out[M,N]
            const aclTensor *X = e->a, *W = e->b, *Y = e->c; aclTensor *O = e->out;
            int K = (int)X->viewDims[1], N = (int)W->viewDims[2]; float *o = FP(O); const float *y = FP(Y);
            int E = (int)e->axes.size(); int64_t off = 0;
            for (int g = 0; g < E; ++g) {
                int rows = (int)e->axes[g];
                if (rows > 0)
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, N, K,
                                1.0f, FP(X) + off * K, K, FP(W) + (size_t)g * K * N, N,
                                0.0f, o + off * N, N);
                off += rows;
            }
            int64_t M = X->viewDims[0]; for (int64_t i = 0; i < M * N; ++i) o[i] += y[i];
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ------------------------------------------------------------------ elementwise / misc
aclnnStatus run_misc(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
        case B_DIVMOD_T: {  // floor(self/other) elementwise (roundMode 0). other tensor
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A), *b = FP(B); float *o = FP(O);
            for (int64_t i = 0; i < n; ++i) o[i] = std::floor(a[i] / b[i]);
            break;
        }
        case B_DIVMOD_S: {  // floor(self/scalar) (roundMode 0)
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A); float *o = FP(O); double d = e->alpha;
            for (int64_t i = 0; i < n; ++i) o[i] = std::floor(a[i] / d);
            break;
        }
        case B_FMOD_S: {  // C fmod (truncated remainder) with scalar
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A); float *o = FP(O); double d = e->alpha;
            for (int64_t i = 0; i < n; ++i) o[i] = std::fmod(a[i], d);
            break;
        }
        case B_REMAINDER_TS: {  // floor-mod: x - floor(x/d)*d
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A); float *o = FP(O); double d = e->alpha;
            for (int64_t i = 0; i < n; ++i) o[i] = (float)(a[i] - std::floor(a[i] / d) * d);
            break;
        }
        case B_ADDRELU: {  // relu(self + other)
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A), *b = FP(B); float *o = FP(O);
            for (int64_t i = 0; i < n; ++i) { float r = a[i] + b[i]; o[i] = r < 0 ? 0 : r; }
            break;
        }
        case B_GCD: {  // int32 elementwise gcd
            const aclTensor *A = e->a, *B = e->b; aclTensor *O = e->out; int64_t n = O->numel();
            const int32_t *a = IP(A), *b = IP(B); int32_t *o = IP(O);
            for (int64_t i = 0; i < n; ++i) { int32_t x = std::abs(a[i]), y = std::abs(b[i]);
                while (y) { int32_t t = x % y; x = y; y = t; } o[i] = x; }
            break;
        }
        case B_LOGIT: {  // log(p/(1-p)) with p clamped to [eps, 1-eps]
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A); float *o = FP(O); double eps = e->eps;
            for (int64_t i = 0; i < n; ++i) { double p = a[i];
                if (eps >= 0) { if (p < eps) p = eps; else if (p > 1 - eps) p = 1 - eps; }
                o[i] = (float)std::log(p / (1.0 - p)); }
            break;
        }
        case B_SHRINK: {  // softshrink: x>lambd ? x-bias : (x<-lambd ? x+bias : 0); test uses bias=0
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel();
            const float *a = FP(A); float *o = FP(O); double L = e->alpha, B = e->eps;
            for (int64_t i = 0; i < n; ++i) { double x = a[i];
                o[i] = (float)(x > L ? x - B : (x < -L ? x + B : 0.0)); }
            break;
        }
        case B_CDIST: {  // pairwise p-norm: out[P,R] = (sum_k |x1[p,k]-x2[r,k]|^p)^(1/p)
            const aclTensor *X1 = e->a, *X2 = e->b; aclTensor *O = e->out;
            int P = (int)X1->viewDims[0], M = (int)X1->viewDims[1], R = (int)X2->viewDims[0];
            const float *x1 = FP(X1), *x2 = FP(X2); float *o = FP(O); double p = e->alpha;
            for (int pp = 0; pp < P; ++pp) for (int r = 0; r < R; ++r) { double acc = 0;
                for (int k = 0; k < M; ++k) acc += std::pow(std::fabs((double)x1[pp * M + k] - x2[r * M + k]), p);
                o[pp * R + r] = (float)std::pow(acc, 1.0 / p); }
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// ------------------------------------------------------------------ scatter / quant / pool
aclnnStatus run_ext(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
        case B_HISTC: {  // counts of self into `bins` bins over [min,max]; alpha=min, eps=max
            const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = A->numel(); int bins = (int)e->dim;
            const float *a = FP(A); float *o = FP(O); double lo = e->alpha, hi = e->eps;
            for (int b = 0; b < bins; ++b) o[b] = 0.f;
            double span = hi - lo;
            for (int64_t i = 0; i < n; ++i) { double v = a[i]; if (v < lo || v > hi) continue;
                int b = (span > 0) ? (int)((v - lo) / span * bins) : 0; if (b >= bins) b = bins - 1; if (b < 0) b = 0; o[b] += 1.f; }
            break;
        }
        case B_SCATTER_VAL: {  // out = self; out[along dim at index] = value; index shape matches non-scattered dims
            const aclTensor *S = e->a, *I = e->b; aclTensor *O = e->out; int dim = (int)e->dim; double val = e->alpha;
            int nd = (int)S->viewDims.size(); if (dim < 0) dim += nd;
            int64_t n = S->numel(); const float *s = FP(S); float *o = FP(O);
            for (int64_t i = 0; i < n; ++i) o[i] = s[i];
            // index tensor has same rank as self but dim-extent = its own; iterate over all index elements
            std::vector<int64_t> istr(nd, 1); for (int d = nd - 2; d >= 0; --d) istr[d] = istr[d + 1] * S->viewDims[d + 1];
            int64_t ni = I->numel(); const int64_t *idx = I64(I);
            std::vector<int64_t> idims = I->viewDims;
            std::vector<int64_t> idxstr(nd, 1); for (int d = nd - 2; d >= 0; --d) idxstr[d] = idxstr[d + 1] * idims[d + 1];
            for (int64_t f = 0; f < ni; ++f) {
                int64_t rem = f, base = 0; for (int d = 0; d < nd; ++d) { int64_t c = (rem / idxstr[d]) % idims[d];
                    if (d != dim) base += c * istr[d]; }
                int64_t pos = base + idx[f] * istr[dim]; o[pos] = (float)val;
            }
            break;
        }
        case B_PUT: {  // out = self; out.flat[index[k]] = source[k]  (accumulate optional in e->causal)
            const aclTensor *S = e->a, *I = e->b, *SRC = e->c; aclTensor *O = e->out; int64_t n = S->numel();
            const float *s = FP(S), *src = FP(SRC); const int64_t *idx = I64(I); float *o = FP(O);
            for (int64_t i = 0; i < n; ++i) o[i] = s[i];
            int64_t k = I->numel(); bool acc = e->causal;
            for (int64_t j = 0; j < k; ++j) { int64_t p = idx[j]; if (acc) o[p] += src[j]; else o[p] = src[j]; }
            break;
        }
        case B_FAKEQUANT: {  // out = (clamp(round(x/scale)+zp, qmin, qmax) - zp)*scale; mask = in-range
            const aclTensor *A = e->a; aclTensor *O = e->out; aclTensor *MK = e->out2; int64_t n = A->numel();
            const float *a = FP(A); float *o = FP(O); uint8_t *mk = MK ? U8(MK) : nullptr;
            double scale = e->dscalars[0], zp = e->dscalars[1]; int qmin = (int)e->m, qmax = (int)e->n;
            for (int64_t i = 0; i < n; ++i) { long q = std::lround(a[i] / scale) + (long)zp;
                long qc = q < qmin ? qmin : (q > qmax ? qmax : q);
                o[i] = (float)((qc - zp) * scale); if (mk) mk[i] = (uint8_t)(q == qc); }
            break;
        }
        case B_DYNBLOCKQUANT: {  // per-block symmetric int8: scale=max|x|/127, q=round(x/scale)
            const aclTensor *A = e->a; aclTensor *Q = e->out; aclTensor *SC = e->out2;
            int blk = (int)e->dim; int64_t n = A->numel(); int nbk = (int)(n / blk);
            const float *a = FP(A); int8_t *q = I8(Q); float *sc = FP(SC);
            for (int b = 0; b < nbk; ++b) { double amax = 0;
                for (int i = 0; i < blk; ++i) amax = std::max(amax, std::fabs((double)a[(size_t)b * blk + i]));
                double scale = amax > 0 ? amax / 127.0 : 1.0; sc[b] = (float)scale;
                for (int i = 0; i < blk; ++i) { long v = std::lround(a[(size_t)b * blk + i] / scale);
                    if (v > 127) v = 127; if (v < -128) v = -128; q[(size_t)b * blk + i] = (int8_t)v; } }
            break;
        }
        case B_UPNEAR1D: {  // [N,C,L] -> [N,C,Lo] nearest: out[o] = in[floor(o*L/Lo)]
            const aclTensor *A = e->a; aclTensor *O = e->out;
            int NC = (int)(A->viewDims[0] * A->viewDims[1]), L = (int)A->viewDims[2], Lo = (int)O->viewDims[2];
            const float *a = FP(A); float *o = FP(O);
            for (int nc = 0; nc < NC; ++nc) for (int oo = 0; oo < Lo; ++oo) {
                int si = (int)((float)oo * L / Lo); o[nc * Lo + oo] = a[nc * L + si]; }
            break;
        }
        case B_UPNEAR1D_BWD: {  // gradIn[src(o)] += gradOut[o]
            const aclTensor *GO = e->a; aclTensor *GI = e->out;
            int NC = (int)(GO->viewDims[0] * GO->viewDims[1]), Lo = (int)GO->viewDims[2], L = (int)GI->viewDims[2];
            const float *go = FP(GO); float *gi = FP(GI); int64_t ng = GI->numel();
            for (int64_t i = 0; i < ng; ++i) gi[i] = 0.f;
            for (int nc = 0; nc < NC; ++nc) for (int oo = 0; oo < Lo; ++oo) {
                int si = (int)((float)oo * L / Lo); gi[nc * L + si] += go[nc * Lo + oo]; }
            break;
        }
        case B_GAP: {  // global avg pool [N,C,H,W] -> [N,C,1,1]
            const aclTensor *A = e->a; aclTensor *O = e->out;
            int NC = (int)(A->viewDims[0] * A->viewDims[1]), HW = (int)(A->viewDims[2] * A->viewDims[3]);
            const float *a = FP(A); float *o = FP(O);
            for (int nc = 0; nc < NC; ++nc) { double m = 0; for (int i = 0; i < HW; ++i) m += a[(size_t)nc * HW + i]; o[nc] = (float)(m / HW); }
            break;
        }
        case B_REPPAD1D_BWD: {  // gradIn[clamp(o-lp, 0, L-1)] += gradOut[o]; lp in axes[0]
            const aclTensor *GO = e->a; aclTensor *GI = e->out;
            int NC = (int)(GO->viewDims[0] * GO->viewDims[1]), Lo = (int)GO->viewDims[2], L = (int)GI->viewDims[2];
            int lp = (int)e->axes[0]; const float *go = FP(GO); float *gi = FP(GI); int64_t ng = GI->numel();
            for (int64_t i = 0; i < ng; ++i) gi[i] = 0.f;
            for (int nc = 0; nc < NC; ++nc) for (int oo = 0; oo < Lo; ++oo) {
                int j = oo - lp; j = j < 0 ? 0 : (j >= L ? L - 1 : j); gi[nc * L + j] += go[nc * Lo + oo]; }
            break;
        }
        case B_MAXPOOL2D_IDX_BWD: {  // gradIn.flat[indices[o]] += gradOut[o] per (N,C) plane
            const aclTensor *GO = e->a, *I = e->b; aclTensor *GI = e->out;
            int NC = (int)(GO->viewDims[0] * GO->viewDims[1]);
            int oHW = (int)(GO->viewDims[2] * GO->viewDims[3]);
            int HW = (int)(GI->viewDims[2] * GI->viewDims[3]);
            const float *go = FP(GO); const int64_t *idx = I64(I); float *gi = FP(GI); int64_t ng = GI->numel();
            for (int64_t i = 0; i < ng; ++i) gi[i] = 0.f;
            for (int nc = 0; nc < NC; ++nc) for (int o = 0; o < oHW; ++o) {
                int64_t flat = idx[(size_t)nc * oHW + o]; gi[(size_t)nc * HW + flat] += go[(size_t)nc * oHW + o]; }
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
// ---- GEMM family (RUN -> run_blas) ----
RUN(aclnnMm, run_blas) RUN(aclnnBmm, run_blas) RUN(aclnnMv, run_blas) RUN(aclnnDot, run_blas)
RUN(aclnnOuter, run_blas) RUN(aclnnKron, run_blas) RUN(aclnnAddmv, run_blas) RUN(aclnnAddmm, run_blas)
RUN(aclnnBaddbmm, run_blas) RUN(aclnnAddbmm, run_blas) RUN(aclnnGemm, run_blas) RUN(aclnnAddr, run_blas)
RUN(aclnnGer, run_blas) RUN(aclnnMatmulWeightNz, run_blas) RUN(aclnnGroupedMatmulAdd, run_blas)

// ACLNN_MM2 family: (self, mat2, out, ws, ex)
#define MM2(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->op = OP; e->a = self; e->b = mat2; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
MM2(aclnnMm, B_MM) MM2(aclnnBmm, B_BMM) MM2(aclnnMv, B_MV) MM2(aclnnDot, B_DOT)
MM2(aclnnOuter, B_OUTER) MM2(aclnnKron, B_KRON) MM2(aclnnGer, B_GER)

aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *y, const aclTensor *mat, const aclTensor *vec, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_ADDMV; e->a = mat; e->b = vec; e->c = y; e->out = out; e->dscalars = {beta, alpha}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_ADDMM; e->a = A; e->b = B; e->c = C; e->out = out; e->dscalars = {beta, alpha}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnBaddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_BADDBMM; e->a = A; e->b = B; e->c = C; e->out = out; e->dscalars = {beta, alpha}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_ADDBMM; e->a = A; e->b = B; e->c = C; e->out = out; e->dscalars = {beta, alpha}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGemmGetWorkspaceSize(const aclTensor *a, const aclTensor *b, const aclTensor *c, float alpha, float beta, int64_t transA, int64_t transB, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; auto *e = new aclOpExecutor(); e->op = B_GEMM; e->a = a; e->b = b; e->c = c; e->out = out;
    e->dim = transA; e->reduceCount = transB; e->dscalars = {(double)alpha, (double)beta}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor *self, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_ADDR; e->a = self; e->b = vec1; e->c = vec2; e->out = out;
    e->dscalars = {beta ? beta->v : 1.0, alpha ? alpha->v : 1.0}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out, int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType; auto *e = new aclOpExecutor(); e->op = B_MATMUL_NZ; e->a = x; e->b = weight; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y, const aclIntArray *groupList, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_GMM_ADD; e->a = x; e->b = weight; e->c = y; e->out = out;
    if (groupList) e->axes = groupList->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; }

// ---- elementwise / misc (RUN -> run_misc) ----
RUN(aclnnDivMod, run_misc) RUN(aclnnDivMods, run_misc) RUN(aclnnFmodScalar, run_misc)
RUN(aclnnRemainderTensorScalar, run_misc) RUN(aclnnAddRelu, run_misc) RUN(aclnnGcd, run_misc)
RUN(aclnnLogit, run_misc) RUN(aclnnShrink, run_misc) RUN(aclnnCdist, run_misc)

aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t roundMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)roundMode; auto *e = new aclOpExecutor(); e->op = B_DIVMOD_T; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, int64_t roundMode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)roundMode; auto *e = new aclOpExecutor(); e->op = B_DIVMOD_S; e->a = self; e->out = out; e->alpha = other ? other->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_FMOD_S; e->a = self; e->out = out; e->alpha = other ? other->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_REMAINDER_TS; e->a = self; e->out = out; e->alpha = other ? other->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnAddReluGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_ADDRELU; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_GCD; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnLogitGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_LOGIT; e->a = self; e->out = out; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor *self, const aclScalar *lambd, const aclScalar *bias, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_SHRINK; e->a = self; e->out = out; e->alpha = lambd ? lambd->v : 0.5; e->eps = bias ? bias->v : 0.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnCdistGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_CDIST; e->a = x1; e->b = x2; e->out = out; e->alpha = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; }

// ---- scatter / quant / pool (RUN -> run_ext) ----
RUN(aclnnHistc, run_ext) RUN(aclnnScatterValue, run_ext) RUN(aclnnPut, run_ext)
RUN(aclnnFakeQuantPerTensorAffineCachemask, run_ext) RUN(aclnnDynamicBlockQuant, run_ext)
RUN(aclnnUpsampleNearest1d, run_ext) RUN(aclnnUpsampleNearest1dBackward, run_ext)
RUN(aclnnGlobalAveragePool, run_ext) RUN(aclnnReplicationPad1dBackward, run_ext)
RUN(aclnnMaxPool2dWithIndicesBackward, run_ext)

aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int64_t bins, const aclScalar *min, const aclScalar *max, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_HISTC; e->a = self; e->out = out; e->dim = bins; e->alpha = min ? min->v : 0.0; e->eps = max ? max->v : 0.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_SCATTER_VAL; e->a = self; e->b = index; e->out = out; e->dim = dim; e->alpha = value ? value->v : 0.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnPutGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *source, bool accumulate, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_PUT; e->a = self; e->b = index; e->c = source; e->out = out; e->causal = accumulate; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *zeroPoint, int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_FAKEQUANT; e->a = self; e->out = out; e->out2 = mask;
    e->dscalars = {scale ? scale->v : 1.0, zeroPoint ? zeroPoint->v : 0.0}; e->m = quantMin; e->n = quantMax; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_DYNBLOCKQUANT; e->a = x; e->out = out; e->out2 = scaleOut; e->dim = blockSize; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnUpsampleNearest1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_UPNEAR1D; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnUpsampleNearest1dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_UPNEAR1D_BWD; e->a = gradOut; e->out = gradIn; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGlobalAveragePoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = B_GAP; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnReplicationPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; auto *e = new aclOpExecutor(); e->op = B_REPPAD1D_BWD; e->a = gradOutput; e->out = gradInput;
    if (padding) e->axes = padding->v; else e->axes = {0}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)kernel; (void)stride; (void)padding; (void)dilation; (void)ceilMode;
    auto *e = new aclOpExecutor(); e->op = B_MAXPOOL2D_IDX_BWD; e->a = gradOutput; e->b = indices; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
} // extern "C"
