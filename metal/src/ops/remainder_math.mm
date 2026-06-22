// test_remainder R9/R10/R11/R12 + generators: extra elementary math, activations, reductions,
// var/mean, generators (eye/linspace/logspace/range), equal, signbits, axpy, ffn, pdist, sinkhorn,
// nantonum, clamp, inplace-tril. Host-side over unified memory (device ptr == host ptr).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *IP64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
uint8_t *UP8(const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

double digamma(double x) { double r = 0; while (x < 6) { r -= 1.0 / x; x += 1.0; } double inv = 1.0 / x, i2 = inv * inv;
    r += std::log(x) - 0.5 * inv - i2 * (1.0 / 12 - i2 * (1.0 / 120 - i2 * (1.0 / 252))); return r; }

// ---- dispatch kinds ----
enum { K_ACOSH, K_ASINH, K_ATANH, K_SINC, K_DIGAMMA, K_FASTGELU, K_SQRELU, K_GELUV2,
       K_SWISH, K_ROUNDDEC, K_NANTONUM, K_CLAMP, K_SUM, K_MAX_FULL, K_MEANV2, K_REDLOGSUM, K_PRODDIM,
       K_MAXDIM, K_VARMEAN, K_VARCORR, K_LOGADDEXP2, K_FLOORDIV, K_RSUB, K_LOGSIGFWD, K_FATRELUMUL,
       K_AXPYV2, K_EQUAL, K_SIGNPACK, K_SIGNUNPACK, K_EYE, K_LINSPACE, K_LOGSPACE, K_RANGE,
       K_FFN, K_PDIST, K_SINKHORN, K_TRIL };

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    case K_ACOSH: case K_ASINH: case K_ATANH: case K_SINC: case K_DIGAMMA:
    case K_FASTGELU: case K_SQRELU: case K_GELUV2: case K_SWISH:
    case K_ROUNDDEC: case K_NANTONUM: case K_CLAMP: case K_LOGSIGFWD: {
        const float *x = FP(e->a); float *y = FP(e->out); int64_t n = e->a->numel();
        for (int64_t i = 0; i < n; i++) {
            double v = x[i], r;
            switch (e->op) {
            case K_ACOSH: r = std::acosh(v); break;
            case K_ASINH: r = std::asinh(v); break;
            case K_ATANH: r = std::atanh(v); break;
            case K_SINC: { if (v == 0) r = 1.0; else { double p = M_PI * v; r = std::sin(p) / p; } } break;
            case K_DIGAMMA: r = digamma(v); break;
            case K_FASTGELU: case K_GELUV2: { double c = 0.7978845608028654 * (v + 0.044715 * v * v * v); r = 0.5 * v * (1.0 + std::tanh(c)); } break;
            case K_SQRELU: { double rl = v > 0 ? v : 0; r = rl * rl; } break;
            case K_SWISH: { double beta = e->alpha; r = v / (1.0 + std::exp(-beta * v)); } break;
            case K_ROUNDDEC: { double m = std::pow(10.0, (double)(int64_t)e->dim); r = std::rint(v * m) / m; } break;
            case K_NANTONUM: { if (std::isnan(v)) r = e->dscalars[0]; else if (std::isinf(v) && v > 0) r = e->dscalars[1]; else if (std::isinf(v) && v < 0) r = e->dscalars[2]; else r = v; } break;
            case K_CLAMP: { double lo = e->dscalars[0], hi = e->dscalars[1]; r = v < lo ? lo : (v > hi ? hi : v); } break;
            case K_LOGSIGFWD: r = std::log(1.0 / (1.0 + std::exp(-v))); break;
            default: r = v;
            }
            y[i] = (float)r;
        }
        if (e->op == K_LOGSIGFWD && e->out2) { float *buf = FP(e->out2); for (int64_t i = 0; i < n; i++) buf[i] = 0.f; }
        return ACLNN_SUCCESS;
    }
    case K_LOGADDEXP2: case K_FLOORDIV: case K_RSUB: case K_FATRELUMUL: case K_AXPYV2: case K_EQUAL: {
        const float *a = FP(e->a), *b = FP(e->b); int64_t n = e->out->numel();
        if (e->op == K_EQUAL) { uint8_t *o = UP8(e->out); for (int64_t i = 0; i < n; i++) o[i] = (a[i] == b[i]) ? 1 : 0; return ACLNN_SUCCESS; }
        float *y = FP(e->out);
        for (int64_t i = 0; i < n; i++) {
            double r;
            switch (e->op) {
            case K_LOGADDEXP2: r = std::log2(std::exp2((double)a[i]) + std::exp2((double)b[i])); break;
            case K_FLOORDIV: r = std::floor((double)a[i] / b[i]); break;
            case K_RSUB: r = (double)b[i] - e->alpha * a[i]; break;            // other - alpha*self
            case K_FATRELUMUL: r = (a[i] > e->alpha ? (double)a[i] : 0.0) * b[i]; break;
            case K_AXPYV2: r = e->alpha * a[i] + b[i]; break;
            default: r = 0;
            }
            y[i] = (float)r;
        }
        return ACLNN_SUCCESS;
    }
    case K_SUM: case K_MAX_FULL: {
        const float *x = FP(e->a); int64_t n = e->a->numel(); double acc = (e->op == K_SUM) ? 0.0 : -INFINITY;
        for (int64_t i = 0; i < n; i++) { if (e->op == K_SUM) acc += x[i]; else acc = std::max(acc, (double)x[i]); }
        FP(e->out)[0] = (float)acc; return ACLNN_SUCCESS;
    }
    case K_MEANV2: case K_REDLOGSUM: {   // reduce along axes[0], no keepdim, out flat over remaining
        int dim = (int)(e->axes.empty() ? 0 : e->axes[0]); int nd = (int)e->a->viewDims.size(); if (dim < 0) dim += nd;
        int64_t outer = 1, D = e->a->viewDims[dim], inner = 1;
        for (int d = 0; d < dim; d++) outer *= e->a->viewDims[d];
        for (int d = dim + 1; d < nd; d++) inner *= e->a->viewDims[d];
        const float *x = FP(e->a); float *y = FP(e->out);
        for (int64_t o = 0; o < outer; o++) for (int64_t ii = 0; ii < inner; ii++) {
            double s = 0; for (int64_t d = 0; d < D; d++) s += x[(o * D + d) * inner + ii];
            y[o * inner + ii] = (float)(e->op == K_MEANV2 ? s / D : std::log(s));
        }
        return ACLNN_SUCCESS;
    }
    case K_PRODDIM: {
        int dim = (int)e->dim; int nd = (int)e->a->viewDims.size(); if (dim < 0) dim += nd;
        int64_t outer = 1, D = e->a->viewDims[dim], inner = 1;
        for (int d = 0; d < dim; d++) outer *= e->a->viewDims[d];
        for (int d = dim + 1; d < nd; d++) inner *= e->a->viewDims[d];
        const float *x = FP(e->a); float *y = FP(e->out);
        for (int64_t o = 0; o < outer; o++) for (int64_t ii = 0; ii < inner; ii++) {
            double p = 1; for (int64_t d = 0; d < D; d++) p *= x[(o * D + d) * inner + ii];
            y[o * inner + ii] = (float)p;
        }
        return ACLNN_SUCCESS;
    }
    case K_MAXDIM: {
        int dim = (int)e->dim; int nd = (int)e->a->viewDims.size(); if (dim < 0) dim += nd;
        int64_t outer = 1, D = e->a->viewDims[dim], inner = 1;
        for (int d = 0; d < dim; d++) outer *= e->a->viewDims[d];
        for (int d = dim + 1; d < nd; d++) inner *= e->a->viewDims[d];
        const float *x = FP(e->a); float *v = FP(e->out); int64_t *idx = IP64(e->out2);
        for (int64_t o = 0; o < outer; o++) for (int64_t ii = 0; ii < inner; ii++) {
            float best = x[(o * D + 0) * inner + ii]; int64_t bi = 0;
            for (int64_t d = 1; d < D; d++) { float c = x[(o * D + d) * inner + ii]; if (c > best) { best = c; bi = d; } }
            v[o * inner + ii] = best; idx[o * inner + ii] = bi;
        }
        return ACLNN_SUCCESS;
    }
    case K_VARMEAN: case K_VARCORR: {
        int dim = (int)(e->axes.empty() ? 0 : e->axes[0]); int nd = (int)e->a->viewDims.size(); if (dim < 0) dim += nd;
        int64_t outer = 1, D = e->a->viewDims[dim], inner = 1;
        for (int d = 0; d < dim; d++) outer *= e->a->viewDims[d];
        for (int d = dim + 1; d < nd; d++) inner *= e->a->viewDims[d];
        double corr = e->alpha;   // correction (Bessel) — 1 for VarMean unbiased & VarCorrection(1)
        const float *x = FP(e->a); float *var = FP(e->out); float *mean = e->out2 ? FP(e->out2) : nullptr;
        for (int64_t o = 0; o < outer; o++) for (int64_t ii = 0; ii < inner; ii++) {
            double m = 0; for (int64_t d = 0; d < D; d++) m += x[(o * D + d) * inner + ii]; m /= D;
            double sq = 0; for (int64_t d = 0; d < D; d++) { double u = x[(o * D + d) * inner + ii] - m; sq += u * u; }
            var[o * inner + ii] = (float)(sq / (D - corr));
            if (mean) mean[o * inner + ii] = (float)m;
        }
        return ACLNN_SUCCESS;
    }
    case K_SIGNPACK: {   // pack sign bits of n floats into ceil(n/8) bytes; bit set if value >= 0
        const float *x = FP(e->a); int64_t n = e->a->numel(); uint8_t *o = UP8(e->out); int64_t nb = (n + 7) / 8;
        for (int64_t b = 0; b < nb; b++) { uint8_t byte = 0; for (int k = 0; k < 8; k++) { int64_t i = b * 8 + k; if (i < n && x[i] >= 0) byte |= (uint8_t)(1u << k); } o[b] = byte; }
        return ACLNN_SUCCESS;
    }
    case K_SIGNUNPACK: {   // unpack bits to +1/-1 floats
        const uint8_t *p = UP8(e->a); float *o = FP(e->out); int64_t n = e->out->numel();
        for (int64_t i = 0; i < n; i++) { uint8_t byte = p[i / 8]; bool bit = (byte >> (i % 8)) & 1u; o[i] = bit ? 1.f : -1.f; }
        return ACLNN_SUCCESS;
    }
    case K_EYE: {
        int R = (int)e->out->viewDims[0], C = e->out->viewDims.size() > 1 ? (int)e->out->viewDims[1] : R;
        float *o = FP(e->out); for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) o[r * C + c] = (r == c) ? 1.f : 0.f;
        return ACLNN_SUCCESS;
    }
    case K_LINSPACE: case K_LOGSPACE: {
        int S = (int)e->out->numel(); double a = e->dscalars[0], b = e->dscalars[1]; float *o = FP(e->out);
        for (int i = 0; i < S; i++) { double t = (S == 1) ? a : a + (b - a) * i / (double)(S - 1);
            o[i] = (float)(e->op == K_LINSPACE ? t : std::pow(e->alpha, t)); }   // alpha=base for logspace
        return ACLNN_SUCCESS;
    }
    case K_RANGE: {
        int S = (int)e->out->numel(); double start = e->dscalars[0], step = e->dscalars[2]; float *o = FP(e->out);
        for (int i = 0; i < S; i++) o[i] = (float)(start + step * i);
        return ACLNN_SUCCESS;
    }
    case K_FFN: {   // relu(x@W1+b1)@W2+b2 ; x[M,K] W1[K,H] b1[H] W2[H,N] b2[N] out[M,N]
        const aclTensor *X = e->a, *W1 = e->b, *B1 = e->c, *W2 = e->inputs[0], *B2 = e->inputs[1];
        int M = (int)X->viewDims[0], K = (int)X->viewDims[1], H = (int)W1->viewDims[1], N = (int)W2->viewDims[1];
        const float *x = FP(X), *w1 = FP(W1), *b1 = FP(B1), *w2 = FP(W2), *b2 = FP(B2); float *o = FP(e->out);
        std::vector<double> h(H);
        for (int m = 0; m < M; m++) {
            for (int j = 0; j < H; j++) { double a = b1[j]; for (int k = 0; k < K; k++) a += (double)x[m * K + k] * w1[k * H + j]; h[j] = std::max(a, 0.0); }
            for (int nn = 0; nn < N; nn++) { double a = b2[nn]; for (int j = 0; j < H; j++) a += h[j] * w2[j * N + nn]; o[m * N + nn] = (float)a; }
        }
        return ACLNN_SUCCESS;
    }
    case K_PDIST: {   // condensed pairwise distance, p in alpha
        const aclTensor *X = e->a; int Nn = (int)X->viewDims[0], Dd = (int)X->viewDims[1]; double p = e->alpha;
        const float *x = FP(X); float *o = FP(e->out); int t = 0;
        for (int i = 0; i < Nn; i++) for (int j = i + 1; j < Nn; j++) {
            double s = 0; for (int d = 0; d < Dd; d++) s += std::pow(std::fabs((double)x[i * Dd + d] - x[j * Dd + d]), p);
            o[t++] = (float)std::pow(s, 1.0 / p);
        }
        return ACLNN_SUCCESS;
    }
    case K_SINKHORN: {   // alternate row/col normalize; iters in dim; finish on col-normalize so columns sum to 1
        const aclTensor *Mt = e->a; int R = (int)Mt->viewDims[0], C = (int)Mt->viewDims[1]; int iters = (int)e->dim;
        const float *m = FP(Mt); float *o = FP(e->out);
        for (int i = 0; i < R * C; i++) o[i] = m[i];
        for (int it = 0; it < iters; it++) {
            for (int r = 0; r < R; r++) { double s = 0; for (int c = 0; c < C; c++) s += o[r * C + c]; if (s > 0) for (int c = 0; c < C; c++) o[r * C + c] /= (float)s; }
            for (int c = 0; c < C; c++) { double s = 0; for (int r = 0; r < R; r++) s += o[r * C + c]; if (s > 0) for (int r = 0; r < R; r++) o[r * C + c] /= (float)s; }
        }
        return ACLNN_SUCCESS;
    }
    case K_TRIL: {   // in-place lower triangular; diagonal offset in dim
        aclTensor *T = e->out; int R = (int)T->viewDims[0], C = (int)T->viewDims[1]; int64_t diag = e->dim;
        float *o = FP(T); for (int r = 0; r < R; r++) for (int c = 0; c < C; c++) if (c > r + diag) o[r * C + c] = 0.f;
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUNFN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUNFN(rm_run)
} // namespace

extern "C" {
// ---- unary math (MATH_UN: self,out) ----
#define UNMATH(NAME, KIND) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e = new aclOpExecutor(); e->op = KIND; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(NAME)
UNMATH(aclnnAcosh, K_ACOSH) UNMATH(aclnnAsinh, K_ASINH) UNMATH(aclnnAtanh, K_ATANH)
UNMATH(aclnnSinc, K_SINC) UNMATH(aclnnDigamma, K_DIGAMMA)
UNMATH(aclnnFastGelu, K_FASTGELU) UNMATH(aclnnSquaredRelu, K_SQRELU)

aclnnStatus aclnnGeluV2GetWorkspaceSize(const aclTensor *self, int64_t approximate, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)approximate; auto *e = new aclOpExecutor(); e->op = K_GELUV2; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnGeluV2)
aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor *self, const aclScalar *beta, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SWISH; e->a = self; e->out = out; e->alpha = beta ? beta->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSwish)
aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(const aclTensor *self, int64_t decimals, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_ROUNDDEC; e->a = self; e->out = out; e->dim = decimals; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnRoundDecimals)
aclnnStatus aclnnNanToNumGetWorkspaceSize(const aclTensor *self, float nan, float posinf, float neginf, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_NANTONUM; e->a = self; e->out = out; e->dscalars = {(double)nan, (double)posinf, (double)neginf}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnNanToNum)
aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor *self, const aclScalar *mn, const aclScalar *mx, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_CLAMP; e->a = self; e->out = out; e->dscalars = {mn->v, mx->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnClamp)
aclnnStatus aclnnLogSigmoidForwardGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *buffer, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LOGSIGFWD; e->a = self; e->out = out; e->out2 = buffer; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLogSigmoidForward)

// ---- binary ----
aclnnStatus aclnnLogAddExp2GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LOGADDEXP2; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLogAddExp2)
aclnnStatus aclnnFloorDivideGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_FLOORDIV; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnFloorDivide)
aclnnStatus aclnnRsubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_RSUB; e->a = self; e->b = other; e->out = out; e->alpha = alpha ? alpha->v : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnRsub)
aclnnStatus aclnnFatreluMulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double threshold, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_FATRELUMUL; e->a = x1; e->b = x2; e->out = out; e->alpha = threshold; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnFatreluMul)
aclnnStatus aclnnAxpyV2GetWorkspaceSize(const aclTensor *self, const aclTensor *other, double alpha, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_AXPYV2; e->a = self; e->b = other; e->out = out; e->alpha = alpha; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnAxpyV2)
aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EQUAL; e->a = self; e->b = other; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnEqual)

// ---- reductions ----
aclnnStatus aclnnSumGetWorkspaceSize(const aclTensor *self, aclDataType dt, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dt; auto *e = new aclOpExecutor(); e->op = K_SUM; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSum)
aclnnStatus aclnnMaxGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MAX_FULL; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMax)
aclnnStatus aclnnMeanV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dt, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)dt; auto *e = new aclOpExecutor(); e->op = K_MEANV2; e->a = self; e->out = out; if (dim) e->axes = dim->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMeanV2)
aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e = new aclOpExecutor(); e->op = K_REDLOGSUM; e->a = self; e->out = out; if (dim) e->axes = dim->v; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnReduceLogSum)
aclnnStatus aclnnProdDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclDataType dt, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; (void)dt; auto *e = new aclOpExecutor(); e->op = K_PRODDIM; e->a = self; e->out = out; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnProdDim)
aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e = new aclOpExecutor(); e->op = K_MAXDIM; e->a = self; e->out = valuesOut; e->out2 = indicesOut; e->dim = dim; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMaxDim)
aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *varOut, aclTensor *meanOut, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e = new aclOpExecutor(); e->op = K_VARMEAN; e->a = self; e->out = varOut; e->out2 = meanOut; if (dim) e->axes = dim->v; e->alpha = 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnVarMean)
aclnnStatus aclnnVarCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e = new aclOpExecutor(); e->op = K_VARCORR; e->a = self; e->out = out; if (dim) e->axes = dim->v; e->alpha = (double)correction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnVarCorrection)

// ---- signbits ----
aclnnStatus aclnnSignBitsPackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SIGNPACK; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSignBitsPack)
aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SIGNUNPACK; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSignBitsUnpack)

// ---- generators ----
aclnnStatus aclnnEyeGetWorkspaceSize(aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EYE; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnEye)
aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)steps; auto *e = new aclOpExecutor(); e->op = K_LINSPACE; e->out = out; e->dscalars = {start->v, end->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLinspace)
aclnnStatus aclnnLogSpaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, double base, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)steps; auto *e = new aclOpExecutor(); e->op = K_LOGSPACE; e->out = out; e->dscalars = {start->v, end->v}; e->alpha = base; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnLogSpace)
aclnnStatus aclnnRangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_RANGE; e->out = out; e->dscalars = {start->v, end->v, step->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnRange)

// ---- misc nn ----
aclnnStatus aclnnFFNGetWorkspaceSize(const aclTensor *x, const aclTensor *w1, const aclTensor *b1, const aclTensor *w2, const aclTensor *b2, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)activation; auto *e = new aclOpExecutor(); e->op = K_FFN; e->a = x; e->b = w1; e->c = b1; e->inputs = {w2, b2}; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnFFN)
aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_PDIST; e->a = self; e->out = out; e->alpha = p; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnPdist)
aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)tau; auto *e = new aclOpExecutor(); e->op = K_SINKHORN; e->a = cost; e->out = out; e->dim = iters; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSinkhorn)
aclnnStatus aclnnInplaceTrilGetWorkspaceSize(aclTensor *selfRef, int64_t diagonal, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_TRIL; e->out = selfRef; e->dim = diagonal; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnInplaceTril)
} // extern "C"
