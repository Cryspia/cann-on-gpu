// test_ops completions: statistical/selection reductions, ternary, SWhere, bitwise, 2-scalar activations,
// and RNG. Host-side over unified memory (exact for deterministic ops; statistical for RNG).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <cstdio>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
// strided element offset (excl. tensor offset) into operand `t` for flattened out index g, honoring NumPy broadcast.
int64_t bcoff(const aclTensor *out, const aclTensor *t, int64_t g) {
    int ond = (int)out->viewDims.size(), tnd = (int)t->viewDims.size(); int64_t off = 0, rem = g;
    for (int d = ond - 1; d >= 0; --d) { int64_t idx = rem % out->viewDims[d]; rem /= out->viewDims[d];
        int td = d - (ond - tnd); if (td >= 0 && t->viewDims[td] != 1) off += idx * t->strides[td]; }
    return off;
}
// single-dim group geometry
void geom(const aclTensor *t, int dim, int64_t &outer, int64_t &D, int64_t &inner) {
    int nd = (int)t->viewDims.size(); if (dim < 0) dim += nd; outer = 1; inner = 1; D = t->viewDims[dim];
    for (int d = 0; d < dim; ++d) outer *= t->viewDims[d]; for (int d = dim + 1; d < nd; ++d) inner *= t->viewDims[d];
}
inline uint32_t pcg(uint32_t v) { uint32_t s = v * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
inline float u01(uint32_t &s) { s = s * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; w = (w >> 22u) ^ w; return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }

// ---- value reductions over a set of axes (e->axes; empty=all). m: 0sum 1mean 2var 3std 4norm(p=alpha) 5lse 6nansum 7nanmean 8all 9any 10countnz ----
aclnnStatus run_red(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int op = (int)e->m; double p = e->alpha;
    int nd = (int)a->viewDims.size(); bool red[8] = {false};
    if (e->axes.empty()) { for (int d = 0; d < nd; ++d) red[d] = true; }
    else for (int64_t ax : e->axes) { int d = (int)(ax < 0 ? ax + nd : ax); red[d] = true; }
    int64_t istr[8]; { int64_t st = 1; for (int d = nd - 1; d >= 0; --d) { istr[d] = st; st *= a->viewDims[d]; } }
    std::vector<int64_t> od, os, rd, rs; int64_t nout = 1, nred = 1;
    for (int d = 0; d < nd; ++d) { if (red[d]) { rd.push_back(a->viewDims[d]); rs.push_back(istr[d]); nred *= a->viewDims[d]; } else { od.push_back(a->viewDims[d]); os.push_back(istr[d]); nout *= a->viewDims[d]; } }
    const float *x = FP(a);
    for (int64_t g = 0; g < nout; ++g) {
        int64_t rem = g, base = 0; for (int i = (int)od.size() - 1; i >= 0; --i) { int64_t id = rem % od[i]; rem /= od[i]; base += id * os[i]; }
        double sum = 0, sumsq = 0, mx = -INFINITY, normacc = 0; int64_t cnt = 0; bool allnz = true, anynz = false;
        for (int64_t r = 0; r < nred; ++r) { int64_t rr = r, off = base; for (int i = (int)rd.size() - 1; i >= 0; --i) { int64_t id = rr % rd[i]; rr /= rd[i]; off += id * rs[i]; }
            float v = x[off];
            if (op == 6 || op == 7) { if (!std::isnan(v)) { sum += v; cnt++; } continue; }
            sum += v; sumsq += (double)v * v; mx = std::max(mx, (double)v); normacc += std::pow(std::fabs((double)v), p);
            if (v == 0) allnz = false; else anynz = true; cnt += (v != 0); }
        double se = 0; if (op == 5) { double s2 = 0; for (int64_t r = 0; r < nred; ++r) { int64_t rr = r, off = base; for (int i = (int)rd.size() - 1; i >= 0; --i) { int64_t id = rr % rd[i]; rr /= rd[i]; off += id * rs[i]; } s2 += std::exp((double)x[off] - mx); } se = mx + std::log(s2); }
        double r; switch (op) { case 0: r = sum; break; case 1: r = sum / nred; break;
            case 2: r = (sumsq - sum * sum / nred) / (nred - 1); break; case 3: r = std::sqrt((sumsq - sum * sum / nred) / (nred - 1)); break;
            case 4: r = std::pow(normacc, 1.0 / p); break; case 5: r = se; break; case 6: r = sum; break; case 7: r = sum / (cnt ? cnt : 1); break; default: r = 0; }
        if (op == 8) ((uint8_t *)o->data + o->offset)[g] = allnz ? 1 : 0;
        else if (op == 9) ((uint8_t *)o->data + o->offset)[g] = anynz ? 1 : 0;
        else if (op == 10) ((int64_t *)o->data + o->offset)[g] = cnt;
        else FP(o)[g] = (float)r;
    }
    return ACLNN_SUCCESS;
}
// argmax/argmin (int64 out) / aminmax (two fp32). m: 0 argmax,1 argmin,2 aminmax
aclnnStatus run_argred(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; int dim = (int)e->dim; int64_t outer, D, inner; geom(a, dim, outer, D, inner); const float *x = FP(a);
    for (int64_t g = 0; g < outer * inner; ++g) { int64_t base = (g / inner) * D * inner + (g % inner);
        float mn = x[base], mx = x[base]; int64_t imn = 0, imx = 0;
        for (int64_t d = 1; d < D; ++d) { float v = x[base + d * inner]; if (v > mx) { mx = v; imx = d; } if (v < mn) { mn = v; imn = d; } }
        if (e->m == 0) ((int64_t *)e->out->data + e->out->offset)[g] = imx;
        else if (e->m == 1) ((int64_t *)e->out->data + e->out->offset)[g] = imn;
        else { FP(e->out)[g] = mn; FP(e->out2)[g] = mx; }
    }
    return ACLNN_SUCCESS;
}
// median/kthvalue/quantile/mode over one dim (fp32). m:0 median,1 kthvalue(k=reduceCount),2 quantile(q=alpha),3 mode
aclnnStatus run_select(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int dim = (int)e->dim; int64_t outer, D, inner; geom(a, dim, outer, D, inner); const float *x = FP(a);
    std::vector<float> v(D);
    for (int64_t g = 0; g < outer * inner; ++g) { int64_t base = (g / inner) * D * inner + (g % inner);
        for (int64_t d = 0; d < D; ++d) v[d] = x[base + d * inner]; std::sort(v.begin(), v.end());
        float r; if (e->m == 0) r = v[(D - 1) / 2]; else if (e->m == 1) r = v[e->reduceCount - 1];
        else if (e->m == 2) { double pp = e->alpha * (D - 1); int lo = (int)std::floor(pp), hi = (int)std::ceil(pp); double f = pp - lo; r = (float)(v[lo] * (1 - f) + v[hi] * f); }
        else { float bv = v[0]; int bc = 0; for (int64_t i = 0; i < D;) { int64_t j = i; while (j < D && v[j] == v[i]) j++; if ((int)(j - i) > bc) { bc = (int)(j - i); bv = v[i]; } i = j; } r = bv; }
        FP(o)[g] = r;
    }
    return ACLNN_SUCCESS;
}
aclnnStatus run_renorm(aclOpExecutor *e, aclrtStream s) {   // p=alpha, dim=dim, maxnorm=eps
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int dim = (int)e->dim; int64_t outer, D, inner; geom(a, dim, outer, D, inner);
    double p = e->alpha, maxn = e->eps; const float *x = FP(a); float *y = FP(o); int64_t n = a->numel();
    for (int64_t i = 0; i < n; ++i) y[i] = x[i];
    for (int64_t j = 0; j < D; ++j) { double nrm = 0;
        for (int64_t oo = 0; oo < outer; ++oo) for (int64_t ii = 0; ii < inner; ++ii) nrm += std::pow(std::fabs((double)x[oo * D * inner + j * inner + ii]), p);
        nrm = std::pow(nrm, 1.0 / p); double sc = nrm > maxn ? maxn / (nrm + 1e-7) : 1.0;
        for (int64_t oo = 0; oo < outer; ++oo) for (int64_t ii = 0; ii < inner; ++ii) { int64_t idx = oo * D * inner + j * inner + ii; y[idx] = (float)(x[idx] * sc); } }
    return ACLNN_SUCCESS;
}
aclnnStatus run_ternary(aclOpExecutor *e, aclrtStream s) {   // m:0 addcmul,1 addcdiv,2 clamptensor
    drain(s); const aclTensor *a = e->a, *b = e->b, *c = e->c; aclTensor *o = e->out; int64_t n = o->numel(); double v = e->alpha;
    const float *sp = FP(a), *t1 = FP(b), *t2 = FP(c); float *y = FP(o);
    for (int64_t i = 0; i < n; ++i) { float av = sp[bcoff(o, a, i)], b1 = t1[bcoff(o, b, i)], c2 = t2[bcoff(o, c, i)];
        // clamptensor follows torch: lower bound wins if min>max (min checked first), so use the same branch order
        if (e->m == 0) y[i] = av + (float)v * b1 * c2; else if (e->m == 1) y[i] = av + (float)v * b1 / c2; else y[i] = av < b1 ? b1 : (av > c2 ? c2 : av); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_swhere(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *cond = e->a, *x = e->b, *y = e->c; aclTensor *o = e->out; int64_t n = o->numel();
    const float *xp = FP(x), *yp = FP(y); float *op = FP(o);
    bool cb = (dtype_size(cond->dtype) == 1); const uint8_t *cu = (const uint8_t *)cond->data + cond->offset; const float *cf = FP(cond);  // bool/uint8/int8 cond = 1 byte
    for (int64_t i = 0; i < n; ++i) { int64_t ci = bcoff(o, cond, i); bool t = cb ? (cu[ci] != 0) : (cf[ci] != 0.f);
        op[i] = t ? xp[bcoff(o, x, i)] : yp[bcoff(o, y, i)]; }
    return ACLNN_SUCCESS;
}
aclnnStatus run_bitwise(aclOpExecutor *e, aclrtStream s) {   // m:0 and,1 or,2 xor,3 not; scalar in alpha (int)
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t n = o->numel(); const int32_t *ap = (const int32_t *)a->data + a->offset; int32_t *op = (int32_t *)o->data + o->offset;
    bool hasB = (e->b != nullptr); const int32_t *bp = hasB ? (const int32_t *)e->b->data + e->b->offset : nullptr; int32_t sc = (int32_t)e->alpha;
    for (int64_t i = 0; i < n; ++i) { int32_t x = ap[bcoff(o, a, i)], y = hasB ? bp[bcoff(o, e->b, i)] : sc;
        op[i] = e->m == 0 ? (x & y) : e->m == 1 ? (x | y) : e->m == 2 ? (x ^ y) : (~x); }
    return ACLNN_SUCCESS;
}
aclnnStatus run_clamp2(aclOpExecutor *e, aclrtStream s) {   // m:0 hardtanh(min=alpha,max=eps), 1 threshold(thr=alpha,val=eps)
    drain(s); const aclTensor *a = e->a; aclTensor *o = e->out; int64_t n = o->numel(); const float *x = FP(a); float *y = FP(o); double lo = e->alpha, hi = e->eps;
    for (int64_t i = 0; i < n; ++i) y[i] = e->m == 0 ? std::min(std::max(x[i], (float)lo), (float)hi) : (x[i] > (float)lo ? x[i] : (float)hi);
    return ACLNN_SUCCESS;
}
aclnnStatus run_rng(aclOpExecutor *e, aclrtStream s) {   // m:0 uniform,1 normal,2 bernoulli,3 dropout,4 randperm
    drain(s); aclTensor *o = e->out; uint32_t seed = (uint32_t)(int64_t)e->dim; int64_t n = o->numel();
    double a0 = e->dscalars.size() > 0 ? e->dscalars[0] : 0, a1 = e->dscalars.size() > 1 ? e->dscalars[1] : 0;
    if (e->m == 4) { int64_t m = (int64_t)a0; int64_t *op = (int64_t *)o->data + o->offset; for (int64_t i = 0; i < m; ++i) op[i] = i;
        for (int64_t i = m - 1; i > 0; --i) { uint32_t st = pcg(seed ^ (uint32_t)(i * 2654435761u)); int64_t j = (int64_t)(u01(st) * (i + 1)); std::swap(op[i], op[j]); } return ACLNN_SUCCESS; }
    float *op = FP(o);
    for (int64_t i = 0; i < n; ++i) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u));
        if (e->m == 0) op[i] = (float)(a0 + u01(st) * (a1 - a0));
        else if (e->m == 1) { float u1 = std::max(u01(st), 1e-7f), u2 = u01(st); op[i] = (float)(a0 + a1 * std::sqrt(-2 * std::log(u1)) * std::cos(6.2831853f * u2)); }
        else if (e->m == 2) op[i] = u01(st) < a0 ? 1.f : 0.f;
        else { float keep = u01(st) >= a0 ? 1.f : 0.f; const float *xp = FP(e->a); op[i] = xp[i] * keep / (1.f - (float)a0); if (e->out2) ((uint8_t *)e->out2->data + e->out2->offset)[i] = (uint8_t)(keep != 0.f); }  // mask is ACL_BOOL (1 byte)
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
int firstdim(const aclIntArray *d) { return (d && !d->v.empty()) ? (int)d->v[0] : 0; }
} // namespace

extern "C" {
// reductions with IntArray dim
#define RED(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; if(dim) e->axes=dim->v; e->m=OP; e->alpha=2.0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_red)
RED(aclnnVar, 2) RED(aclnnStd, 3) RED(aclnnLogSumExp, 5) RED(aclnnAll, 8) RED(aclnnAny, 9) RED(aclnnCountNonzero, 10)
#define RED_DT(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dt, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)keepDim; (void)dt; auto *e=new aclOpExecutor(); e->a=self; e->out=out; if(dim) e->axes=dim->v; e->m=OP; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_red)
RED_DT(aclnnNansum, 6) RED_DT(aclnnNanmean, 7)
aclnnStatus aclnnNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; if(dim) e->axes=dim->v; e->m=4; e->alpha=p; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnNorm, run_red)
// arg / aminmax
#define ARG(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->m=OP; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_argred)
ARG(aclnnArgMax, 0) ARG(aclnnArgMin, 1)
aclnnStatus aclnnAminmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *ws, aclOpExecutor **ex) {
    (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=outMin; e->out2=outMax; e->dim=firstdim(dim); e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnAminmax, run_argred)
// selection
aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->m=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMedian, run_select)
aclnnStatus aclnnKthvalueGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->reduceCount=k; e->m=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnKthvalue, run_select)
aclnnStatus aclnnQuantileGetWorkspaceSize(const aclTensor *self, double q, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->alpha=q; e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnQuantile, run_select)
aclnnStatus aclnnModeGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { (void)keepDim; auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->m=3; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMode, run_select)
aclnnStatus aclnnRenormGetWorkspaceSize(const aclTensor *self, double p, int64_t dim, double maxnorm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->dim=dim; e->alpha=p; e->eps=maxnorm; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnRenorm, run_renorm)
// ternary
#define TERN(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *t1, const aclTensor *t2, const aclScalar *v, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    auto *e=new aclOpExecutor(); e->a=self; e->b=t1; e->c=t2; e->out=out; e->alpha=v?v->v:1.0; e->m=OP; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_ternary)
TERN(aclnnAddcmul, 0) TERN(aclnnAddcdiv, 1)
aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *minT, const aclTensor *maxT, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=minT; e->c=maxT; e->out=out; e->m=2; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnClampTensor, run_ternary)
aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=condition; e->b=self; e->c=other; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnSWhere, run_swhere)
// bitwise
#define BITT(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->b=other; e->out=out; e->m=OP; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_bitwise)
#define BITS(NAME, OP) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->alpha=other->v; e->m=OP; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME, run_bitwise)
BITT(aclnnBitwiseAndTensor, 0) BITT(aclnnBitwiseOrTensor, 1) BITT(aclnnBitwiseXorTensor, 2)
BITS(aclnnBitwiseAndScalar, 0) BITS(aclnnBitwiseOrScalar, 1) BITS(aclnnBitwiseXorScalar, 2)
aclnnStatus aclnnBitwiseNotGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->m=3; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnBitwiseNot, run_bitwise)
// 2-scalar activations
aclnnStatus aclnnHardtanhGetWorkspaceSize(const aclTensor *self, const aclScalar *lo, const aclScalar *hi, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->alpha=lo->v; e->eps=hi->v; e->m=0; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnHardtanh, run_clamp2)
aclnnStatus aclnnThresholdGetWorkspaceSize(const aclTensor *self, const aclScalar *thr, const aclScalar *val, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->alpha=thr->v; e->eps=val->v; e->m=1; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnThreshold, run_clamp2)
// RNG
aclnnStatus aclnnUniformGetWorkspaceSize(aclTensor *out, double from, double to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->out=out; e->dim=seed; e->m=0; e->dscalars={from,to}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnUniform, run_rng)
aclnnStatus aclnnNormalGetWorkspaceSize(aclTensor *out, double mean, double std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->out=out; e->dim=seed; e->m=1; e->dscalars={mean,std}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnNormal, run_rng)
aclnnStatus aclnnBernoulliGetWorkspaceSize(aclTensor *out, double p, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->out=out; e->dim=seed; e->m=2; e->dscalars={p}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnBernoulli, run_rng)
aclnnStatus aclnnDropoutGetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->a=x; e->out=out; e->out2=mask; e->dim=seed; e->m=3; e->dscalars={p}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnDropout, run_rng)
aclnnStatus aclnnRandpermGetWorkspaceSize(int64_t n, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { auto *e=new aclOpExecutor(); e->out=out; e->dim=seed; e->m=4; e->dscalars={(double)n}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnRandperm, run_rng)
} // extern "C"
