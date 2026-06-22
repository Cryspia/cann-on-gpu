// Foreach gap operators: the 57 aclnnForeach* symbols the CUDA backend exports but the Metal
// backend's foreach.mm did not yet cover. Same contract/semantics as cuda/src/ops/foreach_ext.cu.
// Each op applies one elementwise operation across every tensor of an aclTensorList; done host-side
// over unified memory (device ptr == host ptr) after draining the stream, exactly like foreach.mm.
//
// Executor field usage (reusing existing aclOpExecutor slots; internal.h is NOT modified):
//   e->m            : foreach "shape" code (FShape below)
//   e->n            : elementwise math selector (EwKind below)
//   e->tl[0..3]     : participating tensor lists in group order (x, [g1], [g2], [out])
//   e->dscalars     : scalar(s): single -> [s]; per-tensor list -> [s_0 .. s_{N-1}]
//   e->a            : per-tensor scalar source as a 1-D fp32 tensor (ScalarList / addc List variants)
//   e->out          : reduction output (norm)
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
#include <vector>

namespace {

float *hp(const aclTensor *t) { return (float *)t->data + t->offset; }
const aclTensorList *L(aclOpExecutor *e, int i) { return e->tl[i]; }
int64_t LN(const aclTensorList *l) { return (int64_t)l->v.size(); }
const aclTensor *LT(const aclTensorList *l, int64_t i) { return l->v[i]; }

// Foreach "shape": how the lists/scalars feed the per-element math.
enum FShape {
    FS_UNARY = 1,    // out_i = f(x_i)
    FS_SCALAR,       // out_i = g(x_i, s)            s = dscalars[0]
    FS_SCALARLIST,   // out_i = g(x_i, s_i)          s_i = dscalars[i]
    FS_LIST,         // out_i = g(x_i, y_i)          (alpha = dscalars[0] or 1 for add/sub)
    FS_ADDC,         // out_i = x_i + s * (t1_i (*|/) t2_i)   s scalar or per-tensor
    FS_POWSAT,       // out_i = base ^ x_i           base = dscalars[0]
};

// Elementwise math selector. Mirrors elementwise.metal's un_op/bin_op so results match the rest
// of the backend bit-for-bit where the libm primitives agree.
enum EwKind {
    // unary
    E_ABS = 1, E_ACOS, E_ASIN, E_ATAN, E_COS, E_COSH, E_ERF, E_ERFC, E_EXP, E_EXPM1,
    E_LOG, E_LOG10, E_LOG1P, E_LOG2, E_NEG, E_RECIP, E_SIGMOID, E_SIGN, E_SIN, E_SINH,
    E_SQRT, E_TAN, E_TANH, E_ROUND,
    // binary (x op s/y)
    E_ADD, E_SUB, E_MUL, E_DIV, E_MAX, E_MIN, E_POW,
    // addc sub-mode
    E_ADDCMUL, E_ADDCDIV,
};

inline float un_apply(int k, float x) {
    switch (k) {
        case E_ABS:    return std::fabs(x);
        case E_ACOS:   return std::acos(x);
        case E_ASIN:   return std::asin(x);
        case E_ATAN:   return std::atan(x);
        case E_COS:    return std::cos(x);
        case E_COSH:   return std::cosh(x);
        case E_ERF:    return std::erf(x);
        case E_ERFC:   return std::erfc(x);
        case E_EXP:    return std::exp(x);
        case E_EXPM1:  return std::expm1(x);
        case E_LOG:    return std::log(x);
        case E_LOG10:  return std::log10(x);
        case E_LOG1P:  return std::log1p(x);
        case E_LOG2:   return std::log2(x);
        case E_NEG:    return -x;
        case E_RECIP:  return 1.0f / x;
        case E_SIGMOID:return 1.0f / (1.0f + std::exp(-x));
        case E_SIGN:   return (float)((x > 0.0f) - (x < 0.0f));
        case E_SIN:    return std::sin(x);
        case E_SINH:   return std::sinh(x);
        case E_SQRT:   return std::sqrt(x);
        case E_TAN:    return std::tan(x);
        case E_TANH:   return std::tanh(x);
        case E_ROUND:  return std::rint(x);   // banker's rounding (matches FRound / elementwise.metal)
    }
    return x;
}

// x op b, with alpha applied to add/sub (b is the scalar or the second-list element).
inline float bin_apply(int k, float a, float b, float alpha) {
    switch (k) {
        case E_ADD: return a + alpha * b;
        case E_SUB: return a - alpha * b;
        case E_MUL: return a * b;
        case E_DIV: return a / b;
        case E_MAX: return a > b ? a : b;
        case E_MIN: return a < b ? a : b;
        case E_POW: return std::pow(a, b);
    }
    return a;
}

aclnnStatus run_foreach_parity(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];   // drain pending GPU writes
    const int k = (int)e->n;
    switch (e->m) {
        case FS_UNARY: {
            auto *in = L(e, 0), *o = L(e, 1);
            for (int64_t t = 0; t < LN(in); ++t) {
                const float *a = hp(LT(in, t)); float *y = hp((aclTensor *)LT(o, t));
                int64_t n = LT(in, t)->numel();
                for (int64_t j = 0; j < n; ++j) y[j] = un_apply(k, a[j]);
            }
            break; }
        case FS_SCALAR: {
            auto *in = L(e, 0), *o = L(e, 1); float sc = (float)e->dscalars[0];
            for (int64_t t = 0; t < LN(in); ++t) {
                const float *a = hp(LT(in, t)); float *y = hp((aclTensor *)LT(o, t));
                int64_t n = LT(in, t)->numel();
                for (int64_t j = 0; j < n; ++j) y[j] = bin_apply(k, a[j], sc, 1.0f);
            }
            break; }
        case FS_SCALARLIST: {
            auto *in = L(e, 0), *o = L(e, 1);
            for (int64_t t = 0; t < LN(in); ++t) {
                const float *a = hp(LT(in, t)); float *y = hp((aclTensor *)LT(o, t));
                float sc = (float)e->dscalars[t]; int64_t n = LT(in, t)->numel();
                for (int64_t j = 0; j < n; ++j) y[j] = bin_apply(k, a[j], sc, 1.0f);
            }
            break; }
        case FS_LIST: {
            auto *la = L(e, 0), *lb = L(e, 1), *o = L(e, 2);
            float alpha = e->dscalars.empty() ? 1.0f : (float)e->dscalars[0];
            for (int64_t t = 0; t < LN(la); ++t) {
                const float *a = hp(LT(la, t)), *b = hp(LT(lb, t)); float *y = hp((aclTensor *)LT(o, t));
                int64_t n = LT(la, t)->numel();
                for (int64_t j = 0; j < n; ++j) y[j] = bin_apply(k, a[j], b[j], alpha);
            }
            break; }
        case FS_ADDC: {
            auto *lx = L(e, 0), *l1 = L(e, 1), *l2 = L(e, 2), *o = L(e, 3);
            bool perTensor = (int64_t)e->dscalars.size() > 1;
            for (int64_t t = 0; t < LN(lx); ++t) {
                const float *x = hp(LT(lx, t)), *t1 = hp(LT(l1, t)), *t2 = hp(LT(l2, t));
                float *y = hp((aclTensor *)LT(o, t));
                float sc = (float)(perTensor ? e->dscalars[t] : e->dscalars[0]);
                int64_t n = LT(lx, t)->numel();
                for (int64_t j = 0; j < n; ++j)
                    y[j] = x[j] + sc * (k == E_ADDCDIV ? t1[j] / t2[j] : t1[j] * t2[j]);
            }
            break; }
        case FS_POWSAT: {
            auto *in = L(e, 0), *o = L(e, 1); float base = (float)e->dscalars[0];
            for (int64_t t = 0; t < LN(in); ++t) {
                const float *a = hp(LT(in, t)); float *y = hp((aclTensor *)LT(o, t));
                int64_t n = LT(in, t)->numel();
                for (int64_t j = 0; j < n; ++j) y[j] = std::pow(base, a[j]);
            }
            break; }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// Read a per-tensor scalar list provided as a 1-D fp32 unified-memory tensor (length N).
bool read_scalars(const aclTensor *st, int64_t N, std::vector<double> &out) {
    if (!st || !st->data || st->numel() != N || st->dtype != ACL_FLOAT) return false;
    const float *p = hp(st);
    out.assign(p, p + N);
    return true;
}

} // namespace

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; \
    @autoreleasepool { st = run_foreach_parity(e, s); } delete e; return st; }

extern "C" {

// ---- per-tensor unary: (const aclTensorList* x, const aclTensorList* out, ws, ex) ----
#define DEF_UNARY(NAME, EW)                                                                               \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *out,                      \
                                   uint64_t *ws, aclOpExecutor **ex) {                                    \
    if (!ws || !x || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                       \
    auto *e = new aclOpExecutor(); e->m = FS_UNARY; e->n = EW; e->tl[0] = x; e->tl[1] = out;              \
    *ex = e; return ACLNN_SUCCESS; }                                                                      \
RUN(NAME)

// ---- single scalar applied to all (Scalar and ScalarV2 share this) ----
#define DEF_SCALAR(NAME, EW)                                                                              \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclScalar *scalar,                       \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {          \
    if (!ws || !x || !scalar || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                            \
    auto *e = new aclOpExecutor(); e->m = FS_SCALAR; e->n = EW; e->tl[0] = x; e->tl[1] = out;             \
    e->dscalars = {scalar->v}; *ex = e; return ACLNN_SUCCESS; }                                           \
RUN(NAME)

// ---- per-tensor scalar list (scalars given as a 1-D fp32 tensor of length N) ----
#define DEF_SCALARLIST(NAME, EW)                                                                          \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensor *scalars,                      \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {          \
    if (!ws || !x || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                       \
    std::vector<double> sc; if (!read_scalars(scalars, (int64_t)x->v.size(), sc)) return ACLNN_ERR_PARAM_INVALID; \
    auto *e = new aclOpExecutor(); e->m = FS_SCALARLIST; e->n = EW; e->tl[0] = x; e->tl[1] = out;         \
    e->dscalars = sc; *ex = e; return ACLNN_SUCCESS; }                                                    \
RUN(NAME)

// ---- per-tensor binary between two lists ----
#define DEF_LIST(NAME, EW)                                                                                \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y,                        \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {          \
    if (!ws || !x || !y || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                 \
    auto *e = new aclOpExecutor(); e->m = FS_LIST; e->n = EW; e->tl[0] = x; e->tl[1] = y; e->tl[2] = out; \
    *ex = e; return ACLNN_SUCCESS; }                                                                      \
RUN(NAME)

// ---- per-tensor binary between two lists with a scalar alpha (Sub V2) ----
#define DEF_LISTV2(NAME, EW)                                                                              \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y, const aclScalar *alpha,\
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {          \
    if (!ws || !x || !y || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                 \
    auto *e = new aclOpExecutor(); e->m = FS_LIST; e->n = EW; e->tl[0] = x; e->tl[1] = y; e->tl[2] = out; \
    e->dscalars = {alpha ? alpha->v : 1.0}; *ex = e; return ACLNN_SUCCESS; }                              \
RUN(NAME)

// ---- addcmul / addcdiv with one scalar (ScalarV2) ----
#define DEF_ADDC_SCALAR(NAME, EW)                                                                         \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2,\
                                   const aclScalar *scalar, const aclTensorList *out,                     \
                                   uint64_t *ws, aclOpExecutor **ex) {                                    \
    if (!ws || !x || !t1 || !t2 || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                         \
    auto *e = new aclOpExecutor(); e->m = FS_ADDC; e->n = EW;                                             \
    e->tl[0] = x; e->tl[1] = t1; e->tl[2] = t2; e->tl[3] = out;                                           \
    e->dscalars = {scalar ? scalar->v : 1.0}; *ex = e; return ACLNN_SUCCESS; }                            \
RUN(NAME)

// ---- addcmul / addcdiv with per-tensor scalars (ScalarList and List share this) ----
#define DEF_ADDC_LIST(NAME, EW)                                                                           \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2,\
                                   const aclTensor *scalars, const aclTensorList *out,                    \
                                   uint64_t *ws, aclOpExecutor **ex) {                                    \
    if (!ws || !x || !t1 || !t2 || !out) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                         \
    std::vector<double> sc; if (!read_scalars(scalars, (int64_t)x->v.size(), sc)) return ACLNN_ERR_PARAM_INVALID; \
    if (sc.size() == 1) sc.push_back(sc[0]);  /* force per-tensor path even when N==1 */                  \
    auto *e = new aclOpExecutor(); e->m = FS_ADDC; e->n = EW;                                             \
    e->tl[0] = x; e->tl[1] = t1; e->tl[2] = t2; e->tl[3] = out;                                           \
    e->dscalars = sc; *ex = e; return ACLNN_SUCCESS; }                                                    \
RUN(NAME)

// ===== unary (22 of the 24 listed; Sqrt is already in foreach.mm) =====
DEF_UNARY(aclnnForeachAbs, E_ABS)
DEF_UNARY(aclnnForeachAcos, E_ACOS)
DEF_UNARY(aclnnForeachAsin, E_ASIN)
DEF_UNARY(aclnnForeachAtan, E_ATAN)
DEF_UNARY(aclnnForeachCos, E_COS)
DEF_UNARY(aclnnForeachCosh, E_COSH)
DEF_UNARY(aclnnForeachErf, E_ERF)
DEF_UNARY(aclnnForeachErfc, E_ERFC)
DEF_UNARY(aclnnForeachExp, E_EXP)
DEF_UNARY(aclnnForeachExpm1, E_EXPM1)
DEF_UNARY(aclnnForeachLog, E_LOG)
DEF_UNARY(aclnnForeachLog10, E_LOG10)
DEF_UNARY(aclnnForeachLog1p, E_LOG1P)
DEF_UNARY(aclnnForeachLog2, E_LOG2)
DEF_UNARY(aclnnForeachNeg, E_NEG)
DEF_UNARY(aclnnForeachReciprocal, E_RECIP)
DEF_UNARY(aclnnForeachSigmoid, E_SIGMOID)
DEF_UNARY(aclnnForeachSign, E_SIGN)
DEF_UNARY(aclnnForeachSin, E_SIN)
DEF_UNARY(aclnnForeachSinh, E_SINH)
DEF_UNARY(aclnnForeachTan, E_TAN)
DEF_UNARY(aclnnForeachTanh, E_TANH)
DEF_UNARY(aclnnForeachRoundOffNumber, E_ROUND)
DEF_UNARY(aclnnForeachRoundOffNumberV2, E_ROUND)

// ===== single scalar (Scalar + ScalarV2). MulScalar is already in foreach.mm. =====
DEF_SCALAR(aclnnForeachAddScalar, E_ADD)
DEF_SCALAR(aclnnForeachAddScalarV2, E_ADD)
DEF_SCALAR(aclnnForeachSubScalar, E_SUB)
DEF_SCALAR(aclnnForeachSubScalarV2, E_SUB)
DEF_SCALAR(aclnnForeachMulScalarV2, E_MUL)
DEF_SCALAR(aclnnForeachDivScalar, E_DIV)
DEF_SCALAR(aclnnForeachDivScalarV2, E_DIV)
DEF_SCALAR(aclnnForeachMaximumScalar, E_MAX)
DEF_SCALAR(aclnnForeachMaximumScalarV2, E_MAX)
DEF_SCALAR(aclnnForeachMinimumScalar, E_MIN)
DEF_SCALAR(aclnnForeachMinimumScalarV2, E_MIN)
DEF_SCALAR(aclnnForeachPowScalar, E_POW)
DEF_SCALAR(aclnnForeachPowScalarV2, E_POW)

// ===== scalar list. AddScalarList is already in foreach.mm. =====
DEF_SCALARLIST(aclnnForeachSubScalarList, E_SUB)
DEF_SCALARLIST(aclnnForeachMulScalarList, E_MUL)
DEF_SCALARLIST(aclnnForeachDivScalarList, E_DIV)
DEF_SCALARLIST(aclnnForeachMaximumScalarList, E_MAX)
DEF_SCALARLIST(aclnnForeachMinimumScalarList, E_MIN)
DEF_SCALARLIST(aclnnForeachPowScalarList, E_POW)

// ===== list ∘ list. AddList / AddListV2 are already in foreach.mm. =====
DEF_LIST(aclnnForeachSubList, E_SUB)
DEF_LIST(aclnnForeachMulList, E_MUL)
DEF_LIST(aclnnForeachDivList, E_DIV)
DEF_LIST(aclnnForeachMaximumList, E_MAX)
DEF_LIST(aclnnForeachMinimumList, E_MIN)
DEF_LIST(aclnnForeachPowList, E_POW)
DEF_LISTV2(aclnnForeachSubListV2, E_SUB)

// ===== addcmul / addcdiv. AddcmulScalar is already in foreach.mm. =====
DEF_ADDC_SCALAR(aclnnForeachAddcmulScalarV2, E_ADDCMUL)
DEF_ADDC_SCALAR(aclnnForeachAddcdivScalar, E_ADDCDIV)
DEF_ADDC_SCALAR(aclnnForeachAddcdivScalarV2, E_ADDCDIV)
DEF_ADDC_LIST(aclnnForeachAddcmulScalarList, E_ADDCMUL)
DEF_ADDC_LIST(aclnnForeachAddcdivScalarList, E_ADDCDIV)
DEF_ADDC_LIST(aclnnForeachAddcmulList, E_ADDCMUL)
DEF_ADDC_LIST(aclnnForeachAddcdivList, E_ADDCDIV)

} // extern "C"
