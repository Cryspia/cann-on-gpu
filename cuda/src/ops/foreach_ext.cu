// Foreach op family (ops-nn / foreach, 70 ops): apply one elementwise operation across every tensor
// in an aclTensorList. The functional landing loops over the list and reuses the single-tensor
// elementwise dispatch (ew_run_one) per tensor; a few shapes (scalar^tensor, list-lerp, norm,
// non-finite-check, copy/zero) use small dedicated kernels here.
//
// Performance note: this issues one kernel launch per tensor. Fusing the whole list into a single
// multi-tensor kernel (one launch, grid-stride over a packed tensor-meta array) is recorded as a
// backlog item in cuda/TODO.md Part B and intentionally deferred.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace {

constexpr int FT = 256;
inline int64_t fgrid(int64_t n) { return (n + FT - 1) / FT; }

// Variant codes (how the Execute phase interprets the flattened input groups stored in e->inputs)
enum FVariant {
    FV_UNARY = 1,    // [x, out]              per-tensor unary e->op
    FV_SCALAR,       // [x, out]              per-tensor scalar e->op, scalar = dscalars[0]
    FV_SCALARLIST,   // [x, out]              per-tensor scalar e->op, scalar = dscalars[i]
    FV_LIST,         // [x, y, out]           per-tensor binary e->op (alpha = dscalars[0] or 1)
    FV_ADDC,         // [x, t1, t2, out]      addcmul/addcdiv; scalar = dscalars[0] or dscalars[i]
    FV_LERP_SCALAR,  // [x, end, out]         lerp, weight = dscalars[0]
    FV_LERP_LIST,    // [x, end, weight, out] lerp, per-element weight tensor
    FV_POWSAT,       // [x, out]              scalar^tensor, base = dscalars[0]
    FV_COPY,         // [x, out]              out_i = x_i
    FV_ZERO,         // [x]                   x_i = 0 (in place)
    FV_NORM,         // [x] + e->out          out[i] = ||x_i||_p, p = dscalars[0]
    FV_NONFINITE,    // [x] + e->out + e->b   in-place scale by *invScale, set foundInf on non-finite
};

// load/store with float accumulation across fp32 / fp16 / bf16
template <typename T> __device__ inline float fld(T v) { return (float)v; }
template <> __device__ inline float fld(__half v) { return __half2float(v); }
template <> __device__ inline float fld(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ inline T fst(float v) { return (T)v; }
template <> __device__ inline __half fst(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 fst(float v) { return __float2bfloat16(v); }

template <typename T> __global__ void k_lerp3(const T *x, const T *e, const T *w, T *o, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) { float xi = fld(x[i]); o[i] = fst<T>(xi + fld(w[i]) * (fld(e[i]) - xi)); }
}
template <typename T> __global__ void k_powsat(const T *x, T *o, float base, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) o[i] = fst<T>(powf(base, fld(x[i])));
}
template <typename T> __global__ void k_nonfinite(T *x, float *found, const float *invScale, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) { float v = fld(x[i]); if (!isfinite(v)) *found = 1.0f; x[i] = fst<T>(v * (*invScale)); }
}
// L_p norm of one tensor reduced by a single block; writes a single float result
template <typename T> __global__ void k_norm(const T *x, float *out, float p, int64_t n) {
    __shared__ float sh[FT];
    float acc = 0.f;
    for (int64_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = fabsf(fld(x[i]));
        acc += (p == 2.f) ? v * v : powf(v, p);
    }
    sh[threadIdx.x] = acc; __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) { if ((int)threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s]; __syncthreads(); }
    if (threadIdx.x == 0) *out = (p == 2.f) ? sqrtf(sh[0]) : powf(sh[0], 1.0f / p);
}

#define FDISP(KCALL) switch (dt) { \
    case ACL_FLOAT:   KCALL(float); break; \
    case ACL_FLOAT16: KCALL(__half); break; \
    case ACL_BF16:    KCALL(__nv_bfloat16); break; \
    default: st = ACLNN_ERR_PARAM_INVALID; }

// Flatten the participating lists into e->inputs in group order (x, [g1], [g2], [out]).
// All lists must have the same length N; every tensor must be non-null with device data.
aclnnStatus fe_build(int op, int variant, const aclTensorList *x, const aclTensorList *g1,
                     const aclTensorList *g2, const aclTensorList *out,
                     const std::vector<double> &sc, aclOpExecutor **ex) {
    if (!x || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t N = (int64_t)x->v.size();
    if (N == 0) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = op; e->n = variant; e->m = N; e->dscalars = sc;
    auto push = [&](const aclTensorList *L) -> aclnnStatus {
        if (!L || (int64_t)L->v.size() != N) return ACLNN_ERR_PARAM_INVALID;
        for (auto *t : L->v) { if (!t || !t->data) return ACLNN_ERR_PARAM_NULLPTR; e->inputs.push_back(t); }
        return ACLNN_SUCCESS;
    };
    aclnnStatus st = push(x);
    if (st == ACLNN_SUCCESS && g1) st = push(g1);
    if (st == ACLNN_SUCCESS && g2) st = push(g2);
    if (st == ACLNN_SUCCESS && out) st = push(out);
    if (st != ACLNN_SUCCESS) { delete e; return st; }
    *ex = e;
    return ACLNN_SUCCESS;
}

// Read a per-tensor scalar list provided as a 1-D fp32 device tensor (length N) into host doubles.
aclnnStatus read_scalars(const aclTensor *s, int64_t N, std::vector<double> &out) {
    if (!s || !s->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (s->numel() != N || s->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    std::vector<float> h(N);
    if (cudaMemcpy(h.data(), s->data, N * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess)
        return ACLNN_ERR_RUNTIME_ERROR;
    out.assign(h.begin(), h.end());
    return ACLNN_SUCCESS;
}

inline aclTensor *CC(const aclTensor *t) { return const_cast<aclTensor *>(t); }

// Shared Execute: walk the list and dispatch per tensor. Frees the executor (CANN one-shot semantics).
aclnnStatus fe_run(aclOpExecutor *e, void *ws, cudaStream_t s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    const int64_t N = e->m;
    const aclTensor **in = e->inputs.data();
    aclnnStatus st = ACLNN_SUCCESS;
    for (int64_t i = 0; i < N && st == ACLNN_SUCCESS; ++i) {
        switch (e->n) {
            case FV_UNARY:
                st = ew_run_one(e->op, in[i], nullptr, nullptr, CC(in[N + i]), 1.0, nullptr, s); break;
            case FV_SCALAR:
                st = ew_run_one(e->op, in[i], nullptr, nullptr, CC(in[N + i]), e->dscalars[0], nullptr, s); break;
            case FV_SCALARLIST:
                st = ew_run_one(e->op, in[i], nullptr, nullptr, CC(in[N + i]), e->dscalars[i], nullptr, s); break;
            case FV_LIST:
                st = ew_run_one(e->op, in[i], in[N + i], nullptr, CC(in[2 * N + i]),
                                e->dscalars.empty() ? 1.0 : e->dscalars[0], nullptr, s); break;
            case FV_ADDC:
                st = ew_run_one(e->op, in[i], in[N + i], in[2 * N + i], CC(in[3 * N + i]),
                                e->dscalars.size() == 1 ? e->dscalars[0] : e->dscalars[i], nullptr, s); break;
            case FV_LERP_SCALAR:
                st = ew_run_one(OP_LERP, in[i], in[N + i], nullptr, CC(in[2 * N + i]), e->dscalars[0], nullptr, s); break;
            case FV_LERP_LIST: {
                const aclTensor *x = in[i], *end = in[N + i], *w = in[2 * N + i]; aclTensor *o = CC(in[3 * N + i]);
                int64_t n = x->numel(); aclDataType dt = x->dtype;
                #define KL(T) k_lerp3<T><<<fgrid(n), FT, 0, s>>>((const T*)x->data, (const T*)end->data, (const T*)w->data, (T*)o->data, n)
                FDISP(KL)
                #undef KL
                break; }
            case FV_POWSAT: {
                const aclTensor *x = in[i]; aclTensor *o = CC(in[N + i]);
                int64_t n = x->numel(); aclDataType dt = x->dtype;
                #define KP(T) k_powsat<T><<<fgrid(n), FT, 0, s>>>((const T*)x->data, (T*)o->data, (float)e->dscalars[0], n)
                FDISP(KP)
                #undef KP
                break; }
            case FV_COPY: {
                const aclTensor *x = in[i]; aclTensor *o = CC(in[N + i]);
                if (cudaMemcpyAsync(o->data, x->data, x->numel() * dtype_size(x->dtype), cudaMemcpyDeviceToDevice, s) != cudaSuccess)
                    st = ACLNN_ERR_RUNTIME_ERROR;
                break; }
            case FV_ZERO: {
                aclTensor *x = CC(in[i]);
                if (cudaMemsetAsync(x->data, 0, x->numel() * dtype_size(x->dtype), s) != cudaSuccess)
                    st = ACLNN_ERR_RUNTIME_ERROR;
                break; }
            case FV_NORM: {
                const aclTensor *x = in[i]; aclDataType dt = x->dtype;
                int64_t n = x->numel(); float *op = ((float *)e->out->data) + i;
                #define KN(T) k_norm<T><<<1, FT, 0, s>>>((const T*)x->data, op, (float)e->dscalars[0], n)
                FDISP(KN)
                #undef KN
                break; }
            case FV_NONFINITE: {
                aclTensor *x = CC(in[i]); aclDataType dt = x->dtype; int64_t n = x->numel();
                #define KF(T) k_nonfinite<T><<<fgrid(n), FT, 0, s>>>((T*)x->data, (float*)e->out->data, (const float*)e->b->data, n)
                FDISP(KF)
                #undef KF
                break; }
            default: st = ACLNN_ERR_PARAM_INVALID;
        }
    }
    delete e;
    if (st != ACLNN_SUCCESS) return st;
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

extern "C" {

// ---- per-tensor unary ----
#define DEF_UNARY(NAME, OP)                                                                                  \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *out,                         \
                                   uint64_t *ws, aclOpExecutor **ex) {                                       \
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                        \
    return fe_build(OP, FV_UNARY, x, nullptr, nullptr, out, {}, ex); }                                       \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- per-tensor, single scalar applied to all (Scalar and ScalarV2 share this) ----
#define DEF_SCALAR(NAME, OP)                                                                                 \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclScalar *scalar,                          \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {             \
    if (!ws || !scalar) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                             \
    return fe_build(OP, FV_SCALAR, x, nullptr, nullptr, out, {scalar->v}, ex); }                            \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- per-tensor scalar list (scalars given as a 1-D fp32 tensor of length N) ----
#define DEF_SCALARLIST(NAME, OP)                                                                             \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensor *scalars,                         \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {             \
    if (!ws || !x) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                  \
    std::vector<double> sc; aclnnStatus st = read_scalars(scalars, (int64_t)x->v.size(), sc);               \
    if (st != ACLNN_SUCCESS) return st;                                                                      \
    return fe_build(OP, FV_SCALARLIST, x, nullptr, nullptr, out, sc, ex); }                                  \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- per-tensor binary between two lists ----
#define DEF_LIST(NAME, OP)                                                                                   \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y,                           \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {             \
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                        \
    return fe_build(OP, FV_LIST, x, y, nullptr, out, {}, ex); }                                              \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- per-tensor binary between two lists with a scalar alpha (Add/Sub V2) ----
#define DEF_LISTV2(NAME, OP)                                                                                 \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y, const aclScalar *alpha,   \
                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {             \
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                        \
    return fe_build(OP, FV_LIST, x, y, nullptr, out, {alpha ? alpha->v : 1.0}, ex); }                        \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- addcmul / addcdiv with one scalar (Scalar and ScalarV2 share this) ----
#define DEF_ADDC_SCALAR(NAME, OP)                                                                            \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2, \
                                   const aclScalar *scalar, const aclTensorList *out,                        \
                                   uint64_t *ws, aclOpExecutor **ex) {                                       \
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                        \
    return fe_build(OP, FV_ADDC, x, t1, t2, out, {scalar ? scalar->v : 1.0}, ex); }                          \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ---- addcmul / addcdiv with per-tensor scalars (ScalarList and List share this) ----
#define DEF_ADDC_LIST(NAME, OP)                                                                              \
aclnnStatus NAME##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2, \
                                   const aclTensor *scalars, const aclTensorList *out,                       \
                                   uint64_t *ws, aclOpExecutor **ex) {                                       \
    if (!ws || !x) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;                                                  \
    std::vector<double> sc; aclnnStatus st = read_scalars(scalars, (int64_t)x->v.size(), sc);               \
    if (st != ACLNN_SUCCESS) return st;                                                                      \
    return fe_build(OP, FV_ADDC, x, t1, t2, out, sc, ex); }                                                  \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ===== unary (23 + 2 round) =====
DEF_UNARY(aclnnForeachAbs, OP_ABS)
DEF_UNARY(aclnnForeachAcos, OP_ACOS)
DEF_UNARY(aclnnForeachAsin, OP_ASIN)
DEF_UNARY(aclnnForeachAtan, OP_ATAN)
DEF_UNARY(aclnnForeachCos, OP_COS)
DEF_UNARY(aclnnForeachCosh, OP_COSH)
DEF_UNARY(aclnnForeachErf, OP_ERF)
DEF_UNARY(aclnnForeachErfc, OP_ERFC)
DEF_UNARY(aclnnForeachExp, OP_EXP)
DEF_UNARY(aclnnForeachExpm1, OP_EXPM1)
DEF_UNARY(aclnnForeachLog, OP_LOG)
DEF_UNARY(aclnnForeachLog10, OP_LOG10)
DEF_UNARY(aclnnForeachLog1p, OP_LOG1P)
DEF_UNARY(aclnnForeachLog2, OP_LOG2)
DEF_UNARY(aclnnForeachNeg, OP_NEG)
DEF_UNARY(aclnnForeachReciprocal, OP_RECIPROCAL)
DEF_UNARY(aclnnForeachSigmoid, OP_SIGMOID)
DEF_UNARY(aclnnForeachSign, OP_SIGN)
DEF_UNARY(aclnnForeachSin, OP_SIN)
DEF_UNARY(aclnnForeachSinh, OP_SINH)
DEF_UNARY(aclnnForeachSqrt, OP_SQRT)
DEF_UNARY(aclnnForeachTan, OP_TAN)
DEF_UNARY(aclnnForeachTanh, OP_TANH)
DEF_UNARY(aclnnForeachRoundOffNumber, OP_ROUND)
DEF_UNARY(aclnnForeachRoundOffNumberV2, OP_ROUND)

// ===== single scalar (7 + 7 V2) =====
DEF_SCALAR(aclnnForeachAddScalar, OP_ADDS)
DEF_SCALAR(aclnnForeachAddScalarV2, OP_ADDS)
DEF_SCALAR(aclnnForeachSubScalar, OP_SUBS)
DEF_SCALAR(aclnnForeachSubScalarV2, OP_SUBS)
DEF_SCALAR(aclnnForeachMulScalar, OP_MULS)
DEF_SCALAR(aclnnForeachMulScalarV2, OP_MULS)
DEF_SCALAR(aclnnForeachDivScalar, OP_DIVS)
DEF_SCALAR(aclnnForeachDivScalarV2, OP_DIVS)
DEF_SCALAR(aclnnForeachMaximumScalar, OP_CLAMP_MIN)
DEF_SCALAR(aclnnForeachMaximumScalarV2, OP_CLAMP_MIN)
DEF_SCALAR(aclnnForeachMinimumScalar, OP_CLAMP_MAX)
DEF_SCALAR(aclnnForeachMinimumScalarV2, OP_CLAMP_MAX)
DEF_SCALAR(aclnnForeachPowScalar, OP_POWS)
DEF_SCALAR(aclnnForeachPowScalarV2, OP_POWS)

// ===== scalar list (7) =====
DEF_SCALARLIST(aclnnForeachAddScalarList, OP_ADDS)
DEF_SCALARLIST(aclnnForeachSubScalarList, OP_SUBS)
DEF_SCALARLIST(aclnnForeachMulScalarList, OP_MULS)
DEF_SCALARLIST(aclnnForeachDivScalarList, OP_DIVS)
DEF_SCALARLIST(aclnnForeachMaximumScalarList, OP_CLAMP_MIN)
DEF_SCALARLIST(aclnnForeachMinimumScalarList, OP_CLAMP_MAX)
DEF_SCALARLIST(aclnnForeachPowScalarList, OP_POWS)

// ===== list ∘ list (7 + 2 V2) =====
DEF_LIST(aclnnForeachAddList, OP_ADD)
DEF_LIST(aclnnForeachSubList, OP_SUB)
DEF_LIST(aclnnForeachMulList, OP_MUL)
DEF_LIST(aclnnForeachDivList, OP_DIV)
DEF_LIST(aclnnForeachMaximumList, OP_MAXIMUM)
DEF_LIST(aclnnForeachMinimumList, OP_MINIMUM)
DEF_LIST(aclnnForeachPowList, OP_POW)
DEF_LISTV2(aclnnForeachAddListV2, OP_ADD)
DEF_LISTV2(aclnnForeachSubListV2, OP_SUB)

// ===== addcmul / addcdiv (4 scalar + 4 per-tensor) =====
DEF_ADDC_SCALAR(aclnnForeachAddcmulScalar, OP_ADDCMUL)
DEF_ADDC_SCALAR(aclnnForeachAddcmulScalarV2, OP_ADDCMUL)
DEF_ADDC_SCALAR(aclnnForeachAddcdivScalar, OP_ADDCDIV)
DEF_ADDC_SCALAR(aclnnForeachAddcdivScalarV2, OP_ADDCDIV)
DEF_ADDC_LIST(aclnnForeachAddcmulScalarList, OP_ADDCMUL)
DEF_ADDC_LIST(aclnnForeachAddcdivScalarList, OP_ADDCDIV)
DEF_ADDC_LIST(aclnnForeachAddcmulList, OP_ADDCMUL)
DEF_ADDC_LIST(aclnnForeachAddcdivList, OP_ADDCDIV)

// ===== pow(scalar, tensor) =====
aclnnStatus aclnnForeachPowScalarAndTensorGetWorkspaceSize(const aclScalar *scalar, const aclTensorList *x,
                                                           const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws || !scalar) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    return fe_build(0, FV_POWSAT, x, nullptr, nullptr, out, {scalar->v}, ex);
}
aclnnStatus aclnnForeachPowScalarAndTensor(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ===== lerp =====
aclnnStatus aclnnForeachLerpScalarGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclScalar *weight,
                                                   const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    return fe_build(OP_LERP, FV_LERP_SCALAR, x, end, nullptr, out, {weight ? weight->v : 0.0}, ex);
}
aclnnStatus aclnnForeachLerpScalar(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnForeachLerpListGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclTensorList *weight,
                                                 const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    return fe_build(0, FV_LERP_LIST, x, end, weight, out, {}, ex);
}
aclnnStatus aclnnForeachLerpList(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ===== copy / zero =====
aclnnStatus aclnnForeachCopyGetWorkspaceSize(const aclTensorList *x, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    return fe_build(0, FV_COPY, x, nullptr, nullptr, out, {}, ex);
}
aclnnStatus aclnnForeachCopy(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }
aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(const aclTensorList *x, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    return fe_build(0, FV_ZERO, x, nullptr, nullptr, nullptr, {}, ex);
}
aclnnStatus aclnnForeachZeroInplace(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ===== norm (out: 1-D fp32 tensor, one value per input tensor) =====
aclnnStatus aclnnForeachNormGetWorkspaceSize(const aclTensorList *x, const aclScalar *p, aclTensor *out,
                                             uint64_t *ws, aclOpExecutor **ex) {
    if (!ws || !out || !out->data) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    aclnnStatus st = fe_build(0, FV_NORM, x, nullptr, nullptr, nullptr, {p ? p->v : 2.0}, ex);
    if (st == ACLNN_SUCCESS) (*ex)->out = out;
    return st;
}
aclnnStatus aclnnForeachNorm(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

// ===== AMP non-finite check + unscale (in place; foundInf set to 1 on any inf/nan) =====
aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *x, aclTensor *foundInf,
                                                                 const aclTensor *invScale, uint64_t *ws, aclOpExecutor **ex) {
    if (!ws || !foundInf || !foundInf->data || !invScale || !invScale->data) return ACLNN_ERR_PARAM_NULLPTR; *ws = 0;
    aclnnStatus st = fe_build(0, FV_NONFINITE, x, nullptr, nullptr, nullptr, {}, ex);
    if (st == ACLNN_SUCCESS) { (*ex)->out = foundInf; (*ex)->b = invScale; }
    return st;
}
aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fe_run(e, ws, (cudaStream_t)s); }

} // extern "C"
