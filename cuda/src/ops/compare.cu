// Compare / logical / predicate op family (output ACL_BOOL=uint8):
// Greater/Less/Equal/NotEqual/GreaterEqual/LessEqual, LogicalAnd/Or/Not, IsNan/IsFinite, MaskedFill.
// All support broadcasting (indexed via out coordinates + real per-input stride+offset).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int MAXD = 8, TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

struct Desc { int rank; int64_t od[MAXD], as[MAXD], bs[MAXD], aoff, boff; };
inline void fill_desc(const aclTensor *a, const aclTensor *b, const aclTensor *out, Desc &d) {
    int r = (int)out->viewDims.size(), ra = (int)a->viewDims.size(), rb = b ? (int)b->viewDims.size() : 0;
    d.rank = r; d.aoff = a->offset; d.boff = b ? b->offset : 0;
    for (int i = 0; i < r; ++i) {
        d.od[i] = out->viewDims[i];
        int ia = i - (r - ra);
        d.as[i] = (ia >= 0 && a->viewDims[ia] == out->viewDims[i]) ? a->strides[ia] : 0;
        if (b) { int ib = i - (r - rb); d.bs[i] = (ib >= 0 && b->viewDims[ib] == out->viewDims[i]) ? b->strides[ib] : 0; }
        else d.bs[i] = 0;
    }
}
__device__ inline void coords(int64_t i, const Desc &d, int64_t &ia, int64_t &ib) {
    int64_t rem = i; ia = d.aoff; ib = d.boff;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t c = rem % d.od[k]; rem /= d.od[k]; ia += c * d.as[k]; ib += c * d.bs[k]; }
}

template <typename A> struct CGt { __device__ static bool f(A a, A b) { return a > b; } };
template <typename A> struct CLt { __device__ static bool f(A a, A b) { return a < b; } };
template <typename A> struct CEq { __device__ static bool f(A a, A b) { return a == b; } };
template <typename A> struct CNe { __device__ static bool f(A a, A b) { return a != b; } };
template <typename A> struct CGe { __device__ static bool f(A a, A b) { return a >= b; } };
template <typename A> struct CLe { __device__ static bool f(A a, A b) { return a <= b; } };

template <typename T, typename A, template <typename> class C>
__global__ void k_cmp(const T *a, const T *b, uint8_t *o, int64_t n, Desc d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t ia, ib; coords(i, d, ia, ib);
    o[i] = C<A>::f((A)a[ia], (A)b[ib]) ? 1 : 0;
}
// tensor ∘ scalar comparison (self contiguous; scalar in accumulation type A)
template <typename T, typename A, template <typename> class C>
__global__ void k_cmp_s(const T *a, uint8_t *o, A s, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    o[i] = C<A>::f((A)a[i], s) ? 1 : 0;
}
// Logical ops (both inputs and output are uint8 bool)
template <int OP>
__global__ void k_logic(const uint8_t *a, const uint8_t *b, uint8_t *o, int64_t n, Desc d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t ia, ib; coords(i, d, ia, ib);
    bool x = a[ia] != 0, y = b[ib] != 0;
    o[i] = (OP == OP_LAND ? (x && y) : (x || y)) ? 1 : 0;
}
__global__ void k_lnot(const uint8_t *a, uint8_t *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] ? 0 : 1;
}
template <typename T, bool NAN_>
__global__ void k_pred(const T *a, uint8_t *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float v = (float)a[i];
    o[i] = (NAN_ ? isnan(v) : isfinite(v)) ? 1 : 0;
}
// inf predicates: MODE 0 = any inf, 1 = +inf, 2 = -inf
template <typename T, int MODE>
__global__ void k_predinf(const T *a, uint8_t *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float v = (float)a[i];
    bool r = isinf(v) && (MODE == 0 || (MODE == 1 ? v > 0 : v < 0));
    o[i] = r ? 1 : 0;
}
// MaskedFill: out = mask ? value : self  (mask may broadcast to self)
template <typename T>
__global__ void k_maskfill(const T *self, const uint8_t *mask, T *o, T val, int64_t n, Desc d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t is, im; coords(i, d, is, im);   // a=self (contiguous), b=mask
    o[i] = mask[im] ? val : self[is];
}

bool bcast_shape(const std::vector<int64_t> &x, const std::vector<int64_t> &y, std::vector<int64_t> &o) {
    int rx = (int)x.size(), ry = (int)y.size(), r = rx > ry ? rx : ry;
    if (r > MAXD) return false;
    o.assign(r, 1);
    for (int i = 0; i < r; ++i) {
        int64_t dx = (i < r - rx) ? 1 : x[i - (r - rx)], dy = (i < r - ry) ? 1 : y[i - (r - ry)];
        if (dx != dy && dx != 1 && dy != 1) return false;
        o[i] = dx > dy ? dx : dy;
    }
    return true;
}

aclnnStatus make_cmp(int op, const aclTensor *a, const aclTensor *b, aclTensor *out, double val, aclOpExecutor **ex) {
    if (!a || !out || !ex || !a->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (!out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (b) {
        if (!b->data) return ACLNN_ERR_PARAM_NULLPTR;
        std::vector<int64_t> bd;
        if (!bcast_shape(a->viewDims, b->viewDims, bd) || bd != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    } else if (a->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = op; e->a = a; e->b = b; e->out = out; e->alpha = val;
    *ex = e; return ACLNN_SUCCESS;
}

#define CMP_DISPATCH(C)                                                                                            \
    switch (a->dtype) {                                                                                            \
        case ACL_FLOAT:   k_cmp<float,float,C><<<g,TH,0,s>>>((const float*)a->data,(const float*)b->data,po,n,d); break;        \
        case ACL_FLOAT16: k_cmp<__half,float,C><<<g,TH,0,s>>>((const __half*)a->data,(const __half*)b->data,po,n,d); break;     \
        case ACL_BF16:    k_cmp<__nv_bfloat16,float,C><<<g,TH,0,s>>>((const __nv_bfloat16*)a->data,(const __nv_bfloat16*)b->data,po,n,d); break; \
        case ACL_INT32:   k_cmp<int32_t,int64_t,C><<<g,TH,0,s>>>((const int32_t*)a->data,(const int32_t*)b->data,po,n,d); break;\
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }

// scalar comparison dispatch (b is null; e->alpha holds the scalar)
#define CMP_DISPATCH_S(C)                                                                                          \
    switch (a->dtype) {                                                                                            \
        case ACL_FLOAT:   k_cmp_s<float,float,C><<<g,TH,0,s>>>((const float*)a->data,po,(float)e->alpha,n); break;        \
        case ACL_FLOAT16: k_cmp_s<__half,float,C><<<g,TH,0,s>>>((const __half*)a->data,po,(float)e->alpha,n); break;     \
        case ACL_BF16:    k_cmp_s<__nv_bfloat16,float,C><<<g,TH,0,s>>>((const __nv_bfloat16*)a->data,po,(float)e->alpha,n); break; \
        case ACL_INT32:   k_cmp_s<int32_t,int64_t,C><<<g,TH,0,s>>>((const int32_t*)a->data,po,(int64_t)e->alpha,n); break;\
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }

aclnnStatus run_cmp(aclOpExecutor *e, cudaStream_t s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    const aclTensor *a = e->a, *b = e->b; aclTensor *o = e->out;
    int64_t n = o->numel(), g = nb(n);
    uint8_t *po = (uint8_t *)o->data;
    Desc d; fill_desc(a, b, o, d);
    if (o->dtype != ACL_BOOL && o->dtype != ACL_UINT8 && e->op != OP_MASKEDFILL) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    switch (e->op) {
        case OP_GT: if (b) { CMP_DISPATCH(CGt) } else { CMP_DISPATCH_S(CGt) } break;
        case OP_LT: if (b) { CMP_DISPATCH(CLt) } else { CMP_DISPATCH_S(CLt) } break;
        case OP_EQ: if (b) { CMP_DISPATCH(CEq) } else { CMP_DISPATCH_S(CEq) } break;
        case OP_NE: if (b) { CMP_DISPATCH(CNe) } else { CMP_DISPATCH_S(CNe) } break;
        case OP_GE: if (b) { CMP_DISPATCH(CGe) } else { CMP_DISPATCH_S(CGe) } break;
        case OP_LE: if (b) { CMP_DISPATCH(CLe) } else { CMP_DISPATCH_S(CLe) } break;
        case OP_LAND: k_logic<OP_LAND><<<g,TH,0,s>>>((const uint8_t*)a->data,(const uint8_t*)b->data,po,n,d); break;
        case OP_LOR:  k_logic<OP_LOR><<<g,TH,0,s>>>((const uint8_t*)a->data,(const uint8_t*)b->data,po,n,d); break;
        case OP_LNOT: k_lnot<<<g,TH,0,s>>>((const uint8_t*)a->data,po,n); break;
        case OP_ISNAN: case OP_ISFINITE: {
            bool isn = e->op == OP_ISNAN;
            switch (a->dtype) {
                case ACL_FLOAT:   isn ? k_pred<float,true><<<g,TH,0,s>>>((const float*)a->data,po,n)   : k_pred<float,false><<<g,TH,0,s>>>((const float*)a->data,po,n); break;
                case ACL_FLOAT16: isn ? k_pred<__half,true><<<g,TH,0,s>>>((const __half*)a->data,po,n) : k_pred<__half,false><<<g,TH,0,s>>>((const __half*)a->data,po,n); break;
                case ACL_BF16:    isn ? k_pred<__nv_bfloat16,true><<<g,TH,0,s>>>((const __nv_bfloat16*)a->data,po,n) : k_pred<__nv_bfloat16,false><<<g,TH,0,s>>>((const __nv_bfloat16*)a->data,po,n); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        }
        case OP_ISINF: case OP_ISPOSINF: case OP_ISNEGINF: {
            int mode = e->op == OP_ISINF ? 0 : (e->op == OP_ISPOSINF ? 1 : 2);
            #define PINF(T) do { if (mode==0) k_predinf<T,0><<<g,TH,0,s>>>((const T*)a->data,po,n); \
                else if (mode==1) k_predinf<T,1><<<g,TH,0,s>>>((const T*)a->data,po,n); \
                else k_predinf<T,2><<<g,TH,0,s>>>((const T*)a->data,po,n); } while(0)
            switch (a->dtype) {
                case ACL_FLOAT:   PINF(float); break;
                case ACL_FLOAT16: PINF(__half); break;
                case ACL_BF16:    PINF(__nv_bfloat16); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            #undef PINF
            break;
        }
        case OP_MASKEDFILL:
            switch (a->dtype) {
                case ACL_FLOAT:   k_maskfill<float><<<g,TH,0,s>>>((const float*)a->data,(const uint8_t*)b->data,(float*)o->data,(float)e->alpha,n,d); break;
                case ACL_FLOAT16: k_maskfill<__half><<<g,TH,0,s>>>((const __half*)a->data,(const uint8_t*)b->data,(__half*)o->data,(__half)(float)e->alpha,n,d); break;
                case ACL_INT32:   k_maskfill<int32_t><<<g,TH,0,s>>>((const int32_t*)a->data,(const uint8_t*)b->data,(int32_t*)o->data,(int32_t)e->alpha,n,d); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    cudaError_t err = cudaGetLastError(); delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

extern "C" {

#define CMP_OP(name, op)                                                                                           \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (self && other && self->dtype != other->dtype) return ACLNN_ERR_PARAM_INVALID;                              \
    if (ws) *ws = 0; return make_cmp(op, self, other, out, 0.0, ex);                                               \
}                                                                                                                  \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

CMP_OP(aclnnGtTensor, OP_GT)
CMP_OP(aclnnLtTensor, OP_LT)
CMP_OP(aclnnEqTensor, OP_EQ)
CMP_OP(aclnnNeTensor, OP_NE)
CMP_OP(aclnnGeTensor, OP_GE)
CMP_OP(aclnnLeTensor, OP_LE)
CMP_OP(aclnnLogicalAnd, OP_LAND)
CMP_OP(aclnnLogicalOr, OP_LOR)

// tensor ∘ scalar comparisons → bool out (self must be contiguous)
#define CMP_OP_S(name, op)                                                                                          \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID;                                              \
    if (ws) *ws = 0; return make_cmp(op, self, nullptr, out, other ? other->v : 0.0, ex);                          \
}                                                                                                                  \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }
CMP_OP_S(aclnnGtScalar, OP_GT)
CMP_OP_S(aclnnLtScalar, OP_LT)
CMP_OP_S(aclnnEqScalar, OP_EQ)
CMP_OP_S(aclnnNeScalar, OP_NE)
CMP_OP_S(aclnnGeScalar, OP_GE)
CMP_OP_S(aclnnLeScalar, OP_LE)

aclnnStatus aclnnLogicalNotGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_cmp(OP_LNOT, self, nullptr, out, 0.0, ex);
}
aclnnStatus aclnnLogicalNot(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

aclnnStatus aclnnIsNanGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_cmp(OP_ISNAN, self, nullptr, out, 0.0, ex);
}
aclnnStatus aclnnIsNan(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

aclnnStatus aclnnIsFiniteGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_cmp(OP_ISFINITE, self, nullptr, out, 0.0, ex);
}
aclnnStatus aclnnIsFinite(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

#define PRED_OP(name, op) \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (ws) *ws = 0; return make_cmp(op, self, nullptr, out, 0.0, ex); } \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }
PRED_OP(aclnnIsInf, OP_ISINF)
PRED_OP(aclnnIsPosInf, OP_ISPOSINF)
PRED_OP(aclnnIsNegInf, OP_ISNEGINF)

// in-place comparison/logical: result written back into selfRef (selfRef must be BOOL/UINT8)
#define CMP_IP_T(name, op) \
aclnnStatus name##GetWorkspaceSize(aclTensor *self, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    if (self && other && self->dtype != other->dtype) return ACLNN_ERR_PARAM_INVALID; \
    if (ws) *ws = 0; return make_cmp(op, self, other, self, 0.0, ex); } \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }
#define CMP_IP_S(name, op) \
aclnnStatus name##GetWorkspaceSize(aclTensor *self, const aclScalar *other, uint64_t *ws, aclOpExecutor **ex) { \
    if (!self || !self->contiguous()) return ACLNN_ERR_PARAM_INVALID; \
    if (ws) *ws = 0; return make_cmp(op, self, nullptr, self, other ? other->v : 0.0, ex); } \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }
CMP_IP_T(aclnnInplaceGtTensor, OP_GT) CMP_IP_T(aclnnInplaceLtTensor, OP_LT) CMP_IP_T(aclnnInplaceEqTensor, OP_EQ)
CMP_IP_T(aclnnInplaceNeTensor, OP_NE) CMP_IP_T(aclnnInplaceGeTensor, OP_GE) CMP_IP_T(aclnnInplaceLeTensor, OP_LE)
CMP_IP_S(aclnnInplaceGtScalar, OP_GT) CMP_IP_S(aclnnInplaceLtScalar, OP_LT) CMP_IP_S(aclnnInplaceEqScalar, OP_EQ)
CMP_IP_S(aclnnInplaceNeScalar, OP_NE) CMP_IP_S(aclnnInplaceGeScalar, OP_GE) CMP_IP_S(aclnnInplaceLeScalar, OP_LE)
CMP_IP_T(aclnnInplaceLogicalAnd, OP_LAND) CMP_IP_T(aclnnInplaceLogicalOr, OP_LOR)
aclnnStatus aclnnInplaceLogicalNotGetWorkspaceSize(aclTensor *self, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_cmp(OP_LNOT, self, nullptr, self, 0.0, ex);
}
aclnnStatus aclnnInplaceLogicalNot(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

// MaskedFill: self ∘ mask(bool) ∘ value(scalar) → out (same dtype as self)
aclnnStatus aclnnMaskedFillScalarGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclScalar *value,
                                                  aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!mask || (mask->dtype != ACL_BOOL && mask->dtype != ACL_UINT8)) return ACLNN_ERR_PARAM_INVALID;
    if (self && out && self->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
    if (ws) *ws = 0; return make_cmp(OP_MASKEDFILL, self, mask, out, value ? value->v : 0.0, ex);
}
aclnnStatus aclnnMaskedFillScalar(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_cmp(e, (cudaStream_t)s); }

} // extern "C"
