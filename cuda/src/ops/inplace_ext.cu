// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>

namespace _inplace_ext {
// In-place elementwise / activation family (aclnnInplace*): each writes its result back into selfRef.
// Implemented as thin wrappers that stash an elementwise plan (out = self) in GetWorkspaceSize and run
// it via the shared elementwise dispatch (ew_exec) in the Execute phase. fp32 / fp16 / bf16 (and integer
// for the bitwise variant); sub-byte fp4/fp6 in-place is rejected.

namespace {

constexpr int IT = 256;
inline int64_t ig(int64_t n) { return (n + IT - 1) / IT; }
template <typename T> __device__ inline T icvt(float v) { return (T)v; }
template <> __device__ inline __half icvt(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 icvt(float v) { return __float2bfloat16(v); }
template <typename T> __global__ void k_fill(T *o, float v, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; if (i < n) o[i] = icvt<T>(v);
}
// fill the main diagonal of a 2-D [rows,cols] tensor with v
template <typename T> __global__ void k_fill_diag(T *o, float v, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    int64_t d = rows < cols ? rows : cols;
    if (i < d) o[i * cols + i] = icvt<T>(v);
}

inline bool ip_is_subfp(aclDataType d) {
    return d == ACL_FLOAT4_E2M1 || d == ACL_FLOAT4_E1M2 || d == ACL_FLOAT6_E2M3 || d == ACL_FLOAT6_E3M2;
}

// Build an in-place executor: a = out = self, plus optional second/third operands and scalar params.
aclnnStatus ip_build(int op, aclTensor *self, const aclTensor *b, const aclTensor *c, double alpha,
                     const std::vector<double> &ds, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !self->data || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (ip_is_subfp(self->dtype)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = op; e->a = self; e->b = b; e->c = c; e->out = self; e->alpha = alpha; e->dscalars = ds;
    *ws = 0; *ex = e;
    return ACLNN_SUCCESS;
}

} // namespace

extern "C" {

#define IP_UN(NAME, OP)                                                                              \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, uint64_t *ws, aclOpExecutor **ex) {              \
    return ip_build(OP, self, nullptr, nullptr, 1.0, {}, ws, ex); }                                  \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return ew_exec(e, ws, (cudaStream_t)s); }

#define IP_SCALAR(NAME, OP)                                                                          \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclScalar *s, uint64_t *ws, aclOpExecutor **ex) { \
    if (!s) return ACLNN_ERR_PARAM_NULLPTR;                                                          \
    return ip_build(OP, self, nullptr, nullptr, s->v, {}, ws, ex); }                                 \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

#define IP_SCALAR2(NAME, OP)                                                                         \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclScalar *a, const aclScalar *b, uint64_t *ws, aclOpExecutor **ex) { \
    return ip_build(OP, self, nullptr, nullptr, 0.0, {a ? a->v : 0.0, b ? b->v : 0.0}, ws, ex); }    \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

#define IP_TERN(NAME, OP)                                                                            \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclTensor *t1, const aclTensor *t2, const aclScalar *v, uint64_t *ws, aclOpExecutor **ex) { \
    return ip_build(OP, self, t1, t2, v ? v->v : 1.0, {}, ws, ex); }                                 \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

#define IP_BIN(NAME, OP)                                                                             \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    return ip_build(OP, self, other, nullptr, 1.0, {}, ws, ex); }                                    \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

// binary with alpha (self ∘ alpha*other): Add/Sub
#define IP_BIN_ALPHA(NAME, OP)                                                                       \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclTensor *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) { \
    return ip_build(OP, self, other, nullptr, alpha ? alpha->v : 1.0, {}, ws, ex); }                 \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

// tensor-scalar with alpha (self ∘ alpha*scalar): Adds/Subs
#define IP_SCALAR_ALPHA(NAME, OP)                                                                    \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclScalar *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) { \
    if (!other) return ACLNN_ERR_PARAM_NULLPTR;                                                      \
    return ip_build(OP, self, nullptr, nullptr, other->v * (alpha ? alpha->v : 1.0), {}, ws, ex); }  \
aclnnStatus NAME(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return ew_exec(e, ws, (cudaStream_t)st); }

// ---- unary in-place ----
IP_UN(aclnnInplaceAcos, OP_ACOS)
IP_UN(aclnnInplaceCos, OP_COS)
IP_UN(aclnnInplaceErf, OP_ERF)
IP_UN(aclnnInplaceExp, OP_EXP)
IP_UN(aclnnInplaceFloor, OP_FLOOR)
IP_UN(aclnnInplaceLog, OP_LOG)
IP_UN(aclnnInplaceNeg, OP_NEG)
IP_UN(aclnnInplaceReciprocal, OP_RECIPROCAL)
IP_UN(aclnnInplaceRsqrt, OP_RSQRT)
IP_UN(aclnnInplaceSin, OP_SIN)
IP_UN(aclnnInplaceErfinv, OP_ERFINV)
IP_UN(aclnnInplaceHardsigmoid, OP_HARDSIGMOID)
IP_UN(aclnnInplaceHardswish, OP_HARDSWISH)
IP_UN(aclnnInplaceMish, OP_MISH)
IP_UN(aclnnInplaceRelu, OP_RELU)
IP_UN(aclnnInplaceSelu, OP_SELU)
IP_UN(aclnnInplaceSigmoid, OP_SIGMOID)

// ---- one-scalar in-place ----
IP_SCALAR(aclnnInplaceElu, OP_ELU)
IP_SCALAR(aclnnInplaceCelu, OP_CELU)
IP_SCALAR(aclnnInplaceLeakyRelu, OP_LEAKYRELU)
IP_SCALAR(aclnnInplaceClampMax, OP_CLAMP_MAX)

// ---- two-scalar in-place ----
IP_SCALAR2(aclnnInplaceHardtanh, OP_HARDTANH)
IP_SCALAR2(aclnnInplaceThreshold, OP_THRESHOLD)

// ---- ternary in-place (self + value * t1 ∘ t2) ----
IP_TERN(aclnnInplaceAddcmul, OP_ADDCMUL)
IP_TERN(aclnnInplaceAddcdiv, OP_ADDCDIV)

// ---- tensor-bound clamp in-place (elementwise max/min against a tensor) ----
IP_BIN(aclnnInplaceClampMinTensor, OP_MAXIMUM)
IP_BIN(aclnnInplaceClampMaxTensor, OP_MINIMUM)

// ---- bitwise in-place ----
IP_BIN(aclnnInplaceBitwiseAndTensor, OP_BAND)

// ---- lerp with scalar weight: self + weight * (end - self) ----
aclnnStatus aclnnInplaceLerpsGetWorkspaceSize(aclTensor *self, const aclTensor *end, const aclScalar *weight,
                                              uint64_t *ws, aclOpExecutor **ex) {
    return ip_build(OP_LERP, self, end, nullptr, weight ? weight->v : 0.0, {}, ws, ex);
}
aclnnStatus aclnnInplaceLerps(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return ew_exec(e, ws, (cudaStream_t)s); }

// ======================= ops-math / math in-place variants =======================
// ---- unary in-place (elementary math) ----
IP_UN(aclnnInplaceLog2, OP_LOG2)
IP_UN(aclnnInplaceLog10, OP_LOG10)
IP_UN(aclnnInplaceLog1p, OP_LOG1P)
IP_UN(aclnnInplaceExp2, OP_EXP2)
IP_UN(aclnnInplaceExpm1, OP_EXPM1)
IP_UN(aclnnInplaceErfc, OP_ERFC)
IP_UN(aclnnInplaceFrac, OP_FRAC)
IP_UN(aclnnInplaceSinh, OP_SINH)
IP_UN(aclnnInplaceCosh, OP_COSH)
IP_UN(aclnnInplaceTan, OP_TAN)
IP_UN(aclnnInplaceTanh, OP_TANH)
IP_UN(aclnnInplaceSqrt, OP_SQRT)
IP_UN(aclnnInplaceCeil, OP_CEIL)
IP_UN(aclnnInplaceRound, OP_ROUND)
IP_UN(aclnnInplaceTrunc, OP_TRUNC)
IP_UN(aclnnInplaceAsin, OP_ASIN)
IP_UN(aclnnInplaceAtan, OP_ATAN)
IP_UN(aclnnInplaceAcosh, OP_ACOSH)
IP_UN(aclnnInplaceAsinh, OP_ASINH)
IP_UN(aclnnInplaceAtanh, OP_ATANH)
IP_UN(aclnnInplaceSinc, OP_SINC)

// ---- binary in-place (tensor ∘ tensor) ----
IP_BIN(aclnnInplaceMul, OP_MUL)
IP_BIN(aclnnInplaceDiv, OP_DIV)
IP_BIN(aclnnInplacePowTensorTensor, OP_POW)
IP_BIN(aclnnInplaceFmodTensor, OP_FMOD)
IP_BIN(aclnnInplaceRemainderTensorTensor, OP_REMAINDER)
IP_BIN(aclnnInplaceAtan2, OP_ATAN2)
IP_BIN(aclnnInplaceXLogYTensor, OP_XLOGY)
IP_BIN(aclnnInplaceBitwiseOrTensor, OP_BOR)

// ---- binary in-place with alpha ----
IP_BIN_ALPHA(aclnnInplaceAdd, OP_ADD)
IP_BIN_ALPHA(aclnnInplaceSub, OP_SUB)

// ---- tensor-scalar in-place ----
IP_SCALAR(aclnnInplaceMuls, OP_MULS)
IP_SCALAR(aclnnInplaceDivs, OP_DIVS)
IP_SCALAR(aclnnInplacePowTensorScalar, OP_POWS)
IP_SCALAR_ALPHA(aclnnInplaceAdds, OP_ADDS)
IP_SCALAR_ALPHA(aclnnInplaceSubs, OP_SUBS)

// ---- more elementwise in-place (base ops added in later batches) ----
IP_BIN_ALPHA(aclnnInplaceAddV3, OP_ADD)
IP_BIN(aclnnInplaceBitwiseXorTensor, OP_BXOR)
IP_BIN(aclnnInplaceFloorDivide, OP_FLOORDIV)
IP_SCALAR(aclnnInplaceBitwiseAndScalar, OP_BANDS)
IP_SCALAR(aclnnInplaceBitwiseOrScalar, OP_BORS)
IP_SCALAR(aclnnInplaceBitwiseXorScalar, OP_BXORS)
IP_SCALAR(aclnnInplaceFloorDivides, OP_FLOORDIV)
// InplaceRoundDecimals: round to `decimals` places, in place
aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(aclTensor *self, int64_t decimals, uint64_t *ws, aclOpExecutor **ex) {
    return ip_build(OP_ROUNDDEC, self, nullptr, nullptr, (double)decimals, {}, ws, ex);
}
aclnnStatus aclnnInplaceRoundDecimals(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return ew_exec(e, ws, (cudaStream_t)s); }

// ---- zero (in place) ----
aclnnStatus aclnnInplaceZeroGetWorkspaceSize(aclTensor *self, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !self->data || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = 0; e->out = self; *ws = 0; *ex = e;
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceZero(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) {
    cudaError_t err = cudaMemsetAsync(e->out->data, 0, e->out->numel() * dtype_size(e->out->dtype), (cudaStream_t)s);
    delete e;
    return err == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- fill family ----
// e->op: 100 = fill all, 101 = fill diagonal; value in e->alpha
namespace { aclnnStatus fill_build(aclTensor *self, double v, int kind, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !self->data || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = kind; e->out = self; e->alpha = v; *ws = 0; *ex = e; return ACLNN_SUCCESS;
} }
static aclnnStatus fill_run(aclOpExecutor *e, cudaStream_t s) {
    aclTensor *o = e->out; int64_t n = o->numel(); aclDataType dt = o->dtype; float v = (float)e->alpha; aclnnStatus st = ACLNN_SUCCESS;
    if (e->op == 100) {
        #define KFA(T) k_fill<T><<<ig(n), IT, 0, s>>>((T*)o->data, v, n)
        switch (dt) { case ACL_FLOAT: KFA(float); break; case ACL_FLOAT16: KFA(__half); break; case ACL_BF16: KFA(__nv_bfloat16); break;
            case ACL_INT32: KFA(int32_t); break; default: st = ACLNN_ERR_PARAM_INVALID; }
        #undef KFA
    } else {
        int rk = (int)o->viewDims.size(); int64_t rows = rk >= 2 ? o->viewDims[rk-2] : n, cols = rk >= 1 ? o->viewDims[rk-1] : 1;
        int64_t d = rows < cols ? rows : cols;
        #define KFD(T) k_fill_diag<T><<<ig(d), IT, 0, s>>>((T*)o->data, v, rows, cols)
        switch (dt) { case ACL_FLOAT: KFD(float); break; case ACL_FLOAT16: KFD(__half); break; case ACL_BF16: KFD(__nv_bfloat16); break;
            case ACL_INT32: KFD(int32_t); break; default: st = ACLNN_ERR_PARAM_INVALID; }
        #undef KFD
    }
    delete e;
    if (st != ACLNN_SUCCESS) return st;
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
aclnnStatus aclnnInplaceFillScalarGetWorkspaceSize(aclTensor *self, const aclScalar *value, uint64_t *ws, aclOpExecutor **ex) {
    return fill_build(self, value ? value->v : 0.0, 100, ws, ex);
}
aclnnStatus aclnnInplaceFillScalar(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fill_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceOneGetWorkspaceSize(aclTensor *self, uint64_t *ws, aclOpExecutor **ex) {
    return fill_build(self, 1.0, 100, ws, ex);
}
aclnnStatus aclnnInplaceOne(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fill_run(e, (cudaStream_t)s); }
// FillTensor: value comes from a single-element tensor (read to host at plan time)
aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(aclTensor *self, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex) {
    if (!value || !value->data) return ACLNN_ERR_PARAM_NULLPTR;
    double v = 0.0;
    if (value->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, value->data, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return ACLNN_ERR_RUNTIME_ERROR; v = h; }
    else if (value->dtype == ACL_INT32) { int32_t h; if (cudaMemcpy(&h, value->data, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return ACLNN_ERR_RUNTIME_ERROR; v = h; }
    else return ACLNN_ERR_PARAM_INVALID;
    return fill_build(self, v, 100, ws, ex);
}
aclnnStatus aclnnInplaceFillTensor(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fill_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(aclTensor *self, const aclScalar *value, bool wrap, uint64_t *ws, aclOpExecutor **ex) {
    (void)wrap; return fill_build(self, value ? value->v : 0.0, 101, ws, ex);
}
aclnnStatus aclnnInplaceFillDiagonal(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return fill_run(e, (cudaStream_t)s); }

} // extern "C"
} // namespace _inplace_ext
#undef IP_UN
#undef IP_SCALAR
#undef IP_SCALAR2
#undef IP_TERN
#undef IP_BIN
#undef IP_BIN_ALPHA
#undef IP_SCALAR_ALPHA

namespace _inplace2_ext {
// In-place / *Out naming forwards to existing out-of-place bases (gemm-add, copy, lerp, bitwise),
// plus the MaskedFillTensor base (value supplied as a 0-d / scalar tensor).
//   In-place ops forward with out == selfRef; the bases used here write each output element from the
//   matching input element, so aliasing out and self is safe.

// aclnnCopy lives in format_ext.cu (extern "C", not in the public header).
extern "C" {
aclnnStatus aclnnCopyGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnCopy(void *, uint64_t, aclOpExecutor *, aclrtStream);
}

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n){ return (n+TH-1)/TH; }
inline aclnnStatus fin(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
// out[i] = mask[i] ? value[0] : self[i]   (value is a 0-d / scalar tensor)
template <typename T> __global__ void k_mfill_t(const T *self, const uint8_t *mask, const T *value, T *o, int64_t n){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; o[i]= mask[i]? value[0] : self[i];
}
#define DT3(KC) switch(e->out->dtype){case ACL_FLOAT:{using T=float;KC;}break;case ACL_FLOAT16:{using T=__half;KC;}break;default:{using T=__nv_bfloat16;KC;}break;}
} // namespace

extern "C" {

// ---- MaskedFillTensor (base) + in-place ----
aclnnStatus aclnnMaskedFillTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!mask||!value||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e=new aclOpExecutor(); e->a=self; e->b=mask; e->c=value; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMaskedFillTensor(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t n=e->out->numel(),g=nb(n); auto st=(cudaStream_t)s;
    DT3((k_mfill_t<T><<<g,TH,0,st>>>((const T*)e->a->data,(const uint8_t*)e->b->data,(const T*)e->c->data,(T*)e->out->data,n)));
    return fin(e);
}
aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex){
    return aclnnMaskedFillTensorGetWorkspaceSize(selfRef, mask, value, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceMaskedFillTensor(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnMaskedFillTensor(w,wz,e,s); }

// ---- BitwiseAndTensorOut naming variant + in-place ----
aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBitwiseAndTensorGetWorkspaceSize(self, other, out, ws, ex);
}
aclnnStatus aclnnBitwiseAndTensorOut(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBitwiseAndTensor(w,wz,e,s); }
aclnnStatus aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBitwiseAndTensorGetWorkspaceSize(selfRef, other, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceBitwiseAndTensorOut(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBitwiseAndTensor(w,wz,e,s); }

// ---- Lerp in-place (forwards to the scalar-weight Lerp base, out == selfRef) ----
aclnnStatus aclnnInplaceLerpGetWorkspaceSize(aclTensor *selfRef, const aclTensor *end, const aclScalar *weight, uint64_t *ws, aclOpExecutor **ex){
    return aclnnLerpGetWorkspaceSize(selfRef, end, weight, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceLerp(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnLerp(w,wz,e,s); }

// ---- Copy in-place: copy src into selfRef ----
aclnnStatus aclnnInplaceCopyGetWorkspaceSize(aclTensor *selfRef, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex){
    return aclnnCopyGetWorkspaceSize(src, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceCopy(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnCopy(w,wz,e,s); }

// ---- gemm-add in-place: selfRef = beta*selfRef + alpha*(mat1@mat2) ----
aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex){
    return aclnnAddbmmGetWorkspaceSize(selfRef, batch1, batch2, beta, alpha, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceAddbmm(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnAddbmm(w,wz,e,s); }
aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex){
    return aclnnBaddbmmGetWorkspaceSize(selfRef, batch1, batch2, beta, alpha, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceBaddbmm(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnBaddbmm(w,wz,e,s); }
aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mat1, const aclTensor *mat2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex){
    return aclnnAddmmGetWorkspaceSize(selfRef, mat1, mat2, beta, alpha, selfRef, ws, ex);
}
aclnnStatus aclnnInplaceAddmm(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnAddmm(w,wz,e,s); }

} // extern "C"
} // namespace _inplace2_ext
#undef DT3

