// Elementwise op family: binary (Add/Sub/Mul/Div/Maximum/Minimum), tensor-scalar (Adds..ClampMax),
// unary (Exp..Neg), integer bitwise (BitwiseAnd/Or/Not), SWhere.
// Computation is performed in accumulation type A (fp family = float, int32 = int64); result written back as T.
#include "../internal.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "subfp.cuh"

aclnnStatus check_same(const aclTensor *a, const aclTensor *t) {
    if (!a || !t || !a->data || !t->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (a->viewDims != t->viewDims) return ACLNN_ERR_PARAM_INVALID;      // PoC: no broadcast
    if (!a->contiguous() || !t->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (a->format != ACL_FORMAT_ND || t->format != ACL_FORMAT_ND) return ACLNN_ERR_PARAM_INVALID;
    return ACLNN_SUCCESS;
}

namespace {

template <typename A> struct FAdd { __device__ static A f(A a, A b, A al) { return a + al * b; } };
template <typename A> struct FSub { __device__ static A f(A a, A b, A al) { return a - al * b; } };
template <typename A> struct FMul { __device__ static A f(A a, A b, A)    { return a * b; } };
template <typename A> struct FDiv { __device__ static A f(A a, A b, A)    { return a / b; } };
template <typename A> struct FMax { __device__ static A f(A a, A b, A)    { return a > b ? a : b; } };
template <typename A> struct FMin { __device__ static A f(A a, A b, A)    { return a < b ? a : b; } };

template <typename A> struct FExp   { __device__ static A f(A a) { return exp(a); } };
template <> struct FExp<float>      { __device__ static float f(float a) { return expf(a); } };
template <typename A> struct FLog   { __device__ static A f(A a) { return logf(a); } };
template <typename A> struct FAbs   { __device__ static A f(A a) { return a < (A)0 ? (A)(-a) : a; } };
template <typename A> struct FSqrt  { __device__ static A f(A a) { return sqrtf(a); } };
template <typename A> struct FRsqrt { __device__ static A f(A a) { return rsqrtf(a); } };
template <typename A> struct FRecip { __device__ static A f(A a) { return (A)1 / a; } };
template <typename A> struct FRelu  { __device__ static A f(A a) { return a > (A)0 ? a : (A)0; } };
template <typename A> struct FNeg   { __device__ static A f(A a) { return -a; } };
// Activation/elementary functions (computed in accumulation type A=float; fp16/bf16 use float accumulation)
template <typename A> struct FSigmoid  { __device__ static A f(A a) { return (A)1 / ((A)1 + expf(-a)); } };
template <typename A> struct FTanh     { __device__ static A f(A a) { return tanhf(a); } };
template <typename A> struct FErf      { __device__ static A f(A a) { return erff(a); } };
// Use erfc instead of (1+erf): avoids catastrophic cancellation in the negative tail where 1+erf(-y)→0 (since 1+erf(y)=erfc(-y))
template <typename A> struct FGelu     { __device__ static A f(A a) { return (A)0.5 * a * erfcf(-a * (A)0.70710678118654752f); } };
template <typename A> struct FSilu     { __device__ static A f(A a) { return a / ((A)1 + expf(-a)); } };
template <typename A> struct FSoftplus { __device__ static A f(A a) { return a > (A)0 ? a + log1pf(expf(-a)) : log1pf(expf(a)); } };
template <typename A> struct FSin      { __device__ static A f(A a) { return sinf(a); } };
template <typename A> struct FCos      { __device__ static A f(A a) { return cosf(a); } };
template <typename A> struct FTan      { __device__ static A f(A a) { return tanf(a); } };
template <typename A> struct FAtan     { __device__ static A f(A a) { return atanf(a); } };
template <typename A> struct FSign     { __device__ static A f(A a) { return (A)((a > (A)0) - (a < (A)0)); } };
template <typename A> struct FFloor    { __device__ static A f(A a) { return floorf(a); } };
template <typename A> struct FCeil     { __device__ static A f(A a) { return ceilf(a); } };
template <typename A> struct FRound    { __device__ static A f(A a) { return rintf(a); } };   // banker's rounding (CANN/PyTorch round)
template <typename A> struct FTrunc    { __device__ static A f(A a) { return truncf(a); } };
template <typename A> struct FSquare   { __device__ static A f(A a) { return a * a; } };
template <typename A> struct FSinh     { __device__ static A f(A a) { return sinhf(a); } };
template <typename A> struct FCosh     { __device__ static A f(A a) { return coshf(a); } };
template <typename A> struct FAsin     { __device__ static A f(A a) { return asinf(a); } };
template <typename A> struct FAcos     { __device__ static A f(A a) { return acosf(a); } };
template <typename A> struct FErfc     { __device__ static A f(A a) { return erfcf(a); } };
template <typename A> struct FFrac     { __device__ static A f(A a) { return a - truncf(a); } };
template <typename A> struct FLgamma   { __device__ static A f(A a) { return lgammaf(a); } };
template <typename A> struct FPow      { __device__ static A f(A a, A b, A) { return powf(a, b); } };   // power (shared by tensor∘scalar and tensor∘tensor)
template <typename A> struct FFmod     { __device__ static A f(A a, A b, A) { return fmodf(a, b); } };
template <typename A> struct FHypot    { __device__ static A f(A a, A b, A) { return hypotf(a, b); } };

// ---- P1 elementwise math (unary) ----
template <typename A> struct FExpm1  { __device__ static A f(A a) { return expm1f(a); } };
template <typename A> struct FLog1p  { __device__ static A f(A a) { return log1pf(a); } };
template <typename A> struct FLog2   { __device__ static A f(A a) { return log2f(a); } };
template <typename A> struct FLog10  { __device__ static A f(A a) { return log10f(a); } };
template <typename A> struct FExp2   { __device__ static A f(A a) { return exp2f(a); } };
template <typename A> struct FErfinv { __device__ static A f(A a) { return erfinvf(a); } };
// ---- P1 elementwise math (binary, broadcast) ----
template <typename A> struct FAtan2     { __device__ static A f(A a, A b, A) { return atan2f(a, b); } };
// PyTorch/Python remainder: result takes the sign of the divisor (vs fmod which truncates toward zero)
template <typename A> struct FRemainder { __device__ static A f(A a, A b, A) { A r = fmodf(a, b); if (r != (A)0 && ((r < (A)0) != (b < (A)0))) r += b; return r; } };
template <typename A> struct FXlogy     { __device__ static A f(A a, A b, A) { return a == (A)0 ? (A)0 : a * logf(b); } };
// log(exp(a)+exp(b)) numerically stable: max + log1p(exp(min-max))
template <typename A> struct FLogaddexp { __device__ static A f(A a, A b, A) { A m = a > b ? a : b, d = a > b ? b - a : a - b; return m + log1pf(expf(d)); } };
template <typename A> struct FCopysign  { __device__ static A f(A a, A b, A) { return copysignf(a, b); } };
template <typename A> struct FHeaviside { __device__ static A f(A a, A b, A) { return a < (A)0 ? (A)0 : (a > (A)0 ? (A)1 : b); } };
template <typename A> struct FLerp      { __device__ static A f(A a, A b, A w) { return a + w * (b - a); } };   // weight in alpha
// ---- additional elementary math (unary) ----
template <typename A> struct FAcosh  { __device__ static A f(A a) { return acoshf(a); } };
template <typename A> struct FAsinh  { __device__ static A f(A a) { return asinhf(a); } };
template <typename A> struct FAtanh  { __device__ static A f(A a) { return atanhf(a); } };
// normalized sinc: sin(pi x)/(pi x), 1 at 0 (PyTorch/CANN convention)
template <typename A> struct FSinc   { __device__ static A f(A a) { if (a == (A)0) return (A)1; A px = (A)3.14159265358979323846f * a; return sinf(px) / px; } };
// digamma ψ(x): recurrence up to x>=6 then asymptotic series (valid for x>0)
template <typename A> struct FDigamma { __device__ static A f(A a) { float x = (float)a, r = 0.f;
    while (x < 6.f) { r -= 1.f / x; x += 1.f; }
    float inv = 1.f / x, inv2 = inv * inv;
    r += logf(x) - 0.5f * inv - inv2 * (1.f/12.f - inv2 * (1.f/120.f - inv2 * (1.f/252.f)));
    return (A)r; } };
// ---- additional elementary math (binary) ----
template <typename A> struct FLogaddexp2 { __device__ static A f(A a, A b, A) { A m = a > b ? a : b, d = a > b ? b - a : a - b; return m + log2f((A)1 + exp2f(d)); } };
// floor division: floor(a/b); reverse subtraction: b - alpha*a (scalar form s - a with alpha=1)
template <typename A> struct FFloorDiv { __device__ static A f(A a, A b, A) { return floorf(a / b); } };
template <typename A> struct FRsub     { __device__ static A f(A a, A b, A al) { return b - al * a; } };
// round to a given number of decimals (decimals count passed as the scalar argument)
template <typename A> struct FRoundDec { __device__ static A f(A a, A d, A) { A m = powf((A)10, d); return rintf(a * m) / m; } };
// nan_to_num: NaN→s1, +Inf→s2, -Inf→s3
template <typename A> struct FNanToNum { __device__ static A f(A a, A s1, A s2, A s3) { float v = (float)a; return isnan(v) ? s1 : (isinf(v) ? (v > 0 ? s2 : s3) : a); } };

// ---- P2 activation functions (unary, no parameter) ----
template <typename A> struct FMish        { __device__ static A f(A a) { A sp = a > (A)0 ? a + log1pf(expf(-a)) : log1pf(expf(a)); return a * tanhf(sp); } };
template <typename A> struct FHardswish   { __device__ static A f(A a) { A r = a + (A)3; r = r < (A)0 ? (A)0 : (r > (A)6 ? (A)6 : r); return a * r / (A)6; } };
template <typename A> struct FHardsigmoid { __device__ static A f(A a) { A r = a + (A)3; r = r < (A)0 ? (A)0 : (r > (A)6 ? (A)6 : r); return r / (A)6; } };
template <typename A> struct FLogSigmoid  { __device__ static A f(A a) { A z = -a, sp = z > (A)0 ? z + log1pf(expf(-z)) : log1pf(expf(z)); return -sp; } };  // -softplus(-x)
template <typename A> struct FSelu        { __device__ static A f(A a) { const A al = (A)1.6732632423543772, sc = (A)1.0507009873554805; return sc * (a > (A)0 ? a : al * expm1f(a)); } };
template <typename A> struct FTanhshrink  { __device__ static A f(A a) { return a - tanhf(a); } };
template <typename A> struct FRelu6       { __device__ static A f(A a) { return a < (A)0 ? (A)0 : (a > (A)6 ? (A)6 : a); } };
// tanh-approximation GeLU (CANN FastGelu / PyTorch gelu approximate='tanh')
template <typename A> struct FFastGelu     { __device__ static A f(A a) { A c = (A)0.7978845608028654f * (a + (A)0.044715f * a * a * a); return (A)0.5f * a * ((A)1 + tanhf(c)); } };
template <typename A> struct FSquaredRelu  { __device__ static A f(A a) { A r = a > (A)0 ? a : (A)0; return r * r; } };
// Swish(x) = x * sigmoid(beta*x), beta passed as the scalar argument
template <typename A> struct FSwish        { __device__ static A f(A a, A beta, A) { return a / ((A)1 + expf(-beta * a)); } };
// ---- P2 activation functions (unary, one scalar parameter passed as the binary functor's second argument) ----
template <typename A> struct FLeakyRelu { __device__ static A f(A a, A s, A) { return a > (A)0 ? a : s * a; } };
template <typename A> struct FElu       { __device__ static A f(A a, A s, A) { return a > (A)0 ? a : s * expm1f(a); } };
template <typename A> struct FCelu      { __device__ static A f(A a, A s, A) { return a > (A)0 ? a : s * expm1f(a / s); } };
template <typename A> struct FHardshrink{ __device__ static A f(A a, A l, A) { return (a > l || a < -l) ? a : (A)0; } };
template <typename A> struct FSoftshrink{ __device__ static A f(A a, A l, A) { return a > l ? a - l : (a < -l ? a + l : (A)0); } };
template <typename A> struct FPrelu     { __device__ static A f(A a, A w, A) { return a > (A)0 ? a : w * a; } };   // weight tensor (broadcast)
// ---- P2 activation functions (unary, two scalar parameters) ----
template <typename A> struct FHardtanh  { __device__ static A f(A a, A lo, A hi) { return a < lo ? lo : (a > hi ? hi : a); } };
template <typename A> struct FThreshold { __device__ static A f(A a, A th, A val) { return a > th ? a : val; } };
// ---- P1 ternary (self, t1/min, t2/max) ----
template <typename A> struct TAddcmul { __device__ static A f(A a, A b, A c, A s) { return a + s * (b * c); } };
template <typename A> struct TAddcdiv { __device__ static A f(A a, A b, A c, A s) { return a + s * (b / c); } };
template <typename A> struct TClamp   { __device__ static A f(A a, A lo, A hi, A) { return a < lo ? lo : (a > hi ? hi : a); } };

template <typename T, typename A, template <typename> class F>
__global__ void k_bin(const T *a, const T *b, T *o, A al, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i], (A)b[i], al);
}
// General binary descriptor: keyed on out shape (out assumed contiguous, freshly allocated).
// Inputs use real stride+offset (supports non-contiguous views); broadcast dims have stride=0.
// Mixed input dtypes are supported (TA/TB/TO may differ at compile time).
constexpr int MAXD = 8;
struct GBc { int rank; int64_t od[MAXD], as[MAXD], bs[MAXD], aoff, boff; };

template <typename TA, typename TB, typename TO, typename A, template <typename> class F>
__global__ void k_bin_g(const TA *a, const TB *b, TO *o, A al, int64_t n, GBc d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, ia = d.aoff, ib = d.boff;
    for (int k = d.rank - 1; k >= 0; --k) {
        int64_t c = rem % d.od[k]; rem /= d.od[k];
        ia += c * d.as[k]; ib += c * d.bs[k];
    }
    o[i] = (TO)F<A>::f((A)a[ia], (A)b[ib], al);
}

template <typename T, typename A, template <typename> class F>
__global__ void k_scalar(const T *a, T *o, A s, A al, int64_t n) {  // tensor ∘ scalar: reuses binary functor
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i], s, al);
}
template <typename T, typename A, template <typename> class F>
__global__ void k_scalar2(const T *a, T *o, A s1, A s2, int64_t n) {  // unary with two scalar params
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i], s1, s2);
}
template <typename T, typename A, template <typename> class F>
__global__ void k_scalar3(const T *a, T *o, A s1, A s2, A s3, int64_t n) {  // unary with three scalar params
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i], s1, s2, s3);
}
template <typename T, typename A, template <typename> class F>
__global__ void k_tern(const T *a, const T *b, const T *c, T *o, A s, int64_t n) {  // three same-shape inputs + scalar
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i], (A)b[i], (A)c[i], s);
}
template <typename T, typename A, template <typename> class F>
__global__ void k_un(const T *a, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = (T)F<A>::f((A)a[i]);
}
// General unary: input addressed by real stride+offset, supports non-contiguous views (transposed/sliced/Expand).
// Output is contiguous (linear i). out and in have the same shape, so no broadcasting; dimension order matches out.
struct GUn { int rank; int64_t od[8], as[8], aoff; };
template <typename T, typename A, template <typename> class F>
__global__ void k_un_g(const T *a, T *o, GUn d, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int64_t rem = i, ia = d.aoff;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t c = rem % d.od[k]; rem /= d.od[k]; ia += c * d.as[k]; }
    o[i] = (T)F<A>::f((A)a[ia]);
}
template <typename T> __global__ void k_band(const T *a, const T *b, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] & b[i];
}
template <typename T> __global__ void k_bor(const T *a, const T *b, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] | b[i];
}
template <typename T> __global__ void k_bxor(const T *a, const T *b, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = a[i] ^ b[i];
}
template <typename T> __global__ void k_bnot(const T *a, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = ~a[i];
}
// tensor ∘ scalar bitwise (self contiguous; OP 0=and 1=or 2=xor)
template <typename T, int OP> __global__ void k_bit_s(const T *a, T s, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    o[i] = OP == 0 ? (a[i] & s) : (OP == 1 ? (a[i] | s) : (a[i] ^ s));
}
// General three-input broadcast: cond/a/b each addressed via their own broadcast strides relative to out shape
struct GW3 { int rank; int64_t od[8], cs[8], as[8], bs[8], coff, aoff, boff; };
template <typename T> __global__ void k_where_g(const uint8_t *c, const T *a, const T *b, T *o, int64_t n, GW3 d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t rem = i, ic = d.coff, ia = d.aoff, ib = d.boff;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t cc = rem % d.od[k]; rem /= d.od[k];
        ic += cc * d.cs[k]; ia += cc * d.as[k]; ib += cc * d.bs[k]; }
    o[i] = c[ic] ? a[ia] : b[ib];
}
// General two-input integer broadcast (bitwise ops)
template <typename T, int OP> __global__ void k_bit_g(const T *a, const T *b, T *o, int64_t n, GBc d) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    int64_t rem = i, ia = d.aoff, ib = d.boff;
    for (int k = d.rank - 1; k >= 0; --k) { int64_t c = rem % d.od[k]; rem /= d.od[k]; ia += c * d.as[k]; ib += c * d.bs[k]; }
    o[i] = OP == 0 ? (a[ia] & b[ib]) : (OP == 1 ? (a[ia] | b[ib]) : (a[ia] ^ b[ib]));
}
template <typename T> __global__ void k_where(const uint8_t *c, const T *a, const T *b, T *o, int64_t n) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = c[i] ? a[i] : b[i];
}

constexpr int THREADS = 256;
inline int64_t nblocks(int64_t n) { return (n + THREADS - 1) / THREADS; }

inline bool is_bcast_op(int op) {
    return op == OP_ADD || op == OP_SUB || op == OP_MUL || op == OP_DIV || op == OP_MAXIMUM || op == OP_MINIMUM ||
           op == OP_FMOD || op == OP_HYPOT || op == OP_POW ||
           op == OP_ATAN2 || op == OP_REMAINDER || op == OP_XLOGY || op == OP_LOGADDEXP ||
           op == OP_COPYSIGN || op == OP_HEAVISIDE || op == OP_LERP || op == OP_PRELU ||
           op == OP_LOGADDEXP2 || op == OP_FLOORDIV || op == OP_RSUB;   // binary elementary ops: all share the broadcast/mixed-dtype general path
}
// Unary ops covered by DISPATCH_UN (OP_EXP..OP_LGAMMA, excluding scalar OP_POWS); these support non-contiguous input
inline bool is_unary_op(int op) { return op >= OP_EXP && op <= OP_LGAMMA && op != OP_POWS; }
// NumPy/CANN-style broadcast: tail-align dims, dim==1 is stretchable. Returns false if incompatible.
inline bool bcast_dims(const std::vector<int64_t> &x, const std::vector<int64_t> &y, std::vector<int64_t> &o) {
    int rx = (int)x.size(), ry = (int)y.size(), r = rx > ry ? rx : ry;
    if (r > MAXD) return false;
    o.assign(r, 1);
    for (int i = 0; i < r; ++i) {
        int64_t dx = (i < r - rx) ? 1 : x[i - (r - rx)];
        int64_t dy = (i < r - ry) ? 1 : y[i - (r - ry)];
        if (dx != dy && dx != 1 && dy != 1) return false;
        o[i] = dx > dy ? dx : dy;
    }
    return true;
}
// Fill real stride+offset for each input relative to out shape (supports non-contiguous views; broadcast dims get stride 0)
inline void fill_g(const aclTensor *a, const aclTensor *b, const aclTensor *out, GBc &d) {
    int r = (int)out->viewDims.size(), ra = (int)a->viewDims.size(), rb = (int)b->viewDims.size();
    d.rank = r; d.aoff = a->offset; d.boff = b->offset;
    for (int i = 0; i < r; ++i) {
        d.od[i] = out->viewDims[i];
        int ia = i - (r - ra), ib = i - (r - rb);
        d.as[i] = (ia >= 0 && a->viewDims[ia] == out->viewDims[i]) ? a->strides[ia] : 0;
        d.bs[i] = (ib >= 0 && b->viewDims[ib] == out->viewDims[i]) ? b->strides[ib] : 0;
    }
}

// Per-tensor broadcast strides relative to out shape (tail-aligned; mismatched dims get stride 0)
inline void bstrides(const aclTensor *t, const std::vector<int64_t> &od, int64_t *str, int64_t &off) {
    int r = (int)od.size(), rt = (int)t->viewDims.size(); off = t->offset;
    for (int i = 0; i < r; ++i) { int it = i - (r - rt);
        str[i] = (it >= 0 && t->viewDims[it] == od[i]) ? t->strides[it] : 0; }
}
// Three-shape broadcast: first a∘b, then result ∘c
inline bool bcast3(const std::vector<int64_t> &c, const std::vector<int64_t> &a, const std::vector<int64_t> &b, std::vector<int64_t> &o) {
    std::vector<int64_t> ab; if (!bcast_dims(a, b, ab)) return false; return bcast_dims(ab, c, o);
}
inline void fill_g3(const aclTensor *cond, const aclTensor *a, const aclTensor *b, const aclTensor *out, GW3 &d) {
    int r = (int)out->viewDims.size(); d.rank = r;
    for (int i = 0; i < r; ++i) d.od[i] = out->viewDims[i];
    bstrides(cond, out->viewDims, d.cs, d.coff);
    bstrides(a, out->viewDims, d.as, d.aoff);
    bstrides(b, out->viewDims, d.bs, d.boff);
}

inline bool ew_is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool ew_is_fp6(aclDataType t) { return t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline bool ew_is_subfp(aclDataType t) { return ew_is_fp4(t) || ew_is_fp6(t); }
inline int ew_subkind(aclDataType t) {
    return t == ACL_FLOAT4_E2M1 ? SF_FP4E2M1 : t == ACL_FLOAT4_E1M2 ? SF_FP4E1M2 : t == ACL_FLOAT6_E2M3 ? SF_FP6E2M3 : SF_FP6E3M2;
}

aclnnStatus make_exec(int op, const aclTensor *a, const aclTensor *b, const aclTensor *c, aclTensor *out,
                      double alpha, uint64_t *ws, aclOpExecutor **exec) {
    if (!ws || !exec) return ACLNN_ERR_PARAM_NULLPTR;
    // Arithmetic binary via general path: allows broadcast, non-contiguous inputs, mixed dtype
    // (fp4/fp6 still require identical shapes; the unpack path operates on a's numel)
    bool bcast = b && is_bcast_op(op) && !ew_is_subfp(a->dtype);
    if (bcast) {
        if (!a->data || !b->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
        if (!out->contiguous()) return ACLNN_ERR_PARAM_INVALID;   // output must be contiguous (typically a freshly allocated tensor)
        if (a->format != ACL_FORMAT_ND || b->format != ACL_FORMAT_ND || out->format != ACL_FORMAT_ND) return ACLNN_ERR_PARAM_INVALID;
        std::vector<int64_t> bd;
        if (!bcast_dims(a->viewDims, b->viewDims, bd)) return ACLNN_ERR_PARAM_INVALID;
        if (bd != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
        // Mixed dtype: PoC promotes to fp32; out must be fp32. Same dtype: out must match that dtype
        if (a->dtype != b->dtype) { if (out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; }
        else if (out->dtype != a->dtype) return ACLNN_ERR_PARAM_INVALID;
    } else if (is_unary_op(op) && !b && !c) {
        // Unary ops accept non-contiguous inputs (transposed/sliced/Expand views fed directly, avoiding materialization); output must be contiguous
        if (!a || !out || !a->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
        if (a->viewDims != out->viewDims || a->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
        if (a->format != ACL_FORMAT_ND || out->format != ACL_FORMAT_ND) return ACLNN_ERR_PARAM_INVALID;
        if (!out->contiguous() || (int)a->viewDims.size() > MAXD) return ACLNN_ERR_PARAM_INVALID;
    } else if (b && is_bcast_op(op) && ew_is_subfp(a->dtype)) {
        // fp4/fp6 binary broadcast: unpack → fp32 broadcast → encode back; a/b/out share the same subfp dtype
        if (!a->data || !b->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
        if (a->dtype != b->dtype || a->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
        if (a->format != ACL_FORMAT_ND || b->format != ACL_FORMAT_ND || out->format != ACL_FORMAT_ND) return ACLNN_ERR_PARAM_INVALID;
        std::vector<int64_t> bd;
        if (!bcast_dims(a->viewDims, b->viewDims, bd) || bd != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    } else if (op == OP_SWHERE) {
        // SWhere broadcast: cond/self/other each broadcast to out shape; self/other dtype must equal out dtype
        if (!a || !b || !c || !out || !a->data || !b->data || !c->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
        if (!out->contiguous() || b->dtype != out->dtype || c->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
        std::vector<int64_t> bd;
        if (!bcast3(a->viewDims, b->viewDims, c->viewDims, bd) || bd != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    } else if ((op == OP_BAND || op == OP_BOR || op == OP_BXOR) && b) {
        // Bitwise broadcast: a/b integer broadcast to out; same dtype required
        if (!a->data || !b->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
        if (!out->contiguous() || a->dtype != b->dtype || a->dtype != out->dtype) return ACLNN_ERR_PARAM_INVALID;
        std::vector<int64_t> bd;
        if (!bcast_dims(a->viewDims, b->viewDims, bd) || bd != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    } else {
        aclnnStatus st = check_same(a, out);
        if (st == ACLNN_SUCCESS && b) st = check_same(a, b);
        if (st == ACLNN_SUCCESS && c) st = check_same(a, c);
        if (st != ACLNN_SUCCESS) return st;
    }
    auto *e = new aclOpExecutor();
    e->op = op; e->a = a; e->b = b; e->c = c; e->out = const_cast<aclTensor *>(out); e->alpha = alpha;
    // fp4/fp6 takes the unpack path, requiring fp32 scratch buffers (a and b sized by their own numel, out by out numel — broadcast-compatible)
    if (ew_is_subfp(a->dtype)) {
        uint64_t na = a->numel(), nb = b ? b->numel() : 0, no = out->numel();
        *ws = (na + nb + no) * sizeof(float);
    } else *ws = 0;
    *exec = e;
    return ACLNN_SUCCESS;
}

// fp4 (2 values/byte) / fp6 (1 value/byte) ↔ fp32
__global__ void k_subfp_dec(const uint8_t *p, float *o, int64_t n, int k, bool fp4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint8_t code = fp4 ? ((i & 1) ? (p[i / 2] >> 4) : (p[i / 2] & 0xf)) : (p[i] & 0x3f);
    o[i] = subfp_decode(k, code);
}
__global__ void k_subfp_enc(const float *in, uint8_t *o, int64_t n, int k, bool fp4) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (fp4) {
        int64_t nb = (n + 1) / 2;
        if (idx >= nb) return;
        uint8_t lo = subfp_encode(k, in[2 * idx]) & 0xf;
        uint8_t hi = (2 * idx + 1 < n) ? (subfp_encode(k, in[2 * idx + 1]) & 0xf) : 0;
        o[idx] = (uint8_t)(lo | (hi << 4));
    } else {
        if (idx < n) o[idx] = subfp_encode(k, in[idx]);
    }
}

// Vectorized memory access fast path: each thread processes 16B (VEC=16/sizeof(T)) elements,
// compiling to LDG.128 — reduces load instruction count and improves memory-level parallelism.
// On low-bandwidth UMA (GB10) the benefit is ~0 (already saturated), but significant on high-bandwidth
// discrete GPUs (GDDR7/HBM). Pointers are 256B-aligned by cudaMalloc; the vector path is taken only
// when n%VEC==0, otherwise falls back to scalar k_bin/k_un.
template <typename T, int VEC> struct __align__(16) VecT { T d[VEC]; };
template <typename T, typename A, template <typename> class F>
__global__ void k_bin_vec(const T *a, const T *b, T *o, A al, int64_t nvec) {
    constexpr int VEC = 16 / sizeof(T);
    int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (v >= nvec) return;
    VecT<T, VEC> va = ((const VecT<T, VEC> *)a)[v], vb = ((const VecT<T, VEC> *)b)[v], vo;
    #pragma unroll
    for (int k = 0; k < VEC; k++) vo.d[k] = (T)F<A>::f((A)va.d[k], (A)vb.d[k], al);
    ((VecT<T, VEC> *)o)[v] = vo;
}
template <typename T, typename A, template <typename> class F>
__global__ void k_un_vec(const T *a, T *o, int64_t nvec) {
    constexpr int VEC = 16 / sizeof(T);
    int64_t v = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (v >= nvec) return;
    VecT<T, VEC> va = ((const VecT<T, VEC> *)a)[v], vo;
    #pragma unroll
    for (int k = 0; k < VEC; k++) vo.d[k] = (T)F<A>::f((A)va.d[k]);
    ((VecT<T, VEC> *)o)[v] = vo;
}
// ELTWISE_NO_VEC=1 forces the scalar (1 element/thread) path even when the 16B vector path is eligible —
// for same-card A/B measurement of the 128-bit vectorized-access optimization.
static inline bool eltwise_vec_enabled() {
    static int v = -1;
    if (v < 0) { const char *e = getenv("ELTWISE_NO_VEC"); v = (e && e[0] && e[0] != '0') ? 0 : 1; }
    return v != 0;
}
template <typename T, typename A, template <typename> class F>
static inline void bin_fast(const T *a, const T *b, T *o, A al, int64_t n, cudaStream_t s) {
    constexpr int VEC = 16 / sizeof(T);
    if (VEC > 1 && (n % VEC) == 0 && eltwise_vec_enabled()) { int64_t nv = n / VEC, g = (nv + THREADS - 1) / THREADS;
        k_bin_vec<T, A, F><<<g, THREADS, 0, s>>>(a, b, o, al, nv); }
    else { int64_t g = (n + THREADS - 1) / THREADS; k_bin<T, A, F><<<g, THREADS, 0, s>>>(a, b, o, al, n); }
}
template <typename T, typename A, template <typename> class F>
static inline void un_fast(const T *a, T *o, int64_t n, cudaStream_t s) {
    constexpr int VEC = 16 / sizeof(T);
    if (VEC > 1 && (n % VEC) == 0 && eltwise_vec_enabled()) { int64_t nv = n / VEC, g = (nv + THREADS - 1) / THREADS;
        k_un_vec<T, A, F><<<g, THREADS, 0, s>>>(a, o, nv); }
    else { int64_t g = (n + THREADS - 1) / THREADS; k_un<T, A, F><<<g, THREADS, 0, s>>>(a, o, n); }
}
// Unary op dispatch: contiguous input takes the fast path (vector/scalar direct read); otherwise uses stride-based general addressing
template <typename T, typename A, template <typename> class F>
static inline void un_go(const T *a, T *o, bool fast, const GUn &d, int64_t n, cudaStream_t s) {
    if (fast) un_fast<T, A, F>(a, o, n, s);
    else { int64_t g = (n + THREADS - 1) / THREADS; k_un_g<T, A, F><<<g, THREADS, 0, s>>>(a, o, d, n); }
}
inline void fill_un(const aclTensor *a, GUn &d) {
    d.rank = (int)a->viewDims.size(); d.aoff = a->offset;
    for (int i = 0; i < d.rank; ++i) { d.od[i] = a->viewDims[i]; d.as[i] = a->strides[i]; }
}

// dtype dispatch for float + int32 (bitwise family handled separately)
#define DISPATCH_BIN(F)                                                                                            \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   bin_fast<float, float, F>((const float *)pa, (const float *)pb, (float *)po, (float)e->alpha, n, s); break; \
        case ACL_FLOAT16: bin_fast<__half, float, F>((const __half *)pa, (const __half *)pb, (__half *)po, (float)e->alpha, n, s); break; \
        case ACL_BF16:    bin_fast<__nv_bfloat16, float, F>((const __nv_bfloat16 *)pa, (const __nv_bfloat16 *)pb, (__nv_bfloat16 *)po, (float)e->alpha, n, s); break; \
        case ACL_INT32:   bin_fast<int32_t, int64_t, F>((const int32_t *)pa, (const int32_t *)pb, (int32_t *)po, (int64_t)e->alpha, n, s); break; \
        case ACL_FLOAT8_E4M3FN: bin_fast<__nv_fp8_e4m3, float, F>((const __nv_fp8_e4m3 *)pa, (const __nv_fp8_e4m3 *)pb, (__nv_fp8_e4m3 *)po, (float)e->alpha, n, s); break; \
        case ACL_FLOAT8_E5M2:   bin_fast<__nv_fp8_e5m2, float, F>((const __nv_fp8_e5m2 *)pa, (const __nv_fp8_e5m2 *)pb, (__nv_fp8_e5m2 *)po, (float)e->alpha, n, s); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
// General binary launch (broadcast + non-contiguous + mixed dtype). Same dtype: TA=TB=TO;
// mixed dtype: promote to fp32 output (TO=float, ACC=float). Does not delete e; returns false on launch failure.
template <template <typename> class F>
bool launch_g_bin(aclOpExecutor *e, cudaStream_t s) {
    GBc d; fill_g(e->a, e->b, e->out, d);
    const int64_t n = e->out->numel();
    const int64_t g = nblocks(n);
    const void *pa = e->a->data, *pb = e->b->data; void *po = e->out->data;
    aclDataType da = e->a->dtype, db = e->b->dtype;
    if (da == db) {
        switch (da) {
            case ACL_FLOAT:   k_bin_g<float,float,float,float,F><<<g,THREADS,0,s>>>((const float*)pa,(const float*)pb,(float*)po,(float)e->alpha,n,d); return true;
            case ACL_FLOAT16: k_bin_g<__half,__half,__half,float,F><<<g,THREADS,0,s>>>((const __half*)pa,(const __half*)pb,(__half*)po,(float)e->alpha,n,d); return true;
            case ACL_BF16:    k_bin_g<__nv_bfloat16,__nv_bfloat16,__nv_bfloat16,float,F><<<g,THREADS,0,s>>>((const __nv_bfloat16*)pa,(const __nv_bfloat16*)pb,(__nv_bfloat16*)po,(float)e->alpha,n,d); return true;
            case ACL_INT32:   k_bin_g<int32_t,int32_t,int32_t,int64_t,F><<<g,THREADS,0,s>>>((const int32_t*)pa,(const int32_t*)pb,(int32_t*)po,(int64_t)e->alpha,n,d); return true;
            case ACL_FLOAT8_E4M3FN: k_bin_g<__nv_fp8_e4m3,__nv_fp8_e4m3,__nv_fp8_e4m3,float,F><<<g,THREADS,0,s>>>((const __nv_fp8_e4m3*)pa,(const __nv_fp8_e4m3*)pb,(__nv_fp8_e4m3*)po,(float)e->alpha,n,d); return true;
            case ACL_FLOAT8_E5M2:   k_bin_g<__nv_fp8_e5m2,__nv_fp8_e5m2,__nv_fp8_e5m2,float,F><<<g,THREADS,0,s>>>((const __nv_fp8_e5m2*)pa,(const __nv_fp8_e5m2*)pb,(__nv_fp8_e5m2*)po,(float)e->alpha,n,d); return true;
            default: return false;
        }
    }
    #define GB_B(TA) switch (db) {                                                                                  \
        case ACL_FLOAT:   k_bin_g<TA,float,float,float,F><<<g,THREADS,0,s>>>((const TA*)pa,(const float*)pb,(float*)po,(float)e->alpha,n,d); return true; \
        case ACL_FLOAT16: k_bin_g<TA,__half,float,float,F><<<g,THREADS,0,s>>>((const TA*)pa,(const __half*)pb,(float*)po,(float)e->alpha,n,d); return true; \
        case ACL_BF16:    k_bin_g<TA,__nv_bfloat16,float,float,F><<<g,THREADS,0,s>>>((const TA*)pa,(const __nv_bfloat16*)pb,(float*)po,(float)e->alpha,n,d); return true; \
        case ACL_INT32:   k_bin_g<TA,int32_t,float,float,F><<<g,THREADS,0,s>>>((const TA*)pa,(const int32_t*)pb,(float*)po,(float)e->alpha,n,d); return true; \
        default: return false; }
    switch (da) {
        case ACL_FLOAT:   GB_B(float)
        case ACL_FLOAT16: GB_B(__half)
        case ACL_BF16:    GB_B(__nv_bfloat16)
        case ACL_INT32:   GB_B(int32_t)
        default: return false;
    }
    #undef GB_B
}
// Arithmetic binary: same-shape/same-dtype/contiguous takes fast path; otherwise takes general path
#define DO_BIN(F) do { if (fast) { DISPATCH_BIN(F) } else { if (!launch_g_bin<F>(e, s)) { delete e; return ACLNN_ERR_PARAM_INVALID; } } } while (0)
#define DISPATCH_SCALAR(F)                                                                                     \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   k_scalar<float, float, F><<<g, THREADS, 0, s>>>((const float *)pa, (float *)po, (float)e->alpha, 1.f, n); break; \
        case ACL_FLOAT16: k_scalar<__half, float, F><<<g, THREADS, 0, s>>>((const __half *)pa, (__half *)po, (float)e->alpha, 1.f, n); break; \
        case ACL_BF16:    k_scalar<__nv_bfloat16, float, F><<<g, THREADS, 0, s>>>((const __nv_bfloat16 *)pa, (__nv_bfloat16 *)po, (float)e->alpha, 1.f, n); break; \
        case ACL_INT32:   k_scalar<int32_t, int64_t, F><<<g, THREADS, 0, s>>>((const int32_t *)pa, (int32_t *)po, (int64_t)e->alpha, 1, n); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
#define DISPATCH_SCALAR2(F)                                                                                    \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   k_scalar2<float, float, F><<<g, THREADS, 0, s>>>((const float *)pa, (float *)po, (float)e->dscalars[0], (float)e->dscalars[1], n); break; \
        case ACL_FLOAT16: k_scalar2<__half, float, F><<<g, THREADS, 0, s>>>((const __half *)pa, (__half *)po, (float)e->dscalars[0], (float)e->dscalars[1], n); break; \
        case ACL_BF16:    k_scalar2<__nv_bfloat16, float, F><<<g, THREADS, 0, s>>>((const __nv_bfloat16 *)pa, (__nv_bfloat16 *)po, (float)e->dscalars[0], (float)e->dscalars[1], n); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
#define DISPATCH_SCALAR3(F)                                                                                    \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   k_scalar3<float, float, F><<<g, THREADS, 0, s>>>((const float *)pa, (float *)po, (float)e->dscalars[0], (float)e->dscalars[1], (float)e->dscalars[2], n); break; \
        case ACL_FLOAT16: k_scalar3<__half, float, F><<<g, THREADS, 0, s>>>((const __half *)pa, (__half *)po, (float)e->dscalars[0], (float)e->dscalars[1], (float)e->dscalars[2], n); break; \
        case ACL_BF16:    k_scalar3<__nv_bfloat16, float, F><<<g, THREADS, 0, s>>>((const __nv_bfloat16 *)pa, (__nv_bfloat16 *)po, (float)e->dscalars[0], (float)e->dscalars[1], (float)e->dscalars[2], n); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
#define DISPATCH_TERN(F)                                                                                        \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   k_tern<float, float, F><<<g, THREADS, 0, s>>>((const float *)pa, (const float *)pb, (const float *)pc, (float *)po, (float)e->alpha, n); break; \
        case ACL_FLOAT16: k_tern<__half, float, F><<<g, THREADS, 0, s>>>((const __half *)pa, (const __half *)pb, (const __half *)pc, (__half *)po, (float)e->alpha, n); break; \
        case ACL_BF16:    k_tern<__nv_bfloat16, float, F><<<g, THREADS, 0, s>>>((const __nv_bfloat16 *)pa, (const __nv_bfloat16 *)pb, (const __nv_bfloat16 *)pc, (__nv_bfloat16 *)po, (float)e->alpha, n); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
#define DISPATCH_UN(F)                                                                                             \
    switch (e->a->dtype) {                                                                                         \
        case ACL_FLOAT:   un_go<float, float, F>((const float *)pa, (float *)po, ufast, ud, n, s); break;   \
        case ACL_FLOAT16: un_go<__half, float, F>((const __half *)pa, (__half *)po, ufast, ud, n, s); break; \
        case ACL_BF16:    un_go<__nv_bfloat16, float, F>((const __nv_bfloat16 *)pa, (__nv_bfloat16 *)po, ufast, ud, n, s); break; \
        case ACL_FLOAT8_E4M3FN: un_go<__nv_fp8_e4m3, float, F>((const __nv_fp8_e4m3 *)pa, (__nv_fp8_e4m3 *)po, ufast, ud, n, s); break; \
        case ACL_FLOAT8_E5M2:   un_go<__nv_fp8_e5m2, float, F>((const __nv_fp8_e5m2 *)pa, (__nv_fp8_e5m2 *)po, ufast, ud, n, s); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }
#define DISPATCH_BIT(K)                                                                                            \
    switch (e->a->dtype) {                                                                                         \
        case ACL_INT16:  K<int16_t><<<g, THREADS, 0, s>>>((const int16_t *)pa, (const int16_t *)pb, (int16_t *)po, n); break; \
        case ACL_INT32:  K<int32_t><<<g, THREADS, 0, s>>>((const int32_t *)pa, (const int32_t *)pb, (int32_t *)po, n); break; \
        case ACL_UINT8: case ACL_BOOL: K<uint8_t><<<g, THREADS, 0, s>>>((const uint8_t *)pa, (const uint8_t *)pb, (uint8_t *)po, n); break; \
        default: delete e; return ACLNN_ERR_PARAM_INVALID; }

// fp32 kernel (reused by the fp4/fp6 unpack path): runs the same functors on fp32 scratch buffers
aclnnStatus ew_fp32_core(int op, const float *fa, const float *fb, float *fo, int64_t n, float alpha, cudaStream_t s) {
    const int64_t g = nblocks(n);
    switch (op) {
        case OP_ADD: k_bin<float,float,FAdd><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_SUB: k_bin<float,float,FSub><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_MUL: k_bin<float,float,FMul><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_DIV: k_bin<float,float,FDiv><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_MAXIMUM: k_bin<float,float,FMax><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_MINIMUM: k_bin<float,float,FMin><<<g,THREADS,0,s>>>(fa,fb,fo,alpha,n); break;
        case OP_ADDS: k_scalar<float,float,FAdd><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_SUBS: k_scalar<float,float,FSub><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_MULS: k_scalar<float,float,FMul><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_DIVS: k_scalar<float,float,FDiv><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_CLAMP_MIN: k_scalar<float,float,FMax><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_CLAMP_MAX: k_scalar<float,float,FMin><<<g,THREADS,0,s>>>(fa,fo,alpha,1.f,n); break;
        case OP_EXP: k_un<float,float,FExp><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_LOG: k_un<float,float,FLog><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_ABS: k_un<float,float,FAbs><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_SQRT: k_un<float,float,FSqrt><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_RSQRT: k_un<float,float,FRsqrt><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_RECIPROCAL: k_un<float,float,FRecip><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_RELU: k_un<float,float,FRelu><<<g,THREADS,0,s>>>(fa,fo,n); break;
        case OP_NEG: k_un<float,float,FNeg><<<g,THREADS,0,s>>>(fa,fo,n); break;
        default: return ACLNN_ERR_PARAM_INVALID;     // bitwise/SWhere not applicable to fp4/fp6
    }
    return ACLNN_SUCCESS;
}

// fp4/fp6 elementwise: unpack to fp32 → float kernel → encode back (including fp4 packing). Supports binary broadcast.
aclnnStatus run_elementwise_subfp(aclOpExecutor *e, void *ws, cudaStream_t s) {
    const int64_t na = e->a->numel(), nb = e->b ? e->b->numel() : 0, no = e->out->numel();
    int k = ew_subkind(e->a->dtype); bool fp4 = ew_is_fp4(e->a->dtype);
    if (e->b && e->b->dtype != e->a->dtype) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    if (e->out->dtype != e->a->dtype) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    float *fa = (float *)ws, *fb = fa + na, *fo = fb + nb;   // each sized by its own numel (broadcast-compatible layout)
    k_subfp_dec<<<nblocks(na), THREADS, 0, s>>>((const uint8_t *)e->a->data, fa, na, k, fp4);
    if (e->b) k_subfp_dec<<<nblocks(nb), THREADS, 0, s>>>((const uint8_t *)e->b->data, fb, nb, k, fp4);
    aclnnStatus st;
    bool bcast = e->b && (e->a->viewDims != e->out->viewDims || e->b->viewDims != e->out->viewDims);
    if (bcast) {
        // fa/fb are already contiguous fp32 in their respective logical shapes; perform broadcast binary with out shape
        GBc d; d.rank = (int)e->out->viewDims.size(); d.aoff = 0; d.boff = 0;
        int ra = (int)e->a->viewDims.size(), rb = (int)e->b->viewDims.size();
        int64_t as[8], bs[8]; { int64_t acc = 1; for (int i = ra-1; i>=0; --i){ as[i]=acc; acc*=e->a->viewDims[i]; } }
        { int64_t acc = 1; for (int i = rb-1; i>=0; --i){ bs[i]=acc; acc*=e->b->viewDims[i]; } }
        int r = d.rank;
        for (int i = 0; i < r; ++i) { d.od[i] = e->out->viewDims[i];
            int ia = i-(r-ra), ib = i-(r-rb);
            d.as[i] = (ia>=0 && e->a->viewDims[ia]==e->out->viewDims[i]) ? as[ia] : 0;
            d.bs[i] = (ib>=0 && e->b->viewDims[ib]==e->out->viewDims[i]) ? bs[ib] : 0;
        }
        int64_t g = nblocks(no);
        #define SB(F) k_bin_g<float,float,float,float,F><<<g,THREADS,0,s>>>(fa,fb,fo,(float)e->alpha,no,d)
        switch (e->op) {
            case OP_ADD: SB(FAdd); break; case OP_SUB: SB(FSub); break; case OP_MUL: SB(FMul); break;
            case OP_DIV: SB(FDiv); break; case OP_MAXIMUM: SB(FMax); break; case OP_MINIMUM: SB(FMin); break;
            default: delete e; return ACLNN_ERR_PARAM_INVALID;
        }
        #undef SB
        st = ACLNN_SUCCESS;
    } else {
        st = ew_fp32_core(e->op, fa, e->b ? fb : nullptr, fo, no, (float)e->alpha, s);
    }
    if (st != ACLNN_SUCCESS) { delete e; return st; }
    int64_t ge = fp4 ? (((no + 1) / 2 + THREADS - 1) / THREADS) : nblocks(no);
    k_subfp_enc<<<ge, THREADS, 0, s>>>(fo, (uint8_t *)e->out->data, no, k, fp4);
    cudaError_t err = cudaGetLastError();
    delete e;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

aclnnStatus run_elementwise(aclOpExecutor *e, void *ws, cudaStream_t s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    if (ew_is_subfp(e->a->dtype)) return run_elementwise_subfp(e, ws, s);
    const int64_t n = e->out->numel();   // under broadcast out may be larger than a; loop over out
    const int64_t g = nblocks(n);
    const void *pa = e->a->data, *pb = e->b ? e->b->data : nullptr, *pc = e->c ? e->c->data : nullptr;
    void *po = e->out->data;
    // Fast path condition: arithmetic binary with same shape, same dtype, all inputs/output contiguous; otherwise use general launch
    bool fast = e->b && is_bcast_op(e->op) && (e->a->viewDims == e->b->viewDims) &&
                (e->a->dtype == e->b->dtype) && e->a->contiguous() && e->b->contiguous() && e->out->contiguous();
    // Unary: contiguous input takes fast path; otherwise uses stride-based general addressing
    bool ufast = e->a->contiguous(); GUn ud{}; if (!ufast) fill_un(e->a, ud);
    switch (e->op) {
        case OP_ADD:     DO_BIN(FAdd); break;
        case OP_SUB:     DO_BIN(FSub); break;
        case OP_MUL:     DO_BIN(FMul); break;
        case OP_DIV:     DO_BIN(FDiv); break;
        case OP_MAXIMUM: DO_BIN(FMax); break;
        case OP_MINIMUM: DO_BIN(FMin); break;
        case OP_FMOD:    DO_BIN(FFmod); break;
        case OP_HYPOT:   DO_BIN(FHypot); break;
        case OP_POW:     DO_BIN(FPow); break;
        case OP_ADDS:      DISPATCH_SCALAR(FAdd) break;
        case OP_SUBS:      DISPATCH_SCALAR(FSub) break;
        case OP_MULS:      DISPATCH_SCALAR(FMul) break;
        case OP_DIVS:      DISPATCH_SCALAR(FDiv) break;
        case OP_CLAMP_MIN: DISPATCH_SCALAR(FMax) break;   // max(self, s) = lower clamp
        case OP_CLAMP_MAX: DISPATCH_SCALAR(FMin) break;
        case OP_EXP:        DISPATCH_UN(FExp) break;
        case OP_LOG:        DISPATCH_UN(FLog) break;
        case OP_ABS:        DISPATCH_UN(FAbs) break;
        case OP_SQRT:       DISPATCH_UN(FSqrt) break;
        case OP_RSQRT:      DISPATCH_UN(FRsqrt) break;
        case OP_RECIPROCAL: DISPATCH_UN(FRecip) break;
        case OP_RELU:       DISPATCH_UN(FRelu) break;
        case OP_NEG:        DISPATCH_UN(FNeg) break;
        case OP_SIGMOID:    DISPATCH_UN(FSigmoid) break;
        case OP_TANH:       DISPATCH_UN(FTanh) break;
        case OP_ERF:        DISPATCH_UN(FErf) break;
        case OP_GELU:       DISPATCH_UN(FGelu) break;
        case OP_SILU:       DISPATCH_UN(FSilu) break;
        case OP_SOFTPLUS:   DISPATCH_UN(FSoftplus) break;
        case OP_SIN:        DISPATCH_UN(FSin) break;
        case OP_COS:        DISPATCH_UN(FCos) break;
        case OP_TAN:        DISPATCH_UN(FTan) break;
        case OP_ATAN:       DISPATCH_UN(FAtan) break;
        case OP_SIGN:       DISPATCH_UN(FSign) break;
        case OP_FLOOR:      DISPATCH_UN(FFloor) break;
        case OP_CEIL:       DISPATCH_UN(FCeil) break;
        case OP_ROUND:      DISPATCH_UN(FRound) break;
        case OP_TRUNC:      DISPATCH_UN(FTrunc) break;
        case OP_SQUARE:     DISPATCH_UN(FSquare) break;
        case OP_SINH:       DISPATCH_UN(FSinh) break;
        case OP_COSH:       DISPATCH_UN(FCosh) break;
        case OP_ASIN:       DISPATCH_UN(FAsin) break;
        case OP_ACOS:       DISPATCH_UN(FAcos) break;
        case OP_ERFC:       DISPATCH_UN(FErfc) break;
        case OP_FRAC:       DISPATCH_UN(FFrac) break;
        case OP_LGAMMA:     DISPATCH_UN(FLgamma) break;
        case OP_ACOSH:      DISPATCH_UN(FAcosh) break;
        case OP_ASINH:      DISPATCH_UN(FAsinh) break;
        case OP_ATANH:      DISPATCH_UN(FAtanh) break;
        case OP_SINC:       DISPATCH_UN(FSinc) break;
        case OP_DIGAMMA:    DISPATCH_UN(FDigamma) break;
        case OP_FASTGELU:   DISPATCH_UN(FFastGelu) break;
        case OP_SQUAREDRELU: DISPATCH_UN(FSquaredRelu) break;
        case OP_SWISH:      DISPATCH_SCALAR(FSwish) break;
        case OP_POWS:       DISPATCH_SCALAR(FPow) break;
        // ---- P1 elementwise math ----
        case OP_EXPM1:      DISPATCH_UN(FExpm1) break;
        case OP_LOG1P:      DISPATCH_UN(FLog1p) break;
        case OP_LOG2:       DISPATCH_UN(FLog2) break;
        case OP_LOG10:      DISPATCH_UN(FLog10) break;
        case OP_EXP2:       DISPATCH_UN(FExp2) break;
        case OP_ERFINV:     DISPATCH_UN(FErfinv) break;
        case OP_ATAN2:      DO_BIN(FAtan2); break;
        case OP_REMAINDER:  DO_BIN(FRemainder); break;
        case OP_XLOGY:      DO_BIN(FXlogy); break;
        case OP_LOGADDEXP:  DO_BIN(FLogaddexp); break;
        case OP_COPYSIGN:   DO_BIN(FCopysign); break;
        case OP_HEAVISIDE:  DO_BIN(FHeaviside); break;
        case OP_LERP:       DO_BIN(FLerp); break;
        case OP_LOGADDEXP2: DO_BIN(FLogaddexp2); break;
        case OP_FLOORDIV:   if (e->b) { DO_BIN(FFloorDiv); } else { DISPATCH_SCALAR(FFloorDiv) } break;   // scalar variant (aclnnFloorDivides) has b==null
        case OP_RSUB:       if (e->b) { DO_BIN(FRsub); } else { DISPATCH_SCALAR(FRsub) } break;          // scalar variant (aclnnRsubs) has b==null
        case OP_ROUNDDEC:   DISPATCH_SCALAR(FRoundDec) break;
        case OP_NANTONUM:   DISPATCH_SCALAR3(FNanToNum) break;
        case OP_PRELU:      DO_BIN(FPrelu); break;
        case OP_ADDCMUL:    DISPATCH_TERN(TAddcmul) break;
        case OP_ADDCDIV:    DISPATCH_TERN(TAddcdiv) break;
        case OP_CLAMPT:     DISPATCH_TERN(TClamp) break;
        // ---- P2 activations ----
        case OP_MISH:        DISPATCH_UN(FMish) break;
        case OP_HARDSWISH:   DISPATCH_UN(FHardswish) break;
        case OP_HARDSIGMOID: DISPATCH_UN(FHardsigmoid) break;
        case OP_LOGSIGMOID:  DISPATCH_UN(FLogSigmoid) break;
        case OP_SELU:        DISPATCH_UN(FSelu) break;
        case OP_TANHSHRINK:  DISPATCH_UN(FTanhshrink) break;
        case OP_RELU6:       DISPATCH_UN(FRelu6) break;
        case OP_LEAKYRELU:   DISPATCH_SCALAR(FLeakyRelu) break;
        case OP_ELU:         DISPATCH_SCALAR(FElu) break;
        case OP_CELU:        DISPATCH_SCALAR(FCelu) break;
        case OP_HARDSHRINK:  DISPATCH_SCALAR(FHardshrink) break;
        case OP_SOFTSHRINK:  DISPATCH_SCALAR(FSoftshrink) break;
        case OP_HARDTANH:    DISPATCH_SCALAR2(FHardtanh) break;
        case OP_THRESHOLD:   DISPATCH_SCALAR2(FThreshold) break;
        case OP_BAND: case OP_BOR: case OP_BXOR: {
            // same-shape contiguous takes fast path; otherwise stride-based broadcast
            bool bf = (e->a->viewDims == e->b->viewDims) && e->a->contiguous() && e->b->contiguous() && e->out->contiguous();
            if (bf) { if (e->op == OP_BAND) DISPATCH_BIT(k_band) else if (e->op == OP_BOR) DISPATCH_BIT(k_bor) else DISPATCH_BIT(k_bxor) }
            else { GBc bd; fill_g(e->a, e->b, e->out, bd);
                #define BG(T) do { if (e->op == OP_BAND) k_bit_g<T,0><<<g,THREADS,0,s>>>((const T*)pa,(const T*)pb,(T*)po,n,bd); \
                                   else if (e->op == OP_BOR) k_bit_g<T,1><<<g,THREADS,0,s>>>((const T*)pa,(const T*)pb,(T*)po,n,bd); \
                                   else k_bit_g<T,2><<<g,THREADS,0,s>>>((const T*)pa,(const T*)pb,(T*)po,n,bd); } while(0)
                switch (e->a->dtype) {
                    case ACL_INT16: BG(int16_t); break;
                    case ACL_INT32: BG(int32_t); break;
                    case ACL_UINT8: case ACL_BOOL: BG(uint8_t); break;
                    default: delete e; return ACLNN_ERR_PARAM_INVALID;
                }
                #undef BG
            }
            break;
        }
        case OP_BANDS: case OP_BORS: case OP_BXORS: {
            // tensor ∘ scalar bitwise (self contiguous, same dtype out)
            #define BS(T) do { T sv = (T)(int64_t)e->alpha; \
                if (e->op == OP_BANDS) k_bit_s<T,0><<<g,THREADS,0,s>>>((const T*)pa,sv,(T*)po,n); \
                else if (e->op == OP_BORS) k_bit_s<T,1><<<g,THREADS,0,s>>>((const T*)pa,sv,(T*)po,n); \
                else k_bit_s<T,2><<<g,THREADS,0,s>>>((const T*)pa,sv,(T*)po,n); } while(0)
            switch (e->a->dtype) {
                case ACL_INT16: BS(int16_t); break;
                case ACL_INT32: BS(int32_t); break;
                case ACL_UINT8: case ACL_BOOL: BS(uint8_t); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            #undef BS
            break;
        }
        case OP_BNOT:
            switch (e->a->dtype) {
                case ACL_INT16: k_bnot<int16_t><<<g, THREADS, 0, s>>>((const int16_t *)pa, (int16_t *)po, n); break;
                case ACL_INT32: k_bnot<int32_t><<<g, THREADS, 0, s>>>((const int32_t *)pa, (int32_t *)po, n); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            break;
        case OP_SWHERE: {  // e->a = condition(BOOL/UINT8), e->b = self, e->c = other (dtype checked for b/c)
            // same-shape contiguous takes fast path; otherwise three-input broadcast
            bool wf = (e->a->viewDims == e->out->viewDims) && (e->b->viewDims == e->out->viewDims) &&
                      (e->c->viewDims == e->out->viewDims) && e->a->contiguous() && e->b->contiguous() &&
                      e->c->contiguous() && e->out->contiguous();
            GW3 wd; if (!wf) fill_g3(e->a, e->b, e->c, e->out, wd);
            #define WG(T) do { if (wf) k_where<T><<<g,THREADS,0,s>>>((const uint8_t*)pa,(const T*)pb,(const T*)pc,(T*)po,n); \
                               else k_where_g<T><<<g,THREADS,0,s>>>((const uint8_t*)pa,(const T*)pb,(const T*)pc,(T*)po,n,wd); } while(0)
            switch (e->b->dtype) {
                case ACL_FLOAT:   WG(float); break;
                case ACL_FLOAT16: WG(__half); break;
                case ACL_INT32:   WG(int32_t); break;
                default: delete e; return ACLNN_ERR_PARAM_INVALID;
            }
            #undef WG
            break;
        }
        default: delete e; return ACLNN_ERR_PARAM_INVALID;
    }
    cudaError_t err = cudaGetLastError();
    delete e;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

// Single-tensor elementwise dispatch reused by the foreach family: builds a one-shot executor over
// (a, b, c, out, alpha) for the given OpKind and runs it. run_elementwise frees the executor.
// Workspace is only consulted for fp4/fp6 inputs, which foreach does not support (pass ws=nullptr).
aclnnStatus ew_run_one(int op, const aclTensor *a, const aclTensor *b, const aclTensor *c,
                       aclTensor *out, double alpha, void *ws, cudaStream_t s) {
    if (!a || !out) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor();
    e->op = op; e->a = a; e->b = b; e->c = c; e->out = out; e->alpha = alpha;
    return run_elementwise(e, ws, s);
}

// Run a pre-built elementwise executor (Execute phase for ops that stash their plan in GetWorkspaceSize,
// e.g. the in-place family). Frees the executor.
aclnnStatus ew_exec(aclOpExecutor *e, void *ws, cudaStream_t s) { return run_elementwise(e, ws, s); }

extern "C" {

#define IMPL_BIN_ALPHA(name, op)                                                                                     \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha,            \
                                   aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {                               \
    if (self && other && self->dtype != other->dtype && !is_bcast_op(op)) return ACLNN_ERR_PARAM_INVALID;            \
    return make_exec(op, self, other, nullptr, out, alpha ? alpha->v : 1.0, ws, ex);                                 \
}                                                                                                                    \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

#define IMPL_BIN(name, op)                                                                                           \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,                    \
                                   uint64_t *ws, aclOpExecutor **ex) {                                               \
    if (self && other && self->dtype != other->dtype && !is_bcast_op(op)) return ACLNN_ERR_PARAM_INVALID;            \
    return make_exec(op, self, other, nullptr, out, 1.0, ws, ex);                                                    \
}                                                                                                                    \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

#define IMPL_SCALAR_ALPHA(name, op)                                                                                  \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, const aclScalar *alpha,            \
                                   aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {                               \
    if (!other) return ACLNN_ERR_PARAM_NULLPTR;                                                                      \
    return make_exec(op, self, nullptr, nullptr, out, other->v * (alpha ? alpha->v : 1.0), ws, ex);                  \
}                                                                                                                    \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

#define IMPL_SCALAR(name, op)                                                                                        \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out,                    \
                                   uint64_t *ws, aclOpExecutor **ex) {                                               \
    if (!other) return ACLNN_ERR_PARAM_NULLPTR;                                                                      \
    return make_exec(op, self, nullptr, nullptr, out, other->v, ws, ex);                                             \
}                                                                                                                    \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

#define IMPL_UN(name, op)                                                                                            \
aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {        \
    return make_exec(op, self, nullptr, nullptr, out, 1.0, ws, ex);                                                  \
}                                                                                                                    \
aclnnStatus name(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

IMPL_BIN_ALPHA(aclnnAdd, OP_ADD)
IMPL_BIN_ALPHA(aclnnSub, OP_SUB)
IMPL_BIN(aclnnMul, OP_MUL)
IMPL_BIN(aclnnDiv, OP_DIV)
IMPL_BIN(aclnnMaximum, OP_MAXIMUM)
IMPL_BIN(aclnnMinimum, OP_MINIMUM)

IMPL_SCALAR_ALPHA(aclnnAdds, OP_ADDS)
IMPL_SCALAR_ALPHA(aclnnSubs, OP_SUBS)
IMPL_SCALAR(aclnnMuls, OP_MULS)
IMPL_SCALAR(aclnnDivs, OP_DIVS)
IMPL_SCALAR(aclnnClampMin, OP_CLAMP_MIN)
IMPL_SCALAR(aclnnClampMax, OP_CLAMP_MAX)

IMPL_UN(aclnnExp, OP_EXP)
IMPL_UN(aclnnLog, OP_LOG)
IMPL_UN(aclnnAbs, OP_ABS)
IMPL_UN(aclnnSqrt, OP_SQRT)
IMPL_UN(aclnnRsqrt, OP_RSQRT)
IMPL_UN(aclnnReciprocal, OP_RECIPROCAL)
IMPL_UN(aclnnRelu, OP_RELU)
IMPL_UN(aclnnNeg, OP_NEG)

IMPL_UN(aclnnSigmoid, OP_SIGMOID)
IMPL_UN(aclnnTanh, OP_TANH)
IMPL_UN(aclnnErf, OP_ERF)
IMPL_UN(aclnnGelu, OP_GELU)
IMPL_UN(aclnnSilu, OP_SILU)
IMPL_UN(aclnnSoftplus, OP_SOFTPLUS)
IMPL_UN(aclnnSin, OP_SIN)
IMPL_UN(aclnnCos, OP_COS)
IMPL_UN(aclnnTan, OP_TAN)
IMPL_UN(aclnnAtan, OP_ATAN)
IMPL_UN(aclnnSign, OP_SIGN)
IMPL_UN(aclnnFloor, OP_FLOOR)
IMPL_UN(aclnnCeil, OP_CEIL)
IMPL_UN(aclnnRound, OP_ROUND)
IMPL_UN(aclnnTrunc, OP_TRUNC)
IMPL_UN(aclnnSquare, OP_SQUARE)
IMPL_UN(aclnnSinh, OP_SINH)
IMPL_UN(aclnnCosh, OP_COSH)
IMPL_UN(aclnnAsin, OP_ASIN)
IMPL_UN(aclnnAcos, OP_ACOS)
IMPL_UN(aclnnErfc, OP_ERFC)
IMPL_UN(aclnnFrac, OP_FRAC)
IMPL_UN(aclnnLgamma, OP_LGAMMA)
IMPL_SCALAR(aclnnPowTensorScalar, OP_POWS)
IMPL_BIN(aclnnFmod, OP_FMOD)
IMPL_BIN(aclnnHypot, OP_HYPOT)
IMPL_BIN(aclnnPowTensorTensor, OP_POW)

IMPL_BIN(aclnnBitwiseAndTensor, OP_BAND)
IMPL_BIN(aclnnBitwiseOrTensor, OP_BOR)
IMPL_BIN(aclnnBitwiseXorTensor, OP_BXOR)
IMPL_UN(aclnnBitwiseNot, OP_BNOT)
IMPL_SCALAR(aclnnBitwiseAndScalar, OP_BANDS)
IMPL_SCALAR(aclnnBitwiseOrScalar, OP_BORS)
IMPL_SCALAR(aclnnBitwiseXorScalar, OP_BXORS)

// ---- P1 elementwise math ----
IMPL_UN(aclnnExpm1, OP_EXPM1)
IMPL_UN(aclnnLog1p, OP_LOG1P)
IMPL_UN(aclnnLog2, OP_LOG2)
IMPL_UN(aclnnLog10, OP_LOG10)
IMPL_UN(aclnnExp2, OP_EXP2)
IMPL_UN(aclnnErfinv, OP_ERFINV)
IMPL_BIN(aclnnAtan2, OP_ATAN2)
IMPL_BIN(aclnnRemainderTensorTensor, OP_REMAINDER)
IMPL_BIN(aclnnXLogYTensorTensor, OP_XLOGY)
IMPL_BIN(aclnnLogAddExp, OP_LOGADDEXP)
IMPL_BIN(aclnnCopysign, OP_COPYSIGN)
IMPL_BIN(aclnnHeaviside, OP_HEAVISIDE)
IMPL_BIN_ALPHA(aclnnLerp, OP_LERP)
// ---- additional elementary math ----
IMPL_UN(aclnnAcosh, OP_ACOSH)
IMPL_UN(aclnnAsinh, OP_ASINH)
IMPL_UN(aclnnAtanh, OP_ATANH)
IMPL_UN(aclnnSinc, OP_SINC)
IMPL_UN(aclnnDigamma, OP_DIGAMMA)
IMPL_BIN(aclnnLogAddExp2, OP_LOGADDEXP2)
// ---- additional activations ----
IMPL_UN(aclnnFastGelu, OP_FASTGELU)
IMPL_UN(aclnnSquaredRelu, OP_SQUAREDRELU)
IMPL_SCALAR(aclnnSwish, OP_SWISH)
// GeluV2: approximate==0 → exact erf GeLU; approximate==1 → tanh-approx (FastGelu)
aclnnStatus aclnnGeluV2GetWorkspaceSize(const aclTensor *self, int64_t approximate, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_exec(approximate == 1 ? OP_FASTGELU : OP_GELU, self, nullptr, nullptr, out, 1.0, ws, ex);
}
aclnnStatus aclnnGeluV2(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
// ---- more arithmetic ----
IMPL_BIN(aclnnFloorDivide, OP_FLOORDIV)
IMPL_SCALAR(aclnnFloorDivides, OP_FLOORDIV)
IMPL_BIN(aclnnFloorDiv, OP_FLOORDIV)        // experimental alias
IMPL_BIN(aclnnRealDiv, OP_DIV)
IMPL_BIN(aclnnDivV3, OP_DIV)
IMPL_BIN_ALPHA(aclnnRsub, OP_RSUB)
IMPL_SCALAR(aclnnRsubs, OP_RSUB)            // scalar form: other - self
// RoundDecimals: round to `decimals` places (integer parameter)
aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(const aclTensor *self, int64_t decimals, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_exec(OP_ROUNDDEC, self, nullptr, nullptr, out, (double)decimals, ws, ex);
}
aclnnStatus aclnnRoundDecimals(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
// NanToNum: replace NaN/+Inf/-Inf with given values
aclnnStatus aclnnNanToNumGetWorkspaceSize(const aclTensor *self, float nan, float posinf, float neginf, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_exec(OP_NANTONUM, self, nullptr, nullptr, out, 0.0, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->dscalars = {(double)nan, (double)posinf, (double)neginf};
    return st;
}
aclnnStatus aclnnNanToNum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(aclTensor *self, float nan, float posinf, float neginf, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_exec(OP_NANTONUM, self, nullptr, nullptr, self, 0.0, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->dscalars = {(double)nan, (double)posinf, (double)neginf};
    return st;
}
aclnnStatus aclnnInplaceNanToNum(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
// Clamp(self, min, max scalars): clamp to [min,max] (HardTanh); null bound → ±inf
aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor *self, const aclScalar *min, const aclScalar *max, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_exec(OP_HARDTANH, self, nullptr, nullptr, out, 0.0, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->dscalars = {min ? min->v : -1.0/0.0, max ? max->v : 1.0/0.0};
    return st;
}
aclnnStatus aclnnClamp(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
// ClampMaxTensor: elementwise min(self, maxT); ClampMinTensor: elementwise max(self, minT)
IMPL_BIN(aclnnClampMaxTensor, OP_MINIMUM)
IMPL_BIN(aclnnClampMinTensor, OP_MAXIMUM)

// ---- P2 activations (unary, no parameter) ----
IMPL_UN(aclnnMish, OP_MISH)
IMPL_UN(aclnnHardswish, OP_HARDSWISH)
IMPL_UN(aclnnHardsigmoid, OP_HARDSIGMOID)
IMPL_UN(aclnnLogSigmoid, OP_LOGSIGMOID)
IMPL_UN(aclnnSelu, OP_SELU)
IMPL_UN(aclnnTanhshrink, OP_TANHSHRINK)
IMPL_UN(aclnnRelu6, OP_RELU6)
// ---- P2 activations (unary, one scalar parameter) ----
IMPL_SCALAR(aclnnLeakyRelu, OP_LEAKYRELU)
IMPL_SCALAR(aclnnElu, OP_ELU)
IMPL_SCALAR(aclnnCelu, OP_CELU)
IMPL_SCALAR(aclnnHardshrink, OP_HARDSHRINK)
IMPL_SCALAR(aclnnSoftshrink, OP_SOFTSHRINK)
// Prelu: weight is a tensor (per-channel broadcast), so it uses the binary path
IMPL_BIN(aclnnPrelu, OP_PRELU)

// ---- P2 activations (unary, two scalar parameters) ----
aclnnStatus aclnnHardtanhGetWorkspaceSize(const aclTensor *self, const aclScalar *minVal, const aclScalar *maxVal,
                                          aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_exec(OP_HARDTANH, self, nullptr, nullptr, out, 0.0, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->dscalars = { minVal ? minVal->v : -1.0, maxVal ? maxVal->v : 1.0 };
    return st;
}
aclnnStatus aclnnHardtanh(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
aclnnStatus aclnnThresholdGetWorkspaceSize(const aclTensor *self, const aclScalar *threshold, const aclScalar *value,
                                           aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    aclnnStatus st = make_exec(OP_THRESHOLD, self, nullptr, nullptr, out, 0.0, ws, ex);
    if (st == ACLNN_SUCCESS) (*ex)->dscalars = { threshold ? threshold->v : 0.0, value ? value->v : 0.0 };
    return st;
}
aclnnStatus aclnnThreshold(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

// ---- P1 ternary (self, t1/min, t2/max + scalar) ----
aclnnStatus aclnnAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *t1, const aclTensor *t2,
                                         const aclScalar *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_exec(OP_ADDCMUL, self, t1, t2, out, value ? value->v : 1.0, ws, ex);
}
aclnnStatus aclnnAddcmul(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor *self, const aclTensor *t1, const aclTensor *t2,
                                         const aclScalar *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_exec(OP_ADDCDIV, self, t1, t2, out, value ? value->v : 1.0, ws, ex);
}
aclnnStatus aclnnAddcdiv(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }
aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *minT, const aclTensor *maxT,
                                             aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return make_exec(OP_CLAMPT, self, minT, maxT, out, 0.0, ws, ex);
}
aclnnStatus aclnnClampTensor(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self, const aclTensor *other,
                                        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!condition || (condition->dtype != ACL_BOOL && condition->dtype != ACL_UINT8)) return ACLNN_ERR_PARAM_INVALID;
    if (self && other && self->dtype != other->dtype) return ACLNN_ERR_PARAM_INVALID;
    return make_exec(OP_SWHERE, condition, self, other, out, 1.0, ws, ex);
}
aclnnStatus aclnnSWhere(void *ws, uint64_t, aclOpExecutor *e, aclrtStream st) { return run_elementwise(e, ws, (cudaStream_t)st); }

} // extern "C"
