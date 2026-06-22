// Elementwise compute kernels (MSL): generic unary (incl. param ops / tensor-scalar) and generic
// broadcast binary. Computation in float, stored as T. Op selected at runtime via the op code in meta.
#include <metal_stdlib>
#include "ew_ops.h"
using namespace metal;

struct UnMeta { uint n; uint ndim; int op; float p0; float p1; uint odims[8]; uint astr[8]; };
struct BcMeta { uint n; uint ndim; int op; float alpha; uint odims[8]; uint astr[8]; uint bstr[8]; };

inline uint bc_off(constant uint *str, uint gid, constant uint *odims, uint ndim) {
    uint rem = gid, off = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) { uint id = rem % odims[d]; rem /= odims[d]; off += id * str[d]; }
    return off;
}

// --- special functions MSL lacks (or whose CANN/Torch convention differs) ---
inline float log1p_(float x) { float u = 1.f + x; return (u == 1.f) ? x : x * log(u) / (u - 1.f); }   // accurate near 0
inline float expm1_(float x) { float u = exp(x); if (u == 1.f) return x; float um1 = u - 1.f; return (um1 == -1.f) ? -1.f : um1 * x / log(u); }
// erfc tail for x>0 via continued fraction (relative-accurate for x>=1; no cancellation):
//   erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
inline float erfc_tail(float x) {
    float z = x * x, f = x;
    for (int k = 80; k >= 1; --k) f = x + (k * 0.5f) / f;   // backward CF eval; ~80 terms converges down to x~0.8
    return exp(-z) / 1.7724538509055159f / f;
}
inline float erf_(float x) {                 // Taylor for |x|<2 (relative-accurate incl. near 0), tail elsewhere
    float ax = fabs(x);
    if (ax >= 2.f) return x > 0 ? 1.f - erfc_tail(ax) : erfc_tail(ax) - 1.f;
    float sum = 0.f, c = 0.f, t = x, x2 = x * x;   // Kahan-compensated: the alternating series loses ~1 digit otherwise
    for (int n = 0; n < 80; ++n) { float term = t / (2 * n + 1), y = term - c, snew = sum + y; c = (snew - sum) - y; sum = snew; t *= -x2 / (n + 1); if (fabs(t) < 1e-12f) break; }
    return 1.1283791670955126f * sum;        // 2/sqrt(pi)
}
inline float erfc_(float x) {                // direct tail for x>=1 keeps relative accuracy where erfc is small
    if (x >= 1.f) return erfc_tail(x);
    return 1.f - erf_(x);                     // x<1 (incl. negative): no catastrophic cancellation
}
inline float erfinv_(float x) {              // Giles 2010, single precision
    float w = -log((1.f - x) * (1.f + x)), p;
    if (w < 5.f) { w -= 2.5f;
        p = 2.81022636e-08f; p = 3.43273939e-07f + p * w; p = -3.5233877e-06f + p * w; p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w; p = -0.00125372503f + p * w; p = -0.00417768164f + p * w; p = 0.246640727f + p * w; p = 1.50140941f + p * w;
    } else { w = sqrt(w) - 3.f;
        p = -0.000200214257f; p = 0.000100950558f + p * w; p = 0.00134934322f + p * w; p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w; p = -0.0076224613f + p * w; p = 0.00943887047f + p * w; p = 1.00167406f + p * w; p = 2.83297682f + p * w;
    }
    return p * x;
}
inline float lgamma1p(float s) {             // lgamma(1+s), relatively accurate for s in [-0.5,0.5]
    // lgamma(1+s) = (1-gamma)s - log1p(s) + sum_{k>=2} (-1)^k (zeta(k)-1)/k s^k ; the (zeta-1) tail converges fast.
    const float zm1[14] = { 0.6449340668482264f, 0.2020569031595943f, 0.08232323371113819f, 0.03692775514336993f,
        0.01734306198444914f, 0.008349277381922827f, 0.004077356197944339f, 0.002008392826082214f,
        0.0009945751278180853f, 0.0004941886041194646f, 0.0002460865533080883f, 0.0001227133475784891f,
        0.00006124813505870483f, 0.00003058823630702049f };  // zeta(k)-1 for k=2..15
    float corr = 0.f, sk = s * s;
    for (int k = 2; k <= 15; ++k) { float t = zm1[k - 2] / (float)k * sk; corr += (k & 1) ? -t : t; sk *= s; }
    return (1.f - 0.5772156649015329f) * s - log1p_(s) + corr;
}
inline float lgamma_(float x) {              // reduce to [0.5,1.5] by recurrence, keeping relative accuracy near zeros at 1,2
    if (x < 0.5f) return log(fabs(3.14159265358979f / sin(3.14159265358979f * x))) - lgamma_(1.f - x);
    float acc = 0.f;
    while (x > 1.5f) { x -= 1.f; acc += log1p_(x - 1.f); }   // lgamma(x)=log(x-1)+lgamma(x-1)
    return acc + lgamma1p(x - 1.f);
}

inline float un_op(int op, float x, float p0, float p1) {
    switch (op) {
        case U_EXP: return exp(x);            case U_LOG: return log(x);
        case U_ABS: return fabs(x);           case U_SQRT: return sqrt(x);
        case U_RSQRT: return rsqrt(x);        case U_RECIP: return 1.f / x;
        case U_RELU: return x > 0 ? x : 0;    case U_NEG: return -x;
        case U_SIGMOID: return 1.f / (1.f + exp(-x));   case U_TANH: return tanh(x);
        case U_ERF: return erf_(x);           case U_ERFC: return erfc_(x);
        case U_GELU: return 0.5f * x * erfc_(-x * 0.70710678118654752f);   // erfc(-z)=1+erf(z): no cancellation for x<0
        case U_SILU: return x / (1.f + exp(-x));
        case U_SOFTPLUS: return x > 0 ? x + log1p_(exp(-x)) : log1p_(exp(x));
        case U_SIN: return sin(x);            case U_COS: return cos(x);
        case U_TAN: return tan(x);            case U_ATAN: return atan(x);
        case U_SIGN: return (float)((x > 0) - (x < 0));
        case U_FLOOR: return floor(x);        case U_CEIL: return ceil(x);
        case U_ROUND: return rint(x);         case U_TRUNC: return trunc(x);
        case U_SQUARE: return x * x;          case U_SINH: return sinh(x);
        case U_COSH: return cosh(x);          case U_ASIN: return asin(x);
        case U_ACOS: return acos(x);          case U_FRAC: return x - trunc(x);
        case U_LGAMMA: return lgamma_(x);
        case U_EXPM1: return expm1_(x);       case U_LOG1P: return log1p_(x);
        case U_LOG2: return log2(x);          case U_LOG10: return log10(x);
        case U_EXP2: return exp2(x);          case U_ERFINV: return erfinv_(x);
        case U_MISH: return x * tanh(x > 0 ? x + log1p_(exp(-x)) : log1p_(exp(x)));
        case U_HARDSWISH: return x * (x <= -3.f ? 0.f : (x >= 3.f ? 1.f : (x + 3.f) / 6.f));
        case U_HARDSIGMOID: return x <= -3.f ? 0.f : (x >= 3.f ? 1.f : (x + 3.f) / 6.f);
        case U_LOGSIGMOID: return -(x > 0 ? log1p_(exp(-x)) : -x + log1p_(exp(x)));
        case U_SELU: { const float a = 1.6732632423543772f, s = 1.0507009873554805f; return s * (x > 0 ? x : a * expm1_(x)); }
        case U_TANHSHRINK: return x - tanh(x);
        case U_RELU6: return fmin(fmax(x, 0.f), 6.f);
        case U_ADDC: return x + p0;           case U_MULC: return x * p0;
        case U_CLAMP_LO: return fmax(x, p0);  case U_CLAMP_HI: return fmin(x, p0);
        case U_POWS: return pow(x, p0);
        case U_LEAKYRELU: return x > 0 ? x : p0 * x;
        case U_ELU: return x > 0 ? x : p0 * expm1_(x);
        case U_CELU: return fmax(0.f, x) + fmin(0.f, p0 * expm1_(x / p0));
        case U_HARDSHRINK: return fabs(x) > p0 ? x : 0.f;
        case U_SOFTSHRINK: return x > p0 ? x - p0 : (x < -p0 ? x + p0 : 0.f);
        case U_CLAMP_LOHI: return fmin(fmax(x, p0), p1);
        case U_THRESHOLD: return x > p0 ? x : p1;
        default: return x;
    }
}

inline float bin_op(int op, float a, float b, float alpha) {
    switch (op) {
        case B_ADD: return a + alpha * b;     case B_SUB: return a - alpha * b;
        case B_MUL: return a * b;             case B_DIV: return a / b;
        case B_MAX: return fmax(a, b);        case B_MIN: return fmin(a, b);
        case B_POW: return pow(a, b);         case B_FMOD: return fmod(a, b);
        case B_HYPOT: return sqrt(a * a + b * b);   case B_ATAN2: return atan2(a, b);
        case B_REMAINDER: { float r = fmod(a, b); if (r != 0.f && ((r < 0.f) != (b < 0.f))) r += b; return r; }
        case B_XLOGY: return a == 0.f ? 0.f : a * log(b);
        case B_LOGADDEXP: { float m = fmax(a, b), d = -fabs(a - b); return m + log1p_(exp(d)); }
        case B_COPYSIGN: return copysign(a, b);
        case B_HEAVISIDE: return a < 0.f ? 0.f : (a > 0.f ? 1.f : b);
        case B_PRELU: return a > 0.f ? a : b * a;
        case B_LERP: return a + alpha * (b - a);
        default: return a;
    }
}

template <typename T>
kernel void ew_un(device const T *in [[buffer(0)]], device T *out [[buffer(1)]],
                  constant UnMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint ao = bc_off(m.astr, gid, m.odims, m.ndim);   // honor (possibly strided/transposed) input view
    out[gid] = (T)un_op(m.op, (float)in[ao], m.p0, m.p1);
}
template [[host_name("ew_un_f32")]] kernel void ew_un<float>(device const float *, device float *, constant UnMeta &, uint);
template [[host_name("ew_un_f16")]] kernel void ew_un<half>(device const half *, device half *, constant UnMeta &, uint);

template <typename T>
kernel void ew_bin(device const T *a [[buffer(0)]], device const T *b [[buffer(1)]], device T *out [[buffer(2)]],
                   constant BcMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint ao = bc_off(m.astr, gid, m.odims, m.ndim);
    uint bo = bc_off(m.bstr, gid, m.odims, m.ndim);
    out[gid] = (T)bin_op(m.op, (float)a[ao], (float)b[bo], m.alpha);
}
template [[host_name("ew_bin_f32")]] kernel void ew_bin<float>(device const float *, device const float *, device float *, constant BcMeta &, uint);
template [[host_name("ew_bin_f16")]] kernel void ew_bin<half>(device const half *, device const half *, device half *, constant BcMeta &, uint);

// integer add/sub/mul (exact) for int32 — separate so accumulation stays integral
kernel void ew_bin_i32(device const int *a [[buffer(0)]], device const int *b [[buffer(1)]], device int *out [[buffer(2)]],
                       constant BcMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint ao = bc_off(m.astr, gid, m.odims, m.ndim);
    uint bo = bc_off(m.bstr, gid, m.odims, m.ndim);
    int av = a[ao], bv = b[bo], al = (int)m.alpha;
    int r; switch (m.op) { case B_ADD: r = av + al * bv; break; case B_SUB: r = av - al * bv; break;
                           case B_MUL: r = av * bv; break; case B_MAX: r = av > bv ? av : bv; break;
                           case B_MIN: r = av < bv ? av : bv; break; default: r = av; }
    out[gid] = r;
}
