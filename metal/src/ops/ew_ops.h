// Shared elementwise op codes — included by both elementwise.metal (MSL) and elementwise.mm (ObjC++).
// Keep in sync; plain #defines only (no language-specific constructs).
#ifndef EW_OPS_H
#define EW_OPS_H

// ---- unary (ew_un); param ops use p0/p1 ----
#define U_EXP 0
#define U_LOG 1
#define U_ABS 2
#define U_SQRT 3
#define U_RSQRT 4
#define U_RECIP 5
#define U_RELU 6
#define U_NEG 7
#define U_SIGMOID 8
#define U_TANH 9
#define U_ERF 10
#define U_GELU 11
#define U_SILU 12
#define U_SOFTPLUS 13
#define U_SIN 14
#define U_COS 15
#define U_TAN 16
#define U_ATAN 17
#define U_SIGN 18
#define U_FLOOR 19
#define U_CEIL 20
#define U_ROUND 21
#define U_TRUNC 22
#define U_SQUARE 23
#define U_SINH 24
#define U_COSH 25
#define U_ASIN 26
#define U_ACOS 27
#define U_ERFC 28
#define U_FRAC 29
#define U_LGAMMA 30
#define U_EXPM1 31
#define U_LOG1P 32
#define U_LOG2 33
#define U_LOG10 34
#define U_EXP2 35
#define U_ERFINV 36
#define U_MISH 37
#define U_HARDSWISH 38
#define U_HARDSIGMOID 39
#define U_LOGSIGMOID 40
#define U_SELU 41
#define U_TANHSHRINK 42
#define U_RELU6 43
#define U_ADDC 44        // x + p0
#define U_MULC 45        // x * p0
#define U_CLAMP_LO 46    // max(x, p0)
#define U_CLAMP_HI 47    // min(x, p0)
#define U_POWS 48        // pow(x, p0)
#define U_LEAKYRELU 49   // x>0 ? x : p0*x
#define U_ELU 50         // x>0 ? x : p0*(exp(x)-1)
#define U_CELU 51        // max(0,x) + min(0, p0*(exp(x/p0)-1))
#define U_HARDSHRINK 52  // |x|>p0 ? x : 0
#define U_SOFTSHRINK 53  // x>p0 ? x-p0 : (x<-p0 ? x+p0 : 0)
#define U_CLAMP_LOHI 54  // clamp(x, p0, p1)
#define U_THRESHOLD 55   // x>p0 ? x : p1

// ---- binary (ew_bin), broadcast; alpha applies to ADD/SUB/LERP ----
#define B_ADD 0          // a + alpha*b
#define B_SUB 1          // a - alpha*b
#define B_MUL 2
#define B_DIV 3
#define B_MAX 4
#define B_MIN 5
#define B_POW 6
#define B_FMOD 7
#define B_HYPOT 8
#define B_ATAN2 9
#define B_REMAINDER 10
#define B_XLOGY 11
#define B_LOGADDEXP 12
#define B_COPYSIGN 13
#define B_HEAVISIDE 14
#define B_PRELU 15       // a>0 ? a : b*a
#define B_LERP 16        // a + alpha*(b-a)

#endif
