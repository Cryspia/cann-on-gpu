// fp4 / fp6 (sub-byte) device encode/decode. CUDA's __nv_fp4/fp6 rounding differs from Ascend, so
// these are hand-written.
// Decode tables and encode boundaries are **bit-exact against the CANN reference library
// libnnopbase.so (Float4E2M1/E1M2, Float6E2M3/E3M2)**:
//   golden values generated offline via dlopen of the reference library; 50k random samples per
//   format with 0 mismatches (boundaries extracted by bisection to fp32 resolution).
// Encoding: locate the magnitude code via boundary table (Ascend subnormal rounding is non-standard
// RNE; boundaries come directly from the reference), then attach the sign bit (including −0).
// No inf/nan; out-of-range values saturate to the maximum representable magnitude.
// Storage convention: fp4 = 2 elements per byte (element 0 = low nibble, 1 = high nibble);
//   fp6 = 1 byte per element (6-bit code in the low 6 bits, consistent with CUDA __nv_fp6 storage;
//   the true Ascend 4-in-3 packing is not replicated, which does not affect functional/tolerance checks).
#ifndef CANN_ON_GPU_SUBFP_CUH
#define CANN_ON_GPU_SUBFP_CUH
#include <cstdint>

// ---- Decode tables (fp32 bit patterns, full code range including signed half) ----
static __device__ __constant__ uint32_t FP4E2M1_DEC[16] = {
    0x00000000u,0x3f000000u,0x3f800000u,0x3fc00000u,0x40000000u,0x40400000u,0x40800000u,0x40c00000u,
    0x80000000u,0xbf000000u,0xbf800000u,0xbfc00000u,0xc0000000u,0xc0400000u,0xc0800000u,0xc0c00000u};
static __device__ __constant__ uint32_t FP4E1M2_DEC[16] = {
    0x00000000u,0x3e800000u,0x3f000000u,0x3f400000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,
    0x80000000u,0xbe800000u,0xbf000000u,0xbf400000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u};
static __device__ __constant__ uint32_t FP6E2M3_DEC[64] = {
    0x00000000u,0x3e000000u,0x3e800000u,0x3ec00000u,0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,
    0x3f800000u,0x3f900000u,0x3fa00000u,0x3fb00000u,0x3fc00000u,0x3fd00000u,0x3fe00000u,0x3ff00000u,
    0x40000000u,0x40100000u,0x40200000u,0x40300000u,0x40400000u,0x40500000u,0x40600000u,0x40700000u,
    0x40800000u,0x40900000u,0x40a00000u,0x40b00000u,0x40c00000u,0x40d00000u,0x40e00000u,0x40f00000u,
    0x80000000u,0xbe000000u,0xbe800000u,0xbec00000u,0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,
    0xbf800000u,0xbf900000u,0xbfa00000u,0xbfb00000u,0xbfc00000u,0xbfd00000u,0xbfe00000u,0xbff00000u,
    0xc0000000u,0xc0100000u,0xc0200000u,0xc0300000u,0xc0400000u,0xc0500000u,0xc0600000u,0xc0700000u,
    0xc0800000u,0xc0900000u,0xc0a00000u,0xc0b00000u,0xc0c00000u,0xc0d00000u,0xc0e00000u,0xc0f00000u};
static __device__ __constant__ uint32_t FP6E3M2_DEC[64] = {
    0x00000000u,0x3d800000u,0x3e000000u,0x3e400000u,0x3e800000u,0x3ea00000u,0x3ec00000u,0x3ee00000u,
    0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,
    0x40000000u,0x40200000u,0x40400000u,0x40600000u,0x40800000u,0x40a00000u,0x40c00000u,0x40e00000u,
    0x41000000u,0x41200000u,0x41400000u,0x41600000u,0x41800000u,0x41a00000u,0x41c00000u,0x41e00000u,
    0x80000000u,0xbd800000u,0xbe000000u,0xbe400000u,0xbe800000u,0xbea00000u,0xbec00000u,0xbee00000u,
    0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u,
    0xc0000000u,0xc0200000u,0xc0400000u,0xc0600000u,0xc0800000u,0xc0a00000u,0xc0c00000u,0xc0e00000u,
    0xc1000000u,0xc1200000u,0xc1400000u,0xc1600000u,0xc1800000u,0xc1a00000u,0xc1c00000u,0xc1e00000u};

// ---- Encode boundaries (minimum |x| to select magnitude code c; index 1..half-1) ----
static __device__ __constant__ float FP4E2M1_B[8] = {0,0.4375f,0.875f,1.25000012f,1.75f,2.50000024f,3.5f,5.00000048f};
static __device__ __constant__ float FP4E1M2_B[8] = {0,0.234375f,0.46875f,0.6875f,0.9375f,1.12500012f,1.375f,1.62500012f};
static __device__ __constant__ float FP6E2M3_B[32] = {0,0.12109375f,0.2421875f,0.359375f,0.484375f,0.59375f,0.71875f,0.84375f,0.96875f,1.06250012f,1.1875f,1.31250012f,1.4375f,1.56250012f,1.6875f,1.81250012f,1.9375f,2.12500024f,2.375f,2.62500024f,2.875f,3.12500024f,3.375f,3.62500024f,3.875f,4.25000048f,4.75f,5.25000048f,5.75f,6.25000048f,6.75f,7.25000048f};
static __device__ __constant__ float FP6E3M2_B[32] = {0,0.05859375f,0.1171875f,0.171875f,0.234375f,0.28125003f,0.34375f,0.40625003f,0.46875f,0.56250006f,0.6875f,0.81250006f,0.9375f,1.12500012f,1.375f,1.62500012f,1.875f,2.25000024f,2.75f,3.25000024f,3.75f,4.50000048f,5.5f,6.50000048f,7.5f,9.00000095f,11.f,13.000001f,15.f,18.0000019f,22.f,26.0000019f};

enum SubFpKind { SF_FP4E2M1, SF_FP4E1M2, SF_FP6E2M3, SF_FP6E3M2 };

__device__ inline const uint32_t *sf_dec(int k) {
    return k == SF_FP4E2M1 ? FP4E2M1_DEC : k == SF_FP4E1M2 ? FP4E1M2_DEC : k == SF_FP6E2M3 ? FP6E2M3_DEC : FP6E3M2_DEC;
}
__device__ inline const float *sf_bnd(int k) {
    return k == SF_FP4E2M1 ? FP4E2M1_B : k == SF_FP4E1M2 ? FP4E1M2_B : k == SF_FP6E2M3 ? FP6E2M3_B : FP6E3M2_B;
}
__device__ inline int sf_half(int k) { return (k == SF_FP4E2M1 || k == SF_FP4E1M2) ? 8 : 32; }

__device__ inline float subfp_decode(int k, uint8_t code) {
    uint32_t u = sf_dec(k)[code]; float f; __builtin_memcpy(&f, &u, 4); return f;
}
__device__ inline uint8_t subfp_encode(int k, float x) {
    uint32_t xb; __builtin_memcpy(&xb, &x, 4);
    uint8_t sign = (xb >> 31) & 1; float ax = fabsf(x);
    const float *B = sf_bnd(k); int half = sf_half(k);
    if (xb == 0x7fc00000u || (xb & 0x7fffffffu) > 0x7f800000u) return 0;   // NaN → 0 (fp4/fp6 have no NaN)
    int c = 0;
    for (int i = 1; i < half; i++) { if (ax >= B[i]) c = i; else break; }
    return (uint8_t)(sign ? (c | half) : c);                              // includes −0
}

#endif
