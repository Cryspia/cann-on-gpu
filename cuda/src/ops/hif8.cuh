// HiF8 (Huawei HiFloat8, tapered-precision 8-bit) device-side encode/decode.
// CUDA has no corresponding type, so this is hand-written. Decode table and encoding rules are
// **bit-exact against the CANN reference library libnnopbase.so**
//   (golden generated offline via op::HiFloat8::Hifp8ToFloat/BitsFromFp32: 200k random +
//    all adjacent midpoints, 0 mismatches).
// Encoding rules: round to nearest representable value; ties round away from zero (take the larger code);
//   overflow handled via a virtual next-level 49152 (=2^15·1.5) → RNE threshold 40960 → inf;
//   zero is unsigned (-0 bit pattern 0x80 is NaN).
#ifndef CANN_ON_GPU_HIF8_CUH
#define CANN_ON_GPU_HIF8_CUH
#include <cstdint>

static __device__ __constant__ uint32_t HIF8_DEC[256] = {
#include "hif8_table.inc"
};

__device__ inline float hif8_decode(uint8_t b) {
    uint32_t u = HIF8_DEC[b];
    float f; __builtin_memcpy(&f, &u, 4); return f;
}

__device__ inline uint8_t hif8_encode(float x) {
    uint32_t xb; __builtin_memcpy(&xb, &x, 4);
    if ((xb & 0x7fffffffu) > 0x7f800000u) return 0x80;          // NaN → NaN code
    uint8_t sign = (xb >> 31) & 1;
    float ax = fabsf(x);
    if (isinf(ax)) return sign ? 0xef : 0x6f;
    int best = 0; float bestd = 1e30f;
    for (int c = 0; c <= 0x7f; c++) {                            // all finite positive codes (0x6f uses the virtual next-level value)
        uint32_t u = HIF8_DEC[c]; float v; __builtin_memcpy(&v, &u, 4);
        if (c == 0x6f) v = 49152.0f;
        float d = fabsf(v - ax);
        if (d <= bestd) { bestd = d; best = c; }                // ties take the larger code = round away from zero
    }
    if (best == 0) return 0x00;                                  // zero is unsigned
    return (uint8_t)((sign << 7) | best);
}

#endif
