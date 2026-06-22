// Sub-byte / fp8 codec (host C++) — bit-exact with the CANN reference (and the test oracle). Shared by
// cast.mm (fp32<->subfp) and elementwise.mm (subfp Add). Encoding/decoding only; packing handled by callers.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include "acl/acl.h"

namespace subfp {

// ---- fp8 e4m3 (1-4-3, bias 7, no inf) ----
inline uint8_t enc_e4m3(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1; int32_t e = (int32_t)((x >> 23) & 0xFF) - 127; uint32_t m = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint8_t)((sign << 7) | 0x7F);
    int32_t E = e + 7;
    if (E >= 15) return (uint8_t)((sign << 7) | 0x7E);
    if (E <= 0) { int32_t shift = 1 - E; if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t full = m | 0x800000, q = full >> (23 - 3 + shift);
        uint32_t rem = full & ((1u << (23 - 3 + shift)) - 1), half = 1u << (23 - 3 + shift - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return (uint8_t)((sign << 7) | (q & 0x7F)); }
    uint32_t mant = m >> (23 - 3), rem = m & ((1u << (23 - 3)) - 1), half = 1u << (23 - 3 - 1), out = (E << 3) | mant;
    if (rem > half || (rem == half && (mant & 1))) out++;
    if (out >= 0x7F) out = 0x7E;
    return (uint8_t)((sign << 7) | (out & 0x7F));
}
inline float dec_e4m3(uint8_t v) {
    uint32_t sign = (v >> 7) & 1, E = (v >> 3) & 0xF, m = v & 0x7;
    if (E == 0xF && m == 0x7) return sign ? -__builtin_nanf("") : __builtin_nanf("");
    float r = (E == 0) ? m * 0x1p-3f * 0x1p-6f : (1.f + m / 8.f) * __builtin_powf(2.f, (int)E - 7);
    return sign ? -r : r;
}
// ---- fp8 e5m2 (1-5-2, bias 15, inf/nan) ----
inline uint8_t enc_e5m2(float f) {
    uint32_t x; memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1; int32_t E = (int32_t)((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint8_t)((sign << 7) | (m ? 0x7F : 0x7C));
    if (E >= 31) return (uint8_t)((sign << 7) | 0x7C);
    if (E <= 0) { int32_t shift = 1 - E; if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t full = m | 0x800000, q = full >> (23 - 2 + shift);
        uint32_t rem = full & ((1u << (23 - 2 + shift)) - 1), half = 1u << (23 - 2 + shift - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return (uint8_t)((sign << 7) | (q & 0x7F)); }
    uint32_t mant = m >> (23 - 2), rem = m & ((1u << (23 - 2)) - 1), half = 1u << (23 - 2 - 1), out = (E << 2) | mant;
    if (rem > half || (rem == half && (mant & 1))) out++;
    if (out >= 0x7C) out = 0x7C;
    return (uint8_t)((sign << 7) | (out & 0x7F));
}
inline float dec_e5m2(uint8_t v) {
    uint32_t sign = (v >> 7) & 1, E = (v >> 2) & 0x1F, m = v & 0x3;
    if (E == 0x1F) return m ? __builtin_nanf("") : (sign ? -__builtin_inff() : __builtin_inff());
    float r = (E == 0) ? m * 0x1p-2f * 0x1p-14f : (1.f + m / 4.f) * __builtin_powf(2.f, (int)E - 15);
    return sign ? -r : r;
}
// ---- HiF8 (table) ----
static const uint32_t HIF8_DEC[256] = {
#include "hif8_table.inc"
};
inline float hif8_dec(uint8_t b) { float v; uint32_t u = HIF8_DEC[b]; memcpy(&v, &u, 4); return v; }
inline uint8_t hif8_enc(float x) {
    uint32_t xb; memcpy(&xb, &x, 4);
    if ((xb & 0x7fffffffu) > 0x7f800000u) return 0x80;
    uint8_t sign = (xb >> 31) & 1; float ax = std::fabs(x);
    if (std::isinf(ax)) return sign ? 0xef : 0x6f;
    int best = 0; float bestd = 1e30f;
    for (int c = 0; c <= 0x7f; c++) { float v = (c == 0x6f) ? 49152.f : hif8_dec(c); float d = std::fabs(v - ax); if (d <= bestd) { bestd = d; best = c; } }
    if (best == 0) return 0x00;
    return (uint8_t)((sign << 7) | best);
}
// ---- fp4 / fp6 (tables + encode boundaries) ----
static const uint32_t D_4E2M1[16] = {0x00000000u,0x3f000000u,0x3f800000u,0x3fc00000u,0x40000000u,0x40400000u,0x40800000u,0x40c00000u,0x80000000u,0xbf000000u,0xbf800000u,0xbfc00000u,0xc0000000u,0xc0400000u,0xc0800000u,0xc0c00000u};
static const uint32_t D_4E1M2[16] = {0x00000000u,0x3e800000u,0x3f000000u,0x3f400000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,0x80000000u,0xbe800000u,0xbf000000u,0xbf400000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u};
static const float B_4E2M1[8] = {0,0.4375f,0.875f,1.25000012f,1.75f,2.50000024f,3.5f,5.00000048f};
static const float B_4E1M2[8] = {0,0.234375f,0.46875f,0.6875f,0.9375f,1.12500012f,1.375f,1.62500012f};
static const uint32_t D_6E2M3[64] = {0x00000000u,0x3e000000u,0x3e800000u,0x3ec00000u,0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,0x3f800000u,0x3f900000u,0x3fa00000u,0x3fb00000u,0x3fc00000u,0x3fd00000u,0x3fe00000u,0x3ff00000u,0x40000000u,0x40100000u,0x40200000u,0x40300000u,0x40400000u,0x40500000u,0x40600000u,0x40700000u,0x40800000u,0x40900000u,0x40a00000u,0x40b00000u,0x40c00000u,0x40d00000u,0x40e00000u,0x40f00000u,0x80000000u,0xbe000000u,0xbe800000u,0xbec00000u,0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,0xbf800000u,0xbf900000u,0xbfa00000u,0xbfb00000u,0xbfc00000u,0xbfd00000u,0xbfe00000u,0xbff00000u,0xc0000000u,0xc0100000u,0xc0200000u,0xc0300000u,0xc0400000u,0xc0500000u,0xc0600000u,0xc0700000u,0xc0800000u,0xc0900000u,0xc0a00000u,0xc0b00000u,0xc0c00000u,0xc0d00000u,0xc0e00000u,0xc0f00000u};
static const uint32_t D_6E3M2[64] = {0x00000000u,0x3d800000u,0x3e000000u,0x3e400000u,0x3e800000u,0x3ea00000u,0x3ec00000u,0x3ee00000u,0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,0x40000000u,0x40200000u,0x40400000u,0x40600000u,0x40800000u,0x40a00000u,0x40c00000u,0x40e00000u,0x41000000u,0x41200000u,0x41400000u,0x41600000u,0x41800000u,0x41a00000u,0x41c00000u,0x41e00000u,0x80000000u,0xbd800000u,0xbe000000u,0xbe400000u,0xbe800000u,0xbea00000u,0xbec00000u,0xbee00000u,0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u,0xc0000000u,0xc0200000u,0xc0400000u,0xc0600000u,0xc0800000u,0xc0a00000u,0xc0c00000u,0xc0e00000u,0xc1000000u,0xc1200000u,0xc1400000u,0xc1600000u,0xc1800000u,0xc1a00000u,0xc1c00000u,0xc1e00000u};
static const float B_6E2M3[32] = {0,0.12109375f,0.2421875f,0.359375f,0.484375f,0.59375f,0.71875f,0.84375f,0.96875f,1.06250012f,1.1875f,1.31250012f,1.4375f,1.56250012f,1.6875f,1.81250012f,1.9375f,2.12500024f,2.375f,2.62500024f,2.875f,3.12500024f,3.375f,3.62500024f,3.875f,4.25000048f,4.75f,5.25000048f,5.75f,6.25000048f,6.75f,7.25000048f};
static const float B_6E3M2[32] = {0,0.05859375f,0.1171875f,0.171875f,0.234375f,0.28125003f,0.34375f,0.40625003f,0.46875f,0.56250006f,0.6875f,0.81250006f,0.9375f,1.12500012f,1.375f,1.62500012f,1.875f,2.25000024f,2.75f,3.25000024f,3.75f,4.50000048f,5.5f,6.50000048f,7.5f,9.00000095f,11.f,13.000001f,15.f,18.0000019f,22.f,26.0000019f};

inline void subtbl(aclDataType dt, const uint32_t **dec, const float **bnd, int *half) {
    switch (dt) {
        case ACL_FLOAT4_E2M1: *dec = D_4E2M1; *bnd = B_4E2M1; *half = 8; break;
        case ACL_FLOAT4_E1M2: *dec = D_4E1M2; *bnd = B_4E1M2; *half = 8; break;
        case ACL_FLOAT6_E2M3: *dec = D_6E2M3; *bnd = B_6E2M3; *half = 32; break;
        default:              *dec = D_6E3M2; *bnd = B_6E3M2; *half = 32; break;   // ACL_FLOAT6_E3M2
    }
}
inline uint8_t sub_enc(aclDataType dt, float x) {
    const uint32_t *dec; const float *bnd; int half; subtbl(dt, &dec, &bnd, &half);
    uint32_t xb; memcpy(&xb, &x, 4); uint8_t s = (xb >> 31) & 1; float ax = std::fabs(x);
    int c = 0; for (int i = 1; i < half; ++i) { if (ax >= bnd[i]) c = i; else break; }
    return (uint8_t)(s ? (c | half) : c);
}
inline float sub_dec(aclDataType dt, uint8_t c) {
    const uint32_t *dec; const float *bnd; int half; subtbl(dt, &dec, &bnd, &half);
    float v; uint32_t u = dec[c & (dt == ACL_FLOAT4_E2M1 || dt == ACL_FLOAT4_E1M2 ? 0xf : 0x3f)]; memcpy(&v, &u, 4); return v;
}

inline bool is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool is_fp6(aclDataType t) { return t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline bool is_fp8(aclDataType t) { return t == ACL_FLOAT8_E4M3FN || t == ACL_FLOAT8_E5M2 || t == ACL_HIFLOAT8; }
inline bool is_low(aclDataType t) { return is_fp4(t) || is_fp6(t) || is_fp8(t); }

// scalar fp8/hif8 byte codec
inline uint8_t enc8(aclDataType dt, float v) { return dt == ACL_FLOAT8_E4M3FN ? enc_e4m3(v) : dt == ACL_FLOAT8_E5M2 ? enc_e5m2(v) : hif8_enc(v); }
inline float dec8(aclDataType dt, uint8_t b) { return dt == ACL_FLOAT8_E4M3FN ? dec_e4m3(b) : dt == ACL_FLOAT8_E5M2 ? dec_e5m2(b) : hif8_dec(b); }

// read element i from a low-precision buffer as float (handles fp4 nibble packing)
inline float load(aclDataType dt, const uint8_t *base, int64_t i) {
    if (is_fp4(dt)) { uint8_t c = (i & 1) ? (base[i / 2] >> 4) : (base[i / 2] & 0xf); return sub_dec(dt, c); }
    if (is_fp6(dt)) return sub_dec(dt, base[i] & 0x3f);
    return dec8(dt, base[i]);
}
// store float as element i into a low-precision buffer (fp4 packs into nibbles; caller must zero buffer first for fp4)
inline void store(aclDataType dt, uint8_t *base, int64_t i, float v) {
    if (is_fp4(dt)) { uint8_t c = sub_enc(dt, v) & 0xf; if (i & 1) base[i / 2] = (base[i / 2] & 0x0f) | (c << 4); else base[i / 2] = (base[i / 2] & 0xf0) | c; return; }
    if (is_fp6(dt)) { base[i] = sub_enc(dt, v) & 0x3f; return; }
    base[i] = enc8(dt, v);
}
inline int64_t bytes(aclDataType dt, int64_t n) { return is_fp4(dt) ? (n + 1) / 2 : n; }   // fp6/fp8 = 1 byte/elem

} // namespace subfp
