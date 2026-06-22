// Dtype conversion kernels (MSL). Numeric casts via static_cast; to-bool via (x != 0).
#include <metal_stdlib>
using namespace metal;

struct CastMeta { uint n; };

template <typename S, typename D>
kernel void cast_k(device const S *in [[buffer(0)]], device D *out [[buffer(1)]],
                   constant CastMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = (D)in[gid];
}
#define CAST(NAME, S, D) template [[host_name(NAME)]] kernel void cast_k<S, D>(device const S *, device D *, constant CastMeta &, uint);
CAST("cast_f32_f16", float, half)  CAST("cast_f16_f32", half, float)
CAST("cast_f32_i32", float, int)   CAST("cast_i32_f32", int, float)
CAST("cast_f32_i64", float, long)  CAST("cast_i64_f32", long, float)
CAST("cast_f16_i32", half, int)    CAST("cast_i32_f16", int, half)
CAST("cast_i32_i64", int, long)    CAST("cast_i64_i32", long, int)
CAST("cast_f16_i64", half, long)   CAST("cast_i64_f16", long, half)
CAST("cast_b8_f32", uchar, float)  CAST("cast_b8_i32", uchar, int)   CAST("cast_b8_f16", uchar, half)

// Sub-byte (fp8 e4m3/e5m2/HiF8, fp4, fp6) decode -> fp16 via a host-built fp32-bit-pattern table (filled by
// the subfp codec, so it is bit-exact with the host decode). `packed`=1 for fp4 (2 nibbles/byte); `mask`
// selects the code bits (0xff fp8, 0x3f fp6, 0xf fp4). Every code is exact in fp16, so (half) cast is lossless.
struct DecMeta { uint n; uint packed; uint mask; };
kernel void decode_sub_f16(device const uchar *in [[buffer(0)]], device half *out [[buffer(1)]],
                           constant DecMeta &m [[buffer(2)]], device const uint *tbl [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint code = m.packed ? ((gid & 1u) ? (uint)(in[gid >> 1] >> 4) : (uint)(in[gid >> 1] & 0xfu))
                         : ((uint)in[gid] & m.mask);
    out[gid] = (half)as_type<float>(tbl[code]);
}

// bf16 widen/narrow as pure bit manipulation (no native bfloat dependency); matches the host mm_bf2f/mm_f2bf.
// bf16 (stored as ushort) -> f32: place the 16 bits in the top half of a float.
kernel void widen_bf16_f32(device const ushort *in [[buffer(0)]], device float *out [[buffer(1)]],
                           constant CastMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = as_type<float>((uint)in[gid] << 16);
}
// f32 -> bf16 (round to nearest even): add the rounding bias, keep the top 16 bits.
kernel void narrow_f32_bf16(device const float *in [[buffer(0)]], device ushort *out [[buffer(1)]],
                            constant CastMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint x = as_type<uint>(in[gid]);
    x += 0x7FFFu + ((x >> 16) & 1u);
    out[gid] = (ushort)(x >> 16);
}

template <typename S>
kernel void cast_tobool(device const S *in [[buffer(0)]], device uchar *out [[buffer(1)]],
                        constant CastMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = in[gid] != (S)0 ? 1 : 0;
}
template [[host_name("cast_f32_b8")]] kernel void cast_tobool<float>(device const float *, device uchar *, constant CastMeta &, uint);
template [[host_name("cast_f16_b8")]] kernel void cast_tobool<half>(device const half *, device uchar *, constant CastMeta &, uint);
template [[host_name("cast_i32_b8")]] kernel void cast_tobool<int>(device const int *, device uchar *, constant CastMeta &, uint);
