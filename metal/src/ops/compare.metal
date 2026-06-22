// Comparison / logical / predicate / masked-fill kernels (bool output uchar 0/1), MSL.
// Tensor compares + logical-and/or support general broadcasting via per-input strides (stride 0 = broadcast).
// Scalar/unary/predicate/masked-fill assume a contiguous self (the shapes the tests exercise).
#include <metal_stdlib>
using namespace metal;

struct EwMeta  { uint n; uint ndim; int op; uint odims[8]; uint astr[8]; uint bstr[8]; };
struct EwMetaS { uint n; int op; float s; };

// op codes (must match compare.mm): 0 gt, 1 lt, 2 eq, 3 ne, 4 ge, 5 le
template <typename T> inline bool cmp_op(int op, T a, T b) {
    switch (op) { case 0: return a > b; case 1: return a < b; case 2: return a == b;
                  case 3: return a != b; case 4: return a >= b; default: return a <= b; }
}

inline uint bcast_off(constant uint *str, uint gid, constant uint *odims, uint ndim) {
    uint rem = gid, off = 0;
    for (int d = int(ndim) - 1; d >= 0; --d) { uint id = rem % odims[d]; rem /= odims[d]; off += id * str[d]; }
    return off;
}

template <typename T>
kernel void cmp_t(device const T *a [[buffer(0)]], device const T *b [[buffer(1)]],
                  device uchar *out [[buffer(2)]], constant EwMeta &m [[buffer(3)]],
                  uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint ao = bcast_off(m.astr, gid, m.odims, m.ndim);
    uint bo = bcast_off(m.bstr, gid, m.odims, m.ndim);
    out[gid] = cmp_op<T>(m.op, a[ao], b[bo]) ? 1 : 0;
}
template [[host_name("cmp_f32")]] kernel void cmp_t<float>(device const float *, device const float *, device uchar *, constant EwMeta &, uint);
template [[host_name("cmp_f16")]] kernel void cmp_t<half>(device const half *, device const half *, device uchar *, constant EwMeta &, uint);
template [[host_name("cmp_i32")]] kernel void cmp_t<int>(device const int *, device const int *, device uchar *, constant EwMeta &, uint);

// logical and/or on bool (uint8), broadcast. op: 0 and, 1 or
kernel void logic_b(device const uchar *a [[buffer(0)]], device const uchar *b [[buffer(1)]],
                    device uchar *out [[buffer(2)]], constant EwMeta &m [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    bool av = a[bcast_off(m.astr, gid, m.odims, m.ndim)] != 0;
    bool bv = b[bcast_off(m.bstr, gid, m.odims, m.ndim)] != 0;
    out[gid] = (m.op == 0 ? (av && bv) : (av || bv)) ? 1 : 0;
}

// scalar compare (contiguous self): self <cmp> m.s
template <typename T>
kernel void cmps_t(device const T *a [[buffer(0)]], device uchar *out [[buffer(1)]],
                   constant EwMetaS &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = cmp_op<T>(m.op, a[gid], (T)m.s) ? 1 : 0;
}
template [[host_name("cmps_f32")]] kernel void cmps_t<float>(device const float *, device uchar *, constant EwMetaS &, uint);
template [[host_name("cmps_f16")]] kernel void cmps_t<half>(device const half *, device uchar *, constant EwMetaS &, uint);
template [[host_name("cmps_i32")]] kernel void cmps_t<int>(device const int *, device uchar *, constant EwMetaS &, uint);

// logical not (contiguous bool)
kernel void lnot_b(device const uchar *a [[buffer(0)]], device uchar *out [[buffer(1)]],
                   constant EwMetaS &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = a[gid] != 0 ? 0 : 1;
}

// predicates (contiguous). op: 0 isnan, 1 isfinite, 2 isinf, 3 isposinf, 4 isneginf
template <typename T>
kernel void pred_t(device const T *a [[buffer(0)]], device uchar *out [[buffer(1)]],
                   constant EwMetaS &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    float v = (float)a[gid]; bool r;
    switch (m.op) { case 0: r = isnan(v); break; case 1: r = isfinite(v); break; case 2: r = isinf(v); break;
                    case 3: r = isinf(v) && v > 0; break; default: r = isinf(v) && v < 0; }
    out[gid] = r ? 1 : 0;
}
template [[host_name("pred_f32")]] kernel void pred_t<float>(device const float *, device uchar *, constant EwMetaS &, uint);
template [[host_name("pred_f16")]] kernel void pred_t<half>(device const half *, device uchar *, constant EwMetaS &, uint);

// masked fill (contiguous): out = mask ? m.s : self
template <typename T>
kernel void maskfill_t(device const T *a [[buffer(0)]], device const uchar *mask [[buffer(1)]],
                       device T *out [[buffer(2)]], constant EwMetaS &m [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    out[gid] = mask[gid] != 0 ? (T)m.s : a[gid];
}
template [[host_name("maskfill_f32")]] kernel void maskfill_t<float>(device const float *, device const uchar *, device float *, constant EwMetaS &, uint);
template [[host_name("maskfill_f16")]] kernel void maskfill_t<half>(device const half *, device const uchar *, device half *, constant EwMetaS &, uint);
template [[host_name("maskfill_i32")]] kernel void maskfill_t<int>(device const int *, device const uchar *, device int *, constant EwMetaS &, uint);
