// Gated-linear-unit kernels (MSL). SwiGlu: split last dim in half [a|b], out = silu(a) * b.
#include <metal_stdlib>
using namespace metal;

struct GluMeta { uint n; uint D; };   // n = output elements (rows*D), D = half (output) dim

template <typename T>
kernel void swiglu(device const T *in [[buffer(0)]], device T *out [[buffer(1)]],
                   constant GluMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint row = gid / m.D, j = gid % m.D;
    float a = (float)in[row * 2 * m.D + j], b = (float)in[row * 2 * m.D + m.D + j];
    out[gid] = (T)((a / (1.f + exp(-a))) * b);
}
template [[host_name("swiglu_f32")]] kernel void swiglu<float>(device const float *, device float *, constant GluMeta &, uint);
template [[host_name("swiglu_f16")]] kernel void swiglu<half>(device const half *, device half *, constant GluMeta &, uint);
