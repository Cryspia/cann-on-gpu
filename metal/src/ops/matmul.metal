// Matmul post-processing: optional bias broadcast (over the N dimension) + optional activation.
// The GEMM itself is done by MPSMatrixMultiplication (matmul.mm); this fuses the epilogue.
#include <metal_stdlib>
using namespace metal;

struct MMPost { uint M; uint N; int act; int hasBias; };   // act: 0 none, 1 relu, 2 gelu, 3 silu

template <typename T>
kernel void mm_post(device T *c [[buffer(0)]], device const T *bias [[buffer(1)]],
                    constant MMPost &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.M * m.N) return;
    float v = (float)c[gid];
    if (m.hasBias) v += (float)bias[gid % m.N];
    switch (m.act) {
        case 1: v = v > 0 ? v : 0; break;                                   // relu
        case 2: { float t = 0.7978845608f * (v + 0.044715f * v * v * v);    // gelu (tanh approx; MSL has no erfc)
                  v = 0.5f * v * (1.f + tanh(t)); break; }
        case 3: v = v / (1.f + exp(-v)); break;                             // silu
        default: break;
    }
    c[gid] = (T)v;
}
template [[host_name("mm_post_f32")]] kernel void mm_post<float>(device float *, device const float *, constant MMPost &, uint);
template [[host_name("mm_post_f16")]] kernel void mm_post<half>(device half *, device const half *, constant MMPost &, uint);

// W8A8 per-channel dequant epilogue: o[m,n] = raw[m,n] * scale[n] (raw = int8·int8 fp32 sums from MPS),
// optionally accumulated into o. scale is fp16 (CANN-canonical deqScale) or fp32.
struct QScale { uint M; uint N; int accumulate; };
template <typename ST>
kernel void qscale(device float *o [[buffer(0)]], device const float *raw [[buffer(1)]],
                   device const ST *scale [[buffer(2)]], constant QScale &m [[buffer(3)]],
                   uint gid [[thread_position_in_grid]]) {
    if (gid >= m.M * m.N) return;
    float v = raw[gid] * (float)scale[gid % m.N];
    o[gid] = m.accumulate ? o[gid] + v : v;
}
template [[host_name("qscale_f32")]] kernel void qscale<float>(device float *, device const float *, device const float *, constant QScale &, uint);
template [[host_name("qscale_f16")]] kernel void qscale<half>(device float *, device const float *, device const half *, constant QScale &, uint);
