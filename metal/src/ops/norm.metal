// Normalization kernels (MSL): RmsNorm over the last `cols` dims.
// y = x * rsqrt(mean(x^2) + eps) * gamma. One thread per row (cols looped twice).
#include <metal_stdlib>
using namespace metal;

struct RmsMeta { uint rows; uint cols; float eps; };

template <typename T>
kernel void rmsnorm(device const T *x [[buffer(0)]], device const T *g [[buffer(1)]], device T *y [[buffer(2)]],
                    constant RmsMeta &m [[buffer(3)]], uint row [[thread_position_in_grid]]) {
    if (row >= m.rows) return;
    uint base = row * m.cols;
    float ss = 0.f;
    for (uint j = 0; j < m.cols; ++j) { float v = (float)x[base + j]; ss += v * v; }
    float inv = rsqrt(ss / (float)m.cols + m.eps);
    for (uint j = 0; j < m.cols; ++j) y[base + j] = (T)((float)x[base + j] * inv * (float)g[j]);
}
template [[host_name("rmsnorm_f32")]] kernel void rmsnorm<float>(device const float *, device const float *, device float *, constant RmsMeta &, uint);
template [[host_name("rmsnorm_f16")]] kernel void rmsnorm<half>(device const half *, device const half *, device half *, constant RmsMeta &, uint);
