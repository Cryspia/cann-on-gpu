// Softmax / LogSoftmax over an arbitrary dim. Tensor viewed as [outer, D, inner]; one thread per
// (outer,inner) group reduces the D elements at stride `inner`. Numerically stable (subtract max).
#include <metal_stdlib>
using namespace metal;

struct SoftMeta { uint groups; uint D; uint inner; int isLog; };

template <typename T>
kernel void softmax_k(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
                      constant SoftMeta &m [[buffer(2)]], uint g [[thread_position_in_grid]]) {
    if (g >= m.groups) return;
    uint o = g / m.inner, ii = g % m.inner;
    uint base = o * m.D * m.inner + ii;
    float mx = -INFINITY;
    for (uint j = 0; j < m.D; ++j) mx = fmax(mx, (float)x[base + j * m.inner]);
    float sum = 0.f;
    for (uint j = 0; j < m.D; ++j) sum += exp((float)x[base + j * m.inner] - mx);
    float lsum = log(sum);
    for (uint j = 0; j < m.D; ++j) {
        float z = (float)x[base + j * m.inner] - mx;
        out[base + j * m.inner] = (T)(m.isLog ? (z - lsum) : (exp(z) / sum));
    }
}
template [[host_name("softmax_f32")]] kernel void softmax_k<float>(device const float *, device float *, constant SoftMeta &, uint);
template [[host_name("softmax_f16")]] kernel void softmax_k<half>(device const half *, device half *, constant SoftMeta &, uint);
