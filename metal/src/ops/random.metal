// RNG / distribution kernels (MSL). Per-element PCG stream seeded by (seed, gid) — reproducible and
// decorrelated; sufficient for the statistical (sample-mean / frequency) tolerances the tests use.
#include <metal_stdlib>
using namespace metal;

struct RngMeta { uint n; int op; uint seed; float p0; float p1; };

inline uint pcg(uint v) { uint s = v * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
inline float u01(thread uint &s) { s = s * 747796405u + 2891336453u; uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; w = (w >> 22u) ^ w; return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }
inline uint seed_of(uint seed, uint gid) { return pcg((seed * 2654435761u) ^ (gid * 40503u + 1u)); }
inline float gauss(thread uint &s) { float u1 = fmax(u01(s), 1e-7f), u2 = u01(s); return sqrt(-2.f * log(u1)) * cos(6.28318530718f * u2); }

// float distributions: 0 exp(lambda=p0), 1 geometric(p=p0), 2 normal(mean=p0,std=p1), 3 lognormal(p0,p1), 4 cauchy(med=p0,sig=p1)
kernel void rng_f32(device float *out [[buffer(0)]], constant RngMeta &m [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= m.n) return;
    uint s = seed_of(m.seed, i); float r;
    switch (m.op) {
        case 0: { float u = u01(s); r = -log(1.f - u) / m.p0; break; }
        case 1: { float u = u01(s); r = floor(log(1.f - u) / log(1.f - m.p0)) + 1.f; break; }
        case 2: r = m.p0 + m.p1 * gauss(s); break;
        case 3: r = exp(m.p0 + m.p1 * gauss(s)); break;
        default: { float u = u01(s); r = m.p0 + m.p1 * tan(3.14159265358979f * (u - 0.5f)); break; }
    }
    out[i] = r;
}
kernel void rng_randint(device long *out [[buffer(0)]], constant RngMeta &m [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= m.n) return;
    uint s = seed_of(m.seed, i); long lo = (long)m.p0, hi = (long)m.p1;
    out[i] = lo + (long)(u01(s) * (float)(hi - lo));
}
kernel void rng_poisson(device int *out [[buffer(0)]], constant RngMeta &m [[buffer(1)]], uint i [[thread_position_in_grid]]) {
    if (i >= m.n) return;
    uint s = seed_of(m.seed, i); float L = exp(-m.p0), p = 1.f; int k = 0;
    do { k++; p *= u01(s); } while (p > L);
    out[i] = k - 1;
}
kernel void rng_normal_tt(device float *out [[buffer(0)]], device const float *mean [[buffer(1)]], device const float *sd [[buffer(2)]],
                          constant RngMeta &m [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= m.n) return; uint s = seed_of(m.seed, i); out[i] = mean[i] + sd[i] * gauss(s);
}
// probs [C] (p0 = C); each output sample walks the CDF
kernel void rng_multinomial(device long *out [[buffer(0)]], device const float *probs [[buffer(1)]],
                            constant RngMeta &m [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= m.n) return;
    uint s = seed_of(m.seed, i); int C = (int)m.p0; float u = u01(s), acc = 0.f; long idx = C - 1;
    for (int c = 0; c < C; ++c) { acc += probs[c]; if (u < acc) { idx = c; break; } }
    out[i] = idx;
}
