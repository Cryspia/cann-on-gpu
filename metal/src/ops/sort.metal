// Scan kernels (MSL): cumsum/cumprod/cummax/cummin/logcumsumexp along a dim.
// Tensor viewed [outer, D, inner]; one thread per (outer,inner) group scans the D elements at stride `inner`.
#include <metal_stdlib>
using namespace metal;

struct ScanMeta { uint groups; uint D; uint inner; int op; };   // 0 sum,1 prod,2 max,3 min,4 logcumsumexp

kernel void scan_val(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                     constant ScanMeta &m [[buffer(2)]], uint g [[thread_position_in_grid]]) {
    if (g >= m.groups) return;
    uint base = (g / m.inner) * m.D * m.inner + (g % m.inner);
    if (m.op == 0) { float acc = 0.f; for (uint j = 0; j < m.D; ++j) { acc += in[base + j * m.inner]; out[base + j * m.inner] = acc; } }
    else if (m.op == 1) { float acc = 1.f; for (uint j = 0; j < m.D; ++j) { acc *= in[base + j * m.inner]; out[base + j * m.inner] = acc; } }
    else { float mx = 0.f, se = 0.f;   // logcumsumexp, numerically stable online
        for (uint j = 0; j < m.D; ++j) { float v = in[base + j * m.inner];
            if (j == 0) { mx = v; se = 1.f; } else if (v > mx) { se = se * exp(mx - v) + 1.f; mx = v; } else se += exp(v - mx);
            out[base + j * m.inner] = mx + log(se); }
    }
}

kernel void scan_arg(device const float *in [[buffer(0)]], device float *outv [[buffer(1)]], device long *outi [[buffer(2)]],
                     constant ScanMeta &m [[buffer(3)]], uint g [[thread_position_in_grid]]) {
    if (g >= m.groups) return;
    uint base = (g / m.inner) * m.D * m.inner + (g % m.inner);
    float r = in[base]; long idx = 0;
    for (uint j = 0; j < m.D; ++j) {
        float v = in[base + j * m.inner];
        if (j > 0) { if (m.op == 2 ? (v > r) : (v < r)) { r = v; idx = (long)j; } }   // 2 cummax, 3 cummin (strict; ties keep earlier)
        outv[base + j * m.inner] = r; outi[base + j * m.inner] = idx;
    }
}
