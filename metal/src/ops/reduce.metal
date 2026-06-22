// Reduction kernels (MSL). Reduce over an arbitrary axis set; op = sum/mean/max/min/prod.
// One thread per output element; Kahan-compensated sum/mean so large fp32 reductions hold ~1e-5.
#include <metal_stdlib>
using namespace metal;

#define R_SUM 0
#define R_MEAN 1
#define R_MAX 2
#define R_MIN 3
#define R_PROD 4

struct RedMeta {
    uint n_out; uint n_red; uint ndim_out; uint ndim_red; int op;
    uint out_dims[8]; uint out_istr[8];
    uint red_dims[8]; uint red_istr[8];
};

kernel void reduce_f32(device const float *in [[buffer(0)]], device float *out [[buffer(1)]],
                       constant RedMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n_out) return;
    uint rem = gid, base = 0;
    for (int d = int(m.ndim_out) - 1; d >= 0; --d) { uint id = rem % m.out_dims[d]; rem /= m.out_dims[d]; base += id * m.out_istr[d]; }
    float sum = 0.f, c = 0.f, mx = -INFINITY, mn = INFINITY, pr = 1.f;
    for (uint r = 0; r < m.n_red; ++r) {
        uint rr = r, off = base;
        for (int d = int(m.ndim_red) - 1; d >= 0; --d) { uint id = rr % m.red_dims[d]; rr /= m.red_dims[d]; off += id * m.red_istr[d]; }
        float v = in[off];
        float y = v - c, t = sum + y; c = (t - sum) - y; sum = t;   // Kahan
        mx = fmax(mx, v); mn = fmin(mn, v); pr *= v;
    }
    float res;
    switch (m.op) { case R_MEAN: res = sum / (float)m.n_red; break; case R_MAX: res = mx; break;
                    case R_MIN: res = mn; break; case R_PROD: res = pr; break; default: res = sum; }
    out[gid] = res;
}
