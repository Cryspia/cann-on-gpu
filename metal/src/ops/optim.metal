// Optimizer step kernels (MSL): in-place param/state updates. One thread per element (clip = 1 thread).
#include <metal_stdlib>
using namespace metal;

struct OptMeta { uint n; float p[7]; int flag; };

kernel void opt_adam(device float *param [[buffer(0)]], device float *m [[buffer(1)]], device float *v [[buffer(2)]],
                     device const float *g [[buffer(3)]], constant OptMeta &o [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], b1 = o.p[1], b2 = o.p[2], eps = o.p[3], wd = o.p[4], bc1 = o.p[5], bc2 = o.p[6];
    float gi = g[i] + wd * param[i];
    float mi = b1 * m[i] + (1 - b1) * gi, vi = b2 * v[i] + (1 - b2) * gi * gi;
    m[i] = mi; v[i] = vi;
    param[i] -= lr * (mi / bc1) / (sqrt(vi / bc2) + eps);
}
kernel void opt_adagrad(device float *param [[buffer(0)]], device float *ss [[buffer(1)]], device const float *g [[buffer(2)]],
                        constant OptMeta &o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], eps = o.p[1], wd = o.p[2];
    float gi = g[i] + wd * param[i]; float s = ss[i] + gi * gi; ss[i] = s;
    param[i] -= lr * gi / (sqrt(s) + eps);
}
kernel void opt_rmsprop(device float *param [[buffer(0)]], device float *v [[buffer(1)]], device const float *g [[buffer(2)]],
                        constant OptMeta &o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], alpha = o.p[1], eps = o.p[2], wd = o.p[3];
    float gi = g[i] + wd * param[i]; float vi = alpha * v[i] + (1 - alpha) * gi * gi; v[i] = vi;
    param[i] -= lr * gi / (sqrt(vi) + eps);
}
kernel void opt_momentum(device float *param [[buffer(0)]], device float *buf [[buffer(1)]], device const float *g [[buffer(2)]],
                         constant OptMeta &o [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], mu = o.p[1], wd = o.p[2], damp = o.p[3];
    float gi = g[i] + wd * param[i];
    float b = mu * buf[i] + (1 - damp) * gi; buf[i] = b;
    float upd = o.flag ? (gi + mu * b) : b;   // nesterov
    param[i] -= lr * upd;
}
kernel void opt_adamax(device float *param [[buffer(0)]], device float *m [[buffer(1)]], device float *u [[buffer(2)]],
                       device const float *g [[buffer(3)]], constant OptMeta &o [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], b1 = o.p[1], b2 = o.p[2], eps = o.p[3], bc1 = o.p[4];
    float gi = g[i]; float mi = b1 * m[i] + (1 - b1) * gi; float ui = fmax(b2 * u[i], fabs(gi));
    m[i] = mi; u[i] = ui;
    param[i] -= (lr / bc1) * mi / (ui + eps);
}
kernel void opt_adadelta(device float *param [[buffer(0)]], device float *sq [[buffer(1)]], device float *ad [[buffer(2)]],
                         device const float *g [[buffer(3)]], constant OptMeta &o [[buffer(4)]], uint i [[thread_position_in_grid]]) {
    if (i >= o.n) return;
    float lr = o.p[0], rho = o.p[1], eps = o.p[2], wd = o.p[3];
    float gi = g[i] + wd * param[i]; float s = rho * sq[i] + (1 - rho) * gi * gi; sq[i] = s;
    float delta = sqrt(ad[i] + eps) / sqrt(s + eps) * gi;
    ad[i] = rho * ad[i] + (1 - rho) * delta * delta;
    param[i] -= lr * delta;
}
// ClipGradNorm: total L2 norm -> normOut; scale grad in place if norm > maxNorm. Single thread (small grads).
kernel void clip_grad(device float *g [[buffer(0)]], device float *normOut [[buffer(1)]],
                      constant OptMeta &o [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    float ss = 0.f; for (uint k = 0; k < o.n; ++k) ss += g[k] * g[k];
    float tn = sqrt(ss); normOut[0] = tn;
    float sc = tn > o.p[0] ? o.p[0] / (tn + 1e-6f) : 1.f;
    if (sc != 1.f) for (uint k = 0; k < o.n; ++k) g[k] *= sc;
}
