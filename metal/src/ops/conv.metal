// Pooling forward kernels (MSL). 2D max/avg pool over an (kh,kw) window with stride/padding; avg divides by
// the count of in-bounds elements (count_exclude_pad), matching the host pool2d. One thread per output.
#include <metal_stdlib>
using namespace metal;

struct PoolMeta { uint N, C, H, Wd, Ho, Wo; int kh, kw, sh, sw, ph, pw, avg; };

template <typename T>
kernel void pool2d_k(device const T *x [[buffer(0)]], device T *y [[buffer(1)]],
                     constant PoolMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.C * m.Ho * m.Wo) return;
    uint wo = gid % m.Wo, ho = (gid / m.Wo) % m.Ho, c = (gid / (m.Wo * m.Ho)) % m.C, n = gid / (m.Wo * m.Ho * m.C);
    float mx = -1e30, sum = 0; int cnt = 0;
    for (int r = 0; r < m.kh; ++r)
        for (int s = 0; s < m.kw; ++s) {
            int h = (int)ho * m.sh - m.ph + r, w = (int)wo * m.sw - m.pw + s;
            if (h < 0 || h >= (int)m.H || w < 0 || w >= (int)m.Wd) continue;
            float v = (float)x[((n * m.C + c) * m.H + (uint)h) * m.Wd + (uint)w];
            mx = max(mx, v); sum += v; cnt++;
        }
    y[gid] = (T)(m.avg ? sum / (float)cnt : mx);
}
template [[host_name("pool2d_f32")]] kernel void pool2d_k<float>(device const float *, device float *, constant PoolMeta &, uint);
template [[host_name("pool2d_f16")]] kernel void pool2d_k<half>(device const half *, device half *, constant PoolMeta &, uint);
template [[host_name("pool2d_bf16")]] kernel void pool2d_k<bfloat>(device const bfloat *, device bfloat *, constant PoolMeta &, uint);

// 3D max/avg pool, one thread per output; cubic k window, count_exclude_pad average. Matches pool3d.
struct Pool3Meta { uint N, C, D, H, Wd, Do, Ho, Wo; int k, s, p, avg; };
template <typename T>
kernel void pool3d_k(device const T *x [[buffer(0)]], device T *y [[buffer(1)]],
                     constant Pool3Meta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.C * m.Do * m.Ho * m.Wo) return;
    uint ow = gid % m.Wo, oh = (gid / m.Wo) % m.Ho, od = (gid / (m.Wo * m.Ho)) % m.Do, c = (gid / (m.Wo * m.Ho * m.Do)) % m.C, n = gid / (m.Wo * m.Ho * m.Do * m.C);
    float mx = -1e30, sum = 0; int cnt = 0;
    for (int kd = 0; kd < m.k; ++kd) for (int kh = 0; kh < m.k; ++kh) for (int kw = 0; kw < m.k; ++kw) {
        int d = (int)od * m.s - m.p + kd, h = (int)oh * m.s - m.p + kh, w = (int)ow * m.s - m.p + kw;
        if (d < 0 || d >= (int)m.D || h < 0 || h >= (int)m.H || w < 0 || w >= (int)m.Wd) continue;
        float v = (float)x[(((n * m.C + c) * m.D + (uint)d) * m.H + (uint)h) * m.Wd + (uint)w]; mx = max(mx, v); sum += v; cnt++;
    }
    y[gid] = (T)(m.avg ? sum / (float)cnt : mx);
}
template [[host_name("pool3d_f32")]] kernel void pool3d_k<float>(device const float *, device float *, constant Pool3Meta &, uint);
template [[host_name("pool3d_f16")]] kernel void pool3d_k<half>(device const half *, device half *, constant Pool3Meta &, uint);
template [[host_name("pool3d_bf16")]] kernel void pool3d_k<bfloat>(device const bfloat *, device bfloat *, constant Pool3Meta &, uint);

// AdaptiveAvgPool2d: one thread per output; averages the input window [hs,he)×[ws,we). Matches adaptive_avgpool2d.
struct AAMeta { uint N, C, H, Wd, Ho, Wo; };
template <typename T>
kernel void adaptive_avg2d_k(device const T *x [[buffer(0)]], device T *y [[buffer(1)]],
                             constant AAMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.C * m.Ho * m.Wo) return;
    uint ow = gid % m.Wo, oh = (gid / m.Wo) % m.Ho, c = (gid / (m.Wo * m.Ho)) % m.C, n = gid / (m.Wo * m.Ho * m.C);
    uint hs = oh * m.H / m.Ho, he = ((oh + 1) * m.H + m.Ho - 1) / m.Ho, ws = ow * m.Wd / m.Wo, we = ((ow + 1) * m.Wd + m.Wo - 1) / m.Wo;
    float s = 0;
    for (uint h = hs; h < he; ++h) for (uint w = ws; w < we; ++w) s += (float)x[((n * m.C + c) * m.H + h) * m.Wd + w];
    y[gid] = (T)(s / (float)((he - hs) * (we - ws)));
}
template [[host_name("adaptive_avg2d_f32")]] kernel void adaptive_avg2d_k<float>(device const float *, device float *, constant AAMeta &, uint);
template [[host_name("adaptive_avg2d_f16")]] kernel void adaptive_avg2d_k<half>(device const half *, device half *, constant AAMeta &, uint);
template [[host_name("adaptive_avg2d_bf16")]] kernel void adaptive_avg2d_k<bfloat>(device const bfloat *, device bfloat *, constant AAMeta &, uint);

// ---- backward / transposed conv as output-centric gathers (one thread per output element, no atomics) ----
// Each kernel inverts the forward index map: an output position gathers every input/grad position that the
// forward op would have routed to it. Matches the host scatter loops (convt2d / conv_bwd_data/weight).

// Transposed conv2d: Y[n,co,oh,ow] = Σ_{ci,kh,kw} X[n,ci,h,w]·W[ci,co,kh,kw], oh=h*s-p+kh ⇒ h=(oh+p-kh)/s.
struct CTMeta { uint N, Ci, H, Wd, Co, K, Ho, Wo; int s, p; };
template <typename T>
kernel void convt2d_k(device const T *x [[buffer(0)]], device const T *w [[buffer(1)]], device T *y [[buffer(2)]],
                      constant CTMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.Co * m.Ho * m.Wo) return;
    uint ow = gid % m.Wo, oh = (gid / m.Wo) % m.Ho, co = (gid / (m.Wo * m.Ho)) % m.Co, n = gid / (m.Wo * m.Ho * m.Co);
    float acc = 0;
    for (uint ci = 0; ci < m.Ci; ++ci) for (int kh = 0; kh < (int)m.K; ++kh) for (int kw = 0; kw < (int)m.K; ++kw) {
        int hn = (int)oh + m.p - kh, wn = (int)ow + m.p - kw;
        if (hn < 0 || hn % m.s || wn < 0 || wn % m.s) continue;
        int h = hn / m.s, ww = wn / m.s; if (h >= (int)m.H || ww >= (int)m.Wd) continue;
        acc += (float)x[((n * m.Ci + ci) * m.H + (uint)h) * m.Wd + (uint)ww] * (float)w[((ci * m.Co + co) * m.K + (uint)kh) * m.K + (uint)kw];
    }
    y[gid] = (T)acc;
}
template [[host_name("convt2d_f32")]] kernel void convt2d_k<float>(device const float *, device const float *, device float *, constant CTMeta &, uint);
template [[host_name("convt2d_f16")]] kernel void convt2d_k<half>(device const half *, device const half *, device half *, constant CTMeta &, uint);

// conv dgrad: dX[n,ci,h,w] = Σ_{co,r,ss} dY[n,co,ho,wo]·W[co,ci,r,ss], h=ho*s-p+r*d ⇒ ho=(h+p-r*d)/s.
struct CGMeta { uint N, C, H, Wd, Co, KH, KW, Ho, Wo; int s, p, d; };
template <typename T>
kernel void conv_dgrad_k(device const T *dy [[buffer(0)]], device const T *w [[buffer(1)]], device T *dx [[buffer(2)]],
                         constant CGMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.C * m.H * m.Wd) return;
    uint ww = gid % m.Wd, h = (gid / m.Wd) % m.H, ci = (gid / (m.Wd * m.H)) % m.C, n = gid / (m.Wd * m.H * m.C);
    float acc = 0;
    for (uint co = 0; co < m.Co; ++co) for (int r = 0; r < (int)m.KH; ++r) for (int ss = 0; ss < (int)m.KW; ++ss) {
        int hn = (int)h + m.p - r * m.d, wn = (int)ww + m.p - ss * m.d;
        if (hn < 0 || hn % m.s || wn < 0 || wn % m.s) continue;
        int ho = hn / m.s, wo = wn / m.s; if (ho >= (int)m.Ho || wo >= (int)m.Wo) continue;
        acc += (float)dy[((n * m.Co + co) * m.Ho + (uint)ho) * m.Wo + (uint)wo] * (float)w[((co * m.C + ci) * m.KH + (uint)r) * m.KW + (uint)ss];
    }
    dx[gid] = (T)acc;
}
template [[host_name("conv_dgrad_f32")]] kernel void conv_dgrad_k<float>(device const float *, device const float *, device float *, constant CGMeta &, uint);
template [[host_name("conv_dgrad_f16")]] kernel void conv_dgrad_k<half>(device const half *, device const half *, device half *, constant CGMeta &, uint);

// conv wgrad: dW[co,ci,r,ss] = Σ_{n,ho,wo} dY[n,co,ho,wo]·X[n,ci,h,w] (reduction, one thread per dW element).
template <typename T>
kernel void conv_wgrad_k(device const T *x [[buffer(0)]], device const T *dy [[buffer(1)]], device T *dw [[buffer(2)]],
                         constant CGMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.Co * m.C * m.KH * m.KW) return;
    uint ss = gid % m.KW, r = (gid / m.KW) % m.KH, ci = (gid / (m.KW * m.KH)) % m.C, co = gid / (m.KW * m.KH * m.C);
    float acc = 0;
    for (uint n = 0; n < m.N; ++n) for (uint ho = 0; ho < m.Ho; ++ho) for (uint wo = 0; wo < m.Wo; ++wo) {
        int h = (int)ho * m.s - m.p + (int)r * m.d, w = (int)wo * m.s - m.p + (int)ss * m.d;
        if (h < 0 || h >= (int)m.H || w < 0 || w >= (int)m.Wd) continue;
        acc += (float)dy[((n * m.Co + co) * m.Ho + ho) * m.Wo + wo] * (float)x[((n * m.C + ci) * m.H + (uint)h) * m.Wd + (uint)w];
    }
    dw[gid] = (T)acc;
}
template [[host_name("conv_wgrad_f32")]] kernel void conv_wgrad_k<float>(device const float *, device const float *, device float *, constant CGMeta &, uint);
template [[host_name("conv_wgrad_f16")]] kernel void conv_wgrad_k<half>(device const half *, device const half *, device half *, constant CGMeta &, uint);

// pool2d backward: dX[n,c,h,w] gathers grad from covering windows. avg = Σ dY/count_exclude_pad; max =
// dY where (h,w) is the window's first argmax. Matches pool2d_bwd.
struct PBMeta { uint N, C, H, Wd, Ho, Wo; int k, s, p, avg; };
template <typename T>
kernel void pool2d_bwd_k(device const T *x [[buffer(0)]], device const T *dy [[buffer(1)]], device T *dx [[buffer(2)]],
                         constant PBMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.N * m.C * m.H * m.Wd) return;
    uint w = gid % m.Wd, h = (gid / m.Wd) % m.H, c = (gid / (m.Wd * m.H)) % m.C, n = gid / (m.Wd * m.H * m.C);
    float acc = 0;
    for (int r = 0; r < m.k; ++r) for (int ss = 0; ss < m.k; ++ss) {
        int hn = (int)h + m.p - r, wn = (int)w + m.p - ss;
        if (hn < 0 || hn % m.s || wn < 0 || wn % m.s) continue;
        int oh = hn / m.s, ow = wn / m.s; if (oh < 0 || oh >= (int)m.Ho || ow < 0 || ow >= (int)m.Wo) continue;
        float go = (float)dy[((n * m.C + c) * m.Ho + (uint)oh) * m.Wo + (uint)ow];
        if (m.avg) {
            int cnt = 0;
            for (int rr = 0; rr < m.k; ++rr) for (int sss = 0; sss < m.k; ++sss) { int hh = oh * m.s - m.p + rr, wcw = ow * m.s - m.p + sss; if (hh >= 0 && hh < (int)m.H && wcw >= 0 && wcw < (int)m.Wd) cnt++; }
            acc += go / (float)cnt;
        } else {
            float mx = -1e30; int mh = -1, mw = -1;
            for (int rr = 0; rr < m.k; ++rr) for (int sss = 0; sss < m.k; ++sss) { int hh = oh * m.s - m.p + rr, wcw = ow * m.s - m.p + sss; if (hh < 0 || hh >= (int)m.H || wcw < 0 || wcw >= (int)m.Wd) continue; float v = (float)x[((n * m.C + c) * m.H + (uint)hh) * m.Wd + (uint)wcw]; if (v > mx) { mx = v; mh = hh; mw = wcw; } }
            if (mh == (int)h && mw == (int)w) acc += go;
        }
    }
    dx[gid] = (T)acc;
}
template [[host_name("pool2d_bwd_f32")]] kernel void pool2d_bwd_k<float>(device const float *, device const float *, device float *, constant PBMeta &, uint);
template [[host_name("pool2d_bwd_f16")]] kernel void pool2d_bwd_k<half>(device const half *, device const half *, device half *, constant PBMeta &, uint);
