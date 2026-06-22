// RoPE + attention kernels (MSL). Layout BNSD [1, H, s, hd]; cos/sin [s, hd].
#include <metal_stdlib>
using namespace metal;

struct RopeMeta { uint s; uint hd; };
// BNSD attention: q/o [B,Hq,Sq,hd], k/v [B,Hkv,Skv,hd]. GQA Hq%Hkv==0. causal bottom-right (off=Skv-Sq).
// Optional bool mask (nonzero=masked-out): maskMB=0 -> mask [Sq,Skv] broadcast over batch; 1 -> [B,Sq,Skv].
struct AttnMeta { uint B; uint Hq; uint Hkv; uint Sq; uint Skv; uint hd; float scale; int causal; int hasMask; int maskMB; };

// HF/NeoX half-split rotary: out = x*cos + rotate_half(x)*sin, rotate_half = [-x[hd/2:], x[:hd/2]]
template <typename T>
kernel void rope(device const T *x [[buffer(0)]], device const T *cosb [[buffer(1)]], device const T *sinb [[buffer(2)]],
                 device T *out [[buffer(3)]], constant RopeMeta &m [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
    uint hd = m.hd, halfd = hd / 2;
    uint d = gid % hd;
    uint sp = (gid / hd) % m.s;
    uint base = gid - d;
    float xv = (float)x[gid];
    float rot = (d < halfd) ? -(float)x[base + d + halfd] : (float)x[base + d - halfd];
    float c = (float)cosb[sp * hd + d], si = (float)sinb[sp * hd + d];
    out[gid] = (T)(xv * c + rot * si);
}
template [[host_name("rope_f32")]] kernel void rope<float>(device const float *, device const float *, device const float *, device float *, constant RopeMeta &, uint);
template [[host_name("rope_f16")]] kernel void rope<half>(device const half *, device const half *, device const half *, device half *, constant RopeMeta &, uint);

// One thread per (batch b, query head h, query position i). GQA: kv head = h/(Hq/Hkv). Single-pass online
// softmax (running max + rescale) over the unmasked keys — q·k and the V read happen once per key (vs the
// two-pass max-then-sum, which did both twice). Matches the host run_attn online-softmax math. acc[] caps
// hd at 256.
template <typename T>
kernel void attn(device const T *q [[buffer(0)]], device const T *k [[buffer(1)]], device const T *v [[buffer(2)]],
                 device T *out [[buffer(3)]], constant AttnMeta &m [[buffer(4)]], device const uchar *mask [[buffer(5)]],
                 uint gid [[thread_position_in_grid]]) {
    if (gid >= m.B * m.Hq * m.Sq) return;
    uint b = gid / (m.Hq * m.Sq), r = gid % (m.Hq * m.Sq), h = r / m.Sq, i = r % m.Sq, hd = m.hd;
    uint kvh = h / (m.Hq / m.Hkv);
    device const T *qp = q + (((uint64_t)(b * m.Hq + h) * m.Sq) + i) * hd;
    int off = (int)m.Skv - (int)m.Sq;
    uint mrow = (m.maskMB ? b * m.Sq : 0u) + i;       // mask row (broadcast over batch when maskMB==0)
    float mx = -INFINITY, denom = 0.f, acc[256];
    for (uint d = 0; d < hd; ++d) acc[d] = 0.f;
    for (uint j = 0; j < m.Skv; ++j) {
        bool blk = (m.causal && (int)j > (int)i + off) || (m.hasMask && mask[mrow * m.Skv + j] != 0);
        if (blk) continue;
        device const T *kp = k + ((uint64_t)(b * m.Hkv + kvh) * m.Skv + j) * hd;
        float dot = 0.f; for (uint d = 0; d < hd; ++d) dot += (float)qp[d] * (float)kp[d];
        float s = dot * m.scale;
        float nmx = fmax(mx, s), corr = exp(mx - nmx), e = exp(s - nmx);   // first unmasked key: corr=exp(-inf)=0
        denom = denom * corr + e;
        device const T *vp = v + ((uint64_t)(b * m.Hkv + kvh) * m.Skv + j) * hd;
        for (uint d = 0; d < hd; ++d) acc[d] = acc[d] * corr + e * (float)vp[d];
        mx = nmx;
    }
    device T *op = out + (((uint64_t)(b * m.Hq + h) * m.Sq) + i) * hd;
    float inv = 1.f / denom; for (uint d = 0; d < hd; ++d) op[d] = (T)(acc[d] * inv);
}
template [[host_name("attn_f32")]] kernel void attn<float>(device const float *, device const float *, device const float *, device float *, constant AttnMeta &, device const uchar *, uint);
template [[host_name("attn_f16")]] kernel void attn<half>(device const half *, device const half *, device const half *, device half *, constant AttnMeta &, device const uchar *, uint);
template [[host_name("attn_bf16")]] kernel void attn<bfloat>(device const bfloat *, device const bfloat *, device const bfloat *, device bfloat *, constant AttnMeta &, device const uchar *, uint);

// In-place row softmax for the MPS-batched attention path: S holds the (already scaled) QKᵀ scores
// [BH, Sq, Skv]; one thread per (bh, query i) row applies causal / optional bool mask, then softmax over Skv.
// Blocked keys are written 0 so the following PV matmul ignores them. Hq lets the mask row pick the batch.
struct SmMeta { uint BH, Sq, Skv, Hq; int causal, hasMask, maskMB; };
template <typename T>
kernel void attn_softmax(device T *S [[buffer(0)]], device const uchar *mask [[buffer(1)]],
                         constant SmMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.BH * m.Sq) return;
    uint i = gid % m.Sq, bh = gid / m.Sq, b = bh / m.Hq;
    int off = (int)m.Skv - (int)m.Sq;
    uint mrow = (m.maskMB ? b * m.Sq : 0u) + i;
    device T *row = S + (uint64_t)gid * m.Skv;
    float mx = -INFINITY;
    for (uint j = 0; j < m.Skv; ++j)
        if (!((m.causal && (int)j > (int)i + off) || (m.hasMask && mask[mrow * m.Skv + j] != 0))) mx = fmax(mx, (float)row[j]);
    float den = 0.f;
    for (uint j = 0; j < m.Skv; ++j) {
        if ((m.causal && (int)j > (int)i + off) || (m.hasMask && mask[mrow * m.Skv + j] != 0)) { row[j] = (T)0; continue; }
        float e = exp((float)row[j] - mx); row[j] = (T)e; den += e;
    }
    float inv = 1.f / den; for (uint j = 0; j < m.Skv; ++j) row[j] = (T)((float)row[j] * inv);
}
template [[host_name("attn_softmax_f32")]] kernel void attn_softmax<float>(device float *, device const uchar *, constant SmMeta &, uint);
template [[host_name("attn_softmax_f16")]] kernel void attn_softmax<half>(device half *, device const uchar *, constant SmMeta &, uint);

// Replicate KV heads to Q heads (GQA/MQA) while widening to fp32: dst[b,h,*] = src[b, h/(Hq/Hkv), *]. For
// MHA (Hkv==Hq) this is an identity gather. SD = Skv*hd (elements per head). Used by the MPS attention path
// so K/V become uniform-stride [B*Hq, Skv, hd] fp32 batches.
struct GHMeta { uint BH, Hq, Hkv, SD; };
template <typename S>
kernel void gather_heads(device const S *src [[buffer(0)]], device float *dst [[buffer(1)]],
                         constant GHMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.BH * m.SD) return;
    uint bh = gid / m.SD, e = gid % m.SD, b = bh / m.Hq, h = bh % m.Hq, kvh = h / (m.Hq / m.Hkv);
    dst[gid] = (float)src[(uint64_t)(b * m.Hkv + kvh) * m.SD + e];
}
template [[host_name("gather_heads_f32")]] kernel void gather_heads<float>(device const float *, device float *, constant GHMeta &, uint);
template [[host_name("gather_heads_f16")]] kernel void gather_heads<half>(device const half *, device float *, constant GHMeta &, uint);
