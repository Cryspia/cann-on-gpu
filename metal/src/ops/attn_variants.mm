// Attention version variants (V2-V5) + sparse/nsa/ring/rope, host-side over unified memory.
// All FA-style variants forward to the SAME math as the V1 aclnnFlashAttentionScore core
// (see attention.mm / attention.metal attn_f32) so they match it bit-for-bit. The base
// aclnnFlashAttentionScore and aclnnApplyRotaryPosEmb are already defined in attention.mm; only
// the variants listed in the test (which are otherwise "Undefined symbol") are defined here.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ---- FlashAttention core: host-side, bit-identical to attention.mm run_attn (batch B>1, GQA, causal,
//      optional per-batch bool mask, any dtype) so every variant matches the V1 reference exactly. ----
inline uint16_t a_f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e >= 31) { uint32_t exp = (x >> 23) & 0xFF; if (exp == 0xFF && m) return (uint16_t)(s | 0x7E00); return (uint16_t)(s | 0x7C00); }
    if (e <= 0) { if (e < -10) return (uint16_t)s; uint32_t mant = m | 0x800000; int shift = 14 - e;
        uint32_t h = mant >> shift, rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++; return (uint16_t)(s | h); }
    uint32_t r = m & 0x1FFF, h = s | (e << 10) | (m >> 13); if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++; return (uint16_t)h;
}
inline float a_h2f(uint16_t h) {
    uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; memcpy(&x, &f, 4); x |= s; }
    else if (e == 31) x = s | 0x7F800000 | (m << 13);
    else x = s | ((e - 15 + 127) << 23) | (m << 13);
    float f; memcpy(&f, &x, 4); return f;
}
inline uint16_t a_f2bf(float f) { uint32_t x; memcpy(&x, &f, 4); return (uint16_t)((x >> 16) + ((x >> 15) & 1)); }
inline float a_bf2f(uint16_t b) { uint32_t x = ((uint32_t)b) << 16; float f; memcpy(&f, &x, 4); return f; }
inline double a_ld(const aclTensor *t, int64_t i) {
    const uint8_t *base = (const uint8_t *)t->data + (int64_t)t->offset * dtype_size(t->dtype);
    switch (t->dtype) {
        case ACL_FLOAT16: return (double)a_h2f(((const uint16_t *)base)[i]);
        case ACL_BF16:    return (double)a_bf2f(((const uint16_t *)base)[i]);
        case ACL_FLOAT:   return (double)((const float *)base)[i];
        default:          return subfp::is_low(t->dtype) ? (double)subfp::load(t->dtype, base, i) : 0.0;
    }
}
inline void a_st(aclTensor *t, int64_t i, double v) {
    uint8_t *base = (uint8_t *)t->data + (int64_t)t->offset * dtype_size(t->dtype);
    switch (t->dtype) {
        case ACL_FLOAT16: ((uint16_t *)base)[i] = a_f2h((float)v); break;
        case ACL_BF16:    ((uint16_t *)base)[i] = a_f2bf((float)v); break;
        case ACL_FLOAT:   ((float *)base)[i] = (float)v; break;
        default: if (subfp::is_low(t->dtype)) subfp::store(t->dtype, base, i, (float)v); break;
    }
}
aclnnStatus run_attn(aclOpExecutor *e, aclrtStream stream) {
    drain(stream);
    const aclTensor *q = e->a, *k = e->b, *v = e->c, *mask = e->mask; aclTensor *o = e->out;
    if (!q || !k || !v || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t B = q->viewDims[0], Nq = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3];
    int64_t Nkv = k->viewDims[1], Skv = k->viewDims[2]; double scale = e->alpha; bool causal = e->causal; int64_t off = Skv - Sq;
    if (Nkv == 0 || (Nq % Nkv) != 0) return ACLNN_ERR_PARAM_INVALID;
    const uint8_t *mp = mask ? ((const uint8_t *)mask->data + mask->offset) : nullptr; float fscale = (float)scale;
    std::vector<float> acc(D);
    for (int64_t bi = 0; bi < B; ++bi) for (int64_t h = 0; h < Nq; ++h) {
        int64_t kvh = h / (Nq / Nkv), qf = bi * Nq + h, kf = bi * Nkv + kvh;
        for (int64_t i = 0; i < Sq; ++i) {
            float m = -1e30f, l = 0.f; for (int64_t t = 0; t < D; ++t) acc[t] = 0.f;
            for (int64_t j = 0; j < Skv; ++j) {
                float part = 0; for (int64_t t = 0; t < D; ++t) part += (float)a_ld(q, (qf * Sq + i) * D + t) * (float)a_ld(k, (kf * Skv + j) * D + t);
                float score = part * fscale; bool blk = (causal && j > i + off) || (mp && mp[(bi * Sq + i) * Skv + j]);
                float se = blk ? -1e30f : score, mn = std::fmax(m, se), corr = std::exp(m - mn), p = std::exp(se - mn);
                l = l * corr + p; for (int64_t t = 0; t < D; ++t) acc[t] = acc[t] * corr + p * (float)a_ld(v, (kf * Skv + j) * D + t); m = mn;
            }
            float inv = 1.f / l; for (int64_t t = 0; t < D; ++t) a_st(o, (qf * Sq + i) * D + t, (double)(acc[t] * inv));
        }
    }
    return ACLNN_SUCCESS;
}

// ---- RotaryPositionEmbedding (rotate-half): out[:, :h] = x1*c - x2*s; out[:, h:] = x2*c + x1*s ----
// x/cos/sin/out all same shape [..., D]; last dim split in half.
aclnnStatus run_rope_emb(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *cosb = e->b, *sinb = e->c; aclTensor *o = e->out;
    if (!x || !cosb || !sinb || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int nd = (int)x->viewDims.size();
    int64_t D = x->viewDims[nd - 1], h = D / 2, rows = x->numel() / D;
    const float *xp = FP(x), *cp = FP(cosb), *sp = FP(sinb); float *op = FP(o);
    for (int64_t r = 0; r < rows; ++r) {
        const float *xr = xp + r * D, *cr = cp + r * D, *sr = sp + r * D; float *orow = op + r * D;
        for (int64_t d = 0; d < D; ++d) {
            float rot = (d < h) ? -xr[d + h] : xr[d - h];
            orow[d] = xr[d] * cr[d] + rot * sr[d];
        }
    }
    return ACLNN_SUCCESS;
}

// ---- RingAttentionUpdate: online softmax merge of two partial results ----
// out = (o1*exp(l1-m) + o2*exp(l2-m)) / (exp(l1-m)+exp(l2-m)); lse = m + log(sum); m = max(l1,l2)
// rows from lse shape; D = out1.numel()/rows.
aclnnStatus run_ring(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *o1 = e->a, *l1 = e->mean, *o2 = e->b, *l2 = e->rstd;
    aclTensor *out = e->out, *lse = e->out2;
    if (!o1 || !l1 || !o2 || !l2 || !out) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t rows = l1->numel(); if (rows == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = o1->numel() / rows;
    const float *a = FP(o1), *b = FP(o2), *la = FP(l1), *lb = FP(l2); float *oo = FP(out);
    float *ls = lse ? FP(lse) : nullptr;
    for (int64_t r = 0; r < rows; ++r) {
        float m = std::max(la[r], lb[r]);
        float ea = std::exp(la[r] - m), eb = std::exp(lb[r] - m), den = ea + eb;
        for (int64_t d = 0; d < D; ++d) oo[r * D + d] = (a[r * D + d] * ea + b[r * D + d] * eb) / den;
        if (ls) ls[r] = m + std::log(den);
    }
    return ACLNN_SUCCESS;
}

// ---- NsaCompress: block mean-pool over consecutive blockSize rows ----
// x: [Nb*bs, Dd] -> out: [Nb, Dd], out[b,d] = mean over the bs rows of block b.
aclnnStatus run_nsa(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a; aclTensor *o = e->out;
    if (!x || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t bs = e->n; if (bs <= 0) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)x->viewDims.size();
    int64_t D = x->viewDims[nd - 1], totalRows = x->numel() / D, Nb = totalRows / bs;
    const float *xp = FP(x); float *op = FP(o);
    for (int64_t b = 0; b < Nb; ++b)
        for (int64_t d = 0; d < D; ++d) {
            double acc = 0; for (int64_t j = 0; j < bs; ++j) acc += xp[(b * bs + j) * D + d];
            op[b * D + d] = (float)(acc / bs);
        }
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
// ---- FlashAttention-style variants: same simplified contract as V1, forward to shared core ----
#define FA_VARIANT(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, \
            const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *attentionOut, \
            uint64_t *ws, aclOpExecutor **ex) { \
        (void)attenMask; (void)headNum; \
        if (!query || !key || !value || !attentionOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = query; e->b = key; e->c = value; e->out = attentionOut; \
        e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, run_attn)

FA_VARIANT(aclnnFlashAttentionScoreV2)
FA_VARIANT(aclnnFlashAttentionScoreV3)
FA_VARIANT(aclnnFlashAttentionScoreV4)
FA_VARIANT(aclnnPromptFlashAttentionV2)
FA_VARIANT(aclnnPromptFlashAttentionV3)
FA_VARIANT(aclnnFusedInferAttentionScoreV2)
FA_VARIANT(aclnnFusedInferAttentionScoreV5)
FA_VARIANT(aclnnBlockSparseAttention)   // dense fallback == FlashAttentionScore

// ---- IncreFlashAttention variants: same as FA but no causal argument (causal=false) ----
#define IFA_VARIANT(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, \
            const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *attentionOut, \
            uint64_t *ws, aclOpExecutor **ex) { \
        (void)attenMask; (void)headNum; \
        if (!query || !key || !value || !attentionOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = query; e->b = key; e->c = value; e->out = attentionOut; \
        e->alpha = scaleValue; e->causal = false; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, run_attn)

IFA_VARIANT(aclnnIncreFlashAttentionV2)
IFA_VARIANT(aclnnIncreFlashAttentionV3)
IFA_VARIANT(aclnnIncreFlashAttentionV4)

// ---- RotaryPositionEmbedding (single-tensor rotate-half) ----
aclnnStatus aclnnRotaryPositionEmbeddingGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
        int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)mode;
    if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = cos; e->c = sin; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnRotaryPositionEmbedding, run_rope_emb)

// ---- RingAttentionUpdate (online merge of two partial attention outputs) ----
aclnnStatus aclnnRingAttentionUpdateGetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2,
        const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *ws, aclOpExecutor **ex) {
    if (!out1 || !lse1 || !out2 || !lse2 || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = out1; e->mean = lse1; e->b = out2; e->rstd = lse2; e->out = out; e->out2 = lse;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnRingAttentionUpdate, run_ring)

// ---- NsaCompress (block mean-pool) ----
aclnnStatus aclnnNsaCompressGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->n = blockSize; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnNsaCompress, run_nsa)
} // extern "C"
