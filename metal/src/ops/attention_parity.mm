// Attention / FlashAttention / RoPE gap operators — host-side over unified memory, parity with the
// CUDA backend (cuda/src/ops/attention.cu, mla.cu, misc_ext.cu). Only ops NOT already defined in
// attention.mm / attn_variants.mm / nn.mm / moe.mm are implemented here (no duplicate symbols).
//
// The math reuses the same float online-softmax flash core, rotate-half RoPE, online-softmax ring
// merge, RMSNorm+RoPE, block mean-pool, and rel-pos masked softmax as the CUDA reference, so every
// variant matches the canonical cores bit-for-bit / within fp tolerance. Signatures are validated
// against the canonical macros in aclnnop/aclnn_ops.h (this file includes it; it never hand-writes
// op prototypes — the header's *GetWorkspaceSize/* declarations are the source of truth).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
// ---- fp16/bf16 codecs (round-to-nearest-even with subnormals), identical to attention.mm ----
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
// dtype-aware element read/write over unified memory (fp32/fp16/bf16/low-precision via subfp).
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
inline float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// ============================================================================================
//  Cores (each mirrors a CUDA kernel) — driven via aclOpExecutor fields populated below.
// ============================================================================================

// ---- BNSD flash attention: batch B>1, GQA (Nq%Nkv==0), causal (bottom-right), optional bool mask.
//      Identical to attention.mm run_attn — used by all dense FA / sparse / NSA / floyd / rain variants. ----
aclnnStatus core_attn(aclOpExecutor *e, aclrtStream stream) {
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

// ---- rotate-half / interleaved RoPE on a single [rows,D] tensor; cos/sin already gathered per-row.
//      mode 0 = half-split (LLaMA/NeoX): out[d<h]=x*c - x[d+h]*s; out[d>=h]=x*c + x[d-h]*s
//      mode 1 = interleaved (GPT-J):     out[even]=x*c - x[+1]*s; out[odd]=x*c + x[-1]*s
//      Matches cuda k_rope (rows,D variant). ssign=+1 forward. ----
aclnnStatus core_rope(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *cosb = e->b, *sinb = e->c; aclTensor *o = e->out; int mode = (int)e->dim;
    if (!x || !cosb || !sinb || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int nd = (int)x->viewDims.size(); if (nd == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims[nd - 1], h = D / 2, rows = x->numel() / D;
    for (int64_t r = 0; r < rows; ++r) for (int64_t d = 0; d < D; ++d) {
        float c = (float)a_ld(cosb, r * D + d), si = (float)a_ld(sinb, r * D + d), xv = (float)a_ld(x, r * D + d), xp;
        if (mode == 0) { if (d < h) { xp = (float)a_ld(x, r * D + d + h); a_st(o, r * D + d, (double)(xv * c - xp * si)); }
                         else       { xp = (float)a_ld(x, r * D + d - h); a_st(o, r * D + d, (double)(xv * c + xp * si)); } }
        else { int64_t kk = d / 2; if (d % 2 == 0) { xp = (float)a_ld(x, r * D + 2 * kk + 1); a_st(o, r * D + d, (double)(xv * c - xp * si)); }
                                   else            { xp = (float)a_ld(x, r * D + 2 * kk);     a_st(o, r * D + d, (double)(xv * c + xp * si)); } }
    }
    return ACLNN_SUCCESS;
}

// ---- ApplyRotaryPosEmbV2: rope of both q and k, BNSD layout, cos/sin broadcast over [S,D].
//      Matches attention.mm run_rope (mode 0 half-split, mode 1 interleaved). ----
void rope_bnsd(const aclTensor *x, const aclTensor *cosb, const aclTensor *sinb, aclTensor *out, int mode) {
    int64_t S = x->viewDims[2], D = x->viewDims[3], total = x->numel();
    for (int64_t i = 0; i < total; ++i) {
        int64_t d = i % D, sidx = (i / D) % S, base = i - d;
        float c = (float)a_ld(cosb, sidx * D + d), si = (float)a_ld(sinb, sidx * D + d), xv = (float)a_ld(x, i), rh;
        if (mode == 0) { int64_t half = D / 2; rh = (d < half) ? -(float)a_ld(x, base + d + half) : (float)a_ld(x, base + d - half); }
        else           { rh = (d & 1) ? (float)a_ld(x, base + d - 1) : -(float)a_ld(x, base + d + 1); }
        a_st(out, i, (double)(xv * c + rh * si));
    }
}
aclnnStatus core_rope2(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *q = e->a, *k = e->b, *cosb = e->c, *sinb = e->mean; aclTensor *qo = e->out, *ko = e->out2; int mode = (int)e->dim;
    if (!q || !k || !cosb || !sinb || !qo || !ko) return ACLNN_ERR_PARAM_NULLPTR;
    rope_bnsd(q, cosb, sinb, qo, mode); rope_bnsd(k, cosb, sinb, ko, mode);
    return ACLNN_SUCCESS;
}

// ---- RingAttentionUpdate / AttentionUpdate: online-softmax merge of two partials.
//      out = (o1*ea + o2*eb)/(ea+eb), lse = m + log(ea+eb); m = max(l1,l2). Matches cuda k_ring_merge. ----
aclnnStatus core_ring(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *o1 = e->a, *l1 = e->b, *o2 = e->c, *l2 = e->mask; aclTensor *out = e->out, *lse = e->out2;
    if (!o1 || !l1 || !o2 || !l2 || !out) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t rows = l1->numel(); if (rows == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = out->numel() / rows;
    const float *a = FP(o1), *b = FP(o2), *la = FP(l1), *lb = FP(l2); float *oo = FP(out); float *ls = lse ? FP(lse) : nullptr;
    for (int64_t r = 0; r < rows; ++r) {
        float m = std::max(la[r], lb[r]), ea = std::exp(la[r] - m), eb = std::exp(lb[r] - m), den = ea + eb;
        for (int64_t d = 0; d < D; ++d) oo[r * D + d] = (a[r * D + d] * ea + b[r * D + d] * eb) / den;
        if (ls) ls[r] = m + std::log(den);
    }
    return ACLNN_SUCCESS;
}

// ---- KvRmsNormRopeCache / QkvRmsNormRopeCache / NormRopeConcat: RMSNorm(x)*gamma then rotate-half RoPE,
//      written to out (cache). Matches cuda k_rmsnorm_rope. ----
aclnnStatus core_rmsnorm_rope(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *gamma = e->b, *cosb = e->c, *sinb = e->mask; aclTensor *o = e->out;
    if (!x || !cosb || !sinb || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int nd = (int)x->viewDims.size(); if (nd == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims[nd - 1], h = D / 2, rows = x->numel() / D; float eps = (float)e->eps; int mode = (int)e->dim;
    std::vector<float> nrm(D);
    for (int64_t r = 0; r < rows; ++r) {
        float ss = 0; for (int64_t d = 0; d < D; ++d) { float v = (float)a_ld(x, r * D + d); ss += v * v; }
        float rms = 1.f / std::sqrt(ss / D + eps);
        for (int64_t d = 0; d < D; ++d) nrm[d] = (float)a_ld(x, r * D + d) * rms * (gamma ? (float)a_ld(gamma, d) : 1.f);
        for (int64_t d = 0; d < D; ++d) {
            float c = (float)a_ld(cosb, r * D + d), sn = (float)a_ld(sinb, r * D + d), np;
            if (mode == 0) { if (d < h) np = nrm[d] * c - nrm[d + h] * sn; else np = nrm[d] * c + nrm[d - h] * sn; }
            else { int64_t kk = d / 2; if (d % 2 == 0) np = nrm[d] * c - nrm[2 * kk + 1] * sn; else np = nrm[d] * c + nrm[2 * kk] * sn; }
            a_st(o, r * D + d, (double)np);
        }
    }
    return ACLNN_SUCCESS;
}

// ---- NsaCompress / NsaCompressWithCache: block mean-pool over consecutive blockSize rows.
//      x[(Nb*bs),D] -> out[Nb,D]. Matches cuda k_compress. ----
aclnnStatus core_compress(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a; aclTensor *o = e->out;
    if (!x || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t bs = e->n; if (bs <= 0) return ACLNN_ERR_PARAM_INVALID;
    int nd = (int)x->viewDims.size(); if (nd == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims[nd - 1], Nb = (x->numel() / D) / bs;
    for (int64_t b = 0; b < Nb; ++b) for (int64_t d = 0; d < D; ++d) {
        double acc = 0; for (int64_t j = 0; j < bs; ++j) acc += (float)a_ld(x, (b * bs + j) * D + d);
        a_st(o, b * D + d, acc / (double)bs);
    }
    return ACLNN_SUCCESS;
}

// ---- MaskedSoftmaxWithRelPosBias: softmax over last dim of (x*scale + relPosBias + mask).
//      Matches cuda k_relpos_softmax. ----
aclnnStatus core_relpos_softmax(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a, *relpos = e->b, *mask = e->mask; aclTensor *o = e->out; float scale = (float)e->alpha;
    if (!x || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int nd = (int)x->viewDims.size(); if (nd == 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t D = x->viewDims[nd - 1], rows = x->numel() / D;
    for (int64_t r = 0; r < rows; ++r) {
        float mx = -1e30f;
        for (int64_t d = 0; d < D; ++d) { float vv = (float)a_ld(x, r * D + d) * scale + (relpos ? (float)a_ld(relpos, r * D + d) : 0.f) + (mask ? (float)a_ld(mask, r * D + d) : 0.f); mx = std::fmax(mx, vv); }
        float sm = 0; for (int64_t d = 0; d < D; ++d) { float vv = (float)a_ld(x, r * D + d) * scale + (relpos ? (float)a_ld(relpos, r * D + d) : 0.f) + (mask ? (float)a_ld(mask, r * D + d) : 0.f); sm += std::exp(vv - mx); }
        for (int64_t d = 0; d < D; ++d) { float vv = (float)a_ld(x, r * D + d) * scale + (relpos ? (float)a_ld(relpos, r * D + d) : 0.f) + (mask ? (float)a_ld(mask, r * D + d) : 0.f); a_st(o, r * D + d, (double)(std::exp(vv - mx) / sm)); }
    }
    return ACLNN_SUCCESS;
}

// ---- AttentionToFFN / FFNToAttention: layout pass-through (copy). Matches cuda memcpy fallback. ----
aclnnStatus core_passthrough(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    const aclTensor *x = e->a; aclTensor *o = e->out;
    if (!x || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t n = std::min(x->numel(), o->numel());
    if (x->dtype == o->dtype) std::memcpy((uint8_t *)o->data + (int64_t)o->offset * dtype_size(o->dtype),
                                          (const uint8_t *)x->data + (int64_t)x->offset * dtype_size(x->dtype), (size_t)n * dtype_size(x->dtype));
    else for (int64_t i = 0; i < n; ++i) a_st(o, i, a_ld(x, i));
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace

extern "C" {
// ============================================================================================
//  ATT_DECL family (q,k,v,attenMask,scaleValue,headNum,causal,out) -> dense flash core
//  + FA_DECL / IFA / VarLen / Sparse / FusedInfer V3-V4: all reduce to the same SDPA math.
// ============================================================================================
#define GAP_ATT(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, \
            const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, \
            uint64_t *ws, aclOpExecutor **ex) { \
        (void)headNum; \
        if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->out = out; e->mask = attenMask; \
        e->alpha = scaleValue; e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, core_attn)

GAP_ATT(aclnnBlitzSparseAttention)
GAP_ATT(aclnnNsaCompressAttention)
GAP_ATT(aclnnNsaCompressAttentionInfer)
GAP_ATT(aclnnNsaSelectedAttention)
GAP_ATT(aclnnNsaSelectedAttentionInfer)
GAP_ATT(aclnnFusedFloydAttention)
GAP_ATT(aclnnRainFusionAttention)
GAP_ATT(aclnnSparseFlashAttention)
GAP_ATT(aclnnFlashAttentionVarLenScore)
GAP_ATT(aclnnFlashAttentionVarLenScoreV2)
GAP_ATT(aclnnFlashAttentionVarLenScoreV3)
GAP_ATT(aclnnFlashAttentionVarLenScoreV4)
GAP_ATT(aclnnFlashAttentionVarLenScoreV5)
GAP_ATT(aclnnFusedInferAttentionScoreV3)
GAP_ATT(aclnnFusedInferAttentionScoreV4)

// IncreFlashAttention (no causal arg; causal=false) -> dense flash core
aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)headNum;
    if (!q || !k || !v || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->out = out; e->mask = attenMask;
    e->alpha = scaleValue; e->causal = false; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnIncreFlashAttention, core_attn)

// ============================================================================================
//  ROPE_DECL family (x,cos,sin,mode,out) -> single-tensor rotate-half / interleaved RoPE
// ============================================================================================
#define GAP_ROPE(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, \
            int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = cos; e->c = sin; e->out = out; e->dim = mode; \
        *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, core_rope)

GAP_ROPE(aclnnRotaryPositionEmbeddingV2)
GAP_ROPE(aclnnRopeWithSinCosCache)
GAP_ROPE(aclnnRopeWithSinCosCacheV2)

// InterleaveRope (x,cos,sin,out) -> rope mode 1 (interleaved)
aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
        aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->b = cos; e->c = sin; e->out = out; e->dim = 1; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnInterleaveRope, core_rope)

// ApplyRotaryPosEmbV2 (q,k,cos,sin,mode,qOut,kOut) -> rope of both q & k, BNSD broadcast
aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos,
        const aclTensor *sin, int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !cos || !sin || !qOut || !kOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = cos; e->mean = const_cast<aclTensor *>(sin);
    e->out = qOut; e->out2 = kOut; e->dim = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnApplyRotaryPosEmbV2, core_rope2)

// ============================================================================================
//  RING_DECL family (out1,lse1,out2,lse2,out,lse) -> online-softmax merge
// ============================================================================================
#define GAP_RING(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2, \
            const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *ws, aclOpExecutor **ex) { \
        if (!out1 || !lse1 || !out2 || !lse2 || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = out1; e->b = lse1; e->c = out2; e->mask = lse2; e->out = out; e->out2 = lse; \
        *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, core_ring)

GAP_RING(aclnnRingAttentionUpdateV2)
GAP_RING(aclnnAttentionUpdate)

// ============================================================================================
//  NRC_DECL family (x,gamma,cos,sin,eps,mode,out) -> RMSNorm + RoPE into cache
// ============================================================================================
#define GAP_NRC(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, \
            const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->b = gamma; e->c = cos; e->mask = sin; e->out = out; \
        e->eps = eps; e->dim = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, core_rmsnorm_rope)

GAP_NRC(aclnnKvRmsNormRopeCache)
GAP_NRC(aclnnQkvRmsNormRopeCache)
GAP_NRC(aclnnNormRopeConcat)

// ============================================================================================
//  NSAC_DECL family (x,blockSize,out) -> block mean-pool
// ============================================================================================
aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out; e->n = blockSize; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnNsaCompressWithCache, core_compress)

// ============================================================================================
//  A2F_DECL family (x,out) -> layout pass-through
// ============================================================================================
#define GAP_A2F(NAME) \
    aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
        if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
        auto *e = new aclOpExecutor(); e->a = x; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
    RUN(NAME, core_passthrough)

GAP_A2F(aclnnAttentionToFFN)
GAP_A2F(aclnnFFNToAttention)

// ============================================================================================
//  MaskedSoftmaxWithRelPosBias (x,mask,relPosBias,scale,out)
// ============================================================================================
aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(const aclTensor *x, const aclTensor *mask,
        const aclTensor *relPosBias, double scale, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = x; e->mask = mask; e->b = relPosBias; e->out = out; e->alpha = scale;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMaskedSoftmaxWithRelPosBias, core_relpos_softmax)
} // extern "C"
