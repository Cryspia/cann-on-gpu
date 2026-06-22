// RoPE + attention host side: aclnnApplyRotaryPosEmb, aclnnFlashAttentionScore.
#import "../internal.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>   // batched QKᵀ / PV GEMM for the MHA fast path
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace {
// fp16 encode (round-to-nearest-even) WITH subnormal support — matches hardware fp16 (and CUDA's __half cast),
// which the test's getO reader expects for op outputs (the harness's input-packing f2h flushes subnormals, but
// a backend that flushed near-zero outputs to 0 would fail the relative-error check around tiny values).
inline uint16_t a_f2h(float f) {
    uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e >= 31) { uint32_t exp = (x >> 23) & 0xFF; if (exp == 0xFF && m) return (uint16_t)(s | 0x7E00); return (uint16_t)(s | 0x7C00); }
    if (e <= 0) {                                   // subnormal / underflow: shift mantissa with the implicit 1
        if (e < -10) return (uint16_t)s;            // too small even for the smallest subnormal
        uint32_t mant = m | 0x800000; int shift = 14 - e;   // 14 = 24 (fp32 frac+1) - 10 (fp16 frac)
        uint32_t h = mant >> shift; uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++; return (uint16_t)(s | h);
    }
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
// dtype-aware element read/write over unified memory (fp32/fp16/bf16/fp4/fp6/fp8).
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
struct RopeMeta { uint32_t s; uint32_t hd; };
struct AttnMeta { uint32_t B, Hq, Hkv, Sq, Skv, hd; float scale; int32_t causal, hasMask, maskMB; };  // mirrors attention.metal
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
const char *fdt(aclDataType dt) { return dt == ACL_FLOAT ? "f32" : dt == ACL_FLOAT16 ? "f16" : dt == ACL_BF16 ? "bf16" : nullptr; }

// Host-side RoPE for one BNSD tensor: o[i] = x*cos + rotate(x)*sin; cos/sin[S,D] broadcast over B,N.
// mode 0 = half-split (LLaMA/NeoX), mode 1 = interleaved (GPT-J). Float arithmetic matches the CUDA reference.
void rope_one(const aclTensor *x, const aclTensor *cosb, const aclTensor *sinb, aclTensor *out, int mode) {
    int64_t S = x->viewDims[2], D = x->viewDims[3], total = x->numel();
    for (int64_t i = 0; i < total; ++i) {
        int64_t d = i % D, s = (i / D) % S, base = i - d;
        float c = (float)a_ld(cosb, s * D + d), si = (float)a_ld(sinb, s * D + d), xv = (float)a_ld(x, i), rh;
        if (mode == 0) { int64_t half = D / 2; rh = (d < half) ? -(float)a_ld(x, base + d + half) : (float)a_ld(x, base + d - half); }
        else           { rh = (d & 1) ? (float)a_ld(x, base + d - 1) : -(float)a_ld(x, base + d + 1); }
        a_st(out, i, (double)(xv * c + rh * si));
    }
}

aclnnStatus run_rope(aclOpExecutor *e, aclrtStream stream) {
    drain(stream);
    const aclTensor *q = e->a, *k = e->b, *cosb = e->c, *sinb = e->mean; aclTensor *qo = e->out, *ko = e->out2; int mode = (int)e->dim;
    if (!q || !k || !cosb || !sinb || !qo || !ko) return ACLNN_ERR_PARAM_NULLPTR;
    rope_one(q, cosb, sinb, qo, mode);
    rope_one(k, cosb, sinb, ko, mode);
    return ACLNN_SUCCESS;
}

// Matmul-hardware attention (MHA only): QKᵀ and PV run as batched MPS GEMM, with an MSL masked-softmax in
// between, materializing the [BH,Sq,Skv] scores. Computed in fp32 (scores need fp32 precision through the
// softmax for fp16-tolerance accuracy): fp16 q/k/v are widened to fp32 and the fp32 output is narrowed back.
// Gated to fp16/fp32, no GQA (Q/K/V are uniform-stride batches of BH=B*Nq matrices), byte/no mask. Returns
// false (caller uses the per-thread kernel / host) for anything outside that or on setup failure.
struct SmMetaH { uint32_t BH, Sq, Skv, Hq; int32_t causal, hasMask, maskMB; };
static void cast_into(id<MTLComputeCommandEncoder> enc, NSString *kn, id<MTLBuffer> src, size_t so, id<MTLBuffer> dst, size_t doff, uint32_t n) {
    id<MTLComputePipelineState> p = mtl::pipeline(kn);
    [enc setComputePipelineState:p];
    [enc setBuffer:src offset:so atIndex:0]; [enc setBuffer:dst offset:doff atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger tg = p.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}
struct GHMetaH { uint32_t BH, Hq, Hkv, SD; };
static void gather_into(id<MTLComputeCommandEncoder> enc, NSString *kn, id<MTLBuffer> src, size_t so, id<MTLBuffer> dst, size_t doff, GHMetaH gm) {
    id<MTLComputePipelineState> p = mtl::pipeline(kn);
    [enc setComputePipelineState:p];
    [enc setBuffer:src offset:so atIndex:0]; [enc setBuffer:dst offset:doff atIndex:1];
    [enc setBytes:&gm length:sizeof(gm) atIndex:2];
    NSUInteger n = (NSUInteger)gm.BH * gm.SD, tg = p.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}
bool run_attn_mps(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *mask, aclTensor *o,
                  int64_t B, int64_t Nq, int64_t Sq, int64_t Skv, int64_t D, double scale, bool causal, int64_t mb, aclrtStream stream) {
    bool f16;
    if (o->dtype == ACL_FLOAT) f16 = false;
    else if (o->dtype == ACL_FLOAT16) f16 = true;
    else return false;
    if (q->dtype != o->dtype || k->dtype != o->dtype || v->dtype != o->dtype) return false;
    if (mask && dtype_size(mask->dtype) != 1) return false;
    int64_t Nkv = k->viewDims[1], BH = B * Nq;
    size_t qn = (size_t)BH * Sq * D, kn = (size_t)BH * Skv * D, Sn = (size_t)BH * Sq * Skv;
    if (Sn * 4 > ((size_t)2 << 30)) return false;   // cap materialized scores at 2 GB
    id<MTLComputePipelineState> pso = mtl::pipeline(@"attn_softmax_f32");
    id<MTLDevice> dev = mtl::device();
    if (!pso || !dev) return false;
    size_t oq, ok, ov, oo, om = 0;
    id<MTLBuffer> bq = buf_of(q, &oq), bk = buf_of(k, &ok), bv = buf_of(v, &ov), bo = buf_of(o, &oo);
    id<MTLBuffer> bm = mask ? buf_of(mask, &om) : bq;
    if (!bq || !bk || !bv || !bo || !bm) return false;
    // fp32 working buffers. q/o get a temp only for fp16 (widen / narrow). k/v get a temp for fp16 OR GQA
    // (Nkv<Nq): gather_heads widens and replicates kv-heads to Hq heads, so K/V become [B*Nq, Skv, D] fp32.
    bool kvTemp = f16 || (Nkv != Nq);
    void *Sd = mtl::alloc(Sn * 4), *qf = nullptr, *kf = nullptr, *vf = nullptr, *of = nullptr;
    if (f16) { qf = mtl::alloc(qn * 4); of = mtl::alloc(qn * 4); }
    if (kvTemp) { kf = mtl::alloc(kn * 4); vf = mtl::alloc(kn * 4); }
    if (!Sd || (f16 && (!qf || !of)) || (kvTemp && (!kf || !vf))) { mtl::free_(Sd); mtl::free_(qf); mtl::free_(kf); mtl::free_(vf); mtl::free_(of); return false; }
    size_t oS, zq = 0, zk = 0, zv = 0, zo = 0;
    id<MTLBuffer> bS = mtl::bufferFor(Sd, &oS);
    id<MTLBuffer> Bq = f16 ? mtl::bufferFor(qf, &zq) : bq, Bk = kvTemp ? mtl::bufferFor(kf, &zk) : bk;
    id<MTLBuffer> Bv = kvTemp ? mtl::bufferFor(vf, &zv) : bv, Bo = f16 ? mtl::bufferFor(of, &zo) : bo;
    size_t Oq = f16 ? zq : oq, Ok = kvTemp ? zk : ok, Ov = kvTemp ? zv : ov, Oo = f16 ? zo : oo;
    MPSMatrixDescriptor *dQ = [MPSMatrixDescriptor matrixDescriptorWithRows:Sq columns:D matrices:BH rowBytes:D*4 matrixBytes:(NSUInteger)Sq*D*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dK = [MPSMatrixDescriptor matrixDescriptorWithRows:Skv columns:D matrices:BH rowBytes:D*4 matrixBytes:(NSUInteger)Skv*D*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dS = [MPSMatrixDescriptor matrixDescriptorWithRows:Sq columns:Skv matrices:BH rowBytes:Skv*4 matrixBytes:(NSUInteger)Sq*Skv*4 dataType:MPSDataTypeFloat32];
    MPSMatrixDescriptor *dO = [MPSMatrixDescriptor matrixDescriptorWithRows:Sq columns:D matrices:BH rowBytes:D*4 matrixBytes:(NSUInteger)Sq*D*4 dataType:MPSDataTypeFloat32];
    MPSMatrix *mQ = [[MPSMatrix alloc] initWithBuffer:Bq offset:Oq descriptor:dQ];
    MPSMatrix *mK = [[MPSMatrix alloc] initWithBuffer:Bk offset:Ok descriptor:dK];
    MPSMatrix *mS = [[MPSMatrix alloc] initWithBuffer:bS offset:oS descriptor:dS];
    MPSMatrix *mV = [[MPSMatrix alloc] initWithBuffer:Bv offset:Ov descriptor:dK];   // V same shape as K
    MPSMatrix *mO = [[MPSMatrix alloc] initWithBuffer:Bo offset:Oo descriptor:dO];
    MPSMatrixMultiplication *qk = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:Sq resultColumns:Skv interiorColumns:D alpha:scale beta:0.0];
    MPSMatrixMultiplication *pv = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:NO resultRows:Sq resultColumns:D interiorColumns:Skv alpha:1.0 beta:0.0];
    auto *st = (AclStream *)stream; id<MTLCommandQueue> que = st ? st->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [que commandBuffer];
    if (f16 || kvTemp) {   // prep fp32 inputs: widen q (fp16), gather+widen k/v (fp16 / GQA)
        id<MTLComputeCommandEncoder> we = [cb computeCommandEncoder];
        if (f16) cast_into(we, @"cast_f16_f32", bq, oq, Bq, Oq, (uint32_t)qn);
        if (kvTemp) {
            NSString *gk = f16 ? @"gather_heads_f16" : @"gather_heads_f32";
            GHMetaH gm{ (uint32_t)BH, (uint32_t)Nq, (uint32_t)Nkv, (uint32_t)(Skv * D) };
            gather_into(we, gk, bk, ok, Bk, Ok, gm);
            gather_into(we, gk, bv, ov, Bv, Ov, gm);
        }
        [we endEncoding];
    }
    [qk encodeToCommandBuffer:cb leftMatrix:mQ rightMatrix:mK resultMatrix:mS];
    SmMetaH sm{ (uint32_t)BH, (uint32_t)Sq, (uint32_t)Skv, (uint32_t)Nq, causal ? 1 : 0, mask ? 1 : 0, (int32_t)mb };
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bS offset:oS atIndex:0]; [enc setBuffer:bm offset:om atIndex:1]; [enc setBytes:&sm length:sizeof(sm) atIndex:2];
    NSUInteger n = (NSUInteger)(BH * Sq), tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    [pv encodeToCommandBuffer:cb leftMatrix:mS rightMatrix:mV resultMatrix:mO];
    if (f16) {   // narrow fp32 result -> fp16 output
        id<MTLComputeCommandEncoder> ne = [cb computeCommandEncoder];
        cast_into(ne, @"cast_f32_f16", Bo, Oo, bo, oo, (uint32_t)qn);
        [ne endEncoding];
    }
    [cb commit]; [cb waitUntilCompleted];
    bool good = (cb.error == nil);
    mtl::free_(Sd); mtl::free_(qf); mtl::free_(kf); mtl::free_(vf); mtl::free_(of);
    if (st) st->last = cb;
    return good;
}

// Host-side BNSD attention: handles batch B>1, GQA (Hq%Hkv==0), causal (bottom-right aligned), optional
// per-batch bool mask [B,Sq,Skv], and any input/output dtype (fp32/fp16/bf16/fp4/fp8) via the subfp codec.
aclnnStatus run_attn(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *q = e->a, *k = e->b, *v = e->c, *mask = e->mask; aclTensor *o = e->out;
    if (!q || !k || !v || !o) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t B = q->viewDims[0], Nq = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3];
    int64_t Nkv = k->viewDims[1], Skv = k->viewDims[2]; double scale = e->alpha; bool causal = e->causal; int64_t off = Skv - Sq;
    if (Nkv == 0 || (Nq % Nkv) != 0) return ACLNN_ERR_PARAM_INVALID;
    int64_t mb = (mask && mask->numel() / Skv == Sq) ? 0 : 1;   // mask [Sq,Skv] broadcasts over batch; [B,Sq,Skv] is per-batch

    // ---- matmul-hardware fast path (fp16/fp32, incl. GQA): QKᵀ and PV via batched MPS GEMM, with K/V gathered
    // to Hq heads. Larger threshold than the per-thread kernel since the materialized scores + two GEMMs only
    // pay off at scale. ----
    if ((o->dtype == ACL_FLOAT || o->dtype == ACL_FLOAT16) && (!mask || dtype_size(mask->dtype) == 1) &&
        B * Nq * Sq * Skv * D >= (1 << 14)) {
        if (run_attn_mps(q, k, v, mask, o, B, Nq, Sq, Skv, D, scale, causal, mb, stream)) return ACLNN_SUCCESS;
    }

    // ---- GPU fast path: uniform fp32/fp16, hd<=256, byte (bool/uint8) mask or none. Pipelined (no drain).
    // Size-gated: a kernel dispatch costs more than the host loop for small problems, so only take the GPU
    // path once the work (~B*Hq*Sq*Skv*hd) is large enough to amortize it — decode / tiny prefill stay on the
    // host (no regression), real long-sequence prefill goes to the GPU. The threshold still routes the
    // moderate test_attn_variants / torch-oracle attn cases through the GPU so the kernel stays verified. ----
    const char *suf = fdt(o->dtype);
    if (suf && q->dtype == o->dtype && k->dtype == o->dtype && v->dtype == o->dtype && D <= 256 &&
        (!mask || dtype_size(mask->dtype) == 1) && B * Nq * Sq * Skv * D >= 1024) {
        id<MTLComputePipelineState> pso = mtl::pipeline([NSString stringWithFormat:@"attn_%s", suf]);
        size_t oq, ok, ov, oo, om = 0;
        id<MTLBuffer> bq = buf_of(q, &oq), bk = buf_of(k, &ok), bv = buf_of(v, &ov), bo = buf_of(o, &oo);
        id<MTLBuffer> bm = mask ? buf_of(mask, &om) : bq;   // dummy (bq) when no mask — never read (hasMask=0)
        if (pso && bq && bk && bv && bo && bm) {
            AttnMeta gm{ (uint32_t)B, (uint32_t)Nq, (uint32_t)Nkv, (uint32_t)Sq, (uint32_t)Skv, (uint32_t)D,
                         (float)scale, causal ? 1 : 0, mask ? 1 : 0, (int32_t)mb };
            auto *st = (AclStream *)stream; id<MTLCommandQueue> que = st ? st->q : mtl::defaultQueue();
            id<MTLCommandBuffer> cb = [que commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pso];
            [enc setBuffer:bq offset:oq atIndex:0]; [enc setBuffer:bk offset:ok atIndex:1]; [enc setBuffer:bv offset:ov atIndex:2];
            [enc setBuffer:bo offset:oo atIndex:3]; [enc setBytes:&gm length:sizeof(gm) atIndex:4]; [enc setBuffer:bm offset:om atIndex:5];
            NSUInteger n = (NSUInteger)(B * Nq * Sq), tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
            [enc endEncoding]; [cb commit];
            if (st) st->last = cb; else [cb waitUntilCompleted];
            return ACLNN_SUCCESS;
        }
        // fall through to host on any setup failure
    }

    // ---- host fallback: any dtype (bf16/fp8/...), hd>256, non-byte mask. Float online-softmax (flash). ----
    drain(stream);
    const uint8_t *mp = mask ? ((const uint8_t *)mask->data + mask->offset) : nullptr; float fscale = (float)scale;
    std::vector<float> acc(D);
    for (int64_t bi = 0; bi < B; ++bi) for (int64_t h = 0; h < Nq; ++h) {
        int64_t kvh = h / (Nq / Nkv), qf = bi * Nq + h, kf = bi * Nkv + kvh;
        for (int64_t i = 0; i < Sq; ++i) {
            float m = -1e30f, l = 0.f; for (int64_t t = 0; t < D; ++t) acc[t] = 0.f;
            for (int64_t j = 0; j < Skv; ++j) {
                float part = 0; for (int64_t t = 0; t < D; ++t) part += (float)a_ld(q, (qf * Sq + i) * D + t) * (float)a_ld(k, (kf * Skv + j) * D + t);
                float score = part * fscale; bool blk = (causal && j > i + off) || (mp && mp[(mb * bi * Sq + i) * Skv + j]);
                float se = blk ? -1e30f : score, mn = std::fmax(m, se), corr = std::exp(m - mn), p = std::exp(se - mn);
                l = l * corr + p; for (int64_t t = 0; t < D; ++t) acc[t] = acc[t] * corr + p * (float)a_ld(v, (kf * Skv + j) * D + t); m = mn;
            }
            float inv = 1.f / l; for (int64_t t = 0; t < D; ++t) a_st(o, (qf * Sq + i) * D + t, (double)(acc[t] * inv));
        }
    }
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos,
        const aclTensor *sin, int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!q || !k || !cos || !sin || !qOut || !kOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = cos; e->mean = sin; e->out = qOut; e->out2 = kOut; e->dim = mode;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyRotaryPosEmb(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_rope(e, s); } delete e; return st;
}

aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value,
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *attentionOut,
        uint64_t *ws, aclOpExecutor **ex) {
    (void)headNum;
    if (!query || !key || !value || !attentionOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = query; e->b = key; e->c = value; e->out = attentionOut; e->mask = attenMask;
    // scaleValue == 0 means "use the default 1/sqrt(headDim)" (CANN convention; callers may pass it explicitly).
    double D = query->viewDims.empty() ? 1.0 : (double)query->viewDims.back();
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / std::sqrt(D);
    e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnFlashAttentionScore(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_attn(e, s); } delete e; return st;
}
} // extern "C"
