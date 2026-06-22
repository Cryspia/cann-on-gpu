// Matmul family host side via MPSMatrixMultiplication (direct on MTLBuffers — no graph compile).
// Covers aclnnMatmul / aclnnMatmulBias (transA/transB + bias + activation epilogue) / aclnnBatchMatMul.
// fp32 + fp16; computation in the input dtype (MPS picks fp32 accumulation internally).
#import "../internal.h"
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <cmath>
#include <cstring>

namespace {

struct MMPost { uint32_t M; uint32_t N; int32_t act; int32_t hasBias; };

bool mps_dtype(aclDataType dt, MPSDataType *out, size_t *esz) {
    if (dt == ACL_FLOAT)   { *out = MPSDataTypeFloat32; *esz = 4; return true; }
    if (dt == ACL_FLOAT16) { *out = MPSDataTypeFloat16; *esz = 2; return true; }
    return false;
}
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}

// ---- GPU elementwise dtype conversion (for the widen path: bf16 / sub-byte -> fp32/fp16 and back) ----
struct CastMeta { uint32_t n; };   // mirrors cast.metal
// MSL kernel name for a src->dst conversion, or nil if none exists (caller falls back to the host loop).
NSString *cvt_name(aclDataType src, aclDataType dst) {
    if (dst == ACL_FLOAT) {
        if (src == ACL_BF16)    return @"widen_bf16_f32";
        if (src == ACL_FLOAT16) return @"cast_f16_f32";
    } else if (dst == ACL_FLOAT16) {
        if (src == ACL_FLOAT) return @"cast_f32_f16";   // sub-byte sources go through the decode kernel below
    } else if (dst == ACL_BF16 && src == ACL_FLOAT) return @"narrow_f32_bf16";
    return nil;
}
void encode_cvt(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso,
                id<MTLBuffer> src, size_t srcOff, id<MTLBuffer> dst, size_t dstOff, uint32_t n) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:srcOff atIndex:0];
    [enc setBuffer:dst offset:dstOff atIndex:1];
    CastMeta cm{ n }; [enc setBytes:&cm length:sizeof(cm) atIndex:2];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// ---- sub-byte (fp8/fp4/fp6) -> fp16 GPU decode: a host-built fp32-bit-pattern table (filled by the subfp
// codec, hence bit-exact with the host decode) + the decode_sub_f16 kernel. ----
struct DecMeta { uint32_t n, packed, mask; };
// Allocate + fill the decode table (256/64/16 fp32-bit entries) for a sub-byte dtype. Returns nil if dt is
// not sub-byte. *packed/*mask describe the bit layout for decode_sub_f16.
void *sub_decode_table(aclDataType dt, uint32_t *packed, uint32_t *mask) {
    if (!subfp::is_low(dt)) return nullptr;
    *packed = subfp::is_fp4(dt) ? 1 : 0;
    *mask = subfp::is_fp4(dt) ? 0xfu : (subfp::is_fp6(dt) ? 0x3fu : 0xffu);
    int n = subfp::is_fp4(dt) ? 16 : (subfp::is_fp6(dt) ? 64 : 256);
    uint32_t *t = (uint32_t *)mtl::alloc((size_t)n * 4);
    if (!t) return nullptr;
    for (int c = 0; c < n; ++c) {
        float v = subfp::is_fp8(dt) ? subfp::dec8(dt, (uint8_t)c) : subfp::sub_dec(dt, (uint8_t)c);
        std::memcpy(&t[c], &v, 4);
    }
    return t;
}
void encode_dec(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pso, id<MTLBuffer> src, size_t srcOff,
                id<MTLBuffer> tbl, size_t tblOff, id<MTLBuffer> dst, size_t dstOff, uint32_t n, uint32_t packed, uint32_t mask) {
    [enc setComputePipelineState:pso];
    [enc setBuffer:src offset:srcOff atIndex:0];
    [enc setBuffer:dst offset:dstOff atIndex:1];
    DecMeta dm{ n, packed, mask }; [enc setBytes:&dm length:sizeof(dm) atIndex:2];
    [enc setBuffer:tbl offset:tblOff atIndex:3];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = (n ? n : 1);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

// ---- generic host dtype read/write (covers dtypes/mixed-precision MPS can't run directly:
//      bf16, fp8/fp4/fp6, and fp16-in/fp32-out). Host loop over unified memory (exact). ----
inline float mm_h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; std::memcpy(&x, &f, 4); x |= sign; }
    else if (e == 31) x = sign | 0x7F800000u | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}
inline uint16_t mm_f2h(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u; int32_t e = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00u);
    uint32_t r = m & 0x1FFF, h = sign | (e << 10) | (m >> 13);
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}
inline float mm_bf2f(uint16_t h) { uint32_t b = (uint32_t)h << 16; float f; std::memcpy(&f, &b, 4); return f; }
inline uint16_t mm_f2bf(float f) { uint32_t b; std::memcpy(&b, &f, 4); b += 0x7FFF + ((b >> 16) & 1); return (uint16_t)(b >> 16); }
inline float mm_rd(const aclTensor *t, int64_t i) {
    const void *base = t->data; int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:   return ((const float *)base)[off + i];
        case ACL_FLOAT16: return mm_h2f(((const uint16_t *)base)[off + i]);
        case ACL_BF16:    return mm_bf2f(((const uint16_t *)base)[off + i]);
        case ACL_INT8:    return (float)((const int8_t *)base)[off + i];
        default: break;
    }
    if (subfp::is_low(t->dtype)) return subfp::load(t->dtype, (const uint8_t *)base + off, i);
    return 0.f;
}
inline void mm_wr(aclTensor *t, int64_t i, float v) {
    void *base = t->data; int64_t off = t->offset;
    switch (t->dtype) {
        case ACL_FLOAT:   ((float *)base)[off + i] = v; return;
        case ACL_FLOAT16: ((uint16_t *)base)[off + i] = mm_f2h(v); return;
        case ACL_BF16:    ((uint16_t *)base)[off + i] = mm_f2bf(v); return;
        default: return;
    }
}
bool mps_ok(const aclTensor *A, const aclTensor *B, const aclTensor *C) {
    MPSDataType d; size_t e;
    return mps_dtype(A->dtype, &d, &e) && mps_dtype(B->dtype, &d, &e) && mps_dtype(C->dtype, &d, &e) &&
           A->dtype == C->dtype && B->dtype == C->dtype;   // MPS path: uniform fp32/fp16 only
}
// Host GEMM fallback: C[M,N] = op(A) @ op(B) (+ bias) (+ act), per batch. Reads/writes any dtype above.
// bias may be [N] (broadcast per column) or [M,N] (per element); activation: 1=relu, 2=exact-erf gelu.
aclnnStatus gemm_host(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                      bool transA, bool transB, int act, uint32_t batch, int64_t M, int64_t K, int64_t N) {
    int64_t aR = transA ? K : M, aC = transA ? M : K;   // stored A shape per batch
    int64_t bR = transB ? N : K, bC = transB ? K : N;   // stored B shape per batch
    bool bias2d = bias && bias->viewDims.size() == 2;   // [M,N] per-element vs [N] broadcast
    for (uint32_t bi = 0; bi < batch; ++bi) {
        int64_t aBase = (int64_t)bi * aR * aC, bBase = (int64_t)bi * bR * bC, cBase = (int64_t)bi * M * N;
        for (int64_t m = 0; m < M; ++m)
          for (int64_t n = 0; n < N; ++n) {
            double acc = 0;
            for (int64_t k = 0; k < K; ++k) {
              float a = mm_rd(A, aBase + (transA ? k * aC + m : m * aC + k));
              float b = mm_rd(B, bBase + (transB ? n * bC + k : k * bC + n));
              acc += (double)a * b;
            }
            if (bias) acc += mm_rd(bias, bias2d ? m * N + n : n);
            if (act == 1) acc = acc > 0 ? acc : 0;
            else if (act == 2) acc = 0.5 * acc * (1.0 + std::erf(acc * 0.70710678118654752));
            mm_wr(C, cBase + m * N + n, (float)acc);
          }
    }
    return ACLNN_SUCCESS;
}

// Widen a non-MPS-native float GEMM into a native MPS GEMM of dtype `wdt` (defined below): decode A/B/bias
// into `wdt` temps, run the native MPS path, narrow the result back into C. bf16 widens to fp32 (bf16's
// range exceeds fp16); sub-byte fp8/fp4/fp6/HiF8 widen to fp16 (every code is exact in fp16).
static aclnnStatus gemm_widen(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                              bool transA, bool transB, uint32_t batch, int64_t M, int64_t K, int64_t N,
                              aclDataType wdt, aclrtStream stream);

// C[M,N] = op(A) @ op(B) (+ bias[N]) (+ act), batched over `batch` independent matrices.
aclnnStatus gemm(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                 bool transA, bool transB, int act, uint32_t batch,
                 int64_t M, int64_t K, int64_t N, aclrtStream stream) {
    bool bias2d = bias && bias->viewDims.size() == 2;
    // bf16 path: MPSMatrixMultiplication has no bf16, but bf16 is a truncation of fp32 (lossless to widen).
    // Widen A/B/bias to fp32, run the native MPS fp32 GEMM, narrow the result back to bf16. (bf16 +
    // activation / 2-D bias take the exact host path below.)
    if (A->dtype == ACL_BF16 && B->dtype == ACL_BF16 && C->dtype == ACL_BF16 &&
        (!bias || bias->dtype == ACL_BF16) && act == 0 && !bias2d)
        return gemm_widen(A, B, bias, C, transA, transB, batch, M, K, N, ACL_FLOAT, stream);
    // sub-byte float path: fp8 (e4m3/e5m2/HiF8), fp4 and fp6 have no MPS type, but every code decodes
    // losslessly into fp16 (range <= 32768 < 65504, mantissa <= 3 bits <= fp16's 10). Widen A/B/bias to
    // fp16 and run the native MPS fp16 GEMM (fp32 accumulation internally, matching the reference).
    if (subfp::is_low(A->dtype) && subfp::is_low(B->dtype) && act == 0 && !bias2d)
        return gemm_widen(A, B, bias, C, transA, transB, batch, M, K, N, ACL_FLOAT16, stream);
    // Host fallback (exact) for cases the MPS path + epilogue can't run correctly:
    //  - dtypes/mixed precision MPS rejects (mixed sub-byte/float operands, fp16-in/fp32-out, e8m0 scales)
    //  - activation (the MSL epilogue uses a tanh-gelu approx; reference wants exact erf-gelu)
    //  - per-element bias[M,N] (the MSL epilogue only broadcasts bias[N])
    if (!mps_ok(A, B, C) || act != 0 || bias2d) {
        auto *st = (AclStream *)stream; if (st && st->last) [st->last waitUntilCompleted];
        return gemm_host(A, B, bias, C, transA, transB, act, batch, M, K, N);
    }
    MPSDataType dt; size_t esz;
    if (!mps_dtype(C->dtype, &dt, &esz)) return ACLNN_ERR_PARAM_INVALID;
    id<MTLDevice> dev = mtl::device();
    size_t oa, ob, oc; id<MTLBuffer> bufA = buf_of(A, &oa), bufB = buf_of(B, &ob), bufC = buf_of(C, &oc);
    if (!bufA || !bufB || !bufC) return ACLNN_ERR_RUNTIME_ERROR;

    // stored matrix shapes (per batch): A is [aR,aC], B is [bR,bC]
    int64_t aR = transA ? K : M, aC = transA ? M : K;
    int64_t bR = transB ? N : K, bC = transB ? K : N;
    size_t aStride = (size_t)aR * aC * esz, bStride = (size_t)bR * bC * esz, cStride = (size_t)M * N * esz;

    MPSMatrixDescriptor *dA = [MPSMatrixDescriptor matrixDescriptorWithRows:aR columns:aC matrices:batch
                                  rowBytes:aC * esz matrixBytes:aStride dataType:dt];
    MPSMatrixDescriptor *dB = [MPSMatrixDescriptor matrixDescriptorWithRows:bR columns:bC matrices:batch
                                  rowBytes:bC * esz matrixBytes:bStride dataType:dt];
    MPSMatrixDescriptor *dC = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N matrices:batch
                                  rowBytes:N * esz matrixBytes:cStride dataType:dt];
    MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:oa descriptor:dA];
    MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:ob descriptor:dB];
    MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:oc descriptor:dC];
    MPSMatrixMultiplication *mm =
        [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:transA transposeRight:transB
                                             resultRows:M resultColumns:N interiorColumns:K alpha:1.0 beta:0.0];

    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    // A multi-matrix descriptor makes the standard encode process the whole batch.
    [mm encodeToCommandBuffer:cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];

    // bias / activation epilogue
    if (bias || act) {
        NSString *k = (C->dtype == ACL_FLOAT) ? @"mm_post_f32" : @"mm_post_f16";
        id<MTLComputePipelineState> pso = mtl::pipeline(k);
        if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
        size_t obias = 0; id<MTLBuffer> bBias = bias ? buf_of(bias, &obias) : nil;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        for (uint32_t bi = 0; bi < batch; ++bi) {
            MMPost p{ (uint32_t)M, (uint32_t)N, (int32_t)act, bias ? 1 : 0 };
            [enc setBuffer:bufC offset:oc + bi * cStride atIndex:0];
            if (bBias) [enc setBuffer:bBias offset:obias atIndex:1]; else [enc setBuffer:bufC offset:oc atIndex:1];
            [enc setBytes:&p length:sizeof(p) atIndex:2];
            NSUInteger n = (NSUInteger)M * N, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
            [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        }
        [enc endEncoding];
    }
    [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}

// Widen-to-native GEMM: decode A/B/bias into `wdt` temp buffers (wdt = fp32 for bf16, fp16 for sub-byte),
// run the native MPS GEMM at that dtype, then narrow the result back into C. The widen is lossless (bf16 is
// a truncated fp32; every sub-byte code is exact in fp16) and MPS accumulates in fp32, so the result matches
// the reference within tolerance.
//
// Conversion runs on the GPU when an MSL kernel exists for every needed dtype pair (gemm_widen_gpu); the
// host loop (gemm_widen_host) is the fallback for any pair without one.

// Host conversion path: decode/narrow with a dtype-specialized loop (branch on wdt once, not per element).
static aclnnStatus gemm_widen_host(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                                   bool transA, bool transB, uint32_t batch, int64_t M, int64_t K, int64_t N,
                                   aclDataType wdt, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];   // A/B ready in unified memory
    int64_t aR = transA ? K : M, aC = transA ? M : K, bR = transB ? N : K, bC = transB ? K : N;
    size_t aN = (size_t)aR * aC * batch, bN = (size_t)bR * bC * batch, cN = (size_t)M * N * batch;
    size_t es = dtype_size(wdt);
    void *ta = mtl::alloc(aN * es), *tb = mtl::alloc(bN * es), *tc = mtl::alloc(cN * es);
    void *tbias = bias ? mtl::alloc((size_t)N * es) : nullptr;
    if (!ta || !tb || !tc || (bias && !tbias)) { mtl::free_(ta); mtl::free_(tb); mtl::free_(tc); if (tbias) mtl::free_(tbias); return ACLNN_ERR_RUNTIME_ERROR; }
    aclTensor Af = *A, Bf = *B, Cf = *C, Bz;
    Af.data = ta; Af.offset = 0; Af.dtype = wdt;
    Bf.data = tb; Bf.offset = 0; Bf.dtype = wdt;
    Cf.data = tc; Cf.offset = 0; Cf.dtype = wdt;
    if (bias) { Bz = *bias; Bz.data = tbias; Bz.offset = 0; Bz.dtype = wdt; }
    if (wdt == ACL_FLOAT16) {
        uint16_t *pa = (uint16_t *)ta, *pb = (uint16_t *)tb, *pbi = (uint16_t *)tbias;
        for (size_t i = 0; i < aN; ++i) pa[i] = mm_f2h(mm_rd(A, (int64_t)i));
        for (size_t i = 0; i < bN; ++i) pb[i] = mm_f2h(mm_rd(B, (int64_t)i));
        if (bias) for (int64_t i = 0; i < N; ++i) pbi[i] = mm_f2h(mm_rd(bias, i));
    } else {   // ACL_FLOAT
        float *pa = (float *)ta, *pb = (float *)tb, *pbi = (float *)tbias;
        for (size_t i = 0; i < aN; ++i) pa[i] = mm_rd(A, (int64_t)i);
        for (size_t i = 0; i < bN; ++i) pb[i] = mm_rd(B, (int64_t)i);
        if (bias) for (int64_t i = 0; i < N; ++i) pbi[i] = mm_rd(bias, i);
    }
    aclnnStatus st = gemm(&Af, &Bf, bias ? &Bz : nullptr, &Cf, transA, transB, 0, batch, M, K, N, stream);
    if (s && s->last) [s->last waitUntilCompleted];   // result in tc ready
    if (st == ACLNN_SUCCESS) {
        if (wdt == ACL_FLOAT16) { const uint16_t *pc = (const uint16_t *)tc; for (size_t i = 0; i < cN; ++i) mm_wr(C, (int64_t)i, mm_h2f(pc[i])); }
        else                    { const float    *pc = (const float *)tc;    for (size_t i = 0; i < cN; ++i) mm_wr(C, (int64_t)i, pc[i]); }
    }
    mtl::free_(ta); mtl::free_(tb); mtl::free_(tc); if (tbias) mtl::free_(tbias);
    return st;
}

// GPU conversion path: dispatch the decode/widen into temps, the MPS GEMM, and the narrow — all chained on
// the stream's command queue (in-order), with a single host wait at the end (gemm_widen is synchronous, as
// the host path was). Returns false (no GPU kernel for some pair, or misaligned offset) to fall back to host.
static bool gemm_widen_gpu(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                           bool transA, bool transB, uint32_t batch, int64_t M, int64_t K, int64_t N,
                           aclDataType wdt, aclrtStream stream) {
    // A/B convert by simple cast (cvt_name) or, when sub-byte, by the table decode kernel; bias/narrow are
    // always simple casts. Bail (-> host) if any needed kernel is missing.
    bool aLow = subfp::is_low(A->dtype), bLow = subfp::is_low(B->dtype);
    NSString *nA = aLow ? nil : cvt_name(A->dtype, wdt), *nB = bLow ? nil : cvt_name(B->dtype, wdt);
    NSString *nC = cvt_name(wdt, C->dtype), *nBias = bias ? cvt_name(bias->dtype, wdt) : nil;
    if ((!aLow && !nA) || (!bLow && !nB) || !nC || (bias && !nBias)) return false;
    id<MTLComputePipelineState> pDec = (aLow || bLow) ? mtl::pipeline(@"decode_sub_f16") : nil;
    id<MTLComputePipelineState> pA = aLow ? pDec : mtl::pipeline(nA), pB = bLow ? pDec : mtl::pipeline(nB);
    id<MTLComputePipelineState> pC = mtl::pipeline(nC), pBias = bias ? mtl::pipeline(nBias) : nil;
    if (!pA || !pB || !pC || (bias && !pBias)) return false;   // kernel not in the metallib
    size_t oA, oB, oC, oBias = 0;
    id<MTLBuffer> bufA = buf_of(A, &oA), bufB = buf_of(B, &oB), bufC = buf_of(C, &oC);
    id<MTLBuffer> bufBias = bias ? buf_of(bias, &oBias) : nil;
    if (!bufA || !bufB || !bufC || (bias && !bufBias)) return false;
    if ((oA | oB | oC | oBias) & 3u) return false;   // Metal compute buffer offsets must be 4-byte aligned

    int64_t aR = transA ? K : M, aC = transA ? M : K, bR = transB ? N : K, bC = transB ? K : N;
    size_t aN = (size_t)aR * aC * batch, bN = (size_t)bR * bC * batch, cN = (size_t)M * N * batch, es = dtype_size(wdt);
    // sub-byte decode tables (nil for non-sub-byte); freed with the temps after the GPU completes.
    uint32_t aPk = 0, aMk = 0, bPk = 0, bMk = 0;
    void *aTbl = aLow ? sub_decode_table(A->dtype, &aPk, &aMk) : nullptr;
    void *bTbl = bLow ? sub_decode_table(B->dtype, &bPk, &bMk) : nullptr;
    void *ta = mtl::alloc(aN * es), *tb = mtl::alloc(bN * es), *tc = mtl::alloc(cN * es);
    void *tbias = bias ? mtl::alloc((size_t)N * es) : nullptr;
    if (!ta || !tb || !tc || (bias && !tbias) || (aLow && !aTbl) || (bLow && !bTbl)) {
        mtl::free_(ta); mtl::free_(tb); mtl::free_(tc); mtl::free_(tbias); mtl::free_(aTbl); mtl::free_(bTbl); return false;
    }
    size_t za, zb, zc, zbias = 0, zaT = 0, zbT = 0;
    id<MTLBuffer> bta = mtl::bufferFor(ta, &za), btb = mtl::bufferFor(tb, &zb), btc = mtl::bufferFor(tc, &zc);
    id<MTLBuffer> btbias = tbias ? mtl::bufferFor(tbias, &zbias) : nil;
    id<MTLBuffer> baT = aTbl ? mtl::bufferFor(aTbl, &zaT) : nil, bbT = bTbl ? mtl::bufferFor(bTbl, &zbT) : nil;

    auto *s = (AclStream *)stream; id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (aLow) encode_dec(enc, pA, bufA, oA, baT, zaT, bta, za, (uint32_t)aN, aPk, aMk); else encode_cvt(enc, pA, bufA, oA, bta, za, (uint32_t)aN);
        if (bLow) encode_dec(enc, pB, bufB, oB, bbT, zbT, btb, zb, (uint32_t)bN, bPk, bMk); else encode_cvt(enc, pB, bufB, oB, btb, zb, (uint32_t)bN);
        if (bias) encode_cvt(enc, pBias, bufBias, oBias, btbias, zbias, (uint32_t)N);
        [enc endEncoding]; [cb commit]; if (s) s->last = cb;   // MPS GEMM below chains after this on the same queue
    }
    aclTensor Af = *A, Bf = *B, Cf = *C, Bz;
    Af.data = ta; Af.offset = 0; Af.dtype = wdt;
    Bf.data = tb; Bf.offset = 0; Bf.dtype = wdt;
    Cf.data = tc; Cf.offset = 0; Cf.dtype = wdt;
    if (bias) { Bz = *bias; Bz.data = tbias; Bz.offset = 0; Bz.dtype = wdt; }
    aclnnStatus st = gemm(&Af, &Bf, bias ? &Bz : nullptr, &Cf, transA, transB, 0, batch, M, K, N, stream);
    if (st == ACLNN_SUCCESS) {
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            encode_cvt(enc, pC, btc, zc, bufC, oC, (uint32_t)cN);
            [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];   // single wait before freeing temps
            if (s) s->last = cb;
        }
    }
    mtl::free_(ta); mtl::free_(tb); mtl::free_(tc);
    if (tbias) mtl::free_(tbias); if (aTbl) mtl::free_(aTbl); if (bTbl) mtl::free_(bTbl);
    return st == ACLNN_SUCCESS;
}

static aclnnStatus gemm_widen(const aclTensor *A, const aclTensor *B, const aclTensor *bias, aclTensor *C,
                              bool transA, bool transB, uint32_t batch, int64_t M, int64_t K, int64_t N,
                              aclDataType wdt, aclrtStream stream) {
    if (gemm_widen_gpu(A, B, bias, C, transA, transB, batch, M, K, N, wdt, stream)) return ACLNN_SUCCESS;
    return gemm_widen_host(A, B, bias, C, transA, transB, batch, M, K, N, wdt, stream);
}

aclnnStatus run_matmul2d(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *A = e->a, *B = e->b, *bias = e->c; aclTensor *C = e->out;
    if (!A || !B || !C) return ACLNN_ERR_PARAM_NULLPTR;
    if (A->viewDims.size() != 2 || B->viewDims.size() != 2) return ACLNN_ERR_PARAM_INVALID;
    bool tA = (e->stride[0] != 0), tB = (e->stride[1] != 0);   // trans flags stashed in stride[]
    int64_t aR = A->viewDims[0], aC = A->viewDims[1], bR = B->viewDims[0], bC = B->viewDims[1];
    int64_t M = tA ? aC : aR, K = tA ? aR : aC, N = tB ? bR : bC, Kb = tB ? bC : bR;
    if (K != Kb) return ACLNN_ERR_PARAM_INVALID;
    return gemm(A, B, bias, C, tA, tB, (int)e->m, 1, M, K, N, s);
}

aclnnStatus run_bmm(aclOpExecutor *e, aclrtStream s) {
    const aclTensor *A = e->a, *B = e->b; aclTensor *C = e->out;
    if (!A || !B || !C) return ACLNN_ERR_PARAM_NULLPTR;
    if (A->viewDims.size() != 3 || B->viewDims.size() != 3) return ACLNN_ERR_PARAM_INVALID;
    int64_t batch = A->viewDims[0], M = A->viewDims[1], K = A->viewDims[2], N = B->viewDims[2];
    if (B->viewDims[0] != batch || B->viewDims[1] != K) return ACLNN_ERR_PARAM_INVALID;
    return gemm(A, B, nullptr, C, false, false, 0, (uint32_t)batch, M, K, N, s);
}

} // namespace

extern "C" {

aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType;
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = self; e->b = other; e->out = out;
    e->stride[0] = 0; e->stride[1] = 0; e->m = 0;   // no trans, no act
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmul(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    if (aclCaptureRecord(s, &aclnnMatmul, e, w, wss)) return ACLNN_SUCCESS;   // graph capture: record, don't run
    aclnnStatus st; @autoreleasepool { st = run_matmul2d(e, s); } delete e; return st;
}

aclnnStatus aclnnMatmulBiasGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclTensor *bias,
        bool transA, bool transB, int64_t act, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = self; e->b = other; e->c = bias; e->out = out;
    e->stride[0] = transA ? 1 : 0; e->stride[1] = transB ? 1 : 0; e->m = act;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulBias(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_matmul2d(e, s); } delete e; return st;
}

aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex) {
    (void)cubeMathType;
    if (!self || !other || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_MATMUL; e->a = self; e->b = other; e->out = out;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchMatMul(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_bmm(e, s); } delete e; return st;
}

} // extern "C"
