// In-place operator gap fill (aclnnInplace*): the 108 inplace symbols the CUDA backend exports but the
// Metal backend did not. Each computes Xxx in place on selfRef.
//
// Two strategies, mirroring the CUDA backend:
//  (A) DELEGATION — for the vast majority, aclnnInplaceXxx(selfRef,args) == aclnnXxx(selfRef,args,out=selfRef).
//      Both base symbols live in the SAME dylib (declared in aclnnop/aclnn_ops.h). We forward both phases:
//        InplaceXxxGetWorkspaceSize(selfRef,args,ws,ex) -> XxxGetWorkspaceSize(selfRef,args,selfRef,ws,ex)
//        InplaceXxx(w,wss,e,s)                           -> Xxx(w,wss,e,s)
//      The base Execute builds/runs/deletes the executor; we add nothing.
//  (B) HOST-DIRECT — where no pure base op exists in Metal, or where the base kernel's output layout would
//      not match selfRef's dtype (comparisons/bitwise write a fixed width; selfRef must keep its own dtype),
//      we compute over unified memory after draining the stream (contiguous, exact / statistical for RNG).
//      This matches the existing inplace.mm / ops_ext.mm style (FP(t), drain, delete e).
//
// Self-contained: only edits this new file (no internal.h / existing-file changes).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>

// ============================================================================
// (A) DELEGATION to existing base ops (out = selfRef). Base symbols are declared
//     in aclnn_ops.h; we just forward. Macros by argument shape.
// ============================================================================
extern "C" {

// inplace unary: Inplace(selfRef) -> Base(selfRef, out=selfRef)
#define DLG_UN(IP, BASE) \
aclnnStatus IP##GetWorkspaceSize(aclTensor *selfRef, uint64_t *ws, aclOpExecutor **ex) { \
    return BASE##GetWorkspaceSize(selfRef, selfRef, ws, ex); } \
aclnnStatus IP(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return BASE(w, wss, e, s); }

// inplace tensor-other: Inplace(selfRef, other) -> Base(selfRef, other, out=selfRef)
#define DLG_BIN(IP, BASE) \
aclnnStatus IP##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    return BASE##GetWorkspaceSize(selfRef, other, selfRef, ws, ex); } \
aclnnStatus IP(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return BASE(w, wss, e, s); }

// inplace scalar-other: Inplace(selfRef, scalar) -> Base(selfRef, scalar, out=selfRef)
#define DLG_SC(IP, BASE) \
aclnnStatus IP##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *sc, uint64_t *ws, aclOpExecutor **ex) { \
    return BASE##GetWorkspaceSize(selfRef, sc, selfRef, ws, ex); } \
aclnnStatus IP(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return BASE(w, wss, e, s); }

// ---- unary math / activations ----
DLG_UN(aclnnInplaceAcos,       aclnnAcos)
DLG_UN(aclnnInplaceAcosh,      aclnnAcosh)
DLG_UN(aclnnInplaceAsin,       aclnnAsin)
DLG_UN(aclnnInplaceAsinh,      aclnnAsinh)
DLG_UN(aclnnInplaceAtan,       aclnnAtan)
DLG_UN(aclnnInplaceAtanh,      aclnnAtanh)
DLG_UN(aclnnInplaceCeil,       aclnnCeil)
DLG_UN(aclnnInplaceCos,        aclnnCos)
DLG_UN(aclnnInplaceCosh,       aclnnCosh)
DLG_UN(aclnnInplaceErf,        aclnnErf)
DLG_UN(aclnnInplaceErfc,       aclnnErfc)
DLG_UN(aclnnInplaceErfinv,     aclnnErfinv)
DLG_UN(aclnnInplaceExp2,       aclnnExp2)
DLG_UN(aclnnInplaceExpm1,      aclnnExpm1)
DLG_UN(aclnnInplaceFloor,      aclnnFloor)
DLG_UN(aclnnInplaceFrac,       aclnnFrac)
DLG_UN(aclnnInplaceHardsigmoid,aclnnHardsigmoid)
DLG_UN(aclnnInplaceHardswish,  aclnnHardswish)
DLG_UN(aclnnInplaceLog,        aclnnLog)
DLG_UN(aclnnInplaceLog10,      aclnnLog10)
DLG_UN(aclnnInplaceLog1p,      aclnnLog1p)
DLG_UN(aclnnInplaceLog2,       aclnnLog2)
DLG_UN(aclnnInplaceMish,       aclnnMish)
DLG_UN(aclnnInplaceRound,      aclnnRound)
DLG_UN(aclnnInplaceRsqrt,      aclnnRsqrt)
DLG_UN(aclnnInplaceSelu,       aclnnSelu)
DLG_UN(aclnnInplaceSin,        aclnnSin)
DLG_UN(aclnnInplaceSinc,       aclnnSinc)
DLG_UN(aclnnInplaceSinh,       aclnnSinh)
DLG_UN(aclnnInplaceSqrt,       aclnnSqrt)
DLG_UN(aclnnInplaceTan,        aclnnTan)
DLG_UN(aclnnInplaceTrunc,      aclnnTrunc)

// ---- binary arithmetic / math (tensor∘tensor) ----
DLG_BIN(aclnnInplaceAtan2,             aclnnAtan2)
DLG_BIN(aclnnInplaceDiv,               aclnnDiv)
DLG_BIN(aclnnInplaceMul,               aclnnMul)
DLG_BIN(aclnnInplacePowTensorTensor,   aclnnPowTensorTensor)
DLG_BIN(aclnnInplaceRemainderTensorTensor, aclnnRemainderTensorTensor)

// ---- scalar-other math / activations ----
DLG_SC(aclnnInplaceCelu,              aclnnCelu)
DLG_SC(aclnnInplaceDivs,              aclnnDivs)
DLG_SC(aclnnInplaceElu,               aclnnElu)
DLG_SC(aclnnInplaceFmodScalar,        aclnnFmodScalar)
DLG_SC(aclnnInplacePowTensorScalar,   aclnnPowTensorScalar)
DLG_SC(aclnnInplaceRemainderTensorScalar, aclnnRemainderTensorScalar)

} // extern "C" (delegation block 1)

extern "C" {
// ---- ops needing extra fixed args; delegate explicitly ----

// Lerp: Inplace(selfRef, end, weight) -> aclnnLerp(self, other=end, alpha=weight, out=self)
aclnnStatus aclnnInplaceLerpGetWorkspaceSize(aclTensor *selfRef, const aclTensor *end, const aclScalar *weight, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnLerpGetWorkspaceSize(selfRef, end, weight, selfRef, ws, ex); }
aclnnStatus aclnnInplaceLerp(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnLerp(w, wss, e, s); }

// Sub: Inplace(selfRef, other, alpha) -> aclnnSub(self, other, alpha, out=self)
aclnnStatus aclnnInplaceSubGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnSubGetWorkspaceSize(selfRef, other, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceSub(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnSub(w, wss, e, s); }

// Subs: Inplace(selfRef, other, alpha) -> aclnnSubs(self, other, alpha, out=self)
aclnnStatus aclnnInplaceSubsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnSubsGetWorkspaceSize(selfRef, other, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceSubs(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnSubs(w, wss, e, s); }

// Addcdiv: Inplace(selfRef, t1, t2, value) -> aclnnAddcdiv(self, t1, t2, value, out=self)
aclnnStatus aclnnInplaceAddcdivGetWorkspaceSize(aclTensor *selfRef, const aclTensor *t1, const aclTensor *t2, const aclScalar *value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddcdivGetWorkspaceSize(selfRef, t1, t2, value, selfRef, ws, ex); }
aclnnStatus aclnnInplaceAddcdiv(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAddcdiv(w, wss, e, s); }

// AddRelu: Inplace(selfRef, other) -> aclnnAddRelu(self, other, out=self)
aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddReluGetWorkspaceSize(selfRef, other, selfRef, ws, ex); }
aclnnStatus aclnnInplaceAddRelu(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAddRelu(w, wss, e, s); }

// MaskedFillScalar: Inplace(selfRef, mask, value) -> aclnnMaskedFillScalar(self, mask, value, out=self)
aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclScalar *value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMaskedFillScalarGetWorkspaceSize(selfRef, mask, value, selfRef, ws, ex); }
aclnnStatus aclnnInplaceMaskedFillScalar(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnMaskedFillScalar(w, wss, e, s); }

// MaskedFillTensor: Inplace(selfRef, mask, value) -> aclnnMaskedFillTensor(self, mask, value, out=self)
aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMaskedFillTensorGetWorkspaceSize(selfRef, mask, value, selfRef, ws, ex); }
aclnnStatus aclnnInplaceMaskedFillTensor(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnMaskedFillTensor(w, wss, e, s); }

// MaskedScatter: Inplace(self, mask, src) -> aclnnMaskedScatter(self, mask, src, out=self)
aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(aclTensor *self, const aclTensor *mask, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMaskedScatterGetWorkspaceSize(self, mask, src, self, ws, ex); }
aclnnStatus aclnnInplaceMaskedScatter(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnMaskedScatter(w, wss, e, s); }

// NanToNum: Inplace(selfRef, nan, posinf, neginf) -> aclnnNanToNum(self, nan, posinf, neginf, out=self)
aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(aclTensor *selfRef, float nan, float posinf, float neginf, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnNanToNumGetWorkspaceSize(selfRef, nan, posinf, neginf, selfRef, ws, ex); }
aclnnStatus aclnnInplaceNanToNum(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnNanToNum(w, wss, e, s); }

// Cumprod: Inplace(self, dim, dtype) -> aclnnCumprod(self, dim, dtype, out=self)
aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *self, int64_t dim, aclDataType dtype, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnCumprodGetWorkspaceSize(self, dim, dtype, self, ws, ex); }
aclnnStatus aclnnInplaceCumprod(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnCumprod(w, wss, e, s); }

// DivMod: Inplace(selfRef, other, roundMode) -> aclnnDivMod(self, other, roundMode, out=self)
aclnnStatus aclnnInplaceDivModGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, int64_t roundMode, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDivModGetWorkspaceSize(selfRef, other, roundMode, selfRef, ws, ex); }
aclnnStatus aclnnInplaceDivMod(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnDivMod(w, wss, e, s); }

// DivMods: Inplace(selfRef, other, roundMode) -> aclnnDivMods(self, other, roundMode, out=self)
aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, int64_t roundMode, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnDivModsGetWorkspaceSize(selfRef, other, roundMode, selfRef, ws, ex); }
aclnnStatus aclnnInplaceDivMods(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnDivMods(w, wss, e, s); }

// IndexFill: Inplace(selfRef, dim, index, value) -> aclnnIndexFill(self, dim, index, value, out=self)
aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, double value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnIndexFillGetWorkspaceSize(selfRef, dim, index, value, selfRef, ws, ex); }
aclnnStatus aclnnInplaceIndexFill(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnIndexFill(w, wss, e, s); }

// Put: Inplace(selfRef, index, source, accumulate) -> aclnnPut(self, index, source, accumulate, out=self)
aclnnStatus aclnnInplacePutGetWorkspaceSize(aclTensor *selfRef, const aclTensor *index, const aclTensor *source, bool accumulate, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnPutGetWorkspaceSize(selfRef, index, source, accumulate, selfRef, ws, ex); }
aclnnStatus aclnnInplacePut(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnPut(w, wss, e, s); }

// Renorm: Inplace(selfRef, p, dim, maxnorm) -> aclnnRenorm(self, p, dim, maxnorm, out=self)
aclnnStatus aclnnInplaceRenormGetWorkspaceSize(aclTensor *selfRef, double p, int64_t dim, double maxnorm, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRenormGetWorkspaceSize(selfRef, p, dim, maxnorm, selfRef, ws, ex); }
aclnnStatus aclnnInplaceRenorm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnRenorm(w, wss, e, s); }

// RoundDecimals: Inplace(selfRef, decimals) -> aclnnRoundDecimals(self, decimals, out=self)
aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(aclTensor *selfRef, int64_t decimals, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRoundDecimalsGetWorkspaceSize(selfRef, decimals, selfRef, ws, ex); }
aclnnStatus aclnnInplaceRoundDecimals(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnRoundDecimals(w, wss, e, s); }

// Scatter: Inplace(selfRef, dim, index, src, reduce) -> aclnnScatter(self, dim, index, src, reduce, out=self)
aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnScatterGetWorkspaceSize(selfRef, dim, index, src, reduce, selfRef, ws, ex); }
aclnnStatus aclnnInplaceScatter(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnScatter(w, wss, e, s); }

// ScatterUpdate: Inplace(selfRef, index, src) -> aclnnScatterUpdate(self, index, src, out=self)
aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(aclTensor *selfRef, const aclTensor *index, const aclTensor *src, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnScatterUpdateGetWorkspaceSize(selfRef, index, src, selfRef, ws, ex); }
aclnnStatus aclnnInplaceScatterUpdate(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnScatterUpdate(w, wss, e, s); }

// ScatterValue: Inplace(selfRef, dim, index, value) -> aclnnScatterValue(self, dim, index, value, out=self)
aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnScatterValueGetWorkspaceSize(selfRef, dim, index, value, selfRef, ws, ex); }
aclnnStatus aclnnInplaceScatterValue(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnScatterValue(w, wss, e, s); }

// Triu: Inplace(selfRef, diagonal) -> aclnnTriu(self, diagonal, out=self)
aclnnStatus aclnnInplaceTriuGetWorkspaceSize(aclTensor *selfRef, int64_t diagonal, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnTriuGetWorkspaceSize(selfRef, diagonal, selfRef, ws, ex); }
aclnnStatus aclnnInplaceTriu(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnTriu(w, wss, e, s); }

// Threshold: Inplace(selfRef, threshold, value) -> aclnnThreshold(self, threshold, value, out=self)
aclnnStatus aclnnInplaceThresholdGetWorkspaceSize(aclTensor *selfRef, const aclScalar *a, const aclScalar *b, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnThresholdGetWorkspaceSize(selfRef, a, b, selfRef, ws, ex); }
aclnnStatus aclnnInplaceThreshold(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnThreshold(w, wss, e, s); }

// ---- matmul-accumulate family: aclnnInplaceX(selfRef, A, B, beta, alpha) -> aclnnX(selfRef, A, B, beta, alpha, out=selfRef) ----
aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mat1, const aclTensor *mat2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddmmGetWorkspaceSize(selfRef, mat1, mat2, beta, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceAddmm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAddmm(w, wss, e, s); }

aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddbmmGetWorkspaceSize(selfRef, batch1, batch2, beta, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceAddbmm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAddbmm(w, wss, e, s); }

aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnBaddbmmGetWorkspaceSize(selfRef, batch1, batch2, beta, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceBaddbmm(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnBaddbmm(w, wss, e, s); }

// Addr: Inplace(selfRef, vec1, vec2, beta, alpha) -> aclnnAddr(self, vec1, vec2, beta, alpha, out=self)
aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnAddrGetWorkspaceSize(selfRef, vec1, vec2, beta, alpha, selfRef, ws, ex); }
aclnnStatus aclnnInplaceAddr(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnAddr(w, wss, e, s); }

// ---- RNG family that DOES have a writable-out base in Metal: forward (base writes into out=selfRef) ----
// InplaceBernoulli(selfRef, p, seed) -> aclnnBernoulli(out=selfRef, p, seed)
aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(aclTensor *selfRef, double p, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnBernoulliGetWorkspaceSize(selfRef, p, seed, ws, ex); }
aclnnStatus aclnnInplaceBernoulli(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnBernoulli(w, wss, e, s); }
// InplaceNormal(selfRef, mean, std, seed) -> aclnnNormal(out=selfRef, mean, std, seed)
aclnnStatus aclnnInplaceNormalGetWorkspaceSize(aclTensor *selfRef, double mean, double std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnNormalGetWorkspaceSize(selfRef, mean, std, seed, ws, ex); }
aclnnStatus aclnnInplaceNormal(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnNormal(w, wss, e, s); }
// InplaceUniform(selfRef, from, to, seed) -> aclnnUniform(out=selfRef, from, to, seed)
aclnnStatus aclnnInplaceUniformGetWorkspaceSize(aclTensor *selfRef, double from, double to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnUniformGetWorkspaceSize(selfRef, from, to, seed, ws, ex); }
aclnnStatus aclnnInplaceUniform(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { return aclnnUniform(w, wss, e, s); }

} // extern "C" (delegation block 2)


// ============================================================================
// (B) HOST-DIRECT ops. Compute over unified memory after draining the stream.
//     Covers: comparisons (selfRef keeps its dtype), bitwise, and all ops whose
//     base does not exist in Metal (FillTensor, FloorDivides, FmodTensor,
//     IndexFillTensor, XLogY*, ClampMaxTensor, QuantScatterV2, schedulers, and
//     the RNG-with-tensor-args variants + InplaceRandom + RReluWithNoise).
// ============================================================================
namespace {

float   *FP (const aclTensor *t) { return (float *)t->data + t->offset; }
int32_t *I32(const aclTensor *t) { return (int32_t *)t->data + t->offset; }
int64_t *I64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
int8_t  *I8 (const aclTensor *t) { return (int8_t *)t->data + t->offset; }
uint8_t *U8 (const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }

// PCG-ish 32-bit hash + uniform in [0,1) (matches the style used in ops_ext.mm / random.mm)
inline uint32_t pcg(uint32_t v) { uint32_t s = v * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
inline float u01(uint32_t &s) { s = s * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; w = (w >> 22u) ^ w; return ((float)(w >> 8) + 0.5f) * (1.0f / 16777216.0f); }

// Read a single-element host-visible tensor as double (fp32/int32/int64).
double read_scalar_tensor(const aclTensor *t) {
    if (!t || !t->data) return 0.0;
    switch (t->dtype) {
        case ACL_FLOAT: return *FP(t);
        case ACL_INT32: return (double)*I32(t);
        case ACL_INT64: return (double)*I64(t);
        default: return 0.0;
    }
}

// op tags for the host dispatcher
enum GapOp {
    G_CMP = 0,        // comparison: dscalars[0]=cmp code, scalar value in dscalars[1] (scalar form) or tensor e->a
    G_BITWISE,        // bitwise: dscalars[0]=mode(0 and/1 or/2 xor), scalar in dscalars[1] or tensor e->a
    G_LOGICAL,        // logical and/or on bool: dscalars[0]=mode(0 and/1 or)
    G_FILLTENSOR,     // self <- broadcast scalar read from value tensor
    G_FLOORDIVIDES,   // self <- floor(self / scalar)
    G_FMODTENSOR,     // self <- fmod(self, other)
    G_INDEXFILLTENSOR,// self[index along dim0] rows <- scalar-from-tensor
    G_XLOGY_S,        // self <- self==0 ? 0 : self*log(scalar)
    G_XLOGY_T,        // self <- self==0 ? 0 : self*log(other)
    G_CLAMPMAXTENSOR, // self <- min(self, other) (broadcast not needed: same shape)
    G_QUANTSCATTER,   // int8 self[indices] = round(updates/scale)
    G_NOOP,           // scheduler bookkeeping: no tensor change
    G_RNG_RANDOM,     // self <- uniform integer in [from,to) cast to float
    G_RNG_UNIFORM_T,  // self <- uniform float in [from,to) (bounds from tensors / scalars)
    G_RNG_NORMAL_T,   // self <- normal with per-element mean/std tensors
    G_RNG_BERNOULLI_T,// self <- bernoulli with per-element prob tensor
    G_RRELU,          // self <- rrelu(self) writing noise into e->b
};

// Comparison codes: 0 gt,1 lt,2 eq,3 ne,4 ge,5 le
inline bool cmp_apply(int code, double x, double y) {
    switch (code) { case 0: return x > y; case 1: return x < y; case 2: return x == y;
                    case 3: return x != y; case 4: return x >= y; default: return x <= y; }
}

aclnnStatus run_parity(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    aclTensor *self = e->out;
    int64_t n = self ? self->numel() : 0;
    switch ((GapOp)e->m) {

    case G_CMP: {
        // Result written back into self, keeping self's own dtype (PyTorch semantics).
        int code = (int)e->dscalars[0];
        bool isT = (e->a != nullptr);
        // read self values into double, compute, write back as self dtype
        for (int64_t i = 0; i < n; ++i) {
            double x, y;
            // self holds the LHS values; read by dtype
            if (self->dtype == ACL_FLOAT)      x = FP(self)[i];
            else if (self->dtype == ACL_INT32) x = I32(self)[i];
            else                                x = (double)U8(self)[i];   // BOOL/UINT8
            if (isT) {
                if (e->a->dtype == ACL_FLOAT)      y = FP(e->a)[i];
                else if (e->a->dtype == ACL_INT32) y = I32(e->a)[i];
                else                                y = (double)U8(e->a)[i];
            } else y = e->dscalars[1];
            bool r = cmp_apply(code, x, y);
            if (self->dtype == ACL_FLOAT)      FP(self)[i]  = r ? 1.f : 0.f;
            else if (self->dtype == ACL_INT32) I32(self)[i] = r ? 1 : 0;
            else                                U8(self)[i]  = r ? 1 : 0;
        }
        return ACLNN_SUCCESS;
    }

    case G_BITWISE: {
        int mode = (int)e->dscalars[0];
        bool isT = (e->a != nullptr);
        int32_t sc = (int32_t)e->dscalars[1];
        int32_t *x = I32(self);
        const int32_t *o = isT ? I32(e->a) : nullptr;
        for (int64_t i = 0; i < n; ++i) {
            int32_t y = isT ? o[i] : sc;
            x[i] = mode == 0 ? (x[i] & y) : mode == 1 ? (x[i] | y) : (x[i] ^ y);
        }
        return ACLNN_SUCCESS;
    }

    case G_LOGICAL: {
        int mode = (int)e->dscalars[0];           // 0 and, 1 or
        uint8_t *x = U8(self); const uint8_t *o = U8(e->a);
        for (int64_t i = 0; i < n; ++i) {
            bool a = x[i] != 0, b = o[i] != 0;
            x[i] = (mode == 0 ? (a && b) : (a || b)) ? 1 : 0;
        }
        return ACLNN_SUCCESS;
    }

    case G_FILLTENSOR: {
        double v = read_scalar_tensor(e->a);
        if (self->dtype == ACL_FLOAT)      { float *x = FP(self);  for (int64_t i=0;i<n;++i) x[i]=(float)v; }
        else if (self->dtype == ACL_INT32) { int32_t *x = I32(self); for (int64_t i=0;i<n;++i) x[i]=(int32_t)v; }
        else                                { uint8_t *x = U8(self); for (int64_t i=0;i<n;++i) x[i]=(uint8_t)v; }
        return ACLNN_SUCCESS;
    }

    case G_FLOORDIVIDES: {
        double d = e->dscalars[0]; float *x = FP(self);
        for (int64_t i = 0; i < n; ++i) x[i] = std::floor(x[i] / (float)d);
        return ACLNN_SUCCESS;
    }

    case G_FMODTENSOR: {
        float *x = FP(self); const float *o = FP(e->a);
        for (int64_t i = 0; i < n; ++i) x[i] = std::fmod(x[i], o[i]);
        return ACLNN_SUCCESS;
    }

    case G_INDEXFILLTENSOR: {
        // dim-0 row fill (matches CUDA idx0 semantics): rows selected by index[] set to scalar from value tensor.
        double v = read_scalar_tensor(e->b);
        const int64_t *idx = I64(e->a); int64_t L = e->a->numel();
        int64_t rows = self->viewDims.empty() ? 0 : self->viewDims[0];
        int64_t row = rows ? n / rows : n;
        float *x = FP(self);
        for (int64_t l = 0; l < L; ++l) { int64_t r = idx[l]; if (r < 0) r += rows; if (r < 0 || r >= rows) continue;
            for (int64_t c = 0; c < row; ++c) x[r * row + c] = (float)v; }
        return ACLNN_SUCCESS;
    }

    case G_XLOGY_S: {
        double y = e->dscalars[0]; float *x = FP(self); float ly = std::log((float)y);
        for (int64_t i = 0; i < n; ++i) x[i] = x[i] == 0.f ? 0.f : x[i] * ly;
        return ACLNN_SUCCESS;
    }

    case G_XLOGY_T: {
        float *x = FP(self); const float *o = FP(e->a);
        for (int64_t i = 0; i < n; ++i) x[i] = x[i] == 0.f ? 0.f : x[i] * std::log(o[i]);
        return ACLNN_SUCCESS;
    }

    case G_CLAMPMAXTENSOR: {
        float *x = FP(self); const float *o = FP(e->a);
        for (int64_t i = 0; i < n; ++i) x[i] = std::min(x[i], o[i]);
        return ACLNN_SUCCESS;
    }

    case G_QUANTSCATTER: {
        // self int8 [N, D]; indices int64 [K]; updates fp32 [K, D]; self[indices[k], :] = round(updates[k,:]/scale)
        int64_t K = e->b->numel();
        int64_t D = K ? e->c->numel() / K : 0;
        int64_t N = D ? self->numel() / D : 0;
        const int64_t *idx = I64(e->b); const float *upd = FP(e->c); int8_t *dst = I8(self);
        double scale = e->alpha == 0.0 ? 1.0 : e->alpha;
        for (int64_t k = 0; k < K; ++k) { int64_t r = idx[k]; if (r < 0) r += N; if (r < 0 || r >= N) continue;
            for (int64_t d = 0; d < D; ++d) { long q = std::lround(upd[k * D + d] / scale);
                if (q > 127) q = 127; if (q < -128) q = -128; dst[r * D + d] = (int8_t)q; } }
        return ACLNN_SUCCESS;
    }

    case G_NOOP:
        return ACLNN_SUCCESS;   // scheduler bookkeeping; tensor unchanged

    case G_RNG_RANDOM: {
        // integer-valued uniform in [from,to) cast to float; from=dscalars[0], to=dscalars[1], seed=dim
        double lo = e->dscalars[0], hi = e->dscalars[1]; uint32_t seed = (uint32_t)(int64_t)e->dim;
        double span = hi - lo; if (span <= 0) span = 1;
        float *x = FP(self);
        for (int64_t i = 0; i < n; ++i) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u));
            x[i] = (float)((int64_t)(lo + std::floor(u01(st) * span))); }
        return ACLNN_SUCCESS;
    }

    case G_RNG_UNIFORM_T: {
        double lo = e->dscalars[0], hi = e->dscalars[1]; uint32_t seed = (uint32_t)(int64_t)e->dim;
        float *x = FP(self);
        for (int64_t i = 0; i < n; ++i) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u));
            x[i] = (float)(lo + u01(st) * (hi - lo)); }
        return ACLNN_SUCCESS;
    }

    case G_RNG_NORMAL_T: {
        uint32_t seed = (uint32_t)(int64_t)e->dim; float *x = FP(self);
        const float *mean = e->a ? FP(e->a) : nullptr, *stdv = e->b ? FP(e->b) : nullptr;
        for (int64_t i = 0; i < n; ++i) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u));
            float u1 = std::max(u01(st), 1e-7f), u2 = u01(st);
            float z = std::sqrt(-2.f * std::log(u1)) * std::cos(6.2831853f * u2);
            float m = mean ? mean[i] : 0.f, sd = stdv ? stdv[i] : 1.f;
            x[i] = m + sd * z; }
        return ACLNN_SUCCESS;
    }

    case G_RNG_BERNOULLI_T: {
        uint32_t seed = (uint32_t)(int64_t)e->dim; float *x = FP(self); const float *p = FP(e->a);
        for (int64_t i = 0; i < n; ++i) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u));
            x[i] = u01(st) < p[i] ? 1.f : 0.f; }
        return ACLNN_SUCCESS;
    }

    case G_RRELU: {
        // training: noise[i] = (x<0)? U(lower,upper) : 1; out = x>=0? x : x*noise.
        // inference: out = x>=0? x : x*(lower+upper)/2 ; noise filled with that midslope where x<0.
        double lo = e->dscalars[0], hi = e->dscalars[1]; bool training = e->causal;
        uint32_t seed = (uint32_t)(int64_t)e->dim; float *x = FP(self);
        float *noise = e->b ? FP(e->b) : nullptr;
        for (int64_t i = 0; i < n; ++i) {
            float v = x[i];
            if (v >= 0.f) { if (noise) noise[i] = 1.f; x[i] = v; }
            else {
                float slope;
                if (training) { uint32_t st = pcg(((uint32_t)seed * 2654435761u) ^ ((uint32_t)i * 40503u + 1u)); slope = (float)(lo + u01(st) * (hi - lo)); }
                else slope = (float)((lo + hi) * 0.5);
                if (noise) noise[i] = slope; x[i] = v * slope;
            }
        }
        return ACLNN_SUCCESS;
    }

    default: return ACLNN_ERR_PARAM_INVALID;
    }
}

// Common Execute trampoline for host-direct ops.
#define GAP_RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_parity(e, s); } delete e; return st; }

} // namespace

extern "C" {

// ---- comparisons (selfRef keeps its dtype; write 0/1) ----
#define GAP_CMP_S(NAME, CODE) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *scalar, uint64_t *ws, aclOpExecutor **ex) { \
    if (!selfRef || !scalar || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = G_CMP; e->out = selfRef; e->dscalars = {(double)CODE, scalar->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
GAP_RUN(NAME)
#define GAP_CMP_T(NAME, CODE) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = G_CMP; e->out = selfRef; e->a = other; e->dscalars = {(double)CODE, 0.0}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
GAP_RUN(NAME)

GAP_CMP_S(aclnnInplaceGtScalar, 0) GAP_CMP_S(aclnnInplaceLtScalar, 1) GAP_CMP_S(aclnnInplaceEqScalar, 2)
GAP_CMP_S(aclnnInplaceNeScalar, 3) GAP_CMP_S(aclnnInplaceGeScalar, 4) GAP_CMP_S(aclnnInplaceLeScalar, 5)
GAP_CMP_T(aclnnInplaceGtTensor, 0) GAP_CMP_T(aclnnInplaceLtTensor, 1) GAP_CMP_T(aclnnInplaceEqTensor, 2)
GAP_CMP_T(aclnnInplaceNeTensor, 3) GAP_CMP_T(aclnnInplaceGeTensor, 4) GAP_CMP_T(aclnnInplaceLeTensor, 5)

// ---- bitwise (int32 self) ----
#define GAP_BIT_S(NAME, MODE) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *scalar, uint64_t *ws, aclOpExecutor **ex) { \
    if (!selfRef || !scalar || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = G_BITWISE; e->out = selfRef; e->dscalars = {(double)MODE, scalar->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
GAP_RUN(NAME)
#define GAP_BIT_T(NAME, MODE) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = G_BITWISE; e->out = selfRef; e->a = other; e->dscalars = {(double)MODE, 0.0}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
GAP_RUN(NAME)

GAP_BIT_S(aclnnInplaceBitwiseAndScalar, 0) GAP_BIT_S(aclnnInplaceBitwiseOrScalar, 1) GAP_BIT_S(aclnnInplaceBitwiseXorScalar, 2)
GAP_BIT_T(aclnnInplaceBitwiseAndTensor, 0) GAP_BIT_T(aclnnInplaceBitwiseOrTensor, 1) GAP_BIT_T(aclnnInplaceBitwiseXorTensor, 2)
// BitwiseAndTensorOut: alias of AndTensor in-place (same selfRef result; "Out" naming, no separate out param)
GAP_BIT_T(aclnnInplaceBitwiseAndTensorOut, 0)

// ---- logical and/or on bool ----
#define GAP_LOGIC(NAME, MODE) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->m = G_LOGICAL; e->out = selfRef; e->a = other; e->dscalars = {(double)MODE}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
GAP_RUN(NAME)
GAP_LOGIC(aclnnInplaceLogicalAnd, 0) GAP_LOGIC(aclnnInplaceLogicalOr, 1)

// ---- elementwise ops without a Metal base ----
aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !value || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_FILLTENSOR; e->out = selfRef; e->a = value; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceFillTensor)

aclnnStatus aclnnInplaceFloorDividesGetWorkspaceSize(aclTensor *selfRef, const aclScalar *scalar, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !scalar || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_FLOORDIVIDES; e->out = selfRef; e->dscalars = {scalar->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceFloorDivides)

aclnnStatus aclnnInplaceFmodTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_FMODTENSOR; e->out = selfRef; e->a = other; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceFmodTensor)

aclnnStatus aclnnInplaceIndexFillTensorGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclTensor *value, uint64_t *ws, aclOpExecutor **ex) {
    (void)dim; if (!selfRef || !index || !value || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_INDEXFILLTENSOR; e->out = selfRef; e->a = index; e->b = value; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceIndexFillTensor)

aclnnStatus aclnnInplaceXLogYScalarOtherGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_XLOGY_S; e->out = selfRef; e->dscalars = {other->v}; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceXLogYScalarOther)

aclnnStatus aclnnInplaceXLogYTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_XLOGY_T; e->out = selfRef; e->a = other; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceXLogYTensor)

aclnnStatus aclnnInplaceClampMaxTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !other || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_CLAMPMAXTENSOR; e->out = selfRef; e->a = other; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceClampMaxTensor)

aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !indices || !updates || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (selfRef->dtype != ACL_INT8 || updates->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->m = G_QUANTSCATTER; e->out = selfRef; e->b = indices; e->c = updates; e->alpha = scale; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceQuantScatterV2)

// ---- scheduler bookkeeping no-ops ----
aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = G_NOOP; e->out = selfRef; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceAttentionWorkerScheduler)
aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = G_NOOP; e->out = selfRef; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceFfnWorkerScheduler)

// ---- RNG variants with tensor-borne args / integer fill / rrelu ----
aclnnStatus aclnnInplaceRandomGetWorkspaceSize(aclTensor *selfRef, int64_t from, int64_t to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_RNG_RANDOM; e->out = selfRef; e->dscalars = {(double)from, (double)to}; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceRandom)

aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    double lo = from ? read_scalar_tensor(from) : 0.0, hi = to ? read_scalar_tensor(to) : 1.0;
    auto *e = new aclOpExecutor(); e->m = G_RNG_RANDOM; e->out = selfRef; e->dscalars = {lo, hi}; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceRandomTensor)

aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    double lo = from ? read_scalar_tensor(from) : 0.0, hi = to ? read_scalar_tensor(to) : 1.0;
    auto *e = new aclOpExecutor(); e->m = G_RNG_UNIFORM_T; e->out = selfRef; e->dscalars = {lo, hi}; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceUniformTensor)

aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mean, const aclTensor *std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_RNG_NORMAL_T; e->out = selfRef; e->a = mean; e->b = std; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceNormalTensor)

aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *prob, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !prob || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_RNG_BERNOULLI_T; e->out = selfRef; e->a = prob; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceBernoulliTensor)

aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(aclTensor *selfRef, aclTensor *noise, double lower, double upper, bool training, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = G_RRELU; e->out = selfRef; e->b = noise; e->dscalars = {lower, upper}; e->causal = training; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
GAP_RUN(aclnnInplaceRReluWithNoise)

} // extern "C"
