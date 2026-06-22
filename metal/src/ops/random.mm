// RNG / distribution host side. Fills tensors with samples; statistical-tolerance verified.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"

namespace {
struct RngMeta { uint32_t n; int32_t op; uint32_t seed; float p0; float p1; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus launch(NSString *k, NSUInteger n, aclrtStream stream, std::initializer_list<const aclTensor *> ts, const RngMeta &m) {
    id<MTLComputePipelineState> pso = mtl::pipeline(k);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    NSUInteger idx = 0;
    for (const aclTensor *t : ts) { size_t off; id<MTLBuffer> b = buf_of(t, &off); if (!b) { [enc endEncoding]; return ACLNN_ERR_RUNTIME_ERROR; } [enc setBuffer:b offset:off atIndex:idx++]; }
    [enc setBytes:&m length:sizeof(m) atIndex:idx];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
// op>=0: rng_f32; -1 randint; -2 poisson; -3 normal_tt; -4 multinomial
aclnnStatus run_rng(aclOpExecutor *e, aclrtStream s) {
    RngMeta m{ (uint32_t)(e->out ? e->out->numel() : 0), (int32_t)e->m, (uint32_t)(int64_t)e->dim,
               (float)(e->dscalars.size() > 0 ? e->dscalars[0] : 0.0), (float)(e->dscalars.size() > 1 ? e->dscalars[1] : 0.0) };
    switch (e->m) {
        case -1: return launch(@"rng_randint", m.n, s, {e->out}, m);
        case -2: return launch(@"rng_poisson", m.n, s, {e->out}, m);
        case -3: return launch(@"rng_normal_tt", m.n, s, {e->out, e->a, e->b}, m);
        case -4: { m.p0 = (float)e->a->viewDims.back(); return launch(@"rng_multinomial", m.n, s, {e->out, e->a}, m); }
        default: return launch(@"rng_f32", m.n, s, {e->out}, m);
    }
}
#define RUN(NAME) extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_rng(e, s); } delete e; return st; }
} // namespace

extern "C" {
// out + 1 or 2 scalar params + seed
#define DEF_RNG1(NAME, OP) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *out, double a, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { \
    if (!out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = OP; e->out = out; e->dim = seed; e->dscalars = {a}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)
#define DEF_RNG2(NAME, OP) \
aclnnStatus NAME##GetWorkspaceSize(aclTensor *out, double a, double b, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { \
    if (!out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = OP; e->out = out; e->dim = seed; e->dscalars = {a, b}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME)

DEF_RNG1(aclnnExponential, 0)
DEF_RNG1(aclnnGeometric, 1)
DEF_RNG2(aclnnLogNormal, 3)
DEF_RNG2(aclnnCauchy, 4)
DEF_RNG1(aclnnPoisson, -2)

aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(double mean, double std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = 2; e->out = out; e->dim = seed; e->dscalars = {mean, std}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnNormalFloatFloat)
aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(const aclTensor *mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!mean || !std || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = -3; e->out = out; e->a = mean; e->b = std; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnNormalTensorTensor)
aclnnStatus aclnnRandIntGetWorkspaceSize(int64_t low, int64_t high, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = -1; e->out = out; e->dim = seed; e->dscalars = {(double)low, (double)high}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnRandInt)
aclnnStatus aclnnMultinomialGetWorkspaceSize(const aclTensor *probs, int64_t numSamples, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)numSamples; if (!probs || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = -4; e->out = out; e->a = probs; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnMultinomial)
} // extern "C"
