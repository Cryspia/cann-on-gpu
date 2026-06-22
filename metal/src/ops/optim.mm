// Optimizer step host side. In-place param/state updates. ops: Adam/Adagrad/Rmsprop/Momentum/Adamax/Adadelta/ClipGradNorm.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
#include <vector>

namespace {
struct OptMeta { uint32_t n; float p[7]; int32_t flag; };
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus launch(NSString *k, NSUInteger n, aclrtStream stream,
                   std::vector<const aclTensor *> tensors, const OptMeta &m) {
    id<MTLComputePipelineState> pso = mtl::pipeline(k);
    if (!pso) return ACLNN_ERR_RUNTIME_ERROR;
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    for (NSUInteger i = 0; i < tensors.size(); ++i) {
        size_t off; id<MTLBuffer> b = buf_of(tensors[i], &off);
        if (!b) { [enc endEncoding]; return ACLNN_ERR_RUNTIME_ERROR; }
        [enc setBuffer:b offset:off atIndex:i];
    }
    [enc setBytes:&m length:sizeof(m) atIndex:tensors.size()];
    NSUInteger tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding]; [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
aclnnStatus run_optim(aclOpExecutor *e, aclrtStream s) {
    OptMeta m{}; m.flag = e->keepDim ? 1 : 0;
    const aclTensor *param = e->out, *a = e->a, *b = e->b, *g = e->c;
    NSUInteger n = param ? (NSUInteger)param->numel() : 0;
    for (size_t i = 0; i < e->dscalars.size() && i < 7; ++i) m.p[i] = (float)e->dscalars[i];
    m.n = (uint32_t)n;
    switch (e->m) {
        case 0: return launch(@"opt_adam",     n, s, {param, a, b, g}, m);
        case 1: return launch(@"opt_adagrad",  n, s, {param, a, g},    m);
        case 2: return launch(@"opt_rmsprop",  n, s, {param, a, g},    m);
        case 3: return launch(@"opt_momentum", n, s, {param, a, g},    m);
        case 4: return launch(@"opt_adamax",   n, s, {param, a, b, g}, m);
        case 5: return launch(@"opt_adadelta", n, s, {param, a, b, g}, m);
        case 6: { m.n = (uint32_t)(e->a ? e->a->numel() : 0);   // clip: grad in e->a, normOut in e->out
                  return launch(@"clip_grad", 1, s, {e->a, e->out}, m); }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUN(NAME) extern "C" aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_optim(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnApplyAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
        double lr, double b1, double b2, double eps, double wd, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !v || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 0; e->out = param; e->a = m; e->b = v; e->c = grad;
    e->dscalars = {lr, b1, b2, eps, wd, 1.0 - std::pow(b1, (double)step), 1.0 - std::pow(b2, (double)step)};
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdagradGetWorkspaceSize(aclTensor *param, aclTensor *stateSum, const aclTensor *grad,
        double lr, double eps, double wd, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !stateSum || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 1; e->out = param; e->a = stateSum; e->c = grad;
    e->dscalars = {lr, eps, wd}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyRmspropGetWorkspaceSize(aclTensor *param, aclTensor *v, const aclTensor *grad,
        double lr, double alpha, double eps, double wd, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !v || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 2; e->out = param; e->a = v; e->c = grad;
    e->dscalars = {lr, alpha, eps, wd}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyMomentumGetWorkspaceSize(aclTensor *param, aclTensor *buf, const aclTensor *grad,
        double lr, double momentum, double wd, double damp, bool nesterov, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !buf || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 3; e->out = param; e->a = buf; e->c = grad; e->keepDim = nesterov;
    e->dscalars = {lr, momentum, wd, damp}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdamaxGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *u, const aclTensor *grad,
        double lr, double b1, double b2, double eps, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !m || !u || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 4; e->out = param; e->a = m; e->b = u; e->c = grad;
    e->dscalars = {lr, b1, b2, eps, 1.0 - std::pow(b1, (double)step)}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnApplyAdadeltaGetWorkspaceSize(aclTensor *param, aclTensor *squareAvg, aclTensor *accDelta, const aclTensor *grad,
        double lr, double rho, double eps, double wd, uint64_t *ws, aclOpExecutor **ex) {
    if (!param || !squareAvg || !accDelta || !grad || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 5; e->out = param; e->a = squareAvg; e->b = accDelta; e->c = grad;
    e->dscalars = {lr, rho, eps, wd}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnClipGradNormGetWorkspaceSize(aclTensor *grad, double maxNorm, aclTensor *totalNormOut,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!grad || !totalNormOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 6; e->a = grad; e->out = totalNormOut;
    e->dscalars = {maxNorm}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnApplyAdam) RUN(aclnnApplyAdagrad) RUN(aclnnApplyRmsprop) RUN(aclnnApplyMomentum)
RUN(aclnnApplyAdamax) RUN(aclnnApplyAdadelta) RUN(aclnnClipGradNorm)
} // extern "C"
