// Dtype conversion host side. aclnnCast(self, dtype, out).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "subfp.h"
#include <cstring>

namespace {
struct CastMeta { uint32_t n; };
const char *code(aclDataType dt) {
    switch (dt) { case ACL_FLOAT: return "f32"; case ACL_FLOAT16: return "f16"; case ACL_INT32: return "i32";
                  case ACL_INT64: return "i64"; case ACL_BOOL: return "b8"; default: return nullptr; }
}
id<MTLBuffer> buf_of(const aclTensor *t, size_t *off) {
    id<MTLBuffer> b = mtl::bufferFor(t->data, off);
    if (b && off) *off += (size_t)t->offset * dtype_size(t->dtype);
    return b;
}
aclnnStatus run_cast(aclOpExecutor *e, aclrtStream stream) {
    const aclTensor *a = e->a; aclTensor *o = e->out;
    if (!a || !o || !a->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    // fp8 / fp4 / fp6 / hif8 <-> fp32 conversions: host-side codec over unified memory (bit-exact).
    if (subfp::is_low(a->dtype) || subfp::is_low(o->dtype)) {
        auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
        int64_t n = a->numel();
        if (subfp::is_low(o->dtype)) {                       // float -> low
            const float *in = (const float *)a->data + a->offset;
            uint8_t *out = (uint8_t *)o->data;
            memset(out, 0, (size_t)subfp::bytes(o->dtype, n));   // fp4 packs into nibbles
            for (int64_t i = 0; i < n; ++i) subfp::store(o->dtype, out, i, in[i]);
        } else {                                             // low -> float
            const uint8_t *in = (const uint8_t *)a->data;
            float *out = (float *)o->data + o->offset;
            for (int64_t i = 0; i < n; ++i) out[i] = subfp::load(a->dtype, in, i);
        }
        return ACLNN_SUCCESS;
    }
    const char *sc = code(a->dtype), *dc = code(o->dtype);
    if (!sc || !dc) return ACLNN_ERR_PARAM_INVALID;
    NSString *k;
    if (a->dtype == o->dtype) k = nil;   // identity -> plain copy below
    else k = [NSString stringWithFormat:@"cast_%s_%s", sc, dc];
    size_t oa, oo; id<MTLBuffer> ba = buf_of(a, &oa), bo = buf_of(o, &oo);
    if (!ba || !bo) return ACLNN_ERR_RUNTIME_ERROR;
    CastMeta m{ (uint32_t)a->numel() };
    auto *s = (AclStream *)stream;
    id<MTLCommandQueue> q = s ? s->q : mtl::defaultQueue();
    id<MTLCommandBuffer> cb = [q commandBuffer];
    if (!k) {  // same dtype: blit copy
        id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
        [bl copyFromBuffer:ba sourceOffset:oa toBuffer:bo destinationOffset:oo size:(NSUInteger)a->numel() * dtype_size(a->dtype)];
        [bl endEncoding];
    } else {
        id<MTLComputePipelineState> pso = mtl::pipeline(k);
        if (!pso) return ACLNN_ERR_PARAM_INVALID;
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:pso];
        [enc setBuffer:ba offset:oa atIndex:0]; [enc setBuffer:bo offset:oo atIndex:1]; [enc setBytes:&m length:sizeof(m) atIndex:2];
        NSUInteger n = m.n, tg = pso.maxTotalThreadsPerThreadgroup; if (tg > 256) tg = 256; if (tg > n) tg = n ? n : 1;
        [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [enc endEncoding];
    }
    [cb commit];
    if (s) s->last = cb; else [cb waitUntilCompleted];
    return ACLNN_SUCCESS;
}
} // namespace

extern "C" {
aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, aclDataType dtype, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)dtype; if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_CAST; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCast(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    aclnnStatus st; @autoreleasepool { st = run_cast(e, s); } delete e; return st;
}
} // extern "C"
