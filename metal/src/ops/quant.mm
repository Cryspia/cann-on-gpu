// Quantization family: Quantize/Dequantize (int8 affine), DynamicQuant (per-row symmetric int8),
// Fp6Pack/Unpack (4-codes-in-3-bytes). Host compute over unified memory (exact) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
#include <algorithm>

namespace {
const float *fp(const aclTensor *t) { return (const float *)t->data + t->offset; }
float *fpw(const aclTensor *t) { return (float *)t->data + t->offset; }
int8_t *i8(const aclTensor *t) { return (int8_t *)t->data + t->offset; }
const uint8_t *u8(const aclTensor *t) { return (const uint8_t *)t->data + t->offset; }
uint8_t *u8w(const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
inline int clampi(long v) { return (int)(v < -128 ? -128 : v > 127 ? 127 : v); }

aclnnStatus run_quant(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    switch (e->m) {
        case 0: {   // Quantize: q = clamp(round(x*scale + offset))
            const aclTensor *X = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
            int64_t n = X->numel(); const float *x = fp(X); float sc = fp(Sc)[0], of = Of ? fp(Of)[0] : 0.f; int8_t *o = i8(O);
            for (int64_t i = 0; i < n; ++i) o[i] = (int8_t)clampi(std::lrint(x[i] * sc + of));
            break;
        }
        case 1: {   // Dequantize: y = q*scale + offset
            const aclTensor *Q = e->a, *Sc = e->b, *Of = e->c; aclTensor *O = e->out;
            int64_t n = Q->numel(); const int8_t *q = (const int8_t *)Q->data + Q->offset; float sc = fp(Sc)[0], of = Of ? fp(Of)[0] : 0.f; float *o = fpw(O);
            for (int64_t i = 0; i < n; ++i) o[i] = (float)q[i] * sc + of;
            break;
        }
        case 2: {   // DynamicQuant: per-row symmetric int8, scale = amax/127
            const aclTensor *X = e->a; aclTensor *O = e->out, *Sco = e->out2;
            int64_t rows = X->viewDims[0], D = X->numel() / rows; const float *x = fp(X); int8_t *o = i8(O); float *sco = fpw(Sco);
            for (int64_t r = 0; r < rows; ++r) {
                float amax = 0; for (int64_t d = 0; d < D; ++d) amax = std::max(amax, std::fabs(x[r * D + d]));
                float sc = amax / 127.f; if (sc == 0) sc = 1; sco[r] = sc;
                for (int64_t d = 0; d < D; ++d) o[r * D + d] = (int8_t)clampi(std::lrint(x[r * D + d] / sc));
            }
            break;
        }
        case 3: {   // Fp6Pack: 4 six-bit codes -> 3 bytes
            const aclTensor *S = e->a; aclTensor *O = e->out; int64_t n = S->numel(); const uint8_t *in = u8(S); uint8_t *out = u8w(O);
            int64_t groups = (n + 3) / 4;
            for (int64_t gp = 0; gp < groups; ++gp) {
                uint32_t bits = 0;
                for (int j = 0; j < 4; ++j) { int64_t idx = gp * 4 + j; uint32_t c = (idx < n) ? (in[idx] & 0x3f) : 0; bits |= c << (6 * j); }
                out[gp * 3 + 0] = bits & 0xff; out[gp * 3 + 1] = (bits >> 8) & 0xff; out[gp * 3 + 2] = (bits >> 16) & 0xff;
            }
            break;
        }
        case 4: {   // Fp6Unpack: 3 bytes -> 4 six-bit codes
            const aclTensor *P = e->a; aclTensor *O = e->out; int64_t n = O->numel(); const uint8_t *in = u8(P); uint8_t *out = u8w(O);
            int64_t groups = (n + 3) / 4;
            for (int64_t gp = 0; gp < groups; ++gp) {
                uint32_t bits = in[gp * 3 + 0] | (in[gp * 3 + 1] << 8) | (in[gp * 3 + 2] << 16);
                for (int j = 0; j < 4; ++j) { int64_t idx = gp * 4 + j; if (idx < n) out[idx] = (bits >> (6 * j)) & 0x3f; }
            }
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_quant(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnQuantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 0; e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnQuantize)
aclnnStatus aclnnDequantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !scale || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 1; e->a = x; e->b = scale; e->c = offset; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDequantize)
aclnnStatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !scaleOut || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 2; e->a = x; e->out = out; e->out2 = scaleOut; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDynamicQuant)
aclnnStatus aclnnFp6PackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 3; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFp6Pack)
aclnnStatus aclnnFp6UnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 4; e->a = self; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFp6Unpack)
} // extern "C"
