// FFT / signal family: Fft / Ifft (complex, interleaved [..,n,2]) + Rfft / Irfft (real <-> half spectrum).
// Naive DFT host-side over unified memory in double (test sizes are small; tol 1e-4). Transform is the last axis.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>

namespace {
float *fp(const aclTensor *t) { return (float *)t->data + t->offset; }
constexpr double PI = 3.14159265358979323846;

aclnnStatus run_fft(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    const aclTensor *x = e->a; aclTensor *o = e->out;
    if (!x || !o || !x->data || !o->data) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t n = e->n; if (n <= 0) return ACLNN_ERR_PARAM_INVALID;
    const float *in = fp(x); float *out = fp(o);
    switch (e->m) {
        case 0: case 1: {   // fft / ifft: [B,n,2] complex
            double sgn = (e->m == 1) ? +1.0 : -1.0, nrm = (e->m == 1) ? 1.0 / (double)n : 1.0;
            int64_t B = x->numel() / (2 * n);
            for (int64_t b = 0; b < B; ++b) for (int64_t k = 0; k < n; ++k) {
                double re = 0, im = 0;
                for (int64_t j = 0; j < n; ++j) {
                    double ang = sgn * 2 * PI * (double)k * j / n, c = std::cos(ang), si = std::sin(ang);
                    double xr = in[(b * n + j) * 2], xi = in[(b * n + j) * 2 + 1];
                    re += xr * c - xi * si; im += xr * si + xi * c;
                }
                out[(b * n + k) * 2] = (float)(re * nrm); out[(b * n + k) * 2 + 1] = (float)(im * nrm);
            }
            break;
        }
        case 2: {   // rfft: real [B,n] -> [B,nc,2]
            int64_t nc = n / 2 + 1, B = x->numel() / n;
            for (int64_t b = 0; b < B; ++b) for (int64_t k = 0; k < nc; ++k) {
                double re = 0, im = 0;
                for (int64_t j = 0; j < n; ++j) { double ang = -2 * PI * (double)k * j / n; re += in[b * n + j] * std::cos(ang); im += in[b * n + j] * std::sin(ang); }
                out[(b * nc + k) * 2] = (float)re; out[(b * nc + k) * 2 + 1] = (float)im;
            }
            break;
        }
        case 3: {   // irfft: [B,nc,2] -> real [B,n] (hermitian extension, 1/n)
            int64_t nc = n / 2 + 1, B = x->numel() / (2 * nc);
            for (int64_t b = 0; b < B; ++b) for (int64_t j = 0; j < n; ++j) {
                double acc = 0;
                for (int64_t k = 0; k < n; ++k) {
                    int64_t kk = (k < nc) ? k : (n - k);
                    double Xr = in[(b * nc + kk) * 2], Xi = in[(b * nc + kk) * 2 + 1];
                    if (k >= nc) Xi = -Xi;
                    double ang = 2 * PI * (double)k * j / n;
                    acc += Xr * std::cos(ang) - Xi * std::sin(ang);
                }
                out[b * n + j] = (float)(acc / (double)n);
            }
            break;
        }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_fft(e, s); } delete e; return st; }
} // namespace

extern "C" {
#define DEF_FFT(NAME, K) aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { \
    if (!x || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; auto *e = new aclOpExecutor(); e->m = K; e->a = x; e->out = out; e->n = n; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUN(NAME)
DEF_FFT(aclnnFft, 0) DEF_FFT(aclnnIfft, 1) DEF_FFT(aclnnRfft, 2) DEF_FFT(aclnnIrfft, 3)
} // extern "C"
