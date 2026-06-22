// RNN family: single-layer LSTM / GRU forward. Sequential recurrence host-side over unified memory
// (small, exact) after draining the stream. Weights: gates stacked row-major [G*H, *].
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>

namespace {
const float *fp(const aclTensor *t) { return (const float *)t->data + t->offset; }
float *fpw(const aclTensor *t) { return (float *)t->data + t->offset; }
double sigf(double x) { return 1.0 / (1.0 + std::exp(-x)); }

aclnnStatus run_rnn(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    const aclTensor *x = e->inputs[0], *wih = e->inputs[1], *whh = e->inputs[2], *bih = e->inputs[3], *bhh = e->inputs[4];
    int T = (int)x->viewDims[0], B = (int)x->viewDims[1], I = (int)x->viewDims[2], H = (int)whh->viewDims[1];
    const float *X = fp(x), *Wih = fp(wih), *Whh = fp(whh), *Bih = fp(bih), *Bhh = fp(bhh);

    if (e->m == 0) {   // LSTM: gates i,f,g,o (rows g*H+hh)
        const float *h0 = fp(e->inputs[5]), *c0 = fp(e->inputs[6]);
        float *Y = fpw((aclTensor *)e->inputs[7]), *hN = fpw((aclTensor *)e->inputs[8]), *cN = fpw((aclTensor *)e->inputs[9]);
        std::vector<double> h(B * H), c(B * H); for (int i = 0; i < B * H; ++i) { h[i] = h0[i]; c[i] = c0[i]; }
        for (int t = 0; t < T; ++t) {
            std::vector<double> hn(B * H), cn(B * H);
            for (int b = 0; b < B; ++b) for (int hh = 0; hh < H; ++hh) {
                double g[4];
                for (int gate = 0; gate < 4; ++gate) {
                    int row = gate * H + hh; double acc = Bih[row] + Bhh[row];
                    for (int i = 0; i < I; ++i) acc += (double)Wih[row * I + i] * X[(t * B + b) * I + i];
                    for (int k = 0; k < H; ++k) acc += (double)Whh[row * H + k] * h[b * H + k];
                    g[gate] = acc;
                }
                double gi = sigf(g[0]), gf = sigf(g[1]), gg = std::tanh(g[2]), go = sigf(g[3]);
                double cc = gf * c[b * H + hh] + gi * gg; cn[b * H + hh] = cc;
                double hc = go * std::tanh(cc); hn[b * H + hh] = hc; Y[(t * B + b) * H + hh] = (float)hc;
            }
            h = hn; c = cn;
        }
        for (int i = 0; i < B * H; ++i) { hN[i] = (float)h[i]; cN[i] = (float)c[i]; }
    } else {   // GRU: gates r,z,n
        const float *h0 = fp(e->inputs[5]);
        float *Y = fpw((aclTensor *)e->inputs[6]), *hN = fpw((aclTensor *)e->inputs[7]);
        std::vector<double> h(B * H); for (int i = 0; i < B * H; ++i) h[i] = h0[i];
        for (int t = 0; t < T; ++t) {
            std::vector<double> hn(B * H);
            for (int b = 0; b < B; ++b) for (int hh = 0; hh < H; ++hh) {
                double xr = Bih[0 * H + hh], xz = Bih[1 * H + hh], xn = Bih[2 * H + hh];
                double hr = Bhh[0 * H + hh], hz = Bhh[1 * H + hh], hnn = Bhh[2 * H + hh];
                for (int i = 0; i < I; ++i) { double xi = X[(t * B + b) * I + i];
                    xr += (double)Wih[(0 * H + hh) * I + i] * xi; xz += (double)Wih[(1 * H + hh) * I + i] * xi; xn += (double)Wih[(2 * H + hh) * I + i] * xi; }
                for (int k = 0; k < H; ++k) { double hk = h[b * H + k];
                    hr += (double)Whh[(0 * H + hh) * H + k] * hk; hz += (double)Whh[(1 * H + hh) * H + k] * hk; hnn += (double)Whh[(2 * H + hh) * H + k] * hk; }
                double r = sigf(xr + hr), z = sigf(xz + hz), nn = std::tanh(xn + r * hnn);
                double hc = (1 - z) * nn + z * h[b * H + hh]; hn[b * H + hh] = hc; Y[(t * B + b) * H + hh] = (float)hc;
            }
            h = hn;
        }
        for (int i = 0; i < B * H; ++i) hN[i] = (float)h[i];
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_rnn(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnLstmGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih,
        const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN,
        uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 0; e->inputs = {x, wih, whh, bih, bhh, h0, c0, y, hN, cN}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnLstm)
aclnnStatus aclnnGruGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih,
        const aclTensor *bhh, const aclTensor *h0, aclTensor *y, aclTensor *hN, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !y || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->m = 1; e->inputs = {x, wih, whh, bih, bhh, h0, y, hN}; *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
RUN(aclnnGru)
} // extern "C"
