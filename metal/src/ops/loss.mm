// Loss family (P10): forward losses (L1/SmoothL1/Huber/SoftMargin/KlDiv/BCE/MarginRanking/HingeEmbedding),
// loss/activation backward (L1/SmoothL1/Swish/Softsign/Logit/Prelu/BCEWithLogits/CrossEntropyLossGrad),
// fused/structured ops (ScaledMaskedSoftmax/GridSampler2DBackward/ThreeInterpolateBackward/
// ThnnFusedLstmCell/ExpSegsum). Host-side over unified memory (device ptr == host ptr under
// MTLStorageModeShared). Each op uses the standard two-phase aclnn contract: GetWorkspaceSize stashes
// the plan in aclOpExecutor and sets *ws=0; Execute drains the stream, computes, then deletes the
// executor. Math/semantics mirror tests/test_loss.cpp and the CUDA reference (cuda/src/ops/loss.cu).
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
const int64_t *IP64(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
// reduction: 0=none, 1=mean, 2=sum.
double reduce_scalar(const std::vector<double> &ps, int red) {
    double s = 0; for (double v : ps) s += v;
    return red == 1 ? s / (double)ps.size() : s;  // mean or sum
}
// Write a per-element loss vector to `out`: reduction=0(none) -> per-element output; 1(mean)/2(sum) -> scalar.
void emit_loss(aclOpExecutor *e, const std::vector<double> &ps) {
    float *o = FP(e->out);
    if ((int)e->m == 0) { for (size_t i = 0; i < ps.size(); i++) o[i] = (float)ps[i]; }
    else o[0] = (float)reduce_scalar(ps, (int)e->m);
}

enum {
    K_L1, K_SMOOTHL1, K_HUBER, K_SOFTMARGIN, K_KLDIV, K_BCE, K_MARGINRANK, K_HINGEEMB,  // forward
    K_L1BWD, K_SMOOTHL1BWD,                                                              // loss backward
    K_SWISHBWD, K_SOFTSIGNBWD, K_LOGITGRAD, K_PRELUBWD, K_BCELOGITSBWD, K_CEGRAD,        // act/grad backward
    K_SCALEDMASKEDSOFTMAX, K_GRIDSAMPLER2DBWD, K_THREEINTERPBWD, K_LSTMCELL, K_EXPSEGSUM // structured
};

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    // ---- forward scalar losses (reduction in m: 1=mean else sum) ----
    case K_L1: {
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = std::fabs((double)p[i] - t[i]);
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_SMOOTHL1: {   // beta in alpha: |d|<beta ? 0.5*d^2/beta : |d|-0.5*beta
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); double beta = e->alpha; std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) { double d = std::fabs((double)p[i] - t[i]); ps[i] = d < beta ? 0.5 * d * d / beta : d - 0.5 * beta; }
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_HUBER: {   // delta in alpha: |d|<delta ? 0.5*d^2 : delta*(|d|-0.5*delta)
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); double delta = e->alpha; std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) { double d = std::fabs((double)p[i] - t[i]); ps[i] = d < delta ? 0.5 * d * d : delta * (d - 0.5 * delta); }
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_SOFTMARGIN: {   // log(1+exp(-t*p))
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = std::log1p(std::exp(-(double)t[i] * p[i]));
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_KLDIV: {   // input is log-prob, target is prob: t*(log(t)-input)
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = (double)t[i] * (std::log((double)t[i]) - p[i]);
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_BCE: {   // input prob, target prob: -(t*log(p)+(1-t)*log(1-p))
        const float *p = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = -((double)t[i] * std::log((double)p[i]) + (1 - (double)t[i]) * std::log(1 - (double)p[i]));
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_MARGINRANK: {   // a=x1, b=x2, c=y, margin in alpha: max(0, -y*(x1-x2)+margin)
        const float *x1 = FP(e->a), *x2 = FP(e->b), *y = FP(e->c); int64_t n = e->a->numel(); double margin = e->alpha; std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = std::max(0.0, -(double)y[i] * ((double)x1[i] - x2[i]) + margin);
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    case K_HINGEEMB: {   // a=x, b=y, margin in alpha: y>0 ? x : max(0, margin-x)
        const float *x = FP(e->a), *y = FP(e->b); int64_t n = e->a->numel(); double margin = e->alpha; std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = y[i] > 0 ? (double)x[i] : std::max(0.0, margin - x[i]);
        emit_loss(e, ps); return ACLNN_SUCCESS;
    }
    // ---- loss backward: a=gradOut, b=self(p), c=target(t), out=gradInput ----
    case K_L1BWD: {   // gi = go * sign(p-t) [/n if mean]
        const float *go = FP(e->a), *p = FP(e->b), *t = FP(e->c); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o);
        double scale = (e->m == 1) ? 1.0 / n : 1.0; double g0 = go[0]; bool gscalar = (e->a->numel() == 1);
        for (int64_t i = 0; i < n; i++) { double d = (double)p[i] - t[i]; double sg = d > 0 ? 1.0 : (d < 0 ? -1.0 : 0.0); double gg = gscalar ? g0 : go[i]; gi[i] = (float)(gg * sg * scale); }
        return ACLNN_SUCCESS;
    }
    case K_SMOOTHL1BWD: {   // beta in alpha: gi = go * (|d|<beta ? d/beta : sign(d)) [/n if mean]
        const float *go = FP(e->a), *p = FP(e->b), *t = FP(e->c); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o); double beta = e->alpha;
        double scale = (e->m == 1) ? 1.0 / n : 1.0; double g0 = go[0]; bool gscalar = (e->a->numel() == 1);
        for (int64_t i = 0; i < n; i++) { double d = (double)p[i] - t[i]; double ad = std::fabs(d); double dv = ad < beta ? d / beta : (d > 0 ? 1.0 : -1.0); double gg = gscalar ? g0 : go[i]; gi[i] = (float)(gg * dv * scale); }
        return ACLNN_SUCCESS;
    }
    // ---- activation / grad backward: a=gradOut, b=self(x), out=gradInput ----
    case K_SWISHBWD: {   // beta in alpha (=1): s=sigmoid(beta*x); gi = go*(s + beta*x*s*(1-s))
        const float *go = FP(e->a), *x = FP(e->b); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o); double beta = e->alpha;
        for (int64_t i = 0; i < n; i++) { double sx = sigm(beta * x[i]); gi[i] = (float)(go[i] * (sx + beta * x[i] * sx * (1 - sx))); }
        return ACLNN_SUCCESS;
    }
    case K_SOFTSIGNBWD: {   // gi = go / (1+|x|)^2
        const float *go = FP(e->a), *x = FP(e->b); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o);
        for (int64_t i = 0; i < n; i++) { double a = 1 + std::fabs((double)x[i]); gi[i] = (float)(go[i] / (a * a)); }
        return ACLNN_SUCCESS;
    }
    case K_LOGITGRAD: {   // eps in alpha: gi = go / (x*(1-x)), x clamped to [eps,1-eps] when eps>=0
        const float *go = FP(e->a), *x = FP(e->b); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o); double eps = e->alpha;
        for (int64_t i = 0; i < n; i++) { double xv = x[i]; if (eps >= 0) xv = std::min(std::max(xv, eps), 1 - eps); gi[i] = (float)(go[i] / (xv * (1 - xv))); }
        return ACLNN_SUCCESS;
    }
    case K_PRELUBWD: {   // a=gradOut, b=self(x), c=weight(shared shape{1}); out=gradInput, out2=gradWeight
        const float *go = FP(e->a), *x = FP(e->b), *w = FP(e->c); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o);
        double wv = w[0]; double gw = 0;
        for (int64_t i = 0; i < n; i++) { if (x[i] > 0) gi[i] = go[i]; else { gi[i] = (float)(go[i] * wv); gw += (double)go[i] * x[i]; } }
        if (e->out2) FP(e->out2)[0] = (float)gw;
        return ACLNN_SUCCESS;
    }
    case K_BCELOGITSBWD: {   // a=gradOut, b=self(x), c=target(t); reduction none: gi = go*(sigmoid(x)-t)
        const float *go = FP(e->a), *x = FP(e->b), *t = FP(e->c); aclTensor *o = e->out; int64_t n = o->numel(); float *gi = FP(o);
        for (int64_t i = 0; i < n; i++) gi[i] = (float)(go[i] * (sigm(x[i]) - t[i]));
        return ACLNN_SUCCESS;
    }
    case K_CEGRAD: {   // a=logits[N,C], b=target[N] int64; gradOutput scalar in alpha; out=gradInput[N,C]
        const aclTensor *X = e->a; int N = (int)X->viewDims[0], C = (int)X->viewDims[1];
        const float *x = FP(X); const int64_t *tg = IP64(e->b); aclTensor *o = e->out; float *gi = FP(o);
        double go = e->alpha; double scale = (e->m == 1) ? 1.0 / N : 1.0;
        for (int nn = 0; nn < N; nn++) {
            double mx = -1e30; for (int c = 0; c < C; c++) mx = std::max(mx, (double)x[nn * C + c]);
            double sm = 0; for (int c = 0; c < C; c++) sm += std::exp((double)x[nn * C + c] - mx);
            for (int c = 0; c < C; c++) { double so = std::exp((double)x[nn * C + c] - mx) / sm; gi[nn * C + c] = (float)((so - (c == tg[nn] ? 1.0 : 0.0)) * go * scale); }
        }
        return ACLNN_SUCCESS;
    }
    // ---- structured ops ----
    case K_SCALEDMASKEDSOFTMAX: {   // softmax(x*scale) over last dim; mask (optional) sets -inf before softmax
        const aclTensor *X = e->a; int nd = (int)X->viewDims.size(); int64_t D = X->viewDims[nd - 1]; int64_t rows = X->numel() / D;
        const float *x = FP(X); float *z = FP(e->out); double scale = e->alpha; const float *mk = e->mask ? FP(e->mask) : nullptr;
        for (int64_t r = 0; r < rows; r++) {
            double mx = -1e30; for (int64_t d = 0; d < D; d++) { double v = (double)x[r * D + d] * scale; if (mk && mk[r * D + d] != 0) v = -1e30; mx = std::max(mx, v); }
            double sm = 0; std::vector<double> ev(D);
            for (int64_t d = 0; d < D; d++) { double v = (double)x[r * D + d] * scale; if (mk && mk[r * D + d] != 0) v = -1e30; ev[d] = std::exp(v - mx); sm += ev[d]; }
            for (int64_t d = 0; d < D; d++) z[r * D + d] = (float)(ev[d] / sm);
        }
        return ACLNN_SUCCESS;
    }
    case K_GRIDSAMPLER2DBWD: {   // a=gradOut[N,C,oH,oW], b=grid[N,oH,oW,2]; bilinear scatter to out=gradInput[N,C,H,W]
        const aclTensor *GO = e->a, *GR = e->b; aclTensor *O = e->out;
        int N = (int)O->viewDims[0], C = (int)O->viewDims[1], H = (int)O->viewDims[2], W = (int)O->viewDims[3];
        int oH = (int)GO->viewDims[2], oW = (int)GO->viewDims[3]; bool align = (e->dim != 0);
        const float *go = FP(GO), *grid = FP(GR); float *gi = FP(O);
        for (int64_t i = 0, m = O->numel(); i < m; i++) gi[i] = 0.f;
        for (int n = 0; n < N; n++) for (int c = 0; c < C; c++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) {
            const float *g = grid + (((int64_t)n * oH + oh) * oW + ow) * 2; double gx = g[0], gy = g[1];
            double fx, fy;
            if (align) { fx = (gx + 1) * 0.5 * (W - 1); fy = (gy + 1) * 0.5 * (H - 1); }
            else { fx = ((gx + 1) * W - 1) * 0.5; fy = ((gy + 1) * H - 1) * 0.5; }
            int x0 = (int)std::floor(fx), y0 = (int)std::floor(fy), x1 = x0 + 1, y1 = y0 + 1;
            double dx = fx - x0, dy = fy - y0; double gv = go[((int64_t)(n * C + c) * oH + oh) * oW + ow];
            float *p = gi + ((int64_t)n * C + c) * H * W;
            auto add = [&](int yy, int xx, double wgt) { if (yy >= 0 && yy < H && xx >= 0 && xx < W) p[yy * W + xx] += (float)(gv * wgt); };
            add(y0, x0, (1 - dy) * (1 - dx)); add(y0, x1, (1 - dy) * dx); add(y1, x0, dy * (1 - dx)); add(y1, x1, dy * dx);
        }
        return ACLNN_SUCCESS;
    }
    case K_THREEINTERPBWD: {   // a=gradOut[B,C,N], b=indices[B,N,3] int64, c=weight[B,N,3]; out=gradFeatures[B,C,M]
        const aclTensor *GO = e->a; aclTensor *O = e->out;
        int B = (int)GO->viewDims[0], C = (int)GO->viewDims[1], N = (int)GO->viewDims[2], M = (int)O->viewDims[2];
        const float *go = FP(GO); const int64_t *idx = IP64(e->b); const float *wt = FP(e->c); float *gf = FP(O);
        for (int64_t i = 0, m = O->numel(); i < m; i++) gf[i] = 0.f;
        for (int b = 0; b < B; b++) for (int c = 0; c < C; c++) for (int nn = 0; nn < N; nn++) {
            double gv = go[((int64_t)b * C + c) * N + nn]; const int64_t *id = idx + ((int64_t)b * N + nn) * 3; const float *w = wt + ((int64_t)b * N + nn) * 3;
            for (int k = 0; k < 3; k++) { int64_t mm = id[k]; if (mm >= 0 && mm < M) gf[((int64_t)b * C + c) * M + mm] += (float)(gv * w[k]); }
        }
        return ACLNN_SUCCESS;
    }
    case K_LSTMCELL: {   // a=gates[B,4H] (i,f,g,o), b=cprev[B,H]; out=hNew[B,H], out2=cNew[B,H]
        const aclTensor *G = e->a, *CP = e->b; aclTensor *HN = e->out, *CN = e->out2;
        int B = (int)G->viewDims[0], H = (int)CP->viewDims[1];
        const float *gates = FP(G), *cp = FP(CP); float *hn = FP(HN), *cn = FP(CN);
        for (int b = 0; b < B; b++) { const float *g = gates + (int64_t)b * 4 * H;
            for (int h = 0; h < H; h++) { double ig = sigm(g[h]), fg = sigm(g[H + h]), gg = std::tanh((double)g[2 * H + h]), og = sigm(g[3 * H + h]);
                double c = fg * cp[(int64_t)b * H + h] + ig * gg; cn[(int64_t)b * H + h] = (float)c; hn[(int64_t)b * H + h] = (float)(og * std::tanh(c)); } }
        return ACLNN_SUCCESS;
    }
    case K_EXPSEGSUM: {   // a=x[L]; out[i,j]=exp(Σ_{j<k<=i} x[k]) for i>=j, else 0
        const aclTensor *X = e->a; aclTensor *O = e->out; int L = (int)X->numel();
        const float *x = FP(X); float *o = FP(O);
        for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) {
            if (i >= j) { double su = 0; for (int k = j + 1; k <= i; k++) su += x[k]; o[i * L + j] = (float)std::exp(su); }
            else o[i * L + j] = 0.f;
        }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}

#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUN(loss_run)
} // namespace

extern "C" {
// ---- forward scalar losses ----
aclnnStatus aclnnL1LossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_L1; e->a = self; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, double beta, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SMOOTHL1; e->a = self; e->b = target; e->out = out; e->m = reduction; e->alpha = beta; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnHuberLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, double delta, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_HUBER; e->a = self; e->b = target; e->out = out; e->m = reduction; e->alpha = delta; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSoftMarginLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SOFTMARGIN; e->a = self; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnKlDivGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_KLDIV; e->a = self; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnBinaryCrossEntropyGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_BCE; e->a = self; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnMarginRankingLossGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *y, double margin, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MARGINRANK; e->a = x1; e->b = x2; e->c = y; e->out = out; e->alpha = margin; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnHingeEmbeddingLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double margin, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_HINGEEMB; e->a = self; e->b = target; e->out = out; e->alpha = margin; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnL1Loss) RUN(aclnnSmoothL1Loss) RUN(aclnnHuberLoss) RUN(aclnnSoftMarginLoss)
RUN(aclnnKlDiv) RUN(aclnnBinaryCrossEntropy) RUN(aclnnMarginRankingLoss) RUN(aclnnHingeEmbeddingLoss)

// ---- loss backward ----
aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_L1BWD; e->a = gradOut; e->b = self; e->c = target; e->out = gradInput; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target, int64_t reduction, double beta, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SMOOTHL1BWD; e->a = gradOut; e->b = self; e->c = target; e->out = gradInput; e->m = reduction; e->alpha = beta; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnL1LossBackward) RUN(aclnnSmoothL1LossBackward)

// ---- activation / grad backward ----
aclnnStatus aclnnSwishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double beta, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SWISHBWD; e->a = gradOutput; e->b = self; e->out = gradInput; e->alpha = beta; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnSoftsignBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SOFTSIGNBWD; e->a = gradOutput; e->b = self; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnLogitGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double eps, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LOGITGRAD; e->a = gradOutput; e->b = self; e->out = gradInput; e->alpha = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnPreluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, aclTensor *gradInput, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_PRELUBWD; e->a = gradOutput; e->b = self; e->c = weight; e->out = gradInput; e->out2 = gradWeight; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, const aclTensor *posWeight, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)weight; (void)posWeight; (void)reduction; auto *e = new aclOpExecutor(); e->op = K_BCELOGITSBWD; e->a = gradOutput; e->b = self; e->c = target; e->out = gradInput; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double gradOutput, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_CEGRAD; e->a = self; e->b = target; e->out = gradInput; e->alpha = gradOutput; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSwishBackward) RUN(aclnnSoftsignBackward) RUN(aclnnLogitGrad) RUN(aclnnPreluBackward)
RUN(aclnnBinaryCrossEntropyWithLogitsBackward) RUN(aclnnCrossEntropyLossGrad)

// ---- structured ops ----
aclnnStatus aclnnScaledMaskedSoftmaxGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, double scale, bool fixedTriuMask, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)fixedTriuMask; auto *e = new aclOpExecutor(); e->op = K_SCALEDMASKEDSOFTMAX; e->a = x; e->mask = mask; e->out = out; e->alpha = scale; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *gradInput, aclTensor *gradGrid, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)interpolationMode; (void)paddingMode; (void)gradGrid;
    auto *e = new aclOpExecutor(); e->op = K_GRIDSAMPLER2DBWD; e->a = gradOutput; e->b = grid; e->out = gradInput; e->dim = alignCorners ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *indices, const aclTensor *weight, int64_t m, aclTensor *gradFeatures, uint64_t *ws, aclOpExecutor **ex) {
    (void)m; auto *e = new aclOpExecutor(); e->op = K_THREEINTERPBWD; e->a = gradOutput; e->b = indices; e->c = weight; e->out = gradFeatures; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(const aclTensor *gates, const aclTensor *cprev, aclTensor *hNew, aclTensor *cNew, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LSTMCELL; e->a = gates; e->b = cprev; e->out = hNew; e->out2 = cNew; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EXPSEGSUM; e->a = x; e->out = out; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScaledMaskedSoftmax) RUN(aclnnGridSampler2DBackward) RUN(aclnnThreeInterpolateBackward)
RUN(aclnnThnnFusedLstmCell) RUN(aclnnExpSegsum)
} // extern "C"
