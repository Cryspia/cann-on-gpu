// test_remainder R6/R7/R8: loss remainder, optimizer remainder, random/RNN/FFT/EmbeddingBag remainder.
// Host-side over unified memory. Math matches tests/test_remainder.cpp references.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

namespace {
float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
int64_t *IP64(const aclTensor *t) { return (int64_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
double reduce(const std::vector<double> &ps, int red) { double s = 0; for (double v : ps) s += v; return red == 1 ? s / ps.size() : s; }  // 1=mean, else(0/2)=sum
// emit per-element loss vec: reduction=0(none) -> per-element output; 1(mean)/2(sum) -> scalar.
void emit(aclOpExecutor *e, const std::vector<double> &v) {
    float *o = FP(e->out);
    if ((int)e->m == 0) { for (size_t i = 0; i < v.size(); i++) o[i] = (float)v[i]; }
    else o[0] = (float)reduce(v, (int)e->m);
}

enum { K_POISSON, K_GAUSSIAN, K_MLSM, K_MULTIMARGIN, K_TRIPLET, K_COSEMB, K_CTC,
       K_LAMB, K_LARS, K_EMAADAM, K_SAMPLEGAMMA, K_SAMPLEDIRICHLET, K_RNN, K_EMBBAG, K_FFT2, K_STFT };

// Marsaglia-Tsang gamma sampler (shape a, scale 1)
double sample_gamma(double a, std::mt19937 &gen, std::normal_distribution<double> &nd, std::uniform_real_distribution<double> &ud) {
    if (a < 1.0) { double u = ud(gen); return sample_gamma(a + 1.0, gen, nd, ud) * std::pow(u, 1.0 / a); }
    double d = a - 1.0 / 3.0, c = 1.0 / std::sqrt(9.0 * d);
    for (;;) { double x, v; do { x = nd(gen); v = 1.0 + c * x; } while (v <= 0); v = v * v * v; double u = ud(gen);
        if (u < 1.0 - 0.0331 * x * x * x * x) return d * v;
        if (std::log(u) < 0.5 * x * x + d * (1.0 - v + std::log(v))) return d * v; }
}

aclnnStatus run(aclOpExecutor *e, aclrtStream s) {
    drain(s);
    switch (e->op) {
    case K_POISSON: {   // logInput=true: exp(x) - target*x ; reduction in m (1=mean)
        const float *x = FP(e->a), *t = FP(e->b); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) ps[i] = std::exp(x[i]) - (double)t[i] * x[i];
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_GAUSSIAN: {   // 0.5*(log(var) + (x-t)^2/var)
        const float *x = FP(e->a), *t = FP(e->b), *v = FP(e->c); int64_t n = e->a->numel(); std::vector<double> ps(n);
        for (int64_t i = 0; i < n; i++) { double d = x[i] - t[i]; ps[i] = 0.5 * (std::log(v[i]) + d * d / v[i]); }
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_MLSM: {   // multilabel soft margin: per-row mean over C of -[t*log p + (1-t)*log(1-p)]
        const aclTensor *X = e->a; int N = (int)X->viewDims[0], C = (int)X->viewDims[1];
        const float *x = FP(X), *t = FP(e->b); std::vector<double> ps(N);
        for (int nn = 0; nn < N; nn++) { double sum = 0; for (int c = 0; c < C; c++) { double p = 1.0 / (1.0 + std::exp(-x[nn * C + c])); sum += t[nn * C + c] * std::log(p) + (1 - t[nn * C + c]) * std::log(1 - p); } ps[nn] = -sum / C; }
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_MULTIMARGIN: {   // p=1 margin=1: sum_{j!=y} max(0, margin - x[y] + x[j]) / C
        const aclTensor *X = e->a; int N = (int)X->viewDims[0], C = (int)X->viewDims[1];
        const float *x = FP(X); const int64_t *y = IP64(e->b); double margin = e->alpha; std::vector<double> ps(N);
        for (int nn = 0; nn < N; nn++) { double sum = 0; for (int j = 0; j < C; j++) { if (j == y[nn]) continue; double m = margin - x[nn * C + y[nn]] + x[nn * C + j]; if (m > 0) sum += m; } ps[nn] = sum / C; }
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_TRIPLET: {   // p=2 margin: max(0, ||a-p|| - ||a-n|| + margin)
        const aclTensor *A = e->a; int N = (int)A->viewDims[0], D = (int)A->viewDims[1];
        const float *a = FP(A), *p = FP(e->b), *ng = FP(e->c); double margin = e->alpha; std::vector<double> ps(N);
        for (int nn = 0; nn < N; nn++) { double dp = 0, dn = 0; for (int d = 0; d < D; d++) { double u = a[nn * D + d] - p[nn * D + d], v = a[nn * D + d] - ng[nn * D + d]; dp += u * u; dn += v * v; }
            double l = std::sqrt(dp) - std::sqrt(dn) + margin; ps[nn] = l > 0 ? l : 0; }
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_COSEMB: {   // margin: y>0 -> 1-cos ; else max(0, cos-margin)
        const aclTensor *X1 = e->a; int N = (int)X1->viewDims[0], D = (int)X1->viewDims[1];
        const float *x1 = FP(X1), *x2 = FP(e->b), *y = FP(e->c); double margin = e->alpha; std::vector<double> ps(N);
        for (int nn = 0; nn < N; nn++) { double dot = 0, n1 = 0, n2 = 0; for (int d = 0; d < D; d++) { dot += (double)x1[nn * D + d] * x2[nn * D + d]; n1 += (double)x1[nn * D + d] * x1[nn * D + d]; n2 += (double)x2[nn * D + d] * x2[nn * D + d]; }
            double cs = dot / (std::sqrt(n1) * std::sqrt(n2) + 1e-12); ps[nn] = y[nn] > 0 ? 1 - cs : std::max(0.0, cs - margin); }
        emit(e, ps); return ACLNN_SUCCESS;
    }
    case K_CTC: {   // logProbs [T,N,C]; targets [N,Lmax]; blank in dim ; out: per-batch -loglik (mean over batch if reduction)
        const aclTensor *LP = e->a; int T = (int)LP->viewDims[0], Nb = (int)LP->viewDims[1], C = (int)LP->viewDims[2];
        int blank = (int)e->dim; const float *lp = FP(LP); const int64_t *tg = IP64(e->b);
        const int64_t *tl = IP64(e->c); int Lmax = (int)e->b->viewDims[1];
        auto lse = [](double a, double b) { double mx = std::max(a, b), mn = std::min(a, b); return mx + std::log1p(std::exp(mn - mx)); };
        std::vector<double> losses(Nb);
        for (int nb = 0; nb < Nb; nb++) {
            int L = (int)tl[nb], S = 2 * L + 1;
            auto lab = [&](int si) { return (si & 1) ? (int)tg[nb * Lmax + si / 2] : blank; };
            auto LPc = [&](int t, int c) { return (double)lp[(t * Nb + nb) * C + c]; };
            std::vector<double> prev(S, -1e30), cur(S);
            prev[0] = LPc(0, blank); if (S > 1) prev[1] = LPc(0, lab(1));
            for (int t = 1; t < T; t++) { for (int si = 0; si < S; si++) { double v = prev[si]; if (si > 0) v = lse(v, prev[si - 1]); if (si > 1 && lab(si) != blank && lab(si) != lab(si - 2)) v = lse(v, prev[si - 2]); cur[si] = v + LPc(t, lab(si)); } prev = cur; }
            losses[nb] = -lse(prev[S - 1], prev[S - 2]);
        }
        emit(e, losses); return ACLNN_SUCCESS;
    }
    case K_LAMB: {   // param,m,v,grad ; dscalars [lr,b1,b2,eps,wd,step]
        aclTensor *P = e->out; const aclTensor *M = e->b, *V = e->c, *G = e->a;
        double lr = e->dscalars[0], b1 = e->dscalars[1], b2 = e->dscalars[2], eps = e->dscalars[3], wd = e->dscalars[4]; int step = (int)e->dscalars[5];
        double bc1 = 1 - std::pow(b1, step), bc2 = 1 - std::pow(b2, step);
        int64_t n = P->numel(); float *p = FP(P); const float *m = FP(M), *v = FP(V), *g = FP(G);
        std::vector<double> r(n); double pn = 0, rn = 0;
        for (int64_t i = 0; i < n; i++) { double mi = b1 * m[i] + (1 - b1) * g[i]; double vi = b2 * v[i] + (1 - b2) * g[i] * g[i]; r[i] = (mi / bc1) / (std::sqrt(vi / bc2) + eps) + wd * p[i]; pn += p[i] * p[i]; rn += r[i] * r[i]; }
        pn = std::sqrt(pn); rn = std::sqrt(rn); double trust = (pn > 0 && rn > 0) ? pn / rn : 1.0;
        for (int64_t i = 0; i < n; i++) p[i] = (float)(p[i] - lr * trust * r[i]);
        return ACLNN_SUCCESS;
    }
    case K_LARS: {   // param,buf,grad ; dscalars [lr,mu,wd,tc,eps]
        aclTensor *P = e->out; const aclTensor *BUF = e->b, *G = e->a;
        double lr = e->dscalars[0], mu = e->dscalars[1], wd = e->dscalars[2], tc = e->dscalars[3], eps = e->dscalars[4];
        int64_t n = P->numel(); float *p = FP(P); const float *buf = FP(BUF), *g = FP(G);
        double pn = 0, gn = 0; for (int64_t i = 0; i < n; i++) { pn += p[i] * p[i]; gn += g[i] * g[i]; } pn = std::sqrt(pn); gn = std::sqrt(gn);
        double llr = (pn > 0 && gn > 0) ? tc * pn / (gn + wd * pn + eps) : 1.0;
        for (int64_t i = 0; i < n; i++) { double bb = mu * buf[i] + (g[i] + wd * p[i]); p[i] = (float)(p[i] - lr * llr * bb); }
        return ACLNN_SUCCESS;
    }
    case K_EMAADAM: {   // param,m,v,ema,grad ; dscalars [lr,b1,b2,eps,wd,emaDecay,step]
        aclTensor *P = e->out, *EMA = e->out2; const aclTensor *M = e->b, *V = e->c, *G = e->a;
        double lr = e->dscalars[0], b1 = e->dscalars[1], b2 = e->dscalars[2], eps = e->dscalars[3], wd = e->dscalars[4], ed = e->dscalars[5]; int step = (int)e->dscalars[6];
        double bc1 = 1 - std::pow(b1, step), bc2 = 1 - std::pow(b2, step);
        int64_t n = P->numel(); float *p = FP(P); float *ema = FP(EMA); const float *m = FP(M), *v = FP(V), *g = FP(G);
        for (int64_t i = 0; i < n; i++) { double mi = b1 * m[i] + (1 - b1) * g[i]; double vi = b2 * v[i] + (1 - b2) * g[i] * g[i];
            double pp = p[i] - lr * ((mi / bc1) / (std::sqrt(vi / bc2) + eps) + wd * p[i]); double em = ed * ema[i] + (1 - ed) * pp;
            p[i] = (float)pp; ema[i] = (float)em; }
        return ACLNN_SUCCESS;
    }
    case K_SAMPLEGAMMA: {   // alpha tensor, scale in alpha, seed in dim -> Gamma(a, scale)
        const aclTensor *A = e->a; aclTensor *O = e->out; int64_t n = O->numel(); double scale = e->alpha;
        const float *al = FP(A); float *o = FP(O); std::mt19937 gen((uint32_t)e->dim);
        std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
        for (int64_t i = 0; i < n; i++) o[i] = (float)(sample_gamma(al[i], gen, nd, ud) * scale);
        return ACLNN_SUCCESS;
    }
    case K_SAMPLEDIRICHLET: {   // alpha [M,K], seed in dim -> per-row Dirichlet
        const aclTensor *A = e->a; aclTensor *O = e->out; int M = (int)A->viewDims[0], K = (int)A->viewDims[1];
        const float *al = FP(A); float *o = FP(O); std::mt19937 gen((uint32_t)e->dim);
        std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
        for (int r = 0; r < M; r++) { double s = 0; for (int k = 0; k < K; k++) { double gv = sample_gamma(al[r * K + k], gen, nd, ud); o[r * K + k] = (float)gv; s += gv; }
            if (s > 0) for (int k = 0; k < K; k++) o[r * K + k] = (float)(o[r * K + k] / s); }
        return ACLNN_SUCCESS;
    }
    case K_RNN: {   // input [T,B,I], h0 [B,H], wih [H,I], whh [H,H], bih [H], bhh [H]; tanh; out [T,B,H], hn [B,H]
        const aclTensor *X = e->a, *H0 = e->b, *WIH = e->c, *WHH = e->inputs[0], *BIH = e->inputs[1], *BHH = e->inputs[2];
        aclTensor *OUT = e->out, *HN = e->out2;
        int T = (int)X->viewDims[0], B = (int)X->viewDims[1], I = (int)X->viewDims[2], H = (int)H0->viewDims[1];
        const float *x = FP(X), *wih = FP(WIH), *whh = FP(WHH), *bih = FP(BIH), *bhh = FP(BHH);
        float *out = FP(OUT), *hn = FP(HN);
        std::vector<double> h(B * H); { const float *h0 = FP(H0); for (int i = 0; i < B * H; i++) h[i] = h0[i]; }
        for (int t = 0; t < T; t++) { std::vector<double> hnew(B * H);
            for (int b = 0; b < B; b++) for (int j = 0; j < H; j++) { double sum = bih[j] + bhh[j];
                for (int i = 0; i < I; i++) sum += (double)x[(t * B + b) * I + i] * wih[j * I + i];
                for (int k = 0; k < H; k++) sum += h[b * H + k] * whh[j * H + k];
                hnew[b * H + j] = std::tanh(sum); }
            h = hnew; for (int b = 0; b < B; b++) for (int j = 0; j < H; j++) out[(t * B + b) * H + j] = (float)h[b * H + j];
        }
        for (int i = 0; i < B * H; i++) hn[i] = (float)h[i];
        return ACLNN_SUCCESS;
    }
    case K_EMBBAG: {   // weight [V,D], indices [nidx], offsets [B], mode in m (0=sum,1=mean,2=max) -> out [B,D]
        const aclTensor *W = e->a, *IDX = e->b, *OFF = e->c; aclTensor *O = e->out; int mode = (int)e->m;
        int D = (int)W->viewDims[1]; int nidx = (int)IDX->numel(); int B = (int)OFF->numel();
        const float *w = FP(W); const int64_t *idx = IP64(IDX); const int64_t *off = IP64(OFF); float *o = FP(O);
        for (int b = 0; b < B; b++) { int start = (int)off[b], end = (b + 1 < B) ? (int)off[b + 1] : nidx;
            for (int d = 0; d < D; d++) { double acc = (mode == 2) ? -1e30 : 0; int cnt = 0;
                for (int p = start; p < end; p++) { double v = w[idx[p] * D + d]; if (mode == 2) acc = std::max(acc, v); else acc += v; cnt++; }
                if (mode == 1 && cnt) acc /= cnt; o[b * D + d] = (float)acc; }
        }
        return ACLNN_SUCCESS;
    }
    case K_FFT2: {   // x interleaved complex flat = [batch,n0,n1,2]; 2D DFT per batch; inverse normalizes by n0*n1
        const aclTensor *X = e->a; aclTensor *O = e->out; int n0 = (int)e->m, n1 = (int)e->n; bool inv = e->causal;
        int64_t total = X->numel(); int per = n0 * n1 * 2; int batch = (int)(total / per);
        const float *x = FP(X); float *o = FP(O); double sgn = inv ? 1.0 : -1.0;
        for (int bb = 0; bb < batch; bb++) { const float *xb = x + bb * per; float *ob = o + bb * per;
            // DFT over dim n0 then n1 separably
            std::vector<double> re(n0 * n1), im(n0 * n1);
            for (int i = 0; i < n0 * n1; i++) { re[i] = xb[2 * i]; im[i] = xb[2 * i + 1]; }
            // transform along n1 (rows)
            std::vector<double> re1(n0 * n1), im1(n0 * n1);
            for (int r = 0; r < n0; r++) for (int k = 0; k < n1; k++) { double sr = 0, si = 0; for (int j = 0; j < n1; j++) { double ang = sgn * 2 * M_PI * k * j / n1; double c = std::cos(ang), s = std::sin(ang); sr += re[r * n1 + j] * c - im[r * n1 + j] * s; si += re[r * n1 + j] * s + im[r * n1 + j] * c; } re1[r * n1 + k] = sr; im1[r * n1 + k] = si; }
            // transform along n0 (cols)
            std::vector<double> re2(n0 * n1), im2(n0 * n1);
            for (int c = 0; c < n1; c++) for (int k = 0; k < n0; k++) { double sr = 0, si = 0; for (int j = 0; j < n0; j++) { double ang = sgn * 2 * M_PI * k * j / n0; double cc = std::cos(ang), ss = std::sin(ang); sr += re1[j * n1 + c] * cc - im1[j * n1 + c] * ss; si += re1[j * n1 + c] * ss + im1[j * n1 + c] * cc; } re2[k * n1 + c] = sr; im2[k * n1 + c] = si; }
            double norm = inv ? 1.0 / (n0 * n1) : 1.0;
            for (int i = 0; i < n0 * n1; i++) { ob[2 * i] = (float)(re2[i] * norm); ob[2 * i + 1] = (float)(im2[i] * norm); }
        }
        return ACLNN_SUCCESS;
    }
    case K_STFT: {   // x [B,L]; nFft in m, hop in n; window=ones; out flat [B,nFreq,frames,2] laid as d=f*frames+fr
        const aclTensor *X = e->a; aclTensor *O = e->out; int nFft = (int)e->m, hop = (int)e->n;
        int B = (int)X->viewDims[0], L = (int)X->viewDims[1]; int frames = 1 + (L - nFft) / hop, nFreq = nFft / 2 + 1;
        const float *x = FP(X); float *o = FP(O);
        const float *win = e->b ? FP(e->b) : nullptr;
        for (int b = 0; b < B; b++) { const float *xb = x + b * L; float *ob = o + b * nFreq * frames * 2;
            for (int fr = 0; fr < frames; fr++) for (int f = 0; f < nFreq; f++) { double re = 0, im = 0;
                for (int j = 0; j < nFft; j++) { double ang = -2 * M_PI * f * j / nFft; double v = xb[fr * hop + j] * (win ? win[j] : 1.0); re += v * std::cos(ang); im += v * std::sin(ang); }
                int64_t d = (int64_t)f * frames + fr; ob[2 * d] = (float)re; ob[2 * d + 1] = (float)im; }
        }
        return ACLNN_SUCCESS;
    }
    default: return ACLNN_ERR_PARAM_INVALID;
    }
}
#define RUNFN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run(e, s); } delete e; return st; }
RUNFN(rn_run)
} // namespace

extern "C" {
// ---- loss ----
aclnnStatus aclnnPoissonNllLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, bool logInput, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)logInput; auto *e = new aclOpExecutor(); e->op = K_POISSON; e->a = input; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnPoissonNllLoss)
aclnnStatus aclnnGaussianNllLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, const aclTensor *var, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_GAUSSIAN; e->a = input; e->b = target; e->c = var; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnGaussianNllLoss)
aclnnStatus aclnnMultiLabelSoftMarginLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_MLSM; e->a = input; e->b = target; e->out = out; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMultiLabelSoftMarginLoss)
aclnnStatus aclnnMultiMarginLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, double p, double margin, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)p; auto *e = new aclOpExecutor(); e->op = K_MULTIMARGIN; e->a = input; e->b = target; e->out = out; e->alpha = margin; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnMultiMarginLoss)
aclnnStatus aclnnTripletMarginLossGetWorkspaceSize(const aclTensor *anchor, const aclTensor *positive, const aclTensor *negative, double margin, double p, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)p; auto *e = new aclOpExecutor(); e->op = K_TRIPLET; e->a = anchor; e->b = positive; e->c = negative; e->out = out; e->alpha = margin; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnTripletMarginLoss)
aclnnStatus aclnnCosineEmbeddingLossGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *target, double margin, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_COSEMB; e->a = x1; e->b = x2; e->c = target; e->out = out; e->alpha = margin; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnCosineEmbeddingLoss)
aclnnStatus aclnnCtcLossGetWorkspaceSize(const aclTensor *logProbs, const aclTensor *targets, const aclTensor *inputLengths, const aclTensor *targetLengths, int64_t blank, int64_t reduction, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)inputLengths; auto *e = new aclOpExecutor(); e->op = K_CTC; e->a = logProbs; e->b = targets; e->c = targetLengths; e->out = out; e->dim = blank; e->m = reduction; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnCtcLoss)

// ---- optim ----
aclnnStatus aclnnApplyLambGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LAMB; e->out = param; e->b = m; e->c = v; e->a = grad; e->dscalars = {lr, beta1, beta2, eps, weightDecay, (double)step}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnApplyLamb)
aclnnStatus aclnnApplyLarsGetWorkspaceSize(aclTensor *param, aclTensor *buf, const aclTensor *grad, double lr, double momentum, double weightDecay, double trustCoeff, double eps, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_LARS; e->out = param; e->b = buf; e->a = grad; e->dscalars = {lr, momentum, weightDecay, trustCoeff, eps}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnApplyLars)
aclnnStatus aclnnFusedEmaAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, aclTensor *ema, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, double emaDecay, int64_t step, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EMAADAM; e->out = param; e->out2 = ema; e->b = m; e->c = v; e->a = grad; e->dscalars = {lr, beta1, beta2, eps, weightDecay, emaDecay, (double)step}; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnFusedEmaAdam)

// ---- random / rnn / fft / embeddingbag ----
aclnnStatus aclnnSampleGammaGetWorkspaceSize(const aclTensor *alpha, double scale, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SAMPLEGAMMA; e->a = alpha; e->out = out; e->alpha = scale; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSampleGamma)
aclnnStatus aclnnSampleDirichletGetWorkspaceSize(const aclTensor *alpha, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_SAMPLEDIRICHLET; e->a = alpha; e->out = out; e->dim = seed; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnSampleDirichlet)
aclnnStatus aclnnRnnGetWorkspaceSize(const aclTensor *input, const aclTensor *h0, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, int64_t nonlinearity, aclTensor *out, aclTensor *hn, uint64_t *ws, aclOpExecutor **ex) {
    (void)nonlinearity; auto *e = new aclOpExecutor(); e->op = K_RNN; e->a = input; e->b = h0; e->c = wih; e->inputs = {whh, bih, bhh}; e->out = out; e->out2 = hn; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnRnn)
aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_EMBBAG; e->a = weight; e->b = indices; e->c = offsets; e->out = out; e->m = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnEmbeddingBag)
aclnnStatus aclnnFft2GetWorkspaceSize(const aclTensor *x, int64_t n0, int64_t n1, bool inverse, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_FFT2; e->a = x; e->out = out; e->m = n0; e->n = n1; e->causal = inverse; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnFft2)
aclnnStatus aclnnStftGetWorkspaceSize(const aclTensor *x, int64_t nFft, int64_t hop, const aclTensor *window, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e = new aclOpExecutor(); e->op = K_STFT; e->a = x; e->b = window; e->out = out; e->m = nFft; e->n = hop; *ws = 0; *ex = e; return ACLNN_SUCCESS; } RUNFN(aclnnStft)
} // extern "C"
