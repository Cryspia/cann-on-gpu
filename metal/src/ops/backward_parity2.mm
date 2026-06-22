// Backward / gradient operator family — gap fill (part 2: norm / loss / attention / RoPE / MoE /
// deformable-attn / RNN). Host-side over unified memory. Two-phase aclnn contract. Math mirrors the
// CUDA reference (cuda/src/ops/{norm_ext,loss,attention,moe,conv,index_ext}.cu). FP32 throughout.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace bwd_parity2 {
inline float *FP(const aclTensor *t) { return (float *)t->data + t->offset; }
inline const int64_t *IP(const aclTensor *t) { return (const int64_t *)t->data + t->offset; }
inline const uint8_t *U8(const aclTensor *t) { return (const uint8_t *)t->data + t->offset; }
void drain(aclrtStream s) { auto *st = (AclStream *)s; if (st && st->last) [st->last waitUntilCompleted]; }
inline double sigm(double x) { return 1.0 / (1.0 + std::exp(-x)); }
inline void zero(aclTensor *t) { if (!t) return; float *p = FP(t); for (int64_t i = 0, n = t->numel(); i < n; ++i) p[i] = 0.f; }

// ============================ Loss elementwise backward ============================
// modes (mirror cuda k_grad): 4 softmargin, 5 bce, 6 bcelogits d/dx, 7 bcelogits d/dt, 8 kldiv d/din, 9 kldiv d/dt.
aclnnStatus run_lossgrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x = e->b, *t = e->c, *w = e->mask; aclTensor *gi = e->out;
    int mode = (int)e->dim; double invN = e->alpha; int64_t n = gi->numel();
    const float *gp = FP(go), *xp = x ? FP(x) : nullptr, *tp = t ? FP(t) : nullptr, *wp = w ? FP(w) : nullptr;
    float *op = FP(gi);
    for (int64_t i = 0; i < n; ++i) {
        double g = gp[i], xv = xp ? xp[i] : 0.0, r;
        switch (mode) {
            case 4: { double tv = tp[i]; r = g * (-tv * sigm(-tv * xv)); } break;
            case 5: { double tv = tp[i], wv = wp ? wp[i] : 1.0; r = g * wv * (xv - tv) / std::fmax(xv * (1 - xv), 1e-12) * invN; } break;
            case 6: { double tv = tp[i], wv = wp ? wp[i] : 1.0; r = g * wv * (sigm(xv) - tv) * invN; } break;
            case 7: { double wv = wp ? wp[i] : 1.0; r = g * wv * (-xv) * invN; } break;
            case 8: { double tv = tp[i]; r = g * (-tv) * invN; } break;
            default: { double tv = tp[i]; r = g * (std::log(std::fmax(tv, 1e-12)) - xv + 1.0) * invN; }
        }
        op[i] = (float)r;
    }
    return ACLNN_SUCCESS;
}
// NLLLoss backward: gi[n,c] = -g * w[target] * (c==target); g scalar (mean/sum) or per-sample [N].
aclnnStatus run_nllbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *target = e->b, *w = e->mask; aclTensor *gi = e->out;
    int64_t N = e->m, C = e->n; double invN = e->alpha; int goScalar = (int)e->dim;
    const float *gp = FP(go), *wp = w ? FP(w) : nullptr; const int64_t *tp = IP(target); float *op = FP(gi);
    for (int64_t i = 0; i < N * C; ++i) { int64_t nn = i / C, c = i % C, tt = tp[nn];
        double g = goScalar ? gp[0] : gp[nn];
        op[i] = (c == tt) ? (float)(-g * (wp ? wp[tt] : 1.0) * invN) : 0.f;
    }
    return ACLNN_SUCCESS;
}
// CrossEntropyLoss grad (FusedLinearCrossEntropyLossGrad): gi[n,c] = (softmax(x)[n,c] - (c==t))*go*invN.
aclnnStatus run_cegrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *x = e->a, *target = e->b; aclTensor *gi = e->out;
    int64_t N = e->m, C = e->n; double go = e->alpha, invN = e->eps;
    const float *xp = FP(x); const int64_t *tp = IP(target); float *op = FP(gi);
    for (int64_t nn = 0; nn < N; ++nn) { const float *p = xp + nn * C; float *gp = op + nn * C;
        double mx = -1e30; for (int64_t c = 0; c < C; ++c) mx = std::max(mx, (double)p[c]);
        double sm = 0; for (int64_t c = 0; c < C; ++c) sm += std::exp(p[c] - mx);
        int64_t tt = tp[nn];
        for (int64_t c = 0; c < C; ++c) { double so = std::exp(p[c] - mx) / sm; gp[c] = (float)((so - (c == tt ? 1.0 : 0.0)) * go * invN); }
    }
    return ACLNN_SUCCESS;
}
// CtcLossBackward (CUDA placeholder): gradInput = gradLoss * exp(logProbs).
aclnnStatus run_ctcbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gl = e->a, *logp = e->b; aclTensor *gi = e->out;
    int goScalar = (int)e->dim; int64_t n = gi->numel();
    const float *glp = FP(gl), *lp = FP(logp); float *op = FP(gi);
    for (int64_t i = 0; i < n; ++i) op[i] = (float)((goScalar ? glp[0] : glp[i]) * std::exp((double)lp[i]));
    return ACLNN_SUCCESS;
}
// ScaledMaskedSoftmaxBackward: gi = scale*y*(go - Σ go*y) per row over last dim.
aclnnStatus run_smsoftmaxbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *y = e->b; aclTensor *gi = e->out;
    int64_t rows = e->outerCount, D = e->reduceCount; double scale = e->alpha;
    const float *gp = FP(go), *yp = FP(y); float *op = FP(gi);
    for (int64_t r = 0; r < rows; ++r) { const float *g = gp + r * D, *yy = yp + r * D; float *o = op + r * D;
        double dot = 0; for (int64_t d = 0; d < D; ++d) dot += (double)g[d] * yy[d];
        for (int64_t d = 0; d < D; ++d) o[d] = (float)(scale * yy[d] * (g[d] - dot));
    }
    return ACLNN_SUCCESS;
}
// ModulateBackward: forward y=x*(1+scale)+shift. gX=go*(1+scale); gScale=go*x; gShift=go.
aclnnStatus run_modulatebwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x = e->b, *scale = e->c; aclTensor *gx = e->out, *gscale = e->out2;
    aclTensor *gshift = const_cast<aclTensor *>(e->mask);
    int64_t n = x->numel(); const float *gp = FP(go), *xp = FP(x), *sp = FP(scale);
    float *gxp = gx ? FP(gx) : nullptr, *gsp = gscale ? FP(gscale) : nullptr, *ghp = gshift ? FP(gshift) : nullptr;
    for (int64_t i = 0; i < n; ++i) { double g = gp[i];
        if (gxp) gxp[i] = (float)(g * (1.0 + sp[i])); if (gsp) gsp[i] = (float)(g * xp[i]); if (ghp) ghp[i] = (float)g; }
    return ACLNN_SUCCESS;
}
// GroupedBiasAddGrad: gradBias[g,c] = Σ_{r in group g} gradOut[r,c]. groupOffset[G+1] prefix bounds.
aclnnStatus run_groupedbiasaddgrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *off = e->b; aclTensor *gb = e->out;
    int64_t G = e->m, C = e->n; const float *gp = FP(go); const int64_t *op = IP(off); float *bp = FP(gb);
    for (int64_t g = 0; g < G; ++g) { int64_t st = op[g], en = op[g + 1];
        for (int64_t c = 0; c < C; ++c) { double acc = 0; for (int64_t r = st; r < en; ++r) acc += gp[r * C + c]; bp[g * C + c] = (float)acc; } }
    return ACLNN_SUCCESS;
}
// ExpSegsumBackward: gradInput[b,k] = Σ_{i>=k, j<k} gradOut[b,i,j]*out[b,i,j]. out[B,L,L], gradInput[B,L].
aclnnStatus run_expsegsumbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *out = e->b; aclTensor *gi = e->out;
    int64_t B = e->m, L = e->n; const float *gp = FP(go), *op = FP(out); float *xp = FP(gi);
    for (int64_t b = 0; b < B; ++b) for (int64_t k = 0; k < L; ++k) { double acc = 0;
        for (int64_t i = k; i < L; ++i) for (int64_t j = 0; j < k; ++j) acc += (double)gp[(b * L + i) * L + j] * op[(b * L + i) * L + j];
        xp[b * L + k] = (float)acc; }
    return ACLNN_SUCCESS;
}

// ============================ Norm backward ============================
// RmsNormGrad: dx[d]=rcp*gj - (r3/D)*x[d]*B; gj=dy[d]*g[d]; A=Σx²; rcp=1/sqrt(A/D+eps); r3=rcp³; B=Σ gj*x.
//   dgamma[d] = Σ_r dy[d]*x[d]*rcp.
aclnnStatus run_rmsnormgrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *dy = e->a, *x = e->b, *g = e->c; aclTensor *dx = e->out, *dgamma = e->out2;
    int64_t rows = e->outerCount, D = e->reduceCount; double eps = e->eps;
    const float *dyp = FP(dy), *xp = FP(x), *gp = g ? FP(g) : nullptr;
    float *dxp = FP(dx), *dgp = dgamma ? FP(dgamma) : nullptr;
    if (dgp) for (int64_t d = 0; d < D; ++d) dgp[d] = 0.f;
    for (int64_t r = 0; r < rows; ++r) { const float *xr = xp + r * D, *dr = dyp + r * D; float *ox = dxp + r * D;
        double A = 0, B = 0;
        for (int64_t d = 0; d < D; ++d) { double xv = xr[d], gj = (double)dr[d] * (gp ? gp[d] : 1.0); A += xv * xv; B += gj * xv; }
        double rcp = 1.0 / std::sqrt(A / D + eps), r3 = rcp * rcp * rcp;
        for (int64_t d = 0; d < D; ++d) { double gj = (double)dr[d] * (gp ? gp[d] : 1.0);
            ox[d] = (float)(rcp * gj - (r3 / D) * xr[d] * B);
            if (dgp) dgp[d] += (float)((double)dr[d] * xr[d] * rcp); }
    }
    return ACLNN_SUCCESS;
}
// DeepNormGrad: in=alpha*x+gx; LN(in)*gamma+beta. dIn=rs*(dy*gamma - mean(dy*gamma) - xhat*mean(dy*gamma*xhat)).
//   gradX=alpha*dIn; gradGx=dIn; gradGamma=Σ dy*xhat; gradBeta=Σ dy.
aclnnStatus run_deepnormgrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *dy = e->a, *x = e->b, *gx = e->c, *gamma = e->mask;
    aclTensor *gradX = e->out, *gradGx = e->out2;
    aclTensor *gradGamma = e->inputs.size() >= 1 ? const_cast<aclTensor *>(e->inputs[0]) : nullptr;
    aclTensor *gradBeta = e->inputs.size() >= 2 ? const_cast<aclTensor *>(e->inputs[1]) : nullptr;
    int64_t rows = e->outerCount, D = e->reduceCount; double alpha = e->alpha, eps = e->eps;
    const float *dyp = FP(dy), *xp = FP(x), *gxp = FP(gx), *gp = gamma ? FP(gamma) : nullptr;
    float *gxo = gradX ? FP(gradX) : nullptr, *ggxo = gradGx ? FP(gradGx) : nullptr;
    float *ggam = gradGamma ? FP(gradGamma) : nullptr, *gbet = gradBeta ? FP(gradBeta) : nullptr;
    if (ggam) for (int64_t d = 0; d < D; ++d) ggam[d] = 0.f;
    if (gbet) for (int64_t d = 0; d < D; ++d) gbet[d] = 0.f;
    std::vector<double> in(D);
    for (int64_t r = 0; r < rows; ++r) { const float *xr = xp + r * D, *gxr = gxp + r * D, *dr = dyp + r * D;
        double mean = 0; for (int64_t d = 0; d < D; ++d) { in[d] = alpha * xr[d] + gxr[d]; mean += in[d]; } mean /= D;
        double var = 0; for (int64_t d = 0; d < D; ++d) { double t = in[d] - mean; var += t * t; } var /= D;
        double rs = 1.0 / std::sqrt(var + eps);
        double sa = 0, sb = 0;
        for (int64_t d = 0; d < D; ++d) { double xhat = (in[d] - mean) * rs, dyg = (double)dr[d] * (gp ? gp[d] : 1.0); sa += dyg; sb += dyg * xhat; }
        for (int64_t d = 0; d < D; ++d) { double xhat = (in[d] - mean) * rs, dyg = (double)dr[d] * (gp ? gp[d] : 1.0);
            double dIn = rs * (dyg - sa / D - xhat * sb / D);
            if (gxo) gxo[r * D + d] = (float)(alpha * dIn); if (ggxo) ggxo[r * D + d] = (float)dIn;
            if (ggam) ggam[d] += (float)((double)dr[d] * xhat); if (gbet) gbet[d] += (float)dr[d]; }
    }
    return ACLNN_SUCCESS;
}
// BatchNorm backward family (NCHW). FastBatchNormBackward: full bwd from savedMean/savedInvStd.
//   gradInput[i] = invstd*gamma*(gy - sumDy/cnt - xmu*invstd²*sumDyXmu/cnt); per channel.
//   gradGamma[c]=Σ gy*xhat; gradBeta[c]=Σ gy. xhat=(x-mean)*invstd. cnt=N*HW.
aclnnStatus run_fastbnbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gy = e->a, *x = e->b, *gamma = e->c, *mean = e->mean, *invstd = e->rstd;
    aclTensor *gx = e->out, *ggamma = e->out2;
    aclTensor *gbeta = e->inputs.size() >= 1 ? const_cast<aclTensor *>(e->inputs[0]) : nullptr;
    int64_t N = x->viewDims[0], C = x->viewDims[1], total = x->numel(), HW = total / (N * C), cnt = N * HW;
    const float *gyp = FP(gy), *xp = FP(x), *gp = gamma ? FP(gamma) : nullptr, *mp = FP(mean), *isp = FP(invstd);
    float *gxp = gx ? FP(gx) : nullptr, *ggp = ggamma ? FP(ggamma) : nullptr, *gbp = gbeta ? FP(gbeta) : nullptr;
    std::vector<double> sumDy(C, 0), sumDyXmu(C, 0);
    for (int64_t i = 0; i < total; ++i) { int64_t c = (i / HW) % C; double dy = gyp[i], xmu = (double)xp[i] - mp[c];
        sumDy[c] += dy; sumDyXmu[c] += dy * xmu; }
    if (ggp) for (int64_t c = 0; c < C; ++c) ggp[c] = (float)(sumDyXmu[c] * isp[c]);
    if (gbp) for (int64_t c = 0; c < C; ++c) gbp[c] = (float)sumDy[c];
    if (gxp) for (int64_t i = 0; i < total; ++i) { int64_t c = (i / HW) % C;
        double is = isp[c], wv = gp ? gp[c] : 1.0, dy = gyp[i], xmu = (double)xp[i] - mp[c];
        gxp[i] = (float)(is * wv * (dy - sumDy[c] / cnt - xmu * is * is * sumDyXmu[c] / cnt)); }
    return ACLNN_SUCCESS;
}
// BatchNormReduceBackward: produce sumDy, sumDyXmu, gradWeight=sumDyXmu*invstd, gradBias=sumDy.
aclnnStatus run_bnreducebwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x = e->b, *mean = e->mean, *invstd = e->rstd;
    aclTensor *sumDyT = e->out, *sumDyXmuT = e->out2;
    aclTensor *gradW = e->inputs.size() >= 1 ? const_cast<aclTensor *>(e->inputs[0]) : nullptr;
    aclTensor *gradB = e->inputs.size() >= 2 ? const_cast<aclTensor *>(e->inputs[1]) : nullptr;
    int64_t N = x->viewDims[0], C = x->viewDims[1], HW = x->numel() / (N * C);
    const float *gp = FP(go), *xp = FP(x), *mp = FP(mean), *isp = FP(invstd);
    std::vector<double> a(C, 0), b(C, 0);
    for (int64_t i = 0, total = x->numel(); i < total; ++i) { int64_t c = (i / HW) % C; double dy = gp[i]; a[c] += dy; b[c] += dy * ((double)xp[i] - mp[c]); }
    for (int64_t c = 0; c < C; ++c) {
        if (sumDyT) FP(sumDyT)[c] = (float)a[c]; if (sumDyXmuT) FP(sumDyXmuT)[c] = (float)b[c];
        if (gradW) FP(gradW)[c] = (float)(b[c] * isp[c]); if (gradB) FP(gradB)[c] = (float)a[c]; }
    return ACLNN_SUCCESS;
}
// BatchNormElemtBackward: gradInput[i] = invstd*w*(gy - sumDy/cnt - xmu*invstd²*sumDyXmu/cnt).
aclnnStatus run_bnelemtbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x = e->b, *w = e->c, *mean = e->mean, *invstd = e->rstd;
    const aclTensor *sumDy = e->inputs[0], *sumDyXmu = e->inputs[1]; aclTensor *gi = e->out;
    int64_t N = x->viewDims[0], C = x->viewDims[1], total = x->numel(), HW = total / (N * C), cnt = N * HW;
    const float *gp = FP(go), *xp = FP(x), *wp = w ? FP(w) : nullptr, *mp = FP(mean), *isp = FP(invstd);
    const float *sdp = FP(sumDy), *sxp = FP(sumDyXmu); float *op = FP(gi);
    for (int64_t i = 0; i < total; ++i) { int64_t c = (i / HW) % C; double wv = wp ? wp[c] : 1.0, is = isp[c];
        double dy = gp[i], xmu = (double)xp[i] - mp[c];
        op[i] = (float)(is * wv * (dy - sdp[c] / cnt - xmu * is * is * sxp[c] / cnt)); }
    return ACLNN_SUCCESS;
}
// GroupNorm backward (GroupNormSwishGrad reuses it). Block per (n,g) over (C/G)*HW.
//   gradX = rs*(dyg - mean_g(dyg) - xhat*mean_g(dyg*xhat)); dyg=dy*gamma. gradGamma=Σ dy*xhat; gradBeta=Σ dy.
aclnnStatus run_groupnormbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *x = e->b, *mean = e->mean, *rstd = e->rstd, *gamma = e->c;
    aclTensor *gradX = e->out, *gradGamma = e->out2;
    aclTensor *gradBeta = e->inputs.empty() ? nullptr : const_cast<aclTensor *>(e->inputs[0]);
    int64_t N = x->viewDims[0], C = x->viewDims[1], G = e->dim, HW = x->numel() / (N * C), cpg = C / G, cnt = cpg * HW;
    const float *gp = FP(go), *xp = FP(x), *mp = FP(mean), *rp = FP(rstd), *gam = gamma ? FP(gamma) : nullptr;
    float *gxp = FP(gradX), *ggp = gradGamma ? FP(gradGamma) : nullptr, *gbp = gradBeta ? FP(gradBeta) : nullptr;
    if (ggp) for (int64_t c = 0; c < C; ++c) ggp[c] = 0.f;
    if (gbp) for (int64_t c = 0; c < C; ++c) gbp[c] = 0.f;
    for (int64_t n = 0; n < N; ++n) for (int64_t g = 0; g < G; ++g) {
        double m = mp[n * G + g], rs = rp[n * G + g], sa = 0, sb = 0;
        for (int64_t cc = 0; cc < cpg; ++cc) for (int64_t h = 0; h < HW; ++h) { int64_t c = g * cpg + cc, idx = (n * C + c) * HW + h;
            double gv = gam ? gam[c] : 1.0, dy = (double)gp[idx] * gv, xhat = ((double)xp[idx] - m) * rs; sa += dy; sb += dy * xhat; }
        for (int64_t cc = 0; cc < cpg; ++cc) for (int64_t h = 0; h < HW; ++h) { int64_t c = g * cpg + cc, idx = (n * C + c) * HW + h;
            double gv = gam ? gam[c] : 1.0, dy = (double)gp[idx] * gv, xhat = ((double)xp[idx] - m) * rs;
            gxp[idx] = (float)(rs * (dy - sa / cnt - xhat * sb / cnt));
            if (ggp) ggp[c] += (float)((double)gp[idx] * xhat); if (gbp) gbp[c] += (float)gp[idx]; }
    }
    return ACLNN_SUCCESS;
}

// ============================ RoPE grad ============================
// rope grad: inverse rotation = sin negated. cos/sin are per-row [rows,D]. mode 0 half-split / 1 interleaved.
aclnnStatus run_ropegrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *cosb = e->b, *sinb = e->c; aclTensor *gi = e->out;
    int64_t rows = e->outerCount, D = e->reduceCount, half = D / 2; int mode = (int)e->dim;
    const float *xp = FP(go), *cp = FP(cosb), *sp = FP(sinb); float *op = FP(gi);
    for (int64_t r = 0; r < rows; ++r) for (int64_t d = 0; d < D; ++d) {
        int64_t i = r * D + d; double c = cp[i], si = -(double)sp[i], xv = xp[i], xpv;
        if (mode == 0) { if (d < half) { xpv = xp[r * D + d + half]; op[i] = (float)(xv * c - xpv * si); }
                         else { xpv = xp[r * D + d - half]; op[i] = (float)(xv * c + xpv * si); } }
        else { int64_t k = d / 2; if (d % 2 == 0) { xpv = xp[r * D + 2 * k + 1]; op[i] = (float)(xv * c - xpv * si); }
               else { xpv = xp[r * D + 2 * k]; op[i] = (float)(xv * c + xpv * si); } }
    }
    return ACLNN_SUCCESS;
}

// ============================ MoE permute/unpermute grad ============================
// MoeTokenPermuteGrad: gradX[srcIdx[p]] += gradPermX[p]. gradX[T,H], gradPermX[P,H].
aclnnStatus run_moepermutegrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gpx = e->a, *srcIdx = e->b; aclTensor *gx = e->out;
    int64_t T = e->m, H = e->n, P = gpx->viewDims[0];
    const float *gp = FP(gpx); const int64_t *ix = IP(srcIdx); float *xp = FP(gx);
    for (int64_t i = 0; i < T * H; ++i) xp[i] = 0.f;
    for (int64_t p = 0; p < P; ++p) { int64_t t = ix[p]; const float *g = gp + p * H;
        for (int64_t h = 0; h < H; ++h) xp[t * H + h] += g[h]; }
    return ACLNN_SUCCESS;
}
// MoeTokenUnpermuteGrad: gradPermY[p]=gradOut[srcIdx[p]]*w[p]; gradWeight[p]=Σ_h gradOut[srcIdx[p],h]*permY[p,h].
aclnnStatus run_moeunpermutegrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *go = e->a, *permY = e->b, *srcIdx = e->c, *weight = e->mask;
    aclTensor *gpy = e->out, *gw = e->out2;
    int64_t P = e->m, H = e->n;
    const float *gp = FP(go), *yp = FP(permY), *wp = weight ? FP(weight) : nullptr; const int64_t *ix = IP(srcIdx);
    float *gyp = FP(gpy), *gwp = gw ? FP(gw) : nullptr;
    for (int64_t p = 0; p < P; ++p) { int64_t t = ix[p]; double w = wp ? wp[p] : 1.0; const float *o = gp + t * H, *y = yp + p * H;
        float *g = gyp + p * H; double dot = 0;
        for (int64_t h = 0; h < H; ++h) { g[h] = (float)(o[h] * w); dot += (double)o[h] * y[h]; }
        if (gwp) gwp[p] = (float)dot; }
    return ACLNN_SUCCESS;
}

// ============================ FlashAttention score grad (standard MHA) ============================
// BNSD. Recompute P=softmax(scale*QKᵀ + mask/causal); dV=Σ_i P*dO; dP=Σ_d dO*V; dS=P*(dP-Σ P*dP);
//   dQ=scale*Σ_j dS*K; dK=scale*Σ_i dS*Q. causal tail-aligned (Skv-Sq offset).
aclnnStatus run_fagrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *Q = e->a, *K = e->b, *V = e->c, *dO = e->inputs[0], *maskT = e->mask;
    aclTensor *dQ = e->out, *dK = e->out2, *dV = const_cast<aclTensor *>(e->inputs[1]);
    int64_t B = e->ab, N = e->an, Sq = e->asq, Skv = e->askv, D = e->ad; double scale = e->alpha; bool causal = e->causal;
    int64_t maskStride = e->outerCount, off = Skv - Sq;
    const float *q = FP(Q), *k = FP(K), *v = FP(V), *dop = FP(dO);
    const uint8_t *mp = maskT ? U8(maskT) : nullptr;
    float *dq = FP(dQ), *dk = FP(dK), *dv = FP(dV);
    zero(dQ); zero(dK); zero(dV);
    std::vector<double> P(Sq * Skv), dP(Sq * Skv);
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < N; ++h) {
        const float *qb = q + ((b * N + h) * Sq) * D, *kb = k + ((b * N + h) * Skv) * D, *vb = v + ((b * N + h) * Skv) * D;
        const float *dob = dop + ((b * N + h) * Sq) * D;
        float *dqb = dq + ((b * N + h) * Sq) * D, *dkb = dk + ((b * N + h) * Skv) * D, *dvb = dv + ((b * N + h) * Skv) * D;
        const uint8_t *mb = mp ? mp + b * maskStride : nullptr;
        // P = softmax(scale*QKᵀ + mask/causal)
        for (int64_t i = 0; i < Sq; ++i) { double mx = -1e30;
            for (int64_t j = 0; j < Skv; ++j) { double sij = 0; for (int64_t d = 0; d < D; ++d) sij += (double)qb[i * D + d] * kb[j * D + d];
                sij *= scale; bool masked = (mb && mb[i * Skv + j]) || (causal && j > i + off); if (masked) sij = -1e30;
                P[i * Skv + j] = sij; mx = std::max(mx, sij); }
            double se = 0; for (int64_t j = 0; j < Skv; ++j) { double ev = std::exp(P[i * Skv + j] - mx); P[i * Skv + j] = ev; se += ev; }
            for (int64_t j = 0; j < Skv; ++j) P[i * Skv + j] /= se;
        }
        // dV[j,d] = Σ_i P[i,j]*dO[i,d]
        for (int64_t j = 0; j < Skv; ++j) for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t i = 0; i < Sq; ++i) acc += P[i * Skv + j] * dob[i * D + d]; dvb[j * D + d] = (float)acc; }
        // dP[i,j] = Σ_d dO[i,d]*V[j,d]
        for (int64_t i = 0; i < Sq; ++i) for (int64_t j = 0; j < Skv; ++j) { double acc = 0; for (int64_t d = 0; d < D; ++d) acc += (double)dob[i * D + d] * vb[j * D + d]; dP[i * Skv + j] = acc; }
        // dS[i,j] = P[i,j]*(dP[i,j] - Σ_k P[i,k]dP[i,k])  (overwrite dP)
        for (int64_t i = 0; i < Sq; ++i) { double dot = 0; for (int64_t j = 0; j < Skv; ++j) dot += P[i * Skv + j] * dP[i * Skv + j];
            for (int64_t j = 0; j < Skv; ++j) dP[i * Skv + j] = P[i * Skv + j] * (dP[i * Skv + j] - dot); }
        // dQ[i,d] = scale*Σ_j dS[i,j]*K[j,d]
        for (int64_t i = 0; i < Sq; ++i) for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t j = 0; j < Skv; ++j) acc += dP[i * Skv + j] * kb[j * D + d]; dqb[i * D + d] = (float)(acc * scale); }
        // dK[j,d] = scale*Σ_i dS[i,j]*Q[i,d]
        for (int64_t j = 0; j < Skv; ++j) for (int64_t d = 0; d < D; ++d) { double acc = 0; for (int64_t i = 0; i < Sq; ++i) acc += dP[i * Skv + j] * qb[i * D + d]; dkb[j * D + d] = (float)(acc * scale); }
    }
    return ACLNN_SUCCESS;
}

// ============================ MultiScaleDeformableAttention grad ============================
aclnnStatus run_msdagrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *value = e->a, *shapes = e->b, *lstart = e->c;
    const aclTensor *samp = e->inputs[0], *attn = e->inputs[1], *go = e->inputs[2];
    aclTensor *gValue = e->out, *gAttn = e->out2;
    int N = (int)value->viewDims[0], S = (int)value->viewDims[1], nH = (int)value->viewDims[2], hd = (int)value->viewDims[3];
    int Lq = (int)samp->viewDims[1], L = (int)samp->viewDims[3], P = (int)samp->viewDims[4];
    const float *vp = FP(value), *sp = FP(samp), *ap = FP(attn), *gp = FP(go);
    const int64_t *shp = IP(shapes), *lsp = IP(lstart);
    float *gvp = gValue ? FP(gValue) : nullptr, *gap = gAttn ? FP(gAttn) : nullptr;
    if (gValue) zero(gValue); if (gAttn) zero(gAttn);
    for (int n = 0; n < N; ++n) for (int q = 0; q < Lq; ++q) for (int h = 0; h < nH; ++h) for (int l = 0; l < L; ++l) for (int p = 0; p < P; ++p) {
        int H = (int)shp[l * 2], W = (int)shp[l * 2 + 1]; int64_t base = lsp[l];
        int64_t si = ((((int64_t)n * Lq + q) * nH + h) * L + l) * P + p;
        float x = sp[si * 2] * W - 0.5f, y = sp[si * 2 + 1] * H - 0.5f, aw = ap[si];
        int x0 = (int)std::floor(x), y0 = (int)std::floor(y), x1 = x0 + 1, y1 = y0 + 1; float dx = x - x0, dy = y - y0;
        double gattn = 0;
        for (int d = 0; d < hd; ++d) { double g = gp[(((int64_t)n * Lq + q) * nH + h) * hd + d];
            auto add = [&](int yy, int xx, double wgt) { if (yy < 0 || yy >= H || xx < 0 || xx >= W) return;
                int64_t sidx = base + (int64_t)yy * W + xx, vi = (((int64_t)n * S + sidx) * nH + h) * hd + d;
                if (gvp) gvp[vi] += (float)(g * aw * wgt); gattn += g * wgt * vp[vi]; };
            add(y0, x0, (1 - dy) * (1 - dx)); add(y0, x1, (1 - dy) * dx); add(y1, x0, dy * (1 - dx)); add(y1, x1, dy * dx); }
        if (gap) gap[si] = (float)gattn;
    }
    return ACLNN_SUCCESS;
}

// ============================ LightningIndexer grad ============================
// gradWeights[q,h] = Σ_k gradScore[q,k]*relu(q_h·k_h). query[Q,Hh,Dd], key[K,Hh,Dd].
aclnnStatus run_lightninggrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gscore = e->a, *query = e->b, *key = e->c; aclTensor *gw = e->out;
    int Q = (int)e->m, K = (int)e->k, Hh = (int)e->n, Dd = (int)e->reduceCount;
    const float *gs = FP(gscore), *qp = FP(query), *kp = FP(key); float *gwp = FP(gw);
    for (int qi = 0; qi < Q; ++qi) for (int h = 0; h < Hh; ++h) { const float *qrow = qp + ((int64_t)qi * Hh + h) * Dd; double acc = 0;
        for (int k = 0; k < K; ++k) { const float *krow = kp + ((int64_t)k * Hh + h) * Dd; double dot = 0;
            for (int d = 0; d < Dd; ++d) dot += (double)qrow[d] * krow[d]; acc += (double)gs[(int64_t)qi * K + k] * std::fmax(dot, 0.0); }
        gwp[(int64_t)qi * Hh + h] = (float)acc; }
    return ACLNN_SUCCESS;
}

// ============================ RNN: fused LSTM cell backward ============================
aclnnStatus run_lstmcellbwd(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *gates = e->a, *cprev = e->b, *cnew = e->c, *gh = e->mask;
    const aclTensor *gc = e->inputs.empty() ? nullptr : e->inputs[0];
    aclTensor *ggates = e->out, *gcprev = e->out2;
    int64_t B = e->m, Hd = e->n;
    const float *gp = FP(gates), *cp = FP(cprev), *cnp = FP(cnew);
    const float *ghp = gh ? FP(gh) : nullptr, *gcp = gc ? FP(gc) : nullptr;
    float *ggp = FP(ggates), *gcpp = gcprev ? FP(gcprev) : nullptr;
    for (int64_t b = 0; b < B; ++b) for (int64_t h = 0; h < Hd; ++h) { int64_t i = b * Hd + h; const float *g = gp + b * 4 * Hd;
        double ig = sigm(g[h]), fg = sigm(g[Hd + h]), gg = std::tanh(g[2 * Hd + h]), og = sigm(g[3 * Hd + h]);
        double tc = std::tanh((double)cnp[i]);
        double dh = ghp ? ghp[i] : 0.0, dc = (gcp ? gcp[i] : 0.0) + dh * og * (1 - tc * tc);
        double di = dc * gg, df = dc * cp[i], dgc = dc * ig, doo = dh * tc;
        float *gpp = ggp + b * 4 * Hd;
        gpp[h] = (float)(di * ig * (1 - ig));
        gpp[Hd + h] = (float)(df * fg * (1 - fg));
        gpp[2 * Hd + h] = (float)(dgc * (1 - gg * gg));
        gpp[3 * Hd + h] = (float)(doo * og * (1 - og));
        if (gcpp) gcpp[i] = (float)(dc * fg);
    }
    return ACLNN_SUCCESS;
}
// Attention-grad stubs (mirror CUDA: copy gradOut -> gradQ). LstmBackward: zero then copy gradY -> gradX.
aclnnStatus run_copygrad(aclOpExecutor *e, aclrtStream s) {
    drain(s); const aclTensor *src = e->a; aclTensor *out = e->out;
    int64_t n = std::min(src->numel(), out->numel());
    zero(out); std::memcpy(FP(out), FP(src), (size_t)n * sizeof(float));
    return ACLNN_SUCCESS;
}

#define RUN(NAME, FN) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = FN(e, s); } delete e; return st; }
} // namespace bwd_parity2

using namespace bwd_parity2;

extern "C" {
// ---- Loss elementwise backward ----
aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = target; e->mask = weight; e->out = gradInput; e->dim = 5;
    e->alpha = reduction == 1 ? 1.0 / (double)gradInput->numel() : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBinaryCrossEntropyBackward, run_lossgrad)
aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, const aclTensor *posWeight, int64_t reduction, aclTensor *gradTarget, uint64_t *ws, aclOpExecutor **ex) {
    (void)target; (void)posWeight; if (!gradOutput || !gradTarget || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = nullptr; e->mask = weight; e->out = gradTarget; e->dim = 7;
    e->alpha = reduction == 1 ? 1.0 / (double)gradTarget->numel() : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBinaryCrossEntropyWithLogitsTargetBackward, run_lossgrad)
aclnnStatus aclnnKlDivBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)logTarget; if (!gradOutput || !target || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = nullptr; e->c = target; e->mask = nullptr; e->out = gradInput; e->dim = 8;
    e->alpha = reduction == 1 ? 1.0 / (double)gradInput->numel() : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnKlDivBackward, run_lossgrad)
aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *gradTarget, uint64_t *ws, aclOpExecutor **ex) {
    (void)logTarget; if (!gradOutput || !self || !target || !gradTarget || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = target; e->mask = nullptr; e->out = gradTarget; e->dim = 9;
    e->alpha = reduction == 1 ? 1.0 / (double)gradTarget->numel() : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnKlDivTargetBackward, run_lossgrad)
aclnnStatus aclnnSoftMarginLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)reduction; if (!gradOutput || !self || !target || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = self; e->c = target; e->mask = nullptr; e->out = gradInput; e->dim = 4;
    e->alpha = 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnSoftMarginLossBackward, run_lossgrad)
// NLLLoss backward (+2d)
aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)self; (void)ignoreIndex; (void)totalWeight; if (!gradOutput || !target || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = target; e->mask = weight; e->out = gradInput;
    int64_t N = gradInput->viewDims[0]; e->m = N; e->n = gradInput->viewDims[1]; e->dim = reduction == 0 ? 0 : 1;
    e->alpha = reduction == 1 ? 1.0 / (double)N : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnNLLLossBackward, run_nllbwd)
aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnNLLLossBackwardGetWorkspaceSize(gradOutput, self, target, weight, reduction, ignoreIndex, totalWeight, gradInput, ws, ex); }
RUN(aclnnNLLLoss2dBackward, run_nllbwd)
aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double gradOutput, int64_t reduction, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !target || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = self; e->b = target; e->out = gradInput; e->m = self->viewDims[0]; e->n = self->viewDims[1];
    e->alpha = gradOutput; e->eps = reduction == 1 ? 1.0 / (double)e->m : 1.0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFusedLinearCrossEntropyLossGrad, run_cegrad)
aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets, const aclTensor *inputLengths, const aclTensor *targetLengths, const aclTensor *negLogLikelihood, const aclTensor *logAlpha, int64_t blank, bool zeroInfinity, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)targets; (void)inputLengths; (void)targetLengths; (void)negLogLikelihood; (void)logAlpha; (void)blank; (void)zeroInfinity;
    if (!gradOut || !logProbs || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = logProbs; e->out = gradInput; e->dim = (gradOut->numel() == 1) ? 1 : 0; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnCtcLossBackward, run_ctcbwd)
aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *out, const aclTensor *mask, double scale, bool fixedTriuMask, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    (void)mask; (void)fixedTriuMask; if (!gradOutput || !out || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = out->viewDims.back(); auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = out; e->out = gradInput;
    e->reduceCount = D; e->outerCount = out->numel() / D; e->alpha = scale; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnScaledMaskedSoftmaxBackward, run_smsoftmaxbwd)
aclnnStatus aclnnModulateBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *scale, aclTensor *gradX, aclTensor *gradScale, aclTensor *gradShift, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !x || !scale || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = x; e->c = scale; e->out = gradX; e->out2 = gradScale; e->mask = gradShift; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnModulateBackward, run_modulatebwd)
aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *groupOffset, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !groupOffset || !gradBias || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = groupOffset; e->out = gradBias; e->m = gradBias->viewDims[0]; e->n = gradBias->viewDims[1]; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupedBiasAddGrad, run_groupedbiasaddgrad)
aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *groupOffset, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnGroupedBiasAddGradGetWorkspaceSize(gradOutput, groupOffset, gradBias, ws, ex); }
RUN(aclnnGroupedBiasAddGradV2, run_groupedbiasaddgrad)
aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *out, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !out || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t L = gradInput->viewDims.back(); auto *e = new aclOpExecutor(); e->a = gradOutput; e->b = out; e->out = gradInput; e->n = L; e->m = gradInput->numel() / L; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnExpSegsumBackward, run_expsegsumbwd)

// ---- Norm backward ----
aclnnStatus aclnnRmsNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = x->viewDims.back(); auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gamma; e->out = gradX; e->out2 = gradGamma;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->eps = eps; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRmsNormGrad, run_rmsnormgrad)
aclnnStatus aclnnDeepNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gx, const aclTensor *gamma, double alpha, double eps, aclTensor *gradX, aclTensor *gradGx, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !gx || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gx; e->mask = gamma; e->out = gradX; e->out2 = gradGx;
    if (gradGamma) e->inputs.push_back(gradGamma); if (gradBeta) e->inputs.push_back(gradBeta);
    e->alpha = alpha; e->eps = eps; e->reduceCount = x->viewDims.back(); e->outerCount = x->numel() / e->reduceCount; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnDeepNormGrad, run_deepnormgrad)
aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, const aclTensor *savedMean, const aclTensor *savedInvStd, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradY || !x || !savedMean || !savedInvStd || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->b = x; e->c = gamma; e->mean = savedMean; e->rstd = savedInvStd; e->out = gradX; e->out2 = gradGamma;
    if (gradBeta) e->inputs.push_back(gradBeta); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnFastBatchNormBackward, run_fastbnbwd)
aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !self || !mean || !invstd || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->mean = mean; e->rstd = invstd; e->out = sumDy; e->out2 = sumDyXmu;
    if (gradWeight) e->inputs.push_back(gradWeight); if (gradBias) e->inputs.push_back(gradBias); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBatchNormReduceBackward, run_bnreducebwd)
aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, const aclTensor *weight, const aclTensor *sumDy, const aclTensor *sumDyXmu, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !self || !mean || !invstd || !sumDy || !sumDyXmu || !gradInput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->c = weight; e->mean = mean; e->rstd = invstd; e->out = gradInput;
    e->inputs.push_back(sumDy); e->inputs.push_back(sumDyXmu); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnBatchNormElemtBackward, run_bnelemtbwd)
aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma, int64_t numGroups, double swishScale, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *ws, aclOpExecutor **ex) {
    (void)swishScale; if (!gradOut || !self || !mean || !rstd || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = self; e->mean = mean; e->rstd = rstd; e->c = gamma; e->out = gradX; e->out2 = gradGamma;
    if (gradBeta) e->inputs.push_back(gradBeta); e->dim = numGroups; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnGroupNormSwishGrad, run_groupnormbwd)

// ---- RoPE grad ----
aclnnStatus aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !cos || !sin || !out || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = x->viewDims.back(); auto *e = new aclOpExecutor(); e->a = x; e->b = cos; e->c = sin; e->out = out;
    e->reduceCount = D; e->outerCount = x->numel() / D; e->dim = mode; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnRotaryPositionEmbeddingGrad, run_ropegrad)
aclnnStatus aclnnNormRopeConcatBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnRotaryPositionEmbeddingGradGetWorkspaceSize(x, cos, sin, mode, out, ws, ex); }
RUN(aclnnNormRopeConcatBackward, run_ropegrad)

// ---- MoE permute/unpermute grad ----
aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradPermX || !srcIdx || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradPermX; e->b = srcIdx; e->out = gradX; e->m = gradX->viewDims[0]; e->n = gradX->viewDims[1]; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMoeTokenPermuteGrad, run_moepermutegrad)
aclnnStatus aclnnMoeTokenPermuteWithEpGradGetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMoeTokenPermuteGradGetWorkspaceSize(gradPermX, srcIdx, gradX, ws, ex); }
RUN(aclnnMoeTokenPermuteWithEpGrad, run_moepermutegrad)
aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGradGetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMoeTokenPermuteGradGetWorkspaceSize(gradPermX, srcIdx, gradX, ws, ex); }
RUN(aclnnMoeTokenPermuteWithRoutingMapGrad, run_moepermutegrad)
aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !permY || !srcIdx || !gradPermY || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradOut; e->b = permY; e->c = srcIdx; e->mask = weight; e->out = gradPermY; e->out2 = gradWeight;
    e->m = permY->viewDims[0]; e->n = permY->viewDims[1]; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMoeTokenUnpermuteGrad, run_moeunpermutegrad)
aclnnStatus aclnnMoeTokenUnpermuteWithEpGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMoeTokenUnpermuteGradGetWorkspaceSize(gradOut, permY, srcIdx, weight, gradPermY, gradWeight, ws, ex); }
RUN(aclnnMoeTokenUnpermuteWithEpGrad, run_moeunpermutegrad)
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnMoeTokenUnpermuteGradGetWorkspaceSize(gradOut, permY, srcIdx, weight, gradPermY, gradWeight, ws, ex); }
RUN(aclnnMoeTokenUnpermuteWithRoutingMapGrad, run_moeunpermutegrad)

// ---- FlashAttention score grad family ----
#define FAG(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *ws, aclOpExecutor **ex) { \
    (void)headNum; if (!q || !k || !v || !dy || !dq || !dk || !dv || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    if (q->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID; \
    int64_t B = q->viewDims[0], N = q->viewDims[1], Sq = q->viewDims[2], D = q->viewDims[3], Skv = k->viewDims[2]; \
    if (k->viewDims[1] != N) return ACLNN_ERR_PARAM_INVALID; \
    int64_t maskStride = 0; \
    if (attenMask) { if (attenMask->viewDims == std::vector<int64_t>{Sq, Skv}) maskStride = 0; \
        else if (attenMask->viewDims == std::vector<int64_t>{B, Sq, Skv}) maskStride = Sq * Skv; else return ACLNN_ERR_PARAM_INVALID; } \
    auto *e = new aclOpExecutor(); e->a = q; e->b = k; e->c = v; e->out = dq; e->out2 = dk; e->inputs.push_back(dy); e->inputs.push_back(dv); \
    e->mask = attenMask; e->ab = B; e->an = N; e->asq = Sq; e->askv = Skv; e->ad = D; e->outerCount = maskStride; \
    e->alpha = (scaleValue != 0.0) ? scaleValue : 1.0 / std::sqrt((double)D); e->causal = causal; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_fagrad)
FAG(aclnnFlashAttentionScoreGrad)
FAG(aclnnFlashAttentionScoreGradV2)
FAG(aclnnFlashAttentionScoreGradV3)
FAG(aclnnFlashAttentionScoreGradV4)
FAG(aclnnFlashAttentionUnpaddingScoreGrad)
FAG(aclnnFlashAttentionUnpaddingScoreGradV2)
FAG(aclnnFlashAttentionUnpaddingScoreGradV3)
FAG(aclnnFlashAttentionUnpaddingScoreGradV4)
FAG(aclnnFlashAttentionUnpaddingScoreGradV5)
FAG(aclnnQuantFlashAttentionScoreGrad)
FAG(aclnnSparseFlashAttentionGrad)
#undef FAG
// Attention-grad stubs (CUDA copies gradOut -> gradQ; q/k/v ignored)
#define ATTG_STUB(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOut, const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *gradQ, uint64_t *ws, aclOpExecutor **ex) { \
    (void)q; (void)k; (void)v; if (!gradOut || !gradQ || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR; \
    auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradQ; *ws = 0; *ex = e; return ACLNN_SUCCESS; } \
RUN(NAME, run_copygrad)
ATTG_STUB(aclnnNsaSelectedAttentionGrad)
ATTG_STUB(aclnnBlockSparseAttentionGrad)
ATTG_STUB(aclnnFusedFloydAttentionGrad)
#undef ATTG_STUB

// ---- NsaCompressGrad: block mean-pool inverse. gradIn[i] = gradOut[block]/blockSize ----
aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(const aclTensor *gradOut, int64_t blockSize, aclTensor *gradIn, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOut || !gradIn || !ws || !ex || blockSize <= 0) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t D = gradIn->viewDims.back(); auto *e = new aclOpExecutor(); e->a = gradOut; e->out = gradIn; e->k = D; e->m = gradOut->numel() / D; e->n = blockSize; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
aclnnStatus aclnnNsaCompressGrad(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) {
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR;
    @autoreleasepool { drain(s); int64_t Nb = e->m, bs = e->n, D = e->k; const float *go = FP(e->a); float *gi = FP(e->out);
        for (int64_t i = 0, tot = Nb * bs * D; i < tot; ++i) { int64_t d = i % D, row = i / D, b = row / bs; gi[i] = go[b * D + d] / (float)bs; } }
    delete e; return ACLNN_SUCCESS; }

// ---- MultiScaleDeformableAttention grad ----
aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes, const aclTensor *levelStartIndex, const aclTensor *samplingLocations, const aclTensor *attnWeights, const aclTensor *gradOutput, aclTensor *gradValue, aclTensor *gradSampling, aclTensor *gradAttnWeights, uint64_t *ws, aclOpExecutor **ex) {
    if (!value || !gradOutput || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = value; e->b = spatialShapes; e->c = levelStartIndex; e->out = gradValue; e->out2 = gradAttnWeights; e->mask = gradSampling;
    e->inputs.push_back(samplingLocations); e->inputs.push_back(attnWeights); e->inputs.push_back(gradOutput); *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnMultiScaleDeformableAttentionGrad, run_msdagrad)

// ---- LightningIndexer grad ----
aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(const aclTensor *gradIndexScore, const aclTensor *query, const aclTensor *key, aclTensor *gradWeights, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradIndexScore || !query || !key || !gradWeights || !ws || !ex || query->viewDims.size() != 3) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradIndexScore; e->b = query; e->c = key; e->out = gradWeights;
    e->m = query->viewDims[0]; e->k = key->viewDims[0]; e->n = query->viewDims[1]; e->reduceCount = query->viewDims[2]; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLightningIndexerGrad, run_lightninggrad)

// ---- RNN ----
aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(const aclTensor *gradHy, const aclTensor *gradCy, const aclTensor *cprev, const aclTensor *cNew, const aclTensor *gates, aclTensor *gradGates, aclTensor *gradCprev, uint64_t *ws, aclOpExecutor **ex) {
    if (!cprev || !cNew || !gates || !gradGates || !ws || !ex || cprev->viewDims.size() != 2) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gates; e->b = cprev; e->c = cNew; e->mask = gradHy; e->out = gradGates; e->out2 = gradCprev;
    e->inputs.push_back(gradCy); e->m = cprev->viewDims[0]; e->n = cprev->viewDims[1]; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnThnnFusedLstmCellBackward, run_lstmcellbwd)
aclnnStatus aclnnLstmBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *wih, const aclTensor *whh, aclTensor *gradX, uint64_t *ws, aclOpExecutor **ex) {
    (void)x; (void)wih; (void)whh; if (!gradY || !gradX || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->a = gradY; e->out = gradX; *ws = 0; *ex = e; return ACLNN_SUCCESS; }
RUN(aclnnLstmBackward, run_copygrad)
} // extern "C"
