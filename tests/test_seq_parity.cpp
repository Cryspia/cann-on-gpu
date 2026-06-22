// Sequence / expert operator gap cross-check (CUDA parity), self-contained CPU references.
// Covers the 23 ops implemented in metal/src/ops/seq_parity.mm:
//   MoE/MLA (18): MhcPre, MhcPost, MhcSinkhorn, MlaPreprocess, MlaPreprocessV2, MlaProlog,
//                 MlaPrologV2WeightNz, MlaPrologV3WeightNz, MoeFinalizeRoutingV2, MoeFinalizeRoutingV3,
//                 MoeFusedTopk, MoeGatingTopK, MoeGatingTopKSoftmaxV2, MoeInitRoutingV2, MoeInitRoutingV3,
//                 MoeTokenPermuteWithRoutingMap, MoeTokenUnpermuteWithEp, MoeTokenUnpermuteWithRoutingMap
//   RNN/SSM (4):  LSTM, BidirectionLSTM, BidirectionLSTMV2, RecurrentGatedDeltaRule
//   Attention (1): TransformBiasRescaleQkv
// Tolerances: 1e-5 fp32 (statistical/recurrence 1e-4). Index/grouping checks are exact (tol 0).
#include "harness.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
using namespace hn;

// ---- extern "C" prototypes matching what seq_parity.mm declares (== aclnn_ops.h signatures) ----
extern "C" {
aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMhcPre(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMhcPost(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMhcSinkhorn(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnMlaPreprocessGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMlaPreprocess(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMlaPreprocessV2GetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMlaPreprocessV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMlaPrologGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMlaProlog(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMlaPrologV2WeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMlaPrologV2WeightNz(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMlaPrologV3WeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMlaPrologV3WeightNz(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnMoeFusedTopkGetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeFusedTopk(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeGatingTopKGetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeGatingTopK(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeGatingTopKSoftmaxV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnMoeInitRoutingV2GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeInitRoutingV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeInitRoutingV3GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeInitRoutingV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnMoeFinalizeRoutingV2GetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeFinalizeRoutingV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeFinalizeRoutingV3GetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeFinalizeRoutingV3(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeTokenPermuteWithRoutingMap(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeTokenUnpermuteWithEp(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnMoeTokenUnpermuteWithRoutingMap(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnLSTM(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnBidirectionLSTM(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnBidirectionLSTMV2GetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnBidirectionLSTMV2(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnRecurrentGatedDeltaRule(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);

aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(const aclTensor *qkv, const aclTensor *bias, int64_t headDim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex);
aclnnStatus aclnnTransformBiasRescaleQkv(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s);
}

static double sigd(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// ---------------- MhcPre / MhcPost: identity copy ----------------
static void t_mhc_copy() {
    const int N = 37; auto x = randv(N, -3, 3); std::vector<float> ho(N);
    {
        DevBuf dx(N * 4), dout(N * 4); dx.up(x.data());
        auto ti = mk({N}, ACL_FLOAT, dx.p), to = mk({N}, ACL_FLOAT, dout.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMhcPreGetWorkspaceSize(ti, to, w, e); }, aclnnMhcPre);
        dout.down(ho.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, (double)std::fabs(ho[i] - x[i]));
        report("MhcPre (identity)", bad, 0.0); aclDestroyTensor(ti); aclDestroyTensor(to);
    }
    {
        DevBuf dx(N * 4), dout(N * 4); dx.up(x.data());
        auto ti = mk({N}, ACL_FLOAT, dx.p), to = mk({N}, ACL_FLOAT, dout.p);
        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMhcPostGetWorkspaceSize(ti, to, w, e); }, aclnnMhcPost);
        dout.down(ho.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, (double)std::fabs(ho[i] - x[i]));
        report("MhcPost (identity)", bad, 0.0); aclDestroyTensor(ti); aclDestroyTensor(to);
    }
}

// ---------------- MhcSinkhorn: row/col-normalize iterations ----------------
static void t_sinkhorn() {
    const int R = 4, C = 5, iters = 6; auto cost = randv(R * C, 0.1f, 2.0f);
    std::vector<float> ho(R * C);
    DevBuf dc(R * C * 4), dout(R * C * 4); dc.up(cost.data());
    auto tc = mk({R, C}, ACL_FLOAT, dc.p), to = mk({R, C}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMhcSinkhornGetWorkspaceSize(tc, 1.0, iters, to, w, e); }, aclnnMhcSinkhorn);
    dout.down(ho.data());
    // CPU reference: copy then alternate row-norm, col-norm for iters
    std::vector<double> m(R * C); for (int i = 0; i < R * C; i++) m[i] = cost[i];
    for (int it = 0; it < iters; it++) {
        for (int r = 0; r < R; r++) { double s = 0; for (int c = 0; c < C; c++) s += m[r * C + c]; if (s > 0) for (int c = 0; c < C; c++) m[r * C + c] /= s; }
        for (int c = 0; c < C; c++) { double s = 0; for (int r = 0; r < R; r++) s += m[r * C + c]; if (s > 0) for (int r = 0; r < R; r++) m[r * C + c] /= s; }
    }
    double me = 0, mr = 0; for (int i = 0; i < R * C; i++) { me = std::max(me, std::fabs(ho[i] - m[i])); mr = std::max(mr, std::fabs(m[i])); }
    report("MhcSinkhorn", me / (mr + 1e-9), 1e-5);
    aclDestroyTensor(tc); aclDestroyTensor(to);
}

// ---------------- MLA preprocess/prolog: per-row RMSNorm + RoPE (5 ops) ----------------
typedef aclnnStatus (*MlaWs)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, double, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*MlaRun)(void *, uint64_t, aclOpExecutor *, aclrtStream);
static void run_mla_op(const char *name, MlaWs ws, MlaRun run, int mode) {
    const int rows = 6, D = 8, half = D / 2; const double eps = 1e-5;
    auto x = randv(rows * D, -2, 2), gamma = randv(D, 0.5f, 1.5f);
    std::vector<float> cosb(rows * D), sinb(rows * D);
    for (int r = 0; r < rows; r++) for (int d = 0; d < D; d++) { double ang = 0.05 * (r + 1) * (d + 1); cosb[r * D + d] = std::cos(ang); sinb[r * D + d] = std::sin(ang); }
    std::vector<float> ho(rows * D);
    DevBuf dx(rows * D * 4), dg(D * 4), dc(rows * D * 4), ds(rows * D * 4), dout(rows * D * 4);
    dx.up(x.data()); dg.up(gamma.data()); dc.up(cosb.data()); ds.up(sinb.data());
    auto tx = mk({rows, D}, ACL_FLOAT, dx.p), tg = mk({D}, ACL_FLOAT, dg.p),
         tc = mk({rows, D}, ACL_FLOAT, dc.p), ts = mk({rows, D}, ACL_FLOAT, ds.p), to = mk({rows, D}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return ws(tx, tg, tc, ts, eps, mode, to, w, e); }, run);
    dout.down(ho.data());
    std::vector<double> ref(rows * D);
    for (int r = 0; r < rows; r++) {
        const float *p = &x[r * D]; double ss = 0; for (int d = 0; d < D; d++) ss += (double)p[d] * p[d];
        double rms = 1.0 / std::sqrt(ss / D + eps);
        auto nrm = [&](int d) { return p[d] * rms * gamma[d]; };
        for (int d = 0; d < D; d++) {
            double n = nrm(d), c = cosb[r * D + d], sn = sinb[r * D + d], np;
            if (mode == 0) { if (d < half) np = n * c - nrm(d + half) * sn; else np = n * c + nrm(d - half) * sn; }
            else { int k = d / 2; if (d % 2 == 0) np = n * c - nrm(2 * k + 1) * sn; else np = n * c + nrm(2 * k) * sn; }
            ref[r * D + d] = np;
        }
    }
    double me = 0, mr = 0; for (int i = 0; i < rows * D; i++) { me = std::max(me, std::fabs(ho[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    report(name, me / (mr + 1e-9), 1e-5);
    aclDestroyTensor(tx); aclDestroyTensor(tg); aclDestroyTensor(tc); aclDestroyTensor(ts); aclDestroyTensor(to);
}
static void t_mla() {
    run_mla_op("MlaPreprocess (rmsnorm+rope)", aclnnMlaPreprocessGetWorkspaceSize, aclnnMlaPreprocess, 0);
    run_mla_op("MlaPreprocessV2", aclnnMlaPreprocessV2GetWorkspaceSize, aclnnMlaPreprocessV2, 0);
    run_mla_op("MlaProlog", aclnnMlaPrologGetWorkspaceSize, aclnnMlaProlog, 0);
    run_mla_op("MlaPrologV2WeightNz", aclnnMlaPrologV2WeightNzGetWorkspaceSize, aclnnMlaPrologV2WeightNz, 1);  // interleaved mode
    run_mla_op("MlaPrologV3WeightNz", aclnnMlaPrologV3WeightNzGetWorkspaceSize, aclnnMlaPrologV3WeightNz, 0);
}

// ---------------- MoE gating variants: softmax top-K renormalized (3 ops) ----------------
typedef aclnnStatus (*GateWs)(const aclTensor *, int64_t, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
static void run_gate_op(const char *name, GateWs ws, MlaRun run) {
    const int T = 24, E = 8, K = 3; auto lg = randv(T * E, -3, 3);
    std::vector<float> hw(T * K); std::vector<int32_t> hi(T * K);
    DevBuf dl(T * E * 4), dw(T * K * 4), di(T * K * 4); dl.up(lg.data());
    auto tl = mk({T, E}, ACL_FLOAT, dl.p), tw = mk({T, K}, ACL_FLOAT, dw.p); auto ti = mk({T, K}, ACL_INT32, di.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return ws(tl, K, tw, ti, w, e); }, run);
    dw.down(hw.data()); di.down(hi.data());
    double me = 0, mr = 0; int ibad = 0;
    for (int t = 0; t < T; t++) {
        const float *row = &lg[t * E];
        double mx = -1e30; for (int e = 0; e < E; e++) mx = std::max(mx, (double)row[e]);
        double se = 0; for (int e = 0; e < E; e++) se += std::exp((double)row[e] - mx);
        // greedy top-K by logit, low-index tie-break (strict >, ascending scan)
        std::vector<char> used(E, 0); std::vector<int> ord(K); std::vector<double> p(K); double wsum = 0;
        for (int k = 0; k < K; k++) {
            int best = -1; double bv = -1e30;
            for (int e = 0; e < E; e++) { if (used[e]) continue; if ((double)row[e] > bv) { bv = row[e]; best = e; } }
            used[best] = 1; ord[k] = best; p[k] = std::exp((double)row[best] - mx) / se; wsum += p[k];
        }
        for (int k = 0; k < K; k++) { if (hi[t * K + k] != ord[k]) ibad++; double ref = p[k] / wsum; me = std::max(me, std::fabs(hw[t * K + k] - ref)); mr = std::max(mr, std::fabs(ref)); }
    }
    report((std::string(name) + " w").c_str(), me / (mr + 1e-9), 1e-5);
    report((std::string(name) + " idx").c_str(), ibad ? 1.0 : 0.0, 0.0);
    aclDestroyTensor(tl); aclDestroyTensor(tw); aclDestroyTensor(ti);
}
static void t_gating() {
    run_gate_op("MoeFusedTopk", aclnnMoeFusedTopkGetWorkspaceSize, aclnnMoeFusedTopk);
    run_gate_op("MoeGatingTopK", aclnnMoeGatingTopKGetWorkspaceSize, aclnnMoeGatingTopK);
    run_gate_op("MoeGatingTopKSoftmaxV2", aclnnMoeGatingTopKSoftmaxV2GetWorkspaceSize, aclnnMoeGatingTopKSoftmaxV2);
}

// ---------------- MoE InitRouting + FinalizeRouting round-trip (V2 and V3) ----------------
typedef aclnnStatus (*InitWs)(const aclTensor *, const aclTensor *, int64_t, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
typedef aclnnStatus (*FinalWs)(const aclTensor *, const aclTensor *, const aclTensor *, int64_t, aclTensor *, uint64_t *, aclOpExecutor **);
static void run_init_final(const char *tag, InitWs iws, MlaRun irun, FinalWs fws, MlaRun frun) {
    const int T = 16, H = 8, K = 2, E = 4, M = T * K;
    auto x = randv(T * H, -1, 1); std::vector<int32_t> eid(M); for (int i = 0; i < M; i++) eid[i] = rand() % E;
    auto scales = randv(M, 0.1f, 1.0f);
    DevBuf dx(T * H * 4), de(M * 4), dex(M * H * 4), drow(M * 4), dexp(M * 4); dx.up(x.data()); de.up(eid.data());
    auto tx = mk({T, H}, ACL_FLOAT, dx.p), te = mk({T, K}, ACL_INT32, de.p),
         tex = mk({M, H}, ACL_FLOAT, dex.p); auto trow = mk({M}, ACL_INT32, drow.p), texp = mk({M}, ACL_INT32, dexp.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return iws(tx, te, E, tex, trow, texp, w, e); }, irun);
    std::vector<float> hex(M * H); std::vector<int32_t> hrow(M), hexp(M); dex.down(hex.data()); drow.down(hrow.data()); dexp.down(hexp.data());
    // expert ids non-decreasing (grouped), expandedX[p] == x[rowIdx[p]/K], expandedExpertIdx[p] == eid[rowIdx[p]]
    double me = 0, mr = 0; bool ok = true; int prev = -1;
    for (int p = 0; p < M; p++) {
        if (hexp[p] < prev) ok = false; prev = hexp[p];
        int flat = hrow[p], t = flat / K; if (hexp[p] != eid[flat]) ok = false;
        for (int h = 0; h < H; h++) { me = std::max(me, std::fabs((double)hex[p * H + h] - x[t * H + h])); mr = std::max(mr, std::fabs((double)x[t * H + h])); }
    }
    report((std::string(tag) + " InitRouting group").c_str(), ok ? me / (mr + 1e-9) : 1.0, 1e-6);
    // FinalizeRouting: out[t] = sum_k scales[t,k]*expandedY[p(t,k)] (use expandedX as Y)
    DevBuf ds(M * 4), dout(T * H * 4); ds.up(scales.data());
    auto ts = mk({T, K}, ACL_FLOAT, ds.p), tout = mk({T, H}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return fws(tex, trow, ts, K, tout, w, e); }, frun);
    std::vector<float> hout(T * H); dout.down(hout.data());
    std::vector<double> ref(T * H, 0); for (int p = 0; p < M; p++) { int flat = hrow[p], t = flat / K, k = flat % K; for (int h = 0; h < H; h++) ref[t * H + h] += (double)scales[t * K + k] * hex[p * H + h]; }
    double fe = 0, fr = 0; for (int i = 0; i < T * H; i++) { fe = std::max(fe, std::fabs(hout[i] - ref[i])); fr = std::max(fr, std::fabs(ref[i])); }
    report((std::string(tag) + " FinalizeRouting").c_str(), fe / (fr + 1e-9), 1e-5);
    aclDestroyTensor(tx); aclDestroyTensor(te); aclDestroyTensor(tex); aclDestroyTensor(trow); aclDestroyTensor(texp);
    aclDestroyTensor(ts); aclDestroyTensor(tout);
}
static void t_init_final() {
    run_init_final("V2", aclnnMoeInitRoutingV2GetWorkspaceSize, aclnnMoeInitRoutingV2, aclnnMoeFinalizeRoutingV2GetWorkspaceSize, aclnnMoeFinalizeRoutingV2);
    run_init_final("V3", aclnnMoeInitRoutingV3GetWorkspaceSize, aclnnMoeInitRoutingV3, aclnnMoeFinalizeRoutingV3GetWorkspaceSize, aclnnMoeFinalizeRoutingV3);
}

// ---------------- MoE TokenPermuteWithRoutingMap + Unpermute round-trip (WithEp / WithRoutingMap) ----------------
static void t_permute() {
    const int T = 20, H = 6, E = 4; auto x = randv(T * H, -2, 2); std::vector<int32_t> ex(T); for (auto &v : ex) v = rand() % E;
    std::vector<float> hperm(T * H), hout(T * H), hout2(T * H); std::vector<int64_t> hsrc(T);
    DevBuf dx(T * H * 4), dex(T * 4), dperm(T * H * 4), dsrc(T * 8), dout(T * H * 4), dout2(T * H * 4);
    dx.up(x.data()); dex.up(ex.data());
    auto tx = mk({T, H}, ACL_FLOAT, dx.p), tex = mk({T}, ACL_INT32, dex.p),
         tperm = mk({T, H}, ACL_FLOAT, dperm.p); auto tsrc = mk({T}, ACL_INT64, dsrc.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMoeTokenPermuteWithRoutingMapGetWorkspaceSize(tx, tex, E, tperm, tsrc, w, e); }, aclnnMoeTokenPermuteWithRoutingMap);
    dperm.down(hperm.data()); dsrc.down(hsrc.data());
    bool grouped = true; int prev = -1; double cbad = 0;
    for (int p = 0; p < T; p++) { int src = hsrc[p]; int e = ex[src]; if (e < prev) grouped = false; prev = e;
        for (int h = 0; h < H; h++) cbad = std::max(cbad, (double)std::fabs(hperm[p * H + h] - x[src * H + h])); }
    report("MoeTokenPermuteWithRoutingMap grouped", grouped ? cbad : 1.0, 1e-6);
    // WithEp unpermute (weight=null) reconstructs x
    auto tperm2 = mk({T, H}, ACL_FLOAT, dperm.p), tout = mk({T, H}, ACL_FLOAT, dout.p); auto tsrc2 = mk({T}, ACL_INT64, dsrc.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMoeTokenUnpermuteWithEpGetWorkspaceSize(tperm2, tsrc2, nullptr, tout, w, e); }, aclnnMoeTokenUnpermuteWithEp);
    dout.down(hout.data()); double bad = 0; for (int i = 0; i < T * H; i++) bad = std::max(bad, (double)std::fabs(hout[i] - x[i]));
    report("MoeTokenUnpermuteWithEp roundtrip", bad, 1e-5);
    // WithRoutingMap unpermute with weights: out[src[p]] = perm[p]*w[p]
    auto wts = randv(T, 0.2f, 1.5f);
    DevBuf dwt(T * 4); dwt.up(wts.data());
    auto tperm3 = mk({T, H}, ACL_FLOAT, dperm.p), tout3 = mk({T, H}, ACL_FLOAT, dout2.p); auto tsrc3 = mk({T}, ACL_INT64, dsrc.p), twt = mk({T}, ACL_FLOAT, dwt.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMoeTokenUnpermuteWithRoutingMapGetWorkspaceSize(tperm3, tsrc3, twt, tout3, w, e); }, aclnnMoeTokenUnpermuteWithRoutingMap);
    dout2.down(hout2.data());
    std::vector<double> ref(T * H, 0); for (int p = 0; p < T; p++) { int dst = hsrc[p]; for (int h = 0; h < H; h++) ref[dst * H + h] += (double)hperm[p * H + h] * wts[p]; }
    double me = 0, mr = 0; for (int i = 0; i < T * H; i++) { me = std::max(me, std::fabs(hout2[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    report("MoeTokenUnpermuteWithRoutingMap (weighted)", me / (mr + 1e-9), 1e-5);
    aclDestroyTensor(tx); aclDestroyTensor(tex); aclDestroyTensor(tperm); aclDestroyTensor(tsrc);
    aclDestroyTensor(tperm2); aclDestroyTensor(tsrc2); aclDestroyTensor(tout);
    aclDestroyTensor(tperm3); aclDestroyTensor(tsrc3); aclDestroyTensor(twt); aclDestroyTensor(tout3);
}

// ---------------- LSTM / BidirectionLSTM(V2): single-layer forward ----------------
typedef aclnnStatus (*LstmWs)(const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, const aclTensor *, aclTensor *, aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
static void run_lstm_op(const char *name, LstmWs ws, MlaRun run) {
    const int T = 5, B = 3, I = 4, H = 6;
    auto x = randv(T * B * I, -1, 1), wih = randv(4 * H * I, -0.5f, 0.5f), whh = randv(4 * H * H, -0.5f, 0.5f);
    auto bih = randv(4 * H, -0.3f, 0.3f), bhh = randv(4 * H, -0.3f, 0.3f), h0 = randv(B * H, -0.5f, 0.5f), c0 = randv(B * H, -0.5f, 0.5f);
    std::vector<float> hy(T * B * H), hhN(B * H), hcN(B * H);
    DevBuf dx(T * B * I * 4), dwih(4 * H * I * 4), dwhh(4 * H * H * 4), dbih(4 * H * 4), dbhh(4 * H * 4), dh0(B * H * 4), dc0(B * H * 4), dy(T * B * H * 4), dhN(B * H * 4), dcN(B * H * 4);
    dx.up(x.data()); dwih.up(wih.data()); dwhh.up(whh.data()); dbih.up(bih.data()); dbhh.up(bhh.data()); dh0.up(h0.data()); dc0.up(c0.data());
    auto tx = mk({T, B, I}, ACL_FLOAT, dx.p), twih = mk({4 * H, I}, ACL_FLOAT, dwih.p), twhh = mk({4 * H, H}, ACL_FLOAT, dwhh.p),
         tbih = mk({4 * H}, ACL_FLOAT, dbih.p), tbhh = mk({4 * H}, ACL_FLOAT, dbhh.p), th0 = mk({B, H}, ACL_FLOAT, dh0.p), tc0 = mk({B, H}, ACL_FLOAT, dc0.p);
    auto ty = mk({T, B, H}, ACL_FLOAT, dy.p), thN = mk({B, H}, ACL_FLOAT, dhN.p), tcN = mk({B, H}, ACL_FLOAT, dcN.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return ws(tx, twih, twhh, tbih, tbhh, th0, tc0, ty, thN, tcN, w, e); }, run);
    dy.down(hy.data()); dhN.down(hhN.data()); dcN.down(hcN.data());
    // CPU reference (gate order i,f,g,o; rows g*H+hh)
    std::vector<double> h(B * H), c(B * H); for (int i = 0; i < B * H; i++) { h[i] = h0[i]; c[i] = c0[i]; }
    std::vector<double> refY(T * B * H);
    for (int t = 0; t < T; t++) {
        std::vector<double> hn(B * H), cn(B * H);
        for (int b = 0; b < B; b++) for (int hh = 0; hh < H; hh++) {
            double g[4];
            for (int gate = 0; gate < 4; gate++) {
                int row = gate * H + hh; double acc = bih[row] + bhh[row];
                for (int i = 0; i < I; i++) acc += (double)wih[row * I + i] * x[(t * B + b) * I + i];
                for (int kk = 0; kk < H; kk++) acc += (double)whh[row * H + kk] * h[b * H + kk];
                g[gate] = acc;
            }
            double gi = sigd(g[0]), gf = sigd(g[1]), gg = std::tanh(g[2]), go = sigd(g[3]);
            double cc = gf * c[b * H + hh] + gi * gg; cn[b * H + hh] = cc;
            double hc = go * std::tanh(cc); hn[b * H + hh] = hc; refY[(t * B + b) * H + hh] = hc;
        }
        h = hn; c = cn;
    }
    double me = 0, mr = 0; for (int i = 0; i < T * B * H; i++) { me = std::max(me, std::fabs(hy[i] - refY[i])); mr = std::max(mr, std::fabs(refY[i])); }
    double ce = 0, cr = 0; for (int i = 0; i < B * H; i++) { ce = std::max(ce, std::fabs(hcN[i] - c[i])); cr = std::max(cr, std::fabs(c[i])); ce = std::max(ce, std::fabs(hhN[i] - h[i])); cr = std::max(cr, std::fabs(h[i])); }
    report((std::string(name) + " y").c_str(), me / (mr + 1e-9), 1e-4);
    report((std::string(name) + " hN/cN").c_str(), ce / (cr + 1e-9), 1e-4);
    aclDestroyTensor(tx); aclDestroyTensor(twih); aclDestroyTensor(twhh); aclDestroyTensor(tbih); aclDestroyTensor(tbhh);
    aclDestroyTensor(th0); aclDestroyTensor(tc0); aclDestroyTensor(ty); aclDestroyTensor(thN); aclDestroyTensor(tcN);
}
static void t_lstm() {
    run_lstm_op("LSTM", aclnnLSTMGetWorkspaceSize, aclnnLSTM);
    run_lstm_op("BidirectionLSTM(fwd)", aclnnBidirectionLSTMGetWorkspaceSize, aclnnBidirectionLSTM);
    run_lstm_op("BidirectionLSTMV2(fwd)", aclnnBidirectionLSTMV2GetWorkspaceSize, aclnnBidirectionLSTMV2);
}

// ---------------- RecurrentGatedDeltaRule: per (b,head) state recurrence ----------------
static void t_gated_delta() {
    const int B = 2, Hd = 2, L = 5, Dk = 4, Dv = 3;
    auto q = randv(B * Hd * L * Dk, -1, 1), k = randv(B * Hd * L * Dk, -1, 1), v = randv(B * Hd * L * Dv, -1, 1);
    auto beta = randv(B * Hd * L, 0.1f, 0.9f), g = randv(B * Hd * L, 0.5f, 0.99f);
    std::vector<float> hy(B * Hd * L * Dv);
    DevBuf dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), db(beta.size() * 4), dg(g.size() * 4), dy(hy.size() * 4);
    dq.up(q.data()); dk.up(k.data()); dv.up(v.data()); db.up(beta.data()); dg.up(g.data());
    auto tq = mk({B, Hd, L, Dk}, ACL_FLOAT, dq.p), tk = mk({B, Hd, L, Dk}, ACL_FLOAT, dk.p), tv = mk({B, Hd, L, Dv}, ACL_FLOAT, dv.p),
         tb = mk({B, Hd, L}, ACL_FLOAT, db.p), tg = mk({B, Hd, L}, ACL_FLOAT, dg.p), ty = mk({B, Hd, L, Dv}, ACL_FLOAT, dy.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(tq, tk, tv, tb, tg, ty, w, e); }, aclnnRecurrentGatedDeltaRule);
    dy.down(hy.data());
    std::vector<double> ref(hy.size());
    for (int bh = 0; bh < B * Hd; bh++) {
        std::vector<double> S(Dk * Dv, 0);
        for (int t = 0; t < L; t++) {
            long base = (long)bh * L + t; const float *kt = &k[base * Dk], *qt = &q[base * Dk], *vt = &v[base * Dv];
            double gt = g[base], bt = beta[base];
            for (int i = 0; i < Dk * Dv; i++) S[i] *= gt;
            for (int j = 0; j < Dv; j++) { double sk = 0; for (int i = 0; i < Dk; i++) sk += S[i * Dv + j] * kt[i]; double kv = vt[j] - sk; for (int i = 0; i < Dk; i++) S[i * Dv + j] += bt * kt[i] * kv; }
            for (int j = 0; j < Dv; j++) { double yt = 0; for (int i = 0; i < Dk; i++) yt += qt[i] * S[i * Dv + j]; ref[base * Dv + j] = yt; }
        }
    }
    double me = 0, mr = 0; for (size_t i = 0; i < hy.size(); i++) { me = std::max(me, std::fabs(hy[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    report("RecurrentGatedDeltaRule", me / (mr + 1e-9), 1e-4);
    aclDestroyTensor(tq); aclDestroyTensor(tk); aclDestroyTensor(tv); aclDestroyTensor(tb); aclDestroyTensor(tg); aclDestroyTensor(ty);
}

// ---------------- TransformBiasRescaleQkv: add bias, scale Q segment by 1/sqrt(headDim) ----------------
static void t_qkv_rescale() {
    const int seg = 7, D = 8, headDim = 4; const double qscale = 1.0 / std::sqrt((double)headDim);
    auto qkv = randv(3 * seg * D, -1, 1), bias = randv(3 * D, -0.5f, 0.5f);
    std::vector<float> ho(3 * seg * D);
    DevBuf dx(3 * seg * D * 4), db(3 * D * 4), dout(3 * seg * D * 4); dx.up(qkv.data()); db.up(bias.data());
    auto tx = mk({3, seg, D}, ACL_FLOAT, dx.p), tb = mk({3, D}, ACL_FLOAT, db.p), to = mk({3, seg, D}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnTransformBiasRescaleQkvGetWorkspaceSize(tx, tb, headDim, to, w, e); }, aclnnTransformBiasRescaleQkv);
    dout.down(ho.data());
    std::vector<double> ref(3 * seg * D);
    for (int i = 0; i < 3 * seg * D; i++) { int which = i / (seg * D), d = i % D; double v = qkv[i] + bias[which * D + d]; ref[i] = (which == 0) ? v * qscale : v; }
    double me = 0, mr = 0; for (int i = 0; i < 3 * seg * D; i++) { me = std::max(me, std::fabs(ho[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    report("TransformBiasRescaleQkv", me / (mr + 1e-9), 1e-5);
    aclDestroyTensor(tx); aclDestroyTensor(tb); aclDestroyTensor(to);
}

int main() {
    init(); srand(71);
    t_mhc_copy();       // MhcPre, MhcPost
    t_sinkhorn();       // MhcSinkhorn
    t_mla();            // MlaPreprocess, MlaPreprocessV2, MlaProlog, MlaPrologV2WeightNz, MlaPrologV3WeightNz
    t_gating();         // MoeFusedTopk, MoeGatingTopK, MoeGatingTopKSoftmaxV2
    t_init_final();     // MoeInitRoutingV2/V3, MoeFinalizeRoutingV2/V3
    t_permute();        // MoeTokenPermuteWithRoutingMap, MoeTokenUnpermuteWithEp, MoeTokenUnpermuteWithRoutingMap
    t_lstm();           // LSTM, BidirectionLSTM, BidirectionLSTMV2
    t_gated_delta();    // RecurrentGatedDeltaRule
    t_qkv_rescale();    // TransformBiasRescaleQkv
    return finish();
}
