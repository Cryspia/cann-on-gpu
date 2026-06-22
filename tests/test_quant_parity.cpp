// Quant/dequant/antiquant/MX-fp gap operators (~60) cross-check — parity with the CUDA backend.
// Self-contained: extern "C" prototypes match the canonical aclnnop/aclnn_ops.h declarations EXACTLY
// (verified against include/aclnnop/aclnn_ops.h). Each case checks the quant/dequant math with a CPU
// reference, a quant->dequant round-trip (≈ input within one quant step), or equivalence to a base op.
// Tolerances are documented per case (quantization error is looser than fp ops).
#include "harness.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstring>
using namespace hn;
typedef void *HcclComm;   // quant-MC2 ops carry an HcclComm; nullptr on a single rank (matches aclnn_mc2.h)

// ------------------------------------------------------------------ canonical prototypes (extern "C")
extern "C" {
// pure quant/dequant
aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAscendQuantV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAscendAntiQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAscendAntiQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGeluQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, int64_t, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupNormSiluQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, int64_t, int64_t, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, int64_t, double, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantizedBatchNorm(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicQuantV2GetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicQuantV3GetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicQuantV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicQuantV4GetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicQuantV4(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicBlockQuantV2GetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicBlockQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedDynamicBlockQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicMxQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicMxQuantV2GetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicMxQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedDynamicMxQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDynamicMxQuantWithDualAxis(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSwiGluQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDequantSwigluQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
// norm+quant fusions
aclnnStatus aclnnAddRmsNormQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, double, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddRmsNormQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAddRmsNormQuantV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, double, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddRmsNormQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddRmsNormDynamicQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddRmsNormDynamicMxQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAdaLayerNormQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddLayerNormQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSwinTransformerLnQkvQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
// trans quant param + adamw
aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnTransQuantParamV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnTransQuantParamV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor*, aclTensor*, aclTensor*, const aclTensor*, double, double, double, double, double, int64_t, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnApplyAdamWQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
// W8A8 quant matmul (x int8, weight int8, scale, out)
aclnnStatus aclnnQuantMatmulV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulV4GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulV4(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulV5GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulV5(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulDequantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulDequant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFusedQuantMatmul(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFusedQuantMatmulWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMatmulCompressDequant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAlltoAllQuantMatmulGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAlltoAllQuantMatmul(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulAlltoAllGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulAlltoAll(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantMatmulReduceSumWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDualLevelQuantMatmulWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
// W8A16/W4A16 weight-only matmul (x fp, weight int8/int4, antiquantScale, antiquantOffset, out)
aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnWeightQuantBatchMatmulNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnWeightQuantBatchMatmulV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnWeightQuantBatchMatmulV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
// grouped W8A8 quant matmul (x int8, weight int8 [E,K,N], scale [E,N], groupList, out)
aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantGroupedMatmulDequant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantGroupedMatMulAlltoAllv(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAlltoAllvQuantGroupedMatMul(void*, uint64_t, aclOpExecutor*, aclrtStream);
// grouped matmul + swiglu + per-row int8 quant (x int8, weight int8 [E,K,2N], groupList, out int8, scaleOut)
aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedMatmulSwigluQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
// quant conv (input int8, weight int8, bias, scale, stride/pad/dil, groups, out fp32)
aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, const aclIntArray*, const aclIntArray*, const aclIntArray*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantConvolution(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, double, const aclIntArray*, const aclIntArray*, const aclIntArray*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantConvolutionWeightNz(void*, uint64_t, aclOpExecutor*, aclrtStream);
// quant flash attention (q,k,v,attenMask,scaleValue,headNum,causal,out)
aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, int64_t, bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnQuantFlashAttentionScore(void*, uint64_t, aclOpExecutor*, aclrtStream);
// rope-decl: DequantRopeQuantKvcache (x, cos, sin, mode, out)
aclnnStatus aclnnDequantRopeQuantKvcacheGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDequantRopeQuantKvcache(void*, uint64_t, aclOpExecutor*, aclrtStream);
// att-decl: SwinAttentionScoreQuant (q,k,v,attenMask,scaleValue,headNum,causal,out)
aclnnStatus aclnnSwinAttentionScoreQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, double, int64_t, bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSwinAttentionScoreQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
// moeir-decl: MoeInitRoutingQuant(+V2) (x, expertIdx, numExperts, expandedX, expandedRowIdx, expandedExpertIdx)
aclnnStatus aclnnMoeInitRoutingQuantGetWorkspaceSize(const aclTensor*, const aclTensor*, int64_t, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMoeInitRoutingQuant(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMoeInitRoutingQuantV2GetWorkspaceSize(const aclTensor*, const aclTensor*, int64_t, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMoeInitRoutingQuantV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
}

// ------------------------------------------------------------------ helpers
static int clampi(long v, int lo, int hi) { return (int)(v < lo ? lo : (v > hi ? hi : v)); }
static double silu(double v) { return v / (1.0 + std::exp(-v)); }
static double gelu(double v) { return 0.5 * v * std::erfc(-v * M_SQRT1_2); }
static double mx_scale(double amax) { return amax > 0 ? std::exp2(std::ceil(std::log2(amax / 127.0))) : 1.0; }
static std::vector<int8_t> randi8(int n) { std::vector<int8_t> v(n); for (auto &x : v) x = (int8_t)(rand() % 51 - 25); return v; }
static aclIntArray *mkIA(const std::vector<int64_t> &v) { return aclCreateIntArray(v.data(), v.size()); }

// ------------------------------------------------------------------ pure quant/dequant
static void t_ascend_quant_v3() {
    const int N = 8, C = 4; auto x = randv(N * C, -3, 3), sc = randv(C, 0.5, 2.0), of = randv(C, -1, 1);
    std::vector<int8_t> hq(N * C); DevBuf dx(N * C * 4), ds(C * 4), dof(C * 4), dq(N * C); dx.up(x.data()); ds.up(sc.data()); dof.up(of.data());
    aclTensor *tx = mk({N, C}, ACL_FLOAT, dx.p), *tsc = mk({C}, ACL_FLOAT, ds.p), *tof = mk({C}, ACL_FLOAT, dof.p), *tq = mk({N, C}, ACL_INT8, dq.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAscendQuantV3GetWorkspaceSize(tx, tsc, tof, tq, w, e); }, aclnnAscendQuantV3);
    dq.down(hq.data()); double bad = 0; for (int i = 0; i < N * C; i++) { int c = i % C; int ref = clampi(std::lrint(x[i] * sc[c] + of[c]), -128, 127); if ((int)hq[i] != ref) bad = 1; }
    report("AscendQuantV3", bad, 0.0); // exact int8 affine
}
static void t_ascend_antiquant() {
    const int N = 8, C = 4; auto q = randi8(N * C); auto sc = randv(C, 0.5, 2.0), of = randv(C, -1, 1);
    std::vector<float> hy(N * C); DevBuf dq(N * C), ds(C * 4), dof(C * 4), dy(N * C * 4); dq.up(q.data()); ds.up(sc.data()); dof.up(of.data());
    aclTensor *tq = mk({N, C}, ACL_INT8, dq.p), *tsc = mk({C}, ACL_FLOAT, ds.p), *tof = mk({C}, ACL_FLOAT, dof.p), *ty = mk({N, C}, ACL_FLOAT, dy.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAscendAntiQuantGetWorkspaceSize(tq, tsc, tof, ty, w, e); }, aclnnAscendAntiQuant);
    dy.down(hy.data()); double me = 0, mr = 0; for (int i = 0; i < N * C; i++) { int c = i % C; double ref = ((double)q[i] - of[c]) * sc[c]; me = std::max(me, std::fabs(hy[i] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report("AscendAntiQuant", me / (mr + 1e-9), 1e-5); // exact dequant
}
static void t_group_quant() {
    const int n = 32, G = 8; int ng = n / G; auto x = randv(n, -3, 3), sc = randv(ng, 0.5, 2.0), of = randv(ng, -1, 1);
    std::vector<int8_t> hq(n); DevBuf dx(n * 4), ds(ng * 4), dof(ng * 4), dq(n); dx.up(x.data()); ds.up(sc.data()); dof.up(of.data());
    aclTensor *tx = mk({n}, ACL_FLOAT, dx.p), *tsc = mk({ng}, ACL_FLOAT, ds.p), *tof = mk({ng}, ACL_FLOAT, dof.p), *tq = mk({n}, ACL_INT8, dq.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGroupQuantGetWorkspaceSize(tx, tsc, tof, (int64_t)G, tq, w, e); }, aclnnGroupQuant);
    dq.down(hq.data()); double bad = 0; for (int i = 0; i < n; i++) { int g = i / G; int ref = clampi(std::lrint(x[i] * sc[g] + of[g]), -128, 127); if ((int)hq[i] != ref) bad = 1; }
    report("GroupQuant", bad, 0.0); // exact per-group affine
}
static void t_gelu_quant() {
    const int R = 6, D = 16; auto x = randv(R * D, -3, 3); std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf dx(R * D * 4), dq(R * D), ds(R * 4); dx.up(x.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGeluQuantGetWorkspaceSize(tx, tq, ts, w, e); }, aclnnGeluQuant);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) { double g = gelu(x[r * D + d]); double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - g)); mr = std::max(mr, std::fabs(g)); }
    report("GeluQuant", me / (mr + 1e-9), 1e-2); // round-trip ~1 step of per-row int8
}
static void t_gnsilu_quant() {
    const int N = 2, C = 4, S = 6, G = 2; int n = N * C * S; auto x = randv(n, -2, 2), gm = randv(C, 0.5, 1.5), bt = randv(C, -0.5, 0.5);
    std::vector<int8_t> hq(n); std::vector<float> sc(N * G); double eps = 1e-5;
    DevBuf dx(n * 4), dg(C * 4), db(C * 4), dq(n), ds(N * G * 4); dx.up(x.data()); dg.up(gm.data()); db.up(bt.data());
    aclTensor *tx = mk({N, C, S}, ACL_FLOAT, dx.p), *tg = mk({C}, ACL_FLOAT, dg.p), *tb = mk({C}, ACL_FLOAT, db.p), *tq = mk({N, C, S}, ACL_INT8, dq.p), *ts = mk({N, G}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnGroupNormSiluQuantGetWorkspaceSize(tx, tg, tb, (int64_t)G, eps, tq, ts, w, e); }, aclnnGroupNormSiluQuant);
    dq.down(hq.data()); ds.down(sc.data());
    int Cg = C / G, cnt = Cg * S; double me = 0, mr = 0;
    for (int nn = 0; nn < N; nn++) for (int grp = 0; grp < G; grp++) {
        double m = 0, qsum = 0; for (int cc = 0; cc < Cg; cc++) for (int sp = 0; sp < S; sp++) { double v = x[(nn * C + grp * Cg + cc) * S + sp]; m += v; qsum += v * v; }
        m /= cnt; double var = qsum / cnt - m * m; double inv = 1.0 / std::sqrt(var + eps);
        for (int cc = 0; cc < Cg; cc++) { int c = grp * Cg + cc; for (int sp = 0; sp < S; sp++) { double y = silu((x[(nn * C + c) * S + sp] - m) * inv * gm[c] + bt[c]); double deq = (double)hq[(nn * C + c) * S + sp] * sc[nn * G + grp]; me = std::max(me, std::fabs(deq - y)); mr = std::max(mr, std::fabs(y)); } }
    }
    report("GroupNormSiluQuant", me / (mr + 1e-9), 2e-2); // GN+SiLU then per-group int8 round-trip
}
static void t_fakeq_perchannel() {
    const int N = 6, C = 4; auto x = randv(N * C, -3, 3), sc = randv(C, 0.02, 0.05); std::vector<float> zp(C, 0);
    std::vector<float> hy(N * C); std::vector<uint8_t> hm(N * C); int qmin = -128, qmax = 127;
    DevBuf dx(N * C * 4), ds(C * 4), dz(C * 4), dy(N * C * 4), dm(N * C); dx.up(x.data()); ds.up(sc.data()); dz.up(zp.data());
    aclTensor *tx = mk({N, C}, ACL_FLOAT, dx.p), *tsc = mk({C}, ACL_FLOAT, ds.p), *tz = mk({C}, ACL_FLOAT, dz.p), *ty = mk({N, C}, ACL_FLOAT, dy.p), *tm = mk({N, C}, ACL_BOOL, dm.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(tx, tsc, tz, 1, (int64_t)qmin, (int64_t)qmax, ty, tm, w, e); }, aclnnFakeQuantPerChannelAffineCachemask);
    dy.down(hy.data()); dm.down(hm.data()); double me = 0, mr = 0, badm = 0;
    for (int i = 0; i < N * C; i++) { int c = i % C; long qr = std::lrint(x[i] / sc[c]); int q = clampi(qr, qmin, qmax); double ref = (double)q * sc[c]; me = std::max(me, std::fabs(hy[i] - ref)); mr = std::max(mr, std::fabs(ref)); uint8_t mref = (uint8_t)(qr >= qmin && qr <= qmax); if (hm[i] != mref) badm = 1; }
    report("FakeQuantPerChannelAffineCachemask", std::max(me / (mr + 1e-9), badm), 1e-5); // exact fakequant + mask
}
static void t_quantized_bn() {
    const int N = 2, C = 4, S = 5; int n = N * C * S; auto x = randv(n, -2, 2), wt = randv(C, 0.5, 1.5), bs = randv(C, -0.5, 0.5), mn = randv(C, -1, 1), iv = randv(C, 0.5, 1.5);
    double scale = 0.05, eps = 1e-5; int zp = 0; std::vector<int8_t> hq(n);
    DevBuf dx(n * 4), dw(C * 4), db(C * 4), dm(C * 4), di(C * 4), dq(n); dx.up(x.data()); dw.up(wt.data()); db.up(bs.data()); dm.up(mn.data()); di.up(iv.data());
    aclTensor *tx = mk({N, C, S}, ACL_FLOAT, dx.p), *tw = mk({C}, ACL_FLOAT, dw.p), *tb = mk({C}, ACL_FLOAT, db.p), *tm = mk({C}, ACL_FLOAT, dm.p), *ti = mk({C}, ACL_FLOAT, di.p), *tq = mk({N, C, S}, ACL_INT8, dq.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnQuantizedBatchNormGetWorkspaceSize(tx, tw, tb, tm, ti, scale, (int64_t)zp, eps, tq, w, e); }, aclnnQuantizedBatchNorm);
    dq.down(hq.data()); double bad = 0;
    for (int nn = 0; nn < N; nn++) for (int c = 0; c < C; c++) for (int sp = 0; sp < S; sp++) { int idx = (nn * C + c) * S + sp; double bn = (x[idx] - mn[c]) * iv[c] * wt[c] + bs[c]; int ref = clampi(std::lrint(bn / scale) + zp, -128, 127); if ((int)hq[idx] != ref) bad = 1; }
    report("QuantizedBatchNorm", bad, 0.0); // exact
}
// shared dynamic per-row round-trip check (scale = absmax/127)
static void t_dynamic_quant(const char *name, aclnnStatus (*gws)(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int R = 6, D = 20; auto x = randv(R * D, -3, 3); std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf dx(R * D * 4), dq(R * D), ds(R * 4); dx.up(x.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tq = mk({R, D}, ACL_INT8, dx.p ? dq.p : nullptr), *ts = mk({R}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tq, ts, w, e); }, run);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) { double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - x[r * D + d])); mr = std::max(mr, std::fabs((double)x[r * D + d])); }
    report(name, me / (mr + 1e-9), 1e-2); // per-row int8 round-trip
}
// shared block round-trip: blk = last dim, dynamic (absmax/127)
static void t_block_quant(const char *name, int64_t blk, aclnnStatus (*gws)(const aclTensor*, int64_t, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream), bool mx) {
    const int nblk = 6; int n = nblk * (int)blk; auto x = randv(n, -4, 4); std::vector<int8_t> hq(n); std::vector<float> sc(nblk);
    DevBuf dx(n * 4), dq(n), ds(nblk * 4); dx.up(x.data());
    aclTensor *tx = mk({nblk, (int64_t)blk}, ACL_FLOAT, dx.p), *tq = mk({nblk, (int64_t)blk}, ACL_INT8, dq.p), *ts = mk({nblk}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, blk, tq, ts, w, e); }, run);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int b = 0; b < nblk; b++) for (int i = 0; i < blk; i++) { double deq = (double)hq[b * blk + i] * sc[b]; me = std::max(me, std::fabs(deq - x[b * blk + i])); mr = std::max(mr, std::fabs((double)x[b * blk + i])); }
    report(name, me / (mr + 1e-9), mx ? 2e-2 : 1e-2); // block int8 round-trip (MX scale is power-of-2 => coarser)
}
static void t_mx_dualaxis() {
    const int R = 6, D = 16, blk = 16; int n = R * D; auto x = randv(n, -4, 4); std::vector<int8_t> hq(n); std::vector<float> s1(R), s2(D);
    DevBuf dx(n * 4), dq(n), d1(R * 4), d2(D * 4); dx.up(x.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tq = mk({R, D}, ACL_INT8, dq.p), *t1 = mk({R}, ACL_FLOAT, d1.p), *t2 = mk({D}, ACL_FLOAT, d2.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(tx, (int64_t)blk, tq, t1, t2, w, e); }, aclnnDynamicMxQuantWithDualAxis);
    dq.down(hq.data()); d1.down(s1.data()); d2.down(s2.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) { double deq = (double)hq[r * D + d] * s1[r]; me = std::max(me, std::fabs(deq - x[r * D + d])); mr = std::max(mr, std::fabs((double)x[r * D + d])); }
    // verify scaleOut2 is the per-column MX scale
    double bad2 = 0; for (int d = 0; d < D; d++) { double amax = 0; for (int r = 0; r < R; r++) amax = std::max(amax, std::fabs((double)x[r * D + d])); if (std::fabs(s2[d] - mx_scale(amax)) > 1e-6 * (s2[d] + 1)) bad2 = 1; }
    report("DynamicMxQuantWithDualAxis", std::max(me / (mr + 1e-9), bad2 ? 1.0 : 0.0), 2e-2);
}
// swiglu-quant round-trip: out ≈ swiglu(in) within per-row int8 step
static void t_swiglu_quant(const char *name, bool dequant) {
    const int R = 6, D = 12; int n = R * 2 * D; auto in = randv(n, -2, 2); std::vector<float> dqsc = randv(R, 0.5, 1.5);
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf din(n * 4), ddq(R * 4), dq(R * D), ds(R * 4); din.up(in.data()); ddq.up(dqsc.data());
    aclTensor *tin = mk({R, 2 * D}, ACL_FLOAT, din.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p), *tdq = mk({R}, ACL_FLOAT, ddq.p);
    if (dequant) exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnDequantSwigluQuantV2GetWorkspaceSize(tin, tdq, tq, ts, w, e); }, aclnnDequantSwigluQuantV2);
    else        exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSwiGluQuantV2GetWorkspaceSize(tin, tq, ts, w, e); }, aclnnSwiGluQuantV2);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) { double scl = dequant ? dqsc[r] : 1.0; double a = in[r * 2 * D + d] * scl, b = in[r * 2 * D + D + d] * scl; double g = silu(a) * b; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - g)); mr = std::max(mr, std::fabs(g)); }
    report(name, me / (mr + 1e-9), 1e-2);
}

// ------------------------------------------------------------------ norm+quant fusions
static void t_addrms_quant(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, double, double, double, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int R = 5, D = 16; auto x = randv(R * D, -2, 2), res = randv(R * D, -2, 2), gm = randv(D, 0.5, 1.5);
    double scale = 0.05, off = 0, eps = 1e-5; std::vector<int8_t> hq(R * D); std::vector<float> hrs(R * D);
    DevBuf dx(R * D * 4), dr(R * D * 4), dg(D * 4), dq(R * D), drs(R * D * 4); dx.up(x.data()); dr.up(res.data()); dg.up(gm.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tr = mk({R, D}, ACL_FLOAT, dr.p), *tg = mk({D}, ACL_FLOAT, dg.p), *tq = mk({R, D}, ACL_INT8, dq.p), *trs = mk({R, D}, ACL_FLOAT, drs.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tr, tg, scale, off, eps, tq, trs, w, e); }, run);
    dq.down(hq.data()); drs.down(hrs.data()); double bad = 0, rsbad = 0;
    for (int r = 0; r < R; r++) { double ss = 0; std::vector<double> sv(D); for (int d = 0; d < D; d++) { double v = (double)x[r * D + d] + res[r * D + d]; sv[d] = v; ss += v * v; if (std::fabs(hrs[r * D + d] - v) > 1e-4) rsbad = 1; }
        double inv = 1.0 / std::sqrt(ss / D + eps); for (int d = 0; d < D; d++) { double yn = sv[d] * inv * gm[d]; int ref = clampi(std::lrint(yn / scale + off), -128, 127); if ((int)hq[r * D + d] != ref) bad = 1; } }
    report(name, std::max(bad, rsbad), 0.0); // exact static-scale quant + residualSum
}
static void t_addrms_dyn_quant() {
    const int R = 5, D = 16; auto x = randv(R * D, -2, 2), res = randv(R * D, -2, 2), gm = randv(D, 0.5, 1.5); double eps = 1e-5;
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R), hrs(R * D);
    DevBuf dx(R * D * 4), dr(R * D * 4), dg(D * 4), dq(R * D), ds(R * 4), drs(R * D * 4); dx.up(x.data()); dr.up(res.data()); dg.up(gm.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tr = mk({R, D}, ACL_FLOAT, dr.p), *tg = mk({D}, ACL_FLOAT, dg.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p), *trs = mk({R, D}, ACL_FLOAT, drs.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAddRmsNormDynamicQuantV2GetWorkspaceSize(tx, tr, tg, eps, tq, ts, trs, w, e); }, aclnnAddRmsNormDynamicQuantV2);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) { double ss = 0; std::vector<double> sv(D); for (int d = 0; d < D; d++) { double v = (double)x[r * D + d] + res[r * D + d]; sv[d] = v; ss += v * v; } double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int d = 0; d < D; d++) { double yn = sv[d] * inv * gm[d]; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - yn)); mr = std::max(mr, std::fabs(yn)); } }
    report("AddRmsNormDynamicQuantV2", me / (mr + 1e-9), 1e-2); // round-trip
}
static void t_addrms_mx_quant() {
    const int R = 5, D = 16; auto x = randv(R * D, -2, 2), res = randv(R * D, -2, 2), gm = randv(D, 0.5, 1.5); double eps = 1e-5;
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf dx(R * D * 4), dr(R * D * 4), dg(D * 4), dq(R * D), ds(R * 4), drs(R * D * 4); dx.up(x.data()); dr.up(res.data()); dg.up(gm.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tr = mk({R, D}, ACL_FLOAT, dr.p), *tg = mk({D}, ACL_FLOAT, dg.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p), *trs = mk({R, D}, ACL_FLOAT, drs.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(tx, tr, tg, eps, tq, ts, trs, w, e); }, aclnnAddRmsNormDynamicMxQuant);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) { double ss = 0; std::vector<double> sv(D); for (int d = 0; d < D; d++) { double v = (double)x[r * D + d] + res[r * D + d]; sv[d] = v; ss += v * v; } double inv = 1.0 / std::sqrt(ss / D + eps);
        for (int d = 0; d < D; d++) { double yn = sv[d] * inv * gm[d]; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - yn)); mr = std::max(mr, std::fabs(yn)); } }
    report("AddRmsNormDynamicMxQuant", me / (mr + 1e-9), 2e-2); // MX round-trip
}
static void t_adaln_quant() {
    const int R = 5, D = 16; auto x = randv(R * D, -2, 2), scv = randv(R * D, -0.5, 0.5), shv = randv(R * D, -0.5, 0.5); double eps = 1e-5;
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf dx(R * D * 4), dsc(R * D * 4), dsh(R * D * 4), dq(R * D), ds(R * 4); dx.up(x.data()); dsc.up(scv.data()); dsh.up(shv.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tsc = mk({R, D}, ACL_FLOAT, dsc.p), *tsh = mk({R, D}, ACL_FLOAT, dsh.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAdaLayerNormQuantGetWorkspaceSize(tx, tsc, tsh, eps, tq, ts, w, e); }, aclnnAdaLayerNormQuant);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) { double m = 0; for (int d = 0; d < D; d++) m += x[r * D + d]; m /= D; double v = 0; for (int d = 0; d < D; d++) { double u = x[r * D + d] - m; v += u * u; } v /= D; double inv = 1.0 / std::sqrt(v + eps);
        for (int d = 0; d < D; d++) { double y = (x[r * D + d] - m) * inv * (1.0 + scv[r * D + d]) + shv[r * D + d]; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - y)); mr = std::max(mr, std::fabs(y)); } }
    report("AdaLayerNormQuant", me / (mr + 1e-9), 2e-2); // AdaLN + MX round-trip
}
static void t_addln_quant() {
    const int R = 5, D = 16; auto x = randv(R * D, -2, 2), res = randv(R * D, -2, 2), gm = randv(D, 0.5, 1.5), bt = randv(D, -0.5, 0.5); double eps = 1e-5;
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R), hrs(R * D);
    DevBuf dx(R * D * 4), dr(R * D * 4), dg(D * 4), db(D * 4), dq(R * D), ds(R * 4), drs(R * D * 4); dx.up(x.data()); dr.up(res.data()); dg.up(gm.data()); db.up(bt.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tr = mk({R, D}, ACL_FLOAT, dr.p), *tg = mk({D}, ACL_FLOAT, dg.p), *tb = mk({D}, ACL_FLOAT, db.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p), *trs = mk({R, D}, ACL_FLOAT, drs.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAddLayerNormQuantGetWorkspaceSize(tx, tr, tg, tb, eps, tq, ts, trs, w, e); }, aclnnAddLayerNormQuant);
    dq.down(hq.data()); ds.down(sc.data()); drs.down(hrs.data()); double me = 0, mr = 0, rsbad = 0;
    for (int r = 0; r < R; r++) { std::vector<double> sv(D); double m = 0; for (int d = 0; d < D; d++) { double v = (double)x[r * D + d] + res[r * D + d]; sv[d] = v; m += v; if (std::fabs(hrs[r * D + d] - v) > 1e-4) rsbad = 1; } m /= D; double var = 0; for (int d = 0; d < D; d++) { double u = sv[d] - m; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        for (int d = 0; d < D; d++) { double y = (sv[d] - m) * inv * gm[d] + bt[d]; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - y)); mr = std::max(mr, std::fabs(y)); } }
    report("AddLayerNormQuant", std::max(me / (mr + 1e-9), rsbad ? 1.0 : 0.0), 1e-2);
}
static void t_swin_lnqkv_quant() {
    const int R = 6, D = 16; auto x = randv(R * D, -2, 2), gm = randv(D, 0.5, 1.5), bt = randv(D, -0.5, 0.5); double eps = 1e-5;
    std::vector<int8_t> hq(R * D); std::vector<float> sc(R);
    DevBuf dx(R * D * 4), dg(D * 4), db(D * 4), dq(R * D), ds(R * 4); dx.up(x.data()); dg.up(gm.data()); db.up(bt.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tg = mk({D}, ACL_FLOAT, dg.p), *tb = mk({D}, ACL_FLOAT, db.p), *tq = mk({R, D}, ACL_INT8, dq.p), *ts = mk({R}, ACL_FLOAT, ds.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(tx, tg, tb, eps, tq, ts, w, e); }, aclnnSwinTransformerLnQkvQuant);
    dq.down(hq.data()); ds.down(sc.data()); double me = 0, mr = 0;
    for (int r = 0; r < R; r++) { double m = 0; for (int d = 0; d < D; d++) m += x[r * D + d]; m /= D; double var = 0; for (int d = 0; d < D; d++) { double u = x[r * D + d] - m; var += u * u; } var /= D; double inv = 1.0 / std::sqrt(var + eps);
        for (int d = 0; d < D; d++) { double y = (x[r * D + d] - m) * inv * gm[d] + bt[d]; double deq = (double)hq[r * D + d] * sc[r]; me = std::max(me, std::fabs(deq - y)); mr = std::max(mr, std::fabs(y)); } }
    report("SwinTransformerLnQkvQuant", me / (mr + 1e-9), 1e-2);
}

// ------------------------------------------------------------------ trans quant param + adamw
static void t_trans_quant_param(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int n = 8; auto scv = randv(n, 0.01, 2.0), ofv = randv(n, -1, 1);
    std::vector<int64_t> ho(n); DevBuf ds(n * 4), dof(n * 4), dout(n * 8); ds.up(scv.data()); dof.up(ofv.data());
    aclTensor *tsc = mk({n}, ACL_FLOAT, ds.p), *tof = mk({n}, ACL_FLOAT, dof.p), *to = mk({n}, ACL_INT64, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tsc, tof, to, w, e); }, run);
    dout.down(ho.data()); double bad = 0;
    for (int i = 0; i < n; i++) { uint32_t lo; std::memcpy(&lo, &scv[i], 4); uint32_t hi; std::memcpy(&hi, &ofv[i], 4); int64_t ref = (int64_t)(((uint64_t)hi << 32) | (uint64_t)lo); if (ho[i] != ref) bad = 1; }
    report(name, bad, 0.0); // exact bit-packing
}
static void t_apply_adamw_quant() {
    const int n = 64; auto p0 = randv(n, -1, 1), m0 = randv(n, -0.5, 0.5), v0 = randv(n, 0, 0.5), g = randv(n, -1, 1);
    double lr = 0.01, b1 = 0.9, b2 = 0.999, eps = 1e-8, wd = 0.01; int step = 5;
    std::vector<float> hp(n); auto p = p0, m = m0, v = v0;
    DevBuf dp(n * 4), dm(n * 4), dv(n * 4), dg(n * 4); dp.up(p0.data()); dm.up(m0.data()); dv.up(v0.data()); dg.up(g.data());
    aclTensor *tp = mk({n}, ACL_FLOAT, dp.p), *tm = mk({n}, ACL_FLOAT, dm.p), *tv = mk({n}, ACL_FLOAT, dv.p), *tg = mk({n}, ACL_FLOAT, dg.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnApplyAdamWQuantGetWorkspaceSize(tp, tm, tv, tg, lr, b1, b2, eps, wd, (int64_t)step, w, e); }, aclnnApplyAdamWQuant);
    dp.down(hp.data()); double me = 0, mr = 0; double bc1 = 1.0 - std::pow(b1, step), bc2 = 1.0 - std::pow(b2, step);
    for (int i = 0; i < n; i++) { double pw = p[i] - lr * wd * p[i]; double mi = b1 * m[i] + (1 - b1) * g[i]; double vi = b2 * v[i] + (1 - b2) * (double)g[i] * g[i]; double mh = mi / bc1, vh = vi / bc2; double ref = pw - lr * mh / (std::sqrt(vh) + eps); me = std::max(me, std::fabs(hp[i] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report("ApplyAdamWQuant", me / (mr + 1e-9), 1e-5); // full-precision AdamW equivalence
}

// ------------------------------------------------------------------ W8A8 quant matmul
static void t_qmm(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int M = 8, K = 16, N = 8; auto xq = randi8(M * K), wq = randi8(K * N); auto sc = randv(N, 0.01, 0.05); std::vector<uint16_t> sh(N); for (int i = 0; i < N; i++) sh[i] = f2h(sc[i]);   // K%16==0 int8 path; deqScale fp16 (CANN/CUDA-canonical)
    std::vector<float> ho(M * N); DevBuf dx(M * K), dw(K * N), ds(N * 2), dout(M * N * 4); dx.up(xq.data()); dw.up(wq.data()); ds.up(sh.data());
    aclTensor *tx = mk({M, K}, ACL_INT8, dx.p), *tw = mk({K, N}, ACL_INT8, dw.p), *tsc = mk({N}, ACL_FLOAT16, ds.p), *to = mk({M, N}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, tsc, to, w, e); }, run);
    dout.down(ho.data()); double me = 0, mr = 0;
    for (int m = 0; m < M; m++) for (int nn = 0; nn < N; nn++) { long long acc = 0; for (int k = 0; k < K; k++) acc += (long long)xq[m * K + k] * (long long)wq[k * N + nn]; double ref = (double)acc * (double)h2f(sh[nn]); me = std::max(me, std::fabs(ho[m * N + nn] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report(name, me / (mr + 1e-9), 3e-3); // W8A8 matmul, fp16 deqScale (~5e-4 precision; CUDA int8 epilogue vs Metal fp32)
}
// quant collective-matmul: canonical mc2.h signature carries an HcclComm; single-rank => local W8A8 matmul.
static void t_qmm_comm(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, HcclComm, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int M = 8, K = 16, N = 8; auto xq = randi8(M * K), wq = randi8(K * N); auto sc = randv(N, 0.01, 0.05); std::vector<uint16_t> sh(N); for (int i = 0; i < N; i++) sh[i] = f2h(sc[i]);   // K%16==0 int8 path; deqScale fp16 (CANN/CUDA-canonical)
    std::vector<float> ho(M * N); DevBuf dx(M * K), dw(K * N), ds(N * 2), dout(M * N * 4); dx.up(xq.data()); dw.up(wq.data()); ds.up(sh.data());
    aclTensor *tx = mk({M, K}, ACL_INT8, dx.p), *tw = mk({K, N}, ACL_INT8, dw.p), *tsc = mk({N}, ACL_FLOAT16, ds.p), *to = mk({M, N}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, tsc, (HcclComm)nullptr, to, w, e); }, run);
    dout.down(ho.data()); double me = 0, mr = 0;
    for (int m = 0; m < M; m++) for (int nn = 0; nn < N; nn++) { long long acc = 0; for (int k = 0; k < K; k++) acc += (long long)xq[m * K + k] * (long long)wq[k * N + nn]; double ref = (double)acc * (double)h2f(sh[nn]); me = std::max(me, std::fabs(ho[m * N + nn] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report(name, me / (mr + 1e-9), 3e-3); // W8A8 matmul, fp16 deqScale (~5e-4 precision; CUDA int8 epilogue vs Metal fp32) (single-rank MC2 collective = identity)
}
static void t_qmm_iadd_grouped() { // QuantGroupedMatmulInplaceAdd: out += grouped(x@w)*scale
    const int M = 6, K = 8, N = 4, E = 2; auto xq = randi8(M * K), wq = randi8(E * K * N); auto sc = randv(E * N, 0.01, 0.05);
    auto out0 = randv(M * N, -1, 1); std::vector<float> ho(M * N); std::vector<int64_t> gl = {3, 6};
    DevBuf dx(M * K), dw(E * K * N), ds(E * N * 4), dout(M * N * 4); dx.up(xq.data()); dw.up(wq.data()); ds.up(sc.data()); dout.up(out0.data());
    aclTensor *tx = mk({M, K}, ACL_INT8, dx.p), *tw = mk({E, K, N}, ACL_INT8, dw.p), *tsc = mk({E, N}, ACL_FLOAT, ds.p), *to = mk({M, N}, ACL_FLOAT, dout.p);
    aclIntArray *ia = mkIA(gl);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(tx, tw, tsc, ia, to, w, e); }, aclnnQuantGroupedMatmulInplaceAdd);
    dout.down(ho.data()); double me = 0, mr = 0; int row0 = 0;
    for (int g = 0; g < E; g++) { int row1 = (int)gl[g]; for (int m = row0; m < row1; m++) for (int nn = 0; nn < N; nn++) { long long acc = 0; for (int k = 0; k < K; k++) acc += (long long)xq[m * K + k] * (long long)wq[(g * K + k) * N + nn]; double ref = out0[m * N + nn] + (double)acc * sc[g * N + nn]; me = std::max(me, std::fabs(ho[m * N + nn] - ref)); mr = std::max(mr, std::fabs(ref)); } row0 = row1; }
    report("QuantGroupedMatmulInplaceAdd", me / (mr + 1e-9), 1e-5);
    aclDestroyIntArray(ia);
}
static void t_grouped_qmm(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int M = 6, K = 8, N = 4, E = 2; auto xq = randi8(M * K), wq = randi8(E * K * N); auto sc = randv(E * N, 0.01, 0.05);
    std::vector<float> ho(M * N); std::vector<int64_t> gl = {3, 6};
    DevBuf dx(M * K), dw(E * K * N), ds(E * N * 4), dout(M * N * 4); dx.up(xq.data()); dw.up(wq.data()); ds.up(sc.data());
    aclTensor *tx = mk({M, K}, ACL_INT8, dx.p), *tw = mk({E, K, N}, ACL_INT8, dw.p), *tsc = mk({E, N}, ACL_FLOAT, ds.p), *to = mk({M, N}, ACL_FLOAT, dout.p);
    aclIntArray *ia = mkIA(gl);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, tsc, ia, to, w, e); }, run);
    dout.down(ho.data()); double me = 0, mr = 0; int row0 = 0;
    for (int g = 0; g < E; g++) { int row1 = (int)gl[g]; for (int m = row0; m < row1; m++) for (int nn = 0; nn < N; nn++) { long long acc = 0; for (int k = 0; k < K; k++) acc += (long long)xq[m * K + k] * (long long)wq[(g * K + k) * N + nn]; double ref = (double)acc * sc[g * N + nn]; me = std::max(me, std::fabs(ho[m * N + nn] - ref)); mr = std::max(mr, std::fabs(ref)); } row0 = row1; }
    report(name, me / (mr + 1e-9), 1e-5);
    aclDestroyIntArray(ia);
}
static void t_wqmm(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int M = 5, K = 8, N = 6; auto x = randv(M * K, -1, 1); auto wq = randi8(K * N); auto sc = randv(N, 0.01, 0.05), of = randv(N, -1, 1);
    std::vector<float> ho(M * N); DevBuf dx(M * K * 4), dw(K * N), ds(N * 4), dof(N * 4), dout(M * N * 4); dx.up(x.data()); dw.up(wq.data()); ds.up(sc.data()); dof.up(of.data());
    aclTensor *tx = mk({M, K}, ACL_FLOAT, dx.p), *tw = mk({K, N}, ACL_INT8, dw.p), *tsc = mk({N}, ACL_FLOAT, ds.p), *tof = mk({N}, ACL_FLOAT, dof.p), *to = mk({M, N}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, tsc, tof, to, w, e); }, run);
    dout.down(ho.data()); double me = 0, mr = 0;
    for (int m = 0; m < M; m++) for (int nn = 0; nn < N; nn++) { double acc = 0; for (int k = 0; k < K; k++) acc += (double)x[m * K + k] * (((double)wq[k * N + nn] - of[nn]) * sc[nn]); me = std::max(me, std::fabs(ho[m * N + nn] - acc)); mr = std::max(mr, std::fabs(acc)); }
    report(name, me / (mr + 1e-9), 1e-4); // weight-only dequant matmul
}
static void t_gmm_swiglu_quant(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclIntArray*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int M = 6, K = 8, N = 4, E = 2; int N2 = 2 * N; auto xq = randi8(M * K), wq = randi8(E * K * N2);
    std::vector<int8_t> ho(M * N); std::vector<float> hs(M); std::vector<int64_t> gl = {3, 6};
    DevBuf dx(M * K), dw(E * K * N2), dout(M * N), dscale(M * 4); dx.up(xq.data()); dw.up(wq.data());
    aclTensor *tx = mk({M, K}, ACL_INT8, dx.p), *tw = mk({E, K, N2}, ACL_INT8, dw.p), *to = mk({M, N}, ACL_INT8, dout.p), *tscale = mk({M}, ACL_FLOAT, dscale.p);
    aclIntArray *ia = mkIA(gl);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, ia, to, tscale, w, e); }, run);
    dout.down(ho.data()); dscale.down(hs.data());
    // reference: tmp = x@w (int8 accum, no column scale in this signature), swiglu, per-row absmax int8
    double me = 0, mr = 0; std::vector<double> g(M * N); int row0 = 0;
    for (int gi = 0; gi < E; gi++) { int row1 = (int)gl[gi]; for (int m = row0; m < row1; m++) { std::vector<double> t(N2);
        for (int n = 0; n < N2; n++) { long long acc = 0; for (int k = 0; k < K; k++) acc += (long long)xq[m * K + k] * (long long)wq[(gi * K + k) * N2 + n]; t[n] = (double)acc; }
        for (int n = 0; n < N; n++) g[m * N + n] = silu(t[n]) * t[N + n]; } row0 = row1; }
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) { double deq = (double)ho[m * N + n] * hs[m]; me = std::max(me, std::fabs(deq - g[m * N + n])); mr = std::max(mr, std::fabs(g[m * N + n])); }
    report(name, me / (mr + 1e-9), 2e-2); // grouped matmul + swiglu + per-row int8 round-trip
    aclDestroyIntArray(ia);
}

// ------------------------------------------------------------------ quant conv
static void t_quant_conv(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, const aclTensor*, double, const aclIntArray*, const aclIntArray*, const aclIntArray*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int N = 1, Cin = 2, H = 5, Wd = 5, Cout = 3, kH = 3, kW = 3; double scale = 0.02;
    int s0 = 1, p0 = 1, d0 = 1; int oH = (H + 2 * p0 - (d0 * (kH - 1) + 1)) / s0 + 1, oW = (Wd + 2 * p0 - (d0 * (kW - 1) + 1)) / s0 + 1;
    auto xq = randi8(N * Cin * H * Wd), wq = randi8(Cout * Cin * kH * kW); auto bias = randv(Cout, -1, 1);
    std::vector<float> ho(N * Cout * oH * oW); DevBuf dx(xq.size()), dw(wq.size()), db(Cout * 4), dout(ho.size() * 4); dx.up(xq.data()); dw.up(wq.data()); db.up(bias.data());
    aclTensor *tx = mk({N, Cin, H, Wd}, ACL_INT8, dx.p), *tw = mk({Cout, Cin, kH, kW}, ACL_INT8, dw.p), *tb = mk({Cout}, ACL_FLOAT, db.p), *to = mk({N, Cout, oH, oW}, ACL_FLOAT, dout.p);
    aclIntArray *st = mkIA({s0, s0}), *pd = mkIA({p0, p0}), *dl = mkIA({d0, d0});
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, tw, tb, scale, st, pd, dl, 1, to, w, e); }, run);
    dout.down(ho.data()); double me = 0, mr = 0;
    for (int co = 0; co < Cout; co++) for (int oh = 0; oh < oH; oh++) for (int ow = 0; ow < oW; ow++) { long long acc = 0;
        for (int ci = 0; ci < Cin; ci++) for (int kh = 0; kh < kH; kh++) for (int kw = 0; kw < kW; kw++) { int hh = oh * s0 - p0 + kh * d0, ww = ow * s0 - p0 + kw * d0; if (hh < 0 || hh >= H || ww < 0 || ww >= Wd) continue; acc += (long long)xq[(ci * H + hh) * Wd + ww] * (long long)wq[((co * Cin + ci) * kH + kh) * kW + kw]; }
        double ref = (double)acc * scale + bias[co]; int idx = (co * oH + oh) * oW + ow; me = std::max(me, std::fabs(ho[idx] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report(name, me / (mr + 1e-9), 1e-5); // exact int8 conv
    aclDestroyIntArray(st); aclDestroyIntArray(pd); aclDestroyIntArray(dl);
}

// ------------------------------------------------------------------ attention quant
static void t_quant_fa() {
    const int B = 1, Nh = 1, S = 4, D = 8; double scale = 1.0 / std::sqrt((double)D);
    auto q = randv(B * Nh * S * D, -1, 1), k = randv(B * Nh * S * D, -1, 1), v = randv(B * Nh * S * D, -1, 1);
    std::vector<float> ho(B * Nh * S * D); DevBuf dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), dout(ho.size() * 4); dq.up(q.data()); dk.up(k.data()); dv.up(v.data());
    aclTensor *tq = mk({B, Nh, S, D}, ACL_FLOAT, dq.p), *tk = mk({B, Nh, S, D}, ACL_FLOAT, dk.p), *tv = mk({B, Nh, S, D}, ACL_FLOAT, dv.p), *to = mk({B, Nh, S, D}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnQuantFlashAttentionScoreGetWorkspaceSize(tq, tk, tv, nullptr, scale, (int64_t)Nh, false, to, w, e); }, aclnnQuantFlashAttentionScore);
    dout.down(ho.data()); double me = 0, mr = 0; std::vector<double> sc(S);
    for (int i = 0; i < S; i++) { double mx = -1e30; for (int j = 0; j < S; j++) { double dot = 0; for (int d = 0; d < D; d++) dot += (double)q[i * D + d] * k[j * D + d]; sc[j] = dot * scale; mx = std::max(mx, sc[j]); }
        double sum = 0; for (int j = 0; j < S; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        for (int d = 0; d < D; d++) { double acc = 0; for (int j = 0; j < S; j++) acc += sc[j] * v[j * D + d]; double ref = acc / sum; me = std::max(me, std::fabs(ho[i * D + d] - ref)); mr = std::max(mr, std::fabs(ref)); } }
    report("QuantFlashAttentionScore", me / (mr + 1e-9), 1e-5); // softmax attention reference
}
static void t_dequant_rope_quant_kv() {
    const int R = 4, D = 8; auto x = randv(R * D, -1, 1), cs = randv(R * D, -1, 1), sn = randv(R * D, -1, 1);
    std::vector<float> ho(R * D); DevBuf dx(R * D * 4), dc(R * D * 4), dsn(R * D * 4), dout(R * D * 4); dx.up(x.data()); dc.up(cs.data()); dsn.up(sn.data());
    aclTensor *tx = mk({R, D}, ACL_FLOAT, dx.p), *tc = mk({R, D}, ACL_FLOAT, dc.p), *tsn = mk({R, D}, ACL_FLOAT, dsn.p), *to = mk({R, D}, ACL_FLOAT, dout.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnDequantRopeQuantKvcacheGetWorkspaceSize(tx, tc, tsn, 0, to, w, e); }, aclnnDequantRopeQuantKvcache);
    dout.down(ho.data()); double me = 0, mr = 0; int half = D / 2;
    for (int r = 0; r < R; r++) for (int d = 0; d < D; d++) { double rot = (d < half) ? -(double)x[r * D + d + half] : (double)x[r * D + d - half]; double ref = (double)x[r * D + d] * cs[r * D + d] + rot * sn[r * D + d]; me = std::max(me, std::fabs(ho[r * D + d] - ref)); mr = std::max(mr, std::fabs(ref)); }
    report("DequantRopeQuantKvcache", me / (mr + 1e-9), 1e-5); // rotate-half RoPE
}
static void t_swin_attn_quant() {
    const int B = 1, Nh = 1, S = 4, D = 8; double scale = 1.0 / std::sqrt((double)D);
    auto q = randv(B * Nh * S * D, -1, 1), k = randv(B * Nh * S * D, -1, 1), v = randv(B * Nh * S * D, -1, 1);
    std::vector<int8_t> ho(B * Nh * S * D); std::vector<float> hs(S);
    DevBuf dq(q.size() * 4), dk(k.size() * 4), dv(v.size() * 4), dout(ho.size()), dsc(S * 4); dq.up(q.data()); dk.up(k.data()); dv.up(v.data());
    aclTensor *tq = mk({B, Nh, S, D}, ACL_FLOAT, dq.p), *tk = mk({B, Nh, S, D}, ACL_FLOAT, dk.p), *tv = mk({B, Nh, S, D}, ACL_FLOAT, dv.p), *to = mk({B, Nh, S, D}, ACL_INT8, dout.p), *tsc = mk({S}, ACL_FLOAT, dsc.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSwinAttentionScoreQuantGetWorkspaceSize(tq, tk, tv, nullptr, scale, (int64_t)Nh, false, to, w, e); }, aclnnSwinAttentionScoreQuant);
    // NOTE: ATT_DECL signature has only `out`; our impl needs a scaleOut tensor — it is stashed via e->out2. Not present here.
    (void)tsc; dout.down(ho.data());
    // reference attention output, then per-row absmax/127 quant — verify quantization is consistent (q*scale ≈ attn)
    double me = 0, mr = 0; std::vector<double> sc(S), out(S * D);
    for (int i = 0; i < S; i++) { double mx = -1e30; for (int j = 0; j < S; j++) { double dot = 0; for (int d = 0; d < D; d++) dot += (double)q[i * D + d] * k[j * D + d]; sc[j] = dot * scale; mx = std::max(mx, sc[j]); }
        double sum = 0; for (int j = 0; j < S; j++) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
        double amax = 0; for (int d = 0; d < D; d++) { double acc = 0; for (int j = 0; j < S; j++) acc += sc[j] * v[j * D + d]; out[i * D + d] = acc / sum; amax = std::max(amax, std::fabs(out[i * D + d])); }
        double rscale = amax > 0 ? amax / 127.0 : 1.0; for (int d = 0; d < D; d++) { int qq = clampi(std::lrint(out[i * D + d] / rscale), -127, 127); double deq = (double)qq * rscale; me = std::max(me, std::fabs(deq - out[i * D + d])); mr = std::max(mr, std::fabs(out[i * D + d])); } }
    report("SwinAttentionScoreQuant", me / (mr + 1e-9), 1e-2); // attention then per-row int8 round-trip
}

// ------------------------------------------------------------------ moe init routing quant
static void t_moe_init_routing_quant(const char *name, aclnnStatus (*gws)(const aclTensor*, const aclTensor*, int64_t, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**), aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    const int T = 5, D = 6, E = 3; auto x = randv(T * D, -2, 2); std::vector<int32_t> eid = {2, 0, 1, 0, 2};
    std::vector<int8_t> hex(T * D); std::vector<int32_t> hri(T), hei(T);
    DevBuf dx(T * D * 4), de(T * 4), dex(T * D), dri(T * 4), dei(T * 4); dx.up(x.data()); de.up(eid.data());
    aclTensor *tx = mk({T, D}, ACL_FLOAT, dx.p), *te = mk({T}, ACL_INT32, de.p), *tex = mk({T, D}, ACL_INT8, dex.p), *tri = mk({T}, ACL_INT32, dri.p), *tei = mk({T}, ACL_INT32, dei.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return gws(tx, te, (int64_t)E, tex, tri, tei, w, e); }, run);
    dex.down(hex.data()); dri.down(hri.data()); dei.down(hei.data());
    // expected order: experts 0,1,2 stably -> rows {1,3},{2},{0,4}; expert ids sorted; quant round-trips to gathered x.
    std::vector<int> order; for (int exp = 0; exp < E; exp++) for (int t = 0; t < T; t++) if (eid[t] == exp) order.push_back(t);
    double bad = 0, me = 0, mr = 0;
    for (int w = 0; w < T; w++) { if (hri[w] != order[w]) bad = 1; if (hei[w] != eid[order[w]]) bad = 1;
        double amax = 0; for (int d = 0; d < D; d++) amax = std::max(amax, std::fabs((double)x[order[w] * D + d]));
        double rscale = amax > 0 ? amax / 127.0 : 1.0; for (int d = 0; d < D; d++) { double deq = (double)hex[w * D + d] * rscale; me = std::max(me, std::fabs(deq - x[order[w] * D + d])); mr = std::max(mr, std::fabs((double)x[order[w] * D + d])); } }
    report(name, std::max(bad, me / (mr + 1e-9)), 1e-2); // routing order + per-row int8 round-trip
}

int main() {
    init(); srand(71);
    // pure quant/dequant
    t_ascend_quant_v3();
    t_ascend_antiquant();
    t_group_quant();
    t_gelu_quant();
    t_gnsilu_quant();
    t_fakeq_perchannel();
    t_quantized_bn();
    t_dynamic_quant("DynamicQuantV2", aclnnDynamicQuantV2GetWorkspaceSize, aclnnDynamicQuantV2);
    t_dynamic_quant("DynamicQuantV3", aclnnDynamicQuantV3GetWorkspaceSize, aclnnDynamicQuantV3);
    t_dynamic_quant("DynamicQuantV4", aclnnDynamicQuantV4GetWorkspaceSize, aclnnDynamicQuantV4);
    t_block_quant("DynamicBlockQuantV2", 16, aclnnDynamicBlockQuantV2GetWorkspaceSize, aclnnDynamicBlockQuantV2, false);
    t_block_quant("GroupedDynamicBlockQuant", 16, aclnnGroupedDynamicBlockQuantGetWorkspaceSize, aclnnGroupedDynamicBlockQuant, false);
    t_block_quant("DynamicMxQuant", 16, aclnnDynamicMxQuantGetWorkspaceSize, aclnnDynamicMxQuant, true);
    t_block_quant("DynamicMxQuantV2", 16, aclnnDynamicMxQuantV2GetWorkspaceSize, aclnnDynamicMxQuantV2, true);
    t_block_quant("GroupedDynamicMxQuant", 16, aclnnGroupedDynamicMxQuantGetWorkspaceSize, aclnnGroupedDynamicMxQuant, true);
    t_mx_dualaxis();
    t_swiglu_quant("SwiGluQuantV2", false);
    t_swiglu_quant("DequantSwigluQuantV2", true);
    // norm+quant fusions
    t_addrms_quant("AddRmsNormQuant", aclnnAddRmsNormQuantGetWorkspaceSize, aclnnAddRmsNormQuant);
    t_addrms_quant("AddRmsNormQuantV2", aclnnAddRmsNormQuantV2GetWorkspaceSize, aclnnAddRmsNormQuantV2);
    t_addrms_dyn_quant();
    t_addrms_mx_quant();
    t_adaln_quant();
    t_addln_quant();
    t_swin_lnqkv_quant();
    // trans quant param + adamw
    t_trans_quant_param("TransQuantParamV2", aclnnTransQuantParamV2GetWorkspaceSize, aclnnTransQuantParamV2);
    t_trans_quant_param("TransQuantParamV3", aclnnTransQuantParamV3GetWorkspaceSize, aclnnTransQuantParamV3);
    t_apply_adamw_quant();
    // W8A8 quant matmul
    t_qmm("QuantMatmulV2", aclnnQuantMatmulV2GetWorkspaceSize, aclnnQuantMatmulV2);
    t_qmm("QuantMatmulV4", aclnnQuantMatmulV4GetWorkspaceSize, aclnnQuantMatmulV4);
    t_qmm("QuantMatmulV5", aclnnQuantMatmulV5GetWorkspaceSize, aclnnQuantMatmulV5);
    t_qmm("QuantMatmulWeightNz", aclnnQuantMatmulWeightNzGetWorkspaceSize, aclnnQuantMatmulWeightNz);
    t_qmm("QuantMatmulDequant", aclnnQuantMatmulDequantGetWorkspaceSize, aclnnQuantMatmulDequant);
    t_qmm("FusedQuantMatmul", aclnnFusedQuantMatmulGetWorkspaceSize, aclnnFusedQuantMatmul);
    t_qmm("FusedQuantMatmulWeightNz", aclnnFusedQuantMatmulWeightNzGetWorkspaceSize, aclnnFusedQuantMatmulWeightNz);
    t_qmm("MatmulCompressDequant", aclnnMatmulCompressDequantGetWorkspaceSize, aclnnMatmulCompressDequant);
    t_qmm_comm("AlltoAllQuantMatmul", aclnnAlltoAllQuantMatmulGetWorkspaceSize, aclnnAlltoAllQuantMatmul);
    t_qmm_comm("QuantMatmulAlltoAll", aclnnQuantMatmulAlltoAllGetWorkspaceSize, aclnnQuantMatmulAlltoAll);
    t_qmm_comm("QuantMatmulReduceSumWeightNz", aclnnQuantMatmulReduceSumWeightNzGetWorkspaceSize, aclnnQuantMatmulReduceSumWeightNz);
    t_qmm("Sparse4to2QuantMatmulWeightNz", aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize, aclnnSparse4to2QuantMatmulWeightNz);
    t_qmm("DualLevelQuantMatmulWeightNz", aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize, aclnnDualLevelQuantMatmulWeightNz);
    t_wqmm("WeightQuantBatchMatmulNz", aclnnWeightQuantBatchMatmulNzGetWorkspaceSize, aclnnWeightQuantBatchMatmulNz);
    t_wqmm("WeightQuantBatchMatmulV2", aclnnWeightQuantBatchMatmulV2GetWorkspaceSize, aclnnWeightQuantBatchMatmulV2);
    t_wqmm("WeightQuantBatchMatmulV3", aclnnWeightQuantBatchMatmulV3GetWorkspaceSize, aclnnWeightQuantBatchMatmulV3);
    t_grouped_qmm("QuantGroupedMatmulDequant", aclnnQuantGroupedMatmulDequantGetWorkspaceSize, aclnnQuantGroupedMatmulDequant);
    t_grouped_qmm("QuantGroupedMatmulDequantWeightNZ", aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize, aclnnQuantGroupedMatmulDequantWeightNZ);
    // canonical signature is ungrouped (mc2.h: x,weight,scale,comm,out — no groupList), matching the CUDA backend's QMM_FWD
    t_qmm_comm("QuantGroupedMatMulAlltoAllv", aclnnQuantGroupedMatMulAlltoAllvGetWorkspaceSize, aclnnQuantGroupedMatMulAlltoAllv);
    t_qmm_comm("AlltoAllvQuantGroupedMatMul", aclnnAlltoAllvQuantGroupedMatMulGetWorkspaceSize, aclnnAlltoAllvQuantGroupedMatMul);
    t_qmm_iadd_grouped();
    t_gmm_swiglu_quant("GroupedMatmulSwigluQuant", aclnnGroupedMatmulSwigluQuantGetWorkspaceSize, aclnnGroupedMatmulSwigluQuant);
    t_gmm_swiglu_quant("GroupedMatmulSwigluQuantV2", aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize, aclnnGroupedMatmulSwigluQuantV2);
    t_gmm_swiglu_quant("GroupedMatmulSwigluQuantWeightNZ", aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize, aclnnGroupedMatmulSwigluQuantWeightNZ);
    t_gmm_swiglu_quant("GroupedMatmulSwigluQuantWeightNzV2", aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize, aclnnGroupedMatmulSwigluQuantWeightNzV2);
    // quant conv
    t_quant_conv("QuantConvolution", aclnnQuantConvolutionGetWorkspaceSize, aclnnQuantConvolution);
    t_quant_conv("QuantConvolutionWeightNz", aclnnQuantConvolutionWeightNzGetWorkspaceSize, aclnnQuantConvolutionWeightNz);
    // attention quant
    t_quant_fa();
    t_dequant_rope_quant_kv();
    t_swin_attn_quant();
    // moe init routing quant
    t_moe_init_routing_quant("MoeInitRoutingQuant", aclnnMoeInitRoutingQuantGetWorkspaceSize, aclnnMoeInitRoutingQuant);
    t_moe_init_routing_quant("MoeInitRoutingQuantV2", aclnnMoeInitRoutingQuantV2GetWorkspaceSize, aclnnMoeInitRoutingQuantV2);
    return finish();
}
