/*
 * aclnn_mc2.h — mc2 distributed fused operators (Matmul + HCCL collective).
 * Kept separate from aclnn_ops.h because these signatures depend on the HCCL types (HcclComm).
 * Implemented over the real RoCE/HCCL path (NCCL backend); validated on 2 nodes.
 */
#ifndef CANN_ON_GPU_ACLNN_MC2_H
#define CANN_ON_GPU_ACLNN_MC2_H

#include "aclnnop/aclnn_ops.h"
#include "hccl/hccl.h"

#ifdef __cplusplus
extern "C" {
#endif

// MatmulAllReduce: out = AllReduce_SUM( x[M,K] @ weight[K,N] ) (+ bias[N] once). comm = HCCL handle (nullptr → local only).
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// QuantMatmulAllReduce: out = AllReduce_SUM( dequant(x_int8 @ weight_int8, scale[N]) ).
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
    HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MatmulReduceScatter: out[M/nranks,N] = ReduceScatter_SUM( x[M,K] @ weight[K,N] ).
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulReduceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MatmulAllGather: out[M,N] = AllGather( x_shard[M/nranks,K] @ weight[K,N] ) over the row shards.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllGatherGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
    HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllGather(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MatmulAllReduceAddRmsNorm: y = RmsNorm( AllReduce(x@W) + residual )·gamma; residualSum = that sum.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *residual,
    const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MoE expert-parallel token exchange (HcclAlltoAll over [nranks,C,H] capacity layout). Dispatch == Combine (symmetric).
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatch(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombine(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulAllReduceV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulReduceScatterV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulReduceScatterV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV3GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV4GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV2GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV3GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV4GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeDispatchV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV2GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV3GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV4GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- distributed matmul fusions (single-rank degenerate = local; matmul4_ext) ----
#define MC2_MM(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MC2_MM(aclnnAllGatherMatmul) MC2_MM(aclnnAllGatherMatmulV2) MC2_MM(aclnnMatmulAlltoAll) MC2_MM(aclnnAlltoAllMatmul)
MC2_MM(aclnnAlltoAllAllGatherBatchMatMul) MC2_MM(aclnnBatchMatMulReduceScatterAlltoAll)
#undef MC2_MM
#define MC2_GMM(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MC2_GMM(aclnnGroupedMatMulAllReduce) MC2_GMM(aclnnGroupedMatMulAlltoAllv) MC2_GMM(aclnnAlltoAllvGroupedMatMul)
#undef MC2_GMM
#define MC2_QMM(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MC2_QMM(aclnnAlltoAllQuantMatmul) MC2_QMM(aclnnQuantMatmulAlltoAll) MC2_QMM(aclnnAlltoAllvQuantGroupedMatMul)
MC2_QMM(aclnnQuantGroupedMatMulAlltoAllv) MC2_QMM(aclnnQuantReduceScatter) MC2_QMM(aclnnQuantMatmulReduceSumWeightNz)
#undef MC2_QMM
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffset, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantMatmulAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantAllReduceGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantAllReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparse4to2QuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparse4to2QuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDualLevelQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDualLevelQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransSparse4to2ParaGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransSparse4to2Para(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *scale, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *weight, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- MoE distribute combine+rmsnorm + setup/teardown (moe3_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define MOE_ST(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOE_ST(aclnnMoeDistributeDispatchSetup) MOE_ST(aclnnMoeDistributeDispatchTeardown) MOE_ST(aclnnMoeDistributeCombineSetup) MOE_ST(aclnnMoeDistributeCombineTeardown)
#undef MOE_ST
#ifdef __cplusplus
}
#endif
#endif
