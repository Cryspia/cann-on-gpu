/*
 * aclnn_ops.h — shim self-declarations for all aclnn operators (the standard header ships with the
 * CANN ops package and is not available locally). Signatures match the official CANN documentation;
 * elementwise op semantics correspond to AscendC kernels verified by explorer.
 */
#ifndef CANN_ON_GPU_ACLNN_OPS_H
#define CANN_ON_GPU_ACLNN_OPS_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

// aclTensorList element access (acl_meta.h only provides Create/Destroy/Size; this adds element access)
ACL_FUNC_VISIBILITY aclTensor *aclGetTensorListElement(const aclTensorList *tensorList, uint64_t index);

#define ACLNN_BIN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, \
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_BIN_ALPHA(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, \
    const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_SCALAR_ALPHA(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, \
    const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_SCALAR(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, \
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_UN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, \
    uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACLNN_BIN_ALPHA(aclnnSub)
ACLNN_BIN(aclnnMul)
ACLNN_BIN(aclnnDiv)
ACLNN_BIN(aclnnMaximum)
ACLNN_BIN(aclnnMinimum)

ACLNN_SCALAR_ALPHA(aclnnAdds)
ACLNN_SCALAR_ALPHA(aclnnSubs)
ACLNN_SCALAR(aclnnMuls)
ACLNN_SCALAR(aclnnDivs)
ACLNN_SCALAR(aclnnClampMin)
ACLNN_SCALAR(aclnnClampMax)

ACLNN_UN(aclnnExp)
ACLNN_UN(aclnnLog)
ACLNN_UN(aclnnAbs)
ACLNN_UN(aclnnSqrt)
ACLNN_UN(aclnnRsqrt)
ACLNN_UN(aclnnReciprocal)
ACLNN_UN(aclnnRelu)
ACLNN_UN(aclnnNeg)

ACLNN_UN(aclnnSigmoid)
ACLNN_UN(aclnnTanh)
ACLNN_UN(aclnnErf)
ACLNN_UN(aclnnGelu)
ACLNN_UN(aclnnSilu)
ACLNN_UN(aclnnSoftplus)
ACLNN_UN(aclnnSin)
ACLNN_UN(aclnnCos)
ACLNN_UN(aclnnTan)
ACLNN_UN(aclnnAtan)
ACLNN_UN(aclnnSign)
ACLNN_UN(aclnnFloor)
ACLNN_UN(aclnnCeil)
ACLNN_UN(aclnnRound)
ACLNN_UN(aclnnTrunc)
ACLNN_UN(aclnnSquare)
ACLNN_UN(aclnnSinh)
ACLNN_UN(aclnnCosh)
ACLNN_UN(aclnnAsin)
ACLNN_UN(aclnnAcos)
ACLNN_UN(aclnnErfc)
ACLNN_UN(aclnnFrac)
ACLNN_UN(aclnnLgamma)
ACLNN_SCALAR(aclnnPowTensorScalar)
ACLNN_BIN(aclnnFmod)
ACLNN_BIN(aclnnHypot)
ACLNN_BIN(aclnnPowTensorTensor)

ACLNN_BIN(aclnnBitwiseAndTensor)
ACLNN_BIN(aclnnBitwiseOrTensor)
ACLNN_UN(aclnnBitwiseNot)

// ---- elementwise math extensions ----
ACLNN_UN(aclnnExpm1) ACLNN_UN(aclnnLog1p) ACLNN_UN(aclnnLog2) ACLNN_UN(aclnnLog10) ACLNN_UN(aclnnExp2) ACLNN_UN(aclnnErfinv)
ACLNN_BIN(aclnnAtan2) ACLNN_BIN(aclnnRemainderTensorTensor) ACLNN_BIN(aclnnXLogYTensorTensor)
ACLNN_BIN(aclnnLogAddExp) ACLNN_BIN(aclnnCopysign) ACLNN_BIN(aclnnHeaviside)
ACLNN_BIN_ALPHA(aclnnLerp)
// Addcmul/Addcdiv: out = self + value*(t1 op t2); ClampTensor: elementwise clamp to [min,max] tensors
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddcmulGetWorkspaceSize(const aclTensor *self, const aclTensor *t1, const aclTensor *t2, const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddcmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddcdivGetWorkspaceSize(const aclTensor *self, const aclTensor *t1, const aclTensor *t2, const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddcdiv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClampTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *minT, const aclTensor *maxT, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClampTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- activation extensions ----
ACLNN_UN(aclnnMish) ACLNN_UN(aclnnHardswish) ACLNN_UN(aclnnHardsigmoid) ACLNN_UN(aclnnLogSigmoid) ACLNN_UN(aclnnSelu) ACLNN_UN(aclnnTanhshrink) ACLNN_UN(aclnnRelu6)
ACLNN_SCALAR(aclnnLeakyRelu) ACLNN_SCALAR(aclnnElu) ACLNN_SCALAR(aclnnCelu) ACLNN_SCALAR(aclnnHardshrink) ACLNN_SCALAR(aclnnSoftshrink)
ACLNN_BIN(aclnnPrelu)
// Hardtanh(min,max) / Threshold(threshold,value): two scalar parameters
ACL_FUNC_VISIBILITY aclnnStatus aclnnHardtanhGetWorkspaceSize(const aclTensor *self, const aclScalar *minVal, const aclScalar *maxVal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHardtanh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThresholdGetWorkspaceSize(const aclTensor *self, const aclScalar *threshold, const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThreshold(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnCastGetWorkspaceSize(const aclTensor *self, aclDataType dtype,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// fp6 true-Ascend 4-in-3 byte pack/unpack (optional memory layout): self fp6 1 byte/element <-> out ceil(n/4)*3 bytes
ACL_FUNC_VISIBILITY aclnnStatus aclnnFp6PackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFp6Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFp6UnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFp6Unpack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// FRACTAL_NZ <-> ND conversion (2D, 16x16 fractal)
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransDataND2NZGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransDataND2NZ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransDataNZ2NDGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransDataNZ2ND(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Additional fractal format conversions: NC1HWC0 (5HD) and FRACTAL_Z <-> ND/NCHW
#define ACLNN_TRANS(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_TRANS(aclnnTransDataNCHW2NC1HWC0) ACLNN_TRANS(aclnnTransDataNC1HWC0toNCHW)
ACLNN_TRANS(aclnnTransDataND2FZ) ACLNN_TRANS(aclnnTransDataFZ2ND)

// ---- format / data-movement extensions ----
ACLNN_TRANS(aclnnContiguous) ACLNN_TRANS(aclnnCopy) ACLNN_TRANS(aclnnIdentity) ACLNN_TRANS(aclnnViewCopy)
ACL_FUNC_VISIBILITY aclnnStatus aclnnAsStridedGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, const aclIntArray *stride, int64_t storageOffset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAsStrided(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim,
    bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceSum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnAmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim,
    bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim,
    bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAmin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim,
    bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMean(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnProdGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim,
    bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnProd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnArgMaxGetWorkspaceSize(const aclTensor *self, int64_t dim,
    bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnArgMinGetWorkspaceSize(const aclTensor *self, int64_t dim,
    bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnArgMin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- reshape/indexing ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnPermuteGetWorkspaceSize(const aclTensor *self, const aclIntArray *dims,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnSliceGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end,
    int64_t step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSlice(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// StridedSlice: multi-dimensional begin/end/strides (each of length rank)
ACL_FUNC_VISIBILITY aclnnStatus aclnnStridedSliceGetWorkspaceSize(const aclTensor *self, const aclIntArray *begin,
    const aclIntArray *end, const aclIntArray *strides, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStridedSlice(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnTileGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTile(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnConstantPadNdGetWorkspaceSize(const aclTensor *self, const aclIntArray *padding,
    const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConstantPadNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGather(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// GatherV2 (batch_dims; constraint axis==batchDims): self[B...,A,T...] index[B...,I] -> out[B...,I,T...]
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherV2GetWorkspaceSize(const aclTensor *self, int64_t batchDims, int64_t axis,
    const aclTensor *index, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnCatGetWorkspaceSize(const aclTensor *const *tensors, uint64_t num, int64_t dim,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Native aclTensorList interface: aclTensorList type and aclCreateTensorList are provided by acl_meta.h; this adds the list-based Cat variant
ACL_FUNC_VISIBILITY aclnnStatus aclnnCatListGetWorkspaceSize(const aclTensorList *tensors, int64_t dim,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCatList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Stack: N same-shaped tensors stacked along a new dimension dim into out (out has one extra dimension of size N)
ACL_FUNC_VISIBILITY aclnnStatus aclnnStackGetWorkspaceSize(const aclTensorList *tensors, int64_t dim,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnSplitWithSizeGetWorkspaceSize(const aclTensor *self, int64_t dim,
    const aclTensor *const *outputs, uint64_t num, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSplitWithSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Embedding / IndexSelect
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingGetWorkspaceSize(const aclTensor *weight, const aclTensor *ids, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbedding(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingDenseBackwardGetWorkspaceSize(const aclTensor *grad, const aclTensor *ids, aclTensor *gradWeight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingDenseBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Additional index ops
ACL_FUNC_VISIBILITY aclnnStatus aclnnArangeGetWorkspaceSize(double start, double end, double step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnArange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnOneHotGetWorkspaceSize(const aclTensor *ids, int64_t numClasses, double onValue, double offValue, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnOneHot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlipGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlip(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRollGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t shift, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveGetWorkspaceSize(const aclTensor *self, int64_t repeats, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleave(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t outputSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveInt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatGetWorkspaceSize(const aclTensor *self, const aclIntArray *repeats, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLossOutGetWorkspaceSize(const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLossOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// N-dimensional gather/scatter
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherNdGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterNdUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterNdUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterNd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpandGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpand(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterUpdateGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ScatterAdd: out=self, out[index[l],:] += src[l,:]; repeated indices are atomically accumulated (equivalent to index_put accumulate / index_add)
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterAddGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Scatter with reduce mode (0=replace,1=add) + in-place scatter
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatterGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclTensor *src, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatterUpdateGetWorkspaceSize(aclTensor *selfRef, const aclTensor *index, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatterUpdate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// IndexAddV2 + IndexFillTensor + in-place index variants
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexAddV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexCopyGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexFillGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, double value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexFill(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexFillTensorGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceIndexFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedSelectGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, aclTensor *out, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedSelect(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonzeroGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonzero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- indexing extensions ----
#define ACLNN_IDX1(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IDX1(aclnnTril) ACLNN_IDX1(aclnnTriu) ACLNN_IDX1(aclnnDiagonal) ACLNN_IDX1(aclnnDiagFlat)
#undef ACLNN_IDX1
ACL_FUNC_VISIBILITY aclnnStatus aclnnTraceGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTrace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBincountGetWorkspaceSize(const aclTensor *self, int64_t numClasses, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBincount(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSearchSortedGetWorkspaceSize(const aclTensor *sorted, const aclTensor *values, bool right, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSearchSorted(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBucketizeGetWorkspaceSize(const aclTensor *values, const aclTensor *boundaries, bool right, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBucketize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexAddGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexFillGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, double value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexFill(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexCopyGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_SCAT(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_SCAT(aclnnScatterMax) ACLNN_SCAT(aclnnScatterMin) ACLNN_SCAT(aclnnScatterMul)
#undef ACLNN_SCAT
ACL_FUNC_VISIBILITY aclnnStatus aclnnTakeGetWorkspaceSize(const aclTensor *self, const aclTensor *index, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTake(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTakeAlongDimGetWorkspaceSize(const aclTensor *self, const aclTensor *index, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTakeAlongDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedScatterGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *src, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNarrowGetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t length, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNarrow(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_UNIQ(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_UNIQ(aclnnUnique) ACLNN_UNIQ(aclnnUniqueConsecutive)
#undef ACLNN_UNIQ

// ---- shape extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnReshapeGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReshape(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSqueezeGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSqueeze(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnsqueezeGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnsqueeze(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMovedimGetWorkspaceSize(const aclTensor *self, int64_t src, int64_t dst, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMovedim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRot90GetWorkspaceSize(const aclTensor *self, int64_t k, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRot90(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_PAD2D(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_PAD2D(aclnnReflectionPad2d) ACLNN_PAD2D(aclnnReplicationPad2d) ACLNN_PAD2D(aclnnCircularPad2d)
#undef ACLNN_PAD2D
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2ColGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2Col(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCol2ImGetWorkspaceSize(const aclTensor *self, const aclIntArray *outputSize, const aclIntArray *kernel, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCol2Im(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPixelShuffleGetWorkspaceSize(const aclTensor *self, int64_t upscaleFactor, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPixelShuffle(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPixelUnshuffleGetWorkspaceSize(const aclTensor *self, int64_t downscaleFactor, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPixelUnshuffle(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChannelShuffleGetWorkspaceSize(const aclTensor *self, int64_t groups, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChannelShuffle(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- shape composites remainder ----
#define ACLNN_PADX(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_PADX(aclnnReflectionPad1d) ACLNN_PADX(aclnnReplicationPad1d) ACLNN_PADX(aclnnCircularPad1d)
ACLNN_PADX(aclnnReflectionPad3d) ACLNN_PADX(aclnnReplicationPad3d) ACLNN_PADX(aclnnCircularPad3d)
#undef ACLNN_PADX
ACL_FUNC_VISIBILITY aclnnStatus aclnnMeshgridGetWorkspaceSize(const aclTensorList *tensors, const aclTensorList *outs, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMeshgrid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChunkGetWorkspaceSize(const aclTensor *self, int64_t chunks, int64_t dim, const aclTensor *const *outputs, uint64_t num, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChunk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_STACKX(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *const *tensors, uint64_t num, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_STACKX(aclnnVstack) ACLNN_STACKX(aclnnHstack) ACLNN_STACKX(aclnnDstack)
#undef ACLNN_STACKX

// ---- pooling / interpolate / vision extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool1dGetWorkspaceSize(const aclTensor *self, int64_t kernel, int64_t stride, int64_t padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool1dGetWorkspaceSize(const aclTensor *self, int64_t kernel, int64_t stride, int64_t padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSample2dGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSample2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAffineGridGetWorkspaceSize(const aclTensor *theta, int64_t H, int64_t W, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAffineGrid(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNmsGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNms(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- normalization extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnInstanceNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInstanceNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLpNormalizeGetWorkspaceSize(const aclTensor *self, double p, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLpNormalize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLocalResponseNormGetWorkspaceSize(const aclTensor *self, int64_t size, double alpha, double beta, double k, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLocalResponseNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormGatedGetWorkspaceSize(const aclTensor *self, const aclTensor *gate, const aclTensor *weight, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormGated(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, const aclTensor *savedMean, const aclTensor *savedInvStd, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Loss functions (reduction 1=mean/2=sum)
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLossGetWorkspaceSize(const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *pred, const aclTensor *target, int64_t reduction, aclTensor *gradPred, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMseLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogits(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLossGetWorkspaceSize(const aclTensor *logProb, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLossGetWorkspaceSize(const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *logits, const aclTensor *target, int64_t reduction, aclTensor *gradLogits, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- loss extensions ----
#define ACLNN_LOSS(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_LOSS(aclnnL1Loss) ACLNN_LOSS(aclnnKlDiv) ACLNN_LOSS(aclnnBinaryCrossEntropy) ACLNN_LOSS(aclnnSoftMarginLoss)
#undef ACLNN_LOSS
ACL_FUNC_VISIBILITY aclnnStatus aclnnSmoothL1LossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, double beta, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSmoothL1Loss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHuberLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, double delta, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHuberLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnL1LossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnL1LossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSmoothL1LossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *target, int64_t reduction, double beta, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSmoothL1LossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMarginRankingLossGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *y, double margin, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMarginRankingLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHingeEmbeddingLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double margin, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHingeEmbeddingLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// loss remainder
ACL_FUNC_VISIBILITY aclnnStatus aclnnPoissonNllLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, bool logInput, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPoissonNllLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGaussianNllLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, const aclTensor *var, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGaussianNllLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiLabelSoftMarginLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiLabelSoftMarginLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiMarginLossGetWorkspaceSize(const aclTensor *input, const aclTensor *target, double p, double margin, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiMarginLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTripletMarginLossGetWorkspaceSize(const aclTensor *anchor, const aclTensor *positive, const aclTensor *negative, double margin, double p, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTripletMarginLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCosineEmbeddingLossGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, const aclTensor *target, double margin, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCosineEmbeddingLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCtcLossGetWorkspaceSize(const aclTensor *logProbs, const aclTensor *targets, const aclTensor *inputLengths, const aclTensor *targetLengths, int64_t blank, int64_t reduction, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCtcLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Optimizer remainder: LAMB / LARS (layer-wise adaptive trust ratio) and FusedEmaAdam (AdamW + param EMA), fp32 in-place
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyLambGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyLamb(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyLarsGetWorkspaceSize(aclTensor *param, aclTensor *buf, const aclTensor *grad, double lr, double momentum, double weightDecay, double trustCoeff, double eps, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyLars(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedEmaAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, aclTensor *ema, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, double emaDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedEmaAdam(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Optimizer: ApplyAdamW fused in-place update
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamWGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad,
    double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamW(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- optimizer extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdam(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdagradGetWorkspaceSize(aclTensor *param, aclTensor *stateSum, const aclTensor *grad, double lr, double eps, double weightDecay, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdagrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRmspropGetWorkspaceSize(aclTensor *param, aclTensor *v, const aclTensor *grad, double lr, double alpha, double eps, double weightDecay, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRmsprop(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyMomentumGetWorkspaceSize(aclTensor *param, aclTensor *buf, const aclTensor *grad, double lr, double momentum, double weightDecay, double dampening, bool nesterov, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyMomentum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamaxGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *u, const aclTensor *grad, double lr, double beta1, double beta2, double eps, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdadeltaGetWorkspaceSize(aclTensor *param, aclTensor *squareAvg, aclTensor *accDelta, const aclTensor *grad, double lr, double rho, double eps, double weightDecay, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdadelta(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClipGradNormGetWorkspaceSize(aclTensor *grad, double maxNorm, aclTensor *totalNormOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClipGradNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- activation backward coverage ----
#define ACLNN_ABWD(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_ABWD(aclnnReluBackward) ACLNN_ABWD(aclnnGeluBackward) ACLNN_ABWD(aclnnSiluBackward) ACLNN_ABWD(aclnnSoftplusBackward)
ACLNN_ABWD(aclnnHardswishBackward) ACLNN_ABWD(aclnnSigmoidBackward) ACLNN_ABWD(aclnnTanhBackward)
ACLNN_ABWD(aclnnFastGeluBackward) ACLNN_ABWD(aclnnHardsigmoidBackward) ACLNN_ABWD(aclnnHardswishBackwardV2)
ACLNN_ABWD(aclnnLogSigmoidBackward) ACLNN_ABWD(aclnnMishBackward) ACLNN_ABWD(aclnnSeluBackward)
#undef ACLNN_ABWD
// activation backward with extra params
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluBackwardV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, int64_t approximate, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluBackwardV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_ABWD_S(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *p, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_ABWD_S(aclnnHardshrinkBackward) ACLNN_ABWD_S(aclnnSoftshrinkBackward) ACLNN_ABWD_S(aclnnThresholdBackward)
#undef ACLNN_ABWD_S
ACL_FUNC_VISIBILITY aclnnStatus aclnnHardtanhBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclScalar *minVal, const aclScalar *maxVal, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHardtanhBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeakyReluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double negativeSlope, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeakyReluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double alpha, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t dim, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *output, int64_t dim, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSoftmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// pooling / interpolate / gather backward
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherBackwardGetWorkspaceSize(const aclTensor *gradOut, int64_t dim, const aclTensor *index, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithIndicesGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithIndices(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool2dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// vision remainder
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLpPool2dGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *kernel, const aclIntArray *stride, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLpPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSample3dGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSample3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlign(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Random number ops (counter-RNG, statistically verified)
ACL_FUNC_VISIBILITY aclnnStatus aclnnUniformGetWorkspaceSize(aclTensor *out, double from, double to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUniform(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalGetWorkspaceSize(aclTensor *out, double mean, double std, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Normal family (mean/std each scalar or tensor) + in-place
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(double mean, double std, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalFloatFloat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(double mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalFloatTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(const aclTensor *mean, double std, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalTensorFloat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(const aclTensor *mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormalTensorTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNormalGetWorkspaceSize(aclTensor *selfRef, double mean, double std, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mean, const aclTensor *std, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNormalTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBernoulliGetWorkspaceSize(aclTensor *out, double p, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropout(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRandpermGetWorkspaceSize(int64_t n, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRandperm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- random / RNN / FFT remainder (R8) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnSampleGammaGetWorkspaceSize(const aclTensor *alpha, double scale, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSampleGamma(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSampleDirichletGetWorkspaceSize(const aclTensor *alpha, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSampleDirichlet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRnnGetWorkspaceSize(const aclTensor *input, const aclTensor *h0, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, int64_t nonlinearity, aclTensor *out, aclTensor *hn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRnn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingBagGetWorkspaceSize(const aclTensor *weight, const aclTensor *indices, const aclTensor *offsets, int64_t mode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingBag(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFft2GetWorkspaceSize(const aclTensor *x, int64_t n0, int64_t n1, bool inverse, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFft2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStftGetWorkspaceSize(const aclTensor *x, int64_t nFft, int64_t hop, const aclTensor *window, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStft(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- foreach extensions (ops-nn / foreach): one elementwise op applied across every tensor in a list ----
// Unary / round: (x list, out list)
#define ACLNN_FE_UNARY(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_UNARY(aclnnForeachAbs) ACLNN_FE_UNARY(aclnnForeachAcos) ACLNN_FE_UNARY(aclnnForeachAsin) ACLNN_FE_UNARY(aclnnForeachAtan)
ACLNN_FE_UNARY(aclnnForeachCos) ACLNN_FE_UNARY(aclnnForeachCosh) ACLNN_FE_UNARY(aclnnForeachErf) ACLNN_FE_UNARY(aclnnForeachErfc)
ACLNN_FE_UNARY(aclnnForeachExp) ACLNN_FE_UNARY(aclnnForeachExpm1) ACLNN_FE_UNARY(aclnnForeachLog) ACLNN_FE_UNARY(aclnnForeachLog10)
ACLNN_FE_UNARY(aclnnForeachLog1p) ACLNN_FE_UNARY(aclnnForeachLog2) ACLNN_FE_UNARY(aclnnForeachNeg) ACLNN_FE_UNARY(aclnnForeachReciprocal)
ACLNN_FE_UNARY(aclnnForeachSigmoid) ACLNN_FE_UNARY(aclnnForeachSign) ACLNN_FE_UNARY(aclnnForeachSin) ACLNN_FE_UNARY(aclnnForeachSinh)
ACLNN_FE_UNARY(aclnnForeachSqrt) ACLNN_FE_UNARY(aclnnForeachTan) ACLNN_FE_UNARY(aclnnForeachTanh)
ACLNN_FE_UNARY(aclnnForeachRoundOffNumber) ACLNN_FE_UNARY(aclnnForeachRoundOffNumberV2)
ACLNN_FE_UNARY(aclnnForeachCopy)
// Single scalar applied to all tensors: (x list, scalar, out list)
#define ACLNN_FE_SCALAR(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclScalar *scalar, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_SCALAR(aclnnForeachAddScalar) ACLNN_FE_SCALAR(aclnnForeachAddScalarV2) ACLNN_FE_SCALAR(aclnnForeachSubScalar) ACLNN_FE_SCALAR(aclnnForeachSubScalarV2)
ACLNN_FE_SCALAR(aclnnForeachMulScalar) ACLNN_FE_SCALAR(aclnnForeachMulScalarV2) ACLNN_FE_SCALAR(aclnnForeachDivScalar) ACLNN_FE_SCALAR(aclnnForeachDivScalarV2)
ACLNN_FE_SCALAR(aclnnForeachMaximumScalar) ACLNN_FE_SCALAR(aclnnForeachMaximumScalarV2) ACLNN_FE_SCALAR(aclnnForeachMinimumScalar) ACLNN_FE_SCALAR(aclnnForeachMinimumScalarV2)
ACLNN_FE_SCALAR(aclnnForeachPowScalar) ACLNN_FE_SCALAR(aclnnForeachPowScalarV2)
// Per-tensor scalar list (scalars = 1-D fp32 tensor): (x list, scalars, out list)
#define ACLNN_FE_SCALARLIST(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensor *scalars, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_SCALARLIST(aclnnForeachAddScalarList) ACLNN_FE_SCALARLIST(aclnnForeachSubScalarList) ACLNN_FE_SCALARLIST(aclnnForeachMulScalarList) ACLNN_FE_SCALARLIST(aclnnForeachDivScalarList)
ACLNN_FE_SCALARLIST(aclnnForeachMaximumScalarList) ACLNN_FE_SCALARLIST(aclnnForeachMinimumScalarList) ACLNN_FE_SCALARLIST(aclnnForeachPowScalarList)
// List ∘ list: (x list, y list, out list)
#define ACLNN_FE_LIST(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_LIST(aclnnForeachAddList) ACLNN_FE_LIST(aclnnForeachSubList) ACLNN_FE_LIST(aclnnForeachMulList) ACLNN_FE_LIST(aclnnForeachDivList)
ACLNN_FE_LIST(aclnnForeachMaximumList) ACLNN_FE_LIST(aclnnForeachMinimumList) ACLNN_FE_LIST(aclnnForeachPowList)
// List ∘ list with scalar alpha: (x list, y list, alpha, out list)
#define ACLNN_FE_LISTV2(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *y, const aclScalar *alpha, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_LISTV2(aclnnForeachAddListV2) ACLNN_FE_LISTV2(aclnnForeachSubListV2)
// addcmul / addcdiv with one scalar: (x list, t1 list, t2 list, scalar, out list)
#define ACLNN_FE_ADDC_SCALAR(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2, const aclScalar *scalar, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_ADDC_SCALAR(aclnnForeachAddcmulScalar) ACLNN_FE_ADDC_SCALAR(aclnnForeachAddcmulScalarV2) ACLNN_FE_ADDC_SCALAR(aclnnForeachAddcdivScalar) ACLNN_FE_ADDC_SCALAR(aclnnForeachAddcdivScalarV2)
// addcmul / addcdiv with per-tensor scalars (scalars = 1-D fp32 tensor): (x list, t1 list, t2 list, scalars, out list)
#define ACLNN_FE_ADDC_LIST(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2, const aclTensor *scalars, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FE_ADDC_LIST(aclnnForeachAddcmulScalarList) ACLNN_FE_ADDC_LIST(aclnnForeachAddcdivScalarList) ACLNN_FE_ADDC_LIST(aclnnForeachAddcmulList) ACLNN_FE_ADDC_LIST(aclnnForeachAddcdivList)
// pow(scalar, tensor): (scalar, x list, out list)
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachPowScalarAndTensorGetWorkspaceSize(const aclScalar *scalar, const aclTensorList *x, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachPowScalarAndTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// lerp: scalar weight (x list, end list, weight, out list) / per-element weight list (x list, end list, weight list, out list)
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachLerpScalarGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclScalar *weight, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachLerpScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachLerpListGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclTensorList *weight, const aclTensorList *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachLerpList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// zero (in place): (x list)
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(const aclTensorList *x, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachZeroInplace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// norm: per-tensor L_p norm written to a 1-D fp32 out tensor (x list, p, out)
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachNormGetWorkspaceSize(const aclTensorList *x, const aclScalar *p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// AMP non-finite check + unscale in place: (x list, foundInf, invScale)
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *x, aclTensor *foundInf, const aclTensor *invScale, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnForeachNonFiniteCheckAndUnscale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- in-place elementwise / activation (aclnnInplace*): result written back into selfRef ----
#define ACLNN_IP_UN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_UN(aclnnInplaceAcos) ACLNN_IP_UN(aclnnInplaceCos) ACLNN_IP_UN(aclnnInplaceErf) ACLNN_IP_UN(aclnnInplaceExp)
ACLNN_IP_UN(aclnnInplaceFloor) ACLNN_IP_UN(aclnnInplaceLog) ACLNN_IP_UN(aclnnInplaceNeg) ACLNN_IP_UN(aclnnInplaceReciprocal)
ACLNN_IP_UN(aclnnInplaceRsqrt) ACLNN_IP_UN(aclnnInplaceSin) ACLNN_IP_UN(aclnnInplaceErfinv) ACLNN_IP_UN(aclnnInplaceHardsigmoid)
ACLNN_IP_UN(aclnnInplaceHardswish) ACLNN_IP_UN(aclnnInplaceMish) ACLNN_IP_UN(aclnnInplaceRelu) ACLNN_IP_UN(aclnnInplaceSelu)
ACLNN_IP_UN(aclnnInplaceSigmoid)
#define ACLNN_IP_SCALAR(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *scalar, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_SCALAR(aclnnInplaceElu) ACLNN_IP_SCALAR(aclnnInplaceCelu) ACLNN_IP_SCALAR(aclnnInplaceLeakyRelu) ACLNN_IP_SCALAR(aclnnInplaceClampMax)
#define ACLNN_IP_SCALAR2(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *a, const aclScalar *b, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_SCALAR2(aclnnInplaceHardtanh) ACLNN_IP_SCALAR2(aclnnInplaceThreshold)
#define ACLNN_IP_TERN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *t1, const aclTensor *t2, const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_TERN(aclnnInplaceAddcmul) ACLNN_IP_TERN(aclnnInplaceAddcdiv)
#define ACLNN_IP_BIN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_BIN(aclnnInplaceClampMinTensor) ACLNN_IP_BIN(aclnnInplaceClampMaxTensor) ACLNN_IP_BIN(aclnnInplaceBitwiseAndTensor)
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceLerpsGetWorkspaceSize(aclTensor *selfRef, const aclTensor *end, const aclScalar *weight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceLerps(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ops-math / math in-place variants
ACLNN_IP_UN(aclnnInplaceLog2) ACLNN_IP_UN(aclnnInplaceLog10) ACLNN_IP_UN(aclnnInplaceLog1p) ACLNN_IP_UN(aclnnInplaceExp2)
ACLNN_IP_UN(aclnnInplaceExpm1) ACLNN_IP_UN(aclnnInplaceErfc) ACLNN_IP_UN(aclnnInplaceFrac) ACLNN_IP_UN(aclnnInplaceSinh)
ACLNN_IP_UN(aclnnInplaceCosh) ACLNN_IP_UN(aclnnInplaceTan) ACLNN_IP_UN(aclnnInplaceTanh) ACLNN_IP_UN(aclnnInplaceSqrt)
ACLNN_IP_UN(aclnnInplaceCeil) ACLNN_IP_UN(aclnnInplaceRound) ACLNN_IP_UN(aclnnInplaceTrunc) ACLNN_IP_UN(aclnnInplaceAsin)
ACLNN_IP_UN(aclnnInplaceAtan) ACLNN_IP_UN(aclnnInplaceZero)
ACLNN_IP_BIN(aclnnInplaceMul) ACLNN_IP_BIN(aclnnInplaceDiv) ACLNN_IP_BIN(aclnnInplacePowTensorTensor) ACLNN_IP_BIN(aclnnInplaceFmodTensor)
ACLNN_IP_BIN(aclnnInplaceRemainderTensorTensor) ACLNN_IP_BIN(aclnnInplaceAtan2) ACLNN_IP_BIN(aclnnInplaceXLogYTensor) ACLNN_IP_BIN(aclnnInplaceBitwiseOrTensor)
ACLNN_IP_SCALAR(aclnnInplaceMuls) ACLNN_IP_SCALAR(aclnnInplaceDivs) ACLNN_IP_SCALAR(aclnnInplacePowTensorScalar)
#define ACLNN_IP_BIN_ALPHA(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_BIN_ALPHA(aclnnInplaceAdd) ACLNN_IP_BIN_ALPHA(aclnnInplaceSub)
#define ACLNN_IP_SCALAR_ALPHA(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_IP_SCALAR_ALPHA(aclnnInplaceAdds) ACLNN_IP_SCALAR_ALPHA(aclnnInplaceSubs)
ACLNN_IP_UN(aclnnInplaceAcosh) ACLNN_IP_UN(aclnnInplaceAsinh) ACLNN_IP_UN(aclnnInplaceAtanh) ACLNN_IP_UN(aclnnInplaceSinc)
// more in-place elementwise (base ops in elementwise.cu)
ACLNN_IP_BIN_ALPHA(aclnnInplaceAddV3)
ACLNN_IP_BIN(aclnnInplaceBitwiseXorTensor) ACLNN_IP_BIN(aclnnInplaceFloorDivide)
ACLNN_IP_SCALAR(aclnnInplaceBitwiseAndScalar) ACLNN_IP_SCALAR(aclnnInplaceBitwiseOrScalar) ACLNN_IP_SCALAR(aclnnInplaceBitwiseXorScalar) ACLNN_IP_SCALAR(aclnnInplaceFloorDivides)
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRoundDecimalsGetWorkspaceSize(aclTensor *selfRef, int64_t decimals, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRoundDecimals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// in-place comparison / logical (selfRef must be BOOL)
ACLNN_IP_BIN(aclnnInplaceGtTensor) ACLNN_IP_BIN(aclnnInplaceLtTensor) ACLNN_IP_BIN(aclnnInplaceEqTensor)
ACLNN_IP_BIN(aclnnInplaceNeTensor) ACLNN_IP_BIN(aclnnInplaceGeTensor) ACLNN_IP_BIN(aclnnInplaceLeTensor)
ACLNN_IP_BIN(aclnnInplaceLogicalAnd) ACLNN_IP_BIN(aclnnInplaceLogicalOr)
ACLNN_IP_SCALAR(aclnnInplaceGtScalar) ACLNN_IP_SCALAR(aclnnInplaceLtScalar) ACLNN_IP_SCALAR(aclnnInplaceEqScalar)
ACLNN_IP_SCALAR(aclnnInplaceNeScalar) ACLNN_IP_SCALAR(aclnnInplaceGeScalar) ACLNN_IP_SCALAR(aclnnInplaceLeScalar)
ACLNN_IP_UN(aclnnInplaceLogicalNot)
// fill family
ACLNN_IP_UN(aclnnInplaceOne)
ACLNN_IP_SCALAR(aclnnInplaceFillScalar)
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(aclTensor *selfRef, const aclScalar *value, bool wrap, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFillDiagonal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- additional elementary math (unary acosh/asinh/atanh/sinc/digamma, binary logaddexp2) ----
#define ACLNN_MATH_UN(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_MATH_UN(aclnnAcosh) ACLNN_MATH_UN(aclnnAsinh) ACLNN_MATH_UN(aclnnAtanh) ACLNN_MATH_UN(aclnnSinc) ACLNN_MATH_UN(aclnnDigamma)
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogAddExp2GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogAddExp2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- additional activations (FastGelu/SquaredRelu unary, Swish scalar-beta, GeluV2 approximate flag) ----
ACLNN_MATH_UN(aclnnFastGelu) ACLNN_MATH_UN(aclnnSquaredRelu)
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwishGetWorkspaceSize(const aclTensor *self, const aclScalar *beta, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluV2GetWorkspaceSize(const aclTensor *self, int64_t approximate, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- more arithmetic (floor-division, real-division aliases, reverse-sub, round-to-decimals) ----
#define ACLNN_BIN2(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_BIN2(aclnnFloorDivide) ACLNN_BIN2(aclnnFloorDiv) ACLNN_BIN2(aclnnRealDiv) ACLNN_BIN2(aclnnDivV3)
#define ACLNN_SCL2(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_SCL2(aclnnFloorDivides) ACLNN_SCL2(aclnnRsubs)
ACL_FUNC_VISIBILITY aclnnStatus aclnnRsubGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRsub(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoundDecimalsGetWorkspaceSize(const aclTensor *self, int64_t decimals, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoundDecimals(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- tensor generators (Eye / Linspace / LogSpace / Range) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnEyeGetWorkspaceSize(aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEye(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinspace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSpaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, double base, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSpace(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRange(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- bitwise extensions (xor tensor, scalar and/or/xor) ----
ACLNN_BIN2(aclnnBitwiseXorTensor)
ACLNN_SCL2(aclnnBitwiseAndScalar) ACLNN_SCL2(aclnnBitwiseOrScalar) ACLNN_SCL2(aclnnBitwiseXorScalar)

// ---- nan_to_num + in-place tril/triu ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanToNumGetWorkspaceSize(const aclTensor *self, float nan, float posinf, float neginf, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanToNum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNanToNumGetWorkspaceSize(aclTensor *selfRef, float nan, float posinf, float neginf, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceNanToNum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceTrilGetWorkspaceSize(aclTensor *selfRef, int64_t diagonal, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceTril(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceTriuGetWorkspaceSize(aclTensor *selfRef, int64_t diagonal, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceTriu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// clamp (scalar bounds) + tensor-bound clamp
ACL_FUNC_VISIBILITY aclnnStatus aclnnClampGetWorkspaceSize(const aclTensor *self, const aclScalar *min, const aclScalar *max, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClamp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_BIN2(aclnnClampMaxTensor) ACLNN_BIN2(aclnnClampMinTensor)

// ---- random / distribution extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnRandIntGetWorkspaceSize(int64_t low, int64_t high, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRandInt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExponentialGetWorkspaceSize(aclTensor *out, double lambda, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExponential(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeometricGetWorkspaceSize(aclTensor *out, double p, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeometric(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCauchyGetWorkspaceSize(aclTensor *out, double median, double sigma, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCauchy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogNormalGetWorkspaceSize(aclTensor *out, double mean, double std, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogNormal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPoissonGetWorkspaceSize(aclTensor *out, double lambda, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPoisson(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultinomialGetWorkspaceSize(const aclTensor *probs, int64_t numSamples, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultinomial(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Additional reduction ops: All/Any (bool output), Var/Std/LogSumExp, Norm(p)
#define ACLNN_RED(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_RED(aclnnAll) ACLNN_RED(aclnnAny) ACLNN_RED(aclnnVar) ACLNN_RED(aclnnStd) ACLNN_RED(aclnnLogSumExp)
ACL_FUNC_VISIBILITY aclnnStatus aclnnNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor *self, double p, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgVectorNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRenormGetWorkspaceSize(aclTensor *selfRef, double p, int64_t dim, double maxnorm, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRenorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- reduce & scan extensions ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnNansumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNansum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanmeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanmean(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCountNonzeroGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCountNonzero(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMedianGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMedian(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKthvalueGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKthvalue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantileGetWorkspaceSize(const aclTensor *self, double q, int64_t dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantile(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnModeGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRenormGetWorkspaceSize(const aclTensor *self, double p, int64_t dim, double maxnorm, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRenorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- additional reduction variants ----
#define ACLNN_FULLRED(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FULLRED(aclnnMax) ACLNN_FULLRED(aclnnMaxV2) ACLNN_FULLRED(aclnnMin)
ACL_FUNC_VISIBILITY aclnnStatus aclnnSumGetWorkspaceSize(const aclTensor *self, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMeanV2GetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMeanV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnProdDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnProdDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceNansumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceNansum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceLogSumGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReduceLogSum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmaxAllGetWorkspaceSize(const aclTensor *self, aclTensor *outMin, aclTensor *outMax, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmaxAll(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *outMin, aclTensor *outMax, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAminmaxDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMinDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMinDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnVarMeanGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, bool keepDim, aclTensor *varOut, aclTensor *meanOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnVarMean(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *stdOut, aclTensor *meanOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStdMeanCorrection(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCummaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCummax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumminGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCummin(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogcumsumexpGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogcumsumexp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Additional pooling ops: AdaptiveAvgPool2d, MaxPool3d/AvgPool3d
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_POOL3D(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_POOL3D(aclnnMaxPool3d) ACLNN_POOL3D(aclnnAvgPool3d)

// ---- comparison / logical / predicate (bool output) ----
#define ACLNN_CMP(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *other, \
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_PRED(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, aclTensor *out, \
    uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACLNN_CMP(aclnnGtTensor)
ACLNN_CMP(aclnnLtTensor)
ACLNN_CMP(aclnnEqTensor)
ACLNN_CMP(aclnnNeTensor)
ACLNN_CMP(aclnnGeTensor)
ACLNN_CMP(aclnnLeTensor)
ACLNN_CMP(aclnnLogicalAnd)
ACLNN_CMP(aclnnLogicalOr)
// tensor ∘ scalar comparisons → bool out
#define ACLNN_CMP_S(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclScalar *other, \
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_CMP_S(aclnnGtScalar) ACLNN_CMP_S(aclnnLtScalar) ACLNN_CMP_S(aclnnEqScalar)
ACLNN_CMP_S(aclnnNeScalar) ACLNN_CMP_S(aclnnGeScalar) ACLNN_CMP_S(aclnnLeScalar)
ACLNN_PRED(aclnnLogicalNot)
ACLNN_PRED(aclnnIsNan)
ACLNN_PRED(aclnnIsFinite)
ACLNN_PRED(aclnnIsInf)
ACLNN_PRED(aclnnIsPosInf)
ACLNN_PRED(aclnnIsNegInf)
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedFillScalarGetWorkspaceSize(const aclTensor *self, const aclTensor *mask,
    const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedFillScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- sort / scan (along dimension) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumsumGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumprodGetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool stable,
    aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSort(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor *self, int64_t k, int64_t dim, bool largest, bool sorted,
    aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopk(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnSWhereGetWorkspaceSize(const aclTensor *condition, const aclTensor *self,
    const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSWhere(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Gemm: out = alpha·op(A)·op(B) + beta·C, op = transpose per transA/transB (BLAS-3)
ACL_FUNC_VISIBILITY aclnnStatus aclnnGemmGetWorkspaceSize(const aclTensor *a, const aclTensor *b, const aclTensor *c,
    float alpha, float beta, int64_t transA, int64_t transB, aclTensor *out, int8_t cubeMathType,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGemm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// NZ (fractal-layout) matmul family — logical-equivalence (weight treated row-major; forwards to ND cores)
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatMulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatMulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmmWeightNzGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmmWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantBatchMatmulNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantBatchMatmulNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// BatchMatMul: [B,M,K]@[B,K,N]->[B,M,N] (strided batched, fp32/fp16)
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- BLAS extras ----
#define ACLNN_MM2(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *self, const aclTensor *mat2, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_MM2(aclnnMm) ACLNN_MM2(aclnnBmm) ACLNN_MM2(aclnnMv) ACLNN_MM2(aclnnDot) ACLNN_MM2(aclnnVdot) ACLNN_MM2(aclnnInner) ACLNN_MM2(aclnnOuter) ACLNN_MM2(aclnnKron)
#undef ACLNN_MM2
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmvGetWorkspaceSize(const aclTensor *y, const aclTensor *mat, const aclTensor *vec, double beta, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBaddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBaddbmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddbmmGetWorkspaceSize(const aclTensor *C, const aclTensor *A, const aclTensor *B, double beta, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddbmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTensordotGetWorkspaceSize(const aclTensor *A, const aclTensor *B, int64_t naxes, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTensordot(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEinsumGetWorkspaceSize(const char *equation, const aclTensor *A, const aclTensor *B, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEinsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- linear algebra (linalg, cuSOLVER-backed) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnInverseGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDetGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCholeskyGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCholesky(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCross(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, aclTensor *X, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSlogdetGetWorkspaceSize(const aclTensor *A, aclTensor *sign, aclTensor *logabsdet, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSlogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQrGetWorkspaceSize(const aclTensor *A, aclTensor *Q, aclTensor *R, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSvdGetWorkspaceSize(const aclTensor *A, aclTensor *U, aclTensor *S, aclTensor *VT, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSvd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEighGetWorkspaceSize(const aclTensor *A, aclTensor *W, aclTensor *V, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEigh(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixRankGetWorkspaceSize(const aclTensor *A, double tol, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixRank(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstsqGetWorkspaceSize(const aclTensor *A, const aclTensor *B, aclTensor *X, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstsq(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixPowerGetWorkspaceSize(const aclTensor *A, int64_t p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixPower(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPinverseGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPinverse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLuGetWorkspaceSize(const aclTensor *A, aclTensor *LU, aclTensor *pivots, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLuSolveGetWorkspaceSize(const aclTensor *LU, const aclTensor *pivots, const aclTensor *B, aclTensor *X, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLuSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTriangularSolveGetWorkspaceSize(const aclTensor *A, const aclTensor *B, bool upper, bool transpose, bool unitriangular, aclTensor *X, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTriangularSolve(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixExpGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatrixExp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MatmulBias: (transA?A^T:A)@(transB?B^T:B)+bias, act: 0=none/1=ReLU/2=GeLU. bias: nullptr/[N]/[M,N].
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulBiasGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
    const aclTensor *bias, bool transA, bool transB, int64_t act, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulBias(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
    const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation,
    bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *output, int8_t cubeMathType,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Convolution backward + pooling (fp32/fp16)
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackwardDataGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *weight,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
    aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackwardData(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackwardWeightGetWorkspaceSize(const aclTensor *input, const aclTensor *gradOutput,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
    aclTensor *gradWeight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackwardWeight(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
    const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel,
    const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// conv3d (NCDHW)
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution3dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
    aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Transposed convolution ConvTranspose2d
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionTranspose2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
    const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
    aclTensor *output, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionTranspose2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Pooling backward
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
    const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
    const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MXFP8 block-scaled matmul (PoC simplified signature): self[M,K]/other[K,N] are fp8 e4m3/e5m2;
//   selfScale[M,K/32], otherScale[K/32,N] are E8M0 (ACL_FLOAT8_E8M0), one scale 2^(e-127) per 32-element block along K.
//   Implemented by per-block dequantize to fp16 then GEMM (functional/tolerance path, not hardware MX tensor-core). out fp16/fp32.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp8GetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
    const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp8(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MXFP4: self/other are fp4 e2m1 (2 per byte packed), selfScale[M,K/32]/otherScale[K/32,N] are E8M0;
//   per-block dequantize to fp16 then GEMM (functional path). out fp16/fp32.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp4GetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
    const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// MXFP8 hardware tensor-core path: cuBLASLt native VEC32_UE8M0 + swizzled scale. Same signature as aclnnMatmulMxFp8.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp8HwGetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
    const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp8Hw(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// MXFP4 hardware tensor-core path: cuBLASLt native fp4(E2M1) + VEC32_UE8M0; automatically falls back to functional path for cc<10. Same signature as aclnnMatmulMxFp4.
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp4HwGetWorkspaceSize(const aclTensor *self, const aclTensor *selfScale,
    const aclTensor *other, const aclTensor *otherScale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulMxFp4Hw(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Quantized matmul: WeightQuantBatchMatmul (W8A16/W4A16), QuantMatmul (W8A8)
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantBatchMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
    const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnWeightQuantBatchMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Quant matmul version variants (same simplified contract; version bumps forward to the shared core)
#define ACLNN_QMM_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, \
        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_WQBMM_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, \
        const aclTensor *antiquantScale, const aclTensor *antiquantOffset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_QMM_DECL(aclnnQuantMatmulV2)
ACLNN_QMM_DECL(aclnnQuantMatmulV3)
ACLNN_QMM_DECL(aclnnQuantMatmulV4)
ACLNN_QMM_DECL(aclnnQuantMatmulV5)
ACLNN_QMM_DECL(aclnnQuantMatmulDequant)
ACLNN_WQBMM_DECL(aclnnWeightQuantBatchMatmulV2)
ACLNN_WQBMM_DECL(aclnnWeightQuantBatchMatmulV3)
#undef ACLNN_QMM_DECL
#undef ACLNN_WQBMM_DECL

// QuantBatchMatmulInplaceAdd: out += dequant(x_int8 @ weight_int8, scale)
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantBatchMatmulInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Explicit quantize/dequantize
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantizeGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
// DynamicQuant version variants (per-token symmetric int8 quant; forward to the shared core)
#define ACLNN_DQ_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_DQ_DECL(aclnnDynamicQuantV2)
ACLNN_DQ_DECL(aclnnDynamicQuantV3)
ACLNN_DQ_DECL(aclnnDynamicQuantV4)
#undef ACLNN_DQ_DECL
// AdaLayerNorm: y = LayerNorm(x)·(1+scale)+shift (per-row scale/shift)
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift,
    double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormV2GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift,
    double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *scale,
    double eps, aclTensor *gradX, aclTensor *gradScale, aclTensor *gradShift, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- quantization extensions ----
#define ACLNN_QDQ(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_QDQ(aclnnAscendQuant) ACLNN_QDQ(aclnnAscendDequant) ACLNN_QDQ(aclnnAscendAntiQuant) ACLNN_QDQ(aclnnDequantBias)
#undef ACLNN_QDQ
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuantGetWorkspaceSize(const aclTensor *x, double scale, double zeroPoint, int64_t qmin, int64_t qmax, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- sequence / RNN ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstmGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGruGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, aclTensor *y, aclTensor *hN, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGru(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- linear attention / SSM (Qwen3.5-class hybrid backbones) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t activation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSelectiveScanGetWorkspaceSize(const aclTensor *u, const aclTensor *delta, const aclTensor *A, const aclTensor *B, const aclTensor *C, const aclTensor *D, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSelectiveScan(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- MoE routing (mixture-of-experts dispatch/combine) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeGatingTopKSoftmaxGetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeGatingTopKSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeComputeExpertTokensGetWorkspaceSize(const aclTensor *indices, int64_t numExperts, aclTensor *tokensPerExpert, aclTensor *offsets, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeComputeExpertTokens(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenPermuteGetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenPermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenUnpermuteGetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenUnpermute(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// MoE InitRouting (expand+group by top-K expert) / FinalizeRouting (weighted combine) + version/gating/grouped-matmul variants
#define ACLNN_MOE_INIT_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, \
        aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_MOE_FINAL_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *expandedY, const aclTensor *expandedRowIdx, \
        const aclTensor *scales, int64_t k, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_MOE_GATE_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *logits, int64_t k, aclTensor *weights, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_GMM_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, \
        aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_MOE_INIT_DECL(aclnnMoeInitRouting)
ACLNN_MOE_INIT_DECL(aclnnMoeInitRoutingV2)
ACLNN_MOE_INIT_DECL(aclnnMoeInitRoutingV3)
ACLNN_MOE_FINAL_DECL(aclnnMoeFinalizeRouting)
ACLNN_MOE_FINAL_DECL(aclnnMoeFinalizeRoutingV2)
ACLNN_MOE_FINAL_DECL(aclnnMoeFinalizeRoutingV3)
ACLNN_MOE_GATE_DECL(aclnnMoeGatingTopKSoftmaxV2)
ACLNN_MOE_GATE_DECL(aclnnMoeGatingTopK)
ACLNN_MOE_GATE_DECL(aclnnMoeFusedTopk)
ACLNN_GMM_DECL(aclnnGroupedMatmulV2)
ACLNN_GMM_DECL(aclnnGroupedMatmulV3)
ACLNN_GMM_DECL(aclnnGroupedMatmulV4)
ACLNN_GMM_DECL(aclnnGroupedMatmulV5)
#undef ACLNN_MOE_INIT_DECL
#undef ACLNN_MOE_FINAL_DECL
#undef ACLNN_MOE_GATE_DECL
#undef ACLNN_GMM_DECL
// MoE routing gradients (inverse mappings of the forward routing ops)
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeInitRoutingV2GradGetWorkspaceSize(const aclTensor *gradExpandedX, const aclTensor *expandedRowIdx,
    int64_t k, aclTensor *gradX, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeInitRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeFinalizeRoutingV2GradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *expandedY,
    const aclTensor *expandedRowIdx, const aclTensor *scales, int64_t k, aclTensor *gradExpandedY, aclTensor *gradScales,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeFinalizeRoutingV2Grad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenPermuteGradGetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx,
    aclTensor *gradX, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenPermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenUnpermuteGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx,
    const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeTokenUnpermuteGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmbGradGetWorkspaceSize(const aclTensor *gradQ, const aclTensor *gradK, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *gradQOut, aclTensor *gradKOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmbGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedInferAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedInferAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiLatentAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *cKV, const aclTensor *wUK, const aclTensor *wUV, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiLatentAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// ---- FFT / signal (interleaved-complex storage; inverse transforms are 1/n normalized) ----
#define ACLNN_FFT(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, int64_t n, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FFT(aclnnFft) ACLNN_FFT(aclnnIfft) ACLNN_FFT(aclnnRfft) ACLNN_FFT(aclnnIrfft)
#undef ACLNN_FFT

// GroupedMatmul (MoE): x[M,K] grouped by groupList (tokens per expert) @ weight[E,K,N] -> out[M,N]
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight,
    const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// GroupedMatmul weight TensorList variant: weights is a list of per-expert [K,N] tensors
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulWeightListGetWorkspaceSize(const aclTensor *x, const aclTensorList *weights,
    const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulWeightList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Softmax along dim (PoC: last dimension only); semantics match CANN aclnnSoftmax.
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// LogSoftmax (any dimension)
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Norm extensions: RMSNorm (last dimension), GroupNorm (NCHW), BatchNorm (inference)
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, double eps,
    aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
    int64_t numGroups, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Norm-family variants: GemmaRmsNorm ((1+gamma) scale), GroupNormSilu/Swish (fused activation), FastLayerNorm, LayerNormWithImplMode
ACL_FUNC_VISIBILITY aclnnStatus aclnnGemmaRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps,
    aclTensor *yOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGemmaRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSiluGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
    int64_t group, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSilu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSiluV2GetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
    int64_t group, double eps, bool activateSilu, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSiluV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSwishGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
    int64_t group, double eps, double swishScale, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSwish(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps,
    aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormWithImplModeGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
    const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
    int64_t implMode, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormWithImplMode(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta,
    const aclTensor *mean, const aclTensor *var, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Fused ops: AddRmsNorm/AddLayerNorm (outputs y + residual sum), SwiGlu/GeGlu (split last dim in half)
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
    double eps, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// AddRmsNorm fused variants: Cast (second-dtype copy of y) and Inplace (write y back into x)
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNormCastGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
    double eps, aclTensor *y, aclTensor *yCast, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNormCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddRmsNormGetWorkspaceSize(aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
    double eps, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddRmsNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// GroupedMatmulAdd: per-expert GEMM + residual y add
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y,
    const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulAddV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *y,
    const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulAddV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNormGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma,
    const aclTensor *beta, double eps, aclTensor *y, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGlu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeGluGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeGlu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// GLU gradients + GeGluV3 forward. gradOut[...,D], self[...,2D] -> gradIn[...,2D]
#define ACLNN_GLUGRAD_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_GLUGRAD_DECL(aclnnSwiGluGrad)
ACLNN_GLUGRAD_DECL(aclnnGeGluBackward)
ACLNN_GLUGRAD_DECL(aclnnGeGluV3Backward)
#undef ACLNN_GLUGRAD_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeGluV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// SwiGluQuant (silu-GLU + per-token int8 quant) + ClippedSwiglu (clamp-gated)
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGluQuantV2GetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwiGluQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClippedSwigluGetWorkspaceSize(const aclTensor *x, double clipValue, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnClippedSwiglu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// Fused RmsNorm + quant: AddRmsNormDynamicQuant (per-token int8) and AddRmsNormQuant/RmsNormQuant (static scale+offset)
#define ACLNN_ADDRMSDQ_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, \
        double eps, aclTensor *yQuant, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_ADDRMSQ_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, \
        double scale, double offset, double eps, aclTensor *yQuant, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_ADDRMSDQ_DECL(aclnnAddRmsNormDynamicQuant)
ACLNN_ADDRMSDQ_DECL(aclnnAddRmsNormDynamicQuantV2)
ACLNN_ADDRMSQ_DECL(aclnnAddRmsNormQuant)
ACLNN_ADDRMSQ_DECL(aclnnAddRmsNormQuantV2)
#undef ACLNN_ADDRMSDQ_DECL
#undef ACLNN_ADDRMSQ_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double scale, double offset,
    double eps, aclTensor *yQuant, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// DeepNorm (aligned with explorer): y = LayerNorm(alpha*x + gx)*gamma + beta
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeepNormGetWorkspaceSize(const aclTensor *x, const aclTensor *gx, const aclTensor *gamma,
    const aclTensor *beta, double alpha, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeepNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Norm backward + BatchNorm training
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
    double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *gradX, aclTensor *gradGamma, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *gradX, aclTensor *gradResidual, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeQuantBatchMatMulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeQuantBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma,
    const aclIntArray *normalizedShape, double eps, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormTrainingGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta,
    aclTensor *runningMean, aclTensor *runningVar, double momentum, double eps, aclTensor *out, aclTensor *savedMean, aclTensor *savedInvStd,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormTraining(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// High-performance attention (cuBLASLt batched GEMM, standard MHA, fp16/fp32)
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreHighPerf(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// RoPE: ApplyRotaryPosEmb, mode 0=half-split (LLaMA) / 1=interleaved (GPT-J)
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmbGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos,
    const aclTensor *sin, int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmb(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Inference attention: Prompt/Incre routed to contiguous SDPA; PagedAttention with paged KV + blockTable
ACL_FUNC_VISIBILITY aclnnStatus aclnnPromptFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPromptFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIncreFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIncreFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPagedAttentionGetWorkspaceSize(const aclTensor *query, const aclTensor *kCache, const aclTensor *vCache,
    const aclTensor *blockTable, const aclTensor *contextLens, double scaleValue, int64_t numHeads, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPagedAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Attention backward (fp32, standard MHA)
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreBackwardGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v,
    const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal,
    aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// LayerNorm: normalizes over the last normalizedShape.size() dimensions; weight/bias may be null; meanOut/rstdOut may be null.
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
    const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
    uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// FlashAttentionScore (PoC simplified signature): BNSD-layout scaled dot-product attention forward, outputs attentionOut only.
//   q/k/v: [B,N,S,D] (S for k/v may be Skv); attenMask may be null (positions !=0 are blocked); causal=bottom-right-aligned lower triangle;
//   scaleValue is typically 1/sqrt(D). The true CANN aclnnFlashAttentionScore also outputs softmaxMax/Sum etc. for training; omitted here.
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGetWorkspaceSize(const aclTensor *query, const aclTensor *key,
    const aclTensor *value, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal,
    aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

// Attention version variants (V2–V5): same simplified contract as the V1 cores; version bumps forward to the shared kernel.
#define ACLNN_FA_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define ACLNN_IFA_DECL(NAME) \
    ACL_FUNC_VISIBILITY aclnnStatus NAME##GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value, \
        const aclTensor *attenMask, double scaleValue, int64_t headNum, aclTensor *attentionOut, uint64_t *workspaceSize, aclOpExecutor **executor); \
    ACL_FUNC_VISIBILITY aclnnStatus NAME(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACLNN_FA_DECL(aclnnFlashAttentionScoreV2)
ACLNN_FA_DECL(aclnnFlashAttentionScoreV3)
ACLNN_FA_DECL(aclnnFlashAttentionScoreV4)
ACLNN_IFA_DECL(aclnnIncreFlashAttentionV2)
ACLNN_IFA_DECL(aclnnIncreFlashAttentionV3)
ACLNN_IFA_DECL(aclnnIncreFlashAttentionV4)
ACLNN_FA_DECL(aclnnPromptFlashAttentionV2)
ACLNN_FA_DECL(aclnnPromptFlashAttentionV3)
ACLNN_FA_DECL(aclnnFusedInferAttentionScoreV2)
ACLNN_FA_DECL(aclnnFusedInferAttentionScoreV3)
ACLNN_FA_DECL(aclnnFusedInferAttentionScoreV4)
ACLNN_FA_DECL(aclnnFusedInferAttentionScoreV5)
#undef ACLNN_FA_DECL
#undef ACLNN_IFA_DECL

ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamWV2GetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamWV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmbV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *qOut, aclTensor *kOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyRotaryPosEmbV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor *self, int64_t dim, aclDataType dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCumsumV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutV3GetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonzeroV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonzeroV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignV2GetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSliceV2GetWorkspaceSize(const aclTensor *self, int64_t dim, int64_t start, int64_t end, int64_t step, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSliceV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dBackwardV2GetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dBackwardV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest2dV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddV3GetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastGeluV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastGeluV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);

ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceCumprodGetWorkspaceSize(aclTensor *self, int64_t dim, aclDataType dtype, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceCumprod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *self, const aclTensor *mask, const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedScatterGetWorkspaceSize(aclTensor *self, const aclTensor *mask, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBernoulliGetWorkspaceSize(aclTensor *selfRef, double p, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBernoulli(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceUniformGetWorkspaceSize(aclTensor *selfRef, double from, double to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceUniform(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddrGetWorkspaceSize(const aclTensor *self, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddrGetWorkspaceSize(aclTensor *selfRef, const aclTensor *vec1, const aclTensor *vec2, const aclScalar *beta, const aclScalar *alpha, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDivModsGetWorkspaceSize(const aclTensor *self, const aclScalar *other, int64_t roundMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceDivModsGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, int64_t roundMode, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceDivMods(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFmodScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFmodScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFmodScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRemainderTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRemainderTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRemainderTensorScalarGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRemainderTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYScalarOtherGetWorkspaceSize(const aclTensor *self, const aclScalar *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYScalarOther(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceXLogYScalarOtherGetWorkspaceSize(aclTensor *selfRef, const aclScalar *other, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceXLogYScalarOther(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYScalarSelfGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYScalarSelf(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddReluGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddReluGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddRelu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHistcGetWorkspaceSize(const aclTensor *self, int64_t bins, const aclScalar *min, const aclScalar *max, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnHistc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterValueGetWorkspaceSize(const aclTensor *self, int64_t dim, const aclTensor *index, const aclScalar *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatterValueGetWorkspaceSize(aclTensor *selfRef, int64_t dim, const aclTensor *index, const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceScatterValue(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDivModGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t roundMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDivMod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceDivModGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, int64_t roundMode, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceDivMod(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPutGetWorkspaceSize(const aclTensor *self, const aclTensor *index, const aclTensor *source, bool accumulate, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplacePutGetWorkspaceSize(aclTensor *selfRef, const aclTensor *index, const aclTensor *source, bool accumulate, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplacePut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRReluWithNoiseGetWorkspaceSize(const aclTensor *self, aclTensor *noise, double lower, double upper, bool training, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRReluWithNoise(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRReluWithNoiseGetWorkspaceSize(aclTensor *selfRef, aclTensor *noise, double lower, double upper, bool training, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRReluWithNoise(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuantPerTensorAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *zeroPoint, int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuantPerTensorAffineCachemask(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuantPerChannelAffineCachemaskGetWorkspaceSize(const aclTensor *self, const aclTensor *scale, const aclTensor *zeroPoint, int64_t axis, int64_t quantMin, int64_t quantMax, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFakeQuantPerChannelAffineCachemask(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAscendQuantV3GetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAscendQuantV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *offset, int64_t groupSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuantV2GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNormQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddLayerNormQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSiluQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *gamma, const aclTensor *beta, int64_t group, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSiluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantSwigluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantSwigluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *dequantScale, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDequantSwigluQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV3GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV4GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV5GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionVarLenScoreV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantFlashAttentionScoreGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantFlashAttentionScore(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseFlashAttention(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV3GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV4GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV4(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlashAttentionUnpaddingScoreGradV5(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantFlashAttentionScoreGradGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantFlashAttentionScoreGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseFlashAttentionGradGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *dy, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *dq, aclTensor *dk, aclTensor *dv, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseFlashAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution_redlineGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *output, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution_redline(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2DGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact1dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact3dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGlobalAveragePoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGlobalAveragePool(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGlobalMaxPoolGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGlobalMaxPool(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleLinear1dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleLinear1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleTrilinear3dGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleTrilinear3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest3dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact1dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact2dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact3dBackwardGetWorkspaceSize(const aclTensor *gradOut, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearestExact3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleLinear1dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleLinear1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleTrilinear3dBackwardGetWorkspaceSize(const aclTensor *gradOut, bool alignCorners, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleTrilinear3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReflectionPad3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad1dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad1dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReplicationPad3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCircularPad2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCircularPad2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCircularPad3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCircularPad3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithIndicesBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool3dWithArgmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool3dWithArgmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithMaskBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *mask, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithMaskBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveMaxPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, int64_t H, int64_t W, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool3dGetWorkspaceSize(const aclTensor *self, const aclTensor *indices, const aclIntArray *outputSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxUnpool3d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithMaskGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool2dWithMask(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogitGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogit(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftsignGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftsign(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignbitGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignbit(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFmodTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFmodTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnXLogYTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSiluMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSiluMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGeluMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGcdGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGcd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeftShiftGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeftShift(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRightShiftGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRightShift(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsCloseGetWorkspaceSize(const aclTensor *self, const aclTensor *other, double rtol, double atol, bool equalNan, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsClose(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor *self, const aclScalar *scale, const aclScalar *bias, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, double scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedScale(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGerGetWorkspaceSize(const aclTensor *self, const aclTensor *vec2, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor *self, int64_t startDim, int64_t endDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlatten(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnShrinkGetWorkspaceSize(const aclTensor *self, const aclScalar *lambd, const aclScalar *bias, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnShrink(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRemainderScalarTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar *self, const aclTensor *exponent, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPowScalarTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor *self, const aclTensor *end, const aclScalar *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLerps(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxN(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMinNGetWorkspaceSize(const aclTensorList *tensors, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMinN(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCdistGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Inplace / *Out naming forwards + RNG-tensor variants (inplace_ext / inplace2_ext / random) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedFillTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *value, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclTensor *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillScalarGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mask, const aclScalar *value, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceMaskedFillScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBitwiseAndTensorOutGetWorkspaceSize(aclTensor *selfRef, const aclTensor *other, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBitwiseAndTensorOut(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceLerpGetWorkspaceSize(aclTensor *selfRef, const aclTensor *end, const aclScalar *weight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceLerp(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceCopyGetWorkspaceSize(aclTensor *selfRef, const aclTensor *src, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceCopy(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddbmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBaddbmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *batch1, const aclTensor *batch2, double beta, double alpha, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBaddbmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddmmGetWorkspaceSize(aclTensor *selfRef, const aclTensor *mat1, const aclTensor *mat2, double beta, double alpha, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAddmm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(aclTensor *out, const aclTensor *prob, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *prob, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceBernoulliTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRandomGetWorkspaceSize(aclTensor *selfRef, int64_t from, int64_t to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRandom(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceRandomTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceUniformTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- BatchNorm functional decomposition + norm backward + norm naming/quant fusions (norm6_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormStatsGetWorkspaceSize(const aclTensor *self, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormStats(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormElemtGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormElemt(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormReduceGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormReduce(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormReduceBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, aclTensor *sumDy, aclTensor *sumDyXmu, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormReduceBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormElemtBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *invstd, const aclTensor *weight, const aclTensor *sumDy, const aclTensor *sumDyXmu, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormElemtBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormGatherStatsWithCountsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchNormGatherStatsWithCounts(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSyncBatchNormGatherStatsGetWorkspaceSize(const aclTensor *meanAll, const aclTensor *invstdAll, const aclTensor *counts, double eps, aclTensor *mean, aclTensor *invstd, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSyncBatchNormGatherStats(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantizedBatchNormGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, const aclTensor *mean, const aclTensor *invstd, double scale, int64_t zeroPoint, double eps, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantizedBatchNorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastBatchNormBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gamma, const aclTensor *savedMean, const aclTensor *savedInvStd, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFastBatchNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma, int64_t numGroups, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSwishGradGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *self, const aclTensor *mean, const aclTensor *rstd, const aclTensor *gamma, int64_t numGroups, double swishScale, aclTensor *gradX, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupNormSwishGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeepNormGradGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *gx, const aclTensor *gamma, double alpha, double eps, aclTensor *gradX, aclTensor *gradGx, aclTensor *gradGamma, aclTensor *gradBeta, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeepNormGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNorm_noFunctionalGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNorm_noFunctional(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNorm_withFunctionalGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape, const aclTensor *weight, const aclTensor *bias, double eps, aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLayerNorm_withFunctional(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRmsNormDynamicMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNormDynamicMxQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *residual, const aclTensor *gamma, double eps, aclTensor *out, aclTensor *scaleOut, aclTensor *residualSum, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAddRmsNormDynamicMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaLayerNormQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Quant remainder (quant5_ext): MX/block/dual dynamic quant, FlatQuant, SwigluMxQuant, QuantScatter, TransQuantParam, fused ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicBlockMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedDynamicBlockQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedDynamicBlockQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedDynamicMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedDynamicMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicDualLevelMxQuantGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleL1, aclTensor *scaleL2, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicDualLevelMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuantWithDualAxisGetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, aclTensor *scaleOut, aclTensor *scaleOut2, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDynamicMxQuantWithDualAxis(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlatQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFlatQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwigluMxQuantGetWorkspaceSize(const aclTensor *x, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwigluMxQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantScatterGetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantScatter(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantScatterV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, const aclTensor *updates, double scale, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceQuantScatterV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParamGetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParam(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParamV2GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParamV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParamV3GetWorkspaceSize(const aclTensor *scale, const aclTensor *offset, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransQuantParamV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamWQuantGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyAdamWQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwinTransformerLnQkvQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta, double eps, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwinTransformerLnQkvQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Pool backward + MaxPool3dWithArgmax + RoI (pool4_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPoolGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool3dWithArgmaxGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool ceilMode, aclTensor *out, aclTensor *indices, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaxPool3dWithArgmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAvgPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool3dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdaptiveAvgPool3dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignRotatedGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignRotated(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignRotatedGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignRotatedGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignV2BackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, double spatialScale, int64_t samplingRatio, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiAlignV2Backward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiPoolingWithArgMaxGetWorkspaceSize(const aclTensor *self, const aclTensor *rois, double spatialScale, aclTensor *out, aclTensor *argmax, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiPoolingWithArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiPoolingGradWithArgMaxGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *rois, const aclTensor *argmax, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRoiPoolingGradWithArgMax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Conv extensions (conv2_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution_redlineGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *output, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolution_redline(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclIntArray *kernelSize, const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvDepthwise2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackward_redlineGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight, const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvolutionBackward_redline(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbcGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, int64_t pad, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbc(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbcBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbcBackward_redlineGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvTbcBackward_redline(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2col(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIm2colBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnfoldGradGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation, const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnfoldGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *offset, const aclTensor *mask, const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDeformableConv2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, double scale, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantConvolution(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, double scale, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantConvolutionWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateConvolutionWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateConvolutionWeightSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransConvolutionWeight(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnConvertWeightToINT4Pack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t activation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedCausalConv1d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes, const aclTensor *levelStartIndex, const aclTensor *samplingLocations, const aclTensor *attnWeights, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiScaleDeformableAttnFunction(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes, const aclTensor *levelStartIndex, const aclTensor *samplingLocations, const aclTensor *attnWeights, const aclTensor *gradOutput, aclTensor *gradValue, aclTensor *gradSampling, aclTensor *gradAttnWeights, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultiScaleDeformableAttentionGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Loss / activation backward + grad (loss3_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwishBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double beta, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSwishBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftsignBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftsignBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogitGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, double eps, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogitGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPreluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, aclTensor *gradInput, aclTensor *gradWeight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPreluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGlu(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGluBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, int64_t dim, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGluBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *mask, double p, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, const aclTensor *posWeight, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogitsBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, const aclTensor *posWeight, int64_t reduction, aclTensor *gradTarget, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBinaryCrossEntropyWithLogitsTargetBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKlDivBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKlDivBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKlDivTargetBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, bool logTarget, aclTensor *gradTarget, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnKlDivTargetBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftMarginLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftMarginLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLossBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLoss2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, const aclTensor *totalWeight, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLoss2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLoss2dGetWorkspaceSize(const aclTensor *self, const aclTensor *target, const aclTensor *weight, int64_t reduction, int64_t ignoreIndex, aclTensor *out, aclTensor *totalWeight, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNLLLoss2d(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLossGradGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double gradOutput, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCrossEntropyLossGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedLinearCrossEntropyLossGradGetWorkspaceSize(const aclTensor *self, const aclTensor *target, double gradOutput, int64_t reduction, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedLinearCrossEntropyLossGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(const aclTensor *features, const aclTensor *labels, aclTensor *loss, aclTensor *backprop, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSoftmaxCrossEntropyWithLogits(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultilabelMarginLossGetWorkspaceSize(const aclTensor *self, const aclTensor *target, int64_t reduction, aclTensor *out, aclTensor *isTarget, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultilabelMarginLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScaledMaskedSoftmaxGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, double scale, bool fixedTriuMask, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScaledMaskedSoftmax(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScaledMaskedSoftmaxBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *out, const aclTensor *mask, double scale, bool fixedTriuMask, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScaledMaskedSoftmaxBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnModulateGetWorkspaceSize(const aclTensor *x, const aclTensor *scale, const aclTensor *shift, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnModulate(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnModulateBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x, const aclTensor *scale, aclTensor *gradX, aclTensor *gradScale, aclTensor *gradShift, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnModulateBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCdistBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *x1, const aclTensor *x2, double p, const aclTensor *cdist, aclTensor *gradX1, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCdistBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedBiasAddGradGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *groupOffset, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedBiasAddGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedBiasAddGradV2GetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *groupOffset, aclTensor *gradBias, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedBiasAddGradV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveWithDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(const aclTensor *self, int64_t repeats, int64_t dim, int64_t outputSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveIntWithDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *repeats, int64_t outputSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveGradGetWorkspaceSize(const aclTensor *gradOutput, int64_t repeats, int64_t dim, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRepeatInterleaveGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpSegsumGetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpSegsum(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpSegsumBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *out, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpSegsumBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Vision/sequence backward remainder (loss4_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler2D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler3DGetWorkspaceSize(const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler3D(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler2DBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *gradInput, aclTensor *gradGrid, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler2DBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler3DBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *grid, int64_t interpolationMode, int64_t paddingMode, bool alignCorners, aclTensor *gradInput, aclTensor *gradGrid, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGridSampler3DBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dAAGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dAA(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double sH, double sW, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dAAGradGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double sH, double sW, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBicubic2dAAGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dAABackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *outputSize, const aclIntArray *inputSize, bool alignCorners, double sH, double sW, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dAABackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThreeInterpolateBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *indices, const aclTensor *weight, int64_t m, aclTensor *gradFeatures, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThreeInterpolateBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChamferDistanceBackwardGetWorkspaceSize(const aclTensor *gradDist1, const aclTensor *xyz1, const aclTensor *xyz2, const aclTensor *idx1, aclTensor *gradXyz1, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChamferDistanceBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCtcLossBackwardGetWorkspaceSize(const aclTensor *gradOut, const aclTensor *logProbs, const aclTensor *targets, const aclTensor *inputLengths, const aclTensor *targetLengths, const aclTensor *negLogLikelihood, const aclTensor *logAlpha, int64_t blank, bool zeroInfinity, aclTensor *gradInput, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCtcLossBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThnnFusedLstmCellGetWorkspaceSize(const aclTensor *gates, const aclTensor *cprev, aclTensor *hNew, aclTensor *cNew, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThnnFusedLstmCell(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThnnFusedLstmCellBackwardGetWorkspaceSize(const aclTensor *gradHy, const aclTensor *gradCy, const aclTensor *cprev, const aclTensor *cNew, const aclTensor *gates, aclTensor *gradGates, aclTensor *gradCprev, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnThnnFusedLstmCellBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstmBackwardGetWorkspaceSize(const aclTensor *gradY, const aclTensor *x, const aclTensor *wih, const aclTensor *whh, aclTensor *gradX, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLstmBackward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Index/scatter/gather/sort/sampling remainder (index3_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor *self, const aclTensorList *indices, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndex(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor *selfRef, const aclTensorList *indices, const aclTensor *values, bool accumulate, bool unsafe, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIndexPutImpl(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTfScatterAddGetWorkspaceSize(const aclTensor *ref, const aclTensor *indices, const aclTensor *updates, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTfScatterAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterListGetWorkspaceSize(aclTensorList *selfRef, const aclTensorList *indices, const aclTensorList *updates, const aclTensor *mask, int64_t reduce, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterList(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(const aclTensor *input, aclTensor *cache, const aclTensor *slotMapping, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterPaCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor *key, const aclTensor *value, aclTensor *keyCache, aclTensor *valueCache, const aclTensor *slotMapping, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnScatterPaKvCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherPaKvCacheGetWorkspaceSize(const aclTensor *keyCache, const aclTensor *valueCache, const aclTensor *slotMapping, aclTensor *keyOut, aclTensor *valueOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGatherPaKvCache(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSearchSortedsGetWorkspaceSize(const aclTensor *sortedSequence, const aclScalar *value, bool right, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSearchSorteds(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor *self, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUnique2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool sorted, bool returnInverse, bool returnCounts, aclTensor *valuesOut, aclTensor *countOut, aclTensor *inverseOut, aclTensor *countsOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUniqueDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnArgsortGetWorkspaceSize(const aclTensor *self, int64_t dim, bool descending, bool stable, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnArgsort(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingRenormGetWorkspaceSize(aclTensor *selfRef, const aclTensor *indices, double maxNorm, double normType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEmbeddingRenorm(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyTopKTopPGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyTopKTopP(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopKTopPSample(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(const aclTensor *logits, int64_t topk, double topp, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTopKTopPSampleV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAllGatherAddGetWorkspaceSize(const aclTensor *x, const aclTensor *bias, const char *group, int64_t rankSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAllGatherAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLightningIndexerGetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *weights, aclTensor *indexScore, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLightningIndexer(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLightningIndexerGradGetWorkspaceSize(const aclTensor *gradIndexScore, const aclTensor *query, const aclTensor *key, aclTensor *gradWeights, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLightningIndexerGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(const aclTensor *indexScore, aclTensor *lse, aclTensor *probs, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDenseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDenseLightningIndexerGradKLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseLightningIndexerGradKLLossGetWorkspaceSize(const aclTensor *indexScore, const aclTensor *target, aclTensor *gradScore, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSparseLightningIndexerGradKLLoss(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- RNN/FFN + misc fused (misc3_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgCholeskyGetWorkspaceSize(const aclTensor *self, bool upper, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgCholesky(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgQrGetWorkspaceSize(const aclTensor *A, int64_t mode, aclTensor *Q, aclTensor *R, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgQr(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgCrossGetWorkspaceSize(const aclTensor *self, const aclTensor *other, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLinalgCross(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogdetGetWorkspaceSize(const aclTensor *A, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogdet(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLSTM(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBidirectionLSTMGetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBidirectionLSTM(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBidirectionLSTMV2GetWorkspaceSize(const aclTensor *x, const aclTensor *wih, const aclTensor *whh, const aclTensor *bih, const aclTensor *bhh, const aclTensor *h0, const aclTensor *c0, aclTensor *y, aclTensor *hN, aclTensor *cN, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBidirectionLSTMV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor *probs, const aclTensor *numSamples, int64_t seed, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMultinomialTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogicalXor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSigmoidForwardGetWorkspaceSize(const aclTensor *self, aclTensor *out, aclTensor *buffer, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLogSigmoidForward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFatreluMulGetWorkspaceSize(const aclTensor *x1, const aclTensor *x2, double threshold, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFatreluMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAxpyV2GetWorkspaceSize(const aclTensor *self, const aclTensor *other, double alpha, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAxpyV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeftShiftsGetWorkspaceSize(const aclTensor *self, const aclScalar *shift, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnLeftShifts(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor *self, const aclScalar *element, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsInTensorScalar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(const aclScalar *element, const aclTensor *testElements, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIsInScalarTensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnReal(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor *re, const aclTensor *im, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnComplex(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor *abs, const aclTensor *angle, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPolar(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAngleV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnVarCorrectionGetWorkspaceSize(const aclTensor *self, const aclIntArray *dim, int64_t correction, bool keepDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnVarCorrection(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPdistGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPdist(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPdistForwardGetWorkspaceSize(const aclTensor *self, double p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnPdistForward(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFNGetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFN(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFNV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFNV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFNV3GetWorkspaceSize(const aclTensor *x, const aclTensor *weight1, const aclTensor *bias1, const aclTensor *weight2, const aclTensor *bias2, int64_t activation, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFFNV3(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedSoftmaxWithRelPosBiasGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, const aclTensor *relPosBias, double scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMaskedSoftmaxWithRelPosBias(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Matmul/Gemm/Grouped + quant-matmul variants (matmul3_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeBatchMatMulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeBatchMatMul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeBatchMatMulWeightNZGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransposeBatchMatMulWeightNZ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatmulQuantGetWorkspaceSize(const aclTensor *self, const aclTensor *other, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnBatchMatmulQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedMatmulGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulCompressGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulCompress(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulCompressDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMatmulCompressDequant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedQuantMatmulGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedQuantMatmul(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedQuantMatmulWeightNzGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnFusedQuantMatmulWeightNz(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransMatmulWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransMatmulWeight(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateMatmulWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateMatmulWeightSize(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateMatmulWeightSizeV2GetWorkspaceSize(const aclIntArray *weightShape, int64_t dtype, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCalculateMatmulWeightSizeV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNZ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, aclTensor *scaleOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnGroupedMatmulSwigluQuantWeightNzV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulDequantGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulDequant(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulDequantWeightNZ(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- GroupedMatmulFinalizeRouting + QuantGroupedMatmulInplaceAdd (matmul5_ext) ----
#define FR_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
FR_DECL(aclnnGroupedMatmulFinalizeRouting) FR_DECL(aclnnGroupedMatmulFinalizeRoutingV2) FR_DECL(aclnnGroupedMatmulFinalizeRoutingV3)
FR_DECL(aclnnGroupedMatmulFinalizeRoutingWeightNz) FR_DECL(aclnnGroupedMatmulFinalizeRoutingWeightNzV2)
#undef FR_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulInplaceAddGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, const aclIntArray *groupList, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnQuantGroupedMatmulInplaceAdd(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- Attention/MLA/RoPE variants (attn3_ext) ----
#define ROPE_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, int64_t mode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ROPE_DECL(aclnnRotaryPositionEmbedding) ROPE_DECL(aclnnRotaryPositionEmbeddingV2) ROPE_DECL(aclnnRotaryPositionEmbeddingGrad)
ROPE_DECL(aclnnRopeWithSinCosCache) ROPE_DECL(aclnnRopeWithSinCosCacheV2) ROPE_DECL(aclnnDequantRopeQuantKvcache)
ROPE_DECL(aclnnNormRopeConcatBackward)
#undef ROPE_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnInterleaveRopeGetWorkspaceSize(const aclTensor *x, const aclTensor *cos, const aclTensor *sin, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInterleaveRope(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define RING_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *out1, const aclTensor *lse1, const aclTensor *out2, const aclTensor *lse2, aclTensor *out, aclTensor *lse, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
RING_DECL(aclnnRingAttentionUpdate) RING_DECL(aclnnRingAttentionUpdateV2) RING_DECL(aclnnAttentionUpdate)
#undef RING_DECL
#define NRC_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *cos, const aclTensor *sin, double eps, int64_t mode, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
NRC_DECL(aclnnKvRmsNormRopeCache) NRC_DECL(aclnnQkvRmsNormRopeCache) NRC_DECL(aclnnMlaPreprocess) NRC_DECL(aclnnMlaPreprocessV2)
NRC_DECL(aclnnMlaProlog) NRC_DECL(aclnnMlaPrologV2WeightNz) NRC_DECL(aclnnMlaPrologV3WeightNz) NRC_DECL(aclnnNormRopeConcat)
#undef NRC_DECL
#define ATT_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *attenMask, double scaleValue, int64_t headNum, bool causal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ATT_DECL(aclnnBlockSparseAttention) ATT_DECL(aclnnBlitzSparseAttention) ATT_DECL(aclnnNsaCompressAttention) ATT_DECL(aclnnNsaCompressAttentionInfer)
ATT_DECL(aclnnNsaSelectedAttention) ATT_DECL(aclnnNsaSelectedAttentionInfer) ATT_DECL(aclnnFusedFloydAttention) ATT_DECL(aclnnRainFusionAttention) ATT_DECL(aclnnSwinAttentionScoreQuant)
#undef ATT_DECL
#define ATTG_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *gradOut, const aclTensor *q, const aclTensor *k, const aclTensor *v, aclTensor *gradQ, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ATTG_DECL(aclnnNsaSelectedAttentionGrad) ATTG_DECL(aclnnBlockSparseAttentionGrad) ATTG_DECL(aclnnFusedFloydAttentionGrad)
#undef ATTG_DECL
#define NSAC_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, int64_t blockSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
NSAC_DECL(aclnnNsaCompress) NSAC_DECL(aclnnNsaCompressWithCache)
#undef NSAC_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnNsaCompressGradGetWorkspaceSize(const aclTensor *gradOut, int64_t blockSize, aclTensor *gradIn, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNsaCompressGrad(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define A2F_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
A2F_DECL(aclnnAttentionToFFN) A2F_DECL(aclnnFFNToAttention)
#undef A2F_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAttentionWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceAttentionWorkerScheduler(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFfnWorkerSchedulerGetWorkspaceSize(aclTensor *selfRef, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnInplaceFfnWorkerScheduler(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
// ---- MoE EP/routing remainder (moe3_ext) ----
#define MOEP_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertId, int64_t numExperts, aclTensor *permX, aclTensor *srcIdx, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOEP_DECL(aclnnMoeTokenPermuteWithEp) MOEP_DECL(aclnnMoeTokenPermuteWithRoutingMap)
#undef MOEP_DECL
#define MOEPG_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *gradPermX, const aclTensor *srcIdx, aclTensor *gradX, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOEPG_DECL(aclnnMoeTokenPermuteWithEpGrad) MOEPG_DECL(aclnnMoeTokenPermuteWithRoutingMapGrad)
#undef MOEPG_DECL
#define MOEU_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOEU_DECL(aclnnMoeTokenUnpermuteWithEp) MOEU_DECL(aclnnMoeTokenUnpermuteWithRoutingMap)
#undef MOEU_DECL
#define MOEUG_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *gradOut, const aclTensor *permY, const aclTensor *srcIdx, const aclTensor *weight, aclTensor *gradPermY, aclTensor *gradWeight, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOEUG_DECL(aclnnMoeTokenUnpermuteWithEpGrad) MOEUG_DECL(aclnnMoeTokenUnpermuteWithRoutingMapGrad)
#undef MOEUG_DECL
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeUpdateExpertGetWorkspaceSize(const aclTensor *expertIds, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMoeUpdateExpert(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#define MOEIR_DECL(name) \
ACL_FUNC_VISIBILITY aclnnStatus name##GetWorkspaceSize(const aclTensor *x, const aclTensor *expertIdx, int64_t numExperts, aclTensor *expandedX, aclTensor *expandedRowIdx, aclTensor *expandedExpertIdx, uint64_t *workspaceSize, aclOpExecutor **executor); \
ACL_FUNC_VISIBILITY aclnnStatus name(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
MOEIR_DECL(aclnnMoeInitRoutingQuant) MOEIR_DECL(aclnnMoeInitRoutingQuantV2)
#undef MOEIR_DECL
// ---- final tractable remainder (misc4_ext) ----
ACL_FUNC_VISIBILITY aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor *self, int64_t diagonal, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDiag(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMedianDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanMedian(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor *self, int64_t dim, bool keepDim, aclTensor *valuesOut, aclTensor *indicesOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNanMedianDim(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dAAGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleBilinear2dAA(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1dV2GetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnUpsampleNearest1dV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(const aclTensor *q, const aclTensor *k, const aclTensor *v, const aclTensor *beta, const aclTensor *g, aclTensor *y, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnRecurrentGatedDeltaRule(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonMaxSuppressionGetWorkspaceSize(const aclTensor *boxes, const aclTensor *scores, double iouThreshold, aclTensor *keepOut, aclTensor *countOut, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNonMaxSuppression(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcSinkhornGetWorkspaceSize(const aclTensor *cost, double tau, int64_t iters, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcSinkhorn(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEqualGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnEqual(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMaskGetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMask(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMaskV2GetWorkspaceSize(const aclIntArray *shape, double p, int64_t seed, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMaskV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMaskV2TensorGetWorkspaceSize(const aclTensor *shapeRef, double p, int64_t seed, aclTensor *mask, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutGenMaskV2Tensor(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutDoMaskGetWorkspaceSize(const aclTensor *x, const aclTensor *mask, double p, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnDropoutDoMask(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignBitsPackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignBitsPack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignBitsUnpackGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnSignBitsUnpack(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpandvGetWorkspaceSize(const aclTensor *self, const aclIntArray *size, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnExpandv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIouGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnIou(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCIoUGetWorkspaceSize(const aclTensor *boxes1, const aclTensor *boxes2, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnCIoU(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransformBiasRescaleQkvGetWorkspaceSize(const aclTensor *qkv, const aclTensor *bias, int64_t headDim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnTransformBiasRescaleQkv(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNpuFormatCastGetWorkspaceSize(const aclTensor *self, int64_t format, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnNpuFormatCast(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChunkCatGetWorkspaceSize(const aclTensor *self, int64_t chunks, int64_t dim, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnChunkCat(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcPreGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcPre(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcPostGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnMhcPost(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdvanceStepGetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdvanceStep(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdvanceStepV2GetWorkspaceSize(const aclTensor *positions, int64_t numSeqs, int64_t blockSize, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdvanceStepV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyFusedEmaAdamGetWorkspaceSize(aclTensor *param, aclTensor *m, aclTensor *v, aclTensor *ema, const aclTensor *grad, double lr, double beta1, double beta2, double eps, double weightDecay, double emaDecay, int64_t step, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnApplyFusedEmaAdam(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(aclTensor *selfRef, const aclTensor *value, int64_t begin, int64_t end, int64_t stride, uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnStridedSliceAssignV2(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
#undef ACLNN_BIN
#undef ACLNN_BIN_ALPHA
#undef ACLNN_SCALAR_ALPHA
#undef ACLNN_SCALAR
#undef ACLNN_UN

#ifdef __cplusplus
}
#endif

#endif
