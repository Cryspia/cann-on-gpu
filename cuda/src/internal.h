// Shim internal shared definitions: aclTensor/aclScalar structs, error mapping, dtype utilities.
// All header contracts come from $ACL_INCLUDE (acl_base_rt.h / acl_meta.h); do not re-declare them here.
#ifndef CANN_ON_GPU_SHIM_INTERNAL_H
#define CANN_ON_GPU_SHIM_INTERNAL_H

#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// Standard aclnn error codes (consistent with CANN documentation; operator headers are shipped with the ops package and not available locally, so they are defined here in the shim)
constexpr aclnnStatus ACLNN_SUCCESS = 0;
constexpr aclnnStatus ACLNN_ERR_PARAM_NULLPTR = 161001;
constexpr aclnnStatus ACLNN_ERR_PARAM_INVALID = 161002;
constexpr aclnnStatus ACLNN_ERR_RUNTIME_ERROR = 361001;

// CUDA error → aclError (any non-zero value is an error; CUDA code is preserved for debugging)
inline aclError acl_from_cuda(cudaError_t e) {
    return (e == cudaSuccess) ? ACL_SUCCESS : (aclError)(500000 + (int)e);
}
#define ACL_CUDA(call) do { cudaError_t _e = (call); if (_e != cudaSuccess) { \
    fprintf(stderr, "[shim] %s:%d CUDA error %d (%s)\n", __FILE__, __LINE__, (int)_e, cudaGetErrorString(_e)); \
    return acl_from_cuda(_e); } } while (0)

// ---- aclTensor / aclScalar structs (host metadata + device data pointer) ----
struct aclTensor {
    std::vector<int64_t> viewDims, strides, storageDims;
    int64_t offset = 0;            // element offset
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_ND;
    void *data = nullptr;          // device pointer

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : viewDims) n *= d;
        return n;
    }
    bool contiguous() const {
        int64_t s = 1;
        for (int i = (int)viewDims.size() - 1; i >= 0; --i) {
            if (strides[i] != s) return false;
            s *= viewDims[i];
        }
        return true;
    }
};

struct aclScalar {
    double v = 0.0;                // stored as double uniformly; cast to target dtype at use site
    aclDataType dtype = ACL_DT_UNDEFINED;
};

struct aclIntArray { std::vector<int64_t> v; };
struct aclTensorList { std::vector<const aclTensor *> v; };   // native tensor list (Cat/Split, etc.)

// Per-operator execution plan: populated by GetWorkspaceSize, consumed by Execute (CANN executor is one-shot, not repeatable)
struct aclOpExecutor {
    int op = 0;                    // OpKind
    const aclTensor *a = nullptr, *b = nullptr, *c = nullptr;  // c: third input for SWhere / bias for conv
    aclTensor *out = nullptr;
    double alpha = 1.0;            // binary alpha / scalar value for scalar ops / scale for attention
    aclDataType castTo = ACL_DT_UNDEFINED;
    int64_t m = 0, n = 0, k = 0;                       // matmul
    int64_t stride[2] = {1, 1}, pad[2] = {0, 0}, dil[2] = {1, 1};  // conv2d
    // softmax / layernorm
    int64_t dim = -1;              // softmax reduction dimension (already normalized to non-negative)
    int64_t reduceCount = 0;       // number of elements to reduce for softmax/layernorm
    int64_t outerCount = 0;        // number of rows (outer element count)
    std::vector<int64_t> axes;     // reduce: set of reduction dims; shape: permute dims / tile repeats / pad widths
    bool keepDim = false;          // reduce: keep reduced dimensions as size 1
    std::vector<const aclTensor *> inputs;   // input tensor list for concat
    std::vector<aclTensor *> owned;          // temporary view tensors constructed internally (e.g. unit dim inserted by Stack); freed with the executor
    std::vector<double> dscalars;            // generic scalar parameter pack (e.g. optimizer lr/beta/eps/wd/bias-correction)
    double eps = 1e-5;
    const aclTensor *mean = nullptr, *rstd = nullptr;  // optional outputs for layernorm
    aclTensor *out2 = nullptr;     // second output (indices for sort/topk)
    // attention（BNSD）
    int64_t ab = 0, an = 0, asq = 0, askv = 0, ad = 0;
    const aclTensor *mask = nullptr;
    bool causal = false;
};

enum OpKind {
    OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV, OP_MAXIMUM, OP_MINIMUM,
    OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS, OP_CLAMP_MIN, OP_CLAMP_MAX,
    OP_EXP, OP_LOG, OP_ABS, OP_SQRT, OP_RSQRT, OP_RECIPROCAL, OP_RELU, OP_NEG,
    OP_SIGMOID, OP_TANH, OP_ERF, OP_GELU, OP_SILU, OP_SOFTPLUS,
    OP_SIN, OP_COS, OP_TAN, OP_ATAN, OP_SIGN, OP_FLOOR, OP_CEIL, OP_ROUND, OP_TRUNC, OP_SQUARE, OP_POWS,
    OP_SINH, OP_COSH, OP_ASIN, OP_ACOS, OP_ERFC, OP_FRAC, OP_LGAMMA,        // elementary functions (unary)
    OP_FMOD, OP_HYPOT, OP_POW,                                             // binary (tensor∘tensor)
    OP_CAST, OP_REDUCE_SUM, OP_AMAX, OP_AMIN, OP_MEAN, OP_PROD, OP_ARGMAX, OP_ARGMIN,
    OP_ALL, OP_ANY, OP_NORM, OP_VAR, OP_STD, OP_LSE,   // statistical reductions
    OP_BAND, OP_BOR, OP_BNOT, OP_SWHERE,
    OP_MATMUL, OP_CONV2D,
    OP_SOFTMAX, OP_LAYERNORM, OP_ATTENTION,
    OP_PERMUTE, OP_SLICE, OP_TILE, OP_PAD, OP_GATHER, OP_CONCAT, OP_SPLIT, OP_RESHAPE, OP_STRIDED_SLICE,
    OP_GT, OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE,           // comparisons (bool output)
    OP_LAND, OP_LOR, OP_LNOT, OP_ISNAN, OP_ISFINITE, OP_MASKEDFILL,  // logical / predicate / fill
    OP_CUMSUM, OP_CUMPROD, OP_SORT, OP_TOPK,           // sort / scan (along a dimension)
    OP_RMSNORM, OP_GROUPNORM, OP_BATCHNORM,            // additional norm variants
    // ---- P1 elementwise math extensions ----
    OP_EXPM1, OP_LOG1P, OP_LOG2, OP_LOG10, OP_EXP2, OP_ERFINV,          // unary math
    OP_ATAN2, OP_REMAINDER, OP_XLOGY, OP_LOGADDEXP, OP_COPYSIGN, OP_HEAVISIDE, OP_LERP,  // binary math (broadcast)
    OP_ADDCMUL, OP_ADDCDIV, OP_CLAMPT,                                  // ternary (self, t1/min, t2/max)
    // ---- P2 activation extensions ----
    OP_MISH, OP_HARDSWISH, OP_HARDSIGMOID, OP_LOGSIGMOID, OP_SELU, OP_TANHSHRINK, OP_RELU6,  // unary no-param
    OP_LEAKYRELU, OP_ELU, OP_CELU, OP_HARDSHRINK, OP_SOFTSHRINK,        // unary one-scalar (param in alpha)
    OP_HARDTANH, OP_THRESHOLD,                                          // unary two-scalar (params in dscalars)
    OP_PRELU,                                                          // binary (broadcast weight)
    // ---- P3 reduce & scan extensions ----
    OP_NANSUM, OP_NANMEAN, OP_AMINMAX, OP_COUNTNZ,                      // NaN-aware / dual-output / count
    OP_MEDIAN, OP_KTHVALUE, OP_QUANTILE, OP_MODE, OP_RENORM,           // selection / renorm
    OP_CUMMAX, OP_CUMMIN, OP_LCUMSUMEXP,                               // scans (along a dim)
    // ---- additional elementary math ----
    OP_ACOSH, OP_ASINH, OP_ATANH, OP_SINC, OP_DIGAMMA,                 // unary
    OP_LOGADDEXP2,                                                     // binary
    // ---- additional activations ----
    OP_FASTGELU, OP_SQUAREDRELU,                                       // unary no-param
    OP_SWISH,                                                          // unary one-scalar (beta in alpha)
    // ---- more arithmetic ----
    OP_FLOORDIV, OP_RSUB,                                              // binary
    OP_ROUNDDEC,                                                      // unary one-scalar (decimals in alpha)
    OP_ISINF, OP_ISPOSINF, OP_ISNEGINF,                               // predicates (bool out)
    OP_BXOR,                                                          // binary bitwise xor
    OP_BANDS, OP_BORS, OP_BXORS,                                      // tensor ∘ scalar bitwise (scalar in alpha)
    // ---- more reductions ----
    OP_REDUCE_LOGSUM,                                                 // log(sum) along dims
    OP_VARMEAN, OP_STDMEAN,                                           // dual output: out=var/std, out2=mean
    OP_MAXDIM, OP_MINDIM,                                             // dual output: out=values, out2=indices
    OP_NANTONUM,                                                     // unary three-scalar (nan/posinf/neginf in dscalars)
};

// Common validation: non-null, contiguous, ND format, shape consistent (compared against a)
aclnnStatus check_same(const aclTensor *a, const aclTensor *t);

// Single-tensor elementwise dispatch (all dtypes/OpKinds); reused by the foreach family. Defined in elementwise.cu.
// Builds and runs a one-shot executor; pass ws=nullptr for non fp4/fp6 dtypes.
aclnnStatus ew_run_one(int op, const aclTensor *a, const aclTensor *b, const aclTensor *c,
                       aclTensor *out, double alpha, void *ws, cudaStream_t s);
// Run a pre-built elementwise executor (Execute phase); reused by the in-place family. Frees the executor.
aclnnStatus ew_exec(aclOpExecutor *e, void *ws, cudaStream_t s);

inline size_t dtype_size(aclDataType t) {
    switch (t) {
        case ACL_FLOAT: case ACL_INT32: case ACL_UINT32: return 4;
        case ACL_FLOAT16: case ACL_BF16: case ACL_INT16: case ACL_UINT16: return 2;
        case ACL_INT8: case ACL_UINT8: case ACL_BOOL: return 1;
        // fp8 family: 1 byte per element (CUDA __nv_fp8_* and HiF8 are both 8-bit; E8M0 is the microscaling exponent code)
        case ACL_FLOAT8_E4M3FN: case ACL_FLOAT8_E5M2: case ACL_FLOAT8_E8M0: case ACL_HIFLOAT8: return 1;
        case ACL_INT64: case ACL_UINT64: case ACL_DOUBLE: return 8;
        // fp4 (4-bit) / fp6 (6-bit) sub-byte types: dtype_size cannot express the true storage size;
        // returning 1 is only to allow aclCreateTensor to accept them.
        // Actual packed byte counts are computed explicitly in cast (fp4=ceil(n/2), fp6=n); see subfp.cuh.
        case ACL_FLOAT4_E2M1: case ACL_FLOAT4_E1M2: case ACL_FLOAT6_E2M3: case ACL_FLOAT6_E3M2: return 1;
        case ACL_INT4: return 1;   // int4 sub-byte: sentinel (true 2-per-byte packing is computed explicitly in quantized matmul)
        default: return 0;
    }
}

#endif
