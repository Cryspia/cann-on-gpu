// Metal backend internal shared definitions: aclTensor/aclScalar/aclOpExecutor structs (host metadata +
// device pointer), dtype utilities, and the Metal context seam (device / metallib / pipeline cache /
// buffer registry / streams). The aclTensor etc. structs are backend-agnostic — kept identical to the
// CUDA backend (cuda/src/internal.h) so operator plans port unchanged. Header contracts come from
// $ACL_INCLUDE (acl/acl.h, aclnn/acl_meta.h); do not re-declare them here.
//
// This header is Objective-C++ only (included from .mm). It pulls in <Metal/Metal.h>.
#ifndef CANN_ON_GPU_METAL_INTERNAL_H
#define CANN_ON_GPU_METAL_INTERNAL_H

#import <Metal/Metal.h>
#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// Standard aclnn error codes (consistent with CANN docs; operator headers ship with the ops package and
// are not available locally, so they are defined in the shim — identical to the CUDA backend).
constexpr aclnnStatus ACLNN_SUCCESS = 0;
constexpr aclnnStatus ACLNN_ERR_PARAM_NULLPTR = 161001;
constexpr aclnnStatus ACLNN_ERR_PARAM_INVALID = 161002;
constexpr aclnnStatus ACLNN_ERR_RUNTIME_ERROR = 361001;

// ---- aclTensor / aclScalar structs (host metadata + device data pointer) ----
struct aclTensor {
    std::vector<int64_t> viewDims, strides, storageDims;
    int64_t offset = 0;            // element offset
    aclDataType dtype = ACL_DT_UNDEFINED;
    aclFormat format = ACL_FORMAT_ND;
    void *data = nullptr;          // device pointer (== host pointer under MTLStorageModeShared)

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
struct aclTensorList { std::vector<const aclTensor *> v; };

// Per-operator execution plan: populated by GetWorkspaceSize, consumed by Execute (one-shot, like CANN).
struct aclOpExecutor {
    int op = 0;                    // OpKind
    const aclTensor *a = nullptr, *b = nullptr, *c = nullptr;  // c: third input / bias
    aclTensor *out = nullptr;
    double alpha = 1.0;            // binary alpha / scalar value / attention scale
    aclDataType castTo = ACL_DT_UNDEFINED;
    int64_t m = 0, n = 0, k = 0;                       // matmul
    int64_t stride[2] = {1, 1}, pad[2] = {0, 0}, dil[2] = {1, 1};  // conv2d
    int64_t dim = -1;              // softmax/layernorm reduction dim (normalized non-negative)
    int64_t reduceCount = 0;
    int64_t outerCount = 0;
    std::vector<int64_t> axes;
    bool keepDim = false;
    std::vector<const aclTensor *> inputs;
    std::vector<aclTensor *> owned;          // temporaries freed with the executor
    std::vector<double> dscalars;
    double eps = 1e-5;
    const aclTensor *mean = nullptr, *rstd = nullptr;
    aclTensor *out2 = nullptr;
    int64_t ab = 0, an = 0, asq = 0, askv = 0, ad = 0;   // attention (BNSD)
    const aclTensor *mask = nullptr;
    bool causal = false;
    const aclTensorList *tl[4] = {nullptr, nullptr, nullptr, nullptr};  // foreach: input/output tensor lists
};

enum OpKind {
    OP_ADD = 1, OP_SUB, OP_MUL, OP_DIV, OP_MAXIMUM, OP_MINIMUM,
    OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS, OP_CLAMP_MIN, OP_CLAMP_MAX,
    OP_EXP, OP_LOG, OP_ABS, OP_SQRT, OP_RSQRT, OP_RECIPROCAL, OP_RELU, OP_NEG,
    OP_SIGMOID, OP_TANH, OP_ERF, OP_GELU, OP_SILU, OP_SOFTPLUS,
    OP_SIN, OP_COS, OP_TAN, OP_ATAN, OP_SIGN, OP_FLOOR, OP_CEIL, OP_ROUND, OP_TRUNC, OP_SQUARE, OP_POWS,
    OP_CAST, OP_REDUCE_SUM, OP_AMAX, OP_AMIN, OP_MEAN, OP_PROD, OP_ARGMAX, OP_ARGMIN,
    OP_MATMUL, OP_CONV2D, OP_SOFTMAX, OP_LAYERNORM, OP_ATTENTION,
    OP_PERMUTE, OP_SLICE, OP_TILE, OP_PAD, OP_GATHER, OP_CONCAT, OP_SPLIT, OP_RESHAPE,
    OP_GT, OP_LT, OP_EQ, OP_NE, OP_GE, OP_LE,           // comparisons (bool output)
    OP_LAND, OP_LOR, OP_LNOT,                           // logical (bool in/out)
    OP_ISNAN, OP_ISFINITE, OP_ISINF, OP_ISPOSINF, OP_ISNEGINF,  // predicates (bool out)
    OP_MASKEDFILL,                                      // mask ? value : self
    OP_RMSNORM, OP_GROUPNORM, OP_BATCHNORM,
    // (extended as families land — kept sparse; mirror cuda/src/internal.h when needed)
};

inline size_t dtype_size(aclDataType t) {
    switch (t) {
        case ACL_FLOAT: case ACL_INT32: case ACL_UINT32: return 4;
        case ACL_FLOAT16: case ACL_BF16: case ACL_INT16: case ACL_UINT16: return 2;
        case ACL_INT8: case ACL_UINT8: case ACL_BOOL: return 1;
        case ACL_FLOAT8_E4M3FN: case ACL_FLOAT8_E5M2: case ACL_FLOAT8_E8M0: case ACL_HIFLOAT8: return 1;
        case ACL_INT64: case ACL_UINT64: case ACL_DOUBLE: return 8;
        case ACL_FLOAT4_E2M1: case ACL_FLOAT4_E1M2: case ACL_FLOAT6_E2M3: case ACL_FLOAT6_E3M2: return 1;
        case ACL_INT4: return 1;
        default: return 0;
    }
}

// ---- Metal context seam (implemented in context.mm) ----
// A stream is a command queue plus the last-committed command buffer (for synchronize).
struct AclStream {
    id<MTLCommandQueue> q = nil;
    id<MTLCommandBuffer> last = nil;
    bool capturing = false;          // aclmdlRICapture* graph capture in progress on this stream
    void *capgraph = nullptr;        // ShimGraph* currently being recorded (runtime.mm)
};

// Graph capture/replay hook. If `s` is mid-capture, deep-copies the executor (and its tensors; the
// data pointers persist) onto the tape and returns true — the calling op must then NOT run or delete `e`.
// Otherwise returns false and the op runs normally. Defined in runtime.mm. AclExecFn is the Execute ABI.
extern "C" { typedef aclnnStatus (*AclExecFn)(void *, uint64_t, aclOpExecutor *, aclrtStream); }
bool aclCaptureRecord(aclrtStream s, AclExecFn fn, aclOpExecutor *e, void *ws, uint64_t wss);

namespace mtl {
void init();                                   // idempotent: device + default queue + load default.metallib
id<MTLDevice> device();
id<MTLCommandQueue> defaultQueue();
id<MTLComputePipelineState> pipeline(NSString *name);   // get-or-create, cached by function name
void *alloc(size_t n);                         // MTLBuffer (shared) -> host/device pointer; registered
void free_(void *p);
// Find the MTLBuffer backing pointer p; *byteOffset = p - buffer base. nil if p is not a known allocation.
id<MTLBuffer> bufferFor(const void *p, size_t *byteOffset);
}

// MC2 (matmul + collective) ops are declared canonically in <aclnnop/aclnn_mc2.h> (carrying an HcclComm);
// the .mm files that implement them (blas_parity/quant_parity/misc_parity) include that header directly.

#endif
