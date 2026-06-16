// Tensor generators (ops-math): Eye (identity), Linspace, LogSpace, Range. No input tensor — the output
// is produced from scalar parameters stashed in the executor. fp32 / fp16 / bf16 outputs.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int GT = 256;
inline int64_t gg(int64_t n) { return (n + GT - 1) / GT; }

template <typename T> __device__ inline T gst(float v) { return (T)v; }
template <> __device__ inline __half gst(float v) { return __float2half(v); }
template <> __device__ inline __nv_bfloat16 gst(float v) { return __float2bfloat16(v); }

template <typename T> __global__ void k_eye(T *o, int64_t rows, int64_t cols) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int64_t r = i / cols, c = i % cols;
    o[i] = gst<T>(r == c ? 1.f : 0.f);
}
// arithmetic sequence: o[i] = start + i*step  (Linspace passes step=(end-start)/(steps-1); Range passes the given step)
template <typename T> __global__ void k_affine(T *o, float start, float step, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) o[i] = gst<T>(start + (float)i * step);
}
template <typename T> __global__ void k_logspace(T *o, float start, float step, float base, int64_t n) {
    int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
    if (i < n) o[i] = gst<T>(powf(base, start + (float)i * step));
}

enum GKind { G_EYE = 1, G_AFFINE, G_LOGSPACE };

aclnnStatus gen_build(int kind, aclTensor *out, double a, double b, double c, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !out->data || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor();
    e->op = kind; e->out = out; e->dscalars = {a, b, c};
    *ws = 0; *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus gen_run(aclOpExecutor *e, cudaStream_t s) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    aclTensor *o = e->out; int64_t n = o->numel(); aclDataType dt = o->dtype;
    aclnnStatus st = ACLNN_SUCCESS;
    if (e->op == G_EYE) {
        int rk = (int)o->viewDims.size();
        int64_t rows = rk >= 2 ? o->viewDims[rk - 2] : n, cols = rk >= 1 ? o->viewDims[rk - 1] : 1;
        #define KE(T) k_eye<T><<<gg(n), GT, 0, s>>>((T*)o->data, rows, cols)
        switch (dt) { case ACL_FLOAT: KE(float); break; case ACL_FLOAT16: KE(__half); break; case ACL_BF16: KE(__nv_bfloat16); break; default: st = ACLNN_ERR_PARAM_INVALID; }
        #undef KE
    } else if (e->op == G_AFFINE) {
        float start = (float)e->dscalars[0], step = (float)e->dscalars[1];
        #define KA(T) k_affine<T><<<gg(n), GT, 0, s>>>((T*)o->data, start, step, n)
        switch (dt) { case ACL_FLOAT: KA(float); break; case ACL_FLOAT16: KA(__half); break; case ACL_BF16: KA(__nv_bfloat16); break; default: st = ACLNN_ERR_PARAM_INVALID; }
        #undef KA
    } else if (e->op == G_LOGSPACE) {
        float start = (float)e->dscalars[0], step = (float)e->dscalars[1], base = (float)e->dscalars[2];
        #define KL(T) k_logspace<T><<<gg(n), GT, 0, s>>>((T*)o->data, start, step, base, n)
        switch (dt) { case ACL_FLOAT: KL(float); break; case ACL_FLOAT16: KL(__half); break; case ACL_BF16: KL(__nv_bfloat16); break; default: st = ACLNN_ERR_PARAM_INVALID; }
        #undef KL
    } else st = ACLNN_ERR_PARAM_INVALID;
    delete e;
    if (st != ACLNN_SUCCESS) return st;
    return cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // namespace

extern "C" {

// Eye: identity over out's last two dims (1 on the main diagonal)
aclnnStatus aclnnEyeGetWorkspaceSize(aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return gen_build(G_EYE, out, 0, 0, 0, ws, ex);
}
aclnnStatus aclnnEye(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return gen_run(e, (cudaStream_t)s); }

// Linspace: steps evenly spaced values in [start, end] inclusive
aclnnStatus aclnnLinspaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps,
                                          aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!start || !end || steps <= 0) return ACLNN_ERR_PARAM_INVALID;
    double s0 = start->v, step = steps > 1 ? (end->v - start->v) / (double)(steps - 1) : 0.0;
    return gen_build(G_AFFINE, out, s0, step, 0, ws, ex);
}
aclnnStatus aclnnLinspace(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return gen_run(e, (cudaStream_t)s); }

// LogSpace: base ^ linspace(start, end, steps)
aclnnStatus aclnnLogSpaceGetWorkspaceSize(const aclScalar *start, const aclScalar *end, int64_t steps, double base,
                                          aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!start || !end || steps <= 0) return ACLNN_ERR_PARAM_INVALID;
    double s0 = start->v, step = steps > 1 ? (end->v - start->v) / (double)(steps - 1) : 0.0;
    return gen_build(G_LOGSPACE, out, s0, step, base, ws, ex);
}
aclnnStatus aclnnLogSpace(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return gen_run(e, (cudaStream_t)s); }

// Range: start, start+step, ... (half-open, count taken from out->numel)
aclnnStatus aclnnRangeGetWorkspaceSize(const aclScalar *start, const aclScalar *end, const aclScalar *step,
                                       aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!start || !step) return ACLNN_ERR_PARAM_INVALID;
    (void)end;
    return gen_build(G_AFFINE, out, start->v, step->v, 0, ws, ex);
}
aclnnStatus aclnnRange(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return gen_run(e, (cudaStream_t)s); }

} // extern "C"
