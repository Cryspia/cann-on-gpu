// aclnnLayerNorm: normalize over the last K dims. Hand-written kernel; one block per row; double accumulation for mean/var.
// out = (x-mean)*rstd*gamma + beta, rstd = 1/sqrt(var+eps). Optional mean/rstd outputs.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {

constexpr int LT = 256;

template <typename T>
__global__ void layernorm_kernel(const T *x, const T *gamma, const T *beta, T *y,
                                  float *meanOut, float *rstdOut, int64_t cols, double eps) {
    int64_t row = blockIdx.x;
    const T *xr = x + row * cols;
    T *yr = y + row * cols;
    __shared__ double ssum[LT], ssq[LT];
    double s = 0, sq = 0;
    for (int64_t i = threadIdx.x; i < cols; i += LT) { double v = (double)(float)xr[i]; s += v; sq += v * v; }
    ssum[threadIdx.x] = s; ssq[threadIdx.x] = sq;
    __syncthreads();
    for (int st = LT / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) { ssum[threadIdx.x] += ssum[threadIdx.x + st]; ssq[threadIdx.x] += ssq[threadIdx.x + st]; }
        __syncthreads();
    }
    __shared__ double mean, rstd;
    if (threadIdx.x == 0) {
        mean = ssum[0] / cols;
        double var = ssq[0] / cols - mean * mean;
        rstd = 1.0 / sqrt(var + eps);
        if (meanOut) meanOut[row] = (float)mean;
        if (rstdOut) rstdOut[row] = (float)rstd;
    }
    __syncthreads();
    for (int64_t i = threadIdx.x; i < cols; i += LT) {
        double v = ((double)(float)xr[i] - mean) * rstd;
        double g = gamma ? (double)(float)gamma[i] : 1.0;
        double b = beta ? (double)(float)beta[i] : 0.0;
        yr[i] = (T)(v * g + b);
    }
}

} // namespace

extern "C" {

aclnnStatus aclnnLayerNormGetWorkspaceSize(const aclTensor *input, const aclIntArray *normalizedShape,
                                           const aclTensor *weight, const aclTensor *bias, double eps,
                                           aclTensor *out, aclTensor *meanOut, aclTensor *rstdOut,
                                           uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !normalizedShape || !out || !ws || !ex || !input->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (input->dtype != out->dtype || input->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (input->dtype != ACL_FLOAT && input->dtype != ACL_FLOAT16 && input->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (!input->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    const auto &ns = normalizedShape->v;
    const int64_t rank = input->viewDims.size();
    if ((int64_t)ns.size() > rank || ns.empty()) return ACLNN_ERR_PARAM_INVALID;
    int64_t cols = 1;
    for (size_t i = 0; i < ns.size(); i++) {
        if (input->viewDims[rank - ns.size() + i] != ns[i]) return ACLNN_ERR_PARAM_INVALID;  // trailing dims must match
        cols *= ns[i];
    }
    if (weight && (weight->numel() != cols || !weight->data)) return ACLNN_ERR_PARAM_INVALID;
    if (bias && (bias->numel() != cols || !bias->data)) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor();
    e->op = OP_LAYERNORM; e->a = input; e->b = weight; e->c = bias; e->out = out;
    e->mean = meanOut; e->rstd = rstdOut;
    e->reduceCount = cols; e->outerCount = input->numel() / cols; e->eps = eps;
    *ws = 0; *ex = e;
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnLayerNorm(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_LAYERNORM) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    float *mo = e->mean ? (float *)e->mean->data : nullptr;
    float *ro = e->rstd ? (float *)e->rstd->data : nullptr;
    if (e->a->dtype == ACL_FLOAT)
        layernorm_kernel<float><<<e->outerCount, LT, 0, s>>>(
            (const float *)e->a->data, e->b ? (const float *)e->b->data : nullptr,
            e->c ? (const float *)e->c->data : nullptr, (float *)e->out->data, mo, ro, e->reduceCount, e->eps);
    else if (e->a->dtype == ACL_BF16)
        layernorm_kernel<__nv_bfloat16><<<e->outerCount, LT, 0, s>>>(
            (const __nv_bfloat16 *)e->a->data, e->b ? (const __nv_bfloat16 *)e->b->data : nullptr,
            e->c ? (const __nv_bfloat16 *)e->c->data : nullptr, (__nv_bfloat16 *)e->out->data, mo, ro, e->reduceCount, e->eps);
    else
        layernorm_kernel<__half><<<e->outerCount, LT, 0, s>>>(
            (const __half *)e->a->data, e->b ? (const __half *)e->b->data : nullptr,
            e->c ? (const __half *)e->c->data : nullptr, (__half *)e->out->data, mo, ro, e->reduceCount, e->eps);
    cudaError_t err = cudaGetLastError();
    delete e;
    return (err == cudaSuccess) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

} // extern "C"
