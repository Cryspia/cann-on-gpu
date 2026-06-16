// Softmax / LogSoftmax along any dim (fp32/fp16/bf16).
// Dims flattened into [outer, L, inner] (L = reduction dim, stride = inner); one thread per segment for stable softmax.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }

template <typename T, bool LOG>
__global__ void k_softmax_dim(const T *x, T *o, int64_t outer, int64_t L, int64_t inner) {
    int64_t seg = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= outer * inner) return;
    int64_t base = (seg / inner) * L * inner + (seg % inner);
    float mx = -1e30f;
    for (int64_t l = 0; l < L; ++l) mx = fmaxf(mx, (float)x[base + l * inner]);
    float sum = 0;
    for (int64_t l = 0; l < L; ++l) sum += expf((float)x[base + l * inner] - mx);
    float lsum = logf(sum);
    for (int64_t l = 0; l < L; ++l) {
        float e = (float)x[base + l * inner] - mx;
        o[base + l * inner] = (T)(LOG ? e - lsum : expf(e) / sum);
    }
}

// Last-dim softmax fast path (inner==1): one block per row; threads coalesce along L; two-pass warp+block reduction (max/sum).
template <typename T, bool LOG, int TB>
__global__ void k_softmax_row(const T *x, T *o, int64_t rows, int64_t L) {
    int64_t r = blockIdx.x; if (r >= rows) return;
    int t = threadIdx.x; int64_t base = r * L;
    __shared__ float sh[TB / 32];
    float mx = -1e30f;
    for (int64_t l = t; l < L; l += TB) mx = fmaxf(mx, (float)x[base + l]);
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) mx = fmaxf(mx, __shfl_down_sync(0xffffffffu, mx, o2));
    if ((t & 31) == 0) sh[t >> 5] = mx; __syncthreads();
    if (t == 0) { float m = -1e30f; for (int w = 0; w < TB / 32; w++) m = fmaxf(m, sh[w]); sh[0] = m; } __syncthreads();
    mx = sh[0]; __syncthreads();
    float sum = 0;
    for (int64_t l = t; l < L; l += TB) sum += __expf((float)x[base + l] - mx);
    #pragma unroll
    for (int o2 = 16; o2 > 0; o2 >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, o2);
    if ((t & 31) == 0) sh[t >> 5] = sum; __syncthreads();
    if (t == 0) { float s = 0; for (int w = 0; w < TB / 32; w++) s += sh[w]; sh[0] = s; } __syncthreads();
    sum = sh[0]; float lsum = logf(sum);
    for (int64_t l = t; l < L; l += TB) { float e = (float)x[base + l] - mx; o[base + l] = (T)(LOG ? e - lsum : __expf(e) / sum); }
}

void seg_layout(const aclTensor *t, int dim, int64_t &outer, int64_t &L, int64_t &inner) {
    int rank = (int)t->viewDims.size();
    outer = 1; for (int i = 0; i < dim; ++i) outer *= t->viewDims[i];
    L = t->viewDims[dim];
    inner = 1; for (int i = dim + 1; i < rank; ++i) inner *= t->viewDims[i];
}

aclnnStatus make_sm(int op, const aclTensor *self, int64_t dim, aclTensor *out, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    if (self->dtype != out->dtype || self->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    if (self->dtype != ACL_FLOAT && self->dtype != ACL_FLOAT16 && self->dtype != ACL_BF16) return ACLNN_ERR_PARAM_INVALID;
    if (!self->contiguous() || !out->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    int rank = (int)self->viewDims.size();
    int64_t d = dim < 0 ? dim + rank : dim;
    if (d < 0 || d >= rank) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = op; e->a = self; e->out = out; e->dim = d;
    *ex = e; return ACLNN_SUCCESS;
}

aclnnStatus run_sm(aclOpExecutor *e, bool logsm, cudaStream_t s) {
    int64_t outer, L, inner; seg_layout(e->a, (int)e->dim, outer, L, inner);
    int64_t g = nb(outer * inner);
    const void *x = e->a->data; void *o = e->out->data;
    if (inner == 1) {   // last dim: coalesced fast path (one block per row)
        constexpr int TB = 256;
        #define SMR(TYPE) (logsm ? k_softmax_row<TYPE,true,TB><<<(unsigned)outer,TB,0,s>>>((const TYPE*)x,(TYPE*)o,outer,L) \
                                 : k_softmax_row<TYPE,false,TB><<<(unsigned)outer,TB,0,s>>>((const TYPE*)x,(TYPE*)o,outer,L))
        switch (e->a->dtype) {
            case ACL_FLOAT:   SMR(float); break;
            case ACL_FLOAT16: SMR(__half); break;
            default:          SMR(__nv_bfloat16); break;
        }
        #undef SMR
        aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
        delete e; return st;
    }
    switch (e->a->dtype) {
        case ACL_FLOAT:   logsm ? k_softmax_dim<float,true><<<g,TH,0,s>>>((const float*)x,(float*)o,outer,L,inner)
                                : k_softmax_dim<float,false><<<g,TH,0,s>>>((const float*)x,(float*)o,outer,L,inner); break;
        case ACL_FLOAT16: logsm ? k_softmax_dim<__half,true><<<g,TH,0,s>>>((const __half*)x,(__half*)o,outer,L,inner)
                                : k_softmax_dim<__half,false><<<g,TH,0,s>>>((const __half*)x,(__half*)o,outer,L,inner); break;
        default:          logsm ? k_softmax_dim<__nv_bfloat16,true><<<g,TH,0,s>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)o,outer,L,inner)
                                : k_softmax_dim<__nv_bfloat16,false><<<g,TH,0,s>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)o,outer,L,inner); break;
    }
    aclnnStatus st = cudaGetLastError() == cudaSuccess ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    delete e; return st;
}
} // namespace

extern "C" {

aclnnStatus aclnnSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_sm(OP_SOFTMAX, self, dim, out, ex);
}
aclnnStatus aclnnSoftmax(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sm(e, false, (cudaStream_t)s); }

aclnnStatus aclnnLogSoftmaxGetWorkspaceSize(const aclTensor *self, int64_t dim, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (ws) *ws = 0; return make_sm(OP_SOFTMAX, self, dim, out, ex);
}
aclnnStatus aclnnLogSoftmax(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return run_sm(e, true, (cudaStream_t)s); }

} // extern "C"
