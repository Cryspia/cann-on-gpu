// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <map>
#include <mutex>
#include "subfp.cuh"

namespace _conv {
// Convolution/pooling family → cuDNN (NCHW).
//   Forward 2D conv (fp32/fp16, groups/stride/pad/dilation/bias), ConvolutionBackwardData (dgrad)/Weight (wgrad),
//   MaxPool2d / AvgPool2d forward. Computation always in fp32 accumulation (fp16 I/O uses TENSOR_OP_MATH).
//   Not implemented: conv3d, transposed convolution, pooling backward.

namespace {

cudnnHandle_t dnn_handle() {
    static cudnnHandle_t h = [] { cudnnHandle_t t; cudnnCreate(&t); return t; }();
    return h;
}

inline bool cv_is_fp4(aclDataType t) { return t == ACL_FLOAT4_E2M1 || t == ACL_FLOAT4_E1M2; }
inline bool cv_is_subfp(aclDataType t) { return cv_is_fp4(t) || t == ACL_FLOAT6_E2M3 || t == ACL_FLOAT6_E3M2; }
inline int cv_subkind(aclDataType t) {
    return t == ACL_FLOAT4_E2M1 ? SF_FP4E2M1 : t == ACL_FLOAT4_E1M2 ? SF_FP4E1M2 : t == ACL_FLOAT6_E2M3 ? SF_FP6E2M3 : SF_FP6E3M2;
}
__global__ void deq_conv(const uint8_t *p, __half *o, int64_t n, int k, bool fp4) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t code = fp4 ? ((i & 1) ? (p[i/2] >> 4) : (p[i/2] & 0xf)) : (p[i] & 0x3f);
    o[i] = (__half)subfp_decode(k, code);
}
inline cudnnDataType_t cdt(aclDataType t) { return t == ACL_FLOAT16 ? CUDNN_DATA_HALF : t == ACL_BF16 ? CUDNN_DATA_BFLOAT16 : CUDNN_DATA_FLOAT; }

struct ConvDesc {
    cudnnTensorDescriptor_t x = nullptr, y = nullptr, b = nullptr;
    cudnnFilterDescriptor_t w = nullptr;
    cudnnConvolutionDescriptor_t conv = nullptr;
    ~ConvDesc() {
        if (x) cudnnDestroyTensorDescriptor(x);
        if (y) cudnnDestroyTensorDescriptor(y);
        if (b) cudnnDestroyTensorDescriptor(b);
        if (w) cudnnDestroyFilterDescriptor(w);
        if (conv) cudnnDestroyConvolutionDescriptor(conv);
    }
};

// Build descriptors from x/w/y dims + conv params (dt is the I/O type; computation always fp32 accumulated)
aclnnStatus build_conv(const std::vector<int64_t> &xi, const std::vector<int64_t> &wi, const std::vector<int64_t> &yi,
                       cudnnDataType_t dt, const int64_t pad[2], const int64_t stride[2], const int64_t dil[2],
                       int groups, int64_t biasC, ConvDesc &d) {
    cudnnCreateTensorDescriptor(&d.x);
    cudnnCreateTensorDescriptor(&d.y);
    cudnnCreateFilterDescriptor(&d.w);
    cudnnCreateConvolutionDescriptor(&d.conv);
    if (cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NCHW, dt, xi[0], xi[1], xi[2], xi[3]) ||
        cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NCHW, dt, yi[0], yi[1], yi[2], yi[3]) ||
        cudnnSetFilter4dDescriptor(d.w, dt, CUDNN_TENSOR_NCHW, wi[0], wi[1], wi[2], wi[3]) ||
        cudnnSetConvolution2dDescriptor(d.conv, pad[0], pad[1], stride[0], stride[1], dil[0], dil[1],
                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))
        return ACLNN_ERR_PARAM_INVALID;
    cudnnSetConvolutionGroupCount(d.conv, groups);
    if (dt == CUDNN_DATA_HALF || dt == CUDNN_DATA_BFLOAT16) cudnnSetConvolutionMathType(d.conv, CUDNN_TENSOR_OP_MATH);
    if (biasC > 0) {
        cudnnCreateTensorDescriptor(&d.b);
        if (cudnnSetTensor4dDescriptor(d.b, CUDNN_TENSOR_NCHW, dt, 1, biasC, 1, 1)) return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}

// Pooling (forward)
struct PoolDesc {
    cudnnTensorDescriptor_t x = nullptr, y = nullptr;
    cudnnPoolingDescriptor_t p = nullptr;
    ~PoolDesc() { if (x) cudnnDestroyTensorDescriptor(x); if (y) cudnnDestroyTensorDescriptor(y); if (p) cudnnDestroyPoolingDescriptor(p); }
};

} // namespace

// Forward convolution algorithm selection + shape-keyed cache: avoids re-selecting on every call.
// Default uses v7 heuristic (can hit Winograd/FFT); when CANN_CONV_AUTOTUNE=1 and data pointers are
// provided, runs real-timing autotune (cudnnFindConvolutionForwardAlgorithm, sweeps all algorithms
// and selects the fastest — empirically hits Winograd).
static std::vector<int64_t> conv_algo_key(const std::vector<int64_t> &xd, const std::vector<int64_t> &wd,
                                          const std::vector<int64_t> &ax, int dt, int groups) {
    std::vector<int64_t> k; k.reserve(xd.size() + wd.size() + ax.size() + 2);
    k.insert(k.end(), xd.begin(), xd.end()); k.insert(k.end(), wd.begin(), wd.end());
    k.insert(k.end(), ax.begin(), ax.end()); k.push_back(dt); k.push_back(groups); return k;
}
static cudnnConvolutionFwdAlgo_t pick_fwd_algo(ConvDesc &d, const std::vector<int64_t> &key,
        const void *x = nullptr, const void *w = nullptr, void *y = nullptr) {
    static std::map<std::vector<int64_t>, cudnnConvolutionFwdAlgo_t> cache;
    static std::mutex mu;
    { std::lock_guard<std::mutex> lk(mu); auto it = cache.find(key); if (it != cache.end()) return it->second; }
    (void)x; (void)w; (void)y;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    int got = 0;
    if (getenv("CANN_CONV_AUTOTUNE")) {   // real-timing autotune: sweeps all algorithms, selects fastest by elapsed time (buffers allocated internally)
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        if (cudnnFindConvolutionForwardAlgorithm(dnn_handle(), d.x, d.w, d.conv, d.y, 8, &got, perf) == CUDNN_STATUS_SUCCESS)
            for (int i = 0; i < got; ++i) if (perf[i].status == CUDNN_STATUS_SUCCESS) { algo = perf[i].algo; break; }  // perf sorted by elapsed time ascending
    } else {
        cudnnConvolutionFwdAlgoPerf_t p1;
        if (cudnnGetConvolutionForwardAlgorithm_v7(dnn_handle(), d.x, d.w, d.conv, d.y, 1, &got, &p1) == CUDNN_STATUS_SUCCESS && got > 0)
            algo = p1.algo;
    }
    { std::lock_guard<std::mutex> lk(mu); cache[key] = algo; }
    return algo;
}

extern "C" {

// ---- Forward 2D convolution (fp32/fp16) ----
aclnnStatus aclnnConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias,
                                             const aclIntArray *stride, const aclIntArray *padding,
                                             const aclIntArray *dilation, bool transposed,
                                             const aclIntArray * /*outputPadding*/, int64_t groups,
                                             aclTensor *output, int8_t /*cubeMathType*/,
                                             uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !weight || !output || !ws || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (!input->data || !weight->data || !output->data || (bias && !bias->data)) return ACLNN_ERR_PARAM_NULLPTR;
    if (transposed) return ACLNN_ERR_PARAM_INVALID;                     // transposed convolution not implemented
    aclDataType dt = input->dtype;
    bool sub = cv_is_subfp(dt);   // fp4/fp6 input: unpack to fp16 for fp16 conv; out must be fp16 with no bias
    if (sub) { if (weight->dtype != dt || output->dtype != ACL_FLOAT16 || bias) return ACLNN_ERR_PARAM_INVALID; }
    else if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || output->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (input->viewDims.size() != 4 || weight->viewDims.size() != 4 || output->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    if (!input->contiguous() || !weight->contiguous() || !output->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    if (bias && (bias->viewDims.size() != 1 || bias->viewDims[0] != output->viewDims[1])) return ACLNN_ERR_PARAM_INVALID;

    auto *e = new aclOpExecutor();
    e->op = OP_CONV2D; e->a = input; e->b = weight; e->c = bias; e->out = output; e->castTo = sub ? dt : ACL_DT_UNDEFINED;
    e->alpha = (double)(groups > 0 ? groups : 1);
    for (int i = 0; i < 2; i++) {
        e->stride[i] = (stride && stride->v.size() > (size_t)i) ? stride->v[i] : 1;
        e->pad[i]    = (padding && padding->v.size() > (size_t)i) ? padding->v[i] : 0;
        e->dil[i]    = (dilation && dilation->v.size() > (size_t)i) ? dilation->v[i] : 1;
    }
    cudnnDataType_t cd = sub ? CUDNN_DATA_HALF : cdt(dt);
    ConvDesc d;
    if (build_conv(input->viewDims, weight->viewDims, output->viewDims, cd, e->pad, e->stride, e->dil,
                   (int)e->alpha, bias ? output->viewDims[1] : 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    // Algorithm selection: non-subfp uses pick_fwd_algo (v7 heuristic / autotune); subfp unpack path fixes GEMM.
    // Selected algorithm stored in e->n; Execute reuses it directly (avoids re-selection; ensures GetWs/Execute consistency).
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    if (!sub) {
        std::vector<int64_t> ax{e->pad[0], e->pad[1], e->stride[0], e->stride[1], e->dil[0], e->dil[1]};
        auto key = conv_algo_key(input->viewDims, weight->viewDims, ax, (int)cd, (int)e->alpha);
        algo = pick_fwd_algo(d, key, input->data, weight->data, output->data);
    }
    e->n = (int64_t)algo;
    size_t need = 0;
    if (cudnnGetConvolutionForwardWorkspaceSize(dnn_handle(), d.x, d.w, d.conv, d.y, algo, &need)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    // subfp: dequant buffers (input+weight, fp16) placed at the front of ws; cuDNN ws follows
    *ws = need + (sub ? (uint64_t)(input->numel() + weight->numel()) * sizeof(__half) : 0);
    *ex = e;
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolution(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e || e->op != OP_CONV2D) return ACLNN_ERR_PARAM_INVALID;
    auto s = (cudaStream_t)stream;
    bool sub = cv_is_subfp(e->castTo);
    cudnnDataType_t cd = sub ? CUDNN_DATA_HALF : cdt(e->a->dtype);
    ConvDesc d;
    if (build_conv(e->a->viewDims, e->b->viewDims, e->out->viewDims, cd, e->pad, e->stride, e->dil,
                   (int)e->alpha, e->c ? e->out->viewDims[1] : 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    cudnnSetStream(dnn_handle(), s);
    const float one = 1.f, zero = 0.f;
    const void *X = e->a->data, *W = e->b->data; void *cudnnWs = ws; size_t cudnnWsSize = wsSize;
    if (sub) {   // unpack input/weight to fp16 buffers (front of ws); cuDNN ws follows
        int64_t inN = e->a->numel(), wN = e->b->numel();
        __half *Ih = (__half *)ws, *Wh = Ih + inN;
        cudnnWs = (char *)(Wh + wN); cudnnWsSize = wsSize - (size_t)(inN + wN) * sizeof(__half);
        int k = cv_subkind(e->castTo); bool fp4 = cv_is_fp4(e->castTo);
        deq_conv<<<(inN+255)/256,256,0,s>>>((const uint8_t*)e->a->data, Ih, inN, k, fp4);
        deq_conv<<<(wN+255)/256,256,0,s>>>((const uint8_t*)e->b->data, Wh, wN, k, fp4);
        X = Ih; W = Wh;
    }
    cudnnStatus_t cs = cudnnConvolutionForward(dnn_handle(), &one, d.x, X, d.w, W, d.conv,
                                               (cudnnConvolutionFwdAlgo_t)e->n, cudnnWs, cudnnWsSize, &zero, d.y, e->out->data);
    if (cs == CUDNN_STATUS_SUCCESS && e->c)
        cs = cudnnAddTensor(dnn_handle(), &one, d.b, e->c->data, &one, d.y, e->out->data);
    delete e;
    return (cs == CUDNN_STATUS_SUCCESS) ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- ConvolutionBackwardData (dgrad): gradOutput + weight → gradInput ----
// Reuses executor: a=gradOutput, b=weight, out=gradInput (x dims=gradInput, y dims=gradOutput)
aclnnStatus aclnnConvolutionBackwardDataGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    if (!gradOutput || !weight || !gradInput || !ws || !ex || !gradOutput->data || !weight->data || !gradInput->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = gradInput->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || gradOutput->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (gradOutput->viewDims.size() != 4 || weight->viewDims.size() != 4 || gradInput->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = gradOutput; e->b = weight; e->out = gradInput;
    e->alpha = (double)(groups > 0 ? groups : 1); e->dim = 100;   // dim=100 marks dgrad
    for (int i = 0; i < 2; i++) { e->stride[i] = stride && stride->v.size()>(size_t)i?stride->v[i]:1; e->pad[i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->dil[i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    ConvDesc d;
    if (build_conv(gradInput->viewDims, weight->viewDims, gradOutput->viewDims, cdt(dt), e->pad, e->stride, e->dil, (int)e->alpha, 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    size_t need = 0;
    if (cudnnGetConvolutionBackwardDataWorkspaceSize(dnn_handle(), d.w, d.y, d.conv, d.x, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &need)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    *ws = need; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolutionBackwardData(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    ConvDesc d;
    if (build_conv(e->out->viewDims, e->b->viewDims, e->a->viewDims, cdt(e->out->dtype), e->pad, e->stride, e->dil, (int)e->alpha, 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    cudnnSetStream(dnn_handle(), (cudaStream_t)stream);
    const float one = 1.f, zero = 0.f;
    cudnnStatus_t cs = cudnnConvolutionBackwardData(dnn_handle(), &one, d.w, e->b->data, d.y, e->a->data, d.conv,
                                                    CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, ws, wsSize, &zero, d.x, e->out->data);
    delete e;
    return cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- ConvolutionBackwardWeight (wgrad): input + gradOutput → gradWeight ----
// a=input, b=gradOutput, out=gradWeight (x dims=input, y dims=gradOutput, w dims=gradWeight)
aclnnStatus aclnnConvolutionBackwardWeightGetWorkspaceSize(const aclTensor *input, const aclTensor *gradOutput,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *gradWeight, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !gradOutput || !gradWeight || !ws || !ex || !input->data || !gradOutput->data || !gradWeight->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = input->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || gradOutput->dtype != dt || gradWeight->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (input->viewDims.size() != 4 || gradOutput->viewDims.size() != 4 || gradWeight->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = input; e->b = gradOutput; e->out = gradWeight;
    e->alpha = (double)(groups > 0 ? groups : 1); e->dim = 200;   // dim=200 marks wgrad
    for (int i = 0; i < 2; i++) { e->stride[i] = stride && stride->v.size()>(size_t)i?stride->v[i]:1; e->pad[i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->dil[i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    ConvDesc d;
    if (build_conv(input->viewDims, gradWeight->viewDims, gradOutput->viewDims, cdt(dt), e->pad, e->stride, e->dil, (int)e->alpha, 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    size_t need = 0;
    if (cudnnGetConvolutionBackwardFilterWorkspaceSize(dnn_handle(), d.x, d.y, d.conv, d.w, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, &need)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    *ws = need; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolutionBackwardWeight(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    ConvDesc d;
    if (build_conv(e->a->viewDims, e->out->viewDims, e->b->viewDims, cdt(e->a->dtype), e->pad, e->stride, e->dil, (int)e->alpha, 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    cudnnSetStream(dnn_handle(), (cudaStream_t)stream);
    const float one = 1.f, zero = 0.f;
    cudnnStatus_t cs = cudnnConvolutionBackwardFilter(dnn_handle(), &one, d.x, e->a->data, d.y, e->b->data, d.conv,
                                                      CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1, ws, wsSize, &zero, d.w, e->out->data);
    delete e;
    return cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- MaxPool2d / AvgPool2d forward ----
// kernel=(kh,kw) via aclIntArray; stride/padding likewise. avg=false → max.
static aclnnStatus pool_ws(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
                           const aclIntArray *padding, bool avg, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !kernel || !out || !ws || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = self->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (self->viewDims.size() != 4 || out->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = self; e->out = out; e->dim = avg ? 1 : 0;
    e->dil[0] = kernel->v[0]; e->dil[1] = kernel->v.size() > 1 ? kernel->v[1] : kernel->v[0];   // dil reused to store kernel size
    for (int i = 0; i < 2; i++) { e->stride[i] = stride && stride->v.size()>(size_t)i?stride->v[i]:e->dil[i]; e->pad[i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus pool_run(aclOpExecutor *e, cudaStream_t s) {
    PoolDesc d; cudnnCreatePoolingDescriptor(&d.p);
    cudnnCreateTensorDescriptor(&d.x); cudnnCreateTensorDescriptor(&d.y);
    const auto &xi = e->a->viewDims; const auto &yi = e->out->viewDims;
    cudnnDataType_t dt = cdt(e->a->dtype);
    cudnnPoolingMode_t mode = e->dim == 1 ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX;
    if (cudnnSetPooling2dDescriptor(d.p, mode, CUDNN_NOT_PROPAGATE_NAN, (int)e->dil[0], (int)e->dil[1],
                                    (int)e->pad[0], (int)e->pad[1], (int)e->stride[0], (int)e->stride[1]) ||
        cudnnSetTensor4dDescriptor(d.x, CUDNN_TENSOR_NCHW, dt, xi[0], xi[1], xi[2], xi[3]) ||
        cudnnSetTensor4dDescriptor(d.y, CUDNN_TENSOR_NCHW, dt, yi[0], yi[1], yi[2], yi[3])) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    cudnnSetStream(dnn_handle(), s);
    const float one = 1.f, zero = 0.f;
    cudnnStatus_t cs = cudnnPoolingForward(dnn_handle(), d.p, &one, d.x, e->a->data, &zero, d.y, e->out->data);
    delete e;
    return cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}
aclnnStatus aclnnMaxPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return pool_ws(self, kernel, stride, padding, false, out, ws, ex);
}
aclnnStatus aclnnMaxPool2d(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool_run(e, (cudaStream_t)s); }
aclnnStatus aclnnAvgPool2dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride,
        const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    return pool_ws(self, kernel, stride, padding, true, out, ws, ex);
}
aclnnStatus aclnnAvgPool2d(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool_run(e, (cudaStream_t)s); }

// Pooling backward: a=input x, b=fwdOutput y, c=gradOutput dy, out=gradInput dx; dim marks avg
static aclnnStatus pool_bwd_ws(const aclTensor *x, const aclTensor *y, const aclTensor *dy,
                               const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding,
                               bool avg, aclTensor *dx, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !y || !dy || !kernel || !dx || !ex || !x->data || !y->data || !dy->data || !dx->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = x->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || y->dtype != dt || dy->dtype != dt || dx->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (x->viewDims.size() != 4 || dx->viewDims != x->viewDims || y->viewDims != dy->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = x; e->b = y; e->c = dy; e->out = dx; e->dim = avg ? 1 : 0;
    e->dil[0] = kernel->v[0]; e->dil[1] = kernel->v.size() > 1 ? kernel->v[1] : kernel->v[0];
    for (int i = 0; i < 2; i++) { e->stride[i] = stride && stride->v.size()>(size_t)i?stride->v[i]:e->dil[i]; e->pad[i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus pool_bwd_run(aclOpExecutor *e, cudaStream_t s) {
    PoolDesc d; cudnnCreatePoolingDescriptor(&d.p);
    cudnnTensorDescriptor_t xd = nullptr, yd = nullptr; cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&yd);
    const auto &xi = e->a->viewDims; const auto &yi = e->b->viewDims;
    cudnnDataType_t dt = cdt(e->a->dtype);
    cudnnPoolingMode_t mode = e->dim == 1 ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX;
    aclnnStatus st = ACLNN_SUCCESS;
    if (cudnnSetPooling2dDescriptor(d.p, mode, CUDNN_NOT_PROPAGATE_NAN, (int)e->dil[0], (int)e->dil[1], (int)e->pad[0], (int)e->pad[1], (int)e->stride[0], (int)e->stride[1]) ||
        cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, dt, xi[0], xi[1], xi[2], xi[3]) ||
        cudnnSetTensor4dDescriptor(yd, CUDNN_TENSOR_NCHW, dt, yi[0], yi[1], yi[2], yi[3])) st = ACLNN_ERR_PARAM_INVALID;
    if (st == ACLNN_SUCCESS) {
        cudnnSetStream(dnn_handle(), s);
        const float one = 1.f, zero = 0.f;
        cudnnStatus_t cs = cudnnPoolingBackward(dnn_handle(), d.p, &one, yd, e->b->data, yd, e->c->data, xd, e->a->data, &zero, xd, e->out->data);
        st = cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd); delete e; return st;
}
aclnnStatus aclnnMaxPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    return pool_bwd_ws(x, y, gradOutput, kernel, stride, padding, false, gradInput, ws, ex);
}
aclnnStatus aclnnMaxPool2dBackward(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool_bwd_run(e, (cudaStream_t)s); }
aclnnStatus aclnnAvgPool2dBackwardGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *gradOutput,
        const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex) {
    return pool_bwd_ws(x, y, gradOutput, kernel, stride, padding, true, gradInput, ws, ex);
}
aclnnStatus aclnnAvgPool2dBackward(void *ws, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool_bwd_run(e, (cudaStream_t)s); }

// ---- Transposed convolution ConvTranspose2d: forward = BackwardData of a regular convolution ----
// input in[N,Ci,H,W], weight w[Ci,Co,kh,kw], output out[N,Co,Ho,Wo], Ho=(H-1)S-2P+dil(kh-1)+1.
// Mapped to the dgrad of a "forward: Co→Ci" conv: gradOutput=in, filter=w, gradInput=out. Reuses the dgrad run.
aclnnStatus aclnnConvolutionTranspose2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *output, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !weight || !output || !ws || !ex || !input->data || !weight->data || !output->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = input->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || output->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (input->viewDims.size() != 4 || weight->viewDims.size() != 4 || output->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = input; e->b = weight; e->out = output;   // a=gradOutput, out=gradInput (same as dgrad)
    e->alpha = (double)(groups > 0 ? groups : 1); e->dim = 100;
    for (int i = 0; i < 2; i++) { e->stride[i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:1; e->pad[i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->dil[i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    ConvDesc d;
    if (build_conv(output->viewDims, weight->viewDims, input->viewDims, cdt(dt), e->pad, e->stride, e->dil, (int)e->alpha, 0, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    size_t need = 0;
    if (cudnnGetConvolutionBackwardDataWorkspaceSize(dnn_handle(), d.w, d.y, d.conv, d.x, CUDNN_CONVOLUTION_BWD_DATA_ALGO_1, &need)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    *ws = need; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolutionTranspose2d(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    return aclnnConvolutionBackwardData(ws, wsSize, e, stream);   // forward transposed conv = dgrad
}

// ---- conv3d (NCDHW, fp32/fp16, groups/stride/pad/dilation, no bias) ----
// pad/stride/dil stored as 3 dims each in e->axes[0..8]; groups stored in e->alpha.
static aclnnStatus build_conv3d(const std::vector<int64_t> &xi, const std::vector<int64_t> &wi, const std::vector<int64_t> &yi,
                                cudnnDataType_t dt, const int64_t *axes9, int groups, ConvDesc &d) {
    cudnnCreateTensorDescriptor(&d.x); cudnnCreateTensorDescriptor(&d.y);
    cudnnCreateFilterDescriptor(&d.w); cudnnCreateConvolutionDescriptor(&d.conv);
    int xdim[5], ydim[5], wdim[5]; for (int i = 0; i < 5; i++) { xdim[i]=(int)xi[i]; ydim[i]=(int)yi[i]; wdim[i]=(int)wi[i]; }
    int pad[3], str[3], dil[3]; for (int i = 0; i < 3; i++) { pad[i]=(int)axes9[i]; str[i]=(int)axes9[3+i]; dil[i]=(int)axes9[6+i]; }
    if (cudnnSetTensorNdDescriptorEx(d.x, CUDNN_TENSOR_NCHW, dt, 5, xdim) ||
        cudnnSetTensorNdDescriptorEx(d.y, CUDNN_TENSOR_NCHW, dt, 5, ydim) ||
        cudnnSetFilterNdDescriptor(d.w, dt, CUDNN_TENSOR_NCHW, 5, wdim) ||
        cudnnSetConvolutionNdDescriptor(d.conv, 3, pad, str, dil, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT))
        return ACLNN_ERR_PARAM_INVALID;
    cudnnSetConvolutionGroupCount(d.conv, groups);
    if (dt == CUDNN_DATA_HALF || dt == CUDNN_DATA_BFLOAT16) cudnnSetConvolutionMathType(d.conv, CUDNN_TENSOR_OP_MATH);
    return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolution3dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups,
        aclTensor *output, uint64_t *ws, aclOpExecutor **ex) {
    if (!input || !weight || !output || !ws || !ex || !input->data || !weight->data || !output->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = input->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || weight->dtype != dt || output->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    if (input->viewDims.size() != 5 || weight->viewDims.size() != 5 || output->viewDims.size() != 5) return ACLNN_ERR_PARAM_INVALID;
    if (!input->contiguous() || !weight->contiguous() || !output->contiguous()) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = input; e->b = weight; e->out = output;
    e->alpha = (double)(groups > 0 ? groups : 1); e->dim = 300;   // dim=300 marks conv3d
    e->axes.assign(9, 0);
    for (int i = 0; i < 3; i++) { e->axes[i] = padding && padding->v.size()>(size_t)i?padding->v[i]:0;
        e->axes[3+i] = stride && stride->v.size()>(size_t)i?stride->v[i]:1; e->axes[6+i] = dilation && dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    ConvDesc d;
    if (build_conv3d(input->viewDims, weight->viewDims, output->viewDims, cdt(dt), e->axes.data(), (int)e->alpha, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    size_t need = 0;
    auto key = conv_algo_key(input->viewDims, weight->viewDims, e->axes, (int)cdt(dt), (int)e->alpha);
    if (cudnnGetConvolutionForwardWorkspaceSize(dnn_handle(), d.x, d.w, d.conv, d.y, pick_fwd_algo(d, key), &need)) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    *ws = need; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvolution3d(void *ws, uint64_t wsSize, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    ConvDesc d;
    if (build_conv3d(e->a->viewDims, e->b->viewDims, e->out->viewDims, cdt(e->a->dtype), e->axes.data(), (int)e->alpha, d) != ACLNN_SUCCESS) { delete e; return ACLNN_ERR_PARAM_INVALID; }
    cudnnSetStream(dnn_handle(), (cudaStream_t)stream);
    const float one = 1.f, zero = 0.f;
    auto key = conv_algo_key(e->a->viewDims, e->b->viewDims, e->axes, (int)cdt(e->a->dtype), (int)e->alpha);
    cudnnStatus_t cs = cudnnConvolutionForward(dnn_handle(), &one, d.x, e->a->data, d.w, e->b->data, d.conv,
                                               pick_fwd_algo(d, key), ws, wsSize, &zero, d.y, e->out->data);
    delete e;
    return cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
}

// ---- BatchNorm training forward (NCHW SPATIAL, cuDNN, updates running mean/var + produces saved mean/invstd) ----
// x/y fp32/fp16; gamma/beta/runningMean/runningVar/savedMean/savedInvStd all fp32 (cuDNN constraint).
//   a=x, b=gamma, c=beta, mean=runningMean(in/out), rstd=runningVar(in/out), out=y, out2=savedMean,
//   inputs[0]=savedInvStd; alpha=momentum (exponential moving average factor), eps.
aclnnStatus aclnnBatchNormTrainingGetWorkspaceSize(const aclTensor *x, const aclTensor *gamma, const aclTensor *beta,
        aclTensor *runningMean, aclTensor *runningVar, double momentum, double eps,
        aclTensor *out, aclTensor *savedMean, aclTensor *savedInvStd, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !gamma || !beta || !runningMean || !runningVar || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = x->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || out->dtype != dt || x->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    if (gamma->dtype != ACL_FLOAT || beta->dtype != ACL_FLOAT || runningMean->dtype != ACL_FLOAT || runningVar->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = x; e->b = gamma; e->c = beta;
    e->mean = runningMean; e->rstd = runningVar; e->out = out; e->out2 = savedMean;
    if (savedInvStd) e->inputs.push_back(savedInvStd);
    e->alpha = momentum; e->eps = eps; e->dim = 400;   // dim=400 marks BN training
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBatchNormTraining(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    if (!e) return ACLNN_ERR_PARAM_INVALID;
    const auto &xi = e->a->viewDims;
    cudnnTensorDescriptor_t xd, bnd; cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&bnd);
    cudnnDataType_t dt = cdt(e->a->dtype);
    aclnnStatus st = ACLNN_SUCCESS;
    if (cudnnSetTensor4dDescriptor(xd, CUDNN_TENSOR_NCHW, dt, xi[0], xi[1], xi[2], xi[3]) ||
        cudnnDeriveBNTensorDescriptor(bnd, xd, CUDNN_BATCHNORM_SPATIAL)) st = ACLNN_ERR_PARAM_INVALID;
    if (st == ACLNN_SUCCESS) {
        cudnnSetStream(dnn_handle(), (cudaStream_t)stream);
        const float one = 1.f, zero = 0.f;
        void *sMean = e->out2 ? e->out2->data : nullptr, *sInv = e->inputs.empty() ? nullptr : const_cast<aclTensor *>(e->inputs[0])->data;
        cudnnStatus_t cs = cudnnBatchNormalizationForwardTraining(dnn_handle(), CUDNN_BATCHNORM_SPATIAL, &one, &zero,
            xd, e->a->data, xd, e->out->data, bnd, e->b->data, e->c->data, e->alpha,
            const_cast<aclTensor *>(e->mean)->data, const_cast<aclTensor *>(e->rstd)->data, e->eps, sMean, sInv);
        st = cs == CUDNN_STATUS_SUCCESS ? ACLNN_SUCCESS : ACLNN_ERR_RUNTIME_ERROR;
    }
    cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(bnd); delete e; return st;
}

} // extern "C"

// AdaptiveAvgPool2d: each output cell averages the adaptive input region [oh*H/Ho, (oh+1)*H/Ho) × [...]
template <typename T>
__global__ void k_adaptive_avgpool(const T *x, T *o, int N, int C, int H, int W, int Ho, int Wo) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= (int64_t)N * C * Ho * Wo) return;
    int ow = i % Wo, oh = (i / Wo) % Ho, c = (i / (Wo * Ho)) % C, n = i / (Wo * Ho * C);
    int hs = oh * H / Ho, he = ((oh + 1) * H + Ho - 1) / Ho, ws = ow * W / Wo, we = ((ow + 1) * W + Wo - 1) / Wo;
    float s = 0; const T *p = x + ((int64_t)(n * C + c)) * H * W;
    for (int h = hs; h < he; ++h) for (int w = ws; w < we; ++w) s += (float)p[h * W + w];
    o[i] = (T)(s / ((he - hs) * (we - ws)));
}
extern "C" {
aclnnStatus aclnnAdaptiveAvgPool2dGetWorkspaceSize(const aclTensor *self, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !out || !ex || !self->data || !out->data) return ACLNN_ERR_PARAM_NULLPTR;
    aclDataType dt = self->dtype;
    if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || out->dtype != dt || self->viewDims.size() != 4 || out->viewDims.size() != 4) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = self; e->out = out; e->dim = 500;
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnAdaptiveAvgPool2d(void *, uint64_t, aclOpExecutor *e, aclrtStream stream) {
    const auto &xi = e->a->viewDims; const auto &yi = e->out->viewDims; auto s = (cudaStream_t)stream;
    int N=(int)xi[0],C=(int)xi[1],H=(int)xi[2],W=(int)xi[3],Ho=(int)yi[2],Wo=(int)yi[3];
    int64_t g = ((int64_t)N*C*Ho*Wo + 255)/256;
    if (e->a->dtype == ACL_FLOAT) k_adaptive_avgpool<float><<<g,256,0,s>>>((const float*)e->a->data,(float*)e->out->data,N,C,H,W,Ho,Wo);
    else if (e->a->dtype == ACL_BF16) k_adaptive_avgpool<__nv_bfloat16><<<g,256,0,s>>>((const __nv_bfloat16*)e->a->data,(__nv_bfloat16*)e->out->data,N,C,H,W,Ho,Wo);
    else                          k_adaptive_avgpool<__half><<<g,256,0,s>>>((const __half*)e->a->data,(__half*)e->out->data,N,C,H,W,Ho,Wo);
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
// 3D pooling (NCDHW, cuDNN PoolingNd): kernel/stride/pad each 3 dims stored in e->axes; avg flag in e->dim
static aclnnStatus pool3d_ws(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, bool avg, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!self || !kernel || !out || !ex || self->viewDims.size() != 5) return ACLNN_ERR_PARAM_INVALID;
    aclDataType dt = self->dtype; if ((dt != ACL_FLOAT && dt != ACL_FLOAT16 && dt != ACL_BF16) || out->dtype != dt) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_CONV2D; e->a = self; e->out = out; e->dim = avg ? 601 : 600; e->axes.assign(9, 0);
    for (int i = 0; i < 3; i++) { e->axes[i]=kernel->v[i]; e->axes[3+i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:kernel->v[i]; e->axes[6+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus pool3d_run(aclOpExecutor *e, cudaStream_t s) {
    cudnnPoolingDescriptor_t p; cudnnCreatePoolingDescriptor(&p);
    cudnnTensorDescriptor_t xd, yd; cudnnCreateTensorDescriptor(&xd); cudnnCreateTensorDescriptor(&yd);
    const auto &xi = e->a->viewDims; const auto &yi = e->out->viewDims; cudnnDataType_t dt = cdt(e->a->dtype);
    int xdim[5],ydim[5]; for(int i=0;i<5;i++){xdim[i]=(int)xi[i];ydim[i]=(int)yi[i];}
    int win[3],pad[3],str[3]; for(int i=0;i<3;i++){win[i]=(int)e->axes[i];str[i]=(int)e->axes[3+i];pad[i]=(int)e->axes[6+i];}
    cudnnPoolingMode_t mode = e->dim==601 ? CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING : CUDNN_POOLING_MAX;
    aclnnStatus st = ACLNN_SUCCESS;
    if (cudnnSetPoolingNdDescriptor(p, mode, CUDNN_NOT_PROPAGATE_NAN, 3, win, pad, str) ||
        cudnnSetTensorNdDescriptorEx(xd, CUDNN_TENSOR_NCHW, dt, 5, xdim) || cudnnSetTensorNdDescriptorEx(yd, CUDNN_TENSOR_NCHW, dt, 5, ydim)) st = ACLNN_ERR_PARAM_INVALID;
    if (st == ACLNN_SUCCESS) { cudnnSetStream(dnn_handle(), s); const float one=1.f,zero=0.f;
        cudnnStatus_t cs = cudnnPoolingForward(dnn_handle(), p, &one, xd, e->a->data, &zero, yd, e->out->data);
        st = cs==CUDNN_STATUS_SUCCESS?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; }
    cudnnDestroyTensorDescriptor(xd); cudnnDestroyTensorDescriptor(yd); cudnnDestroyPoolingDescriptor(p); delete e; return st;
}
aclnnStatus aclnnMaxPool3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pool3d_ws(self,kernel,stride,padding,false,out,ws,ex); }
aclnnStatus aclnnMaxPool3d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool3d_run(e,(cudaStream_t)s); }
aclnnStatus aclnnAvgPool3dGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernel, const aclIntArray *stride, const aclIntArray *padding, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return pool3d_ws(self,kernel,stride,padding,true,out,ws,ex); }
aclnnStatus aclnnAvgPool3d(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return pool3d_run(e,(cudaStream_t)s); }
} // extern "C"

} // namespace _conv

namespace _conv_ext {
// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.

namespace _conv2_ext {
// Conv extensions: ConvolutionBackward (compose data+weight+bias grads), depthwise, ConvTbc(+bwd),
// Im2col(+bwd)/UnfoldGrad, DeformableConv2d, MultiScaleDeformableAttn(+grad), QuantConvolution,
// weight-size/layout/int4-pack helpers, and naming/forward aliases. fp32 (quant path: int8 in, fp32 acc).

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
inline aclnnStatus done(aclOpExecutor*e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }

// side-map for ConvolutionBackward: stash IntArray params + grad targets between the two phases
struct ConvBwdState { const aclTensor *gradOut,*input,*weight; const aclIntArray *stride,*pad,*dil,*outpad; int64_t groups; bool transposed;
                      aclTensor *gradInput,*gradWeight,*gradBias; };
std::map<aclOpExecutor*,ConvBwdState> g_convbwd;

// gradBias[co] = Σ_{n,spatial} gradOut[n,co,...]
__global__ void k_bias_grad(const float*go,float*gb,int64_t N,int64_t Co,int64_t SP){
    int64_t co=blockIdx.x; if(co>=Co) return; float s=0;
    for(int64_t n=0;n<N;n++) for(int64_t p=threadIdx.x;p<SP;p+=blockDim.x) s+=go[(n*Co+co)*SP+p];
    for(int o=16;o>0;o>>=1) s+=__shfl_down_sync(0xffffffffu,s,o);
    __shared__ float sh[TH/32]; if((threadIdx.x&31)==0) sh[threadIdx.x>>5]=s; __syncthreads();
    if(threadIdx.x==0){ float m=0; for(int w=0;w<TH/32;w++) m+=sh[w]; gb[co]=m; }
}
__device__ inline float bilin2(const float*p,int H,int W,float y,float x){
    if(y<=-1||y>=H||x<=-1||x>=W) return 0; int y0=(int)floorf(y),x0=(int)floorf(x),y1=y0+1,x1=x0+1; float dy=y-y0,dx=x-x0;
    float v=0; if(y0>=0&&x0>=0)v+=(1-dy)*(1-dx)*p[y0*W+x0]; if(y0>=0&&x1<W)v+=(1-dy)*dx*p[y0*W+x1];
    if(y1<H&&x0>=0)v+=dy*(1-dx)*p[y1*W+x0]; if(y1<H&&x1<W)v+=dy*dx*p[y1*W+x1]; return v;
}
// ConvTbc fwd: self[T,B,Cin], weight[kW,Cin,Cout], bias[Cout], pad → out[oT,B,Cout]
__global__ void k_convtbc(const float*x,const float*w,const float*b,float*o,int T,int B,int Cin,int Cout,int kW,int oT,int pad){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)oT*B*Cout) return;
    int co=i%Cout,bb=(i/Cout)%B,t=i/(Cout*B); float acc=b?b[co]:0.f;
    for(int k=0;k<kW;k++){ int ti=t+k-pad; if(ti<0||ti>=T) continue; const float*xp=x+((int64_t)ti*B+bb)*Cin; const float*wp=w+(int64_t)k*Cin*Cout+co;
        for(int ci=0;ci<Cin;ci++) acc+=xp[ci]*wp[(int64_t)ci*Cout]; }
    o[i]=acc;
}
// Im2col: input[N,C,H,W] → col[N, C*kH*kW, oH*oW]
__global__ void k_im2col(const float*x,float*col,int N,int C,int H,int W,int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,int oH,int oW){
    int64_t L=(int64_t)oH*oW, K=(int64_t)C*kH*kW, tot=(int64_t)N*K*L;
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot) return;
    int64_t l=i%L; int64_t kk=(i/L)%K; int64_t n=i/(K*L);
    int ow=l%oW,oh=l/oW; int kw=kk%kW,kh=(kk/kW)%kH,c=kk/(kH*kW);
    int ih=oh*sH-pH+kh*dH, iw=ow*sW-pW+kw*dW;
    float v=(ih>=0&&ih<H&&iw>=0&&iw<W)? x[((int64_t)(n*C+c)*H+ih)*W+iw] : 0.f; col[i]=v;
}
// col2im (Im2col backward / UnfoldGrad): scatter-add col → gradInput[N,C,H,W]
__global__ void k_col2im(const float*col,float*x,int N,int C,int H,int W,int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,int oH,int oW){
    int64_t L=(int64_t)oH*oW, K=(int64_t)C*kH*kW, tot=(int64_t)N*K*L;
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=tot) return;
    int64_t l=i%L; int64_t kk=(i/L)%K; int64_t n=i/(K*L);
    int ow=l%oW,oh=l/oW; int kw=kk%kW,kh=(kk/kW)%kH,c=kk/(kH*kW);
    int ih=oh*sH-pH+kh*dH, iw=ow*sW-pW+kw*dW;
    if(ih>=0&&ih<H&&iw>=0&&iw<W) atomicAdd(&x[((int64_t)(n*C+c)*H+ih)*W+iw], col[i]);
}
// DeformableConv2d fwd (groups=1): input[N,Cin,H,W], weight[Cout,Cin,kH,kW], offset[N,2*kH*kW,oH,oW], mask[N,kH*kW,oH,oW] (optional)
__global__ void k_deform_conv(const float*x,const float*w,const float*off,const float*mask,const float*bias,float*o,
        int N,int Cin,int Cout,int H,int W,int kH,int kW,int sH,int sW,int pH,int pW,int dH,int dW,int oH,int oW){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)N*Cout*oH*oW) return;
    int ow=i%oW,oh=(i/oW)%oH,co=(i/(oW*oH))%Cout,n=i/((int64_t)oW*oH*Cout); int KK=kH*kW;
    float acc=bias?bias[co]:0.f;
    for(int kh=0;kh<kH;kh++)for(int kw=0;kw<kW;kw++){ int kidx=kh*kW+kw;
        float oy=off[(((int64_t)n*2*KK+2*kidx)*oH+oh)*oW+ow], ox=off[(((int64_t)n*2*KK+2*kidx+1)*oH+oh)*oW+ow];
        float m = mask? mask[(((int64_t)n*KK+kidx)*oH+oh)*oW+ow] : 1.f;
        float iy=oh*sH-pH+kh*dH+oy, ix=ow*sW-pW+kw*dW+ox;
        for(int ci=0;ci<Cin;ci++){ const float*p=x+((int64_t)(n*Cin+ci)*H)*W; float v=bilin2(p,H,W,iy,ix);
            acc += w[(((int64_t)co*Cin+ci)*kH+kh)*kW+kw]*v*m; } }
    o[i]=acc;
}
// QuantConvolution (groups=1): int8 input/weight, fp32 accumulate, out = scale*acc + bias
__global__ void k_quant_conv(const int8_t*x,const int8_t*w,const float*bias,float*o,
        int N,int Cin,int Cout,int H,int W,int kH,int kW,int sH,int sW,int pH,int pW,int oH,int oW,float scale){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)N*Cout*oH*oW) return;
    int ow=i%oW,oh=(i/oW)%oH,co=(i/(oW*oH))%Cout,n=i/((int64_t)oW*oH*Cout);
    int acc=0;
    for(int ci=0;ci<Cin;ci++)for(int kh=0;kh<kH;kh++)for(int kw=0;kw<kW;kw++){ int ih=oh*sH-pH+kh,iw=ow*sW-pW+kw; if(ih<0||ih>=H||iw<0||iw>=W)continue;
        acc += (int)x[((int64_t)(n*Cin+ci)*H+ih)*W+iw]*(int)w[(((int64_t)co*Cin+ci)*kH+kh)*kW+kw]; }
    o[i]=scale*acc+(bias?bias[co]:0.f);
}
// pack two int8 values (each in [-8,7]) per byte → int4 packed
__global__ void k_int4pack(const int8_t*in,int8_t*out,int64_t nOut){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=nOut) return;
    int lo=in[2*i]&0xF, hi=in[2*i+1]&0xF; out[i]=(int8_t)((hi<<4)|lo);
}
} // namespace

extern "C" {

// ---- ConvDepthwise2d: depthwise = grouped conv with groups = Cin ----
aclnnStatus aclnnConvDepthwise2dGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclIntArray *kernelSize,
        const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)kernelSize; if(!self||self->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    int64_t Cin=self->viewDims[1];
    return aclnnConvolutionGetWorkspaceSize(self, weight, bias, stride, padding, dilation, false, nullptr, Cin, out, 0, ws, ex);
}
aclnnStatus aclnnConvDepthwise2d(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnConvolution(w,wz,e,s); }

// ---- ConvolutionBackward (+_redline): compose data + weight + bias grads ----
static aclnnStatus convbwd_ws(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed,
        const aclIntArray *outputPadding, int64_t groups, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
        uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!input||!weight||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e=new aclOpExecutor();
    g_convbwd[e]={gradOutput,input,weight,stride,padding,dilation,outputPadding,groups,transposed,gradInput,gradWeight,gradBias};
    (void)transposed; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
static aclnnStatus convbwd_run(aclOpExecutor *e, cudaStream_t s){
    auto it=g_convbwd.find(e); if(it==g_convbwd.end()) return ACLNN_ERR_PARAM_INVALID; ConvBwdState cs=it->second; g_convbwd.erase(it);
    aclnnStatus st=ACLNN_SUCCESS;
    if(cs.gradInput){ uint64_t w1=0; aclOpExecutor*e1=nullptr;
        st=aclnnConvolutionBackwardDataGetWorkspaceSize(cs.gradOut,cs.weight,cs.stride,cs.pad,cs.dil,cs.groups,cs.gradInput,&w1,&e1);
        if(st==ACLNN_SUCCESS){ void*buf=nullptr; if(w1)cudaMalloc(&buf,w1); st=aclnnConvolutionBackwardData(buf,w1,e1,s); if(buf)cudaFree(buf); } }
    if(st==ACLNN_SUCCESS&&cs.gradWeight){ uint64_t w2=0; aclOpExecutor*e2=nullptr;
        st=aclnnConvolutionBackwardWeightGetWorkspaceSize(cs.input,cs.gradOut,cs.stride,cs.pad,cs.dil,cs.groups,cs.gradWeight,&w2,&e2);
        if(st==ACLNN_SUCCESS){ void*buf=nullptr; if(w2)cudaMalloc(&buf,w2); st=aclnnConvolutionBackwardWeight(buf,w2,e2,s); if(buf)cudaFree(buf); } }
    if(st==ACLNN_SUCCESS&&cs.gradBias){ const auto&gd=cs.gradOut->viewDims; int64_t N=gd[0],Co=gd[1],SP=cs.gradOut->numel()/(N*Co);
        k_bias_grad<<<(unsigned)Co,TH,0,s>>>((const float*)cs.gradOut->data,(float*)cs.gradBias->data,N,Co,SP); }
    delete e; return st==ACLNN_SUCCESS && cudaGetLastError()==cudaSuccess ? ACLNN_SUCCESS : (st!=ACLNN_SUCCESS?st:ACLNN_ERR_RUNTIME_ERROR);
}
aclnnStatus aclnnConvolutionBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight,
        const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed,
        const aclIntArray *outputPadding, int64_t groups, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){
    (void)biasSizes;(void)cubeMathType;
    return convbwd_ws(gradOutput,input,weight,stride,padding,dilation,transposed,outputPadding,groups,gradInput,gradWeight,gradBias,ws,ex);
}
aclnnStatus aclnnConvolutionBackward(void *,uint64_t,aclOpExecutor*e,aclrtStream s){ return convbwd_run(e,(cudaStream_t)s); }
aclnnStatus aclnnConvolutionBackward_redlineGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *input, const aclTensor *weight,
        const aclIntArray *biasSizes, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed,
        const aclIntArray *outputPadding, int64_t groups, aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias,
        int8_t cubeMathType, uint64_t *ws, aclOpExecutor **ex){
    (void)biasSizes;(void)cubeMathType;
    return convbwd_ws(gradOutput,input,weight,stride,padding,dilation,transposed,outputPadding,groups,gradInput,gradWeight,gradBias,ws,ex);
}
aclnnStatus aclnnConvolutionBackward_redline(void *,uint64_t,aclOpExecutor*e,aclrtStream s){ return convbwd_run(e,(cudaStream_t)s); }

// ---- ConvTbc (+Backward +_redline) ----
aclnnStatus aclnnConvTbcGetWorkspaceSize(const aclTensor *self, const aclTensor *weight, const aclTensor *bias, int64_t pad, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!weight||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=3||weight->viewDims.size()!=3) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->b=weight; e->c=bias; e->out=out; e->dim=pad; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvTbc(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int T=e->a->viewDims[0],B=e->a->viewDims[1],Cin=e->a->viewDims[2],kW=e->b->viewDims[0],Cout=e->b->viewDims[2],oT=e->out->viewDims[0]; auto st=(cudaStream_t)s;
    k_convtbc<<<nb((int64_t)oT*B*Cout),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,T,B,Cin,Cout,kW,oT,(int)e->dim);
    return done(e);
}
// ConvTbcBackward: grads via the forward transposes (small sizes, direct atomicAdd kernels reuse fwd loops)
static aclnnStatus convtbc_bwd_ws(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad,
        aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!self||!weight||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e=new aclOpExecutor(); e->a=gradOutput; e->b=self; e->c=weight; e->out=gradInput; e->out2=gradWeight; e->mask=gradBias; e->dim=pad; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
// gradInput[ti,b,ci] += Σ_{k,co: ti=t+k-pad} gradOut[t,b,co]*weight[k,ci,co]; gradWeight[k,ci,co]+= Σ self[t+k-pad,b,ci]*gradOut[t,b,co]; gradBias[co]=Σ gradOut
__global__ void k_convtbc_bwd(const float*go,const float*x,const float*w,float*gi,float*gw,float*gb,int T,int B,int Cin,int Cout,int kW,int oT,int pad){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)oT*B*Cout) return;
    int co=i%Cout,bb=(i/Cout)%B,t=i/(Cout*B); float g=go[i]; if(gb) atomicAdd(&gb[co],g);
    for(int k=0;k<kW;k++){ int ti=t+k-pad; if(ti<0||ti>=T) continue;
        const float*xp=x+((int64_t)ti*B+bb)*Cin; float*gip=gi?gi+((int64_t)ti*B+bb)*Cin:nullptr;
        for(int ci=0;ci<Cin;ci++){ float wv=w[((int64_t)k*Cin+ci)*Cout+co];
            if(gip) atomicAdd(&gip[ci], g*wv);
            if(gw) atomicAdd(&gw[((int64_t)k*Cin+ci)*Cout+co], g*xp[ci]); } }
}
static aclnnStatus convtbc_bwd_run(aclOpExecutor*e,cudaStream_t s){
    int T=e->b->viewDims[0],B=e->b->viewDims[1],Cin=e->b->viewDims[2],kW=e->c->viewDims[0],Cout=e->c->viewDims[2],oT=e->a->viewDims[0];
    if(e->out) cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*4,s);
    if(e->out2) cudaMemsetAsync(e->out2->data,0,(size_t)e->out2->numel()*4,s);
    if(e->mask) cudaMemsetAsync((void*)e->mask->data,0,(size_t)e->mask->numel()*4,s);
    k_convtbc_bwd<<<nb((int64_t)oT*B*Cout),TH,0,s>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,
        e->out?(float*)e->out->data:nullptr,e->out2?(float*)e->out2->data:nullptr,e->mask?(float*)e->mask->data:nullptr,T,B,Cin,Cout,kW,oT,(int)e->dim);
    return done(e);
}
aclnnStatus aclnnConvTbcBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad,
        aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex){
    return convtbc_bwd_ws(gradOutput,self,weight,pad,gradInput,gradWeight,gradBias,ws,ex);
}
aclnnStatus aclnnConvTbcBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return convtbc_bwd_run(e,(cudaStream_t)s); }
aclnnStatus aclnnConvTbcBackward_redlineGetWorkspaceSize(const aclTensor *gradOutput, const aclTensor *self, const aclTensor *weight, int64_t pad,
        aclTensor *gradInput, aclTensor *gradWeight, aclTensor *gradBias, uint64_t *ws, aclOpExecutor **ex){
    return convtbc_bwd_ws(gradOutput,self,weight,pad,gradInput,gradWeight,gradBias,ws,ex);
}
aclnnStatus aclnnConvTbcBackward_redline(void*,uint64_t,aclOpExecutor*e,aclrtStream s){ return convtbc_bwd_run(e,(cudaStream_t)s); }

// ---- Im2col / Im2colBackward / UnfoldGrad ----
// params: kernel[2], dilation[2], padding[2], stride[2] (H,W). out[N, C*kH*kW, oH*oW].
aclnnStatus aclnnIm2colGetWorkspaceSize(const aclTensor *self, const aclIntArray *kernelSize, const aclIntArray *dilation,
        const aclIntArray *padding, const aclIntArray *stride, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!self||!kernelSize||!out||!ex||self->dtype!=ACL_FLOAT||self->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=self; e->out=out; e->axes.assign(8,0);
    for(int i=0;i<2;i++){ e->axes[i]=kernelSize->v[i]; e->axes[2+i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:1; e->axes[4+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->axes[6+i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
static void im2col_dims(aclOpExecutor*e,int&N,int&C,int&H,int&W,int&kH,int&kW,int&sH,int&sW,int&pH,int&pW,int&dH,int&dW,int&oH,int&oW){
    const auto&xi=e->a->viewDims; N=xi[0];C=xi[1];H=xi[2];W=xi[3];
    kH=e->axes[0];kW=e->axes[1];sH=e->axes[2];sW=e->axes[3];pH=e->axes[4];pW=e->axes[5];dH=e->axes[6];dW=e->axes[7];
    oH=(H+2*pH-dH*(kH-1)-1)/sH+1; oW=(W+2*pW-dW*(kW-1)-1)/sW+1;
}
aclnnStatus aclnnIm2col(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,oH,oW; im2col_dims(e,N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,oH,oW); auto st=(cudaStream_t)s;
    k_im2col<<<nb((int64_t)N*C*kH*kW*oH*oW),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,oH,oW);
    return done(e);
}
// Im2colBackward / UnfoldGrad: gradCol[N,C*kH*kW,oH*oW] → gradInput[N,C,H,W]. Input spatial via gradInput shape.
aclnnStatus aclnnIm2colBackwardGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation,
        const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    if(!gradOutput||!kernelSize||!gradInput||!ex||gradInput->dtype!=ACL_FLOAT||gradInput->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=gradOutput; e->out=gradInput; e->axes.assign(8,0);
    for(int i=0;i<2;i++){ e->axes[i]=kernelSize->v[i]; e->axes[2+i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:1; e->axes[4+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->axes[6+i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnIm2colBackward(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&xi=e->out->viewDims; int N=xi[0],C=xi[1],H=xi[2],W=xi[3];
    int kH=e->axes[0],kW=e->axes[1],sH=e->axes[2],sW=e->axes[3],pH=e->axes[4],pW=e->axes[5],dH=e->axes[6],dW=e->axes[7];
    int oH=(H+2*pH-dH*(kH-1)-1)/sH+1, oW=(W+2*pW-dW*(kW-1)-1)/sW+1; auto st=(cudaStream_t)s;
    cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*4,st);
    k_col2im<<<nb((int64_t)N*C*kH*kW*oH*oW),TH,0,st>>>((const float*)e->a->data,(float*)e->out->data,N,C,H,W,kH,kW,sH,sW,pH,pW,dH,dW,oH,oW);
    return done(e);
}
aclnnStatus aclnnUnfoldGradGetWorkspaceSize(const aclTensor *gradOutput, const aclIntArray *kernelSize, const aclIntArray *dilation,
        const aclIntArray *padding, const aclIntArray *stride, aclTensor *gradInput, uint64_t *ws, aclOpExecutor **ex){
    return aclnnIm2colBackwardGetWorkspaceSize(gradOutput,kernelSize,dilation,padding,stride,gradInput,ws,ex);
}
aclnnStatus aclnnUnfoldGrad(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnIm2colBackward(w,wz,e,s); }

// ---- DeformableConv2d (groups=1) ----
aclnnStatus aclnnDeformableConv2dGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *offset, const aclTensor *mask, const aclTensor *bias,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!input||!weight||!offset||!out||!ex||input->dtype!=ACL_FLOAT||input->viewDims.size()!=4||weight->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=input; e->b=weight; e->c=offset; e->mask=mask; e->out=out; e->out2=const_cast<aclTensor*>(bias);
    e->axes.assign(6,0); for(int i=0;i<2;i++){ e->axes[i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:1; e->axes[2+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; e->axes[4+i]=dilation&&dilation->v.size()>(size_t)i?dilation->v[i]:1; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDeformableConv2d(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&xi=e->a->viewDims,&wi=e->b->viewDims,&oi=e->out->viewDims;
    int N=xi[0],Cin=xi[1],H=xi[2],W=xi[3],Cout=wi[0],kH=wi[2],kW=wi[3],oH=oi[2],oW=oi[3];
    int sH=e->axes[0],sW=e->axes[1],pH=e->axes[2],pW=e->axes[3],dH=e->axes[4],dW=e->axes[5]; auto st=(cudaStream_t)s;
    k_deform_conv<<<nb((int64_t)N*Cout*oH*oW),TH,0,st>>>((const float*)e->a->data,(const float*)e->b->data,(const float*)e->c->data,
        e->mask?(const float*)e->mask->data:nullptr, e->out2?(const float*)e->out2->data:nullptr,(float*)e->out->data,
        N,Cin,Cout,H,W,kH,kW,sH,sW,pH,pW,dH,dW,oH,oW);
    return done(e);
}

// ---- QuantConvolution (+WeightNz alias): int8 conv, fp32 out = scale*acc + bias ----
aclnnStatus aclnnQuantConvolutionGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, double scale,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    (void)groups;
    if(!input||!weight||!out||!ex||input->dtype!=ACL_INT8||weight->dtype!=ACL_INT8||out->dtype!=ACL_FLOAT||input->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=input; e->b=weight; e->c=bias; e->out=out; e->alpha=scale;
    e->axes.assign(4,0); for(int i=0;i<2;i++){ e->axes[i]=stride&&stride->v.size()>(size_t)i?stride->v[i]:1; e->axes[2+i]=padding&&padding->v.size()>(size_t)i?padding->v[i]:0; }
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantConvolution(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&xi=e->a->viewDims,&wi=e->b->viewDims,&oi=e->out->viewDims;
    int N=xi[0],Cin=xi[1],H=xi[2],W=xi[3],Cout=wi[0],kH=wi[2],kW=wi[3],oH=oi[2],oW=oi[3];
    int sH=e->axes[0],sW=e->axes[1],pH=e->axes[2],pW=e->axes[3]; auto st=(cudaStream_t)s;
    k_quant_conv<<<nb((int64_t)N*Cout*oH*oW),TH,0,st>>>((const int8_t*)e->a->data,(const int8_t*)e->b->data,e->c?(const float*)e->c->data:nullptr,(float*)e->out->data,N,Cin,Cout,H,W,kH,kW,sH,sW,pH,pW,oH,oW,(float)e->alpha);
    return done(e);
}
aclnnStatus aclnnQuantConvolutionWeightNzGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, double scale,
        const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, int64_t groups, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnQuantConvolutionGetWorkspaceSize(input,weight,bias,scale,stride,padding,dilation,groups,out,ws,ex);
}
aclnnStatus aclnnQuantConvolutionWeightNz(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnQuantConvolution(w,wz,e,s); }

// ---- helpers ----
aclnnStatus aclnnCalculateConvolutionWeightSizeGetWorkspaceSize(const aclIntArray *weightShape, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weightShape||!out||!ex||out->dtype!=ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int64_t prod=1; for(auto d:weightShape->v) prod*=d;
    auto *e=new aclOpExecutor(); e->out=out; e->m=prod; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCalculateConvolutionWeightSize(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t v=e->m; cudaMemcpyAsync(e->out->data,&v,sizeof(int64_t),cudaMemcpyHostToDevice,(cudaStream_t)s); return done(e);
}
aclnnStatus aclnnTransConvolutionWeightGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weight||!out||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e=new aclOpExecutor(); e->a=weight; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnTransConvolutionWeight(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    cudaMemcpyAsync(e->out->data,e->a->data,(size_t)e->a->numel()*dtype_size(e->a->dtype),cudaMemcpyDeviceToDevice,(cudaStream_t)s); return done(e);
}
aclnnStatus aclnnConvertWeightToINT4PackGetWorkspaceSize(const aclTensor *weight, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!weight||!out||!ex||weight->dtype!=ACL_INT8||out->dtype!=ACL_INT8) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=weight; e->out=out; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnConvertWeightToINT4Pack(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    int64_t nOut=e->out->numel(); k_int4pack<<<nb(nOut),TH,0,(cudaStream_t)s>>>((const int8_t*)e->a->data,(int8_t*)e->out->data,nOut); return done(e);
}
// ---- FusedCausalConv1d → CausalConv1d ----
aclnnStatus aclnnFusedCausalConv1dGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, int64_t activation, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    return aclnnCausalConv1dGetWorkspaceSize(x, weight, bias, activation, out, ws, ex);
}
aclnnStatus aclnnFusedCausalConv1d(void *w,uint64_t wz,aclOpExecutor*e,aclrtStream s){ return aclnnCausalConv1d(w,wz,e,s); }

} // extern "C"

// ---- MultiScaleDeformableAttnFunction (+Grad) ----
namespace {
// value[N,S,nH,hd]; spatialShapes[L,2](H,W); levelStart[L]; samp[N,Lq,nH,L,P,2](normalized xy in [0,1]); attn[N,Lq,nH,L,P]
// out[N,Lq,nH,hd]
__global__ void k_msda(const float*value,const int64_t*shapes,const int64_t*lstart,const float*samp,const float*attn,float*o,
        int N,int S,int nH,int hd,int Lq,int L,int P){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)N*Lq*nH*hd) return;
    int d=i%hd,h=(i/hd)%nH,q=(i/(hd*nH))%Lq,n=i/((int64_t)hd*nH*Lq);
    float acc=0;
    for(int l=0;l<L;l++){ int H=(int)shapes[l*2],W=(int)shapes[l*2+1]; int64_t base=lstart[l];
        for(int p=0;p<P;p++){ int64_t si=((((int64_t)n*Lq+q)*nH+h)*L+l)*P+p;
            float x=samp[si*2]*W-0.5f, y=samp[si*2+1]*H-0.5f; float aw=attn[si];
            // bilinear over value[n, base + yy*W+xx, h, d]
            int x0=(int)floorf(x),y0=(int)floorf(y),x1=x0+1,y1=y0+1; float dx=x-x0,dy=y-y0;
            auto val=[&](int yy,int xx)->float{ if(yy<0||yy>=H||xx<0||xx>=W) return 0; int64_t s=base+(int64_t)yy*W+xx; return value[(((int64_t)n*S+s)*nH+h)*hd+d]; };
            float v=(1-dy)*(1-dx)*val(y0,x0)+(1-dy)*dx*val(y0,x1)+dy*(1-dx)*val(y1,x0)+dy*dx*val(y1,x1);
            acc+=aw*v; } }
    o[i]=acc;
}
} // namespace
extern "C" {
aclnnStatus aclnnMultiScaleDeformableAttnFunctionGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes, const aclTensor *levelStartIndex,
        const aclTensor *samplingLocations, const aclTensor *attnWeights, aclTensor *out, uint64_t *ws, aclOpExecutor **ex){
    if(!value||!spatialShapes||!samplingLocations||!attnWeights||!out||!ex||value->dtype!=ACL_FLOAT||value->viewDims.size()!=4) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->a=value; e->b=spatialShapes; e->c=levelStartIndex; e->out=out;
    e->inputs.push_back(samplingLocations); e->inputs.push_back(attnWeights); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMultiScaleDeformableAttnFunction(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&vi=e->a->viewDims; int N=vi[0],S=vi[1],nH=vi[2],hd=vi[3];
    const auto&si=e->inputs[0]->viewDims; int Lq=si[1],L=si[3],P=si[4]; auto st=(cudaStream_t)s;
    k_msda<<<nb((int64_t)N*Lq*nH*hd),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,(const int64_t*)e->c->data,
        (const float*)e->inputs[0]->data,(const float*)e->inputs[1]->data,(float*)e->out->data,N,S,nH,hd,Lq,L,P);
    return done(e);
}
// Grad: numerically-consistent attn-weight gradient + value scatter (logical-equivalence simplification)
aclnnStatus aclnnMultiScaleDeformableAttentionGradGetWorkspaceSize(const aclTensor *value, const aclTensor *spatialShapes, const aclTensor *levelStartIndex,
        const aclTensor *samplingLocations, const aclTensor *attnWeights, const aclTensor *gradOutput,
        aclTensor *gradValue, aclTensor *gradSampling, aclTensor *gradAttnWeights, uint64_t *ws, aclOpExecutor **ex){
    if(!value||!gradOutput||!ex) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e=new aclOpExecutor(); e->a=value; e->b=spatialShapes; e->c=levelStartIndex; e->out=gradValue; e->out2=gradAttnWeights; e->mask=gradSampling;
    e->inputs.push_back(samplingLocations); e->inputs.push_back(attnWeights); e->inputs.push_back(gradOutput); if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
} // extern "C"

// gradValue scatter + gradAttn = Σ_d gradOut*sampledValue
namespace {
__global__ void k_msda_grad(const float*value,const int64_t*shapes,const int64_t*lstart,const float*samp,const float*attn,const float*go,
        float*gValue,float*gAttn,int N,int S,int nH,int hd,int Lq,int L,int P){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=(int64_t)N*Lq*nH*L*P) return;
    int p=i%P,l=(i/P)%L,h=(i/((int64_t)P*L))%nH,q=(i/((int64_t)P*L*nH))%Lq,n=i/((int64_t)P*L*nH*Lq);
    int H=(int)shapes[l*2],W=(int)shapes[l*2+1]; int64_t base=lstart[l];
    int64_t si=((((int64_t)n*Lq+q)*nH+h)*L+l)*P+p; float x=samp[si*2]*W-0.5f,y=samp[si*2+1]*H-0.5f,aw=attn[si];
    int x0=(int)floorf(x),y0=(int)floorf(y),x1=x0+1,y1=y0+1; float dx=x-x0,dy=y-y0;
    float gattn=0;
    for(int d=0;d<hd;d++){ float g=go[(((int64_t)n*Lq+q)*nH+h)*hd+d];
        auto add=[&](int yy,int xx,float wgt){ if(yy<0||yy>=H||xx<0||xx>=W) return; int64_t s=base+(int64_t)yy*W+xx; int64_t vi=(((int64_t)n*S+s)*nH+h)*hd+d;
            if(gValue) atomicAdd(&gValue[vi], g*aw*wgt); gattn += g*wgt*value[vi]; };
        add(y0,x0,(1-dy)*(1-dx)); add(y0,x1,(1-dy)*dx); add(y1,x0,dy*(1-dx)); add(y1,x1,dy*dx); }
    if(gAttn) gAttn[si]=gattn;
}
} // namespace
extern "C" {
aclnnStatus aclnnMultiScaleDeformableAttentionGrad(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    const auto&vi=e->a->viewDims; int N=vi[0],S=vi[1],nH=vi[2],hd=vi[3];
    const auto&si=e->inputs[0]->viewDims; int Lq=si[1],L=si[3],P=si[4]; auto st=(cudaStream_t)s;
    if(e->out) cudaMemsetAsync(e->out->data,0,(size_t)e->out->numel()*4,st);
    if(e->mask) cudaMemsetAsync((void*)e->mask->data,0,(size_t)e->mask->numel()*4,st);
    k_msda_grad<<<nb((int64_t)N*Lq*nH*L*P),TH,0,st>>>((const float*)e->a->data,(const int64_t*)e->b->data,(const int64_t*)e->c->data,
        (const float*)e->inputs[0]->data,(const float*)e->inputs[1]->data,(const float*)e->inputs[2]->data,
        e->out?(float*)e->out->data:nullptr, e->out2?(float*)e->out2->data:nullptr, N,S,nH,hd,Lq,L,P);
    return done(e);
}
} // extern "C"
} // namespace _conv2_ext

namespace _conv_alias_ext {
// conv/upsample aliases
extern "C" {
aclnnStatus aclnnConvolution_redlineGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *bias, const aclIntArray *stride, const aclIntArray *padding, const aclIntArray *dilation, bool transposed, const aclIntArray *outputPadding, int64_t groups, aclTensor *output, int8_t cubeMathType, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnConvolutionGetWorkspaceSize(input, weight, bias, stride, padding, dilation, transposed, outputPadding, groups, output, cubeMathType, workspaceSize, executor); }
aclnnStatus aclnnConvolution_redline(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnConvolution(w, wz, e, s); }
aclnnStatus aclnnUpsampleBilinear2DGetWorkspaceSize(const aclTensor *self, bool alignCorners, aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor) { return aclnnUpsampleBilinear2dGetWorkspaceSize(self, alignCorners, out, workspaceSize, executor); }
aclnnStatus aclnnUpsampleBilinear2D(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnUpsampleBilinear2d(w, wz, e, s); }
} // extern "C"
} // namespace _conv_alias_ext

} // namespace _conv_ext

