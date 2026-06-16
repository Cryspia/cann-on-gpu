// m_out.cu — merged family translation unit.
// Consolidated from per-feature source files; each former file is isolated in its own
// named namespace so file-local helpers cannot collide. extern "C" aclnn exports keep
// C linkage and bind to the global declarations in the API headers.
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>

namespace _random {
// Random op family: Uniform/Normal/Bernoulli/Dropout/Randperm.
// Uses a counter-based hash RNG (reproducible: seed+index→uniform); no cuRAND dependency.
// Validation checks statistical properties (mean/variance/proportion/range/permutation),
// not bit-exact matching against Ascend (random ops are not bit-for-bit comparable).

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__device__ inline uint32_t hash32(uint32_t x) { x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x; }
__device__ inline float u01(uint64_t seed, int64_t i) {
    uint32_t h = hash32((uint32_t)seed ^ hash32((uint32_t)i ^ (uint32_t)(i >> 32) ^ (uint32_t)(seed >> 32)));
    return (h >> 8) * (1.0f / 16777216.0f);   // result in [0, 1)
}
__global__ void k_uniform(float *o, int64_t n, float lo, float hi, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = lo + (hi - lo) * u01(seed, i);
}
__global__ void k_normal(float *o, int64_t n, float mean, float std, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float u1 = fmaxf(u01(seed, 2*i), 1e-7f), u2 = u01(seed, 2*i+1);
    o[i] = mean + std * sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);   // Box-Muller transform
}
// normal with optional per-element mean/std tensors (null → use scalar)
__global__ void k_normal_t(float *o, int64_t n, float meanS, float stdS, const float *meanT, const float *stdT, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    float u1 = fmaxf(u01(seed, 2*i), 1e-7f), u2 = u01(seed, 2*i+1);
    float z = sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
    float m = meanT ? meanT[i] : meanS, s = stdT ? stdT[i] : stdS;
    o[i] = m + s * z;
}
__global__ void k_bernoulli(float *o, int64_t n, float p, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = u01(seed, i) < p ? 1.f : 0.f;
}
// Bernoulli with a per-element probability tensor.
__global__ void k_bernoulli_t(float *o, const float *p, int64_t n, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = u01(seed, i) < p[i] ? 1.f : 0.f;
}
// Random integer fill: uniform in [lo, hi) cast to the float output, dispatched on numel.
__global__ void k_random(float *o, int64_t n, double lo, double hi, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = (float)(int64_t)(lo + (hi - lo) * (double)u01(seed, i));
}
__global__ void k_dropout(const float *x, float *o, uint8_t *mask, int64_t n, float p, uint64_t seed) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    uint8_t keep = u01(seed, i) < (1.f - p) ? 1 : 0; mask[i] = keep;
    o[i] = keep ? x[i] / (1.f - p) : 0.f;
}
__global__ void k_randperm(int64_t *o, int64_t n, uint64_t seed) {   // single-thread Fisher-Yates shuffle
    for (int64_t i = 0; i < n; ++i) o[i] = i;
    for (int64_t i = n - 1; i > 0; --i) { int64_t j = (int64_t)(u01(seed, i) * (i + 1)); if (j > i) j = i;
        int64_t t = o[i]; o[i] = o[j]; o[j] = t; }
}
} // namespace

extern "C" {

aclnnStatus aclnnUniformGetWorkspaceSize(aclTensor *out, double from, double to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->dscalars = {from, to}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnUniform(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_uniform<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(float)e->dscalars[1],(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
aclnnStatus aclnnNormalGetWorkspaceSize(aclTensor *out, double mean, double std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->dscalars = {mean, std}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
// Normal family: mean/std each scalar or tensor. e->b=meanT, e->c=stdT (nullable); dscalars={meanS,stdS}; m=seed
static aclnnStatus normal_build(aclTensor *out, double meanS, double stdS, const aclTensor *meanT, const aclTensor *stdT, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->b = meanT; e->c = stdT; e->dscalars = {meanS, stdS}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus normal_run(aclOpExecutor *e, cudaStream_t s) {
    int64_t n = e->out->numel();
    k_normal_t<<<nb(n),TH,0,s>>>((float*)e->out->data, n, (float)e->dscalars[0], (float)e->dscalars[1],
        e->b ? (const float*)e->b->data : nullptr, e->c ? (const float*)e->c->data : nullptr, (uint64_t)e->m);
    aclnnStatus st = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st;
}
aclnnStatus aclnnNormalFloatFloatGetWorkspaceSize(double mean, double std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return normal_build(out, mean, std, nullptr, nullptr, seed, ws, ex); }
aclnnStatus aclnnNormalFloatFloat(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }
aclnnStatus aclnnNormalFloatTensorGetWorkspaceSize(double mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return normal_build(out, mean, 0, nullptr, std, seed, ws, ex); }
aclnnStatus aclnnNormalFloatTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }
aclnnStatus aclnnNormalTensorFloatGetWorkspaceSize(const aclTensor *mean, double std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return normal_build(out, 0, std, mean, nullptr, seed, ws, ex); }
aclnnStatus aclnnNormalTensorFloat(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }
aclnnStatus aclnnNormalTensorTensorGetWorkspaceSize(const aclTensor *mean, const aclTensor *std, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return normal_build(out, 0, 0, mean, std, seed, ws, ex); }
aclnnStatus aclnnNormalTensorTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceNormalGetWorkspaceSize(aclTensor *self, double mean, double std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { return normal_build(self, mean, std, nullptr, nullptr, seed, ws, ex); }
aclnnStatus aclnnInplaceNormal(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }
aclnnStatus aclnnInplaceNormalTensorGetWorkspaceSize(aclTensor *self, const aclTensor *mean, const aclTensor *std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) { return normal_build(self, 0, 0, mean, std, seed, ws, ex); }
aclnnStatus aclnnInplaceNormalTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return normal_run(e, (cudaStream_t)s); }

aclnnStatus aclnnNormal(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_normal<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(float)e->dscalars[1],(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
aclnnStatus aclnnBernoulliGetWorkspaceSize(aclTensor *out, double p, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->dscalars = {p}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBernoulli(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_bernoulli<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
aclnnStatus aclnnDropoutGetWorkspaceSize(const aclTensor *x, double p, int64_t seed, aclTensor *out, aclTensor *mask, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !mask || !ex || x->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->a = x; e->out = out; e->out2 = mask; e->dscalars = {p}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnDropout(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->a->numel(); k_dropout<<<nb(n),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,(uint8_t*)e->out2->data,n,(float)e->dscalars[0],(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
// BernoulliTensor / InplaceBernoulliTensor: per-element probability tensor (e->a = prob, e->out = result)
aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(aclTensor *out, const aclTensor *prob, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !prob || !ex || out->dtype != ACL_FLOAT || prob->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->a = prob; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnBernoulliTensor(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_bernoulli_t<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,(const float*)e->a->data,n,(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
aclnnStatus aclnnInplaceBernoulliTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *prob, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    return aclnnBernoulliTensorGetWorkspaceSize(selfRef, prob, seed, ws, ex);
}
aclnnStatus aclnnInplaceBernoulliTensor(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnBernoulliTensor(w, wz, e, s); }

// InplaceRandom: fill self with integers uniform in [from, to)
aclnnStatus aclnnInplaceRandomGetWorkspaceSize(aclTensor *selfRef, int64_t from, int64_t to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!selfRef || !ex || selfRef->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = selfRef; e->dscalars = {(double)from, (double)to}; e->m = seed;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnInplaceRandom(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n = e->out->numel(); k_random<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,e->dscalars[0],e->dscalars[1],(uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}
// InplaceRandomTensor: bounds carried via single-element tensors (read at plan time)
aclnnStatus aclnnInplaceRandomTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    double lo = 0.0, hi = 1.0;
    if (from && from->data && from->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, from->data, 4, cudaMemcpyDeviceToHost) == cudaSuccess) lo = h; }
    if (to && to->data && to->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, to->data, 4, cudaMemcpyDeviceToHost) == cudaSuccess) hi = h; }
    return aclnnInplaceRandomGetWorkspaceSize(selfRef, (int64_t)lo, (int64_t)hi, seed, ws, ex);
}
aclnnStatus aclnnInplaceRandomTensor(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnInplaceRandom(w, wz, e, s); }
// InplaceUniformTensor: like Uniform but bounds via single-element tensors
aclnnStatus aclnnInplaceUniformTensorGetWorkspaceSize(aclTensor *selfRef, const aclTensor *from, const aclTensor *to, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    double lo = 0.0, hi = 1.0;
    if (from && from->data && from->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, from->data, 4, cudaMemcpyDeviceToHost) == cudaSuccess) lo = h; }
    if (to && to->data && to->dtype == ACL_FLOAT) { float h; if (cudaMemcpy(&h, to->data, 4, cudaMemcpyDeviceToHost) == cudaSuccess) hi = h; }
    return aclnnUniformGetWorkspaceSize(selfRef, lo, hi, seed, ws, ex);
}
aclnnStatus aclnnInplaceUniformTensor(void *w, uint64_t wz, aclOpExecutor *e, aclrtStream s) { return aclnnUniform(w, wz, e, s); }

aclnnStatus aclnnRandpermGetWorkspaceSize(int64_t n, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_NULLPTR;
    auto *e = new aclOpExecutor(); e->op = OP_REDUCE_SUM; e->out = out; e->m = seed; e->n = n;
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRandperm(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    k_randperm<<<1,1,0,(cudaStream_t)s>>>((int64_t*)e->out->data, e->n, (uint64_t)e->m);
    aclnnStatus r = cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return r;
}

} // extern "C"
} // namespace _random

namespace _random_ext {
// Random / distribution extensions (P12): RandInt, Exponential, Geometric, Cauchy, LogNormal, Poisson, Multinomial.
// Counter-based hash RNG (reproducible). Validation is statistical (mean/range), not bit-exact.

namespace {
constexpr int TH = 256;
inline int64_t nb(int64_t n) { return (n + TH - 1) / TH; }
__device__ inline uint32_t hash32(uint32_t x){ x^=x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; x^=x>>16; return x; }
__device__ inline float u01(uint64_t seed, int64_t i){ uint32_t h=hash32((uint32_t)seed ^ hash32((uint32_t)i ^ (uint32_t)(i>>32) ^ (uint32_t)(seed>>32))); return (h>>8)*(1.0f/16777216.0f); }

__global__ void k_randint(int64_t *o, int64_t n, int64_t lo, int64_t hi, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; o[i]=lo+(int64_t)(u01(seed,i)*(hi-lo));
}
__global__ void k_exponential(float *o, int64_t n, float lambda, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float u=fmaxf(u01(seed,i),1e-7f); o[i]=-logf(u)/lambda;
}
__global__ void k_geometric(float *o, int64_t n, float p, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; float u=fmaxf(u01(seed,i),1e-7f); o[i]=floorf(logf(u)/log1pf(-p))+1.f;
}
__global__ void k_cauchy(float *o, int64_t n, float median, float sigma, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return; o[i]=median+sigma*tanf(3.14159265f*(u01(seed,i)-0.5f));
}
__global__ void k_lognormal(float *o, int64_t n, float mean, float std, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float u1=fmaxf(u01(seed,2*i),1e-7f),u2=u01(seed,2*i+1); float z=sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2); o[i]=expf(mean+std*z);
}
__global__ void k_poisson(int32_t *o, int64_t n, float lambda, uint64_t seed) {
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    float L=expf(-lambda); int k=0; float pp=1.f; int64_t c=0;
    do { pp *= u01(seed, i*64 + c); c++; k++; } while (pp > L && c < 10000);
    o[i]=k-1;
}
// Multinomial: probs[rows,C] (need not be normalized) -> out[rows,numSamples] int64 by CDF inversion
__global__ void k_multinomial(const float *probs, int64_t *o, int64_t rows, int64_t C, int64_t ns, uint64_t seed) {
    int64_t idx=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(idx>=rows*ns) return;
    int64_t r=idx/ns; const float *p=probs+r*C;
    double tot=0; for(int64_t c=0;c<C;c++) tot+=p[c];
    double target=u01(seed, idx)*tot, acc=0; int64_t pick=C-1;
    for(int64_t c=0;c<C;c++){ acc+=p[c]; if(acc>=target){ pick=c; break; } }
    o[idx]=pick;
}
inline aclnnStatus done(aclOpExecutor *e){ aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR; delete e; return st; }
} // namespace

extern "C" {

aclnnStatus aclnnRandIntGetWorkspaceSize(int64_t low, int64_t high, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={(double)low,(double)high};
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnRandInt(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_randint<<<nb(n),TH,0,(cudaStream_t)s>>>((int64_t*)e->out->data,n,(int64_t)e->dscalars[0],(int64_t)e->dscalars[1],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnExponentialGetWorkspaceSize(aclTensor *out, double lambda, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={lambda}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnExponential(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_exponential<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnGeometricGetWorkspaceSize(aclTensor *out, double p, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={p}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnGeometric(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_geometric<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnCauchyGetWorkspaceSize(aclTensor *out, double median, double sigma, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={median,sigma}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnCauchy(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_cauchy<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(float)e->dscalars[1],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnLogNormalGetWorkspaceSize(aclTensor *out, double mean, double std, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID; auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={mean,std}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnLogNormal(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_lognormal<<<nb(n),TH,0,(cudaStream_t)s>>>((float*)e->out->data,n,(float)e->dscalars[0],(float)e->dscalars[1],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnPoissonGetWorkspaceSize(aclTensor *out, double lambda, int64_t seed, uint64_t *ws, aclOpExecutor **ex) {
    if (!out || !ex || out->dtype != ACL_INT32) return ACLNN_ERR_PARAM_INVALID; auto *e=new aclOpExecutor(); e->out=out; e->m=seed; e->dscalars={lambda}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnPoisson(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t n=e->out->numel(); k_poisson<<<nb(n),TH,0,(cudaStream_t)s>>>((int32_t*)e->out->data,n,(float)e->dscalars[0],(uint64_t)e->m); return done(e);
}
aclnnStatus aclnnMultinomialGetWorkspaceSize(const aclTensor *probs, int64_t numSamples, int64_t seed, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!probs || !out || !ex || probs->dtype != ACL_FLOAT || out->dtype != ACL_INT64) return ACLNN_ERR_PARAM_INVALID;
    int rank=(int)probs->viewDims.size(); auto *e=new aclOpExecutor(); e->a=probs; e->out=out; e->m=seed;
    e->n=probs->viewDims[rank-1]; e->outerCount=probs->numel()/e->n; e->k=numSamples;
    if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMultinomial(void *, uint64_t, aclOpExecutor *e, aclrtStream s) {
    int64_t rows=e->outerCount,C=e->n,ns=e->k; int64_t g=nb(rows*ns);
    k_multinomial<<<g,TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(int64_t*)e->out->data,rows,C,ns,(uint64_t)e->m); return done(e);
}

} // extern "C"
} // namespace _random_ext

namespace _rand2_ext {
// Random remainder (R8): SampleGamma (Marsaglia-Tsang), SampleDirichlet (Gamma + last-dim normalize).
// Counter-based hash RNG (no cuRAND); validation is statistical, not bit-exact.

namespace {
constexpr int TH=256; inline int64_t nb(int64_t n){return (n+TH-1)/TH;}
__device__ inline uint32_t hash32(uint32_t x){x^=x>>16;x*=0x7feb352dU;x^=x>>15;x*=0x846ca68bU;x^=x>>16;return x;}
__device__ inline float u01(uint64_t seed,uint64_t i){uint32_t h=hash32((uint32_t)seed^hash32((uint32_t)i^(uint32_t)(i>>32)^(uint32_t)(seed>>32)));return (h>>8)*(1.0f/16777216.0f);}
// Marsaglia-Tsang gamma sampler, shape a>0, unit scale.
__device__ float gen_gamma(float a,uint64_t seed,uint64_t idx,uint32_t&ctr){
    float boost=1.f; uint64_t base=idx*65536ULL;
    if(a<1.f){ float u=fmaxf(u01(seed,base+ctr++),1e-7f); boost=powf(u,1.f/a); a+=1.f; }
    float d=a-1.f/3.f, c=1.f/sqrtf(9.f*d);
    for(int it=0;it<256;it++){
        float u1=fmaxf(u01(seed,base+ctr++),1e-7f),u2=u01(seed,base+ctr++);
        float x=sqrtf(-2.f*logf(u1))*cosf(6.2831853f*u2);
        float v=1.f+c*x; if(v<=0.f)continue; v=v*v*v;
        float u=fmaxf(u01(seed,base+ctr++),1e-7f);
        if(logf(u) < 0.5f*x*x + d - d*v + d*logf(v)) return d*v*boost;
    }
    return d*boost;
}
__global__ void k_gamma(const float*alpha,float*out,int64_t n,float scale,uint64_t seed){
    int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; uint32_t ctr=0; out[i]=gen_gamma(alpha[i],seed,i,ctr)*scale;
}
// Dirichlet: M rows of length K; sample gamma per element then normalize across K.
__global__ void k_dirichlet(const float*alpha,float*out,int64_t M,int64_t K,uint64_t seed){
    int64_t r=(int64_t)blockIdx.x*blockDim.x+threadIdx.x; if(r>=M)return; uint32_t ctr=0; double s=0;
    for(int64_t k=0;k<K;k++){ float g=gen_gamma(alpha[r*K+k],seed,r*K+k,ctr); out[r*K+k]=g; s+=g; }
    float inv=(s>0)?(float)(1.0/s):0.f; for(int64_t k=0;k<K;k++) out[r*K+k]*=inv;
}
inline aclnnStatus done(aclOpExecutor*e){aclnnStatus st=cudaGetLastError()==cudaSuccess?ACLNN_SUCCESS:ACLNN_ERR_RUNTIME_ERROR;delete e;return st;}
} // namespace

extern "C" {

aclnnStatus aclnnSampleGammaGetWorkspaceSize(const aclTensor*alpha,double scale,int64_t seed,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!alpha||!out||!ex||alpha->dtype!=ACL_FLOAT) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=alpha; e->out=out; e->m=alpha->numel(); e->alpha=scale; e->dscalars={(double)seed}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSampleGamma(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    k_gamma<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->m,(float)e->alpha,(uint64_t)e->dscalars[0]); return done(e);
}
aclnnStatus aclnnSampleDirichletGetWorkspaceSize(const aclTensor*alpha,int64_t seed,aclTensor*out,uint64_t*ws,aclOpExecutor**ex){
    if(!alpha||!out||!ex||alpha->dtype!=ACL_FLOAT||alpha->viewDims.empty()) return ACLNN_ERR_PARAM_INVALID;
    auto*e=new aclOpExecutor(); e->a=alpha; e->out=out; int64_t K=alpha->viewDims.back(); e->k=K; e->m=alpha->numel()/K; e->dscalars={(double)seed}; if(ws)*ws=0; *ex=e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnSampleDirichlet(void*,uint64_t,aclOpExecutor*e,aclrtStream s){
    k_dirichlet<<<nb(e->m),TH,0,(cudaStream_t)s>>>((const float*)e->a->data,(float*)e->out->data,e->m,e->k,(uint64_t)e->dscalars[0]); return done(e);
}

} // extern "C"
} // namespace _rand2_ext

