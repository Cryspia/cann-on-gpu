// mc2 distributed fused ops over the real 2-node RoCE/HCCL path (NCCL backend).
//   MatmulAllReduce / QuantMatmulAllReduce: K-split tensor parallel — local (quant) matmul + HcclAllReduce(SUM) → full x@W.
//   MatmulReduceScatter: K-split partial [M,N] → HcclReduceScatter → each rank gets rows [M/nranks, N] of the sum.
// Local matmul reuses the exported aclnnMatmul/aclnnQuantMatmul cores (nested executor stashed in a side-map,
// same pattern as blas2_ext einsum). comm==nullptr degrades to local only. Validated on 2 nodes (tools/mc2_2node).
#include "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_mc2.h"
#include "hccl/hccl.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <map>

namespace {
enum Mc2Kind { MC2_ALLREDUCE, MC2_REDUCESCATTER, MC2_ALLGATHER };
struct Mc2State { aclOpExecutor *inner; HcclComm comm; int kind; const aclTensor *bias; aclTensor *temp; uint64_t mmWs; };
std::map<aclOpExecutor *, Mc2State> g_mc2;   // stash nested executor + comm between the two phases

inline HcclDataType hccl_dt(aclDataType d) {
    return d == ACL_FLOAT ? HCCL_DATA_TYPE_FP32 : d == ACL_FLOAT16 ? HCCL_DATA_TYPE_FP16 : HCCL_DATA_TYPE_BFP16;
}
inline int dt_size(aclDataType d) { return d == ACL_FLOAT ? 4 : 2; }
template <typename T>
__global__ void k_bias_add(T *out, const T *bias, int64_t M, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; if (i >= M * N) return;
    out[i] = (T)((float)out[i] + (float)bias[i % N]);
}
inline void launch_bias(aclDataType dt, void *out, const void *bias, int64_t M, int64_t N, cudaStream_t s) {
    int64_t g = (M*N + 255) / 256;
    switch (dt) {
        case ACL_FLOAT:   k_bias_add<float><<<g,256,0,s>>>((float*)out,(const float*)bias,M,N); break;
        case ACL_FLOAT16: k_bias_add<__half><<<g,256,0,s>>>((__half*)out,(const __half*)bias,M,N); break;
        default:          k_bias_add<__nv_bfloat16><<<g,256,0,s>>>((__nv_bfloat16*)out,(const __nv_bfloat16*)bias,M,N); break;
    }
}
} // namespace

extern "C" {

// MatmulAllReduce: out = AllReduce_SUM( x[M,K] @ weight[K,N] ) (+ bias[N] once).
aclnnStatus aclnnMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (bias && (bias->dtype != out->dtype || bias->viewDims.size() != 1 || bias->viewDims[0] != out->viewDims.back())) return ACLNN_ERR_PARAM_INVALID;
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnMatmulGetWorkspaceSize(x, weight, out, 1, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) return st;
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = bias; e->out = out;
    g_mc2[e] = { inner, comm, MC2_ALLREDUCE, bias, nullptr, mmWs };
    if (ws) *ws = mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulAllReduce(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    aclnnStatus st = aclnnMatmul(ws, wsz, ms.inner, s);
    if (st == ACLNN_SUCCESS && ms.comm) {
        if (HcclAllReduce(e->out->data, e->out->data, (uint64_t)e->out->numel(), hccl_dt(e->out->dtype), HCCL_REDUCE_SUM, ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    }
    if (st == ACLNN_SUCCESS && ms.bias) {
        int64_t N = e->out->viewDims.back(), M = e->out->numel()/N;
        launch_bias(e->out->dtype, e->out->data, ms.bias->data, M, N, (cudaStream_t)s);
        if (cudaGetLastError() != cudaSuccess) st = ACLNN_ERR_RUNTIME_ERROR;
    }
    delete e; return st;
}

// QuantMatmulAllReduce: out = AllReduce_SUM( dequant(x_int8[M,K] @ weight_int8[K,N], scale[N]) ).
// scale is linear over the K-reduction, so per-rank scale·partial summed across ranks == scale·full. (TP K-split)
aclnnStatus aclnnQuantMatmulAllReduceGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !scale || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnQuantMatmulGetWorkspaceSize(x, weight, scale, out, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) return st;
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = scale; e->out = out;
    g_mc2[e] = { inner, comm, MC2_ALLREDUCE, nullptr, nullptr, mmWs };
    if (ws) *ws = mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnQuantMatmulAllReduce(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    aclnnStatus st = aclnnQuantMatmul(ws, wsz, ms.inner, s);
    if (st == ACLNN_SUCCESS && ms.comm) {
        if (HcclAllReduce(e->out->data, e->out->data, (uint64_t)e->out->numel(), hccl_dt(e->out->dtype), HCCL_REDUCE_SUM, ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    }
    delete e; return st;
}

// MatmulReduceScatter: local partial = x[M,K] @ weight[K,N] (full [M,N]); HcclReduceScatter(SUM) → out[M/nranks, N].
// Local matmul writes to a workspace temp [M,N]; temp->data is patched in run (dummy non-null at build to pass validation).
aclnnStatus aclnnMatmulReduceScatterGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)bias;
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t M = x->viewDims[0], N = weight->viewDims.back();
    // temp full-output [M,N], same dtype as out; dummy non-null data so aclnnMatmul validation passes; patched in run.
    aclTensor *temp = aclCreateTensor(std::vector<int64_t>{M,N}.data(), 2, out->dtype, nullptr, 0, ACL_FORMAT_ND,
                                      std::vector<int64_t>{M,N}.data(), 2, (void*)16);
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnMatmulGetWorkspaceSize(x, weight, temp, 1, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) { aclDestroyTensor(temp); return st; }
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->out = out; e->m = M; e->n = N;
    uint64_t tempBytes = (uint64_t)M * N * dt_size(out->dtype);
    g_mc2[e] = { inner, comm, MC2_REDUCESCATTER, nullptr, temp, mmWs };
    if (ws) *ws = tempBytes + mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulReduceScatter(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    int64_t M = e->m, N = e->n; uint64_t tempBytes = (uint64_t)M * N * dt_size(e->out->dtype);
    ms.temp->data = ws;                                   // patch local-matmul output into the workspace temp
    aclnnStatus st = aclnnMatmul((char*)ws + tempBytes, wsz - tempBytes, ms.inner, s);   // partial [M,N] → temp
    if (st == ACLNN_SUCCESS && ms.comm) {
        if (HcclReduceScatter(ms.temp->data, e->out->data, (uint64_t)e->out->numel(), hccl_dt(e->out->dtype), HCCL_REDUCE_SUM, ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    } else if (st == ACLNN_SUCCESS) {                      // single-node: copy first M_local rows
        cudaMemcpyAsync(e->out->data, ms.temp->data, (size_t)e->out->numel()*dt_size(e->out->dtype), cudaMemcpyDeviceToDevice, (cudaStream_t)s);
    }
    aclDestroyTensor(ms.temp); delete e; return st;
}

// MatmulAllGather: each rank computes its row-shard x[Mr,K] @ weight[K,N] -> temp[Mr,N]; HcclAllGather concatenates
// the shards along rows -> out[M,N] = X_full @ W. (row-parallel: x sharded over M, weight replicated)
aclnnStatus aclnnMatmulAllGatherGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias,
        HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)bias;
    if (!x || !weight || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t Mr = x->viewDims[0], N = weight->viewDims.back();
    aclTensor *temp = aclCreateTensor(std::vector<int64_t>{Mr,N}.data(), 2, out->dtype, nullptr, 0, ACL_FORMAT_ND,
                                      std::vector<int64_t>{Mr,N}.data(), 2, (void*)16);
    aclOpExecutor *inner = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnMatmulGetWorkspaceSize(x, weight, temp, 1, &mmWs, &inner);
    if (st != ACLNN_SUCCESS) { aclDestroyTensor(temp); return st; }
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->out = out; e->m = Mr; e->n = N;
    uint64_t tempBytes = (uint64_t)Mr * N * dt_size(out->dtype);
    g_mc2[e] = { inner, comm, MC2_ALLGATHER, nullptr, temp, mmWs };
    if (ws) *ws = tempBytes + mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulAllGather(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    int64_t Mr = e->m, N = e->n; uint64_t tempBytes = (uint64_t)Mr * N * dt_size(e->out->dtype);
    ms.temp->data = ws;
    aclnnStatus st = aclnnMatmul((char*)ws + tempBytes, wsz - tempBytes, ms.inner, s);   // local shard [Mr,N] → temp
    if (st == ACLNN_SUCCESS && ms.comm) {
        if (HcclAllGather(ms.temp->data, e->out->data, (uint64_t)(Mr*N), hccl_dt(e->out->dtype), ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    } else if (st == ACLNN_SUCCESS) {
        cudaMemcpyAsync(e->out->data, ms.temp->data, (size_t)Mr*N*dt_size(e->out->dtype), cudaMemcpyDeviceToDevice, (cudaStream_t)s);
    }
    aclDestroyTensor(ms.temp); delete e; return st;
}

// MoeDistributeDispatch / Combine: expert-parallel token exchange via HcclAlltoAll. Capacity layout:
// x is [nranks, C, H] — block j holds the C tokens this rank sends to rank j; out[j] = the C tokens
// received from rank j. Dispatch (scatter tokens to expert-owning ranks) and Combine (return results)
// are the same symmetric AlltoAll over this layout. (Routing/permute is done by the single-card Moe* ops.)
static aclnnStatus moe_a2a_ws(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !out || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    if (x->dtype != out->dtype || x->viewDims != out->viewDims) return ACLNN_ERR_PARAM_INVALID;
    auto *e = new aclOpExecutor(); e->a = x; e->out = out;
    g_mc2[e] = { nullptr, comm, MC2_ALLREDUCE, nullptr, nullptr, 0 };
    if (ws) *ws = 0; *ex = e; return ACLNN_SUCCESS;
}
static aclnnStatus moe_a2a_run(aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    aclnnStatus st = ACLNN_SUCCESS;
    if (ms.comm) {
        uint32_t nr = 0; HcclGetRankSize(ms.comm, &nr); if (nr == 0) nr = 1;
        uint64_t per = (uint64_t)e->a->numel() / nr;     // elements sent to / received from each rank
        if (HcclAlltoAll(e->a->data, per, hccl_dt(e->a->dtype), e->out->data, per, hccl_dt(e->a->dtype), ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    } else {
        cudaMemcpyAsync(e->out->data, e->a->data, (size_t)e->a->numel()*dt_size(e->a->dtype), cudaMemcpyDeviceToDevice, (cudaStream_t)s);
    }
    delete e; return st;
}
aclnnStatus aclnnMoeDistributeDispatchGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return moe_a2a_ws(x, comm, out, ws, ex); }
aclnnStatus aclnnMoeDistributeDispatch(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return moe_a2a_run(e, s); }
aclnnStatus aclnnMoeDistributeCombineGetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return moe_a2a_ws(x, comm, out, ws, ex); }
aclnnStatus aclnnMoeDistributeCombine(void *, uint64_t, aclOpExecutor *e, aclrtStream s) { return moe_a2a_run(e, s); }

// Version-variant forwards (same simplified contract as the base mc2 op).
#define MC2_MM_VAR(NAME, BASE) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *bias, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return BASE##GetWorkspaceSize(x, weight, bias, comm, out, ws, ex); } \
aclnnStatus NAME(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return BASE(ws, wsz, e, s); }
#define MC2_QMM_VAR(NAME) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *scale, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return aclnnQuantMatmulAllReduceGetWorkspaceSize(x, weight, scale, comm, out, ws, ex); } \
aclnnStatus NAME(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return aclnnQuantMatmulAllReduce(ws, wsz, e, s); }
#define MC2_A2A_VAR(NAME, BASE) \
aclnnStatus NAME##GetWorkspaceSize(const aclTensor *x, HcclComm comm, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) { return BASE##GetWorkspaceSize(x, comm, out, ws, ex); } \
aclnnStatus NAME(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) { return BASE(ws, wsz, e, s); }
MC2_MM_VAR(aclnnMatmulAllReduceV2, aclnnMatmulAllReduce)
MC2_MM_VAR(aclnnMatmulReduceScatterV2, aclnnMatmulReduceScatter)
MC2_QMM_VAR(aclnnQuantMatmulAllReduceV2)
MC2_QMM_VAR(aclnnQuantMatmulAllReduceV3)
MC2_QMM_VAR(aclnnQuantMatmulAllReduceV4)
MC2_A2A_VAR(aclnnMoeDistributeDispatchV2, aclnnMoeDistributeDispatch)
MC2_A2A_VAR(aclnnMoeDistributeDispatchV3, aclnnMoeDistributeDispatch)
MC2_A2A_VAR(aclnnMoeDistributeDispatchV4, aclnnMoeDistributeDispatch)
MC2_A2A_VAR(aclnnMoeDistributeCombineV2, aclnnMoeDistributeCombine)
MC2_A2A_VAR(aclnnMoeDistributeCombineV3, aclnnMoeDistributeCombine)
MC2_A2A_VAR(aclnnMoeDistributeCombineV4, aclnnMoeDistributeCombine)

// MatmulAllReduceAddRmsNorm: y = RmsNorm( AllReduce(x@W) + residual )·gamma; also outputs residualSum.
// Tail-fusion = MatmulAllReduce (into a temp) → nested aclnnAddRmsNorm. Reuses both verified pieces.
aclnnStatus aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(const aclTensor *x, const aclTensor *weight, const aclTensor *residual,
        const aclTensor *gamma, double eps, HcclComm comm, aclTensor *y, aclTensor *residualSum, uint64_t *ws, aclOpExecutor **ex) {
    if (!x || !weight || !residual || !y || !residualSum || !ex) return ACLNN_ERR_PARAM_NULLPTR;
    int64_t M = x->viewDims[0], N = weight->viewDims.back();
    aclTensor *temp = aclCreateTensor(std::vector<int64_t>{M,N}.data(), 2, y->dtype, nullptr, 0, ACL_FORMAT_ND,
                                      std::vector<int64_t>{M,N}.data(), 2, (void*)16);
    aclOpExecutor *mm = nullptr; uint64_t mmWs = 0;
    aclnnStatus st = aclnnMatmulGetWorkspaceSize(x, weight, temp, 1, &mmWs, &mm);
    if (st != ACLNN_SUCCESS) { aclDestroyTensor(temp); return st; }
    auto *e = new aclOpExecutor(); e->a = x; e->b = weight; e->c = gamma; e->mask = residual; e->out = y; e->out2 = residualSum;
    e->m = M; e->n = N; e->eps = eps;
    g_mc2[e] = { mm, comm, MC2_ALLREDUCE, nullptr, temp, mmWs };
    if (ws) *ws = (uint64_t)M*N*dt_size(y->dtype) + mmWs; *ex = e; return ACLNN_SUCCESS;
}
aclnnStatus aclnnMatmulAllReduceAddRmsNorm(void *ws, uint64_t wsz, aclOpExecutor *e, aclrtStream s) {
    auto it = g_mc2.find(e); if (it == g_mc2.end()) return ACLNN_ERR_PARAM_INVALID;
    Mc2State ms = it->second; g_mc2.erase(it);
    int64_t M = e->m, N = e->n; uint64_t tempBytes = (uint64_t)M*N*dt_size(e->out->dtype);
    ms.temp->data = ws;
    aclnnStatus st = aclnnMatmul((char*)ws + tempBytes, wsz - tempBytes, ms.inner, s);     // local partial → temp
    if (st == ACLNN_SUCCESS && ms.comm) {
        if (HcclAllReduce(ms.temp->data, ms.temp->data, (uint64_t)(M*N), hccl_dt(e->out->dtype), HCCL_REDUCE_SUM, ms.comm, s) != HCCL_SUCCESS) st = ACLNN_ERR_RUNTIME_ERROR;
    }
    if (st == ACLNN_SUCCESS) {   // tail RmsNorm: y = rms(temp+residual)·gamma, residualSum = temp+residual
        aclOpExecutor *rn = nullptr; uint64_t rnWs = 0;
        st = aclnnAddRmsNormGetWorkspaceSize(ms.temp, e->mask, e->c, e->eps, e->out, e->out2, &rnWs, &rn);
        if (st == ACLNN_SUCCESS) st = aclnnAddRmsNorm(nullptr, rnWs, rn, s);   // AddRmsNorm needs no workspace
    }
    aclDestroyTensor(ms.temp); delete e; return st;
}

} // extern "C"
