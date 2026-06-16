// aclrt* / aclInit → near 1:1 mapping to CUDA Runtime.
// Signature contract comes from $ACL_INCLUDE/acl/acl_rt.h (enforced via include, not copied manually).
#include "internal.h"
#include <unordered_map>
#include <vector>
#include <mutex>

// ---- Caching device memory allocator ----
// cudaMalloc/cudaFree are synchronous and slow; inference backends that alloc/free heavily each step
// will be bottlenecked. This reuses allocations by size-bucket:
// free does not return memory to the driver but places it in a free list; malloc prefers reusing a
// same-sized block. Set env ACL_NO_CACHE_ALLOC=1 to fall back to pass-through for comparison.
namespace {
struct CacheAlloc {
    std::mutex mu;
    std::unordered_map<size_t, std::vector<void*>> freeList;   // bucket → free blocks
    std::unordered_map<void*, size_t> live;                    // ptr → bucket (looked up on free)
    bool enabled = (getenv("ACL_NO_CACHE_ALLOC") == nullptr);
    // bucket: below 1 MB round up to power-of-two; 1 MB and above align up to 2 MB (caps waste at <2 MB)
    static size_t bucketOf(size_t n) {
        if (n == 0) n = 1;
        if (n < (1u << 20)) { size_t b = 1; while (b < n) b <<= 1; return b; }
        const size_t G = 2u << 20; return (n + G - 1) / G * G;
    }
    cudaError_t alloc(void **p, size_t n) {
        size_t b = bucketOf(n);
        if (enabled) { std::lock_guard<std::mutex> lk(mu);
            auto it = freeList.find(b);
            if (it != freeList.end() && !it->second.empty()) { *p = it->second.back(); it->second.pop_back(); live[*p] = b; return cudaSuccess; } }
        cudaError_t e = cudaMalloc(p, b);
        if (e == cudaSuccess && enabled) { std::lock_guard<std::mutex> lk(mu); live[*p] = b; }
        return e;
    }
    cudaError_t free(void *p) {
        if (!p) return cudaSuccess;
        if (enabled) { std::lock_guard<std::mutex> lk(mu);
            auto it = live.find(p);
            if (it != live.end()) { freeList[it->second].push_back(p); live.erase(it); return cudaSuccess; } }
        return cudaFree(p);
    }
};
CacheAlloc &g_alloc() { static CacheAlloc a; return a; }
} // namespace

extern "C" {

aclError aclInit(const char *configPath) {
    (void)configPath;                  // acl.json config is not supported and not needed here
    return acl_from_cuda(cudaFree(0)); // trigger CUDA primary context initialization
}

aclError aclFinalize() {
    return ACL_SUCCESS;                // primary context is reclaimed on process exit
}

aclError aclrtSetDevice(int32_t deviceId)   { return acl_from_cuda(cudaSetDevice(deviceId)); }
aclError aclrtResetDevice(int32_t deviceId) { (void)deviceId; return ACL_SUCCESS; }
aclError aclrtGetDevice(int32_t *deviceId)  { return acl_from_cuda(cudaGetDevice(deviceId)); }

aclError aclrtGetDeviceCount(uint32_t *count) {
    int n = 0;
    ACL_CUDA(cudaGetDeviceCount(&n));
    *count = (uint32_t)n;
    return ACL_SUCCESS;
}

// CUDA runtime uses the primary context; ACL explicit contexts are transparent no-ops in the shim
aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
    ACL_CUDA(cudaSetDevice(deviceId));
    *context = (aclrtContext)(intptr_t)(deviceId + 1);
    return ACL_SUCCESS;
}
aclError aclrtDestroyContext(aclrtContext context)   { (void)context; return ACL_SUCCESS; }
aclError aclrtSetCurrentContext(aclrtContext context){ (void)context; return ACL_SUCCESS; }

aclError aclrtGetRunMode(aclrtRunMode *runMode) { *runMode = ACL_HOST; return ACL_SUCCESS; }

aclError aclrtCreateStream(aclrtStream *stream) {
    return acl_from_cuda(cudaStreamCreate(reinterpret_cast<cudaStream_t *>(stream)));
}
aclError aclrtDestroyStream(aclrtStream stream) {
    return acl_from_cuda(cudaStreamDestroy((cudaStream_t)stream));
}
aclError aclrtSynchronizeStream(aclrtStream stream) {
    return acl_from_cuda(cudaStreamSynchronize((cudaStream_t)stream));
}
aclError aclrtSynchronizeDevice(void) { return acl_from_cuda(cudaDeviceSynchronize()); }

aclError aclrtCreateEvent(aclrtEvent *event) {
    return acl_from_cuda(cudaEventCreate(reinterpret_cast<cudaEvent_t *>(event)));
}
aclError aclrtDestroyEvent(aclrtEvent event) { return acl_from_cuda(cudaEventDestroy((cudaEvent_t)event)); }
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) {
    return acl_from_cuda(cudaEventRecord((cudaEvent_t)event, (cudaStream_t)stream));
}
aclError aclrtSynchronizeEvent(aclrtEvent event) { return acl_from_cuda(cudaEventSynchronize((cudaEvent_t)event)); }
aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) {
    if (!ms) return ACLNN_ERR_PARAM_NULLPTR;
    return acl_from_cuda(cudaEventElapsedTime(ms, (cudaEvent_t)startEvent, (cudaEvent_t)endEvent));
}

// GB10 has unified memory, but we keep explicit device allocation to faithfully preserve aclrt semantics
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
    (void)policy;
    return acl_from_cuda(g_alloc().alloc(devPtr, size));   // caching allocator
}
aclError aclrtFree(void *devPtr) { return acl_from_cuda(g_alloc().free(devPtr)); }

aclError aclrtMallocHost(void **hostPtr, size_t size) { return acl_from_cuda(cudaMallocHost(hostPtr, size)); }
aclError aclrtFreeHost(void *hostPtr)                 { return acl_from_cuda(cudaFreeHost(hostPtr)); }

static cudaMemcpyKind to_cuda_kind(aclrtMemcpyKind k) {
    switch (k) {
        case ACL_MEMCPY_HOST_TO_HOST:     return cudaMemcpyHostToHost;
        case ACL_MEMCPY_HOST_TO_DEVICE:   return cudaMemcpyHostToDevice;
        case ACL_MEMCPY_DEVICE_TO_HOST:   return cudaMemcpyDeviceToHost;
        case ACL_MEMCPY_DEVICE_TO_DEVICE: return cudaMemcpyDeviceToDevice;
        default:                          return cudaMemcpyDefault;
    }
}

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    if (count > destMax) return ACLNN_ERR_PARAM_INVALID;
    return acl_from_cuda(cudaMemcpy(dst, src, count, to_cuda_kind(kind)));
}
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                          aclrtMemcpyKind kind, aclrtStream stream) {
    if (count > destMax) return ACLNN_ERR_PARAM_INVALID;
    return acl_from_cuda(cudaMemcpyAsync(dst, src, count, to_cuda_kind(kind), (cudaStream_t)stream));
}
aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
    if (count > maxCount) return ACLNN_ERR_PARAM_INVALID;
    return acl_from_cuda(cudaMemset(devPtr, value, count));
}

size_t aclDataTypeSize(aclDataType dataType) { return dtype_size(dataType); }

// ---- CUDA Graph capture/replay: aclmdlRICapture* → cudaStreamBeginCapture/EndCapture + cudaGraph(Exec).
// Autoregressive decoding launches tens of small kernels per token; launch overhead dominates.
// Capturing once and replaying many times significantly reduces that cost.
// Note: operators must not call cudaMalloc during capture (illegal). Use the caching allocator
// with a warm-up pass beforehand to pre-populate the cache.
namespace { struct ShimGraph { cudaGraph_t graph = nullptr; cudaGraphExec_t exec = nullptr; }; }
static cudaStreamCaptureMode cap_mode(aclmdlRICaptureMode m) {
    switch (m) { case ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL: return cudaStreamCaptureModeThreadLocal;
                 case ACL_MODEL_RI_CAPTURE_MODE_RELAXED:      return cudaStreamCaptureModeRelaxed;
                 default:                                     return cudaStreamCaptureModeGlobal; }
}
aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode) {
    return acl_from_cuda(cudaStreamBeginCapture((cudaStream_t)stream, cap_mode(mode)));
}
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI) {
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone; unsigned long long id = 0;
    cudaError_t e = cudaStreamGetCaptureInfo((cudaStream_t)stream, &cs, &id);
    if (status) *status = (cs == cudaStreamCaptureStatusActive) ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE
                        : (cs == cudaStreamCaptureStatusInvalidated) ? ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED
                        : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    if (modelRI) *modelRI = nullptr;   // capture is still in progress; no completed model yet
    return acl_from_cuda(e);
}
aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI) {
    if (!modelRI) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    cudaGraph_t g = nullptr;
    cudaError_t e = cudaStreamEndCapture((cudaStream_t)stream, &g);
    if (e != cudaSuccess) return acl_from_cuda(e);
    auto *sg = new ShimGraph(); sg->graph = g;
    cudaError_t ie = cudaGraphInstantiate(&sg->exec, g, 0);
    if (ie != cudaSuccess) { cudaGraphDestroy(g); delete sg; return acl_from_cuda(ie); }
    *modelRI = (aclmdlRI)sg; return ACL_SUCCESS;
}
aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) {
    if (!modelRI) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    return acl_from_cuda(cudaGraphLaunch(((ShimGraph *)modelRI)->exec, (cudaStream_t)stream));
}
aclError aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout) {
    (void)timeout; if (!modelRI) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    cudaError_t e = cudaGraphLaunch(((ShimGraph *)modelRI)->exec, 0);
    if (e != cudaSuccess) return acl_from_cuda(e);
    return acl_from_cuda(cudaStreamSynchronize(0));
}
aclError aclmdlRIDestroy(aclmdlRI modelRI) {
    if (!modelRI) return ACL_SUCCESS;
    auto *sg = (ShimGraph *)modelRI;
    if (sg->exec) cudaGraphExecDestroy(sg->exec);
    if (sg->graph) cudaGraphDestroy(sg->graph);
    delete sg; return ACL_SUCCESS;
}
// Graph update/task-group (re-capture covers the override semantics): currently a transparent no-op
aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) { (void)mode; return ACL_SUCCESS; }

// ---- Miscellaneous runtime APIs ----

// 2D pitched copy → cudaMemcpy2D (width/height in bytes/rows; pitch is row stride in bytes)
aclError aclrtMemcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, aclrtMemcpyKind kind) {
    return acl_from_cuda(cudaMemcpy2D(dst, dpitch, src, spitch, width, height, to_cuda_kind(kind)));
}
aclError aclrtMemcpy2dAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                            size_t width, size_t height, aclrtMemcpyKind kind, aclrtStream stream) {
    return acl_from_cuda(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height,
                                           to_cuda_kind(kind), (cudaStream_t)stream));
}
aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream) {
    if (count > maxCount) return ACLNN_ERR_PARAM_INVALID;
    return acl_from_cuda(cudaMemsetAsync(devPtr, value, count, (cudaStream_t)stream));
}

// Create stream with priority/flags: smaller ACL priority value means higher priority, consistent with CUDA, passed through directly
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
    unsigned cf = (flag != 0) ? cudaStreamNonBlocking : cudaStreamDefault;
    return acl_from_cuda(cudaStreamCreateWithPriority(reinterpret_cast<cudaStream_t *>(stream), cf, (int)priority));
}
aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority) {
    int lo = 0, hi = 0; ACL_CUDA(cudaDeviceGetStreamPriorityRange(&lo, &hi));
    if (leastPriority) *leastPriority = lo; if (greatestPriority) *greatestPriority = hi;
    return ACL_SUCCESS;
}
aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) { (void)stream; (void)mode; return ACL_SUCCESS; }

// Event wait (cross-stream synchronization)
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) {
    return acl_from_cuda(cudaStreamWaitEvent((cudaStream_t)stream, (cudaEvent_t)event, 0));
}

// Peer access (multi-GPU)
aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId) {
    int can = 0; ACL_CUDA(cudaDeviceCanAccessPeer(&can, deviceId, peerDeviceId));
    if (canAccessPeer) *canAccessPeer = can; return ACL_SUCCESS;
}
aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) {
    cudaError_t e = cudaDeviceEnablePeerAccess(peerDeviceId, flags);
    if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); return ACL_SUCCESS; }   // idempotent
    return acl_from_cuda(e);
}
aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId) {
    return acl_from_cuda(cudaDeviceDisablePeerAccess(peerDeviceId));
}

// Host callback into stream: aclrtCallback and cudaHostFn_t share the same signature (void(*)(void*)), so we pass through directly
aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream) {
    (void)blockType;   // blocking semantics are handled internally by cudaLaunchHostFunc's callback thread
    return acl_from_cuda(cudaLaunchHostFunc((cudaStream_t)stream, (cudaHostFn_t)fn, userData));
}
// Ascend callbacks rely on a subscribe/poll thread; CUDA triggers callbacks from an internal runtime thread, so subscribe/poll are transparent no-ops
aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream) { (void)threadId; (void)stream; return ACL_SUCCESS; }
aclError aclrtProcessReport(int32_t timeout) { (void)timeout; return ACL_SUCCESS; }

} // extern "C"
