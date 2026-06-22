// aclrt* / aclInit -> Metal runtime. Signature contract from $ACL_INCLUDE/acl/acl_rt.h.
// Unified memory: aclrtMalloc returns an MTLBuffer's shared contents pointer (host-addressable AND the
// device pointer), so H2D/D2H copies are plain memcpy. Correctness relies on the caller synchronizing the
// stream before reading back (as the ACL contract requires), which the tests do.
#import "internal.h"
#include <cstring>

namespace { inline aclError ok() { return ACL_SUCCESS; }
            inline aclError errn() { return (aclError)ACLNN_ERR_RUNTIME_ERROR; }
            inline aclError errp() { return (aclError)ACLNN_ERR_PARAM_NULLPTR; } }

extern "C" {

aclError aclInit(const char *configPath) { (void)configPath; mtl::init(); return mtl::device() ? ok() : errn(); }
aclError aclFinalize() { return ok(); }

// Single Metal device; context APIs are transparent no-ops in the shim.
aclError aclrtSetDevice(int32_t deviceId)   { (void)deviceId; mtl::init(); return mtl::device() ? ok() : errn(); }
aclError aclrtResetDevice(int32_t deviceId) { (void)deviceId; return ok(); }
aclError aclrtGetDevice(int32_t *deviceId)  { if (deviceId) *deviceId = 0; return ok(); }
aclError aclrtGetDeviceCount(uint32_t *count) { if (count) *count = mtl::device() ? 1 : 0; return ok(); }

aclError aclrtCreateContext(aclrtContext *context, int32_t deviceId) {
    (void)deviceId; if (!context) return errp(); *context = (aclrtContext)(intptr_t)1; return ok();
}
aclError aclrtDestroyContext(aclrtContext context)    { (void)context; return ok(); }
aclError aclrtSetCurrentContext(aclrtContext context) { (void)context; return ok(); }

// Stream = command queue + last-committed command buffer (for synchronize).
aclError aclrtCreateStream(aclrtStream *stream) {
    if (!stream) return errp();
    mtl::init();
    auto *s = new AclStream();
    s->q = [mtl::device() newCommandQueue];
    *stream = (aclrtStream)s;
    return s->q ? ok() : errn();
}
aclError aclrtDestroyStream(aclrtStream stream) {
    if (stream) { auto *s = (AclStream *)stream; s->last = nil; s->q = nil; delete s; }
    return ok();
}
aclError aclrtSynchronizeStream(aclrtStream stream) {
    if (stream) { auto *s = (AclStream *)stream; if (s->last) [s->last waitUntilCompleted]; }
    return ok();
}
aclError aclrtSynchronizeDevice(void) { return ok(); }   // per-stream sync covers correctness in the shim

// Memory: shared MTLBuffer; the returned pointer is both host- and device-addressable.
aclError aclrtMalloc(void **devPtr, size_t size, aclrtMemMallocPolicy policy) {
    (void)policy; if (!devPtr) return errp();
    void *p = mtl::alloc(size); if (!p) return errn();
    *devPtr = p; return ok();
}
aclError aclrtFree(void *devPtr) { mtl::free_(devPtr); return ok(); }

aclError aclrtMallocHost(void **hostPtr, size_t size) {
    if (!hostPtr) return errp(); *hostPtr = malloc(size); return *hostPtr ? ok() : errn();
}
aclError aclrtFreeHost(void *hostPtr) { free(hostPtr); return ok(); }

aclError aclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, aclrtMemcpyKind kind) {
    (void)kind; if (count > destMax) return (aclError)ACLNN_ERR_PARAM_INVALID;
    memcpy(dst, src, count); return ok();
}
aclError aclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count,
                          aclrtMemcpyKind kind, aclrtStream stream) {
    (void)kind; (void)stream; if (count > destMax) return (aclError)ACLNN_ERR_PARAM_INVALID;
    memcpy(dst, src, count); return ok();   // unified memory; async copy is treated as synchronous
}
aclError aclrtMemset(void *devPtr, size_t maxCount, int32_t value, size_t count) {
    if (count > maxCount) return (aclError)ACLNN_ERR_PARAM_INVALID;
    memset(devPtr, value, count); return ok();
}
aclError aclrtMemsetAsync(void *devPtr, size_t maxCount, int32_t value, size_t count, aclrtStream stream) {
    (void)stream; if (count > maxCount) return (aclError)ACLNN_ERR_PARAM_INVALID;
    memset(devPtr, value, count); return ok();
}

size_t aclDataTypeSize(aclDataType dataType) { return dtype_size(dataType); }

// ---- 2D pitched copy (row-by-row over unified memory) ----
aclError aclrtMemcpy2d(void *dst, size_t dpitch, const void *src, size_t spitch,
                       size_t width, size_t height, aclrtMemcpyKind kind) {
    (void)kind;
    for (size_t r = 0; r < height; ++r)
        memcpy((char *)dst + r * dpitch, (const char *)src + r * spitch, width);
    return ok();
}
aclError aclrtMemcpy2dAsync(void *dst, size_t dpitch, const void *src, size_t spitch,
                            size_t width, size_t height, aclrtMemcpyKind kind, aclrtStream stream) {
    (void)stream; return aclrtMemcpy2d(dst, dpitch, src, spitch, width, height, kind);
}

// ---- Stream priority / config ----
aclError aclrtCreateStreamWithConfig(aclrtStream *stream, uint32_t priority, uint32_t flag) {
    (void)priority; (void)flag; return aclrtCreateStream(stream);   // single GPU; priority is advisory
}
aclError aclrtDeviceGetStreamPriorityRange(int32_t *leastPriority, int32_t *greatestPriority) {
    if (leastPriority) *leastPriority = 0;
    if (greatestPriority) *greatestPriority = 0;   // Metal has no public stream-priority range
    return ok();
}
aclError aclrtSetStreamFailureMode(aclrtStream stream, uint64_t mode) { (void)stream; (void)mode; return ok(); }

// ---- Events. In the current synchronous model (host memcpy + per-stream sync) record/wait are no-ops:
// program-order correctness already holds. Wired to MTLSharedEvent when the GPU-async decode path needs it. ----
aclError aclrtCreateEvent(aclrtEvent *event) { if (!event) return errp(); *event = (aclrtEvent)new uint64_t(0); return ok(); }
aclError aclrtDestroyEvent(aclrtEvent event) { delete (uint64_t *)event; return ok(); }
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream) { (void)event; (void)stream; return ok(); }
aclError aclrtSynchronizeEvent(aclrtEvent event) { (void)event; return ok(); }
aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) { (void)stream; (void)event; return ok(); }
aclError aclrtEventElapsedTime(float *ms, aclrtEvent startEvent, aclrtEvent endEvent) {
    (void)startEvent; (void)endEvent; if (ms) *ms = 0.f; return ok();
}

// ---- Peer access (single device: cannot peer-access self) ----
aclError aclrtDeviceCanAccessPeer(int32_t *canAccessPeer, int32_t deviceId, int32_t peerDeviceId) {
    (void)deviceId; (void)peerDeviceId; if (canAccessPeer) *canAccessPeer = 0; return ok();
}
aclError aclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags) { (void)peerDeviceId; (void)flags; return ok(); }
aclError aclrtDeviceDisablePeerAccess(int32_t peerDeviceId) { (void)peerDeviceId; return ok(); }

// ---- Host callback into stream: drain pending stream work, then invoke synchronously ----
aclError aclrtLaunchCallback(aclrtCallback fn, void *userData, aclrtCallbackBlockType blockType, aclrtStream stream) {
    (void)blockType;
    auto *s = (AclStream *)stream;
    if (s && s->last) [s->last waitUntilCompleted];
    if (fn) fn(userData);
    return ok();
}
aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream) { (void)threadId; (void)stream; return ok(); }
aclError aclrtProcessReport(int32_t timeout) { (void)timeout; return ok(); }

} // extern "C"
