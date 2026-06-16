// Runtime edge-case API cross-check: Memcpy2d / MemsetAsync / stream creation with priority /
// cross-stream event wait / peer access query / host callback into stream. Pure ACL client, no CUDA headers.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include "acl/acl.h"

static int g_pass = 0, g_fail = 0;
static void report(const char *name, bool ok) {
    (ok ? g_pass : g_fail)++;
    printf("%-34s %s\n", name, ok ? "PASS" : "FAIL");
}
#define CHECK(x) do { int __r=(int)(x); if(__r!=0){ printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,__r); } } while(0)

// 2D pitched copy: packed host [H,W] -> device pitched (pitch>W) -> packed host; readback must match
static void t_memcpy2d() {
    const int H = 4, W = 3; const size_t pitch = 8 * sizeof(float);   // device row stride: 8 elements
    std::vector<float> src(H*W), back(H*W, -1);
    for (int i=0;i<H*W;i++) src[i] = (float)(i+1);
    void *dev = nullptr; CHECK(aclrtMalloc(&dev, pitch*H, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy2d(dev, pitch, src.data(), W*sizeof(float), W*sizeof(float), H, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy2d(back.data(), W*sizeof(float), dev, pitch, W*sizeof(float), H, ACL_MEMCPY_DEVICE_TO_HOST));
    report("Memcpy2d round-trip", memcmp(src.data(), back.data(), H*W*sizeof(float))==0);
    aclrtFree(dev);
}

// MemsetAsync: zero-fill then readback must be all zeros
static void t_memset_async() {
    const int n = 256; void *dev=nullptr; CHECK(aclrtMalloc(&dev, n*4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream s; CHECK(aclrtCreateStream(&s));
    std::vector<float> ones(n, 7.0f); CHECK(aclrtMemcpyAsync(dev, n*4, ones.data(), n*4, ACL_MEMCPY_HOST_TO_DEVICE, s));
    CHECK(aclrtMemsetAsync(dev, n*4, 0, n*4, s));
    std::vector<float> back(n, -1); CHECK(aclrtMemcpyAsync(back.data(), n*4, dev, n*4, ACL_MEMCPY_DEVICE_TO_HOST, s));
    CHECK(aclrtSynchronizeStream(s));
    bool ok=true; for (float v : back) if (v!=0.f) ok=false;
    report("MemsetAsync zero", ok);
    aclrtDestroyStream(s); aclrtFree(dev);
}

// Stream creation with priority + priority range query: passes if stream creation succeeds and synchronization works
static void t_stream_priority() {
    int32_t least=0, greatest=0; CHECK(aclrtDeviceGetStreamPriorityRange(&least, &greatest));
    aclrtStream s=nullptr; aclError e = aclrtCreateStreamWithConfig(&s, (uint32_t)greatest, 0);
    bool ok = (e==ACL_SUCCESS) && s!=nullptr && (greatest<=least);
    if (s) { CHECK(aclrtSynchronizeStream(s)); aclrtDestroyStream(s); }
    report("CreateStreamWithConfig+range", ok);
}

// Cross-stream event wait: stream A writes data then records event; stream B waits for event then copies out, must see A's result
static void t_stream_wait_event() {
    const int n=64; void *dev=nullptr; CHECK(aclrtMalloc(&dev, n*4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclrtStream a,b; CHECK(aclrtCreateStream(&a)); CHECK(aclrtCreateStream(&b));
    aclrtEvent ev; CHECK(aclrtCreateEvent(&ev));
    std::vector<float> vin(n, 3.5f), back(n, -1);
    CHECK(aclrtMemcpyAsync(dev, n*4, vin.data(), n*4, ACL_MEMCPY_HOST_TO_DEVICE, a));
    CHECK(aclrtRecordEvent(ev, a));
    CHECK(aclrtStreamWaitEvent(b, ev));
    CHECK(aclrtMemcpyAsync(back.data(), n*4, dev, n*4, ACL_MEMCPY_DEVICE_TO_HOST, b));
    CHECK(aclrtSynchronizeStream(b));
    bool ok=true; for (float v:back) if (v!=3.5f) ok=false;
    report("StreamWaitEvent cross-stream", ok);
    aclrtDestroyEvent(ev); aclrtDestroyStream(a); aclrtDestroyStream(b); aclrtFree(dev);
}

// Peer access query: in single-GPU environment querying self gives 0 (cannot peer-access self); passes if API does not error
static void t_peer_query() {
    uint32_t cnt=0; CHECK(aclrtGetDeviceCount(&cnt));
    int32_t can=-1; aclError e = aclrtDeviceCanAccessPeer(&can, 0, 0);
    report("DeviceCanAccessPeer query", e==ACL_SUCCESS && can==0);
}

// Host callback into stream: callback sets userData flag; must be triggered after stream synchronization
static int g_cb_hit = 0;
static void my_callback(void *ud) { *(int*)ud = 1; }
static void t_launch_callback() {
    g_cb_hit = 0;
    aclrtStream s; CHECK(aclrtCreateStream(&s));
    CHECK(aclrtSubscribeReport(0, s));
    aclError e = aclrtLaunchCallback(my_callback, &g_cb_hit, ACL_CALLBACK_BLOCK, s);
    CHECK(aclrtSynchronizeStream(s));
    report("LaunchCallback fired", e==ACL_SUCCESS && g_cb_hit==1);
    aclrtDestroyStream(s);
}

int main() {
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0));
    t_memcpy2d();
    t_memset_async();
    t_stream_priority();
    t_stream_wait_event();
    t_peer_query();
    t_launch_callback();
    CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
