// Runtime PTX modular demo: replicates the CANN operator binary "runtime load/dispatch" architecture.
//   NVRTC compiles the operator CUDA source string to PTX -> CUDA Driver API cuModuleLoadData loads it at runtime ->
//   cuModuleGetFunction retrieves the kernel -> cuLaunchKernel launches it (contrast with the current shim which statically links kernels into .so).
//   This chain is the equivalent of a real CANN runtime loading a .o operator binary and launching it.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <cuda.h>
#include <nvrtc.h>

#define CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char *s; cuGetErrorString(r, &s); \
    fprintf(stderr, "CU FAIL %s:%d %s\n", __FILE__, __LINE__, s); exit(1); } } while (0)
#define NV(x) do { nvrtcResult r = (x); if (r != NVRTC_SUCCESS) { \
    fprintf(stderr, "NVRTC FAIL %s:%d %s\n", __FILE__, __LINE__, nvrtcGetErrorString(r)); exit(1); } } while (0)

// CUDA source for one "operator" (compiled at runtime, not linked into .so)
static const char *kVadd = R"(
extern "C" __global__ void vadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
)";

int main() {
    CU(cuInit(0));
    CUdevice dev; CU(cuDeviceGet(&dev, 0));
    int major = 0, minor = 0;
    CU(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
    CU(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));
    CUcontext ctx; CU(cuCtxCreate(&ctx, nullptr, 0, dev));   // CUDA 13: cuCtxCreate_v4(ctx, params=null, flags, dev) -- using legacy form
    printf("[driver] device CC = sm_%d%d\n", major, minor);

    // 1) NVRTC compile operator source at runtime -> PTX (arch follows device CC)
    nvrtcProgram prog;
    NV(nvrtcCreateProgram(&prog, kVadd, "vadd.cu", 0, nullptr, nullptr));
    char arch[32]; snprintf(arch, sizeof arch, "--gpu-architecture=compute_%d%d", major, minor);
    const char *opts[] = {arch};
    nvrtcResult cr = nvrtcCompileProgram(prog, 1, opts);
    if (cr != NVRTC_SUCCESS) { size_t ls; nvrtcGetProgramLogSize(prog, &ls); std::vector<char> log(ls); nvrtcGetProgramLog(prog, log.data()); fprintf(stderr, "%s\n", log.data()); return 1; }
    size_t ptxSize; NV(nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptx(ptxSize); NV(nvrtcGetPTX(prog, ptx.data()));
    NV(nvrtcDestroyProgram(&prog));
    printf("[nvrtc] compiled vadd -> PTX (%zu bytes)\n", ptxSize);

    // 2) Driver API loads PTX module at runtime + retrieves function (equivalent to CANN loading an operator binary)
    CUmodule mod; CU(cuModuleLoadData(&mod, ptx.data()));
    CUfunction fn; CU(cuModuleGetFunction(&fn, mod, "vadd"));
    printf("[driver] cuModuleLoadData + cuModuleGetFunction OK\n");

    // 3) Data setup + cuLaunchKernel
    const int n = 1 << 20;
    std::vector<float> ha(n), hb(n), hc(n);
    for (int i = 0; i < n; i++) { ha[i] = (float)i * 0.5f; hb[i] = (float)(n - i); }
    CUdeviceptr da, db, dc;
    CU(cuMemAlloc(&da, n * 4)); CU(cuMemAlloc(&db, n * 4)); CU(cuMemAlloc(&dc, n * 4));
    CU(cuMemcpyHtoD(da, ha.data(), n * 4)); CU(cuMemcpyHtoD(db, hb.data(), n * 4));
    int nn = n; void *args[] = {&da, &db, &dc, &nn};
    int threads = 256, blocks = (n + threads - 1) / threads;
    CU(cuLaunchKernel(fn, blocks, 1, 1, threads, 1, 1, 0, nullptr, args, nullptr));
    CU(cuCtxSynchronize());
    CU(cuMemcpyDtoH(hc.data(), dc, n * 4));

    long bad = 0; for (int i = 0; i < n; i++) if (hc[i] != ha[i] + hb[i]) bad++;
    printf("[launch] vadd n=%d bad=%ld -> %s\n", n, bad, bad ? "FAIL" : "PASS");

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    CU(cuModuleUnload(mod)); CU(cuCtxDestroy(ctx));
    printf("== RUNTIME PTX MODULE: %s ==\n", bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
