// Pure ACL client: written entirely with AscendCL API, linked against the shim's libascendcl.so running on GPU.
// Validates the two-phase aclnnAdd call with CPU double reference and tolerance cross-check.
// Case 1 replicates the explorer/cannsim Add golden scenario (x=1, y=2, n=8*2048, expected 3).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FAIL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

static uint16_t f2h(float f) { // float -> fp16 (round-to-nearest), used for test inputs
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)sign;
    if (e >= 31) return (uint16_t)(sign | 0x7C00);
    uint32_t r = m & 0x1FFF, h = sign | (e << 10) | (m >> 13);
    if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++;
    return (uint16_t)h;
}
static float h2f(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x;
    if (e == 0) { float f = m * 0x1p-24f; __builtin_memcpy(&x, &f, 4); x |= sign; }
    else if (e == 31) x = sign | 0x7F800000 | (m << 13);
    else x = sign | ((e - 15 + 127) << 23) | (m << 13);
    float f; __builtin_memcpy(&f, &x, 4); return f;
}

template <typename T>
static int run_case(const char *name, int64_t n, float alphaVal, double rtol,
                    aclDataType dt, T (*enc)(float), float (*dec)(T), aclrtStream stream,
                    bool constInput) {
    size_t bytes = n * sizeof(T);
    std::vector<T> hx(n), hy(n), hz(n);
    std::vector<double> ref(n);
    srand(42);
    for (int64_t i = 0; i < n; i++) {
        float a = constInput ? 1.0f : (rand() / (float)RAND_MAX) * 4.f - 2.f;
        float b = constInput ? 2.0f : (rand() / (float)RAND_MAX) * 4.f - 2.f;
        hx[i] = enc(a); hy[i] = enc(b);
        ref[i] = (double)dec(hx[i]) + (double)alphaVal * (double)dec(hy[i]);
    }
    void *dx, *dy, *dz;
    CHECK(aclrtMalloc(&dx, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dy, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dz, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx, bytes, hx.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(dy, bytes, hy.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t dims[1] = {n};
    aclTensor *tx = aclCreateTensor(dims, 1, dt, nullptr, 0, ACL_FORMAT_ND, dims, 1, dx);
    aclTensor *ty = aclCreateTensor(dims, 1, dt, nullptr, 0, ACL_FORMAT_ND, dims, 1, dy);
    aclTensor *tz = aclCreateTensor(dims, 1, dt, nullptr, 0, ACL_FORMAT_ND, dims, 1, dz);
    aclScalar *alpha = aclCreateScalar(&alphaVal, ACL_FLOAT);

    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnAddGetWorkspaceSize(tx, ty, alpha, tz, &ws, &ex));
    void *wsp = nullptr;
    if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnAdd(wsp, ws, ex, stream));
    CHECK(aclrtSynchronizeStream(stream));
    CHECK(aclrtMemcpy(hz.data(), bytes, dz, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    int errors = 0;
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) {
        double got = dec(hz[i]);
        double rel = std::fabs(got - ref[i]) / (std::fabs(ref[i]) + 1e-12);
        if (rel > maxrel) maxrel = rel;
        if (rel > rtol) { if (errors < 3) printf("  idx %ld got %g ref %g\n", (long)i, got, ref[i]); errors++; }
    }
    printf("%-22s n=%-7ld alpha=%-4g maxrel=%.2e tol=%.0e -> %s\n",
           name, (long)n, alphaVal, maxrel, rtol, errors ? "FAIL" : "PASS");

    aclDestroyScalar(alpha);
    aclDestroyTensor(tx); aclDestroyTensor(ty); aclDestroyTensor(tz);
    if (wsp) aclrtFree(wsp);
    aclrtFree(dx); aclrtFree(dy); aclrtFree(dz);
    return errors ? 1 : 0;
}

static float idf(float f) { return f; }
static int32_t enci(float f) { return (int32_t)f; }
static float deci(int32_t i) { return (float)i; }

int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    CHECK(aclrtCreateStream(&stream));

    int fails = 0;
    // case 1: explorer/cannsim golden scenario (constant 1+2=3, n=8*2048)
    fails += run_case<float>("fp32 golden(1+2)", 8 * 2048, 1.0f, 1e-6, ACL_FLOAT, idf, idf, stream, true);
    fails += run_case<float>("fp32 random", 1 << 20, 1.0f, 1e-6, ACL_FLOAT, idf, idf, stream, false);
    fails += run_case<float>("fp32 alpha=2.5", 1 << 20, 2.5f, 1e-6, ACL_FLOAT, idf, idf, stream, false);
    fails += run_case<uint16_t>("fp16 random", 1 << 20, 1.0f, 2e-3, ACL_FLOAT16, f2h, h2f, stream, false);
    fails += run_case<int32_t>("int32 random", 1 << 20, 1.0f, 0.0, ACL_INT32, enci, deci, stream, false);

    CHECK(aclrtDestroyStream(stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf(fails ? "ADD ON GPU: FAILED (%d)\n" : "ADD ON GPU: ALL PASSED\n", fails);
    return fails ? 1 : 0;
}
