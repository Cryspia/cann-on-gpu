// Cross-check harness shared header: device buffers, tensor construction, two-phase execution, fp16 encode/decode, reporting,
//   and golden bridge (prefer reading golden .bin produced by cannsim/explorer; fall back to CPU reference if missing).
// All test clients can #include this header to avoid repeating boilerplate in each file.
#ifndef CANN_ON_GPU_TEST_HARNESS_H
#define CANN_ON_GPU_TEST_HARNESS_H

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <functional>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

#define HCHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

namespace hn {

inline aclrtStream g_stream;
inline int g_pass = 0, g_fail = 0;

inline void init() { HCHECK(aclInit(nullptr)); HCHECK(aclrtSetDevice(0)); HCHECK(aclrtCreateStream(&g_stream)); }
inline int finish() { HCHECK(aclrtDestroyStream(g_stream)); HCHECK(aclrtResetDevice(0)); HCHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail); return g_fail ? 1 : 0; }

struct DevBuf {
    void *p = nullptr; size_t bytes = 0;
    explicit DevBuf(size_t b) : bytes(b) { HCHECK(aclrtMalloc(&p, b, ACL_MEM_MALLOC_HUGE_FIRST)); }
    ~DevBuf() { if (p) aclrtFree(p); }
    DevBuf(const DevBuf &) = delete; DevBuf &operator=(const DevBuf &) = delete;
    void up(const void *h) { HCHECK(aclrtMemcpy(p, bytes, h, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void down(void *h) const { HCHECK(aclrtMemcpy(h, bytes, p, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

inline aclTensor *mk(const std::vector<int64_t> &dims, aclDataType dt, void *data) {
    return aclCreateTensor(dims.data(), dims.size(), dt, nullptr, 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

inline void report(const std::string &name, double err, double tol) {
    bool ok = err <= tol; (ok ? g_pass : g_fail)++;
    printf("%-32s err=%.2e tol=%.0e %s\n", name.c_str(), err, tol, ok ? "PASS" : "FAIL");
}
inline double relerr(double got, double ref, double atol = 1e-9) { return std::fabs(got - ref) / (std::fabs(ref) + atol); }
// Whole-matrix/tensor normalized error (avoids inflation by near-zero elements; use max|err|/max|ref| normalization)
inline double norm_err(const std::vector<float> &got, const std::vector<double> &ref) {
    double me = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); i++) { me = std::max(me, std::fabs(got[i] - ref[i])); mr = std::max(mr, std::fabs(ref[i])); }
    return me / (mr + 1e-9);
}

inline uint16_t f2h(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int32_t e = ((x >> 23) & 0xFF) - 127 + 15; uint32_t m = x & 0x7FFFFF;
    if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7C00); uint32_t r = m & 0x1FFF, h = s | (e << 10) | (m >> 13); if (r > 0x1000 || (r == 0x1000 && (h & 1))) h++; return (uint16_t)h; }
inline float h2f(uint16_t h) { uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, x; if (e == 0) { float f = m * 0x1p-24f; memcpy(&x, &f, 4); x |= s; }
    else if (e == 31) x = s | 0x7F800000 | (m << 13); else x = s | ((e - 15 + 127) << 23) | (m << 13); float f; memcpy(&f, &x, 4); return f; }

inline std::vector<float> randv(int64_t n, float lo = -1, float hi = 1) { std::vector<float> v(n); for (auto &x : v) x = lo + (hi - lo) * (rand() / (float)RAND_MAX); return v; }

// Two-phase execute + workspace allocation
template <typename GetWs>
inline void exec2(GetWs getws, aclnnStatus (*run)(void *, uint64_t, aclOpExecutor *, aclrtStream)) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    HCHECK(getws(&ws, &ex));
    void *wsp = nullptr; if (ws) HCHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    HCHECK(run(wsp, ws, ex, g_stream));
    HCHECK(aclrtSynchronizeStream(g_stream));
    if (wsp) aclrtFree(wsp);
}

// ---- golden bridge ----
// Convention: cannsim/explorer writes a golden output tensor (fp32 raw) to $CANN_GOLDEN_DIR/<key>.bin.
//   If present, use it as reference (against Ascend simulation results); otherwise fall back to caller-provided CPU reference (cpu_ref). Returns reference vector.
inline bool load_golden(const std::string &key, std::vector<float> &out) {
    const char *dir = getenv("CANN_GOLDEN_DIR"); if (!dir) return false;
    std::string path = std::string(dir) + "/" + key + ".bin";
    FILE *f = fopen(path.c_str(), "rb"); if (!f) return false;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(sz / sizeof(float));
    bool ok = fread(out.data(), sizeof(float), out.size(), f) == out.size();
    fclose(f); return ok;
}
inline void save_golden(const std::string &key, const std::vector<float> &v) {
    const char *dir = getenv("CANN_GOLDEN_DIR"); if (!dir) return;
    std::string path = std::string(dir) + "/" + key + ".bin";
    FILE *f = fopen(path.c_str(), "wb"); if (!f) return;
    fwrite(v.data(), sizeof(float), v.size(), f); fclose(f);
}
// Prefer golden; fall back to CPU reference. Source tag is printed after the case name.
inline std::vector<double> golden_or_cpu(const std::string &key, std::function<std::vector<double>()> cpu_ref, std::string &src) {
    std::vector<float> g;
    if (load_golden(key, g)) { src = "[golden]"; return std::vector<double>(g.begin(), g.end()); }
    src = "[cpu]"; return cpu_ref();
}

} // namespace hn
#endif
