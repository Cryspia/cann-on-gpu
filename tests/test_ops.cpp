// Full operator cross-check: pure ACL client -> shim -> GPU; CPU double reference + tolerance, targeting cannsim semantics (non bit-exact).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <functional>
#include <string>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

static aclrtStream g_stream;
static int g_pass = 0, g_fail = 0;

static uint16_t f2h(float f) {
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

struct DevBuf {
    void *p = nullptr; size_t bytes = 0;
    DevBuf(size_t b) : bytes(b) { CHECK(aclrtMalloc(&p, b, ACL_MEM_MALLOC_HUGE_FIRST)); }
    ~DevBuf() { aclrtFree(p); }
    void up(const void *h)   { CHECK(aclrtMemcpy(p, bytes, h, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void down(void *h) const { CHECK(aclrtMemcpy(h, bytes, p, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

static aclTensor *mk(const std::vector<int64_t> &dims, aclDataType dt, void *data) {
    return aclCreateTensor(dims.data(), dims.size(), dt, nullptr, 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}

static void report(const std::string &name, double maxrel, double tol) {
    bool ok = maxrel <= tol;
    (ok ? g_pass : g_fail)++;
    printf("%-28s maxrel=%.2e tol=%.0e %s\n", name.c_str(), maxrel, tol, ok ? "PASS" : "FAIL");
}

// Two-phase execution + workspace allocation
template <typename GetWs>
static void exec2(GetWs getws, aclnnStatus (*run)(void *, uint64_t, aclOpExecutor *, aclrtStream)) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(getws(&ws, &ex));
    void *wsp = nullptr;
    if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    if (wsp) aclrtFree(wsp);
}

static std::vector<float> randv(int64_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto &x : v) x = lo + (hi - lo) * (rand() / (float)RAND_MAX);
    return v;
}

// atol floor: when fp16 multiply results land in the subnormal range pure relative error diverges; absolute error is ~ULP, so use |ref|+atol as denominator
static double relerr(double got, double ref, double atol = 1e-12) {
    return std::fabs(got - ref) / (std::fabs(ref) + atol);
}

// fp32 / fp16 binary：aclnnXxx(self, other, [alpha], out)
static void t_bin_f(const char *name, bool fp16, bool hasAlpha, float alpha,
                    std::function<aclnnStatus(aclTensor*, aclTensor*, aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                    aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                    std::function<double(double, double, double)> ref, float lo, float hi, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, lo, hi), fb = randv(n, lo, hi);
    aclDataType dt = fp16 ? ACL_FLOAT16 : ACL_FLOAT;
    size_t esz = fp16 ? 2 : 4, bytes = n * esz;
    std::vector<uint8_t> ha(bytes), hb(bytes), hz(bytes);
    for (int64_t i = 0; i < n; i++) {
        if (fp16) { ((uint16_t*)ha.data())[i] = f2h(fa[i]); ((uint16_t*)hb.data())[i] = f2h(fb[i]); }
        else      { ((float*)ha.data())[i] = fa[i];         ((float*)hb.data())[i] = fb[i]; }
    }
    DevBuf da(bytes), db(bytes), dz(bytes);
    da.up(ha.data()); db.up(hb.data());
    aclTensor *ta = mk({n}, dt, da.p), *tb = mk({n}, dt, db.p), *tz = mk({n}, dt, dz.p);
    aclScalar *al = hasAlpha ? aclCreateScalar(&alpha, ACL_FLOAT) : nullptr;
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tb, al, tz, w, e); }, run);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) {
        double x = fp16 ? h2f(((uint16_t*)ha.data())[i]) : ((float*)ha.data())[i];
        double y = fp16 ? h2f(((uint16_t*)hb.data())[i]) : ((float*)hb.data())[i];
        double got = fp16 ? h2f(((uint16_t*)hz.data())[i]) : ((float*)hz.data())[i];
        maxrel = std::max(maxrel, relerr(got, ref(x, y, alpha), fp16 ? 1e-2 : 1e-12));
    }
    report(std::string(name) + (fp16 ? " fp16" : " fp32"), maxrel, tol);
    if (al) aclDestroyScalar(al);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// fp32 tensor∘scalar：aclnnXxxs(self, scalar, [alpha], out)
static void t_scalar_f(const char *name, float sval, bool hasAlpha, float alpha,
                       std::function<aclnnStatus(aclTensor*, aclScalar*, aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                       aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                       std::function<double(double, double)> ref, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, -2, 2);
    size_t bytes = n * 4;
    std::vector<float> hz(n);
    DevBuf da(bytes), dz(bytes);
    da.up(fa.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    aclScalar *s = aclCreateScalar(&sval, ACL_FLOAT);
    aclScalar *al = hasAlpha ? aclCreateScalar(&alpha, ACL_FLOAT) : nullptr;
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, s, al, tz, w, e); }, run);
    dz.down(hz.data());
    double maxrel = 0;
    double sEff = sval * (hasAlpha ? alpha : 1.0f);
    for (int64_t i = 0; i < n; i++) maxrel = std::max(maxrel, relerr(hz[i], ref(fa[i], sEff)));
    report(name, maxrel, tol);
    aclDestroyScalar(s); if (al) aclDestroyScalar(al);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

static void t_un_f(const char *name,
                   aclnnStatus (*getws)(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**),
                   aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                   std::function<double(double)> ref, float lo, float hi, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, lo, hi);
    size_t bytes = n * 4;
    std::vector<float> hz(n);
    DevBuf da(bytes), dz(bytes);
    da.up(fa.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tz, w, e); }, run);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) maxrel = std::max(maxrel, relerr(hz[i], ref(fa[i])));
    report(name, maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

static void t_bit32(const char *name,
                    std::function<aclnnStatus(aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                    aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                    std::function<int32_t(int32_t, int32_t)> ref, bool unary) {
    const int64_t n = 1 << 18;
    size_t bytes = n * 4;
    std::vector<int32_t> ha(n), hb(n), hz(n);
    for (int64_t i = 0; i < n; i++) { ha[i] = rand() - RAND_MAX / 2; hb[i] = rand() - RAND_MAX / 2; }
    DevBuf da(bytes), db(bytes), dz(bytes);
    da.up(ha.data()); db.up(hb.data());
    aclTensor *ta = mk({n}, ACL_INT32, da.p), *tb = mk({n}, ACL_INT32, db.p), *tz = mk({n}, ACL_INT32, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, unary ? nullptr : tb, tz, w, e); }, run);
    dz.down(hz.data());
    int64_t bad = 0;
    for (int64_t i = 0; i < n; i++) if (hz[i] != ref(ha[i], hb[i])) bad++;
    report(std::string(name) + " int32", bad ? 1.0 : 0.0, 0.0);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

static void t_cast(const char *name, aclDataType from, aclDataType to, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, -100, 100);
    size_t fsz = (from == ACL_FLOAT16) ? 2 : 4, tsz = (to == ACL_FLOAT16) ? 2 : 4;
    std::vector<uint8_t> ha(n * fsz), hz(n * tsz);
    for (int64_t i = 0; i < n; i++) {
        if (from == ACL_FLOAT) ((float*)ha.data())[i] = fa[i];
        else if (from == ACL_FLOAT16) ((uint16_t*)ha.data())[i] = f2h(fa[i]);
        else ((int32_t*)ha.data())[i] = (int32_t)fa[i];
    }
    DevBuf da(n * fsz), dz(n * tsz);
    da.up(ha.data());
    aclTensor *ta = mk({n}, from, da.p), *tz = mk({n}, to, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnCastGetWorkspaceSize(ta, to, tz, w, e); }, aclnnCast);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) {
        double src = (from == ACL_FLOAT) ? ((float*)ha.data())[i]
                   : (from == ACL_FLOAT16) ? h2f(((uint16_t*)ha.data())[i]) : ((int32_t*)ha.data())[i];
        double ref = (to == ACL_INT32) ? (double)(int32_t)src : src;
        double got = (to == ACL_FLOAT) ? ((float*)hz.data())[i]
                   : (to == ACL_FLOAT16) ? h2f(((uint16_t*)hz.data())[i]) : ((int32_t*)hz.data())[i];
        maxrel = std::max(maxrel, relerr(got, ref));
    }
    report(name, maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

static void t_where() {
    const int64_t n = 1 << 18;
    auto fa = randv(n, -2, 2), fb = randv(n, -2, 2);
    std::vector<uint8_t> hc(n);
    std::vector<float> hz(n);
    for (auto &c : hc) c = rand() & 1;
    DevBuf dc(n), da(n * 4), db(n * 4), dz(n * 4);
    dc.up(hc.data()); da.up(fa.data()); db.up(fb.data());
    aclTensor *tc = mk({n}, ACL_BOOL, dc.p), *ta = mk({n}, ACL_FLOAT, da.p),
              *tb = mk({n}, ACL_FLOAT, db.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSWhereGetWorkspaceSize(tc, ta, tb, tz, w, e); }, aclnnSWhere);
    dz.down(hz.data());
    int64_t bad = 0;
    for (int64_t i = 0; i < n; i++) if (hz[i] != (hc[i] ? fa[i] : fb[i])) bad++;
    report("SWhere fp32", bad ? 1.0 : 0.0, 0.0);
    aclDestroyTensor(tc); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// SWhere broadcast: cond[M,1], self[M,N], other scalar broadcast[1,1] -> out[M,N]
static void t_where_bcast() {
    const int64_t M = 8, N = 5;
    std::vector<uint8_t> hc(M); for (int64_t i=0;i<M;i++) hc[i]=(i%2);
    auto fa = randv(M*N, -2, 2); std::vector<float> fb(1, -7.0f);
    std::vector<float> hz(M*N);
    DevBuf dc(M), da(M*N*4), db(4), dz(M*N*4);
    dc.up(hc.data()); da.up(fa.data()); db.up(fb.data());
    aclTensor *tc = mk({M,1}, ACL_BOOL, dc.p), *ta = mk({M,N}, ACL_FLOAT, da.p),
              *tb = mk({1,1}, ACL_FLOAT, db.p), *tz = mk({M,N}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSWhereGetWorkspaceSize(tc, ta, tb, tz, w, e); }, aclnnSWhere);
    dz.down(hz.data());
    int64_t bad = 0;
    for (int64_t i=0;i<M;i++) for (int64_t j=0;j<N;j++) { float ref = hc[i] ? fa[i*N+j] : fb[0]; if (hz[i*N+j]!=ref) bad++; }
    report("SWhere bcast cond[M,1] other[1,1]", bad?1.0:0.0, 0.0);
    aclDestroyTensor(tc); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// BitwiseAnd broadcast: a[M,N] & b[1,N] (column-wise broadcast)
static void t_band_bcast() {
    const int64_t M = 6, N = 4;
    std::vector<int32_t> a(M*N), b(N), hz(M*N);
    for (int64_t i=0;i<M*N;i++) a[i]=(int32_t)(i*7+3);
    for (int64_t j=0;j<N;j++) b[j]=(int32_t)(0xF0 | j);
    DevBuf da(M*N*4), db(N*4), dz(M*N*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({M,N},ACL_INT32,da.p), *tb=mk({1,N},ACL_INT32,db.p), *tz=mk({M,N},ACL_INT32,dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnBitwiseAndTensorGetWorkspaceSize(ta, tb, tz, w, e); }, aclnnBitwiseAndTensor);
    dz.down(hz.data());
    int64_t bad=0; for (int64_t i=0;i<M;i++) for (int64_t j=0;j<N;j++) if (hz[i*N+j]!=(a[i*N+j]&b[j])) bad++;
    report("BitwiseAnd bcast [M,N]&[1,N]", bad?1.0:0.0, 0.0);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// Non-contiguous input: a is a transposed view of [M,N] (strides={1,M}), b/out contiguous [M,N].
static void t_strided_add() {
    const int64_t M = 32, N = 48;
    auto phys_a = randv(M * N, -2, 2), fb = randv(M * N, -2, 2);   // phys_a has physical layout [N,M]
    std::vector<float> hz(M * N);
    DevBuf da(M*N*4), db(M*N*4), dz(M*N*4);
    da.up(phys_a.data()); db.up(fb.data());
    int64_t vdims[2] = {M, N}, vstr[2] = {1, M};   // a[i,j] = phys_a[i*1 + j*M] (transposed view)
    aclTensor *ta = aclCreateTensor(vdims, 2, ACL_FLOAT, vstr, 0, ACL_FORMAT_ND, vdims, 2, da.p);
    aclTensor *tb = mk({M, N}, ACL_FLOAT, db.p), *tz = mk({M, N}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAddGetWorkspaceSize(ta, tb, nullptr, tz, w, e); }, aclnnAdd);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < N; j++) {
        double ref = phys_a[i + j * M] + fb[i * N + j];
        maxrel = std::max(maxrel, relerr(hz[i * N + j], ref));
    }
    report("Add transposed-view fp32", maxrel, 1e-6);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// Non-contiguous unary: transposed view fed directly into Sigmoid without materializing
static void t_strided_unary() {
    const int64_t M = 32, N = 48;
    auto phys_a = randv(M * N, -3, 3);   // physical layout [N,M]
    std::vector<float> hz(M * N);
    DevBuf da(M*N*4), dz(M*N*4); da.up(phys_a.data());
    int64_t vdims[2] = {M, N}, vstr[2] = {1, M};   // a[i,j] = phys_a[i + j*M] (transposed view)
    aclTensor *ta = aclCreateTensor(vdims, 2, ACL_FLOAT, vstr, 0, ACL_FORMAT_ND, vdims, 2, da.p);
    aclTensor *tz = mk({M, N}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnSigmoidGetWorkspaceSize(ta, tz, w, e); }, aclnnSigmoid);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < N; j++) {
        double x = phys_a[i + j * M], ref = 1.0 / (1.0 + std::exp(-x));
        maxrel = std::max(maxrel, relerr(hz[i * N + j], ref));
    }
    report("Sigmoid transposed-view fp32", maxrel, 1e-6);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Mixed dtype: a(fp16) + b(fp32) -> out(fp32) (PoC promotes to fp32)
static void t_mixed_add() {
    const int64_t n = 1 << 16;
    auto fa = randv(n, -2, 2), fb = randv(n, -2, 2);
    std::vector<uint16_t> ha(n); for (int64_t i = 0; i < n; i++) ha[i] = f2h(fa[i]);
    std::vector<float> hz(n);
    DevBuf da(n*2), db(n*4), dz(n*4);
    da.up(ha.data()); db.up(fb.data());
    aclTensor *ta = mk({n}, ACL_FLOAT16, da.p), *tb = mk({n}, ACL_FLOAT, db.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnMulGetWorkspaceSize(ta, tb, tz, w, e); }, aclnnMul);
    dz.down(hz.data());
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) maxrel = std::max(maxrel, relerr(hz[i], (double)h2f(ha[i]) * fb[i]));
    report("Mul fp16*fp32->fp32", maxrel, 1e-6);   // fp32 compute; only a is quantized to fp16
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// Arbitrary-dim reduction: reduce shape along axes (keepDim optional). redfn receives the reduction slice and returns the reference value.
static void t_redx(const char *name, std::vector<int64_t> shape, std::vector<int64_t> axes, bool keepDim,
                   std::function<aclnnStatus(aclTensor*, aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                   aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                   std::function<double(std::vector<double>&)> redfn, double tol) {
    int rank = (int)shape.size();
    int64_t n = 1; for (auto d : shape) n *= d;
    auto fa = randv(n, 0.5, 1.8);
    std::vector<bool> isred(rank, false); for (auto a : axes) isred[a] = true;
    std::vector<int64_t> kept, red, istr(rank); { int64_t acc = 1; for (int i = rank-1; i>=0; --i){ istr[i]=acc; acc*=shape[i]; } }
    std::vector<int64_t> ksz, rsz, kst, rst;
    for (int i = 0; i < rank; ++i) if (isred[i]) { rsz.push_back(shape[i]); rst.push_back(istr[i]); }
                                  else            { ksz.push_back(shape[i]); kst.push_back(istr[i]); }
    int64_t nout = 1; for (auto d : ksz) nout *= d;
    int64_t rcount = 1; for (auto d : rsz) rcount *= d;
    // output tensor dimensions
    std::vector<int64_t> odims;
    for (int i = 0; i < rank; ++i) { if (isred[i]) { if (keepDim) odims.push_back(1); } else odims.push_back(shape[i]); }
    if (odims.empty()) odims.push_back(1);

    DevBuf da(n*4), dz(std::max<int64_t>(nout,1)*4);
    da.up(fa.data());
    aclTensor *ta = mk(shape, ACL_FLOAT, da.p), *tz = mk(odims, ACL_FLOAT, dz.p);
    std::vector<int64_t> ax = axes;
    aclIntArray *dim = aclCreateIntArray(ax.data(), ax.size());
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, dim, tz, w, e); }, run);
    std::vector<float> got(nout); dz.down(got.data());
    double maxrel = 0;
    for (int64_t oi = 0; oi < nout; ++oi) {
        int64_t rem = oi, base = 0;
        for (int i = (int)ksz.size()-1; i >= 0; --i) { int64_t c = rem % ksz[i]; rem /= ksz[i]; base += c * kst[i]; }
        std::vector<double> slice;
        for (int64_t j = 0; j < rcount; ++j) {
            int64_t rr = j, off = base;
            for (int i = (int)rsz.size()-1; i >= 0; --i) { int64_t c = rr % rsz[i]; rr /= rsz[i]; off += c * rst[i]; }
            slice.push_back(fa[off]);
        }
        maxrel = std::max(maxrel, relerr(got[oi], redfn(slice)));
    }
    report(name, maxrel, tol);
    aclDestroyIntArray(dim); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// argmax/argmin (along a single dim, returns int64 indices)
static void t_arg(const char *name, std::vector<int64_t> shape, int64_t dim, bool ismax,
                  std::function<aclnnStatus(aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                  aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    int rank = (int)shape.size();
    int64_t n = 1; for (auto d : shape) n *= d;
    auto fa = randv(n, -2, 2);
    std::vector<int64_t> istr(rank); { int64_t acc=1; for (int i=rank-1;i>=0;--i){istr[i]=acc; acc*=shape[i];} }
    std::vector<int64_t> ksz, kst; for (int i=0;i<rank;++i) if (i!=dim){ksz.push_back(shape[i]); kst.push_back(istr[i]);}
    int64_t nout=1; for (auto d:ksz) nout*=d;
    std::vector<int64_t> odims; for (int i=0;i<rank;++i) if (i!=dim) odims.push_back(shape[i]); if (odims.empty()) odims.push_back(1);
    DevBuf da(n*4), dz(std::max<int64_t>(nout,1)*8);
    da.up(fa.data());
    aclTensor *ta = mk(shape, ACL_FLOAT, da.p), *tz = mk(odims, ACL_INT64, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, dim, tz, w, e); }, run);
    std::vector<int64_t> got(nout); dz.down(got.data());
    int64_t bad = 0;
    for (int64_t oi=0; oi<nout; ++oi) {
        int64_t rem=oi, base=0;
        for (int i=(int)ksz.size()-1;i>=0;--i){int64_t c=rem%ksz[i]; rem/=ksz[i]; base+=c*kst[i];}
        int64_t bestj=0; double best=fa[base];
        for (int64_t j=0;j<shape[dim];++j){ double v=fa[base+j*istr[dim]]; if (ismax? v>best : v<best){best=v; bestj=j;} }
        if (got[oi]!=bestj) bad++;
    }
    report(name, bad?1.0:0.0, 0.0);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Broadcast binary: a[da] ∘ b[db] -> out[dout], CPU NumPy-style broadcast reference
static void t_bcast(const char *name, bool fp16,
                    const std::vector<int64_t> &da_, const std::vector<int64_t> &db_, const std::vector<int64_t> &dout,
                    std::function<aclnnStatus(aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                    aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                    std::function<double(double, double)> ref, double tol) {
    auto numel = [](const std::vector<int64_t> &d) { int64_t n = 1; for (auto x : d) n *= x; return n; };
    int64_t na = numel(da_), nb = numel(db_), no = numel(dout);
    auto fa = randv(na, 0.5, 2), fb = randv(nb, 0.5, 2);
    size_t esz = fp16 ? 2 : 4;
    std::vector<uint8_t> ha(na * esz), hb(nb * esz), hz(no * esz);
    auto pack = [&](const std::vector<float> &src, std::vector<uint8_t> &dst) {
        for (size_t i = 0; i < src.size(); i++)
            if (fp16) { uint16_t h = f2h(src[i]); __builtin_memcpy(&dst[i*2], &h, 2); }
            else __builtin_memcpy(&dst[i*4], &src[i], 4);
    };
    pack(fa, ha); pack(fb, hb);
    DevBuf da(ha.size()), db(hb.size()), dz(hz.size());
    da.up(ha.data()); db.up(hb.data());
    aclDataType dt = fp16 ? ACL_FLOAT16 : ACL_FLOAT;
    aclTensor *ta = mk(da_, dt, da.p), *tb = mk(db_, dt, db.p), *tz = mk(dout, dt, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tb, tz, w, e); }, run);
    dz.down(hz.data());
    // CPU reference: recover a/b broadcast indices from the multi-dim out index
    int r = (int)dout.size(), ra = (int)da_.size(), rb = (int)db_.size();
    double maxrel = 0;
    for (int64_t i = 0; i < no; i++) {
        int64_t rem = i, ia = 0, ib = 0, sa = 1, sb = 1;
        std::vector<int64_t> coord(r);
        for (int d = r - 1; d >= 0; --d) { coord[d] = rem % dout[d]; rem /= dout[d]; }
        for (int d = ra - 1; d >= 0; --d) { int od = d + (r - ra); int64_t c = (da_[d] == 1) ? 0 : coord[od]; ia += c * sa; sa *= da_[d]; }
        for (int d = rb - 1; d >= 0; --d) { int od = d + (r - rb); int64_t c = (db_[d] == 1) ? 0 : coord[od]; ib += c * sb; sb *= db_[d]; }
        double got = fp16 ? h2f(((uint16_t*)hz.data())[i]) : ((float*)hz.data())[i];
        maxrel = std::max(maxrel, relerr(got, ref(fa[ia], fb[ib])));
    }
    report(std::string(name) + (fp16 ? " fp16" : " fp32"), maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// fp32 binary op (normalized error metric, avoids relative-error inflation on near-zero results such as Fmod)
static void t_binop_f32(const char *name,
                        std::function<aclnnStatus(aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                        aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                        std::function<double(double, double)> ref, float lo, float hi, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, lo, hi), fb = randv(n, lo, hi);
    std::vector<float> hz(n);
    DevBuf da(n*4), db(n*4), dz(n*4); da.up(fa.data()); db.up(fb.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tb = mk({n}, ACL_FLOAT, db.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tb, tz, w, e); }, run);
    dz.down(hz.data());
    double me = 0, mr = 0;
    for (int64_t i = 0; i < n; i++) { double r = ref(fa[i], fb[i]); me = std::max(me, std::fabs(hz[i]-r)); mr = std::max(mr, std::fabs(r)); }
    report(name, me/(mr+1e-9), tol);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

template <typename GetWs>
static void t_reduce(const char *name, GetWs getws,
                     aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                     std::function<double(const std::vector<float>&)> ref, double tol) {
    const int64_t n = 1 << 20;
    auto fa = randv(n, -1, 1);
    DevBuf da(n * 4), dz(4);
    da.up(fa.data());
    int64_t odim[1] = {1};
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p);
    aclTensor *tz = aclCreateTensor(odim, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, odim, 1, dz.p);
    int64_t dims0[1] = {0};
    aclIntArray *dim = aclCreateIntArray(dims0, 1);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, dim, tz, w, e); }, run);
    float got;
    dz.down(&got);
    report(name, relerr(got, ref(fa)), tol);
    aclDestroyIntArray(dim); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Random number family: statistical property verification (non bit-exact)
static void t_random(){
    const int64_t n = 1<<20;
    // Uniform[-2,3]
    { DevBuf d(n*4); aclTensor*t=mk({n},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUniformGetWorkspaceSize(t,-2.0,3.0,123,w,e);},aclnnUniform);
      std::vector<float> h(n); d.down(h.data()); double s=0; bool inr=true; for(auto v:h){s+=v; if(v<-2||v>=3)inr=false;}
      report("Uniform[-2,3] mean", std::fabs(s/n - 0.5)/0.5 + (inr?0:1), 2e-2); aclDestroyTensor(t); }
    // Normal(1, 2)
    { DevBuf d(n*4); aclTensor*t=mk({n},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNormalGetWorkspaceSize(t,1.0,2.0,7,w,e);},aclnnNormal);
      std::vector<float> h(n); d.down(h.data()); double s=0; for(auto v:h)s+=v; double mu=s/n; double var=0; for(auto v:h)var+=(v-mu)*(v-mu); var/=n;
      report("Normal(1,2) mean/std", std::fabs(mu-1.0)+std::fabs(std::sqrt(var)-2.0), 5e-2); aclDestroyTensor(t); }
    // Bernoulli(0.3)
    { DevBuf d(n*4); aclTensor*t=mk({n},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBernoulliGetWorkspaceSize(t,0.3,99,w,e);},aclnnBernoulli);
      std::vector<float> h(n); d.down(h.data()); double s=0; for(auto v:h)s+=v; report("Bernoulli(0.3) frac", std::fabs(s/n-0.3), 2e-2); aclDestroyTensor(t); }
    // Dropout(0.4)
    { auto x=randv(n,1,2); DevBuf dx(n*4),dy(n*4),dm(n); dx.up(x.data()); aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*ty=mk({n},ACL_FLOAT,dy.p),*tm=mk({n},ACL_BOOL,dm.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDropoutGetWorkspaceSize(tx,0.4,55,ty,tm,w,e);},aclnnDropout);
      std::vector<float> hy(n); std::vector<uint8_t> hm(n); dy.down(hy.data()); dm.down(hm.data());
      double kept=0,bad=0; for(int64_t i=0;i<n;i++){kept+=hm[i]; double ref=hm[i]?x[i]/0.6f:0.f; if(std::fabs(hy[i]-ref)>1e-4)bad++;}
      report("Dropout(0.4)", std::fabs(kept/n-0.6)+(bad?1:0), 2e-2); aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(tm); }
    // Randperm(1000)
    { int64_t m=1000; DevBuf d(m*8); aclTensor*t=aclCreateTensor(&m,1,ACL_INT64,nullptr,0,ACL_FORMAT_ND,&m,1,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRandpermGetWorkspaceSize(m,3,t,w,e);},aclnnRandperm);
      std::vector<int64_t> h(m); d.down(h.data()); std::vector<int> seen(m,0); int64_t bad=0; for(auto v:h){if(v<0||v>=m||seen[v])bad++;else seen[v]=1;}
      report("Randperm(1000) is perm", bad?1.0:0.0, 0); aclDestroyTensor(t); }
}

// All/Any: reduce along a dim to bool (uint8). Injects a zero to verify the logic.
static void t_allany() {
    const int64_t R=3, C=4; std::vector<float> fa(R*C);
    for (int64_t i=0;i<R*C;i++) fa[i]=1.0f;            // all non-zero by default
    fa[1*C+2]=0.0f;                                    // row 1 has one zero
    // reduce along dim=1 -> output [R]
    DevBuf da(R*C*4), dz(R);
    da.up(fa.data());
    int64_t ax[1]={1}; aclIntArray *dim=aclCreateIntArray(ax,1);
    auto runcase=[&](bool isAll){
        aclTensor *ta=mk({R,C},ACL_FLOAT,da.p), *tz=mk({R},ACL_BOOL,dz.p);
        if(isAll) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAllGetWorkspaceSize(ta,dim,false,tz,w,e);},aclnnAll);
        else      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAnyGetWorkspaceSize(ta,dim,false,tz,w,e);},aclnnAny);
        std::vector<uint8_t> got(R); dz.down(got.data());
        int64_t bad=0;
        for(int64_t r=0;r<R;r++){ bool all=true,any=false; for(int64_t c=0;c<C;c++){bool nz=fa[r*C+c]!=0.f; all=all&&nz; any=any||nz;}
            uint8_t ref = isAll? (all?1:0):(any?1:0); if(got[r]!=ref) bad++; }
        report(isAll?"All [3,4] ax{1}":"Any [3,4] ax{1}", bad?1.0:0.0, 0);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    };
    runcase(true); runcase(false);
    aclDestroyIntArray(dim);
}

// normalized unary test (max|err|/max|ref|): for ops whose output crosses zero with fp32 cancellation (e.g. Tanhshrink)
static void t_un_norm(const char *name, aclnnStatus (*getws)(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**),
                      aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                      std::function<double(double)> ref, float lo, float hi, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, lo, hi);
    std::vector<float> hz(n);
    DevBuf da(n*4), dz(n*4); da.up(fa.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tz, w, e); }, run);
    dz.down(hz.data());
    double me = 0, mr = 0;
    for (int64_t i = 0; i < n; i++) { double r = ref(fa[i]); me = std::max(me, std::fabs(hz[i]-r)); mr = std::max(mr, std::fabs(r)); }
    report(name, me/(mr+1e-9), tol);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}
// Lerp normalized test (output crosses zero): self + w*(end-self)
static void t_lerp(float wv, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, -2, 2), fb = randv(n, -2, 2);
    std::vector<float> hz(n);
    DevBuf da(n*4), db(n*4), dz(n*4); da.up(fa.data()); db.up(fb.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tb = mk({n}, ACL_FLOAT, db.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    aclScalar *w = aclCreateScalar(&wv, ACL_FLOAT);
    exec2([&](uint64_t *ww, aclOpExecutor **e) { return aclnnLerpGetWorkspaceSize(ta, tb, w, tz, ww, e); }, aclnnLerp);
    dz.down(hz.data());
    double me = 0, mr = 0;
    for (int64_t i = 0; i < n; i++) { double r = fa[i] + wv*(fb[i]-fa[i]); me = std::max(me, std::fabs(hz[i]-r)); mr = std::max(mr, std::fabs(r)); }
    report("Lerp w=0.3", me/(mr+1e-9), tol);
    aclDestroyScalar(w); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}
// erfinv round-trip: verify erf(erfinv(x)) == x (avoids needing a CPU erfinv reference)
static void t_erfinv() {
    const int64_t n = 1 << 16;
    auto fa = randv(n, -0.95f, 0.95f);
    std::vector<float> hz(n);
    DevBuf da(n*4), dz(n*4); da.up(fa.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnErfinvGetWorkspaceSize(ta, tz, w, e); }, aclnnErfinv);
    dz.down(hz.data());
    double me = 0; for (int64_t i = 0; i < n; i++) me = std::max(me, std::fabs(std::erf((double)hz[i]) - fa[i]));
    report("Erfinv (erf round-trip)", me, 1e-5);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}
// two-scalar unary activation (Hardtanh/Threshold)
static void t_scalar2_f(const char *name, float s1, float s2,
                        std::function<aclnnStatus(aclTensor*, aclScalar*, aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                        aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                        std::function<double(double)> ref, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, -4, 4);
    std::vector<float> hz(n);
    DevBuf da(n*4), dz(n*4); da.up(fa.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    aclScalar *a = aclCreateScalar(&s1, ACL_FLOAT), *b = aclCreateScalar(&s2, ACL_FLOAT);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, a, b, tz, w, e); }, run);
    dz.down(hz.data());
    double mr = 0; for (int64_t i = 0; i < n; i++) mr = std::max(mr, relerr(hz[i], ref(fa[i])));
    report(name, mr, tol);
    aclDestroyScalar(a); aclDestroyScalar(b); aclDestroyTensor(ta); aclDestroyTensor(tz);
}
// ternary (self, t1, t2) + scalar value (Addcmul/Addcdiv); for ClampTensor t1=min,t2=max, value ignored
static void t_tern_f(const char *name, float value, float lo, float hi,
                     std::function<aclnnStatus(aclTensor*, aclTensor*, aclTensor*, aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                     aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream),
                     std::function<double(double, double, double, double)> ref, double tol) {
    const int64_t n = 1 << 18;
    auto fa = randv(n, lo, hi), fb = randv(n, lo, hi), fc = randv(n, 0.3f, hi);
    std::vector<float> hz(n);
    DevBuf da(n*4), db(n*4), dc(n*4), dz(n*4); da.up(fa.data()); db.up(fb.data()); dc.up(fc.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tb = mk({n}, ACL_FLOAT, db.p), *tc = mk({n}, ACL_FLOAT, dc.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    aclScalar *v = aclCreateScalar(&value, ACL_FLOAT);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tb, tc, v, tz, w, e); }, run);
    dz.down(hz.data());
    double me = 0, mr = 0;
    for (int64_t i = 0; i < n; i++) { double r = ref(fa[i], fb[i], fc[i], value); me = std::max(me, std::fabs(hz[i]-r)); mr = std::max(mr, std::fabs(r)); }
    report(name, me/(mr+1e-9), tol);
    aclDestroyScalar(v); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc); aclDestroyTensor(tz);
}

// P3 reduce/selection over dim=1 of an [R,C] matrix -> out[R]; reducer maps one row (sorted copy available) to a scalar
static void t_dimred(const char *name, int R, int C, std::function<float(std::vector<float>)> rdc, double tol,
                     std::function<aclnnStatus(aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**)> getws,
                     aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream), float lo = -3, float hi = 3) {
    auto fa = randv((int64_t)R*C, lo, hi);
    std::vector<float> hz(R);
    DevBuf da((int64_t)R*C*4), dz(R*4); da.up(fa.data());
    aclTensor *ta = mk({R, C}, ACL_FLOAT, da.p), *tz = mk({R}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return getws(ta, tz, w, e); }, run);
    dz.down(hz.data());
    double me = 0, mr = 0;
    for (int r = 0; r < R; r++) { std::vector<float> row(fa.begin()+(int64_t)r*C, fa.begin()+(int64_t)r*C+C);
        double ref = rdc(row); me = std::max(me, std::fabs(hz[r]-ref)); mr = std::max(mr, std::fabs(ref)); }
    report(name, me/(mr+1e-9), tol);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

static void t_p3() {
    const int R = 16, C = 32;
    auto dim1 = []{ int64_t d[1]={1}; return aclCreateIntArray(d,1); };
    // value reductions
    t_dimred("Nansum dim1", R, C, [](std::vector<float> v){ double s=0; for(float x:v) s+=x; return (float)s; }, 1e-5,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ static aclIntArray*d=dim1(); return aclnnNansumGetWorkspaceSize(a,d,false,ACL_FLOAT,o,w,e); }, aclnnNansum);
    t_dimred("Nanmean dim1", R, C, [](std::vector<float> v){ double s=0; for(float x:v) s+=x; return (float)(s/v.size()); }, 1e-5,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ static aclIntArray*d=dim1(); return aclnnNanmeanGetWorkspaceSize(a,d,false,ACL_FLOAT,o,w,e); }, aclnnNanmean);
    t_dimred("Median dim1", R, C, [](std::vector<float> v){ std::sort(v.begin(),v.end()); return v[(v.size()-1)/2]; }, 0,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ return aclnnMedianGetWorkspaceSize(a,1,false,o,w,e); }, aclnnMedian);
    t_dimred("Kthvalue k=5 dim1", R, C, [](std::vector<float> v){ std::sort(v.begin(),v.end()); return v[4]; }, 0,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ return aclnnKthvalueGetWorkspaceSize(a,5,1,false,o,w,e); }, aclnnKthvalue);
    t_dimred("Quantile q=0.3 dim1", R, C, [](std::vector<float> v){ std::sort(v.begin(),v.end()); double p=0.3*(v.size()-1); int lo=(int)std::floor(p),hi=(int)std::ceil(p); double f=p-lo; return (float)(v[lo]*(1-f)+v[hi]*f); }, 1e-5,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ return aclnnQuantileGetWorkspaceSize(a,0.3,1,false,o,w,e); }, aclnnQuantile);
    // Mode: integer-valued data so ties are frequent; reference = smallest most-frequent value
    t_dimred("Mode dim1", R, C, [](std::vector<float> v){ std::sort(v.begin(),v.end()); float bv=v[0]; int bc=0; for(size_t i=0;i<v.size();){ size_t j=i; while(j<v.size()&&v[j]==v[i]) j++; if((int)(j-i)>bc){bc=j-i;bv=v[i];} i=j;} return bv; }, 0,
             [&](aclTensor*a,aclTensor*o,uint64_t*w,aclOpExecutor**e){ return aclnnModeGetWorkspaceSize(a,1,false,o,w,e); }, aclnnMode, -3.99, 3.99);
    // CountNonzero (int64 out) and Aminmax (two outputs) tested directly
    {
        auto fa = randv((int64_t)R*C, -2, 2);
        for (int i = 0; i < R; i++) fa[(int64_t)i*C + (i%C)] = 0.f;   // plant a known zero per row
        std::vector<int64_t> hc(R);
        DevBuf da((int64_t)R*C*4), dc(R*8); da.up(fa.data());
        aclTensor *ta = mk({R,C}, ACL_FLOAT, da.p), *tc = mk({R}, ACL_INT64, dc.p);
        aclIntArray *d = dim1();
        exec2([&](uint64_t*w,aclOpExecutor**e){ return aclnnCountNonzeroGetWorkspaceSize(ta,d,false,tc,w,e); }, aclnnCountNonzero);
        dc.down(hc.data());
        int64_t bad = 0; for (int r=0;r<R;r++){ int64_t cnt=0; for(int c=0;c<C;c++) if(fa[(int64_t)r*C+c]!=0) cnt++; if(hc[r]!=cnt) bad++; }
        report("CountNonzero dim1", bad?1.0:0.0, 0);
        aclDestroyTensor(ta); aclDestroyTensor(tc); aclDestroyIntArray(d);
    }
    {
        auto fa = randv((int64_t)R*C, -3, 3);
        std::vector<float> hmin(R), hmax(R);
        DevBuf da((int64_t)R*C*4), dmin(R*4), dmax(R*4); da.up(fa.data());
        aclTensor *ta = mk({R,C}, ACL_FLOAT, da.p), *tmin = mk({R}, ACL_FLOAT, dmin.p), *tmax = mk({R}, ACL_FLOAT, dmax.p);
        aclIntArray *d = dim1();
        exec2([&](uint64_t*w,aclOpExecutor**e){ return aclnnAminmaxGetWorkspaceSize(ta,d,false,tmin,tmax,w,e); }, aclnnAminmax);
        dmin.down(hmin.data()); dmax.down(hmax.data());
        double bad = 0; for(int r=0;r<R;r++){ float mn=fa[(int64_t)r*C],mx=mn; for(int c=0;c<C;c++){mn=std::min(mn,fa[(int64_t)r*C+c]);mx=std::max(mx,fa[(int64_t)r*C+c]);} bad=std::max(bad,(double)std::max(std::fabs(hmin[r]-mn),std::fabs(hmax[r]-mx))); }
        report("Aminmax dim1", bad, 0);
        aclDestroyTensor(ta); aclDestroyTensor(tmin); aclDestroyTensor(tmax); aclDestroyIntArray(d);
    }
    // Renorm: p=2 along dim0 of [R,C]; each row scaled so its L2 norm <= maxnorm
    {
        const float mn = 2.0f;
        auto fa = randv((int64_t)R*C, -2, 2);
        std::vector<float> hz((int64_t)R*C);
        DevBuf da((int64_t)R*C*4), dz((int64_t)R*C*4); da.up(fa.data());
        aclTensor *ta = mk({R,C}, ACL_FLOAT, da.p), *tz = mk({R,C}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){ return aclnnRenormGetWorkspaceSize(ta,2.0,0,mn,tz,w,e); }, aclnnRenorm);
        dz.down(hz.data());
        double me=0,mr=0;
        for(int r=0;r<R;r++){ double nrm=0; for(int c=0;c<C;c++) nrm+=(double)fa[(int64_t)r*C+c]*fa[(int64_t)r*C+c]; nrm=std::sqrt(nrm);
            double sc = nrm>mn ? mn/(nrm+1e-7) : 1.0;
            for(int c=0;c<C;c++){ double ref=fa[(int64_t)r*C+c]*sc; me=std::max(me,std::fabs(hz[(int64_t)r*C+c]-ref)); mr=std::max(mr,std::fabs(ref)); } }
        report("Renorm p2 dim0", me/(mr+1e-9), 1e-5);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
}

int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    CHECK(aclrtCreateStream(&g_stream));
    srand(7);
    t_p3();

    for (int fp16 = 0; fp16 <= 1; fp16++) {
        double tol = fp16 ? 2e-3 : 1e-6;
        t_bin_f("Add a=2.5", fp16, true, 2.5f, aclnnAddGetWorkspaceSize, aclnnAdd,
                [](double x, double y, double a) { return x + a * y; }, -2, 2, tol);
        t_bin_f("Sub a=1.5", fp16, true, 1.5f, aclnnSubGetWorkspaceSize, aclnnSub,
                [](double x, double y, double a) { return x - a * y; }, -2, 2, tol);
        auto noalpha = [](auto f) { return f; };
        t_bin_f("Mul", fp16, false, 1, [](aclTensor *a, aclTensor *b, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                    return aclnnMulGetWorkspaceSize(a, b, o, w, e); }, aclnnMul,
                [](double x, double y, double) { return x * y; }, -2, 2, tol);
        t_bin_f("Div", fp16, false, 1, [](aclTensor *a, aclTensor *b, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                    return aclnnDivGetWorkspaceSize(a, b, o, w, e); }, aclnnDiv,
                [](double x, double y, double) { return x / y; }, 0.5, 2, tol);
        t_bin_f("Maximum", fp16, false, 1, [](aclTensor *a, aclTensor *b, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                    return aclnnMaximumGetWorkspaceSize(a, b, o, w, e); }, aclnnMaximum,
                [](double x, double y, double) { return std::max(x, y); }, -2, 2, tol);
        t_bin_f("Minimum", fp16, false, 1, [](aclTensor *a, aclTensor *b, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                    return aclnnMinimumGetWorkspaceSize(a, b, o, w, e); }, aclnnMinimum,
                [](double x, double y, double) { return std::min(x, y); }, -2, 2, tol);
        (void)noalpha;
    }

    t_scalar_f("Adds s=1.5 a=2", 1.5f, true, 2.f, aclnnAddsGetWorkspaceSize, aclnnAdds,
               [](double x, double s) { return x + s; }, 1e-6);
    t_scalar_f("Subs s=1.5 a=2", 1.5f, true, 2.f, aclnnSubsGetWorkspaceSize, aclnnSubs,
               [](double x, double s) { return x - s; }, 1e-6);
    t_scalar_f("Muls s=1.5", 1.5f, false, 1, [](aclTensor *a, aclScalar *s, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                   return aclnnMulsGetWorkspaceSize(a, s, o, w, e); }, aclnnMuls,
               [](double x, double s) { return x * s; }, 1e-6);
    t_scalar_f("Divs s=1.5", 1.5f, false, 1, [](aclTensor *a, aclScalar *s, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                   return aclnnDivsGetWorkspaceSize(a, s, o, w, e); }, aclnnDivs,
               [](double x, double s) { return x / s; }, 1e-6);
    t_scalar_f("ClampMin s=0.5", 0.5f, false, 1, [](aclTensor *a, aclScalar *s, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                   return aclnnClampMinGetWorkspaceSize(a, s, o, w, e); }, aclnnClampMin,
               [](double x, double s) { return std::max(x, s); }, 0);
    t_scalar_f("ClampMax s=0.5", 0.5f, false, 1, [](aclTensor *a, aclScalar *s, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                   return aclnnClampMaxGetWorkspaceSize(a, s, o, w, e); }, aclnnClampMax,
               [](double x, double s) { return std::min(x, s); }, 0);

    t_un_f("Exp", aclnnExpGetWorkspaceSize, aclnnExp, [](double x) { return std::exp(x); }, -2, 2, 1e-6);
    t_un_f("Log", aclnnLogGetWorkspaceSize, aclnnLog, [](double x) { return std::log(x); }, 0.1, 4, 1e-5);
    t_un_f("Abs", aclnnAbsGetWorkspaceSize, aclnnAbs, [](double x) { return std::fabs(x); }, -2, 2, 0);
    t_un_f("Sqrt", aclnnSqrtGetWorkspaceSize, aclnnSqrt, [](double x) { return std::sqrt(x); }, 0.01, 4, 1e-6);
    t_un_f("Rsqrt", aclnnRsqrtGetWorkspaceSize, aclnnRsqrt, [](double x) { return 1 / std::sqrt(x); }, 0.01, 4, 1e-5);
    t_un_f("Reciprocal", aclnnReciprocalGetWorkspaceSize, aclnnReciprocal, [](double x) { return 1 / x; }, 0.2, 4, 1e-6);
    t_un_f("Relu", aclnnReluGetWorkspaceSize, aclnnRelu, [](double x) { return x > 0 ? x : 0; }, -2, 2, 0);
    t_un_f("Neg", aclnnNegGetWorkspaceSize, aclnnNeg, [](double x) { return -x; }, -2, 2, 0);

    t_un_f("Sigmoid", aclnnSigmoidGetWorkspaceSize, aclnnSigmoid, [](double x) { return 1/(1+std::exp(-x)); }, -4, 4, 1e-6);
    t_un_f("Tanh", aclnnTanhGetWorkspaceSize, aclnnTanh, [](double x) { return std::tanh(x); }, -4, 4, 1e-6);
    t_un_f("Erf", aclnnErfGetWorkspaceSize, aclnnErf, [](double x) { return std::erf(x); }, -3, 3, 1e-6);
    t_un_f("Gelu", aclnnGeluGetWorkspaceSize, aclnnGelu, [](double x) { return 0.5*x*(1+std::erf(x*0.70710678118654752)); }, -4, 4, 1e-5);
    t_un_f("Silu", aclnnSiluGetWorkspaceSize, aclnnSilu, [](double x) { return x/(1+std::exp(-x)); }, -4, 4, 1e-6);
    t_un_f("Softplus", aclnnSoftplusGetWorkspaceSize, aclnnSoftplus, [](double x) { return std::log1p(std::exp(-std::fabs(x)))+std::max(x,0.0); }, -4, 4, 1e-6);
    t_un_f("Sin", aclnnSinGetWorkspaceSize, aclnnSin, [](double x) { return std::sin(x); }, -3, 3, 1e-6);
    t_un_f("Cos", aclnnCosGetWorkspaceSize, aclnnCos, [](double x) { return std::cos(x); }, -3, 3, 1e-6);
    t_un_f("Tan", aclnnTanGetWorkspaceSize, aclnnTan, [](double x) { return std::tan(x); }, -1, 1, 1e-5);
    t_un_f("Atan", aclnnAtanGetWorkspaceSize, aclnnAtan, [](double x) { return std::atan(x); }, -4, 4, 1e-6);
    t_un_f("Sign", aclnnSignGetWorkspaceSize, aclnnSign, [](double x) { return (x>0)-(x<0); }, -2, 2, 0);
    t_un_f("Floor", aclnnFloorGetWorkspaceSize, aclnnFloor, [](double x) { return std::floor(x); }, -4, 4, 0);
    t_un_f("Ceil", aclnnCeilGetWorkspaceSize, aclnnCeil, [](double x) { return std::ceil(x); }, -4, 4, 0);
    t_un_f("Round", aclnnRoundGetWorkspaceSize, aclnnRound, [](double x) { return std::nearbyint(x); }, -4, 4, 0);
    t_un_f("Trunc", aclnnTruncGetWorkspaceSize, aclnnTrunc, [](double x) { return std::trunc(x); }, -4, 4, 0);
    t_un_f("Square", aclnnSquareGetWorkspaceSize, aclnnSquare, [](double x) { return x*x; }, -3, 3, 1e-6);
    t_scalar_f("Pow s=3", 3.0f, false, 1, [](aclTensor *a, aclScalar *s, aclScalar *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                   return aclnnPowTensorScalarGetWorkspaceSize(a, s, o, w, e); }, aclnnPowTensorScalar,
               [](double x, double s) { return std::pow(x, s); }, 1e-5);

    // Elementary functions added to the explorer math library
    t_un_f("Sinh", aclnnSinhGetWorkspaceSize, aclnnSinh, [](double x){return std::sinh(x);}, -2, 2, 1e-5);
    t_un_f("Cosh", aclnnCoshGetWorkspaceSize, aclnnCosh, [](double x){return std::cosh(x);}, -2, 2, 1e-5);
    t_un_f("Asin", aclnnAsinGetWorkspaceSize, aclnnAsin, [](double x){return std::asin(x);}, -0.9, 0.9, 1e-6);
    t_un_f("Acos", aclnnAcosGetWorkspaceSize, aclnnAcos, [](double x){return std::acos(x);}, -0.9, 0.9, 1e-6);
    t_un_f("Erfc", aclnnErfcGetWorkspaceSize, aclnnErfc, [](double x){return std::erfc(x);}, -2, 2, 1e-6);
    t_un_f("Frac", aclnnFracGetWorkspaceSize, aclnnFrac, [](double x){return x-std::trunc(x);}, -4, 4, 1e-6);
    t_un_f("Lgamma", aclnnLgammaGetWorkspaceSize, aclnnLgamma, [](double x){return std::lgamma(x);}, 0.5, 4, 1e-5);
    t_binop_f32("Fmod", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnFmodGetWorkspaceSize(a,b,o,w,e);}, aclnnFmod,
                [](double x,double y){return std::fmod(x,y);}, 0.5, 3, 1e-5);
    t_binop_f32("Hypot", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHypotGetWorkspaceSize(a,b,o,w,e);}, aclnnHypot,
                [](double x,double y){return std::hypot(x,y);}, -2, 2, 1e-6);
    t_binop_f32("PowTensorTensor", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnPowTensorTensorGetWorkspaceSize(a,b,o,w,e);}, aclnnPowTensorTensor,
                [](double x,double y){return std::pow(x,y);}, 0.5, 2, 1e-5);

    t_bit32("BitwiseAnd", [](aclTensor *a, aclTensor *b, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                return aclnnBitwiseAndTensorGetWorkspaceSize(a, b, o, w, e); }, aclnnBitwiseAndTensor,
            [](int32_t x, int32_t y) { return x & y; }, false);
    t_bit32("BitwiseOr", [](aclTensor *a, aclTensor *b, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                return aclnnBitwiseOrTensorGetWorkspaceSize(a, b, o, w, e); }, aclnnBitwiseOrTensor,
            [](int32_t x, int32_t y) { return x | y; }, false);
    t_bit32("BitwiseNot", [](aclTensor *a, aclTensor *, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                return aclnnBitwiseNotGetWorkspaceSize(a, o, w, e); }, aclnnBitwiseNot,
            [](int32_t x, int32_t) { return ~x; }, true);
    t_bit32("BitwiseXor", [](aclTensor *a, aclTensor *b, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                return aclnnBitwiseXorTensorGetWorkspaceSize(a, b, o, w, e); }, aclnnBitwiseXorTensor,
            [](int32_t x, int32_t y) { return x ^ y; }, false);
    // scalar bitwise (mask 0x0F0F0F0F)
    { const int64_t n = 1<<16; std::vector<int32_t> ha(n), hz(n); for(int64_t i=0;i<n;i++) ha[i]=rand()-RAND_MAX/2;
      int32_t S=0x0F0F0F0F; auto sc=aclCreateScalar(&S,ACL_INT32);
      auto run_s=[&](const char*nm, aclnnStatus(*ws)(const aclTensor*,const aclScalar*,aclTensor*,uint64_t*,aclOpExecutor**), aclnnStatus(*rn)(void*,uint64_t,aclOpExecutor*,aclrtStream), std::function<int32_t(int32_t)> ref){
        DevBuf da(n*4),dz(n*4); da.up(ha.data()); auto ta=mk({n},ACL_INT32,da.p), tz=mk({n},ACL_INT32,dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return ws(ta,sc,tz,w,e);}, rn); dz.down(hz.data());
        int64_t bad=0; for(int64_t i=0;i<n;i++) if(hz[i]!=ref(ha[i]))bad++; report(std::string(nm)+" int32", bad?1.0:0.0, 0.0); aclDestroyTensor(ta);aclDestroyTensor(tz); };
      run_s("BitwiseAndScalar", aclnnBitwiseAndScalarGetWorkspaceSize, aclnnBitwiseAndScalar, [&](int32_t x){return x & S;});
      run_s("BitwiseOrScalar",  aclnnBitwiseOrScalarGetWorkspaceSize,  aclnnBitwiseOrScalar,  [&](int32_t x){return x | S;});
      run_s("BitwiseXorScalar", aclnnBitwiseXorScalarGetWorkspaceSize, aclnnBitwiseXorScalar, [&](int32_t x){return x ^ S;}); }

    // ---- elementwise math extensions ----
    t_un_f("Expm1", aclnnExpm1GetWorkspaceSize, aclnnExpm1, [](double x){return std::expm1(x);}, -2, 2, 1e-6);
    t_un_f("Log1p", aclnnLog1pGetWorkspaceSize, aclnnLog1p, [](double x){return std::log1p(x);}, -0.9, 4, 1e-6);
    t_un_f("Log2", aclnnLog2GetWorkspaceSize, aclnnLog2, [](double x){return std::log2(x);}, 0.1, 4, 1e-6);
    t_un_f("Log10", aclnnLog10GetWorkspaceSize, aclnnLog10, [](double x){return std::log10(x);}, 0.1, 4, 1e-6);
    t_un_f("Exp2", aclnnExp2GetWorkspaceSize, aclnnExp2, [](double x){return std::exp2(x);}, -2, 2, 1e-6);
    t_erfinv();
    t_binop_f32("Atan2", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnAtan2GetWorkspaceSize(a,b,o,w,e);}, aclnnAtan2,
                [](double x,double y){return std::atan2(x,y);}, -2, 2, 1e-6);
    t_binop_f32("Remainder", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnRemainderTensorTensorGetWorkspaceSize(a,b,o,w,e);}, aclnnRemainderTensorTensor,
                [](double x,double y){double r=std::fmod(x,y); if(r!=0 && ((r<0)!=(y<0))) r+=y; return r;}, 0.5, 3, 1e-5);
    t_binop_f32("Xlogy", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnXLogYTensorTensorGetWorkspaceSize(a,b,o,w,e);}, aclnnXLogYTensorTensor,
                [](double x,double y){return x==0?0:x*std::log(y);}, 0.1, 3, 1e-5);
    t_binop_f32("Logaddexp", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnLogAddExpGetWorkspaceSize(a,b,o,w,e);}, aclnnLogAddExp,
                [](double x,double y){double m=std::max(x,y); return m+std::log1p(std::exp(std::min(x,y)-m));}, -3, 3, 1e-6);
    t_binop_f32("Copysign", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnCopysignGetWorkspaceSize(a,b,o,w,e);}, aclnnCopysign,
                [](double x,double y){return std::copysign(x,y);}, -2, 2, 0);
    t_binop_f32("Heaviside", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHeavisideGetWorkspaceSize(a,b,o,w,e);}, aclnnHeaviside,
                [](double x,double y){return x<0?0:(x>0?1:y);}, -2, 2, 0);
    t_lerp(0.3f, 1e-6);
    t_tern_f("Addcmul v=0.5", 0.5f, -2, 2, [](aclTensor*a,aclTensor*b,aclTensor*c,aclScalar*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnAddcmulGetWorkspaceSize(a,b,c,v,o,w,e);}, aclnnAddcmul,
             [](double a,double b,double c,double v){return a+v*(b*c);}, 1e-6);
    t_tern_f("Addcdiv v=0.5", 0.5f, -2, 2, [](aclTensor*a,aclTensor*b,aclTensor*c,aclScalar*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnAddcdivGetWorkspaceSize(a,b,c,v,o,w,e);}, aclnnAddcdiv,
             [](double a,double b,double c,double v){return a+v*(b/c);}, 1e-5);
    t_tern_f("ClampTensor", 0.f, -2, 2, [](aclTensor*a,aclTensor*b,aclTensor*c,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnClampTensorGetWorkspaceSize(a,b,c,o,w,e);}, aclnnClampTensor,
             [](double a,double b,double c,double){return a<b?b:(a>c?c:a);}, 0);

    // ---- activation extensions ----
    t_un_f("Mish", aclnnMishGetWorkspaceSize, aclnnMish, [](double x){double sp=x>0?x+std::log1p(std::exp(-x)):std::log1p(std::exp(x)); return x*std::tanh(sp);}, -4, 4, 1e-6);
    t_un_f("Hardswish", aclnnHardswishGetWorkspaceSize, aclnnHardswish, [](double x){double r=std::min(std::max(x+3,0.0),6.0); return x*r/6;}, -4, 4, 1e-6);
    t_un_f("Hardsigmoid", aclnnHardsigmoidGetWorkspaceSize, aclnnHardsigmoid, [](double x){return std::min(std::max(x+3,0.0),6.0)/6;}, -4, 4, 1e-6);
    t_un_f("LogSigmoid", aclnnLogSigmoidGetWorkspaceSize, aclnnLogSigmoid, [](double x){return std::min(x,0.0)-std::log1p(std::exp(-std::fabs(x)));}, -4, 4, 1e-6);
    t_un_f("Selu", aclnnSeluGetWorkspaceSize, aclnnSelu, [](double x){double al=1.6732632423543772,sc=1.0507009873554805; return sc*(x>0?x:al*std::expm1(x));}, -4, 4, 1e-6);
    t_un_norm("Tanhshrink", aclnnTanhshrinkGetWorkspaceSize, aclnnTanhshrink, [](double x){return x-std::tanh(x);}, -3, 3, 1e-6);
    t_un_f("Relu6", aclnnRelu6GetWorkspaceSize, aclnnRelu6, [](double x){return std::min(std::max(x,0.0),6.0);}, -4, 8, 0);
    t_scalar_f("LeakyRelu s=0.1", 0.1f, false, 1, [](aclTensor*a,aclScalar*s,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnLeakyReluGetWorkspaceSize(a,s,o,w,e);}, aclnnLeakyRelu,
               [](double x,double s){return x>0?x:s*x;}, 1e-6);
    t_scalar_f("Elu a=1.0", 1.0f, false, 1, [](aclTensor*a,aclScalar*s,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnEluGetWorkspaceSize(a,s,o,w,e);}, aclnnElu,
               [](double x,double s){return x>0?x:s*std::expm1(x);}, 1e-6);
    t_scalar_f("Celu a=1.0", 1.0f, false, 1, [](aclTensor*a,aclScalar*s,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnCeluGetWorkspaceSize(a,s,o,w,e);}, aclnnCelu,
               [](double x,double s){return x>0?x:s*std::expm1(x/s);}, 1e-6);
    t_scalar_f("Hardshrink l=0.5", 0.5f, false, 1, [](aclTensor*a,aclScalar*s,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardshrinkGetWorkspaceSize(a,s,o,w,e);}, aclnnHardshrink,
               [](double x,double l){return (x>l||x<-l)?x:0;}, 0);
    t_scalar_f("Softshrink l=0.5", 0.5f, false, 1, [](aclTensor*a,aclScalar*s,aclScalar*,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnSoftshrinkGetWorkspaceSize(a,s,o,w,e);}, aclnnSoftshrink,
               [](double x,double l){return x>l?x-l:(x<-l?x+l:0);}, 1e-6);
    t_binop_f32("Prelu", [](aclTensor*a,aclTensor*b,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnPreluGetWorkspaceSize(a,b,o,w,e);}, aclnnPrelu,
                [](double x,double wt){return x>0?x:wt*x;}, -2, 2, 1e-6);
    t_scalar2_f("Hardtanh[-1,1]", -1.f, 1.f, [](aclTensor*a,aclScalar*lo,aclScalar*hi,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnHardtanhGetWorkspaceSize(a,lo,hi,o,w,e);}, aclnnHardtanh,
                [](double x){return std::min(std::max(x,-1.0),1.0);}, 0);
    t_scalar2_f("Threshold th=0 v=-1", 0.f, -1.f, [](aclTensor*a,aclScalar*t,aclScalar*v,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnThresholdGetWorkspaceSize(a,t,v,o,w,e);}, aclnnThreshold,
                [](double x){return x>0?x:-1.0;}, 0);

    t_cast("Cast fp32->fp16", ACL_FLOAT, ACL_FLOAT16, 5e-4);
    t_cast("Cast fp16->fp32", ACL_FLOAT16, ACL_FLOAT, 0);
    t_cast("Cast fp32->int32", ACL_FLOAT, ACL_INT32, 0);
    t_cast("Cast int32->fp32", ACL_INT32, ACL_FLOAT, 0);

    t_where();

    // Broadcast: [M,N]+[N] (bias add), [M,1]+[1,N] (outer-product style), [B,1,N]*[B,M,1], scalar-dim stretch
    auto add_ws = [](aclTensor *a, aclTensor *b, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnAddGetWorkspaceSize(a, b, nullptr, o, w, e); };
    auto mul_ws = [](aclTensor *a, aclTensor *b, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnMulGetWorkspaceSize(a, b, o, w, e); };
    auto add_ref = [](double x, double y) { return x + y; };
    auto mul_ref = [](double x, double y) { return x * y; };
    for (int fp16 = 0; fp16 <= 1; fp16++) {
        double tol = fp16 ? 2e-3 : 1e-6;
        t_bcast("Bcast Add [M,N]+[N]", fp16, {64, 128}, {128}, {64, 128}, add_ws, aclnnAdd, add_ref, tol);
        t_bcast("Bcast Add [M,1]+[1,N]", fp16, {64, 1}, {1, 128}, {64, 128}, add_ws, aclnnAdd, add_ref, tol);
        t_bcast("Bcast Mul [B,1,N]*[B,M,1]", fp16, {8, 1, 32}, {8, 16, 1}, {8, 16, 32}, mul_ws, aclnnMul, mul_ref, tol);
        t_bcast("Bcast Mul [M,N]*scalar1", fp16, {32, 48}, {1, 1}, {32, 48}, mul_ws, aclnnMul, mul_ref, tol);
        t_bcast("Bcast Add [1,N]+[M,N]", fp16, {1, 128}, {64, 128}, {64, 128}, add_ws, aclnnAdd, add_ref, tol);
    }

    // Non-contiguous views + mixed dtype
    t_where_bcast();
    t_band_bcast();
    t_strided_add();
    t_strided_unary();
    t_mixed_add();

    t_reduce("ReduceSum fp32 n=1M", [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                 return aclnnReduceSumGetWorkspaceSize(a, d, false, ACL_FLOAT, o, w, e); }, aclnnReduceSum,
             [](const std::vector<float> &v) { double s = 0; for (auto x : v) s += x; return s; }, 1e-5);
    t_reduce("Amax fp32 n=1M", [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                 return aclnnAmaxGetWorkspaceSize(a, d, false, o, w, e); }, aclnnAmax,
             [](const std::vector<float> &v) { double m = v[0]; for (auto x : v) m = std::max(m, (double)x); return m; }, 0);
    t_reduce("Amin fp32 n=1M", [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
                 return aclnnAminGetWorkspaceSize(a, d, false, o, w, e); }, aclnnAmin,
             [](const std::vector<float> &v) { double m = v[0]; for (auto x : v) m = std::min(m, (double)x); return m; }, 0);

    // Arbitrary-dim subset reduction + keepDim + Mean/Prod/Max/Min
    auto sum_ws  = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnReduceSumGetWorkspaceSize(a, d, false, ACL_FLOAT, o, w, e); };
    auto sum_wsK = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnReduceSumGetWorkspaceSize(a, d, true, ACL_FLOAT, o, w, e); };
    auto mean_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnMeanGetWorkspaceSize(a, d, false, ACL_FLOAT, o, w, e); };
    auto prod_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnProdGetWorkspaceSize(a, d, false, ACL_FLOAT, o, w, e); };
    auto amax_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnAmaxGetWorkspaceSize(a, d, false, o, w, e); };
    auto amin_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnAminGetWorkspaceSize(a, d, false, o, w, e); };
    auto rsum = [](std::vector<double> &v) { double s = 0; for (auto x : v) s += x; return s; };
    auto rmean = [](std::vector<double> &v) { double s = 0; for (auto x : v) s += x; return s / v.size(); };
    auto rprod = [](std::vector<double> &v) { double p = 1; for (auto x : v) p *= x; return p; };
    auto rmax = [](std::vector<double> &v) { double m = v[0]; for (auto x : v) m = std::max(m, x); return m; };
    auto rmin = [](std::vector<double> &v) { double m = v[0]; for (auto x : v) m = std::min(m, x); return m; };

    t_redx("ReduceSum [2,3,4] ax{1}",     {2,3,4}, {1},   false, sum_ws,  aclnnReduceSum, rsum, 1e-6);
    t_redx("ReduceSum [2,3,4] ax{0,2}",   {2,3,4}, {0,2}, false, sum_ws,  aclnnReduceSum, rsum, 1e-6);
    t_redx("ReduceSum [2,3,4] ax{2} keep",{2,3,4}, {2},   true,  sum_wsK, aclnnReduceSum, rsum, 1e-6);
    t_redx("Mean [4,5,6] ax{1}",          {4,5,6}, {1},   false, mean_ws, aclnnMean,      rmean, 1e-6);
    t_redx("Mean [4,5,6] ax{0,1,2}",      {4,5,6}, {0,1,2},false,mean_ws, aclnnMean,      rmean, 1e-6);
    t_redx("Prod [3,4] ax{1}",            {3,4},   {1},   false, prod_ws, aclnnProd,      rprod, 1e-5);
    t_redx("Amax [2,3,4] ax{1}",          {2,3,4}, {1},   false, amax_ws, aclnnAmax,      rmax, 0);
    t_redx("Amin [2,3,4] ax{0,2}",        {2,3,4}, {0,2}, false, amin_ws, aclnnAmin,      rmin, 0);

    auto argmax_ws = [](aclTensor *a, int64_t d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnArgMaxGetWorkspaceSize(a, d, false, o, w, e); };
    auto argmin_ws = [](aclTensor *a, int64_t d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnArgMinGetWorkspaceSize(a, d, false, o, w, e); };
    t_arg("ArgMax [6,8] dim1", {6,8}, 1, true,  argmax_ws, aclnnArgMax);
    t_arg("ArgMin [6,8] dim0", {6,8}, 0, false, argmin_ws, aclnnArgMin);
    t_arg("ArgMax [4,5,6] dim2", {4,5,6}, 2, true, argmax_ws, aclnnArgMax);

    // Statistical reductions: Var/Std unbiased /(n-1), LogSumExp, Norm p=2
    auto var_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnVarGetWorkspaceSize(a, d, false, o, w, e); };
    auto std_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnStdGetWorkspaceSize(a, d, false, o, w, e); };
    auto lse_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnLogSumExpGetWorkspaceSize(a, d, false, o, w, e); };
    auto norm2_ws = [](aclTensor *a, aclIntArray *d, aclTensor *o, uint64_t *w, aclOpExecutor **e) {
        return aclnnNormGetWorkspaceSize(a, 2.0, d, false, o, w, e); };
    auto rvar = [](std::vector<double> &v) { double m=0; for(auto x:v)m+=x; m/=v.size(); double s=0; for(auto x:v)s+=(x-m)*(x-m); return s/(v.size()-1); };
    auto rstd = [](std::vector<double> &v) { double m=0; for(auto x:v)m+=x; m/=v.size(); double s=0; for(auto x:v)s+=(x-m)*(x-m); return std::sqrt(s/(v.size()-1)); };
    auto rlse = [](std::vector<double> &v) { double mx=v[0]; for(auto x:v)mx=std::max(mx,x); double s=0; for(auto x:v)s+=std::exp(x-mx); return mx+std::log(s); };
    auto rnorm2 = [](std::vector<double> &v) { double s=0; for(auto x:v)s+=x*x; return std::sqrt(s); };
    t_redx("Var [4,5,6] ax{1}",      {4,5,6}, {1},   false, var_ws,   aclnnVar,       rvar, 1e-5);
    t_redx("Std [4,5,6] ax{0,2}",    {4,5,6}, {0,2}, false, std_ws,   aclnnStd,       rstd, 1e-5);
    t_redx("LogSumExp [2,3,4] ax{2}",{2,3,4}, {2},   false, lse_ws,   aclnnLogSumExp, rlse, 1e-5);
    t_redx("Norm2 [2,3,4] ax{1}",    {2,3,4}, {1},   false, norm2_ws, aclnnNorm,      rnorm2, 1e-5);
    t_allany();

    t_random();
    CHECK(aclrtDestroyStream(g_stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
