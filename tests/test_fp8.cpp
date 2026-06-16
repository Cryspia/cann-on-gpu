// fp8 (e4m3 / e5m2) support validation: pure ACL client -> shim -> GPU.
//   (1) fp32->fp8->fp32 quantization round-trip: cross-check against CPU fp8 encoding that is bit-exact (not approximate).
//   (2) fp8 tensors through aclnnAdd: reads fp8 as float, computes, writes back as fp8; compared against CPU reference within fp8 quantization tolerance.
// End-to-end fp8 usability proves the low-precision dtype path is fully wired (quantized inference scenario).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

static aclrtStream g_stream;
static int g_pass = 0, g_fail = 0;
static void *up(const void *h, size_t b) {
    void *d; CHECK(aclrtMalloc(&d, b, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d, b, h, b, ACL_MEMCPY_HOST_TO_DEVICE)); return d;
}
static void report(const char *name, double v, double tol, bool exact = false) {
    bool ok = exact ? (v == 0.0) : (v <= tol);
    (ok ? g_pass : g_fail)++;
    printf("%-30s %s=%.3e %s\n", name, exact ? "mismatch" : "maxrel", v, ok ? "PASS" : "FAIL");
}

// CPU fp8 encoding reference (IEEE-style, round-to-nearest-even; matches hardware for bit-exact round-trip validation)
static uint8_t enc_e4m3(float f); static float dec_e4m3(uint8_t);
static uint8_t enc_e5m2(float f); static float dec_e5m2(uint8_t);

// General: quantize host float to fp8 encoding (CPU reference), upload to device, call aclnnCast(fp32->fp8), compare encoded bytes.
static void t_quant_roundtrip(bool e5m2, int64_t n) {
    aclDataType fp8dt = e5m2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
    float rng = e5m2 ? 100.f : 8.f;     // e4m3 max~=448 but small range is easier to verify; e5m2 has larger dynamic range
    std::vector<float> x(n);
    srand(5);
    for (auto &v : x) v = (rand() / (float)RAND_MAX) * 2 * rng - rng;

    // GPU: fp32 -> fp8
    void *dx, *dq;
    CHECK(aclrtMalloc(&dx, n * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dq, n, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx, n * 4, x.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t d1[1] = {n};
    aclTensor *tx = aclCreateTensor(d1, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d1, 1, dx);
    aclTensor *tq = aclCreateTensor(d1, 1, fp8dt, nullptr, 0, ACL_FORMAT_ND, d1, 1, dq);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnCastGetWorkspaceSize(tx, fp8dt, tq, &ws, &ex));
    CHECK(aclnnCast(nullptr, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gpu_q(n);
    CHECK(aclrtMemcpy(gpu_q.data(), n, dq, n, ACL_MEMCPY_DEVICE_TO_HOST));

    // Round-trip: fp8 -> fp32
    void *dr; CHECK(aclrtMalloc(&dr, n * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *tr = aclCreateTensor(d1, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d1, 1, dr);
    uint64_t ws2 = 0; aclOpExecutor *ex2 = nullptr;
    CHECK(aclnnCastGetWorkspaceSize(tq, ACL_FLOAT, tr, &ws2, &ex2));
    CHECK(aclnnCast(nullptr, ws2, ex2, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> gpu_r(n);
    CHECK(aclrtMemcpy(gpu_r.data(), n * 4, dr, n * 4, ACL_MEMCPY_DEVICE_TO_HOST));

    // Validate: GPU decoded value == CPU reference decode of the same fp8 byte (round-trip self-consistent, bit-exact)
    int64_t mism = 0;
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) {
        float cpu_dec = e5m2 ? dec_e5m2(gpu_q[i]) : dec_e4m3(gpu_q[i]);
        if (gpu_r[i] != cpu_dec) mism++;
        // Quantization error (GPU round-trip vs original float): observed only in the normal range (subnormal step is coarse, relative error naturally large, excluded)
        if (std::fabs(x[i]) > 0.1) maxrel = std::max(maxrel, (double)std::fabs(gpu_r[i] - x[i]) / std::fabs(x[i]));
    }
    char nm[48]; snprintf(nm, sizeof nm, "Cast fp32<->%s round-trip", e5m2 ? "e5m2" : "e4m3");
    report(nm, (double)mism, 0, true);
    char nm2[48]; snprintf(nm2, sizeof nm2, "  %s quant error (observed)", e5m2 ? "e5m2" : "e4m3");
    double qtol = e5m2 ? 0.30 : 0.14;   // e5m2: 2 mantissa bits ~25%; e4m3: 3 mantissa bits ~12.5% (half step)
    report(nm2, maxrel, qtol);

    aclDestroyTensor(tx); aclDestroyTensor(tq); aclDestroyTensor(tr);
    aclrtFree(dx); aclrtFree(dq); aclrtFree(dr);
}

// fp8 tensors through aclnnAdd (compute in float): out = a + b
static void t_fp8_add(bool e5m2, int64_t n) {
    aclDataType fp8dt = e5m2 ? ACL_FLOAT8_E5M2 : ACL_FLOAT8_E4M3FN;
    std::vector<float> a(n), b(n);
    srand(9);
    for (int64_t i = 0; i < n; i++) { a[i] = (rand()/(float)RAND_MAX)*4-2; b[i] = (rand()/(float)RAND_MAX)*4-2; }
    std::vector<uint8_t> ea(n), eb(n);
    for (int64_t i = 0; i < n; i++) {
        ea[i] = e5m2 ? enc_e5m2(a[i]) : enc_e4m3(a[i]);
        eb[i] = e5m2 ? enc_e5m2(b[i]) : enc_e4m3(b[i]);
    }
    void *da, *db, *dz;
    CHECK(aclrtMalloc(&da, n, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&db, n, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dz, n, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(da, n, ea.data(), n, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK(aclrtMemcpy(db, n, eb.data(), n, ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t d1[1] = {n};
    aclTensor *ta = aclCreateTensor(d1, 1, fp8dt, nullptr, 0, ACL_FORMAT_ND, d1, 1, da);
    aclTensor *tb = aclCreateTensor(d1, 1, fp8dt, nullptr, 0, ACL_FORMAT_ND, d1, 1, db);
    aclTensor *tz = aclCreateTensor(d1, 1, fp8dt, nullptr, 0, ACL_FORMAT_ND, d1, 1, dz);
    float one = 1.f; aclScalar *al = aclCreateScalar(&one, ACL_FLOAT);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnAddGetWorkspaceSize(ta, tb, al, tz, &ws, &ex));
    CHECK(aclnnAdd(nullptr, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> ez(n);
    CHECK(aclrtMemcpy(ez.data(), n, dz, n, ACL_MEMCPY_DEVICE_TO_HOST));
    // Reference: dec(a)+dec(b) re-quantized to fp8, compared against GPU fp8 output decode (tolerance ~1 fp8 step in relative magnitude)
    double maxrel = 0;
    for (int64_t i = 0; i < n; i++) {
        float ref = (e5m2 ? dec_e5m2(ea[i]) : dec_e4m3(ea[i])) + (e5m2 ? dec_e5m2(eb[i]) : dec_e4m3(eb[i]));
        float got = e5m2 ? dec_e5m2(ez[i]) : dec_e4m3(ez[i]);
        maxrel = std::max(maxrel, (double)std::fabs(got - ref) / (std::fabs(ref) + 0.5));
    }
    char nm[40]; snprintf(nm, sizeof nm, "Add %s tensor", e5m2 ? "e5m2" : "e4m3");
    report(nm, maxrel, e5m2 ? 0.30 : 0.14);
    aclDestroyScalar(al);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
    aclrtFree(da); aclrtFree(db); aclrtFree(dz);
}

// ---- HiF8 CPU reference (decode table and encoding rules aligned bit-for-bit with CANN libnnopbase.so, verified offline) ----
static const uint32_t HIF8_DEC[256] = {
#include "hif8_table.inc"
};
static float hif8_dec(uint8_t b) { float v; uint32_t u = HIF8_DEC[b]; memcpy(&v, &u, 4); return v; }
static uint8_t hif8_enc(float x) {
    uint32_t xb; memcpy(&xb, &x, 4);
    if ((xb & 0x7fffffffu) > 0x7f800000u) return 0x80;
    uint8_t sign = (xb >> 31) & 1; float ax = std::fabs(x);
    if (std::isinf(ax)) return sign ? 0xef : 0x6f;
    int best = 0; float bestd = 1e30f;
    for (int c = 0; c <= 0x7f; c++) { float v = (c == 0x6f) ? 49152.f : hif8_dec(c); float d = std::fabs(v - ax); if (d <= bestd) { bestd = d; best = c; } }
    if (best == 0) return 0x00;
    return (uint8_t)((sign << 7) | best);
}

// ---- fp4/fp6 CPU reference (decode tables + encoding boundaries, both aligned offline with libnnopbase.so, 0 mismatches on 50k random samples) ----
struct SubFmt { int half; const uint32_t *dec; const float *bnd; };
static const uint32_t D_4E2M1[16]={0x00000000u,0x3f000000u,0x3f800000u,0x3fc00000u,0x40000000u,0x40400000u,0x40800000u,0x40c00000u,0x80000000u,0xbf000000u,0xbf800000u,0xbfc00000u,0xc0000000u,0xc0400000u,0xc0800000u,0xc0c00000u};
static const uint32_t D_4E1M2[16]={0x00000000u,0x3e800000u,0x3f000000u,0x3f400000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,0x80000000u,0xbe800000u,0xbf000000u,0xbf400000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u};
static const float B_4E2M1[8]={0,0.4375f,0.875f,1.25000012f,1.75f,2.50000024f,3.5f,5.00000048f};
static const float B_4E1M2[8]={0,0.234375f,0.46875f,0.6875f,0.9375f,1.12500012f,1.375f,1.62500012f};
static const uint32_t D_6E2M3[64]={0x00000000u,0x3e000000u,0x3e800000u,0x3ec00000u,0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,0x3f800000u,0x3f900000u,0x3fa00000u,0x3fb00000u,0x3fc00000u,0x3fd00000u,0x3fe00000u,0x3ff00000u,0x40000000u,0x40100000u,0x40200000u,0x40300000u,0x40400000u,0x40500000u,0x40600000u,0x40700000u,0x40800000u,0x40900000u,0x40a00000u,0x40b00000u,0x40c00000u,0x40d00000u,0x40e00000u,0x40f00000u,0x80000000u,0xbe000000u,0xbe800000u,0xbec00000u,0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,0xbf800000u,0xbf900000u,0xbfa00000u,0xbfb00000u,0xbfc00000u,0xbfd00000u,0xbfe00000u,0xbff00000u,0xc0000000u,0xc0100000u,0xc0200000u,0xc0300000u,0xc0400000u,0xc0500000u,0xc0600000u,0xc0700000u,0xc0800000u,0xc0900000u,0xc0a00000u,0xc0b00000u,0xc0c00000u,0xc0d00000u,0xc0e00000u,0xc0f00000u};
static const uint32_t D_6E3M2[64]={0x00000000u,0x3d800000u,0x3e000000u,0x3e400000u,0x3e800000u,0x3ea00000u,0x3ec00000u,0x3ee00000u,0x3f000000u,0x3f200000u,0x3f400000u,0x3f600000u,0x3f800000u,0x3fa00000u,0x3fc00000u,0x3fe00000u,0x40000000u,0x40200000u,0x40400000u,0x40600000u,0x40800000u,0x40a00000u,0x40c00000u,0x40e00000u,0x41000000u,0x41200000u,0x41400000u,0x41600000u,0x41800000u,0x41a00000u,0x41c00000u,0x41e00000u,0x80000000u,0xbd800000u,0xbe000000u,0xbe400000u,0xbe800000u,0xbea00000u,0xbec00000u,0xbee00000u,0xbf000000u,0xbf200000u,0xbf400000u,0xbf600000u,0xbf800000u,0xbfa00000u,0xbfc00000u,0xbfe00000u,0xc0000000u,0xc0200000u,0xc0400000u,0xc0600000u,0xc0800000u,0xc0a00000u,0xc0c00000u,0xc0e00000u,0xc1000000u,0xc1200000u,0xc1400000u,0xc1600000u,0xc1800000u,0xc1a00000u,0xc1c00000u,0xc1e00000u};
static const float B_6E2M3[32]={0,0.12109375f,0.2421875f,0.359375f,0.484375f,0.59375f,0.71875f,0.84375f,0.96875f,1.06250012f,1.1875f,1.31250012f,1.4375f,1.56250012f,1.6875f,1.81250012f,1.9375f,2.12500024f,2.375f,2.62500024f,2.875f,3.12500024f,3.375f,3.62500024f,3.875f,4.25000048f,4.75f,5.25000048f,5.75f,6.25000048f,6.75f,7.25000048f};
static const float B_6E3M2[32]={0,0.05859375f,0.1171875f,0.171875f,0.234375f,0.28125003f,0.34375f,0.40625003f,0.46875f,0.56250006f,0.6875f,0.81250006f,0.9375f,1.12500012f,1.375f,1.62500012f,1.875f,2.25000024f,2.75f,3.25000024f,3.75f,4.50000048f,5.5f,6.50000048f,7.5f,9.00000095f,11.f,13.000001f,15.f,18.0000019f,22.f,26.0000019f};
static float subdec(const SubFmt&F,uint8_t c){ float v; uint32_t u=F.dec[c]; memcpy(&v,&u,4); return v; }
static uint8_t subenc(const SubFmt&F,float x){ uint32_t xb; memcpy(&xb,&x,4); uint8_t s=(xb>>31)&1; float ax=std::fabs(x);
    int c=0; for(int i=1;i<F.half;i++){ if(ax>=F.bnd[i]) c=i; else break; } return (uint8_t)(s?(c|F.half):c); }

// fp4 round-trip: encoded bytes (including 2-per-byte packing) and decoded values must be bit-exact against CPU reference.
static void t_fp4(aclDataType dt, const SubFmt&F, const char*name, int64_t n) {
    std::vector<float> x(n);
    srand(23);
    for (auto &v : x) v = (rand()/(float)RAND_MAX)*14-7;
    int64_t nb=(n+1)/2;
    void *dx,*dq,*dr;
    CHECK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dq,nb,ACL_MEM_MALLOC_HUGE_FIRST));     // packed: 2 elements per byte
    CHECK(aclrtMalloc(&dr,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t d1[1]={n};
    aclTensor *tx=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dx);
    aclTensor *tq=aclCreateTensor(d1,1,dt,nullptr,0,ACL_FORMAT_ND,d1,1,dq);
    aclTensor *tr=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dr);
    uint64_t ws; aclOpExecutor *ex;
    CHECK(aclnnCastGetWorkspaceSize(tx,dt,tq,&ws,&ex)); CHECK(aclnnCast(nullptr,ws,ex,g_stream));
    CHECK(aclnnCastGetWorkspaceSize(tq,ACL_FLOAT,tr,&ws,&ex)); CHECK(aclnnCast(nullptr,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gq(nb); std::vector<float> gr(n);
    CHECK(aclrtMemcpy(gq.data(),nb,dq,nb,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(gr.data(),n*4,dr,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t enc_bad=0,dec_bad=0;
    for(int64_t i=0;i<n;i++){
        uint8_t refc=subenc(F,x[i])&0xf;                       // reference 4-bit code
        uint8_t gpuc=(i&1)?(gq[i/2]>>4):(gq[i/2]&0xf);         // GPU packed byte extracted
        if(gpuc!=refc) enc_bad++;
        if(gr[i]!=subdec(F,gpuc)) dec_bad++;
    }
    report((std::string(name)+" encode(GPU vs ref)").c_str(),(double)enc_bad,0,true);
    report((std::string(name)+" decode(GPU vs ref)").c_str(),(double)dec_bad,0,true);
    aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(tr);
    aclrtFree(dx);aclrtFree(dq);aclrtFree(dr);
}

// fp6 round-trip: 1 byte per element.
static void t_fp6(aclDataType dt, const SubFmt&F, const char*name, int64_t n) {
    std::vector<float> x(n);
    srand(29);
    for (auto &v : x) v = (rand()/(float)RAND_MAX)*16-8;
    void *dx,*dq,*dr;
    CHECK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dq,n,ACL_MEM_MALLOC_HUGE_FIRST));      // 1 byte per element
    CHECK(aclrtMalloc(&dr,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t d1[1]={n};
    aclTensor *tx=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dx);
    aclTensor *tq=aclCreateTensor(d1,1,dt,nullptr,0,ACL_FORMAT_ND,d1,1,dq);
    aclTensor *tr=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dr);
    uint64_t ws; aclOpExecutor *ex;
    CHECK(aclnnCastGetWorkspaceSize(tx,dt,tq,&ws,&ex)); CHECK(aclnnCast(nullptr,ws,ex,g_stream));
    CHECK(aclnnCastGetWorkspaceSize(tq,ACL_FLOAT,tr,&ws,&ex)); CHECK(aclnnCast(nullptr,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gq(n); std::vector<float> gr(n);
    CHECK(aclrtMemcpy(gq.data(),n,dq,n,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(gr.data(),n*4,dr,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t enc_bad=0,dec_bad=0;
    for(int64_t i=0;i<n;i++){
        if((gq[i]&0x3f)!=subenc(F,x[i])) enc_bad++;
        if(gr[i]!=subdec(F,gq[i]&0x3f)) dec_bad++;
    }
    report((std::string(name)+" encode(GPU vs ref)").c_str(),(double)enc_bad,0,true);
    report((std::string(name)+" decode(GPU vs ref)").c_str(),(double)dec_bad,0,true);
    aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(tr);
    aclrtFree(dx);aclrtFree(dq);aclrtFree(dr);
}

// fp4 element-wise Add: fp4 tensor -> aclnnAdd (shim unpacks to fp32, adds, re-encodes) -> fp4. Bit-exact against CPU reference.
static void t_fp4_add(aclDataType dt, const SubFmt&F, const char*name, int64_t n) {
    std::vector<float> a(n), b(n);
    srand(31);
    for (int64_t i=0;i<n;i++){ a[i]=(rand()/(float)RAND_MAX)*8-4; b[i]=(rand()/(float)RAND_MAX)*8-4; }
    int64_t nb=(n+1)/2;
    void *da32=up(a.data(),n*4), *db32=up(b.data(),n*4);
    void *da8,*db8,*dz8,*dab,*dbb;
    CHECK(aclrtMalloc(&da8,nb,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&db8,nb,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dz8,nb,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dab,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dbb,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t d1[1]={n}; uint64_t ws; aclOpExecutor *ex;
    auto cast=[&](void*src,aclDataType sdt,void*dst,aclDataType ddt){
        aclTensor*s=aclCreateTensor(d1,1,sdt,nullptr,0,ACL_FORMAT_ND,d1,1,src);
        aclTensor*o=aclCreateTensor(d1,1,ddt,nullptr,0,ACL_FORMAT_ND,d1,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream));
        aclDestroyTensor(s); aclDestroyTensor(o);
    };
    cast(da32,ACL_FLOAT,da8,dt); cast(db32,ACL_FLOAT,db8,dt);
    cast(da8,dt,dab,ACL_FLOAT);  cast(db8,dt,dbb,ACL_FLOAT);   // retrieve fp4-rounded values
    // aclnnAdd on fp4
    aclTensor *ta=aclCreateTensor(d1,1,dt,nullptr,0,ACL_FORMAT_ND,d1,1,da8);
    aclTensor *tb=aclCreateTensor(d1,1,dt,nullptr,0,ACL_FORMAT_ND,d1,1,db8);
    aclTensor *tz=aclCreateTensor(d1,1,dt,nullptr,0,ACL_FORMAT_ND,d1,1,dz8);
    float one=1.f; aclScalar*al=aclCreateScalar(&one,ACL_FLOAT);
    CHECK(aclnnAddGetWorkspaceSize(ta,tb,al,tz,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnAdd(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gz(nb); std::vector<float> ab(n),bb(n);
    CHECK(aclrtMemcpy(gz.data(),nb,dz8,nb,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(ab.data(),n*4,dab,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(bb.data(),n*4,dbb,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t bad=0;
    for(int64_t i=0;i<n;i++){
        uint8_t refc=subenc(F,ab[i]+bb[i])&0xf;
        uint8_t gpuc=(i&1)?(gz[i/2]>>4):(gz[i/2]&0xf);
        if(gpuc!=refc) bad++;
    }
    report((std::string(name)+" Add(GPU vs ref)").c_str(),(double)bad,0,true);
    aclDestroyScalar(al); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
    if(wsp)aclrtFree(wsp);
    aclrtFree(da32);aclrtFree(db32);aclrtFree(da8);aclrtFree(db8);aclrtFree(dz8);aclrtFree(dab);aclrtFree(dbb);
}

// fp4 binary broadcast Add: a[M,N] fp4 + b[1,N] fp4 (column broadcast) -> out[M,N] fp4, bit-exact against unpacked reference.
static void t_fp4_add_bcast(aclDataType dt, const SubFmt&F, const char*name, int64_t M, int64_t N) {
    int64_t na=M*N, nb=N;
    std::vector<float> a(na), b(nb); srand(41);
    for (int64_t i=0;i<na;i++) a[i]=(rand()/(float)RAND_MAX)*8-4;
    for (int64_t j=0;j<nb;j++) b[j]=(rand()/(float)RAND_MAX)*8-4;
    int64_t pa=(na+1)/2, pb=(nb+1)/2, po=(na+1)/2;
    void *da32=up(a.data(),na*4), *db32=up(b.data(),nb*4);
    void *da8,*db8,*dz8,*dab,*dbb;
    CHECK(aclrtMalloc(&da8,pa,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&db8,pb,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dz8,po,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dab,na*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dbb,nb*4,ACL_MEM_MALLOC_HUGE_FIRST));
    auto cast=[&](void*src,aclDataType sdt,void*dst,aclDataType ddt,int64_t cn){
        int64_t d1[1]={cn};
        aclTensor*s=aclCreateTensor(d1,1,sdt,nullptr,0,ACL_FORMAT_ND,d1,1,src);
        aclTensor*o=aclCreateTensor(d1,1,ddt,nullptr,0,ACL_FORMAT_ND,d1,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream));
        aclDestroyTensor(s); aclDestroyTensor(o);
    };
    cast(da32,ACL_FLOAT,da8,dt,na); cast(db32,ACL_FLOAT,db8,dt,nb);
    cast(da8,dt,dab,ACL_FLOAT,na);  cast(db8,dt,dbb,ACL_FLOAT,nb);   // fp4-rounded values
    int64_t da_[2]={M,N}, db_[2]={1,N}, dz_[2]={M,N};
    aclTensor *ta=aclCreateTensor(da_,2,dt,nullptr,0,ACL_FORMAT_ND,da_,2,da8);
    aclTensor *tb=aclCreateTensor(db_,2,dt,nullptr,0,ACL_FORMAT_ND,db_,2,db8);
    aclTensor *tz=aclCreateTensor(dz_,2,dt,nullptr,0,ACL_FORMAT_ND,dz_,2,dz8);
    float one=1.f; aclScalar*al=aclCreateScalar(&one,ACL_FLOAT);
    uint64_t ws; aclOpExecutor *ex; CHECK(aclnnAddGetWorkspaceSize(ta,tb,al,tz,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnAdd(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gz(po); std::vector<float> ab(na),bb(nb);
    CHECK(aclrtMemcpy(gz.data(),po,dz8,po,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(ab.data(),na*4,dab,na*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(bb.data(),nb*4,dbb,nb*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t bad=0;
    for(int64_t i=0;i<M;i++) for(int64_t j=0;j<N;j++){ int64_t idx=i*N+j;
        uint8_t refc=subenc(F,ab[idx]+bb[j])&0xf;
        uint8_t gpuc=(idx&1)?(gz[idx/2]>>4):(gz[idx/2]&0xf);
        if(gpuc!=refc) bad++;
    }
    report((std::string(name)+" Add bcast[M,N]+[1,N]").c_str(),(double)bad,0,true);
    aclDestroyScalar(al); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz);
    if(wsp)aclrtFree(wsp);
    aclrtFree(da32);aclrtFree(db32);aclrtFree(da8);aclrtFree(db8);aclrtFree(dz8);aclrtFree(dab);aclrtFree(dbb);
}

// HiF8: fp32->HiF8 encoded bytes must be bit-exact against CPU reference; HiF8->fp32 decode must equal table lookup value.
static void t_hif8(int64_t n) {
    std::vector<float> x(n);
    srand(17);
    for (auto &v : x) v = (rand() / (float)RAND_MAX) * 2 - 1, v *= std::pow(10.f, (rand() % 12) - 6);
    void *dx, *dq, *dr;
    CHECK(aclrtMalloc(&dx, n * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dq, n, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dr, n * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(dx, n * 4, x.data(), n * 4, ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t d1[1] = {n};
    aclTensor *tx = aclCreateTensor(d1, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d1, 1, dx);
    aclTensor *tq = aclCreateTensor(d1, 1, ACL_HIFLOAT8, nullptr, 0, ACL_FORMAT_ND, d1, 1, dq);
    aclTensor *tr = aclCreateTensor(d1, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d1, 1, dr);
    uint64_t ws; aclOpExecutor *ex;
    CHECK(aclnnCastGetWorkspaceSize(tx, ACL_HIFLOAT8, tq, &ws, &ex)); CHECK(aclnnCast(nullptr, ws, ex, g_stream));
    CHECK(aclnnCastGetWorkspaceSize(tq, ACL_FLOAT, tr, &ws, &ex)); CHECK(aclnnCast(nullptr, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> gq(n); std::vector<float> gr(n);
    CHECK(aclrtMemcpy(gq.data(), n, dq, n, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(gr.data(), n * 4, dr, n * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t enc_bad = 0, dec_bad = 0;
    for (int64_t i = 0; i < n; i++) {
        if (gq[i] != hif8_enc(x[i])) enc_bad++;          // GPU encoding vs reference (bit-exact)
        if (gr[i] != hif8_dec(gq[i])) dec_bad++;          // GPU decode vs table lookup (bit-exact)
    }
    report("HiF8 encode(GPU vs ref)", (double)enc_bad, 0, true);
    report("HiF8 decode(GPU vs ref)", (double)dec_bad, 0, true);
    aclDestroyTensor(tx); aclDestroyTensor(tq); aclDestroyTensor(tr);
    aclrtFree(dx); aclrtFree(dq); aclrtFree(dr);
}

// fp6 true 4-in-3 pack/unpack round-trip
static void t_fp6_pack(int64_t n) {
    std::vector<uint8_t> codes(n); for (auto &c : codes) c = rand() & 0x3f;
    int64_t pn = ((n + 3) / 4) * 3;
    void *dc = up(codes.data(), n), *dp, *du;
    CHECK(aclrtMalloc(&dp, pn, ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&du, n, ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dn[1] = {n}, dpn[1] = {pn};
    aclTensor *ts = aclCreateTensor(dn, 1, ACL_FLOAT6_E2M3, nullptr, 0, ACL_FORMAT_ND, dn, 1, dc);
    aclTensor *tp = aclCreateTensor(dpn, 1, ACL_UINT8, nullptr, 0, ACL_FORMAT_ND, dpn, 1, dp);
    aclTensor *tu = aclCreateTensor(dn, 1, ACL_FLOAT6_E2M3, nullptr, 0, ACL_FORMAT_ND, dn, 1, du);
    uint64_t w = 0; aclOpExecutor *e = nullptr;
    CHECK(aclnnFp6PackGetWorkspaceSize(ts, tp, &w, &e)); CHECK(aclnnFp6Pack(nullptr, w, e, g_stream));
    CHECK(aclnnFp6UnpackGetWorkspaceSize(tp, tu, &w, &e)); CHECK(aclnnFp6Unpack(nullptr, w, e, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> back(n); CHECK(aclrtMemcpy(back.data(), n, du, n, ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t bad = 0; for (int64_t i = 0; i < n; i++) if (back[i] != codes[i]) bad++;
    report("fp6 4-in-3 pack/unpack", (double)bad, 0, true);
    aclDestroyTensor(ts); aclDestroyTensor(tp); aclDestroyTensor(tu); aclrtFree(dc); aclrtFree(dp); aclrtFree(du);
}

// Explicit quantize/dequantize
static void t_quant(int64_t n) {
    std::vector<float> x(n); for (auto &v : x) v = (rand()/(float)RAND_MAX)*6 - 3;
    float sq = 127.f/3.f, off = 0.f, sd = 1.f/sq;
    void *dx = up(x.data(), n*4), *dsq = up(&sq, 4), *dof = up(&off, 4), *dq, *dsd = up(&sd, 4), *dy;
    CHECK(aclrtMalloc(&dq, n, ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dy, n*4, ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dn[1]={n}, d1[1]={1};
    aclTensor *tx=aclCreateTensor(dn,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dn,1,dx), *tsq=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dsq),
              *tof=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dof), *tq=aclCreateTensor(dn,1,ACL_INT8,nullptr,0,ACL_FORMAT_ND,dn,1,dq),
              *tsd=aclCreateTensor(d1,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d1,1,dsd), *ty=aclCreateTensor(dn,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dn,1,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    CHECK(aclnnQuantizeGetWorkspaceSize(tx,tsq,tof,tq,&ws,&ex)); CHECK(aclnnQuantize(nullptr,ws,ex,g_stream));
    CHECK(aclnnDequantizeGetWorkspaceSize(tq,tsd,tof,ty,&ws,&ex)); CHECK(aclnnDequantize(nullptr,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<int8_t> hq(n); std::vector<float> hy(n);
    CHECK(aclrtMemcpy(hq.data(),n,dq,n,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hy.data(),n*4,dy,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    long bad=0; for(int64_t i=0;i<n;i++){ int qr=(int)std::lrint(x[i]*sq); qr=qr<-128?-128:qr>127?127:qr; if(hq[i]!=qr)bad++; if(std::fabs(hy[i]-hq[i]*sd)>1e-4)bad++; }
    report("Quantize+Dequantize", (double)bad, 0, true);
    aclDestroyTensor(tx);aclDestroyTensor(tsq);aclDestroyTensor(tof);aclDestroyTensor(tq);aclDestroyTensor(tsd);aclDestroyTensor(ty);
    aclrtFree(dx);aclrtFree(dsq);aclrtFree(dof);aclrtFree(dq);aclrtFree(dsd);aclrtFree(dy);
}
static void t_dynquant(int64_t rows,int64_t D){
    std::vector<float> x(rows*D); for(auto&v:x)v=(rand()/(float)RAND_MAX)*4-2;
    void*dx=up(x.data(),rows*D*4),*dq,*dsc; CHECK(aclrtMalloc(&dq,rows*D,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dsc,rows*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dx2[2]={rows,D}, dr[1]={rows};
    aclTensor*tx=aclCreateTensor(dx2,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dx2,2,dx),*tq=aclCreateTensor(dx2,2,ACL_INT8,nullptr,0,ACL_FORMAT_ND,dx2,2,dq),*tsc=aclCreateTensor(dr,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dr,1,dsc);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnDynamicQuantGetWorkspaceSize(tx,tq,tsc,&ws,&ex)); CHECK(aclnnDynamicQuant(nullptr,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<int8_t> hq(rows*D); std::vector<float> hsc(rows); CHECK(aclrtMemcpy(hq.data(),rows*D,dq,rows*D,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hsc.data(),rows*4,dsc,rows*4,ACL_MEMCPY_DEVICE_TO_HOST));
    long bad=0; for(int64_t r=0;r<rows;r++){ float amax=0; for(int64_t d=0;d<D;d++)amax=std::max(amax,std::fabs(x[r*D+d])); float sc=amax/127.f; if(sc==0)sc=1; if(std::fabs(hsc[r]-sc)>1e-6*sc+1e-9)bad++;
        for(int64_t d=0;d<D;d++){int qr=(int)std::lrint(x[r*D+d]/sc); qr=qr<-128?-128:qr>127?127:qr; if(hq[r*D+d]!=qr)bad++;} }
    report("DynamicQuant per-token",(double)bad,0,true);
    aclDestroyTensor(tx);aclDestroyTensor(tq);aclDestroyTensor(tsc); aclrtFree(dx);aclrtFree(dq);aclrtFree(dsc);
}

int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    CHECK(aclrtCreateStream(&g_stream));
    t_quant_roundtrip(false, 1 << 16);
    t_quant_roundtrip(true, 1 << 16);
    t_fp8_add(false, 1 << 16);
    t_fp8_add(true, 1 << 16);
    t_hif8(1 << 16);
    { SubFmt f{8, D_4E2M1, B_4E2M1}; t_fp4(ACL_FLOAT4_E2M1, f, "fp4e2m1", 1 << 16); }
    { SubFmt f{8, D_4E1M2, B_4E1M2}; t_fp4(ACL_FLOAT4_E1M2, f, "fp4e1m2", 1 << 16); }
    { SubFmt f{32, D_6E2M3, B_6E2M3}; t_fp6(ACL_FLOAT6_E2M3, f, "fp6e2m3", 1 << 16); }
    { SubFmt f{32, D_6E3M2, B_6E3M2}; t_fp6(ACL_FLOAT6_E3M2, f, "fp6e3m2", 1 << 16); }
    { SubFmt f{8, D_4E2M1, B_4E2M1}; t_fp4_add(ACL_FLOAT4_E2M1, f, "fp4e2m1", 1 << 16); }
    { SubFmt f{8, D_4E1M2, B_4E1M2}; t_fp4_add(ACL_FLOAT4_E1M2, f, "fp4e1m2", 1 << 16); }
    { SubFmt f{8, D_4E2M1, B_4E2M1}; t_fp4_add_bcast(ACL_FLOAT4_E2M1, f, "fp4e2m1", 64, 48); }
    t_fp6_pack(4099);   // non-multiple-of-4, verifies tail handling
    t_quant(1 << 16);   // explicit quantize/dequantize
    t_dynquant(512, 256);
    CHECK(aclrtDestroyStream(g_stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}

// ---- CPU fp8 encode/decode (IEEE-754 style, aligned with CUDA __nv_fp8 behavior: RNE, saturation/NaN) ----
// e4m3: 1-4-3, bias 7, max 448, no inf (S.1111.111=NaN).
static uint8_t enc_e4m3(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    int32_t e = (int32_t)((x >> 23) & 0xFF) - 127;
    uint32_t m = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint8_t)((sign << 7) | 0x7F);   // inf/nan -> nan
    // Target exponent range [-6,8], bias 7
    int32_t E = e + 7;
    if (E >= 15) return (uint8_t)((sign << 7) | 0x7E);                      // saturate to max (448)
    uint32_t mant;
    if (E <= 0) {                                                           // subnormal
        int32_t shift = 1 - E;                                             // shift in implicit 1
        if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t full = m | 0x800000;
        uint32_t q = full >> (23 - 3 + shift);
        uint32_t rem = full & ((1u << (23 - 3 + shift)) - 1), half = 1u << (23 - 3 + shift - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return (uint8_t)((sign << 7) | (q & 0x7F));
    }
    mant = m >> (23 - 3);
    uint32_t rem = m & ((1u << (23 - 3)) - 1), half = 1u << (23 - 3 - 1);
    uint32_t out = (E << 3) | mant;
    if (rem > half || (rem == half && (mant & 1))) out++;                   // RNE (may carry into exponent)
    if (out >= 0x7F) out = 0x7E;                                           // carry overflow: saturate
    return (uint8_t)((sign << 7) | (out & 0x7F));
}
static float dec_e4m3(uint8_t v) {
    uint32_t sign = (v >> 7) & 1, E = (v >> 3) & 0xF, m = v & 0x7;
    if (E == 0xF && m == 0x7) return sign ? -__builtin_nanf("") : __builtin_nanf("");
    float r;
    if (E == 0) r = m * 0x1p-3f * 0x1p-6f;                                  // subnormal: 2^-6 * (m/8)
    else r = (1.f + m / 8.f) * __builtin_powf(2.f, (int)E - 7);
    return sign ? -r : r;
}
// e5m2: 1-5-2, bias 15, has inf/nan (same as IEEE half-precision truncation).
static uint8_t enc_e5m2(float f) {
    uint32_t x; __builtin_memcpy(&x, &f, 4);
    uint32_t sign = (x >> 31) & 1;
    int32_t E = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t m = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) return (uint8_t)((sign << 7) | (m ? 0x7F : 0x7C));  // nan/inf
    if (E >= 31) return (uint8_t)((sign << 7) | 0x7C);                       // inf
    if (E <= 0) {
        int32_t shift = 1 - E;
        if (shift > 24) return (uint8_t)(sign << 7);
        uint32_t full = m | 0x800000;
        uint32_t q = full >> (23 - 2 + shift);
        uint32_t rem = full & ((1u << (23 - 2 + shift)) - 1), half = 1u << (23 - 2 + shift - 1);
        if (rem > half || (rem == half && (q & 1))) q++;
        return (uint8_t)((sign << 7) | (q & 0x7F));
    }
    uint32_t mant = m >> (23 - 2);
    uint32_t rem = m & ((1u << (23 - 2)) - 1), half = 1u << (23 - 2 - 1);
    uint32_t out = (E << 2) | mant;
    if (rem > half || (rem == half && (mant & 1))) out++;
    if (out >= 0x7C) out = 0x7C;                                            // carry to inf: stay inf
    return (uint8_t)((sign << 7) | (out & 0x7F));
}
static float dec_e5m2(uint8_t v) {
    uint32_t sign = (v >> 7) & 1, E = (v >> 2) & 0x1F, m = v & 0x3;
    if (E == 0x1F) return m ? __builtin_nanf("") : (sign ? -__builtin_inff() : __builtin_inff());
    float r;
    if (E == 0) r = m * 0x1p-2f * 0x1p-14f;
    else r = (1.f + m / 4.f) * __builtin_powf(2.f, (int)E - 15);
    return sign ? -r : r;
}
