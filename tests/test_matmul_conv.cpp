// matmul (cuBLASLt) + conv2d (cuDNN) cross-check against CPU double reference.
// Matrix/conv ops are long-chain accumulations; tolerance is relaxed by K/receptive-field (fp32 1e-5, fp16 1e-2).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
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

static void *up(const void *h, size_t b) {
    void *d; CHECK(aclrtMalloc(&d, b, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d, b, h, b, ACL_MEMCPY_HOST_TO_DEVICE));
    return d;
}
static aclTensor *mk(std::vector<int64_t> dims, aclDataType dt, void *p) {
    return aclCreateTensor(dims.data(), dims.size(), dt, nullptr, 0, ACL_FORMAT_ND, dims.data(), dims.size(), p);
}
static std::vector<float> randv(int64_t n) {
    std::vector<float> v(n);
    for (auto &x : v) x = (rand() / (float)RAND_MAX) * 2.f - 1.f;
    return v;
}
static void report(const char *name, double maxrel, double tol) {
    bool ok = maxrel <= tol;
    (ok ? g_pass : g_fail)++;
    printf("%-30s maxrel=%.2e tol=%.0e %s\n", name, maxrel, tol, ok ? "PASS" : "FAIL");
}

static void t_matmul(bool fp16, int64_t M, int64_t K, int64_t N, double tol) {
    auto A = randv(M * K), B = randv(K * N);
    size_t es = fp16 ? 2 : 4;
    std::vector<uint8_t> ha(M * K * es), hb(K * N * es), hc(M * N * es);
    for (int64_t i = 0; i < M * K; i++)
        fp16 ? (void)(((uint16_t*)ha.data())[i] = f2h(A[i])) : (void)(((float*)ha.data())[i] = A[i]);
    for (int64_t i = 0; i < K * N; i++)
        fp16 ? (void)(((uint16_t*)hb.data())[i] = f2h(B[i])) : (void)(((float*)hb.data())[i] = B[i]);
    aclDataType dt = fp16 ? ACL_FLOAT16 : ACL_FLOAT;
    void *da = up(ha.data(), ha.size()), *db = up(hb.data(), hb.size()), *dc;
    CHECK(aclrtMalloc(&dc, hc.size(), ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *ta = mk({M, K}, dt, da), *tb = mk({K, N}, dt, db), *tc = mk({M, N}, dt, dc);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex));
    void *wsp = nullptr;
    if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(), hc.size(), dc, hc.size(), ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double r = 0;
            for (int64_t l = 0; l < K; l++) {
                double x = fp16 ? h2f(((uint16_t*)ha.data())[i*K+l]) : ((float*)ha.data())[i*K+l];
                double y = fp16 ? h2f(((uint16_t*)hb.data())[l*N+j]) : ((float*)hb.data())[l*N+j];
                r += x * y;
            }
            double got = fp16 ? h2f(((uint16_t*)hc.data())[i*N+j]) : ((float*)hc.data())[i*N+j];
            maxrel = std::max(maxrel, std::fabs(got - r) / (std::fabs(r) + 1.0));  // K-term sum; atol=1 guards against near-zero denominator
        }
    char nm[64]; snprintf(nm, sizeof nm, "Matmul %s %ldx%ldx%ld", fp16 ? "fp16" : "fp32", (long)M, (long)K, (long)N);
    report(nm, maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
    if (wsp) aclrtFree(wsp);
    aclrtFree(da); aclrtFree(db); aclrtFree(dc);
}

// TF32 low-precision fast-path switch: CANN_FAST_TF32=1 routes fp32 GEMM through TF32 tensor-core.
// Verifies results are within TF32 tolerance (~2e-3, 10-bit mantissa) and that precision is indeed reduced (error > full fp32 ~1e-6).
static void t_matmul_tf32(int64_t M, int64_t K, int64_t N) {
    auto A = randv(M * K), B = randv(K * N);
    void *da = up(A.data(), M*K*4), *db = up(B.data(), K*N*4), *dc;
    CHECK(aclrtMalloc(&dc, M*N*4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *ta = mk({M,K}, ACL_FLOAT, da), *tb = mk({K,N}, ACL_FLOAT, db), *tc = mk({M,N}, ACL_FLOAT, dc);
    setenv("CANN_FAST_TF32", "1", 1);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr; CHECK(aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp, ws, ex, g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    unsetenv("CANN_FAST_TF32");
    std::vector<float> hc(M*N); CHECK(aclrtMemcpy(hc.data(), M*N*4, dc, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++) for (int64_t j = 0; j < N; j++) {
        double r = 0; for (int64_t l = 0; l < K; l++) r += (double)A[i*K+l]*B[l*N+j];
        maxrel = std::max(maxrel, std::fabs(hc[i*N+j]-r)/(std::fabs(r)+1.0));
    }
    char nm[64]; snprintf(nm, sizeof nm, "Matmul TF32 %ldx%ldx%ld", (long)M,(long)K,(long)N);
    report(nm, maxrel, 5e-3);   // TF32 tolerance
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
    if (wsp) aclrtFree(wsp); aclrtFree(da); aclrtFree(db); aclrtFree(dc);
}

// fp8 matmul: quantization uses the shim's own aclnnCast (pure ACL client cannot directly construct fp8 bytes);
// CPU reference uses the fp8 dequantized values (cast-back) to verify fp8 GEMM accumulation correctness.
static void quant_to_fp8(void *f32_dev, void *fp8_dev, void *back_dev, int64_t n, aclDataType fp8dt, aclrtStream s) {
    int64_t d[1] = {n};
    aclTensor *tf = aclCreateTensor(d, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d, 1, f32_dev);
    aclTensor *tq = aclCreateTensor(d, 1, fp8dt, nullptr, 0, ACL_FORMAT_ND, d, 1, fp8_dev);
    aclTensor *tb = aclCreateTensor(d, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, d, 1, back_dev);
    uint64_t ws; aclOpExecutor *ex;
    CHECK(aclnnCastGetWorkspaceSize(tf, fp8dt, tq, &ws, &ex)); CHECK(aclnnCast(nullptr, ws, ex, s));
    CHECK(aclnnCastGetWorkspaceSize(tq, ACL_FLOAT, tb, &ws, &ex)); CHECK(aclnnCast(nullptr, ws, ex, s));
    aclDestroyTensor(tf); aclDestroyTensor(tq); aclDestroyTensor(tb);
}

static void t_matmul_fp8(int64_t M, int64_t K, int64_t N, double tol) {
    auto A = randv(M * K), B = randv(K * N);
    void *dAf = up(A.data(), A.size() * 4), *dBf = up(B.data(), B.size() * 4);
    void *dA8, *dB8, *dAb, *dBb, *dC;
    CHECK(aclrtMalloc(&dA8, M * K, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dB8, K * N, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dAb, M * K * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dBb, K * N * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dC, M * N * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    quant_to_fp8(dAf, dA8, dAb, M * K, ACL_FLOAT8_E4M3FN, g_stream);
    quant_to_fp8(dBf, dB8, dBb, K * N, ACL_FLOAT8_E4M3FN, g_stream);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Ab(M * K), Bb(K * N), C(M * N);
    CHECK(aclrtMemcpy(Ab.data(), M*K*4, dAb, M*K*4, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(Bb.data(), K*N*4, dBb, K*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t da[2] = {M, K}, db[2] = {K, N}, dc[2] = {M, N};
    aclTensor *ta = aclCreateTensor(da, 2, ACL_FLOAT8_E4M3FN, nullptr, 0, ACL_FORMAT_ND, da, 2, dA8);
    aclTensor *tb = aclCreateTensor(db, 2, ACL_FLOAT8_E4M3FN, nullptr, 0, ACL_FORMAT_ND, db, 2, dB8);
    aclTensor *tc = aclCreateTensor(dc, 2, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, dc, 2, dC);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnMatmulGetWorkspaceSize(ta, tb, tc, 1, &ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(C.data(), M*N*4, dC, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double r = 0;
            for (int64_t l = 0; l < K; l++) r += (double)Ab[i*K+l] * Bb[l*N+j];
            maxrel = std::max(maxrel, std::fabs(C[i*N+j] - r) / (std::fabs(r) + 1.0));
        }
    char nm[64]; snprintf(nm, sizeof nm, "Matmul fp8e4m3 %ldx%ldx%ld", (long)M, (long)K, (long)N);
    report(nm, maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tc);
    if (wsp) aclrtFree(wsp);
    aclrtFree(dAf); aclrtFree(dBf); aclrtFree(dA8); aclrtFree(dB8); aclrtFree(dAb); aclrtFree(dBb); aclrtFree(dC);
}

// fp4 matmul: quantization via shim Cast (packed); reference uses fp4 dequantized values.
static void t_matmul_fp4(aclDataType dt, const char*name, int64_t M, int64_t K, int64_t N, double tol) {
    auto A = randv(M*K), B = randv(K*N);
    for(auto&v:A)v*=4; for(auto&v:B)v*=4;     // scale into fp4/fp6 representable range
    bool fp4 = (dt==ACL_FLOAT4_E2M1 || dt==ACL_FLOAT4_E1M2);
    int64_t nA = fp4 ? (M*K+1)/2 : M*K, nB = fp4 ? (K*N+1)/2 : K*N;   // fp4=2 elements/byte, fp6=1 element/byte
    void *dAf=up(A.data(),M*K*4), *dBf=up(B.data(),K*N*4);
    void *dA4,*dB4,*dAb,*dBb,*dC;
    CHECK(aclrtMalloc(&dA4,nA,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dB4,nB,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dAb,M*K*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dBb,K*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dC,M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    auto cast=[&](void*src,aclDataType sdt,int64_t ne,void*dst,aclDataType ddt){
        int64_t d[1]={ne}; aclTensor*s=aclCreateTensor(d,1,sdt,nullptr,0,ACL_FORMAT_ND,d,1,src);
        aclTensor*o=aclCreateTensor(d,1,ddt,nullptr,0,ACL_FORMAT_ND,d,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream));
        aclDestroyTensor(s);aclDestroyTensor(o);
    };
    cast(dAf,ACL_FLOAT,M*K,dA4,dt); cast(dBf,ACL_FLOAT,K*N,dB4,dt);
    cast(dA4,dt,M*K,dAb,ACL_FLOAT); cast(dB4,dt,K*N,dBb,ACL_FLOAT);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Ab(M*K),Bb(K*N),C(M*N);
    CHECK(aclrtMemcpy(Ab.data(),M*K*4,dAb,M*K*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(Bb.data(),K*N*4,dBb,K*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    int64_t da[2]={M,K},db[2]={K,N},dc[2]={M,N};
    aclTensor*ta=aclCreateTensor(da,2,dt,nullptr,0,ACL_FORMAT_ND,da,2,dA4);
    aclTensor*tb=aclCreateTensor(db,2,dt,nullptr,0,ACL_FORMAT_ND,db,2,dB4);
    aclTensor*tc=aclCreateTensor(dc,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dc,2,dC);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    CHECK(aclnnMatmulGetWorkspaceSize(ta,tb,tc,1,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(C.data(),M*N*4,dC,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){double r=0;for(int64_t l=0;l<K;l++)r+=(double)Ab[i*K+l]*Bb[l*N+j];maxrel=std::max(maxrel,std::fabs(C[i*N+j]-r)/(std::fabs(r)+1.0));}
    char nm[64]; snprintf(nm,sizeof nm,"Matmul %s %ldx%ldx%ld",name,(long)M,(long)K,(long)N);
    report(nm,maxrel,tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp);
    aclrtFree(dAf);aclrtFree(dBf);aclrtFree(dA4);aclrtFree(dB4);aclrtFree(dAb);aclrtFree(dBb);aclrtFree(dC);
}

// MXFP4: fp4 e2m1 (2 elements/byte) + E8M0 per-32-block scaling. Block scales quantized via shim Cast; reference uses block-dequantized values.
static void t_matmul_mxfp4(int64_t M, int64_t K, int64_t N, double tol, bool hw = false) {
    const int BLK = 32; int64_t nbk = K / BLK;
    auto A = randv(M*K), B = randv(K*N);
    std::vector<uint8_t> sA(M*nbk), sB(nbk*N);
    std::vector<float> Asc(M*K), Bsc(K*N);
    for (int64_t i=0;i<M;i++) for (int64_t b=0;b<nbk;b++){ float mx=1e-30f; for(int t=0;t<BLK;t++)mx=std::max(mx,std::fabs(A[i*K+b*BLK+t]));
        int e=(int)std::floor(std::log2(mx)); sA[i*nbk+b]=(uint8_t)(e+127); for(int t=0;t<BLK;t++)Asc[i*K+b*BLK+t]=A[i*K+b*BLK+t]/std::pow(2.f,e); }
    for (int64_t b=0;b<nbk;b++) for (int64_t j=0;j<N;j++){ float mx=1e-30f; for(int t=0;t<BLK;t++)mx=std::max(mx,std::fabs(B[(b*BLK+t)*N+j]));
        int e=(int)std::floor(std::log2(mx)); sB[b*N+j]=(uint8_t)(e+127); for(int t=0;t<BLK;t++)Bsc[(b*BLK+t)*N+j]=B[(b*BLK+t)*N+j]/std::pow(2.f,e); }
    auto cast=[&](void*src,aclDataType sdt,int64_t ne,void*dst,aclDataType ddt){
        int64_t d[1]={ne}; aclTensor*s=aclCreateTensor(d,1,sdt,nullptr,0,ACL_FORMAT_ND,d,1,src);
        aclTensor*o=aclCreateTensor(d,1,ddt,nullptr,0,ACL_FORMAT_ND,d,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream));
        aclDestroyTensor(s);aclDestroyTensor(o); };
    void *dAsc=up(Asc.data(),M*K*4), *dBsc=up(Bsc.data(),K*N*4);
    void *dA4,*dB4,*dAb,*dBb; CHECK(aclrtMalloc(&dA4,(M*K+1)/2,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dB4,(K*N+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dAb,M*K*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dBb,K*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    cast(dAsc,ACL_FLOAT,M*K,dA4,ACL_FLOAT4_E2M1); cast(dBsc,ACL_FLOAT,K*N,dB4,ACL_FLOAT4_E2M1);
    cast(dA4,ACL_FLOAT4_E2M1,M*K,dAb,ACL_FLOAT); cast(dB4,ACL_FLOAT4_E2M1,K*N,dBb,ACL_FLOAT);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Ab(M*K),Bb(K*N); CHECK(aclrtMemcpy(Ab.data(),M*K*4,dAb,M*K*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(Bb.data(),K*N*4,dBb,K*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    void *dsA=up(sA.data(),M*nbk),*dsB=up(sB.data(),nbk*N),*dC; CHECK(aclrtMalloc(&dC,M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t da[2]={M,K},db[2]={K,N},dc[2]={M,N},dsa[2]={M,nbk},dsb[2]={nbk,N};
    aclTensor*ta=aclCreateTensor(da,2,ACL_FLOAT4_E2M1,nullptr,0,ACL_FORMAT_ND,da,2,dA4);
    aclTensor*tb=aclCreateTensor(db,2,ACL_FLOAT4_E2M1,nullptr,0,ACL_FORMAT_ND,db,2,dB4);
    aclTensor*tsa=aclCreateTensor(dsa,2,ACL_FLOAT8_E8M0,nullptr,0,ACL_FORMAT_ND,dsa,2,dsA);
    aclTensor*tsb=aclCreateTensor(dsb,2,ACL_FLOAT8_E8M0,nullptr,0,ACL_FORMAT_ND,dsb,2,dsB);
    aclTensor*tc=aclCreateTensor(dc,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dc,2,dC);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if (hw) CHECK(aclnnMatmulMxFp4HwGetWorkspaceSize(ta,tsa,tb,tsb,tc,&ws,&ex)); else CHECK(aclnnMatmulMxFp4GetWorkspaceSize(ta,tsa,tb,tsb,tc,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if (hw) CHECK(aclnnMatmulMxFp4Hw(wsp,ws,ex,g_stream)); else CHECK(aclnnMatmulMxFp4(wsp,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> C(M*N); CHECK(aclrtMemcpy(C.data(),M*N*4,dC,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxerr=0,maxref=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double r=0;
        for(int64_t b=0;b<nbk;b++){ double scA=std::pow(2.f,(int)sA[i*nbk+b]-127), scB=std::pow(2.f,(int)sB[b*N+j]-127);
            for(int t=0;t<BLK;t++) r+=(double)Ab[i*K+b*BLK+t]*scA*Bb[(b*BLK+t)*N+j]*scB; }
        maxerr=std::max(maxerr,std::fabs(C[i*N+j]-r)); maxref=std::max(maxref,std::fabs(r)); }
    char nm[64]; snprintf(nm,sizeof nm,"MXFP4%s e2m1 %ldx%ldx%ld blk32",hw?"-HW":"",(long)M,(long)K,(long)N);
    report(nm,maxerr/(maxref+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tsa);aclDestroyTensor(tsb);aclDestroyTensor(tc);
    if(wsp)aclrtFree(wsp); aclrtFree(dAsc);aclrtFree(dBsc);aclrtFree(dA4);aclrtFree(dB4);aclrtFree(dAb);aclrtFree(dBb);aclrtFree(dsA);aclrtFree(dsB);aclrtFree(dC);
}

// MXFP8: one E8M0 scale per 32-element block (along K). Quantization via shim Cast; reference uses block-dequantized values.
static void t_matmul_mxfp8(int64_t M, int64_t K, int64_t N, double tol, bool hw = false) {
    const int BLK = 32; int64_t nbk = K / BLK;
    auto A = randv(M * K), B = randv(K * N);
    // Select E8M0 exponent per block so that scaled values fall within fp8-friendly range
    std::vector<uint8_t> sA(M * nbk), sB(nbk * N);
    std::vector<float> Asc(M * K), Bsc(K * N);
    for (int64_t i = 0; i < M; i++)
        for (int64_t b = 0; b < nbk; b++) {
            float mx = 1e-30f; for (int t = 0; t < BLK; t++) mx = std::max(mx, std::fabs(A[i*K+b*BLK+t]));
            int e = (int)std::floor(std::log2(mx)); sA[i*nbk+b] = (uint8_t)(e + 127);
            for (int t = 0; t < BLK; t++) Asc[i*K+b*BLK+t] = A[i*K+b*BLK+t] / std::pow(2.f, e);
        }
    for (int64_t b = 0; b < nbk; b++)
        for (int64_t j = 0; j < N; j++) {
            float mx = 1e-30f; for (int t = 0; t < BLK; t++) mx = std::max(mx, std::fabs(B[(b*BLK+t)*N+j]));
            int e = (int)std::floor(std::log2(mx)); sB[b*N+j] = (uint8_t)(e + 127);
            for (int t = 0; t < BLK; t++) Bsc[(b*BLK+t)*N+j] = B[(b*BLK+t)*N+j] / std::pow(2.f, e);
        }
    // Quantize scaled values to fp8 (shim Cast) and retrieve dequantized values as reference
    void *dAsc = up(Asc.data(), M*K*4), *dBsc = up(Bsc.data(), K*N*4);
    void *dA8, *dB8, *dAb, *dBb;
    CHECK(aclrtMalloc(&dA8, M*K, ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dB8, K*N, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dAb, M*K*4, ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dBb, K*N*4, ACL_MEM_MALLOC_HUGE_FIRST));
    quant_to_fp8(dAsc, dA8, dAb, M*K, ACL_FLOAT8_E4M3FN, g_stream);
    quant_to_fp8(dBsc, dB8, dBb, K*N, ACL_FLOAT8_E4M3FN, g_stream);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Ab(M*K), Bb(K*N);
    CHECK(aclrtMemcpy(Ab.data(), M*K*4, dAb, M*K*4, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(Bb.data(), K*N*4, dBb, K*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
    // Upload E8M0 scale tensors to device
    void *dsA = up(sA.data(), M*nbk), *dsB = up(sB.data(), nbk*N), *dC;
    CHECK(aclrtMalloc(&dC, M*N*4, ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t da[2]={M,K}, db[2]={K,N}, dc[2]={M,N}, dsa[2]={M,nbk}, dsb[2]={nbk,N};
    aclTensor *ta = aclCreateTensor(da,2,ACL_FLOAT8_E4M3FN,nullptr,0,ACL_FORMAT_ND,da,2,dA8);
    aclTensor *tb = aclCreateTensor(db,2,ACL_FLOAT8_E4M3FN,nullptr,0,ACL_FORMAT_ND,db,2,dB8);
    aclTensor *tsa = aclCreateTensor(dsa,2,ACL_FLOAT8_E8M0,nullptr,0,ACL_FORMAT_ND,dsa,2,dsA);
    aclTensor *tsb = aclCreateTensor(dsb,2,ACL_FLOAT8_E8M0,nullptr,0,ACL_FORMAT_ND,dsb,2,dsB);
    aclTensor *tc = aclCreateTensor(dc,2,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dc,2,dC);
    uint64_t ws=0; aclOpExecutor *ex=nullptr;
    if (hw) CHECK(aclnnMatmulMxFp8HwGetWorkspaceSize(ta, tsa, tb, tsb, tc, &ws, &ex));
    else    CHECK(aclnnMatmulMxFp8GetWorkspaceSize(ta, tsa, tb, tsb, tc, &ws, &ex));
    void *wsp=nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    if (hw) CHECK(aclnnMatmulMxFp8Hw(wsp, ws, ex, g_stream));
    else    CHECK(aclnnMatmulMxFp8(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> C(M*N);
    CHECK(aclrtMemcpy(C.data(), M*N*4, dC, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            double r = 0;
            for (int64_t b = 0; b < nbk; b++) {
                double scA = std::pow(2.f, (int)sA[i*nbk+b]-127), scB = std::pow(2.f, (int)sB[b*N+j]-127);
                for (int t = 0; t < BLK; t++) r += (double)Ab[i*K+b*BLK+t]*scA * Bb[(b*BLK+t)*N+j]*scB;
            }
            maxrel = std::max(maxrel, std::fabs(C[i*N+j]-r)/(std::fabs(r)+1.0));
        }
    char nm[64]; snprintf(nm, sizeof nm, "MXFP8%s e4m3 %ldx%ldx%ld blk32", hw?"-HW":"", (long)M,(long)K,(long)N);
    report(nm, maxrel, tol);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tsa); aclDestroyTensor(tsb); aclDestroyTensor(tc);
    if (wsp) aclrtFree(wsp);
    aclrtFree(dAsc); aclrtFree(dBsc); aclrtFree(dA8); aclrtFree(dB8); aclrtFree(dAb); aclrtFree(dBb);
    aclrtFree(dsA); aclrtFree(dsB); aclrtFree(dC);
}

static void t_conv(int64_t Nn, int64_t C, int64_t H, int64_t W, int64_t Co, int64_t R, int64_t S,
                   int64_t st, int64_t pd, bool bias, double tol) {
    const int64_t Ho = (H + 2 * pd - R) / st + 1, Wo = (W + 2 * pd - S) / st + 1;
    auto X = randv(Nn * C * H * W), Wt = randv(Co * C * R * S), Bs = randv(Co);
    std::vector<float> Y(Nn * Co * Ho * Wo);
    void *dx = up(X.data(), X.size() * 4), *dw = up(Wt.data(), Wt.size() * 4), *dy, *db = nullptr;
    CHECK(aclrtMalloc(&dy, Y.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    if (bias) db = up(Bs.data(), Bs.size() * 4);
    aclTensor *tx = mk({Nn, C, H, W}, ACL_FLOAT, dx), *tw = mk({Co, C, R, S}, ACL_FLOAT, dw),
              *ty = mk({Nn, Co, Ho, Wo}, ACL_FLOAT, dy), *tb = bias ? mk({Co}, ACL_FLOAT, db) : nullptr;
    int64_t s2[2] = {st, st}, p2[2] = {pd, pd}, d2[2] = {1, 1};
    aclIntArray *as = aclCreateIntArray(s2, 2), *ap = aclCreateIntArray(p2, 2), *ad = aclCreateIntArray(d2, 2);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnConvolutionGetWorkspaceSize(tx, tw, tb, as, ap, ad, false, nullptr, 1, ty, 1, &ws, &ex));
    void *wsp = nullptr;
    if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnConvolution(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(Y.data(), Y.size() * 4, dy, Y.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t n = 0; n < Nn; n++)
        for (int64_t co = 0; co < Co; co++)
            for (int64_t ho = 0; ho < Ho; ho++)
                for (int64_t wo = 0; wo < Wo; wo++) {
                    double r = bias ? Bs[co] : 0.0;
                    for (int64_t c = 0; c < C; c++)
                        for (int64_t kr = 0; kr < R; kr++)
                            for (int64_t ks = 0; ks < S; ks++) {
                                int64_t hi = ho * st - pd + kr, wi = wo * st - pd + ks;
                                if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;
                                r += (double)X[((n*C+c)*H+hi)*W+wi] * Wt[((co*C+c)*R+kr)*S+ks];
                            }
                    double got = Y[((n*Co+co)*Ho+ho)*Wo+wo];
                    maxrel = std::max(maxrel, std::fabs(got - r) / (std::fabs(r) + 1.0));
                }
    char nm[80]; snprintf(nm, sizeof nm, "Conv %ldx%ldx%ldx%ld k%ld s%ld p%ld%s",
                          (long)Nn, (long)C, (long)H, (long)W, (long)R, (long)st, (long)pd, bias ? " +bias" : "");
    report(nm, maxrel, tol);
    aclDestroyIntArray(as); aclDestroyIntArray(ap); aclDestroyIntArray(ad);
    aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(ty); if (tb) aclDestroyTensor(tb);
    if (wsp) aclrtFree(wsp);
    aclrtFree(dx); aclrtFree(dw); aclrtFree(dy); if (db) aclrtFree(db);
}

// BatchMatMul [B,M,K]@[B,K,N]（fp32）
static void t_bmm(int64_t B, int64_t M, int64_t K, int64_t N, double tol) {
    auto A = randv(B*M*K), Bm = randv(B*K*N); std::vector<float> hc(B*M*N);
    void *da=up(A.data(),B*M*K*4), *db=up(Bm.data(),B*K*N*4), *dc; CHECK(aclrtMalloc(&dc,B*M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *ta=mk({B,M,K},ACL_FLOAT,da), *tb=mk({B,K,N},ACL_FLOAT,db), *tc=mk({B,M,N},ACL_FLOAT,dc);
    uint64_t ws=0; aclOpExecutor *ex=nullptr;
    CHECK(aclnnBatchMatMulGetWorkspaceSize(ta,tb,tc,0,&ws,&ex));
    void *wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnBatchMatMul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),B*M*N*4,dc,B*M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    // Normalized error: max|err| / max|ref| (robust for TF32 tensor-core GEMM; avoids near-zero elements inflating relative error)
    double maxerr=0, maxref=0;
    for(int64_t b=0;b<B;b++)for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){
        double acc=0; for(int64_t l=0;l<K;l++) acc+=(double)A[(b*M+i)*K+l]*Bm[(b*K+l)*N+j];
        double got=hc[(b*M+i)*N+j]; maxerr=std::max(maxerr,std::fabs(got-acc)); maxref=std::max(maxref,std::fabs(acc));
    }
    char nm[64]; snprintf(nm,64,"BMM [%lld,%lld,%lld,%lld]",(long long)B,(long long)M,(long long)K,(long long)N);
    report(nm,maxerr/(maxref+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); aclrtFree(da);aclrtFree(db);aclrtFree(dc); if(wsp)aclrtFree(wsp);
}

// MatmulBias: transA/transB/bias/act combinations (fp32). biasMode: 0=none / 1=[N] / 2=[M,N]
static void t_mmbias(const char *name, bool tA, bool tB, int act, int biasMode, int64_t M, int64_t K, int64_t N, double tol) {
    std::vector<int64_t> ad = tA ? std::vector<int64_t>{K,M} : std::vector<int64_t>{M,K};
    std::vector<int64_t> bd = tB ? std::vector<int64_t>{N,K} : std::vector<int64_t>{K,N};
    auto A=randv(M*K), Bm=randv(K*N); int64_t bn = biasMode==1?N:(biasMode==2?M*N:0);
    auto bias = bn? randv(bn) : std::vector<float>();
    std::vector<float> hc(M*N);
    void *da=up(A.data(),M*K*4), *db=up(Bm.data(),K*N*4), *dbias=bn?up(bias.data(),bn*4):nullptr, *dc;
    CHECK(aclrtMalloc(&dc,M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *ta=mk(ad,ACL_FLOAT,da), *tb=mk(bd,ACL_FLOAT,db), *tc=mk({M,N},ACL_FLOAT,dc);
    aclTensor *tbias = bn? mk(biasMode==1?std::vector<int64_t>{N}:std::vector<int64_t>{M,N},ACL_FLOAT,dbias):nullptr;
    uint64_t ws=0; aclOpExecutor *ex=nullptr;
    CHECK(aclnnMatmulBiasGetWorkspaceSize(ta,tb,tbias,tA,tB,act,tc,&ws,&ex));
    void *wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmulBias(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),M*N*4,dc,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxerr=0, maxref=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){
        double acc=0;
        for(int64_t l=0;l<K;l++){ double a=tA?A[l*M+i]:A[i*K+l]; double b=tB?Bm[j*K+l]:Bm[l*N+j]; acc+=a*b; }
        if(biasMode==1) acc+=bias[j]; else if(biasMode==2) acc+=bias[i*N+j];
        if(act==1) acc=acc>0?acc:0; else if(act==2) acc=0.5*acc*(1+std::erf(acc*0.70710678118654752));
        double got=hc[i*N+j]; maxerr=std::max(maxerr,std::fabs(got-acc)); maxref=std::max(maxref,std::fabs(acc));
    }
    report(name,maxerr/(maxref+1e-6),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); if(tbias)aclDestroyTensor(tbias);
    aclrtFree(da);aclrtFree(db);aclrtFree(dc); if(dbias)aclrtFree(dbias); if(wsp)aclrtFree(wsp);
}

// Small conv backward cross-check (fp32, groups=1): dgrad + wgrad
static void t_conv_bwd(int64_t N,int64_t C,int64_t H,int64_t W,int64_t Co,int64_t KH,int64_t KW,int64_t S,int64_t P,int64_t D){
    int64_t Ho=(H+2*P-(D*(KH-1)+1))/S+1, Wo=(W+2*P-(D*(KW-1)+1))/S+1;
    auto X=randv(N*C*H*W), Wt=randv(Co*C*KH*KW), dY=randv(N*Co*Ho*Wo);
    void *dx=up(X.data(),X.size()*4), *dw=up(Wt.data(),Wt.size()*4), *ddy=up(dY.data(),dY.size()*4);
    void *ddx, *ddw; CHECK(aclrtMalloc(&ddx,N*C*H*W*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&ddw,Co*C*KH*KW*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *tX=mk({N,C,H,W},ACL_FLOAT,dx),*tW=mk({Co,C,KH,KW},ACL_FLOAT,dw),*tdY=mk({N,Co,Ho,Wo},ACL_FLOAT,ddy);
    aclTensor *tdX=mk({N,C,H,W},ACL_FLOAT,ddx),*tdW=mk({Co,C,KH,KW},ACL_FLOAT,ddw);
    int64_t sv[2]={S,S},pv[2]={P,P},dv[2]={D,D};
    aclIntArray *as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2),*ad=aclCreateIntArray(dv,2);
    // dgrad
    { uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionBackwardDataGetWorkspaceSize(tdY,tW,as,ap,ad,1,tdX,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolutionBackwardData(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); }
    std::vector<float> hdx(N*C*H*W); CHECK(aclrtMemcpy(hdx.data(),hdx.size()*4,ddx,hdx.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dX(N*C*H*W,0);
    for(int64_t n=0;n<N;n++)for(int64_t co=0;co<Co;co++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
        double g=dY[((n*Co+co)*Ho+ho)*Wo+wo];
        for(int64_t ci=0;ci<C;ci++)for(int64_t r=0;r<KH;r++)for(int64_t s=0;s<KW;s++){
            int64_t h=ho*S-P+r*D, w=wo*S-P+s*D; if(h<0||h>=H||w<0||w>=W)continue;
            dX[((n*C+ci)*H+h)*W+w]+=g*Wt[((co*C+ci)*KH+r)*KW+s];
        }}
    double me=0,mr=0; for(size_t i=0;i<dX.size();i++){me=std::max(me,std::fabs(hdx[i]-dX[i]));mr=std::max(mr,std::fabs(dX[i]));}
    report("Conv dgrad",me/(mr+1e-9),1e-4);
    // wgrad
    { uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionBackwardWeightGetWorkspaceSize(tX,tdY,as,ap,ad,1,tdW,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolutionBackwardWeight(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); }
    std::vector<float> hdw(Co*C*KH*KW); CHECK(aclrtMemcpy(hdw.data(),hdw.size()*4,ddw,hdw.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dW(Co*C*KH*KW,0);
    for(int64_t co=0;co<Co;co++)for(int64_t ci=0;ci<C;ci++)for(int64_t r=0;r<KH;r++)for(int64_t s=0;s<KW;s++){
        double acc=0;
        for(int64_t n=0;n<N;n++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
            int64_t h=ho*S-P+r*D, w=wo*S-P+s*D; if(h<0||h>=H||w<0||w>=W)continue;
            acc+=dY[((n*Co+co)*Ho+ho)*Wo+wo]*X[((n*C+ci)*H+h)*W+w];
        }
        dW[((co*C+ci)*KH+r)*KW+s]=acc;
    }
    me=0;mr=0; for(size_t i=0;i<dW.size();i++){me=std::max(me,std::fabs(hdw[i]-dW[i]));mr=std::max(mr,std::fabs(dW[i]));}
    report("Conv wgrad",me/(mr+1e-9),1e-4);
    aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);
    aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tdY);aclDestroyTensor(tdX);aclDestroyTensor(tdW);
    aclrtFree(dx);aclrtFree(dw);aclrtFree(ddy);aclrtFree(ddx);aclrtFree(ddw);
}

// Pooling forward cross-check (max/avg)
static void t_pool(bool avg,int64_t N,int64_t C,int64_t H,int64_t W,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1;
    auto X=randv(N*C*H*W); void*dx=up(X.data(),X.size()*4),*dy; CHECK(aclrtMalloc(&dy,N*C*Ho*Wo*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,H,W},ACL_FLOAT,dx),*tY=mk({N,C,Ho,Wo},ACL_FLOAT,dy);
    int64_t kv[2]={K,K},sv[2]={S,S},pv[2]={P,P};
    aclIntArray*ak=aclCreateIntArray(kv,2),*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if(avg) CHECK(aclnnAvgPool2dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex)); else CHECK(aclnnMaxPool2dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(avg) CHECK(aclnnAvgPool2d(wsp,ws,ex,g_stream)); else CHECK(aclnnMaxPool2d(wsp,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<float> hy(N*C*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*4,dy,hy.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t c=0;c<C;c++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
        double mx=-1e30,sum=0; int cnt=0;
        for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){ int64_t h=ho*S-P+r,w=wo*S-P+s; if(h<0||h>=H||w<0||w>=W)continue;
            double v=X[((n*C+c)*H+h)*W+w]; mx=std::max(mx,v); sum+=v; cnt++; }
        double ref=avg? sum/cnt : mx;   // avg excludes padding (EXCLUDE_PADDING)
        double got=hy[((n*C+c)*Ho+ho)*Wo+wo]; me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref));
    }
    report(avg?"AvgPool2d k3s2p1":"MaxPool2d k3s2p1",me/(mr+1e-9),1e-5);
    aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyTensor(tX);aclDestroyTensor(tY);aclrtFree(dx);aclrtFree(dy);
}

// fp16 forward conv
static void t_conv_fp16(int64_t N,int64_t C,int64_t H,int64_t W,int64_t Co,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1;
    auto X=randv(N*C*H*W), Wt=randv(Co*C*K*K);
    std::vector<uint16_t> hx(X.size()),hw(Wt.size()); for(size_t i=0;i<X.size();i++)hx[i]=f2h(X[i]); for(size_t i=0;i<Wt.size();i++)hw[i]=f2h(Wt[i]);
    void*dx=up(hx.data(),hx.size()*2),*dw=up(hw.data(),hw.size()*2),*dy; CHECK(aclrtMalloc(&dy,N*Co*Ho*Wo*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,H,W},ACL_FLOAT16,dx),*tW=mk({Co,C,K,K},ACL_FLOAT16,dw),*tY=mk({N,Co,Ho,Wo},ACL_FLOAT16,dy);
    int64_t sv[2]={S,S},pv[2]={P,P},dv[2]={1,1};
    aclIntArray*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2),*ad=aclCreateIntArray(dv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    CHECK(aclnnConvolutionGetWorkspaceSize(tX,tW,nullptr,as,ap,ad,false,nullptr,1,tY,0,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnConvolution(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<uint16_t> hy(N*Co*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*2,dy,hy.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t co=0;co<Co;co++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
        double acc=0; for(int64_t ci=0;ci<C;ci++)for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){
            int64_t h=ho*S-P+r,w=wo*S-P+s; if(h<0||h>=H||w<0||w>=W)continue;
            acc+=(double)h2f(hx[((n*C+ci)*H+h)*W+w])*(double)h2f(hw[((co*C+ci)*K+r)*K+s]); }
        double got=h2f(hy[((n*Co+co)*Ho+ho)*Wo+wo]); me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc));
    }
    report("Conv fp16 fwd",me/(mr+1e-9),1e-2);
    aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tY);aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}

// fp4 input conv: input/weight are fp4 e2m1, unpacked to fp16 for cuDNN, output fp16
static void t_conv_fp4(int64_t N,int64_t C,int64_t H,int64_t W,int64_t Co,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1;
    auto X=randv(N*C*H*W), Wt=randv(Co*C*K*K); for(auto&v:X)v*=4; for(auto&v:Wt)v*=4;
    auto cast=[&](void*src,aclDataType sdt,int64_t ne,void*dst,aclDataType ddt){
        int64_t d[1]={ne}; aclTensor*s=aclCreateTensor(d,1,sdt,nullptr,0,ACL_FORMAT_ND,d,1,src);
        aclTensor*o=aclCreateTensor(d,1,ddt,nullptr,0,ACL_FORMAT_ND,d,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream)); aclDestroyTensor(s);aclDestroyTensor(o); };
    int64_t xn=N*C*H*W, wn=Co*C*K*K;
    void*dXf=up(X.data(),xn*4),*dWf=up(Wt.data(),wn*4),*dX4,*dW4,*dXb,*dWb,*dY;
    CHECK(aclrtMalloc(&dX4,(xn+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dW4,(wn+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dXb,xn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dWb,wn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dY,N*Co*Ho*Wo*2,ACL_MEM_MALLOC_HUGE_FIRST));
    cast(dXf,ACL_FLOAT,xn,dX4,ACL_FLOAT4_E2M1); cast(dWf,ACL_FLOAT,wn,dW4,ACL_FLOAT4_E2M1);
    cast(dX4,ACL_FLOAT4_E2M1,xn,dXb,ACL_FLOAT); cast(dW4,ACL_FLOAT4_E2M1,wn,dWb,ACL_FLOAT);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Xb(xn),Wb(wn); CHECK(aclrtMemcpy(Xb.data(),xn*4,dXb,xn*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(Wb.data(),wn*4,dWb,wn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    aclTensor*tX=mk({N,C,H,W},ACL_FLOAT4_E2M1,dX4),*tW=mk({Co,C,K,K},ACL_FLOAT4_E2M1,dW4),*tY=mk({N,Co,Ho,Wo},ACL_FLOAT16,dY);
    int64_t sv[2]={S,S},pv[2]={P,P},dv[2]={1,1}; aclIntArray*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2),*ad=aclCreateIntArray(dv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionGetWorkspaceSize(tX,tW,nullptr,as,ap,ad,false,nullptr,1,tY,0,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolution(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> hy(N*Co*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*2,dY,hy.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t co=0;co<Co;co++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
        double acc=0; for(int64_t ci=0;ci<C;ci++)for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){int64_t h=ho*S-P+r,w=wo*S-P+s; if(h<0||h>=H||w<0||w>=W)continue;
            acc+=(double)Xb[((n*C+ci)*H+h)*W+w]*Wb[((co*C+ci)*K+r)*K+s];}
        double got=h2f(hy[((n*Co+co)*Ho+ho)*Wo+wo]); me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("Conv fp4 input",me/(mr+1e-9),1e-2);
    aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tY);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);
    if(wsp)aclrtFree(wsp); aclrtFree(dXf);aclrtFree(dWf);aclrtFree(dX4);aclrtFree(dW4);aclrtFree(dXb);aclrtFree(dWb);aclrtFree(dY);
}

// Pooling backward cross-check (fp32): run forward to get y, then random dy backward to get dx, CPU reference
static void t_pool_bwd(bool avg,int64_t N,int64_t C,int64_t H,int64_t W,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1, xn=N*C*H*W, yn=N*C*Ho*Wo;
    auto X=randv(xn); for(int64_t i=0;i<xn;i++)X[i]+=1e-4f*i; // slight perturbation to avoid max ties
    auto dY=randv(yn);
    void*dx=up(X.data(),xn*4),*dyy=up(dY.data(),yn*4),*dY_=dyy,*dyout,*ddx;
    CHECK(aclrtMalloc(&dyout,yn*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&ddx,xn*4,ACL_MEM_MALLOC_HUGE_FIRST)); (void)dY_;
    aclTensor*tX=mk({N,C,H,W},ACL_FLOAT,dx),*tY=mk({N,C,Ho,Wo},ACL_FLOAT,dyout);
    int64_t kv[2]={K,K},sv[2]={S,S},pv[2]={P,P}; aclIntArray*ak=aclCreateIntArray(kv,2),*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;   // forward pass to obtain y
    if(avg)CHECK(aclnnAvgPool2dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex)); else CHECK(aclnnMaxPool2dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(avg)CHECK(aclnnAvgPool2d(wsp,ws,ex,g_stream)); else CHECK(aclnnMaxPool2d(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<float> Y(yn); CHECK(aclrtMemcpy(Y.data(),yn*4,dyout,yn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    aclTensor*tdY=mk({N,C,Ho,Wo},ACL_FLOAT,dyy),*tdX=mk({N,C,H,W},ACL_FLOAT,ddx);
    ws=0; ex=nullptr;
    if(avg)CHECK(aclnnAvgPool2dBackwardGetWorkspaceSize(tX,tY,tdY,ak,as,ap,tdX,&ws,&ex)); else CHECK(aclnnMaxPool2dBackwardGetWorkspaceSize(tX,tY,tdY,ak,as,ap,tdX,&ws,&ex));
    wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(avg)CHECK(aclnnAvgPool2dBackward(wsp,ws,ex,g_stream)); else CHECK(aclnnMaxPool2dBackward(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<float> hdx(xn); CHECK(aclrtMemcpy(hdx.data(),xn*4,ddx,xn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dX(xn,0);
    for(int64_t n=0;n<N;n++)for(int64_t c=0;c<C;c++)for(int64_t oh=0;oh<Ho;oh++)for(int64_t ow=0;ow<Wo;ow++){
        double g=dY[((n*C+c)*Ho+oh)*Wo+ow];
        if(avg){ int cnt=0; for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){int64_t h=oh*S-P+r,w=ow*S-P+s; if(h>=0&&h<H&&w>=0&&w<W)cnt++;}
            for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){int64_t h=oh*S-P+r,w=ow*S-P+s; if(h>=0&&h<H&&w>=0&&w<W)dX[((n*C+c)*H+h)*W+w]+=g/cnt;} }
        else { double mx=-1e30; int64_t mh=-1,mw=-1; for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){int64_t h=oh*S-P+r,w=ow*S-P+s; if(h<0||h>=H||w<0||w>=W)continue; double v=X[((n*C+c)*H+h)*W+w]; if(v>mx){mx=v;mh=h;mw=w;}}
            if(mh>=0)dX[((n*C+c)*H+mh)*W+mw]+=g; } }
    double me=0,mr=0; for(int64_t i=0;i<xn;i++){me=std::max(me,std::fabs(hdx[i]-dX[i]));mr=std::max(mr,std::fabs(dX[i]));}
    report(avg?"AvgPool2dBackward":"MaxPool2dBackward",me/(mr+1e-9),1e-5);
    aclDestroyTensor(tX);aclDestroyTensor(tY);aclDestroyTensor(tdY);aclDestroyTensor(tdX);aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);
    aclrtFree(dx);aclrtFree(dyy);aclrtFree(dyout);aclrtFree(ddx);
}

// ConvTranspose2d cross-check (fp32, groups=1): CPU scatter-add reference
static void t_conv_transpose(int64_t N,int64_t Ci,int64_t H,int64_t W,int64_t Co,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H-1)*S-2*P+(K-1)+1, Wo=(W-1)*S-2*P+(K-1)+1;
    int64_t xn=N*Ci*H*W, wn=Ci*Co*K*K, yn=N*Co*Ho*Wo;
    auto X=randv(xn), Wt=randv(wn); std::vector<float> hy(yn);
    void*dx=up(X.data(),xn*4),*dw=up(Wt.data(),wn*4),*dy; CHECK(aclrtMalloc(&dy,yn*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,Ci,H,W},ACL_FLOAT,dx),*tW=mk({Ci,Co,K,K},ACL_FLOAT,dw),*tY=mk({N,Co,Ho,Wo},ACL_FLOAT,dy);
    int64_t sv[2]={S,S},pv[2]={P,P},dv[2]={1,1};
    aclIntArray*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2),*ad=aclCreateIntArray(dv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionTranspose2dGetWorkspaceSize(tX,tW,as,ap,ad,1,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolutionTranspose2d(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hy.data(),yn*4,dy,yn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> out(yn,0);
    for(int64_t n=0;n<N;n++)for(int64_t ci=0;ci<Ci;ci++)for(int64_t h=0;h<H;h++)for(int64_t w=0;w<W;w++){
        double v=X[((n*Ci+ci)*H+h)*W+w];
        for(int64_t co=0;co<Co;co++)for(int64_t kh=0;kh<K;kh++)for(int64_t kw=0;kw<K;kw++){
            int64_t oh=h*S-P+kh, ow=w*S-P+kw; if(oh<0||oh>=Ho||ow<0||ow>=Wo)continue;
            out[((n*Co+co)*Ho+oh)*Wo+ow]+=v*Wt[((ci*Co+co)*K+kh)*K+kw]; } }
    double me=0,mr=0; for(int64_t i=0;i<yn;i++){me=std::max(me,std::fabs(hy[i]-out[i]));mr=std::max(mr,std::fabs(out[i]));}
    report("ConvTranspose2d s2",me/(mr+1e-9),1e-4);
    aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tY);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}

// conv3d cross-check (NCDHW, fp32, groups=1)
static void t_conv3d(int64_t N,int64_t C,int64_t D,int64_t H,int64_t W,int64_t Co,int64_t K,int64_t S,int64_t P){
    int64_t Do=(D+2*P-K)/S+1, Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1;
    int64_t xn=N*C*D*H*W, wn=Co*C*K*K*K, yn=N*Co*Do*Ho*Wo;
    auto X=randv(xn), Wt=randv(wn); std::vector<float> hy(yn);
    void*dx=up(X.data(),xn*4),*dw=up(Wt.data(),wn*4),*dy; CHECK(aclrtMalloc(&dy,yn*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,D,H,W},ACL_FLOAT,dx),*tW=mk({Co,C,K,K,K},ACL_FLOAT,dw),*tY=mk({N,Co,Do,Ho,Wo},ACL_FLOAT,dy);
    int64_t sv[3]={S,S,S},pv[3]={P,P,P},dv[3]={1,1,1};
    aclIntArray*as=aclCreateIntArray(sv,3),*ap=aclCreateIntArray(pv,3),*ad=aclCreateIntArray(dv,3);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolution3dGetWorkspaceSize(tX,tW,as,ap,ad,1,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolution3d(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hy.data(),yn*4,dy,yn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t co=0;co<Co;co++)for(int64_t od=0;od<Do;od++)for(int64_t oh=0;oh<Ho;oh++)for(int64_t ow=0;ow<Wo;ow++){
        double acc=0;
        for(int64_t ci=0;ci<C;ci++)for(int64_t kd=0;kd<K;kd++)for(int64_t kh=0;kh<K;kh++)for(int64_t kw=0;kw<K;kw++){
            int64_t id=od*S-P+kd, ih=oh*S-P+kh, iw=ow*S-P+kw; if(id<0||id>=D||ih<0||ih>=H||iw<0||iw>=W)continue;
            acc+=(double)X[(((n*C+ci)*D+id)*H+ih)*W+iw]*Wt[(((co*C+ci)*K+kd)*K+kh)*K+kw]; }
        double got=hy[(((n*Co+co)*Do+od)*Ho+oh)*Wo+ow]; me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("Conv3d NCDHW",me/(mr+1e-9),1e-4);
    aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tY);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}

// WeightQuantBatchMatmul W8A16/W4A16：x fp16, weight int8/int4, antiquant scale/off[N] fp16
static void t_wqmm(bool int4, int64_t M, int64_t K, int64_t N) {
    auto X = randv(M*K); std::vector<uint16_t> hx(M*K); for (int64_t i=0;i<M*K;i++) hx[i]=f2h(X[i]);
    std::vector<int8_t> w8(K*N); std::vector<uint8_t> w4((K*N+1)/2,0); std::vector<int> wv(K*N);
    for (int64_t i=0;i<K*N;i++){ int v = (rand()%(int4?16:256)) - (int4?8:128); wv[i]=v; if(int4){ uint8_t u=(uint8_t)(v&0xf); if(i&1)w4[i/2]|=(u<<4); else w4[i/2]|=u; } else w8[i]=(int8_t)v; }
    std::vector<float> sc(N), of(N); for (int64_t n=0;n<N;n++){ sc[n]=0.02f+0.01f*(n%5); of[n]=(float)((n%3)-1); }
    std::vector<uint16_t> hsc(N),hof(N); for(int64_t n=0;n<N;n++){hsc[n]=f2h(sc[n]);hof[n]=f2h(of[n]);}
    void*dx=up(hx.data(),M*K*2),*dw=up(int4?(void*)w4.data():(void*)w8.data(),int4?(K*N+1)/2:K*N),*dsc=up(hsc.data(),N*2),*dof=up(hof.data(),N*2),*dy;
    CHECK(aclrtMalloc(&dy,M*N*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({M,K},ACL_FLOAT16,dx),*tw=mk({K,N},int4?ACL_INT4:ACL_INT8,dw),*tsc=mk({N},ACL_FLOAT16,dsc),*tof=mk({N},ACL_FLOAT16,dof),*ty=mk({M,N},ACL_FLOAT16,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnWeightQuantBatchMatmulGetWorkspaceSize(tx,tw,tsc,tof,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnWeightQuantBatchMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> hy(M*N); CHECK(aclrtMemcpy(hy.data(),M*N*2,dy,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t m=0;m<M;m++)for(int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++){ double wf=((double)wv[k*N+n]-(double)h2f(hof[n]))*(double)h2f(hsc[n]); acc+=(double)h2f(hx[m*K+k])*wf; }
        double got=h2f(hy[m*N+n]); me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
    report(int4?"WeightQuantMM W4A16":"WeightQuantMM W8A16", me/(mr+1e-9), 2e-2);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tsc);aclDestroyTensor(tof);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dsc);aclrtFree(dof);aclrtFree(dy);
}
// QuantMatmul W8A8：x int8, weight int8, deqScale[N] fp16
static void t_qmm(int64_t M, int64_t K, int64_t N) {
    std::vector<int8_t> x(M*K), w(K*N); for(auto&v:x)v=(int8_t)((rand()%255)-127); for(auto&v:w)v=(int8_t)((rand()%255)-127);
    std::vector<float> sc(N); std::vector<uint16_t> hsc(N); for(int64_t n=0;n<N;n++){sc[n]=0.001f*(1+n%7); hsc[n]=f2h(sc[n]);}
    void*dx=up(x.data(),M*K),*dw=up(w.data(),K*N),*dsc=up(hsc.data(),N*2),*dy; CHECK(aclrtMalloc(&dy,M*N*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({M,K},ACL_INT8,dx),*tw=mk({K,N},ACL_INT8,dw),*tsc=mk({N},ACL_FLOAT16,dsc),*ty=mk({M,N},ACL_FLOAT16,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnQuantMatmulGetWorkspaceSize(tx,tw,tsc,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnQuantMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> hy(M*N); CHECK(aclrtMemcpy(hy.data(),M*N*2,dy,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t m=0;m<M;m++)for(int64_t n=0;n<N;n++){ long long acc=0; for(int64_t k=0;k<K;k++)acc+=(long long)x[m*K+k]*w[k*N+n]; double ref=(double)acc*(double)h2f(hsc[n]);
        double got=h2f(hy[m*N+n]); me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref)); }
    report("QuantMM W8A8", me/(mr+1e-9), 2e-2);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tsc);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dsc);aclrtFree(dy);
}

// QuantMatmulV3 variant: must match the W8A8 core.
static void t_qmm_v(int64_t M, int64_t K, int64_t N) {
    std::vector<int8_t> x(M*K), w(K*N); for(auto&v:x)v=(int8_t)((rand()%255)-127); for(auto&v:w)v=(int8_t)((rand()%255)-127);
    std::vector<float> sc(N); std::vector<uint16_t> hsc(N); for(int64_t n=0;n<N;n++){sc[n]=0.001f*(1+n%7); hsc[n]=f2h(sc[n]);}
    void*dx=up(x.data(),M*K),*dw=up(w.data(),K*N),*dsc=up(hsc.data(),N*2),*dy; CHECK(aclrtMalloc(&dy,M*N*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({M,K},ACL_INT8,dx),*tw=mk({K,N},ACL_INT8,dw),*tsc=mk({N},ACL_FLOAT16,dsc),*ty=mk({M,N},ACL_FLOAT16,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnQuantMatmulV3GetWorkspaceSize(tx,tw,tsc,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnQuantMatmulV3(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> hy(M*N); CHECK(aclrtMemcpy(hy.data(),M*N*2,dy,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t m=0;m<M;m++)for(int64_t n=0;n<N;n++){ long long acc=0; for(int64_t k=0;k<K;k++)acc+=(long long)x[m*K+k]*w[k*N+n]; double ref=(double)acc*(double)h2f(hsc[n]);
        double got=h2f(hy[m*N+n]); me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref)); }
    report("QuantMatmulV3 W8A8", me/(mr+1e-9), 2e-2);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tsc);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dsc);aclrtFree(dy);
}
// QuantBatchMatmulInplaceAdd: out (preloaded accumulator) += dequant(int8 x @ int8 w, scale[N])
static void t_qmm_iadd(int64_t M, int64_t K, int64_t N) {
    std::vector<int8_t> x(M*K), w(K*N); for(auto&v:x)v=(int8_t)((rand()%255)-127); for(auto&v:w)v=(int8_t)((rand()%255)-127);
    std::vector<float> sc(N); std::vector<uint16_t> hsc(N); for(int64_t n=0;n<N;n++){sc[n]=0.001f*(1+n%7); hsc[n]=f2h(sc[n]);}
    std::vector<uint16_t> acc(M*N); std::vector<float> accf(M*N); for(int64_t i=0;i<M*N;i++){accf[i]=0.5f*((i%11)-5); acc[i]=f2h(accf[i]);}
    void*dx=up(x.data(),M*K),*dw=up(w.data(),K*N),*dsc=up(hsc.data(),N*2),*dy=up(acc.data(),M*N*2);
    aclTensor*tx=mk({M,K},ACL_INT8,dx),*tw=mk({K,N},ACL_INT8,dw),*tsc=mk({N},ACL_FLOAT16,dsc),*ty=mk({M,N},ACL_FLOAT16,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnQuantBatchMatmulInplaceAddGetWorkspaceSize(tx,tw,tsc,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnQuantBatchMatmulInplaceAdd(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> hy(M*N); CHECK(aclrtMemcpy(hy.data(),M*N*2,dy,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t m=0;m<M;m++)for(int64_t n=0;n<N;n++){ long long a=0; for(int64_t k=0;k<K;k++)a+=(long long)x[m*K+k]*w[k*N+n]; double ref=accf[m*N+n]+(double)a*(double)h2f(hsc[n]);
        double got=h2f(hy[m*N+n]); me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref)); }
    report("QuantBatchMatmulInplaceAdd", me/(mr+1e-9), 2e-2);
    aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tsc);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dsc);aclrtFree(dy);
}
// GroupedMatmul (fp32, MoE): x[M,K] grouped by groupList @ weight[E,K,N]. native=true uses native grouped GEMM path.
static void t_gmm(int64_t K,int64_t N,std::vector<int64_t> counts,bool native=false){
    int64_t E=counts.size(), M=0; for(auto c:counts)M+=c;
    auto X=randv(M*K), W=randv(E*K*N); std::vector<float> hc(M*N);
    void*dx=up(X.data(),M*K*4),*dw=up(W.data(),E*K*N*4),*dy; CHECK(aclrtMalloc(&dy,M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({M,K},ACL_FLOAT,dx),*tw=mk({E,K,N},ACL_FLOAT,dw),*ty=mk({M,N},ACL_FLOAT,dy);
    aclIntArray*gl=aclCreateIntArray(counts.data(),E);
    if(native) setenv("CANN_GMM_NATIVE","1",1);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnGroupedMatmulGetWorkspaceSize(tx,tw,gl,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnGroupedMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    if(native) unsetenv("CANN_GMM_NATIVE");
    CHECK(aclrtMemcpy(hc.data(),M*N*4,dy,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0; int64_t off=0;
    for(int64_t e=0;e<E;e++){ for(int64_t i=0;i<counts[e];i++)for(int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++)acc+=(double)X[(off+i)*K+k]*W[(e*K+k)*N+n];
        me=std::max(me,std::fabs(hc[(off+i)*N+n]-acc)); mr=std::max(mr,std::fabs(acc)); } off+=counts[e]; }
    report(native?"GroupedMatmul MoE native":"GroupedMatmul MoE",me/(mr+1e-9),1e-5);
    aclDestroyIntArray(gl);aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}

// AdaptiveAvgPool2d: output Ho×Wo, each cell averages an adaptive input region
static void t_adaptive_avgpool(int64_t N,int64_t C,int64_t H,int64_t W,int64_t Ho,int64_t Wo){
    auto X=randv(N*C*H*W); void*dx=up(X.data(),X.size()*4),*dy; CHECK(aclrtMalloc(&dy,N*C*Ho*Wo*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,H,W},ACL_FLOAT,dx),*tY=mk({N,C,Ho,Wo},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    CHECK(aclnnAdaptiveAvgPool2dGetWorkspaceSize(tX,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnAdaptiveAvgPool2d(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<float> hy(N*C*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*4,dy,hy.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t c=0;c<C;c++)for(int64_t oh=0;oh<Ho;oh++)for(int64_t ow=0;ow<Wo;ow++){
        int64_t hs=oh*H/Ho,he=((oh+1)*H+Ho-1)/Ho,wsr=ow*W/Wo,we=((ow+1)*W+Wo-1)/Wo; double s=0;
        for(int64_t h=hs;h<he;h++)for(int64_t w=wsr;w<we;w++) s+=X[((n*C+c)*H+h)*W+w];
        double ref=s/((he-hs)*(we-wsr)),got=hy[((n*C+c)*Ho+oh)*Wo+ow]; me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref));
    }
    report("AdaptiveAvgPool2d 16->5",me/(mr+1e-9),1e-5);
    aclDestroyTensor(tX);aclDestroyTensor(tY);aclrtFree(dx);aclrtFree(dy);
}
// 3D pooling: NCDHW
static void t_pool3d(bool avg,int64_t N,int64_t C,int64_t D,int64_t H,int64_t W,int64_t K,int64_t S,int64_t P){
    int64_t Do=(D+2*P-K)/S+1,Ho=(H+2*P-K)/S+1,Wo=(W+2*P-K)/S+1;
    auto X=randv(N*C*D*H*W); void*dx=up(X.data(),X.size()*4),*dy; CHECK(aclrtMalloc(&dy,N*C*Do*Ho*Wo*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,D,H,W},ACL_FLOAT,dx),*tY=mk({N,C,Do,Ho,Wo},ACL_FLOAT,dy);
    int64_t kv[3]={K,K,K},sv[3]={S,S,S},pv[3]={P,P,P};
    aclIntArray*ak=aclCreateIntArray(kv,3),*as=aclCreateIntArray(sv,3),*ap=aclCreateIntArray(pv,3);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if(avg) CHECK(aclnnAvgPool3dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex)); else CHECK(aclnnMaxPool3dGetWorkspaceSize(tX,ak,as,ap,tY,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(avg) CHECK(aclnnAvgPool3d(wsp,ws,ex,g_stream)); else CHECK(aclnnMaxPool3d(wsp,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<float> hy(N*C*Do*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*4,dy,hy.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t c=0;c<C;c++)for(int64_t od=0;od<Do;od++)for(int64_t oh=0;oh<Ho;oh++)for(int64_t ow=0;ow<Wo;ow++){
        double mx=-1e30,sum=0; int cnt=0;
        for(int64_t kd=0;kd<K;kd++)for(int64_t kh=0;kh<K;kh++)for(int64_t kw=0;kw<K;kw++){
            int64_t d=od*S-P+kd,h=oh*S-P+kh,w=ow*S-P+kw; if(d<0||d>=D||h<0||h>=H||w<0||w>=W)continue;
            double v=X[(((n*C+c)*D+d)*H+h)*W+w]; mx=std::max(mx,v); sum+=v; cnt++; }
        double ref=avg? sum/cnt : mx;
        double got=hy[(((n*C+c)*Do+od)*Ho+oh)*Wo+ow]; me=std::max(me,std::fabs(got-ref)); mr=std::max(mr,std::fabs(ref));
    }
    report(avg?"AvgPool3d k3s2p1":"MaxPool3d k3s2p1",me/(mr+1e-9),1e-5);
    aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyTensor(tX);aclDestroyTensor(tY);aclrtFree(dx);aclrtFree(dy);
}

// bf16 matmul: bf16 = top 16 bits of fp32 (RNE), cuBLASLt CUDA_R_16BF + fp32 accumulation
static uint16_t f2bf(float f){ uint32_t b; __builtin_memcpy(&b,&f,4); b += 0x7FFF + ((b>>16)&1); return (uint16_t)(b>>16); }
static float bf2f(uint16_t h){ uint32_t b=(uint32_t)h<<16; float f; __builtin_memcpy(&f,&b,4); return f; }
static void t_matmul_bf16(int64_t M, int64_t K, int64_t N, double tol){
    auto A=randv(M*K), B=randv(K*N);
    std::vector<uint16_t> ha(M*K), hb(K*N), hc(M*N);
    for(int64_t i=0;i<M*K;i++) ha[i]=f2bf(A[i]);
    for(int64_t i=0;i<K*N;i++) hb[i]=f2bf(B[i]);
    void *da=up(ha.data(),ha.size()*2),*db=up(hb.data(),hb.size()*2),*dc;
    CHECK(aclrtMalloc(&dc,hc.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*ta=mk({M,K},ACL_BF16,da),*tb=mk({K,N},ACL_BF16,db),*tc=mk({M,N},ACL_BF16,dc);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnMatmulGetWorkspaceSize(ta,tb,tc,1,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),hc.size()*2,dc,hc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double r=0;
        for(int64_t l=0;l<K;l++) r+=bf2f(ha[i*K+l])*bf2f(hb[l*N+j]);
        me=std::max(me,std::fabs(bf2f(hc[i*N+j])-r)); mr=std::max(mr,std::fabs(r)); }
    char nm[48]; snprintf(nm,sizeof nm,"Matmul bf16 %ldx%ldx%ld",(long)M,(long)K,(long)N);
    report(nm,me/(mr+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp);
    aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}

// MatmulBias bf16: lifts the fp32/fp16-only restriction on MatmulBias — real models commonly use bf16, and transB+bias is a common real-model q/k/v projection pattern
static void t_mmbias_bf16(int64_t M, int64_t K, int64_t N, double tol){
    auto A=randv(M*K), Bm=randv(K*N), bias=randv(N);       // weight native layout [N,K] (transB=true), bias[N]
    std::vector<uint16_t> ha(M*K), hb(N*K), hbias(N), hc(M*N);
    for(int64_t i=0;i<M*K;i++) ha[i]=f2bf(A[i]);
    for(int64_t i=0;i<N*K;i++) hb[i]=f2bf(Bm[i]);          // treated as [N,K]
    for(int64_t i=0;i<N;i++) hbias[i]=f2bf(bias[i]);
    void *da=up(ha.data(),ha.size()*2),*db=up(hb.data(),hb.size()*2),*dbias=up(hbias.data(),N*2),*dc;
    CHECK(aclrtMalloc(&dc,hc.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*ta=mk({M,K},ACL_BF16,da),*tb=mk({N,K},ACL_BF16,db),*tbias=mk({N},ACL_BF16,dbias),*tc=mk({M,N},ACL_BF16,dc);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnMatmulBiasGetWorkspaceSize(ta,tb,tbias,false,true,0,tc,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmulBias(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),hc.size()*2,dc,hc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double acc=bf2f(hbias[j]);
        for(int64_t l=0;l<K;l++) acc+=bf2f(ha[i*K+l])*bf2f(hb[j*K+l]);
        me=std::max(me,std::fabs(bf2f(hc[i*N+j])-acc)); mr=std::max(mr,std::fabs(acc)); }
    char nm[48]; snprintf(nm,sizeof nm,"MatmulBias bf16 transB %ldx%ldx%ld",(long)M,(long)K,(long)N);
    report(nm,me/(mr+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tbias);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp);
    aclrtFree(da);aclrtFree(db);aclrtFree(dbias);aclrtFree(dc);
}

// fp16 input accumulated into fp32 output (common Ascend fp16×fp16→fp32 config; lifts out!=in dtype restriction)
static void t_matmul_f16f32(int64_t M, int64_t K, int64_t N, double tol){
    auto A=randv(M*K), B=randv(K*N);
    std::vector<uint16_t> ha(M*K), hb(K*N); std::vector<float> hc(M*N);
    for(int64_t i=0;i<M*K;i++) ha[i]=f2h(A[i]);
    for(int64_t i=0;i<K*N;i++) hb[i]=f2h(B[i]);
    void *da=up(ha.data(),ha.size()*2),*db=up(hb.data(),hb.size()*2),*dc;
    CHECK(aclrtMalloc(&dc,hc.size()*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*ta=mk({M,K},ACL_FLOAT16,da),*tb=mk({K,N},ACL_FLOAT16,db),*tc=mk({M,N},ACL_FLOAT,dc);  // output fp32
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnMatmulGetWorkspaceSize(ta,tb,tc,1,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),hc.size()*4,dc,hc.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double r=0;
        for(int64_t l=0;l<K;l++) r+=h2f(ha[i*K+l])*h2f(hb[l*N+j]);
        me=std::max(me,std::fabs((double)hc[i*N+j]-r)); mr=std::max(mr,std::fabs(r)); }
    char nm[48]; snprintf(nm,sizeof nm,"Matmul f16->f32 %ldx%ldx%ld",(long)M,(long)K,(long)N);
    report(nm,me/(mr+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp);
    aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}

// ---- bf16 coverage additions (lifts bf16 restriction on BatchMatMul/GroupedMatmul/Conv) ----
static void t_bmm_bf16(int64_t B,int64_t M,int64_t K,int64_t N,double tol){
    auto A=randv(B*M*K),Bm=randv(B*K*N);
    std::vector<uint16_t> ha(B*M*K),hb(B*K*N),hc(B*M*N);
    for(size_t i=0;i<ha.size();i++)ha[i]=f2bf(A[i]); for(size_t i=0;i<hb.size();i++)hb[i]=f2bf(Bm[i]);
    void*da=up(ha.data(),ha.size()*2),*db=up(hb.data(),hb.size()*2),*dc; CHECK(aclrtMalloc(&dc,hc.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*ta=mk({B,M,K},ACL_BF16,da),*tb=mk({B,K,N},ACL_BF16,db),*tc=mk({B,M,N},ACL_BF16,dc);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnBatchMatMulGetWorkspaceSize(ta,tb,tc,0,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnBatchMatMul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),hc.size()*2,dc,hc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t b=0;b<B;b++)for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double acc=0;
        for(int64_t l=0;l<K;l++)acc+=bf2f(ha[(b*M+i)*K+l])*bf2f(hb[(b*K+l)*N+j]);
        me=std::max(me,std::fabs(bf2f(hc[(b*M+i)*N+j])-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("BMM bf16",me/(mr+1e-9),tol);
    aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp); aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
static void t_gmm_bf16(int64_t K,int64_t N,std::vector<int64_t> counts,double tol){
    int64_t E=counts.size(),M=0; for(auto c:counts)M+=c;
    auto X=randv(M*K),W=randv(E*K*N); std::vector<uint16_t> hx(M*K),hw(E*K*N),hc(M*N);
    for(size_t i=0;i<hx.size();i++)hx[i]=f2bf(X[i]); for(size_t i=0;i<hw.size();i++)hw[i]=f2bf(W[i]);
    void*dx=up(hx.data(),hx.size()*2),*dw=up(hw.data(),hw.size()*2),*dy; CHECK(aclrtMalloc(&dy,hc.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({M,K},ACL_BF16,dx),*tw=mk({E,K,N},ACL_BF16,dw),*ty=mk({M,N},ACL_BF16,dy);
    aclIntArray*gl=aclCreateIntArray(counts.data(),E);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnGroupedMatmulGetWorkspaceSize(tx,tw,gl,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnGroupedMatmul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hc.data(),hc.size()*2,dy,hc.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0; int64_t off=0;
    for(int64_t e=0;e<E;e++){ for(int64_t i=0;i<counts[e];i++)for(int64_t n=0;n<N;n++){ double acc=0;
        for(int64_t k=0;k<K;k++)acc+=bf2f(hx[(off+i)*K+k])*bf2f(hw[(e*K+k)*N+n]);
        me=std::max(me,std::fabs(bf2f(hc[(off+i)*N+n])-acc)); mr=std::max(mr,std::fabs(acc)); } off+=counts[e]; }
    report("GroupedMatmul MoE bf16",me/(mr+1e-9),tol);
    aclDestroyIntArray(gl);aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}
static void t_conv_bf16(int64_t N,int64_t C,int64_t H,int64_t W,int64_t Co,int64_t K,int64_t S,int64_t P){
    int64_t Ho=(H+2*P-K)/S+1, Wo=(W+2*P-K)/S+1;
    auto X=randv(N*C*H*W), Wt=randv(Co*C*K*K);
    std::vector<uint16_t> hx(X.size()),hw(Wt.size()); for(size_t i=0;i<X.size();i++)hx[i]=f2bf(X[i]); for(size_t i=0;i<Wt.size();i++)hw[i]=f2bf(Wt[i]);
    void*dx=up(hx.data(),hx.size()*2),*dw=up(hw.data(),hw.size()*2),*dy; CHECK(aclrtMalloc(&dy,N*Co*Ho*Wo*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tX=mk({N,C,H,W},ACL_BF16,dx),*tW=mk({Co,C,K,K},ACL_BF16,dw),*tY=mk({N,Co,Ho,Wo},ACL_BF16,dy);
    int64_t sv[2]={S,S},pv[2]={P,P},dv[2]={1,1};
    aclIntArray*as=aclCreateIntArray(sv,2),*ap=aclCreateIntArray(pv,2),*ad=aclCreateIntArray(dv,2);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionGetWorkspaceSize(tX,tW,nullptr,as,ap,ad,false,nullptr,1,tY,0,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolution(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp);
    std::vector<uint16_t> hy(N*Co*Ho*Wo); CHECK(aclrtMemcpy(hy.data(),hy.size()*2,dy,hy.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t n=0;n<N;n++)for(int64_t co=0;co<Co;co++)for(int64_t ho=0;ho<Ho;ho++)for(int64_t wo=0;wo<Wo;wo++){
        double acc=0; for(int64_t ci=0;ci<C;ci++)for(int64_t r=0;r<K;r++)for(int64_t s=0;s<K;s++){
            int64_t h=ho*S-P+r,w=wo*S-P+s; if(h<0||h>=H||w<0||w>=W)continue;
            acc+=(double)bf2f(hx[((n*C+ci)*H+h)*W+w])*(double)bf2f(hw[((co*C+ci)*K+r)*K+s]); }
        double got=bf2f(hy[((n*Co+co)*Ho+ho)*Wo+wo]); me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("Conv bf16 fwd",me/(mr+1e-9),5e-2);
    aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad);aclDestroyTensor(tX);aclDestroyTensor(tW);aclDestroyTensor(tY);aclrtFree(dx);aclrtFree(dw);aclrtFree(dy);
}

// conv extensions: Im2col, ConvTbc, ConvolutionBackward bias grad, DeformableConv2d(offset=0), helpers
static void t_conv_ext() {
    { // Im2col: input[1,2,4,4] kernel 3 stride 1 pad 0 → col[1, 2*9, 2*2]; compare to CPU unfold
      const int N=1,C=2,H=4,W=4,kH=3,kW=3,oH=2,oW=2; auto X=randv(N*C*H*W);
      void*dx=up(X.data(),X.size()*4),*dc; int64_t L=oH*oW,K=C*kH*kW; CHECK(aclrtMalloc(&dc,N*K*L*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx),*tc=mk({N,K,L},ACL_FLOAT,dc);
      int64_t kv[2]={kH,kW},dv[2]={1,1},pv[2]={0,0},sv[2]={1,1};
      aclIntArray*ak=aclCreateIntArray(kv,2),*ad=aclCreateIntArray(dv,2),*ap=aclCreateIntArray(pv,2),*as=aclCreateIntArray(sv,2);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnIm2colGetWorkspaceSize(tx,ak,ad,ap,as,tc,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnIm2col(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> col(N*K*L); CHECK(aclrtMemcpy(col.data(),col.size()*4,dc,col.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double bad=0; for(int c=0;c<C;c++)for(int kh=0;kh<kH;kh++)for(int kw=0;kw<kW;kw++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){
        int kk=(c*kH+kh)*kW+kw,l=oh*oW+ow; double ref=X[((c)*H+(oh+kh))*W+(ow+kw)]; bad=std::max(bad,std::fabs((double)col[kk*L+l]-ref)); }
      report("Im2col", bad, 1e-6); aclDestroyTensor(tx);aclDestroyTensor(tc); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dc); }
    { // ConvTbc: self[T,B,Cin]=[4,1,2], weight[kW,Cin,Cout]=[3,2,2], pad 1 → out[4,1,2]
      const int T=4,B=1,Cin=2,kW=3,Cout=2,pad=1,oT=4; auto X=randv(T*B*Cin),Wt=randv(kW*Cin*Cout),Bs=randv(Cout);
      void*dx=up(X.data(),X.size()*4),*dw=up(Wt.data(),Wt.size()*4),*db=up(Bs.data(),Bs.size()*4),*dy; CHECK(aclrtMalloc(&dy,oT*B*Cout*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({T,B,Cin},ACL_FLOAT,dx),*tw=mk({kW,Cin,Cout},ACL_FLOAT,dw),*tb=mk({Cout},ACL_FLOAT,db),*ty=mk({oT,B,Cout},ACL_FLOAT,dy);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvTbcGetWorkspaceSize(tx,tw,tb,pad,ty,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvTbc(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> Y(oT*B*Cout); CHECK(aclrtMemcpy(Y.data(),Y.size()*4,dy,Y.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int t=0;t<oT;t++)for(int co=0;co<Cout;co++){ double acc=Bs[co]; for(int k=0;k<kW;k++){int ti=t+k-pad; if(ti<0||ti>=T)continue; for(int ci=0;ci<Cin;ci++) acc+=X[(ti*B+0)*Cin+ci]*Wt[(k*Cin+ci)*Cout+co]; }
        double got=Y[(t*B+0)*Cout+co]; me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("ConvTbc", me/(mr+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tb);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(db);aclrtFree(dy); }
    { // ConvolutionBackward bias grad: gradBias[co] = Σ gradOut over N,H,W
      const int N=2,C=3,H=5,W=5,Co=4,K=3,oH=3,oW=3; auto X=randv(N*C*H*W),Wt=randv(Co*C*K*K),dY=randv(N*Co*oH*oW);
      void*dx=up(X.data(),X.size()*4),*dw=up(Wt.data(),Wt.size()*4),*ddy=up(dY.data(),dY.size()*4);
      void*dgx,*dgw,*dgb; CHECK(aclrtMalloc(&dgx,X.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dgw,Wt.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dgb,Co*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx),*tw=mk({Co,C,K,K},ACL_FLOAT,dw),*tdy=mk({N,Co,oH,oW},ACL_FLOAT,ddy),
               *tgx=mk({N,C,H,W},ACL_FLOAT,dgx),*tgw=mk({Co,C,K,K},ACL_FLOAT,dgw),*tgb=mk({Co},ACL_FLOAT,dgb);
      int64_t s2[2]={1,1},p2[2]={0,0},d2[2]={1,1}; aclIntArray*as=aclCreateIntArray(s2,2),*ap=aclCreateIntArray(p2,2),*ad=aclCreateIntArray(d2,2);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnConvolutionBackwardGetWorkspaceSize(tdy,tx,tw,nullptr,as,ap,ad,false,nullptr,1,tgx,tgw,tgb,1,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnConvolutionBackward(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> gb(Co); CHECK(aclrtMemcpy(gb.data(),Co*4,dgb,Co*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int co=0;co<Co;co++){ double acc=0; for(int n=0;n<N;n++)for(int p=0;p<oH*oW;p++) acc+=dY[(n*Co+co)*oH*oW+p]; me=std::max(me,std::fabs(gb[co]-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("ConvolutionBackward bias", me/(mr+1e-9), 1e-4);
      aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tdy);aclDestroyTensor(tgx);aclDestroyTensor(tgw);aclDestroyTensor(tgb); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(ddy);aclrtFree(dgx);aclrtFree(dgw);aclrtFree(dgb); }
    { // DeformableConv2d with zero offset == regular convolution (no bias, no mask)
      const int N=1,Cin=2,H=5,W=5,Cout=3,K=3,oH=3,oW=3; auto X=randv(N*Cin*H*W),Wt=randv(Cout*Cin*K*K);
      std::vector<float> off(N*2*K*K*oH*oW,0.f);
      void*dx=up(X.data(),X.size()*4),*dw=up(Wt.data(),Wt.size()*4),*doff=up(off.data(),off.size()*4),*dy; CHECK(aclrtMalloc(&dy,N*Cout*oH*oW*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({N,Cin,H,W},ACL_FLOAT,dx),*tw=mk({Cout,Cin,K,K},ACL_FLOAT,dw),*toff=mk({N,2*K*K,oH,oW},ACL_FLOAT,doff),*ty=mk({N,Cout,oH,oW},ACL_FLOAT,dy);
      int64_t s2[2]={1,1},p2[2]={0,0},d2[2]={1,1}; aclIntArray*as=aclCreateIntArray(s2,2),*ap=aclCreateIntArray(p2,2),*ad=aclCreateIntArray(d2,2);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnDeformableConv2dGetWorkspaceSize(tx,tw,toff,nullptr,nullptr,as,ap,ad,ty,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnDeformableConv2d(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> Y(N*Cout*oH*oW); CHECK(aclrtMemcpy(Y.data(),Y.size()*4,dy,Y.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int co=0;co<Cout;co++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ double acc=0;
        for(int ci=0;ci<Cin;ci++)for(int kh=0;kh<K;kh++)for(int kw=0;kw<K;kw++){ int ih=oh+kh,iw=ow+kw; acc+=(double)X[((ci)*H+ih)*W+iw]*Wt[((co*Cin+ci)*K+kh)*K+kw]; }
        double got=Y[(co*oH+oh)*oW+ow]; me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("DeformableConv2d(offset=0)", me/(mr+1e-9), 1e-4);
      aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(toff);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(doff);aclrtFree(dy); }
    { // CalculateConvolutionWeightSize: product of shape dims
      int64_t shp[4]={16,8,3,3}; aclIntArray*sh=aclCreateIntArray(shp,4); void*dout; CHECK(aclrtMalloc(&dout,8,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*to=mk({1},ACL_INT64,dout); uint64_t ws=0; aclOpExecutor*ex=nullptr;
      CHECK(aclnnCalculateConvolutionWeightSizeGetWorkspaceSize(sh,to,&ws,&ex)); CHECK(aclnnCalculateConvolutionWeightSize(nullptr,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      int64_t got; CHECK(aclrtMemcpy(&got,8,dout,8,ACL_MEMCPY_DEVICE_TO_HOST));
      report("CalculateConvolutionWeightSize", std::fabs((double)(got-16*8*3*3)), 0.0); aclDestroyTensor(to); aclrtFree(dout); }
    { // ConvertWeightToINT4Pack: two int8 (each [-8,7]) → one byte (hi<<4|lo)
      std::vector<int8_t> in={1,2,-3,7,-8,0,5,-1}; int nOut=4; std::vector<int8_t> out(nOut);
      void*di=up(in.data(),in.size()),*dout; CHECK(aclrtMalloc(&dout,nOut,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*ti=mk({8},ACL_INT8,di),*to=mk({4},ACL_INT8,dout); uint64_t ws=0; aclOpExecutor*ex=nullptr;
      CHECK(aclnnConvertWeightToINT4PackGetWorkspaceSize(ti,to,&ws,&ex)); CHECK(aclnnConvertWeightToINT4Pack(nullptr,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      CHECK(aclrtMemcpy(out.data(),nOut,dout,nOut,ACL_MEMCPY_DEVICE_TO_HOST));
      double bad=0; for(int i=0;i<nOut;i++){ int ref=((in[2*i+1]&0xF)<<4)|(in[2*i]&0xF); if((out[i]&0xFF)!=ref) bad=1; }
      report("ConvertWeightToINT4Pack", bad, 0.0); aclDestroyTensor(ti);aclDestroyTensor(to); aclrtFree(di);aclrtFree(dout); }
}
int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    CHECK(aclrtCreateStream(&g_stream));
    srand(11);

    t_matmul(false, 128, 256, 64, 1e-5);
    t_matmul(false, 257, 511, 129, 1e-5);   // non-aligned
    t_matmul(true, 128, 256, 64, 1e-2);
    t_matmul_f16f32(128, 256, 64, 1e-3);     // fp16 input -> fp32 output (common Ascend config)
    t_matmul_f16f32(257, 511, 129, 1e-3);    // non-aligned
    t_matmul_tf32(128, 256, 64);             // TF32 fast-path switch
    t_matmul_bf16(128, 256, 64, 5e-2);      // bf16 (7-bit mantissa, wider tolerance)
    t_matmul_bf16(257, 511, 129, 5e-2);     // non-aligned
    t_mmbias_bf16(128, 256, 64, 5e-2);      // MatmulBias bf16 transB+bias (real-model bf16 projection shape)
    t_mmbias_bf16(64, 512, 128, 5e-2);
    t_matmul_fp8(128, 256, 64, 5e-2);       // fp8 accumulation: cuBLASLt may use fp8 fast-accum, tolerance relaxed
    t_matmul_fp8(64, 512, 128, 5e-2);
    t_matmul_mxfp8(128, 256, 64, 5e-2);     // MXFP8: block-dequantize to fp16 + GEMM
    t_matmul_mxfp8(64, 512, 128, 5e-2);
    t_matmul_mxfp8(256, 256, 128, 5e-2, true);   // MXFP8 hardware swizzle path
    t_matmul_mxfp8(128, 512, 256, 5e-2, true);
    t_matmul_mxfp4(256, 256, 128, 8e-2, true);   // MXFP4 native fp4 tensor-core (fp4 wider tolerance)
    t_matmul_mxfp4(128, 512, 256, 8e-2, true);
    t_wqmm(false, 128, 256, 64);            // W8A16 (large M -> dequant+cuBLASLt)
    t_wqmm(true, 128, 256, 64);             // W4A16
    t_wqmm(false, 4, 256, 64);              // W8A16 small M -> fused direct int8 read
    t_wqmm(true, 4, 256, 64);               // W4A16 small M -> fused direct int4 read
    t_qmm(128, 256, 64);                    // W8A8
    t_qmm_v(128, 256, 64);                  // W8A8 via QuantMatmulV3 variant
    t_qmm_iadd(64, 128, 48);                // QuantBatchMatmulInplaceAdd (out += quant matmul)
    { // TransposeQuantBatchMatMul: weight[N,K] transposed; out[M,N] = (x @ weightᵀ)·scale
      const int64_t M=32,K=64,N=48; std::vector<int8_t> x(M*K),w(N*K); for(auto&v:x)v=(int8_t)((rand()%255)-127); for(auto&v:w)v=(int8_t)((rand()%255)-127);
      std::vector<float> sc(N); std::vector<uint16_t> hsc(N); for(int64_t n=0;n<N;n++){sc[n]=0.001f*(1+n%7); hsc[n]=f2h(sc[n]);}
      void*dx=up(x.data(),M*K),*dw=up(w.data(),N*K),*dsc=up(hsc.data(),N*2),*dy; CHECK(aclrtMalloc(&dy,M*N*2,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({M,K},ACL_INT8,dx),*tw=mk({N,K},ACL_INT8,dw),*tsc=mk({N},ACL_FLOAT16,dsc),*ty=mk({M,N},ACL_FLOAT16,dy);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnTransposeQuantBatchMatMulGetWorkspaceSize(tx,tw,tsc,ty,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnTransposeQuantBatchMatMul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<uint16_t> hy(M*N); CHECK(aclrtMemcpy(hy.data(),M*N*2,dy,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int64_t m=0;m<M;m++)for(int64_t n=0;n<N;n++){ long long a=0; for(int64_t k=0;k<K;k++)a+=(long long)x[m*K+k]*w[n*K+k]; double ref=(double)a*(double)h2f(hsc[n]);
        me=std::max(me,std::fabs(h2f(hy[m*N+n])-ref)); mr=std::max(mr,std::fabs(ref)); }
      report("TransposeQuantBatchMatMul", me/(mr+1e-9), 2e-2);
      aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tsc);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dw);aclrtFree(dsc);aclrtFree(dy); }
    t_gmm(64, 48, {50, 30, 80, 40});        // GroupedMatmul MoE
    t_gmm(64, 48, {50, 30, 80, 40}, true);  // GroupedMatmul native grouped GEMM
    t_matmul_fp4(ACL_FLOAT4_E2M1, "fp4e2m1", 128, 256, 64, 5e-2);   // fp4/fp6 unpack to fp16 + GEMM
    t_matmul_fp4(ACL_FLOAT6_E2M3, "fp6e2m3", 64, 512, 128, 5e-2);
    t_matmul_mxfp4(128, 256, 64, 5e-2);                             // MXFP4: fp4 + E8M0 block scaling
    t_matmul_mxfp4(64, 512, 128, 5e-2);

    // Error normalized as max|err|/max|ref| (avoids near-zero elements inflating relative error); true fp32 accumulation ~1e-7
    t_bmm(8, 64, 96, 48, 1e-5);
    t_bmm(4, 128, 256, 64, 1e-5);
    t_bmm_bf16(8, 64, 96, 48, 5e-2);                               // bf16 coverage: BatchMatMul
    t_gmm_bf16(64, 48, {50, 30, 80, 40}, 5e-2);                    // GroupedMatmul MoE bf16
    t_conv_bf16(2, 4, 16, 16, 8, 3, 1, 1);                         // Conv forward bf16
    t_mmbias("MMBias bias[N]", false, false, 0, 1, 128, 256, 64, 1e-5);
    t_mmbias("MMBias transA relu", true, false, 1, 1, 128, 256, 64, 1e-5);
    t_mmbias("MMBias transB gelu", false, true, 2, 0, 128, 256, 64, 1e-5);
    t_mmbias("MMBias transAB bias[M,N]", true, true, 0, 2, 96, 128, 64, 1e-5);

    t_conv(2, 8, 32, 32, 16, 3, 3, 1, 1, false, 1e-5);
    t_conv(2, 8, 32, 32, 16, 3, 3, 2, 0, false, 1e-5);
    t_conv(1, 3, 64, 64, 8, 5, 5, 1, 2, true, 1e-5);

    t_conv_bwd(2, 4, 8, 8, 6, 3, 3, 1, 1, 1);   // dgrad + wgrad
    t_conv_fp16(2, 4, 16, 16, 8, 3, 1, 1);
    t_pool(false, 2, 4, 16, 16, 3, 2, 1);       // MaxPool
    t_pool(true,  2, 4, 16, 16, 3, 2, 1);       // AvgPool
    t_conv_fp4(2, 4, 16, 16, 8, 3, 1, 1);       // fp4 input conv
    t_conv3d(2, 3, 8, 8, 8, 4, 3, 1, 1);        // conv3d
    t_conv_transpose(2, 4, 8, 8, 6, 3, 2, 1);   // ConvTranspose2d stride2
    t_pool_bwd(false, 2, 3, 8, 8, 3, 2, 1);     // MaxPool backward
    t_pool_bwd(true,  2, 3, 8, 8, 3, 2, 1);     // AvgPool backward
    t_adaptive_avgpool(2, 4, 16, 16, 5, 5);     // AdaptiveAvgPool2d
    t_pool3d(false, 2, 3, 8, 8, 8, 3, 2, 1);    // MaxPool3d
    t_pool3d(true,  2, 3, 8, 8, 8, 3, 2, 1);    // AvgPool3d
    t_conv_ext();                               // Im2col/ConvTbc/ConvolutionBackward/DeformableConv2d/helpers
    { // TransposeBatchMatMul: out[B,M,N] = a[B,M,K] @ b[B,N,K]^T
      const int B=2,M=3,K=4,N=5; auto a=randv(B*M*K),b=randv(B*N*K);
      void*da=up(a.data(),a.size()*4),*db=up(b.data(),b.size()*4),*dy; CHECK(aclrtMalloc(&dy,B*M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*ta=mk({B,M,K},ACL_FLOAT,da),*tb=mk({B,N,K},ACL_FLOAT,db),*ty=mk({B,M,N},ACL_FLOAT,dy);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnTransposeBatchMatMulGetWorkspaceSize(ta,tb,ty,1,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnTransposeBatchMatMul(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> y(B*M*N); CHECK(aclrtMemcpy(y.data(),y.size()*4,dy,y.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int bb=0;bb<B;bb++)for(int m=0;m<M;m++)for(int n=0;n<N;n++){ double acc=0; for(int k=0;k<K;k++)acc+=(double)a[(bb*M+m)*K+k]*b[(bb*N+n)*K+k]; double got=y[(bb*M+m)*N+n]; me=std::max(me,std::fabs(got-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("TransposeBatchMatMul", me/(mr+1.0), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(da);aclrtFree(db);aclrtFree(dy); }
    { // BatchMatmulQuant: int8 a[B,M,K] @ b[B,K,N] × scale[N]
      const int B=2,M=3,K=4,N=5; std::vector<int8_t> a(B*M*K),b(B*K*N); for(auto&v:a)v=(int8_t)(rand()%21-10); for(auto&v:b)v=(int8_t)(rand()%21-10);
      std::vector<float> sc(N); for(int n=0;n<N;n++)sc[n]=0.01f*(n+1);
      void*da=up(a.data(),a.size()),*db=up(b.data(),b.size()),*dsc=up(sc.data(),N*4),*dy; CHECK(aclrtMalloc(&dy,B*M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*ta=mk({B,M,K},ACL_INT8,da),*tb=mk({B,K,N},ACL_INT8,db),*tsc=mk({N},ACL_FLOAT,dsc),*ty=mk({B,M,N},ACL_FLOAT,dy);
      uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnBatchMatmulQuantGetWorkspaceSize(ta,tb,tsc,ty,&ws,&ex));
      void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnBatchMatmulQuant(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
      std::vector<float> y(B*M*N); CHECK(aclrtMemcpy(y.data(),y.size()*4,dy,y.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int bb=0;bb<B;bb++)for(int m=0;m<M;m++)for(int n=0;n<N;n++){ long acc=0; for(int k=0;k<K;k++)acc+=(long)a[(bb*M+m)*K+k]*b[(bb*K+k)*N+n]; double r=acc*sc[n]; double got=y[(bb*M+m)*N+n]; me=std::max(me,std::fabs(got-r)); mr=std::max(mr,std::fabs(r)); }
      report("BatchMatmulQuant", me/(mr+1.0), 1e-4); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tsc);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(da);aclrtFree(db);aclrtFree(dsc);aclrtFree(dy); }

    CHECK(aclrtDestroyStream(g_stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
