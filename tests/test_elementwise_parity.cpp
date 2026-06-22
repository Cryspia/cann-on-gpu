// Cross-check for the elementwise / misc / activation / generator operator gap (CUDA-parity set landed in
// metal/src/ops/elementwise_parity.mm). Self-contained: each case computes a double-precision CPU reference
// (or equivalence to an existing base op) and reports err vs tolerance. 48 ops covered.
//
// All ops are pointwise / small misc ops on unified memory; tolerances 1e-5 (fp32 round-trip) unless a
// looser bound is genuinely warranted (documented inline). Predicate/bitwise/index ops are exact (tol 0).
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
using namespace hn;

// Some gap ops are not declared in the shared canonical header build the test links against in all configs;
// declare the ones we exercise via extern "C" with signatures matching the canonical header EXACTLY.
extern "C" {
aclnnStatus aclnnSoftsignGetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSoftsign(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRealGetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnReal(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAngleV2GetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAngleV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSignbitGetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSignbit(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnScaleGetWorkspaceSize(const aclTensor*, const aclScalar*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnScale(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFmodTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFmodTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnXLogYTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnXLogYTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSiluMulGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSiluMul(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGeluMulGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGeluMul(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFloorDivGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFloorDiv(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRealDivGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRealDiv(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDivV3GetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDivV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnClampMaxTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnClampMaxTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnClampMinTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnClampMinTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnLogicalXorGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnLogicalXor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnComplexGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnComplex(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnPolarGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnPolar(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnLeftShiftGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnLeftShift(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRightShiftGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRightShift(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnLeftShiftsGetWorkspaceSize(const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnLeftShifts(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIsCloseGetWorkspaceSize(const aclTensor*, const aclTensor*, double, double, bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIsClose(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnLerpsGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnLerps(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnPowScalarTensorGetWorkspaceSize(const aclScalar*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnPowScalarTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRemainderScalarTensorGetWorkspaceSize(const aclScalar*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRemainderScalarTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnXLogYScalarOtherGetWorkspaceSize(const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnXLogYScalarOther(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnXLogYScalarSelfGetWorkspaceSize(const aclScalar*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnXLogYScalarSelf(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFloorDividesGetWorkspaceSize(const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFloorDivides(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRsubsGetWorkspaceSize(const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRsubs(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIsInTensorScalarGetWorkspaceSize(const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIsInTensorScalar(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIsInScalarTensorGetWorkspaceSize(const aclScalar*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIsInScalarTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMaxNGetWorkspaceSize(const aclTensorList*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMaxN(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMinNGetWorkspaceSize(const aclTensorList*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMinN(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGluGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGlu(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnModulateGetWorkspaceSize(const aclTensor*, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnModulate(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMinDimGetWorkspaceSize(const aclTensor*, int64_t, bool, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMinDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSqueezeGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSqueeze(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnUnsqueezeGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnUnsqueeze(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnBernoulliTensorGetWorkspaceSize(aclTensor*, const aclTensor*, int64_t, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnBernoulliTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAddV3GetWorkspaceSize(const aclTensor*, const aclTensor*, const aclScalar*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAddV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFastGeluV2GetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFastGeluV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGeGluV3GetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGeGluV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnHstackGetWorkspaceSize(const aclTensor* const*, uint64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnHstack(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDstackGetWorkspaceSize(const aclTensor* const*, uint64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDstack(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnVdotGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnVdot(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnInnerGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnInner(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMaxV2GetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMaxV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMinGetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMin(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMultinomialTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMultinomialTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
}

// ---- small helpers ----
static aclScalar* scl(float v) { return aclCreateScalar(&v, ACL_FLOAT); }
static double sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }
static double gelu_erf(double x) { return 0.5 * x * std::erfc(-x * 0.70710678118654752440); }

// run a unary float op (self->out, fp32) and compare to a CPU reference fn
template <class Mk, class Ref>
static void t_unary(const char* name, std::vector<float> x, Mk mkws, aclnnStatus (*run)(void*,uint64_t,aclOpExecutor*,aclrtStream), Ref ref, double tol=1e-5) {
    int64_t n = x.size(); std::vector<float> y(n); DevBuf dx(n*4), dy(n*4); dx.up(x.data());
    aclTensor* tx = mk({n}, ACL_FLOAT, dx.p), * ty = mk({n}, ACL_FLOAT, dy.p);
    exec2([&](uint64_t* w, aclOpExecutor** e){ return mkws(tx, ty, w, e); }, run);
    dy.down(y.data());
    double bad = 0; for (int64_t i = 0; i < n; i++) bad = std::max(bad, std::fabs((double)y[i] - ref((double)x[i])));
    report(name, bad, tol); aclDestroyTensor(tx); aclDestroyTensor(ty);
}

// run a binary float op (a,b->out, fp32, equal shapes)
template <class Mk, class Ref>
static void t_binary(const char* name, std::vector<float> a, std::vector<float> b, Mk mkws, aclnnStatus (*run)(void*,uint64_t,aclOpExecutor*,aclrtStream), Ref ref, double tol=1e-5) {
    int64_t n = a.size(); std::vector<float> y(n); DevBuf da(n*4), db(n*4), dy(n*4); da.up(a.data()); db.up(b.data());
    aclTensor* ta = mk({n}, ACL_FLOAT, da.p), * tb = mk({n}, ACL_FLOAT, db.p), * ty = mk({n}, ACL_FLOAT, dy.p);
    exec2([&](uint64_t* w, aclOpExecutor** e){ return mkws(ta, tb, ty, w, e); }, run);
    dy.down(y.data());
    double bad = 0; for (int64_t i = 0; i < n; i++) bad = std::max(bad, std::fabs((double)y[i] - ref((double)a[i], (double)b[i])));
    report(name, bad, tol); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty);
}

int main() {
    init();
    srand(1234);

    // ===================== unary float =====================
    t_unary("Softsign", randv(2048, -8, 8), aclnnSoftsignGetWorkspaceSize, aclnnSoftsign, [](double v){ return v/(1.0+std::fabs(v)); });

    // Real: interleaved complex [N,2] -> [N]
    { const int N = 512; auto c = randv(2*N, -3, 3); std::vector<float> y(N); DevBuf dc(2*N*4), dy(N*4); dc.up(c.data());
      aclTensor* tc = mk({N,2}, ACL_FLOAT, dc.p), * ty = mk({N}, ACL_FLOAT, dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnRealGetWorkspaceSize(tc, ty, w, e); }, aclnnReal);
      dy.down(y.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, std::fabs((double)y[i] - c[2*i]));
      report("Real", bad, 0); aclDestroyTensor(tc); aclDestroyTensor(ty); }

    // AngleV2: interleaved complex -> atan2(im,re)
    { const int N = 512; auto c = randv(2*N, -3, 3); std::vector<float> y(N); DevBuf dc(2*N*4), dy(N*4); dc.up(c.data());
      aclTensor* tc = mk({N,2}, ACL_FLOAT, dc.p), * ty = mk({N}, ACL_FLOAT, dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnAngleV2GetWorkspaceSize(tc, ty, w, e); }, aclnnAngleV2);
      dy.down(y.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, std::fabs((double)y[i] - std::atan2((double)c[2*i+1], (double)c[2*i])));
      report("AngleV2", bad, 1e-5); aclDestroyTensor(tc); aclDestroyTensor(ty); }

    // Signbit: bool out
    { const int N = 1000; auto x = randv(N, -2, 2); x[0]=-0.0f; x[1]=0.0f; std::vector<uint8_t> y(N); DevBuf dx(N*4), dy(N); dx.up(x.data());
      aclTensor* tx = mk({N}, ACL_FLOAT, dx.p), * ty = mk({N}, ACL_BOOL, dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnSignbitGetWorkspaceSize(tx, ty, w, e); }, aclnnSignbit);
      dy.down(y.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, std::fabs((double)y[i] - (std::signbit((double)x[i]) ? 1 : 0)));
      report("Signbit", bad, 0); aclDestroyTensor(tx); aclDestroyTensor(ty); }

    // Scale: out = self*scale + bias
    { const int N = 2048; auto x = randv(N, -4, 4); float sc = 2.5f, bi = -0.75f; std::vector<float> y(N); DevBuf dx(N*4), dy(N*4); dx.up(x.data());
      aclTensor* tx = mk({N}, ACL_FLOAT, dx.p), * ty = mk({N}, ACL_FLOAT, dy.p); auto a = scl(sc), b = scl(bi);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnScaleGetWorkspaceSize(tx, a, b, ty, w, e); }, aclnnScale);
      dy.down(y.data()); double bad = 0; for (int i = 0; i < N; i++) bad = std::max(bad, std::fabs((double)y[i] - ((double)x[i]*sc + bi)));
      report("Scale", bad, 1e-5); aclDestroyScalar(a); aclDestroyScalar(b); aclDestroyTensor(tx); aclDestroyTensor(ty); }

    // ===================== binary float =====================
    t_binary("FmodTensor", randv(2048,-9,9), randv(2048,1,5), aclnnFmodTensorGetWorkspaceSize, aclnnFmodTensor, [](double a,double b){ return std::fmod(a,b); });
    t_binary("XLogYTensor", randv(2048,-2,2), randv(2048,0.1,5), aclnnXLogYTensorGetWorkspaceSize, aclnnXLogYTensor, [](double a,double b){ return a==0?0.0:a*std::log(b); });
    t_binary("SiluMul", randv(2048,-6,6), randv(2048,-3,3), aclnnSiluMulGetWorkspaceSize, aclnnSiluMul, [](double a,double b){ return (a*sig(a))*b; });
    t_binary("GeluMul", randv(2048,-6,6), randv(2048,-3,3), aclnnGeluMulGetWorkspaceSize, aclnnGeluMul, [](double a,double b){ return gelu_erf(a)*b; });
    t_binary("FloorDiv", randv(2048,-9,9), randv(2048,1,5), aclnnFloorDivGetWorkspaceSize, aclnnFloorDiv, [](double a,double b){ return std::floor(a/b); });
    t_binary("RealDiv", randv(2048,-5,5), randv(2048,1,5), aclnnRealDivGetWorkspaceSize, aclnnRealDiv, [](double a,double b){ return a/b; });
    t_binary("DivV3", randv(2048,-5,5), randv(2048,1,5), aclnnDivV3GetWorkspaceSize, aclnnDivV3, [](double a,double b){ return a/b; });
    t_binary("ClampMaxTensor", randv(2048,-5,5), randv(2048,-1,1), aclnnClampMaxTensorGetWorkspaceSize, aclnnClampMaxTensor, [](double a,double b){ return a<b?a:b; });
    t_binary("ClampMinTensor", randv(2048,-5,5), randv(2048,-1,1), aclnnClampMinTensorGetWorkspaceSize, aclnnClampMinTensor, [](double a,double b){ return a>b?a:b; });

    // LogicalXor: bool in/out
    { const int N = 1000; std::vector<uint8_t> a(N), b(N), y(N); for (int i=0;i<N;i++){ a[i]=rand()&1; b[i]=rand()&1; }
      DevBuf da(N), db(N), dy(N); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_BOOL,da.p), *tb=mk({N},ACL_BOOL,db.p), *ty=mk({N},ACL_BOOL,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnLogicalXorGetWorkspaceSize(ta,tb,ty,w,e); }, aclnnLogicalXor);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,(double)std::abs((int)y[i]-((a[i]!=0)^(b[i]!=0))));
      report("LogicalXor", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }

    // Complex: (re,im) -> interleaved [N,2]
    { const int N=512; auto re=randv(N,-3,3), im=randv(N,-3,3); std::vector<float> y(2*N);
      DevBuf dr(N*4), di(N*4), dy(2*N*4); dr.up(re.data()); di.up(im.data());
      aclTensor* tr=mk({N},ACL_FLOAT,dr.p), *ti=mk({N},ACL_FLOAT,di.p), *ty=mk({N,2},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnComplexGetWorkspaceSize(tr,ti,ty,w,e); }, aclnnComplex);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++){ bad=std::max(bad,std::fabs((double)y[2*i]-re[i])); bad=std::max(bad,std::fabs((double)y[2*i+1]-im[i])); }
      report("Complex", bad, 0); aclDestroyTensor(tr); aclDestroyTensor(ti); aclDestroyTensor(ty); }

    // Polar: (abs,angle) -> interleaved
    { const int N=512; auto ab=randv(N,0.1,3), an=randv(N,-3,3); std::vector<float> y(2*N);
      DevBuf da(N*4), dn(N*4), dy(2*N*4); da.up(ab.data()); dn.up(an.data());
      aclTensor* ta=mk({N},ACL_FLOAT,da.p), *tn=mk({N},ACL_FLOAT,dn.p), *ty=mk({N,2},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnPolarGetWorkspaceSize(ta,tn,ty,w,e); }, aclnnPolar);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++){ bad=std::max(bad,std::fabs((double)y[2*i]-ab[i]*std::cos((double)an[i]))); bad=std::max(bad,std::fabs((double)y[2*i+1]-ab[i]*std::sin((double)an[i]))); }
      report("Polar", bad, 1e-5); aclDestroyTensor(ta); aclDestroyTensor(tn); aclDestroyTensor(ty); }

    // ===================== int32 shifts =====================
    { const int N=512; std::vector<int32_t> a(N), b(N), y(N); for(int i=0;i<N;i++){ a[i]=(rand()%1000)-500; b[i]=rand()%5; }
      DevBuf da(N*4), db(N*4), dy(N*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_INT32,da.p), *tb=mk({N},ACL_INT32,db.p), *ty=mk({N},ACL_INT32,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnLeftShiftGetWorkspaceSize(ta,tb,ty,w,e); }, aclnnLeftShift);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,(double)std::abs((long)y[i]-((long)a[i]<<b[i])));
      report("LeftShift", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    { const int N=512; std::vector<int32_t> a(N), b(N), y(N); for(int i=0;i<N;i++){ a[i]=(rand()%100000)-50000; b[i]=rand()%5; }
      DevBuf da(N*4), db(N*4), dy(N*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_INT32,da.p), *tb=mk({N},ACL_INT32,db.p), *ty=mk({N},ACL_INT32,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnRightShiftGetWorkspaceSize(ta,tb,ty,w,e); }, aclnnRightShift);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,(double)std::abs((long)y[i]-((long)a[i]>>b[i])));
      report("RightShift", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    { const int N=512; int sh=3; std::vector<int32_t> a(N), y(N); for(int i=0;i<N;i++) a[i]=(rand()%2000)-1000;
      DevBuf da(N*4), dy(N*4); da.up(a.data()); aclTensor* ta=mk({N},ACL_INT32,da.p), *ty=mk({N},ACL_INT32,dy.p);
      int32_t s=sh; auto ss=aclCreateScalar(&s,ACL_INT32);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnLeftShiftsGetWorkspaceSize(ta,ss,ty,w,e); }, aclnnLeftShifts);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,(double)std::abs((long)y[i]-((long)a[i]<<sh)));
      report("LeftShifts", bad, 0); aclDestroyScalar(ss); aclDestroyTensor(ta); aclDestroyTensor(ty); }

    // ===================== IsClose / Lerps =====================
    { const int N=1000; auto a=randv(N,-2,2); std::vector<float> b(N); for(int i=0;i<N;i++) b[i]= (i%3==0)? a[i] : a[i]+ ( (i%2)? 1e-4f : 0.5f );
      double rt=1e-3, at=1e-5; std::vector<uint8_t> y(N); DevBuf da(N*4), db(N*4), dy(N); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_FLOAT,da.p), *tb=mk({N},ACL_FLOAT,db.p), *ty=mk({N},ACL_BOOL,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnIsCloseGetWorkspaceSize(ta,tb,rt,at,false,ty,w,e); }, aclnnIsClose);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++){ int ref=(std::fabs((double)a[i]-b[i])<=at+rt*std::fabs((double)b[i])); bad=std::max(bad,(double)std::abs((int)y[i]-ref)); }
      report("IsClose", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    { const int N=2048; auto a=randv(N,-3,3), b=randv(N,-3,3); float wt=0.35f; std::vector<float> y(N);
      DevBuf da(N*4), db(N*4), dy(N*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_FLOAT,da.p), *tb=mk({N},ACL_FLOAT,db.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sw=scl(wt);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnLerpsGetWorkspaceSize(ta,tb,sw,ty,w,e); }, aclnnLerps);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-((double)a[i]+wt*((double)b[i]-a[i]))));
      report("Lerps", bad, 1e-5); aclDestroyScalar(sw); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }

    // ===================== scalar/tensor mixes =====================
    { const int N=2048; auto t=randv(N,-2,2); float s=2.0f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnPowScalarTensorGetWorkspaceSize(sc,tt,ty,w,e); }, aclnnPowScalarTensor);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-std::pow((double)s,(double)t[i])));
      report("PowScalarTensor", bad, 1e-4); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }  // pow round-trip in fp32
    { const int N=2048; auto t=randv(N,1,5); float s=7.3f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnRemainderScalarTensorGetWorkspaceSize(sc,tt,ty,w,e); }, aclnnRemainderScalarTensor);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++){ double r=(double)s-std::floor((double)s/t[i])*t[i]; bad=std::max(bad,std::fabs((double)y[i]-r)); }
      report("RemainderScalarTensor", bad, 1e-4); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    { const int N=2048; auto t=randv(N,-3,3); float s=4.0f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnXLogYScalarOtherGetWorkspaceSize(tt,sc,ty,w,e); }, aclnnXLogYScalarOther);
      dy.down(y.data()); double bad=0; double L=std::log((double)s); for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-(double)t[i]*L));
      report("XLogYScalarOther", bad, 1e-5); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    { const int N=2048; auto t=randv(N,0.1,5); float s=2.5f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnXLogYScalarSelfGetWorkspaceSize(sc,tt,ty,w,e); }, aclnnXLogYScalarSelf);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-(double)s*std::log((double)t[i])));
      report("XLogYScalarSelf", bad, 1e-5); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    { const int N=2048; auto t=randv(N,-9,9); float s=4.0f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnFloorDividesGetWorkspaceSize(tt,sc,ty,w,e); }, aclnnFloorDivides);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-std::floor((double)t[i]/s)));
      report("FloorDivides", bad, 0); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    { const int N=2048; auto t=randv(N,-3,3); float s=1.5f; std::vector<float> y(N); DevBuf dt(N*4), dy(N*4); dt.up(t.data());
      aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_FLOAT,dy.p); auto sc=scl(s);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnRsubsGetWorkspaceSize(tt,sc,ty,w,e); }, aclnnRsubs);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-((double)s-(double)t[i])));
      report("Rsubs", bad, 1e-5); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    // IsInTensorScalar: place the target value at known indices
    { const int N=1000; auto t=randv(N,-2,2); float v=1.234567f; for(int i=0;i<N;i+=7) t[i]=v; std::vector<uint8_t> y(N);
      DevBuf dt(N*4), dy(N); dt.up(t.data()); aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *ty=mk({N},ACL_BOOL,dy.p); auto sc=scl(v);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnIsInTensorScalarGetWorkspaceSize(tt,sc,ty,w,e); }, aclnnIsInTensorScalar);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,(double)std::abs((int)y[i]-((int)(t[i]==v))));
      report("IsInTensorScalar", bad, 0); aclDestroyScalar(sc); aclDestroyTensor(tt); aclDestroyTensor(ty); }
    // IsInScalarTensor: scalar 1 present, scalar 99 absent -> two checks
    { const int N=64; auto t=randv(N,-2,2); t[10]=1.0f; uint8_t y0=0,y1=0;
      DevBuf dt(N*4), do0(1), do1(1); dt.up(t.data()); aclTensor* tt=mk({N},ACL_FLOAT,dt.p), *t0=mk({1},ACL_BOOL,do0.p), *t1=mk({1},ACL_BOOL,do1.p);
      auto s1=scl(1.0f), s2=scl(99.0f);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnIsInScalarTensorGetWorkspaceSize(s1,tt,t0,w,e); }, aclnnIsInScalarTensor);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnIsInScalarTensorGetWorkspaceSize(s2,tt,t1,w,e); }, aclnnIsInScalarTensor);
      do0.down(&y0); do1.down(&y1); double bad = std::abs((int)y0-1) + std::abs((int)y1-0);
      report("IsInScalarTensor", bad, 0); aclDestroyScalar(s1); aclDestroyScalar(s2); aclDestroyTensor(tt); aclDestroyTensor(t0); aclDestroyTensor(t1); }

    // ===================== MaxN / MinN over tensor list =====================
    { const int K=4, N=512; std::vector<std::vector<float>> src(K); std::vector<DevBuf*> bufs; std::vector<aclTensor*> ts;
      for (int k=0;k<K;k++){ src[k]=randv(N,-5,5); auto* d=new DevBuf(N*4); d->up(src[k].data()); bufs.push_back(d); ts.push_back(mk({N},ACL_FLOAT,d->p)); }
      const aclTensor* arr[4]={ts[0],ts[1],ts[2],ts[3]}; auto* tl=aclCreateTensorList(arr,K);
      std::vector<float> ymx(N), ymn(N); DevBuf dmx(N*4), dmn(N*4); aclTensor* tmx=mk({N},ACL_FLOAT,dmx.p), *tmn=mk({N},ACL_FLOAT,dmn.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMaxNGetWorkspaceSize(tl,tmx,w,e); }, aclnnMaxN);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMinNGetWorkspaceSize(tl,tmn,w,e); }, aclnnMinN);
      dmx.down(ymx.data()); dmn.down(ymn.data()); double bx=0,bn=0;
      for(int i=0;i<N;i++){ double mx=src[0][i],mn=src[0][i]; for(int k=1;k<K;k++){ mx=std::max(mx,(double)src[k][i]); mn=std::min(mn,(double)src[k][i]); } bx=std::max(bx,std::fabs((double)ymx[i]-mx)); bn=std::max(bn,std::fabs((double)ymn[i]-mn)); }
      report("MaxN", bx, 0); report("MinN", bn, 0);
      aclDestroyTensorList(tl); aclDestroyTensor(tmx); aclDestroyTensor(tmn); for(auto* t:ts) aclDestroyTensor(t); for(auto* d:bufs) delete d; }

    // ===================== Glu / Modulate =====================
    // Glu: in[R,2D] -> out[R,D] = first*sigmoid(second)
    { const int R=8, D=16; auto x=randv(R*2*D,-3,3); std::vector<float> y(R*D); DevBuf dx(R*2*D*4), dy(R*D*4); dx.up(x.data());
      aclTensor* tx=mk({R,2*D},ACL_FLOAT,dx.p), *ty=mk({R,D},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnGluGetWorkspaceSize(tx,-1,ty,w,e); }, aclnnGlu);
      dy.down(y.data()); double bad=0; for(int r=0;r<R;r++)for(int d=0;d<D;d++){ const float* p=&x[r*2*D]; double ref=p[d]*sig(p[D+d]); bad=std::max(bad,std::fabs((double)y[r*D+d]-ref)); }
      report("Glu", bad, 1e-6); aclDestroyTensor(tx); aclDestroyTensor(ty); }
    // Modulate: x*(1+scale)+shift, all same shape
    { const int N=1024; auto x=randv(N,-3,3), sc=randv(N,-1,1), sh=randv(N,-2,2); std::vector<float> y(N);
      DevBuf dx(N*4), ds(N*4), dh(N*4), dy(N*4); dx.up(x.data()); ds.up(sc.data()); dh.up(sh.data());
      aclTensor* tx=mk({N},ACL_FLOAT,dx.p), *ts=mk({N},ACL_FLOAT,ds.p), *th=mk({N},ACL_FLOAT,dh.p), *ty=mk({N},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnModulateGetWorkspaceSize(tx,ts,th,ty,w,e); }, aclnnModulate);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-((double)x[i]*(1.0+sc[i])+sh[i])));
      report("Modulate", bad, 1e-5); aclDestroyTensor(tx); aclDestroyTensor(ts); aclDestroyTensor(th); aclDestroyTensor(ty); }

    // ===================== MinDim (values + indices) =====================
    { const int O=6, D=10; auto x=randv(O*D,-5,5); std::vector<float> v(O); std::vector<int64_t> idx(O);
      DevBuf dx(O*D*4), dv(O*4), di(O*8); dx.up(x.data());
      aclTensor* tx=mk({O,D},ACL_FLOAT,dx.p), *tv=mk({O},ACL_FLOAT,dv.p), *ti=mk({O},ACL_INT64,di.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMinDimGetWorkspaceSize(tx,1,false,tv,ti,w,e); }, aclnnMinDim);
      dv.down(v.data()); di.down(idx.data()); double bad=0;
      for(int o=0;o<O;o++){ double best=x[o*D]; int bi=0; for(int d=1;d<D;d++) if(x[o*D+d]<best){ best=x[o*D+d]; bi=d; } bad=std::max(bad,std::fabs((double)v[o]-best)); bad=std::max(bad,(double)std::llabs(idx[o]-bi)); }
      report("MinDim", bad, 0); aclDestroyTensor(tx); aclDestroyTensor(tv); aclDestroyTensor(ti); }

    // ===================== Squeeze / Unsqueeze (data preserved) =====================
    { const int N=120; auto x=randv(N,-2,2); std::vector<float> y(N); DevBuf dx(N*4), dy(N*4); dx.up(x.data());
      aclTensor* tx=mk({1,N,1},ACL_FLOAT,dx.p), *ty=mk({N},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnSqueezeGetWorkspaceSize(tx,0,ty,w,e); }, aclnnSqueeze);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-x[i]));
      report("Squeeze", bad, 0); aclDestroyTensor(tx); aclDestroyTensor(ty); }
    { const int N=120; auto x=randv(N,-2,2); std::vector<float> y(N); DevBuf dx(N*4), dy(N*4); dx.up(x.data());
      aclTensor* tx=mk({N},ACL_FLOAT,dx.p), *ty=mk({1,N},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnUnsqueezeGetWorkspaceSize(tx,0,ty,w,e); }, aclnnUnsqueeze);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-x[i]));
      report("Unsqueeze", bad, 0); aclDestroyTensor(tx); aclDestroyTensor(ty); }

    // ===================== BernoulliTensor (statistical: mean ~ p) =====================
    { const int N=200000; std::vector<float> p(N), y(N); for(int i=0;i<N;i++) p[i]=0.3f; DevBuf dp(N*4), dy(N*4); dp.up(p.data());
      aclTensor* tp=mk({N},ACL_FLOAT,dp.p), *ty=mk({N},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnBernoulliTensorGetWorkspaceSize(ty,tp,12345,w,e); }, aclnnBernoulliTensor);
      dy.down(y.data()); double m=0; for(int i=0;i<N;i++){ if(y[i]!=0.f&&y[i]!=1.f) m=1e9; m+=y[i]; } double mean=m/N;
      report("BernoulliTensor(mean~p)", std::fabs(mean-0.3), 1e-2); aclDestroyTensor(tp); aclDestroyTensor(ty); }  // RNG: statistical 1% tol

    // ===================== alias forwards =====================
    // AddV3 == Add(self,other,alpha=2): a + 2*b
    { const int N=2048; auto a=randv(N,-3,3), b=randv(N,-3,3); std::vector<float> y(N); DevBuf da(N*4), db(N*4), dy(N*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_FLOAT,da.p), *tb=mk({N},ACL_FLOAT,db.p), *ty=mk({N},ACL_FLOAT,dy.p); auto al=scl(2.0f);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnAddV3GetWorkspaceSize(ta,tb,al,ty,w,e); }, aclnnAddV3);
      dy.down(y.data()); double bad=0; for(int i=0;i<N;i++) bad=std::max(bad,std::fabs((double)y[i]-((double)a[i]+2.0*b[i])));
      report("AddV3", bad, 1e-5); aclDestroyScalar(al); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    // FastGeluV2 == FastGelu (tanh-approx gelu)
    t_unary("FastGeluV2", randv(2048,-6,6), aclnnFastGeluV2GetWorkspaceSize, aclnnFastGeluV2,
            [](double v){ double c=0.7978845608028654*(v+0.044715*v*v*v); return 0.5*v*(1.0+std::tanh(c)); }, 1e-5);  // FastGelu == tanh-approx GELU
    // GeGluV3 == GeGlu: in[R,2D]->out[R,D] = first*gelu(second)
    { const int R=8, D=16; auto x=randv(R*2*D,-3,3); std::vector<float> y(R*D); DevBuf dx(R*2*D*4), dy(R*D*4); dx.up(x.data());
      aclTensor* tx=mk({R,2*D},ACL_FLOAT,dx.p), *ty=mk({R,D},ACL_FLOAT,dy.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnGeGluV3GetWorkspaceSize(tx,ty,w,e); }, aclnnGeGluV3);
      dy.down(y.data()); double bad=0; for(int r=0;r<R;r++)for(int d=0;d<D;d++){ const float* p=&x[r*2*D]; double ref=gelu_erf(p[d])*p[D+d]; bad=std::max(bad,std::fabs((double)y[r*D+d]-ref)); }  // GeGlu == gelu(first)*second (erf-gelu)
      report("GeGluV3", bad, 1e-5); aclDestroyTensor(tx); aclDestroyTensor(ty); }
    // Hstack: two [3,4] along dim1 -> [3,8]
    { const int R=3, C=4; auto a=randv(R*C,-1,1), b=randv(R*C,-1,1); std::vector<float> y(R*2*C);
      DevBuf da(R*C*4), db(R*C*4), dy(R*2*C*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({R,C},ACL_FLOAT,da.p), *tb=mk({R,C},ACL_FLOAT,db.p), *ty=mk({R,2*C},ACL_FLOAT,dy.p); const aclTensor* ins[2]={ta,tb};
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnHstackGetWorkspaceSize(ins,2,ty,w,e); }, aclnnHstack);
      dy.down(y.data()); double bad=0; for(int r=0;r<R;r++){ for(int c=0;c<C;c++){ bad=std::max(bad,std::fabs((double)y[r*2*C+c]-a[r*C+c])); bad=std::max(bad,std::fabs((double)y[r*2*C+C+c]-b[r*C+c])); } }
      report("Hstack", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    // Dstack: two [2,3] -> [2,3,2] (cat along new last dim, here dim2)
    { const int A=2, B=3; auto a=randv(A*B,-1,1), b=randv(A*B,-1,1); std::vector<float> y(A*B*2);
      DevBuf da(A*B*4), db(A*B*4), dy(A*B*2*4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({A,B,1},ACL_FLOAT,da.p), *tb=mk({A,B,1},ACL_FLOAT,db.p), *ty=mk({A,B,2},ACL_FLOAT,dy.p); const aclTensor* ins[2]={ta,tb};
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnDstackGetWorkspaceSize(ins,2,ty,w,e); }, aclnnDstack);
      dy.down(y.data()); double bad=0; for(int i=0;i<A*B;i++){ bad=std::max(bad,std::fabs((double)y[2*i]-a[i])); bad=std::max(bad,std::fabs((double)y[2*i+1]-b[i])); }
      report("Dstack", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(ty); }
    // Vdot / Inner == Dot (1-D)
    { const int N=257; auto a=randv(N,-2,2), b=randv(N,-2,2); float yv=0,yi=0; DevBuf da(N*4), db(N*4), dv(4), di(4); da.up(a.data()); db.up(b.data());
      aclTensor* ta=mk({N},ACL_FLOAT,da.p), *tb=mk({N},ACL_FLOAT,db.p), *tv=mk({1},ACL_FLOAT,dv.p), *tin=mk({1},ACL_FLOAT,di.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnVdotGetWorkspaceSize(ta,tb,tv,w,e); }, aclnnVdot);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnInnerGetWorkspaceSize(ta,tb,tin,w,e); }, aclnnInner);
      dv.down(&yv); di.down(&yi); double ref=0; for(int i=0;i<N;i++) ref+=(double)a[i]*b[i];
      report("Vdot", relerr(yv,ref), 1e-5); report("Inner", relerr(yi,ref), 1e-5);
      aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tv); aclDestroyTensor(tin); }
    // MaxV2 / Min full reduce -> scalar
    { const int N=4096; auto x=randv(N,-9,9); float ymx=0,ymn=0; DevBuf dx(N*4), dmx(4), dmn(4); dx.up(x.data());
      aclTensor* tx=mk({N},ACL_FLOAT,dx.p), *tmx=mk({1},ACL_FLOAT,dmx.p), *tmn=mk({1},ACL_FLOAT,dmn.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMaxV2GetWorkspaceSize(tx,tmx,w,e); }, aclnnMaxV2);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMinGetWorkspaceSize(tx,tmn,w,e); }, aclnnMin);
      dmx.down(&ymx); dmn.down(&ymn); double rmx=-1e30,rmn=1e30; for(int i=0;i<N;i++){ rmx=std::max(rmx,(double)x[i]); rmn=std::min(rmn,(double)x[i]); }
      report("MaxV2", std::fabs((double)ymx-rmx), 1e-5); report("Min", std::fabs((double)ymn-rmn), 1e-5);
      aclDestroyTensor(tx); aclDestroyTensor(tmx); aclDestroyTensor(tmn); }
    // MultinomialTensor: numSamples carried as a tensor; every drawn index must be a valid class with prob>0
    { const int C=5, NS=64; std::vector<float> probs(C,0.f); probs[1]=0.5f; probs[3]=0.5f; int64_t nsv=NS; std::vector<int64_t> out(NS);
      DevBuf dp(C*4), dns(8), dout(NS*8); dp.up(probs.data()); dns.up(&nsv);
      aclTensor* tp=mk({C},ACL_FLOAT,dp.p), *tns=mk({1},ACL_INT64,dns.p), *to=mk({NS},ACL_INT64,dout.p);
      exec2([&](uint64_t* w, aclOpExecutor** e){ return aclnnMultinomialTensorGetWorkspaceSize(tp,tns,777,to,w,e); }, aclnnMultinomialTensor);
      dout.down(out.data()); double bad=0; for(int i=0;i<NS;i++){ if(out[i]!=1 && out[i]!=3) bad=1; }  // only classes 1,3 have mass
      report("MultinomialTensor", bad, 0); aclDestroyTensor(tp); aclDestroyTensor(tns); aclDestroyTensor(to); }

    return finish();
}
