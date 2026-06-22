// Cross-check for the 57 "gap" foreach ops (the aclnnForeach* symbols the CUDA backend exports that
// the Metal backend's foreach.mm did not already cover). Each op applies one elementwise operation
// across every tensor of an aclTensorList; references are computed on the CPU. Self-contained: the
// public aclnn_ops.h header in this tree declares only the original handful of foreach ops, so the
// gap-op prototypes are declared here directly (they match cuda/src/ops/foreach_ext.cu exactly).
#include "harness.h"
#include <cmath>
using namespace hn;

// ---- prototypes for the 57 gap ops (not in the shared aclnn_ops.h) ----
extern "C" {
typedef aclnnStatus (*FRun)(void *, uint64_t, aclOpExecutor *, aclrtStream);

#define P_UNARY(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_SCALAR(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclScalar*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_SCALARLIST(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensor*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_LIST(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensorList*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_LISTV2(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensorList*, const aclScalar*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_ADDC_SCALAR(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclScalar*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);
#define P_ADDC_LIST(N) \
  aclnnStatus N##GetWorkspaceSize(const aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensor*, const aclTensorList*, uint64_t*, aclOpExecutor**); \
  aclnnStatus N(void*, uint64_t, aclOpExecutor*, aclrtStream);

// unary (24)
P_UNARY(aclnnForeachAbs) P_UNARY(aclnnForeachAcos) P_UNARY(aclnnForeachAsin) P_UNARY(aclnnForeachAtan)
P_UNARY(aclnnForeachCos) P_UNARY(aclnnForeachCosh) P_UNARY(aclnnForeachErf) P_UNARY(aclnnForeachErfc)
P_UNARY(aclnnForeachExp) P_UNARY(aclnnForeachExpm1) P_UNARY(aclnnForeachLog) P_UNARY(aclnnForeachLog10)
P_UNARY(aclnnForeachLog1p) P_UNARY(aclnnForeachLog2) P_UNARY(aclnnForeachNeg) P_UNARY(aclnnForeachReciprocal)
P_UNARY(aclnnForeachSigmoid) P_UNARY(aclnnForeachSign) P_UNARY(aclnnForeachSin) P_UNARY(aclnnForeachSinh)
P_UNARY(aclnnForeachTan) P_UNARY(aclnnForeachTanh) P_UNARY(aclnnForeachRoundOffNumber) P_UNARY(aclnnForeachRoundOffNumberV2)
// single scalar (13)
P_SCALAR(aclnnForeachAddScalar) P_SCALAR(aclnnForeachAddScalarV2) P_SCALAR(aclnnForeachSubScalar) P_SCALAR(aclnnForeachSubScalarV2)
P_SCALAR(aclnnForeachMulScalarV2) P_SCALAR(aclnnForeachDivScalar) P_SCALAR(aclnnForeachDivScalarV2)
P_SCALAR(aclnnForeachMaximumScalar) P_SCALAR(aclnnForeachMaximumScalarV2) P_SCALAR(aclnnForeachMinimumScalar)
P_SCALAR(aclnnForeachMinimumScalarV2) P_SCALAR(aclnnForeachPowScalar) P_SCALAR(aclnnForeachPowScalarV2)
// scalar list (6)
P_SCALARLIST(aclnnForeachSubScalarList) P_SCALARLIST(aclnnForeachMulScalarList) P_SCALARLIST(aclnnForeachDivScalarList)
P_SCALARLIST(aclnnForeachMaximumScalarList) P_SCALARLIST(aclnnForeachMinimumScalarList) P_SCALARLIST(aclnnForeachPowScalarList)
// list (7)
P_LIST(aclnnForeachSubList) P_LIST(aclnnForeachMulList) P_LIST(aclnnForeachDivList)
P_LIST(aclnnForeachMaximumList) P_LIST(aclnnForeachMinimumList) P_LIST(aclnnForeachPowList)
P_LISTV2(aclnnForeachSubListV2)
// addc (7)
P_ADDC_SCALAR(aclnnForeachAddcmulScalarV2) P_ADDC_SCALAR(aclnnForeachAddcdivScalar) P_ADDC_SCALAR(aclnnForeachAddcdivScalarV2)
P_ADDC_LIST(aclnnForeachAddcmulScalarList) P_ADDC_LIST(aclnnForeachAddcdivScalarList)
P_ADDC_LIST(aclnnForeachAddcmulList) P_ADDC_LIST(aclnnForeachAddcdivList)
} // extern "C"

namespace {

// A list of fp32 tensors held on device (unified memory), plus host bookkeeping.
struct FList {
    std::vector<DevBuf *> bufs;
    std::vector<aclTensor *> ts;
    std::vector<int64_t> sizes;
    aclTensorList *list = nullptr;
};

FList make_list(const std::vector<std::vector<float>> &data) {
    FList l;
    for (auto &d : data) {
        auto *b = new DevBuf(d.size() * sizeof(float));
        b->up(d.data());
        l.bufs.push_back(b);
        l.sizes.push_back((int64_t)d.size());
        l.ts.push_back(mk({(int64_t)d.size()}, ACL_FLOAT, b->p));
    }
    l.list = aclCreateTensorList(l.ts.data(), l.ts.size());
    return l;
}
FList zeros_like(const std::vector<std::vector<float>> &data) {
    std::vector<std::vector<float>> z;
    for (auto &d : data) z.emplace_back(d.size(), 0.f);
    return make_list(z);
}

std::vector<float> dl(const DevBuf *b, int64_t n) { std::vector<float> v(n); b->down(v.data()); return v; }

// max |got-ref| / max|ref| over a whole list
double list_err(const FList &got, const std::vector<std::vector<float>> &ref) {
    double me = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        auto g = dl(got.bufs[i], got.sizes[i]);
        for (size_t j = 0; j < ref[i].size(); j++) {
            me = std::max(me, (double)std::fabs(g[j] - ref[i][j]));
            mr = std::max(mr, (double)std::fabs(ref[i][j]));
        }
    }
    return me / (mr + 1e-9);
}

// Three input lists of different shapes (3 tensors each). Values chosen positive where the op
// needs it (log/sqrt/pow/acos/asin domain): all in (0, 1] except where noted.
const std::vector<std::vector<float>> A = {{0.2f, 0.5f, 0.8f}, {0.1f, 0.3f, 0.6f, 0.9f}, {0.4f, 0.7f}};
const std::vector<std::vector<float>> B = {{0.7f, 0.3f, 0.9f}, {0.5f, 0.8f, 0.2f, 0.6f}, {0.1f, 0.5f}};
const std::vector<std::vector<float>> C = {{0.3f, 0.6f, 0.4f}, {0.9f, 0.2f, 0.7f, 0.5f}, {0.8f, 0.3f}};
// signed variant for ops fine with negatives (abs/sign/neg/sin/...): includes negatives and zero.
const std::vector<std::vector<float>> S = {{-1.5f, 0.f, 2.3f}, {-0.7f, 1.1f, -2.6f, 0.5f}, {3.4f, -0.4f}};

std::vector<std::vector<float>> apply_un(const std::vector<std::vector<float>> &in, float (*f)(float)) {
    auto r = in; for (auto &v : r) for (auto &e : v) e = f(e); return r;
}

// Run a unary op and compare against CPU reference f over the chosen input domain.
void test_un(const char *name,
             aclnnStatus (*ws)(const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **),
             FRun run, const std::vector<std::vector<float>> &in, float (*f)(float), double tol) {
    FList x = make_list(in);
    FList o = zeros_like(in);
    exec2([&](uint64_t *w, aclOpExecutor **ex) { return ws(x.list, o.list, w, ex); }, run);
    report(name, list_err(o, apply_un(in, f)), tol);
}

// Run a single-scalar op: out_i = g(x_i, s)
void test_scalar(const char *name,
                 aclnnStatus (*ws)(const aclTensorList *, const aclScalar *, const aclTensorList *, uint64_t *, aclOpExecutor **),
                 FRun run, const std::vector<std::vector<float>> &in, float s, float (*g)(float, float), double tol) {
    auto sc = aclCreateScalar(&s, ACL_FLOAT);
    FList x = make_list(in), o = zeros_like(in);
    exec2([&](uint64_t *w, aclOpExecutor **ex) { return ws(x.list, sc, o.list, w, ex); }, run);
    auto ref = in; for (auto &v : ref) for (auto &e : v) e = g(e, s);
    report(name, list_err(o, ref), tol);
}

// Run a scalar-list op: out_i = g(x_i, s_i)
void test_scalarlist(const char *name,
                     aclnnStatus (*ws)(const aclTensorList *, const aclTensor *, const aclTensorList *, uint64_t *, aclOpExecutor **),
                     FRun run, const std::vector<std::vector<float>> &in, const std::vector<float> &scal,
                     float (*g)(float, float), double tol) {
    DevBuf sb(scal.size() * sizeof(float)); sb.up(scal.data());
    auto st = mk({(int64_t)scal.size()}, ACL_FLOAT, sb.p);
    FList x = make_list(in), o = zeros_like(in);
    exec2([&](uint64_t *w, aclOpExecutor **ex) { return ws(x.list, st, o.list, w, ex); }, run);
    auto ref = in; for (size_t i = 0; i < ref.size(); i++) for (auto &e : ref[i]) e = g(e, scal[i]);
    report(name, list_err(o, ref), tol);
}

// Run a list op: out_i = g(x_i, y_i)
void test_list(const char *name,
               aclnnStatus (*ws)(const aclTensorList *, const aclTensorList *, const aclTensorList *, uint64_t *, aclOpExecutor **),
               FRun run, const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b,
               float (*g)(float, float), double tol) {
    FList x = make_list(a), y = make_list(b), o = zeros_like(a);
    exec2([&](uint64_t *w, aclOpExecutor **ex) { return ws(x.list, y.list, o.list, w, ex); }, run);
    auto ref = a; for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] = g(a[i][j], b[i][j]);
    report(name, list_err(o, ref), tol);
}

} // namespace

int main() {
    init();

    // ============ unary (24) ============
    test_un("ForeachAbs", aclnnForeachAbsGetWorkspaceSize,   aclnnForeachAbs,   S, [](float x){return std::fabs(x);}, 1e-6);
    test_un("ForeachAcos", aclnnForeachAcosGetWorkspaceSize,  aclnnForeachAcos,  A, [](float x){return std::acos(x);}, 1e-5);
    test_un("ForeachAsin", aclnnForeachAsinGetWorkspaceSize,  aclnnForeachAsin,  A, [](float x){return std::asin(x);}, 1e-5);
    test_un("ForeachAtan", aclnnForeachAtanGetWorkspaceSize,  aclnnForeachAtan,  S, [](float x){return std::atan(x);}, 1e-6);
    test_un("ForeachCos", aclnnForeachCosGetWorkspaceSize,   aclnnForeachCos,   S, [](float x){return std::cos(x);},  1e-6);
    test_un("ForeachCosh", aclnnForeachCoshGetWorkspaceSize,  aclnnForeachCosh,  S, [](float x){return std::cosh(x);}, 1e-5);
    test_un("ForeachErf", aclnnForeachErfGetWorkspaceSize,   aclnnForeachErf,   S, [](float x){return std::erf(x);},  1e-5);
    test_un("ForeachErfc", aclnnForeachErfcGetWorkspaceSize,  aclnnForeachErfc,  S, [](float x){return std::erfc(x);}, 1e-5);
    test_un("ForeachExp", aclnnForeachExpGetWorkspaceSize,   aclnnForeachExp,   S, [](float x){return std::exp(x);},  1e-5);
    test_un("ForeachExpm1", aclnnForeachExpm1GetWorkspaceSize, aclnnForeachExpm1, S, [](float x){return std::expm1(x);},1e-5);
    test_un("ForeachLog", aclnnForeachLogGetWorkspaceSize,   aclnnForeachLog,   A, [](float x){return std::log(x);},  1e-6);
    test_un("ForeachLog10", aclnnForeachLog10GetWorkspaceSize, aclnnForeachLog10, A, [](float x){return std::log10(x);},1e-6);
    test_un("ForeachLog1p", aclnnForeachLog1pGetWorkspaceSize, aclnnForeachLog1p, A, [](float x){return std::log1p(x);},1e-6);
    test_un("ForeachLog2", aclnnForeachLog2GetWorkspaceSize,  aclnnForeachLog2,  A, [](float x){return std::log2(x);}, 1e-6);
    test_un("ForeachNeg", aclnnForeachNegGetWorkspaceSize,   aclnnForeachNeg,   S, [](float x){return -x;},           0.0);
    test_un("ForeachReciprocal", aclnnForeachReciprocalGetWorkspaceSize, aclnnForeachReciprocal, A, [](float x){return 1.0f/x;}, 1e-6);
    test_un("ForeachSigmoid",aclnnForeachSigmoidGetWorkspaceSize,aclnnForeachSigmoid,S,[](float x){return 1.0f/(1.0f+std::exp(-x));},1e-6);
    test_un("ForeachSign", aclnnForeachSignGetWorkspaceSize,  aclnnForeachSign,  S, [](float x){return (float)((x>0)-(x<0));}, 0.0);
    test_un("ForeachSin", aclnnForeachSinGetWorkspaceSize,   aclnnForeachSin,   S, [](float x){return std::sin(x);},  1e-6);
    test_un("ForeachSinh", aclnnForeachSinhGetWorkspaceSize,  aclnnForeachSinh,  S, [](float x){return std::sinh(x);}, 1e-5);
    test_un("ForeachTan", aclnnForeachTanGetWorkspaceSize,   aclnnForeachTan,   A, [](float x){return std::tan(x);},  1e-6);
    test_un("ForeachTanh", aclnnForeachTanhGetWorkspaceSize,  aclnnForeachTanh,  S, [](float x){return std::tanh(x);}, 1e-6);
    test_un("ForeachRoundOffNumber", aclnnForeachRoundOffNumberGetWorkspaceSize,   aclnnForeachRoundOffNumber,   S, [](float x){return std::rint(x);}, 0.0);
    test_un("ForeachRoundOffNumberV2", aclnnForeachRoundOffNumberV2GetWorkspaceSize, aclnnForeachRoundOffNumberV2, S, [](float x){return std::rint(x);}, 0.0);

    // ============ single scalar (13) ============
    test_scalar("ForeachAddScalar",     aclnnForeachAddScalarGetWorkspaceSize,     aclnnForeachAddScalar,     S, 2.5f, [](float x,float s){return x+s;}, 1e-6);
    test_scalar("ForeachAddScalarV2",   aclnnForeachAddScalarV2GetWorkspaceSize,   aclnnForeachAddScalarV2,   S, 2.5f, [](float x,float s){return x+s;}, 1e-6);
    test_scalar("ForeachSubScalar",     aclnnForeachSubScalarGetWorkspaceSize,     aclnnForeachSubScalar,     S, 1.3f, [](float x,float s){return x-s;}, 1e-6);
    test_scalar("ForeachSubScalarV2",   aclnnForeachSubScalarV2GetWorkspaceSize,   aclnnForeachSubScalarV2,   S, 1.3f, [](float x,float s){return x-s;}, 1e-6);
    test_scalar("ForeachMulScalarV2",   aclnnForeachMulScalarV2GetWorkspaceSize,   aclnnForeachMulScalarV2,   S, 3.0f, [](float x,float s){return x*s;}, 1e-6);
    test_scalar("ForeachDivScalar",     aclnnForeachDivScalarGetWorkspaceSize,     aclnnForeachDivScalar,     S, 4.0f, [](float x,float s){return x/s;}, 1e-6);
    test_scalar("ForeachDivScalarV2",   aclnnForeachDivScalarV2GetWorkspaceSize,   aclnnForeachDivScalarV2,   S, 4.0f, [](float x,float s){return x/s;}, 1e-6);
    test_scalar("ForeachMaximumScalar", aclnnForeachMaximumScalarGetWorkspaceSize, aclnnForeachMaximumScalar, S, 0.5f, [](float x,float s){return x>s?x:s;}, 0.0);
    test_scalar("ForeachMaximumScalarV2",aclnnForeachMaximumScalarV2GetWorkspaceSize,aclnnForeachMaximumScalarV2,S,0.5f,[](float x,float s){return x>s?x:s;}, 0.0);
    test_scalar("ForeachMinimumScalar", aclnnForeachMinimumScalarGetWorkspaceSize, aclnnForeachMinimumScalar, S, 0.5f, [](float x,float s){return x<s?x:s;}, 0.0);
    test_scalar("ForeachMinimumScalarV2",aclnnForeachMinimumScalarV2GetWorkspaceSize,aclnnForeachMinimumScalarV2,S,0.5f,[](float x,float s){return x<s?x:s;}, 0.0);
    test_scalar("ForeachPowScalar",     aclnnForeachPowScalarGetWorkspaceSize,     aclnnForeachPowScalar,     A, 2.0f, [](float x,float s){return std::pow(x,s);}, 1e-5);
    test_scalar("ForeachPowScalarV2",   aclnnForeachPowScalarV2GetWorkspaceSize,   aclnnForeachPowScalarV2,   A, 2.0f, [](float x,float s){return std::pow(x,s);}, 1e-5);

    // ============ scalar list (6) — per-tensor scalars {1.0, -0.5, 2.0} ============
    const std::vector<float> SL = {1.0f, -0.5f, 2.0f};
    test_scalarlist("ForeachSubScalarList",     aclnnForeachSubScalarListGetWorkspaceSize,     aclnnForeachSubScalarList,     S, SL, [](float x,float s){return x-s;}, 1e-6);
    test_scalarlist("ForeachMulScalarList",     aclnnForeachMulScalarListGetWorkspaceSize,     aclnnForeachMulScalarList,     S, SL, [](float x,float s){return x*s;}, 1e-6);
    test_scalarlist("ForeachDivScalarList",     aclnnForeachDivScalarListGetWorkspaceSize,     aclnnForeachDivScalarList,     S, {2.f,4.f,0.5f}, [](float x,float s){return x/s;}, 1e-6);
    test_scalarlist("ForeachMaximumScalarList", aclnnForeachMaximumScalarListGetWorkspaceSize, aclnnForeachMaximumScalarList, S, SL, [](float x,float s){return x>s?x:s;}, 0.0);
    test_scalarlist("ForeachMinimumScalarList", aclnnForeachMinimumScalarListGetWorkspaceSize, aclnnForeachMinimumScalarList, S, SL, [](float x,float s){return x<s?x:s;}, 0.0);
    test_scalarlist("ForeachPowScalarList",     aclnnForeachPowScalarListGetWorkspaceSize,     aclnnForeachPowScalarList,     A, {2.f,3.f,0.5f}, [](float x,float s){return std::pow(x,s);}, 1e-5);

    // ============ list ∘ list (7) ============
    test_list("ForeachSubList",     aclnnForeachSubListGetWorkspaceSize,     aclnnForeachSubList,     S, B, [](float a,float b){return a-b;}, 1e-6);
    test_list("ForeachMulList",     aclnnForeachMulListGetWorkspaceSize,     aclnnForeachMulList,     S, B, [](float a,float b){return a*b;}, 1e-6);
    test_list("ForeachDivList",     aclnnForeachDivListGetWorkspaceSize,     aclnnForeachDivList,     S, B, [](float a,float b){return a/b;}, 1e-6);
    test_list("ForeachMaximumList", aclnnForeachMaximumListGetWorkspaceSize, aclnnForeachMaximumList, S, B, [](float a,float b){return a>b?a:b;}, 0.0);
    test_list("ForeachMinimumList", aclnnForeachMinimumListGetWorkspaceSize, aclnnForeachMinimumList, S, B, [](float a,float b){return a<b?a:b;}, 0.0);
    test_list("ForeachPowList",     aclnnForeachPowListGetWorkspaceSize,     aclnnForeachPowList,     A, B, [](float a,float b){return std::pow(a,b);}, 1e-5);
    // SubListV2: a - alpha*b
    {
        float al = 2.0f; auto alpha = aclCreateScalar(&al, ACL_FLOAT);
        FList x = make_list(S), y = make_list(B), o = zeros_like(S);
        exec2([&](uint64_t *w, aclOpExecutor **ex){ return aclnnForeachSubListV2GetWorkspaceSize(x.list, y.list, alpha, o.list, w, ex); }, aclnnForeachSubListV2);
        auto ref = S; for (size_t i=0;i<ref.size();i++) for (size_t j=0;j<ref[i].size();j++) ref[i][j] = S[i][j] - al*B[i][j];
        report("ForeachSubListV2", list_err(o, ref), 1e-6);
    }

    // ============ addcmul / addcdiv (7) ============
    // out_i = x_i + s * (t1_i (* or /) t2_i)
    {
        float s = 0.5f; auto sc = aclCreateScalar(&s, ACL_FLOAT);
        FList x = make_list(A), t1 = make_list(B), t2 = make_list(C), o = zeros_like(A);
        exec2([&](uint64_t *w, aclOpExecutor **ex){ return aclnnForeachAddcmulScalarV2GetWorkspaceSize(x.list,t1.list,t2.list,sc,o.list,w,ex); }, aclnnForeachAddcmulScalarV2);
        auto ref = A; for (size_t i=0;i<ref.size();i++) for (size_t j=0;j<ref[i].size();j++) ref[i][j] = A[i][j] + s*B[i][j]*C[i][j];
        report("ForeachAddcmulScalarV2", list_err(o, ref), 1e-6);
    }
    {
        float s = 0.5f; auto sc = aclCreateScalar(&s, ACL_FLOAT);
        FList x = make_list(A), t1 = make_list(B), t2 = make_list(C), o = zeros_like(A);
        exec2([&](uint64_t *w, aclOpExecutor **ex){ return aclnnForeachAddcdivScalarGetWorkspaceSize(x.list,t1.list,t2.list,sc,o.list,w,ex); }, aclnnForeachAddcdivScalar);
        auto ref = A; for (size_t i=0;i<ref.size();i++) for (size_t j=0;j<ref[i].size();j++) ref[i][j] = A[i][j] + s*B[i][j]/C[i][j];
        report("ForeachAddcdivScalar", list_err(o, ref), 1e-6);
    }
    {
        float s = 0.75f; auto sc = aclCreateScalar(&s, ACL_FLOAT);
        FList x = make_list(A), t1 = make_list(B), t2 = make_list(C), o = zeros_like(A);
        exec2([&](uint64_t *w, aclOpExecutor **ex){ return aclnnForeachAddcdivScalarV2GetWorkspaceSize(x.list,t1.list,t2.list,sc,o.list,w,ex); }, aclnnForeachAddcdivScalarV2);
        auto ref = A; for (size_t i=0;i<ref.size();i++) for (size_t j=0;j<ref[i].size();j++) ref[i][j] = A[i][j] + s*B[i][j]/C[i][j];
        report("ForeachAddcdivScalarV2", list_err(o, ref), 1e-6);
    }
    // per-tensor scalar list {0.5, 1.0, -0.5}
    const std::vector<float> AC = {0.5f, 1.0f, -0.5f};
    auto addc_list_test = [&](const char *name, auto ws, FRun run, bool isdiv) {
        DevBuf sb(AC.size()*sizeof(float)); sb.up(AC.data());
        auto stt = mk({(int64_t)AC.size()}, ACL_FLOAT, sb.p);
        FList x = make_list(A), t1 = make_list(B), t2 = make_list(C), o = zeros_like(A);
        exec2([&](uint64_t *w, aclOpExecutor **ex){ return ws(x.list,t1.list,t2.list,stt,o.list,w,ex); }, run);
        auto ref = A;
        for (size_t i=0;i<ref.size();i++) for (size_t j=0;j<ref[i].size();j++)
            ref[i][j] = A[i][j] + AC[i]*(isdiv ? B[i][j]/C[i][j] : B[i][j]*C[i][j]);
        report(name, list_err(o, ref), 1e-6);
    };
    addc_list_test("ForeachAddcmulScalarList", aclnnForeachAddcmulScalarListGetWorkspaceSize, aclnnForeachAddcmulScalarList, false);
    addc_list_test("ForeachAddcdivScalarList", aclnnForeachAddcdivScalarListGetWorkspaceSize, aclnnForeachAddcdivScalarList, true);
    addc_list_test("ForeachAddcmulList",       aclnnForeachAddcmulListGetWorkspaceSize,       aclnnForeachAddcmulList,       false);
    addc_list_test("ForeachAddcdivList",       aclnnForeachAddcdivListGetWorkspaceSize,       aclnnForeachAddcdivList,       true);

    return finish();
}
