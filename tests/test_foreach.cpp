// Cross-check for the foreach op family (ops-nn / foreach): each op applies one elementwise
// operation across every tensor in a list. References are computed on the CPU.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
using namespace hn;

namespace {

// A list of fp32 tensors held on device, plus host copies of the original data.
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

std::vector<float> dl(const DevBuf *b, int64_t n) { std::vector<float> v(n); b->down(v.data()); return v; }

// max |got-ref| / max|ref| over a whole list
double list_err(const FList &got, const std::vector<std::vector<float>> &ref) {
    double me = 0, mr = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        auto g = dl(got.bufs[i], got.sizes[i]);
        for (size_t j = 0; j < ref[i].size(); j++) { me = std::max(me, (double)std::fabs(g[j] - ref[i][j])); mr = std::max(mr, (double)std::fabs(ref[i][j])); }
    }
    return me / (mr + 1e-9);
}

const std::vector<std::vector<float>> A = {{1.f, 2.f, 3.f, 4.f, 5.f}, {0.5f, 1.5f, 2.5f, 3.5f}};
const std::vector<std::vector<float>> B = {{2.f, 1.f, 0.5f, 2.f, 1.f}, {1.f, 2.f, 1.f, 0.5f}};

} // namespace

int main() {
    init();

    // ---- unary: sqrt ----
    {
        auto x = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachSqrtGetWorkspaceSize(x.list, o.list, ws, ex); }, aclnnForeachSqrt);
        std::vector<std::vector<float>> ref = A;
        for (auto &v : ref) for (auto &e : v) e = std::sqrt(e);
        report("ForeachSqrt", list_err(o, ref), 1e-6);
    }

    // ---- single scalar: mul by 2.5 ----
    {
        float s = 2.5f; auto sc = aclCreateScalar(&s, ACL_FLOAT);
        auto x = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachMulScalarGetWorkspaceSize(x.list, sc, o.list, ws, ex); }, aclnnForeachMulScalar);
        std::vector<std::vector<float>> ref = A;
        for (auto &v : ref) for (auto &e : v) e *= s;
        report("ForeachMulScalar", list_err(o, ref), 1e-6);
    }

    // ---- per-tensor scalar list: add {10, -1} ----
    {
        std::vector<float> scal = {10.f, -1.f};
        DevBuf sb(scal.size() * sizeof(float)); sb.up(scal.data());
        auto st = mk({2}, ACL_FLOAT, sb.p);
        auto x = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachAddScalarListGetWorkspaceSize(x.list, st, o.list, ws, ex); }, aclnnForeachAddScalarList);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (auto &e : ref[i]) e += scal[i];
        report("ForeachAddScalarList", list_err(o, ref), 1e-6);
    }

    // ---- list + list ----
    {
        auto x = make_list(A), y = make_list(B), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachAddListGetWorkspaceSize(x.list, y.list, o.list, ws, ex); }, aclnnForeachAddList);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] += B[i][j];
        report("ForeachAddList", list_err(o, ref), 1e-6);
    }

    // ---- list + alpha*list (V2) ----
    {
        float a = 3.f; auto al = aclCreateScalar(&a, ACL_FLOAT);
        auto x = make_list(A), y = make_list(B), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachAddListV2GetWorkspaceSize(x.list, y.list, al, o.list, ws, ex); }, aclnnForeachAddListV2);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] += a * B[i][j];
        report("ForeachAddListV2", list_err(o, ref), 1e-6);
    }

    // ---- addcmul scalar: x + s*t1*t2 ----
    {
        float s = 0.5f; auto sc = aclCreateScalar(&s, ACL_FLOAT);
        auto x = make_list(A), t1 = make_list(B), t2 = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachAddcmulScalarGetWorkspaceSize(x.list, t1.list, t2.list, sc, o.list, ws, ex); }, aclnnForeachAddcmulScalar);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] += s * B[i][j] * A[i][j];
        report("ForeachAddcmulScalar", list_err(o, ref), 1e-6);
    }

    // ---- lerp scalar: x + w*(end-x) ----
    {
        float w = 0.25f; auto wt = aclCreateScalar(&w, ACL_FLOAT);
        auto x = make_list(A), end = make_list(B), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachLerpScalarGetWorkspaceSize(x.list, end.list, wt, o.list, ws, ex); }, aclnnForeachLerpScalar);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] += w * (B[i][j] - A[i][j]);
        report("ForeachLerpScalar", list_err(o, ref), 1e-6);
    }

    // ---- lerp list: per-element weight ----
    {
        auto x = make_list(A), end = make_list(B), wl = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachLerpListGetWorkspaceSize(x.list, end.list, wl.list, o.list, ws, ex); }, aclnnForeachLerpList);
        std::vector<std::vector<float>> ref = A;
        for (size_t i = 0; i < ref.size(); i++) for (size_t j = 0; j < ref[i].size(); j++) ref[i][j] += A[i][j] * (B[i][j] - A[i][j]);
        report("ForeachLerpList", list_err(o, ref), 1e-6);
    }

    // ---- pow(scalar, tensor): base^x ----
    {
        float base = 2.f; auto sc = aclCreateScalar(&base, ACL_FLOAT);
        auto x = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachPowScalarAndTensorGetWorkspaceSize(sc, x.list, o.list, ws, ex); }, aclnnForeachPowScalarAndTensor);
        std::vector<std::vector<float>> ref = A;
        for (auto &v : ref) for (auto &e : v) e = std::pow(base, e);
        report("ForeachPowScalarAndTensor", list_err(o, ref), 1e-5);
    }

    // ---- copy ----
    {
        auto x = make_list(A), o = make_list({std::vector<float>(5), std::vector<float>(4)});
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachCopyGetWorkspaceSize(x.list, o.list, ws, ex); }, aclnnForeachCopy);
        report("ForeachCopy", list_err(o, A), 0.0);
    }

    // ---- zero (in place) ----
    {
        auto x = make_list(A);
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachZeroInplaceGetWorkspaceSize(x.list, ws, ex); }, aclnnForeachZeroInplace);
        report("ForeachZeroInplace", list_err(x, {std::vector<float>(5, 0.f), std::vector<float>(4, 0.f)}), 0.0);
    }

    // ---- norm: L2 per tensor → 1-D out ----
    {
        auto x = make_list(A);
        float p = 2.f; auto pp = aclCreateScalar(&p, ACL_FLOAT);
        DevBuf ob(2 * sizeof(float));
        auto ot = mk({2}, ACL_FLOAT, ob.p);
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachNormGetWorkspaceSize(x.list, pp, ot, ws, ex); }, aclnnForeachNorm);
        auto got = dl(&ob, 2);
        double me = 0, mr = 0;
        for (size_t i = 0; i < A.size(); i++) { double ss = 0; for (float e : A[i]) ss += (double)e * e; double r = std::sqrt(ss); me = std::max(me, std::fabs(got[i] - r)); mr = std::max(mr, std::fabs(r)); }
        report("ForeachNorm", me / (mr + 1e-9), 1e-6);
    }

    // ---- non-finite check + unscale (in place) ----
    {
        std::vector<std::vector<float>> data = {{2.f, 4.f, 6.f}, {8.f, INFINITY, 10.f}};
        auto x = make_list(data);
        float inv = 0.5f; DevBuf ib(sizeof(float)); ib.up(&inv);
        auto it = mk({1}, ACL_FLOAT, ib.p);
        float zero = 0.f; DevBuf fb(sizeof(float)); fb.up(&zero);
        auto ft = mk({1}, ACL_FLOAT, fb.p);
        exec2([&](uint64_t *ws, aclOpExecutor **ex) { return aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(x.list, ft, it, ws, ex); }, aclnnForeachNonFiniteCheckAndUnscale);
        auto found = dl(&fb, 1);
        auto g0 = dl(x.bufs[0], 3);
        double err = std::fabs(found[0] - 1.0);                    // inf present → flag set
        for (int j = 0; j < 3; j++) err = std::max(err, (double)std::fabs(g0[j] - data[0][j] * inv));  // finite tensor scaled
        report("ForeachNonFiniteCheckAndUnscale", err, 1e-6);
    }

    return finish();
}
