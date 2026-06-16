// Demonstrates harness.h engineering + golden bridge: test cases only declare data/op/reference, all boilerplate lives in harness.
// Reference source: if CANN_GOLDEN_DIR is set and <key>.bin (produced by cannsim/explorer) is found, cross-check against Ascend golden;
//   otherwise fall back to CPU reference. Source tag [golden]/[cpu] is printed after the case name.
#include "harness.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"
using namespace hn;

// Element-wise Add (harness-driven): golden preferred, CPU fallback
static void case_add(int64_t n) {
    auto a = randv(n, -2, 2), b = randv(n, -2, 2);
    DevBuf da(n*4), db(n*4), dz(n*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p), *tb = mk({n}, ACL_FLOAT, db.p), *tz = mk({n}, ACL_FLOAT, dz.p);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnAddGetWorkspaceSize(ta, tb, nullptr, tz, w, e); }, aclnnAdd);
    std::vector<float> got(n); dz.down(got.data());
    std::string src;
    std::string key = "add_n" + std::to_string(n);
    auto ref = golden_or_cpu(key, [&]{ std::vector<double> r(n); for (int64_t i=0;i<n;i++) r[i]=(double)a[i]+b[i]; return r; }, src);
    report("Add " + src, norm_err(got, ref), 1e-6);
    save_golden(key, got);   // Demo: save this run's output as golden (in practice golden should come from cannsim/explorer)
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// ReduceSum over all elements (harness-driven)
static void case_reducesum(int64_t n) {
    auto a = randv(n, -1, 1);
    DevBuf da(n*4), dz(4); da.up(a.data());
    int64_t od[1] = {1};
    aclTensor *ta = mk({n}, ACL_FLOAT, da.p);
    aclTensor *tz = aclCreateTensor(od, 1, ACL_FLOAT, nullptr, 0, ACL_FORMAT_ND, od, 1, dz.p);
    int64_t d0[1] = {0}; aclIntArray *dim = aclCreateIntArray(d0, 1);
    exec2([&](uint64_t *w, aclOpExecutor **e) { return aclnnReduceSumGetWorkspaceSize(ta, dim, false, ACL_FLOAT, tz, w, e); }, aclnnReduceSum);
    std::vector<float> got(1); dz.down(got.data());
    std::string src;
    auto ref = golden_or_cpu("reducesum_n" + std::to_string(n), [&]{ double s=0; for (auto x:a) s+=x; return std::vector<double>{s}; }, src);
    report("ReduceSum " + src, norm_err(got, ref), 1e-5);
    aclDestroyIntArray(dim); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

int main() {
    init(); srand(21);
    case_add(1 << 16);
    case_reducesum(1 << 20);
    return finish();
}
