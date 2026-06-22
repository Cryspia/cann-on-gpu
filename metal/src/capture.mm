// aclmdlRICapture*: stream graph capture/replay. The Metal backend has no native graph object, so during
// capture we record a tape of (Execute fn, deep-copied executor) and re-run it on ExecuteAsync. The deep
// copy clones each tensor's metadata but KEEPS its data pointer (unified memory persists for the lifetime of
// the allocation), so replay reads whatever the input buffers currently hold — matching CUDA-graph recompute
// semantics (the test mutates inputs between replays and expects recomputation). Only ops that call the
// aclCaptureRecord() hook are recordable; that covers Add/Silu (elementwise RUN_DECL), Matmul and RmsNorm —
// the ops the decode-step graph test captures.
#import "internal.h"
#include "acl/acl.h"
#include <vector>
#include <tuple>

namespace {
struct ShimGraph {
    std::vector<aclTensor *> tensors;        // cloned tensors (metadata copied; data pointer shared with original)
    std::vector<aclOpExecutor *> tmpls;      // cloned executor templates
    std::vector<aclTensorList *> tlists;     // cloned tensor lists
    std::vector<std::tuple<AclExecFn, void *, uint64_t, aclOpExecutor *>> tape;
};
aclTensor *clone_tensor(ShimGraph *g, const aclTensor *t) {
    if (!t) return nullptr; auto *c = new aclTensor(*t); g->tensors.push_back(c); return c;
}
// Deep-clone an executor: copy all scalar/vector state, then redirect every tensor pointer to a fresh clone.
aclOpExecutor *clone_exec(ShimGraph *g, const aclOpExecutor *e) {
    auto *c = new aclOpExecutor(*e);
    c->a = clone_tensor(g, e->a); c->b = clone_tensor(g, e->b); c->c = clone_tensor(g, e->c);
    c->out = clone_tensor(g, e->out); c->out2 = clone_tensor(g, e->out2);
    c->mean = clone_tensor(g, e->mean); c->rstd = clone_tensor(g, e->rstd); c->mask = clone_tensor(g, e->mask);
    for (auto &p : c->inputs) p = clone_tensor(g, p);
    for (int i = 0; i < 4; ++i)
        if (e->tl[i]) { auto *nl = new aclTensorList(*e->tl[i]); for (auto &tp : nl->v) tp = clone_tensor(g, tp); c->tl[i] = nl; g->tlists.push_back(nl); }
    c->owned.clear();                        // replay execs share these tensors; don't let Execute free them
    g->tmpls.push_back(c);
    return c;
}
} // namespace

// C++ linkage (matches the declaration in internal.h).
bool aclCaptureRecord(aclrtStream s, AclExecFn fn, aclOpExecutor *e, void *ws, uint64_t wss) {
    auto *st = (AclStream *)s;
    if (!st || !st->capturing) return false;
    auto *g = (ShimGraph *)st->capgraph;
    auto *tmpl = clone_exec(g, e);
    g->tape.emplace_back(fn, ws, wss, tmpl);
    for (auto *t : e->owned) delete t;        // mirror the normal Execute cleanup of the original executor
    delete e;
    return true;
}

extern "C" {

aclError aclmdlRICaptureBegin(aclrtStream stream, aclmdlRICaptureMode mode) {
    (void)mode; auto *st = (AclStream *)stream; if (!st) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    if (st->last) [st->last waitUntilCompleted];           // drain pending work before recording
    st->capturing = true; st->capgraph = new ShimGraph(); return ACL_SUCCESS;
}

aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus *status, aclmdlRI *modelRI) {
    auto *st = (AclStream *)stream;
    if (status) *status = (st && st->capturing) ? ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE : ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    if (modelRI) *modelRI = nullptr;
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureEnd(aclrtStream stream, aclmdlRI *modelRI) {
    auto *st = (AclStream *)stream; if (!st || !st->capturing) return (aclError)ACLNN_ERR_RUNTIME_ERROR;
    st->capturing = false; auto *g = (ShimGraph *)st->capgraph; st->capgraph = nullptr;
    if (modelRI) *modelRI = (aclmdlRI)g;
    return ACL_SUCCESS;
}

aclError aclmdlRIExecuteAsync(aclmdlRI modelRI, aclrtStream stream) {
    auto *g = (ShimGraph *)modelRI; if (!g) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    for (auto &entry : g->tape) {
        AclExecFn fn = std::get<0>(entry); void *ws = std::get<1>(entry); uint64_t wss = std::get<2>(entry);
        aclOpExecutor *tmpl = std::get<3>(entry);
        auto *e2 = new aclOpExecutor(*tmpl); e2->owned.clear();   // shares tmpl's tensors; Execute deletes only e2
        fn(ws, wss, e2, stream);                                  // capturing==false now, so it runs for real
    }
    return ACL_SUCCESS;
}

aclError aclmdlRIExecute(aclmdlRI modelRI, int32_t timeout) { (void)timeout; return aclmdlRIExecuteAsync(modelRI, nullptr); }

aclError aclmdlRIDestroy(aclmdlRI modelRI) {
    auto *g = (ShimGraph *)modelRI; if (!g) return ACL_SUCCESS;
    for (auto *e : g->tmpls) delete e;
    for (auto *l : g->tlists) delete l;
    for (auto *t : g->tensors) delete t;
    delete g;
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode *mode) { (void)mode; return ACL_SUCCESS; }

} // extern "C"
