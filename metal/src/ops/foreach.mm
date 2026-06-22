// foreach multi-tensor family: one elementwise op applied across every tensor in a list.
// Done host-side over unified memory (small lists; exact) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>

namespace {
float *hp(const aclTensor *t) { return (float *)t->data + t->offset; }
const aclTensorList *L(aclOpExecutor *e, int i) { return e->tl[i]; }
int64_t LN(const aclTensorList *l) { return (int64_t)l->v.size(); }
const aclTensor *LT(const aclTensorList *l, int64_t i) { return l->v[i]; }

aclnnStatus run_foreach(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];   // drain pending GPU writes
    double s0 = e->dscalars.size() > 0 ? e->dscalars[0] : 0.0;
    switch (e->m) {
        case 0: { auto *in = L(e,0), *o = L(e,1); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=std::sqrt(a[j]); } break; }
        case 1: { auto *in = L(e,0), *o = L(e,1); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=a[j]*(float)s0; } break; }
        case 2: { auto *in = L(e,0), *o = L(e,1); const float*sc=hp(e->a); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=a[j]+sc[t]; } break; }
        case 3: { auto *la=L(e,0),*lb=L(e,1),*o=L(e,2); for (int64_t t=0;t<LN(la);++t){ const float*a=hp(LT(la,t)),*b=hp(LT(lb,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(la,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=a[j]+b[j]; } break; }
        case 4: { auto *la=L(e,0),*lb=L(e,1),*o=L(e,2); for (int64_t t=0;t<LN(la);++t){ const float*a=hp(LT(la,t)),*b=hp(LT(lb,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(la,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=a[j]+(float)s0*b[j]; } break; }
        case 5: { auto *lx=L(e,0),*l1=L(e,1),*l2=L(e,2),*o=L(e,3); for (int64_t t=0;t<LN(lx);++t){ const float*x=hp(LT(lx,t)),*t1=hp(LT(l1,t)),*t2=hp(LT(l2,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(lx,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=x[j]+(float)s0*t1[j]*t2[j]; } break; }
        case 6: { auto *lx=L(e,0),*le=L(e,1),*o=L(e,2); for (int64_t t=0;t<LN(lx);++t){ const float*x=hp(LT(lx,t)),*en=hp(LT(le,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(lx,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=x[j]+(float)s0*(en[j]-x[j]); } break; }
        case 7: { auto *lx=L(e,0),*le=L(e,1),*lw=L(e,2),*o=L(e,3); for (int64_t t=0;t<LN(lx);++t){ const float*x=hp(LT(lx,t)),*en=hp(LT(le,t)),*w=hp(LT(lw,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(lx,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=x[j]+w[j]*(en[j]-x[j]); } break; }
        case 8: { auto *in=L(e,0),*o=L(e,1); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=std::pow((float)s0,a[j]); } break; }
        case 9: { auto *in=L(e,0),*o=L(e,1); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); float*y=hp((aclTensor*)LT(o,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=a[j]; } break; }
        case 10: { auto *in=L(e,0); for (int64_t t=0;t<LN(in);++t){ float*y=hp((aclTensor*)LT(in,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j) y[j]=0.f; } break; }
        case 11: { auto *in=L(e,0); float*out=hp(e->out); for (int64_t t=0;t<LN(in);++t){ const float*a=hp(LT(in,t)); int64_t n=LT(in,t)->numel(); double ss=0; for(int64_t j=0;j<n;++j) ss+=(double)a[j]*a[j]; out[t]=(float)std::sqrt(ss); } break; }
        case 12: { auto *in=L(e,0); float*found=hp(e->a); float inv=*hp(e->b); int bad=0; for (int64_t t=0;t<LN(in);++t){ float*y=hp((aclTensor*)LT(in,t)); int64_t n=LT(in,t)->numel(); for(int64_t j=0;j<n;++j){ if(!std::isfinite(y[j])) bad=1; y[j]*=inv; } } if(bad) found[0]=1.f; break; }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_foreach(e, s); } delete e; return st; }
} // namespace

extern "C" {
aclnnStatus aclnnForeachSqrtGetWorkspaceSize(const aclTensorList *in, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=0; e->tl[0]=in; e->tl[1]=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachSqrt)
aclnnStatus aclnnForeachMulScalarGetWorkspaceSize(const aclTensorList *in, const aclScalar *s, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=1; e->tl[0]=in; e->tl[1]=out; e->dscalars={s->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachMulScalar)
aclnnStatus aclnnForeachAddScalarListGetWorkspaceSize(const aclTensorList *in, const aclTensor *scalars, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=2; e->tl[0]=in; e->tl[1]=out; e->a=scalars; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachAddScalarList)
aclnnStatus aclnnForeachAddListGetWorkspaceSize(const aclTensorList *a, const aclTensorList *b, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=3; e->tl[0]=a; e->tl[1]=b; e->tl[2]=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachAddList)
aclnnStatus aclnnForeachAddListV2GetWorkspaceSize(const aclTensorList *a, const aclTensorList *b, const aclScalar *alpha, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=4; e->tl[0]=a; e->tl[1]=b; e->tl[2]=out; e->dscalars={alpha->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachAddListV2)
aclnnStatus aclnnForeachAddcmulScalarGetWorkspaceSize(const aclTensorList *x, const aclTensorList *t1, const aclTensorList *t2, const aclScalar *s, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=5; e->tl[0]=x; e->tl[1]=t1; e->tl[2]=t2; e->tl[3]=out; e->dscalars={s->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachAddcmulScalar)
aclnnStatus aclnnForeachLerpScalarGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclScalar *w, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=6; e->tl[0]=x; e->tl[1]=end; e->tl[2]=out; e->dscalars={w->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachLerpScalar)
aclnnStatus aclnnForeachLerpListGetWorkspaceSize(const aclTensorList *x, const aclTensorList *end, const aclTensorList *w, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=7; e->tl[0]=x; e->tl[1]=end; e->tl[2]=w; e->tl[3]=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachLerpList)
aclnnStatus aclnnForeachPowScalarAndTensorGetWorkspaceSize(const aclScalar *base, const aclTensorList *x, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=8; e->tl[0]=x; e->tl[1]=out; e->dscalars={base->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachPowScalarAndTensor)
aclnnStatus aclnnForeachCopyGetWorkspaceSize(const aclTensorList *in, const aclTensorList *out, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=9; e->tl[0]=in; e->tl[1]=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachCopy)
aclnnStatus aclnnForeachZeroInplaceGetWorkspaceSize(const aclTensorList *in, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=10; e->tl[0]=in; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachZeroInplace)
aclnnStatus aclnnForeachNormGetWorkspaceSize(const aclTensorList *in, const aclScalar *p, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    (void)p; auto *e=new aclOpExecutor(); e->m=11; e->tl[0]=in; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachNorm)
aclnnStatus aclnnForeachNonFiniteCheckAndUnscaleGetWorkspaceSize(const aclTensorList *in, aclTensor *foundInf, const aclTensor *invScale, uint64_t *ws, aclOpExecutor **ex) {
    auto *e=new aclOpExecutor(); e->m=12; e->tl[0]=in; e->a=foundInf; e->b=invScale; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnForeachNonFiniteCheckAndUnscale)
} // extern "C"
