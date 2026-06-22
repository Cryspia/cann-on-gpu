// In-place elementwise / activation family + a few out-of-place tensor ops (MaskedFillTensor,
// BitwiseAndTensorOut). Host compute over unified memory (contiguous, exact) after draining the stream.
#import "../internal.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>

namespace {
float *fp(const aclTensor *t) { return (float *)t->data + t->offset; }
uint8_t *u8(const aclTensor *t) { return (uint8_t *)t->data + t->offset; }
int32_t *i32(const aclTensor *t) { return (int32_t *)t->data + t->offset; }

aclnnStatus run_inplace(aclOpExecutor *e, aclrtStream stream) {
    auto *s = (AclStream *)stream; if (s && s->last) [s->last waitUntilCompleted];
    double a = e->dscalars.size() > 0 ? e->dscalars[0] : 0.0, b = e->dscalars.size() > 1 ? e->dscalars[1] : 0.0;
    const aclTensor *self = e->out ? e->out : e->a;
    int64_t n = self ? self->numel() : 0;
    switch (e->m) {
        case 0: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=std::exp(x[i]); break; }
        case 1: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=-x[i]; break; }
        case 2: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=1.f/x[i]; break; }
        case 3: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=x[i]>0?x[i]:0.f; break; }
        case 4: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=1.f/(1.f+std::exp(-x[i])); break; }
        case 5: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=std::tanh(x[i]); break; }
        case 6: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=x[i]>0?x[i]:(float)a*x[i]; break; }
        case 7: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=std::min(x[i],(float)a); break; }
        case 8: { float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=std::min(std::max(x[i],(float)a),(float)b); break; }
        case 9: { float *x = fp(self); const float *t1=fp(e->a),*t2=fp(e->b); for (int64_t i=0;i<n;++i) x[i]+= (float)a*t1[i]*t2[i]; break; }
        case 10:{ float *x = fp(self); const float *o=fp(e->a); for (int64_t i=0;i<n;++i) x[i]=std::max(x[i],o[i]); break; }
        case 11:{ float *x = fp(self); const float *en=fp(e->a); for (int64_t i=0;i<n;++i) x[i]+=(float)a*(en[i]-x[i]); break; }
        case 12:{ float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]+=(float)a; break; }              // adds (a = alpha*other)
        case 13:{ float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]*=(float)a; break; }              // muls
        case 14:{ float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=0.f; break; }
        case 15:{ float *x = fp(self); const float *o=fp(e->a); for (int64_t i=0;i<n;++i) x[i]+=(float)a*o[i]; break; }  // add/addv3
        case 16:{ float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=(float)a; break; }               // fill scalar
        case 17:{ float *x = fp(self); for (int64_t i=0;i<n;++i) x[i]=1.f; break; }
        case 18:{ float *x = fp(self); const float *o=fp(e->a); for (int64_t i=0;i<n;++i) x[i]=std::floor(x[i]/o[i]); break; }
        case 19:{ float *x = fp(self); int64_t R=self->viewDims[0], C=self->viewDims.size()>1?self->viewDims[1]:1, d=std::min(R,C); for (int64_t i=0;i<d;++i) x[i*C+i]=(float)a; break; }
        case 20:{ uint8_t *x = u8(self); for (int64_t i=0;i<n;++i) x[i]=x[i]?0:1; break; }
        case 21:{ const float *sf=fp(e->a); const uint8_t *m=u8(e->b); float val=*fp(e->c); float *o=fp(e->out); int64_t nn=e->a->numel(); for (int64_t i=0;i<nn;++i) o[i]=m[i]?val:sf[i]; break; }
        case 22:{ float *d=fp(e->out); const float *src=fp(e->a); for (int64_t i=0;i<n;++i) d[i]=src[i]; break; }
        case 23:{ const int32_t *x=i32(e->a),*o2=i32(e->b); int32_t *o=i32(e->out); int64_t nn=e->a->numel(); for (int64_t i=0;i<nn;++i) o[i]=x[i]&o2[i]; break; }
        default: return ACLNN_ERR_PARAM_INVALID;
    }
    return ACLNN_SUCCESS;
}
#define RUN(NAME) aclnnStatus NAME(void *w, uint64_t wss, aclOpExecutor *e, aclrtStream s) { \
    (void)w; (void)wss; if (!e) return ACLNN_ERR_PARAM_NULLPTR; aclnnStatus st; @autoreleasepool { st = run_inplace(e, s); } delete e; return st; }
} // namespace

extern "C" {
#define IP_UN(NAME, K) aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, uint64_t *ws, aclOpExecutor **ex) { \
    if(!self||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=K; e->out=self; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME)
#define IP_SC(NAME, K) aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclScalar *s0, uint64_t *ws, aclOpExecutor **ex) { \
    if(!self||!s0||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=K; e->out=self; e->dscalars={s0->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME)
#define IP_BIN(NAME, K) aclnnStatus NAME##GetWorkspaceSize(aclTensor *self, const aclTensor *other, uint64_t *ws, aclOpExecutor **ex) { \
    if(!self||!other||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=K; e->out=self; e->a=other; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(NAME)

IP_UN(aclnnInplaceExp,0) IP_UN(aclnnInplaceNeg,1) IP_UN(aclnnInplaceReciprocal,2) IP_UN(aclnnInplaceRelu,3)
IP_UN(aclnnInplaceSigmoid,4) IP_UN(aclnnInplaceTanh,5) IP_UN(aclnnInplaceZero,14) IP_UN(aclnnInplaceOne,17)
IP_UN(aclnnInplaceLogicalNot,20)
IP_SC(aclnnInplaceLeakyRelu,6) IP_SC(aclnnInplaceClampMax,7) IP_SC(aclnnInplaceMuls,13) IP_SC(aclnnInplaceFillScalar,16)
IP_BIN(aclnnInplaceClampMinTensor,10) IP_BIN(aclnnInplaceFloorDivide,18) IP_BIN(aclnnInplaceCopy,22)

aclnnStatus aclnnInplaceHardtanhGetWorkspaceSize(aclTensor *self, const aclScalar *lo, const aclScalar *hi, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!lo||!hi||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=8; e->out=self; e->dscalars={lo->v,hi->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceHardtanh)
aclnnStatus aclnnInplaceAddcmulGetWorkspaceSize(aclTensor *self, const aclTensor *t1, const aclTensor *t2, const aclScalar *v, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!t1||!t2||!v||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=9; e->out=self; e->a=t1; e->b=t2; e->dscalars={v->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceAddcmul)
aclnnStatus aclnnInplaceLerpsGetWorkspaceSize(aclTensor *self, const aclTensor *end, const aclScalar *w, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!end||!w||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=11; e->out=self; e->a=end; e->dscalars={w->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceLerps)
aclnnStatus aclnnInplaceAddsGetWorkspaceSize(aclTensor *self, const aclScalar *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!other||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=12; e->out=self; e->dscalars={(alpha?alpha->v:1.0)*other->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceAdds)
aclnnStatus aclnnInplaceAddGetWorkspaceSize(aclTensor *self, const aclTensor *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!other||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=15; e->out=self; e->a=other; e->dscalars={alpha?alpha->v:1.0}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceAdd)
aclnnStatus aclnnInplaceAddV3GetWorkspaceSize(aclTensor *self, const aclTensor *other, const aclScalar *alpha, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!other||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=15; e->out=self; e->a=other; e->dscalars={alpha?alpha->v:1.0}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceAddV3)
aclnnStatus aclnnInplaceFillDiagonalGetWorkspaceSize(aclTensor *self, const aclScalar *v, bool wrap, uint64_t *ws, aclOpExecutor **ex) {
    (void)wrap; if(!self||!v||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=19; e->out=self; e->dscalars={v->v}; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnInplaceFillDiagonal)
aclnnStatus aclnnMaskedFillTensorGetWorkspaceSize(const aclTensor *self, const aclTensor *mask, const aclTensor *value, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!mask||!value||!out||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=21; e->a=self; e->b=mask; e->c=value; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnMaskedFillTensor)
aclnnStatus aclnnBitwiseAndTensorOutGetWorkspaceSize(const aclTensor *self, const aclTensor *other, aclTensor *out, uint64_t *ws, aclOpExecutor **ex) {
    if(!self||!other||!out||!ws||!ex) return ACLNN_ERR_PARAM_NULLPTR; auto*e=new aclOpExecutor(); e->m=23; e->a=self; e->b=other; e->out=out; *ws=0; *ex=e; return ACLNN_SUCCESS; } RUN(aclnnBitwiseAndTensorOut)
} // extern "C"
