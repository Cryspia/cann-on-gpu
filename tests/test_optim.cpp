// Optimizer extensions (P11) cross-check: one update step vs CPU reference. Adam/Adagrad/Rmsprop/Momentum/Adamax/Adadelta/ClipGradNorm.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_adam() {
    const int n=256; double lr=0.01,b1=0.9,b2=0.999,eps=1e-8,wd=0.01; int step=3;
    auto p=randv(n,-1,1),m=randv(n,-0.1,0.1),v=randv(n,0,0.1),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),dm(n*4),dv(n*4),dg(n*4); dp.up(p.data());dm.up(m.data());dv.up(v.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tm=mk({n},ACL_FLOAT,dm.p),*tv=mk({n},ACL_FLOAT,dv.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdamGetWorkspaceSize(tp,tm,tv,tg,lr,b1,b2,eps,wd,step,w,e);}, aclnnApplyAdam);
    dp.down(hp.data()); double bc1=1-std::pow(b1,step),bc2=1-std::pow(b2,step),me=0,mr=0;
    for(int i=0;i<n;i++){double gi=g[i]+wd*p[i];double mi=b1*m[i]+(1-b1)*gi,vi=b2*v[i]+(1-b2)*gi*gi;double ref=p[i]-lr*(mi/bc1)/(std::sqrt(vi/bc2)+eps);me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyAdam", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(tg);
}
static void t_adagrad() {
    const int n=256; double lr=0.1,eps=1e-10,wd=0.0; auto p=randv(n,-1,1),ss=randv(n,0,0.5),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),ds(n*4),dg(n*4); dp.up(p.data());ds.up(ss.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tss=mk({n},ACL_FLOAT,ds.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdagradGetWorkspaceSize(tp,tss,tg,lr,eps,wd,w,e);}, aclnnApplyAdagrad);
    dp.down(hp.data()); double me=0,mr=0; for(int i=0;i<n;i++){double gi=g[i];double s=ss[i]+gi*gi;double ref=p[i]-lr*gi/(std::sqrt(s)+eps);me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyAdagrad", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tss);aclDestroyTensor(tg);
}
static void t_rmsprop() {
    const int n=256; double lr=0.01,alpha=0.99,eps=1e-8,wd=0; auto p=randv(n,-1,1),v=randv(n,0,0.5),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),dv(n*4),dg(n*4); dp.up(p.data());dv.up(v.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tv=mk({n},ACL_FLOAT,dv.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyRmspropGetWorkspaceSize(tp,tv,tg,lr,alpha,eps,wd,w,e);}, aclnnApplyRmsprop);
    dp.down(hp.data()); double me=0,mr=0; for(int i=0;i<n;i++){double gi=g[i];double vi=alpha*v[i]+(1-alpha)*gi*gi;double ref=p[i]-lr*gi/(std::sqrt(vi)+eps);me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyRmsprop", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tv);aclDestroyTensor(tg);
}
static void t_momentum() {
    const int n=256; double lr=0.01,mu=0.9,wd=0,damp=0; auto p=randv(n,-1,1),buf=randv(n,-0.1,0.1),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),db(n*4),dg(n*4); dp.up(p.data());db.up(buf.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tb=mk({n},ACL_FLOAT,db.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyMomentumGetWorkspaceSize(tp,tb,tg,lr,mu,wd,damp,false,w,e);}, aclnnApplyMomentum);
    dp.down(hp.data()); double me=0,mr=0; for(int i=0;i<n;i++){double gi=g[i];double b=mu*buf[i]+gi;double ref=p[i]-lr*b;me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyMomentum", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tb);aclDestroyTensor(tg);
}
static void t_adamax() {
    const int n=256; double lr=0.002,b1=0.9,b2=0.999,eps=1e-8; int step=3; auto p=randv(n,-1,1),m=randv(n,-0.1,0.1),u=randv(n,0,0.5),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),dm(n*4),du(n*4),dg(n*4); dp.up(p.data());dm.up(m.data());du.up(u.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tm=mk({n},ACL_FLOAT,dm.p),*tu=mk({n},ACL_FLOAT,du.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdamaxGetWorkspaceSize(tp,tm,tu,tg,lr,b1,b2,eps,step,w,e);}, aclnnApplyAdamax);
    dp.down(hp.data()); double bc1=1-std::pow(b1,step),me=0,mr=0;
    for(int i=0;i<n;i++){double gi=g[i];double mi=b1*m[i]+(1-b1)*gi;double ui=std::max(b2*u[i],std::fabs(gi));double ref=p[i]-(lr/bc1)*mi/(ui+eps);me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyAdamax", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tu);aclDestroyTensor(tg);
}
static void t_adadelta() {
    const int n=256; double lr=1.0,rho=0.9,eps=1e-6,wd=0; auto p=randv(n,-1,1),sq=randv(n,0,0.3),ad=randv(n,0,0.3),g=randv(n,-1,1);
    std::vector<float> hp(n); DevBuf dp(n*4),dsq(n*4),dad(n*4),dg(n*4); dp.up(p.data());dsq.up(sq.data());dad.up(ad.data());dg.up(g.data());
    aclTensor *tp=mk({n},ACL_FLOAT,dp.p),*tsq=mk({n},ACL_FLOAT,dsq.p),*tad=mk({n},ACL_FLOAT,dad.p),*tg=mk({n},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdadeltaGetWorkspaceSize(tp,tsq,tad,tg,lr,rho,eps,wd,w,e);}, aclnnApplyAdadelta);
    dp.down(hp.data()); double me=0,mr=0;
    for(int i=0;i<n;i++){double gi=g[i];double s=rho*sq[i]+(1-rho)*gi*gi;double delta=std::sqrt(ad[i]+eps)/std::sqrt(s+eps)*gi;double ref=p[i]-lr*delta;me=std::max(me,std::fabs(hp[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ApplyAdadelta", me/(mr+1e-9), 1e-5); aclDestroyTensor(tp);aclDestroyTensor(tsq);aclDestroyTensor(tad);aclDestroyTensor(tg);
}
static void t_clip() {
    const int n=256; double maxNorm=1.0; auto g=randv(n,-1,1); std::vector<float> hg(n),hnorm(1);
    double tn=0; for(int i=0;i<n;i++) tn+=(double)g[i]*g[i]; tn=std::sqrt(tn);
    DevBuf dg(n*4),dnorm(4); dg.up(g.data());
    aclTensor *tg=mk({n},ACL_FLOAT,dg.p),*tnorm=mk({1},ACL_FLOAT,dnorm.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnClipGradNormGetWorkspaceSize(tg,maxNorm,tnorm,w,e);}, aclnnClipGradNorm);
    dg.down(hg.data()); dnorm.down(hnorm.data());
    double sc = tn>maxNorm? maxNorm/(tn+1e-6):1.0, me=0,mr=0;
    for(int i=0;i<n;i++){double ref=g[i]*sc;me=std::max(me,std::fabs(hg[i]-ref));mr=std::max(mr,std::fabs(ref));}
    report("ClipGradNorm scale", me/(mr+1e-9), 1e-5);
    report("ClipGradNorm total", std::fabs(hnorm[0]-tn)/(tn+1e-9), 1e-5);
    aclDestroyTensor(tg);aclDestroyTensor(tnorm);
}
int main() {
    init(); srand(29);
    t_adam(); t_adagrad(); t_rmsprop(); t_momentum(); t_adamax(); t_adadelta(); t_clip();
    return finish();
}
