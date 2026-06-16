// Random / distribution extensions (P12) cross-check: statistical (sample mean / frequency vs theory).
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_randint() {
    const int n=1<<16; int64_t lo=5,hi=15; std::vector<int64_t> h(n); DevBuf d(n*8);
    aclTensor *t=mk({n},ACL_INT64,d.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRandIntGetWorkspaceSize(lo,hi,7,t,w,e);}, aclnnRandInt);
    d.down(h.data()); double m=0; bool range=true; for(int i=0;i<n;i++){m+=h[i]; if(h[i]<lo||h[i]>=hi)range=false;} m/=n;
    report("RandInt mean", range? std::fabs(m-(lo+hi-1)/2.0)/((lo+hi-1)/2.0) : 1.0, 2e-2); aclDestroyTensor(t);
}
static void t_dist(const char *name, double theoMean, double tol, std::function<aclnnStatus(aclTensor*,uint64_t*,aclOpExecutor**)> getws, aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)) {
    const int n=1<<18; std::vector<float> h(n); DevBuf d(n*4); aclTensor *t=mk({n},ACL_FLOAT,d.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return getws(t,w,e);}, r);
    d.down(h.data()); double m=0; for(int i=0;i<n;i++) m+=h[i]; m/=n;
    report(name, std::fabs(m-theoMean)/(std::fabs(theoMean)+1e-9), tol); aclDestroyTensor(t);
}
static void t_poisson() {
    const int n=1<<18; double lam=4.0; std::vector<int32_t> h(n); DevBuf d(n*4); aclTensor *t=mk({n},ACL_INT32,d.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPoissonGetWorkspaceSize(t,lam,7,w,e);}, aclnnPoisson);
    d.down(h.data()); double m=0; for(int i=0;i<n;i++) m+=h[i]; m/=n;
    report("Poisson mean", std::fabs(m-lam)/lam, 3e-2); aclDestroyTensor(t);
}
static void t_multinomial() {
    const int C=4, ns=1<<16; float probs[C]={0.1f,0.2f,0.3f,0.4f}; std::vector<int64_t> h(ns);
    DevBuf dp(C*4), do_(ns*8); dp.up(probs);
    aclTensor *tp=mk({1,C},ACL_FLOAT,dp.p),*to=mk({1,ns},ACL_INT64,do_.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMultinomialGetWorkspaceSize(tp,ns,7,to,w,e);}, aclnnMultinomial);
    do_.down(h.data()); std::vector<double> freq(C,0); for(int i=0;i<ns;i++) freq[h[i]]+=1.0/ns;
    double bad=0; for(int c=0;c<C;c++) bad=std::max(bad,std::fabs(freq[c]-probs[c]));
    report("Multinomial freq", bad, 1e-2); aclDestroyTensor(tp); aclDestroyTensor(to);
}
int main() {
    init(); srand(31);
    t_randint();
    t_dist("Exponential lam=2 mean", 0.5, 2e-2, [](aclTensor*t,uint64_t*w,aclOpExecutor**e){return aclnnExponentialGetWorkspaceSize(t,2.0,7,w,e);}, aclnnExponential);
    t_dist("Geometric p=0.25 mean", 4.0, 3e-2, [](aclTensor*t,uint64_t*w,aclOpExecutor**e){return aclnnGeometricGetWorkspaceSize(t,0.25,7,w,e);}, aclnnGeometric);
    t_dist("LogNormal(0,0.5) mean", std::exp(0.0+0.5*0.25), 3e-2, [](aclTensor*t,uint64_t*w,aclOpExecutor**e){return aclnnLogNormalGetWorkspaceSize(t,0.0,0.5,7,w,e);}, aclnnLogNormal);
    // Normal family: mean/std scalar+tensor combos
    t_dist("NormalFloatFloat mean", 5.0, 2e-2, [](aclTensor*t,uint64_t*w,aclOpExecutor**e){return aclnnNormalFloatFloatGetWorkspaceSize(5.0,2.0,7,t,w,e);}, aclnnNormalFloatFloat);
    { const int n=1<<18; std::vector<float> mean(n,3.0f), sd(n,1.0f); std::vector<float> h(n);
      DevBuf dm(n*4),ds(n*4),dz(n*4); dm.up(mean.data()); ds.up(sd.data());
      auto tm=mk({n},ACL_FLOAT,dm.p),ts=mk({n},ACL_FLOAT,ds.p),tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNormalTensorTensorGetWorkspaceSize(tm,ts,7,tz,w,e);}, aclnnNormalTensorTensor);
      dz.down(h.data()); double m=0; for(int i=0;i<n;i++) m+=h[i]; m/=n;
      report("NormalTensorTensor mean", std::fabs(m-3.0)/3.0, 2e-2); aclDestroyTensor(tm);aclDestroyTensor(ts);aclDestroyTensor(tz); }
    t_poisson();
    t_multinomial();
    // Cauchy: heavy-tailed (mean undefined); just check it runs and median ~ param via fraction within sigma
    { const int n=1<<18; double med=1.0,sig=2.0; std::vector<float> h(n); DevBuf d(n*4); aclTensor *t=mk({n},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCauchyGetWorkspaceSize(t,med,sig,7,w,e);}, aclnnCauchy);
      d.down(h.data()); int64_t below=0; for(int i=0;i<n;i++) if(h[i]<med) below++;
      report("Cauchy median split", std::fabs((double)below/n-0.5), 2e-2); aclDestroyTensor(t); }
    return finish();
}
