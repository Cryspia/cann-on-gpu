// FFT / signal (P17) cross-check: Fft vs naive DFT, Fft/Ifft round-trip, Rfft/Irfft round-trip. Interleaved complex.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

static void t_fft_forward() {
    const int n=8; auto c=randv(2*n,-1,1);   // interleaved complex
    std::vector<float> hz(2*n); DevBuf dx(2*n*4),dz(2*n*4); dx.up(c.data());
    aclTensor *tx=mk({n,2},ACL_FLOAT,dx.p),*tz=mk({n,2},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFftGetWorkspaceSize(tx,n,tz,w,e);}, aclnnFft);
    dz.down(hz.data()); double me=0,mr=0;
    for(int k=0;k<n;k++){ double re=0,im=0; for(int j=0;j<n;j++){ double ang=-2*M_PI*k*j/n; double cr=c[2*j],ci=c[2*j+1]; re+=cr*std::cos(ang)-ci*std::sin(ang); im+=cr*std::sin(ang)+ci*std::cos(ang); }
        me=std::max(me,(double)std::fabs(hz[2*k]-re)); me=std::max(me,(double)std::fabs(hz[2*k+1]-im)); mr=std::max({mr,std::fabs(re),std::fabs(im)}); }
    report("Fft vs DFT", me/(mr+1e-9), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tz);
}
static void t_fft_roundtrip() {
    const int n=16, B=3; auto c=randv(2*n*B,-1,1); std::vector<float> mid(2*n*B),hz(2*n*B);
    DevBuf dx(2*n*B*4),dm(2*n*B*4),dz(2*n*B*4); dx.up(c.data());
    aclTensor *tx=mk({B,n,2},ACL_FLOAT,dx.p),*tm=mk({B,n,2},ACL_FLOAT,dm.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFftGetWorkspaceSize(tx,n,tm,w,e);}, aclnnFft);
    aclTensor *tm2=mk({B,n,2},ACL_FLOAT,dm.p),*tz=mk({B,n,2},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIfftGetWorkspaceSize(tm2,n,tz,w,e);}, aclnnIfft);
    dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<2*n*B;i++){me=std::max(me,(double)std::fabs(hz[i]-c[i]));mr=std::max(mr,std::fabs((double)c[i]));}
    report("Fft/Ifft roundtrip", me/(mr+1e-9), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tm);aclDestroyTensor(tm2);aclDestroyTensor(tz);
}
static void t_rfft_roundtrip() {
    const int n=16, B=2; int nc=n/2+1; auto x=randv(n*B,-1,1); std::vector<float> spec(2*nc*B),hz(n*B);
    DevBuf dx(n*B*4),ds(2*nc*B*4),dz(n*B*4); dx.up(x.data());
    aclTensor *tx=mk({B,n},ACL_FLOAT,dx.p),*ts=mk({B,nc,2},ACL_FLOAT,ds.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRfftGetWorkspaceSize(tx,n,ts,w,e);}, aclnnRfft);
    aclTensor *ts2=mk({B,nc,2},ACL_FLOAT,ds.p),*tz=mk({B,n},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIrfftGetWorkspaceSize(ts2,n,tz,w,e);}, aclnnIrfft);
    dz.down(hz.data()); double me=0,mr=0; for(int i=0;i<n*B;i++){me=std::max(me,(double)std::fabs(hz[i]-x[i]));mr=std::max(mr,std::fabs((double)x[i]));}
    report("Rfft/Irfft roundtrip", me/(mr+1e-9), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(ts);aclDestroyTensor(ts2);aclDestroyTensor(tz);
}
int main(){ init(); srand(61); t_fft_forward(); t_fft_roundtrip(); t_rfft_roundtrip(); return finish(); }
