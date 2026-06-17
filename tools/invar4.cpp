// Invariant test for FFT family: ifft(fft(x))==x (C2C) and irfft(rfft(x))==x (R2C/C2R), since the
// inverse transforms are 1/n normalized. Complex tensors are interleaved (real,imag) floats. cuFFT-backed.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double ncmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string op=argv[1]; const int64_t B=4, n=8;
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    double e=0, tol=1e-5;
    if(op=="c2c"){
        int64_t W=2*n, N=B*W;                       // interleaved complex
        std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.21+0.3);
        void*dx=upf(x),*dt=mal(N*4),*dy=mal(N*4);
        void*wp=nullptr; CK(aclnnFftGetWorkspaceSize(T({B,W},dx),n,T({B,W},dt),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnFft(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ws=0;wp=nullptr; CK(aclnnIfftGetWorkspaceSize(T({B,W},dt),n,T({B,W},dy),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnIfft(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,N),x);
    } else { // r2c/c2r round-trip
        int64_t Wc=2*(n/2+1), Nr=B*n, Nc=B*Wc;      // real [B,n] -> complex [B,2(n/2+1)] -> real [B,n]
        std::vector<float> x(Nr); for(int64_t i=0;i<Nr;i++)x[i]=(float)std::cos(i*0.17+0.2);
        void*dx=upf(x),*dt=mal(Nc*4),*dy=mal(Nr*4);
        void*wp=nullptr; CK(aclnnRfftGetWorkspaceSize(T({B,n},dx),n,T({B,Wc},dt),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnRfft(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ws=0;wp=nullptr; CK(aclnnIrfftGetWorkspaceSize(T({B,Wc},dt),n,T({B,n},dy),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnIrfft(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,Nr),x);
    }
    printf("[%s] roundtrip normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
