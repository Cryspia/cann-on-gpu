// Invariant test for TransData format conversions: convert(x) then convert_back == x (exact, since
// dims are chosen as multiples of the 16-tile so there is no padding). Validates the fractal/5HD layouts
// (incl. the NZ de-swizzle path) are self-consistent. No torch reference needed.
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
    std::string op=argv[1];
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    auto run=[&](auto getws,auto fn,aclTensor*in,aclTensor*o){ ws=0; void*wp=nullptr; CK(getws(in,o,&ws,&ex)); if(ws)wp=mal(ws); CK(fn(wp,ws,ex,s)); };
    double e=0;
    if(op=="nz"){
        const int64_t M=32,N=48; int64_t NN=M*N;
        std::vector<float> x(NN); for(int64_t i=0;i<NN;i++)x[i]=(float)std::sin(i*0.07+0.1);
        void*dx=upf(x),*dnz=mal(NN*4),*dy=mal(NN*4);
        aclTensor*tx=T({M,N},dx),*tnz=T({N/16,M/16,16,16},dnz),*ty=T({M,N},dy);
        run(aclnnTransDataND2NZGetWorkspaceSize,aclnnTransDataND2NZ,tx,tnz);
        run(aclnnTransDataNZ2NDGetWorkspaceSize,aclnnTransDataNZ2ND,tnz,ty);
        CK(aclrtSynchronizeStream(s)); e=ncmp(dnf(dy,NN),x);
    } else if(op=="fz"){
        const int64_t M=32,N=48; int64_t NN=M*N;
        std::vector<float> x(NN); for(int64_t i=0;i<NN;i++)x[i]=(float)std::cos(i*0.05+0.3);
        void*dx=upf(x),*dfz=mal(NN*4),*dy=mal(NN*4);
        aclTensor*tx=T({M,N},dx),*tfz=T({M/16,N/16,16,16},dfz),*ty=T({M,N},dy);
        run(aclnnTransDataND2FZGetWorkspaceSize,aclnnTransDataND2FZ,tx,tfz);
        run(aclnnTransDataFZ2NDGetWorkspaceSize,aclnnTransDataFZ2ND,tfz,ty);
        CK(aclrtSynchronizeStream(s)); e=ncmp(dnf(dy,NN),x);
    } else { // nc1hwc0 (C multiple of 16 -> exact)
        const int64_t Nn=2,Cc=16,Hh=3,Ww=4; int64_t NN=Nn*Cc*Hh*Ww;
        std::vector<float> x(NN); for(int64_t i=0;i<NN;i++)x[i]=(float)std::sin(i*0.09+0.2);
        void*dx=upf(x),*dm=mal(NN*4),*dy=mal(NN*4);
        aclTensor*tx=T({Nn,Cc,Hh,Ww},dx),*tm=T({Nn,Cc/16,Hh,Ww,16},dm),*ty=T({Nn,Cc,Hh,Ww},dy);
        run(aclnnTransDataNCHW2NC1HWC0GetWorkspaceSize,aclnnTransDataNCHW2NC1HWC0,tx,tm);
        run(aclnnTransDataNC1HWC0toNCHWGetWorkspaceSize,aclnnTransDataNC1HWC0toNCHW,tm,ty);
        CK(aclrtSynchronizeStream(s)); e=ncmp(dnf(dy,NN),x);
    }
    double tol=1e-6; printf("[%s] roundtrip normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
