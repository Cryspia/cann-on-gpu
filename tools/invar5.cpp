// Invariant test for Cast / fp6 packing.
//   cast_fp16/bf16: fp32 -> half/bf16 -> fp32 ≈ x within that format's precision.
//   fp6pack: Fp6Pack/Fp6Unpack is lossless bit-packing (4 fp6 -> 3 bytes), so unpack(pack(q))==q exactly;
//            we verify cast→fp6→pack→unpack→fp32 equals cast→fp6→fp32 (the lossy quant is identical, the
//            pack/unpack adds no error). fp6quant: cast fp32→fp6(E2M3)→fp32 within fp6 precision.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t N=64;
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double ncmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static uint64_t ws; static aclOpExecutor* ex;
static void castT(void* din,aclDataType sdt,std::vector<int64_t> dd,void* dout,aclDataType ddt,aclrtStream s){ ws=0; void*wp=nullptr; CK(aclnnCastGetWorkspaceSize(T(dd,din,sdt),ddt,T(dd,dout,ddt),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnCast(wp,ws,ex,s)); }

int main(int argc,char**argv){
    std::string op=argv[1];
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)(0.5*std::sin(i*0.19+0.3)+0.5);   // ~[0,1], positive
    void* dx=upf(x);
    double e=0, tol=0;
    if(op=="cast_fp16"||op=="cast_bf16"){
        aclDataType mid = (op=="cast_fp16")?ACL_FLOAT16:ACL_BF16;
        void* dmid=mal(N*2), *dy=mal(N*4);
        castT(dx,ACL_FLOAT,{N},dmid,mid,s); CK(aclrtSynchronizeStream(s));
        castT(dmid,mid,{N},dy,ACL_FLOAT,s); CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,N),x); tol=(op=="cast_fp16")?1e-3:1e-2;
    } else if(op=="fp6quant"){
        void* dq=mal(N), *dy=mal(N*4);
        castT(dx,ACL_FLOAT,{N},dq,ACL_FLOAT6_E2M3,s); CK(aclrtSynchronizeStream(s));
        castT(dq,ACL_FLOAT6_E2M3,{N},dy,ACL_FLOAT,s); CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,N),x); tol=0.12;   // E2M3: 3 mantissa bits
    } else { // fp6pack: pack/unpack adds zero error vs direct cast round-trip
        void* dq=mal(N), *dpk=mal((N/4)*3), *dq2=mal(N), *dy=mal(N*4), *dy0=mal(N*4);
        castT(dx,ACL_FLOAT,{N},dq,ACL_FLOAT6_E2M3,s); CK(aclrtSynchronizeStream(s));
        castT(dq,ACL_FLOAT6_E2M3,{N},dy0,ACL_FLOAT,s); CK(aclrtSynchronizeStream(s));   // direct (no pack)
        void*wp=nullptr; ws=0; CK(aclnnFp6PackGetWorkspaceSize(T({N},dq,ACL_FLOAT6_E2M3),T({(N/4)*3},dpk,ACL_UINT8),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnFp6Pack(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        wp=nullptr; ws=0; CK(aclnnFp6UnpackGetWorkspaceSize(T({(N/4)*3},dpk,ACL_UINT8),T({N},dq2,ACL_FLOAT6_E2M3),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnFp6Unpack(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        castT(dq2,ACL_FLOAT6_E2M3,{N},dy,ACL_FLOAT,s); CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,N),dnf(dy0,N)); tol=1e-6;   // exact: pack/unpack is lossless
    }
    printf("[%s] err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
