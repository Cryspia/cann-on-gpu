// Invariant test for MoE token routing: unpermute(permute(x)) == x. MoeTokenPermute groups tokens by
// expert (srcIdx[p] = original token at permuted position p); MoeTokenUnpermute scatters them back with
// weight (default 1). With top-1 routing each token appears once, so the round-trip is exact identity.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t Tn=12, Dn=8, En=3;
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi32(const std::vector<int32_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double ncmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(){
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    int64_t ND=Tn*Dn;
    std::vector<float> x(ND); for(int64_t i=0;i<ND;i++)x[i]=(float)std::sin(i*0.23+0.4);
    std::vector<int32_t> eid(Tn); for(int64_t t=0;t<Tn;t++)eid[t]=(int32_t)(t%En);
    void*dx=upf(x),*deid=upi32(eid),*dperm=mal(ND*4),*dsrc=mal(Tn*8),*dout=mal(ND*4);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wp=nullptr;
    CK(aclnnMoeTokenPermuteGetWorkspaceSize(T({Tn,Dn},dx),T({Tn},deid,ACL_INT32),En,T({Tn,Dn},dperm),T({Tn},dsrc,ACL_INT64),&ws,&ex)); if(ws)wp=mal(ws);
    CK(aclnnMoeTokenPermute(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
    CK(aclrtMemset(dout,ND*4,0,ND*4));   // unpermute scatters via atomicAdd -> zero first
    ws=0; wp=nullptr;
    CK(aclnnMoeTokenUnpermuteGetWorkspaceSize(T({Tn,Dn},dperm),T({Tn},dsrc,ACL_INT64),nullptr,T({Tn,Dn},dout),&ws,&ex)); if(ws)wp=mal(ws);
    CK(aclnnMoeTokenUnpermute(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
    double e=ncmp(dnf(dout,ND),x); double tol=1e-6;
    printf("[moe_permute] unpermute(permute(x))==x  err=%.3e (tol=%.0e)  %s\n",e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
