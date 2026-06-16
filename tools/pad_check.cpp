// Independent forward check for padding ops vs PyTorch (torch_pad.py). Symmetric pads.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static std::vector<float> g(int64_t n){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*0.21+0.3); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    auto INSZ=[&](const std::string&o)->int64_t{ if(o=="constantpad")return 8; if(o=="reflect1d"||o=="replicate1d")return 10; if(o=="reflect3d"||o=="replicate3d")return 64; return 50; };
    if(mode=="gen"){ wf(pre+".x",g(INSZ(op))); printf("[gen] %s\n",op.c_str()); return 0; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    auto x=rf(pre+".x",INSZ(op)); void*dx=upf(x); int64_t on=0; void*dout=nullptr; ex=nullptr; ws=0; wsp=nullptr;
    if(op=="constantpad"){ double v=0.5; aclScalar*sv=aclCreateScalar(&v,ACL_DOUBLE); dout=mal(1*2*6*4); on=12; CK(aclnnConstantPadNdGetWorkspaceSize(T({1,2,4},dx),IA({0,0,0,0,1,1}),sv,T({1,2,6},dout),&ws,&ex)); run(aclnnConstantPadNd,s); }
    else if(op=="reflect1d"){ dout=mal(1*2*9*4); on=18; CK(aclnnReflectionPad1dGetWorkspaceSize(T({1,2,5},dx),IA({2,2}),T({1,2,9},dout),&ws,&ex)); run(aclnnReflectionPad1d,s); }
    else if(op=="replicate1d"){ dout=mal(1*2*9*4); on=18; CK(aclnnReplicationPad1dGetWorkspaceSize(T({1,2,5},dx),IA({2,2}),T({1,2,9},dout),&ws,&ex)); run(aclnnReplicationPad1d,s); }
    else if(op=="reflect2d"){ dout=mal(1*2*9*9*4); on=162; CK(aclnnReflectionPad2dGetWorkspaceSize(T({1,2,5,5},dx),IA({2,2,2,2}),T({1,2,9,9},dout),&ws,&ex)); run(aclnnReflectionPad2d,s); }
    else if(op=="replicate2d"){ dout=mal(1*2*9*9*4); on=162; CK(aclnnReplicationPad2dGetWorkspaceSize(T({1,2,5,5},dx),IA({2,2,2,2}),T({1,2,9,9},dout),&ws,&ex)); run(aclnnReplicationPad2d,s); }
    else if(op=="circular2d"){ dout=mal(1*2*9*9*4); on=162; CK(aclnnCircularPad2dGetWorkspaceSize(T({1,2,5,5},dx),IA({2,2,2,2}),T({1,2,9,9},dout),&ws,&ex)); run(aclnnCircularPad2d,s); }
    else if(op=="reflect3d"){ dout=mal(1*1*6*6*6*4); on=216; CK(aclnnReflectionPad3dGetWorkspaceSize(T({1,1,4,4,4},dx),IA({1,1,1,1,1,1}),T({1,1,6,6,6},dout),&ws,&ex)); run(aclnnReflectionPad3d,s); }
    else if(op=="replicate3d"){ dout=mal(1*1*6*6*6*4); on=216; CK(aclnnReplicationPad3dGetWorkspaceSize(T({1,1,4,4,4},dx),IA({1,1,1,1,1,1}),T({1,1,6,6,6},dout),&ws,&ex)); run(aclnnReplicationPad3d,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
