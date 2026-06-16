// Independent forward check for the pooling family vs PyTorch (torch_pool.py). pad=0 throughout.
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
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static std::vector<float> gen(int64_t n){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*0.21+0.3); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    auto INSZ=[&](const std::string&o)->int64_t{
        if(o=="avgpool3d"||o=="adaptiveavgpool3d"||o=="maxpool3d"||o=="maxpool3dargmax") return 1*2*4*4*4;
        if(o=="avgpool1d") return 1*2*8; if(o=="adaptivemaxpool2d"||o=="lppool2d") return 1*2*4*4;
        if(o=="channelshuffle") return 1*4*2*2; if(o=="im2col") return 1*2*4*4; if(o=="col2im") return 1*8*4; return 0; };
    if(mode=="gen"){ wf(pre+".x",gen(INSZ(op))); printf("[gen] %s\n",op.c_str()); return 0; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    auto x=rf(pre+".x",INSZ(op)); void*dx=upf(x); int64_t on=0; void*dout=nullptr; ex=nullptr;
    auto run=[&](aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream)){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); };
    ws=0;wsp=nullptr;
    if(op=="avgpool3d"){ dout=mal(1*2*2*2*2*4); on=16; CK(aclnnAvgPool3dGetWorkspaceSize(T({1,2,4,4,4},dx),IA({2,2,2}),IA({2,2,2}),IA({0,0,0}),T({1,2,2,2,2},dout),&ws,&ex)); run(aclnnAvgPool3d); }
    else if(op=="avgpool1d"){ dout=mal(1*2*4*4); on=8; CK(aclnnAvgPool1dGetWorkspaceSize(T({1,2,8},dx),2,2,0,T({1,2,4},dout),&ws,&ex)); run(aclnnAvgPool1d); }
    else if(op=="adaptiveavgpool3d"){ dout=mal(1*2*2*2*2*4); on=16; CK(aclnnAdaptiveAvgPool3dGetWorkspaceSize(T({1,2,4,4,4},dx),T({1,2,2,2,2},dout),&ws,&ex)); run(aclnnAdaptiveAvgPool3d); }
    else if(op=="adaptivemaxpool2d"){ dout=mal(1*2*2*2*4); void*di=mal(1*2*2*2*8); on=8; CK(aclnnAdaptiveMaxPool2dGetWorkspaceSize(T({1,2,4,4},dx),T({1,2,2,2},dout),T({1,2,2,2},di,ACL_INT64),&ws,&ex)); run(aclnnAdaptiveMaxPool2d); }
    else if(op=="maxpool3d"){ dout=mal(1*2*2*2*2*4); on=16; CK(aclnnMaxPool3dGetWorkspaceSize(T({1,2,4,4,4},dx),IA({2,2,2}),IA({2,2,2}),IA({0,0,0}),T({1,2,2,2,2},dout),&ws,&ex)); run(aclnnMaxPool3d); }
    else if(op=="maxpool3dargmax"){ dout=mal(1*2*2*2*2*4); void*di=mal(1*2*2*2*2*8); on=16; CK(aclnnMaxPool3dWithArgmaxGetWorkspaceSize(T({1,2,4,4,4},dx),IA({2,2,2}),IA({2,2,2}),IA({0,0,0}),IA({1,1,1}),false,T({1,2,2,2,2},dout),T({1,2,2,2,2},di,ACL_INT64),&ws,&ex)); run(aclnnMaxPool3dWithArgmax); }
    else if(op=="channelshuffle"){ dout=mal(1*4*2*2*4); on=16; CK(aclnnChannelShuffleGetWorkspaceSize(T({1,4,2,2},dx),2,T({1,4,2,2},dout),&ws,&ex)); run(aclnnChannelShuffle); }
    else if(op=="im2col"){ dout=mal(1*8*4*4); on=32; CK(aclnnIm2colGetWorkspaceSize(T({1,2,4,4},dx),IA({2,2}),IA({1,1}),IA({0,0}),IA({2,2}),T({1,8,4},dout),&ws,&ex)); run(aclnnIm2col); }
    else if(op=="col2im"){ dout=mal(1*2*4*4*4); on=32; CK(aclnnCol2ImGetWorkspaceSize(T({1,8,4},dx),IA({4,4}),IA({2,2}),IA({1,1}),IA({0,0}),IA({2,2}),T({1,2,4,4},dout),&ws,&ex)); run(aclnnCol2Im); }
    else if(op=="lppool2d"){ dout=mal(1*2*2*2*4); on=8; CK(aclnnLpPool2dGetWorkspaceSize(T({1,2,4,4},dx),2.0,IA({2,2}),IA({2,2}),T({1,2,2,2},dout),&ws,&ex)); run(aclnnLpPool2d); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
