// Independent forward check for reduction / softmax ops vs PyTorch (torch_reduce.py).
// x[Rr,Cc] reduced/scanned along dim=1. argmax/argmin produce int64 indices (compared as float).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t Rr=6,Cc=5; static const double P=2.0;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t n){ std::vector<float> v(n); CK(aclrtMemcpy(v.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<float> dni(void*d,int64_t n){ std::vector<int64_t> t(n); CK(aclrtMemcpy(t.data(),n*8,d,n*8,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)t[i]; return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3]; int64_t NT=Rr*Cc;
    if(mode=="gen"){ std::vector<float> x(NT); for(int64_t i=0;i<NT;i++)x[i]=(float)(1.0+0.6*std::sin(i*0.37+0.2)); wf(pre+".x",x); printf("[gen] %s\n",op.c_str()); return 0; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    auto x=rf(pre+".x",NT); void*dx=upf(x); aclTensor*tx=T({Rr,Cc},dx);
    bool ew = (op=="softmax"||op=="logsoftmax"||op=="cumsum"||op=="cumprod");   // out [Rr,Cc]
    bool arg= (op=="argmax"||op=="argmin");                                    // out int64 [Rr]
    int64_t on = ew? NT : Rr;
    void* dout = mal(on * (arg?8:4));
    aclTensor* to = T(ew?std::vector<int64_t>{Rr,Cc}:std::vector<int64_t>{Rr}, dout, arg?ACL_INT64:ACL_FLOAT);
    if(op=="softmax"){ WS(aclnnSoftmaxGetWorkspaceSize(tx,1,to,&ws,&ex)); CK(aclnnSoftmax(wsp,ws,ex,s)); }
    else if(op=="logsoftmax"){ WS(aclnnLogSoftmaxGetWorkspaceSize(tx,1,to,&ws,&ex)); CK(aclnnLogSoftmax(wsp,ws,ex,s)); }
    else if(op=="cumsum"){ WS(aclnnCumsumGetWorkspaceSize(tx,1,ACL_FLOAT,to,&ws,&ex)); CK(aclnnCumsum(wsp,ws,ex,s)); }
    else if(op=="cumprod"){ WS(aclnnCumprodGetWorkspaceSize(tx,1,ACL_FLOAT,to,&ws,&ex)); CK(aclnnCumprod(wsp,ws,ex,s)); }
    else if(op=="logsumexp"){ WS(aclnnLogSumExpGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); CK(aclnnLogSumExp(wsp,ws,ex,s)); }
    else if(op=="var"){ WS(aclnnVarGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); CK(aclnnVar(wsp,ws,ex,s)); }
    else if(op=="std"){ WS(aclnnStdGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); CK(aclnnStd(wsp,ws,ex,s)); }
    else if(op=="normp"){ WS(aclnnNormGetWorkspaceSize(tx,P,IA({1}),false,to,&ws,&ex)); CK(aclnnNorm(wsp,ws,ex,s)); }
    else if(op=="amax"){ WS(aclnnAmaxGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); CK(aclnnAmax(wsp,ws,ex,s)); }
    else if(op=="amin"){ WS(aclnnAminGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); CK(aclnnAmin(wsp,ws,ex,s)); }
    else if(op=="argmax"){ WS(aclnnArgMaxGetWorkspaceSize(tx,1,false,to,&ws,&ex)); CK(aclnnArgMax(wsp,ws,ex,s)); }
    else if(op=="argmin"){ WS(aclnnArgMinGetWorkspaceSize(tx,1,false,to,&ws,&ex)); CK(aclnnArgMin(wsp,ws,ex,s)); }
    else if(op=="median"){ WS(aclnnMedianDimGetWorkspaceSize(tx,1,false,to,nullptr,&ws,&ex)); CK(aclnnMedianDim(wsp,ws,ex,s)); }
    else { fprintf(stderr,"unknown op %s\n",op.c_str()); return 2; }
    CK(aclrtSynchronizeStream(s));
    double e=cmp(arg?dni(dout,on):dn(dout,on), rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
