// Independent check for optimizers vs PyTorch (torch_opt.py). Fresh state (m=v=buf=0, step=1); compares
// the in-place-updated param against torch.optim's one step (builtins) or closed-form (lamb/lars/fusedema).
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
static const double LR=0.01,B1=0.9,B2=0.999,EPS=1e-8,WD=0.01,MOM=0.9,ALPHA=0.99,RHO=0.9,TRUST=0.001,EMAD=0.999;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(void*p){ int64_t d[1]={N}; return aclCreateTensor(d,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d,1,p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* zeros(){ void*d; CK(aclrtMalloc(&d,N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemset(d,N*4,0,N*4)); return d; }
static std::vector<float> dn(void*d){ std::vector<float> v(N); CK(aclrtMemcpy(v.data(),N*4,d,N*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){ std::vector<float> P(N),G(N); for(int64_t i=0;i<N;i++){ P[i]=(float)(0.5*std::sin(i*0.11+0.3)); G[i]=(float)(0.3*std::cos(i*0.07+0.2)); } wf(pre+".p",P); wf(pre+".g",G); printf("[gen] %s\n",op.c_str()); return 0; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    auto P=rf(pre+".p",N),G=rf(pre+".g",N); void*dp=upf(P),*dg=upf(G);
    auto W=[&](int rc){ CK(rc); if(ws){CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));} };
    if(op=="adam"){ W(aclnnApplyAdamGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(dg),LR,B1,B2,EPS,WD,1,&ws,&ex)); CK(aclnnApplyAdam(wsp,ws,ex,s)); }
    else if(op=="adamw"){ W(aclnnApplyAdamWGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(dg),LR,B1,B2,EPS,WD,1,&ws,&ex)); CK(aclnnApplyAdamW(wsp,ws,ex,s)); }
    else if(op=="adagrad"){ W(aclnnApplyAdagradGetWorkspaceSize(T(dp),T(zeros()),T(dg),LR,EPS,WD,&ws,&ex)); CK(aclnnApplyAdagrad(wsp,ws,ex,s)); }
    else if(op=="rmsprop"){ W(aclnnApplyRmspropGetWorkspaceSize(T(dp),T(zeros()),T(dg),LR,ALPHA,EPS,WD,&ws,&ex)); CK(aclnnApplyRmsprop(wsp,ws,ex,s)); }
    else if(op=="adamax"){ W(aclnnApplyAdamaxGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(dg),LR,B1,B2,EPS,1,&ws,&ex)); CK(aclnnApplyAdamax(wsp,ws,ex,s)); }
    else if(op=="adadelta"){ W(aclnnApplyAdadeltaGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(dg),LR,RHO,EPS,WD,&ws,&ex)); CK(aclnnApplyAdadelta(wsp,ws,ex,s)); }
    else if(op=="momentum"){ W(aclnnApplyMomentumGetWorkspaceSize(T(dp),T(zeros()),T(dg),LR,MOM,WD,0.0,false,&ws,&ex)); CK(aclnnApplyMomentum(wsp,ws,ex,s)); }
    else if(op=="lamb"){ W(aclnnApplyLambGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(dg),LR,B1,B2,EPS,WD,1,&ws,&ex)); CK(aclnnApplyLamb(wsp,ws,ex,s)); }
    else if(op=="lars"){ W(aclnnApplyLarsGetWorkspaceSize(T(dp),T(zeros()),T(dg),LR,MOM,WD,TRUST,EPS,&ws,&ex)); CK(aclnnApplyLars(wsp,ws,ex,s)); }
    else if(op=="fusedemaadam"){ W(aclnnApplyFusedEmaAdamGetWorkspaceSize(T(dp),T(zeros()),T(zeros()),T(zeros()),T(dg),LR,B1,B2,EPS,WD,EMAD,1,&ws,&ex)); CK(aclnnApplyFusedEmaAdam(wsp,ws,ex,s)); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    CK(aclrtSynchronizeStream(s));
    double e=cmp(dn(dp),rf(pre+".out",N)); double tol=2e-4;
    printf("[%s] param normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
