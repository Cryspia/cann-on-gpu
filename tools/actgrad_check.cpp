// Independent backward-gradient cross-check: compares each activation *Backward shim op against the
// gradient computed by PyTorch autograd (torch_actgrad.py) — an oracle independent of our hand-derived
// analytic f', so it catches the "shim and test share the same wrong derivative" class of bug.
//   gen   <op> <prefix>           write prefix.x (forward input) + prefix.go (upstream grad)
//   check <op> <prefix> <ref.bin> compute val (x, or f(x) for output-based ops), run aclnnXxxBackward,
//                                 compare gradInput against the torch-autograd reference.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t NE=4096;

static bool out_based(const std::string&o){ return o=="sigmoid"||o=="tanh"; }  // backward takes the forward OUTPUT
static double fwd(const std::string&o,double x){
    if(o=="sigmoid") return 1.0/(1.0+std::exp(-x));
    if(o=="tanh")    return std::tanh(x);
    return x;
}
typedef aclnnStatus (*BWs)(const aclTensor*,const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**);
typedef aclnnStatus (*Run)(void*,uint64_t,aclOpExecutor*,aclrtStream);
static BWs bws(const std::string&o){
    if(o=="relu")return aclnnReluBackwardGetWorkspaceSize; if(o=="gelu")return aclnnGeluBackwardGetWorkspaceSize;
    if(o=="silu")return aclnnSiluBackwardGetWorkspaceSize; if(o=="softplus")return aclnnSoftplusBackwardGetWorkspaceSize;
    if(o=="hardswish")return aclnnHardswishBackwardGetWorkspaceSize; if(o=="sigmoid")return aclnnSigmoidBackwardGetWorkspaceSize;
    if(o=="tanh")return aclnnTanhBackwardGetWorkspaceSize; if(o=="fastgelu")return aclnnFastGeluBackwardGetWorkspaceSize;
    if(o=="hardsigmoid")return aclnnHardsigmoidBackwardGetWorkspaceSize; if(o=="logsigmoid")return aclnnLogSigmoidBackwardGetWorkspaceSize;
    if(o=="mish")return aclnnMishBackwardGetWorkspaceSize; if(o=="selu")return aclnnSeluBackwardGetWorkspaceSize;
    return nullptr;
}
static Run brun(const std::string&o){
    if(o=="relu")return aclnnReluBackward; if(o=="gelu")return aclnnGeluBackward; if(o=="silu")return aclnnSiluBackward;
    if(o=="softplus")return aclnnSoftplusBackward; if(o=="hardswish")return aclnnHardswishBackward;
    if(o=="sigmoid")return aclnnSigmoidBackward; if(o=="tanh")return aclnnTanhBackward; if(o=="fastgelu")return aclnnFastGeluBackward;
    if(o=="hardsigmoid")return aclnnHardsigmoidBackward; if(o=="logsigmoid")return aclnnLogSigmoidBackward;
    if(o=="mish")return aclnnMishBackward; if(o=="selu")return aclnnSeluBackward;
    return nullptr;
}
static void wbin(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rbin(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* mkt(int64_t n,void*d){ int64_t dm[1]={n}; return aclCreateTensor(dm,1,ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,dm,1,d); }

int main(int argc,char**argv){
    if(argc<4){ fprintf(stderr,"usage: actgrad_check gen|check <op> <prefix> [ref.bin]\n"); return 1; }
    std::string mode=argv[1], op=argv[2], pre=argv[3];
    if(mode=="gen"){
        std::vector<float> x(NE), go(NE);
        for(int64_t i=0;i<NE;i++){ x[i]=(float)(2.5*std::sin(i*0.013)); go[i]=(float)(1.3*std::cos(i*0.017)); }
        wbin(pre+".x",x); wbin(pre+".go",go); printf("[gen] %s\n",op.c_str()); return 0;
    }
    // check
    std::string ref=argv[4];
    auto x=rbin(pre+".x",NE), go=rbin(pre+".go",NE), r=rbin(ref,NE);
    std::vector<float> val(NE); for(int64_t i=0;i<NE;i++) val[i]=(float)fwd(op,x[i]);   // x, or f(x) for out-based
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    void *dg,*dv,*dz; CK(aclrtMalloc(&dg,NE*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dv,NE*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,NE*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(dg,NE*4,go.data(),NE*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dv,NE*4,val.data(),NE*4,ACL_MEMCPY_HOST_TO_DEVICE));
    aclTensor *tg=mkt(NE,dg),*tv=mkt(NE,dv),*tz=mkt(NE,dz);
    BWs gw=bws(op); Run run=brun(op); if(!gw||!run){ fprintf(stderr,"unsupported backward op %s\n",op.c_str()); return 2; }
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(gw(tg,tv,tz,&ws,&ex));
    void*wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(run(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
    std::vector<float> z(NE); CK(aclrtMemcpy(z.data(),NE*4,dz,NE*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0; for(int64_t i=0;i<NE;i++){ me=std::max(me,(double)std::fabs(z[i]-r[i])); mr=std::max(mr,(double)std::fabs(r[i])); }
    double tol = (op=="gelu"||op=="fastgelu"||op=="mish")?3e-3:2e-4;
    double e=me/(mr+1e-9);
    printf("[%s] shim gI[0]=%.6f torch[0]=%.6f  normalized_err=%.3e (tol=%.0e)  %s\n", op.c_str(), z[0], r[0], e, tol, e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
