// Independent backward check (activation/softmax) vs PyTorch autograd (torch_bwd.py).
// Activation *Backward(gradOut, self, [param], gradIn) uses self=x; Softmax/LogSoftmax backward use the
// forward output f(x). Compares gradInput to torch x.grad.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t N=4096,R=8,C=16;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclScalar* S(double v){ return aclCreateScalar(&v,ACL_DOUBLE); }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    bool sm=(op=="softmaxbwd"||op=="logsoftmaxbwd"); int64_t n=sm?R*C:N;
    if(mode=="gen"){ std::vector<float> x(n),go(n); for(int64_t i=0;i<n;i++){ x[i]=(float)(2.5*std::sin(i*0.013)); go[i]=(float)(1.3*std::cos(i*0.017)); } wf(pre+".x",x); wf(pre+".go",go); printf("[gen] %s\n",op.c_str()); return 0; }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    auto x=rf(pre+".x",n),go=rf(pre+".go",n); auto ref=rf(pre+".gi",n);
    void*dgo=upf(go),*dgi=mal(n*4);
    if(sm){ // forward output f(x) computed on host, fed as the op's "output"
        std::vector<float> out(n);
        for(int64_t r=0;r<R;r++){ double mx=-1e30; for(int64_t c=0;c<C;c++)mx=std::max(mx,(double)x[r*C+c]); double sum=0; for(int64_t c=0;c<C;c++)sum+=std::exp(x[r*C+c]-mx);
            for(int64_t c=0;c<C;c++){ double smv=std::exp(x[r*C+c]-mx)/sum; out[r*C+c]=(float)(op=="softmaxbwd"?smv:std::log(smv)); } }
        void*dout=upf(out); aclTensor*tgo=T({R,C},dgo),*to=T({R,C},dout),*tgi=T({R,C},dgi);
        if(op=="softmaxbwd"){ CK(aclnnSoftmaxBackwardGetWorkspaceSize(tgo,to,1,tgi,&ws,&ex)); run(aclnnSoftmaxBackward,s); }
        else { CK(aclnnLogSoftmaxBackwardGetWorkspaceSize(tgo,to,1,tgi,&ws,&ex)); run(aclnnLogSoftmaxBackward,s); }
    } else {
        void*dx=upf(x); aclTensor*tgo=T({N},dgo),*tx=T({N},dx),*tgi=T({N},dgi);
        if(op=="elu"){ CK(aclnnEluBackwardGetWorkspaceSize(tgo,tx,1.0,tgi,&ws,&ex)); run(aclnnEluBackward,s); }
        else if(op=="hardtanh"){ CK(aclnnHardtanhBackwardGetWorkspaceSize(tgo,tx,S(-0.5),S(0.5),tgi,&ws,&ex)); run(aclnnHardtanhBackward,s); }
        else if(op=="leakyrelu"){ CK(aclnnLeakyReluBackwardGetWorkspaceSize(tgo,tx,0.1,tgi,&ws,&ex)); run(aclnnLeakyReluBackward,s); }
        else if(op=="softshrink"){ CK(aclnnSoftshrinkBackwardGetWorkspaceSize(tgo,tx,S(0.5),tgi,&ws,&ex)); run(aclnnSoftshrinkBackward,s); }
        else if(op=="threshold"){ CK(aclnnThresholdBackwardGetWorkspaceSize(tgo,tx,S(0.0),tgi,&ws,&ex)); run(aclnnThresholdBackward,s); }
        else if(op=="hardshrink"){ CK(aclnnHardshrinkBackwardGetWorkspaceSize(tgo,tx,S(0.5),tgi,&ws,&ex)); run(aclnnHardshrinkBackward,s); }
        else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    }
    double e=cmp(dn(dgi,n),ref); double tol=2e-4;
    printf("[%s] gradInput normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
