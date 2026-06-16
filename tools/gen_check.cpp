// Independent forward check for generators / activations / RNN vs PyTorch (torch_gen.py).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t S=3,Bt=2,I=4,H=5;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclScalar* S_(double v){ return aclCreateScalar(&v,ACL_DOUBLE); }
static std::vector<float> g(int64_t n,double f,double p){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*f+p); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="hardtanh"||op=="logsigmoidfwd") wf(pre+".a",g(16,0.4,0.2));
        else if(op=="renorm") wf(pre+".a",g(20,0.3,0.1));
        else if(op=="lstm"||op=="gru"){ int G=(op=="lstm")?4:3;
            wf(pre+".x",g(S*Bt*I,0.3,0.1)); wf(pre+".wih",g(G*H*I,0.2,0.3)); wf(pre+".whh",g(G*H*H,0.15,0.5));
            wf(pre+".bih",g(G*H,0.1,0.2)); wf(pre+".bhh",g(G*H,0.12,0.4)); wf(pre+".h0",g(Bt*H,0.25,0.6)); if(op=="lstm")wf(pre+".c0",g(Bt*H,0.22,0.7)); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    int64_t on=0; void*dout=nullptr;
    if(op=="arange"){ dout=mal(9*4); on=9; CK(aclnnArangeGetWorkspaceSize(0.5,5.0,0.5,T({9},dout),&ws,&ex)); run(aclnnArange,s); }
    else if(op=="eye"){ dout=mal(16*4); on=16; CK(aclnnEyeGetWorkspaceSize(T({4,4},dout),&ws,&ex)); run(aclnnEye,s); }
    else if(op=="linspace"){ dout=mal(8*4); on=8; CK(aclnnLinspaceGetWorkspaceSize(S_(0.0),S_(1.0),8,T({8},dout),&ws,&ex)); run(aclnnLinspace,s); }
    else if(op=="hardtanh"){ auto a=rf(pre+".a",16); void*da=upf(a); dout=mal(16*4); on=16; CK(aclnnHardtanhGetWorkspaceSize(T({16},da),S_(-0.5),S_(0.5),T({16},dout),&ws,&ex)); run(aclnnHardtanh,s); }
    else if(op=="logsigmoidfwd"){ auto a=rf(pre+".a",16); void*da=upf(a); dout=mal(16*4); void*buf=mal(16*4); on=16; CK(aclnnLogSigmoidForwardGetWorkspaceSize(T({16},da),T({16},dout),T({16},buf),&ws,&ex)); run(aclnnLogSigmoidForward,s); }
    else if(op=="renorm"){ auto a=rf(pre+".a",20); void*da=upf(a); dout=mal(20*4); on=20; CK(aclnnRenormGetWorkspaceSize(T({4,5},da),2.0,0,1.0,T({4,5},dout),&ws,&ex)); run(aclnnRenorm,s); }
    else if(op=="lstm"){ auto x=rf(pre+".x",S*Bt*I),wih=rf(pre+".wih",4*H*I),whh=rf(pre+".whh",4*H*H),bih=rf(pre+".bih",4*H),bhh=rf(pre+".bhh",4*H),h0=rf(pre+".h0",Bt*H),c0=rf(pre+".c0",Bt*H);
        void*dx=upf(x),*dwih=upf(wih),*dwhh=upf(whh),*dbih=upf(bih),*dbhh=upf(bhh),*dh0=upf(h0),*dc0=upf(c0); dout=mal(S*Bt*H*4); void*dhn=mal(Bt*H*4),*dcn=mal(Bt*H*4); on=S*Bt*H;
        CK(aclnnLstmGetWorkspaceSize(T({S,Bt,I},dx),T({4*H,I},dwih),T({4*H,H},dwhh),T({4*H},dbih),T({4*H},dbhh),T({Bt,H},dh0),T({Bt,H},dc0),T({S,Bt,H},dout),T({Bt,H},dhn),T({Bt,H},dcn),&ws,&ex)); run(aclnnLstm,s); }
    else if(op=="gru"){ auto x=rf(pre+".x",S*Bt*I),wih=rf(pre+".wih",3*H*I),whh=rf(pre+".whh",3*H*H),bih=rf(pre+".bih",3*H),bhh=rf(pre+".bhh",3*H),h0=rf(pre+".h0",Bt*H);
        void*dx=upf(x),*dwih=upf(wih),*dwhh=upf(whh),*dbih=upf(bih),*dbhh=upf(bhh),*dh0=upf(h0); dout=mal(S*Bt*H*4); void*dhn=mal(Bt*H*4); on=S*Bt*H;
        CK(aclnnGruGetWorkspaceSize(T({S,Bt,I},dx),T({3*H,I},dwih),T({3*H,H},dwhh),T({3*H},dbih),T({3*H},dbhh),T({Bt,H},dh0),T({S,Bt,H},dout),T({Bt,H},dhn),&ws,&ex)); run(aclnnGru,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
