// Independent forward check for the BLAS/matmul family vs PyTorch (torch_blas.py). beta=0.5, alpha=2.0.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t M=4,K=5,N=6,Bb=3; static const double BETA=0.5,ALPHA=2.0;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclScalar* S(double v){ return aclCreateScalar(&v,ACL_DOUBLE); }
static std::vector<float> g(int64_t n,double f,double p){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*f+p); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
#define RUN(call) do{ ws=0; wsp=nullptr; CK(call##GetWorkspaceSize); if(ws)wsp=mal(ws);}while(0)

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    auto WRITE=[&](const char*nm,std::vector<float>v){ wf(pre+nm,v); };
    if(mode=="gen"){
        if(op=="addmm"){ WRITE(".c",g(M*N,0.11,0.1)); WRITE(".a",g(M*K,0.07,0.2)); WRITE(".b",g(K*N,0.05,0.3)); }
        else if(op=="addbmm"){ WRITE(".c",g(M*N,0.11,0.1)); WRITE(".a",g(Bb*M*K,0.07,0.2)); WRITE(".b",g(Bb*K*N,0.05,0.3)); }
        else if(op=="addmv"){ WRITE(".c",g(M,0.11,0.1)); WRITE(".a",g(M*K,0.07,0.2)); WRITE(".b",g(K,0.05,0.3)); }
        else if(op=="addr"){ WRITE(".c",g(M*N,0.11,0.1)); WRITE(".a",g(M,0.07,0.2)); WRITE(".b",g(N,0.05,0.3)); }
        else if(op=="baddbmm"){ WRITE(".c",g(Bb*M*N,0.11,0.1)); WRITE(".a",g(Bb*M*K,0.07,0.2)); WRITE(".b",g(Bb*K*N,0.05,0.3)); }
        else if(op=="bmm"||op=="batchmatmul"){ WRITE(".a",g(Bb*M*K,0.07,0.2)); WRITE(".b",g(Bb*K*N,0.05,0.3)); }
        else if(op=="ger"){ WRITE(".a",g(M,0.07,0.2)); WRITE(".b",g(N,0.05,0.3)); }
        else if(op=="inner"){ WRITE(".a",g(8,0.07,0.2)); WRITE(".b",g(8,0.05,0.3)); }
        else if(op=="kron"){ WRITE(".a",g(2*3,0.07,0.2)); WRITE(".b",g(2*2,0.05,0.3)); }
        else if(op=="mm"){ WRITE(".a",g(M*K,0.07,0.2)); WRITE(".b",g(K*N,0.05,0.3)); }
        else if(op=="mv"){ WRITE(".a",g(M*K,0.07,0.2)); WRITE(".b",g(K,0.05,0.3)); }
        else if(op=="tensordot"){ WRITE(".a",g(3*4,0.07,0.2)); WRITE(".b",g(4*5,0.05,0.3)); }
        else if(op=="vdot"){ WRITE(".a",g(8,0.07,0.2)); WRITE(".b",g(8,0.05,0.3)); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    int64_t on=0; void* dout=nullptr; ex=nullptr;
    auto fin=[&](aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){ CK(run(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); };
    if(op=="addmm"){ auto C=rf(pre+".c",M*N),A=rf(pre+".a",M*K),B=rf(pre+".b",K*N); void*dc=upf(C),*da=upf(A),*db=upf(B); dout=mal(M*N*4); on=M*N;
        ws=0;wsp=nullptr; CK(aclnnAddmmGetWorkspaceSize(T({M,N},dc),T({M,K},da),T({K,N},db),BETA,ALPHA,T({M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnAddmm); }
    else if(op=="addbmm"){ auto C=rf(pre+".c",M*N),A=rf(pre+".a",Bb*M*K),B=rf(pre+".b",Bb*K*N); void*dc=upf(C),*da=upf(A),*db=upf(B); dout=mal(M*N*4); on=M*N;
        ws=0;wsp=nullptr; CK(aclnnAddbmmGetWorkspaceSize(T({M,N},dc),T({Bb,M,K},da),T({Bb,K,N},db),BETA,ALPHA,T({M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnAddbmm); }
    else if(op=="addmv"){ auto C=rf(pre+".c",M),A=rf(pre+".a",M*K),B=rf(pre+".b",K); void*dc=upf(C),*da=upf(A),*db=upf(B); dout=mal(M*4); on=M;
        ws=0;wsp=nullptr; CK(aclnnAddmvGetWorkspaceSize(T({M},dc),T({M,K},da),T({K},db),BETA,ALPHA,T({M},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnAddmv); }
    else if(op=="addr"){ auto C=rf(pre+".c",M*N),A=rf(pre+".a",M),B=rf(pre+".b",N); void*dc=upf(C),*da=upf(A),*db=upf(B); dout=mal(M*N*4); on=M*N;
        ws=0;wsp=nullptr; CK(aclnnAddrGetWorkspaceSize(T({M,N},dc),T({M},da),T({N},db),S(BETA),S(ALPHA),T({M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnAddr); }
    else if(op=="baddbmm"){ auto C=rf(pre+".c",Bb*M*N),A=rf(pre+".a",Bb*M*K),B=rf(pre+".b",Bb*K*N); void*dc=upf(C),*da=upf(A),*db=upf(B); dout=mal(Bb*M*N*4); on=Bb*M*N;
        ws=0;wsp=nullptr; CK(aclnnBaddbmmGetWorkspaceSize(T({Bb,M,N},dc),T({Bb,M,K},da),T({Bb,K,N},db),BETA,ALPHA,T({Bb,M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnBaddbmm); }
    else if(op=="bmm"){ auto A=rf(pre+".a",Bb*M*K),B=rf(pre+".b",Bb*K*N); void*da=upf(A),*db=upf(B); dout=mal(Bb*M*N*4); on=Bb*M*N;
        ws=0;wsp=nullptr; CK(aclnnBmmGetWorkspaceSize(T({Bb,M,K},da),T({Bb,K,N},db),T({Bb,M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnBmm); }
    else if(op=="batchmatmul"){ auto A=rf(pre+".a",Bb*M*K),B=rf(pre+".b",Bb*K*N); void*da=upf(A),*db=upf(B); dout=mal(Bb*M*N*4); on=Bb*M*N;
        ws=0;wsp=nullptr; CK(aclnnBatchMatMulGetWorkspaceSize(T({Bb,M,K},da),T({Bb,K,N},db),T({Bb,M,N},dout),0,&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnBatchMatMul); }
    else if(op=="ger"){ auto A=rf(pre+".a",M),B=rf(pre+".b",N); void*da=upf(A),*db=upf(B); dout=mal(M*N*4); on=M*N;
        ws=0;wsp=nullptr; CK(aclnnGerGetWorkspaceSize(T({M},da),T({N},db),T({M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnGer); }
    else if(op=="inner"){ auto A=rf(pre+".a",8),B=rf(pre+".b",8); void*da=upf(A),*db=upf(B); dout=mal(4); on=1;
        ws=0;wsp=nullptr; CK(aclnnInnerGetWorkspaceSize(T({8},da),T({8},db),T({1},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnInner); }
    else if(op=="kron"){ auto A=rf(pre+".a",6),B=rf(pre+".b",4); void*da=upf(A),*db=upf(B); dout=mal(6*4*4); on=24;
        ws=0;wsp=nullptr; CK(aclnnKronGetWorkspaceSize(T({2,3},da),T({2,2},db),T({4,6},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnKron); }
    else if(op=="mm"){ auto A=rf(pre+".a",M*K),B=rf(pre+".b",K*N); void*da=upf(A),*db=upf(B); dout=mal(M*N*4); on=M*N;
        ws=0;wsp=nullptr; CK(aclnnMmGetWorkspaceSize(T({M,K},da),T({K,N},db),T({M,N},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnMm); }
    else if(op=="mv"){ auto A=rf(pre+".a",M*K),B=rf(pre+".b",K); void*da=upf(A),*db=upf(B); dout=mal(M*4); on=M;
        ws=0;wsp=nullptr; CK(aclnnMvGetWorkspaceSize(T({M,K},da),T({K},db),T({M},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnMv); }
    else if(op=="tensordot"){ auto A=rf(pre+".a",12),B=rf(pre+".b",20); void*da=upf(A),*db=upf(B); dout=mal(3*5*4); on=15;
        ws=0;wsp=nullptr; CK(aclnnTensordotGetWorkspaceSize(T({3,4},da),T({4,5},db),1,T({3,5},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnTensordot); }
    else if(op=="vdot"){ auto A=rf(pre+".a",8),B=rf(pre+".b",8); void*da=upf(A),*db=upf(B); dout=mal(4); on=1;
        ws=0;wsp=nullptr; CK(aclnnVdotGetWorkspaceSize(T({8},da),T({8},db),T({1},dout),&ws,&ex)); if(ws)wsp=mal(ws); fin(aclnnVdot); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
