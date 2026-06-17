// Invariant tests: shape round-trips (op∘inverse==identity) and cross-op self-consistency relations.
//   permute_inv: permute(permute(x,[1,0,2]),[1,0,2])==x      flip2: flip(flip(x))==x
//   roll_inv: roll(roll(x,+s),-s)==x                          squeeze_unsq: unsqueeze(squeeze(x))==x
//   cumsum_consistency: diff(cumsum(x))==x & cumsum[0]==x[0]  sort_argsort: gather(x,argsort(x))==sort(x).values
//   onehot_consistency: row sum==1 & argmax==ids              logsoftmax_softmax: exp(logsoftmax(x))==softmax(x)
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<int64_t> dni(void*d,int64_t k){ std::vector<int64_t> v(k); CK(aclrtMemcpy(v.data(),k*8,d,k*8,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static double amax(const std::vector<float>&z,const std::vector<float>&r){ double m=0; for(size_t i=0;i<z.size();i++)m=std::max(m,(double)std::fabs(z[i]-r[i])); return m; }
static uint64_t ws; static aclOpExecutor* ex;

int main(int argc,char**argv){
    std::string op=argv[1];
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    auto W=[&](int rc)->void*{ ws=0; void*wp=nullptr; CK(rc); if(ws)wp=mal(ws); return wp; };
    double e=0, tol=1e-5; const int64_t Rr=4,Cc=6;
    if(op=="permute_inv"){
        const int64_t A=3,B=4,C=5; int64_t N=A*B*C; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.13);
        void*dx=upf(x),*d1=mal(N*4),*d2=mal(N*4);
        {void*wp=W(aclnnPermuteGetWorkspaceSize(T({A,B,C},dx),IA({1,0,2}),T({B,A,C},d1),&ws,&ex)); CK(aclnnPermute(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnPermuteGetWorkspaceSize(T({B,A,C},d1),IA({1,0,2}),T({A,B,C},d2),&ws,&ex)); CK(aclnnPermute(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        e=amax(dnf(d2,N),x);
    } else if(op=="flip2"){
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.13); void*dx=upf(x),*d1=mal(N*4),*d2=mal(N*4);
        {void*wp=W(aclnnFlipGetWorkspaceSize(T({Rr,Cc},dx),1,T({Rr,Cc},d1),&ws,&ex)); CK(aclnnFlip(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnFlipGetWorkspaceSize(T({Rr,Cc},d1),1,T({Rr,Cc},d2),&ws,&ex)); CK(aclnnFlip(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        e=amax(dnf(d2,N),x);
    } else if(op=="roll_inv"){
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.13); void*dx=upf(x),*d1=mal(N*4),*d2=mal(N*4);
        {void*wp=W(aclnnRollGetWorkspaceSize(T({Rr,Cc},dx),1,2,T({Rr,Cc},d1),&ws,&ex)); CK(aclnnRoll(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnRollGetWorkspaceSize(T({Rr,Cc},d1),1,-2,T({Rr,Cc},d2),&ws,&ex)); CK(aclnnRoll(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        e=amax(dnf(d2,N),x);
    } else if(op=="squeeze_unsq"){
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.13); void*dx=upf(x),*d1=mal(N*4),*d2=mal(N*4);
        {void*wp=W(aclnnSqueezeGetWorkspaceSize(T({Rr,1,Cc},dx),1,T({Rr,Cc},d1),&ws,&ex)); CK(aclnnSqueeze(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnUnsqueezeGetWorkspaceSize(T({Rr,Cc},d1),1,T({Rr,1,Cc},d2),&ws,&ex)); CK(aclnnUnsqueeze(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        e=amax(dnf(d2,N),x);
    } else if(op=="cumsum_consistency"){
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)(0.5+std::sin(i*0.3)); void*dx=upf(x),*dy=mal(N*4);
        {void*wp=W(aclnnCumsumGetWorkspaceSize(T({Rr,Cc},dx),1,ACL_FLOAT,T({Rr,Cc},dy),&ws,&ex)); CK(aclnnCumsum(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        auto y=dnf(dy,N); double m=0; for(int64_t r=0;r<Rr;r++){ m=std::max(m,(double)std::fabs(y[r*Cc]-x[r*Cc])); for(int64_t c=1;c<Cc;c++) m=std::max(m,(double)std::fabs((y[r*Cc+c]-y[r*Cc+c-1])-x[r*Cc+c])); } e=m;
    } else if(op=="sort_argsort"){
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)(3.0*std::sin(i*0.9)+0.31*i); void*dx=upf(x),*dv=mal(N*4),*di=mal(N*8),*dai=mal(N*8);
        {void*wp=W(aclnnSortGetWorkspaceSize(T({Rr,Cc},dx),1,false,true,T({Rr,Cc},dv),T({Rr,Cc},di,ACL_INT64),&ws,&ex)); CK(aclnnSort(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnArgsortGetWorkspaceSize(T({Rr,Cc},dx),1,false,true,T({Rr,Cc},dai,ACL_INT64),&ws,&ex)); CK(aclnnArgsort(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        auto val=dnf(dv,N); auto ai=dni(dai,N); std::vector<float> gath(N); for(int64_t r=0;r<Rr;r++)for(int64_t c=0;c<Cc;c++)gath[r*Cc+c]=x[r*Cc+ai[r*Cc+c]]; e=amax(gath,val);
    } else if(op=="onehot_consistency"){
        const int64_t L=8, NCl=5; std::vector<int64_t> ids(L); for(int64_t i=0;i<L;i++)ids[i]=(i*3+1)%NCl; void*dids=upi(ids),*dout=mal(L*NCl*4);
        {void*wp=W(aclnnOneHotGetWorkspaceSize(T({L},dids,ACL_INT64),NCl,1.0,0.0,T({L,NCl},dout),&ws,&ex)); CK(aclnnOneHot(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        auto o=dnf(dout,L*NCl); double m=0; for(int64_t i=0;i<L;i++){ double sum=0; int am=0; float best=-1e9; for(int64_t c=0;c<NCl;c++){ sum+=o[i*NCl+c]; if(o[i*NCl+c]>best){best=o[i*NCl+c];am=(int)c;} } m=std::max(m,std::fabs(sum-1.0)); m=std::max(m,(double)std::fabs((double)(am-ids[i]))); } e=m;
    } else { // logsoftmax_softmax: exp(logsoftmax)==softmax
        int64_t N=Rr*Cc; std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.3); void*dx=upf(x),*dl=mal(N*4),*dsm=mal(N*4);
        {void*wp=W(aclnnLogSoftmaxGetWorkspaceSize(T({Rr,Cc},dx),1,T({Rr,Cc},dl),&ws,&ex)); CK(aclnnLogSoftmax(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        {void*wp=W(aclnnSoftmaxGetWorkspaceSize(T({Rr,Cc},dx),1,T({Rr,Cc},dsm),&ws,&ex)); CK(aclnnSoftmax(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));}
        auto l=dnf(dl,N); auto sm=dnf(dsm,N); std::vector<float> el(N); for(int64_t i=0;i<N;i++)el[i]=std::exp(l[i]); e=amax(el,sm);
    }
    printf("[%s] err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
