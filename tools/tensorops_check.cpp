// Independent check for elementwise / tensor-manip / sort-select ops vs PyTorch (torch_tensorops.py).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t N=64,n=5,Rr=4,Cc=6,NC=5;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wu(const std::string&p,const std::vector<uint8_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),1,v.size(),f); fclose(f); }
static void wi(const std::string&p,const std::vector<int64_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),8,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static int64_t ri1(const std::string&p){ int64_t x=0; FILE*f=fopen(p.c_str(),"rb"); if(f){ if(fread(&x,8,1,f)!=1)x=0; fclose(f);} return x; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upu(const std::vector<uint8_t>&v){ void*d; CK(aclrtMalloc(&d,v.size(),ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size(),v.data(),v.size(),ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<float> dni(void*d,int64_t k){ std::vector<int64_t> t(k); CK(aclrtMemcpy(t.data(),k*8,d,k*8,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)t[i]; return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclScalar* S(double v){ return aclCreateScalar(&v,ACL_DOUBLE); }
static bool rep(const char*op,double e,double tol){ printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op,e,tol,e<=tol?"PASS":"FAIL"); return e<=tol; }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    auto sinv=[&](int64_t k,double a,double f,double ph){ std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)(a*std::sin(i*f+ph)); return v; };
    auto distinct=[&](int64_t k){ std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)(3.0*std::sin(i*0.9)+0.31*i); return v; }; // strictly varied (no ties)
    if(mode=="gen"){
        if(op=="clamp"||op=="threshold"){ wf(pre+".x",sinv(N,0.8,0.13,0.2)); }
        else if(op=="lerp"){ wf(pre+".x",sinv(N,1.0,0.13,0.0)); wf(pre+".end",sinv(N,1.0,0.11,1.0)); }
        else if(op=="addcmul"||op=="addcdiv"){ wf(pre+".x",sinv(N,1.0,0.13,0.0)); wf(pre+".t1",sinv(N,1.0,0.11,0.5)); auto t2=sinv(N,1.0,0.09,1.0); for(auto&z:t2)z+=2.0f; wf(pre+".t2",t2); }
        else if(op=="swhere"){ wf(pre+".x",sinv(N,1.0,0.13,0.0)); wf(pre+".y",sinv(N,1.0,0.11,2.0)); std::vector<uint8_t> c(N); for(int64_t i=0;i<N;i++)c[i]=(i%3==0)?1:0; wu(pre+".c",c); }
        else if(op=="maskedfill"){ wf(pre+".x",sinv(N,1.0,0.13,0.0)); std::vector<uint8_t> m(N); for(int64_t i=0;i<N;i++)m[i]=(i%3==0)?1:0; wu(pre+".m",m); }
        else if(op=="maskedselect"){ wf(pre+".x",sinv(N,1.0,0.13,0.0)); std::vector<uint8_t> m(N); for(int64_t i=0;i<N;i++)m[i]=(i%3==0)?1:0; wu(pre+".m",m); }
        else if(op=="tril"||op=="triu"){ wf(pre+".A",sinv(n*n,1.0,0.3,0.1)); }
        else if(op=="flip"||op=="roll"){ wf(pre+".x",sinv(Rr*Cc,1.0,0.3,0.1)); }
        else if(op=="onehot"){ std::vector<int64_t> ids(8); for(int i=0;i<8;i++)ids[i]=i%NC; wi(pre+".ids",ids); }
        else { wf(pre+".x",distinct(Rr*Cc)); }   // sort/topk/argsort/kthvalue
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    double tol=2e-4; bool ok=true;
    if(op=="clamp"||op=="threshold"){ auto x=rf(pre+".x",N); void*dx=upf(x),*dout=mal(N*4); aclTensor*tx=T({N},dx),*to=T({N},dout);
        if(op=="clamp"){ WS(aclnnClampGetWorkspaceSize(tx,S(-0.3),S(0.5),to,&ws,&ex)); CK(aclnnClamp(wsp,ws,ex,s)); }
        else { WS(aclnnThresholdGetWorkspaceSize(tx,S(0.0),S(-1.0),to,&ws,&ex)); CK(aclnnThreshold(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),cmp(dn(dout,N),rf(pre+".out",N)),tol); }
    else if(op=="lerp"){ auto x=rf(pre+".x",N),e2=rf(pre+".end",N); void*dx=upf(x),*de=upf(e2),*dout=mal(N*4); aclTensor*tx=T({N},dx),*te=T({N},de),*to=T({N},dout);
        WS(aclnnLerpGetWorkspaceSize(tx,te,S(0.3),to,&ws,&ex)); CK(aclnnLerp(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("lerp",cmp(dn(dout,N),rf(pre+".out",N)),tol); }
    else if(op=="addcmul"||op=="addcdiv"){ auto x=rf(pre+".x",N),t1=rf(pre+".t1",N),t2=rf(pre+".t2",N); void*dx=upf(x),*d1=upf(t1),*d2=upf(t2),*dout=mal(N*4); aclTensor*tx=T({N},dx),*tt1=T({N},d1),*tt2=T({N},d2),*to=T({N},dout);
        if(op=="addcmul"){ WS(aclnnAddcmulGetWorkspaceSize(tx,tt1,tt2,S(0.5),to,&ws,&ex)); CK(aclnnAddcmul(wsp,ws,ex,s)); } else { WS(aclnnAddcdivGetWorkspaceSize(tx,tt1,tt2,S(0.5),to,&ws,&ex)); CK(aclnnAddcdiv(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),cmp(dn(dout,N),rf(pre+".out",N)),tol); }
    else if(op=="swhere"){ auto x=rf(pre+".x",N),y=rf(pre+".y",N); std::vector<uint8_t> c(N); { FILE*f=fopen((pre+".c").c_str(),"rb"); if(fread(c.data(),1,N,f)!=(size_t)N){} fclose(f);} void*dx=upf(x),*dy=upf(y),*dc=upu(c),*dout=mal(N*4);
        aclTensor*tc=T({N},dc,ACL_UINT8),*tx=T({N},dx),*ty=T({N},dy),*to=T({N},dout); WS(aclnnSWhereGetWorkspaceSize(tc,tx,ty,to,&ws,&ex)); CK(aclnnSWhere(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("swhere",cmp(dn(dout,N),rf(pre+".out",N)),tol); }
    else if(op=="maskedfill"){ auto x=rf(pre+".x",N); std::vector<uint8_t> m(N); { FILE*f=fopen((pre+".m").c_str(),"rb"); if(fread(m.data(),1,N,f)!=(size_t)N){} fclose(f);} void*dx=upf(x),*dm=upu(m),*dout=mal(N*4);
        aclTensor*tx=T({N},dx),*tm=T({N},dm,ACL_UINT8),*to=T({N},dout); WS(aclnnMaskedFillScalarGetWorkspaceSize(tx,tm,S(9.0),to,&ws,&ex)); CK(aclnnMaskedFillScalar(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("maskedfill",cmp(dn(dout,N),rf(pre+".out",N)),tol); }
    else if(op=="maskedselect"){ auto x=rf(pre+".x",N); std::vector<uint8_t> m(N); { FILE*f=fopen((pre+".m").c_str(),"rb"); if(fread(m.data(),1,N,f)!=(size_t)N){} fclose(f);} void*dx=upf(x),*dm=upu(m),*dout=mal(N*4),*dcnt=mal(8);
        aclTensor*tx=T({N},dx),*tm=T({N},dm,ACL_UINT8),*to=T({N},dout),*tcnt=T({1},dcnt,ACL_INT64); WS(aclnnMaskedSelectGetWorkspaceSize(tx,tm,to,tcnt,&ws,&ex)); CK(aclnnMaskedSelect(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        int64_t cnt=0; CK(aclrtMemcpy(&cnt,8,dcnt,8,ACL_MEMCPY_DEVICE_TO_HOST)); int64_t rc=ri1(pre+".cnt");
        if(cnt!=rc){ printf("[maskedselect] count %ld vs torch %ld  FAIL\n",(long)cnt,(long)rc); ok=false; }
        else { auto ref=rf(pre+".out",rc); auto got=dn(dout,rc); ok&=rep("maskedselect",cmp(got,ref),tol); } }
    else if(op=="tril"||op=="triu"){ auto A=rf(pre+".A",n*n); void*dA=upf(A),*dout=mal(n*n*4); aclTensor*tA=T({n,n},dA),*to=T({n,n},dout);
        if(op=="tril"){ WS(aclnnTrilGetWorkspaceSize(tA,0,to,&ws,&ex)); CK(aclnnTril(wsp,ws,ex,s)); } else { WS(aclnnTriuGetWorkspaceSize(tA,0,to,&ws,&ex)); CK(aclnnTriu(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),cmp(dn(dout,n*n),rf(pre+".out",n*n)),tol); }
    else if(op=="flip"||op=="roll"){ auto x=rf(pre+".x",Rr*Cc); void*dx=upf(x),*dout=mal(Rr*Cc*4); aclTensor*tx=T({Rr,Cc},dx),*to=T({Rr,Cc},dout);
        if(op=="flip"){ WS(aclnnFlipGetWorkspaceSize(tx,1,to,&ws,&ex)); CK(aclnnFlip(wsp,ws,ex,s)); } else { WS(aclnnRollGetWorkspaceSize(tx,1,2,to,&ws,&ex)); CK(aclnnRoll(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),cmp(dn(dout,Rr*Cc),rf(pre+".out",Rr*Cc)),tol); }
    else if(op=="onehot"){ std::vector<int64_t> ids(8); { FILE*f=fopen((pre+".ids").c_str(),"rb"); if(fread(ids.data(),8,8,f)!=8){} fclose(f);} void*dids=upi(ids),*dout=mal(8*NC*4); aclTensor*ti=T({8},dids,ACL_INT64),*to=T({8,NC},dout);
        WS(aclnnOneHotGetWorkspaceSize(ti,NC,1.0,0.0,to,&ws,&ex)); CK(aclnnOneHot(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("onehot",cmp(dn(dout,8*NC),rf(pre+".out",8*NC)),tol); }
    else { // sort/topk/argsort/kthvalue
        auto x=rf(pre+".x",Rr*Cc); void*dx=upf(x); aclTensor*tx=T({Rr,Cc},dx);
        if(op=="sort"){ void*dv=mal(Rr*Cc*4),*di=mal(Rr*Cc*8); aclTensor*tv=T({Rr,Cc},dv),*tid=T({Rr,Cc},di,ACL_INT64); WS(aclnnSortGetWorkspaceSize(tx,1,false,true,tv,tid,&ws,&ex)); CK(aclnnSort(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("sort",cmp(dn(dv,Rr*Cc),rf(pre+".out",Rr*Cc)),tol); }
        else if(op=="topk"){ int64_t kk=3; void*dv=mal(Rr*kk*4),*di=mal(Rr*kk*8); aclTensor*tv=T({Rr,kk},dv),*tid=T({Rr,kk},di,ACL_INT64); WS(aclnnTopkGetWorkspaceSize(tx,kk,1,true,true,tv,tid,&ws,&ex)); CK(aclnnTopk(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("topk",cmp(dn(dv,Rr*kk),rf(pre+".out",Rr*kk)),tol); }
        else if(op=="argsort"){ void*di=mal(Rr*Cc*8); aclTensor*tid=T({Rr,Cc},di,ACL_INT64); WS(aclnnArgsortGetWorkspaceSize(tx,1,false,true,tid,&ws,&ex)); CK(aclnnArgsort(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("argsort",cmp(dni(di,Rr*Cc),rf(pre+".out",Rr*Cc)),tol); }
        else { void*dout=mal(Rr*4); aclTensor*to=T({Rr},dout); WS(aclnnKthvalueGetWorkspaceSize(tx,2,1,false,to,&ws,&ex)); CK(aclnnKthvalue(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ok&=rep("kthvalue",cmp(dn(dout,Rr),rf(pre+".out",Rr)),tol); }
    }
    return ok?0:1;
}
