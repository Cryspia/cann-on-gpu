// Independent forward check for reduction/scan extras vs PyTorch (torch_red.py). x[Rr,Cc], dim=1.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t Rr=4,Cc=6;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3]; int64_t NT=Rr*Cc;
    if(mode=="gen"){
        std::vector<float> x(NT);
        for(int64_t r=0;r<Rr;r++)for(int64_t c=0;c<Cc;c++) x[r*Cc+c]=(float)(0.6+std::sin((r*Cc+c)*0.4));
        if(op=="all"||op=="any"||op=="countnonzero"){ for(int64_t r=0;r<Rr;r+=2) x[r*Cc+0]=0.f; }       // inject zeros
        if(op=="nansum"||op=="nanmean"){ for(int64_t r=0;r<Rr;r++) x[r*Cc+1]=NAN; }                       // inject NaN
        if(op=="mode"){ for(int64_t r=0;r<Rr;r++)for(int64_t c=0;c<Cc;c++) x[r*Cc+c]=(c<3)?1.f:(float)(c+1); } // mode=1
        wf(pre+".x",x); printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    auto x=rf(pre+".x",NT); void*dx=upf(x); aclTensor*tx=T({Rr,Cc},dx); ex=nullptr; ws=0; wsp=nullptr;
    bool ew=(op=="cummax"||op=="cummin"||op=="logcumsumexp"); int64_t on=ew?NT:Rr;
    if(op=="aminmax"){ void*dmn=mal(Rr*4),*dmx=mal(Rr*4); CK(aclnnAminmaxGetWorkspaceSize(tx,IA({1}),false,T({Rr},dmn),T({Rr},dmx),&ws,&ex)); run(aclnnAminmax,s);
        double e1=cmp(dn(dmn,Rr),rf(pre+".outmin",Rr)), e2=cmp(dn(dmx,Rr),rf(pre+".outmax",Rr)); bool ok=e1<=2e-4&&e2<=2e-4;
        printf("[aminmax] min=%.2e max=%.2e %s\n",e1,e2,ok?"PASS":"FAIL"); return ok?0:1; }
    // bool ops → ACL_BOOL out (1 byte); countnonzero → ACL_INT64 out. Read with the right width, compare as float.
    if(op=="all"||op=="any"){ void*db=mal(Rr); aclTensor*tb=T({Rr},db,ACL_BOOL);
        if(op=="all"){ CK(aclnnAllGetWorkspaceSize(tx,IA({1}),false,tb,&ws,&ex)); run(aclnnAll,s); } else { CK(aclnnAnyGetWorkspaceSize(tx,IA({1}),false,tb,&ws,&ex)); run(aclnnAny,s); }
        std::vector<uint8_t> b(Rr); CK(aclrtMemcpy(b.data(),Rr,db,Rr,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> z(Rr); for(int64_t i=0;i<Rr;i++)z[i]=(float)b[i];
        double e=cmp(z,rf(pre+".out",Rr)); printf("[%s] out normalized_err=%.3e (tol=2e-04)  %s\n",op.c_str(),e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    if(op=="countnonzero"){ void*dc=mal(Rr*8); aclTensor*tc=T({Rr},dc,ACL_INT64); CK(aclnnCountNonzeroGetWorkspaceSize(tx,IA({1}),false,tc,&ws,&ex)); run(aclnnCountNonzero,s);
        std::vector<int64_t> c(Rr); CK(aclrtMemcpy(c.data(),Rr*8,dc,Rr*8,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> z(Rr); for(int64_t i=0;i<Rr;i++)z[i]=(float)c[i];
        double e=cmp(z,rf(pre+".out",Rr)); printf("[countnonzero] out normalized_err=%.3e (tol=2e-04)  %s\n",e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    void*dout=mal(on*4),*di=mal(on*8);
    aclTensor*to=T(ew?std::vector<int64_t>{Rr,Cc}:std::vector<int64_t>{Rr},dout);
    if(false){}
    else if(op=="maxdim"){ CK(aclnnMaxDimGetWorkspaceSize(tx,1,false,to,T({Rr},di,ACL_INT64),&ws,&ex)); run(aclnnMaxDim,s); }
    else if(op=="mindim"){ CK(aclnnMinDimGetWorkspaceSize(tx,1,false,to,T({Rr},di,ACL_INT64),&ws,&ex)); run(aclnnMinDim,s); }
    else if(op=="cummax"){ CK(aclnnCummaxGetWorkspaceSize(tx,1,to,T({Rr,Cc},di,ACL_INT64),&ws,&ex)); run(aclnnCummax,s); }
    else if(op=="cummin"){ CK(aclnnCumminGetWorkspaceSize(tx,1,to,T({Rr,Cc},di,ACL_INT64),&ws,&ex)); run(aclnnCummin,s); }
    else if(op=="logcumsumexp"){ CK(aclnnLogcumsumexpGetWorkspaceSize(tx,1,to,&ws,&ex)); run(aclnnLogcumsumexp,s); }
    else if(op=="mode"){ CK(aclnnModeGetWorkspaceSize(tx,1,false,to,&ws,&ex)); run(aclnnMode,s); }
    else if(op=="nansum"){ CK(aclnnNansumGetWorkspaceSize(tx,IA({1}),false,ACL_FLOAT,to,&ws,&ex)); run(aclnnNansum,s); }
    else if(op=="nanmean"){ CK(aclnnNanmeanGetWorkspaceSize(tx,IA({1}),false,ACL_FLOAT,to,&ws,&ex)); run(aclnnNanmean,s); }
    else if(op=="quantile"){ CK(aclnnQuantileGetWorkspaceSize(tx,0.5,1,false,to,&ws,&ex)); run(aclnnQuantile,s); }
    else if(op=="countnonzero"){ CK(aclnnCountNonzeroGetWorkspaceSize(tx,IA({1}),false,to,&ws,&ex)); run(aclnnCountNonzero,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
