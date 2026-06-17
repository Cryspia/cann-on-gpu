// Independent forward check for normalization ops vs PyTorch (torch_norm.py).
//   gen   <op> <prefix>   write inputs (.x/.w/.b/.g/.res/.m/.v float)
//   check <op> <prefix>   run the shim, compare output(s) against torch.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t R=8,Dd=16;        // layernorm/rmsnorm/addrmsnorm
static const int64_t N=2,C=4,Hh=3,G=2; // 4D norms
static const double EPS=1e-5;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t n){ std::vector<float> v(n); CK(aclrtMemcpy(v.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static bool rep(const char*op,const char*w,double e,double tol){ printf("[%s] %-4s normalized_err=%.3e (tol=%.0e)  %s\n",op,w,e,tol,e<=tol?"PASS":"FAIL"); return e<=tol; }
static void fill(std::vector<float>&v,double a,double f,double ph){ for(size_t i=0;i<v.size();i++)v[i]=(float)(a*std::sin(i*f+ph)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    int64_t FOUR=N*C*Hh*Hh, RD=R*Dd;
    if(mode=="gen"){
        if(op=="layernorm"||op=="rmsnorm"){ std::vector<float> x(RD),w(Dd),b(Dd); fill(x,2.0,0.013,0); fill(w,1.0,0.05,0.5); fill(b,0.4,0.07,1.0);
            for(auto&z:w)z+=1.0f; wf(pre+".x",x); wf(pre+(op=="rmsnorm"?".g":".w"),w); if(op=="layernorm")wf(pre+".b",b); }
        else if(op=="addrmsnorm"){ std::vector<float> x(RD),res(RD),g(Dd); fill(x,2.0,0.013,0); fill(res,1.0,0.017,0.3); fill(g,1.0,0.05,0.5); for(auto&z:g)z+=1.0f;
            wf(pre+".x",x); wf(pre+".res",res); wf(pre+".g",g); }
        else { std::vector<float> x(FOUR),w(C),b(C),m(C),v(C); fill(x,2.0,0.013,0); fill(w,0.5,0.3,0.5); for(auto&z:w)z+=1.0f; fill(b,0.3,0.4,1.0);
            fill(m,0.5,0.6,0.2); fill(v,0.3,0.5,0.1); for(auto&z:v)z=std::fabs(z)+0.5f;   // var>0
            wf(pre+".x",x); wf(pre+".w",w); wf(pre+".b",b); if(op=="batchnorm"){ wf(pre+".m",m); wf(pre+".v",v);} }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    bool ok=true;
    if(op=="layernorm"){
        auto x=rf(pre+".x",RD),w=rf(pre+".w",Dd),b=rf(pre+".b",Dd);
        void*dx=upf(x),*dw=upf(w),*db=upf(b),*dout=mal(RD*4),*dm=mal(R*4),*dr=mal(R*4);
        aclTensor*tx=T({R,Dd},dx),*tw=T({Dd},dw),*tb=T({Dd},db),*to=T({R,Dd},dout),*tm=T({R},dm),*trr=T({R},dr);
        WS(aclnnLayerNormGetWorkspaceSize(tx,IA({Dd}),tw,tb,EPS,to,tm,trr,&ws,&ex)); CK(aclnnLayerNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("layernorm","out",cmp(dn(dout,RD),rf(pre+".out",RD)),2e-4);
    } else if(op=="rmsnorm"){
        auto x=rf(pre+".x",RD),g=rf(pre+".g",Dd);
        void*dx=upf(x),*dg=upf(g),*dout=mal(RD*4),*dr=mal(R*4);
        aclTensor*tx=T({R,Dd},dx),*tg=T({Dd},dg),*to=T({R,Dd},dout),*trr=T({R},dr);
        WS(aclnnRmsNormGetWorkspaceSize(tx,tg,EPS,to,&ws,&ex)); (void)trr; CK(aclnnRmsNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("rmsnorm","out",cmp(dn(dout,RD),rf(pre+".out",RD)),2e-4);
    } else if(op=="addrmsnorm"){
        auto x=rf(pre+".x",RD),res=rf(pre+".res",RD),g=rf(pre+".g",Dd);
        void*dx=upf(x),*dres=upf(res),*dg=upf(g),*dy=mal(RD*4),*drs=mal(RD*4);
        aclTensor*tx=T({R,Dd},dx),*tres=T({R,Dd},dres),*tg=T({Dd},dg),*ty=T({R,Dd},dy),*trs=T({R,Dd},drs);
        WS(aclnnAddRmsNormGetWorkspaceSize(tx,tres,tg,EPS,ty,trs,&ws,&ex)); CK(aclnnAddRmsNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("addrmsnorm","y",cmp(dn(dy,RD),rf(pre+".y",RD)),2e-4);
        ok&=rep("addrmsnorm","rs",cmp(dn(drs,RD),rf(pre+".rs",RD)),2e-4);
    } else { // 4D norms
        auto x=rf(pre+".x",FOUR),w=rf(pre+".w",C),b=rf(pre+".b",C);
        void*dx=upf(x),*dw=upf(w),*db=upf(b),*dout=mal(FOUR*4);
        aclTensor*tx=T({N,C,Hh,Hh},dx),*tw=T({C},dw),*tb=T({C},db),*to=T({N,C,Hh,Hh},dout);
        if(op=="groupnorm"){ WS(aclnnGroupNormGetWorkspaceSize(tx,tw,tb,G,EPS,to,&ws,&ex)); CK(aclnnGroupNorm(wsp,ws,ex,s)); }
        else if(op=="instancenorm"){ WS(aclnnInstanceNormGetWorkspaceSize(tx,tw,tb,EPS,to,&ws,&ex)); CK(aclnnInstanceNorm(wsp,ws,ex,s)); }
        else { auto m=rf(pre+".m",C),v=rf(pre+".v",C); void*dm=upf(m),*dv=upf(v); aclTensor*tm=T({C},dm),*tv=T({C},dv);
               WS(aclnnBatchNormGetWorkspaceSize(tx,tw,tb,tm,tv,EPS,to,&ws,&ex)); CK(aclnnBatchNorm(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),"out",cmp(dn(dout,FOUR),rf(pre+".out",FOUR)),2e-4);
    }
    return ok?0:1;
}
