// Independent backward check for loss + norm gradients vs PyTorch autograd (torch_lossnorm_grad.py).
//   Losses (reduction=none): gradInput = gradOut ⊙ dLoss/dself.  inputs self,target,gradOut → gradInput.
//   RmsNorm/LayerNorm backward: y=norm(x,gamma[,beta]); y.backward(gradY) → gradX, gradGamma[, gradBeta].
// gen writes the inputs; the python oracle writes the reference grads; check runs the shim + compares.
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
static const int64_t N=4096;          // loss element count
static const int64_t R=16, D=64;      // norm rows x lastdim

static bool is_norm(const std::string&o){ return o=="rmsnorm"||o=="layernorm"; }
static void wb(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rb(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* up(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static std::vector<float> down(void*d,int64_t n){ std::vector<float> v(n); CK(aclrtMemcpy(v.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string mode=argv[1], op=argv[2], pre=argv[3];
    if(mode=="gen"){
        if(!is_norm(op)){
            std::vector<float> self(N),tgt(N),go(N);
            for(int64_t i=0;i<N;i++){
                go[i]=(float)(1.3*std::cos(i*0.017));
                if(op=="bce"){ self[i]=(float)(0.5+0.4*std::sin(i*0.013)); tgt[i]=(float)(0.5+0.4*std::cos(i*0.011)); }   // (0.1,0.9)
                else if(op=="kldiv"){ self[i]=(float)(-1.0+std::sin(i*0.013)); tgt[i]=(float)(0.2+0.15*std::cos(i*0.011)); } // self=log-prob, tgt>0
                else { self[i]=(float)(2.0*std::sin(i*0.013)); tgt[i]=(float)(2.0*std::cos(i*0.011)); }
            }
            wb(pre+".self",self); wb(pre+".tgt",tgt); wb(pre+".go",go);
        } else {
            std::vector<float> x(R*D),gamma(D),beta(D),gy(R*D);
            for(int64_t i=0;i<R*D;i++){ x[i]=(float)(2.0*std::sin(i*0.013)); gy[i]=(float)(1.1*std::cos(i*0.017)); }
            for(int64_t i=0;i<D;i++){ gamma[i]=(float)(1.0+0.3*std::sin(i*0.05)); beta[i]=(float)(0.2*std::cos(i*0.07)); }
            wb(pre+".x",x); wb(pre+".gamma",gamma); wb(pre+".beta",beta); wb(pre+".gy",gy);
        }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    // check
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    double eps=1e-5; uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr;
    if(!is_norm(op)){
        auto self=rb(pre+".self",N), tgt=rb(pre+".tgt",N), go=rb(pre+".go",N), ref=rb(pre+".gi",N);
        void *dg=up(go),*ds=up(self),*dt=up(tgt),*dz; CK(aclrtMalloc(&dz,N*4,ACL_MEM_MALLOC_HUGE_FIRST));
        aclTensor *tg=T({N},dg),*ts=T({N},ds),*tt=T({N},dt),*tz=T({N},dz);
        if(op=="l1") CK(aclnnL1LossBackwardGetWorkspaceSize(tg,ts,tt,0,tz,&ws,&ex));
        else if(op=="smoothl1") CK(aclnnSmoothL1LossBackwardGetWorkspaceSize(tg,ts,tt,0,1.0,tz,&ws,&ex));
        else if(op=="mse") CK(aclnnMseLossBackwardGetWorkspaceSize(tg,ts,tt,0,tz,&ws,&ex));
        else if(op=="bce") CK(aclnnBinaryCrossEntropyBackwardGetWorkspaceSize(tg,ts,tt,nullptr,0,tz,&ws,&ex));
        else if(op=="kldiv") CK(aclnnKlDivBackwardGetWorkspaceSize(tg,ts,tt,0,false,tz,&ws,&ex));
        else if(op=="softmargin") CK(aclnnSoftMarginLossBackwardGetWorkspaceSize(tg,ts,tt,0,tz,&ws,&ex));
        else { fprintf(stderr,"unknown loss %s\n",op.c_str()); return 2; }
        if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if(op=="l1") CK(aclnnL1LossBackward(wsp,ws,ex,s));
        else if(op=="smoothl1") CK(aclnnSmoothL1LossBackward(wsp,ws,ex,s));
        else if(op=="mse") CK(aclnnMseLossBackward(wsp,ws,ex,s));
        else if(op=="bce") CK(aclnnBinaryCrossEntropyBackward(wsp,ws,ex,s));
        else if(op=="kldiv") CK(aclnnKlDivBackward(wsp,ws,ex,s));
        else CK(aclnnSoftMarginLossBackward(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        double e=cmp(down(dz,N),ref); double tol=2e-4;
        printf("[%s] gradInput normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
        return e<=tol?0:1;
    } else {
        auto x=rb(pre+".x",R*D), gamma=rb(pre+".gamma",D), gy=rb(pre+".gy",R*D);
        auto rgx=rb(pre+".gx",R*D), rgg=rb(pre+".gg",D);
        void *dgy=up(gy),*dx=up(x),*dga=up(gamma),*dgx,*dgg; CK(aclrtMalloc(&dgx,R*D*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dgg,D*4,ACL_MEM_MALLOC_HUGE_FIRST));
        aclTensor *tgy=T({R,D},dgy),*tx=T({R,D},dx),*tga=T({D},dga),*tgx=T({R,D},dgx),*tgg=T({D},dgg);
        if(op=="rmsnorm"){ CK(aclnnRmsNormBackwardGetWorkspaceSize(tgy,tx,tga,eps,tgx,tgg,&ws,&ex)); if(ws)CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclnnRmsNormBackward(wsp,ws,ex,s)); }
        else { int64_t ns[1]={D}; aclIntArray*nsh=aclCreateIntArray(ns,1); void*dgb; CK(aclrtMalloc(&dgb,D*4,ACL_MEM_MALLOC_HUGE_FIRST)); aclTensor*tgb=T({D},dgb);
               CK(aclnnLayerNormBackwardGetWorkspaceSize(tgy,tx,tga,nsh,eps,tgx,tgg,tgb,&ws,&ex)); if(ws)CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclnnLayerNormBackward(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
               auto rgb=rb(pre+".gb",D); double eb=cmp(down(dgb,D),rgb); printf("[layernorm] gradBeta  normalized_err=%.3e %s\n",eb,eb<=2e-4?"PASS":"FAIL"); }
        CK(aclrtSynchronizeStream(s));
        double ex_=cmp(down(dgx,R*D),rgx), eg=cmp(down(dgg,D),rgg); double tol=2e-3;   // norm grads accumulate; looser
        bool ok = ex_<=tol && eg<=tol;
        printf("[%s] gradX=%.3e gradGamma=%.3e (tol=%.0e)  %s\n",op.c_str(),ex_,eg,tol,ok?"PASS":"FAIL");
        return ok?0:1;
    }
}
