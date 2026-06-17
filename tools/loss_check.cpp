// Independent forward check for loss ops across reduction modes vs PyTorch (torch_loss.py).
//   gen|check <op> <prefix> <reduction>   reduction: 0=none,1=mean,2=sum
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t N=64,C=5,D=8; static const double M=0.5;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wi(const std::string&p,const std::vector<int64_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),8,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3]; int red=atoi(argv[4]);
    bool pw=(op=="l1"||op=="smoothl1"||op=="mse"||op=="bce"||op=="kldiv"||op=="softmargin");
    if(mode=="gen"){
        if(pw){ std::vector<float> self(N),tgt(N);
            for(int64_t i=0;i<N;i++){
                if(op=="bce"){ self[i]=(float)(0.5+0.4*std::sin(i*0.13)); tgt[i]=(float)(0.5+0.4*std::cos(i*0.11)); }
                else if(op=="kldiv"){ self[i]=(float)(-1.0+std::sin(i*0.13)); tgt[i]=(float)(0.3+0.15*std::cos(i*0.11)); }
                else if(op=="softmargin"){ self[i]=(float)(2.0*std::sin(i*0.13)); tgt[i]=(i%2)?1.f:-1.f; }
                else { self[i]=(float)(2.0*std::sin(i*0.13)); tgt[i]=(float)(2.0*std::cos(i*0.11)); }
            } wf(pre+".self",self); wf(pre+".tgt",tgt);
        } else if(op=="nllloss"){ std::vector<float> lp(N*C); std::vector<int64_t> tgt(N);
            for(int64_t i=0;i<N*C;i++)lp[i]=(float)(-1.0-std::fabs(std::sin(i*0.07)));   // log-probs (negative)
            for(int64_t i=0;i<N;i++)tgt[i]=i%C; wf(pre+".lp",lp); wi(pre+".tgt",tgt);
        } else if(op=="cosine"){ std::vector<float> x1(N*D),x2(N*D),tgt(N);
            for(int64_t i=0;i<N*D;i++){x1[i]=(float)(std::sin(i*0.09));x2[i]=(float)(std::cos(i*0.07));}
            for(int64_t i=0;i<N;i++)tgt[i]=(i%2)?1.f:-1.f; wf(pre+".x1",x1);wf(pre+".x2",x2);wf(pre+".tgt",tgt);
        } else { std::vector<float> x1(N),x2(N),y(N);
            for(int64_t i=0;i<N;i++){x1[i]=(float)(std::sin(i*0.09));x2[i]=(float)(std::cos(i*0.07));y[i]=(i%2)?1.f:-1.f;}
            wf(pre+".x1",x1);wf(pre+".x2",x2);wf(pre+".y",y);
        }
        printf("[gen] %s r%d\n",op.c_str(),red); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    int64_t on = (red==0)? N : 1;
    void* dout = mal((on<1?1:on)*4);
    if(pw){
        auto self=rf(pre+".self",N),tgt=rf(pre+".tgt",N); void*ds=upf(self),*dt=upf(tgt);
        aclTensor*tself=T({N},ds),*tt=T({N},dt),*to=T(red==0?std::vector<int64_t>{N}:std::vector<int64_t>{1},dout);
        if(op=="l1"){ WS(aclnnL1LossGetWorkspaceSize(tself,tt,red,to,&ws,&ex)); CK(aclnnL1Loss(wsp,ws,ex,s)); }
        else if(op=="smoothl1"){ WS(aclnnSmoothL1LossGetWorkspaceSize(tself,tt,red,1.0,to,&ws,&ex)); CK(aclnnSmoothL1Loss(wsp,ws,ex,s)); }
        else if(op=="mse"){ WS(aclnnMseLossGetWorkspaceSize(tself,tt,red,to,&ws,&ex)); CK(aclnnMseLoss(wsp,ws,ex,s)); }
        else if(op=="bce"){ WS(aclnnBinaryCrossEntropyGetWorkspaceSize(tself,tt,red,to,&ws,&ex)); CK(aclnnBinaryCrossEntropy(wsp,ws,ex,s)); }
        else if(op=="kldiv"){ WS(aclnnKlDivGetWorkspaceSize(tself,tt,red,to,&ws,&ex)); CK(aclnnKlDiv(wsp,ws,ex,s)); }
        else { WS(aclnnSoftMarginLossGetWorkspaceSize(tself,tt,red,to,&ws,&ex)); CK(aclnnSoftMarginLoss(wsp,ws,ex,s)); }
    } else if(op=="nllloss"){
        auto lp=rf(pre+".lp",N*C); std::vector<int64_t> tgt(N); { FILE*f=fopen((pre+".tgt").c_str(),"rb"); if((int64_t)fread(tgt.data(),8,N,f)!=N){fprintf(stderr,"short tgt\n");return 1;} fclose(f); }
        void*dlp=upf(lp),*dt=upi(tgt); aclTensor*tlp=T({N,C},dlp),*tt=T({N},dt,ACL_INT64),*to=T(red==0?std::vector<int64_t>{N}:std::vector<int64_t>{1},dout);
        WS(aclnnNLLLossGetWorkspaceSize(tlp,tt,red,to,&ws,&ex)); CK(aclnnNLLLoss(wsp,ws,ex,s));
    } else if(op=="cosine"){
        auto x1=rf(pre+".x1",N*D),x2=rf(pre+".x2",N*D),tgt=rf(pre+".tgt",N); void*dx1=upf(x1),*dx2=upf(x2),*dt=upf(tgt);
        aclTensor*tx1=T({N,D},dx1),*tx2=T({N,D},dx2),*tt=T({N},dt),*to=T(red==0?std::vector<int64_t>{N}:std::vector<int64_t>{1},dout);
        WS(aclnnCosineEmbeddingLossGetWorkspaceSize(tx1,tx2,tt,M,red,to,&ws,&ex)); CK(aclnnCosineEmbeddingLoss(wsp,ws,ex,s));
    } else { // marginranking
        auto x1=rf(pre+".x1",N),x2=rf(pre+".x2",N),y=rf(pre+".y",N); void*dx1=upf(x1),*dx2=upf(x2),*dy=upf(y);
        aclTensor*tx1=T({N},dx1),*tx2=T({N},dx2),*ty=T({N},dy),*to=T(red==0?std::vector<int64_t>{N}:std::vector<int64_t>{1},dout);
        WS(aclnnMarginRankingLossGetWorkspaceSize(tx1,tx2,ty,M,red,to,&ws,&ex)); CK(aclnnMarginRankingLoss(wsp,ws,ex,s));
    }
    CK(aclrtSynchronizeStream(s));
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s r%d] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),red,e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
