// Independent backward check for conv / pool / index gradients vs PyTorch autograd (torch_nn_grad.py).
// These backward ops route or accumulate the upstream gradient; their main test cases exercise one
// reduction/shape, so an independent autograd oracle catches "happy-path-only" bugs (cf. the loss-bwd
// reduction=none and digamma asymptotic bugs both surfaced this way).
//   gen   <op> <prefix>   write the inputs (float .x/.w/.b/.go/.grad + int64 .ids/.idx)
//   check <op> <prefix>   run the shim backward, compare each grad against the torch reference
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

// shapes shared with torch_nn_grad.py
static const int64_t N=2,Cin=3,H=6,Cout=4,K=3;            // conv (stride1 pad1 -> Hout=H)
static const int64_t PN=2,PC=3,PH=8;                       // pool (8x8 -> 4x4)
static const int64_t V=10,D=5,Ln=12;                       // embedding
static const int64_t GA=6,GB=4,GL=5;                       // gather=index_select: x[GA,GB], idx[GL] along dim0

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wi(const std::string&p,const std::vector<int64_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),8,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<int64_t> ri(const std::string&p,int64_t n){ std::vector<int64_t> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),8,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t bytes){ void*d; CK(aclrtMalloc(&d,bytes,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t n){ std::vector<float> v(n); CK(aclrtMemcpy(v.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static bool rep(const char*op,const char*what,double e,double tol){ printf("[%s] %-9s normalized_err=%.3e (tol=%.0e)  %s\n",op,what,e,tol,e<=tol?"PASS":"FAIL"); return e<=tol; }

int main(int argc,char**argv){
    std::string mode=argv[1], op=argv[2], pre=argv[3];
    if(mode=="gen"){
        if(op=="conv"){
            std::vector<float> x(N*Cin*H*H),w(Cout*Cin*K*K),b(Cout),go(N*Cout*H*H);
            for(size_t i=0;i<x.size();i++)x[i]=(float)(1.5*std::sin(i*0.021));
            for(size_t i=0;i<w.size();i++)w[i]=(float)(0.7*std::cos(i*0.033));
            for(size_t i=0;i<b.size();i++)b[i]=(float)(0.2*i-0.3);
            for(size_t i=0;i<go.size();i++)go[i]=(float)(1.1*std::cos(i*0.017));
            wf(pre+".x",x);wf(pre+".w",w);wf(pre+".b",b);wf(pre+".go",go);
        } else if(op=="avgpool"||op=="maxpool"||op=="adaptiveavg"){
            std::vector<float> x(PN*PC*PH*PH),go(PN*PC*4*4);
            for(size_t i=0;i<x.size();i++)x[i]=(float)(2.0*std::sin(i*0.019));
            for(size_t i=0;i<go.size();i++)go[i]=(float)(1.3*std::cos(i*0.023));
            wf(pre+".x",x);wf(pre+".go",go);
        } else if(op=="embedding"){
            std::vector<float> w(V*D),grad(Ln*D); std::vector<int64_t> ids(Ln);
            for(size_t i=0;i<w.size();i++)w[i]=0.f;                        // weight content irrelevant to grad
            for(size_t i=0;i<grad.size();i++)grad[i]=(float)(1.0*std::sin(i*0.07));
            for(int64_t i=0;i<Ln;i++)ids[i]=(int64_t)((i*7+3)%V);          // repeats -> exercises accumulation
            wf(pre+".w",w);wf(pre+".grad",grad);wi(pre+".ids",ids);
        } else if(op=="gather"){
            std::vector<float> x(GA*GB),go(GL*GB); std::vector<int64_t> idx(GL);
            for(size_t i=0;i<x.size();i++)x[i]=0.f;
            for(size_t i=0;i<go.size();i++)go[i]=(float)(1.0+std::sin(i*0.11));
            for(int64_t l=0;l<GL;l++)idx[l]=(int64_t)((l*2+1)%GA);   // repeats -> exercises scatter accumulation
            wf(pre+".x",x);wf(pre+".go",go);wi(pre+".idx",idx);
        } else { fprintf(stderr,"gen: unknown op %s\n",op.c_str()); return 2; }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr;
    auto WS=[&](int rc){ CK(rc); if(ws){wsp=mal(ws);} };
    bool ok=true;
    if(op=="conv"){
        auto x=rf(pre+".x",N*Cin*H*H),w=rf(pre+".w",Cout*Cin*K*K),b=rf(pre+".b",Cout),go=rf(pre+".go",N*Cout*H*H);
        void*dx=upf(x),*dw=upf(w),*dgo=upf(go);
        void*dgi=mal(x.size()*4),*dgw=mal(w.size()*4),*dgb=mal(b.size()*4);
        aclTensor *tgo=T({N,Cout,H,H},dgo),*tx=T({N,Cin,H,H},dx),*tw=T({Cout,Cin,K,K},dw);
        aclTensor *tgi=T({N,Cin,H,H},dgi),*tgw=T({Cout,Cin,K,K},dgw),*tgb=T({Cout},dgb);
        WS(aclnnConvolutionBackwardGetWorkspaceSize(tgo,tx,tw,IA({Cout}),IA({1,1}),IA({1,1}),IA({1,1}),false,IA({0,0}),1,tgi,tgw,tgb,0,&ws,&ex));
        CK(aclnnConvolutionBackward(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("conv","gradInput", cmp(dn(dgi,x.size()), rf(pre+".gi",x.size())),2e-3);
        ok&=rep("conv","gradWeight",cmp(dn(dgw,w.size()), rf(pre+".gw",w.size())),2e-3);
        ok&=rep("conv","gradBias",  cmp(dn(dgb,b.size()), rf(pre+".gb",b.size())),2e-3);
    } else if(op=="avgpool"||op=="maxpool"||op=="adaptiveavg"){
        int64_t IN=PN*PC*PH*PH, ON=PN*PC*4*4;
        auto x=rf(pre+".x",IN),go=rf(pre+".go",ON);
        void*dx=upf(x),*dgo=upf(go),*dgi=mal(IN*4);
        aclTensor *tx=T({PN,PC,PH,PH},dx),*tgo=T({PN,PC,4,4},dgo),*tgi=T({PN,PC,PH,PH},dgi);
        if(op=="adaptiveavg"){
            WS(aclnnAdaptiveAvgPool2dBackwardGetWorkspaceSize(tgo,tgi,&ws,&ex)); CK(aclnnAdaptiveAvgPool2dBackward(wsp,ws,ex,s));
        } else if(op=="avgpool"){
            void*dy=mal(ON*4); aclTensor*ty=T({PN,PC,4,4},dy);                       // forward output y (shape only)
            WS(aclnnAvgPool2dGetWorkspaceSize(tx,IA({2,2}),IA({2,2}),IA({0,0}),ty,&ws,&ex)); CK(aclnnAvgPool2d(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ws=0;wsp=nullptr;
            WS(aclnnAvgPool2dBackwardGetWorkspaceSize(tx,ty,tgo,IA({2,2}),IA({2,2}),IA({0,0}),tgi,&ws,&ex)); CK(aclnnAvgPool2dBackward(wsp,ws,ex,s));
        } else { // maxpool: run forward to obtain argmax indices, then backward
            void*dy=mal(ON*4),*didx=mal(ON*8); aclTensor*ty=T({PN,PC,4,4},dy),*tidx=T({PN,PC,4,4},didx,ACL_INT64);
            WS(aclnnMaxPool2dWithIndicesGetWorkspaceSize(tx,IA({2,2}),IA({2,2}),IA({0,0}),ty,tidx,&ws,&ex)); CK(aclnnMaxPool2dWithIndices(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); ws=0;wsp=nullptr;
            WS(aclnnMaxPool2dWithIndicesBackwardGetWorkspaceSize(tgo,tx,tidx,IA({2,2}),IA({2,2}),IA({0,0}),IA({1,1}),false,tgi,&ws,&ex)); CK(aclnnMaxPool2dWithIndicesBackward(wsp,ws,ex,s));
        }
        CK(aclrtSynchronizeStream(s));
        ok&=rep(op.c_str(),"gradInput",cmp(dn(dgi,IN),rf(pre+".gi",IN)),2e-3);
    } else if(op=="embedding"){
        auto grad=rf(pre+".grad",Ln*D); auto ids=ri(pre+".ids",Ln);
        void*dgrad=upf(grad),*dids=upi(ids),*dgw=mal(V*D*4);
        aclTensor *tgrad=T({Ln,D},dgrad),*tids=T({Ln},dids,ACL_INT64),*tgw=T({V,D},dgw);
        WS(aclnnEmbeddingDenseBackwardGetWorkspaceSize(tgrad,tids,tgw,&ws,&ex)); CK(aclnnEmbeddingDenseBackward(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("embedding","gradWeight",cmp(dn(dgw,V*D),rf(pre+".gw",V*D)),2e-4);
    } else if(op=="gather"){   // index_select convention: 1-D index along dim 0
        auto go=rf(pre+".go",GL*GB); auto idx=ri(pre+".idx",GL);
        void*dgo=upf(go),*didx=upi(idx),*dgi=mal(GA*GB*4);
        aclTensor *tgo=T({GL,GB},dgo),*tidx=T({GL},didx,ACL_INT64),*tgi=T({GA,GB},dgi);
        WS(aclnnGatherBackwardGetWorkspaceSize(tgo,0,tidx,tgi,&ws,&ex)); CK(aclnnGatherBackward(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ok&=rep("gather","gradInput",cmp(dn(dgi,GA*GB),rf(pre+".gi",GA*GB)),2e-4);
    } else { fprintf(stderr,"check: unknown op %s\n",op.c_str()); return 2; }
    return ok?0:1;
}
