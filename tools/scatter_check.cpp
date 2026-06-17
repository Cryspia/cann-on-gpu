// Independent forward check for the scatter / index_put family vs PyTorch (torch_scatter.py).
// Each op is checked against the convention the shim ACTUALLY implements (verified from the impl):
// Scatter/ScatterAdd are 1-D-index-along-dim0 (index_copy/index_add), not torch.scatter; ScatterValue
// is the real per-dim full-shape-index torch.scatter_. Replace-mode ops use unique indices (duplicate
// writes race / are order-undefined). This catches happy-path-only bugs the same way the loss/gather sweeps did.
//   gen   <op> <prefix>   write inputs (.self/.src float, .idx int64)
//   check <op> <prefix>   run the shim, compare .out against the torch reference
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

static const int64_t SV=6,SROW=4,SL=5;      // dim0 ops
static const int64_t FV=6,FROW=4,FL=3;      // indexfill
static const int64_t VR=4,VC=6,VK=3;        // scattervalue
static const double ALPHA=2.0,FILLV=3.14,SVALUE=2.71;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wi(const std::string&p,const std::vector<int64_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),8,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t n){ std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<int64_t> ri(const std::string&p,int64_t n){ std::vector<int64_t> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),8,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t n){ std::vector<float> v(n); CK(aclrtMemcpy(v.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string mode=argv[1], op=argv[2], pre=argv[3];
    bool iput = (op=="indexput"||op=="indexputadd");
    bool dim0 = iput || op=="scatter"||op=="scatteradd"||op=="indexadd"||op=="indexcopy";
    bool repl = (op=="scatter"||op=="indexcopy"||op=="indexput"); // replace-mode -> unique idx
    if(mode=="gen"){
        if(dim0){
            std::vector<float> self(SV*SROW),src(SL*SROW); std::vector<int64_t> idx(SL);
            for(size_t i=0;i<self.size();i++)self[i]=(float)(1.7*std::sin(i*0.021));
            for(size_t i=0;i<src.size();i++)src[i]=(float)(0.9*std::cos(i*0.031)+0.3);
            int64_t uniq[SL]={0,2,4,1,5}, dup[SL]={0,2,2,4,0};
            for(int64_t i=0;i<SL;i++)idx[i]=repl?uniq[i]:dup[i];
            wf(pre+".self",self);wf(pre+".src",src);wi(pre+".idx",idx);
        } else if(op=="indexfill"){
            std::vector<float> self(FV*FROW); std::vector<int64_t> idx={1,3,5};
            for(size_t i=0;i<self.size();i++)self[i]=(float)(1.7*std::sin(i*0.021));
            wf(pre+".self",self);wi(pre+".idx",idx);
        } else if(op=="scattervalue"){
            std::vector<float> self(VR*VC); std::vector<int64_t> idx(VR*VK);
            for(size_t i=0;i<self.size();i++)self[i]=(float)(1.7*std::sin(i*0.021));
            int64_t cols[VK]={1,3,5};                    // per-row unique target columns
            for(int64_t r=0;r<VR;r++)for(int64_t k=0;k<VK;k++)idx[r*VK+k]=cols[k];
            wf(pre+".self",self);wi(pre+".idx",idx);
        } else { fprintf(stderr,"gen: unknown %s\n",op.c_str()); return 2; }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr;
    auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    double e=0; int64_t on=0;
    if(dim0){
        auto self=rf(pre+".self",SV*SROW),src=rf(pre+".src",SL*SROW); auto idx=ri(pre+".idx",SL);
        void*ds=upf(self),*dsrc=upf(src),*didx=upi(idx),*dout=mal(SV*SROW*4); on=SV*SROW;
        aclTensor *tself=T({SV,SROW},ds),*tsrc=T({SL,SROW},dsrc),*tidx=T({SL},didx,ACL_INT64),*tout=T({SV,SROW},dout);
        void* result = iput ? ds : dout;       // IndexPutImpl is in-place (mutates selfRef buffer ds)
        if(iput){ aclTensor* iarr[1]={tidx}; aclTensorList* il=aclCreateTensorList(iarr,1);
            WS(aclnnIndexPutImplGetWorkspaceSize(tself,il,tsrc,op=="indexputadd",false,&ws,&ex)); CK(aclnnIndexPutImpl(wsp,ws,ex,s)); }
        else if(op=="scatter"){    WS(aclnnScatterGetWorkspaceSize(tself,0,tidx,tsrc,0,tout,&ws,&ex)); CK(aclnnScatter(wsp,ws,ex,s)); }
        else if(op=="scatteradd"){ WS(aclnnScatterAddGetWorkspaceSize(tself,tidx,tsrc,tout,&ws,&ex)); CK(aclnnScatterAdd(wsp,ws,ex,s)); }
        else if(op=="indexadd"){   WS(aclnnIndexAddGetWorkspaceSize(tself,0,tidx,tsrc,ALPHA,tout,&ws,&ex)); CK(aclnnIndexAdd(wsp,ws,ex,s)); }
        else {                     WS(aclnnIndexCopyGetWorkspaceSize(tself,0,tidx,tsrc,tout,&ws,&ex)); CK(aclnnIndexCopy(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s)); e=cmp(dn(result,on),rf(pre+".out",on));
    } else if(op=="indexfill"){
        auto self=rf(pre+".self",FV*FROW); auto idx=ri(pre+".idx",FL);
        void*ds=upf(self),*didx=upi(idx),*dout=mal(FV*FROW*4); on=FV*FROW;
        aclTensor *tself=T({FV,FROW},ds),*tidx=T({FL},didx,ACL_INT64),*tout=T({FV,FROW},dout);
        WS(aclnnIndexFillGetWorkspaceSize(tself,0,tidx,FILLV,tout,&ws,&ex)); CK(aclnnIndexFill(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s)); e=cmp(dn(dout,on),rf(pre+".out",on));
    } else { // scattervalue
        auto self=rf(pre+".self",VR*VC); auto idx=ri(pre+".idx",VR*VK);
        void*ds=upf(self),*didx=upi(idx),*dout=mal(VR*VC*4); on=VR*VC;
        aclTensor *tself=T({VR,VC},ds),*tidx=T({VR,VK},didx,ACL_INT64),*tout=T({VR,VC},dout);
        aclScalar*val=aclCreateScalar((void*)&SVALUE,ACL_DOUBLE);
        WS(aclnnScatterValueGetWorkspaceSize(tself,1,tidx,val,tout,&ws,&ex)); CK(aclnnScatterValue(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s)); e=cmp(dn(dout,on),rf(pre+".out",on));
    }
    double tol=2e-4; printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
