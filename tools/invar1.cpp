// Invariant test for the quant family: dequant(quant(x)) ≈ x (error bounded by one quant step).
// Quantize:  q=clamp(round(x*scale+offset),-128,127);  Dequantize: y=(q-offset)*scale  -> so the
// inverse uses 1/scale on dequant (offset cancels). DynamicQuant returns a per-row scale; we dequant
// on the host with it. No torch reference needed — the round-trip is its own oracle.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t R=8, C=16;
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<int8_t> dni8(void*d,int64_t k){ std::vector<int8_t> v(k); CK(aclrtMemcpy(v.data(),k,d,k,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double ncmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string op=argv[1]; int64_t N=R*C;
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    std::vector<float> x(N); for(int64_t i=0;i<N;i++)x[i]=(float)std::sin(i*0.137+0.3);   // in [-1,1]
    void* dx=upf(x);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ ws=0;wsp=nullptr; CK(rc); if(ws)wsp=mal(ws); };
    double e=0, tol=0;
    if(op=="dynamicquant"){
        void* dq=mal(N), *dsc=mal(R*4);
        aclTensor *tx=T({R,C},dx),*tq=T({R,C},dq,ACL_INT8),*tsc=T({R},dsc);
        WS(aclnnDynamicQuantGetWorkspaceSize(tx,tq,tsc,&ws,&ex)); CK(aclnnDynamicQuant(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        auto q=dni8(dq,N); auto sc=dnf(dsc,R);
        std::vector<float> y(N); for(int64_t r=0;r<R;r++)for(int64_t c=0;c<C;c++)y[r*C+c]=(float)q[r*C+c]*sc[r];  // host dequant
        e=ncmp(y,x); tol=1.0/127.0;   // one quant step
    } else {
        bool perchan = (op=="quantize_pc"||op=="ascendquant_pc"||op=="ascendantiquant");
        bool ascend  = (op=="ascendquant_pc"||op=="ascendantiquant");
        int64_t SL = perchan ? C : 1;
        std::vector<float> scQ(SL), scD(SL), off(SL,0.f);
        for(int64_t i=0;i<SL;i++){ scQ[i]=perchan?(float)(90.0+10.0*std::sin(i)):110.f; scD[i]=1.f/scQ[i]; }
        void *dscQ=upf(scQ),*dscD=upf(scD),*doff=upf(off),*dq=mal(N),*dy=mal(N*4);
        aclTensor *tx=T({R,C},dx),*tscQ=T({SL},dscQ),*tscD=T({SL},dscD),*toff=T({SL},doff),*tq=T({R,C},dq,ACL_INT8),*ty=T({R,C},dy);
        if(ascend){ WS(aclnnAscendQuantGetWorkspaceSize(tx,tscQ,toff,tq,&ws,&ex)); CK(aclnnAscendQuant(wsp,ws,ex,s)); }
        else { WS(aclnnQuantizeGetWorkspaceSize(tx,tscQ,toff,tq,&ws,&ex)); CK(aclnnQuantize(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s));
        if(op=="ascendantiquant"){ WS(aclnnAscendAntiQuantGetWorkspaceSize(tq,tscD,toff,ty,&ws,&ex)); CK(aclnnAscendAntiQuant(wsp,ws,ex,s)); }
        else if(ascend){ WS(aclnnAscendDequantGetWorkspaceSize(tq,tscD,toff,ty,&ws,&ex)); CK(aclnnAscendDequant(wsp,ws,ex,s)); }
        else { WS(aclnnDequantizeGetWorkspaceSize(tq,tscD,toff,ty,&ws,&ex)); CK(aclnnDequantize(wsp,ws,ex,s)); }
        CK(aclrtSynchronizeStream(s));
        e=ncmp(dnf(dy,N),x); tol=2.0/90.0;   // ~one quant step (min scale 90)
    }
    printf("[%s] roundtrip normalized_err=%.3e (tol=%.2e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
