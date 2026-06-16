// Independent forward check for vision/conv/interpolation vs PyTorch (torch_vision.py).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static std::vector<float> g(int64_t n,double f,double p){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*f+p); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="convolution"){ wf(pre+".x",g(1*3*6*6,0.05,0.2)); wf(pre+".w",g(4*3*3*3,0.07,0.4)); wf(pre+".b",g(4,0.3,0.1)); }
        else if(op=="gridsampler2d"){ wf(pre+".x",g(1*2*4*4,0.11,0.2)); std::vector<float> grid(1*3*3*2); for(size_t i=0;i<grid.size();i++)grid[i]=(float)(0.8*std::sin(i*0.5)); wf(pre+".g",grid); }
        else if(op=="pixelshuffle"){ wf(pre+".x",g(1*8*2*2,0.11,0.2)); }
        else if(op=="pixelunshuffle"){ wf(pre+".x",g(1*2*4*4,0.11,0.2)); }
        else if(op=="upnearest3d"||op=="uptrilinear3d"){ wf(pre+".x",g(1*2*2*2*2,0.11,0.2)); }
        else if(op=="upbilinear2d"){ wf(pre+".x",g(1*2*2*2,0.11,0.2)); }
        else if(op=="upbicubic2d"){ wf(pre+".x",g(1*2*3*3,0.11,0.2)); }
        else if(op=="upnearest1d"||op=="uplinear1d"){ wf(pre+".x",g(1*2*4,0.11,0.2)); }
        else if(op=="affinegrid"){ std::vector<float> th={1,0,0.1f, 0,1,-0.1f}; wf(pre+".x",th); }
        else if(op=="lrn"){ wf(pre+".x",g(1*4*2*2,0.11,0.2)); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    int64_t on=0; void*dout=nullptr; ex=nullptr; ws=0; wsp=nullptr;
    if(op=="convolution"){ auto x=rf(pre+".x",108),w=rf(pre+".w",108),b=rf(pre+".b",4); void*dx=upf(x),*dw=upf(w),*db=upf(b); dout=mal(1*4*6*6*4); on=144;
        CK(aclnnConvolutionGetWorkspaceSize(T({1,3,6,6},dx),T({4,3,3,3},dw),T({4},db),IA({1,1}),IA({1,1}),IA({1,1}),false,IA({0,0}),1,T({1,4,6,6},dout),0,&ws,&ex)); run(aclnnConvolution,s); }
    else if(op=="gridsampler2d"){ auto x=rf(pre+".x",32),gr=rf(pre+".g",18); void*dx=upf(x),*dg=upf(gr); dout=mal(1*2*3*3*4); on=18;
        CK(aclnnGridSampler2DGetWorkspaceSize(T({1,2,4,4},dx),T({1,3,3,2},dg),0,0,false,T({1,2,3,3},dout),&ws,&ex)); run(aclnnGridSampler2D,s); }
    else if(op=="pixelshuffle"){ auto x=rf(pre+".x",32); void*dx=upf(x); dout=mal(32*4); on=32; CK(aclnnPixelShuffleGetWorkspaceSize(T({1,8,2,2},dx),2,T({1,2,4,4},dout),&ws,&ex)); run(aclnnPixelShuffle,s); }
    else if(op=="pixelunshuffle"){ auto x=rf(pre+".x",32); void*dx=upf(x); dout=mal(32*4); on=32; CK(aclnnPixelUnshuffleGetWorkspaceSize(T({1,2,4,4},dx),2,T({1,8,2,2},dout),&ws,&ex)); run(aclnnPixelUnshuffle,s); }
    else if(op=="upnearest3d"){ auto x=rf(pre+".x",16); void*dx=upf(x); dout=mal(128*4); on=128; CK(aclnnUpsampleNearest3dGetWorkspaceSize(T({1,2,2,2,2},dx),T({1,2,4,4,4},dout),&ws,&ex)); run(aclnnUpsampleNearest3d,s); }
    else if(op=="uptrilinear3d"){ auto x=rf(pre+".x",16); void*dx=upf(x); dout=mal(128*4); on=128; CK(aclnnUpsampleTrilinear3dGetWorkspaceSize(T({1,2,2,2,2},dx),false,T({1,2,4,4,4},dout),&ws,&ex)); run(aclnnUpsampleTrilinear3d,s); }
    else if(op=="upbilinear2d"){ auto x=rf(pre+".x",8); void*dx=upf(x); dout=mal(32*4); on=32; CK(aclnnUpsampleBilinear2dGetWorkspaceSize(T({1,2,2,2},dx),false,T({1,2,4,4},dout),&ws,&ex)); run(aclnnUpsampleBilinear2d,s); }
    else if(op=="upbicubic2d"){ auto x=rf(pre+".x",18); void*dx=upf(x); dout=mal(72*4); on=72; CK(aclnnUpsampleBicubic2dGetWorkspaceSize(T({1,2,3,3},dx),false,T({1,2,6,6},dout),&ws,&ex)); run(aclnnUpsampleBicubic2d,s); }
    else if(op=="upnearest1d"){ auto x=rf(pre+".x",8); void*dx=upf(x); dout=mal(16*4); on=16; CK(aclnnUpsampleNearest1dGetWorkspaceSize(T({1,2,4},dx),T({1,2,8},dout),&ws,&ex)); run(aclnnUpsampleNearest1d,s); }
    else if(op=="uplinear1d"){ auto x=rf(pre+".x",8); void*dx=upf(x); dout=mal(16*4); on=16; CK(aclnnUpsampleLinear1dGetWorkspaceSize(T({1,2,4},dx),false,T({1,2,8},dout),&ws,&ex)); run(aclnnUpsampleLinear1d,s); }
    else if(op=="affinegrid"){ auto th=rf(pre+".x",6); void*dt=upf(th); dout=mal(1*3*3*2*4); on=18; CK(aclnnAffineGridGetWorkspaceSize(T({1,2,3},dt),3,3,false,T({1,3,3,2},dout),&ws,&ex)); run(aclnnAffineGrid,s); }
    else if(op=="lrn"){ auto x=rf(pre+".x",16); void*dx=upf(x); dout=mal(16*4); on=16; CK(aclnnLocalResponseNormGetWorkspaceSize(T({1,4,2,2},dx),3,1e-4,0.75,1.0,T({1,4,2,2},dout),&ws,&ex)); run(aclnnLocalResponseNorm,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
