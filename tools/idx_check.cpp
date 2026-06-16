// Independent forward check for indexing/shape extras vs PyTorch (torch_idx.py).
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
static void wi(const std::string&p,const std::vector<int64_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),8,v.size(),f); fclose(f); }
static void wi32(const std::string&p,const std::vector<int32_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<int64_t> ri(const std::string&p,int64_t k){ std::vector<int64_t> v(k); FILE*f=fopen(p.c_str(),"rb"); if(fread(v.data(),8,k,f)!=(size_t)k){} fclose(f); return v; }
static std::vector<int32_t> ri32(const std::string&p,int64_t k){ std::vector<int32_t> v(k); FILE*f=fopen(p.c_str(),"rb"); if(fread(v.data(),4,k,f)!=(size_t)k){} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi32(const std::vector<int32_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<float> dni(void*d,int64_t k){ std::vector<int64_t> t(k); CK(aclrtMemcpy(t.data(),k*8,d,k*8,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)t[i]; return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static aclIntArray* IA(std::vector<int64_t> v){ return aclCreateIntArray(v.data(),(int64_t)v.size()); }
static std::vector<float> g(int64_t n){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*0.31+0.2); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="take"){ wf(pre+".a",g(12)); wi(pre+".idx",{0,5,11,3,7}); }
        else if(op=="takealongdim"){ wf(pre+".a",g(12)); std::vector<int64_t> idx(12); for(int64_t i=0;i<12;i++)idx[i]=(i*3+1)%4; wi(pre+".idx",idx); }
        else if(op=="indexselect"){ wf(pre+".a",g(20)); wi(pre+".idx",{0,2,3}); }
        else if(op=="bincount"){ std::vector<int32_t> a={0,1,2,1,4,3,2,1}; wi32(pre+".ai",a); }
        else if(op=="histc"){ wf(pre+".a",g(16)); }
        else if(op=="narrow"||op=="rot90"||op=="flatten"||op=="repeatinterleave"){ wf(pre+".a",g(op=="narrow"?24:(op=="repeatinterleave"?12:12))); }
        else if(op=="diagonal"){ wf(pre+".a",g(16)); }
        else if(op=="tile"||op=="repeat"){ wf(pre+".a",g(6)); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    int kind=0; int64_t on=0; void*dout=nullptr;
    if(op=="take"){ auto a=rf(pre+".a",12); auto idx=ri(pre+".idx",5); void*da=upf(a),*di=upi(idx); dout=mal(5*4); on=5; CK(aclnnTakeGetWorkspaceSize(T({3,4},da),T({5},di,ACL_INT64),T({5},dout),&ws,&ex)); run(aclnnTake,s); }
    else if(op=="takealongdim"){ auto a=rf(pre+".a",12); auto idx=ri(pre+".idx",12); void*da=upf(a),*di=upi(idx); dout=mal(12*4); on=12; CK(aclnnTakeAlongDimGetWorkspaceSize(T({3,4},da),T({3,4},di,ACL_INT64),1,T({3,4},dout),&ws,&ex)); run(aclnnTakeAlongDim,s); }
    else if(op=="indexselect"){ auto a=rf(pre+".a",20); auto idx=ri(pre+".idx",3); void*da=upf(a),*di=upi(idx); dout=mal(3*5*4); on=15; CK(aclnnIndexSelectGetWorkspaceSize(T({4,5},da),0,T({3},di,ACL_INT64),T({3,5},dout),&ws,&ex)); run(aclnnIndexSelect,s); }
    else if(op=="bincount"){ auto a=ri32(pre+".ai",8); void*da=upi32(a); dout=mal(5*8); kind=1; on=5; CK(aclnnBincountGetWorkspaceSize(T({8},da,ACL_INT32),5,T({5},dout,ACL_INT64),&ws,&ex)); run(aclnnBincount,s); }
    else if(op=="histc"){ auto a=rf(pre+".a",16); void*da=upf(a); double mn=-1,mx=1; dout=mal(4*4); on=4; CK(aclnnHistcGetWorkspaceSize(T({16},da),4,aclCreateScalar(&mn,ACL_DOUBLE),aclCreateScalar(&mx,ACL_DOUBLE),T({4},dout),&ws,&ex)); run(aclnnHistc,s); }
    else if(op=="narrow"){ auto a=rf(pre+".a",24); void*da=upf(a); dout=mal(4*3*4); on=12; CK(aclnnNarrowGetWorkspaceSize(T({4,6},da),1,1,3,T({4,3},dout),&ws,&ex)); run(aclnnNarrow,s); }
    else if(op=="rot90"){ auto a=rf(pre+".a",12); void*da=upf(a); dout=mal(12*4); on=12; CK(aclnnRot90GetWorkspaceSize(T({3,4},da),1,T({4,3},dout),&ws,&ex)); run(aclnnRot90,s); }
    else if(op=="flatten"){ auto a=rf(pre+".a",12); void*da=upf(a); dout=mal(12*4); on=12; CK(aclnnFlattenGetWorkspaceSize(T({3,4},da),0,1,T({12},dout),&ws,&ex)); run(aclnnFlatten,s); }
    else if(op=="diagonal"){ auto a=rf(pre+".a",16); void*da=upf(a); dout=mal(4*4); on=4; CK(aclnnDiagonalGetWorkspaceSize(T({4,4},da),0,T({4},dout),&ws,&ex)); run(aclnnDiagonal,s); }
    else if(op=="tile"){ auto a=rf(pre+".a",6); void*da=upf(a); dout=mal(24*4); on=24; CK(aclnnTileGetWorkspaceSize(T({2,3},da),IA({2,2}),T({4,6},dout),&ws,&ex)); run(aclnnTile,s); }
    else if(op=="repeat"){ auto a=rf(pre+".a",6); void*da=upf(a); dout=mal(24*4); on=24; CK(aclnnRepeatGetWorkspaceSize(T({2,3},da),IA({2,2}),T({4,6},dout),&ws,&ex)); run(aclnnRepeat,s); }
    else if(op=="repeatinterleave"){ auto a=rf(pre+".a",12); void*da=upf(a); dout=mal(24*4); on=24; CK(aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(T({3,4},da),2,1,8,T({3,8},dout),&ws,&ex)); run(aclnnRepeatInterleaveIntWithDim,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(kind==1?dni(dout,on):dn(dout,on),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
