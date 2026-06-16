// Independent check for tensor-list/dynamic-shape ops vs PyTorch (torch_dyn.py).
// scatternd/scatterndupdate: scatter at nd-indices. unique/nonzero: dynamic count — compare countOut + values.
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
static int64_t rc(const std::string&p){ int64_t x=0; FILE*f=fopen(p.c_str(),"rb"); if(f){if(fread(&x,8,1,f)!=1)x=0; fclose(f);} return x; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int64_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*8,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*8,v.data(),v.size()*8,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemset(d,b<4?4:b,0,b<4?4:b)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<float> dni(void*d,int64_t k){ std::vector<int64_t> t(k); CK(aclrtMemcpy(t.data(),k*8,d,k*8,ACL_MEMCPY_DEVICE_TO_HOST)); std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)t[i]; return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static std::vector<float> g(int64_t n){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*0.4+0.2); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="scatternd"){ wf(pre+".u",g(6)); }
        else if(op=="scatterndupdate"){ wf(pre+".s",g(12)); wf(pre+".u",g(6)); }
        else if(op=="unique"){ std::vector<float> x={1,2,2,3,1,4,3,2}; wf(pre+".x",x); }
        else if(op=="nonzero"){ std::vector<float> x={0,1.5f,0,2.5f,0,0,3.5f,4.5f}; wf(pre+".x",x); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    if(op=="scatternd"){ auto u=rf(pre+".u",6); std::vector<int64_t> idx={0,2}; void*du=upf(u),*di=upi(idx),*dout=mal(12*4);
        CK(aclnnScatterNdGetWorkspaceSize(T({2,1},di,ACL_INT64),T({2,3},du),T({4,3},dout),&ws,&ex)); run(aclnnScatterNd,s);
        double e=cmp(dn(dout,12),rf(pre+".out",12)); printf("[scatternd] out=%.3e %s\n",e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    if(op=="scatterndupdate"){ auto sf=rf(pre+".s",12),u=rf(pre+".u",6); std::vector<int64_t> idx={0,2}; void*ds=upf(sf),*du=upf(u),*di=upi(idx),*dout=mal(12*4);
        CK(aclnnScatterNdUpdateGetWorkspaceSize(T({4,3},ds),T({2,1},di,ACL_INT64),T({2,3},du),T({4,3},dout),&ws,&ex)); run(aclnnScatterNdUpdate,s);
        double e=cmp(dn(dout,12),rf(pre+".out",12)); printf("[scatterndupdate] out=%.3e %s\n",e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    if(op=="unique"){ auto x=rf(pre+".x",8); void*dx=upf(x),*dv=mal(8*4),*dcnt=mal(8),*dinv=mal(8*8),*dcc=mal(8*8);
        CK(aclnnUniqueGetWorkspaceSize(T({8},dx),T({8},dv),T({1},dcnt,ACL_INT64),T({8},dinv,ACL_INT64),T({8},dcc,ACL_INT64),&ws,&ex)); run(aclnnUnique,s);
        int64_t cnt=0; CK(aclrtMemcpy(&cnt,8,dcnt,8,ACL_MEMCPY_DEVICE_TO_HOST)); int64_t rcnt=rc(pre+".cnt");
        if(cnt!=rcnt){ printf("[unique] count %ld vs %ld FAIL\n",(long)cnt,(long)rcnt); return 1; }
        double e=cmp(dn(dv,cnt),rf(pre+".out",cnt)); printf("[unique] count=%ld vals=%.3e %s\n",(long)cnt,e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    if(op=="nonzero"){ auto x=rf(pre+".x",8); void*dx=upf(x),*dout=mal(8*8),*dcnt=mal(8);
        CK(aclnnNonzeroV2GetWorkspaceSize(T({8},dx),T({8},dout,ACL_INT64),T({1},dcnt,ACL_INT64),&ws,&ex)); run(aclnnNonzeroV2,s);
        int64_t cnt=0; CK(aclrtMemcpy(&cnt,8,dcnt,8,ACL_MEMCPY_DEVICE_TO_HOST)); int64_t rcnt=rc(pre+".cnt");
        if(cnt!=rcnt){ printf("[nonzero] count %ld vs %ld FAIL\n",(long)cnt,(long)rcnt); return 1; }
        double e=cmp(dni(dout,cnt),rf(pre+".out",cnt)); printf("[nonzero] count=%ld idx=%.3e %s\n",(long)cnt,e,e<=2e-4?"PASS":"FAIL"); return e<=2e-4?0:1; }
    fprintf(stderr,"unknown %s\n",op.c_str()); return 2;
}
