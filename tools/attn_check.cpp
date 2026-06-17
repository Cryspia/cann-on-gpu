// Independent check for FlashAttentionScore vs PyTorch SDPA (torch_attn.py). BNSD layout, GQA, causal,
// optional bool mask (nonzero=masked-out). Covers flash path (small) and the fp32 cuBLASLt perf path (perf).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
struct Cfg{ int64_t B,Nq,Nkv,Sq,Skv,D; int causal,hasmask; };
static Cfg cfg(const std::string&c){
    if(c=="plain") return {2,2,2,4,4,8,0,0};
    if(c=="causal")return {2,2,2,4,4,8,1,0};
    if(c=="gqa")   return {2,4,2,4,4,8,0,0};
    if(c=="mask")  return {2,2,2,4,4,8,0,1};
    return {1,2,2,16,16,16,0,0}; // perf
}
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wb(const std::string&p,const std::vector<uint8_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),1,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<uint8_t> ru(const std::string&p,int64_t k){ std::vector<uint8_t> v(k); FILE*f=fopen(p.c_str(),"rb"); if((int64_t)fread(v.data(),1,k,f)!=k){fprintf(stderr,"short mask\n");exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upu(const std::vector<uint8_t>&v){ void*d; CK(aclrtMalloc(&d,v.size(),ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size(),v.data(),v.size(),ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }

int main(int argc,char**argv){
    std::string mode=argv[1],cs=argv[2],pre=argv[3]; Cfg c=cfg(cs);
    int64_t QN=c.B*c.Nq*c.Sq*c.D, KN=c.B*c.Nkv*c.Skv*c.D;
    if(mode=="gen"){
        std::vector<float> q(QN),k(KN),v(KN);
        for(int64_t i=0;i<QN;i++)q[i]=(float)(0.5*std::sin(i*0.05+0.1));
        for(int64_t i=0;i<KN;i++){ k[i]=(float)(0.5*std::cos(i*0.04+0.2)); v[i]=(float)(0.4*std::sin(i*0.03+0.5)); }
        wf(pre+".q",q); wf(pre+".k",k); wf(pre+".v",v);
        if(c.hasmask){ std::vector<uint8_t> m(c.Sq*c.Skv,0); for(int64_t i=0;i<c.Sq;i++)for(int64_t j=0;j<c.Skv;j++) m[i*c.Skv+j]=(j>i+1)?1:0; wb(pre+".mask",m); }
        printf("[gen] %s\n",cs.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    auto q=rf(pre+".q",QN),k=rf(pre+".k",KN),v=rf(pre+".v",KN);
    void*dq=upf(q),*dk=upf(k),*dv=upf(v),*dout=mal(QN*4);
    aclTensor*tq=T({c.B,c.Nq,c.Sq,c.D},dq),*tk=T({c.B,c.Nkv,c.Skv,c.D},dk),*tv=T({c.B,c.Nkv,c.Skv,c.D},dv),*to=T({c.B,c.Nq,c.Sq,c.D},dout);
    aclTensor* tm=nullptr; if(c.hasmask){ auto m=ru(pre+".mask",c.Sq*c.Skv); void*dm=upu(m); tm=T({c.Sq,c.Skv},dm,ACL_UINT8); }
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr;
    CK(aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,tm,0.0,c.Nq,c.causal!=0,to,&ws,&ex)); if(ws)wsp=mal(ws);
    CK(aclnnFlashAttentionScore(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
    double e=cmp(dn(dout,QN),rf(pre+".out",QN)); double tol=1e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",cs.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
