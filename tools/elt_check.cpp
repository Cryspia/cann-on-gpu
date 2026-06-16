// Independent forward check for elementwise long-tail + comparison/bitwise vs PyTorch (torch_elt.py).
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t N=32;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static void wi(const std::string&p,const std::vector<int32_t>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<int32_t> rfi(const std::string&p,int64_t k){ std::vector<int32_t> v(k); FILE*f=fopen(p.c_str(),"rb"); if(fread(v.data(),4,k,f)!=(size_t)k){} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* upi(const std::vector<int32_t>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static std::vector<float> g(int64_t n,double f,double p){ std::vector<float> v(n); for(int64_t i=0;i<n;i++)v[i]=(float)std::sin(i*f+p); return v; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }
// kind: 0 float,1 int32,2 bool
static std::vector<float> readout(void*d,int64_t n,int kind){ std::vector<float> z(n);
    if(kind==0){ CK(aclrtMemcpy(z.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); }
    else if(kind==1){ std::vector<int32_t> t(n); CK(aclrtMemcpy(t.data(),n*4,d,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); for(int64_t i=0;i<n;i++)z[i]=(float)t[i]; }
    else if(kind==2){ std::vector<uint8_t> t(n); CK(aclrtMemcpy(t.data(),n,d,n,ACL_MEMCPY_DEVICE_TO_HOST)); for(int64_t i=0;i<n;i++)z[i]=(float)t[i]; }
    else { std::vector<int64_t> t(n); CK(aclrtMemcpy(t.data(),n*8,d,n*8,ACL_MEMCPY_DEVICE_TO_HOST)); for(int64_t i=0;i<n;i++)z[i]=(float)t[i]; }
    return z; }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="gcd"){ std::vector<int32_t> a(N),b(N); for(int64_t i=0;i<N;i++){a[i]=(int32_t)((i%7+1)*3);b[i]=(int32_t)((i%5+1)*4);} wi(pre+".ai",a); wi(pre+".bi",b); }
        else if(op=="logit"){ std::vector<float> a(N); for(int64_t i=0;i<N;i++)a[i]=(float)(0.5+0.4*std::sin(i*0.3)); wf(pre+".a",a); }
        else if(op=="nantonum"){ auto a=g(N,0.3,0.1); a[2]=NAN; a[5]=INFINITY; a[9]=-INFINITY; wf(pre+".a",a); }
        else if(op=="softsign"||op=="rounddecimals"||op=="signbit"||op=="logicalnot"){ wf(pre+".a",g(N,0.37,0.2)); }
        else if(op=="equal"||op=="isclose"){ auto a=g(N,0.37,0.2); auto b=a; b[3]+=1.0f; b[7]+=1e-5f; wf(pre+".a",a); wf(pre+".b",b); }
        else if(op=="logicalxor"){ std::vector<uint8_t> a(N),b(N); for(int64_t i=0;i<N;i++){ a[i]=(i%3==0)?0:1; b[i]=(i%4==0)?0:1; } FILE*fa=fopen((pre+".a").c_str(),"wb");fwrite(a.data(),1,N,fa);fclose(fa); FILE*fb=fopen((pre+".b").c_str(),"wb");fwrite(b.data(),1,N,fb);fclose(fb); }
        else if(op=="bucketize"){ wf(pre+".a",g(8,0.5,0.1)); std::vector<float> bnd={-0.8f,-0.3f,0.1f,0.5f,0.9f}; wf(pre+".bnd",bnd); }
        else if(op=="searchsorted"){ wf(pre+".a",g(8,0.5,0.1)); std::vector<float> srt={-0.8f,-0.3f,0.1f,0.5f,0.9f}; wf(pre+".srt",srt); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr;
    int kind=0; int64_t on=N; void*dout=nullptr;
    if(op=="gcd"){ auto a=rfi(pre+".ai",N),b=rfi(pre+".bi",N); void*da=upi(a),*db=upi(b); dout=mal(N*4); kind=1; CK(aclnnGcdGetWorkspaceSize(T({N},da,ACL_INT32),T({N},db,ACL_INT32),T({N},dout,ACL_INT32),&ws,&ex)); run(aclnnGcd,s); }
    else if(op=="logit"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N*4); CK(aclnnLogitGetWorkspaceSize(T({N},da),1e-6,T({N},dout),&ws,&ex)); run(aclnnLogit,s); }
    else if(op=="nantonum"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N*4); CK(aclnnNanToNumGetWorkspaceSize(T({N},da),0.f,1e4f,-1e4f,T({N},dout),&ws,&ex)); run(aclnnNanToNum,s); }
    else if(op=="softsign"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N*4); CK(aclnnSoftsignGetWorkspaceSize(T({N},da),T({N},dout),&ws,&ex)); run(aclnnSoftsign,s); }
    else if(op=="rounddecimals"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N*4); CK(aclnnRoundDecimalsGetWorkspaceSize(T({N},da),2,T({N},dout),&ws,&ex)); run(aclnnRoundDecimals,s); }
    else if(op=="signbit"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N); kind=2; CK(aclnnSignbitGetWorkspaceSize(T({N},da),T({N},dout,ACL_BOOL),&ws,&ex)); run(aclnnSignbit,s); }
    else if(op=="equal"){ auto a=rf(pre+".a",N),b=rf(pre+".b",N); void*da=upf(a),*db=upf(b); dout=mal(N); kind=2; CK(aclnnEqualGetWorkspaceSize(T({N},da),T({N},db),T({N},dout,ACL_BOOL),&ws,&ex)); run(aclnnEqual,s); }
    else if(op=="isclose"){ auto a=rf(pre+".a",N),b=rf(pre+".b",N); void*da=upf(a),*db=upf(b); dout=mal(N); kind=2; CK(aclnnIsCloseGetWorkspaceSize(T({N},da),T({N},db),1e-3,1e-3,false,T({N},dout,ACL_BOOL),&ws,&ex)); run(aclnnIsClose,s); }
    else if(op=="logicalnot"){ auto a=rf(pre+".a",N); void*da=upf(a); dout=mal(N); kind=2; CK(aclnnLogicalNotGetWorkspaceSize(T({N},da),T({N},dout,ACL_BOOL),&ws,&ex)); run(aclnnLogicalNot,s); }
    else if(op=="logicalxor"){ std::vector<uint8_t> a(N),b(N); {FILE*f=fopen((pre+".a").c_str(),"rb");if(fread(a.data(),1,N,f)!=(size_t)N){}fclose(f);} {FILE*f=fopen((pre+".b").c_str(),"rb");if(fread(b.data(),1,N,f)!=(size_t)N){}fclose(f);}
        void*da=mal(N),*db=mal(N); CK(aclrtMemcpy(da,N,a.data(),N,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(db,N,b.data(),N,ACL_MEMCPY_HOST_TO_DEVICE)); dout=mal(N); kind=2;
        CK(aclnnLogicalXorGetWorkspaceSize(T({N},da,ACL_BOOL),T({N},db,ACL_BOOL),T({N},dout,ACL_BOOL),&ws,&ex)); run(aclnnLogicalXor,s); }
    else if(op=="bucketize"){ auto a=rf(pre+".a",8),bnd=rf(pre+".bnd",5); void*da=upf(a),*dbnd=upf(bnd); dout=mal(8*8); kind=3; on=8; CK(aclnnBucketizeGetWorkspaceSize(T({8},da),T({5},dbnd),false,T({8},dout,ACL_INT64),&ws,&ex)); run(aclnnBucketize,s); }
    else if(op=="searchsorted"){ auto a=rf(pre+".a",8),srt=rf(pre+".srt",5); void*da=upf(a),*dsrt=upf(srt); dout=mal(8*8); kind=3; on=8; CK(aclnnSearchSortedGetWorkspaceSize(T({5},dsrt),T({8},da),false,T({8},dout,ACL_INT64),&ws,&ex)); run(aclnnSearchSorted,s); }
    else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
    double e=cmp(readout(dout,on,kind),rf(pre+".out",on)); double tol=2e-4;
    printf("[%s] out normalized_err=%.3e (tol=%.0e)  %s\n",op.c_str(),e,tol,e<=tol?"PASS":"FAIL");
    return e<=tol?0:1;
}
