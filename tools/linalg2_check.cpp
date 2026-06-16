// Independent check for linalg extras / distance / complex vs PyTorch (torch_linalg2.py).
// eigh: A@V==V@diag(W) + W vs eigvalsh. lulusolve/lstsq: A@X==B reconstruction. others: direct compare.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t n=4;
static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p,aclDataType dt=ACL_FLOAT){ return aclCreateTensor(d.data(),(int64_t)d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b<4?4:b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static std::vector<float> mm(const std::vector<float>&A,const std::vector<float>&B,int M,int K,int N){ std::vector<float> C(M*N,0); for(int i=0;i<M;i++)for(int p=0;p<K;p++){float a=A[i*K+p];for(int j=0;j<N;j++)C[i*N+j]+=a*B[p*N+j];} return C; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static bool rep(const char*op,const char*w,double e,double tol){ printf("[%s] %-7s err=%.3e (tol=%.0e)  %s\n",op,w,e,tol,e<=tol?"PASS":"FAIL"); return e<=tol; }
static std::vector<float> g(int64_t k,double f,double p){ std::vector<float> v(k); for(int64_t i=0;i<k;i++)v[i]=(float)std::sin(i*f+p); return v; }
static std::vector<float> spd(){ std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++){ float v=0.3f*std::cos((std::min(i,j)*7+std::max(i,j))*0.5f); A[i*n+j]=(i==j)?(float)(n+1):v;} return A; }
static uint64_t ws; static aclOpExecutor* ex; static void* wsp;
static void run(aclnnStatus(*r)(void*,uint64_t,aclOpExecutor*,aclrtStream),aclrtStream s){ if(ws)wsp=mal(ws); CK(r(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s)); }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="eigh"||op=="logdet"){ wf(pre+".A",spd()); }
        else if(op=="lulusolve"||op=="lstsq"){ std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++)A[i*n+j]=(i==j)?(float)(n+1):(float)(0.3*std::sin((i*n+j)*0.4)); wf(pre+".A",A); wf(pre+".B",g(n*n,0.55,0.7)); }
        else if(op=="matrixexp"){ wf(pre+".A",g(n*n,0.3,0.1)); }
        else if(op=="pinverse"){ std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++)A[i*n+j]=(i==j)?(float)(n+1):(float)(0.3*std::sin((i*n+j)*0.4)); wf(pre+".A",A); }
        else if(op=="cdist"){ wf(pre+".a",g(3*4,0.3,0.1)); wf(pre+".b",g(5*4,0.25,0.5)); }
        else if(op=="pdist"){ wf(pre+".a",g(4*3,0.3,0.1)); }
        else if(op=="complex"||op=="polar"){ wf(pre+".a",g(5,0.3,0.1)); wf(pre+".b",g(5,0.25,0.5)); }
        else if(op=="real"){ wf(pre+".a",g(5*2,0.3,0.1)); }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s)); ex=nullptr; ws=0; wsp=nullptr; bool ok=true;
    if(op=="eigh"){ auto A=rf(pre+".A",n*n); void*dA=upf(A),*dW=mal(n*4),*dV=mal(n*n*4); CK(aclnnEighGetWorkspaceSize(T({n,n},dA),T({n},dW),T({n,n},dV),&ws,&ex)); run(aclnnEigh,s);
        auto W=dn(dW,n),V=dn(dV,n*n); std::vector<float> AV=mm(A,V,n,n,n), VD(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++)VD[i*n+j]=V[i*n+j]*W[j];
        ok&=rep("eigh","A@V=VD",cmp(AV,VD),2e-3); ok&=rep("eigh","W",cmp(W,rf(pre+".W",n)),2e-3); }
    else if(op=="lulusolve"){ auto A=rf(pre+".A",n*n),B=rf(pre+".B",n*n); void*dA=upf(A),*dB=upf(B),*dLU=mal(n*n*4),*dpiv=mal(n*4),*dX=mal(n*n*4);
        CK(aclnnLuGetWorkspaceSize(T({n,n},dA),T({n,n},dLU),T({n},dpiv,ACL_INT32),&ws,&ex)); run(aclnnLu,s); ws=0;wsp=nullptr;
        CK(aclnnLuSolveGetWorkspaceSize(T({n,n},dLU),T({n},dpiv,ACL_INT32),T({n,n},dB),T({n,n},dX),&ws,&ex)); run(aclnnLuSolve,s);
        auto X=dn(dX,n*n); ok&=rep("lulusolve","A@X=B",cmp(mm(A,X,n,n,n),B),2e-3); }
    else if(op=="lstsq"){ auto A=rf(pre+".A",n*n),B=rf(pre+".B",n*n); void*dA=upf(A),*dB=upf(B),*dX=mal(n*n*4);
        CK(aclnnLstsqGetWorkspaceSize(T({n,n},dA),T({n,n},dB),T({n,n},dX),&ws,&ex)); run(aclnnLstsq,s);
        auto X=dn(dX,n*n); ok&=rep("lstsq","A@X=B",cmp(mm(A,X,n,n,n),B),2e-3); }
    else { // direct-compare
        void*dout=nullptr; int64_t on=0;
        if(op=="matrixexp"||op=="pinverse"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(n*n*4); on=n*n;
            if(op=="matrixexp"){ CK(aclnnMatrixExpGetWorkspaceSize(T({n,n},dA),T({n,n},dout),&ws,&ex)); run(aclnnMatrixExp,s);} else { CK(aclnnPinverseGetWorkspaceSize(T({n,n},dA),T({n,n},dout),&ws,&ex)); run(aclnnPinverse,s);} }
        else if(op=="logdet"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(4); on=1; CK(aclnnLogdetGetWorkspaceSize(T({n,n},dA),T({1},dout),&ws,&ex)); run(aclnnLogdet,s); }
        else if(op=="cdist"){ auto a=rf(pre+".a",12),b=rf(pre+".b",20); void*da=upf(a),*db=upf(b); dout=mal(3*5*4); on=15; CK(aclnnCdistGetWorkspaceSize(T({3,4},da),T({5,4},db),2.0,T({3,5},dout),&ws,&ex)); run(aclnnCdist,s); }
        else if(op=="pdist"){ auto a=rf(pre+".a",12); void*da=upf(a); dout=mal(6*4); on=6; CK(aclnnPdistGetWorkspaceSize(T({4,3},da),2.0,T({6},dout),&ws,&ex)); run(aclnnPdist,s); }
        else if(op=="complex"){ auto a=rf(pre+".a",5),b=rf(pre+".b",5); void*da=upf(a),*db=upf(b); dout=mal(10*4); on=10; CK(aclnnComplexGetWorkspaceSize(T({5},da),T({5},db),T({5,2},dout),&ws,&ex)); run(aclnnComplex,s); }
        else if(op=="polar"){ auto a=rf(pre+".a",5),b=rf(pre+".b",5); for(auto&x:a)x=std::fabs(x); void*da=upf(a),*db=upf(b); dout=mal(10*4); on=10; CK(aclnnPolarGetWorkspaceSize(T({5},da),T({5},db),T({5,2},dout),&ws,&ex)); run(aclnnPolar,s); }
        else if(op=="real"){ auto a=rf(pre+".a",10); void*da=upf(a); dout=mal(5*4); on=5; CK(aclnnRealGetWorkspaceSize(T({5,2},da),T({5},dout),&ws,&ex)); run(aclnnReal,s); }
        else { fprintf(stderr,"unknown %s\n",op.c_str()); return 2; }
        ok&=rep(op.c_str(),"out",cmp(dn(dout,on),rf(pre+".out",on)),2e-3);
    }
    return ok?0:1;
}
