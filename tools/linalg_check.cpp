// Independent check for linalg ops vs PyTorch (torch_linalg.py). Unique outputs are compared directly
// to torch.linalg; qr/svd are non-unique so they are validated by reconstruction (Q@R==A, U@diag(S)@VT==A,
// plus svd singular values vs torch). SPD/diag-dominant matrices are generated for the solve/factor ops.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t n=4, c3=3, m3=3;

static void wf(const std::string&p,const std::vector<float>&v){ FILE*f=fopen(p.c_str(),"wb"); fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rf(const std::string&p,int64_t k){ std::vector<float> v(k); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);} if((int64_t)fread(v.data(),4,k,f)!=k){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dn(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double cmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static std::vector<float> mm(const std::vector<float>&A,const std::vector<float>&B,int M,int K,int N){ std::vector<float> C(M*N,0); for(int i=0;i<M;i++)for(int p=0;p<K;p++){float a=A[i*K+p];for(int j=0;j<N;j++)C[i*N+j]+=a*B[p*N+j];} return C; }
static bool rep(const char*op,const char*w,double e,double tol){ printf("[%s] %-6s normalized_err=%.3e (tol=%.0e)  %s\n",op,w,e,tol,e<=tol?"PASS":"FAIL"); return e<=tol; }

int main(int argc,char**argv){
    std::string mode=argv[1],op=argv[2],pre=argv[3];
    if(mode=="gen"){
        if(op=="cross"){ std::vector<float> u(c3),v(c3); for(int i=0;i<c3;i++){u[i]=(float)(1.0+std::sin(i*0.7));v[i]=(float)(0.5+std::cos(i*0.9));} wf(pre+".u",u);wf(pre+".v",v); }
        else if(op=="dot"){ std::vector<float> u(n),v(n); for(int i=0;i<n;i++){u[i]=(float)(1.0+std::sin(i*0.7));v[i]=(float)(0.5+std::cos(i*0.9));} wf(pre+".u",u);wf(pre+".v",v); }
        else if(op=="outer"){ std::vector<float> u(n),v(m3); for(int i=0;i<n;i++)u[i]=(float)(1.0+std::sin(i*0.7)); for(int i=0;i<m3;i++)v[i]=(float)(0.5+std::cos(i*0.9)); wf(pre+".u",u);wf(pre+".v",v); }
        else { // matrix ops
            std::vector<float> A(n*n);
            bool spd = (op=="inverse"||op=="det"||op=="slogdet"||op=="cholesky"||op=="solve");
            bool tri = (op=="triangularsolve");
            for(int i=0;i<n;i++)for(int j=0;j<n;j++){
                if(spd){ float v=0.3f*std::cos((std::min(i,j)*7+std::max(i,j))*0.5f); A[i*n+j]=(i==j)?(float)(n+1):v; } // symmetric diag-dominant SPD
                else if(tri){ A[i*n+j]=(i==j)?(float)(n):(float)(0.4*std::sin((i*n+j)*0.4)); }                          // well-conditioned (upper used)
                else { A[i*n+j]=(float)(1.0+0.5*std::sin((i*n+j)*0.4)); }                                              // general
            }
            wf(pre+".A",A);
            if(op=="solve"||op=="triangularsolve"){ std::vector<float> B(n*n); for(int i=0;i<n*n;i++)B[i]=(float)(0.7+0.6*std::cos(i*0.55)); wf(pre+".B",B); }
        }
        printf("[gen] %s\n",op.c_str()); return 0;
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    uint64_t ws=0; aclOpExecutor*ex=nullptr; void*wsp=nullptr; auto WS=[&](int rc){ CK(rc); if(ws)wsp=mal(ws); };
    bool ok=true; double tol=2e-3;
    if(op=="qr"){
        auto A=rf(pre+".A",n*n); void*dA=upf(A),*dQ=mal(n*n*4),*dR=mal(n*n*4);
        aclTensor*tA=T({n,n},dA),*tQ=T({n,n},dQ),*tR=T({n,n},dR);
        WS(aclnnQrGetWorkspaceSize(tA,tQ,tR,&ws,&ex)); CK(aclnnQr(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        auto recon=mm(dn(dQ,n*n),dn(dR,n*n),n,n,n); ok&=rep("qr","Q@R=A",cmp(recon,A),tol);
    } else if(op=="svd"){
        auto A=rf(pre+".A",n*n); void*dA=upf(A),*dU=mal(n*n*4),*dS=mal(n*4),*dV=mal(n*n*4);
        aclTensor*tA=T({n,n},dA),*tU=T({n,n},dU),*tS=T({n},dS),*tV=T({n,n},dV);
        WS(aclnnSvdGetWorkspaceSize(tA,tU,tS,tV,&ws,&ex)); CK(aclnnSvd(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        auto U=dn(dU,n*n),S=dn(dS,n),VT=dn(dV,n*n);
        std::vector<float> US(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++)US[i*n+j]=U[i*n+j]*S[j];
        ok&=rep("svd","USV=A",cmp(mm(US,VT,n,n,n),A),tol);
        ok&=rep("svd","S",cmp(S,rf(pre+".S",n)),tol);
    } else {
        // direct-compare ops
        void *dout=nullptr; int64_t on=0; const char* outf=".out";
        if(op=="inverse"||op=="matrixpower"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(n*n*4); on=n*n; aclTensor*tA=T({n,n},dA),*to=T({n,n},dout);
            if(op=="inverse"){ WS(aclnnInverseGetWorkspaceSize(tA,to,&ws,&ex)); CK(aclnnInverse(wsp,ws,ex,s)); } else { WS(aclnnMatrixPowerGetWorkspaceSize(tA,3,to,&ws,&ex)); CK(aclnnMatrixPower(wsp,ws,ex,s)); } }
        else if(op=="cholesky"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(n*n*4); on=n*n; aclTensor*tA=T({n,n},dA),*to=T({n,n},dout); WS(aclnnCholeskyGetWorkspaceSize(tA,to,&ws,&ex)); CK(aclnnCholesky(wsp,ws,ex,s)); }
        else if(op=="det"||op=="trace"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(4); on=1; aclTensor*tA=T({n,n},dA),*to=T({1},dout);
            if(op=="det"){ WS(aclnnDetGetWorkspaceSize(tA,to,&ws,&ex)); CK(aclnnDet(wsp,ws,ex,s)); } else { WS(aclnnTraceGetWorkspaceSize(tA,to,&ws,&ex)); CK(aclnnTrace(wsp,ws,ex,s)); } }
        else if(op=="diag"){ auto A=rf(pre+".A",n*n); void*dA=upf(A); dout=mal(n*4); on=n; aclTensor*tA=T({n,n},dA),*to=T({n},dout); WS(aclnnDiagGetWorkspaceSize(tA,0,to,&ws,&ex)); CK(aclnnDiag(wsp,ws,ex,s)); }
        else if(op=="slogdet"){ auto A=rf(pre+".A",n*n); void*dA=upf(A),*dsg=mal(4),*dla=mal(4); aclTensor*tA=T({n,n},dA),*tsg=T({1},dsg),*tla=T({1},dla);
            WS(aclnnSlogdetGetWorkspaceSize(tA,tsg,tla,&ws,&ex)); CK(aclnnSlogdet(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
            ok&=rep("slogdet","sign",cmp(dn(dsg,1),rf(pre+".sign",1)),tol); ok&=rep("slogdet","logabs",cmp(dn(dla,1),rf(pre+".logabs",1)),tol); return ok?0:1; }
        else if(op=="solve"||op=="triangularsolve"){ auto A=rf(pre+".A",n*n),B=rf(pre+".B",n*n); void*dA=upf(A),*dB=upf(B); dout=mal(n*n*4); on=n*n; aclTensor*tA=T({n,n},dA),*tB=T({n,n},dB),*to=T({n,n},dout);
            if(op=="solve"){ WS(aclnnSolveGetWorkspaceSize(tA,tB,to,&ws,&ex)); CK(aclnnSolve(wsp,ws,ex,s)); } else { WS(aclnnTriangularSolveGetWorkspaceSize(tA,tB,true,false,false,to,&ws,&ex)); CK(aclnnTriangularSolve(wsp,ws,ex,s)); } }
        else if(op=="cross"){ auto u=rf(pre+".u",c3),v=rf(pre+".v",c3); void*du=upf(u),*dv=upf(v); dout=mal(c3*4); on=c3; aclTensor*tu=T({c3},du),*tv=T({c3},dv),*to=T({c3},dout); WS(aclnnCrossGetWorkspaceSize(tu,tv,to,&ws,&ex)); CK(aclnnCross(wsp,ws,ex,s)); }
        else if(op=="dot"){ auto u=rf(pre+".u",n),v=rf(pre+".v",n); void*du=upf(u),*dv=upf(v); dout=mal(4); on=1; aclTensor*tu=T({n},du),*tv=T({n},dv),*to=T({1},dout); WS(aclnnDotGetWorkspaceSize(tu,tv,to,&ws,&ex)); CK(aclnnDot(wsp,ws,ex,s)); }
        else if(op=="outer"){ auto u=rf(pre+".u",n),v=rf(pre+".v",m3); void*du=upf(u),*dv=upf(v); dout=mal(n*m3*4); on=n*m3; aclTensor*tu=T({n},du),*tv=T({m3},dv),*to=T({n,m3},dout); WS(aclnnOuterGetWorkspaceSize(tu,tv,to,&ws,&ex)); CK(aclnnOuter(wsp,ws,ex,s)); }
        else { fprintf(stderr,"unknown op %s\n",op.c_str()); return 2; }
        CK(aclrtSynchronizeStream(s)); ok&=rep(op.c_str(),outf,cmp(dn(dout,on),rf(pre+outf,on)),tol);
    }
    return ok?0:1;
}
