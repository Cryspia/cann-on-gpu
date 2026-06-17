// Invariant test for the RoPE family. RoPE is an orthogonal per-pair rotation, so:
//   (a) round-trip: rope(rope(x, sin), -sin) == x   (inverse = negated sin, per the Grad op)
//   (b) norm preservation: ||rope(x)|| == ||x||      (rotation preserves L2 norm)
// cos/sin are built with c²+s²=1 and equal angle for the two partners of each rotated pair (per mode),
// so the rotation is well-formed. No torch reference needed.
#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"
#define CK(x) do{int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)
static const int64_t Rw=6, D=8;
static aclTensor* T(std::vector<int64_t> d,void*p){ return aclCreateTensor(d.data(),(int64_t)d.size(),ACL_FLOAT,nullptr,0,ACL_FORMAT_ND,d.data(),(int64_t)d.size(),p); }
static void* upf(const std::vector<float>&v){ void*d; CK(aclrtMalloc(&d,v.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(d,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }
static void* mal(int64_t b){ void*d; CK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static std::vector<float> dnf(void*d,int64_t k){ std::vector<float> v(k); CK(aclrtMemcpy(v.data(),k*4,d,k*4,ACL_MEMCPY_DEVICE_TO_HOST)); return v; }
static double ncmp(const std::vector<float>&z,const std::vector<float>&r){ double me=0,mr=0; for(size_t i=0;i<z.size();i++){me=std::max(me,(double)std::fabs(z[i]-r[i]));mr=std::max(mr,(double)std::fabs(r[i]));} return me/(mr+1e-9); }
static double l2(const std::vector<float>&v){ double s=0; for(float x:v)s+=(double)x*x; return std::sqrt(s); }

int main(int argc,char**argv){
    std::string op=argv[1];
    bool inter = (op=="interleave"||op=="rope1");           // interleaved pairing (2k,2k+1)
    int64_t mode = inter?1:0;
    int64_t N=Rw*D;
    std::vector<float> x(N),x2(N),cs(N),sn(N),snn(N);
    for(int64_t i=0;i<N;i++){ x[i]=(float)std::sin(i*0.17+0.1); x2[i]=(float)std::cos(i*0.13+0.4); }
    int64_t half=D/2;
    for(int64_t r=0;r<Rw;r++)for(int64_t d=0;d<D;d++){
        int64_t pidx = inter ? d/2 : d%half;               // pair index: partners share the same angle
        double th = 0.3 + 0.2*r + 0.5*pidx;
        cs[r*D+d]=(float)std::cos(th); sn[r*D+d]=(float)std::sin(th); snn[r*D+d]=-sn[r*D+d];
    }
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); aclrtStream s; CK(aclrtCreateStream(&s));
    void *dx=upf(x),*dcs=upf(cs),*dsn=upf(sn),*dsnn=upf(snn);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    auto rope=[&](void* din,void* dsin,void* dout){ ws=0; void*wp=nullptr;
        if(op=="ropecache"){ CK(aclnnRopeWithSinCosCacheGetWorkspaceSize(T({Rw,D},din),T({Rw,D},dcs),T({Rw,D},dsin),mode,T({Rw,D},dout),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnRopeWithSinCosCache(wp,ws,ex,s)); }
        else if(op=="interleave"){ CK(aclnnInterleaveRopeGetWorkspaceSize(T({Rw,D},din),T({Rw,D},dcs),T({Rw,D},dsin),T({Rw,D},dout),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnInterleaveRope(wp,ws,ex,s)); }
        else { CK(aclnnRotaryPositionEmbeddingGetWorkspaceSize(T({Rw,D},din),T({Rw,D},dcs),T({Rw,D},dsin),mode,T({Rw,D},dout),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnRotaryPositionEmbedding(wp,ws,ex,s)); }
    };
    bool ok=true; double tol=2e-5;
    if(op=="applyrope"){
        // ApplyRotaryPosEmb requires 4D q/k [B,Nh,S,D] and cos/sin [S,D] (shared across B,Nh).
        const int64_t B=2,Nh=2,Sq=3; int64_t QN=B*Nh*Sq*D;
        std::vector<float> q(QN),kk(QN),c2(Sq*D),s2(Sq*D),s2n(Sq*D);
        for(int64_t i=0;i<QN;i++){ q[i]=(float)std::sin(i*0.11+0.2); kk[i]=(float)std::cos(i*0.09+0.5); }
        for(int64_t si=0;si<Sq;si++)for(int64_t d=0;d<D;d++){ int64_t p=mode?d/2:d%half; double th=0.2+0.3*si+0.4*p; c2[si*D+d]=(float)std::cos(th); s2[si*D+d]=(float)std::sin(th); s2n[si*D+d]=-s2[si*D+d]; }
        void*dq=upf(q),*dk=upf(kk),*dc2=upf(c2),*ds2=upf(s2),*ds2n=upf(s2n);
        void*dqf=mal(QN*4),*dkf=mal(QN*4),*dqb=mal(QN*4),*dkb=mal(QN*4),*wp=nullptr;
        std::vector<int64_t> sh{B,Nh,Sq,D}, cd{Sq,D};
        CK(aclnnApplyRotaryPosEmbGetWorkspaceSize(T(sh,dq),T(sh,dk),T(cd,dc2),T(cd,ds2),mode,T(sh,dqf),T(sh,dkf),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnApplyRotaryPosEmb(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        ws=0;wp=nullptr; CK(aclnnApplyRotaryPosEmbGetWorkspaceSize(T(sh,dqf),T(sh,dkf),T(cd,dc2),T(cd,ds2n),mode,T(sh,dqb),T(sh,dkb),&ws,&ex)); if(ws)wp=mal(ws); CK(aclnnApplyRotaryPosEmb(wp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        double eq=ncmp(dnf(dqb,QN),q), ek=ncmp(dnf(dkb,QN),kk), nq=std::fabs(l2(dnf(dqf,QN))-l2(q))/l2(q);
        bool pk=(eq<=tol&&ek<=tol&&nq<=tol);
        printf("[applyrope] q_rt=%.2e k_rt=%.2e q_norm=%.2e (tol=%.0e)  %s\n",eq,ek,nq,tol,pk?"PASS":"FAIL");
        return pk?0:1;
    }
    void*dfwd=mal(N*4),*dback=mal(N*4);
    rope(dx,dsn,dfwd); CK(aclrtSynchronizeStream(s));
    rope(dfwd,dsnn,dback); CK(aclrtSynchronizeStream(s));
    auto fwd=dnf(dfwd,N); double rt=ncmp(dnf(dback,N),x); double nrm=std::fabs(l2(fwd)-l2(x))/l2(x);
    ok = rt<=tol && nrm<=tol;
    printf("[%s] roundtrip=%.3e norm_preserve=%.3e (tol=%.0e)  %s\n",op.c_str(),rt,nrm,tol,ok?"PASS":"FAIL");
    return ok?0:1;
}
