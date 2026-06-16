// mc2 MatmulAllReduce end-to-end over RoCE/HCCL (2 nodes). Tensor-parallel: weight/x split along K
// across ranks; each rank computes a partial matmul; HcclAllReduce sums them to the full x@W.
// Usage: mc2_2node <rank> <nranks> <rootinfo_file>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <unistd.h>
#include <cstdint>
#include "acl/acl.h"
#include "hccl/hccl.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_mc2.h"

#define CK(x) do { int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d code=%d\n",__FILE__,__LINE__,_r);exit(1);} } while(0)
static int g_rank, g_nranks, g_bad = 0;
static HcclComm g_comm; static aclrtStream g_stream;
static aclTensor *T(std::vector<int64_t> d, void *p, aclDataType dt=ACL_FLOAT) { return aclCreateTensor(d.data(), d.size(), dt, nullptr, 0, ACL_FORMAT_ND, d.data(), d.size(), p); }
static void *dev(int64_t b) { void *d; CK(aclrtMalloc(&d, b, ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
// minimal IEEE half codec for verifying fp16 outputs
static float h2f(uint16_t h){ uint32_t s=(h>>15)&1,e=(h>>10)&0x1f,m=h&0x3ff,bits; if(!e){ if(!m){bits=s<<31;} else { e=127-15+1; while(!(m&0x400)){m<<=1;e--;} m&=0x3ff; bits=(s<<31)|(e<<23)|(m<<13);} } else if(e==0x1f){ bits=(s<<31)|0x7f800000|(m<<13);} else { bits=(s<<31)|((e-15+127)<<23)|(m<<13);} float f; __builtin_memcpy(&f,&bits,4); return f; }
static uint16_t f2h(float f){ uint32_t b; __builtin_memcpy(&b,&f,4); uint32_t s=(b>>16)&0x8000,e=((b>>23)&0xff),m=b&0x7fffff; if(e<103)return s; if(e>142)return s|0x7c00; int ne=e-112; return s|(ne<<10)|(m>>13); }

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s rank nranks idfile\n", argv[0]); return 1; }
    g_rank = atoi(argv[1]); g_nranks = atoi(argv[2]); const char *idf = argv[3];
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&g_stream));
    HcclRootInfo root;
    if (g_rank == 0) { CK(HcclGetRootInfo(&root)); FILE *f = fopen(idf, "wb"); fwrite(&root, sizeof root, 1, f); fclose(f); }
    else { for (int i = 0; i < 600; i++) { FILE *f = fopen(idf, "rb"); if (f && fread(&root, sizeof root, 1, f) == 1) { fclose(f); goto got; } if (f) fclose(f); usleep(100000); } return 1; got:; }
    CK(HcclCommInitRootInfo(g_nranks, &root, g_rank, &g_comm));

    {
        const int64_t M = 32, K = 64, N = 48;            // global problem
        const int64_t Kr = K / g_nranks;                 // each rank owns a K-slice (assume K % nranks == 0)
        // Deterministic global x[M,K], W[K,N] — identical on every rank (no seed divergence).
        std::vector<float> x(M*K), W(K*N);
        for (int64_t i = 0; i < M*K; i++) x[i] = std::sin(0.01*i + 1.0);
        for (int64_t i = 0; i < K*N; i++) W[i] = std::cos(0.013*i + 0.5);
        // This rank's slice: xr[M,Kr] = x[:, rank*Kr:(rank+1)*Kr]; Wr[Kr,N] = W[rank*Kr:(rank+1)*Kr, :]
        std::vector<float> xr(M*Kr), Wr(Kr*N);
        for (int64_t m = 0; m < M; m++) for (int64_t k = 0; k < Kr; k++) xr[m*Kr+k] = x[m*K + g_rank*Kr + k];
        for (int64_t k = 0; k < Kr; k++) for (int64_t n = 0; n < N; n++) Wr[k*N+n] = W[(g_rank*Kr+k)*N + n];
        void *dx = dev(M*Kr*4), *dw = dev(Kr*N*4), *dout = dev(M*N*4);
        CK(aclrtMemcpy(dx, M*Kr*4, xr.data(), M*Kr*4, ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw, Kr*N*4, Wr.data(), Kr*N*4, ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx = T({M,Kr}, dx), *tw = T({Kr,N}, dw), *to = T({M,N}, dout);
        uint64_t ws = 0; aclOpExecutor *ex = nullptr;
        CK(aclnnMatmulAllReduceGetWorkspaceSize(tx, tw, nullptr, g_comm, to, &ws, &ex));
        void *wsp = nullptr; if (ws) wsp = dev(ws);
        CK(aclnnMatmulAllReduce(wsp, ws, ex, g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(), M*N*4, dout, M*N*4, ACL_MEMCPY_DEVICE_TO_HOST));
        // reference: full x@W
        double me = 0, mr = 0;
        for (int64_t m = 0; m < M; m++) for (int64_t n = 0; n < N; n++) {
            double acc = 0; for (int64_t k = 0; k < K; k++) acc += (double)x[m*K+k]*W[k*N+n];
            me = std::max(me, std::fabs(got[m*N+n]-acc)); mr = std::max(mr, std::fabs(acc));
        }
        double rel = me/(mr+1e-9); if (rel > 1e-4) g_bad++;
        printf("[rank%d] MatmulAllReduce rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(to);
    }

    // MatmulReduceScatter: K-split partial [M,N] → reduce-scatter → out[M/nranks, N] (rank's row block of full x@W)
    {
        const int64_t M = 32, K = 64, N = 48, Kr = K / g_nranks, Mloc = M / g_nranks;
        std::vector<float> x(M*K), W(K*N);
        for (int64_t i=0;i<M*K;i++) x[i]=std::sin(0.01*i+1.0);
        for (int64_t i=0;i<K*N;i++) W[i]=std::cos(0.013*i+0.5);
        std::vector<float> xr(M*Kr), Wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=x[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) Wr[k*N+n]=W[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr*4), *dw=dev(Kr*N*4), *dout=dev(Mloc*N*4);
        CK(aclrtMemcpy(dx,M*Kr*4,xr.data(),M*Kr*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw,Kr*N*4,Wr.data(),Kr*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx), *tw=T({Kr,N},dw), *to=T({Mloc,N},dout);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnMatmulReduceScatterGetWorkspaceSize(tx,tw,nullptr,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws);
        CK(aclnnMatmulReduceScatter(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(Mloc*N); CK(aclrtMemcpy(got.data(),Mloc*N*4,dout,Mloc*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<Mloc;m++) for (int64_t n=0;n<N;n++){ int64_t gm=g_rank*Mloc+m; double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)x[gm*K+k]*W[k*N+n];
            me=std::max(me,std::fabs(got[m*N+n]-acc)); mr=std::max(mr,std::fabs(acc)); }
        double rel=me/(mr+1e-9); if(rel>1e-4) g_bad++;
        printf("[rank%d] MatmulReduceScatter rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(to);
    }

    // QuantMatmulAllReduce: int8 K-split; per-rank scale·partial summed == scale·full (scale linear over K)
    {
        const int64_t M = 16, K = 64, N = 32, Kr = K / g_nranks;
        std::vector<int8_t> X(M*K), Wt(K*N);
        for (int64_t i=0;i<M*K;i++) X[i]=(int8_t)(((i*7+3)%201)-100);
        for (int64_t i=0;i<K*N;i++) Wt[i]=(int8_t)(((i*5+1)%201)-100);
        std::vector<uint16_t> sc(N); std::vector<float> scf(N); for (int64_t n=0;n<N;n++){ scf[n]=0.0005f*(1+n%9); sc[n]=f2h(scf[n]); }
        std::vector<int8_t> xr(M*Kr), wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=X[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) wr[k*N+n]=Wt[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr), *dw=dev(Kr*N), *dsc=dev(N*2), *dout=dev(M*N*2);
        CK(aclrtMemcpy(dx,M*Kr,xr.data(),M*Kr,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw,Kr*N,wr.data(),Kr*N,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dsc,N*2,sc.data(),N*2,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx,ACL_INT8), *tw=T({Kr,N},dw,ACL_INT8), *tsc=T({N},dsc,ACL_FLOAT16), *to=T({M,N},dout,ACL_FLOAT16);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnQuantMatmulAllReduceGetWorkspaceSize(tx,tw,tsc,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws);
        CK(aclnnQuantMatmulAllReduce(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<uint16_t> got(M*N); CK(aclrtMemcpy(got.data(),M*N*2,dout,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<M;m++) for (int64_t n=0;n<N;n++){ long long acc=0; for(int64_t k=0;k<K;k++) acc+=(long long)X[m*K+k]*Wt[k*N+n]; double ref=(double)acc*scf[n];
            me=std::max(me,std::fabs(h2f(got[m*N+n])-ref)); mr=std::max(mr,std::fabs(ref)); }
        double rel=me/(mr+1e-9); if(rel>3e-2) g_bad++;
        printf("[rank%d] QuantMatmulAllReduce rel=%.3e %s\n", g_rank, rel, rel<=3e-2?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tsc); aclDestroyTensor(to);
    }
    // WeightQuantMatmulAllReduce (W8A16): fp16 x, int8 weight, per-N antiquant scale+offset; K-split TP →
    //   each rank computes x_shard @ dequant(w_shard) and AllReduce(SUM) == full x @ dequant(W).
    {
        const int64_t M = 16, K = 64, N = 32, Kr = K / g_nranks;
        std::vector<float> Xf(M*K); std::vector<int8_t> Wt(K*N);
        for (int64_t i=0;i<M*K;i++) Xf[i]=0.01f*(float)(((i*7+3)%41)-20);
        for (int64_t i=0;i<K*N;i++) Wt[i]=(int8_t)(((i*5+1)%201)-100);
        std::vector<float> scf(N), off(N); for (int64_t n=0;n<N;n++){ scf[n]=0.002f*(1+n%7); off[n]=(float)((n%3)-1); }
        std::vector<uint16_t> Xh(M*K); for(int64_t i=0;i<M*K;i++) Xh[i]=f2h(Xf[i]);
        std::vector<uint16_t> sc(N),of(N); for(int64_t n=0;n<N;n++){ sc[n]=f2h(scf[n]); of[n]=f2h(off[n]); }
        std::vector<uint16_t> xr(M*Kr); std::vector<int8_t> wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=Xh[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) wr[k*N+n]=Wt[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr*2), *dw=dev(Kr*N), *dsc=dev(N*2), *dof=dev(N*2), *dout=dev(M*N*2);
        CK(aclrtMemcpy(dx,M*Kr*2,xr.data(),M*Kr*2,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw,Kr*N,wr.data(),Kr*N,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dsc,N*2,sc.data(),N*2,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dof,N*2,of.data(),N*2,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx,ACL_FLOAT16), *tw=T({Kr,N},dw,ACL_INT8), *tsc=T({N},dsc,ACL_FLOAT16), *tof=T({N},dof,ACL_FLOAT16), *to=T({M,N},dout,ACL_FLOAT16);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnWeightQuantMatmulAllReduceGetWorkspaceSize(tx,tw,tsc,tof,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws);
        CK(aclnnWeightQuantMatmulAllReduce(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<uint16_t> got(M*N); CK(aclrtMemcpy(got.data(),M*N*2,dout,M*N*2,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<M;m++) for (int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++){ double wf=((double)Wt[k*N+n]-off[n])*scf[n]; acc+=(double)Xf[m*K+k]*wf; }
            me=std::max(me,std::fabs(h2f(got[m*N+n])-acc)); mr=std::max(mr,std::fabs(acc)); }
        double rel=me/(mr+1e-9); if(rel>3e-2) g_bad++;
        printf("[rank%d] WeightQuantMatmulAllReduce rel=%.3e %s\n", g_rank, rel, rel<=3e-2?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(tsc); aclDestroyTensor(tof); aclDestroyTensor(to);
    }
    // MatmulAllGather: row-sharded x; each rank computes its [Mr,N] block; AllGather → out[M,N] = X_full @ W
    {
        const int64_t M = 32, K = 48, N = 40, Mr = M / g_nranks;
        std::vector<float> x(M*K), W(K*N);
        for (int64_t i=0;i<M*K;i++) x[i]=std::sin(0.017*i+0.3);
        for (int64_t i=0;i<K*N;i++) W[i]=std::cos(0.011*i+0.7);
        std::vector<float> xr(Mr*K);
        for (int64_t m=0;m<Mr;m++) for (int64_t k=0;k<K;k++) xr[m*K+k]=x[(g_rank*Mr+m)*K+k];
        void *dx=dev(Mr*K*4), *dw=dev(K*N*4), *dout=dev(M*N*4);
        CK(aclrtMemcpy(dx,Mr*K*4,xr.data(),Mr*K*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw,K*N*4,W.data(),K*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({Mr,K},dx), *tw=T({K,N},dw), *to=T({M,N},dout);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnMatmulAllGatherGetWorkspaceSize(tx,tw,nullptr,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws);
        CK(aclnnMatmulAllGather(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(),M*N*4,dout,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<M;m++) for (int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)x[m*K+k]*W[k*N+n];
            me=std::max(me,std::fabs(got[m*N+n]-acc)); mr=std::max(mr,std::fabs(acc)); }
        double rel=me/(mr+1e-9); if(rel>1e-4) g_bad++;
        printf("[rank%d] MatmulAllGather rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(to);
    }
    // MatmulAllReduceAddRmsNorm: y = RmsNorm(AllReduce(x@W) + residual)·gamma (K-split TP + tail rmsnorm)
    {
        const int64_t M = 16, K = 64, N = 32, Kr = K / g_nranks; double eps = 1e-6;
        std::vector<float> x(M*K), W(K*N), res(M*N), gam(N);
        for (int64_t i=0;i<M*K;i++) x[i]=std::sin(0.02*i+0.2);
        for (int64_t i=0;i<K*N;i++) W[i]=std::cos(0.015*i+0.4);
        for (int64_t i=0;i<M*N;i++) res[i]=0.1*std::sin(0.03*i);
        for (int64_t n=0;n<N;n++) gam[n]=0.5+0.5*std::cos(0.1*n);
        std::vector<float> xr(M*Kr), Wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=x[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) Wr[k*N+n]=W[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr*4),*dw=dev(Kr*N*4),*dres=dev(M*N*4),*dg=dev(N*4),*dy=dev(M*N*4),*dsum=dev(M*N*4);
        CK(aclrtMemcpy(dx,M*Kr*4,xr.data(),M*Kr*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dw,Kr*N*4,Wr.data(),Kr*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dres,M*N*4,res.data(),M*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dg,N*4,gam.data(),N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx),*tw=T({Kr,N},dw),*tres=T({M,N},dres),*tg=T({N},dg),*ty=T({M,N},dy),*tsum=T({M,N},dsum);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,tres,tg,eps,g_comm,ty,tsum,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws);
        CK(aclnnMatmulAllReduceAddRmsNorm(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(),M*N*4,dy,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<M;m++){ double ss=0; std::vector<double> s(N); for(int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)x[m*K+k]*W[k*N+n]; s[n]=acc+res[m*N+n]; ss+=s[n]*s[n]; }
            double inv=1.0/std::sqrt(ss/N+eps); for(int64_t n=0;n<N;n++){ double ref=s[n]*inv*gam[n]; me=std::max(me,std::fabs(got[m*N+n]-ref)); mr=std::max(mr,std::fabs(ref)); } }
        double rel=me/(mr+1e-9); if(rel>1e-3) g_bad++;
        printf("[rank%d] MatmulAllReduceAddRmsNorm rel=%.3e %s\n", g_rank, rel, rel<=1e-3?"PASS":"FAIL");
        aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tres);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
    }
    // AllGatherMatmul: AllGather row-shard x[Mr,K] → xfull[M,K], then xfull@W → out[M,N]. Ref = concat(shards)@W.
    {
        const int64_t Mr=8, K=32, N=16, M=Mr*g_nranks;
        auto vx=[](int r,int m,int k){ return (float)std::sin(0.02*(r*100 + m*32 + k)); };
        std::vector<float> xr(Mr*K), W(K*N);
        for (int64_t m=0;m<Mr;m++) for (int64_t k=0;k<K;k++) xr[m*K+k]=vx((int)g_rank,(int)m,(int)k);
        for (int64_t k=0;k<K;k++) for (int64_t n=0;n<N;n++) W[k*N+n]=std::cos(0.015*(k*N+n));
        void *dx=dev(Mr*K*4),*dw=dev(K*N*4),*dout=dev(M*N*4);
        CK(aclrtMemcpy(dx,Mr*K*4,xr.data(),Mr*K*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dw,K*N*4,W.data(),K*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({Mr,K},dx),*tw=T({K,N},dw),*to=T({M,N},dout);
        uint64_t ws=0; aclOpExecutor *ex=nullptr; CK(aclnnAllGatherMatmulGetWorkspaceSize(tx,tw,nullptr,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws); CK(aclnnAllGatherMatmul(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(),M*N*4,dout,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr2=0;
        for (int64_t i=0;i<M;i++){ int r=(int)(i/Mr), m=(int)(i%Mr); for (int64_t n=0;n<N;n++){ double acc=0; for (int64_t k=0;k<K;k++) acc+=(double)vx(r,m,(int)k)*W[k*N+n]; me=std::max(me,std::fabs(got[i*N+n]-acc)); mr2=std::max(mr2,std::fabs(acc)); } }
        double rel=me/(mr2+1e-9); if(rel>1e-4) g_bad++; printf("[rank%d] AllGatherMatmul rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(to);
    }
    // GroupedMatMulAllReduce (E=1 group → plain matmul): K-split local grouped matmul → AllReduce(SUM) = full x@W.
    {
        const int64_t M=16, K=64, N=32, Kr=K/g_nranks;
        std::vector<float> x(M*K), W(K*N);
        for (int64_t i=0;i<M*K;i++) x[i]=std::sin(0.02*i+0.2);
        for (int64_t i=0;i<K*N;i++) W[i]=std::cos(0.015*i+0.4);
        std::vector<float> xr(M*Kr), Wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=x[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) Wr[k*N+n]=W[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr*4),*dw=dev(Kr*N*4),*dout=dev(M*N*4);
        CK(aclrtMemcpy(dx,M*Kr*4,xr.data(),M*Kr*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dw,Kr*N*4,Wr.data(),Kr*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx),*tw=T({1,Kr,N},dw),*to=T({M,N},dout);
        int64_t gl[1]={M}; aclIntArray *tgl=aclCreateIntArray(gl,1);
        uint64_t ws=0; aclOpExecutor *ex=nullptr; CK(aclnnGroupedMatMulAllReduceGetWorkspaceSize(tx,tw,tgl,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws); CK(aclnnGroupedMatMulAllReduce(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(),M*N*4,dout,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr2=0; for (int64_t m=0;m<M;m++) for (int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)x[m*K+k]*W[k*N+n]; me=std::max(me,std::fabs(got[m*N+n]-acc)); mr2=std::max(mr2,std::fabs(acc)); }
        double rel=me/(mr2+1e-9); if(rel>1e-4) g_bad++; printf("[rank%d] GroupedMatMulAllReduce rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(tw); aclDestroyTensor(to);
    }
    // QuantAllReduce: real HcclAllReduce(SUM) of per-rank x. rank r holds (r+1)*v → sum = (Σ r+1)·v.
    {
        const int64_t M=8, N=16; int64_t NN=M*N;
        std::vector<float> v(NN), x(NN); for (int64_t i=0;i<NN;i++){ v[i]=std::sin(0.03*i+0.1); x[i]=(float)((g_rank+1)*v[i]); }
        void *dx=dev(NN*4), *dout=dev(NN*4); CK(aclrtMemcpy(dx,NN*4,x.data(),NN*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,N},dx), *to=T({M,N},dout);
        uint64_t ws=0; aclOpExecutor *ex=nullptr; CK(aclnnQuantAllReduceGetWorkspaceSize(tx,g_comm,to,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws); CK(aclnnQuantAllReduce(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(NN); CK(aclrtMemcpy(got.data(),NN*4,dout,NN*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double sc=g_nranks*(g_nranks+1)/2.0, me=0, mr=0; for (int64_t i=0;i<NN;i++){ double ref=sc*v[i]; me=std::max(me,std::fabs(got[i]-ref)); mr=std::max(mr,std::fabs(ref)); }
        double rel=me/(mr+1e-9); if(rel>1e-4) g_bad++; printf("[rank%d] QuantAllReduce rel=%.3e %s\n", g_rank, rel, rel<=1e-4?"PASS":"FAIL");
        aclDestroyTensor(tx); aclDestroyTensor(to);
    }
    // WeightQuantMatmulAllReduceAddRmsNorm: y = RmsNorm(AllReduce(x@W)+residual)·gamma (matmul4 fusion, scale ignored→plain matmul).
    {
        const int64_t M=16, K=64, N=32, Kr=K/g_nranks; double eps=1e-6;
        std::vector<float> x(M*K), W(K*N), res(M*N), gam(N);
        for (int64_t i=0;i<M*K;i++) x[i]=std::sin(0.02*i+0.2);
        for (int64_t i=0;i<K*N;i++) W[i]=std::cos(0.015*i+0.4);
        for (int64_t i=0;i<M*N;i++) res[i]=0.1*std::sin(0.03*i);
        for (int64_t n=0;n<N;n++) gam[n]=0.5+0.5*std::cos(0.1*n);
        std::vector<float> xr(M*Kr), Wr(Kr*N);
        for (int64_t m=0;m<M;m++) for (int64_t k=0;k<Kr;k++) xr[m*Kr+k]=x[m*K+g_rank*Kr+k];
        for (int64_t k=0;k<Kr;k++) for (int64_t n=0;n<N;n++) Wr[k*N+n]=W[(g_rank*Kr+k)*N+n];
        void *dx=dev(M*Kr*4),*dw=dev(Kr*N*4),*dres=dev(M*N*4),*dg=dev(N*4),*dy=dev(M*N*4),*dsum=dev(M*N*4);
        CK(aclrtMemcpy(dx,M*Kr*4,xr.data(),M*Kr*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dw,Kr*N*4,Wr.data(),Kr*N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dres,M*N*4,res.data(),M*N*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dg,N*4,gam.data(),N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({M,Kr},dx),*tw=T({Kr,N},dw),*tres=T({M,N},dres),*tg=T({N},dg),*ty=T({M,N},dy),*tsum=T({M,N},dsum);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnWeightQuantMatmulAllReduceAddRmsNormGetWorkspaceSize(tx,tw,nullptr,tres,tg,eps,g_comm,ty,tsum,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws); CK(aclnnWeightQuantMatmulAllReduceAddRmsNorm(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(M*N); CK(aclrtMemcpy(got.data(),M*N*4,dy,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t m=0;m<M;m++){ double ss=0; std::vector<double> sv(N); for(int64_t n=0;n<N;n++){ double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)x[m*K+k]*W[k*N+n]; sv[n]=acc+res[m*N+n]; ss+=sv[n]*sv[n]; }
            double inv=1.0/std::sqrt(ss/N+eps); for(int64_t n=0;n<N;n++){ double ref=sv[n]*inv*gam[n]; me=std::max(me,std::fabs(got[m*N+n]-ref)); mr=std::max(mr,std::fabs(ref)); } }
        double rel=me/(mr+1e-9); if(rel>1e-3) g_bad++; printf("[rank%d] WeightQuantMatmulAllReduceAddRmsNorm rel=%.3e %s\n", g_rank, rel, rel<=1e-3?"PASS":"FAIL");
        aclDestroyTensor(tx);aclDestroyTensor(tw);aclDestroyTensor(tres);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
    }
    // MoeDistributeDispatch (AlltoAll): x[nranks,C,H], block j → rank j; recv block j = rank j's block r. Combine = inverse.
    {
        const int64_t nr = g_nranks, C = 4, H = 8;
        // Per-element unique tag encode(src,dst,elem): block j element i = (src*100+dst)*1000+i. After the
        // AlltoAll, my block j must hold rank j's block destined for me, element-for-element — catches
        // intra-block permutation/stride bugs a uniform-per-block tag would miss.
        std::vector<float> x(nr*C*H); for (int64_t j=0;j<nr;j++) for (int64_t i=0;i<C*H;i++) x[j*C*H+i] = (float)((g_rank*100 + j)*1000 + i);
        void *dx=dev(nr*C*H*4), *dd=dev(nr*C*H*4), *dc=dev(nr*C*H*4);
        CK(aclrtMemcpy(dx,nr*C*H*4,x.data(),nr*C*H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({nr,C,H},dx), *td=T({nr,C,H},dd), *tc=T({nr,C,H},dc);
        uint64_t ws=0; aclOpExecutor *ex=nullptr;
        CK(aclnnMoeDistributeDispatchGetWorkspaceSize(tx,g_comm,td,&ws,&ex)); CK(aclnnMoeDistributeDispatch(nullptr,0,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(nr*C*H); CK(aclrtMemcpy(got.data(),nr*C*H*4,dd,nr*C*H*4,ACL_MEMCPY_DEVICE_TO_HOST));
        int bad=0; for (int64_t j=0;j<nr;j++) for (int64_t i=0;i<C*H;i++) if (got[j*C*H+i] != (float)((j*100 + g_rank)*1000 + i)) bad++;   // recv block j = sender j's block (dst=me), element i preserved
        if (bad) g_bad++; printf("[rank%d] MoeDistributeDispatch %s\n", g_rank, bad?"FAIL":"PASS");
        // Combine round-trip: dispatch(dispatch(x)) == x
        CK(aclnnMoeDistributeCombineGetWorkspaceSize(td,g_comm,tc,&ws,&ex)); CK(aclnnMoeDistributeCombine(nullptr,0,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> rt(nr*C*H); CK(aclrtMemcpy(rt.data(),nr*C*H*4,dc,nr*C*H*4,ACL_MEMCPY_DEVICE_TO_HOST));
        int bad2=0; for (int64_t i=0;i<nr*C*H;i++) if (rt[i]!=x[i]) bad2++;
        if (bad2) g_bad++; printf("[rank%d] MoeDistributeCombine roundtrip %s\n", g_rank, bad2?"FAIL":"PASS");
        aclDestroyTensor(tx); aclDestroyTensor(td); aclDestroyTensor(tc);
    }
    // MoeDistributeCombineAddRmsNorm: AlltoAll combine (block j ← rank j's block-for-me) then RmsNorm(combined+residual)·gamma over H.
    {
        const int64_t nr=g_nranks, C=4, H=8; int64_t NN=nr*C*H; double eps=1e-6;
        auto val=[](int src,int dst,int c,int h){ return (float)std::sin(0.1*(src*7+dst*3+c)+0.05*h); };
        std::vector<float> x(NN), res(NN), gam(H);
        for (int64_t j=0;j<nr;j++) for (int64_t c=0;c<C;c++) for (int64_t h=0;h<H;h++) x[(j*C+c)*H+h]=val((int)g_rank,(int)j,(int)c,(int)h);  // my block j (dst=j)
        for (int64_t i=0;i<NN;i++) res[i]=0.1f*std::cos(0.07*i);
        for (int64_t h=0;h<H;h++) gam[h]=0.5f+0.5f*std::sin(0.2*h);
        void *dx=dev(NN*4),*dres=dev(NN*4),*dg=dev(H*4),*dy=dev(NN*4),*dsum=dev(NN*4);
        CK(aclrtMemcpy(dx,NN*4,x.data(),NN*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dres,NN*4,res.data(),NN*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dg,H*4,gam.data(),H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=T({nr,C,H},dx),*tres=T({nr,C,H},dres),*tg=T({H},dg),*ty=T({nr,C,H},dy),*tsum=T({nr,C,H},dsum);
        uint64_t ws=0; aclOpExecutor *ex=nullptr; CK(aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize(tx,tres,tg,eps,g_comm,ty,tsum,&ws,&ex));
        void *wsp=nullptr; if(ws) wsp=dev(ws); CK(aclnnMoeDistributeCombineAddRmsNorm(wsp,ws,ex,g_stream)); CK(aclrtSynchronizeStream(g_stream));
        std::vector<float> got(NN); CK(aclrtMemcpy(got.data(),NN*4,dy,NN*4,ACL_MEMCPY_DEVICE_TO_HOST));
        double me=0,mr=0;
        for (int64_t j=0;j<nr;j++) for (int64_t c=0;c<C;c++){ double ss=0; std::vector<double> sv(H);
            for (int64_t h=0;h<H;h++){ double comb=val((int)j,(int)g_rank,(int)c,(int)h); sv[h]=comb+res[(j*C+c)*H+h]; ss+=sv[h]*sv[h]; }  // combined block j = rank j's data for me
            double inv=1.0/std::sqrt(ss/H+eps); for (int64_t h=0;h<H;h++){ double ref=sv[h]*inv*gam[h]; me=std::max(me,std::fabs(got[(j*C+c)*H+h]-ref)); mr=std::max(mr,std::fabs(ref)); } }
        double rel=me/(mr+1e-9); if(rel>1e-3) g_bad++; printf("[rank%d] MoeDistributeCombineAddRmsNorm rel=%.3e %s\n", g_rank, rel, rel<=1e-3?"PASS":"FAIL");
        aclDestroyTensor(tx);aclDestroyTensor(tres);aclDestroyTensor(tg);aclDestroyTensor(ty);aclDestroyTensor(tsum);
    }
    CK(HcclCommDestroy(g_comm)); CK(aclrtDestroyStream(g_stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    printf("[rank%d] %s\n", g_rank, g_bad ? "== FAILED ==" : "== ALL PASSED ==");
    return g_bad ? 1 : 0;
}
