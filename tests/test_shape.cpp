// Shape/index operator cross-check: Permute/Slice/Tile/ConstantPadNd/Gather/Cat. Pure ACL client + CPU reference.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

static aclrtStream g_stream;
static int g_pass = 0, g_fail = 0;

struct DevBuf {
    void *p = nullptr; size_t bytes = 0;
    DevBuf(size_t b) : bytes(b) { CHECK(aclrtMalloc(&p, b, ACL_MEM_MALLOC_HUGE_FIRST)); }
    ~DevBuf() { aclrtFree(p); }
    void up(const void *h)   { CHECK(aclrtMemcpy(p, bytes, h, bytes, ACL_MEMCPY_HOST_TO_DEVICE)); }
    void down(void *h) const { CHECK(aclrtMemcpy(h, bytes, p, bytes, ACL_MEMCPY_DEVICE_TO_HOST)); }
};

static aclTensor *mk(const std::vector<int64_t> &dims, aclDataType dt, void *data) {
    return aclCreateTensor(dims.data(), dims.size(), dt, nullptr, 0, ACL_FORMAT_ND, dims.data(), dims.size(), data);
}
static void report(const std::string &name, int64_t bad) {
    (bad == 0 ? g_pass : g_fail)++;
    printf("%-30s mismatch=%lld %s\n", name.c_str(), (long long)bad, bad == 0 ? "PASS" : "FAIL");
}
template <typename GetWs>
static void exec2(GetWs getws, aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(getws(&ws, &ex));
    void *wsp = nullptr;
    if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    if (wsp) aclrtFree(wsp);
}
static std::vector<float> iota(int64_t n) { std::vector<float> v(n); for (int64_t i=0;i<n;i++) v[i]=(float)(i+1); return v; }

// Permute [A,B,C] -> dims{2,0,1} => out[C,A,B]
static void t_permute() {
    const int64_t A=3,B=4,C=5; auto in=iota(A*B*C);
    std::vector<float> hz(A*B*C);
    DevBuf da(A*B*C*4), dz(A*B*C*4); da.up(in.data());
    aclTensor *ta=mk({A,B,C},ACL_FLOAT,da.p), *tz=mk({C,A,B},ACL_FLOAT,dz.p);
    int64_t pd[3]={2,0,1}; aclIntArray *dims=aclCreateIntArray(pd,3);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(ta,dims,tz,w,e);},aclnnPermute);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t c=0;c<C;c++) for (int64_t a=0;a<A;a++) for (int64_t b=0;b<B;b++) {
        float ref = in[(a*B+b)*C + c];
        if (hz[(c*A+a)*B + b] != ref) bad++;
    }
    report("Permute [3,4,5]->{2,0,1}", bad);
    aclDestroyIntArray(dims); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Slice along dim1 [1,7) step2 on [4,8]
static void t_slice() {
    const int64_t M=4,N=8; auto in=iota(M*N);
    int64_t start=1,end=7,step=2,len=(end-start+step-1)/step;
    std::vector<float> hz(M*len);
    DevBuf da(M*N*4), dz(M*len*4); da.up(in.data());
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *tz=mk({M,len},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSliceGetWorkspaceSize(ta,1,start,end,step,tz,w,e);},aclnnSlice);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<M;i++) for (int64_t j=0;j<len;j++) if (hz[i*len+j]!=in[i*N + start+j*step]) bad++;
    report("Slice [4,8] dim1 [1,7)/2", bad);
    aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Tile [2,3] repeats{2,3} -> [4,9]
static void t_tile() {
    const int64_t M=2,N=3; auto in=iota(M*N);
    int64_t r0=2,r1=3; int64_t OM=M*r0,ON=N*r1;
    std::vector<float> hz(OM*ON);
    DevBuf da(M*N*4), dz(OM*ON*4); da.up(in.data());
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *tz=mk({OM,ON},ACL_FLOAT,dz.p);
    int64_t rp[2]={r0,r1}; aclIntArray *reps=aclCreateIntArray(rp,2);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTileGetWorkspaceSize(ta,reps,tz,w,e);},aclnnTile);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<OM;i++) for (int64_t j=0;j<ON;j++) if (hz[i*ON+j]!=in[(i%M)*N + (j%N)]) bad++;
    report("Tile [2,3] x{2,3}", bad);
    aclDestroyIntArray(reps); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Pad [2,3] padding{1,1,2,0} value=-1 -> [4,5]
static void t_pad() {
    const int64_t M=2,N=3; auto in=iota(M*N);
    int64_t lo0=1,hi0=1,lo1=2,hi1=0; int64_t OM=M+lo0+hi0,ON=N+lo1+hi1; float val=-1;
    std::vector<float> hz(OM*ON);
    DevBuf da(M*N*4), dz(OM*ON*4); da.up(in.data());
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *tz=mk({OM,ON},ACL_FLOAT,dz.p);
    int64_t pw[4]={lo0,hi0,lo1,hi1}; aclIntArray *pad=aclCreateIntArray(pw,4);
    aclScalar *sv=aclCreateScalar(&val,ACL_FLOAT);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnConstantPadNdGetWorkspaceSize(ta,pad,sv,tz,w,e);},aclnnConstantPadNd);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<OM;i++) for (int64_t j=0;j<ON;j++) {
        int64_t ii=i-lo0, jj=j-lo1;
        float ref=(ii>=0&&ii<M&&jj>=0&&jj<N)? in[ii*N+jj] : val;
        if (hz[i*ON+j]!=ref) bad++;
    }
    report("Pad [2,3] {1,1,2,0} v=-1", bad);
    aclDestroyIntArray(pad); aclDestroyScalar(sv); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// Gather along dim0 index{2,0,2,1} on [4,3]
static void t_gather() {
    const int64_t M=4,N=3; auto in=iota(M*N);
    int64_t idx[4]={2,0,2,1}; int64_t L=4;
    std::vector<float> hz(L*N);
    DevBuf da(M*N*4), di(L*8), dz(L*N*4); da.up(in.data()); di.up(idx);
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *ti=mk({L},ACL_INT64,di.p), *tz=mk({L,N},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherGetWorkspaceSize(ta,0,ti,tz,w,e);},aclnnGather);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t l=0;l<L;l++) for (int64_t j=0;j<N;j++) if (hz[l*N+j]!=in[idx[l]*N+j]) bad++;
    report("Gather [4,3] dim0 idx{2,0,2,1}", bad);
    aclDestroyTensor(ta); aclDestroyTensor(ti); aclDestroyTensor(tz);
}

// StridedSlice: [6,8] begin{1,2} end{6,8} strides{2,3} -> [3,2]
static void t_strided_slice() {
    const int64_t M=6,N=8; auto in=iota(M*N);
    int64_t b[2]={1,2}, en[2]={6,8}, sp[2]={2,3};
    int64_t L0=(en[0]-b[0]+sp[0]-1)/sp[0], L1=(en[1]-b[1]+sp[1]-1)/sp[1];
    std::vector<float> hz(L0*L1);
    DevBuf da(M*N*4), dz(L0*L1*4); da.up(in.data());
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *tz=mk({L0,L1},ACL_FLOAT,dz.p);
    aclIntArray *ba=aclCreateIntArray(b,2), *ea=aclCreateIntArray(en,2), *sa=aclCreateIntArray(sp,2);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnStridedSliceGetWorkspaceSize(ta,ba,ea,sa,tz,w,e);},aclnnStridedSlice);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<L0;i++) for (int64_t j=0;j<L1;j++)
        if (hz[i*L1+j]!=in[(b[0]+i*sp[0])*N + (b[1]+j*sp[1])]) bad++;
    report("StridedSlice [6,8] str{2,3}", bad);
    aclDestroyIntArray(ba);aclDestroyIntArray(ea);aclDestroyIntArray(sa); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// GatherV2 batch_dims=1: self[2,4,3] index[2,2] axis=1 -> out[2,2,3]
static void t_gather_v2() {
    const int64_t B=2,A=4,T=3,I=2; auto in=iota(B*A*T);
    int64_t idx[B*I]={3,0, 1,2};   // batch0 selects rows {3,0}, batch1 selects rows {1,2}
    std::vector<float> hz(B*I*T);
    DevBuf da(B*A*T*4), di(B*I*8), dz(B*I*T*4); da.up(in.data()); di.up(idx);
    aclTensor *ta=mk({B,A,T},ACL_FLOAT,da.p), *ti=mk({B,I},ACL_INT64,di.p), *tz=mk({B,I,T},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherV2GetWorkspaceSize(ta,1,1,ti,tz,w,e);},aclnnGatherV2);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t bb=0;bb<B;bb++) for (int64_t ii=0;ii<I;ii++) for (int64_t t=0;t<T;t++) {
        int64_t g=idx[bb*I+ii];
        if (hz[(bb*I+ii)*T+t] != in[(bb*A+g)*T+t]) bad++;
    }
    report("GatherV2 batch_dims=1 [2,4,3]", bad);
    aclDestroyTensor(ta); aclDestroyTensor(ti); aclDestroyTensor(tz);
}

// Cat along dim1: [2,2] + [2,3] -> [2,5]
static void t_cat() {
    const int64_t M=2,N1=2,N2=3,ON=N1+N2;
    auto a=iota(M*N1); std::vector<float> b(M*N2); for (int64_t i=0;i<M*N2;i++) b[i]=100+i;
    std::vector<float> hz(M*ON);
    DevBuf da(M*N1*4), db(M*N2*4), dz(M*ON*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({M,N1},ACL_FLOAT,da.p), *tb=mk({M,N2},ACL_FLOAT,db.p), *tz=mk({M,ON},ACL_FLOAT,dz.p);
    const aclTensor *list[2]={ta,tb};
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCatGetWorkspaceSize(list,2,1,tz,w,e);},aclnnCat);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<M;i++) {
        for (int64_t j=0;j<N1;j++) if (hz[i*ON+j]!=a[i*N1+j]) bad++;
        for (int64_t j=0;j<N2;j++) if (hz[i*ON+N1+j]!=b[i*N2+j]) bad++;
    }
    report("Cat dim1 [2,2]+[2,3]", bad);
    aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// Native aclTensorList Cat: aclCreateTensorList -> aclnnCatList, result must match the array version
static void t_cat_list() {
    const int64_t M=2,N1=2,N2=3,ON=N1+N2;
    auto a=iota(M*N1); std::vector<float> b(M*N2); for (int64_t i=0;i<M*N2;i++) b[i]=100+i;
    std::vector<float> hz(M*ON);
    DevBuf da(M*N1*4), db(M*N2*4), dz(M*ON*4); da.up(a.data()); db.up(b.data());
    aclTensor *ta=mk({M,N1},ACL_FLOAT,da.p), *tb=mk({M,N2},ACL_FLOAT,db.p), *tz=mk({M,ON},ACL_FLOAT,dz.p);
    const aclTensor *arr[2]={ta,tb};
    aclTensorList *list = aclCreateTensorList(arr, 2);
    uint64_t lsz=0; aclGetTensorListSize(list, &lsz);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCatListGetWorkspaceSize(list,1,tz,w,e);},aclnnCatList);
    dz.down(hz.data());
    int64_t bad = (lsz!=2);
    for (int64_t i=0;i<M;i++) {
        for (int64_t j=0;j<N1;j++) if (hz[i*ON+j]!=a[i*N1+j]) bad++;
        for (int64_t j=0;j<N2;j++) if (hz[i*ON+N1+j]!=b[i*N2+j]) bad++;
    }
    report("CatList(aclTensorList) dim1", bad);
    aclDestroyTensorList(list); aclDestroyTensor(ta); aclDestroyTensor(tb); aclDestroyTensor(tz);
}

// Stack: 3 tensors of [2,3] stacked along new dim1 -> [2,3,3]
static void t_stack() {
    const int64_t M=2,N=3,K=3;   // K tensors
    std::vector<std::vector<float>> ins(K);
    for (int64_t k=0;k<K;k++){ ins[k].resize(M*N); for(int64_t i=0;i<M*N;i++) ins[k][i]=(float)(k*100+i+1); }
    std::vector<DevBuf*> db; std::vector<aclTensor*> ts;
    for (int64_t k=0;k<K;k++){ auto *d=new DevBuf(M*N*4); d->up(ins[k].data()); db.push_back(d);
        ts.push_back(mk({M,N},ACL_FLOAT,d->p)); }
    DevBuf dz(M*K*N*4);
    aclTensor *tz=mk({M,K,N},ACL_FLOAT,dz.p);   // new dim=1 inserted with size=K
    aclTensorList *list=aclCreateTensorList(ts.data(), K);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnStackGetWorkspaceSize(list,1,tz,w,e);},aclnnStack);
    std::vector<float> hz(M*K*N); dz.down(hz.data());
    int64_t bad=0;
    for (int64_t i=0;i<M;i++) for (int64_t k=0;k<K;k++) for (int64_t j=0;j<N;j++)
        if (hz[(i*K+k)*N+j] != ins[k][i*N+j]) bad++;
    report("Stack 3x[2,3] dim1->[2,3,3]", bad);
    aclDestroyTensorList(list); for (auto*t:ts) aclDestroyTensor(t); aclDestroyTensor(tz);
    for (auto*d:db) delete d;
}

// GroupedMatmul weight TensorList variant: x[4,3] split into two groups by groupList{1,3}, each matmul'd with expert W[3,2]
static void t_gmm_weightlist() {
    const int64_t M=4,K=3,N=2,E=2;
    int64_t groups[E]={1,3};
    auto x=iota(M*K);
    std::vector<std::vector<float>> ws(E);
    for (int64_t e=0;e<E;e++){ ws[e].resize(K*N); for(int64_t i=0;i<K*N;i++) ws[e][i]=(float)((e+1)*10+i); }
    DevBuf dx(M*K*4), dz(M*N*4); dx.up(x.data());
    std::vector<DevBuf*> dw; std::vector<aclTensor*> wt;
    for (int64_t e=0;e<E;e++){ auto*d=new DevBuf(K*N*4); d->up(ws[e].data()); dw.push_back(d);
        wt.push_back(mk({K,N},ACL_FLOAT,d->p)); }
    aclTensor *tx=mk({M,K},ACL_FLOAT,dx.p), *tz=mk({M,N},ACL_FLOAT,dz.p);
    aclTensorList *wl=aclCreateTensorList(wt.data(), E);
    aclIntArray *gl=aclCreateIntArray(groups, E);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGroupedMatmulWeightListGetWorkspaceSize(tx,wl,gl,tz,w,e);},aclnnGroupedMatmulWeightList);
    std::vector<float> hz(M*N); dz.down(hz.data());
    // CPU reference: row r belongs to expert ei (by groupList prefix sum)
    int64_t bad=0; int64_t off=0;
    for (int64_t e=0;e<E;e++){ for (int64_t r=0;r<groups[e];r++){ int64_t row=off+r;
        for (int64_t n=0;n<N;n++){ float acc=0; for(int64_t k=0;k<K;k++) acc+=x[row*K+k]*ws[e][k*N+n];
            if (std::fabs(hz[row*N+n]-acc) > 1e-3f) bad++; } } off+=groups[e]; }
    report("GroupedMatmul weightlist [4,3]x2", bad);
    aclDestroyTensorList(wl); aclDestroyIntArray(gl); aclDestroyTensor(tx); aclDestroyTensor(tz);
    for (auto*t:wt) aclDestroyTensor(t); for (auto*d:dw) delete d;
}

// Split along dim1: [2,5] -> [2,2]+[2,3] (inverse of Cat)
static void t_split() {
    const int64_t M=2,N=5; auto in=iota(M*N);
    int64_t N1=2,N2=3;
    std::vector<float> h1(M*N1), h2(M*N2);
    DevBuf da(M*N*4), d1(M*N1*4), d2(M*N2*4); da.up(in.data());
    aclTensor *ta=mk({M,N},ACL_FLOAT,da.p), *t1=mk({M,N1},ACL_FLOAT,d1.p), *t2=mk({M,N2},ACL_FLOAT,d2.p);
    const aclTensor *outs[2]={t1,t2};
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSplitWithSizeGetWorkspaceSize(ta,1,outs,2,w,e);},aclnnSplitWithSize);
    d1.down(h1.data()); d2.down(h2.data());
    int64_t bad=0;
    for (int64_t i=0;i<M;i++) {
        for (int64_t j=0;j<N1;j++) if (h1[i*N1+j]!=in[i*N+j]) bad++;
        for (int64_t j=0;j<N2;j++) if (h2[i*N2+j]!=in[i*N+N1+j]) bad++;
    }
    report("Split dim1 [2,5]->2+3", bad);
    aclDestroyTensor(ta); aclDestroyTensor(t1); aclDestroyTensor(t2);
}

// Permute int32 (verifies dtype-agnostic byte-width shuffle)
static void t_permute_i32() {
    const int64_t A=2,B=3; std::vector<int32_t> in(A*B); for (int64_t i=0;i<A*B;i++) in[i]=(int32_t)(i*7+1);
    std::vector<int32_t> hz(A*B);
    DevBuf da(A*B*4), dz(A*B*4); da.up(in.data());
    aclTensor *ta=mk({A,B},ACL_INT32,da.p), *tz=mk({B,A},ACL_INT32,dz.p);
    int64_t pd[2]={1,0}; aclIntArray *dims=aclCreateIntArray(pd,2);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPermuteGetWorkspaceSize(ta,dims,tz,w,e);},aclnnPermute);
    dz.down(hz.data());
    int64_t bad=0;
    for (int64_t a=0;a<A;a++) for (int64_t b=0;b<B;b++) if (hz[b*A+a]!=in[a*B+b]) bad++;
    report("Permute int32 [2,3]->{1,0}", bad);
    aclDestroyIntArray(dims); aclDestroyTensor(ta); aclDestroyTensor(tz);
}

// FRACTAL_NZ <-> ND: round-trip + per-element match with CPU NZ layout
static void t_nz(int64_t M, int64_t N) {
    const int F=16; int64_t M1=(M+F-1)/F, N1=(N+F-1)/F, nzn=N1*M1*F*F;
    auto X=iota(M*N); std::vector<float> nz(nzn), back(M*N);
    DevBuf dx(M*N*4), dnz(nzn*4), dback(M*N*4); dx.up(X.data());
    aclTensor*tx=mk({M,N},ACL_FLOAT,dx.p),*tnz=mk({N1,M1,F,F},ACL_FLOAT,dnz.p),*tb=mk({M,N},ACL_FLOAT,dback.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataND2NZGetWorkspaceSize(tx,tnz,w,e);},aclnnTransDataND2NZ);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataNZ2NDGetWorkspaceSize(tnz,tb,w,e);},aclnnTransDataNZ2ND);
    dnz.down(nz.data()); dback.down(back.data());
    int64_t bad=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){
        int64_t i1=i/F,i0=i%F,j1=j/F,j0=j%F, off=((j1*M1+i1)*F+i0)*F+j0;
        if(nz[off]!=X[i*N+j]) bad++;            // NZ layout correct
        if(back[i*N+j]!=X[i*N+j]) bad++;        // round-trip restored
    }
    char nm[48]; snprintf(nm,48,"NZ<->ND [%lld,%lld]",(long long)M,(long long)N);
    report(nm,bad);
    aclDestroyTensor(tx);aclDestroyTensor(tnz);aclDestroyTensor(tb);
}

// NCHW <-> NC1HWC0 (5HD) layout + round-trip
static void t_5hd(int64_t N,int64_t C,int64_t H,int64_t W){
    const int F=16; int64_t C1=(C+F-1)/F, fhn=N*C1*H*W*F;
    auto X=iota(N*C*H*W); std::vector<float> hd(fhn), back(N*C*H*W);
    DevBuf dx(N*C*H*W*4), dh(fhn*4), db(N*C*H*W*4); dx.up(X.data());
    aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx.p),*th=mk({N,C1,H,W,F},ACL_FLOAT,dh.p),*tb=mk({N,C,H,W},ACL_FLOAT,db.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataNCHW2NC1HWC0GetWorkspaceSize(tx,th,w,e);},aclnnTransDataNCHW2NC1HWC0);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataNC1HWC0toNCHWGetWorkspaceSize(th,tb,w,e);},aclnnTransDataNC1HWC0toNCHW);
    dh.down(hd.data()); db.down(back.data());
    int64_t bad=0;
    for(int64_t n=0;n<N;n++)for(int64_t c=0;c<C;c++)for(int64_t h=0;h<H;h++)for(int64_t w=0;w<W;w++){
        int64_t c1=c/F,c0=c%F, off=((((n*C1+c1)*H+h)*W+w)*F)+c0, nd=((n*C+c)*H+h)*W+w;
        if(hd[off]!=X[nd]) bad++;
        if(back[nd]!=X[nd]) bad++;
    }
    char nm[48]; snprintf(nm,48,"5HD<->NCHW [%lld,%lld,%lld,%lld]",(long long)N,(long long)C,(long long)H,(long long)W);
    report(nm,bad);
    aclDestroyTensor(tx);aclDestroyTensor(th);aclDestroyTensor(tb);
}
// ND <-> FRACTAL_Z layout + round-trip
static void t_fz(int64_t K,int64_t N){
    const int F=16; int64_t K1=(K+F-1)/F, N1=(N+F-1)/F, fzn=K1*N1*F*F;
    auto X=iota(K*N); std::vector<float> fz(fzn), back(K*N);
    DevBuf dx(K*N*4), df(fzn*4), db(K*N*4); dx.up(X.data());
    aclTensor*tx=mk({K,N},ACL_FLOAT,dx.p),*tf=mk({K1,N1,F,F},ACL_FLOAT,df.p),*tb=mk({K,N},ACL_FLOAT,db.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataND2FZGetWorkspaceSize(tx,tf,w,e);},aclnnTransDataND2FZ);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTransDataFZ2NDGetWorkspaceSize(tf,tb,w,e);},aclnnTransDataFZ2ND);
    df.down(fz.data()); db.down(back.data());
    int64_t bad=0;
    for(int64_t k=0;k<K;k++)for(int64_t n=0;n<N;n++){
        int64_t k1=k/F,k0=k%F,n1=n/F,n0=n%F, off=((k1*N1+n1)*F+k0)*F+n0;
        if(fz[off]!=X[k*N+n]) bad++;
        if(back[k*N+n]!=X[k*N+n]) bad++;
    }
    char nm[48]; snprintf(nm,48,"FZ<->ND [%lld,%lld]",(long long)K,(long long)N);
    report(nm,bad);
    aclDestroyTensor(tx);aclDestroyTensor(tf);aclDestroyTensor(tb);
}

// Embedding + EmbeddingDenseBackward
static void t_embed(int64_t V,int64_t D,int64_t L){
    auto W=iota(V*D); std::vector<int64_t> ids(L); for(auto&x:ids)x=rand()%V;
    auto grad=std::vector<float>(L*D); for(int64_t i=0;i<L*D;i++)grad[i]=(float)((i%13)-6);
    std::vector<float> ho(L*D),hgw(V*D);
    DevBuf dw(V*D*4),di(L*8),dout(L*D*4),dg(L*D*4),dgw(V*D*4); dw.up(W.data()); di.up(ids.data()); dg.up(grad.data());
    aclTensor*tw=mk({V,D},ACL_FLOAT,dw.p),*tid=mk({L},ACL_INT64,di.p),*to=mk({L,D},ACL_FLOAT,dout.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(tw,tid,to,w,e);},aclnnEmbedding);
    dout.down(ho.data());
    int64_t bad=0; for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++)if(ho[l*D+d]!=W[ids[l]*D+d])bad++;
    report("Embedding",bad);
    aclTensor*tg=mk({L,D},ACL_FLOAT,dg.p),*tgw=mk({V,D},ACL_FLOAT,dgw.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEmbeddingDenseBackwardGetWorkspaceSize(tg,tid,tgw,w,e);},aclnnEmbeddingDenseBackward);
    dgw.down(hgw.data());
    std::vector<double> gw(V*D,0); for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++)gw[ids[l]*D+d]+=grad[l*D+d];
    int64_t bad2=0; for(int64_t i=0;i<V*D;i++)if(std::fabs(hgw[i]-gw[i])>1e-4)bad2++;
    report("EmbeddingDenseBackward",bad2);
    aclDestroyTensor(tw);aclDestroyTensor(tid);aclDestroyTensor(to);aclDestroyTensor(tg);aclDestroyTensor(tgw);
}

// Index operator completeness coverage
static void t_index50(){
    // Arange
    { int64_t n=10; std::vector<float> h(n); DevBuf d(n*4); aclTensor*t=mk({n},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnArangeGetWorkspaceSize(2.0,7.0,0.5,t,w,e);},aclnnArange); d.down(h.data());
      int64_t bad=0; for(int64_t i=0;i<n;i++)if(h[i]!=2.0f+i*0.5f)bad++; report("Arange",bad); aclDestroyTensor(t); }
    // OneHot
    { int64_t L=5,C=4; int64_t ids[5]={2,0,3,1,2}; std::vector<float> h(L*C); DevBuf di(L*8),dz(L*C*4); di.up(ids);
      aclTensor*tid=mk({L},ACL_INT64,di.p),*to=mk({L,C},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnOneHotGetWorkspaceSize(tid,C,1.0,0.0,to,w,e);},aclnnOneHot); dz.down(h.data());
      int64_t bad=0; for(int64_t l=0;l<L;l++)for(int64_t c=0;c<C;c++)if(h[l*C+c]!=(c==ids[l]?1.f:0.f))bad++; report("OneHot",bad); aclDestroyTensor(tid);aclDestroyTensor(to); }
    // Flip dim1
    { int64_t M=3,N=4; auto in=iota(M*N); std::vector<float> h(M*N); DevBuf da(M*N*4),dz(M*N*4); da.up(in.data());
      aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFlipGetWorkspaceSize(ta,1,tz,w,e);},aclnnFlip); dz.down(h.data());
      int64_t bad=0; for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++)if(h[i*N+j]!=in[i*N+(N-1-j)])bad++; report("Flip dim1",bad); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // Roll dim1 shift 1
    { int64_t M=3,N=4; auto in=iota(M*N); std::vector<float> h(M*N); DevBuf da(M*N*4),dz(M*N*4); da.up(in.data());
      aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRollGetWorkspaceSize(ta,1,1,tz,w,e);},aclnnRoll); dz.down(h.data());
      int64_t bad=0; for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++)if(h[i*N+j]!=in[i*N+((j-1+N)%N)])bad++; report("Roll dim1 +1",bad); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // RepeatInterleave 1D x3
    { int64_t n=6,r=3; auto in=iota(n); std::vector<float> h(n*r); DevBuf da(n*4),dz(n*r*4); da.up(in.data());
      aclTensor*ta=mk({n},ACL_FLOAT,da.p),*tz=mk({n*r},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatInterleaveGetWorkspaceSize(ta,r,tz,w,e);},aclnnRepeatInterleave); dz.down(h.data());
      int64_t bad=0; for(int64_t i=0;i<n*r;i++)if(h[i]!=in[i/r])bad++; report("RepeatInterleave x3",bad); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // Expand [1,4]->[3,4]
    { int64_t N=4,M=3; auto in=iota(N); std::vector<float> h(M*N); DevBuf da(N*4),dz(M*N*4); da.up(in.data());
      aclTensor*ta=mk({1,N},ACL_FLOAT,da.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnExpandGetWorkspaceSize(ta,tz,w,e);},aclnnExpand); dz.down(h.data());
      int64_t bad=0; for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++)if(h[i*N+j]!=in[j])bad++; report("Expand [1,4]->[3,4]",bad); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // ScatterUpdate dim0
    { int64_t V=5,D=3,L=2; auto self=iota(V*D); int64_t idx[2]={1,3}; std::vector<float> src(L*D); for(int64_t i=0;i<L*D;i++)src[i]=100+i; std::vector<float> h(V*D);
      DevBuf ds(V*D*4),di(L*8),dsrc(L*D*4),dz(V*D*4); ds.up(self.data()); di.up(idx); dsrc.up(src.data());
      aclTensor*tself=mk({V,D},ACL_FLOAT,ds.p),*tid=mk({L},ACL_INT64,di.p),*tsrc=mk({L,D},ACL_FLOAT,dsrc.p),*to=mk({V,D},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterUpdateGetWorkspaceSize(tself,tid,tsrc,to,w,e);},aclnnScatterUpdate); dz.down(h.data());
      std::vector<float> ref=self; for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++)ref[idx[l]*D+d]=src[l*D+d];
      int64_t bad=0; for(int64_t i=0;i<V*D;i++)if(h[i]!=ref[i])bad++; report("ScatterUpdate dim0",bad); aclDestroyTensor(tself);aclDestroyTensor(tid);aclDestroyTensor(tsrc);aclDestroyTensor(to); }
    // ScatterAdd dim0 with duplicate indices (idx={1,3,1}) -> row 1 accumulates two src contributions
    { int64_t V=5,D=3,L=3; auto self=iota(V*D); int64_t idx[3]={1,3,1}; std::vector<float> src(L*D); for(int64_t i=0;i<L*D;i++)src[i]=(float)(100+i); std::vector<float> h(V*D);
      DevBuf ds(V*D*4),di(L*8),dsrc(L*D*4),dz(V*D*4); ds.up(self.data()); di.up(idx); dsrc.up(src.data());
      aclTensor*tself=mk({V,D},ACL_FLOAT,ds.p),*tid=mk({L},ACL_INT64,di.p),*tsrc=mk({L,D},ACL_FLOAT,dsrc.p),*to=mk({V,D},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterAddGetWorkspaceSize(tself,tid,tsrc,to,w,e);},aclnnScatterAdd); dz.down(h.data());
      std::vector<float> ref=self; for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++)ref[idx[l]*D+d]+=src[l*D+d];
      int64_t bad=0; for(int64_t i=0;i<V*D;i++)if(std::fabs(h[i]-ref[i])>1e-3f)bad++; report("ScatterAdd dim0 dup-idx",bad); aclDestroyTensor(tself);aclDestroyTensor(tid);aclDestroyTensor(tsrc);aclDestroyTensor(to); }
    // MaskedSelect
    { int64_t n=10; auto x=iota(n); std::vector<uint8_t> m(n); for(int64_t i=0;i<n;i++)m[i]=(i%3==0); std::vector<float> h(n); int64_t cnt;
      DevBuf dx(n*4),dm(n),dz(n*4),dc(8); dx.up(x.data()); dm.up(m.data());
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tm=mk({n},ACL_BOOL,dm.p),*to=mk({n},ACL_FLOAT,dz.p),*tc=mk({1},ACL_INT64,dc.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedSelectGetWorkspaceSize(tx,tm,to,tc,w,e);},aclnnMaskedSelect); dz.down(h.data()); dc.down(&cnt);
      int64_t pos=0,bad=0; for(int64_t i=0;i<n;i++)if(m[i]){if(h[pos]!=x[i])bad++;pos++;} if(cnt!=pos)bad++; report("MaskedSelect",bad); aclDestroyTensor(tx);aclDestroyTensor(tm);aclDestroyTensor(to);aclDestroyTensor(tc); }
    // Nonzero
    { int64_t n=8; float xv[8]={0,3,0,0,5,1,0,2}; std::vector<int64_t> h(n); int64_t cnt;
      DevBuf dx(n*4),dz(n*8),dc(8); dx.up(xv);
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*to=mk({n},ACL_INT64,dz.p),*tc=mk({1},ACL_INT64,dc.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNonzeroGetWorkspaceSize(tx,to,tc,w,e);},aclnnNonzero); dz.down(h.data()); dc.down(&cnt);
      int64_t pos=0,bad=0; for(int64_t i=0;i<n;i++)if(xv[i]!=0){if(h[pos]!=i)bad++;pos++;} if(cnt!=pos)bad++; report("Nonzero",bad); aclDestroyTensor(tx);aclDestroyTensor(to);aclDestroyTensor(tc); }
}

int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    CHECK(aclrtCreateStream(&g_stream));
    t_permute(); t_slice(); t_strided_slice(); t_tile(); t_pad(); t_gather(); t_gather_v2(); t_cat(); t_cat_list(); t_stack(); t_gmm_weightlist(); t_split(); t_permute_i32();
    t_embed(20, 8, 12);
    t_index50();
    t_nz(32, 48);           // evenly divisible
    t_nz(20, 37);           // non-divisible (with padding)
    t_5hd(2, 32, 4, 5);     // NC1HWC0 evenly divisible
    t_5hd(1, 20, 3, 3);     // NC1HWC0 with channel padding
    t_fz(48, 32);           // FRACTAL_Z evenly divisible
    t_fz(20, 37);           // FRACTAL_Z with padding
    CHECK(aclrtDestroyStream(g_stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
