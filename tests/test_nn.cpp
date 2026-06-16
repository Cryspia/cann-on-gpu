// Cross-check: softmax (cuDNN) / layernorm (kernel) / flash-attention (naive batched kernel + stable softmax).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r = (int)(x); if (__r != 0) { \
    printf("[FATAL] %s:%d ret=%d\n", __FILE__, __LINE__, __r); exit(1); } } while (0)

static aclrtStream g_stream;
static int g_pass = 0, g_fail = 0;

static void *up(const void *h, size_t b) {
    void *d; CHECK(aclrtMalloc(&d, b, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMemcpy(d, b, h, b, ACL_MEMCPY_HOST_TO_DEVICE));
    return d;
}
static aclTensor *mk(std::vector<int64_t> dims, aclDataType dt, void *p) {
    return aclCreateTensor(dims.data(), dims.size(), dt, nullptr, 0, ACL_FORMAT_ND, dims.data(), dims.size(), p);
}
static std::vector<float> randv(int64_t n, float lo = -1, float hi = 1) {
    std::vector<float> v(n);
    for (auto &x : v) x = lo + (hi - lo) * (rand() / (float)RAND_MAX);
    return v;
}
static void report(const char *name, double maxrel, double tol) {
    bool ok = maxrel <= tol;
    (ok ? g_pass : g_fail)++;
    printf("%-32s maxrel=%.2e tol=%.0e %s\n", name, maxrel, tol, ok ? "PASS" : "FAIL");
}
static double rel(double got, double ref, double atol = 1e-9) { return std::fabs(got - ref) / (std::fabs(ref) + atol); }
template <typename GetWs>
static void exec2(GetWs getws, aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream)) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr; CHECK(getws(&ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(run(wsp, ws, ex, g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if (wsp) aclrtFree(wsp);
}
static uint16_t f2h(float f){ uint32_t x; __builtin_memcpy(&x,&f,4); uint32_t s=(x>>16)&0x8000; int32_t e=((x>>23)&0xFF)-127+15; uint32_t m=x&0x7FFFFF;
    if(e<=0)return (uint16_t)s; if(e>=31)return (uint16_t)(s|0x7C00); uint32_t r=m&0x1FFF,h=s|(e<<10)|(m>>13); if(r>0x1000||(r==0x1000&&(h&1)))h++; return (uint16_t)h; }
static float h2f(uint16_t h){ uint32_t s=(h&0x8000)<<16,e=(h>>10)&0x1F,m=h&0x3FF,x; if(e==0){float f=m*0x1p-24f;__builtin_memcpy(&x,&f,4);x|=s;}
    else if(e==31)x=s|0x7F800000|(m<<13); else x=s|((e-15+127)<<23)|(m<<13); float f;__builtin_memcpy(&f,&x,4); return f; }
static uint16_t f2bf(float f){ uint32_t x; __builtin_memcpy(&x,&f,4); return (uint16_t)((x>>16)+((x>>15)&1)); }
static float bf2f(uint16_t b){ uint32_t x=((uint32_t)b)<<16; float f; __builtin_memcpy(&f,&x,4); return f; }

// General attention cross-check (GQA: Nkv<=Nq evenly divides; fp16 I/O; maskMode: 0 none / 1 per-batch [B,Sq,Skv])
static void t_attn2(const char *name, bool fp16, int64_t B, int64_t Nq, int64_t Nkv, int64_t Sq, int64_t Skv, int64_t D,
                    bool causal, int maskMode, double tol) {
    auto Q=randv(B*Nq*Sq*D), K=randv(B*Nkv*Skv*D), V=randv(B*Nkv*Skv*D);
    double scale=1.0/std::sqrt((double)D); int64_t off=Skv-Sq;
    std::vector<uint8_t> mask; if(maskMode==1){ mask.resize(B*Sq*Skv); for(auto&m:mask)m=rand()&1; }
    size_t esz=fp16?2:4;
    std::vector<uint8_t> qb(Q.size()*esz),kb(K.size()*esz),vb(V.size()*esz);
    auto pack=[&](const std::vector<float>&src,std::vector<uint8_t>&dst){ for(size_t i=0;i<src.size();i++) if(fp16){uint16_t h=f2h(src[i]);__builtin_memcpy(&dst[i*2],&h,2);} else __builtin_memcpy(&dst[i*4],&src[i],4); };
    pack(Q,qb);pack(K,kb);pack(V,vb);
    void*dq=up(qb.data(),qb.size()),*dk=up(kb.data(),kb.size()),*dv=up(vb.data(),vb.size()),*doO,*dm=nullptr;
    CHECK(aclrtMalloc(&doO,Q.size()*esz,ACL_MEM_MALLOC_HUGE_FIRST)); if(maskMode==1)dm=up(mask.data(),mask.size());
    aclDataType dt=fp16?ACL_FLOAT16:ACL_FLOAT;
    aclTensor*tq=mk({B,Nq,Sq,D},dt,dq),*tk=mk({B,Nkv,Skv,D},dt,dk),*tv=mk({B,Nkv,Skv,D},dt,dv),*to=mk({B,Nq,Sq,D},dt,doO);
    aclTensor*tm=maskMode==1?mk({B,Sq,Skv},ACL_BOOL,dm):nullptr;
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    CHECK(aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,tm,scale,Nq,causal,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnFlashAttentionScore(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> ob(Q.size()*esz); CHECK(aclrtMemcpy(ob.data(),ob.size(),doO,ob.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    auto getO=[&](int64_t i){ return fp16? (double)h2f(((uint16_t*)ob.data())[i]) : (double)((float*)ob.data())[i]; };
    auto qv=[&](int64_t i){ return fp16? (double)h2f(f2h(Q[i])) : (double)Q[i]; };  // after fp16 quantization
    auto kv=[&](int64_t i){ return fp16? (double)h2f(f2h(K[i])) : (double)K[i]; };
    auto vv=[&](int64_t i){ return fp16? (double)h2f(f2h(V[i])) : (double)V[i]; };
    double maxrel=0; std::vector<double> s(Skv);
    for(int64_t bi=0;bi<B;bi++)for(int64_t h=0;h<Nq;h++){
        int64_t kvh=h/(Nq/Nkv), qf=bi*Nq+h, kf=bi*Nkv+kvh;
        for(int64_t i=0;i<Sq;i++){
            double mx=-1e30;
            for(int64_t j=0;j<Skv;j++){ double d=0; for(int64_t t=0;t<D;t++) d+=qv((qf*Sq+i)*D+t)*kv((kf*Skv+j)*D+t); d*=scale;
                bool blk=(causal&&j>i+off)||(maskMode==1&&mask[(bi*Sq+i)*Skv+j]); if(blk)d=-1e30; s[j]=d; mx=std::max(mx,d); }
            double sum=0; for(int64_t j=0;j<Skv;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
            for(int64_t t=0;t<D;t++){ double o=0; for(int64_t j=0;j<Skv;j++) o+=s[j]/sum*vv((kf*Skv+j)*D+t);
                maxrel=std::max(maxrel,rel(getO((qf*Sq+i)*D+t),o,1e-3)); }
        }
    }
    report(name,maxrel,tol);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to); if(tm)aclDestroyTensor(tm);
    if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(doO); if(dm)aclrtFree(dm);
}

static void t_softmax(int64_t N, int64_t L, double tol) {
    auto x = randv(N * L, -3, 3);
    std::vector<float> y(N * L);
    void *dx = up(x.data(), x.size() * 4), *dy;
    CHECK(aclrtMalloc(&dy, y.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *tx = mk({N, L}, ACL_FLOAT, dx), *ty = mk({N, L}, ACL_FLOAT, dy);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnSoftmaxGetWorkspaceSize(tx, -1, ty, &ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnSoftmax(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(), y.size() * 4, dy, y.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t i = 0; i < N; i++) {
        double mx = -1e30; for (int64_t j = 0; j < L; j++) mx = std::max(mx, (double)x[i*L+j]);
        double sum = 0; for (int64_t j = 0; j < L; j++) sum += std::exp(x[i*L+j] - mx);
        for (int64_t j = 0; j < L; j++) maxrel = std::max(maxrel, rel(y[i*L+j], std::exp(x[i*L+j]-mx)/sum, 1e-7));
    }
    char nm[48]; snprintf(nm, sizeof nm, "Softmax %ldx%ld", (long)N, (long)L);
    report(nm, maxrel, tol);
    aclDestroyTensor(tx); aclDestroyTensor(ty); if (wsp) aclrtFree(wsp);
    aclrtFree(dx); aclrtFree(dy);
}

static void t_layernorm(int64_t rows, int64_t D, bool affine, double tol) {
    auto x = randv(rows * D, -2, 2), g = randv(D, 0.5, 1.5), b = randv(D, -0.5, 0.5);
    std::vector<float> y(rows * D);
    const double eps = 1e-5;
    void *dx = up(x.data(), x.size() * 4), *dy, *dg = nullptr, *db = nullptr;
    CHECK(aclrtMalloc(&dy, y.size() * 4, ACL_MEM_MALLOC_HUGE_FIRST));
    if (affine) { dg = up(g.data(), D * 4); db = up(b.data(), D * 4); }
    aclTensor *tx = mk({rows, D}, ACL_FLOAT, dx), *ty = mk({rows, D}, ACL_FLOAT, dy);
    aclTensor *tg = affine ? mk({D}, ACL_FLOAT, dg) : nullptr, *tb = affine ? mk({D}, ACL_FLOAT, db) : nullptr;
    int64_t ns[1] = {D}; aclIntArray *nsh = aclCreateIntArray(ns, 1);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnLayerNormGetWorkspaceSize(tx, nsh, tg, tb, eps, ty, nullptr, nullptr, &ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnLayerNorm(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(), y.size() * 4, dy, y.size() * 4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    for (int64_t r = 0; r < rows; r++) {
        double mean = 0; for (int64_t j = 0; j < D; j++) mean += x[r*D+j]; mean /= D;
        double var = 0; for (int64_t j = 0; j < D; j++) { double d = x[r*D+j]-mean; var += d*d; } var /= D;
        double rstd = 1.0 / std::sqrt(var + eps);
        for (int64_t j = 0; j < D; j++) {
            double ref = (x[r*D+j]-mean)*rstd*(affine?g[j]:1.0) + (affine?b[j]:0.0);
            maxrel = std::max(maxrel, rel(y[r*D+j], ref, 1e-6));
        }
    }
    char nm[56]; snprintf(nm, sizeof nm, "LayerNorm %ldx%ld%s", (long)rows, (long)D, affine ? " affine" : "");
    report(nm, maxrel, tol);
    aclDestroyIntArray(nsh); aclDestroyTensor(tx); aclDestroyTensor(ty);
    if (tg) aclDestroyTensor(tg); if (tb) aclDestroyTensor(tb);
    if (wsp) aclrtFree(wsp); aclrtFree(dx); aclrtFree(dy); if (dg) aclrtFree(dg); if (db) aclrtFree(db);
}

static void t_attention(int64_t B, int64_t Nh, int64_t Sq, int64_t Skv, int64_t D, bool causal, double tol) {
    auto Q = randv(B*Nh*Sq*D), K = randv(B*Nh*Skv*D), V = randv(B*Nh*Skv*D);
    std::vector<float> O(B*Nh*Sq*D);
    double scale = 1.0 / std::sqrt((double)D);
    void *dq = up(Q.data(), Q.size()*4), *dk = up(K.data(), K.size()*4), *dv = up(V.data(), V.size()*4), *doO;
    CHECK(aclrtMalloc(&doO, O.size()*4, ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor *tq = mk({B,Nh,Sq,D}, ACL_FLOAT, dq), *tk = mk({B,Nh,Skv,D}, ACL_FLOAT, dk),
              *tv = mk({B,Nh,Skv,D}, ACL_FLOAT, dv), *to = mk({B,Nh,Sq,D}, ACL_FLOAT, doO);
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    CHECK(aclnnFlashAttentionScoreGetWorkspaceSize(tq, tk, tv, nullptr, scale, Nh, causal, to, &ws, &ex));
    void *wsp = nullptr; if (ws) CHECK(aclrtMalloc(&wsp, ws, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclnnFlashAttentionScore(wsp, ws, ex, g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(O.data(), O.size()*4, doO, O.size()*4, ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel = 0;
    int64_t off = Skv - Sq;
    std::vector<double> s(Skv);
    for (int64_t bn = 0; bn < B*Nh; bn++)
        for (int64_t i = 0; i < Sq; i++) {
            const float *q = Q.data() + (bn*Sq+i)*D;
            double mx = -1e30;
            for (int64_t j = 0; j < Skv; j++) {
                double d = 0; const float *kk = K.data() + (bn*Skv+j)*D;
                for (int64_t t = 0; t < D; t++) d += (double)q[t]*kk[t];
                d *= scale;
                if (causal && j > i + off) d = -1e30;
                s[j] = d; mx = std::max(mx, d);
            }
            double sum = 0; for (int64_t j = 0; j < Skv; j++) { s[j] = std::exp(s[j]-mx); sum += s[j]; }
            for (int64_t t = 0; t < D; t++) {
                double o = 0;
                for (int64_t j = 0; j < Skv; j++) o += s[j]/sum * V[(bn*Skv+j)*D+t];
                maxrel = std::max(maxrel, rel(O[(bn*Sq+i)*D+t], o, 1e-5));
            }
        }
    char nm[72]; snprintf(nm, sizeof nm, "Attn B%ldN%ldSq%ldSkv%ldD%ld%s",
                          (long)B,(long)Nh,(long)Sq,(long)Skv,(long)D, causal?" causal":"");
    report(nm, maxrel, tol);
    aclDestroyTensor(tq); aclDestroyTensor(tk); aclDestroyTensor(tv); aclDestroyTensor(to);
    if (wsp) aclrtFree(wsp); aclrtFree(dq); aclrtFree(dk); aclrtFree(dv); aclrtFree(doO);
}

// Arbitrary-dim softmax / logsoftmax cross-check (fp32)
static void t_softmaxdim(const char *name, std::vector<int64_t> shape, int dim, bool log) {
    int rank=shape.size(); int64_t n=1; for(auto d:shape)n*=d;
    auto X=randv(n,-2,2); std::vector<float> hz(n);
    int64_t outer=1; for(int i=0;i<dim;i++)outer*=shape[i]; int64_t L=shape[dim]; int64_t inner=1; for(int i=dim+1;i<rank;i++)inner*=shape[i];
    void*dx=up(X.data(),n*4),*dy; CHECK(aclrtMalloc(&dy,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk(shape,ACL_FLOAT,dx),*ty=mk(shape,ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if(log)CHECK(aclnnLogSoftmaxGetWorkspaceSize(tx,dim,ty,&ws,&ex)); else CHECK(aclnnSoftmaxGetWorkspaceSize(tx,dim,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(log)CHECK(aclnnLogSoftmax(wsp,ws,ex,g_stream)); else CHECK(aclnnSoftmax(wsp,ws,ex,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream)); CHECK(aclrtMemcpy(hz.data(),n*4,dy,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t o=0;o<outer;o++)for(int64_t in=0;in<inner;in++){ int64_t base=o*L*inner+in;
        double mx=-1e30; for(int64_t l=0;l<L;l++)mx=std::max(mx,(double)X[base+l*inner]);
        double sum=0; for(int64_t l=0;l<L;l++)sum+=std::exp(X[base+l*inner]-mx);
        for(int64_t l=0;l<L;l++){ double ref=log? (X[base+l*inner]-mx-std::log(sum)) : std::exp(X[base+l*inner]-mx)/sum;
            maxrel=std::max(maxrel,rel(hz[base+l*inner],ref,1e-6)); } }
    report(name,maxrel,1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dy);
}
static void t_rmsnorm(int64_t rows,int64_t D){
    auto X=randv(rows*D,-2,2),G=randv(D,0.5,1.5); std::vector<float> hz(rows*D); double eps=1e-6;
    void*dx=up(X.data(),X.size()*4),*dg=up(G.data(),G.size()*4),*dy; CHECK(aclrtMalloc(&dy,X.size()*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({rows,D},ACL_FLOAT,dx),*tg=mk({D},ACL_FLOAT,dg),*ty=mk({rows,D},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnRmsNormGetWorkspaceSize(tx,tg,eps,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnRmsNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hz.data(),hz.size()*4,dy,hz.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t r=0;r<rows;r++){ double ss=0; for(int64_t d=0;d<D;d++)ss+=(double)X[r*D+d]*X[r*D+d]; double inv=1.0/std::sqrt(ss/D+eps);
        for(int64_t d=0;d<D;d++){ double ref=X[r*D+d]*inv*G[d]; maxrel=std::max(maxrel,rel(hz[r*D+d],ref,1e-5)); } }
    report("RMSNorm",maxrel,1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dg);aclrtFree(dy);
}
static void t_groupnorm(int64_t N,int64_t C,int64_t H,int64_t W,int64_t G){
    int64_t HW=H*W,total=N*C*HW; auto X=randv(total,-2,2),gm=randv(C,0.5,1.5),bt=randv(C,-0.5,0.5);
    std::vector<float> hz(total); double eps=1e-5;
    void*dx=up(X.data(),total*4),*dg=up(gm.data(),C*4),*db=up(bt.data(),C*4),*dy; CHECK(aclrtMalloc(&dy,total*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx),*tg=mk({C},ACL_FLOAT,dg),*tb=mk({C},ACL_FLOAT,db),*ty=mk({N,C,H,W},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnGroupNormGetWorkspaceSize(tx,tg,tb,G,eps,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnGroupNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hz.data(),total*4,dy,total*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0; int64_t Cg=C/G;
    for(int64_t n=0;n<N;n++)for(int64_t grp=0;grp<G;grp++){ double sum=0,sq=0; int64_t cnt=Cg*HW;
        for(int64_t cc=0;cc<Cg;cc++){int64_t c=grp*Cg+cc; for(int64_t hw=0;hw<HW;hw++){double v=X[(n*C+c)*HW+hw];sum+=v;sq+=v*v;}}
        double mean=sum/cnt,var=sq/cnt-mean*mean,inv=1.0/std::sqrt(var+eps);
        for(int64_t cc=0;cc<Cg;cc++){int64_t c=grp*Cg+cc; for(int64_t hw=0;hw<HW;hw++){double ref=(X[(n*C+c)*HW+hw]-mean)*inv*gm[c]+bt[c];
            maxrel=std::max(maxrel,rel(hz[(n*C+c)*HW+hw],ref,1e-4));}} }
    report("GroupNorm",maxrel,1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dg);aclrtFree(db);aclrtFree(dy);
}
static void t_batchnorm(int64_t N,int64_t C,int64_t H,int64_t W){
    int64_t HW=H*W,total=N*C*HW; auto X=randv(total,-2,2),gm=randv(C,0.5,1.5),bt=randv(C,-0.5,0.5),mn=randv(C,-1,1),vr=randv(C,0.5,2);
    std::vector<float> hz(total); double eps=1e-5;
    void*dx=up(X.data(),total*4),*dg=up(gm.data(),C*4),*db=up(bt.data(),C*4),*dm=up(mn.data(),C*4),*dv=up(vr.data(),C*4),*dy;
    CHECK(aclrtMalloc(&dy,total*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx),*tg=mk({C},ACL_FLOAT,dg),*tb=mk({C},ACL_FLOAT,db),*tm=mk({C},ACL_FLOAT,dm),*tv=mk({C},ACL_FLOAT,dv),*ty=mk({N,C,H,W},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnBatchNormGetWorkspaceSize(tx,tg,tb,tm,tv,eps,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnBatchNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hz.data(),total*4,dy,total*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t i=0;i<total;i++){ int64_t c=(i/HW)%C; double ref=(X[i]-mn[c])/std::sqrt(vr[c]+eps)*gm[c]+bt[c]; maxrel=std::max(maxrel,rel(hz[i],ref,1e-4)); }
    report("BatchNorm infer",maxrel,1e-4);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dg);aclrtFree(db);aclrtFree(dm);aclrtFree(dv);aclrtFree(dy);
}

// fp4 input attention: Q/K/V in fp4 e2m1, unpacked to fp16, output fp16
static void t_attn_fp4(int64_t B,int64_t Nh,int64_t Sq,int64_t Skv,int64_t D){
    auto cast=[&](void*src,aclDataType sdt,int64_t ne,void*dst,aclDataType ddt){
        int64_t d[1]={ne}; aclTensor*s=aclCreateTensor(d,1,sdt,nullptr,0,ACL_FORMAT_ND,d,1,src);
        aclTensor*o=aclCreateTensor(d,1,ddt,nullptr,0,ACL_FORMAT_ND,d,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream)); aclDestroyTensor(s);aclDestroyTensor(o); };
    int64_t qn=B*Nh*Sq*D, kn=B*Nh*Skv*D; double scale=1.0/std::sqrt((double)D);
    auto Q=randv(qn,-1.5,1.5),Kk=randv(kn,-1.5,1.5),V=randv(kn,-1.5,1.5);
    void *dQf=up(Q.data(),qn*4),*dKf=up(Kk.data(),kn*4),*dVf=up(V.data(),kn*4);
    void *dQ4,*dK4,*dV4,*dQb,*dKb,*dVb,*dO;
    CHECK(aclrtMalloc(&dQ4,(qn+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dK4,(kn+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dV4,(kn+1)/2,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dQb,qn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dKb,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dVb,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dO,qn*2,ACL_MEM_MALLOC_HUGE_FIRST));
    cast(dQf,ACL_FLOAT,qn,dQ4,ACL_FLOAT4_E2M1);cast(dKf,ACL_FLOAT,kn,dK4,ACL_FLOAT4_E2M1);cast(dVf,ACL_FLOAT,kn,dV4,ACL_FLOAT4_E2M1);
    cast(dQ4,ACL_FLOAT4_E2M1,qn,dQb,ACL_FLOAT);cast(dK4,ACL_FLOAT4_E2M1,kn,dKb,ACL_FLOAT);cast(dV4,ACL_FLOAT4_E2M1,kn,dVb,ACL_FLOAT);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Qb(qn),Kb(kn),Vb(kn); CHECK(aclrtMemcpy(Qb.data(),qn*4,dQb,qn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(Kb.data(),kn*4,dKb,kn*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(Vb.data(),kn*4,dVb,kn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    aclTensor*tq=mk({B,Nh,Sq,D},ACL_FLOAT4_E2M1,dQ4),*tk=mk({B,Nh,Skv,D},ACL_FLOAT4_E2M1,dK4),*tv=mk({B,Nh,Skv,D},ACL_FLOAT4_E2M1,dV4),*to=mk({B,Nh,Sq,D},ACL_FLOAT16,dO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,false,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnFlashAttentionScore(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> ob(qn); CHECK(aclrtMemcpy(ob.data(),qn*2,dO,qn*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0; std::vector<double> s(Skv);
    for(int64_t bn=0;bn<B*Nh;bn++)for(int64_t i=0;i<Sq;i++){ double mx=-1e30;
        for(int64_t j=0;j<Skv;j++){double d=0;for(int64_t t=0;t<D;t++)d+=(double)Qb[(bn*Sq+i)*D+t]*Kb[(bn*Skv+j)*D+t]; d*=scale; s[j]=d; mx=std::max(mx,d);}
        double sum=0; for(int64_t j=0;j<Skv;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
        for(int64_t t=0;t<D;t++){double o=0;for(int64_t j=0;j<Skv;j++)o+=s[j]/sum*Vb[(bn*Skv+j)*D+t]; maxrel=std::max(maxrel,rel(h2f(ob[(bn*Sq+i)*D+t]),o,1e-2));}}
    report("Attn fp4 input",maxrel,5e-3);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to); if(wsp)aclrtFree(wsp);
    aclrtFree(dQf);aclrtFree(dKf);aclrtFree(dVf);aclrtFree(dQ4);aclrtFree(dK4);aclrtFree(dV4);aclrtFree(dQb);aclrtFree(dKb);aclrtFree(dVb);aclrtFree(dO);
}

// fp8 flash attention: fp8 e4m3 input Q/K/V -> decode to fp16 -> fp16 flash, output fp16. CPU reference uses fp8 cast-back values.
static void t_attn_fp8(int64_t B,int64_t Nh,int64_t Sq,int64_t Skv,int64_t D){
    auto cast=[&](void*src,aclDataType sdt,int64_t ne,void*dst,aclDataType ddt){
        int64_t d[1]={ne}; aclTensor*s=aclCreateTensor(d,1,sdt,nullptr,0,ACL_FORMAT_ND,d,1,src);
        aclTensor*o=aclCreateTensor(d,1,ddt,nullptr,0,ACL_FORMAT_ND,d,1,dst);
        uint64_t w; aclOpExecutor*e; CHECK(aclnnCastGetWorkspaceSize(s,ddt,o,&w,&e)); CHECK(aclnnCast(nullptr,w,e,g_stream)); aclDestroyTensor(s);aclDestroyTensor(o); };
    int64_t qn=B*Nh*Sq*D, kn=B*Nh*Skv*D; double scale=1.0/std::sqrt((double)D);
    auto Q=randv(qn,-1.5,1.5),Kk=randv(kn,-1.5,1.5),V=randv(kn,-1.5,1.5);
    void *dQf=up(Q.data(),qn*4),*dKf=up(Kk.data(),kn*4),*dVf=up(V.data(),kn*4);
    void *dQ8,*dK8,*dV8,*dQb,*dKb,*dVb,*dO;
    CHECK(aclrtMalloc(&dQ8,qn,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dK8,kn,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dV8,kn,ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK(aclrtMalloc(&dQb,qn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dKb,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dVb,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dO,qn*2,ACL_MEM_MALLOC_HUGE_FIRST));
    cast(dQf,ACL_FLOAT,qn,dQ8,ACL_FLOAT8_E4M3FN);cast(dKf,ACL_FLOAT,kn,dK8,ACL_FLOAT8_E4M3FN);cast(dVf,ACL_FLOAT,kn,dV8,ACL_FLOAT8_E4M3FN);
    cast(dQ8,ACL_FLOAT8_E4M3FN,qn,dQb,ACL_FLOAT);cast(dK8,ACL_FLOAT8_E4M3FN,kn,dKb,ACL_FLOAT);cast(dV8,ACL_FLOAT8_E4M3FN,kn,dVb,ACL_FLOAT);
    CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> Qb(qn),Kb(kn),Vb(kn); CHECK(aclrtMemcpy(Qb.data(),qn*4,dQb,qn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK(aclrtMemcpy(Kb.data(),kn*4,dKb,kn*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(Vb.data(),kn*4,dVb,kn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    aclTensor*tq=mk({B,Nh,Sq,D},ACL_FLOAT8_E4M3FN,dQ8),*tk=mk({B,Nh,Skv,D},ACL_FLOAT8_E4M3FN,dK8),*tv=mk({B,Nh,Skv,D},ACL_FLOAT8_E4M3FN,dV8),*to=mk({B,Nh,Sq,D},ACL_FLOAT16,dO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnFlashAttentionScoreGetWorkspaceSize(tq,tk,tv,nullptr,scale,Nh,false,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnFlashAttentionScore(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint16_t> ob(qn); CHECK(aclrtMemcpy(ob.data(),qn*2,dO,qn*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0; std::vector<double> s(Skv);
    for(int64_t bn=0;bn<B*Nh;bn++)for(int64_t i=0;i<Sq;i++){ double mx=-1e30;
        for(int64_t j=0;j<Skv;j++){double d=0;for(int64_t t=0;t<D;t++)d+=(double)Qb[(bn*Sq+i)*D+t]*Kb[(bn*Skv+j)*D+t]; d*=scale; s[j]=d; mx=std::max(mx,d);}
        double sum=0; for(int64_t j=0;j<Skv;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
        for(int64_t t=0;t<D;t++){double o=0;for(int64_t j=0;j<Skv;j++)o+=s[j]/sum*Vb[(bn*Skv+j)*D+t]; maxrel=std::max(maxrel,rel(h2f(ob[(bn*Sq+i)*D+t]),o,1e-2));}}
    report("Attn fp8(e4m3) input",maxrel,5e-3);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to); if(wsp)aclrtFree(wsp);
    aclrtFree(dQf);aclrtFree(dKf);aclrtFree(dVf);aclrtFree(dQ8);aclrtFree(dK8);aclrtFree(dV8);aclrtFree(dQb);aclrtFree(dKb);aclrtFree(dVb);aclrtFree(dO);
}

// RMSNorm backward (fp32): cross-check dx + dgamma
static void t_rmsnorm_bwd(int64_t rows,int64_t D){
    auto x=randv(rows*D,-2,2),gm=randv(D,0.5,1.5),dy=randv(rows*D,-1,1); double eps=1e-6;
    std::vector<float> hdx(rows*D),hdg(D);
    void*dx_=up(x.data(),x.size()*4),*dg=up(gm.data(),D*4),*ddy=up(dy.data(),dy.size()*4),*odx,*odg;
    CHECK(aclrtMalloc(&odx,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&odg,D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tdy=mk({rows,D},ACL_FLOAT,ddy),*tx=mk({rows,D},ACL_FLOAT,dx_),*tg=mk({D},ACL_FLOAT,dg),*todx=mk({rows,D},ACL_FLOAT,odx),*todg=mk({D},ACL_FLOAT,odg);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnRmsNormBackwardGetWorkspaceSize(tdy,tx,tg,eps,todx,todg,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnRmsNormBackward(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hdx.data(),hdx.size()*4,odx,hdx.size()*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hdg.data(),D*4,odg,D*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dg_ref(D,0); double mx_dx=0,mr_dx=0;
    for(int64_t r=0;r<rows;r++){ double ss=0; for(int64_t d=0;d<D;d++)ss+=(double)x[r*D+d]*x[r*D+d]; double rcp=1.0/std::sqrt(ss/D+eps),r3=rcp*rcp*rcp;
        double dot=0; for(int64_t d=0;d<D;d++)dot+=(double)dy[r*D+d]*gm[d]*x[r*D+d];
        for(int64_t d=0;d<D;d++){ double ref=rcp*(dy[r*D+d]*gm[d])-(r3/D)*x[r*D+d]*dot; mx_dx=std::max(mx_dx,std::fabs(hdx[r*D+d]-ref)); mr_dx=std::max(mr_dx,std::fabs(ref)); dg_ref[d]+=dy[r*D+d]*x[r*D+d]*rcp; } }
    double mx_dg=0,mr_dg=0; for(int64_t d=0;d<D;d++){mx_dg=std::max(mx_dg,std::fabs(hdg[d]-dg_ref[d]));mr_dg=std::max(mr_dg,std::fabs(dg_ref[d]));}
    report("RMSNorm bwd dx",mx_dx/(mr_dx+1e-9),1e-5); report("RMSNorm bwd dgamma",mx_dg/(mr_dg+1e-9),1e-4);
    aclDestroyTensor(tdy);aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(todx);aclDestroyTensor(todg);
    if(wsp)aclrtFree(wsp); aclrtFree(dx_);aclrtFree(dg);aclrtFree(ddy);aclrtFree(odx);aclrtFree(odg);
}
// LayerNorm backward (fp32): cross-check dx + dgamma + dbeta
static void t_layernorm_bwd(int64_t rows,int64_t D){
    auto x=randv(rows*D,-2,2),gm=randv(D,0.5,1.5),dy=randv(rows*D,-1,1); double eps=1e-5;
    std::vector<float> hdx(rows*D),hdg(D),hdb(D);
    void*dx_=up(x.data(),x.size()*4),*dg=up(gm.data(),D*4),*ddy=up(dy.data(),dy.size()*4),*odx,*odg,*odb;
    CHECK(aclrtMalloc(&odx,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&odg,D*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&odb,D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t ns[1]={D}; aclIntArray*nsh=aclCreateIntArray(ns,1);
    aclTensor*tdy=mk({rows,D},ACL_FLOAT,ddy),*tx=mk({rows,D},ACL_FLOAT,dx_),*tg=mk({D},ACL_FLOAT,dg),*todx=mk({rows,D},ACL_FLOAT,odx),*todg=mk({D},ACL_FLOAT,odg),*todb=mk({D},ACL_FLOAT,odb);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnLayerNormBackwardGetWorkspaceSize(tdy,tx,tg,nsh,eps,todx,todg,todb,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnLayerNormBackward(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hdx.data(),hdx.size()*4,odx,hdx.size()*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hdg.data(),D*4,odg,D*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hdb.data(),D*4,odb,D*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dg_ref(D,0),db_ref(D,0); double mx_dx=0,mr_dx=0;
    for(int64_t r=0;r<rows;r++){ double mean=0; for(int64_t d=0;d<D;d++)mean+=x[r*D+d]; mean/=D; double var=0; for(int64_t d=0;d<D;d++){double t=x[r*D+d]-mean;var+=t*t;} var/=D; double rstd=1.0/std::sqrt(var+eps);
        double mg=0,mgx=0; for(int64_t d=0;d<D;d++){double xhat=(x[r*D+d]-mean)*rstd,gi=(double)dy[r*D+d]*gm[d];mg+=gi;mgx+=gi*xhat;} mg/=D;mgx/=D;
        for(int64_t d=0;d<D;d++){double xhat=(x[r*D+d]-mean)*rstd,gi=(double)dy[r*D+d]*gm[d]; double ref=rstd*(gi-mg-xhat*mgx); mx_dx=std::max(mx_dx,std::fabs(hdx[r*D+d]-ref)); mr_dx=std::max(mr_dx,std::fabs(ref)); dg_ref[d]+=dy[r*D+d]*xhat; db_ref[d]+=dy[r*D+d];} }
    double mx_dg=0,mr_dg=0,mx_db=0,mr_db=0; for(int64_t d=0;d<D;d++){mx_dg=std::max(mx_dg,std::fabs(hdg[d]-dg_ref[d]));mr_dg=std::max(mr_dg,std::fabs(dg_ref[d]));mx_db=std::max(mx_db,std::fabs(hdb[d]-db_ref[d]));mr_db=std::max(mr_db,std::fabs(db_ref[d]));}
    report("LayerNorm bwd dx",mx_dx/(mr_dx+1e-9),1e-5); report("LayerNorm bwd dgamma",mx_dg/(mr_dg+1e-9),1e-4); report("LayerNorm bwd dbeta",mx_db/(mr_db+1e-9),1e-4);
    aclDestroyIntArray(nsh);aclDestroyTensor(tdy);aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(todx);aclDestroyTensor(todg);aclDestroyTensor(todb);
    if(wsp)aclrtFree(wsp); aclrtFree(dx_);aclrtFree(dg);aclrtFree(ddy);aclrtFree(odx);aclrtFree(odg);aclrtFree(odb);
}
// BatchNorm training forward (fp32): cross-check y + savedMean + savedInvStd
static void t_batchnorm_train(int64_t N,int64_t C,int64_t H,int64_t W){
    int64_t HW=H*W,total=N*C*HW; auto X=randv(total,-2,2),gm=randv(C,0.5,1.5),bt=randv(C,-0.5,0.5);
    std::vector<float> rm(C,0),rv(C,1),hy(total),sm(C),si(C); double eps=1e-5,mom=0.1;
    void*dx=up(X.data(),total*4),*dg=up(gm.data(),C*4),*db=up(bt.data(),C*4),*drm=up(rm.data(),C*4),*drv=up(rv.data(),C*4),*dy,*dsm,*dsi;
    CHECK(aclrtMalloc(&dy,total*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dsm,C*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&dsi,C*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({N,C,H,W},ACL_FLOAT,dx),*tg=mk({C},ACL_FLOAT,dg),*tb=mk({C},ACL_FLOAT,db),*trm=mk({C},ACL_FLOAT,drm),*trv=mk({C},ACL_FLOAT,drv),*ty=mk({N,C,H,W},ACL_FLOAT,dy),*tsm=mk({C},ACL_FLOAT,dsm),*tsi=mk({C},ACL_FLOAT,dsi);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnBatchNormTrainingGetWorkspaceSize(tx,tg,tb,trm,trv,mom,eps,ty,tsm,tsi,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnBatchNormTraining(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hy.data(),total*4,dy,total*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(sm.data(),C*4,dsm,C*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(si.data(),C*4,dsi,C*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double my=0,mr=0,msm=0,msi=0; int64_t cnt=N*HW;
    for(int64_t c=0;c<C;c++){ double mean=0; for(int64_t n=0;n<N;n++)for(int64_t hw=0;hw<HW;hw++)mean+=X[(n*C+c)*HW+hw]; mean/=cnt;
        double var=0; for(int64_t n=0;n<N;n++)for(int64_t hw=0;hw<HW;hw++){double t=X[(n*C+c)*HW+hw]-mean;var+=t*t;} var/=cnt; double inv=1.0/std::sqrt(var+eps);
        msm=std::max(msm,std::fabs(sm[c]-mean)); msi=std::max(msi,std::fabs(si[c]-inv));
        for(int64_t n=0;n<N;n++)for(int64_t hw=0;hw<HW;hw++){double ref=(X[(n*C+c)*HW+hw]-mean)*inv*gm[c]+bt[c]; my=std::max(my,std::fabs(hy[(n*C+c)*HW+hw]-ref)); mr=std::max(mr,std::fabs(ref));} }
    report("BatchNorm train y",my/(mr+1e-9),1e-4); report("BatchNorm train savedMean",msm,1e-4); report("BatchNorm train savedInvStd",msi,1e-3);
    aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(trm);aclDestroyTensor(trv);aclDestroyTensor(ty);aclDestroyTensor(tsm);aclDestroyTensor(tsi);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dg);aclrtFree(db);aclrtFree(drm);aclrtFree(drv);aclrtFree(dy);aclrtFree(dsm);aclrtFree(dsi);
}

// Attention backward (fp32, standard MHA): cross-check dQ/dK/dV
static void t_attn_bwd(int64_t B,int64_t N,int64_t Sq,int64_t Skv,int64_t D,bool causal){
    int64_t qn=B*N*Sq*D, kn=B*N*Skv*D; double scale=1.0/std::sqrt((double)D); int64_t off=Skv-Sq;
    auto Q=randv(qn),K=randv(kn),V=randv(kn),dO=randv(qn);
    std::vector<float> hdq(qn),hdk(kn),hdv(kn);
    void*dq_=up(Q.data(),qn*4),*dk_=up(K.data(),kn*4),*dv_=up(V.data(),kn*4),*ddo=up(dO.data(),qn*4),*odq,*odk,*odv;
    CHECK(aclrtMalloc(&odq,qn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&odk,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));CHECK(aclrtMalloc(&odv,kn*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,N,Sq,D},ACL_FLOAT,dq_),*tk=mk({B,N,Skv,D},ACL_FLOAT,dk_),*tv=mk({B,N,Skv,D},ACL_FLOAT,dv_),*tdo=mk({B,N,Sq,D},ACL_FLOAT,ddo);
    aclTensor*todq=mk({B,N,Sq,D},ACL_FLOAT,odq),*todk=mk({B,N,Skv,D},ACL_FLOAT,odk),*todv=mk({B,N,Skv,D},ACL_FLOAT,odv);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnFlashAttentionScoreBackwardGetWorkspaceSize(tq,tk,tv,tdo,nullptr,scale,N,causal,todq,todk,todv,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnFlashAttentionScoreBackward(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hdq.data(),qn*4,odq,qn*4,ACL_MEMCPY_DEVICE_TO_HOST));CHECK(aclrtMemcpy(hdk.data(),kn*4,odk,kn*4,ACL_MEMCPY_DEVICE_TO_HOST));CHECK(aclrtMemcpy(hdv.data(),kn*4,odv,kn*4,ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<double> dQ(qn,0),dK(kn,0),dV(kn,0);
    std::vector<double> P(Sq*Skv),dP(Sq*Skv);
    for(int64_t bn=0;bn<B*N;bn++){
        for(int64_t i=0;i<Sq;i++){ double mx=-1e30; for(int64_t j=0;j<Skv;j++){double s=0;for(int64_t d=0;d<D;d++)s+=(double)Q[(bn*Sq+i)*D+d]*K[(bn*Skv+j)*D+d];s*=scale; if(causal&&j>i+off)s=-1e30; P[i*Skv+j]=s; mx=std::max(mx,s);} double sm=0; for(int64_t j=0;j<Skv;j++){P[i*Skv+j]=std::exp(P[i*Skv+j]-mx);sm+=P[i*Skv+j];} for(int64_t j=0;j<Skv;j++)P[i*Skv+j]/=sm; }
        for(int64_t i=0;i<Sq;i++)for(int64_t j=0;j<Skv;j++){double dp=0;for(int64_t d=0;d<D;d++)dp+=(double)dO[(bn*Sq+i)*D+d]*V[(bn*Skv+j)*D+d];dP[i*Skv+j]=dp;}
        for(int64_t j=0;j<Skv;j++)for(int64_t d=0;d<D;d++){double a=0;for(int64_t i=0;i<Sq;i++)a+=P[i*Skv+j]*dO[(bn*Sq+i)*D+d];dV[(bn*Skv+j)*D+d]=a;}
        for(int64_t i=0;i<Sq;i++){double dot=0;for(int64_t j=0;j<Skv;j++)dot+=P[i*Skv+j]*dP[i*Skv+j]; for(int64_t j=0;j<Skv;j++)dP[i*Skv+j]=P[i*Skv+j]*(dP[i*Skv+j]-dot);}
        for(int64_t i=0;i<Sq;i++)for(int64_t d=0;d<D;d++){double a=0;for(int64_t j=0;j<Skv;j++)a+=dP[i*Skv+j]*K[(bn*Skv+j)*D+d];dQ[(bn*Sq+i)*D+d]=a*scale;}
        for(int64_t j=0;j<Skv;j++)for(int64_t d=0;d<D;d++){double a=0;for(int64_t i=0;i<Sq;i++)a+=dP[i*Skv+j]*Q[(bn*Sq+i)*D+d];dK[(bn*Skv+j)*D+d]=a*scale;}
    }
    auto ne=[&](std::vector<float>&g,std::vector<double>&r){double me=0,mr=0;for(size_t i=0;i<r.size();i++){me=std::max(me,std::fabs(g[i]-r[i]));mr=std::max(mr,std::fabs(r[i]));}return me/(mr+1e-9);};
    std::string tag=causal?" causal":"";
    report(("Attn bwd dQ"+tag).c_str(),ne(hdq,dQ),2e-3); report(("Attn bwd dK"+tag).c_str(),ne(hdk,dK),2e-3); report(("Attn bwd dV"+tag).c_str(),ne(hdv,dV),2e-3);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(tdo);aclDestroyTensor(todq);aclDestroyTensor(todk);aclDestroyTensor(todv);
    if(wsp)aclrtFree(wsp); aclrtFree(dq_);aclrtFree(dk_);aclrtFree(dv_);aclrtFree(ddo);aclrtFree(odq);aclrtFree(odk);aclrtFree(odv);
}

// Attention high-perf version (batched GEMM): cross-check against CPU reference (fp16/fp32, standard MHA)
static void t_attn_perf(bool fp16,int64_t B,int64_t N,int64_t Sq,int64_t Skv,int64_t D,bool causal,double tol){
    int64_t qn=B*N*Sq*D, kn=B*N*Skv*D; double scale=1.0/std::sqrt((double)D); int64_t off=Skv-Sq;
    auto Q=randv(qn),K=randv(kn),V=randv(kn); size_t esz=fp16?2:4;
    std::vector<uint8_t> qb(qn*esz),kb(kn*esz),vb(kn*esz);
    auto pack=[&](const std::vector<float>&src,std::vector<uint8_t>&dst){for(size_t i=0;i<src.size();i++) if(fp16){uint16_t h=f2h(src[i]);__builtin_memcpy(&dst[i*2],&h,2);} else __builtin_memcpy(&dst[i*4],&src[i],4);};
    pack(Q,qb);pack(K,kb);pack(V,vb);
    void*dq=up(qb.data(),qb.size()),*dk=up(kb.data(),kb.size()),*dv=up(vb.data(),vb.size()),*doO; CHECK(aclrtMalloc(&doO,qn*esz,ACL_MEM_MALLOC_HUGE_FIRST));
    aclDataType dt=fp16?ACL_FLOAT16:ACL_FLOAT;
    aclTensor*tq=mk({B,N,Sq,D},dt,dq),*tk=mk({B,N,Skv,D},dt,dk),*tv=mk({B,N,Skv,D},dt,dv),*to=mk({B,N,Sq,D},dt,doO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(tq,tk,tv,nullptr,scale,N,causal,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnFlashAttentionScoreHighPerf(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<uint8_t> ob(qn*esz); CHECK(aclrtMemcpy(ob.data(),ob.size(),doO,ob.size(),ACL_MEMCPY_DEVICE_TO_HOST));
    auto qv=[&](int64_t i){return fp16?(double)h2f(f2h(Q[i])):(double)Q[i];}; auto kv=[&](int64_t i){return fp16?(double)h2f(f2h(K[i])):(double)K[i];}; auto vv=[&](int64_t i){return fp16?(double)h2f(f2h(V[i])):(double)V[i];};
    auto getO=[&](int64_t i){return fp16?(double)h2f(((uint16_t*)ob.data())[i]):(double)((float*)ob.data())[i];};
    double maxrel=0; std::vector<double> s(Skv);
    for(int64_t bn=0;bn<B*N;bn++)for(int64_t i=0;i<Sq;i++){ double mx=-1e30;
        for(int64_t j=0;j<Skv;j++){double d=0;for(int64_t t=0;t<D;t++)d+=qv((bn*Sq+i)*D+t)*kv((bn*Skv+j)*D+t); d*=scale; if(causal&&j>i+off)d=-1e30; s[j]=d; mx=std::max(mx,d);}
        double sum=0; for(int64_t j=0;j<Skv;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
        for(int64_t t=0;t<D;t++){double o=0;for(int64_t j=0;j<Skv;j++)o+=s[j]/sum*vv((bn*Skv+j)*D+t); maxrel=std::max(maxrel,rel(getO((bn*Sq+i)*D+t),o,1e-2));} }
    report(fp16?"Attn perf fp16":"Attn perf fp32",maxrel,tol);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to); if(wsp)aclrtFree(wsp);
    aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(doO);
}

// DeepNorm（fp32）：y = LayerNorm(alpha·x + gx)·gamma + beta
static void t_deepnorm(int64_t rows,int64_t D){
    auto x=randv(rows*D,-2,2),gx=randv(rows*D,-2,2),gm=randv(D,0.5,1.5),bt=randv(D,-0.5,0.5);
    std::vector<float> hz(rows*D); double alpha=0.8,eps=1e-5;
    void*dx=up(x.data(),x.size()*4),*dgx=up(gx.data(),gx.size()*4),*dg=up(gm.data(),D*4),*db=up(bt.data(),D*4),*dy;
    CHECK(aclrtMalloc(&dy,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({rows,D},ACL_FLOAT,dx),*tgx=mk({rows,D},ACL_FLOAT,dgx),*tg=mk({D},ACL_FLOAT,dg),*tb=mk({D},ACL_FLOAT,db),*ty=mk({rows,D},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnDeepNormGetWorkspaceSize(tx,tgx,tg,tb,alpha,eps,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnDeepNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hz.data(),hz.size()*4,dy,hz.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;   // normalized error (output xhat*gamma+beta crosses zero; per-element relative error is inflated near zero)
    for(int64_t r=0;r<rows;r++){ double mean=0; for(int64_t d=0;d<D;d++)mean+=alpha*x[r*D+d]+gx[r*D+d]; mean/=D;
        double var=0; for(int64_t d=0;d<D;d++){double t=alpha*x[r*D+d]+gx[r*D+d]-mean;var+=t*t;} var/=D; double rstd=1.0/std::sqrt(var+eps);
        for(int64_t d=0;d<D;d++){double tmp=alpha*x[r*D+d]+gx[r*D+d],ref=(tmp-mean)*rstd*gm[d]+bt[d]; me=std::max(me,std::fabs(hz[r*D+d]-ref)); mr=std::max(mr,std::fabs(ref));} }
    report("DeepNorm",me/(mr+1e-9),1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tgx);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(ty);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dgx);aclrtFree(dg);aclrtFree(db);aclrtFree(dy);
}

// Fused norm (fp32): AddRmsNorm / AddLayerNorm (cross-check y + residual sum)
static void t_addnorm(bool layer,int64_t rows,int64_t D){
    auto x=randv(rows*D,-2,2),res=randv(rows*D,-2,2),gm=randv(D,0.5,1.5),bt=randv(D,-0.5,0.5); double eps=1e-5;
    std::vector<float> hy(rows*D),hs(rows*D);
    void*dx=up(x.data(),x.size()*4),*dr=up(res.data(),res.size()*4),*dg=up(gm.data(),D*4),*db=up(bt.data(),D*4),*dy,*dsum;
    CHECK(aclrtMalloc(&dy,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dsum,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({rows,D},ACL_FLOAT,dx),*tr=mk({rows,D},ACL_FLOAT,dr),*tg=mk({D},ACL_FLOAT,dg),*tb=mk({D},ACL_FLOAT,db),*ty=mk({rows,D},ACL_FLOAT,dy),*tsum=mk({rows,D},ACL_FLOAT,dsum);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if(layer)CHECK(aclnnAddLayerNormGetWorkspaceSize(tx,tr,tg,tb,eps,ty,tsum,&ws,&ex)); else CHECK(aclnnAddRmsNormGetWorkspaceSize(tx,tr,tg,eps,ty,tsum,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(layer)CHECK(aclnnAddLayerNorm(wsp,ws,ex,g_stream)); else CHECK(aclnnAddRmsNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hy.data(),hy.size()*4,dy,hy.size()*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hs.data(),hs.size()*4,dsum,hs.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double my=0,mr=0; long badsum=0;
    for(int64_t r=0;r<rows;r++){ std::vector<double> t(D); for(int64_t d=0;d<D;d++){t[d]=x[r*D+d]+res[r*D+d]; if(std::fabs(hs[r*D+d]-t[d])>1e-4)badsum++;}
        if(layer){ double mean=0;for(auto v:t)mean+=v;mean/=D; double var=0;for(auto v:t)var+=(v-mean)*(v-mean);var/=D; double rstd=1/std::sqrt(var+eps);
            for(int64_t d=0;d<D;d++){double ref=(t[d]-mean)*rstd*gm[d]+bt[d]; my=std::max(my,std::fabs(hy[r*D+d]-ref));mr=std::max(mr,std::fabs(ref));}}
        else { double ss=0;for(auto v:t)ss+=v*v; double inv=1/std::sqrt(ss/D+eps);
            for(int64_t d=0;d<D;d++){double ref=t[d]*inv*gm[d]; my=std::max(my,std::fabs(hy[r*D+d]-ref));mr=std::max(mr,std::fabs(ref));}} }
    report(layer?"AddLayerNorm":"AddRmsNorm", std::max(my/(mr+1e-9),(double)badsum), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tg);aclDestroyTensor(tb);aclDestroyTensor(ty);aclDestroyTensor(tsum);
    if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dr);aclrtFree(dg);aclrtFree(db);aclrtFree(dy);aclrtFree(dsum);
}
// SwiGlu / GeGlu（fp32）
static void t_glu(bool gelu,int64_t rows,int64_t D){
    auto in=randv(rows*2*D,-3,3); std::vector<float> ho(rows*D);
    void*di=up(in.data(),in.size()*4),*dy; CHECK(aclrtMalloc(&dy,rows*D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*ti=mk({rows,2*D},ACL_FLOAT,di),*ty=mk({rows,D},ACL_FLOAT,dy);
    uint64_t ws=0; aclOpExecutor*ex=nullptr;
    if(gelu)CHECK(aclnnGeGluGetWorkspaceSize(ti,ty,&ws,&ex)); else CHECK(aclnnSwiGluGetWorkspaceSize(ti,ty,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
    if(gelu)CHECK(aclnnGeGlu(wsp,ws,ex,g_stream)); else CHECK(aclnnSwiGlu(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(ho.data(),ho.size()*4,dy,ho.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t r=0;r<rows;r++)for(int64_t d=0;d<D;d++){ double a=in[r*2*D+d],b=in[r*2*D+D+d];
        double act=gelu? 0.5*a*(1+std::erf(a*0.70710678118654752)) : a/(1+std::exp(-a)); double ref=act*b;
        me=std::max(me,std::fabs(ho[r*D+d]-ref)); mr=std::max(mr,std::fabs(ref)); }
    report(gelu?"GeGlu":"SwiGlu", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(ti);aclDestroyTensor(ty); if(wsp)aclrtFree(wsp); aclrtFree(di);aclrtFree(dy);
}

// RoPE (fp32): half-split / interleaved modes, cross-check against CPU
static void t_rope(int mode,int64_t B,int64_t Nq,int64_t Nk,int64_t S,int64_t D){
    auto Q=randv(B*Nq*S*D),K=randv(B*Nk*S*D); std::vector<float> cs(S*D),sn(S*D);
    for(int64_t i=0;i<S*D;i++){cs[i]=std::cos(0.01*i);sn[i]=std::sin(0.013*i);}
    std::vector<float> qo(Q.size()),ko(K.size());
    void*dq=up(Q.data(),Q.size()*4),*dk=up(K.data(),K.size()*4),*dc=up(cs.data(),cs.size()*4),*ds=up(sn.data(),sn.size()*4),*dqo,*dko;
    CHECK(aclrtMalloc(&dqo,Q.size()*4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dko,K.size()*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,Nq,S,D},ACL_FLOAT,dq),*tk=mk({B,Nk,S,D},ACL_FLOAT,dk),*tc=mk({S,D},ACL_FLOAT,dc),*ts=mk({S,D},ACL_FLOAT,ds),*tqo=mk({B,Nq,S,D},ACL_FLOAT,dqo),*tko=mk({B,Nk,S,D},ACL_FLOAT,dko);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,mode,tqo,tko,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnApplyRotaryPosEmb(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(qo.data(),qo.size()*4,dqo,qo.size()*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(ko.data(),ko.size()*4,dko,ko.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    auto chk=[&](std::vector<float>&X,std::vector<float>&out,int64_t Nh)->double{ double mr=0;
        for(int64_t b=0;b<B;b++)for(int64_t h=0;h<Nh;h++)for(int64_t s=0;s<S;s++)for(int64_t d=0;d<D;d++){ int64_t base=((b*Nh+h)*S+s)*D;
            double rh; if(mode==0){int64_t half=D/2; rh=(d<half)?-X[base+d+half]:X[base+d-half];} else {rh=(d&1)?X[base+d-1]:-X[base+d+1];}
            double ref=X[base+d]*cs[s*D+d]+rh*sn[s*D+d]; mr=std::max(mr,rel(out[base+d],ref,1e-5)); } return mr; };
    double e1=chk(Q,qo,Nq),e2=chk(K,ko,Nk);
    report(mode==0?"RoPE half-split":"RoPE interleaved",std::max(e1,e2),mode==0?1e-4:5e-4);  // interleaved near-zero cancellation inflates per-element relative error (kernel correct, half-split 2e-6)
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tc);aclDestroyTensor(ts);aclDestroyTensor(tqo);aclDestroyTensor(tko);
    if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dc);aclrtFree(ds);aclrtFree(dqo);aclrtFree(dko);
}

// RoPE bf16: extend ApplyRotaryPosEmb to support bf16 (real models commonly use bf16)
static void t_rope_bf16(int64_t B,int64_t Nq,int64_t Nk,int64_t S,int64_t D){
    auto Q=randv(B*Nq*S*D),K=randv(B*Nk*S*D); std::vector<float> cs(S*D),sn(S*D);
    for(int64_t i=0;i<S*D;i++){cs[i]=std::cos(0.01*i);sn[i]=std::sin(0.013*i);}
    std::vector<uint16_t> hq(Q.size()),hk(K.size()),hc(cs.size()),hs(sn.size()),qo(Q.size()),ko(K.size());
    for(size_t i=0;i<Q.size();i++)hq[i]=f2bf(Q[i]); for(size_t i=0;i<K.size();i++)hk[i]=f2bf(K[i]);
    for(size_t i=0;i<cs.size();i++){hc[i]=f2bf(cs[i]);hs[i]=f2bf(sn[i]);}
    void*dq=up(hq.data(),hq.size()*2),*dk=up(hk.data(),hk.size()*2),*dc=up(hc.data(),hc.size()*2),*ds=up(hs.data(),hs.size()*2),*dqo,*dko;
    CHECK(aclrtMalloc(&dqo,Q.size()*2,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dko,K.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,Nq,S,D},ACL_BF16,dq),*tk=mk({B,Nk,S,D},ACL_BF16,dk),*tc=mk({S,D},ACL_BF16,dc),*ts=mk({S,D},ACL_BF16,ds),*tqo=mk({B,Nq,S,D},ACL_BF16,dqo),*tko=mk({B,Nk,S,D},ACL_BF16,dko);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnApplyRotaryPosEmbGetWorkspaceSize(tq,tk,tc,ts,0,tqo,tko,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnApplyRotaryPosEmb(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(qo.data(),qo.size()*2,dqo,qo.size()*2,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(ko.data(),ko.size()*2,dko,ko.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    auto chk=[&](std::vector<float>&X,std::vector<uint16_t>&out,int64_t Nh)->double{ double mr=0;
        for(int64_t b=0;b<B;b++)for(int64_t h=0;h<Nh;h++)for(int64_t s=0;s<S;s++)for(int64_t d=0;d<D;d++){ int64_t base=((b*Nh+h)*S+s)*D; int64_t half=D/2;
            double rh=(d<half)?-bf2f(f2bf(X[base+d+half])):bf2f(f2bf(X[base+d-half]));
            double ref=bf2f(f2bf(X[base+d]))*bf2f(hc[s*D+d])+rh*bf2f(hs[s*D+d]); mr=std::max(mr,rel(bf2f(out[base+d]),ref,5e-2)); } return mr; };
    double e1=chk(Q,qo,Nq),e2=chk(K,ko,Nk);
    report("RoPE bf16 half-split",std::max(e1,e2),3e-2);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tc);aclDestroyTensor(ts);aclDestroyTensor(tqo);aclDestroyTensor(tko);
    if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dc);aclrtFree(ds);aclrtFree(dqo);aclrtFree(dko);
}

// PagedAttention (fp32, GQA, decode): paged KV + blockTable, cross-check against CPU online attention
static void t_paged(int64_t B,int64_t Nq,int64_t Nkv,int64_t Sq,int64_t D,int64_t blockSize,int64_t maxBlocks){
    double scale=1.0/std::sqrt((double)D); int64_t blocks=B*maxBlocks;
    auto Q=randv(B*Nq*Sq*D), Kc=randv(blocks*blockSize*Nkv*D), Vc=randv(blocks*blockSize*Nkv*D);
    std::vector<int32_t> bt(B*maxBlocks), cl(B);
    for(int64_t b=0;b<B;b++){ for(int64_t j=0;j<maxBlocks;j++) bt[b*maxBlocks+j]=(int32_t)(b*maxBlocks+j); cl[b]=(int32_t)(maxBlocks*blockSize - 1 - b*3); if(cl[b]<1)cl[b]=blockSize; }
    void*dq=up(Q.data(),Q.size()*4),*dk=up(Kc.data(),Kc.size()*4),*dv=up(Vc.data(),Vc.size()*4),*dbt=up(bt.data(),bt.size()*4),*dcl=up(cl.data(),cl.size()*4),*doO;
    CHECK(aclrtMalloc(&doO,B*Nq*Sq*D*4,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,Nq,Sq,D},ACL_FLOAT,dq),*tk=mk({blocks,blockSize,Nkv,D},ACL_FLOAT,dk),*tv=mk({blocks,blockSize,Nkv,D},ACL_FLOAT,dv);
    aclTensor*tbt=mk({B,maxBlocks},ACL_INT32,dbt),*tcl=mk({B},ACL_INT32,dcl),*to=mk({B,Nq,Sq,D},ACL_FLOAT,doO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnPagedAttentionGetWorkspaceSize(tq,tk,tv,tbt,tcl,scale,Nq,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnPagedAttention(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    std::vector<float> O(B*Nq*Sq*D); CHECK(aclrtMemcpy(O.data(),O.size()*4,doO,O.size()*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t b=0;b<B;b++)for(int64_t h=0;h<Nq;h++){ int64_t kvh=h/(Nq/Nkv); int64_t L=cl[b];
        for(int64_t i=0;i<Sq;i++){ std::vector<double> sc(L); double mx=-1e30;
            for(int64_t p=0;p<L;p++){ int64_t blk=bt[b*maxBlocks+p/blockSize],off=p%blockSize; double d=0;
                for(int64_t t=0;t<D;t++)d+=(double)Q[(((b*Nq+h)*Sq)+i)*D+t]*Kc[((blk*blockSize+off)*Nkv+kvh)*D+t]; d*=scale; sc[p]=d; mx=std::max(mx,d);}
            double sum=0; for(int64_t p=0;p<L;p++){sc[p]=std::exp(sc[p]-mx);sum+=sc[p];}
            for(int64_t t=0;t<D;t++){ double o=0; for(int64_t p=0;p<L;p++){int64_t blk=bt[b*maxBlocks+p/blockSize],off=p%blockSize; o+=sc[p]/sum*Vc[((blk*blockSize+off)*Nkv+kvh)*D+t];}
                maxrel=std::max(maxrel,rel(O[(((b*Nq+h)*Sq)+i)*D+t],o,1e-4)); } } }
    report("PagedAttention GQA",maxrel,2e-3);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(tbt);aclDestroyTensor(tcl);aclDestroyTensor(to);
    if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(dbt);aclrtFree(dcl);aclrtFree(doO);
}

// ApplyAdamW: single-step update, cross-check param/m/v
static void t_adamw(int64_t n){
    auto p=randv(n,-1,1),m=randv(n,-0.1,0.1),v=randv(n,0,0.5),g=randv(n,-1,1);
    double lr=1e-3,b1=0.9,b2=0.999,eps=1e-8,wd=0.01; int64_t step=1;
    void*dp=up(p.data(),n*4),*dm=up(m.data(),n*4),*dv=up(v.data(),n*4),*dg=up(g.data(),n*4);
    aclTensor*tp=mk({n},ACL_FLOAT,dp),*tm=mk({n},ACL_FLOAT,dm),*tv=mk({n},ACL_FLOAT,dv),*tg=mk({n},ACL_FLOAT,dg);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyAdamWGetWorkspaceSize(tp,tm,tv,tg,lr,b1,b2,eps,wd,step,w,e);},aclnnApplyAdamW);
    std::vector<float> hp(n),hm(n),hv(n); CHECK(aclrtMemcpy(hp.data(),n*4,dp,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hm.data(),n*4,dm,n*4,ACL_MEMCPY_DEVICE_TO_HOST)); CHECK(aclrtMemcpy(hv.data(),n*4,dv,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double bc1=1-std::pow(b1,step),bc2=1-std::pow(b2,step),ep=0,em=0,ev=0,mr=0;
    for(int64_t i=0;i<n;i++){ double mi=b1*m[i]+(1-b1)*g[i],vi=b2*v[i]+(1-b2)*(double)g[i]*g[i]; double mh=mi/bc1,vh=vi/bc2; double pp=p[i]-lr*(mh/(std::sqrt(vh)+eps)+wd*p[i]);
        ep=std::max(ep,std::fabs(hp[i]-pp)); em=std::max(em,std::fabs(hm[i]-mi)); ev=std::max(ev,std::fabs(hv[i]-vi)); mr=std::max(mr,std::fabs(pp)); }
    report("ApplyAdamW", std::max(ep/(mr+1e-9), std::max(em, ev)), 1e-5);
    aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(tg); aclrtFree(dp);aclrtFree(dm);aclrtFree(dv);aclrtFree(dg);
}

// Loss functions
static void t_loss(){
    int64_t n=4096; auto p=randv(n,-1,1),t=randv(n,-1,1); float go=1.5f;
    // MSE + backward
    { void*dp=up(p.data(),n*4),*dt=up(t.data(),n*4),*dout,*dgo=up(&go,4),*dgp; CHECK(aclrtMalloc(&dout,4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dgp,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tp=mk({n},ACL_FLOAT,dp),*tt=mk({n},ACL_FLOAT,dt),*to=mk({1},ACL_FLOAT,dout),*tgo=mk({1},ACL_FLOAT,dgo),*tgp=mk({n},ACL_FLOAT,dgp);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMseLossGetWorkspaceSize(tp,tt,1,to,w,e);},aclnnMseLoss); float ho; CHECK(aclrtMemcpy(&ho,4,dout,4,ACL_MEMCPY_DEVICE_TO_HOST));
      double s=0; for(int64_t i=0;i<n;i++)s+=(double)(p[i]-t[i])*(p[i]-t[i]); s/=n; report("MSELoss mean",rel(ho,s),1e-5);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMseLossBackwardGetWorkspaceSize(tgo,tp,tt,1,tgp,w,e);},aclnnMseLossBackward); std::vector<float> hg(n); CHECK(aclrtMemcpy(hg.data(),n*4,dgp,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int64_t i=0;i<n;i++){double r=go*2.0*(p[i]-t[i])/n; me=std::max(me,std::fabs(hg[i]-r));mr=std::max(mr,std::fabs(r));} report("MSELoss bwd",me/(mr+1e-9),1e-4);
      aclDestroyTensor(tp);aclDestroyTensor(tt);aclDestroyTensor(to);aclDestroyTensor(tgo);aclDestroyTensor(tgp); aclrtFree(dp);aclrtFree(dt);aclrtFree(dout);aclrtFree(dgo);aclrtFree(dgp); }
    // BCEWithLogits
    { auto tg=randv(n,0,1); void*dx=up(p.data(),n*4),*dtg=up(tg.data(),n*4),*dout; CHECK(aclrtMalloc(&dout,4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tx=mk({n},ACL_FLOAT,dx),*tt=mk({n},ACL_FLOAT,dtg),*to=mk({1},ACL_FLOAT,dout);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBinaryCrossEntropyWithLogitsGetWorkspaceSize(tx,tt,1,to,w,e);},aclnnBinaryCrossEntropyWithLogits); float ho; CHECK(aclrtMemcpy(&ho,4,dout,4,ACL_MEMCPY_DEVICE_TO_HOST));
      double s=0; for(int64_t i=0;i<n;i++)s+=std::max((double)p[i],0.0)-(double)p[i]*tg[i]+std::log1p(std::exp(-std::fabs((double)p[i]))); s/=n; report("BCEWithLogits",rel(ho,s),1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(to); aclrtFree(dx);aclrtFree(dtg);aclrtFree(dout); }
    // NLL + CrossEntropy + CE bwd
    { int64_t N=256,C=10; auto lg=randv(N*C,-2,2); std::vector<int64_t> tgt(N); for(auto&v:tgt)v=rand()%C;
      // logprob = logsoftmax(lg) for NLL
      std::vector<float> lp(N*C); for(int64_t r=0;r<N;r++){double mx=-1e30;for(int64_t c=0;c<C;c++)mx=std::max(mx,(double)lg[r*C+c]);double se=0;for(int64_t c=0;c<C;c++)se+=std::exp(lg[r*C+c]-mx);for(int64_t c=0;c<C;c++)lp[r*C+c]=(float)(lg[r*C+c]-mx-std::log(se));}
      void*dlp=up(lp.data(),N*C*4),*dlg=up(lg.data(),N*C*4),*dtgt=up(tgt.data(),N*8),*do1,*do2,*dgo=up(&go,4),*dgl;
      CHECK(aclrtMalloc(&do1,4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&do2,4,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMalloc(&dgl,N*C*4,ACL_MEM_MALLOC_HUGE_FIRST));
      aclTensor*tlp=mk({N,C},ACL_FLOAT,dlp),*tlg=mk({N,C},ACL_FLOAT,dlg),*tt=mk({N},ACL_INT64,dtgt),*to1=mk({1},ACL_FLOAT,do1),*to2=mk({1},ACL_FLOAT,do2),*tgo=mk({1},ACL_FLOAT,dgo),*tgl=mk({N,C},ACL_FLOAT,dgl);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNLLLossGetWorkspaceSize(tlp,tt,1,to1,w,e);},aclnnNLLLoss); float hnll; CHECK(aclrtMemcpy(&hnll,4,do1,4,ACL_MEMCPY_DEVICE_TO_HOST));
      double snll=0; for(int64_t r=0;r<N;r++)snll+=-lp[r*C+tgt[r]]; snll/=N; report("NLLLoss",rel(hnll,snll),1e-5);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCrossEntropyLossGetWorkspaceSize(tlg,tt,1,to2,w,e);},aclnnCrossEntropyLoss); float hce; CHECK(aclrtMemcpy(&hce,4,do2,4,ACL_MEMCPY_DEVICE_TO_HOST));
      report("CrossEntropy(=NLL∘logsm)",rel(hce,snll),1e-4);   // CE(logits)=NLL(logsoftmax)
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCrossEntropyLossBackwardGetWorkspaceSize(tgo,tlg,tt,1,tgl,w,e);},aclnnCrossEntropyLossBackward); std::vector<float> hg(N*C); CHECK(aclrtMemcpy(hg.data(),N*C*4,dgl,N*C*4,ACL_MEMCPY_DEVICE_TO_HOST));
      double me=0,mr=0; for(int64_t r=0;r<N;r++)for(int64_t c=0;c<C;c++){double sm=std::exp(lp[r*C+c]); double ref=go*(sm-(c==tgt[r]?1.0:0.0))/N; me=std::max(me,std::fabs(hg[r*C+c]-ref));mr=std::max(mr,std::fabs(ref));} report("CrossEntropy bwd",me/(mr+1e-9),1e-4);
      aclDestroyTensor(tlp);aclDestroyTensor(tlg);aclDestroyTensor(tt);aclDestroyTensor(to1);aclDestroyTensor(to2);aclDestroyTensor(tgo);aclDestroyTensor(tgl);
      aclrtFree(dlp);aclrtFree(dlg);aclrtFree(dtgt);aclrtFree(do1);aclrtFree(do2);aclrtFree(dgo);aclrtFree(dgl); }
}

// PagedAttention bf16: extend PagedAttention to support bf16
static void t_paged_bf16(int64_t B,int64_t Nq,int64_t Nkv,int64_t Sq,int64_t D,int64_t blockSize,int64_t maxBlocks){
    double scale=1.0/std::sqrt((double)D); int64_t blocks=B*maxBlocks;
    auto Q=randv(B*Nq*Sq*D), Kc=randv(blocks*blockSize*Nkv*D), Vc=randv(blocks*blockSize*Nkv*D);
    std::vector<int32_t> bt(B*maxBlocks), cl(B);
    for(int64_t b=0;b<B;b++){ for(int64_t j=0;j<maxBlocks;j++) bt[b*maxBlocks+j]=(int32_t)(b*maxBlocks+j); cl[b]=(int32_t)(maxBlocks*blockSize-1-b*3); if(cl[b]<1)cl[b]=blockSize; }
    std::vector<uint16_t> hq(Q.size()),hk(Kc.size()),hv(Vc.size()),hoO(B*Nq*Sq*D);
    for(size_t i=0;i<Q.size();i++)hq[i]=f2bf(Q[i]); for(size_t i=0;i<Kc.size();i++){hk[i]=f2bf(Kc[i]);hv[i]=f2bf(Vc[i]);}
    void*dq=up(hq.data(),hq.size()*2),*dk=up(hk.data(),hk.size()*2),*dv=up(hv.data(),hv.size()*2),*dbt=up(bt.data(),bt.size()*4),*dcl=up(cl.data(),cl.size()*4),*doO;
    CHECK(aclrtMalloc(&doO,hoO.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,Nq,Sq,D},ACL_BF16,dq),*tk=mk({blocks,blockSize,Nkv,D},ACL_BF16,dk),*tv=mk({blocks,blockSize,Nkv,D},ACL_BF16,dv);
    aclTensor*tbt=mk({B,maxBlocks},ACL_INT32,dbt),*tcl=mk({B},ACL_INT32,dcl),*to=mk({B,Nq,Sq,D},ACL_BF16,doO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnPagedAttentionGetWorkspaceSize(tq,tk,tv,tbt,tcl,scale,Nq,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnPagedAttention(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hoO.data(),hoO.size()*2,doO,hoO.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0;
    for(int64_t b=0;b<B;b++)for(int64_t h=0;h<Nq;h++){ int64_t kvh=h/(Nq/Nkv),L=cl[b];
        for(int64_t i=0;i<Sq;i++){ std::vector<double> sc(L); double mx=-1e30;
            for(int64_t p=0;p<L;p++){ int64_t blk=bt[b*maxBlocks+p/blockSize],off=p%blockSize; double d=0;
                for(int64_t t=0;t<D;t++)d+=(double)bf2f(hq[(((b*Nq+h)*Sq)+i)*D+t])*bf2f(hk[((blk*blockSize+off)*Nkv+kvh)*D+t]); d*=scale; sc[p]=d; mx=std::max(mx,d);}
            double sum=0; for(int64_t p=0;p<L;p++){sc[p]=std::exp(sc[p]-mx);sum+=sc[p];}
            for(int64_t t=0;t<D;t++){ double o=0; for(int64_t p=0;p<L;p++){int64_t blk=bt[b*maxBlocks+p/blockSize],off=p%blockSize; o+=sc[p]/sum*bf2f(hv[((blk*blockSize+off)*Nkv+kvh)*D+t]);}
                maxrel=std::max(maxrel,rel(bf2f(hoO[(((b*Nq+h)*Sq)+i)*D+t]),o,1e-2)); } } }
    report("PagedAttention GQA bf16",maxrel,6e-2);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(tbt);aclDestroyTensor(tcl);aclDestroyTensor(to);
    if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(dbt);aclrtFree(dcl);aclrtFree(doO);
}

// ---- bf16 coverage: extend LayerNorm / FlashAttentionScoreHighPerf to support bf16 ----
static void t_layernorm_bf16(int64_t rows,int64_t D){
    auto x=randv(rows*D,-2,2),g=randv(D,0.5,1.5),b=randv(D,-0.5,0.5);
    std::vector<uint16_t> hx(rows*D),hg(D),hb(D),hy(rows*D);
    for(int64_t i=0;i<rows*D;i++)hx[i]=f2bf(x[i]); for(int64_t i=0;i<D;i++){hg[i]=f2bf(g[i]);hb[i]=f2bf(b[i]);}
    const double eps=1e-5;
    void*dx=up(hx.data(),hx.size()*2),*dy,*dg=up(hg.data(),D*2),*db=up(hb.data(),D*2); CHECK(aclrtMalloc(&dy,hy.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tx=mk({rows,D},ACL_BF16,dx),*ty=mk({rows,D},ACL_BF16,dy),*tg=mk({D},ACL_BF16,dg),*tb=mk({D},ACL_BF16,db);
    int64_t ns[1]={D}; aclIntArray*nsh=aclCreateIntArray(ns,1);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnLayerNormGetWorkspaceSize(tx,nsh,tg,tb,eps,ty,nullptr,nullptr,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnLayerNorm(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(hy.data(),hy.size()*2,dy,hy.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t r=0;r<rows;r++){ double mean=0; for(int64_t j=0;j<D;j++)mean+=bf2f(hx[r*D+j]); mean/=D;
        double var=0; for(int64_t j=0;j<D;j++){double d=bf2f(hx[r*D+j])-mean;var+=d*d;} var/=D; double rstd=1.0/std::sqrt(var+eps);
        for(int64_t j=0;j<D;j++){ double ref=(bf2f(hx[r*D+j])-mean)*rstd*bf2f(hg[j])+bf2f(hb[j]);
            me=std::max(me,std::fabs(bf2f(hy[r*D+j])-ref)); mr=std::max(mr,std::fabs(ref)); } }
    report("LayerNorm bf16",me/(mr+1e-9),3e-2);
    aclDestroyIntArray(nsh);aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(tg);aclDestroyTensor(tb); if(wsp)aclrtFree(wsp); aclrtFree(dx);aclrtFree(dy);aclrtFree(dg);aclrtFree(db);
}
static void t_attn_perf_bf16(int64_t B,int64_t N,int64_t Sq,int64_t Skv,int64_t D,bool causal){
    int64_t qn=B*N*Sq*D,kn=B*N*Skv*D; double scale=1.0/std::sqrt((double)D); int64_t off=Skv-Sq;
    auto Q=randv(qn),K=randv(kn),Vv=randv(kn);
    std::vector<uint16_t> qb(qn),kb(kn),vb(kn),ob(qn);
    for(int64_t i=0;i<qn;i++)qb[i]=f2bf(Q[i]); for(int64_t i=0;i<kn;i++){kb[i]=f2bf(K[i]);vb[i]=f2bf(Vv[i]);}
    void*dq=up(qb.data(),qb.size()*2),*dk=up(kb.data(),kb.size()*2),*dv=up(vb.data(),vb.size()*2),*doO; CHECK(aclrtMalloc(&doO,qn*2,ACL_MEM_MALLOC_HUGE_FIRST));
    aclTensor*tq=mk({B,N,Sq,D},ACL_BF16,dq),*tk=mk({B,N,Skv,D},ACL_BF16,dk),*tv=mk({B,N,Skv,D},ACL_BF16,dv),*to=mk({B,N,Sq,D},ACL_BF16,doO);
    uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(aclnnFlashAttentionScoreHighPerfGetWorkspaceSize(tq,tk,tv,nullptr,scale,N,causal,to,&ws,&ex));
    void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclnnFlashAttentionScoreHighPerf(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(ob.data(),ob.size()*2,doO,ob.size()*2,ACL_MEMCPY_DEVICE_TO_HOST));
    double maxrel=0; std::vector<double> s(Skv);
    for(int64_t bn=0;bn<B*N;bn++)for(int64_t i=0;i<Sq;i++){ double mx=-1e30;
        for(int64_t j=0;j<Skv;j++){double d=0;for(int64_t t=0;t<D;t++)d+=bf2f(qb[(bn*Sq+i)*D+t])*bf2f(kb[(bn*Skv+j)*D+t]); d*=scale; if(causal&&j>i+off)d=-1e30; s[j]=d; mx=std::max(mx,d);}
        double sum=0; for(int64_t j=0;j<Skv;j++){s[j]=std::exp(s[j]-mx);sum+=s[j];}
        for(int64_t t=0;t<D;t++){double o=0;for(int64_t j=0;j<Skv;j++)o+=s[j]/sum*bf2f(vb[(bn*Skv+j)*D+t]); maxrel=std::max(maxrel,rel(bf2f(ob[(bn*Sq+i)*D+t]),o,1e-2));} }
    report("Attn perf bf16",maxrel,6e-2);
    aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tv);aclDestroyTensor(to); if(wsp)aclrtFree(wsp); aclrtFree(dq);aclrtFree(dk);aclrtFree(dv);aclrtFree(doO);
}

int main() {
    CHECK(aclInit(nullptr));
    CHECK(aclrtSetDevice(0));
    CHECK(aclrtCreateStream(&g_stream));
    srand(13);

    t_softmax(1024, 128, 1e-5);
    t_softmax(64, 1000, 1e-5);
    t_layernorm(2048, 256, false, 1e-5);
    t_layernorm(2048, 768, true, 1e-5);
    t_attention(2, 4, 64, 64, 32, false, 2e-3);
    t_attention(2, 4, 64, 64, 32, true, 2e-3);
    t_attention(1, 8, 48, 96, 64, false, 2e-3);   // cross-attention Sq!=Skv

    t_attn2("Attn GQA Nq8 Nkv2", false, 2, 8, 2, 64, 64, 32, false, 0, 2e-3);   // GQA
    t_attn2("Attn MQA Nq8 Nkv1", false, 1, 8, 1, 48, 48, 64, true, 0, 2e-3);    // MQA + causal
    t_attn2("Attn fp16", true, 2, 4, 4, 64, 64, 32, false, 0, 3e-3);            // fp16 I/O
    t_attn2("Attn fp16 D64 1tile", true, 1, 1, 1, 16, 16, 64, false, 0, 5e-3);  // WMMA single tile
    t_attn2("Attn fp16 D64 wmma", true, 2, 4, 4, 128, 128, 64, false, 0, 5e-3); // WMMA tensor-core path
    t_attn2("Attn fp16 D64 wmma causal", true, 2, 4, 4, 96, 96, 64, true, 0, 5e-3);
    t_attn2("Attn per-batch mask", false, 2, 4, 4, 32, 48, 32, false, 1, 2e-3); // per-batch mask

    t_softmaxdim("Softmax dim1 [8,16,4]", {8,16,4}, 1, false);   // non-last dim
    t_softmaxdim("Softmax dim0 [16,32]", {16,32}, 0, false);
    t_softmaxdim("LogSoftmax dim2 [4,8,16]", {4,8,16}, 2, true);
    t_rmsnorm(2048, 768);
    t_groupnorm(2, 8, 8, 8, 4);     // GroupNorm G=4
    t_groupnorm(2, 6, 4, 4, 6);     // InstanceNorm (G=C)
    t_batchnorm(2, 8, 8, 8);
    t_attn_fp4(2, 4, 48, 48, 32);   // fp4 input attention
    t_attn_fp8(2, 4, 48, 48, 32);   // fp8 e4m3 input attention
    t_rmsnorm_bwd(1024, 256);
    t_layernorm_bwd(1024, 256);
    t_batchnorm_train(4, 8, 8, 8);
    t_deepnorm(2048, 256);                 // added to align with explorer
    t_attn_bwd(2, 2, 32, 32, 16, false);
    t_attn_bwd(2, 2, 48, 48, 16, true);    // causal
    t_attn_perf(false, 2, 4, 64, 64, 32, false, 2e-3);  // high-perf fp32
    t_attn_perf(true,  2, 4, 64, 64, 32, false, 1e-2);  // high-perf fp16
    t_attn_perf_bf16(2, 4, 64, 64, 32, false);          // high-perf bf16
    t_attn_perf_bf16(1, 8, 48, 96, 64, true);           // cross-attention + causal bf16
    t_attn_perf(false, 1, 8, 48, 96, 64, true, 2e-3);   // cross-attention + causal
    t_paged(2, 8, 2, 1, 64, 16, 4);   // PagedAttention (decode, GQA)
    t_paged(2, 4, 4, 2, 32, 8, 6);
    t_paged_bf16(2, 8, 2, 1, 64, 16, 4);  // PagedAttention bf16
    t_rope(0, 2, 8, 2, 12, 16);       // RoPE half-split (GQA: Nq=8,Nk=2)
    t_rope(1, 2, 4, 4, 12, 16);       // RoPE interleaved
    t_rope_bf16(2, 8, 2, 12, 16);     // RoPE bf16 (real model bf16 path)
    t_layernorm_bf16(8, 64);          // LayerNorm bf16
    t_addnorm(false, 1024, 256);      // AddRmsNorm
    t_addnorm(true, 1024, 256);       // AddLayerNorm
    t_glu(false, 1024, 256);          // SwiGlu
    t_glu(true, 1024, 256);           // GeGlu
    t_loss();                         // loss functions
    t_adamw(1 << 16);                 // optimizer

    CHECK(aclrtDestroyStream(g_stream));
    CHECK(aclrtResetDevice(0));
    CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
