// Legacy single-op interface cross-check: aclopExecuteV2 routes by opType to aclnn.
// Uses aclCreateTensorDesc/aclCreateDataBuffer/aclopSetAttr* to assemble calls; CPU double reference.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include "acl/acl.h"
#include "acl/acl_op.h"

static aclrtStream g_stream;
static int g_pass = 0, g_fail = 0;
#define CHECK(x) do { int __r=(int)(x); if(__r!=0){ printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,__r); } } while(0)
static void report(const char *nm, double err, double tol) {
    bool ok = err <= tol; (ok?g_pass:g_fail)++;
    printf("%-26s err=%.2e tol=%.0e %s\n", nm, err, tol, ok?"PASS":"FAIL");
}
static std::vector<float> randv(int64_t n){ std::vector<float> v(n); for(auto&x:v)x=(rand()/(float)RAND_MAX)*2-1; return v; }
static void *up(const void *h, size_t b){ void*d; CHECK(aclrtMalloc(&d,b,ACL_MEM_MALLOC_HUGE_FIRST)); CHECK(aclrtMemcpy(d,b,h,b,ACL_MEMCPY_HOST_TO_DEVICE)); return d; }

// Unary Abs: [n]
static void t_abs(int64_t n){
    auto x=randv(n); std::vector<float> y(n);
    void*dx=up(x.data(),n*4),*dy; CHECK(aclrtMalloc(&dy,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dims[1]={n};
    aclTensorDesc*id=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*od=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND);
    aclDataBuffer*ib=aclCreateDataBuffer(dx,n*4),*ob=aclCreateDataBuffer(dy,n*4);
    aclTensorDesc*idesc[1]={id},*odesc[1]={od}; aclDataBuffer*ibuf[1]={ib},*obuf[1]={ob};
    CHECK(aclopExecuteV2("Abs",1,idesc,ibuf,1,odesc,obuf,nullptr,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(),n*4,dy,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0; for(int64_t i=0;i<n;i++) me=std::max(me,(double)std::fabs(y[i]-std::fabs(x[i])));
    report("aclop Abs",me,1e-6);
    aclDestroyTensorDesc(id);aclDestroyTensorDesc(od);aclDestroyDataBuffer(ib);aclDestroyDataBuffer(ob);aclrtFree(dx);aclrtFree(dy);
}
// Binary Add: [n]
static void t_add(int64_t n){
    auto a=randv(n),b=randv(n); std::vector<float> c(n);
    void*da=up(a.data(),n*4),*db=up(b.data(),n*4),*dc; CHECK(aclrtMalloc(&dc,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dims[1]={n};
    aclTensorDesc*da_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*db_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*dc_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND);
    aclDataBuffer*ba=aclCreateDataBuffer(da,n*4),*bb=aclCreateDataBuffer(db,n*4),*bc=aclCreateDataBuffer(dc,n*4);
    aclTensorDesc*idesc[2]={da_,db_},*odesc[1]={dc_}; aclDataBuffer*ibuf[2]={ba,bb},*obuf[1]={bc};
    CHECK(aclopExecuteV2("Add",2,idesc,ibuf,1,odesc,obuf,nullptr,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(c.data(),n*4,dc,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0; for(int64_t i=0;i<n;i++) me=std::max(me,(double)std::fabs(c[i]-(a[i]+b[i])));
    report("aclop Add",me,1e-6);
    aclDestroyTensorDesc(da_);aclDestroyTensorDesc(db_);aclDestroyTensorDesc(dc_);
    aclDestroyDataBuffer(ba);aclDestroyDataBuffer(bb);aclDestroyDataBuffer(bc);aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
// MatMul: [M,K]@[K,N]
static void t_matmul(int64_t M,int64_t K,int64_t N){
    auto a=randv(M*K),b=randv(K*N); std::vector<float> c(M*N);
    void*da=up(a.data(),M*K*4),*db=up(b.data(),K*N*4),*dc; CHECK(aclrtMalloc(&dc,M*N*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t ad[2]={M,K},bd[2]={K,N},cd[2]={M,N};
    aclTensorDesc*da_=aclCreateTensorDesc(ACL_FLOAT,2,ad,ACL_FORMAT_ND),*db_=aclCreateTensorDesc(ACL_FLOAT,2,bd,ACL_FORMAT_ND),*dc_=aclCreateTensorDesc(ACL_FLOAT,2,cd,ACL_FORMAT_ND);
    aclDataBuffer*ba=aclCreateDataBuffer(da,M*K*4),*bb=aclCreateDataBuffer(db,K*N*4),*bc=aclCreateDataBuffer(dc,M*N*4);
    aclTensorDesc*idesc[2]={da_,db_},*odesc[1]={dc_}; aclDataBuffer*ibuf[2]={ba,bb},*obuf[1]={bc};
    CHECK(aclopExecuteV2("MatMul",2,idesc,ibuf,1,odesc,obuf,nullptr,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(c.data(),M*N*4,dc,M*N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t i=0;i<M;i++)for(int64_t j=0;j<N;j++){ double acc=0; for(int64_t k=0;k<K;k++) acc+=(double)a[i*K+k]*b[k*N+j];
        me=std::max(me,std::fabs(c[i*N+j]-acc)); mr=std::max(mr,std::fabs(acc)); }
    report("aclop MatMul",me/(mr+1e-9),1e-5);
    aclDestroyTensorDesc(da_);aclDestroyTensorDesc(db_);aclDestroyTensorDesc(dc_);
    aclDestroyDataBuffer(ba);aclDestroyDataBuffer(bb);aclDestroyDataBuffer(bc);aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
// ReduceSum: [R,C] along axis=1, keep_dims=false -> [R] (axes/keep_dims passed via attr)
static void t_reducesum(int64_t R,int64_t C){
    auto x=randv(R*C); std::vector<float> y(R);
    void*dx=up(x.data(),R*C*4),*dy; CHECK(aclrtMalloc(&dy,R*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t id_[2]={R,C},od_[1]={R};
    aclTensorDesc*idd=aclCreateTensorDesc(ACL_FLOAT,2,id_,ACL_FORMAT_ND),*odd=aclCreateTensorDesc(ACL_FLOAT,1,od_,ACL_FORMAT_ND);
    aclDataBuffer*ib=aclCreateDataBuffer(dx,R*C*4),*ob=aclCreateDataBuffer(dy,R*4);
    aclopAttr*attr=aclopCreateAttr(); int64_t axes[1]={1};
    CHECK(aclopSetAttrListInt(attr,"axes",1,axes));
    CHECK(aclopSetAttrBool(attr,"keep_dims",0));
    aclTensorDesc*idesc[1]={idd},*odesc[1]={odd}; aclDataBuffer*ibuf[1]={ib},*obuf[1]={ob};
    CHECK(aclopExecuteV2("ReduceSum",1,idesc,ibuf,1,odesc,obuf,attr,g_stream));
    CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(),R*4,dy,R*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0; for(int64_t r=0;r<R;r++){ double s=0; for(int64_t c=0;c<C;c++) s+=x[r*C+c];
        me=std::max(me,std::fabs(y[r]-s)); mr=std::max(mr,std::fabs(s)); }
    report("aclop ReduceSum+attr",me/(mr+1e-9),1e-5);
    aclopDestroyAttr(attr);aclDestroyTensorDesc(idd);aclDestroyTensorDesc(odd);aclDestroyDataBuffer(ib);aclDestroyDataBuffer(ob);aclrtFree(dx);aclrtFree(dy);
}

// Unary Tanh (aclop routing)
static void t_tanh(int64_t n){
    auto x=randv(n); std::vector<float> y(n);
    void*dx=up(x.data(),n*4),*dy; CHECK(aclrtMalloc(&dy,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dims[1]={n};
    aclTensorDesc*id=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*od=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND);
    aclDataBuffer*ib=aclCreateDataBuffer(dx,n*4),*ob=aclCreateDataBuffer(dy,n*4);
    aclTensorDesc*idesc[1]={id},*odesc[1]={od}; aclDataBuffer*ibuf[1]={ib},*obuf[1]={ob};
    CHECK(aclopExecuteV2("Tanh",1,idesc,ibuf,1,odesc,obuf,nullptr,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(),n*4,dy,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0; for(int64_t i=0;i<n;i++) me=std::max(me,(double)std::fabs(y[i]-std::tanh(x[i])));
    report("aclop Tanh",me,1e-6);
    aclDestroyTensorDesc(id);aclDestroyTensorDesc(od);aclDestroyDataBuffer(ib);aclDestroyDataBuffer(ob);aclrtFree(dx);aclrtFree(dy);
}
// Binary Maximum (aclop routing)
static void t_maximum(int64_t n){
    auto a=randv(n),b=randv(n); std::vector<float> c(n);
    void*da=up(a.data(),n*4),*db=up(b.data(),n*4),*dc; CHECK(aclrtMalloc(&dc,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dims[1]={n};
    aclTensorDesc*da_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*db_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND),*dc_=aclCreateTensorDesc(ACL_FLOAT,1,dims,ACL_FORMAT_ND);
    aclDataBuffer*ba=aclCreateDataBuffer(da,n*4),*bb=aclCreateDataBuffer(db,n*4),*bc=aclCreateDataBuffer(dc,n*4);
    aclTensorDesc*idesc[2]={da_,db_},*odesc[1]={dc_}; aclDataBuffer*ibuf[2]={ba,bb},*obuf[1]={bc};
    CHECK(aclopExecuteV2("Maximum",2,idesc,ibuf,1,odesc,obuf,nullptr,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(c.data(),n*4,dc,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0; for(int64_t i=0;i<n;i++) me=std::max(me,(double)std::fabs(c[i]-std::max(a[i],b[i])));
    report("aclop Maximum",me,1e-6);
    aclDestroyTensorDesc(da_);aclDestroyTensorDesc(db_);aclDestroyTensorDesc(dc_);
    aclDestroyDataBuffer(ba);aclDestroyDataBuffer(bb);aclDestroyDataBuffer(bc);aclrtFree(da);aclrtFree(db);aclrtFree(dc);
}
// Softmax (aclop routing + attr "axis"=-1, last dim)
static void t_softmax(int64_t R,int64_t C){
    auto x=randv(R*C); std::vector<float> y(R*C);
    void*dx=up(x.data(),R*C*4),*dy; CHECK(aclrtMalloc(&dy,R*C*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t dims[2]={R,C};
    aclTensorDesc*id=aclCreateTensorDesc(ACL_FLOAT,2,dims,ACL_FORMAT_ND),*od=aclCreateTensorDesc(ACL_FLOAT,2,dims,ACL_FORMAT_ND);
    aclDataBuffer*ib=aclCreateDataBuffer(dx,R*C*4),*ob=aclCreateDataBuffer(dy,R*C*4);
    aclopAttr*attr=aclopCreateAttr(); CHECK(aclopSetAttrInt(attr,"axis",-1));
    aclTensorDesc*idesc[1]={id},*odesc[1]={od}; aclDataBuffer*ibuf[1]={ib},*obuf[1]={ob};
    CHECK(aclopExecuteV2("Softmax",1,idesc,ibuf,1,odesc,obuf,attr,g_stream)); CHECK(aclrtSynchronizeStream(g_stream));
    CHECK(aclrtMemcpy(y.data(),R*C*4,dy,R*C*4,ACL_MEMCPY_DEVICE_TO_HOST));
    double me=0,mr=0;
    for(int64_t r=0;r<R;r++){ double mx=-1e30; for(int64_t c=0;c<C;c++)mx=std::max(mx,(double)x[r*C+c]);
        double den=0; for(int64_t c=0;c<C;c++)den+=std::exp(x[r*C+c]-mx);
        for(int64_t c=0;c<C;c++){double ref=std::exp(x[r*C+c]-mx)/den; me=std::max(me,std::fabs(y[r*C+c]-ref)); mr=std::max(mr,ref);} }
    report("aclop Softmax+attr",me/(mr+1e-9),1e-5);
    aclopDestroyAttr(attr);aclDestroyTensorDesc(id);aclDestroyTensorDesc(od);aclDestroyDataBuffer(ib);aclDestroyDataBuffer(ob);aclrtFree(dx);aclrtFree(dy);
}

int main(){
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(31);
    t_abs(4096);
    t_add(4096);
    t_matmul(128,256,64);
    t_reducesum(64,128);
    t_tanh(4096);
    t_maximum(4096);
    t_softmax(64,128);
    CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
    return g_fail?1:0;
}
