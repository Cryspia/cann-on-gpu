// Sort/scan cross-check: Cumsum/Cumprod (along a dimension), Sort (values + indices), Topk. CPU reference uses (value, index) total order.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_ops.h"

#define CHECK(x) do { int __r=(int)(x); if(__r!=0){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,__r);exit(1);} } while(0)
static aclrtStream g_stream; static int g_pass=0,g_fail=0;
struct DevBuf{ void*p=nullptr; size_t bytes=0; DevBuf(size_t b):bytes(b){CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST));}
  ~DevBuf(){aclrtFree(p);} void up(const void*h){CHECK(aclrtMemcpy(p,bytes,h,bytes,ACL_MEMCPY_HOST_TO_DEVICE));}
  void down(void*h)const{CHECK(aclrtMemcpy(h,bytes,p,bytes,ACL_MEMCPY_DEVICE_TO_HOST));} };
static aclTensor*mk(const std::vector<int64_t>&d,aclDataType dt,void*data){return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),data);}
static void report(const std::string&n,double err,double tol){bool ok=err<=tol;(ok?g_pass:g_fail)++;printf("%-28s err=%.2e tol=%.0e %s\n",n.c_str(),err,tol,ok?"PASS":"FAIL");}
static void reportb(const std::string&n,int64_t bad){(bad==0?g_pass:g_fail)++;printf("%-28s mismatch=%lld %s\n",n.c_str(),(long long)bad,bad==0?"PASS":"FAIL");}
template<typename G> static void exec2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
  uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex)); void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); }
static std::vector<float> randv(int64_t n,float lo,float hi){std::vector<float> v(n);for(auto&x:v)x=lo+(hi-lo)*(rand()/(float)RAND_MAX);return v;}

// Cumsum along dim1 on [outer=4, L=8, inner=3] (dim=1)
static void t_cumsum(){
  const int64_t O=4,L=8,I=3; auto in=randv(O*L*I,-1,1); std::vector<float> hz(O*L*I);
  DevBuf da(O*L*I*4),dz(O*L*I*4); da.up(in.data());
  aclTensor*ta=mk({O,L,I},ACL_FLOAT,da.p),*tz=mk({O,L,I},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCumsumGetWorkspaceSize(ta,1,ACL_FLOAT,tz,w,e);},aclnnCumsum);
  dz.down(hz.data());
  double maxe=0;
  for(int64_t o=0;o<O;o++)for(int64_t i=0;i<I;i++){double acc=0;for(int64_t l=0;l<L;l++){acc+=in[(o*L+l)*I+i];double g=hz[(o*L+l)*I+i];maxe=std::max(maxe,std::fabs(g-acc)/(std::fabs(acc)+1e-9));}}
  report("Cumsum dim1 [4,8,3]",maxe,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_cumprod(){
  const int64_t O=3,L=6; auto in=randv(O*L,0.7,1.3); std::vector<float> hz(O*L);
  DevBuf da(O*L*4),dz(O*L*4); da.up(in.data());
  aclTensor*ta=mk({O,L},ACL_FLOAT,da.p),*tz=mk({O,L},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCumprodGetWorkspaceSize(ta,1,ACL_FLOAT,tz,w,e);},aclnnCumprod);
  dz.down(hz.data());
  double maxe=0;
  for(int64_t o=0;o<O;o++){double acc=1;for(int64_t l=0;l<L;l++){acc*=in[o*L+l];maxe=std::max(maxe,std::fabs(hz[o*L+l]-acc)/(std::fabs(acc)+1e-9));}}
  report("Cumprod dim1 [3,6]",maxe,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// Sort along dim1 descending on [3,7]
static void t_sort(){
  const int64_t M=3,N=7; auto in=randv(M*N,-2,2); for(int64_t i=0;i<M;i++) in[i*N+2]=in[i*N+5]; // introduce ties
  std::vector<float> hv(M*N); std::vector<int64_t> hi(M*N);
  DevBuf da(M*N*4),dv(M*N*4),di(M*N*8); da.up(in.data());
  aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tv=mk({M,N},ACL_FLOAT,dv.p),*ti=mk({M,N},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSortGetWorkspaceSize(ta,1,true,true,tv,ti,w,e);},aclnnSort);
  dv.down(hv.data()); di.down(hi.data());
  int64_t bad=0;
  for(int64_t r=0;r<M;r++){
    std::vector<int64_t> idx(N); for(int64_t j=0;j<N;j++) idx[j]=j;
    std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){float va=in[r*N+a],vb=in[r*N+b]; if(va==vb)return a<b; return va>vb;});
    for(int64_t j=0;j<N;j++){ if(hi[r*N+j]!=idx[j]) bad++; if(hv[r*N+j]!=in[r*N+idx[j]]) bad++; }
  }
  reportb("Sort dim1 desc [3,7]",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(ti);
}
// Topk k=3 along dim1 largest on [4,10]
static void t_topk(){
  const int64_t M=4,N=10,K=3; auto in=randv(M*N,-3,3);
  std::vector<float> hv(M*K); std::vector<int64_t> hi(M*K);
  DevBuf da(M*N*4),dv(M*K*4),di(M*K*8); da.up(in.data());
  aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tv=mk({M,K},ACL_FLOAT,dv.p),*ti=mk({M,K},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTopkGetWorkspaceSize(ta,K,1,true,true,tv,ti,w,e);},aclnnTopk);
  dv.down(hv.data()); di.down(hi.data());
  int64_t bad=0;
  for(int64_t r=0;r<M;r++){
    std::vector<int64_t> idx(N); for(int64_t j=0;j<N;j++) idx[j]=j;
    std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){float va=in[r*N+a],vb=in[r*N+b]; if(va==vb)return a<b; return va>vb;});
    for(int64_t j=0;j<K;j++){ if(hi[r*K+j]!=idx[j]) bad++; if(hv[r*K+j]!=in[r*N+idx[j]]) bad++; }
  }
  reportb("Topk k=3 dim1 [4,10]",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(ti);
}

// Large-L sort (bitonic path, non-power-of-2 L): Sort [2, 5000] desc + Topk k=8
static void t_sort_large(){
  const int64_t M=2,N=5000; auto in=randv(M*N,-5,5);
  std::vector<float> hv(M*N); std::vector<int64_t> hi(M*N);
  DevBuf da(M*N*4),dv(M*N*4),di(M*N*8); da.up(in.data());
  aclTensor*ta=mk({M,N},ACL_FLOAT,da.p),*tv=mk({M,N},ACL_FLOAT,dv.p),*ti=mk({M,N},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSortGetWorkspaceSize(ta,1,true,true,tv,ti,w,e);},aclnnSort);
  dv.down(hv.data()); di.down(hi.data());
  int64_t bad=0;
  for(int64_t r=0;r<M;r++){
    std::vector<int64_t> idx(N); for(int64_t j=0;j<N;j++) idx[j]=j;
    std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){float va=in[r*N+a],vb=in[r*N+b]; if(va==vb)return a<b; return va>vb;});
    for(int64_t j=0;j<N;j++){ if(hi[r*N+j]!=idx[j]) bad++; if(hv[r*N+j]!=in[r*N+idx[j]]) bad++; }
  }
  reportb("Sort large [2,5000] desc",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(ti);
}
// Multi-dim inner!=1 (falls back to O(L^2) path): Sort [4,6,5] along dim1 (inner=5)
static void t_sort_strided(){
  const int64_t O=4,L=6,I=5; auto in=randv(O*L*I,-2,2);
  std::vector<float> hv(O*L*I); std::vector<int64_t> hi(O*L*I);
  DevBuf da(O*L*I*4),dv(O*L*I*4),di(O*L*I*8); da.up(in.data());
  aclTensor*ta=mk({O,L,I},ACL_FLOAT,da.p),*tv=mk({O,L,I},ACL_FLOAT,dv.p),*ti=mk({O,L,I},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSortGetWorkspaceSize(ta,1,false,true,tv,ti,w,e);},aclnnSort);
  dv.down(hv.data()); di.down(hi.data());
  int64_t bad=0;
  for(int64_t o=0;o<O;o++)for(int64_t i=0;i<I;i++){
    std::vector<int64_t> idx(L); for(int64_t l=0;l<L;l++) idx[l]=l;
    std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){float va=in[(o*L+a)*I+i],vb=in[(o*L+b)*I+i]; if(va==vb)return a<b; return va<vb;});
    for(int64_t l=0;l<L;l++){ if(hi[(o*L+l)*I+i]!=idx[l]) bad++; if(hv[(o*L+l)*I+i]!=in[(o*L+idx[l])*I+i]) bad++; }
  }
  reportb("Sort strided [4,6,5] dim1 asc",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(ti);
}

// Cummax/Cummin/Logcumsumexp along dim1 of [O,L]
static void t_scans(){
  const int64_t O=5,L=12; auto in=randv(O*L,-2,2);
  std::vector<float> hmax(O*L),hmin(O*L),hlse(O*L); std::vector<int64_t> himax(O*L),himin(O*L);
  DevBuf da(O*L*4),dvmax(O*L*4),dvmin(O*L*4),dlse(O*L*4),dimax(O*L*8),dimin(O*L*8); da.up(in.data());
  aclTensor*ta=mk({O,L},ACL_FLOAT,da.p);
  aclTensor*tvmax=mk({O,L},ACL_FLOAT,dvmax.p),*timax=mk({O,L},ACL_INT64,dimax.p);
  aclTensor*tvmin=mk({O,L},ACL_FLOAT,dvmin.p),*timin=mk({O,L},ACL_INT64,dimin.p);
  aclTensor*tlse=mk({O,L},ACL_FLOAT,dlse.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCummaxGetWorkspaceSize(ta,1,tvmax,timax,w,e);},aclnnCummax);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCumminGetWorkspaceSize(ta,1,tvmin,timin,w,e);},aclnnCummin);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogcumsumexpGetWorkspaceSize(ta,1,tlse,w,e);},aclnnLogcumsumexp);
  dvmax.down(hmax.data()); dvmin.down(hmin.data()); dlse.down(hlse.data()); dimax.down(himax.data()); dimin.down(himin.data());
  double emax=0,emin=0,else_=0; int64_t ibad=0;
  for(int64_t o=0;o<O;o++){ float rmx=in[o*L],rmn=in[o*L]; int64_t imx=0,imn=0; double m=-1e300,se=0;
    for(int64_t l=0;l<L;l++){ float v=in[o*L+l];
      if(v>rmx){rmx=v;imx=l;} if(v<rmn){rmn=v;imn=l;}
      if(l==0){m=v;se=1;} else if(v>m){se=se*std::exp(m-v)+1;m=v;} else se+=std::exp(v-m);
      emax=std::max(emax,(double)std::fabs(hmax[o*L+l]-rmx)); emin=std::max(emin,(double)std::fabs(hmin[o*L+l]-rmn));
      double lse=m+std::log(se); else_=std::max(else_,std::fabs(hlse[o*L+l]-lse)/(std::fabs(lse)+1e-9));
      if(himax[o*L+l]!=imx||himin[o*L+l]!=imn) ibad++; } }
  report("Cummax dim1 [5,12]",emax,0); report("Cummin dim1 [5,12]",emin,0);
  reportb("Cummax/min indices",ibad); report("Logcumsumexp dim1 [5,12]",else_,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tvmax);aclDestroyTensor(timax);aclDestroyTensor(tvmin);aclDestroyTensor(timin);aclDestroyTensor(tlse);
}

int main(){
  CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(5);
  t_cumsum(); t_cumprod(); t_sort(); t_topk(); t_sort_large(); t_sort_strided(); t_scans();
  CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
  printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
  return g_fail?1:0;
}
