// Self-contained cross-check for the 39 index/shape/reduce/sort gap operators (CUDA parity).
// Each op has a CPU reference (or equivalence to a base op). Index/integer ops are exact (tol 0);
// float reductions use 1e-5/1e-6. Prototypes are self-declared extern "C" matching index2_parity.mm.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>

#include "acl/acl.h"
#include "aclnn/acl_meta.h"

// ---- self-declared op prototypes (none in aclnnop/aclnn_ops.h) ----
extern "C" {
typedef struct aclTensor aclTensor;
typedef struct aclScalar aclScalar;
typedef struct aclIntArray aclIntArray;
typedef struct aclTensorList aclTensorList;
typedef struct aclOpExecutor aclOpExecutor;

// index/shape
aclnnStatus aclnnIndexSelectGetWorkspaceSize(const aclTensor*, int64_t, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIndexSelect(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIndexGetWorkspaceSize(const aclTensor*, const aclTensorList*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIndex(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnGatherV3GetWorkspaceSize(const aclTensor*, int64_t, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGatherV3(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIndexAddV2GetWorkspaceSize(const aclTensor*, int64_t, const aclTensor*, const aclTensor*, double, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIndexAddV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIndexFillTensorGetWorkspaceSize(const aclTensor*, int64_t, const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIndexFillTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnScatterNdGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnScatterNd(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnScatterListGetWorkspaceSize(aclTensorList*, const aclTensorList*, const aclTensorList*, const aclTensor*, int64_t, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnScatterList(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnIndexPutImplGetWorkspaceSize(aclTensor*, const aclTensorList*, const aclTensor*, bool, bool, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnIndexPutImpl(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnScatterPaCacheGetWorkspaceSize(const aclTensor*, aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnScatterPaCache(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnScatterPaKvCacheGetWorkspaceSize(const aclTensor*, const aclTensor*, aclTensor*, aclTensor*, const aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnScatterPaKvCache(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMaskedScaleGetWorkspaceSize(const aclTensor*, const aclTensor*, double, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMaskedScale(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnExpandvGetWorkspaceSize(const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnExpandv(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRepeatGetWorkspaceSize(const aclTensor*, const aclIntArray*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRepeat(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDiagGetWorkspaceSize(const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDiag(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRepeatInterleaveIntGetWorkspaceSize(const aclTensor*, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRepeatInterleaveInt(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRepeatInterleaveWithDimGetWorkspaceSize(const aclTensor*, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRepeatInterleaveWithDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(const aclTensor*, int64_t, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRepeatInterleaveIntWithDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnRepeatInterleaveTensorGetWorkspaceSize(const aclTensor*, const aclTensor*, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnRepeatInterleaveTensor(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnChunkGetWorkspaceSize(const aclTensor*, int64_t, int64_t, const aclTensor* const*, uint64_t, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnChunk(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnChunkCatGetWorkspaceSize(const aclTensor*, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnChunkCat(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnFlattenGetWorkspaceSize(const aclTensor*, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnFlatten(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnSliceV2GetWorkspaceSize(const aclTensor*, int64_t, int64_t, int64_t, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnSliceV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnStridedSliceAssignV2GetWorkspaceSize(aclTensor*, const aclTensor*, int64_t, int64_t, int64_t, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnStridedSliceAssignV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnDenseLightningIndexerSoftmaxLse(void*, uint64_t, aclOpExecutor*, aclrtStream);

// reduce/sort
aclnnStatus aclnnAminmaxAllGetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAminmaxAll(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnAminmaxDimGetWorkspaceSize(const aclTensor*, int64_t, bool, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnAminmaxDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnReduceNansumGetWorkspaceSize(const aclTensor*, const aclIntArray*, bool, aclDataType, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnReduceNansum(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnStdMeanCorrectionGetWorkspaceSize(const aclTensor*, const aclIntArray*, int64_t, bool, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnStdMeanCorrection(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnCumsumV2GetWorkspaceSize(const aclTensor*, int64_t, aclDataType, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnCumsumV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnMedianDimGetWorkspaceSize(const aclTensor*, int64_t, bool, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnMedianDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnNanMedianGetWorkspaceSize(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnNanMedian(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnNanMedianDimGetWorkspaceSize(const aclTensor*, int64_t, bool, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnNanMedianDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnArgsortGetWorkspaceSize(const aclTensor*, int64_t, bool, bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnArgsort(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnBucketizeGetWorkspaceSize(const aclTensor*, const aclTensor*, bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnBucketize(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnUnique2GetWorkspaceSize(const aclTensor*, bool, bool, bool, aclTensor*, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnUnique2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnUniqueDimGetWorkspaceSize(const aclTensor*, int64_t, bool, bool, bool, aclTensor*, aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnUniqueDim(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnTopKTopPSampleGetWorkspaceSize(const aclTensor*, int64_t, double, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnTopKTopPSample(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnTopKTopPSampleV2GetWorkspaceSize(const aclTensor*, int64_t, double, int64_t, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnTopKTopPSampleV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
aclnnStatus aclnnNonzeroV2GetWorkspaceSize(const aclTensor*, aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnNonzeroV2(void*, uint64_t, aclOpExecutor*, aclrtStream);
}

#define CHECK(x) do { int __r=(int)(x); if(__r!=0){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,__r);exit(1);} } while(0)
static aclrtStream g_stream; static int g_pass=0,g_fail=0;
struct DevBuf{ void*p=nullptr; size_t bytes=0; DevBuf(size_t b):bytes(b){CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST));}
  ~DevBuf(){aclrtFree(p);} void up(const void*h){CHECK(aclrtMemcpy(p,bytes,h,bytes,ACL_MEMCPY_HOST_TO_DEVICE));}
  void down(void*h)const{CHECK(aclrtMemcpy(h,bytes,p,bytes,ACL_MEMCPY_DEVICE_TO_HOST));} };
static aclTensor*mk(const std::vector<int64_t>&d,aclDataType dt,void*data){return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),data);}
static void report(const std::string&n,double err,double tol){bool ok=err<=tol;(ok?g_pass:g_fail)++;printf("%-32s err=%.2e tol=%.0e %s\n",n.c_str(),err,tol,ok?"PASS":"FAIL");}
static void reportb(const std::string&n,int64_t bad){(bad==0?g_pass:g_fail)++;printf("%-32s mismatch=%lld %s\n",n.c_str(),(long long)bad,bad==0?"PASS":"FAIL");}
template<typename G> static void exec2(G getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
  uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex)); void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
  CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); }
static std::vector<float> randv(int64_t n,float lo,float hi){std::vector<float> v(n);for(auto&x:v)x=lo+(hi-lo)*(rand()/(float)RAND_MAX);return v;}

// ============================== INDEX / SHAPE ==============================

// IndexSelect along dim1: out[r,j] = self[r, idx[j]]
static void t_index_select(){
  const int64_t R=4,N=6,K=3; auto in=randv(R*N,-2,2); int64_t idx[K]={5,1,3}; std::vector<float> ho(R*K);
  DevBuf da(R*N*4),di(K*8),dz(R*K*4); da.up(in.data()); di.up(idx);
  auto*ta=mk({R,N},ACL_FLOAT,da.p),*tidx=mk({K},ACL_INT64,di.p),*tz=mk({R,K},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexSelectGetWorkspaceSize(ta,1,tidx,tz,w,e);},aclnnIndexSelect);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<R;r++)for(int64_t j=0;j<K;j++) e=std::max(e,(double)std::fabs(ho[r*K+j]-in[r*N+idx[j]]));
  report("IndexSelect dim1",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tidx);aclDestroyTensor(tz);
}
// Index (advanced, dim0): out[k]=self[idx[k]]
static void t_index(){
  const int64_t V=5,D=3,K=4; auto in=randv(V*D,-2,2); int64_t idx[K]={3,0,4,1}; std::vector<float> ho(K*D);
  DevBuf da(V*D*4),di(K*8),dz(K*D*4); da.up(in.data()); di.up(idx);
  auto*ta=mk({V,D},ACL_FLOAT,da.p),*tidx=mk({K},ACL_INT64,di.p),*tz=mk({K,D},ACL_FLOAT,dz.p);
  aclTensorList* lst=aclCreateTensorList(&tidx,1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexGetWorkspaceSize(ta,lst,tz,w,e);},aclnnIndex);
  dz.down(ho.data()); double e=0; for(int64_t k=0;k<K;k++)for(int64_t d=0;d<D;d++) e=std::max(e,(double)std::fabs(ho[k*D+d]-in[idx[k]*D+d]));
  report("Index advanced dim0",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// GatherV3 == Gather along dim1
static void t_gatherv3(){
  const int64_t R=3,N=5,K=2; auto in=randv(R*N,-2,2); int64_t idx[K]={4,2}; std::vector<float> ho(R*K);
  DevBuf da(R*N*4),di(K*8),dz(R*K*4); da.up(in.data()); di.up(idx);
  auto*ta=mk({R,N},ACL_FLOAT,da.p),*tidx=mk({K},ACL_INT64,di.p),*tz=mk({R,K},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherV3GetWorkspaceSize(ta,1,tidx,tz,w,e);},aclnnGatherV3);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<R;r++)for(int64_t j=0;j<K;j++) e=std::max(e,(double)std::fabs(ho[r*K+j]-in[r*N+idx[j]]));
  report("GatherV3 dim1",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tidx);aclDestroyTensor(tz);
}
// IndexAddV2: out = self; out[idx[l],:] += alpha*src[l,:]
static void t_index_add_v2(){
  const int64_t V=5,D=3,L=3; auto self=randv(V*D,-1,1),src=randv(L*D,-1,1); int64_t idx[L]={0,2,2}; double al=0.5; std::vector<float> ho(V*D);
  DevBuf ds(V*D*4),dsrc(L*D*4),di(L*8),dz(V*D*4); ds.up(self.data()); dsrc.up(src.data()); di.up(idx);
  auto*ta=mk({V,D},ACL_FLOAT,ds.p),*tsrc=mk({L,D},ACL_FLOAT,dsrc.p),*tidx=mk({L},ACL_INT64,di.p),*tz=mk({V,D},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexAddV2GetWorkspaceSize(ta,0,tidx,tsrc,al,tz,w,e);},aclnnIndexAddV2);
  dz.down(ho.data()); auto ref=self; for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++) ref[idx[l]*D+d]+=al*src[l*D+d];
  double e=0; for(int64_t i=0;i<V*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("IndexAddV2 dim0",e,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tsrc);aclDestroyTensor(tidx);aclDestroyTensor(tz);
}
// IndexFillTensor: out[idx[l],:] = value (single-element tensor)
static void t_index_fill_tensor(){
  const int64_t V=5,D=3,L=2; auto self=randv(V*D,-1,1); int64_t idx[L]={1,4}; float val=7.0f; std::vector<float> ho(V*D);
  DevBuf ds(V*D*4),dval(4),di(L*8),dz(V*D*4); ds.up(self.data()); dval.up(&val); di.up(idx);
  auto*ta=mk({V,D},ACL_FLOAT,ds.p),*tval=mk({1},ACL_FLOAT,dval.p),*tidx=mk({L},ACL_INT64,di.p),*tz=mk({V,D},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexFillTensorGetWorkspaceSize(ta,0,tidx,tval,tz,w,e);},aclnnIndexFillTensor);
  dz.down(ho.data()); auto ref=self; for(int64_t l=0;l<L;l++)for(int64_t d=0;d<D;d++) ref[idx[l]*D+d]=val;
  double e=0; for(int64_t i=0;i<V*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("IndexFillTensor dim0",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tval);aclDestroyTensor(tidx);aclDestroyTensor(tz);
}
// ScatterNd: out[idx[r]] = upd[r] into zero tensor (2D out [4,3], K=1 coord)
static void t_scatternd(){
  const int64_t Rows=2,M=4,D=3; std::vector<float> upd=randv(Rows*D,-1,1); int64_t idx[Rows*1]={3,1}; std::vector<float> ho(M*D);
  DevBuf di(Rows*8),du(Rows*D*4),dz(M*D*4); di.up(idx); du.up(upd.data());
  auto*tidx=mk({Rows,1},ACL_INT64,di.p),*tupd=mk({Rows,D},ACL_FLOAT,du.p),*tz=mk({M,D},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterNdGetWorkspaceSize(tidx,tupd,tz,w,e);},aclnnScatterNd);
  dz.down(ho.data()); std::vector<float> ref(M*D,0.f); for(int64_t r=0;r<Rows;r++)for(int64_t d=0;d<D;d++) ref[idx[r]*D+d]=upd[r*D+d];
  double e=0; for(int64_t i=0;i<M*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("ScatterNd (zero+scatter)",e,0);
  aclDestroyTensor(tidx);aclDestroyTensor(tupd);aclDestroyTensor(tz);
}
// ScatterList: self[idx[k],:] = upd[k,:] (in place), single tensor in lists
static void t_scatter_list(){
  const int64_t N=4,D=3,K=2; auto self=randv(N*D,-1,1),upd=randv(K*D,-1,1); int64_t idx[K]={0,3}; std::vector<float> ho(N*D);
  DevBuf ds(N*D*4),di(K*8),du(K*D*4); ds.up(self.data()); di.up(idx); du.up(upd.data());
  auto*ts=mk({N,D},ACL_FLOAT,ds.p),*ti=mk({K},ACL_INT64,di.p),*tu=mk({K,D},ACL_FLOAT,du.p);
  aclTensorList *sl=aclCreateTensorList(&ts,1),*il=aclCreateTensorList(&ti,1),*ul=aclCreateTensorList(&tu,1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterListGetWorkspaceSize(sl,il,ul,nullptr,0,w,e);},aclnnScatterList);
  ds.down(ho.data()); auto ref=self; for(int64_t k=0;k<K;k++)for(int64_t d=0;d<D;d++) ref[idx[k]*D+d]=upd[k*D+d];
  double e=0; for(int64_t i=0;i<N*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("ScatterList (overwrite)",e,0);
}
// IndexPutImpl: selfRef[idx[k],:] += values[k,:] (accumulate=true)
static void t_index_put(){
  const int64_t N=4,D=3,K=3; auto self=randv(N*D,-1,1),vals=randv(K*D,-1,1); int64_t idx[K]={1,1,3}; std::vector<float> ho(N*D);
  DevBuf ds(N*D*4),di(K*8),dv(K*D*4); ds.up(self.data()); di.up(idx); dv.up(vals.data());
  auto*ts=mk({N,D},ACL_FLOAT,ds.p),*ti=mk({K},ACL_INT64,di.p),*tv=mk({K,D},ACL_FLOAT,dv.p);
  aclTensorList* il=aclCreateTensorList(&ti,1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexPutImplGetWorkspaceSize(ts,il,tv,true,false,w,e);},aclnnIndexPutImpl);
  ds.down(ho.data()); auto ref=self; for(int64_t k=0;k<K;k++)for(int64_t d=0;d<D;d++) ref[idx[k]*D+d]+=vals[k*D+d];
  double e=0; for(int64_t i=0;i<N*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("IndexPutImpl (accum)",e,1e-5);
}
// ScatterPaCache: cache[slot[k],:] = input[k,:]
static void t_scatter_pa_cache(){
  const int64_t N=5,D=4,K=3; auto cache=randv(N*D,-1,1),input=randv(K*D,-1,1); int64_t slot[K]={4,0,2}; std::vector<float> ho(N*D);
  DevBuf dc(N*D*4),di(K*D*4),ds(K*8); dc.up(cache.data()); di.up(input.data()); ds.up(slot);
  auto*tc=mk({N,D},ACL_FLOAT,dc.p),*tin=mk({K,D},ACL_FLOAT,di.p),*tsl=mk({K},ACL_INT64,ds.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterPaCacheGetWorkspaceSize(tin,tc,tsl,w,e);},aclnnScatterPaCache);
  dc.down(ho.data()); auto ref=cache; for(int64_t k=0;k<K;k++)for(int64_t d=0;d<D;d++) ref[slot[k]*D+d]=input[k*D+d];
  double e=0; for(int64_t i=0;i<N*D;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("ScatterPaCache",e,0);
  aclDestroyTensor(tc);aclDestroyTensor(tin);aclDestroyTensor(tsl);
}
// ScatterPaKvCache: keyCache[slot]/valueCache[slot] = key/value
static void t_scatter_pa_kv(){
  const int64_t N=5,D=4,K=2; auto kc=randv(N*D,-1,1),vc=randv(N*D,-1,1),key=randv(K*D,-1,1),val=randv(K*D,-1,1); int64_t slot[K]={1,4};
  std::vector<float> hk(N*D),hv(N*D);
  DevBuf dkc(N*D*4),dvc(N*D*4),dk(K*D*4),dv(K*D*4),ds(K*8); dkc.up(kc.data());dvc.up(vc.data());dk.up(key.data());dv.up(val.data());ds.up(slot);
  auto*tkc=mk({N,D},ACL_FLOAT,dkc.p),*tvc=mk({N,D},ACL_FLOAT,dvc.p),*tk=mk({K,D},ACL_FLOAT,dk.p),*tv=mk({K,D},ACL_FLOAT,dv.p),*tsl=mk({K},ACL_INT64,ds.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterPaKvCacheGetWorkspaceSize(tk,tv,tkc,tvc,tsl,w,e);},aclnnScatterPaKvCache);
  dkc.down(hk.data()); dvc.down(hv.data()); auto rk=kc,rv=vc; for(int64_t k=0;k<K;k++)for(int64_t d=0;d<D;d++){rk[slot[k]*D+d]=key[k*D+d];rv[slot[k]*D+d]=val[k*D+d];}
  double e=0; for(int64_t i=0;i<N*D;i++){e=std::max(e,(double)std::fabs(hk[i]-rk[i]));e=std::max(e,(double)std::fabs(hv[i]-rv[i]));}
  report("ScatterPaKvCache",e,0);
}
// MaskedScale: out = self * mask * scale
static void t_masked_scale(){
  const int64_t N=12; auto self=randv(N,-2,2); std::vector<uint8_t> mask(N); for(int64_t i=0;i<N;i++) mask[i]=(i%3==0)?1:0; double sc=2.5; std::vector<float> ho(N);
  DevBuf ds(N*4),dm(N),dz(N*4); ds.up(self.data()); dm.up(mask.data());
  auto*ts=mk({N},ACL_FLOAT,ds.p),*tm=mk({N},ACL_UINT8,dm.p),*tz=mk({N},ACL_FLOAT,dz.p);   // mask is a uint8 dropout mask
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedScaleGetWorkspaceSize(ts,tm,sc,tz,w,e);},aclnnMaskedScale);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<N;i++) e=std::max(e,(double)std::fabs(ho[i]-self[i]*(float)mask[i]*(float)sc)); report("MaskedScale",e,1e-5);
  aclDestroyTensor(ts);aclDestroyTensor(tm);aclDestroyTensor(tz);
}
// Expandv: out[i] = in[i % inN] (flat broadcast-tile)
static void t_expandv(){
  const int64_t inN=4,rep=3,outN=inN*rep; auto in=randv(inN,-2,2); std::vector<float> ho(outN);
  DevBuf di(inN*4),dz(outN*4); di.up(in.data());
  auto*ta=mk({inN},ACL_FLOAT,di.p),*tz=mk({outN},ACL_FLOAT,dz.p);
  std::vector<int64_t> szv={outN}; aclIntArray* sz=aclCreateIntArray(szv.data(),1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnExpandvGetWorkspaceSize(ta,sz,tz,w,e);},aclnnExpandv);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<outN;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i%inN])); report("Expandv",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// Repeat (per-dim tile): [2,3] repeated x[2,2] -> [4,6]
static void t_repeat(){
  const int64_t M=2,N=3,rM=2,rN=2; auto in=randv(M*N,-2,2); int64_t oM=M*rM,oN=N*rN; std::vector<float> ho(oM*oN);
  DevBuf di(M*N*4),dz(oM*oN*4); di.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,di.p),*tz=mk({oM,oN},ACL_FLOAT,dz.p);
  std::vector<int64_t> rv={rM,rN}; aclIntArray* rep=aclCreateIntArray(rv.data(),2);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatGetWorkspaceSize(ta,rep,tz,w,e);},aclnnRepeat);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<oM;r++)for(int64_t c=0;c<oN;c++) e=std::max(e,(double)std::fabs(ho[r*oN+c]-in[(r%M)*N+(c%N)]));
  report("Repeat (tile)",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// Diag: 1D->diagflat (matrix) and 2D->diagonal (vector)
static void t_diag(){
  const int64_t L=4; auto v=randv(L,-2,2); std::vector<float> hm(L*L);
  DevBuf dv(L*4),dm(L*L*4); dv.up(v.data());
  auto*tv=mk({L},ACL_FLOAT,dv.p),*tm=mk({L,L},ACL_FLOAT,dm.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDiagGetWorkspaceSize(tv,0,tm,w,e);},aclnnDiag);
  dm.down(hm.data()); double e=0; for(int64_t r=0;r<L;r++)for(int64_t c=0;c<L;c++) e=std::max(e,(double)std::fabs(hm[r*L+c]-((r==c)?v[r]:0.f)));
  report("Diag (diagflat 1D)",e,0);
  // 2D->vector
  const int64_t M=3,N=4; auto mat=randv(M*N,-2,2); int64_t len=std::min(M,N); std::vector<float> hd(len);
  DevBuf dmat(M*N*4),dd(len*4); dmat.up(mat.data());
  auto*tmat=mk({M,N},ACL_FLOAT,dmat.p),*td=mk({len},ACL_FLOAT,dd.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDiagGetWorkspaceSize(tmat,0,td,w,e);},aclnnDiag);
  dd.down(hd.data()); double e2=0; for(int64_t i=0;i<len;i++) e2=std::max(e2,(double)std::fabs(hd[i]-mat[i*N+i]));
  report("Diag (diagonal 2D)",e2,0);
  aclDestroyTensor(tv);aclDestroyTensor(tm);aclDestroyTensor(tmat);aclDestroyTensor(td);
}
// RepeatInterleaveInt: out[i] = in[i/rep]
static void t_repeat_il_int(){
  const int64_t N=4,rep=3,outN=N*rep; auto in=randv(N,-2,2); std::vector<float> ho(outN);
  DevBuf di(N*4),dz(outN*4); di.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,di.p),*tz=mk({outN},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatInterleaveIntGetWorkspaceSize(ta,rep,outN,tz,w,e);},aclnnRepeatInterleaveInt);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<outN;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i/rep])); report("RepeatInterleaveInt",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_repeat_il_withdim(){
  const int64_t N=5,rep=2,outN=N*rep; auto in=randv(N,-2,2); std::vector<float> ho(outN);
  DevBuf di(N*4),dz(outN*4); di.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,di.p),*tz=mk({outN},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatInterleaveWithDimGetWorkspaceSize(ta,rep,0,tz,w,e);},aclnnRepeatInterleaveWithDim);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<outN;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i/rep])); report("RepeatInterleaveWithDim",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_repeat_il_intwithdim(){
  const int64_t N=3,rep=4,outN=N*rep; auto in=randv(N,-2,2); std::vector<float> ho(outN);
  DevBuf di(N*4),dz(outN*4); di.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,di.p),*tz=mk({outN},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatInterleaveIntWithDimGetWorkspaceSize(ta,rep,0,outN,tz,w,e);},aclnnRepeatInterleaveIntWithDim);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<outN;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i/rep])); report("RepeatInterleaveIntWithDim",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_repeat_il_tensor(){
  const int64_t N=4,rep=2,outN=N*rep; auto in=randv(N,-2,2); std::vector<float> ho(outN); int64_t repv[N]={rep,rep,rep,rep};
  DevBuf di(N*4),dr(N*8),dz(outN*4); di.up(in.data()); dr.up(repv);
  auto*ta=mk({N},ACL_FLOAT,di.p),*trep=mk({N},ACL_INT64,dr.p),*tz=mk({outN},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRepeatInterleaveTensorGetWorkspaceSize(ta,trep,outN,tz,w,e);},aclnnRepeatInterleaveTensor);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<outN;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i/rep])); report("RepeatInterleaveTensor",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(trep);aclDestroyTensor(tz);
}
// Chunk: split [6,3] along dim0 into [2,3]+[2,3]+[2,3]
static void t_chunk(){
  const int64_t M=6,N=3,C=3,cm=M/C; auto in=randv(M*N,-2,2);
  DevBuf da(M*N*4),d0(cm*N*4),d1(cm*N*4),d2(cm*N*4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*t0=mk({cm,N},ACL_FLOAT,d0.p),*t1=mk({cm,N},ACL_FLOAT,d1.p),*t2=mk({cm,N},ACL_FLOAT,d2.p);
  const aclTensor* outs[3]={t0,t1,t2};
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnChunkGetWorkspaceSize(ta,C,0,outs,3,w,e);},aclnnChunk);
  std::vector<float> h0(cm*N),h1(cm*N),h2(cm*N); d0.down(h0.data());d1.down(h1.data());d2.down(h2.data());
  double e=0; for(int64_t i=0;i<cm*N;i++){e=std::max(e,(double)std::fabs(h0[i]-in[i]));e=std::max(e,(double)std::fabs(h1[i]-in[cm*N+i]));e=std::max(e,(double)std::fabs(h2[i]-in[2*cm*N+i]));}
  report("Chunk dim0",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(t0);aclDestroyTensor(t1);aclDestroyTensor(t2);
}
// ChunkCat: identity copy (layout cast)
static void t_chunkcat(){
  const int64_t N=12; auto in=randv(N,-2,2); std::vector<float> ho(N);
  DevBuf di(N*4),dz(N*4); di.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,di.p),*tz=mk({N},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnChunkCatGetWorkspaceSize(ta,3,0,tz,w,e);},aclnnChunkCat);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<N;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i])); report("ChunkCat (copy)",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// Flatten: identity copy
static void t_flatten(){
  const int64_t M=3,N=4; auto in=randv(M*N,-2,2); std::vector<float> ho(M*N);
  DevBuf di(M*N*4),dz(M*N*4); di.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,di.p),*tz=mk({M*N},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFlattenGetWorkspaceSize(ta,0,1,tz,w,e);},aclnnFlatten);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<M*N;i++) e=std::max(e,(double)std::fabs(ho[i]-in[i])); report("Flatten (copy)",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// SliceV2: self[1:7:2] along dim0 of [8] -> [3]
static void t_slice_v2(){
  const int64_t N=8; auto in=randv(N,-2,2); int64_t st=1,en=7,step=2; int64_t len=(en-st+step-1)/step; std::vector<float> ho(len);
  DevBuf di(N*4),dz(len*4); di.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,di.p),*tz=mk({len},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSliceV2GetWorkspaceSize(ta,0,st,en,step,tz,w,e);},aclnnSliceV2);
  dz.down(ho.data()); double e=0; for(int64_t i=0;i<len;i++) e=std::max(e,(double)std::fabs(ho[i]-in[st+i*step])); report("SliceV2 [1:7:2]",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// StridedSliceAssignV2: self[begin:begin+vn] = value (contiguous)
static void t_slice_assign(){
  const int64_t N=8,vn=3,begin=2; auto self=randv(N,-2,2),val=randv(vn,5,6); std::vector<float> ho(N);
  DevBuf ds(N*4),dv(vn*4); ds.up(self.data()); dv.up(val.data());
  auto*ts=mk({N},ACL_FLOAT,ds.p),*tv=mk({vn},ACL_FLOAT,dv.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnStridedSliceAssignV2GetWorkspaceSize(ts,tv,begin,begin+vn,1,w,e);},aclnnStridedSliceAssignV2);
  ds.down(ho.data()); auto ref=self; for(int64_t i=0;i<vn;i++) ref[begin+i]=val[i];
  double e=0; for(int64_t i=0;i<N;i++) e=std::max(e,(double)std::fabs(ho[i]-ref[i])); report("StridedSliceAssignV2",e,0);
  aclDestroyTensor(ts);aclDestroyTensor(tv);
}
// DenseLightningIndexerSoftmaxLse: lse[q]=log Σ_k exp(score[q,k]); probs row-softmax
static void t_lse(){
  const int64_t Q=3,K=5; auto sc=randv(Q*K,-2,2); std::vector<float> hl(Q),hp(Q*K);
  DevBuf ds(Q*K*4),dl(Q*4),dp(Q*K*4); ds.up(sc.data());
  auto*ta=mk({Q,K},ACL_FLOAT,ds.p),*tl=mk({Q},ACL_FLOAT,dl.p),*tp=mk({Q,K},ACL_FLOAT,dp.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDenseLightningIndexerSoftmaxLseGetWorkspaceSize(ta,tl,tp,w,e);},aclnnDenseLightningIndexerSoftmaxLse);
  dl.down(hl.data()); dp.down(hp.data()); double e=0;
  for(int64_t q=0;q<Q;q++){ double mx=-1e30; for(int64_t k=0;k<K;k++) mx=std::max(mx,(double)sc[q*K+k]); double sm=0; for(int64_t k=0;k<K;k++) sm+=std::exp(sc[q*K+k]-mx);
    double lse=mx+std::log(sm); e=std::max(e,std::fabs(hl[q]-lse)/(std::fabs(lse)+1e-9)); for(int64_t k=0;k<K;k++) e=std::max(e,std::fabs(hp[q*K+k]-std::exp(sc[q*K+k]-mx)/sm)); }
  report("DenseLightningSoftmaxLse",e,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tl);aclDestroyTensor(tp);
}

// ============================== REDUCE / SORT ==============================

// AminmaxAll: global min/max over [4,5]
static void t_aminmax_all(){
  const int64_t M=4,N=5; auto in=randv(M*N,-3,3); float hmn,hmx;
  DevBuf da(M*N*4),dmn(4),dmx(4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tmn=mk({1},ACL_FLOAT,dmn.p),*tmx=mk({1},ACL_FLOAT,dmx.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAminmaxAllGetWorkspaceSize(ta,tmn,tmx,w,e);},aclnnAminmaxAll);
  dmn.down(&hmn); dmx.down(&hmx); float rmn=in[0],rmx=in[0]; for(auto v:in){rmn=std::min(rmn,v);rmx=std::max(rmx,v);}
  report("AminmaxAll",std::max(std::fabs(hmn-rmn),std::fabs(hmx-rmx)),1e-6);
  aclDestroyTensor(ta);aclDestroyTensor(tmn);aclDestroyTensor(tmx);
}
// AminmaxDim: min/max along dim1 of [3,6] -> [3]
static void t_aminmax_dim(){
  const int64_t M=3,N=6; auto in=randv(M*N,-3,3); std::vector<float> hmn(M),hmx(M);
  DevBuf da(M*N*4),dmn(M*4),dmx(M*4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tmn=mk({M},ACL_FLOAT,dmn.p),*tmx=mk({M},ACL_FLOAT,dmx.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAminmaxDimGetWorkspaceSize(ta,1,false,tmn,tmx,w,e);},aclnnAminmaxDim);
  dmn.down(hmn.data()); dmx.down(hmx.data()); double e=0;
  for(int64_t r=0;r<M;r++){ float rmn=in[r*N],rmx=in[r*N]; for(int64_t c=0;c<N;c++){rmn=std::min(rmn,in[r*N+c]);rmx=std::max(rmx,in[r*N+c]);} e=std::max(e,(double)std::fabs(hmn[r]-rmn)); e=std::max(e,(double)std::fabs(hmx[r]-rmx)); }
  report("AminmaxDim dim1",e,1e-6);
  aclDestroyTensor(ta);aclDestroyTensor(tmn);aclDestroyTensor(tmx);
}
// ReduceNansum along dim1 (with a NaN injected)
static void t_reduce_nansum(){
  const int64_t M=3,N=5; auto in=randv(M*N,-2,2); in[1*N+2]=NAN; std::vector<float> ho(M);
  DevBuf da(M*N*4),dz(M*4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M},ACL_FLOAT,dz.p);
  std::vector<int64_t> dv={1}; aclIntArray* dim=aclCreateIntArray(dv.data(),1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReduceNansumGetWorkspaceSize(ta,dim,false,ACL_FLOAT,tz,w,e);},aclnnReduceNansum);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<M;r++){ double s=0; for(int64_t c=0;c<N;c++) if(!std::isnan(in[r*N+c])) s+=in[r*N+c]; e=std::max(e,std::fabs(ho[r]-s)); }
  report("ReduceNansum dim1",e,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// StdMeanCorrection along dim1 of [4,8] (unbiased ddof=1)
static void t_stdmean(){
  const int64_t M=4,N=8; auto in=randv(M*N,-2,2); std::vector<float> hstd(M),hmean(M);
  DevBuf da(M*N*4),dstd(M*4),dmean(M*4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tstd=mk({M},ACL_FLOAT,dstd.p),*tmean=mk({M},ACL_FLOAT,dmean.p);
  std::vector<int64_t> dv={1}; aclIntArray* dim=aclCreateIntArray(dv.data(),1);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnStdMeanCorrectionGetWorkspaceSize(ta,dim,1,false,tstd,tmean,w,e);},aclnnStdMeanCorrection);
  dstd.down(hstd.data()); dmean.down(hmean.data()); double e=0;
  for(int64_t r=0;r<M;r++){ double s=0,sq=0; for(int64_t c=0;c<N;c++){s+=in[r*N+c];sq+=(double)in[r*N+c]*in[r*N+c];} double mean=s/N,var=(sq-s*s/N)/(N-1),std=std::sqrt(var);
    e=std::max(e,std::fabs(hmean[r]-mean)/(std::fabs(mean)+1e-6)); e=std::max(e,std::fabs(hstd[r]-std)/(std::fabs(std)+1e-6)); }
  report("StdMeanCorrection dim1",e,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tstd);aclDestroyTensor(tmean);
}
// CumsumV2 along dim1 of [4,7]
static void t_cumsum_v2(){
  const int64_t M=4,N=7; auto in=randv(M*N,-1,1); std::vector<float> ho(M*N);
  DevBuf da(M*N*4),dz(M*N*4); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCumsumV2GetWorkspaceSize(ta,1,ACL_FLOAT,tz,w,e);},aclnnCumsumV2);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<M;r++){ double acc=0; for(int64_t c=0;c<N;c++){acc+=in[r*N+c]; e=std::max(e,std::fabs(ho[r*N+c]-acc)/(std::fabs(acc)+1e-9));} }
  report("CumsumV2 dim1",e,1e-5);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// MedianDim: lower-median along dim1 of [3,7]
static void t_median_dim(){
  const int64_t M=3,N=7; auto in=randv(M*N,-3,3); std::vector<float> ho(M); std::vector<int64_t> hi(M);
  DevBuf da(M*N*4),dz(M*4),di(M*8); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M},ACL_FLOAT,dz.p),*tidx=mk({M},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMedianDimGetWorkspaceSize(ta,1,false,tz,tidx,w,e);},aclnnMedianDim);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<M;r++){ std::vector<float> v(in.begin()+r*N,in.begin()+r*N+N); std::sort(v.begin(),v.end()); e=std::max(e,(double)std::fabs(ho[r]-v[(N-1)/2])); }
  report("MedianDim dim1",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);aclDestroyTensor(tidx);
}
// NanMedian: lower-median over whole tensor
static void t_nan_median(){
  const int64_t N=11; auto in=randv(N,-3,3); float ho;
  DevBuf da(N*4),dz(4); da.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,da.p),*tz=mk({1},ACL_FLOAT,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNanMedianGetWorkspaceSize(ta,tz,w,e);},aclnnNanMedian);
  dz.down(&ho); auto v=in; std::sort(v.begin(),v.end()); report("NanMedian (all)",std::fabs(ho-v[(N-1)/2]),0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
// NanMedianDim: along dim1 of [2,9]
static void t_nan_median_dim(){
  const int64_t M=2,N=9; auto in=randv(M*N,-3,3); std::vector<float> ho(M); std::vector<int64_t> hi(M);
  DevBuf da(M*N*4),dz(M*4),di(M*8); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tz=mk({M},ACL_FLOAT,dz.p),*tidx=mk({M},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNanMedianDimGetWorkspaceSize(ta,1,false,tz,tidx,w,e);},aclnnNanMedianDim);
  dz.down(ho.data()); double e=0; for(int64_t r=0;r<M;r++){ std::vector<float> v(in.begin()+r*N,in.begin()+r*N+N); std::sort(v.begin(),v.end()); e=std::max(e,(double)std::fabs(ho[r]-v[(N-1)/2])); }
  report("NanMedianDim dim1",e,0);
  aclDestroyTensor(ta);aclDestroyTensor(tz);aclDestroyTensor(tidx);
}
// Argsort descending along dim1 of [3,7] (with ties)
static void t_argsort(){
  const int64_t M=3,N=7; auto in=randv(M*N,-2,2); for(int64_t r=0;r<M;r++) in[r*N+2]=in[r*N+5]; std::vector<int64_t> hi(M*N);
  DevBuf da(M*N*4),di(M*N*8); da.up(in.data());
  auto*ta=mk({M,N},ACL_FLOAT,da.p),*tidx=mk({M,N},ACL_INT64,di.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnArgsortGetWorkspaceSize(ta,1,true,true,tidx,w,e);},aclnnArgsort);
  di.down(hi.data()); int64_t bad=0;
  for(int64_t r=0;r<M;r++){ std::vector<int64_t> idx(N); for(int64_t j=0;j<N;j++) idx[j]=j;
    std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){float va=in[r*N+a],vb=in[r*N+b]; if(va==vb)return a<b; return va>vb;});
    for(int64_t j=0;j<N;j++) if(hi[r*N+j]!=idx[j]) bad++; }
  reportb("Argsort dim1 desc",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tidx);
}
// Bucketize: int64 out = count of boundaries (right=false) below each value
static void t_bucketize(){
  const int64_t B=4,NV=6; float bnd[B]={-1,0,1,2}; auto val=randv(NV,-2,3); std::vector<int64_t> ho(NV);
  DevBuf db(B*4),dv(NV*4),dz(NV*8); db.up(bnd); dv.up(val.data());
  auto*tb=mk({B},ACL_FLOAT,db.p),*tv=mk({NV},ACL_FLOAT,dv.p),*tz=mk({NV},ACL_INT64,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBucketizeGetWorkspaceSize(tv,tb,false,tz,w,e);},aclnnBucketize);
  dz.down(ho.data()); int64_t bad=0; for(int64_t i=0;i<NV;i++){ int64_t r=0; for(int64_t j=0;j<B;j++) if(bnd[j]<val[i]) r++; if(ho[i]!=r) bad++; }
  reportb("Bucketize (left)",bad);
  aclDestroyTensor(tb);aclDestroyTensor(tv);aclDestroyTensor(tz);
}
// Unique2 with counts + inverse on a 1D tensor with duplicates
static void t_unique2(){
  const int64_t N=8; float in[N]={3,1,2,1,3,3,2,5}; int64_t vcnt; std::vector<float> hv(N); std::vector<int64_t> hc(N),hinv(N);
  DevBuf da(N*4),dv(N*4),dcnt(8),dcounts(N*8),dinv(N*8); da.up(in);
  auto*ta=mk({N},ACL_FLOAT,da.p),*tv=mk({N},ACL_FLOAT,dv.p),*tcnt=mk({1},ACL_INT64,dcnt.p),*tcounts=mk({N},ACL_INT64,dcounts.p),*tinv=mk({N},ACL_INT64,dinv.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUnique2GetWorkspaceSize(ta,true,true,true,tv,tcnt,tinv,tcounts,w,e);},aclnnUnique2);
  dcnt.down(&vcnt); dv.down(hv.data()); dcounts.down(hc.data()); dinv.down(hinv.data());
  // reference: sorted unique {1,2,3,5}, counts {2,2,3,1}
  std::vector<float> su(in,in+N); std::sort(su.begin(),su.end()); std::vector<float> ru; std::vector<int64_t> rc; for(int64_t i=0;i<N;i++){ if(i==0||su[i]!=su[i-1]){ru.push_back(su[i]);rc.push_back(1);} else rc.back()++; }
  int64_t bad=0; if(vcnt!=(int64_t)ru.size()) bad++; for(size_t i=0;i<ru.size();i++){ if(hv[i]!=ru[i]) bad++; if(hc[i]!=rc[i]) bad++; }
  for(int64_t i=0;i<N;i++){ int64_t pos=std::lower_bound(ru.begin(),ru.end(),in[i])-ru.begin(); if(hinv[i]!=pos) bad++; }
  reportb("Unique2 (vals/counts/inv)",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(tcnt);aclDestroyTensor(tcounts);aclDestroyTensor(tinv);
}
// UniqueDim: alias of Unique (sorted dedup)
static void t_unique_dim(){
  const int64_t N=6; float in[N]={4,2,4,2,1,1}; int64_t vcnt; std::vector<float> hv(N);
  DevBuf da(N*4),dv(N*4),dcnt(8); da.up(in);
  auto*ta=mk({N},ACL_FLOAT,da.p),*tv=mk({N},ACL_FLOAT,dv.p),*tcnt=mk({1},ACL_INT64,dcnt.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUniqueDimGetWorkspaceSize(ta,0,true,false,false,tv,tcnt,nullptr,nullptr,w,e);},aclnnUniqueDim);
  dcnt.down(&vcnt); dv.down(hv.data()); std::vector<float> su(in,in+N); std::sort(su.begin(),su.end()); su.erase(std::unique(su.begin(),su.end()),su.end());
  int64_t bad=0; if(vcnt!=(int64_t)su.size()) bad++; for(size_t i=0;i<su.size();i++) if(hv[i]!=su[i]) bad++;
  reportb("UniqueDim",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tv);aclDestroyTensor(tcnt);
}
// TopKTopPSample: filter logits then sample; verify the sampled token is within the kept (top-k) set.
static void t_topk_topp_sample(){
  const int64_t R=4,V=10,topk=3; double topp=1.0; int64_t seed=123; auto in=randv(R*V,-3,3); std::vector<int64_t> ho(R);
  DevBuf da(R*V*4),dz(R*8); da.up(in.data());
  auto*ta=mk({R,V},ACL_FLOAT,da.p),*tz=mk({R},ACL_INT64,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTopKTopPSampleGetWorkspaceSize(ta,topk,topp,seed,tz,w,e);},aclnnTopKTopPSample);
  dz.down(ho.data()); int64_t bad=0;
  for(int64_t r=0;r<R;r++){ // build top-k set
    std::vector<int64_t> idx(V); for(int64_t j=0;j<V;j++) idx[j]=j; std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){return in[r*V+a]>in[r*V+b];});
    bool ok=false; for(int64_t k=0;k<topk;k++) if(ho[r]==idx[k]) ok=true; if(!ok) bad++; }
  reportb("TopKTopPSample (in top-k)",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_topk_topp_sample_v2(){
  const int64_t R=4,V=8,topk=2; double topp=1.0; int64_t seed=77; auto in=randv(R*V,-3,3); std::vector<int64_t> ho(R);
  DevBuf da(R*V*4),dz(R*8); da.up(in.data());
  auto*ta=mk({R,V},ACL_FLOAT,da.p),*tz=mk({R},ACL_INT64,dz.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTopKTopPSampleV2GetWorkspaceSize(ta,topk,topp,seed,tz,w,e);},aclnnTopKTopPSampleV2);
  dz.down(ho.data()); int64_t bad=0;
  for(int64_t r=0;r<R;r++){ std::vector<int64_t> idx(V); for(int64_t j=0;j<V;j++) idx[j]=j; std::sort(idx.begin(),idx.end(),[&](int64_t a,int64_t b){return in[r*V+a]>in[r*V+b];});
    bool ok=false; for(int64_t k=0;k<topk;k++) if(ho[r]==idx[k]) ok=true; if(!ok) bad++; }
  reportb("TopKTopPSampleV2 (in top-k)",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tz);
}

// NonzeroV2: out[k] = flat index of nonzero elements; countOut[0] = count
static void t_nonzero_v2(){
  const int64_t N=10; std::vector<float> in(N,0.f); in[1]=1; in[4]=-2; in[7]=3; in[9]=0.5f; std::vector<int64_t> ho(N); int64_t cnt;
  DevBuf da(N*4),dz(N*8),dc(8); da.up(in.data());
  auto*ta=mk({N},ACL_FLOAT,da.p),*tz=mk({N},ACL_INT64,dz.p),*tc=mk({1},ACL_INT64,dc.p);
  exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNonzeroV2GetWorkspaceSize(ta,tz,tc,w,e);},aclnnNonzeroV2);
  dz.down(ho.data()); dc.down(&cnt); std::vector<int64_t> ref; for(int64_t i=0;i<N;i++) if(in[i]!=0.f) ref.push_back(i);
  int64_t bad=0; if(cnt!=(int64_t)ref.size()) bad++; for(size_t i=0;i<ref.size();i++) if(ho[i]!=ref[i]) bad++;
  reportb("NonzeroV2",bad);
  aclDestroyTensor(ta);aclDestroyTensor(tz);aclDestroyTensor(tc);
}

int main(){
  CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(7);
  // index / shape (25)
  t_index_select(); t_index(); t_gatherv3(); t_index_add_v2(); t_index_fill_tensor();
  t_scatternd(); t_scatter_list(); t_index_put(); t_scatter_pa_cache(); t_scatter_pa_kv();
  t_masked_scale(); t_expandv(); t_repeat(); t_diag();
  t_repeat_il_int(); t_repeat_il_withdim(); t_repeat_il_intwithdim(); t_repeat_il_tensor();
  t_chunk(); t_chunkcat(); t_flatten(); t_slice_v2(); t_slice_assign(); t_lse(); t_nonzero_v2();
  // reduce / sort (14)
  t_aminmax_all(); t_aminmax_dim(); t_reduce_nansum(); t_stdmean(); t_cumsum_v2();
  t_median_dim(); t_nan_median(); t_nan_median_dim(); t_argsort(); t_bucketize();
  t_unique2(); t_unique_dim(); t_topk_topp_sample(); t_topk_topp_sample_v2();
  CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
  printf("== %d PASS, %d FAIL ==\n", g_pass, g_fail);
  return g_fail?1:0;
}
