// Format / data-movement extensions (P19) cross-check: Contiguous (from transposed view), AsStrided, ViewCopy, Copy.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;

// strided view tensor over `data`: viewDims/strides explicit
static aclTensor *mkv(const std::vector<int64_t>&vd, const std::vector<int64_t>&st, int64_t off, const std::vector<int64_t>&sd, void*data){
    return aclCreateTensor(vd.data(), vd.size(), ACL_FLOAT, st.data(), off, ACL_FORMAT_ND, sd.data(), sd.size(), data);
}

static void t_contiguous() {
    const int M=3,N=4; auto fa=randv(M*N,-2,2);   // row-major [M,N]
    std::vector<float> hz(N*M); DevBuf da(M*N*4),dz(N*M*4); da.up(fa.data());
    aclTensor *tv=mkv({N,M},{1,(int64_t)N},0,{M,N},da.p);   // transposed view [N,M], strides {1,N}
    aclTensor *tz=mk({N,M},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnContiguousGetWorkspaceSize(tv,tz,w,e);}, aclnnContiguous);
    dz.down(hz.data()); double bad=0;
    for(int i=0;i<N;i++)for(int j=0;j<M;j++) bad=std::max(bad,(double)std::fabs(hz[i*M+j]-fa[j*N+i]));
    report("Contiguous (transpose view)", bad, 0);
    aclDestroyTensor(tv); aclDestroyTensor(tz);
}
static void t_asstrided() {
    const int n=12; auto fa=randv(n,-2,2); std::vector<float> hz(9); DevBuf da(n*4),dz(9*4); da.up(fa.data());
    int64_t sz[2]={3,3}, st[2]={4,1}; aclIntArray *asz=aclCreateIntArray(sz,2),*ast=aclCreateIntArray(st,2);
    aclTensor *ta=mk({n},ACL_FLOAT,da.p),*tz=mk({3,3},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAsStridedGetWorkspaceSize(ta,asz,ast,0,tz,w,e);}, aclnnAsStrided);
    dz.down(hz.data()); double bad=0; for(int i=0;i<3;i++)for(int j=0;j<3;j++) bad=std::max(bad,(double)std::fabs(hz[i*3+j]-fa[i*4+j]));
    report("AsStrided", bad, 0); aclDestroyIntArray(asz);aclDestroyIntArray(ast);aclDestroyTensor(ta);aclDestroyTensor(tz);
}
static void t_viewcopy_copy() {
    const int n=24; auto fa=randv(n,-2,2);
    { std::vector<float> hz(n); DevBuf da(n*4),dz(n*4); da.up(fa.data());
      aclTensor *ta=mk({2,12},ACL_FLOAT,da.p),*tz=mk({4,6},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnViewCopyGetWorkspaceSize(ta,tz,w,e);}, aclnnViewCopy);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::fabs(hz[i]-fa[i]));
      report("ViewCopy", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    { std::vector<float> hz(n); DevBuf da(n*4),dz(n*4); da.up(fa.data());
      aclTensor *ta=mk({n},ACL_FLOAT,da.p),*tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCopyGetWorkspaceSize(ta,tz,w,e);}, aclnnCopy);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::fabs(hz[i]-fa[i]));
      report("Copy", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}
int main() {
    init(); srand(37);
    t_contiguous(); t_asstrided(); t_viewcopy_copy();
    return finish();
}
