// Cross-check for the remaining in-scope ops (R1..R8): shape composites, linalg/BLAS/index/vision/loss/optim/random/RNN/FFT remainders.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;
static int rfl(int x,int n){ if(n==1)return 0; while(x<0||x>=n){if(x<0)x=-x; if(x>=n)x=2*(n-1)-x;} return x; }
static int rep(int x,int n){ return x<0?0:(x>=n?n-1:x); }
static int cir(int x,int n){ return ((x%n)+n)%n; }

// ---------------- R1: shape composites ----------------
static void t_r1() {
    // pad1d on [outer=2, W=5], pad {2,1}
    { const int O=2,W=5,pl=2,pr=1,oW=W+pl+pr; auto x=randv(O*W,-2,2); int64_t pad[2]={pl,pr}; aclIntArray*ap=aclCreateIntArray(pad,2);
      for(int mode=0;mode<3;mode++){ std::vector<float> hz(O*oW); DevBuf dx(O*W*4),dz(O*oW*4); dx.up(x.data());
        aclTensor*tx=mk({O,W},ACL_FLOAT,dx.p),*tz=mk({O,oW},ACL_FLOAT,dz.p);
        if(mode==0)exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReflectionPad1dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnReflectionPad1d);
        else if(mode==1)exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReplicationPad1dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnReplicationPad1d);
        else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCircularPad1dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnCircularPad1d);
        dz.down(hz.data()); double bad=0; for(int o=0;o<O;o++)for(int ow=0;ow<oW;ow++){int iw=mode==0?rfl(ow-pl,W):mode==1?rep(ow-pl,W):cir(ow-pl,W); bad=std::max(bad,(double)std::fabs(hz[o*oW+ow]-x[o*W+iw]));}
        report(std::string("Pad1d mode")+std::to_string(mode),bad,0); aclDestroyTensor(tx);aclDestroyTensor(tz); }
      aclDestroyIntArray(ap); }
    // pad3d on [1,1,D,H,W]=[2,3,4], pad {1,1,1,1,1,1}
    { const int Dd=2,H=3,W=4; auto x=randv(Dd*H*W,-2,2); int64_t pad[6]={1,1,1,1,1,1}; aclIntArray*ap=aclCreateIntArray(pad,6);
      int oD=Dd+2,oH=H+2,oW=W+2; for(int mode=0;mode<3;mode++){ std::vector<float> hz(oD*oH*oW); DevBuf dx(Dd*H*W*4),dz(oD*oH*oW*4); dx.up(x.data());
        aclTensor*tx=mk({1,1,Dd,H,W},ACL_FLOAT,dx.p),*tz=mk({1,1,oD,oH,oW},ACL_FLOAT,dz.p);
        if(mode==0)exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReflectionPad3dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnReflectionPad3d);
        else if(mode==1)exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReplicationPad3dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnReplicationPad3d);
        else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCircularPad3dGetWorkspaceSize(tx,ap,tz,w,e);},aclnnCircularPad3d);
        dz.down(hz.data()); double bad=0;
        for(int od=0;od<oD;od++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ int id=mode==0?rfl(od-1,Dd):mode==1?rep(od-1,Dd):cir(od-1,Dd),ih=mode==0?rfl(oh-1,H):mode==1?rep(oh-1,H):cir(oh-1,H),iw=mode==0?rfl(ow-1,W):mode==1?rep(ow-1,W):cir(ow-1,W);
            bad=std::max(bad,(double)std::fabs(hz[(od*oH+oh)*oW+ow]-x[(id*H+ih)*W+iw])); }
        report(std::string("Pad3d mode")+std::to_string(mode),bad,0); aclDestroyTensor(tx);aclDestroyTensor(tz); }
      aclDestroyIntArray(ap); }
    // meshgrid 2-input x[A],y[B] (ij)
    { const int A=3,B=4; auto x=randv(A,-2,2),y=randv(B,-2,2); std::vector<float> h0(A*B),h1(A*B);
      DevBuf dx(A*4),dy(B*4),d0(A*B*4),d1(A*B*4); dx.up(x.data()); dy.up(y.data());
      aclTensor*tx=mk({A},ACL_FLOAT,dx.p),*ty=mk({B},ACL_FLOAT,dy.p),*t0=mk({A,B},ACL_FLOAT,d0.p),*t1=mk({A,B},ACL_FLOAT,d1.p);
      const aclTensor*ins[2]={tx,ty}; aclTensorList*til=aclCreateTensorList(ins,2); const aclTensor*outs[2]={t0,t1}; aclTensorList*tol=aclCreateTensorList(outs,2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMeshgridGetWorkspaceSize(til,tol,w,e);},aclnnMeshgrid);
      d0.down(h0.data()); d1.down(h1.data()); double bad=0; for(int a=0;a<A;a++)for(int b=0;b<B;b++){bad=std::max(bad,(double)std::fabs(h0[a*B+b]-x[a])); bad=std::max(bad,(double)std::fabs(h1[a*B+b]-y[b]));}
      report("Meshgrid 2-input",bad,0); aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(t0);aclDestroyTensor(t1); }
    // vstack(dim0) two [2,3]
    { auto a=randv(6,-1,1),b=randv(6,-1,1); std::vector<float> hz(12); DevBuf da(24),db(24),dz(48); da.up(a.data()); db.up(b.data());
      aclTensor*ta=mk({2,3},ACL_FLOAT,da.p),*tb=mk({2,3},ACL_FLOAT,db.p),*tz=mk({4,3},ACL_FLOAT,dz.p); const aclTensor*ins[2]={ta,tb};
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnVstackGetWorkspaceSize(ins,2,tz,w,e);},aclnnVstack);
      dz.down(hz.data()); double bad=0; for(int i=0;i<6;i++){bad=std::max(bad,(double)std::fabs(hz[i]-a[i]));bad=std::max(bad,(double)std::fabs(hz[6+i]-b[i]));}
      report("Vstack",bad,0); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
}

// ---------------- R2: linalg remainder ----------------
static void t_r2() {
    // MatrixRank: full-rank random [4,4] -> 4 ; rank-1 outer product u v^T -> 1
    { const int n=4; auto A=randv(n*n,-1,1); for(int i=0;i<n;i++)A[i*n+i]+=2; std::vector<int64_t> hr(1); DevBuf da(n*n*4),dr(8); da.up(A.data());
      aclTensor*ta=mk({n,n},ACL_FLOAT,da.p),*tr=mk({1},ACL_INT64,dr.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatrixRankGetWorkspaceSize(ta,0.0,tr,w,e);},aclnnMatrixRank);
      dr.down(hr.data()); report("MatrixRank full",(double)std::llabs(hr[0]-n),0); aclDestroyTensor(ta);aclDestroyTensor(tr); }
    { const int n=4; auto u=randv(n,-1,1),v=randv(n,-1,1); std::vector<float> A(n*n); for(int i=0;i<n;i++)for(int j=0;j<n;j++)A[i*n+j]=u[i]*v[j];
      std::vector<int64_t> hr(1); DevBuf da(n*n*4),dr(8); da.up(A.data()); aclTensor*ta=mk({n,n},ACL_FLOAT,da.p),*tr=mk({1},ACL_INT64,dr.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatrixRankGetWorkspaceSize(ta,0.0,tr,w,e);},aclnnMatrixRank);
      dr.down(hr.data()); report("MatrixRank rank-1",(double)std::llabs(hr[0]-1),0); aclDestroyTensor(ta);aclDestroyTensor(tr); }
    // Lstsq: A[6,3], B[6,2]; verify A^T(A x - B) ~ 0
    { const int m=6,n=3,nr=2; auto A=randv(m*n,-1,1),B=randv(m*nr,-1,1); std::vector<float> hX(n*nr);
      DevBuf da(m*n*4),db(m*nr*4),dx(n*nr*4); da.up(A.data()); db.up(B.data());
      aclTensor*ta=mk({m,n},ACL_FLOAT,da.p),*tb=mk({m,nr},ACL_FLOAT,db.p),*tx=mk({n,nr},ACL_FLOAT,dx.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLstsqGetWorkspaceSize(ta,tb,tx,w,e);},aclnnLstsq);
      dx.down(hX.data()); double resid=0;
      for(int i=0;i<n;i++)for(int r=0;r<nr;r++){ double g=0; for(int mm=0;mm<m;mm++){ double ax=0; for(int k=0;k<n;k++) ax+=(double)A[mm*n+k]*hX[k*nr+r]; g+=(double)A[mm*n+i]*(ax-B[mm*nr+r]); } resid=std::max(resid,std::fabs(g)); }
      report("Lstsq (normal-eq resid)", resid, 1e-3); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tx); }
    // MatrixPower A[3,3]^3
    { const int n=3,p=3; auto A=randv(n*n,-0.6,0.6); std::vector<float> hz(n*n); DevBuf da(n*n*4),dz(n*n*4); da.up(A.data());
      aclTensor*ta=mk({n,n},ACL_FLOAT,da.p),*tz=mk({n,n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMatrixPowerGetWorkspaceSize(ta,p,tz,w,e);},aclnnMatrixPower);
      dz.down(hz.data()); std::vector<double> M(A.begin(),A.end());
      for(int s=1;s<p;s++){ std::vector<double> t(n*n,0); for(int i=0;i<n;i++)for(int j=0;j<n;j++){double acc=0;for(int k=0;k<n;k++)acc+=M[i*n+k]*A[k*n+j];t[i*n+j]=acc;} M=t; }
      double me=0,mr=0; for(int i=0;i<n*n;i++){me=std::max(me,std::fabs(hz[i]-M[i]));mr=std::max(mr,std::fabs(M[i]));}
      report("MatrixPower p=3", me/(mr+1e-9), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}

// ---------------- R3: BLAS remainder ----------------
static double nerr2(const std::vector<float>&g,const std::vector<double>&r){double me=0,mr=0;for(size_t i=0;i<r.size();i++){me=std::max(me,std::fabs(g[i]-r[i]));mr=std::max(mr,std::fabs(r[i]));}return me/(mr+1e-9);}
static void t_r3() {
    // Tensordot A[2,3,4] . B[4,5] over 1 axis -> [2,3,5]
    { const int A0=2,A1=3,C=4,N=5; auto A=randv(A0*A1*C,-1,1),B=randv(C*N,-1,1); std::vector<float> hz(A0*A1*N);
      DevBuf da(A0*A1*C*4),db(C*N*4),dz(A0*A1*N*4); da.up(A.data()); db.up(B.data());
      aclTensor*ta=mk({A0,A1,C},ACL_FLOAT,da.p),*tb=mk({C,N},ACL_FLOAT,db.p),*tz=mk({A0,A1,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTensordotGetWorkspaceSize(ta,tb,1,tz,w,e);},aclnnTensordot);
      dz.down(hz.data()); std::vector<double> ref(A0*A1*N,0);
      for(int a=0;a<A0*A1;a++)for(int n=0;n<N;n++){double s=0;for(int c=0;c<C;c++)s+=(double)A[a*C+c]*B[c*N+n];ref[a*N+n]=s;}
      report("Tensordot",nerr2(hz,ref),1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    auto einsum=[&](const char*eq,std::vector<int64_t> as,std::vector<int64_t> bs,std::vector<int64_t> os,std::function<void(const std::vector<float>&,const std::vector<float>&,std::vector<double>&)> ref){
      int64_t an=1;for(auto x:as)an*=x; int64_t bn=1;for(auto x:bs)bn*=x; int64_t on=1;for(auto x:os)on*=x;
      auto A=randv(an,-1,1),B=randv(bn,-1,1); std::vector<float> hz(on); DevBuf da(an*4),db(bn*4),dz(on*4); da.up(A.data()); db.up(B.data());
      aclTensor*ta=mk(as,ACL_FLOAT,da.p),*tb=mk(bs,ACL_FLOAT,db.p),*tz=mk(os,ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEinsumGetWorkspaceSize(eq,ta,tb,tz,w,e);},aclnnEinsum);
      dz.down(hz.data()); std::vector<double> r(on,0); ref(A,B,r); report(std::string("Einsum ")+eq,nerr2(hz,r),1e-5);
      aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); };
    einsum("ij,jk->ik",{3,4},{4,5},{3,5},[](auto&A,auto&B,auto&r){for(int i=0;i<3;i++)for(int k=0;k<5;k++){double s=0;for(int j=0;j<4;j++)s+=(double)A[i*4+j]*B[j*5+k];r[i*5+k]=s;}});
    einsum("ij,jk->ki",{3,4},{4,5},{5,3},[](auto&A,auto&B,auto&r){for(int i=0;i<3;i++)for(int k=0;k<5;k++){double s=0;for(int j=0;j<4;j++)s+=(double)A[i*4+j]*B[j*5+k];r[k*3+i]=s;}});
    einsum("bij,bjk->bik",{2,3,4},{2,4,5},{2,3,5},[](auto&A,auto&B,auto&r){for(int b=0;b<2;b++)for(int i=0;i<3;i++)for(int k=0;k<5;k++){double s=0;for(int j=0;j<4;j++)s+=(double)A[(b*3+i)*4+j]*B[(b*4+j)*5+k];r[(b*3+i)*5+k]=s;}});
}

// ---------------- R4: Unique / UniqueConsecutive ----------------
static void t_r4() {
    // Unique: [3,1,2,1,3,3] -> values[1,2,3] count=3 counts[2,1,3]; inverse maps original->uidx
    { float x[6]={3,1,2,1,3,3}; const int n=6; std::vector<float> hv(n); std::vector<int64_t> hc(1),hcnt(n),hinv(n);
      DevBuf dx(n*4),dv(n*4),dc(8),dcnt(n*8),dinv(n*8); dx.up(x);
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tv=mk({n},ACL_FLOAT,dv.p),*tc=mk({1},ACL_INT64,dc.p),*tinv=mk({n},ACL_INT64,dinv.p),*tcnt=mk({n},ACL_INT64,dcnt.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUniqueGetWorkspaceSize(tx,tv,tc,tinv,tcnt,w,e);},aclnnUnique);
      dv.down(hv.data()); dc.down(hc.data()); dcnt.down(hcnt.data()); dinv.down(hinv.data());
      bool ok = hc[0]==3 && hv[0]==1 && hv[1]==2 && hv[2]==3 && hcnt[0]==2 && hcnt[1]==1 && hcnt[2]==3;
      double iv=0; for(int i=0;i<n;i++){ int exp=(x[i]==1?0:x[i]==2?1:2); iv=std::max(iv,(double)std::llabs(hinv[i]-exp)); }
      report("Unique values+count+counts", ok?0.0:1.0, 0); report("Unique inverse", iv, 0);
      aclDestroyTensor(tx);aclDestroyTensor(tv);aclDestroyTensor(tc);aclDestroyTensor(tinv);aclDestroyTensor(tcnt); }
    // UniqueConsecutive: [1,1,2,3,3,1] -> [1,2,3,1] count=4 counts[2,1,2,1]
    { float x[6]={1,1,2,3,3,1}; const int n=6; std::vector<float> hv(n); std::vector<int64_t> hc(1),hcnt(n);
      DevBuf dx(n*4),dv(n*4),dc(8),dcnt(n*8); dx.up(x);
      aclTensor*tx=mk({n},ACL_FLOAT,dx.p),*tv=mk({n},ACL_FLOAT,dv.p),*tc=mk({1},ACL_INT64,dc.p),*tcnt=mk({n},ACL_INT64,dcnt.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUniqueConsecutiveGetWorkspaceSize(tx,tv,tc,nullptr,tcnt,w,e);},aclnnUniqueConsecutive);
      dv.down(hv.data()); dc.down(hc.data()); dcnt.down(hcnt.data());
      bool ok = hc[0]==4 && hv[0]==1&&hv[1]==2&&hv[2]==3&&hv[3]==1 && hcnt[0]==2&&hcnt[1]==1&&hcnt[2]==2&&hcnt[3]==1;
      report("UniqueConsecutive", ok?0.0:1.0, 0);
      aclDestroyTensor(tx);aclDestroyTensor(tv);aclDestroyTensor(tc);aclDestroyTensor(tcnt); }
}

// ---------------- R5: vision remainder ----------------
static void t_r5() {
    // AdaptiveMaxPool3d [1,1,4,4,4]->[2,2,2]
    { const int D=4,H=4,W=4,oD=2,oH=2,oW=2; auto x=randv(D*H*W,-2,2); std::vector<float> hz(oD*oH*oW); DevBuf dx(D*H*W*4),dz(oD*oH*oW*4); dx.up(x.data());
      aclTensor*tx=mk({1,1,D,H,W},ACL_FLOAT,dx.p),*tz=mk({1,1,oD,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaptiveMaxPool3dGetWorkspaceSize(tx,tz,w,e);},aclnnAdaptiveMaxPool3d);
      dz.down(hz.data()); double bad=0;
      for(int od=0;od<oD;od++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){double best=-1e30; for(int d=od*D/oD;d<(od+1)*D/oD;d++)for(int h=oh*H/oH;h<(oh+1)*H/oH;h++)for(int w=ow*W/oW;w<(ow+1)*W/oW;w++) best=std::max(best,(double)x[(d*H+h)*W+w]); bad=std::max(bad,std::fabs(hz[(od*oH+oh)*oW+ow]-best));}
      report("AdaptiveMaxPool3d",bad,0); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    // LpPool2d [1,1,4,4] k2 s2 p2
    { const int H=4,W=4,k=2,oH=2,oW=2; auto x=randv(H*W,-2,2); std::vector<float> hz(oH*oW); int64_t kk[2]={k,k},st[2]={k,k}; aclIntArray*ak=aclCreateIntArray(kk,2),*as=aclCreateIntArray(st,2);
      DevBuf dx(H*W*4),dz(oH*oW*4); dx.up(x.data()); aclTensor*tx=mk({1,1,H,W},ACL_FLOAT,dx.p),*tz=mk({1,1,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLpPool2dGetWorkspaceSize(tx,2.0,ak,as,tz,w,e);},aclnnLpPool2d);
      dz.down(hz.data()); double me=0,mr=0; for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){double s=0;for(int a=0;a<k;a++)for(int b=0;b<k;b++)s+=(double)x[(oh*2+a)*W+(ow*2+b)]*x[(oh*2+a)*W+(ow*2+b)]; double r=std::sqrt(s); me=std::max(me,std::fabs(hz[oh*oW+ow]-r)); mr=std::max(mr,r);}
      report("LpPool2d p=2",me/(mr+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); aclDestroyIntArray(ak);aclDestroyIntArray(as); }
    // UpsampleBicubic2d [1,1,3,3]->[6,6] vs CPU
    { const int H=3,W=3,oH=6,oW=6; bool al=false; auto x=randv(H*W,-2,2); std::vector<float> hz(oH*oW); DevBuf dx(H*W*4),dz(oH*oW*4); dx.up(x.data());
      aclTensor*tx=mk({1,1,H,W},ACL_FLOAT,dx.p),*tz=mk({1,1,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleBicubic2dGetWorkspaceSize(tx,al,tz,w,e);},aclnnUpsampleBicubic2d);
      dz.down(hz.data()); auto cub=[](float t){t=std::fabs(t);float a=-0.75f; if(t<=1)return ((a+2)*t-(a+3))*t*t+1; if(t<2)return (((t-5)*t+8)*t-4)*a; return 0.f;};
      double me=0,mr=0; for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ float fh=(oh+0.5f)*H/oH-0.5f,fw=(ow+0.5f)*W/oW-0.5f; int y0=(int)std::floor(fh),x0=(int)std::floor(fw); float dy=fh-y0,dx=fw-x0; double acc=0;
          for(int m=-1;m<=2;m++){float wy=cub(dy-m); int yy=std::min(std::max(y0+m,0),H-1); for(int nn=-1;nn<=2;nn++){float wx=cub(dx-nn); int xx=std::min(std::max(x0+nn,0),W-1); acc+=(double)wy*wx*x[yy*W+xx];}} me=std::max(me,std::fabs(hz[oh*oW+ow]-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("UpsampleBicubic2d",me/(mr+1e-9),1e-4); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    // GridSample3d identity (align_corners) reproduces input
    { const int D=2,H=3,Wd=3; auto x=randv(D*H*Wd,-2,2); std::vector<float> grid(D*H*Wd*3),hz(D*H*Wd);
      for(int d=0;d<D;d++)for(int h=0;h<H;h++)for(int w=0;w<Wd;w++){int o=((d*H+h)*Wd+w)*3; grid[o]=Wd>1?2.f*w/(Wd-1)-1:0; grid[o+1]=H>1?2.f*h/(H-1)-1:0; grid[o+2]=D>1?2.f*d/(D-1)-1:0;}
      DevBuf dx(D*H*Wd*4),dg(D*H*Wd*3*4),dz(D*H*Wd*4); dx.up(x.data()); dg.up(grid.data());
      aclTensor*tx=mk({1,1,D,H,Wd},ACL_FLOAT,dx.p),*tg=mk({1,D,H,Wd,3},ACL_FLOAT,dg.p),*tz=mk({1,1,D,H,Wd},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGridSample3dGetWorkspaceSize(tx,tg,true,tz,w,e);},aclnnGridSample3d);
      dz.down(hz.data()); double bad=0; for(int i=0;i<D*H*Wd;i++) bad=std::max(bad,(double)std::fabs(hz[i]-x[i]));
      report("GridSample3d identity",bad,1e-5); aclDestroyTensor(tx);aclDestroyTensor(tg);aclDestroyTensor(tz); }
    // RoiAlign: 1 roi over full [1,1,4,4], ph=pw=2, ratio=2, scale=1 -> vs CPU same formula
    { const int H=4,W=4,ph=2,pw=2,ratio=2; auto x=randv(H*W,-2,2); float roi[5]={0,0,0,(float)W,(float)H}; std::vector<float> hz(ph*pw);
      DevBuf dx(H*W*4),dr(5*4),dz(ph*pw*4); dx.up(x.data()); dr.up(roi);
      aclTensor*tx=mk({1,1,H,W},ACL_FLOAT,dx.p),*tr=mk({1,5},ACL_FLOAT,dr.p),*tz=mk({1,1,ph,pw},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRoiAlignGetWorkspaceSize(tx,tr,1.0,ratio,tz,w,e);},aclnnRoiAlign);
      dz.down(hz.data()); auto bil=[&](float fy,float fx){ if(fy<-1||fy>H||fx<-1||fx>W)return 0.f; fy=std::max(fy,0.f);fx=std::max(fx,0.f); int y0=(int)fy,x0=(int)fx,y1=std::min(y0+1,H-1),x1=std::min(x0+1,W-1); float dy=fy-y0,dx=fx-x0; return (1-dy)*(1-dx)*x[y0*W+x0]+(1-dy)*dx*x[y0*W+x1]+dy*(1-dx)*x[y1*W+x0]+dy*dx*x[y1*W+x1]; };
      double me=0,mr=0; float rw=W,rh=H,bw=rw/pw,bh=rh/ph; for(int py=0;py<ph;py++)for(int px=0;px<pw;px++){double s=0;for(int iy=0;iy<ratio;iy++)for(int ix=0;ix<ratio;ix++){float yy=0+py*bh+(iy+0.5f)*bh/ratio,xx=0+px*bw+(ix+0.5f)*bw/ratio; s+=bil(yy,xx);} double r=s/(ratio*ratio); me=std::max(me,std::fabs(hz[py*pw+px]-r)); mr=std::max(mr,std::fabs(r));}
      report("RoiAlign",me/(mr+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tz); }
}

// ---------------- R6: loss remainder ----------------
static void t_r6() {
    const int N=8;
    auto meanloss=[&](const std::vector<double>&ps){ double s=0; for(double v:ps)s+=v; return s/ps.size(); };
    // PoissonNLL (logInput) mean
    { auto x=randv(N,-1,1),t=randv(N,0,2); std::vector<float> hz(1); DevBuf dx(N*4),dt(N*4),dz(4); dx.up(x.data()); dt.up(t.data());
      aclTensor*tx=mk({N},ACL_FLOAT,dx.p),*tt=mk({N},ACL_FLOAT,dt.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPoissonNllLossGetWorkspaceSize(tx,tt,true,1,tz,w,e);},aclnnPoissonNllLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int i=0;i<N;i++)ps[i]=std::exp(x[i])-t[i]*x[i]; double r=meanloss(ps);
      report("PoissonNLL",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tz); }
    // GaussianNLL mean
    { auto x=randv(N,-1,1),t=randv(N,-1,1),v=randv(N,0.2,2); std::vector<float> hz(1); DevBuf dx(N*4),dt(N*4),dv(N*4),dz(4); dx.up(x.data()); dt.up(t.data()); dv.up(v.data());
      aclTensor*tx=mk({N},ACL_FLOAT,dx.p),*tt=mk({N},ACL_FLOAT,dt.p),*tv=mk({N},ACL_FLOAT,dv.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGaussianNllLossGetWorkspaceSize(tx,tt,tv,1,tz,w,e);},aclnnGaussianNllLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int i=0;i<N;i++){double d=x[i]-t[i];ps[i]=0.5*(std::log(v[i])+d*d/v[i]);} double r=meanloss(ps);
      report("GaussianNLL",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tv);aclDestroyTensor(tz); }
    // MultiLabelSoftMargin [N,C] mean
    { const int C=5; auto x=randv(N*C,-2,2); std::vector<float> t(N*C); for(auto&v:t)v=(rand()&1)?1.f:0.f; std::vector<float> hz(1);
      DevBuf dx(N*C*4),dt(N*C*4),dz(4); dx.up(x.data()); dt.up(t.data()); aclTensor*tx=mk({N,C},ACL_FLOAT,dx.p),*tt=mk({N,C},ACL_FLOAT,dt.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMultiLabelSoftMarginLossGetWorkspaceSize(tx,tt,1,tz,w,e);},aclnnMultiLabelSoftMarginLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int n=0;n<N;n++){double s=0;for(int c=0;c<C;c++){double p=1/(1+std::exp(-x[n*C+c]));s+=t[n*C+c]*std::log(p)+(1-t[n*C+c])*std::log(1-p);}ps[n]=-s/C;} double r=meanloss(ps);
      report("MultiLabelSoftMargin",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(tt);aclDestroyTensor(tz); }
    // MultiMargin [N,C] p=1 margin=1 mean
    { const int C=5; auto x=randv(N*C,-2,2); std::vector<int64_t> y(N); for(auto&v:y)v=rand()%C; std::vector<float> hz(1);
      DevBuf dx(N*C*4),dy(N*8),dz(4); dx.up(x.data()); dy.up(y.data()); aclTensor*tx=mk({N,C},ACL_FLOAT,dx.p),*ty=mk({N},ACL_INT64,dy.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMultiMarginLossGetWorkspaceSize(tx,ty,1.0,1.0,1,tz,w,e);},aclnnMultiMarginLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int n=0;n<N;n++){double s=0;for(int j=0;j<C;j++){if(j==y[n])continue;double m=1.0-x[n*C+y[n]]+x[n*C+j];if(m>0)s+=m;}ps[n]=s/C;} double r=meanloss(ps);
      report("MultiMargin",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(tz); }
    // TripletMargin p=2 margin=1 mean
    { const int D=6; auto a=randv(N*D,-1,1),p=randv(N*D,-1,1),ng=randv(N*D,-1,1); std::vector<float> hz(1);
      DevBuf da(N*D*4),dp(N*D*4),dn(N*D*4),dz(4); da.up(a.data()); dp.up(p.data()); dn.up(ng.data());
      aclTensor*ta=mk({N,D},ACL_FLOAT,da.p),*tp=mk({N,D},ACL_FLOAT,dp.p),*tn=mk({N,D},ACL_FLOAT,dn.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTripletMarginLossGetWorkspaceSize(ta,tp,tn,1.0,2.0,1,tz,w,e);},aclnnTripletMarginLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int n=0;n<N;n++){double dp_=0,dn_=0;for(int d=0;d<D;d++){double u=a[n*D+d]-p[n*D+d],v=a[n*D+d]-ng[n*D+d];dp_+=u*u;dn_+=v*v;}double l=std::sqrt(dp_)-std::sqrt(dn_)+1.0;ps[n]=l>0?l:0;} double r=meanloss(ps);
      report("TripletMargin",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(ta);aclDestroyTensor(tp);aclDestroyTensor(tn);aclDestroyTensor(tz); }
    // CosineEmbedding margin=0 mean (y in {+1,-1})
    { const int D=6; auto x1=randv(N*D,-1,1),x2=randv(N*D,-1,1); std::vector<float> y(N); for(auto&v:y)v=(rand()&1)?1.f:-1.f; std::vector<float> hz(1);
      DevBuf d1(N*D*4),d2(N*D*4),dy(N*4),dz(4); d1.up(x1.data()); d2.up(x2.data()); dy.up(y.data());
      aclTensor*t1=mk({N,D},ACL_FLOAT,d1.p),*t2=mk({N,D},ACL_FLOAT,d2.p),*ty=mk({N},ACL_FLOAT,dy.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCosineEmbeddingLossGetWorkspaceSize(t1,t2,ty,0.0,1,tz,w,e);},aclnnCosineEmbeddingLoss);
      dz.down(hz.data()); std::vector<double> ps(N); for(int n=0;n<N;n++){double dot=0,n1=0,n2=0;for(int d=0;d<D;d++){dot+=(double)x1[n*D+d]*x2[n*D+d];n1+=(double)x1[n*D+d]*x1[n*D+d];n2+=(double)x2[n*D+d]*x2[n*D+d];}double cs=dot/(std::sqrt(n1)*std::sqrt(n2)+1e-12);ps[n]=y[n]>0?1-cs:std::max(0.0,cs);} double r=meanloss(ps);
      report("CosineEmbedding",std::fabs(hz[0]-r)/(std::fabs(r)+1e-9),1e-5); aclDestroyTensor(t1);aclDestroyTensor(t2);aclDestroyTensor(ty);aclDestroyTensor(tz); }
    // CTC: T=5,C=4(blank0),targets=[1,2,3](L=3),lens full; compare to independent CPU log-forward
    { const int T=5,Nb=1,C=4,Lmax=3,blank=0; std::vector<float> lp(T*Nb*C); for(int t=0;t<T;t++){ auto row=randv(C,-1,1); double mx=-1e30; for(int c=0;c<C;c++)mx=std::max(mx,(double)row[c]); double se=0; for(int c=0;c<C;c++)se+=std::exp(row[c]-mx); for(int c=0;c<C;c++) lp[(t*Nb)*C+c]=(float)(row[c]-mx-std::log(se)); }
      int64_t tg[3]={1,2,3}, il[1]={T}, tl[1]={3}; std::vector<float> hz(1);
      DevBuf dlp(T*Nb*C*4),dtg(3*8),dil(8),dtl(8),dz(4); dlp.up(lp.data()); dtg.up(tg); dil.up(il); dtl.up(tl);
      aclTensor*tlp=mk({T,Nb,C},ACL_FLOAT,dlp.p),*ttg=mk({Nb,Lmax},ACL_INT64,dtg.p),*til=mk({Nb},ACL_INT64,dil.p),*ttl=mk({Nb},ACL_INT64,dtl.p),*tz=mk({1},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCtcLossGetWorkspaceSize(tlp,ttg,til,ttl,blank,1,tz,w,e);},aclnnCtcLoss);
      dz.down(hz.data());
      // CPU CTC log-forward
      int L=3,S=2*L+1; auto lab=[&](int s){return (s&1)?(int)tg[s/2]:blank;}; auto LPc=[&](int t,int c){return (double)lp[(t*Nb)*C+c];};
      std::vector<double> prev(S,-1e30),cur(S); auto lse=[&](double a,double b){double mx=std::max(a,b),mn=std::min(a,b);return mx+std::log1p(std::exp(mn-mx));};
      prev[0]=LPc(0,blank); prev[1]=LPc(0,lab(1));
      for(int t=1;t<T;t++){for(int s=0;s<S;s++){double v=prev[s];if(s>0)v=lse(v,prev[s-1]);if(s>1&&lab(s)!=blank&&lab(s)!=lab(s-2))v=lse(v,prev[s-2]);cur[s]=v+LPc(t,lab(s));}prev=cur;}
      double ref=-lse(prev[S-1],prev[S-2]);
      report("CTCLoss",std::fabs(hz[0]-ref)/(std::fabs(ref)+1e-9),1e-4); aclDestroyTensor(tlp);aclDestroyTensor(ttg);aclDestroyTensor(til);aclDestroyTensor(ttl);aclDestroyTensor(tz); }
}

// ---------------- R7: optimizer remainder ----------------
static void t_r7() {
    const int N=64; const float lr=0.01f,b1=0.9f,b2=0.999f,eps=1e-8f,wd=0.01f; const int step=1;
    float bc1=1-std::pow(b1,step),bc2=1-std::pow(b2,step);
    // ApplyLamb
    { auto p=randv(N,-1,1),m=randv(N,0,0.1),v=randv(N,0,0.1),g=randv(N,-1,1); std::vector<float> out(N);
      DevBuf dp(N*4),dm(N*4),dv(N*4),dg(N*4); dp.up(p.data()); dm.up(m.data()); dv.up(v.data()); dg.up(g.data());
      aclTensor*tp=mk({N},ACL_FLOAT,dp.p),*tm=mk({N},ACL_FLOAT,dm.p),*tv=mk({N},ACL_FLOAT,dv.p),*tg=mk({N},ACL_FLOAT,dg.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyLambGetWorkspaceSize(tp,tm,tv,tg,lr,b1,b2,eps,wd,step,w,e);},aclnnApplyLamb);
      dp.down(out.data());
      std::vector<double> r(N); double pn=0,rn=0; for(int i=0;i<N;i++){double mi=b1*m[i]+(1-b1)*g[i];double vi=b2*v[i]+(1-b2)*g[i]*g[i];r[i]=(mi/bc1)/(std::sqrt(vi/bc2)+eps)+wd*p[i];pn+=p[i]*p[i];rn+=r[i]*r[i];}
      pn=std::sqrt(pn);rn=std::sqrt(rn); double trust=(pn>0&&rn>0)?pn/rn:1.0; double mx=0,mr=0; for(int i=0;i<N;i++){double e=p[i]-lr*trust*r[i];mx=std::max(mx,std::fabs(out[i]-e));mr=std::max(mr,std::fabs(e));}
      report("ApplyLamb",mx/(mr+1e-9),1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(tg); }
    // ApplyLars
    { const float mu=0.9f,tc=0.001f; auto p=randv(N,-1,1),buf=randv(N,-0.1,0.1),g=randv(N,-1,1); std::vector<float> out(N);
      DevBuf dp(N*4),db(N*4),dg(N*4); dp.up(p.data()); db.up(buf.data()); dg.up(g.data());
      aclTensor*tp=mk({N},ACL_FLOAT,dp.p),*tb=mk({N},ACL_FLOAT,db.p),*tg=mk({N},ACL_FLOAT,dg.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyLarsGetWorkspaceSize(tp,tb,tg,lr,mu,wd,tc,eps,w,e);},aclnnApplyLars);
      dp.down(out.data());
      double pn=0,gn=0; for(int i=0;i<N;i++){pn+=p[i]*p[i];gn+=g[i]*g[i];} pn=std::sqrt(pn);gn=std::sqrt(gn);
      double llr=(pn>0&&gn>0)?tc*pn/(gn+wd*pn+eps):1.0; double mx=0,mr=0; for(int i=0;i<N;i++){double bb=mu*buf[i]+(g[i]+wd*p[i]);double e=p[i]-lr*llr*bb;mx=std::max(mx,std::fabs(out[i]-e));mr=std::max(mr,std::fabs(e));}
      report("ApplyLars",mx/(mr+1e-9),1e-5); aclDestroyTensor(tp);aclDestroyTensor(tb);aclDestroyTensor(tg); }
    // FusedEmaAdam
    { const float ed=0.999f; auto p=randv(N,-1,1),m=randv(N,0,0.1),v=randv(N,0,0.1),ema=randv(N,-1,1),g=randv(N,-1,1); std::vector<float> op(N),oe(N);
      DevBuf dp(N*4),dm(N*4),dv(N*4),de(N*4),dg(N*4); dp.up(p.data()); dm.up(m.data()); dv.up(v.data()); de.up(ema.data()); dg.up(g.data());
      aclTensor*tp=mk({N},ACL_FLOAT,dp.p),*tm=mk({N},ACL_FLOAT,dm.p),*tv=mk({N},ACL_FLOAT,dv.p),*te=mk({N},ACL_FLOAT,de.p),*tg=mk({N},ACL_FLOAT,dg.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFusedEmaAdamGetWorkspaceSize(tp,tm,tv,te,tg,lr,b1,b2,eps,wd,ed,step,w,e);},aclnnFusedEmaAdam);
      dp.down(op.data()); de.down(oe.data());
      double mx=0,mr=0; for(int i=0;i<N;i++){double mi=b1*m[i]+(1-b1)*g[i];double vi=b2*v[i]+(1-b2)*g[i]*g[i];double pp=p[i]-lr*((mi/bc1)/(std::sqrt(vi/bc2)+eps)+wd*p[i]);double em=ed*ema[i]+(1-ed)*pp;mx=std::max(mx,std::max(std::fabs(op[i]-pp),std::fabs(oe[i]-em)));mr=std::max(mr,std::max(std::fabs(pp),std::fabs(em)));}
      report("FusedEmaAdam",mx/(mr+1e-9),1e-5); aclDestroyTensor(tp);aclDestroyTensor(tm);aclDestroyTensor(tv);aclDestroyTensor(te);aclDestroyTensor(tg); }
}

// ---------------- R8: random + RNN + FFT remainder ----------------
static void t_r8() {
    // SampleGamma: mean and variance of Gamma(a,scale=1) are both a
    { const int N=40000; const double a=3.0; std::vector<float> al(N,(float)a),out(N); DevBuf da(N*4),dz(N*4); da.up(al.data());
      aclTensor*ta=mk({N},ACL_FLOAT,da.p),*tz=mk({N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSampleGammaGetWorkspaceSize(ta,1.0,7,tz,w,e);},aclnnSampleGamma);
      dz.down(out.data()); double m=0; for(float v:out)m+=v; m/=N; double var=0; for(float v:out)var+=(v-m)*(v-m); var/=N;
      double err=std::max(std::fabs(m-a)/a,std::fabs(var-a)/a); report("SampleGamma(mean/var)",err,0.05); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // SampleDirichlet: rows sum to 1, column means ~ a_k/sum
    { const int M=20000,K=3; float ak[K]={1,2,3}; double sa=6; std::vector<float> al(M*K); for(int r=0;r<M;r++)for(int k=0;k<K;k++)al[r*K+k]=ak[k];
      std::vector<float> out(M*K); DevBuf da(M*K*4),dz(M*K*4); da.up(al.data()); aclTensor*ta=mk({M,K},ACL_FLOAT,da.p),*tz=mk({M,K},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSampleDirichletGetWorkspaceSize(ta,11,tz,w,e);},aclnnSampleDirichlet);
      dz.down(out.data()); double sumerr=0; std::vector<double> cm(K,0); for(int r=0;r<M;r++){double s=0;for(int k=0;k<K;k++){s+=out[r*K+k];cm[k]+=out[r*K+k];}sumerr=std::max(sumerr,std::fabs(s-1.0));}
      double me=0; for(int k=0;k<K;k++){cm[k]/=M; me=std::max(me,std::fabs(cm[k]-ak[k]/sa));} report("SampleDirichlet",std::max(sumerr,me),0.02); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // Rnn vanilla tanh
    { const int T=3,B=2,I=4,H=5; auto x=randv(T*B*I,-1,1),h0=randv(B*H,-0.5,0.5),wih=randv(H*I,-0.5,0.5),whh=randv(H*H,-0.5,0.5),bih=randv(H,-0.2,0.2),bhh=randv(H,-0.2,0.2);
      std::vector<float> out(T*B*H),hn(B*H); DevBuf dx(T*B*I*4),dh(B*H*4),dwi(H*I*4),dwh(H*H*4),dbi(H*4),dbh(H*4),doo(T*B*H*4),dhn(B*H*4);
      dx.up(x.data()); dh.up(h0.data()); dwi.up(wih.data()); dwh.up(whh.data()); dbi.up(bih.data()); dbh.up(bhh.data());
      aclTensor*tx=mk({T,B,I},ACL_FLOAT,dx.p),*th=mk({B,H},ACL_FLOAT,dh.p),*twi=mk({H,I},ACL_FLOAT,dwi.p),*twh=mk({H,H},ACL_FLOAT,dwh.p),*tbi=mk({H},ACL_FLOAT,dbi.p),*tbh=mk({H},ACL_FLOAT,dbh.p),*to=mk({T,B,H},ACL_FLOAT,doo.p),*thn=mk({B,H},ACL_FLOAT,dhn.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRnnGetWorkspaceSize(tx,th,twi,twh,tbi,tbh,0,to,thn,w,e);},aclnnRnn);
      doo.down(out.data()); dhn.down(hn.data());
      std::vector<double> h(B*H); for(int b=0;b<B;b++)for(int j=0;j<H;j++)h[b*H+j]=h0[b*H+j]; double mx=0,mr=0;
      for(int t=0;t<T;t++){ std::vector<double> hn2(B*H); for(int b=0;b<B;b++)for(int j=0;j<H;j++){double s=bih[j]+bhh[j];for(int i=0;i<I;i++)s+=x[(t*B+b)*I+i]*wih[j*I+i];for(int k=0;k<H;k++)s+=h[b*H+k]*whh[j*H+k];hn2[b*H+j]=std::tanh(s);} h=hn2;
        for(int b=0;b<B;b++)for(int j=0;j<H;j++){double e=h[b*H+j];mx=std::max(mx,std::fabs(out[(t*B+b)*H+j]-e));mr=std::max(mr,std::fabs(e));}}
      report("Rnn",mx/(mr+1e-9),1e-5); aclDestroyTensor(tx);aclDestroyTensor(th);aclDestroyTensor(twi);aclDestroyTensor(twh);aclDestroyTensor(tbi);aclDestroyTensor(tbh);aclDestroyTensor(to);aclDestroyTensor(thn); }
    // EmbeddingBag sum/mean/max
    { const int V=10,D=4,B=3; auto w=randv(V*D,-1,1); int64_t idx[7]={1,3,5, 2,2, 7,0}; int64_t off[3]={0,3,5}; // bags: [1,3,5],[2,2],[7,0]
      DevBuf dw(V*D*4),di(7*8),dof(3*8); dw.up(w.data()); di.up(idx); dof.up(off);
      for(int mode=0;mode<3;mode++){ std::vector<float> out(B*D); DevBuf dz(B*D*4);
        aclTensor*tw=mk({V,D},ACL_FLOAT,dw.p),*ti=mk({7},ACL_INT64,di.p),*tof=mk({3},ACL_INT64,dof.p),*tz=mk({B,D},ACL_FLOAT,dz.p);
        exec2([&](uint64_t*ww,aclOpExecutor**e){return aclnnEmbeddingBagGetWorkspaceSize(tw,ti,tof,mode,tz,ww,e);},aclnnEmbeddingBag);
        dz.down(out.data()); double mx=0,mr=0;
        for(int b=0;b<B;b++){int s=off[b],e2=(b+1<B)?off[b+1]:7; for(int d=0;d<D;d++){double acc=(mode==2)?-1e30:0; int cnt=0; for(int p=s;p<e2;p++){double v=w[idx[p]*D+d]; if(mode==2)acc=std::max(acc,v);else acc+=v; cnt++;} if(mode==1&&cnt)acc/=cnt; double e=acc; mx=std::max(mx,std::fabs(out[b*D+d]-e));mr=std::max(mr,std::fabs(e));}}
        const char*nm[3]={"EmbeddingBag(sum)","EmbeddingBag(mean)","EmbeddingBag(max)"}; report(nm[mode],mx/(mr+1e-9),1e-5);
        aclDestroyTensor(tw);aclDestroyTensor(ti);aclDestroyTensor(tof);aclDestroyTensor(tz); } }
    // Fft2 forward+inverse round trip == identity
    { const int n0=4,n1=4,batch=2; const int NC=batch*n0*n1*2; auto in=randv(NC,-1,1); std::vector<float> mid(NC),back(NC);
      DevBuf di(NC*4),dm(NC*4),db(NC*4); di.up(in.data());
      aclTensor*ti=mk({NC},ACL_FLOAT,di.p),*tm=mk({NC},ACL_FLOAT,dm.p),*tb=mk({NC},ACL_FLOAT,db.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFft2GetWorkspaceSize(ti,n0,n1,false,tm,w,e);},aclnnFft2);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFft2GetWorkspaceSize(tm,n0,n1,true,tb,w,e);},aclnnFft2);
      db.down(back.data()); double mx=0,mr=0; for(int i=0;i<NC;i++){mx=std::max(mx,std::fabs((double)back[i]-in[i]));mr=std::max(mr,std::fabs((double)in[i]));}
      report("Fft2(roundtrip)",mx/(mr+1e-9),1e-4); aclDestroyTensor(ti);aclDestroyTensor(tm);aclDestroyTensor(tb); }
    // Stft vs CPU per-frame DFT (window=ones)
    { const int B=1,L=16,nFft=8,hop=4; int frames=1+(L-nFft)/hop, nFreq=nFft/2+1; auto sig=randv(B*L,-1,1);
      std::vector<float> out(B*nFreq*frames*2); DevBuf ds(B*L*4),dz(B*nFreq*frames*2*4); ds.up(sig.data());
      aclTensor*ts=mk({B,L},ACL_FLOAT,ds.p),*tz=mk({B*nFreq*frames*2},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnStftGetWorkspaceSize(ts,nFft,hop,nullptr,tz,w,e);},aclnnStft);
      dz.down(out.data()); double mx=0,mr=0;
      for(int fr=0;fr<frames;fr++)for(int f=0;f<nFreq;f++){double re=0,im=0; for(int j=0;j<nFft;j++){double ang=-2*M_PI*f*j/nFft; double v=sig[fr*hop+j]; re+=v*std::cos(ang); im+=v*std::sin(ang);} int64_t d=((int64_t)f*frames+fr); mx=std::max(mx,std::max(std::fabs(out[2*d]-re),std::fabs(out[2*d+1]-im)));mr=std::max(mr,std::max(std::fabs(re),std::fabs(im)));}
      report("Stft",mx/(mr+1e-9),1e-4); aclDestroyTensor(ts);aclDestroyTensor(tz); }
}

// ---------------- R9: additional elementary math (acosh/asinh/atanh/sinc/digamma, logaddexp2) ----------------
static void t_r9() {
    const int64_t n = 4096;
    auto run_un = [&](const char *name, std::vector<float> in, aclnnStatus(*ws)(const aclTensor*,aclTensor*,uint64_t*,aclOpExecutor**),
                      aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream), std::function<double(double)> ref) {
        DevBuf di(n*4), doo(n*4); di.up(in.data());
        auto ti=mk({n},ACL_FLOAT,di.p), to=mk({n},ACL_FLOAT,doo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return ws(ti,to,w,e);}, run);
        std::vector<float> got(n); doo.down(got.data());
        std::vector<double> r(n); for(int64_t i=0;i<n;i++) r[i]=ref(in[i]);
        report(name, norm_err(got,r), 1e-5); aclDestroyTensor(ti); aclDestroyTensor(to);
    };
    run_un("Acosh", randv(n,1.1f,5.f), aclnnAcoshGetWorkspaceSize, aclnnAcosh, [](double v){return std::acosh(v);});
    run_un("Asinh", randv(n,-5.f,5.f), aclnnAsinhGetWorkspaceSize, aclnnAsinh, [](double v){return std::asinh(v);});
    run_un("Atanh", randv(n,-0.9f,0.9f), aclnnAtanhGetWorkspaceSize, aclnnAtanh, [](double v){return std::atanh(v);});
    run_un("Sinc",  randv(n,-3.f,3.f), aclnnSincGetWorkspaceSize, aclnnSinc, [](double v){ if(v==0)return 1.0; double p=M_PI*v; return std::sin(p)/p;});
    run_un("Digamma", randv(n,0.5f,8.f), aclnnDigammaGetWorkspaceSize, aclnnDigamma, [](double x){ double r=0; while(x<6){r-=1.0/x;x+=1.0;} double inv=1.0/x,i2=inv*inv; r+=std::log(x)-0.5*inv-i2*(1.0/12-i2*(1.0/120-i2*(1.0/252))); return r;});
    // binary logaddexp2: log2(2^a + 2^b)
    { auto a=randv(n,-3.f,3.f), b=randv(n,-3.f,3.f); DevBuf da(n*4),db(n*4),dz(n*4); da.up(a.data()); db.up(b.data());
      auto ta=mk({n},ACL_FLOAT,da.p), tb=mk({n},ACL_FLOAT,db.p), tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogAddExp2GetWorkspaceSize(ta,tb,tz,w,e);}, aclnnLogAddExp2);
      std::vector<float> got(n); dz.down(got.data());
      std::vector<double> r(n); for(int64_t i=0;i<n;i++) r[i]=std::log2(std::exp2((double)a[i])+std::exp2((double)b[i]));
      report("LogAddExp2", norm_err(got,r), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    // additional activations
    run_un("FastGelu", randv(n,-4.f,4.f), aclnnFastGeluGetWorkspaceSize, aclnnFastGelu, [](double a){ double c=0.7978845608028654*(a+0.044715*a*a*a); return 0.5*a*(1.0+std::tanh(c)); });
    run_un("SquaredRelu", randv(n,-4.f,4.f), aclnnSquaredReluGetWorkspaceSize, aclnnSquaredRelu, [](double a){ double r=a>0?a:0; return r*r; });
    run_un("GeluV2(tanh)", randv(n,-4.f,4.f), [](const aclTensor*s,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnGeluV2GetWorkspaceSize(s,1,o,w,e);}, aclnnGeluV2, [](double a){ double c=0.7978845608028654*(a+0.044715*a*a*a); return 0.5*a*(1.0+std::tanh(c)); });
    // swish (beta=1.5): x*sigmoid(beta*x)
    { auto x=randv(n,-4.f,4.f); DevBuf dx(n*4),dz(n*4); dx.up(x.data());
      auto tx=mk({n},ACL_FLOAT,dx.p), tz=mk({n},ACL_FLOAT,dz.p);
      float beta=1.5f; auto sb=aclCreateScalar(&beta,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSwishGetWorkspaceSize(tx,sb,tz,w,e);}, aclnnSwish);
      std::vector<float> got(n); dz.down(got.data());
      std::vector<double> r(n); for(int64_t i=0;i<n;i++){ double v=x[i]; r[i]=v/(1.0+std::exp(-1.5*v)); }
      report("Swish(beta=1.5)", norm_err(got,r), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    // floor divide (binary): floor(a/b)
    { auto a=randv(n,-8.f,8.f), b=randv(n,1.f,4.f); DevBuf da(n*4),db(n*4),dz(n*4); da.up(a.data()); db.up(b.data());
      auto ta=mk({n},ACL_FLOAT,da.p), tb=mk({n},ACL_FLOAT,db.p), tz=mk({n},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFloorDivideGetWorkspaceSize(ta,tb,tz,w,e);}, aclnnFloorDivide);
      std::vector<float> got(n); dz.down(got.data());
      std::vector<double> r(n); for(int64_t i=0;i<n;i++) r[i]=std::floor((double)a[i]/b[i]);
      report("FloorDivide", norm_err(got,r), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    // rsub (binary, alpha=2): other - 2*self
    { auto a=randv(n,-4.f,4.f), b=randv(n,-4.f,4.f); DevBuf da(n*4),db(n*4),dz(n*4); da.up(a.data()); db.up(b.data());
      auto ta=mk({n},ACL_FLOAT,da.p), tb=mk({n},ACL_FLOAT,db.p), tz=mk({n},ACL_FLOAT,dz.p);
      float al=2.f; auto sa=aclCreateScalar(&al,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRsubGetWorkspaceSize(ta,tb,sa,tz,w,e);}, aclnnRsub);
      std::vector<float> got(n); dz.down(got.data());
      std::vector<double> r(n); for(int64_t i=0;i<n;i++) r[i]=(double)b[i]-2.0*a[i];
      report("Rsub(alpha=2)", norm_err(got,r), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(tz); }
    // round to 2 decimals
    run_un("RoundDecimals(2)", randv(n,-5.f,5.f), [](const aclTensor*s,aclTensor*o,uint64_t*w,aclOpExecutor**e){return aclnnRoundDecimalsGetWorkspaceSize(s,2,o,w,e);}, aclnnRoundDecimals, [](double a){ double m=100.0; return std::rint(a*m)/m; });
    // IsInf / IsPosInf / IsNegInf
    { std::vector<float> in(n); for(int64_t i=0;i<n;i++){ int k=i%4; in[i]= k==0?INFINITY : k==1?-INFINITY : k==2?(float)(i%7)-3.f : NAN; }
      DevBuf di(n*4); di.up(in.data());
      for(int mode=0;mode<3;mode++){ DevBuf doo(n); auto ti=mk({n},ACL_FLOAT,di.p), to=mk({n},ACL_BOOL,doo.p);
        auto ws = mode==0?aclnnIsInfGetWorkspaceSize : mode==1?aclnnIsPosInfGetWorkspaceSize : aclnnIsNegInfGetWorkspaceSize;
        auto rn = mode==0?aclnnIsInf : mode==1?aclnnIsPosInf : aclnnIsNegInf;
        exec2([&](uint64_t*w,aclOpExecutor**e){return ws(ti,to,w,e);}, rn);
        std::vector<uint8_t> got(n); doo.down(got.data()); double bad=0;
        for(int64_t i=0;i<n;i++){ uint8_t r = (mode==0?std::isinf(in[i]):mode==1?(std::isinf(in[i])&&in[i]>0):(std::isinf(in[i])&&in[i]<0))?1:0; if(got[i]!=r)bad=1; }
        report(std::string("IsInf mode")+std::to_string(mode), bad, 0); aclDestroyTensor(ti);aclDestroyTensor(to); } }
    // NanToNum: NaN→0, +Inf→100, -Inf→-100
    { std::vector<float> in={1.f, NAN, INFINITY, -INFINITY, 2.5f, -3.f}; int m=in.size(); DevBuf di(m*4),doo(m*4); di.up(in.data());
      auto ti=mk({m},ACL_FLOAT,di.p), to=mk({m},ACL_FLOAT,doo.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNanToNumGetWorkspaceSize(ti,0.f,100.f,-100.f,to,w,e);}, aclnnNanToNum);
      std::vector<float> got(m); doo.down(got.data()); double bad=0;
      float exp[6]={1.f,0.f,100.f,-100.f,2.5f,-3.f}; for(int i=0;i<m;i++) bad=std::max(bad,(double)std::fabs(got[i]-exp[i]));
      report("NanToNum", bad, 1e-5); aclDestroyTensor(ti);aclDestroyTensor(to); }
    // InplaceTril on [3,3] ones (diagonal 0): keep lower triangle
    { const int N=3; std::vector<float> o(N*N,1.f); DevBuf b(N*N*4); b.up(o.data()); auto t=mk({N,N},ACL_FLOAT,b.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceTrilGetWorkspaceSize(t,0,w,e);}, aclnnInplaceTril);
      std::vector<float> got(N*N); b.down(got.data()); double bad=0;
      for(int r=0;r<N;r++)for(int c=0;c<N;c++) bad=std::max(bad,(double)std::fabs(got[r*N+c]-(c<=r?1.f:0.f)));
      report("InplaceTril", bad, 0); aclDestroyTensor(t); }
    // Clamp to [-1, 2]
    run_un("Clamp[-1,2]", randv(4096,-5.f,5.f), [](const aclTensor*s,aclTensor*o,uint64_t*w,aclOpExecutor**e){ float lo=-1.f,hi=2.f; auto a=aclCreateScalar(&lo,ACL_FLOAT),b=aclCreateScalar(&hi,ACL_FLOAT); return aclnnClampGetWorkspaceSize(s,a,b,o,w,e);}, aclnnClamp, [](double v){ return v<-1?-1.0:(v>2?2.0:v); });
    // ---- generators ----
    // Eye [4,5]
    { const int R=4,C=5; DevBuf d(R*C*4); auto to=mk({R,C},ACL_FLOAT,d.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEyeGetWorkspaceSize(to,w,e);}, aclnnEye);
      std::vector<float> got(R*C); d.down(got.data()); double bad=0;
      for(int r=0;r<R;r++)for(int c=0;c<C;c++) bad=std::max(bad,(double)std::fabs(got[r*C+c]-(r==c?1.f:0.f)));
      report("Eye[4,5]", bad, 0); aclDestroyTensor(to); }
    // Linspace(0,1,steps=11)
    { const int S=11; DevBuf d(S*4); auto to=mk({S},ACL_FLOAT,d.p);
      float a=0.f,b=1.f; auto sa=aclCreateScalar(&a,ACL_FLOAT),sb=aclCreateScalar(&b,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLinspaceGetWorkspaceSize(sa,sb,S,to,w,e);}, aclnnLinspace);
      std::vector<float> got(S); d.down(got.data());
      std::vector<double> r(S); for(int i=0;i<S;i++) r[i]=i/10.0;
      report("Linspace(0,1,11)", norm_err(got,r), 1e-6); aclDestroyTensor(to); }
    // LogSpace(0,3,steps=4,base=2) → 1,2,4,8
    { const int S=4; DevBuf d(S*4); auto to=mk({S},ACL_FLOAT,d.p);
      float a=0.f,b=3.f; auto sa=aclCreateScalar(&a,ACL_FLOAT),sb=aclCreateScalar(&b,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogSpaceGetWorkspaceSize(sa,sb,S,2.0,to,w,e);}, aclnnLogSpace);
      std::vector<float> got(S); d.down(got.data());
      std::vector<double> r={1,2,4,8};
      report("LogSpace(0,3,4,b2)", norm_err(got,r), 1e-6); aclDestroyTensor(to); }
    // Range(2, _, step=0.5) over 6 elements → 2,2.5,...,4.5
    { const int S=6; DevBuf d(S*4); auto to=mk({S},ACL_FLOAT,d.p);
      float a=2.f,b=5.f,st=0.5f; auto sa=aclCreateScalar(&a,ACL_FLOAT),sb=aclCreateScalar(&b,ACL_FLOAT),ss=aclCreateScalar(&st,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRangeGetWorkspaceSize(sa,sb,ss,to,w,e);}, aclnnRange);
      std::vector<float> got(S); d.down(got.data());
      std::vector<double> r(S); for(int i=0;i<S;i++) r[i]=2.0+0.5*i;
      report("Range(2,step0.5)", norm_err(got,r), 1e-6); aclDestroyTensor(to); }
}

// ---------------- R10: additional reductions ----------------
static void t_r10() {
    const int R=8, C=16; auto x=randv(R*C, 0.3f, 2.0f);   // positive (logsum/prod safe)
    auto col = [&](int r){ std::vector<double> v(C); for(int c=0;c<C;c++) v[c]=x[r*C+c]; return v; };
    // full Sum / Max / Min
    { DevBuf dx(R*C*4),dz(4); dx.up(x.data()); auto tx=mk({R,C},ACL_FLOAT,dx.p), tz=mk({},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSumGetWorkspaceSize(tx,ACL_FLOAT,tz,w,e);}, aclnnSum);
      float got; dz.down(&got); double r=0; for(float v:x)r+=v; report("Sum(full)", std::fabs(got-r)/(r+1e-9), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    { DevBuf dx(R*C*4),dz(4); dx.up(x.data()); auto tx=mk({R,C},ACL_FLOAT,dx.p), tz=mk({},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxGetWorkspaceSize(tx,tz,w,e);}, aclnnMax);
      float got; dz.down(&got); double r=-1e30; for(float v:x)r=std::max(r,(double)v); report("Max(full)", std::fabs(got-r), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    // MeanV2 along dim 1 → [R]
    { aclIntArray* ad=aclCreateIntArray((const int64_t[]){1},1); DevBuf dx(R*C*4),dz(R*4); dx.up(x.data());
      auto tx=mk({R,C},ACL_FLOAT,dx.p), tz=mk({R},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMeanV2GetWorkspaceSize(tx,ad,false,ACL_FLOAT,tz,w,e);}, aclnnMeanV2);
      std::vector<float> got(R); dz.down(got.data()); std::vector<double> r(R); for(int i=0;i<R;i++){double s=0;for(double v:col(i))s+=v; r[i]=s/C;}
      report("MeanV2(dim1)", norm_err(got,r), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); aclDestroyIntArray(ad); }
    // ReduceLogSum along dim 1 → [R]
    { aclIntArray* ad=aclCreateIntArray((const int64_t[]){1},1); DevBuf dx(R*C*4),dz(R*4); dx.up(x.data());
      auto tx=mk({R,C},ACL_FLOAT,dx.p), tz=mk({R},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReduceLogSumGetWorkspaceSize(tx,ad,false,tz,w,e);}, aclnnReduceLogSum);
      std::vector<float> got(R); dz.down(got.data()); std::vector<double> r(R); for(int i=0;i<R;i++){double s=0;for(double v:col(i))s+=v; r[i]=std::log(s);}
      report("ReduceLogSum(dim1)", norm_err(got,r), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tz); aclDestroyIntArray(ad); }
    // ProdDim along dim 1 → [R]
    { DevBuf dx(R*C*4),dz(R*4); dx.up(x.data()); auto tx=mk({R,C},ACL_FLOAT,dx.p), tz=mk({R},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnProdDimGetWorkspaceSize(tx,1,false,ACL_FLOAT,tz,w,e);}, aclnnProdDim);
      std::vector<float> got(R); dz.down(got.data()); std::vector<double> r(R); for(int i=0;i<R;i++){double p=1;for(double v:col(i))p*=v; r[i]=p;}
      report("ProdDim(dim1)", norm_err(got,r), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tz); }
    // MaxDim along dim 1 → values[R] + indices[R]
    { DevBuf dx(R*C*4),dv(R*4),di(R*8); dx.up(x.data()); auto tx=mk({R,C},ACL_FLOAT,dx.p), tv=mk({R},ACL_FLOAT,dv.p), ti=mk({R},ACL_INT64,di.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxDimGetWorkspaceSize(tx,1,false,tv,ti,w,e);}, aclnnMaxDim);
      std::vector<float> gv(R); std::vector<int64_t> gi(R); dv.down(gv.data()); di.down(gi.data());
      double bad=0; for(int i=0;i<R;i++){ auto cv=col(i); int am=0; for(int c=1;c<C;c++) if(cv[c]>cv[am])am=c; bad=std::max(bad,std::fabs(gv[i]-cv[am])); if(gi[i]!=am)bad=std::max(bad,1.0); }
      report("MaxDim(values+idx)", bad, 1e-5); aclDestroyTensor(tx);aclDestroyTensor(tv);aclDestroyTensor(ti); }
    // VarMean along dim 1 → var[R] + mean[R]
    { aclIntArray* ad=aclCreateIntArray((const int64_t[]){1},1); DevBuf dx(R*C*4),dvar(R*4),dmean(R*4); dx.up(x.data());
      auto tx=mk({R,C},ACL_FLOAT,dx.p), tvar=mk({R},ACL_FLOAT,dvar.p), tmean=mk({R},ACL_FLOAT,dmean.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnVarMeanGetWorkspaceSize(tx,ad,false,tvar,tmean,w,e);}, aclnnVarMean);
      std::vector<float> gv(R),gm(R); dvar.down(gv.data()); dmean.down(gm.data());
      std::vector<double> rv(R),rm(R); for(int i=0;i<R;i++){auto cv=col(i); double m=0;for(double v:cv)m+=v;m/=C; double s=0;for(double v:cv)s+=(v-m)*(v-m); rv[i]=s/(C-1); rm[i]=m;}
      report("VarMean(var)", norm_err(gv,rv), 1e-4); report("VarMean(mean)", norm_err(gm,rm), 1e-5);
      aclDestroyTensor(tx);aclDestroyTensor(tvar);aclDestroyTensor(tmean); aclDestroyIntArray(ad); }
}

static void t_r11(){
    { // LogSigmoidForward: out = log(sigmoid(x))
      const int n=64; auto x=randv(n,-3,3); DevBuf dx(n*4),do_(n*4),db(n*4); dx.up(x.data());
      auto tx=mk({n},ACL_FLOAT,dx.p),to=mk({n},ACL_FLOAT,do_.p),tb=mk({n},ACL_FLOAT,db.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLogSigmoidForwardGetWorkspaceSize(tx,to,tb,w,e);}, aclnnLogSigmoidForward);
      std::vector<float> o(n); do_.down(o.data()); std::vector<double> rf(n); for(int i=0;i<n;i++) rf[i]=std::log(1.0/(1.0+std::exp(-x[i])));
      report("LogSigmoidForward", norm_err(o,rf), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(to);aclDestroyTensor(tb); }
    { // FatreluMul: (a>thr?a:0)*b
      const int n=64; auto a=randv(n,-2,2),b=randv(n,-2,2); double thr=0.3;
      DevBuf da(n*4),db(n*4),do_(n*4); da.up(a.data()); db.up(b.data());
      auto ta=mk({n},ACL_FLOAT,da.p),tb=mk({n},ACL_FLOAT,db.p),to=mk({n},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFatreluMulGetWorkspaceSize(ta,tb,thr,to,w,e);}, aclnnFatreluMul);
      std::vector<float> o(n); do_.down(o.data()); std::vector<double> rf(n); for(int i=0;i<n;i++) rf[i]=(a[i]>thr?a[i]:0.0)*b[i];
      report("FatreluMul", norm_err(o,rf), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(to); }
    { // AxpyV2: alpha*x + y
      const int n=64; auto x=randv(n,-2,2),y=randv(n,-2,2); double al=1.7;
      DevBuf dx(n*4),dy(n*4),do_(n*4); dx.up(x.data()); dy.up(y.data());
      auto tx=mk({n},ACL_FLOAT,dx.p),ty=mk({n},ACL_FLOAT,dy.p),to=mk({n},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAxpyV2GetWorkspaceSize(tx,ty,al,to,w,e);}, aclnnAxpyV2);
      std::vector<float> o(n); do_.down(o.data()); std::vector<double> rf(n); for(int i=0;i<n;i++) rf[i]=al*x[i]+y[i];
      report("AxpyV2", norm_err(o,rf), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(ty);aclDestroyTensor(to); }
    { // VarCorrection: per-row variance with correction=1 (Bessel)
      const int R=4,C=8; auto x=randv(R*C,-2,2); DevBuf dx(R*C*4),dv(R*4); dx.up(x.data());
      auto tx=mk({R,C},ACL_FLOAT,dx.p),tv=mk({R},ACL_FLOAT,dv.p); aclIntArray*ad=aclCreateIntArray((const int64_t[]){1},1);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnVarCorrectionGetWorkspaceSize(tx,ad,1,false,tv,w,e);}, aclnnVarCorrection);
      std::vector<float> v(R); dv.down(v.data()); std::vector<double> rf(R); for(int r=0;r<R;r++){ double m=0;for(int c=0;c<C;c++)m+=x[r*C+c];m/=C; double s=0;for(int c=0;c<C;c++){double u=x[r*C+c]-m;s+=u*u;} rf[r]=s/(C-1); }
      report("VarCorrection", norm_err(v,rf), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tv);aclDestroyIntArray(ad); }
    { // Pdist: condensed pairwise L2 within X[N,D]
      const int N=4,D=3; auto x=randv(N*D,-2,2); int T=N*(N-1)/2; DevBuf dx(N*D*4),do_(T*4); dx.up(x.data());
      auto tx=mk({N,D},ACL_FLOAT,dx.p),to=mk({T},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPdistGetWorkspaceSize(tx,2.0,to,w,e);}, aclnnPdist);
      std::vector<float> o(T); do_.down(o.data()); std::vector<double> rf; for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){ double s=0;for(int d=0;d<D;d++){double u=x[i*D+d]-x[j*D+d];s+=u*u;} rf.push_back(std::sqrt(s)); }
      report("Pdist", norm_err(o,rf), 1e-5); aclDestroyTensor(tx);aclDestroyTensor(to); }
    { // FFN: relu(x@W1+b1)@W2+b2
      const int M=3,K=4,Hd=5,N=2; auto x=randv(M*K,-1,1),W1=randv(K*Hd,-1,1),b1=randv(Hd,-1,1),W2=randv(Hd*N,-1,1),b2=randv(N,-1,1);
      DevBuf dx(M*K*4),dw1(K*Hd*4),db1(Hd*4),dw2(Hd*N*4),db2(N*4),do_(M*N*4); dx.up(x.data());dw1.up(W1.data());db1.up(b1.data());dw2.up(W2.data());db2.up(b2.data());
      auto tx=mk({M,K},ACL_FLOAT,dx.p),tw1=mk({K,Hd},ACL_FLOAT,dw1.p),tb1=mk({Hd},ACL_FLOAT,db1.p),tw2=mk({Hd,N},ACL_FLOAT,dw2.p),tb2=mk({N},ACL_FLOAT,db2.p),to=mk({M,N},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnFFNGetWorkspaceSize(tx,tw1,tb1,tw2,tb2,0,to,w,e);}, aclnnFFN);
      std::vector<float> o(M*N); do_.down(o.data()); std::vector<double> rf(M*N);
      for(int m=0;m<M;m++){ std::vector<double> h(Hd); for(int j=0;j<Hd;j++){ double a=b1[j]; for(int k=0;k<K;k++)a+=x[m*K+k]*W1[k*Hd+j]; h[j]=std::max(a,0.0); }
        for(int nn=0;nn<N;nn++){ double a=b2[nn]; for(int j=0;j<Hd;j++)a+=h[j]*W2[j*N+nn]; rf[m*N+nn]=a; } }
      report("FFN", norm_err(o,rf), 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tw1);aclDestroyTensor(tb1);aclDestroyTensor(tw2);aclDestroyTensor(tb2);aclDestroyTensor(to); }
    { // Sinkhorn: result rows/cols approximately balanced after iterations (col sums ≈ const after final col-normalize)
      const int R=3,C=3; auto m=randv(R*C,0.1,2.0); DevBuf dm(R*C*4),do_(R*C*4); dm.up(m.data());
      auto tm=mk({R,C},ACL_FLOAT,dm.p),to=mk({R,C},ACL_FLOAT,do_.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSinkhornGetWorkspaceSize(tm,1.0,20,to,w,e);}, aclnnSinkhorn);
      std::vector<float> o(R*C); do_.down(o.data()); double bad=0; for(int c=0;c<C;c++){ double s=0; for(int r=0;r<R;r++)s+=o[r*C+c]; bad=std::max(bad,std::fabs(s-1.0)); }
      report("Sinkhorn(col-sum=1)", bad, 1e-3); aclDestroyTensor(tm);aclDestroyTensor(to); }
}
static void t_r12(){
    { // Equal elementwise
      std::vector<float> a={1,2,3,4,5},b={1,9,3,9,5}; int n=5; DevBuf da(n*4),db(n*4),dout(n); da.up(a.data()); db.up(b.data());
      auto ta=mk({n},ACL_FLOAT,da.p),tb=mk({n},ACL_FLOAT,db.p),to=mk({n},ACL_BOOL,dout.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEqualGetWorkspaceSize(ta,tb,to,w,e);}, aclnnEqual);
      std::vector<uint8_t> o(n); dout.down(o.data()); double bad=0; for(int i=0;i<n;i++) if(o[i]!=(a[i]==b[i])) bad=1;
      report("Equal", bad, 0.0); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(to); }
    { // Iou of axis-aligned boxes
      std::vector<float> a={0,0,2,2, 0,0,4,4},b={1,1,3,3, 2,2,6,6}; int N=2; DevBuf da(N*4*4),db(N*4*4),dout(N*4); da.up(a.data()); db.up(b.data());
      auto ta=mk({N,4},ACL_FLOAT,da.p),tb=mk({N,4},ACL_FLOAT,db.p),to=mk({N},ACL_FLOAT,dout.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIouGetWorkspaceSize(ta,tb,to,w,e);}, aclnnIou);
      std::vector<float> o(N); dout.down(o.data()); double me=0,mr=0;
      for(int i=0;i<N;i++){ const float*A=&a[i*4],*B=&b[i*4]; double ix1=std::max(A[0],B[0]),iy1=std::max(A[1],B[1]),ix2=std::min(A[2],B[2]),iy2=std::min(A[3],B[3]);
        double iw=std::max(ix2-ix1,0.0),ih=std::max(iy2-iy1,0.0),inter=iw*ih,ua=(A[2]-A[0])*(A[3]-A[1])+(B[2]-B[0])*(B[3]-B[1])-inter; double r=ua>0?inter/ua:0; me=std::max(me,std::fabs(o[i]-r)); mr=std::max(mr,std::fabs(r)); }
      report("Iou", me/(mr+1e-9), 1e-5); aclDestroyTensor(ta);aclDestroyTensor(tb);aclDestroyTensor(to); }
    { // SignBitsPack/Unpack roundtrip
      const int n=8; std::vector<float> x={-1,2,-3,4,-5,6,-7,8}; DevBuf dx(n*4),dp(1),du(n*4); dx.up(x.data());
      auto tx=mk({n},ACL_FLOAT,dx.p),tp=mk({1},ACL_UINT8,dp.p),tu=mk({n},ACL_FLOAT,du.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSignBitsPackGetWorkspaceSize(tx,tp,w,e);}, aclnnSignBitsPack);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSignBitsUnpackGetWorkspaceSize(tp,tu,w,e);}, aclnnSignBitsUnpack);
      std::vector<float> u(n); du.down(u.data()); double bad=0; for(int i=0;i<n;i++){ float ref=x[i]<0?-1.f:1.f; if(u[i]!=ref) bad=1; }
      report("SignBitsPack/Unpack", bad, 0.0); aclDestroyTensor(tx);aclDestroyTensor(tp);aclDestroyTensor(tu); }
}
int main(){ init(); srand(101); t_r1(); t_r2(); t_r3(); t_r4(); t_r5(); t_r6(); t_r7(); t_r8(); t_r9(); t_r10(); t_r11(); t_r12(); return finish(); }
