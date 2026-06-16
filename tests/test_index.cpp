// Indexing extensions (P4) cross-check: triangular, trace, diagonal/diagflat, bincount, searchsorted/bucketize,
// index_add/fill/copy, scatter max/min/mul, take/take_along_dim, masked_scatter, narrow. CPU double reference + tolerance.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <algorithm>
using namespace hn;

static void t_tri() {
    const int M = 6, N = 7; auto fa = randv(M*N, -2, 2);
    for (int lower = 0; lower <= 1; lower++) for (int kk = -1; kk <= 1; kk++) {
        std::vector<float> hz(M*N);
        DevBuf da(M*N*4), dz(M*N*4); da.up(fa.data());
        aclTensor *ta = mk({M,N}, ACL_FLOAT, da.p), *tz = mk({M,N}, ACL_FLOAT, dz.p);
        if (lower) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTrilGetWorkspaceSize(ta,kk,tz,w,e);}, aclnnTril);
        else       exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTriuGetWorkspaceSize(ta,kk,tz,w,e);}, aclnnTriu);
        dz.down(hz.data());
        double bad = 0;
        for (int r=0;r<M;r++) for (int c=0;c<N;c++) { bool keep = lower ? (c<=r+kk) : (c>=r+kk); double ref = keep?fa[r*N+c]:0; bad=std::max(bad,(double)std::fabs(hz[r*N+c]-ref)); }
        report(std::string(lower?"Tril k=":"Triu k=")+std::to_string(kk), bad, 0);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
}
static void t_trace_diag() {
    const int M = 5, N = 6; auto fa = randv(M*N, -3, 3);
    { std::vector<float> hz(1); DevBuf da(M*N*4), dz(4); da.up(fa.data());
      aclTensor *ta = mk({M,N}, ACL_FLOAT, da.p), *tz = mk({1}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTraceGetWorkspaceSize(ta,tz,w,e);}, aclnnTrace);
      dz.down(hz.data()); double ref=0; for(int i=0;i<std::min(M,N);i++) ref+=fa[i*N+i];
      report("Trace [5,6]", std::fabs(hz[0]-ref)/(std::fabs(ref)+1e-9), 1e-6);
      aclDestroyTensor(ta); aclDestroyTensor(tz); }
    for (int off = -1; off <= 2; off++) {
        int roff = off>=0?0:-off, coff = off>=0?off:0; int len = std::min(M-roff, N-coff);
        std::vector<float> hz(len); DevBuf da(M*N*4), dz(len*4); da.up(fa.data());
        aclTensor *ta = mk({M,N}, ACL_FLOAT, da.p), *tz = mk({(int64_t)len}, ACL_FLOAT, dz.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDiagonalGetWorkspaceSize(ta,off,tz,w,e);}, aclnnDiagonal);
        dz.down(hz.data()); double bad=0; for(int i=0;i<len;i++) bad=std::max(bad,(double)std::fabs(hz[i]-fa[(roff+i)*N+(coff+i)]));
        report(std::string("Diagonal off=")+std::to_string(off), bad, 0);
        aclDestroyTensor(ta); aclDestroyTensor(tz);
    }
    { const int L=5, off=1, S=L+off; auto v=randv(L,-2,2); std::vector<float> hz(S*S);
      DevBuf da(L*4), dz(S*S*4); da.up(v.data());
      aclTensor *ta = mk({L}, ACL_FLOAT, da.p), *tz = mk({S,S}, ACL_FLOAT, dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnDiagFlatGetWorkspaceSize(ta,off,tz,w,e);}, aclnnDiagFlat);
      dz.down(hz.data()); double bad=0;
      for(int r=0;r<S;r++)for(int c=0;c<S;c++){ double ref=(c==r+off && r<L)?v[r]:0; bad=std::max(bad,(double)std::fabs(hz[r*S+c]-ref)); }
      report("DiagFlat off=1", bad, 0); aclDestroyTensor(ta); aclDestroyTensor(tz); }
}
static void t_bincount_search() {
    const int n = 64, C = 10;
    std::vector<int32_t> x(n); for (auto &v : x) v = rand() % C;
    std::vector<int64_t> hc(C);
    DevBuf dx(n*4), dc(C*8); dx.up(x.data());
    aclTensor *tx = mk({n}, ACL_INT32, dx.p), *tc = mk({C}, ACL_INT64, dc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBincountGetWorkspaceSize(tx,C,tc,w,e);}, aclnnBincount);
    dc.down(hc.data());
    std::vector<int64_t> ref(C,0); for (int v : x) ref[v]++;
    int64_t bad=0; for(int i=0;i<C;i++) if(hc[i]!=ref[i]) bad++;
    report("Bincount", bad?1.0:0.0, 0);
    aclDestroyTensor(tx); aclDestroyTensor(tc);
    // searchsorted: boundaries sorted, values arbitrary
    const int B = 8, nv = 32;
    std::vector<float> bnd(B); for(int i=0;i<B;i++) bnd[i]=i*1.0f - 4;
    auto val = randv(nv, -5, 5);
    for (int right = 0; right <= 1; right++) {
        std::vector<int64_t> ho(nv);
        DevBuf db(B*4), dvv(nv*4), dout(nv*8); db.up(bnd.data()); dvv.up(val.data());
        aclTensor *tb = mk({B}, ACL_FLOAT, db.p), *tv = mk({nv}, ACL_FLOAT, dvv.p), *to = mk({nv}, ACL_INT64, dout.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSearchSortedGetWorkspaceSize(tb,tv,(bool)right,to,w,e);}, aclnnSearchSorted);
        dout.down(ho.data());
        int64_t bad2=0; for(int i=0;i<nv;i++){ int64_t r=0; for(int j=0;j<B;j++){ if(right?(bnd[j]<=val[i]):(bnd[j]<val[i])) r++; } if(ho[i]!=r) bad2++; }
        report(std::string("SearchSorted right=")+std::to_string(right), bad2?1.0:0.0, 0);
        aclDestroyTensor(tb); aclDestroyTensor(tv); aclDestroyTensor(to);
    }
}
static void t_index_scatter() {
    const int V = 8, row = 4, L = 5;
    auto self = randv(V*row, -2, 2), src = randv(L*row, -2, 2);
    std::vector<int64_t> idx(L); for (auto &v : idx) v = rand() % V;
    std::vector<int64_t> uniq(L); for (int i=0;i<L;i++) uniq[i]=i;   // distinct rows: IndexCopy/IndexFill are order-sensitive on duplicates
    auto run_idx = [&](const char *name, int kind, double param) {
        const std::vector<int64_t> &ix = (kind==0) ? idx : uniq;     // IndexAdd accumulates (duplicates OK)
        std::vector<float> hz(V*row);
        DevBuf ds(V*row*4), di(L*8), dsrc(L*row*4), dz(V*row*4); ds.up(self.data()); di.up(ix.data()); dsrc.up(src.data());
        aclTensor *tself=mk({V,row},ACL_FLOAT,ds.p),*tidx=mk({L},ACL_INT64,di.p),*tsrc=mk({L,row},ACL_FLOAT,dsrc.p),*tz=mk({V,row},ACL_FLOAT,dz.p);
        if (kind==0) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexAddGetWorkspaceSize(tself,0,tidx,tsrc,param,tz,w,e);}, aclnnIndexAdd);
        else if (kind==1) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexFillGetWorkspaceSize(tself,0,tidx,param,tz,w,e);}, aclnnIndexFill);
        else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIndexCopyGetWorkspaceSize(tself,0,tidx,tsrc,tz,w,e);}, aclnnIndexCopy);
        dz.down(hz.data());
        std::vector<double> ref(self.begin(), self.end());
        for (int l=0;l<L;l++) for (int c=0;c<row;c++) { int64_t r=ix[l];
            if(kind==0) ref[r*row+c]+=param*src[l*row+c]; else if(kind==1) ref[r*row+c]=param; else ref[r*row+c]=src[l*row+c]; }
        double bad=0; for(int i=0;i<V*row;i++) bad=std::max(bad,(double)std::fabs(hz[i]-ref[i]));
        report(name, bad, kind==0?1e-5:0.0);   // index_add: duplicate-index float accum tolerance
        aclDestroyTensor(tself);aclDestroyTensor(tidx);aclDestroyTensor(tsrc);aclDestroyTensor(tz);
    };
    run_idx("IndexAdd a=0.5", 0, 0.5);
    run_idx("IndexFill v=9", 1, 9.0);
    run_idx("IndexCopy", 2, 0);
    // scatter max/min/mul: use unique indices to make result deterministic
    std::vector<int64_t> uidx(L); for(int i=0;i<L;i++) uidx[i]=i;   // distinct
    auto run_scat = [&](const char *name, int kind) {
        std::vector<float> hz(V*row);
        DevBuf ds(V*row*4), di(L*8), dsrc(L*row*4), dz(V*row*4); ds.up(self.data()); di.up(uidx.data()); dsrc.up(src.data());
        aclTensor *tself=mk({V,row},ACL_FLOAT,ds.p),*tidx=mk({L},ACL_INT64,di.p),*tsrc=mk({L,row},ACL_FLOAT,dsrc.p),*tz=mk({V,row},ACL_FLOAT,dz.p);
        if(kind==0) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterMaxGetWorkspaceSize(tself,tidx,tsrc,tz,w,e);}, aclnnScatterMax);
        else if(kind==1) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterMinGetWorkspaceSize(tself,tidx,tsrc,tz,w,e);}, aclnnScatterMin);
        else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterMulGetWorkspaceSize(tself,tidx,tsrc,tz,w,e);}, aclnnScatterMul);
        dz.down(hz.data());
        std::vector<double> ref(self.begin(), self.end());
        for(int l=0;l<L;l++)for(int c=0;c<row;c++){int64_t r=uidx[l]; double s=src[l*row+c];
            if(kind==0) ref[r*row+c]=std::max(ref[r*row+c],s); else if(kind==1) ref[r*row+c]=std::min(ref[r*row+c],s); else ref[r*row+c]*=s; }
        double bad=0; for(int i=0;i<V*row;i++) bad=std::max(bad,(double)std::fabs(hz[i]-ref[i]));
        report(name, bad, 1e-5); aclDestroyTensor(tself);aclDestroyTensor(tidx);aclDestroyTensor(tsrc);aclDestroyTensor(tz);
    };
    run_scat("ScatterMax", 0); run_scat("ScatterMin", 1); run_scat("ScatterMul", 2);
}
static void t_take() {
    const int M=4, N=5, n=M*N; auto fa = randv(n, -3, 3);
    { const int k=7; std::vector<int64_t> idx(k); for(auto&v:idx) v=rand()%n; std::vector<float> hz(k);
      DevBuf da(n*4), di(k*8), dz(k*4); da.up(fa.data()); di.up(idx.data());
      aclTensor *ta=mk({M,N},ACL_FLOAT,da.p),*ti=mk({k},ACL_INT64,di.p),*tz=mk({k},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTakeGetWorkspaceSize(ta,ti,tz,w,e);}, aclnnTake);
      dz.down(hz.data()); double bad=0; for(int i=0;i<k;i++) bad=std::max(bad,(double)std::fabs(hz[i]-fa[idx[i]]));
      report("Take", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(ti);aclDestroyTensor(tz); }
    // take_along_dim along dim1: index [M,N] each in [0,N)
    { std::vector<int64_t> idx(n); for(auto&v:idx) v=rand()%N; std::vector<float> hz(n);
      DevBuf da(n*4), di(n*8), dz(n*4); da.up(fa.data()); di.up(idx.data());
      aclTensor *ta=mk({M,N},ACL_FLOAT,da.p),*ti=mk({M,N},ACL_INT64,di.p),*tz=mk({M,N},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTakeAlongDimGetWorkspaceSize(ta,ti,1,tz,w,e);}, aclnnTakeAlongDim);
      dz.down(hz.data()); double bad=0; for(int r=0;r<M;r++)for(int c=0;c<N;c++) bad=std::max(bad,(double)std::fabs(hz[r*N+c]-fa[r*N+idx[r*N+c]]));
      report("TakeAlongDim dim1", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(ti);aclDestroyTensor(tz); }
    // masked_scatter
    { const int n2=20; auto self=randv(n2,-1,1), src=randv(n2,5,6); std::vector<uint8_t> mask(n2); for(auto&m:mask) m=rand()&1;
      std::vector<float> hz(n2); DevBuf ds(n2*4), dm(n2), dsrc(n2*4), dz(n2*4); ds.up(self.data()); dm.up(mask.data()); dsrc.up(src.data());
      aclTensor *tself=mk({n2},ACL_FLOAT,ds.p),*tm=mk({n2},ACL_BOOL,dm.p),*tsrc=mk({n2},ACL_FLOAT,dsrc.p),*tz=mk({n2},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedScatterGetWorkspaceSize(tself,tm,tsrc,tz,w,e);}, aclnnMaskedScatter);
      dz.down(hz.data()); std::vector<double> ref(self.begin(),self.end()); int pos=0; for(int i=0;i<n2;i++) if(mask[i]) ref[i]=src[pos++];
      double bad=0; for(int i=0;i<n2;i++) bad=std::max(bad,(double)std::fabs(hz[i]-ref[i]));
      report("MaskedScatter", bad, 0); aclDestroyTensor(tself);aclDestroyTensor(tm);aclDestroyTensor(tsrc);aclDestroyTensor(tz); }
    // narrow along dim0
    { const int M2=6,N2=4,start=2,len=3; auto fa2=randv(M2*N2,-2,2); std::vector<float> hz(len*N2);
      DevBuf da(M2*N2*4), dz(len*N2*4); da.up(fa2.data());
      aclTensor *ta=mk({M2,N2},ACL_FLOAT,da.p),*tz=mk({len,N2},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNarrowGetWorkspaceSize(ta,0,start,len,tz,w,e);}, aclnnNarrow);
      dz.down(hz.data()); double bad=0; for(int r=0;r<len;r++)for(int c=0;c<N2;c++) bad=std::max(bad,(double)std::fabs(hz[r*N2+c]-fa2[(start+r)*N2+c]));
      report("Narrow dim0", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}

static void t_shape_ext() {
    // Reshape / Squeeze / Unsqueeze: data preserved
    { const int n=24; auto fa=randv(n,-2,2); std::vector<float> hz(n); DevBuf da(n*4),dz(n*4); da.up(fa.data());
      aclTensor *ta=mk({4,6},ACL_FLOAT,da.p),*tz=mk({2,3,4},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReshapeGetWorkspaceSize(ta,tz,w,e);}, aclnnReshape);
      dz.down(hz.data()); double bad=0; for(int i=0;i<n;i++) bad=std::max(bad,(double)std::fabs(hz[i]-fa[i]));
      report("Reshape", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // Movedim: [2,3,4] move dim0->dim2 => [3,4,2]
    { const int A=2,B=3,C=4; auto fa=randv(A*B*C,-2,2); std::vector<float> hz(A*B*C);
      DevBuf da(A*B*C*4),dz(A*B*C*4); da.up(fa.data());
      aclTensor *ta=mk({A,B,C},ACL_FLOAT,da.p),*tz=mk({B,C,A},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMovedimGetWorkspaceSize(ta,0,2,tz,w,e);}, aclnnMovedim);
      dz.down(hz.data()); double bad=0;
      for(int a=0;a<A;a++)for(int b=0;b<B;b++)for(int c=0;c<C;c++) bad=std::max(bad,(double)std::fabs(hz[(b*C+c)*A+a]-fa[(a*B+b)*C+c]));
      report("Movedim 0->2", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // Rot90 k=1 on [3,4]
    for (int k=1;k<=3;k++) { const int H=3,W=4; auto fa=randv(H*W,-2,2);
      int oH=(k%2)?W:H, oW=(k%2)?H:W; std::vector<float> hz(oH*oW);
      DevBuf da(H*W*4),dz(oH*oW*4); da.up(fa.data());
      aclTensor *ta=mk({H,W},ACL_FLOAT,da.p),*tz=mk({oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRot90GetWorkspaceSize(ta,k,tz,w,e);}, aclnnRot90);
      dz.down(hz.data()); double bad=0;
      for(int r=0;r<oH;r++)for(int c=0;c<oW;c++){ int ih,iw; if(k==1){ih=c;iw=W-1-r;} else if(k==2){ih=H-1-r;iw=W-1-c;} else {ih=H-1-c;iw=r;}
          bad=std::max(bad,(double)std::fabs(hz[r*oW+c]-fa[ih*W+iw])); }
      report(std::string("Rot90 k=")+std::to_string(k), bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // 2D pads on [1,1,4,4], pad=1 each side
    { const int H=4,W=4,p=1,oH=H+2*p,oW=W+2*p; auto fa=randv(H*W,-2,2);
      int64_t pads[4]={p,p,p,p}; aclIntArray *pad=aclCreateIntArray(pads,4);
      auto refl=[&](int x,int n){ if(n==1)return 0; while(x<0||x>=n){if(x<0)x=-x; if(x>=n)x=2*(n-1)-x;} return x; };
      auto repl=[&](int x,int n){ return x<0?0:(x>=n?n-1:x); };
      auto circ=[&](int x,int n){ return ((x%n)+n)%n; };
      for(int mode=0;mode<3;mode++){
        std::vector<float> hz(oH*oW); DevBuf da(H*W*4),dz(oH*oW*4); da.up(fa.data());
        aclTensor *ta=mk({1,1,H,W},ACL_FLOAT,da.p),*tz=mk({1,1,oH,oW},ACL_FLOAT,dz.p);
        if(mode==0) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReflectionPad2dGetWorkspaceSize(ta,pad,tz,w,e);}, aclnnReflectionPad2d);
        else if(mode==1) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnReplicationPad2dGetWorkspaceSize(ta,pad,tz,w,e);}, aclnnReplicationPad2d);
        else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnCircularPad2dGetWorkspaceSize(ta,pad,tz,w,e);}, aclnnCircularPad2d);
        dz.down(hz.data()); double bad=0;
        for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ int ih=oh-p,iw=ow-p;
            if(mode==0){ih=refl(ih,H);iw=refl(iw,W);} else if(mode==1){ih=repl(ih,H);iw=repl(iw,W);} else {ih=circ(ih,H);iw=circ(iw,W);}
            bad=std::max(bad,(double)std::fabs(hz[oh*oW+ow]-fa[ih*W+iw])); }
        report(std::string("Pad2d mode")+std::to_string(mode), bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz);
      }
      aclDestroyIntArray(pad); }
    // Im2Col on [1,2,4,4], k=3 s=1 p=0 d=1 -> [1, 2*9, 4]  (oH=oW=2)
    { const int N=1,C=2,H=4,W=4,kh=3,kw=3; auto fa=randv(C*H*W,-2,2);
      int oH=2,oW=2,L=oH*oW,K=C*kh*kw; std::vector<float> hz(K*L);
      int64_t kk[2]={kh,kw},st[2]={1,1},pd[2]={0,0},dl[2]={1,1};
      aclIntArray *ak=aclCreateIntArray(kk,2),*as=aclCreateIntArray(st,2),*ap=aclCreateIntArray(pd,2),*ad=aclCreateIntArray(dl,2);
      DevBuf da(C*H*W*4),dz(K*L*4); da.up(fa.data());
      aclTensor *ta=mk({N,C,H,W},ACL_FLOAT,da.p),*tz=mk({N,K,L},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnIm2ColGetWorkspaceSize(ta,ak,ad,ap,as,tz,w,e);}, aclnnIm2Col);
      dz.down(hz.data()); double bad=0;
      for(int kr=0;kr<K;kr++)for(int l=0;l<L;l++){ int ow=l%oW,oh=l/oW,kj=kr%kw,ki=(kr/kw)%kh,c=kr/(kh*kw);
          int ih=oh+ki,iw=ow+kj; double ref=fa[(c*H+ih)*W+iw]; bad=std::max(bad,(double)std::fabs(hz[kr*L+l]-ref)); }
      report("Im2Col", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz);
      aclDestroyIntArray(ak);aclDestroyIntArray(as);aclDestroyIntArray(ap);aclDestroyIntArray(ad); }
    // PixelShuffle r=2: [1,4,2,2] -> [1,1,4,4], then Unshuffle back
    { const int N=1,C=1,r=2,H=2,W=2,Crr=C*r*r; auto fa=randv(Crr*H*W,-2,2);
      std::vector<float> hz(C*H*r*W*r), hb(Crr*H*W);
      DevBuf da(Crr*H*W*4),dz(C*H*r*W*r*4),db(Crr*H*W*4); da.up(fa.data());
      aclTensor *ta=mk({N,Crr,H,W},ACL_FLOAT,da.p),*tz=mk({N,C,H*r,W*r},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPixelShuffleGetWorkspaceSize(ta,r,tz,w,e);}, aclnnPixelShuffle);
      aclTensor *tz2=mk({N,C,H*r,W*r},ACL_FLOAT,dz.p),*tb=mk({N,Crr,H,W},ACL_FLOAT,db.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnPixelUnshuffleGetWorkspaceSize(tz2,r,tb,w,e);}, aclnnPixelUnshuffle);
      db.down(hb.data()); double bad=0; for(int i=0;i<Crr*H*W;i++) bad=std::max(bad,(double)std::fabs(hb[i]-fa[i]));
      report("PixelShuffle/Unshuffle roundtrip", bad, 0);
      aclDestroyTensor(ta);aclDestroyTensor(tz);aclDestroyTensor(tz2);aclDestroyTensor(tb); }
    // ChannelShuffle groups=2 on [1,4,1]
    { const int N=1,C=4,HW=2,g=2; auto fa=randv(C*HW,-2,2); std::vector<float> hz(C*HW);
      DevBuf da(C*HW*4),dz(C*HW*4); da.up(fa.data());
      aclTensor *ta=mk({N,C,HW},ACL_FLOAT,da.p),*tz=mk({N,C,HW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnChannelShuffleGetWorkspaceSize(ta,g,tz,w,e);}, aclnnChannelShuffle);
      dz.down(hz.data()); double bad=0; int cpg=C/g;
      for(int c=0;c<C;c++){ int src=(c%g)*cpg+c/g; for(int h=0;h<HW;h++) bad=std::max(bad,(double)std::fabs(hz[c*HW+h]-fa[src*HW+h])); }
      report("ChannelShuffle g=2", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}

static void t_scatter2() {
    const int A=5, L=3, row=4;
    std::vector<float> self(A*row); for(int i=0;i<A*row;i++) self[i]=(float)i*0.1f;
    std::vector<float> src(L*row); for(int i=0;i<L*row;i++) src[i]=1.0f+(float)i;
    std::vector<int64_t> idx={1,3,1};   // duplicate index 1 → tests add accumulation
    // Scatter add
    { DevBuf ds(A*row*4),dsr(L*row*4),di(L*8),dz(A*row*4); ds.up(self.data()); dsr.up(src.data()); di.up(idx.data());
      auto tself=mk({A,row},ACL_FLOAT,ds.p),tsrc=mk({L,row},ACL_FLOAT,dsr.p),tidx=mk({L},ACL_INT64,di.p),tz=mk({A,row},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterGetWorkspaceSize(tself,0,tidx,tsrc,1,tz,w,e);}, aclnnScatter);
      std::vector<float> got(A*row); dz.down(got.data());
      std::vector<float> ref=self; for(int l=0;l<L;l++)for(int c=0;c<row;c++) ref[idx[l]*row+c]+=src[l*row+c];
      double bad=0; for(int i=0;i<A*row;i++) bad=std::max(bad,(double)std::fabs(got[i]-ref[i]));
      report("Scatter(add)", bad, 1e-5); aclDestroyTensor(tself);aclDestroyTensor(tsrc);aclDestroyTensor(tidx);aclDestroyTensor(tz); }
    // InplaceIndexCopy
    { DevBuf ds(A*row*4),dsr(L*row*4),di(L*8); ds.up(self.data()); dsr.up(src.data());
      std::vector<int64_t> idx2={0,2,4}; di.up(idx2.data());
      auto tself=mk({A,row},ACL_FLOAT,ds.p),tsrc=mk({L,row},ACL_FLOAT,dsr.p),tidx=mk({L},ACL_INT64,di.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceIndexCopyGetWorkspaceSize(tself,0,tidx,tsrc,w,e);}, aclnnInplaceIndexCopy);
      std::vector<float> got(A*row); ds.down(got.data());
      std::vector<float> ref=self; for(int l=0;l<L;l++)for(int c=0;c<row;c++) ref[idx2[l]*row+c]=src[l*row+c];
      double bad=0; for(int i=0;i<A*row;i++) bad=std::max(bad,(double)std::fabs(got[i]-ref[i]));
      report("InplaceIndexCopy", bad, 1e-5); aclDestroyTensor(tself);aclDestroyTensor(tsrc);aclDestroyTensor(tidx); }
    // GatherNd: data[4,3], indices[2,1] K=1 → rows {1,3} → out[2,3]
    { std::vector<float> data(12); for(int i=0;i<12;i++) data[i]=(float)i;
      std::vector<int64_t> idx={1,3};
      DevBuf dd(12*4),di(2*8),dz(6*4); dd.up(data.data()); di.up(idx.data());
      auto td=mk({4,3},ACL_FLOAT,dd.p),ti=mk({2,1},ACL_INT64,di.p),tz=mk({2,3},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherNdGetWorkspaceSize(td,ti,tz,w,e);}, aclnnGatherNd);
      std::vector<float> got(6); dz.down(got.data()); double bad=0;
      for(int r=0;r<2;r++)for(int c=0;c<3;c++) bad=std::max(bad,(double)std::fabs(got[r*3+c]-data[idx[r]*3+c]));
      report("GatherNd", bad, 1e-5); aclDestroyTensor(td);aclDestroyTensor(ti);aclDestroyTensor(tz); }
    // ScatterNdUpdate: data[4,3], indices[2,1] {0,2}, updates[2,3] → rows 0,2 overwritten
    { std::vector<float> data(12); for(int i=0;i<12;i++) data[i]=(float)i;
      std::vector<int64_t> idx={0,2}; std::vector<float> upd={100,101,102, 200,201,202};
      DevBuf dd(12*4),di(2*8),du(6*4),dz(12*4); dd.up(data.data()); di.up(idx.data()); du.up(upd.data());
      auto td=mk({4,3},ACL_FLOAT,dd.p),ti=mk({2,1},ACL_INT64,di.p),tu=mk({2,3},ACL_FLOAT,du.p),tz=mk({4,3},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnScatterNdUpdateGetWorkspaceSize(td,ti,tu,tz,w,e);}, aclnnScatterNdUpdate);
      std::vector<float> got(12); dz.down(got.data());
      std::vector<float> ref=data; ref[0]=100;ref[1]=101;ref[2]=102; ref[6]=200;ref[7]=201;ref[8]=202;
      double bad=0; for(int i=0;i<12;i++) bad=std::max(bad,(double)std::fabs(got[i]-ref[i]));
      report("ScatterNdUpdate", bad, 1e-5); aclDestroyTensor(td);aclDestroyTensor(ti);aclDestroyTensor(tu);aclDestroyTensor(tz); }
}
static void t_index3() {
    { // TfScatterAdd: ref[idx[k]] += updates[k] (rows)
      const int N=4,D=3,K=3; std::vector<float> ref(N*D,0.f),upd=randv(K*D,-2,2); std::vector<int64_t> idx={1,1,3};
      DevBuf dr(N*D*4),di(K*8),du(K*D*4),dout(N*D*4); dr.up(ref.data()); di.up(idx.data()); du.up(upd.data());
      auto tr=mk({N,D},ACL_FLOAT,dr.p),ti=mk({K},ACL_INT64,di.p),tu=mk({K,D},ACL_FLOAT,du.p),to=mk({N,D},ACL_FLOAT,dout.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnTfScatterAddGetWorkspaceSize(tr,ti,tu,to,w,e);}, aclnnTfScatterAdd);
      std::vector<float> got(N*D); dout.down(got.data()); std::vector<double> rf(N*D,0.0);
      for(int k=0;k<K;k++)for(int d=0;d<D;d++) rf[idx[k]*D+d]+=upd[k*D+d];
      double bad=0; for(int i=0;i<N*D;i++) bad=std::max(bad,std::fabs(got[i]-rf[i]));
      report("TfScatterAdd", bad, 1e-5); aclDestroyTensor(tr);aclDestroyTensor(ti);aclDestroyTensor(tu);aclDestroyTensor(to); }
    { // SearchSorteds: count of sorted <= value
      std::vector<float> seq={1,3,5,7,9}; float v=6.0f; int64_t out;
      DevBuf ds(seq.size()*4),dout(8); ds.up(seq.data());
      auto tseq=mk({(int64_t)seq.size()},ACL_FLOAT,ds.p),to=mk({1},ACL_INT64,dout.p); auto sc=aclCreateScalar(&v,ACL_FLOAT);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnSearchSortedsGetWorkspaceSize(tseq,sc,true,to,w,e);}, aclnnSearchSorteds);
      dout.down(&out); report("SearchSorteds", std::fabs((double)(out-3)), 0.0); aclDestroyTensor(tseq);aclDestroyTensor(to); }
    { // EmbeddingRenorm: rows exceeding max L2 norm get scaled to maxnorm
      const int V=4,D=3; std::vector<float> emb=randv(V*D,-3,3); std::vector<int64_t> idx={0,2}; double maxn=1.0;
      DevBuf de(V*D*4),di(2*8); de.up(emb.data()); di.up(idx.data());
      auto te=mk({V,D},ACL_FLOAT,de.p),ti=mk({2},ACL_INT64,di.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnEmbeddingRenormGetWorkspaceSize(te,ti,maxn,2.0,w,e);}, aclnnEmbeddingRenorm);
      std::vector<float> got(V*D); de.down(got.data()); double bad=0;
      for(int r=0;r<V;r++){ bool renorm=(r==0||r==2); double n0=0; for(int d=0;d<D;d++)n0+=emb[r*D+d]*emb[r*D+d]; n0=std::sqrt(n0);
        for(int d=0;d<D;d++){ double ref = (renorm&&n0>maxn)? emb[r*D+d]*(maxn/(n0+1e-7)) : emb[r*D+d]; bad=std::max(bad,std::fabs(got[r*D+d]-ref)); } }
      report("EmbeddingRenorm", bad, 1e-4); aclDestroyTensor(te);aclDestroyTensor(ti); }
    { // ApplyTopKTopP: keep top-2 logits, rest → -inf
      const int R=1,V=5; std::vector<float> lg={1.0f,3.0f,2.0f,5.0f,4.0f}; DevBuf dl(V*4),dout(V*4); dl.up(lg.data());
      auto tl=mk({R,V},ACL_FLOAT,dl.p),to=mk({R,V},ACL_FLOAT,dout.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnApplyTopKTopPGetWorkspaceSize(tl,2,1.0,to,w,e);}, aclnnApplyTopKTopP);
      std::vector<float> o(V); dout.down(o.data()); double bad=0; // top-2 = {5,4} kept; others -inf
      bool keep[5]={false,false,false,true,true};
      for(int v=0;v<V;v++){ if(keep[v]){ if(std::fabs(o[v]-lg[v])>1e-5) bad=1; } else { if(o[v]>-1e29f) bad=1; } }
      report("ApplyTopKTopP", bad, 0.0); aclDestroyTensor(tl);aclDestroyTensor(to); }
    { // GatherPaKvCache: out[i] = cache[slot[i]]
      const int N=5,D=2,K=3; auto kc=randv(N*D,-2,2),vc=randv(N*D,-2,2); std::vector<int64_t> slot={4,0,2};
      DevBuf dkc(N*D*4),dvc(N*D*4),dsl(K*8),dko(K*D*4),dvo(K*D*4); dkc.up(kc.data()); dvc.up(vc.data()); dsl.up(slot.data());
      auto tkc=mk({N,D},ACL_FLOAT,dkc.p),tvc=mk({N,D},ACL_FLOAT,dvc.p),tsl=mk({K},ACL_INT64,dsl.p),tko=mk({K,D},ACL_FLOAT,dko.p),tvo=mk({K,D},ACL_FLOAT,dvo.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGatherPaKvCacheGetWorkspaceSize(tkc,tvc,tsl,tko,tvo,w,e);}, aclnnGatherPaKvCache);
      std::vector<float> ko(K*D),vo(K*D); dko.down(ko.data()); dvo.down(vo.data()); double bad=0;
      for(int k=0;k<K;k++)for(int d=0;d<D;d++){ bad=std::max(bad,(double)std::fabs(ko[k*D+d]-kc[slot[k]*D+d])); bad=std::max(bad,(double)std::fabs(vo[k*D+d]-vc[slot[k]*D+d])); }
      report("GatherPaKvCache", bad, 1e-6); aclDestroyTensor(tkc);aclDestroyTensor(tvc);aclDestroyTensor(tsl);aclDestroyTensor(tko);aclDestroyTensor(tvo); }
    { // LightningIndexer: score[q,k] = Σ_h wq[q,h]*relu(q_h·k_h)
      const int Q=2,K=3,H=2,D=2; auto q=randv(Q*H*D,-1,1),key=randv(K*H*D,-1,1),wq=randv(Q*H,0,1);
      DevBuf dq(Q*H*D*4),dk(K*H*D*4),dw(Q*H*4),ds(Q*K*4); dq.up(q.data()); dk.up(key.data()); dw.up(wq.data());
      auto tq=mk({Q,H,D},ACL_FLOAT,dq.p),tk=mk({K,H,D},ACL_FLOAT,dk.p),tw=mk({Q,H},ACL_FLOAT,dw.p),ts=mk({Q,K},ACL_FLOAT,ds.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLightningIndexerGetWorkspaceSize(tq,tk,tw,ts,w,e);}, aclnnLightningIndexer);
      std::vector<float> sc(Q*K); ds.down(sc.data()); double me=0,mr=0;
      for(int qi=0;qi<Q;qi++)for(int ki=0;ki<K;ki++){ double acc=0; for(int h=0;h<H;h++){ double dot=0; for(int d=0;d<D;d++)dot+=q[(qi*H+h)*D+d]*key[(ki*H+h)*D+d]; acc+=wq[qi*H+h]*std::max(dot,0.0); }
        me=std::max(me,std::fabs(sc[qi*K+ki]-acc)); mr=std::max(mr,std::fabs(acc)); }
      report("LightningIndexer", me/(mr+1e-9), 1e-5); aclDestroyTensor(tq);aclDestroyTensor(tk);aclDestroyTensor(tw);aclDestroyTensor(ts); }
}
int main() {
    init(); srand(11);
    t_tri(); t_trace_diag(); t_bincount_search(); t_index_scatter(); t_take(); t_shape_ext(); t_scatter2(); t_index3();
    return finish();
}
