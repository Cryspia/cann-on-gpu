// Pooling / interpolate / vision (P6) cross-check: upsample nearest/bilinear, adaptive max2d / avg3d,
// max/avg pool1d, grid_sample2d, affine_grid, NMS. CPU double reference + tolerance.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <algorithm>
#include <cmath>
using namespace hn;

static void t_upsample() {
    const int C=2,H=3,W=3,oH=6,oW=6; auto fa=randv(C*H*W,-2,2);
    { std::vector<float> hz(C*oH*oW); DevBuf da(C*H*W*4),dz(C*oH*oW*4); da.up(fa.data());
      aclTensor *ta=mk({1,C,H,W},ACL_FLOAT,da.p),*tz=mk({1,C,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleNearest2dGetWorkspaceSize(ta,tz,w,e);}, aclnnUpsampleNearest2d);
      dz.down(hz.data()); double bad=0;
      for(int c=0;c<C;c++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ int ih=std::min(oh*H/oH,H-1),iw=std::min(ow*W/oW,W-1);
          bad=std::max(bad,(double)std::fabs(hz[(c*oH+oh)*oW+ow]-fa[(c*H+ih)*W+iw])); }
      report("UpsampleNearest2d", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    for (int align=0; align<=1; align++) {
      std::vector<float> hz(C*oH*oW); DevBuf da(C*H*W*4),dz(C*oH*oW*4); da.up(fa.data());
      aclTensor *ta=mk({1,C,H,W},ACL_FLOAT,da.p),*tz=mk({1,C,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnUpsampleBilinear2dGetWorkspaceSize(ta,(bool)align,tz,w,e);}, aclnnUpsampleBilinear2d);
      dz.down(hz.data()); double me=0,mr=0;
      for(int c=0;c<C;c++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ float fh,fw;
          if(align){fh=oH>1?(float)oh*(H-1)/(oH-1):0;fw=oW>1?(float)ow*(W-1)/(oW-1):0;} else {fh=(oh+0.5f)*H/oH-0.5f;fw=(ow+0.5f)*W/oW-0.5f;}
          fh=fh<0?0:fh;fw=fw<0?0:fw; int h0=(int)fh,w0=(int)fw,h1=std::min(h0+1,H-1),w1=std::min(w0+1,W-1); float dh=fh-h0,dw=fw-w0;
          const float*p=&fa[c*H*W]; double ref=p[h0*W+w0]*(1-dh)*(1-dw)+p[h0*W+w1]*(1-dh)*dw+p[h1*W+w0]*dh*(1-dw)+p[h1*W+w1]*dh*dw;
          me=std::max(me,std::fabs(hz[(c*oH+oh)*oW+ow]-ref)); mr=std::max(mr,std::fabs(ref)); }
      report(std::string("UpsampleBilinear2d align=")+std::to_string(align), me/(mr+1e-9), 1e-5);
      aclDestroyTensor(ta);aclDestroyTensor(tz); }
}
static void t_adaptive() {
    const int C=2,H=8,W=6,oH=3,oW=2; auto fa=randv(C*H*W,-2,2);
    { std::vector<float> hz(C*oH*oW); DevBuf da(C*H*W*4),dz(C*oH*oW*4); da.up(fa.data());
      aclTensor *ta=mk({1,C,H,W},ACL_FLOAT,da.p),*tz=mk({1,C,oH,oW},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaptiveMaxPool2dGetWorkspaceSize(ta,tz,nullptr,w,e);}, aclnnAdaptiveMaxPool2d);
      dz.down(hz.data()); double bad=0;
      for(int c=0;c<C;c++)for(int oh=0;oh<oH;oh++)for(int ow=0;ow<oW;ow++){ int hs=oh*H/oH,he=(oh+1)*H/oH+((oh+1)*H%oH?1:0),ws=ow*W/oW,we=(ow+1)*W/oW+((ow+1)*W%oW?1:0);
          double best=-1e30; for(int h=hs;h<he;h++)for(int w=ws;w<we;w++) best=std::max(best,(double)fa[(c*H+h)*W+w]);
          bad=std::max(bad,std::fabs(hz[(c*oH+oh)*oW+ow]-best)); }
      report("AdaptiveMaxPool2d", bad, 0); aclDestroyTensor(ta);aclDestroyTensor(tz); }
    // AdaptiveAvgPool3d [1,1,4,4,4]->[1,1,2,2,2]
    { const int D=4,H3=4,W3=4,oD=2,oH3=2,oW3=2; auto v=randv(D*H3*W3,-2,2); std::vector<float> hz(oD*oH3*oW3);
      DevBuf da(D*H3*W3*4),dz(oD*oH3*oW3*4); da.up(v.data());
      aclTensor *ta=mk({1,1,D,H3,W3},ACL_FLOAT,da.p),*tz=mk({1,1,oD,oH3,oW3},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAdaptiveAvgPool3dGetWorkspaceSize(ta,tz,w,e);}, aclnnAdaptiveAvgPool3d);
      dz.down(hz.data()); double me=0,mr=0;
      for(int od=0;od<oD;od++)for(int oh=0;oh<oH3;oh++)for(int ow=0;ow<oW3;ow++){ int ds=od*D/oD,de=(od+1)*D/oD+((od+1)*D%oD?1:0),hs=oh*H3/oH3,he=(oh+1)*H3/oH3+((oh+1)*H3%oH3?1:0),ws=ow*W3/oW3,we=(ow+1)*W3/oW3+((ow+1)*W3%oW3?1:0);
          double s=0; int cnt=0; for(int d=ds;d<de;d++)for(int h=hs;h<he;h++)for(int w=ws;w<we;w++){s+=v[(d*H3+h)*W3+w];cnt++;} double ref=s/cnt;
          me=std::max(me,std::fabs(hz[(od*oH3+oh)*oW3+ow]-ref)); mr=std::max(mr,std::fabs(ref)); }
      report("AdaptiveAvgPool3d", me/(mr+1e-9), 1e-6); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}
static void t_pool1d() {
    const int C=2,L=10,k=3,st=2,pad=0; int oL=(L+2*pad-k)/st+1; auto fa=randv(C*L,-2,2);
    for (int mx=0; mx<=1; mx++) {
      std::vector<float> hz(C*oL); DevBuf da(C*L*4),dz(C*oL*4); da.up(fa.data());
      aclTensor *ta=mk({1,C,L},ACL_FLOAT,da.p),*tz=mk({1,C,oL},ACL_FLOAT,dz.p);
      if(mx) exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxPool1dGetWorkspaceSize(ta,k,st,pad,tz,w,e);}, aclnnMaxPool1d);
      else exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAvgPool1dGetWorkspaceSize(ta,k,st,pad,tz,w,e);}, aclnnAvgPool1d);
      dz.down(hz.data()); double me=0,mr=0;
      for(int c=0;c<C;c++)for(int ol=0;ol<oL;ol++){ double acc=mx?-1e30:0; int cnt=0; for(int j=0;j<k;j++){int l=ol*st-pad+j; if(l<0||l>=L)continue; double vv=fa[c*L+l]; if(mx)acc=std::max(acc,vv); else acc+=vv; cnt++;}
          double ref=mx?acc:acc/(cnt?cnt:1); me=std::max(me,std::fabs(hz[c*oL+ol]-ref)); mr=std::max(mr,std::fabs(ref)); }
      report(std::string(mx?"MaxPool1d":"AvgPool1d"), me/(mr+1e-9), 1e-6); aclDestroyTensor(ta);aclDestroyTensor(tz); }
}
static void t_grid_affine() {
    // AffineGrid: identity theta -> grid equals base coords; then GridSample with identity grid reproduces input
    const int N=1,C=1,H=4,W=4; float theta[6]={1,0,0, 0,1,0};
    std::vector<float> hgrid(H*W*2);
    DevBuf dt(6*4), dg(H*W*2*4); dt.up(theta);
    aclTensor *tt=mk({N,2,3},ACL_FLOAT,dt.p),*tg=mk({N,H,W,2},ACL_FLOAT,dg.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAffineGridGetWorkspaceSize(tt,H,W,true,tg,w,e);}, aclnnAffineGrid);
    dg.down(hgrid.data());
    double bad=0; for(int h=0;h<H;h++)for(int w=0;w<W;w++){ float xn=W>1?(float)w/(W-1)*2-1:0, yn=H>1?(float)h/(H-1)*2-1:0;
        bad=std::max(bad,(double)std::fabs(hgrid[(h*W+w)*2+0]-xn)); bad=std::max(bad,(double)std::fabs(hgrid[(h*W+w)*2+1]-yn)); }
    report("AffineGrid identity", bad, 1e-6);
    // GridSample with this identity grid should reproduce input (align_corners=true)
    auto fa=randv(C*H*W,-2,2); std::vector<float> hz(C*H*W);
    DevBuf da(C*H*W*4),dz(C*H*W*4); da.up(fa.data());
    aclTensor *ta=mk({N,C,H,W},ACL_FLOAT,da.p),*tg2=mk({N,H,W,2},ACL_FLOAT,dg.p),*tz=mk({N,C,H,W},ACL_FLOAT,dz.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGridSample2dGetWorkspaceSize(ta,tg2,true,tz,w,e);}, aclnnGridSample2d);
    dz.down(hz.data()); double bad2=0; for(int i=0;i<C*H*W;i++) bad2=std::max(bad2,(double)std::fabs(hz[i]-fa[i]));
    report("GridSample2d identity", bad2, 1e-5);
    aclDestroyTensor(tt);aclDestroyTensor(tg);aclDestroyTensor(ta);aclDestroyTensor(tg2);aclDestroyTensor(tz);
}
static void t_nms() {
    // 4 boxes: 0&1 overlap heavily, 2 separate, 3 overlaps 0 slightly. scores pick 0 then 2 then 3.
    const int M=4; float boxes[16]={0,0,10,10, 1,1,11,11, 50,50,60,60, 8,8,18,18};
    float scores[4]={0.9f,0.8f,0.7f,0.6f};
    DevBuf db(M*4*4), dsc(M*4), dk(M*8), dc(8); db.up(boxes); dsc.up(scores);
    aclTensor *tb=mk({M,4},ACL_FLOAT,db.p),*ts=mk({M},ACL_FLOAT,dsc.p),*tk=mk({M},ACL_INT64,dk.p),*tc=mk({1},ACL_INT64,dc.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnNmsGetWorkspaceSize(tb,ts,0.5,tk,tc,w,e);}, aclnnNms);
    std::vector<int64_t> keep(M); int64_t cnt; dk.down(keep.data()); dc.down(&cnt);
    // expected: keep 0 (suppress 1 iou>0.5), keep 2, box3 iou with 0 = inter(8,8,10,10)=4 / (100+100-4)=0.02<0.5 keep
    bool ok = (cnt==3) && keep[0]==0 && keep[1]==2 && keep[2]==3;
    report("NMS iou=0.5", ok?0.0:1.0, 0);
    aclDestroyTensor(tb);aclDestroyTensor(ts);aclDestroyTensor(tk);aclDestroyTensor(tc);
}

static void t_pool_new() {
    { // MaxPool3dWithArgmax: [1,1,2,2,2] kernel 2 → max + argmax over the 8 elements
      std::vector<float> x={1,2,3,9,4,5,6,7}; DevBuf dx(8*4),do_(4),di(8); dx.up(x.data());
      aclTensor *tx=mk({1,1,2,2,2},ACL_FLOAT,dx.p),*to=mk({1,1,1,1,1},ACL_FLOAT,do_.p),*ti=mk({1,1,1,1,1},ACL_INT64,di.p);
      int64_t kv[3]={2,2,2},sv[3]={1,1,1},pv[3]={0,0,0};
      aclIntArray *k=aclCreateIntArray(kv,3),*s=aclCreateIntArray(sv,3),*p=aclCreateIntArray(pv,3);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaxPool3dWithArgmaxGetWorkspaceSize(tx,k,s,p,nullptr,false,to,ti,w,e);}, aclnnMaxPool3dWithArgmax);
      float ov; do_.down(&ov); int64_t oi; di.down(&oi);
      report("MaxPool3dWithArgmax", (std::fabs(ov-9.0)+std::fabs((double)(oi-3)))/9.0, 1e-6);
      aclDestroyTensor(tx);aclDestroyTensor(to);aclDestroyTensor(ti); }
    { // AvgPool3dBackward: gradOut [1,1,1,1,1]=8 over kernel 2 → each of 8 inputs gets 1.0
      float go=8.f; DevBuf dgo(4),dgi(8*4); dgo.up(&go);
      aclTensor *tgo=mk({1,1,1,1,1},ACL_FLOAT,dgo.p),*tgi=mk({1,1,2,2,2},ACL_FLOAT,dgi.p);
      int64_t kv[3]={2,2,2},sv[3]={1,1,1},pv[3]={0,0,0};
      aclIntArray *k=aclCreateIntArray(kv,3),*s=aclCreateIntArray(sv,3),*p=aclCreateIntArray(pv,3);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnAvgPool3dBackwardGetWorkspaceSize(tgo,k,s,p,tgi,w,e);}, aclnnAvgPool3dBackward);
      std::vector<float> gi(8); dgi.down(gi.data()); double bad=0; for(int i=0;i<8;i++) bad=std::max(bad,(double)std::fabs(gi[i]-1.0));
      report("AvgPool3dBackward", bad, 1e-6); aclDestroyTensor(tgo);aclDestroyTensor(tgi); }
    { // RoiAlignRotated on a constant image → output equals that constant
      const int C=1,H=8,W=8,K=1,ph=2,pw=2; std::vector<float> x(C*H*W,5.f);
      std::vector<float> rois={0, 4,4, 4,4, 0.5f}; // batch,cx,cy,w,h,theta
      DevBuf dx(C*H*W*4),dr(6*4),dz(K*C*ph*pw*4); dx.up(x.data()); dr.up(rois.data());
      aclTensor *tx=mk({1,C,H,W},ACL_FLOAT,dx.p),*tr=mk({K,6},ACL_FLOAT,dr.p),*tz=mk({K,C,ph,pw},ACL_FLOAT,dz.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRoiAlignRotatedGetWorkspaceSize(tx,tr,1.0,2,tz,w,e);}, aclnnRoiAlignRotated);
      std::vector<float> z(K*C*ph*pw); dz.down(z.data()); double bad=0; for(float v:z) bad=std::max(bad,(double)std::fabs(v-5.0));
      report("RoiAlignRotated(const)", bad, 1e-4); aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tz); }
    { // RoiPoolingWithArgMax: whole-image roi, 1x1 pool → global max + argmax
      const int C=1,H=4,W=4,K=1; auto x=randv(C*H*W,-3,3);
      std::vector<float> rois={0,0,0,(float)(W-1),(float)(H-1)};
      DevBuf dx(C*H*W*4),dr(5*4),dz(4),da(8); dx.up(x.data()); dr.up(rois.data());
      aclTensor *tx=mk({1,C,H,W},ACL_FLOAT,dx.p),*tr=mk({K,5},ACL_FLOAT,dr.p),*tz=mk({K,C,1,1},ACL_FLOAT,dz.p),*ta=mk({K,C,1,1},ACL_INT64,da.p);
      exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnRoiPoolingWithArgMaxGetWorkspaceSize(tx,tr,1.0,tz,ta,w,e);}, aclnnRoiPoolingWithArgMax);
      float ov; dz.down(&ov); int64_t oi; da.down(&oi);
      double mx=-1e30; int64_t mi=0; for(int i=0;i<H*W;i++) if(x[i]>mx){mx=x[i];mi=i;}
      report("RoiPoolingWithArgMax", (std::fabs(ov-mx)+std::fabs((double)(oi-mi)))/(std::fabs(mx)+1e-9), 1e-6);
      aclDestroyTensor(tx);aclDestroyTensor(tr);aclDestroyTensor(tz);aclDestroyTensor(ta); }
}
int main() {
    init(); srand(13);
    t_upsample(); t_adaptive(); t_pool1d(); t_grid_affine(); t_nms(); t_pool_new();
    return finish();
}
