// Sequence / RNN (P16) cross-check: single-layer LSTM/GRU forward vs CPU double reference.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <vector>
#include <cmath>
#include <algorithm>
using namespace hn;
static double sig(double x){return 1.0/(1.0+std::exp(-x));}

static void t_lstm() {
    const int T=3,B=2,I=4,H=5;
    auto x=randv(T*B*I,-1,1),wih=randv(4*H*I,-0.5,0.5),whh=randv(4*H*H,-0.5,0.5),bih=randv(4*H,-0.2,0.2),bhh=randv(4*H,-0.2,0.2),h0=randv(B*H,-0.3,0.3),c0=randv(B*H,-0.3,0.3);
    std::vector<float> hy(T*B*H),hhN(B*H),hcN(B*H);
    DevBuf dx(T*B*I*4),dwih(4*H*I*4),dwhh(4*H*H*4),dbih(4*H*4),dbhh(4*H*4),dh0(B*H*4),dc0(B*H*4),dy(T*B*H*4),dhN(B*H*4),dcN(B*H*4);
    dx.up(x.data());dwih.up(wih.data());dwhh.up(whh.data());dbih.up(bih.data());dbhh.up(bhh.data());dh0.up(h0.data());dc0.up(c0.data());
    aclTensor *tx=mk({T,B,I},ACL_FLOAT,dx.p),*twih=mk({4*H,I},ACL_FLOAT,dwih.p),*twhh=mk({4*H,H},ACL_FLOAT,dwhh.p),
        *tbih=mk({4*H},ACL_FLOAT,dbih.p),*tbhh=mk({4*H},ACL_FLOAT,dbhh.p),*th0=mk({B,H},ACL_FLOAT,dh0.p),*tc0=mk({B,H},ACL_FLOAT,dc0.p),
        *ty=mk({T,B,H},ACL_FLOAT,dy.p),*thN=mk({B,H},ACL_FLOAT,dhN.p),*tcN=mk({B,H},ACL_FLOAT,dcN.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnLstmGetWorkspaceSize(tx,twih,twhh,tbih,tbhh,th0,tc0,ty,thN,tcN,w,e);}, aclnnLstm);
    dy.down(hy.data()); dhN.down(hhN.data()); dcN.down(hcN.data());
    // CPU
    std::vector<double> h(B*H),c(B*H); for(int i=0;i<B*H;i++){h[i]=h0[i];c[i]=c0[i];}
    std::vector<double> yref(T*B*H);
    for(int t=0;t<T;t++){ std::vector<double> hn(B*H),cn(B*H);
        for(int b=0;b<B;b++)for(int hh=0;hh<H;hh++){ double g[4];
            for(int gate=0;gate<4;gate++){ int row=gate*H+hh; double acc=bih[row]+bhh[row];
                for(int i=0;i<I;i++) acc+=(double)wih[row*I+i]*x[(t*B+b)*I+i];
                for(int k=0;k<H;k++) acc+=(double)whh[row*H+k]*h[b*H+k]; g[gate]=acc; }
            double gi=sig(g[0]),gf=sig(g[1]),gg=std::tanh(g[2]),go=sig(g[3]);
            double cc=gf*c[b*H+hh]+gi*gg; cn[b*H+hh]=cc; double hc=go*std::tanh(cc); hn[b*H+hh]=hc; yref[(t*B+b)*H+hh]=hc; }
        h=hn; c=cn; }
    double me=0,mr=0; for(int i=0;i<T*B*H;i++){me=std::max(me,std::fabs(hy[i]-yref[i]));mr=std::max(mr,std::fabs(yref[i]));}
    for(int i=0;i<B*H;i++){me=std::max(me,std::fabs(hhN[i]-h[i]));me=std::max(me,std::fabs(hcN[i]-c[i]));}
    report("LSTM forward", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(twih);aclDestroyTensor(twhh);aclDestroyTensor(tbih);aclDestroyTensor(tbhh);aclDestroyTensor(th0);aclDestroyTensor(tc0);aclDestroyTensor(ty);aclDestroyTensor(thN);aclDestroyTensor(tcN);
}
static void t_gru() {
    const int T=3,B=2,I=4,H=5;
    auto x=randv(T*B*I,-1,1),wih=randv(3*H*I,-0.5,0.5),whh=randv(3*H*H,-0.5,0.5),bih=randv(3*H,-0.2,0.2),bhh=randv(3*H,-0.2,0.2),h0=randv(B*H,-0.3,0.3);
    std::vector<float> hy(T*B*H),hhN(B*H);
    DevBuf dx(T*B*I*4),dwih(3*H*I*4),dwhh(3*H*H*4),dbih(3*H*4),dbhh(3*H*4),dh0(B*H*4),dy(T*B*H*4),dhN(B*H*4);
    dx.up(x.data());dwih.up(wih.data());dwhh.up(whh.data());dbih.up(bih.data());dbhh.up(bhh.data());dh0.up(h0.data());
    aclTensor *tx=mk({T,B,I},ACL_FLOAT,dx.p),*twih=mk({3*H,I},ACL_FLOAT,dwih.p),*twhh=mk({3*H,H},ACL_FLOAT,dwhh.p),
        *tbih=mk({3*H},ACL_FLOAT,dbih.p),*tbhh=mk({3*H},ACL_FLOAT,dbhh.p),*th0=mk({B,H},ACL_FLOAT,dh0.p),*ty=mk({T,B,H},ACL_FLOAT,dy.p),*thN=mk({B,H},ACL_FLOAT,dhN.p);
    exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnGruGetWorkspaceSize(tx,twih,twhh,tbih,tbhh,th0,ty,thN,w,e);}, aclnnGru);
    dy.down(hy.data()); dhN.down(hhN.data());
    std::vector<double> h(B*H); for(int i=0;i<B*H;i++) h[i]=h0[i]; std::vector<double> yref(T*B*H);
    for(int t=0;t<T;t++){ std::vector<double> hn(B*H);
        for(int b=0;b<B;b++)for(int hh=0;hh<H;hh++){ double xr=bih[0*H+hh],xz=bih[1*H+hh],xn=bih[2*H+hh],hr=bhh[0*H+hh],hz=bhh[1*H+hh],hn_=bhh[2*H+hh];
            for(int i=0;i<I;i++){double xi=x[(t*B+b)*I+i]; xr+=(double)wih[(0*H+hh)*I+i]*xi; xz+=(double)wih[(1*H+hh)*I+i]*xi; xn+=(double)wih[(2*H+hh)*I+i]*xi;}
            for(int k=0;k<H;k++){double hk=h[b*H+k]; hr+=(double)whh[(0*H+hh)*H+k]*hk; hz+=(double)whh[(1*H+hh)*H+k]*hk; hn_+=(double)whh[(2*H+hh)*H+k]*hk;}
            double r=sig(xr+hr),z=sig(xz+hz),nn=std::tanh(xn+r*hn_); double hc=(1-z)*nn+z*h[b*H+hh]; hn[b*H+hh]=hc; yref[(t*B+b)*H+hh]=hc; }
        h=hn; }
    double me=0,mr=0; for(int i=0;i<T*B*H;i++){me=std::max(me,std::fabs(hy[i]-yref[i]));mr=std::max(mr,std::fabs(yref[i]));}
    for(int i=0;i<B*H;i++) me=std::max(me,std::fabs(hhN[i]-h[i]));
    report("GRU forward", me/(mr+1e-9), 1e-5);
    aclDestroyTensor(tx);aclDestroyTensor(twih);aclDestroyTensor(twhh);aclDestroyTensor(tbih);aclDestroyTensor(tbhh);aclDestroyTensor(th0);aclDestroyTensor(ty);aclDestroyTensor(thN);
}
int main(){ init(); srand(47); t_lstm(); t_gru(); return finish(); }
