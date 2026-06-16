// Cross-check for the in-place elementwise / activation family (aclnnInplace*): each modifies selfRef.
// References computed on the CPU.
#include "harness.h"
#include "aclnnop/aclnn_ops.h"
#include <cmath>
using namespace hn;

namespace {
const std::vector<float> X = {-1.5f, -0.5f, 0.25f, 0.5f, 1.0f, 2.0f, 3.0f, 4.0f};

// run an in-place op on a fresh copy of X and compare against ref
double check(std::function<void(aclTensor *)> run, std::function<float(float)> ref) {
    DevBuf b(X.size() * sizeof(float)); b.up(X.data());
    auto t = mk({(int64_t)X.size()}, ACL_FLOAT, b.p);
    run(t);
    std::vector<float> got(X.size()); b.down(got.data());
    double me = 0, mr = 0;
    for (size_t i = 0; i < X.size(); i++) { double r = ref(X[i]); me = std::max(me, std::fabs(got[i] - r)); mr = std::max(mr, std::fabs(r)); }
    return me / (mr + 1e-9);
}
} // namespace

int main() {
    init();

    report("InplaceExp", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceExpGetWorkspaceSize(t,w,e);}, aclnnInplaceExp); },
                                [](float v){ return std::exp(v); }), 1e-6);
    report("InplaceNeg", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceNegGetWorkspaceSize(t,w,e);}, aclnnInplaceNeg); },
                                [](float v){ return -v; }), 1e-6);
    report("InplaceReciprocal", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceReciprocalGetWorkspaceSize(t,w,e);}, aclnnInplaceReciprocal); },
                                [](float v){ return 1.0f / v; }), 1e-6);
    report("InplaceRelu", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceReluGetWorkspaceSize(t,w,e);}, aclnnInplaceRelu); },
                                [](float v){ return v > 0 ? v : 0.f; }), 1e-6);
    report("InplaceSigmoid", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceSigmoidGetWorkspaceSize(t,w,e);}, aclnnInplaceSigmoid); },
                                [](float v){ return 1.0f / (1.0f + std::exp(-v)); }), 1e-6);

    // one-scalar: leaky relu (negative slope 0.1)
    report("InplaceLeakyRelu", check([](aclTensor *t){ float a=0.1f; auto s=aclCreateScalar(&a,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceLeakyReluGetWorkspaceSize(t,s,w,e);}, aclnnInplaceLeakyRelu); },
                                [](float v){ return v > 0 ? v : 0.1f * v; }), 1e-6);
    // one-scalar: clamp max at 1.0
    report("InplaceClampMax", check([](aclTensor *t){ float a=1.0f; auto s=aclCreateScalar(&a,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceClampMaxGetWorkspaceSize(t,s,w,e);}, aclnnInplaceClampMax); },
                                [](float v){ return v < 1.0f ? v : 1.0f; }), 1e-6);
    // two-scalar: hardtanh [-1, 2]
    report("InplaceHardtanh", check([](aclTensor *t){ float lo=-1.f,hi=2.f; auto a=aclCreateScalar(&lo,ACL_FLOAT),bb=aclCreateScalar(&hi,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceHardtanhGetWorkspaceSize(t,a,bb,w,e);}, aclnnInplaceHardtanh); },
                                [](float v){ return v < -1.f ? -1.f : (v > 2.f ? 2.f : v); }), 1e-6);

    // ternary: addcmul with t1=t2=X, value=0.5 → x + 0.5*x*x
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data());
        DevBuf b1(X.size()*sizeof(float)); b1.up(X.data());
        DevBuf b2(X.size()*sizeof(float)); b2.up(X.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), t1=mk({(int64_t)X.size()},ACL_FLOAT,b1.p), t2=mk({(int64_t)X.size()},ACL_FLOAT,b2.p);
        float v=0.5f; auto sv=aclCreateScalar(&v,ACL_FLOAT);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceAddcmulGetWorkspaceSize(t,t1,t2,sv,w,e);}, aclnnInplaceAddcmul);
        std::vector<float> got(X.size()); b.down(got.data());
        double me=0,mr=0; for(size_t i=0;i<X.size();i++){ double r=X[i]+0.5*X[i]*X[i]; me=std::max(me,std::fabs(got[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report("InplaceAddcmul", me/(mr+1e-9), 1e-6);
    }

    // tensor clamp-min against a constant tensor of 0.5 → max(x, 0.5)
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data());
        std::vector<float> half(X.size(), 0.5f); DevBuf bh(X.size()*sizeof(float)); bh.up(half.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), m=mk({(int64_t)X.size()},ACL_FLOAT,bh.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceClampMinTensorGetWorkspaceSize(t,m,w,e);}, aclnnInplaceClampMinTensor);
        std::vector<float> got(X.size()); b.down(got.data());
        double me=0,mr=0; for(size_t i=0;i<X.size();i++){ double r=std::max(X[i],0.5f); me=std::max(me,std::fabs(got[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report("InplaceClampMinTensor", me/(mr+1e-9), 1e-6);
    }

    // lerp scalar: self + 0.25*(end-self), end = 2*X
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data());
        std::vector<float> end(X.size()); for(size_t i=0;i<X.size();i++) end[i]=2*X[i];
        DevBuf be(X.size()*sizeof(float)); be.up(end.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), e2=mk({(int64_t)X.size()},ACL_FLOAT,be.p);
        float w=0.25f; auto wt=aclCreateScalar(&w,ACL_FLOAT);
        exec2([&](uint64_t*w2,aclOpExecutor**e){return aclnnInplaceLerpsGetWorkspaceSize(t,e2,wt,w2,e);}, aclnnInplaceLerps);
        std::vector<float> got(X.size()); b.down(got.data());
        double me=0,mr=0; for(size_t i=0;i<X.size();i++){ double r=X[i]+0.25*(end[i]-X[i]); me=std::max(me,std::fabs(got[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report("InplaceLerps", me/(mr+1e-9), 1e-6);
    }

    // unary math: tanh
    report("InplaceTanh", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceTanhGetWorkspaceSize(t,w,e);}, aclnnInplaceTanh); },
                                [](float v){ return std::tanh(v); }), 1e-6);
    // scalar with alpha: adds (other=3, alpha=2) → self + 6
    report("InplaceAdds", check([](aclTensor *t){ float o=3.f,a=2.f; auto so=aclCreateScalar(&o,ACL_FLOAT),sa=aclCreateScalar(&a,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceAddsGetWorkspaceSize(t,so,sa,w,e);}, aclnnInplaceAdds); },
                                [](float v){ return v + 6.f; }), 1e-6);
    // scalar: muls by 3
    report("InplaceMuls", check([](aclTensor *t){ float o=3.f; auto so=aclCreateScalar(&o,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceMulsGetWorkspaceSize(t,so,w,e);}, aclnnInplaceMuls); },
                                [](float v){ return v * 3.f; }), 1e-6);
    // zero
    report("InplaceZero", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceZeroGetWorkspaceSize(t,w,e);}, aclnnInplaceZero); },
                                [](float){ return 0.f; }), 0.0);

    // binary with alpha: add (other = X, alpha = 2) → self + 2*self = 3*self
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data());
        DevBuf bo(X.size()*sizeof(float)); bo.up(X.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), o=mk({(int64_t)X.size()},ACL_FLOAT,bo.p);
        float a=2.f; auto sa=aclCreateScalar(&a,ACL_FLOAT);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceAddGetWorkspaceSize(t,o,sa,w,e);}, aclnnInplaceAdd);
        std::vector<float> got(X.size()); b.down(got.data());
        double me=0,mr=0; for(size_t i=0;i<X.size();i++){ double r=3.0*X[i]; me=std::max(me,std::fabs(got[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report("InplaceAdd", me/(mr+1e-9), 1e-6);
    }

    // fill scalar / one
    report("InplaceFillScalar", check([](aclTensor *t){ float v=7.f; auto s=aclCreateScalar(&v,ACL_FLOAT);
                                exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceFillScalarGetWorkspaceSize(t,s,w,e);}, aclnnInplaceFillScalar); },
                                [](float){ return 7.f; }), 0.0);
    report("InplaceOne", check([](aclTensor *t){ exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceOneGetWorkspaceSize(t,w,e);}, aclnnInplaceOne); },
                                [](float){ return 1.f; }), 0.0);

    // InplaceAddV3 (other=X, alpha=1) → 2*self
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data()); DevBuf bo(X.size()*sizeof(float)); bo.up(X.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), o=mk({(int64_t)X.size()},ACL_FLOAT,bo.p);
        float a=1.f; auto sa=aclCreateScalar(&a,ACL_FLOAT);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceAddV3GetWorkspaceSize(t,o,sa,w,e);}, aclnnInplaceAddV3);
        std::vector<float> got(X.size()); b.down(got.data()); double me=0,mr=0;
        for(size_t i=0;i<X.size();i++){ double r=2.0*X[i]; me=std::max(me,std::fabs(got[i]-r)); mr=std::max(mr,std::fabs(r)); }
        report("InplaceAddV3", me/(mr+1e-9), 1e-6);
    }
    // InplaceFloorDivide (other = const 2) → floor(self/2)
    {
        DevBuf b(X.size()*sizeof(float)); b.up(X.data()); std::vector<float> two(X.size(),2.f); DevBuf bo(X.size()*sizeof(float)); bo.up(two.data());
        auto t=mk({(int64_t)X.size()},ACL_FLOAT,b.p), o=mk({(int64_t)X.size()},ACL_FLOAT,bo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceFloorDivideGetWorkspaceSize(t,o,w,e);}, aclnnInplaceFloorDivide);
        std::vector<float> got(X.size()); b.down(got.data()); double bad=0;
        for(size_t i=0;i<X.size();i++) bad=std::max(bad,(double)std::fabs(got[i]-std::floor(X[i]/2.f)));
        report("InplaceFloorDivide", bad, 1e-5);
    }
    // InplaceFillDiagonal on [3,3] with 9
    {
        const int N=3; std::vector<float> z(N*N,0.f); DevBuf b(N*N*4); b.up(z.data());
        auto t=mk({N,N},ACL_FLOAT,b.p); float v=9.f; auto s=aclCreateScalar(&v,ACL_FLOAT);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceFillDiagonalGetWorkspaceSize(t,s,false,w,e);}, aclnnInplaceFillDiagonal);
        std::vector<float> got(N*N); b.down(got.data()); double bad=0;
        for(int r=0;r<N;r++)for(int c=0;c<N;c++) bad=std::max(bad,(double)std::fabs(got[r*N+c]-(r==c?9.f:0.f)));
        report("InplaceFillDiagonal", bad, 0.0);
    }
    // InplaceLogicalNot on a bool tensor
    {
        std::vector<uint8_t> bv={1,0,1,1,0,0,1,0}; DevBuf b(bv.size()); b.up(bv.data());
        auto t=mk({(int64_t)bv.size()},ACL_BOOL,b.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceLogicalNotGetWorkspaceSize(t,w,e);}, aclnnInplaceLogicalNot);
        std::vector<uint8_t> got(bv.size()); b.down(got.data()); double bad=0;
        for(size_t i=0;i<bv.size();i++) if(got[i]!=(bv[i]?0:1)) bad=1;
        report("InplaceLogicalNot", bad, 0.0);
    }

    // MaskedFillTensor: out = mask ? value[0] : self
    {
        std::vector<float> self={1,2,3,4,5,6,7,8}; std::vector<uint8_t> m={1,0,1,0,1,0,1,0}; float val=-9.f;
        DevBuf bs(self.size()*4); bs.up(self.data()); DevBuf bm(m.size()); bm.up(m.data());
        DevBuf bv(4); bv.up(&val); DevBuf bo(self.size()*4);
        auto ts=mk({(int64_t)self.size()},ACL_FLOAT,bs.p), tm=mk({(int64_t)m.size()},ACL_BOOL,bm.p),
             tv=mk({1},ACL_FLOAT,bv.p), to=mk({(int64_t)self.size()},ACL_FLOAT,bo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnMaskedFillTensorGetWorkspaceSize(ts,tm,tv,to,w,e);}, aclnnMaskedFillTensor);
        std::vector<float> got(self.size()); bo.down(got.data()); double bad=0;
        for(size_t i=0;i<self.size();i++) bad=std::max(bad,(double)std::fabs(got[i]-(m[i]?val:self[i])));
        report("MaskedFillTensor", bad, 0.0);
    }
    // InplaceCopy: selfRef <- src
    {
        std::vector<float> dst(X.size(),0.f), src=X; DevBuf bd(X.size()*4); bd.up(dst.data()); DevBuf bsrc(X.size()*4); bsrc.up(src.data());
        auto td=mk({(int64_t)X.size()},ACL_FLOAT,bd.p), tsrc=mk({(int64_t)X.size()},ACL_FLOAT,bsrc.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnInplaceCopyGetWorkspaceSize(td,tsrc,w,e);}, aclnnInplaceCopy);
        std::vector<float> got(X.size()); bd.down(got.data()); double bad=0;
        for(size_t i=0;i<X.size();i++) bad=std::max(bad,(double)std::fabs(got[i]-X[i]));
        report("InplaceCopy", bad, 0.0);
    }
    // BitwiseAndTensorOut (int32)
    {
        std::vector<int32_t> a={0xF0,0x0F,0xFF,0x33,12,7,255,128}, bb={0x3C,0x3C,0x0F,0x11,10,3,15,64};
        DevBuf ba(a.size()*4); ba.up(a.data()); DevBuf bbf(bb.size()*4); bbf.up(bb.data()); DevBuf bo(a.size()*4);
        auto ta=mk({(int64_t)a.size()},ACL_INT32,ba.p), tb=mk({(int64_t)bb.size()},ACL_INT32,bbf.p), to=mk({(int64_t)a.size()},ACL_INT32,bo.p);
        exec2([&](uint64_t*w,aclOpExecutor**e){return aclnnBitwiseAndTensorOutGetWorkspaceSize(ta,tb,to,w,e);}, aclnnBitwiseAndTensorOut);
        std::vector<int32_t> got(a.size()); bo.down(got.data()); double bad=0;
        for(size_t i=0;i<a.size();i++) if(got[i]!=(a[i]&bb[i])) bad=1;
        report("BitwiseAndTensorOut", bad, 0.0);
    }

    return finish();
}
