// cannsim golden comparison (44 ops):
// Compares shim results against the "true Ascend golden" computed by cann-api-explorer
// running on cannsim card-free simulation.
//   gen   <op> <prefix>            generate deterministic fp32 inputs for op into <prefix>.x[/.y/.g/.b]
//   check <op> <prefix> <out.bin>  read the same inputs + Ascend golden, run the corresponding
//                                  shim aclnn op, and compare using normalized error
//
// Supported ops (share the same fp32 raw inputs with the explorer unit GOLDEN hook, bit-exact):
//   elementwise binary: add/sub/mul/div/max/min/power/hypot/fmod                       N=16384
//   elementwise unary:  relu/exp/sqrt/abs/neg/silu/gelu/sigmoid/tanh/erf/log/sin/cos/
//                       rsqrt/reciprocal/acos/asin/atan/cosh/sinh/tan/ceil/floor/round/
//                       trunc/sign/erfc/frac/lgamma                                    N=16384
//   softmax/logsoftmax  x[8,64] last dim                      layernorm x[8,64]+gamma+beta
//   rmsnorm  x[8,64]+gamma   matmul A[64,64]@B[64,64] fp16->fp32   reducesum x[256]->scalar
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

#define CK(x) do { int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} } while(0)

enum Kind { K_BIN, K_UN, K_SOFTMAX, K_LOGSOFTMAX, K_LAYERNORM, K_RMSNORM, K_MATMUL, K_REDUCESUM,
            K_SORT, K_TOPK, K_GROUPNORM, K_DEEPNORM, K_REDUCE, K_SHIFT, K_BOOL, K_SCALAR, K_QUANT };
static const int64_t QUANT_N = 256;   // fp32 -> int8 per-tensor quant: ascendquant(scale1/off0), quantize(scale2/off1)
static const int64_t SHIFT_N = 64;   // int32 bit-shift: x0=value, x1=shift amount
// scalar-elementwise: the explorer kernels bake a scalar (adds/subs/muls/divs/maxs/mins use 2.0;
// leakyrelu negSlope=0.1; axpy = 2*x + 1) — feed the same baked scalar to the shim.
static bool is_scalar_op(const std::string&o){ return o=="adds"||o=="subs"||o=="muls"||o=="divs"||o=="maxs"||o=="mins"||o=="leakyrelu"||o=="axpy"; }
// full-axis reductions to a scalar: reducemax/reducemin (256), sum/mean/reduceprod (64)
static bool is_reduce(const std::string&o){ return o=="reducemax"||o=="reducemin"||o=="sum"||o=="mean"||o=="reduceprod"; }
static int64_t reduce_n(const std::string&o){ return (o=="reducemax"||o=="reducemin")?256:64; }
static bool is_bin(const std::string&o){ return o=="add"||o=="sub"||o=="mul"||o=="div"||o=="max"||o=="min"||o=="power"||o=="hypot"||o=="fmod"; }
static Kind kind_of(const std::string &op){
    if (is_bin(op)) return K_BIN;
    if (op=="softmax") return K_SOFTMAX;
    if (op=="logsoftmax") return K_LOGSOFTMAX;
    if (op=="layernorm") return K_LAYERNORM;
    if (op=="rmsnorm") return K_RMSNORM;
    if (op=="matmul") return K_MATMUL;
    if (op=="reducesum") return K_REDUCESUM;
    if (is_reduce(op)) return K_REDUCE;
    if (op=="shiftleft"||op=="shiftright") return K_SHIFT;
    if (op=="is_finite"||op=="is_inf"||op=="is_nan") return K_BOOL;
    if (is_scalar_op(op)) return K_SCALAR;
    if (op=="ascendquant"||op=="quantize") return K_QUANT;
    if (op=="sort") return K_SORT;
    if (op=="topk") return K_TOPK;
    if (op=="groupnorm") return K_GROUPNORM;
    if (op=="deepnorm") return K_DEEPNORM;
    return K_UN;
}
// Input domain: POS positive range (safe for sqrt/exp/log/div/rsqrt/reciprocal/lgamma/power/fmod/reducesum),
//               UNIT [-1,1] (domain for asin/acos), GEN default [-2.5,2.5]
enum Dom { D_POS, D_UNIT, D_GEN, D_GT1 };
static Dom dom_of(const std::string &op){
    if (op=="sqrt"||op=="exp"||op=="log"||op=="ln"||op=="log2"||op=="log10"||op=="digamma"||op=="div"||op=="rsqrt"||op=="reciprocal"||op=="lgamma"||op=="power"||op=="fmod"||op=="reducesum"||op=="log1p") return D_POS;
    if (op=="asin"||op=="acos"||op=="atanh"||op=="erfinv") return D_UNIT;
    if (op=="acosh") return D_GT1;   // acosh domain x>=1
    return D_GEN;  // asinh defined everywhere
}
static double tol_of(const std::string &op){
    if (op=="matmul") return 2e-2;
    if (op=="gelu"||op=="erf"||op=="erfc"||op=="tan"||op=="power"||op=="lgamma"||op=="frac"||op=="fmod") return 3e-3;
    if (op=="layernorm"||op=="rmsnorm") return 2e-4;
    if (op=="silu"||op=="sigmoid"||op=="tanh"||op=="sin"||op=="cos"||op=="log"||op=="logsoftmax"||
        op=="asin"||op=="acos"||op=="atan"||op=="sinh"||op=="cosh"||op=="hypot"||op=="reducesum"||op=="ln"||
        op=="acosh"||op=="asinh"||op=="atanh"||op=="log2"||op=="log10") return 2e-4;
    if (op=="digamma") return 3e-3;   // series/asymptotic approximation
    if (op=="sum"||op=="mean"||op=="reduceprod") return 2e-4;   // reducemax/reducemin fall through to exact 1e-5
    if (op=="adds"||op=="subs"||op=="muls"||op=="divs"||op=="maxs"||op=="mins"||op=="leakyrelu"||op=="axpy") return 2e-4;
    if (op=="groupnorm"||op=="deepnorm") return 2e-4;
    return 1e-5;   // sort/topk: sorted values are exact
}

static const int64_t NE = 16384;
static int64_t scalar_n(const std::string&o){ return (o=="leakyrelu"||o=="axpy")?64:NE; }   // these two units use N=64
static const int64_t SM_ROWS=8, SM_COLS=64;
static const int64_t LN_A=8, LN_R=64;
static const int64_t RN_BSH=512, RN_H=64;
static const int64_t MM_M=64, MM_N=64, MM_K=64;
static const int64_t RS_N=256;
static const int64_t ST_N=32;                       // sort
static const int64_t TK_N=32, TK_K=4;               // topk
static const int64_t GN_TOTAL=64, GN_C=4, GN_HW=16, GN_G=2;   // groupnorm [1,4,16] 2 groups
static const int64_t DN_H=8;                         // deepnorm

static float gval(int64_t i, Dom d, bool y){
    double base = y ? std::cos(i*0.017) : std::sin(i*0.013);
    if (d==D_POS)  return (float)(base*(y?1.0:1.5) + (y?1.5:2.0));   // [0.5,3.5]/[0.5,2.5]
    if (d==D_UNIT) return (float)(base*0.9);                          // [-0.9,0.9]
    if (d==D_GT1)  return (float)(std::fabs(base)*3.0 + 1.05);        // [1.05,4.05] (acosh x>=1)
    return (float)(base*(y?2.0:2.5));                                 // [-2.5,2.5]
}

static void wbin(const std::string &p, const std::vector<float> &v){
    FILE*f=fopen(p.c_str(),"wb"); if(!f){perror(p.c_str());exit(1);} fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<float> rbin(const std::string &p, int64_t n){
    std::vector<float> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);}
    if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static void wbin_i32(const std::string &p, const std::vector<int32_t> &v){
    FILE*f=fopen(p.c_str(),"wb"); if(!f){perror(p.c_str());exit(1);} fwrite(v.data(),4,v.size(),f); fclose(f); }
static std::vector<int32_t> rbin_i32(const std::string &p, int64_t n){
    std::vector<int32_t> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);}
    if((int64_t)fread(v.data(),4,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static std::vector<uint8_t> rbin_u8(const std::string &p, int64_t n){
    std::vector<uint8_t> v(n); FILE*f=fopen(p.c_str(),"rb"); if(!f){perror(p.c_str());exit(1);}
    if((int64_t)fread(v.data(),1,n,f)!=n){fprintf(stderr,"short read %s\n",p.c_str());exit(1);} fclose(f); return v; }
static uint16_t f2h(float f){
    uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t mant=x&0x7fffffu;
    if(exp<=0) return (uint16_t)sign;
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    uint16_t h=(uint16_t)(sign | ((uint32_t)exp<<10) | (mant>>13));
    if(mant&0x1000u) h++;
    return h;
}
static aclTensor* mkt(std::vector<int64_t> dims, aclDataType dt, void*d){
    return aclCreateTensor(dims.data(),(int64_t)dims.size(),dt,nullptr,0,ACL_FORMAT_ND,dims.data(),(int64_t)dims.size(),d);
}

// =========================== gen ===========================
static int do_gen(const std::string &op, const std::string &prefix){
    Kind k = kind_of(op); Dom d = dom_of(op);
    auto P=[&](const char*s){ return prefix+s; };
    if (k==K_BIN || k==K_UN){
        std::vector<float> x(NE),y(NE);
        for(int64_t i=0;i<NE;i++){ x[i]=gval(i,d,false); y[i]=gval(i,d,true); }
        wbin(P(".x"),x); if(k==K_BIN) wbin(P(".y"),y);
    } else if (k==K_SOFTMAX || k==K_LOGSOFTMAX){
        std::vector<float> x(SM_ROWS*SM_COLS);
        for(int64_t i=0;i<(int64_t)x.size();i++) x[i]=gval(i,D_GEN,false);
        wbin(P(".x"),x);
    } else if (k==K_LAYERNORM){
        std::vector<float> x(LN_A*LN_R), g(LN_R), b(LN_R);
        for(int64_t i=0;i<(int64_t)x.size();i++) x[i]=gval(i,D_GEN,false);
        for(int64_t i=0;i<LN_R;i++){ g[i]=1.0f+0.3f*std::sin(i*0.05f); b[i]=0.2f*std::cos(i*0.07f); }
        wbin(P(".x"),x); wbin(P(".g"),g); wbin(P(".b"),b);
    } else if (k==K_RMSNORM){
        std::vector<float> x(RN_BSH), g(RN_H);
        for(int64_t i=0;i<RN_BSH;i++) x[i]=gval(i,D_GEN,false);
        for(int64_t i=0;i<RN_H;i++) g[i]=1.0f+0.3f*std::sin(i*0.05f);
        wbin(P(".x"),x); wbin(P(".g"),g);
    } else if (k==K_MATMUL){
        std::vector<float> a(MM_M*MM_K), b(MM_K*MM_N);
        for(int64_t i=0;i<(int64_t)a.size();i++) a[i]=std::sin(i*0.011f)*0.9f;
        for(int64_t i=0;i<(int64_t)b.size();i++) b[i]=std::cos(i*0.013f)*0.9f;
        wbin(P(".x"),a); wbin(P(".y"),b);
    } else if (k==K_REDUCESUM){
        std::vector<float> x(RS_N);
        for(int64_t i=0;i<RS_N;i++) x[i]=gval(i,D_POS,false);
        wbin(P(".x"),x);
    } else if (k==K_REDUCE){
        int64_t n=reduce_n(op); std::vector<float> x(n);
        // reduceprod: keep near 1.0 so a 64-wide product stays in fp32 range; others use the general range
        for(int64_t i=0;i<n;i++) x[i] = (op=="reduceprod") ? (float)(1.0+0.02*std::sin(i*0.013)) : gval(i,D_GEN,false);
        wbin(P(".x"),x);
    } else if (k==K_SHIFT){
        std::vector<int32_t> x(SHIFT_N), y(SHIFT_N);
        for(int64_t i=0;i<SHIFT_N;i++){ x[i]=(int32_t)((i*7+1)%97); y[i]=(int32_t)(i%5); }   // value, shift 0..4
        wbin_i32(P(".x"),x); wbin_i32(P(".y"),y);
    } else if (k==K_BOOL){
        std::vector<float> x(NE);
        for(int64_t i=0;i<NE;i++){ float v=gval(i,D_GEN,false);
            if (i%16==0) v=INFINITY; else if (i%16==5) v=-INFINITY; else if (i%16==9) v=NAN;   // exercise inf/-inf/nan
            x[i]=v; }
        wbin(P(".x"),x);
    } else if (k==K_SCALAR){
        int64_t n=scalar_n(op); std::vector<float> x(n);
        for(int64_t i=0;i<n;i++) x[i]=gval(i,D_GEN,false);   // divs scalar=2 is the divisor (never zero), safe
        wbin(P(".x"),x);
    } else if (k==K_QUANT){
        std::vector<float> x(QUANT_N);
        for(int64_t i=0;i<QUANT_N;i++) x[i]=gval(i,D_GEN,false);
        wbin(P(".x"),x);
    } else if (k==K_SORT || k==K_TOPK){
        int64_t n = (k==K_SORT)?ST_N:TK_N; std::vector<float> x(n);
        for(int64_t i=0;i<n;i++) x[i]=gval(i*7,D_GEN,false);   // shuffle to avoid monotone sequence
        wbin(P(".x"),x);
    } else if (k==K_GROUPNORM){
        std::vector<float> x(GN_TOTAL), g(GN_C), b(GN_C);
        for(int64_t i=0;i<GN_TOTAL;i++) x[i]=gval(i,D_GEN,false);
        for(int64_t i=0;i<GN_C;i++){ g[i]=1.0f+0.3f*std::sin(i*0.7f); b[i]=0.2f*std::cos(i*0.5f); }
        wbin(P(".x"),x); wbin(P(".g"),g); wbin(P(".b"),b);
    } else if (k==K_DEEPNORM){
        std::vector<float> x(DN_H), gx(DN_H), g(DN_H), b(DN_H);
        for(int64_t i=0;i<DN_H;i++){ x[i]=gval(i,D_GEN,false); gx[i]=gval(i,D_GEN,true);
                                     g[i]=1.0f+0.3f*std::sin(i*0.5f); b[i]=0.2f*std::cos(i*0.4f); }
        wbin(P(".x"),x); wbin(P(".y"),gx); wbin(P(".g"),g); wbin(P(".b"),b);
    }
    printf("[gen] %s inputs -> %s.*\n", op.c_str(), prefix.c_str());
    return 0;
}

// Unary shim dispatch: returns GetWorkspaceSize / Execute function pointers
typedef aclnnStatus (*UnWs)(const aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
typedef aclnnStatus (*Run)(void*, uint64_t, aclOpExecutor*, aclrtStream);
static UnWs un_ws(const std::string&op){
    if(op=="relu")return aclnnReluGetWorkspaceSize; if(op=="exp")return aclnnExpGetWorkspaceSize;
    if(op=="sqrt")return aclnnSqrtGetWorkspaceSize; if(op=="abs")return aclnnAbsGetWorkspaceSize;
    if(op=="neg")return aclnnNegGetWorkspaceSize; if(op=="silu")return aclnnSiluGetWorkspaceSize;
    if(op=="gelu")return aclnnGeluGetWorkspaceSize; if(op=="sigmoid")return aclnnSigmoidGetWorkspaceSize;
    if(op=="tanh")return aclnnTanhGetWorkspaceSize; if(op=="erf")return aclnnErfGetWorkspaceSize;
    if(op=="log")return aclnnLogGetWorkspaceSize; if(op=="sin")return aclnnSinGetWorkspaceSize;
    if(op=="cos")return aclnnCosGetWorkspaceSize; if(op=="rsqrt")return aclnnRsqrtGetWorkspaceSize;
    if(op=="reciprocal")return aclnnReciprocalGetWorkspaceSize; if(op=="acos")return aclnnAcosGetWorkspaceSize;
    if(op=="asin")return aclnnAsinGetWorkspaceSize; if(op=="atan")return aclnnAtanGetWorkspaceSize;
    if(op=="cosh")return aclnnCoshGetWorkspaceSize; if(op=="sinh")return aclnnSinhGetWorkspaceSize;
    if(op=="tan")return aclnnTanGetWorkspaceSize; if(op=="ceil")return aclnnCeilGetWorkspaceSize;
    if(op=="floor")return aclnnFloorGetWorkspaceSize; if(op=="round")return aclnnRoundGetWorkspaceSize;
    if(op=="trunc")return aclnnTruncGetWorkspaceSize; if(op=="sign")return aclnnSignGetWorkspaceSize;
    if(op=="erfc")return aclnnErfcGetWorkspaceSize; if(op=="frac")return aclnnFracGetWorkspaceSize;
    if(op=="lgamma")return aclnnLgammaGetWorkspaceSize; if(op=="ln")return aclnnLogGetWorkspaceSize; if(op=="rint")return aclnnRoundGetWorkspaceSize;
    if(op=="acosh")return aclnnAcoshGetWorkspaceSize; if(op=="asinh")return aclnnAsinhGetWorkspaceSize;
    if(op=="atanh")return aclnnAtanhGetWorkspaceSize; if(op=="log2")return aclnnLog2GetWorkspaceSize;
    if(op=="log10")return aclnnLog10GetWorkspaceSize; if(op=="digamma")return aclnnDigammaGetWorkspaceSize;
    if(op=="sinc")return aclnnSincGetWorkspaceSize; if(op=="erfinv")return aclnnErfinvGetWorkspaceSize; if(op=="expm1")return aclnnExpm1GetWorkspaceSize; if(op=="log1p")return aclnnLog1pGetWorkspaceSize; if(op=="exp2")return aclnnExp2GetWorkspaceSize;
    return nullptr;
}
static Run un_run(const std::string&op){
    if(op=="relu")return aclnnRelu; if(op=="exp")return aclnnExp; if(op=="sqrt")return aclnnSqrt;
    if(op=="abs")return aclnnAbs; if(op=="neg")return aclnnNeg; if(op=="silu")return aclnnSilu;
    if(op=="gelu")return aclnnGelu; if(op=="sigmoid")return aclnnSigmoid; if(op=="tanh")return aclnnTanh;
    if(op=="erf")return aclnnErf; if(op=="log")return aclnnLog; if(op=="sin")return aclnnSin;
    if(op=="cos")return aclnnCos; if(op=="rsqrt")return aclnnRsqrt; if(op=="reciprocal")return aclnnReciprocal;
    if(op=="acos")return aclnnAcos; if(op=="asin")return aclnnAsin; if(op=="atan")return aclnnAtan;
    if(op=="cosh")return aclnnCosh; if(op=="sinh")return aclnnSinh; if(op=="tan")return aclnnTan;
    if(op=="ceil")return aclnnCeil; if(op=="floor")return aclnnFloor; if(op=="round")return aclnnRound;
    if(op=="trunc")return aclnnTrunc; if(op=="sign")return aclnnSign; if(op=="erfc")return aclnnErfc;
    if(op=="frac")return aclnnFrac; if(op=="lgamma")return aclnnLgamma; if(op=="ln")return aclnnLog; if(op=="rint")return aclnnRound;
    if(op=="acosh")return aclnnAcosh; if(op=="asinh")return aclnnAsinh; if(op=="atanh")return aclnnAtanh;
    if(op=="log2")return aclnnLog2; if(op=="log10")return aclnnLog10; if(op=="digamma")return aclnnDigamma;
    if(op=="sinc")return aclnnSinc; if(op=="erfinv")return aclnnErfinv; if(op=="expm1")return aclnnExpm1; if(op=="log1p")return aclnnLog1p; if(op=="exp2")return aclnnExp2;
    return nullptr;
}

// =========================== check ===========================
static int do_check(const std::string &op, const std::string &prefix, const std::string &goldenf){
    Kind k = kind_of(op);
    auto P=[&](const char*s){ return prefix+s; };
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0));
    aclrtStream s; CK(aclrtCreateStream(&s));
    std::vector<float> z, golden;

    if (k==K_BIN || k==K_UN){
        auto x=rbin(P(".x"),NE); golden=rbin(goldenf,NE);
        std::vector<float> y; if(k==K_BIN) y=rbin(P(".y"),NE);
        void *dx,*dy=nullptr,*dz;
        CK(aclrtMalloc(&dx,NE*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,NE*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,NE*4,x.data(),NE*4,ACL_MEMCPY_HOST_TO_DEVICE));
        if(k==K_BIN){ CK(aclrtMalloc(&dy,NE*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(dy,NE*4,y.data(),NE*4,ACL_MEMCPY_HOST_TO_DEVICE)); }
        aclTensor *tx=mkt({NE},ACL_FLOAT,dx),*tz=mkt({NE},ACL_FLOAT,dz),*ty=(k==K_BIN)?mkt({NE},ACL_FLOAT,dy):nullptr;
        uint64_t ws=0; aclOpExecutor*ex=nullptr; Run run=nullptr;
        if(k==K_BIN){
            if      (op=="add") CK(aclnnAddGetWorkspaceSize(tx,ty,nullptr,tz,&ws,&ex));
            else if (op=="sub") CK(aclnnSubGetWorkspaceSize(tx,ty,nullptr,tz,&ws,&ex));
            else if (op=="mul") CK(aclnnMulGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="div") CK(aclnnDivGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="max") CK(aclnnMaximumGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="min") CK(aclnnMinimumGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="power")CK(aclnnPowTensorTensorGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="hypot")CK(aclnnHypotGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            else if (op=="fmod") CK(aclnnFmodGetWorkspaceSize(tx,ty,tz,&ws,&ex));
            run = op=="add"?aclnnAdd:op=="sub"?aclnnSub:op=="mul"?aclnnMul:op=="div"?aclnnDiv:
                  op=="max"?aclnnMaximum:op=="min"?aclnnMinimum:op=="power"?aclnnPowTensorTensor:
                  op=="hypot"?aclnnHypot:aclnnFmod;
        } else {
            UnWs uw=un_ws(op); if(!uw){ fprintf(stderr,"unsupported un op %s\n",op.c_str()); return 1; }
            CK(uw(tx,tz,&ws,&ex)); run=un_run(op);
        }
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(run(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(NE); CK(aclrtMemcpy(z.data(),NE*4,dz,NE*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_SCALAR){
        int64_t n=scalar_n(op); auto x=rbin(P(".x"),n); golden=rbin(goldenf,n);
        void *dx,*dz; CK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({n},ACL_FLOAT,dx),*tz=mkt({n},ACL_FLOAT,dz);
        float s2=2.0f, sslope=0.1f, one=1.0f; aclScalar *S2=aclCreateScalar(&s2,ACL_FLOAT), *AL=aclCreateScalar(&one,ACL_FLOAT), *SL=aclCreateScalar(&sslope,ACL_FLOAT);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; void*dy=nullptr; aclTensor*ty=nullptr;
        if      (op=="adds") CK(aclnnAddsGetWorkspaceSize(tx,S2,AL,tz,&ws,&ex));
        else if (op=="subs") CK(aclnnSubsGetWorkspaceSize(tx,S2,AL,tz,&ws,&ex));
        else if (op=="muls") CK(aclnnMulsGetWorkspaceSize(tx,S2,tz,&ws,&ex));
        else if (op=="divs") CK(aclnnDivsGetWorkspaceSize(tx,S2,tz,&ws,&ex));
        else if (op=="maxs") CK(aclnnClampMinGetWorkspaceSize(tx,S2,tz,&ws,&ex));   // max(x,2)
        else if (op=="mins") CK(aclnnClampMaxGetWorkspaceSize(tx,S2,tz,&ws,&ex));   // min(x,2)
        else if (op=="leakyrelu") CK(aclnnLeakyReluGetWorkspaceSize(tx,SL,tz,&ws,&ex));
        else { // axpy: out = 2*x + 1  (y preset to 1.0)
            std::vector<float> yv(n,1.0f); CK(aclrtMalloc(&dy,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMemcpy(dy,n*4,yv.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
            ty=mkt({n},ACL_FLOAT,dy); CK(aclnnAxpyV2GetWorkspaceSize(tx,ty,2.0,tz,&ws,&ex)); }
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if      (op=="adds") CK(aclnnAdds(wsp,ws,ex,s));
        else if (op=="subs") CK(aclnnSubs(wsp,ws,ex,s));
        else if (op=="muls") CK(aclnnMuls(wsp,ws,ex,s));
        else if (op=="divs") CK(aclnnDivs(wsp,ws,ex,s));
        else if (op=="maxs") CK(aclnnClampMin(wsp,ws,ex,s));
        else if (op=="mins") CK(aclnnClampMax(wsp,ws,ex,s));
        else if (op=="leakyrelu") CK(aclnnLeakyRelu(wsp,ws,ex,s));
        else CK(aclnnAxpyV2(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        z.resize(n); CK(aclrtMemcpy(z.data(),n*4,dz,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_QUANT){
        auto x=rbin(P(".x"),QUANT_N); auto gu=rbin_u8(goldenf,QUANT_N);   // golden int8 stored as bytes
        float scl=(op=="quantize")?2.0f:1.0f, off=(op=="quantize")?1.0f:0.0f;
        void *dx,*dsc,*dof,*dz; CK(aclrtMalloc(&dx,QUANT_N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dsc,4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dof,4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,QUANT_N,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,QUANT_N*4,x.data(),QUANT_N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dsc,4,&scl,4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dof,4,&off,4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({QUANT_N,1},ACL_FLOAT,dx),*tsc=mkt({1},ACL_FLOAT,dsc),*tof=mkt({1},ACL_FLOAT,dof),*tz=mkt({QUANT_N,1},ACL_INT8,dz);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        if(op=="quantize") CK(aclnnQuantizeGetWorkspaceSize(tx,tsc,tof,tz,&ws,&ex)); else CK(aclnnAscendQuantGetWorkspaceSize(tx,tsc,tof,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if(op=="quantize") CK(aclnnQuantize(wsp,ws,ex,s)); else CK(aclnnAscendQuant(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        std::vector<int8_t> zi(QUANT_N); CK(aclrtMemcpy(zi.data(),QUANT_N,dz,QUANT_N,ACL_MEMCPY_DEVICE_TO_HOST));
        int bad=0; for(int64_t i=0;i<QUANT_N;i++) if(zi[i]!=(int8_t)gu[i]) bad++;
        printf("[%s] shim z[0]=%d golden[0]=%d  mismatches=%d/%ld  %s\n", op.c_str(), zi[0], (int8_t)gu[0], bad, (long)QUANT_N, bad==0?"PASS":"FAIL");
        return bad==0?0:1;
    }
    else if (k==K_SOFTMAX || k==K_LOGSOFTMAX){
        int64_t n=SM_ROWS*SM_COLS; auto x=rbin(P(".x"),n); golden=rbin(goldenf,n);
        void *dx,*dz; CK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({SM_ROWS,SM_COLS},ACL_FLOAT,dx),*tz=mkt({SM_ROWS,SM_COLS},ACL_FLOAT,dz);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        if(k==K_SOFTMAX) CK(aclnnSoftmaxGetWorkspaceSize(tx,-1,tz,&ws,&ex));
        else             CK(aclnnLogSoftmaxGetWorkspaceSize(tx,-1,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK((k==K_SOFTMAX?aclnnSoftmax:aclnnLogSoftmax)(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(n); CK(aclrtMemcpy(z.data(),n*4,dz,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_LAYERNORM){
        int64_t n=LN_A*LN_R; auto x=rbin(P(".x"),n), g=rbin(P(".g"),LN_R), b=rbin(P(".b"),LN_R); golden=rbin(goldenf,n);
        void *dx,*dz,*dg,*db;
        CK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dg,LN_R*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&db,LN_R*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dg,LN_R*4,g.data(),LN_R*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(db,LN_R*4,b.data(),LN_R*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({LN_A,LN_R},ACL_FLOAT,dx),*tz=mkt({LN_A,LN_R},ACL_FLOAT,dz);
        aclTensor *tg=mkt({LN_R},ACL_FLOAT,dg),*tb=mkt({LN_R},ACL_FLOAT,db);
        int64_t ns[1]={LN_R}; aclIntArray *nsh=aclCreateIntArray(ns,1);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        CK(aclnnLayerNormGetWorkspaceSize(tx,nsh,tg,tb,1e-5,tz,nullptr,nullptr,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnLayerNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(n); CK(aclrtMemcpy(z.data(),n*4,dz,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_RMSNORM){
        int64_t n=RN_BSH; auto x=rbin(P(".x"),n), g=rbin(P(".g"),RN_H); golden=rbin(goldenf,n);
        int64_t rows=RN_BSH/RN_H;
        void *dx,*dz,*dg;
        CK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dg,RN_H*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dg,RN_H*4,g.data(),RN_H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({rows,RN_H},ACL_FLOAT,dx),*tz=mkt({rows,RN_H},ACL_FLOAT,dz),*tg=mkt({RN_H},ACL_FLOAT,dg);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnRmsNormGetWorkspaceSize(tx,tg,1e-5,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnRmsNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(n); CK(aclrtMemcpy(z.data(),n*4,dz,n*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_MATMUL){
        auto a=rbin(P(".x"),MM_M*MM_K), b=rbin(P(".y"),MM_K*MM_N); golden=rbin(goldenf,MM_M*MM_N);
        std::vector<uint16_t> ah(MM_M*MM_K), bh(MM_K*MM_N);
        for(size_t i=0;i<ah.size();i++) ah[i]=f2h(a[i]);
        for(size_t i=0;i<bh.size();i++) bh[i]=f2h(b[i]);
        void *da,*db,*dc;
        CK(aclrtMalloc(&da,ah.size()*2,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&db,bh.size()*2,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dc,MM_M*MM_N*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(da,ah.size()*2,ah.data(),ah.size()*2,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(db,bh.size()*2,bh.data(),bh.size()*2,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *ta=mkt({MM_M,MM_K},ACL_FLOAT16,da),*tb=mkt({MM_K,MM_N},ACL_FLOAT16,db),*tc=mkt({MM_M,MM_N},ACL_FLOAT,dc);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnMatmulGetWorkspaceSize(ta,tb,tc,1,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnMatmul(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(MM_M*MM_N); CK(aclrtMemcpy(z.data(),MM_M*MM_N*4,dc,MM_M*MM_N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_REDUCESUM){
        auto x=rbin(P(".x"),RS_N); golden=rbin(goldenf,1);
        void *dx,*dz; CK(aclrtMalloc(&dx,RS_N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,RS_N*4,x.data(),RS_N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({RS_N},ACL_FLOAT,dx),*tz=mkt({1},ACL_FLOAT,dz);
        int64_t da[1]={0}; aclIntArray *dim=aclCreateIntArray(da,1);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnReduceSumGetWorkspaceSize(tx,dim,true,ACL_FLOAT,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnReduceSum(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(1); CK(aclrtMemcpy(z.data(),4,dz,4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_REDUCE){
        int64_t n=reduce_n(op); auto x=rbin(P(".x"),n); golden=rbin(goldenf,1);
        void *dx,*dz; CK(aclrtMalloc(&dx,n*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,n*4,x.data(),n*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({n},ACL_FLOAT,dx),*tz=mkt({1},ACL_FLOAT,dz);
        int64_t da[1]={0}; aclIntArray *dim=aclCreateIntArray(da,1);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        if      (op=="reducemax") CK(aclnnAmaxGetWorkspaceSize(tx,dim,true,tz,&ws,&ex));
        else if (op=="reducemin") CK(aclnnAminGetWorkspaceSize(tx,dim,true,tz,&ws,&ex));
        else if (op=="mean")      CK(aclnnMeanGetWorkspaceSize(tx,dim,true,ACL_FLOAT,tz,&ws,&ex));
        else if (op=="reduceprod")CK(aclnnProdGetWorkspaceSize(tx,dim,true,ACL_FLOAT,tz,&ws,&ex));
        else                      CK(aclnnReduceSumGetWorkspaceSize(tx,dim,true,ACL_FLOAT,tz,&ws,&ex));   // sum
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if      (op=="reducemax") CK(aclnnAmax(wsp,ws,ex,s));
        else if (op=="reducemin") CK(aclnnAmin(wsp,ws,ex,s));
        else if (op=="mean")      CK(aclnnMean(wsp,ws,ex,s));
        else if (op=="reduceprod")CK(aclnnProd(wsp,ws,ex,s));
        else                      CK(aclnnReduceSum(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        z.resize(1); CK(aclrtMemcpy(z.data(),4,dz,4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_SHIFT){
        auto x=rbin_i32(P(".x"),SHIFT_N), y=rbin_i32(P(".y"),SHIFT_N); auto gi=rbin_i32(goldenf,SHIFT_N);
        void *dx,*dy,*dz; CK(aclrtMalloc(&dx,SHIFT_N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dy,SHIFT_N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,SHIFT_N*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,SHIFT_N*4,x.data(),SHIFT_N*4,ACL_MEMCPY_HOST_TO_DEVICE)); CK(aclrtMemcpy(dy,SHIFT_N*4,y.data(),SHIFT_N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({SHIFT_N},ACL_INT32,dx),*ty=mkt({SHIFT_N},ACL_INT32,dy),*tz=mkt({SHIFT_N},ACL_INT32,dz);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        if(op=="shiftleft") CK(aclnnLeftShiftGetWorkspaceSize(tx,ty,tz,&ws,&ex)); else CK(aclnnRightShiftGetWorkspaceSize(tx,ty,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if(op=="shiftleft") CK(aclnnLeftShift(wsp,ws,ex,s)); else CK(aclnnRightShift(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        std::vector<int32_t> zi(SHIFT_N); CK(aclrtMemcpy(zi.data(),SHIFT_N*4,dz,SHIFT_N*4,ACL_MEMCPY_DEVICE_TO_HOST));
        int bad=0; for(int64_t i=0;i<SHIFT_N;i++) if(zi[i]!=gi[i]) bad++;
        printf("[%s] shim z[0]=%d golden[0]=%d  mismatches=%d/%ld  %s\n", op.c_str(), zi[0], gi[0], bad, (long)SHIFT_N, bad==0?"PASS":"FAIL");
        return bad==0?0:1;
    }
    else if (k==K_BOOL){
        auto x=rbin(P(".x"),NE); auto gu=rbin_u8(goldenf,NE);
        void *dx,*dz; CK(aclrtMalloc(&dx,NE*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,NE,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,NE*4,x.data(),NE*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({NE},ACL_FLOAT,dx),*tz=mkt({NE},ACL_BOOL,dz);
        uint64_t ws=0; aclOpExecutor*ex=nullptr;
        if      (op=="is_finite") CK(aclnnIsFiniteGetWorkspaceSize(tx,tz,&ws,&ex));
        else if (op=="is_inf")    CK(aclnnIsInfGetWorkspaceSize(tx,tz,&ws,&ex));
        else                      CK(aclnnIsNanGetWorkspaceSize(tx,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        if      (op=="is_finite") CK(aclnnIsFinite(wsp,ws,ex,s));
        else if (op=="is_inf")    CK(aclnnIsInf(wsp,ws,ex,s));
        else                      CK(aclnnIsNan(wsp,ws,ex,s));
        CK(aclrtSynchronizeStream(s));
        std::vector<uint8_t> zu(NE); CK(aclrtMemcpy(zu.data(),NE,dz,NE,ACL_MEMCPY_DEVICE_TO_HOST));
        int bad=0; for(int64_t i=0;i<NE;i++) if((zu[i]!=0)!=(gu[i]!=0)) bad++;
        printf("[%s] shim[0]=%d golden[0]=%d  mismatches=%d/%ld  %s\n", op.c_str(), zu[0], gu[0], bad, (long)NE, bad==0?"PASS":"FAIL");
        return bad==0?0:1;
    }
    else if (k==K_SORT){
        auto x=rbin(P(".x"),ST_N); golden=rbin(goldenf,ST_N);   // golden = Ascend ascending-sort values
        void *dx,*dv,*di; CK(aclrtMalloc(&dx,ST_N*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dv,ST_N*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&di,ST_N*8,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,ST_N*4,x.data(),ST_N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({ST_N},ACL_FLOAT,dx),*tv=mkt({ST_N},ACL_FLOAT,dv),*ti=mkt({ST_N},ACL_INT64,di);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnSortGetWorkspaceSize(tx,0,false,false,tv,ti,&ws,&ex));   // ascending
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnSort(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(ST_N); CK(aclrtMemcpy(z.data(),ST_N*4,dv,ST_N*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_TOPK){
        auto x=rbin(P(".x"),TK_N); golden=rbin(goldenf,TK_K);   // golden = Ascend top-K largest values
        void *dx,*dv,*di; CK(aclrtMalloc(&dx,TK_N*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dv,TK_K*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&di,TK_K*8,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,TK_N*4,x.data(),TK_N*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({TK_N},ACL_FLOAT,dx),*tv=mkt({TK_K},ACL_FLOAT,dv),*ti=mkt({TK_K},ACL_INT64,di);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnTopkGetWorkspaceSize(tx,TK_K,0,true,true,tv,ti,&ws,&ex));   // largest K, sorted
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnTopk(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(TK_K); CK(aclrtMemcpy(z.data(),TK_K*4,dv,TK_K*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_GROUPNORM){
        auto x=rbin(P(".x"),GN_TOTAL), g=rbin(P(".g"),GN_C), b=rbin(P(".b"),GN_C); golden=rbin(goldenf,GN_TOTAL);
        void *dx,*dz,*dg,*db;
        CK(aclrtMalloc(&dx,GN_TOTAL*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dz,GN_TOTAL*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dg,GN_C*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&db,GN_C*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,GN_TOTAL*4,x.data(),GN_TOTAL*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dg,GN_C*4,g.data(),GN_C*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(db,GN_C*4,b.data(),GN_C*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({1,GN_C,GN_HW},ACL_FLOAT,dx),*tz=mkt({1,GN_C,GN_HW},ACL_FLOAT,dz);
        aclTensor *tg=mkt({GN_C},ACL_FLOAT,dg),*tb=mkt({GN_C},ACL_FLOAT,db);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnGroupNormGetWorkspaceSize(tx,tg,tb,GN_G,1e-5,tz,&ws,&ex));
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnGroupNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(GN_TOTAL); CK(aclrtMemcpy(z.data(),GN_TOTAL*4,dz,GN_TOTAL*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }
    else if (k==K_DEEPNORM){
        auto x=rbin(P(".x"),DN_H), gx=rbin(P(".y"),DN_H), g=rbin(P(".g"),DN_H), b=rbin(P(".b"),DN_H); golden=rbin(goldenf,DN_H);
        void *dx,*dgx,*dz,*dg,*db;
        CK(aclrtMalloc(&dx,DN_H*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dgx,DN_H*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMalloc(&dz,DN_H*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&dg,DN_H*4,ACL_MEM_MALLOC_HUGE_FIRST)); CK(aclrtMalloc(&db,DN_H*4,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclrtMemcpy(dx,DN_H*4,x.data(),DN_H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dgx,DN_H*4,gx.data(),DN_H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(dg,DN_H*4,g.data(),DN_H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        CK(aclrtMemcpy(db,DN_H*4,b.data(),DN_H*4,ACL_MEMCPY_HOST_TO_DEVICE));
        aclTensor *tx=mkt({1,DN_H},ACL_FLOAT,dx),*tgx=mkt({1,DN_H},ACL_FLOAT,dgx),*tz=mkt({1,DN_H},ACL_FLOAT,dz);
        aclTensor *tg=mkt({DN_H},ACL_FLOAT,dg),*tb=mkt({DN_H},ACL_FLOAT,db);
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CK(aclnnDeepNormGetWorkspaceSize(tx,tgx,tg,tb,2.0,1e-5,tz,&ws,&ex));  // alpha=2 (matches explorer)
        void *wsp=nullptr; if(ws) CK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CK(aclnnDeepNorm(wsp,ws,ex,s)); CK(aclrtSynchronizeStream(s));
        z.resize(DN_H); CK(aclrtMemcpy(z.data(),DN_H*4,dz,DN_H*4,ACL_MEMCPY_DEVICE_TO_HOST));
    }

    double me=0,mr=0; int64_t n=(int64_t)z.size();
    for(int64_t i=0;i<n;i++){ me=std::max(me,(double)std::fabs(z[i]-golden[i])); mr=std::max(mr,(double)std::fabs(golden[i])); }
    double err=me/(mr+1e-9), tol=tol_of(op);
    printf("[%s] shim z[0]=%.6f golden[0]=%.6f  normalized_err=%.3e (tol=%.0e)  %s\n",
           op.c_str(), z[0], golden[0], err, tol, err<=tol?"PASS":"FAIL");
    return err<=tol?0:1;
}

int main(int argc, char **argv){
    if (argc<2){ fprintf(stderr,"usage: %s gen|check ...\n",argv[0]); return 1; }
    if (!strcmp(argv[1],"gen")){
        if (argc<4){ fprintf(stderr,"usage: %s gen <op> <prefix>\n",argv[0]); return 1; }
        return do_gen(argv[2], argv[3]);
    }
    if (!strcmp(argv[1],"check")){
        if (argc<5){ fprintf(stderr,"usage: %s check <op> <prefix> <out.bin>\n",argv[0]); return 1; }
        return do_check(argv[2], argv[3], argv[4]);
    }
    fprintf(stderr,"unknown mode %s\n",argv[1]); return 1;
}
