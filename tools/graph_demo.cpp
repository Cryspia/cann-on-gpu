// Equivalent model container demo / self-test: build an Embedding->RMSNorm->Matmul->ReLU->Matmul graph,
// serialize to disk -> reload -> execute; compare against "direct shim op-by-op" results
// (should be bit-exact, since the same set of operators is used).
// Demonstrates "load model -> execute" equivalent capability (non-Ascend, non-.om format).
#include "cann_graph.h"
#include <cmath>
#include <cstdlib>
using namespace cann_graph;

static aclrtStream g_stream;
#define CHECK(x) do{int _r=(int)(x); if(_r){printf("[FATAL] %s:%d ret=%d\n",__FILE__,__LINE__,_r);exit(1);} }while(0)

// Dimensions: V vocab size, D model dim, H hidden dim, S sequence length
static const int64_t V=16, D=8, H=12, S=3;
static const double EPS=1e-6;

int main(){
    CHECK(aclInit(nullptr)); CHECK(aclrtSetDevice(0)); CHECK(aclrtCreateStream(&g_stream)); srand(7);
    auto rv=[&](int64_t n,double sc){ std::vector<float> v(n); for(auto&x:v)x=(float)((rand()/(double)RAND_MAX*2-1)*sc); return v; };

    // weights
    auto Wemb=rv(V*D,0.5); std::vector<float> g1(D); for(auto&x:g1)x=1.0f+(float)((rand()/(double)RAND_MAX*2-1)*0.1);
    auto W1=rv(D*H,0.4), W2=rv(H*D,0.4);
    std::vector<int64_t> ids(S); for(auto&x:ids)x=rand()%V;

    // ---- build graph ----
    Graph g;
    auto addT=[&](int kind,int dtype,std::vector<int64_t> dims,int64_t off=-1){ g.tensors.push_back({kind,dtype,dims,off}); return (int)g.tensors.size()-1; };
    auto addW=[&](const std::vector<float>&w,std::vector<int64_t> dims){ int64_t off=g.blob.size(); g.blob.insert(g.blob.end(),w.begin(),w.end()); return addT(GK_WEIGHT,ACL_FLOAT,dims,off); };
    int tIds = addT(GK_INPUT, ACL_INT64, {S});
    int tWemb= addW(Wemb,{V,D});
    int tG1  = addW(g1,{D});
    int tW1  = addW(W1,{D,H});
    int tW2  = addW(W2,{H,D});
    int tX   = addT(GK_ACT,ACL_FLOAT,{S,D});
    int tHn  = addT(GK_ACT,ACL_FLOAT,{S,D});
    int tA   = addT(GK_ACT,ACL_FLOAT,{S,H});
    int tR   = addT(GK_ACT,ACL_FLOAT,{S,H});
    int tY   = addT(GK_ACT,ACL_FLOAT,{S,D});
    g.inputs={tIds}; g.outputs={tY};
    g.nodes.push_back({GOP_EMBEDDING,{tWemb,tIds},tX,0});
    g.nodes.push_back({GOP_RMSNORM, {tX,tG1},     tHn,EPS});
    g.nodes.push_back({GOP_MATMUL,  {tHn,tW1},    tA,0});
    g.nodes.push_back({GOP_RELU,    {tA},         tR,0});
    g.nodes.push_back({GOP_MATMUL,  {tR,tW2},     tY,0});

    // ---- serialize -> reload ----
    const char*path="/tmp/cann_graph_demo.cgm";
    if(!save(g,path)){ printf("[FATAL] save failed\n"); return 1; }
    Graph g2; if(!load(g2,path)){ printf("[FATAL] load failed\n"); return 1; }
    printf("[graph] save->load: tensors %zu->%zu, nodes %zu->%zu, blob %zu->%zu floats\n",
           g.tensors.size(),g2.tensors.size(),g.nodes.size(),g2.nodes.size(),g.blob.size(),g2.blob.size());

    // ---- execute the loaded graph ----
    Runtime rt; rt.stream=g_stream;
    std::map<int,const void*> inputs; inputs[tIds]=ids.data();
    auto outs=rt.execute(g2,inputs);
    std::vector<float> gy(S*D); CHECK(aclrtMemcpy(gy.data(),S*D*4,outs[tY],S*D*4,ACL_MEMCPY_DEVICE_TO_HOST));
    rt.free_all();

    // ---- reference: direct op-by-op shim calls ----
    std::vector<void*> frees;
    auto dal=[&](size_t b){ void*p; CHECK(aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST)); frees.push_back(p); return p; };
    auto dup=[&](const std::vector<float>&v){ void*p=dal(v.size()*4); CHECK(aclrtMemcpy(p,v.size()*4,v.data(),v.size()*4,ACL_MEMCPY_HOST_TO_DEVICE)); return p; };
    auto Tt=[&](std::vector<int64_t> d,void*p,aclDataType dt){ return aclCreateTensor(d.data(),d.size(),dt,nullptr,0,ACL_FORMAT_ND,d.data(),d.size(),p); };
    auto run2=[&](auto getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
        uint64_t ws=0; aclOpExecutor*ex=nullptr; CHECK(getws(&ws,&ex));
        void*wsp=nullptr; if(ws)CHECK(aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK(run(wsp,ws,ex,g_stream)); CHECK(aclrtSynchronizeStream(g_stream)); if(wsp)aclrtFree(wsp); };
    void*dWemb=dup(Wemb),*dG1=dup(g1),*dW1=dup(W1),*dW2=dup(W2);
    void*dids=dal(S*8); CHECK(aclrtMemcpy(dids,S*8,ids.data(),S*8,ACL_MEMCPY_HOST_TO_DEVICE));
    void*dX=dal(S*D*4),*dHn=dal(S*D*4),*dA=dal(S*H*4),*dR=dal(S*H*4),*dY=dal(S*D*4);
    { aclTensor*w=Tt({V,D},dWemb,ACL_FLOAT),*id=Tt({S},dids,ACL_INT64),*o=Tt({S,D},dX,ACL_FLOAT);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(w,id,o,ws,e);},aclnnEmbedding); }
    { aclTensor*x=Tt({S,D},dX,ACL_FLOAT),*gg=Tt({D},dG1,ACL_FLOAT),*o=Tt({S,D},dHn,ACL_FLOAT);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(x,gg,EPS,o,ws,e);},aclnnRmsNorm); }
    { aclTensor*a=Tt({S,D},dHn,ACL_FLOAT),*b=Tt({D,H},dW1,ACL_FLOAT),*o=Tt({S,H},dA,ACL_FLOAT);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(a,b,o,1,ws,e);},aclnnMatmul); }
    { aclTensor*a=Tt({S,H},dA,ACL_FLOAT),*o=Tt({S,H},dR,ACL_FLOAT);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnReluGetWorkspaceSize(a,o,ws,e);},aclnnRelu); }
    { aclTensor*a=Tt({S,H},dR,ACL_FLOAT),*b=Tt({H,D},dW2,ACL_FLOAT),*o=Tt({S,D},dY,ACL_FLOAT);
      run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(a,b,o,1,ws,e);},aclnnMatmul); }
    std::vector<float> ry(S*D); CHECK(aclrtMemcpy(ry.data(),S*D*4,dY,S*D*4,ACL_MEMCPY_DEVICE_TO_HOST));
    for(void*p:frees) aclrtFree(p);

    // ---- comparison ----
    double me=0; for(int64_t i=0;i<S*D;i++) me=std::max(me,(double)std::fabs(gy[i]-ry[i]));
    bool ok = me==0.0;   // same operators, same weights -> must be exactly equal
    printf("model container load->execute vs direct op-by-op: max|diff|=%.3e  %s\n", me, ok?"PASS":"FAIL");
    printf("== %d PASS, %d FAIL ==\n", ok?1:0, ok?0:1);

    CHECK(aclrtDestroyStream(g_stream)); CHECK(aclrtResetDevice(0)); CHECK(aclFinalize());
    return ok?0:1;
}
