// Equivalent model container: minimal operator-graph IR + binary serialization + sequential executor,
// implemented as a custom non-.om format.
// Purpose: provide "load model -> execute" equivalent capability (as opposed to implementing the
// Ascend-proprietary aclmdl/.om black box).
// Design: graph = tensor table + node table (topological order) + weight blob.
// execute() dispatches each node to the corresponding shim aclnn operator.
// Control flow: currently sequential (DAG single pass); loop/branch support can be added via
// new node types (decode loop deferred).
// Pure ACL client interface; links only libascendcl.so — same linking style as Ascend applications.
#ifndef CANN_GRAPH_H
#define CANN_GRAPH_H
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <map>
#include <string>
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_add.h"
#include "aclnnop/aclnn_ops.h"

namespace cann_graph {

// Supported operators (sufficient to express a transformer/MLP block; extensible as needed)
enum GOp { GOP_EMBEDDING=1, GOP_RMSNORM, GOP_MATMUL, GOP_ADD, GOP_RELU, GOP_SWIGLU };
// Tensor role
enum GKind { GK_INPUT=0, GK_WEIGHT=1, GK_ACT=2 };

struct GTensor {
    int kind;                       // GKind
    int dtype;                      // aclDataType
    std::vector<int64_t> dims;
    int64_t blobOffset;             // GK_WEIGHT: float offset into blob (fp32 weights only); -1 otherwise
};
struct GNode {
    int op;                         // GOp
    std::vector<int> ins;           // input tensor ids
    int out;                        // output tensor id
    double attr;                    // scalar attribute (e.g. rmsnorm eps)
};
struct Graph {
    std::vector<GTensor> tensors;
    std::vector<GNode> nodes;
    std::vector<int> inputs, outputs;   // tensor ids
    std::vector<float> blob;            // all fp32 weights flattened
};

// ---- Binary serialization (little-endian, fixed layout; self-describing, loadable by another process) ----
static const uint32_t MAGIC = 0x43474D31;   // "CGM1"
inline void wr(FILE*f,const void*p,size_t n){ fwrite(p,1,n,f); }
inline void wi(FILE*f,int64_t v){ wr(f,&v,8); }
template<typename T> inline void rd(FILE*f,T*p,size_t n){ if(fread(p,1,n,f)!=n){fprintf(stderr,"[cann_graph] read err\n");exit(1);} }
inline int64_t ri(FILE*f){ int64_t v; rd(f,&v,8); return v; }

inline bool save(const Graph&g,const std::string&path){
    FILE*f=fopen(path.c_str(),"wb"); if(!f) return false;
    uint32_t m=MAGIC; wr(f,&m,4);
    wi(f,(int64_t)g.tensors.size());
    for(auto&t:g.tensors){ wi(f,t.kind); wi(f,t.dtype); wi(f,(int64_t)t.dims.size());
        for(auto d:t.dims) wi(f,d); wi(f,t.blobOffset); }
    wi(f,(int64_t)g.nodes.size());
    for(auto&n:g.nodes){ wi(f,n.op); wi(f,(int64_t)n.ins.size()); for(int x:n.ins) wi(f,x); wi(f,n.out); wr(f,&n.attr,8); }
    wi(f,(int64_t)g.inputs.size()); for(int x:g.inputs) wi(f,x);
    wi(f,(int64_t)g.outputs.size()); for(int x:g.outputs) wi(f,x);
    wi(f,(int64_t)g.blob.size()); if(!g.blob.empty()) wr(f,g.blob.data(),g.blob.size()*4);
    fclose(f); return true;
}
inline bool load(Graph&g,const std::string&path){
    FILE*f=fopen(path.c_str(),"rb"); if(!f) return false;
    uint32_t m; rd(f,&m,4); if(m!=MAGIC){ fclose(f); return false; }
    int64_t nt=ri(f); g.tensors.resize(nt);
    for(auto&t:g.tensors){ t.kind=(int)ri(f); t.dtype=(int)ri(f); int64_t nd=ri(f); t.dims.resize(nd);
        for(auto&d:t.dims) d=ri(f); t.blobOffset=ri(f); }
    int64_t nn=ri(f); g.nodes.resize(nn);
    for(auto&n:g.nodes){ n.op=(int)ri(f); int64_t ni=ri(f); n.ins.resize(ni); for(auto&x:n.ins) x=(int)ri(f); n.out=(int)ri(f); rd(f,&n.attr,8); }
    int64_t nin=ri(f); g.inputs.resize(nin); for(auto&x:g.inputs) x=(int)ri(f);
    int64_t no=ri(f); g.outputs.resize(no); for(auto&x:g.outputs) x=(int)ri(f);
    int64_t nb=ri(f); g.blob.resize(nb); if(nb) rd(f,g.blob.data(),nb*4);
    fclose(f); return true;
}

// ---- Runtime executor ----
struct Runtime {
    const Graph* g=nullptr; aclrtStream stream;
    std::vector<void*> dev;        // device pointer for each tensor id
    std::vector<void*> frees;
    int64_t numel(const GTensor&t)const{ int64_t n=1; for(auto d:t.dims) n*=d; return n; }
    size_t bytes(const GTensor&t)const{ size_t es=(t.dtype==ACL_INT64)?8:(t.dtype==ACL_FLOAT?4:4); return numel(t)*es; }
    void* alloc(size_t b){ void*p; aclrtMalloc(&p,b,ACL_MEM_MALLOC_HUGE_FIRST); frees.push_back(p); return p; }
    aclTensor* T(const GTensor&t,void*p){ return aclCreateTensor(t.dims.data(),t.dims.size(),(aclDataType)t.dtype,nullptr,0,ACL_FORMAT_ND,t.dims.data(),t.dims.size(),p); }
    template<typename G2> void run2(G2 getws, aclnnStatus(*run)(void*,uint64_t,aclOpExecutor*,aclrtStream)){
        uint64_t ws=0; aclOpExecutor*ex=nullptr; getws(&ws,&ex);
        void*wsp=nullptr; if(ws) aclrtMalloc(&wsp,ws,ACL_MEM_MALLOC_HUGE_FIRST);
        run(wsp,ws,ex,stream); aclrtSynchronizeStream(stream); if(wsp) aclrtFree(wsp);
    }
    // inputData: input tensor id -> host pointer (matching tensor dtype/numel). Returns output id -> device pointer.
    std::map<int,void*> execute(const Graph&graph, const std::map<int,const void*>&inputData){
        g=&graph; dev.assign(g->tensors.size(),nullptr);
        // allocate + upload weights/inputs
        for(size_t i=0;i<g->tensors.size();++i){ const GTensor&t=g->tensors[i]; void*p=alloc(bytes(t));
            dev[i]=p;
            if(t.kind==GK_WEIGHT){ aclrtMemcpy(p,bytes(t),g->blob.data()+t.blobOffset,bytes(t),ACL_MEMCPY_HOST_TO_DEVICE); }
            else if(t.kind==GK_INPUT){ auto it=inputData.find((int)i); if(it!=inputData.end()) aclrtMemcpy(p,bytes(t),it->second,bytes(t),ACL_MEMCPY_HOST_TO_DEVICE); }
        }
        // execute nodes in topological order
        for(const GNode&n:g->nodes){
            const GTensor&ot=g->tensors[n.out]; void*o=dev[n.out];
            switch(n.op){
                case GOP_EMBEDDING:{ const GTensor&w=g->tensors[n.ins[0]]; const GTensor&id=g->tensors[n.ins[1]];
                    aclTensor*tw=T(w,dev[n.ins[0]]),*tid=T(id,dev[n.ins[1]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnEmbeddingGetWorkspaceSize(tw,tid,to,ws,e);},aclnnEmbedding); break; }
                case GOP_RMSNORM:{ const GTensor&x=g->tensors[n.ins[0]]; const GTensor&gg=g->tensors[n.ins[1]];
                    aclTensor*tx=T(x,dev[n.ins[0]]),*tg=T(gg,dev[n.ins[1]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnRmsNormGetWorkspaceSize(tx,tg,n.attr,to,ws,e);},aclnnRmsNorm); break; }
                case GOP_MATMUL:{ const GTensor&a=g->tensors[n.ins[0]]; const GTensor&b=g->tensors[n.ins[1]];
                    aclTensor*ta=T(a,dev[n.ins[0]]),*tb=T(b,dev[n.ins[1]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnMatmulGetWorkspaceSize(ta,tb,to,1,ws,e);},aclnnMatmul); break; }
                case GOP_ADD:{ const GTensor&a=g->tensors[n.ins[0]]; const GTensor&b=g->tensors[n.ins[1]];
                    aclTensor*ta=T(a,dev[n.ins[0]]),*tb=T(b,dev[n.ins[1]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnAddGetWorkspaceSize(ta,tb,nullptr,to,ws,e);},aclnnAdd); break; }
                case GOP_RELU:{ const GTensor&a=g->tensors[n.ins[0]];
                    aclTensor*ta=T(a,dev[n.ins[0]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnReluGetWorkspaceSize(ta,to,ws,e);},aclnnRelu); break; }
                case GOP_SWIGLU:{ const GTensor&a=g->tensors[n.ins[0]];
                    aclTensor*ta=T(a,dev[n.ins[0]]),*to=T(ot,o);
                    run2([&](uint64_t*ws,aclOpExecutor**e){return aclnnSwiGluGetWorkspaceSize(ta,to,ws,e);},aclnnSwiGlu); break; }
                default: fprintf(stderr,"[cann_graph] unknown op %d\n",n.op); exit(1);
            }
        }
        std::map<int,void*> outs; for(int id:g->outputs) outs[id]=dev[id]; return outs;
    }
    void free_all(){ for(void*p:frees) aclrtFree(p); frees.clear(); }
};

} // namespace cann_graph
#endif
