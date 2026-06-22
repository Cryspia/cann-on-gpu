// Legacy single-operator interface: aclTensorDesc/aclDataBuffer/aclopAttr + aclopExecuteV2 /
// aclopCompileAndExecute routing by opType string to already-implemented aclnn kernels. Ported from the
// CUDA backend's aclop.cu (workspace alloc swapped to the Metal allocator; our aclnn ops use ws=0).
#import "internal.h"
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "aclnnop/aclnn_ops.h"
#include "aclnnop/aclnn_add.h"
#include <string>
#include <vector>
#include <map>

struct aclTensorDesc { aclDataType dtype = ACL_DT_UNDEFINED; aclFormat format = ACL_FORMAT_ND; std::vector<int64_t> dims; };
struct aclDataBuffer { void *data = nullptr; size_t size = 0; };
struct aclopAttr {
    std::map<std::string, int64_t> ints; std::map<std::string, float> floats;
    std::map<std::string, uint8_t> bools; std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<int64_t>> listInts;
};

extern "C" {
aclTensorDesc *aclCreateTensorDesc(aclDataType dataType, int numDims, const int64_t *dims, aclFormat format) {
    auto *d = new aclTensorDesc(); d->dtype = dataType; d->format = format;
    if (dims && numDims > 0) d->dims.assign(dims, dims + numDims); return d;
}
void aclDestroyTensorDesc(const aclTensorDesc *desc) { delete desc; }
aclDataType aclGetTensorDescType(const aclTensorDesc *desc) { return desc ? desc->dtype : ACL_DT_UNDEFINED; }
aclFormat aclGetTensorDescFormat(const aclTensorDesc *desc) { return desc ? desc->format : ACL_FORMAT_ND; }
size_t aclGetTensorDescNumDims(const aclTensorDesc *desc) { return desc ? desc->dims.size() : 0; }
int64_t aclGetTensorDescDim(const aclTensorDesc *desc, size_t index) { return (desc && index < desc->dims.size()) ? desc->dims[index] : -1; }
aclError aclGetTensorDescDimV2(const aclTensorDesc *desc, size_t index, int64_t *dimSize) {
    if (!desc || !dimSize || index >= desc->dims.size()) return (aclError)ACLNN_ERR_PARAM_INVALID; *dimSize = desc->dims[index]; return ACL_SUCCESS;
}
aclDataBuffer *aclCreateDataBuffer(void *data, size_t size) { auto *b = new aclDataBuffer(); b->data = data; b->size = size; return b; }
aclError aclDestroyDataBuffer(const aclDataBuffer *dataBuffer) { delete dataBuffer; return ACL_SUCCESS; }
void *aclGetDataBufferAddr(const aclDataBuffer *dataBuffer) { return dataBuffer ? dataBuffer->data : nullptr; }
size_t aclGetDataBufferSizeV2(const aclDataBuffer *dataBuffer) { return dataBuffer ? dataBuffer->size : 0; }
aclopAttr *aclopCreateAttr() { return new aclopAttr(); }
void aclopDestroyAttr(const aclopAttr *attr) { delete attr; }
aclError aclopSetAttrBool(aclopAttr *a, const char *n, uint8_t v) { if(!a||!n)return (aclError)ACLNN_ERR_PARAM_INVALID; a->bools[n]=v; return ACL_SUCCESS; }
aclError aclopSetAttrInt(aclopAttr *a, const char *n, int64_t v) { if(!a||!n)return (aclError)ACLNN_ERR_PARAM_INVALID; a->ints[n]=v; return ACL_SUCCESS; }
aclError aclopSetAttrFloat(aclopAttr *a, const char *n, float v) { if(!a||!n)return (aclError)ACLNN_ERR_PARAM_INVALID; a->floats[n]=v; return ACL_SUCCESS; }
aclError aclopSetAttrString(aclopAttr *a, const char *n, const char *v) { if(!a||!n||!v)return (aclError)ACLNN_ERR_PARAM_INVALID; a->strs[n]=v; return ACL_SUCCESS; }
aclError aclopSetAttrDataType(aclopAttr *a, const char *n, aclDataType v) { if(!a||!n)return (aclError)ACLNN_ERR_PARAM_INVALID; a->ints[n]=(int64_t)v; return ACL_SUCCESS; }
aclError aclopSetAttrListInt(aclopAttr *a, const char *n, int cnt, const int64_t *v) { if(!a||!n||(cnt>0&&!v))return (aclError)ACLNN_ERR_PARAM_INVALID; a->listInts[n].assign(v, v+cnt); return ACL_SUCCESS; }
aclError aclopSetAttrListBool(aclopAttr *a, const char *n, int cnt, const uint8_t *v) { (void)cnt;(void)v; return a&&n?ACL_SUCCESS:(aclError)ACLNN_ERR_PARAM_INVALID; }
aclError aclopSetAttrListFloat(aclopAttr *a, const char *n, int cnt, const float *v) { (void)cnt;(void)v; return a&&n?ACL_SUCCESS:(aclError)ACLNN_ERR_PARAM_INVALID; }
} // extern "C"

namespace {
aclTensor *mk_t(const aclTensorDesc *d, void *data) {
    if (!d) return nullptr;
    return aclCreateTensor(d->dims.data(), d->dims.size(), d->dtype, nullptr, 0, d->format, d->dims.data(), d->dims.size(), data);
}
template <typename GetWs>
aclnnStatus two_stage(GetWs getws, aclnnStatus (*run)(void*, uint64_t, aclOpExecutor*, aclrtStream), aclrtStream s) {
    uint64_t ws = 0; aclOpExecutor *ex = nullptr;
    aclnnStatus st = getws(&ws, &ex); if (st != ACLNN_SUCCESS) return st;
    void *wsp = ws ? mtl::alloc(ws) : nullptr;
    st = run(wsp, ws, ex, s);
    if (wsp) mtl::free_(wsp);
    return st;
}
aclError route(const std::string &op, int ni, const aclTensorDesc *const inD[], void *const inBuf[],
               int no, const aclTensorDesc *const outD[], void *const outBuf[], const aclopAttr *attr, aclrtStream stream) {
    if (no < 1) return (aclError)ACLNN_ERR_PARAM_INVALID;
    aclTensor *out = mk_t(outD[0], outBuf[0]);
    aclTensor *a = ni >= 1 ? mk_t(inD[0], inBuf[0]) : nullptr;
    aclTensor *b = ni >= 2 ? mk_t(inD[1], inBuf[1]) : nullptr;
    aclnnStatus st = ACLNN_ERR_PARAM_INVALID;
    #define UN(NAME, FN) if (op == NAME) { st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return FN##GetWorkspaceSize(a,out,w,e);}, FN, stream); }
    UN("Abs", aclnnAbs) else UN("Exp", aclnnExp) else UN("Sqrt", aclnnSqrt)
    else UN("Relu", aclnnRelu) else UN("Neg", aclnnNeg) else UN("Sigmoid", aclnnSigmoid) else UN("Tanh", aclnnTanh)
    else if (op == "Add") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnAddGetWorkspaceSize(a,b,nullptr,out,w,e);}, aclnnAdd, stream);
    else if (op == "Sub") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnSubGetWorkspaceSize(a,b,nullptr,out,w,e);}, aclnnSub, stream);
    else if (op == "Mul") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnMulGetWorkspaceSize(a,b,out,w,e);}, aclnnMul, stream);
    else if (op == "Div") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnDivGetWorkspaceSize(a,b,out,w,e);}, aclnnDiv, stream);
    else if (op == "Maximum") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnMaximumGetWorkspaceSize(a,b,out,w,e);}, aclnnMaximum, stream);
    else if (op == "Minimum") st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnMinimumGetWorkspaceSize(a,b,out,w,e);}, aclnnMinimum, stream);
    else if (op == "Softmax" || op == "SoftmaxV2") {
        int64_t axis = -1;
        if (attr) { auto it = attr->ints.find("axis"); if (it==attr->ints.end()) it = attr->ints.find("dim"); if (it != attr->ints.end()) axis = it->second; }
        st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnSoftmaxGetWorkspaceSize(a,axis,out,w,e);}, aclnnSoftmax, stream);
    }
    else if (op == "MatMul" || op == "MatMulV2" || op == "BatchMatMul")
        st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnMatmulGetWorkspaceSize(a,b,out,1,w,e);}, aclnnMatmul, stream);
    else if (op == "ReduceSum" || op == "ReduceSumD") {
        std::vector<int64_t> axes; bool keep = false;
        if (attr) { auto it = attr->listInts.find("axes"); if (it != attr->listInts.end()) axes = it->second;
                    auto kb = attr->bools.find("keep_dims"); if (kb != attr->bools.end()) keep = kb->second != 0; }
        if (axes.empty()) for (size_t i = 0; i < inD[0]->dims.size(); ++i) axes.push_back((int64_t)i);
        aclIntArray *dim = aclCreateIntArray(axes.data(), axes.size());
        st = two_stage([&](uint64_t*w,aclOpExecutor**e){ return aclnnReduceSumGetWorkspaceSize(a,dim,keep,outD[0]->dtype,out,w,e);}, aclnnReduceSum, stream);
        aclDestroyIntArray(dim);
    }
    #undef UN
    if (a) aclDestroyTensor(a); if (b) aclDestroyTensor(b); if (out) aclDestroyTensor(out);
    return st == ACLNN_SUCCESS ? ACL_SUCCESS : (aclError)st;
}
} // namespace

extern "C" {
aclError aclopExecuteV2(const char *opType, int numInputs, aclTensorDesc *inputDesc[], aclDataBuffer *inputs[],
                        int numOutputs, aclTensorDesc *outputDesc[], aclDataBuffer *outputs[], aclopAttr *attr, aclrtStream stream) {
    if (!opType) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    std::vector<void*> in(numInputs), outp(numOutputs);
    for (int i = 0; i < numInputs; ++i) in[i] = inputs ? aclGetDataBufferAddr(inputs[i]) : nullptr;
    for (int i = 0; i < numOutputs; ++i) outp[i] = outputs ? aclGetDataBufferAddr(outputs[i]) : nullptr;
    aclnnStatus st; @autoreleasepool { st = route(opType, numInputs, inputDesc, in.data(), numOutputs, outputDesc, outp.data(), attr, stream); }
    return (aclError)st;
}
aclError aclopCompileAndExecute(const char *opType,
    int numInputs, const aclTensorDesc *const inputDesc[], const aclDataBuffer *const inputs[],
    int numOutputs, const aclTensorDesc *const outputDesc[], aclDataBuffer *const outputs[],
    const aclopAttr *attr, aclopEngineType engineType, aclopCompileType compileFlag, const char *opPath, aclrtStream stream) {
    (void)engineType; (void)compileFlag; (void)opPath; if (!opType) return (aclError)ACLNN_ERR_PARAM_NULLPTR;
    std::vector<void*> in(numInputs), outp(numOutputs);
    for (int i = 0; i < numInputs; ++i) in[i] = inputs ? aclGetDataBufferAddr(inputs[i]) : nullptr;
    for (int i = 0; i < numOutputs; ++i) outp[i] = outputs ? aclGetDataBufferAddr(outputs[i]) : nullptr;
    aclnnStatus st; @autoreleasepool { st = route(opType, numInputs, inputDesc, in.data(), numOutputs, outputDesc, outp.data(), attr, stream); }
    return (aclError)st;
}
} // extern "C"
