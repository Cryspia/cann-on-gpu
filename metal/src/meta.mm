// Implementation of the acl_meta.h contract: aclTensor / aclScalar / aclOpExecutor lifecycle and queries.
// Backend-agnostic (host metadata only) — kept identical to the CUDA backend's meta.cu.
#import "internal.h"
#include <cstring>

extern "C" {

aclTensor *aclCreateTensor(const int64_t *viewDims, uint64_t viewDimsNum, aclDataType dataType,
                           const int64_t *stride, int64_t offset, aclFormat format,
                           const int64_t *storageDims, uint64_t storageDimsNum, void *tensorData) {
    if ((viewDimsNum && !viewDims) || dtype_size(dataType) == 0) return nullptr;
    auto *t = new aclTensor();
    t->viewDims.assign(viewDims, viewDims + viewDimsNum);
    if (stride) t->strides.assign(stride, stride + viewDimsNum);
    else {                                  // default: contiguous layout
        t->strides.resize(viewDimsNum);
        int64_t s = 1;
        for (int i = (int)viewDimsNum - 1; i >= 0; --i) { t->strides[i] = s; s *= viewDims[i]; }
    }
    if (storageDims) t->storageDims.assign(storageDims, storageDims + storageDimsNum);
    t->offset = offset;
    t->dtype = dataType;
    t->format = format;
    t->data = tensorData;
    return t;
}

aclScalar *aclCreateScalar(void *value, aclDataType dataType) {
    if (!value) return nullptr;
    auto *s = new aclScalar();
    s->dtype = dataType;
    switch (dataType) {
        case ACL_FLOAT:  s->v = *(const float *)value; break;
        case ACL_DOUBLE: s->v = *(const double *)value; break;
        case ACL_INT32:  s->v = *(const int32_t *)value; break;
        case ACL_INT64:  s->v = *(const int64_t *)value; break;
        case ACL_INT8:   s->v = *(const int8_t *)value; break;
        case ACL_UINT8:  s->v = *(const uint8_t *)value; break;
        case ACL_BOOL:   s->v = *(const uint8_t *)value ? 1.0 : 0.0; break;
        case ACL_FLOAT16: {                 // IEEE fp16 -> double (scalar decoded on the host)
            uint16_t h = *(const uint16_t *)value;
            uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
            double d;
            if (exp == 0) d = man * 0x1p-24;
            else if (exp == 31) d = man ? __builtin_nan("") : __builtin_inf();
            else d = (1.0 + man * 0x1p-10) * __builtin_pow(2.0, (int)exp - 15);
            s->v = sign ? -d : d;
            break;
        }
        default: delete s; return nullptr;
    }
    return s;
}

aclIntArray *aclCreateIntArray(const int64_t *value, uint64_t size) {
    if (size && !value) return nullptr;
    auto *a = new aclIntArray();
    a->v.assign(value, value + size);
    return a;
}
aclnnStatus aclDestroyIntArray(const aclIntArray *array) { delete array; return ACLNN_SUCCESS; }

aclTensorList *aclCreateTensorList(const aclTensor *const *value, uint64_t size) {
    if (size && !value) return nullptr;
    auto *l = new aclTensorList();
    for (uint64_t i = 0; i < size; ++i) l->v.push_back(value[i]);
    return l;
}
aclnnStatus aclDestroyTensorList(const aclTensorList *array) { delete array; return ACLNN_SUCCESS; }
aclnnStatus aclGetTensorListSize(const aclTensorList *tensorList, uint64_t *size) {
    if (!tensorList || !size) return ACLNN_ERR_PARAM_NULLPTR;
    *size = (uint64_t)tensorList->v.size(); return ACLNN_SUCCESS;
}

aclnnStatus aclDestroyTensor(const aclTensor *tensor) { delete tensor; return ACLNN_SUCCESS; }
aclnnStatus aclDestroyScalar(const aclScalar *scalar) { delete scalar; return ACLNN_SUCCESS; }
aclnnStatus aclDestroyAclOpExecutor(aclOpExecutor *executor) {
    if (executor) for (auto *t : executor->owned) delete t;
    delete executor; return ACLNN_SUCCESS;
}

aclTensor *aclGetTensorListElement(const aclTensorList *tensorList, uint64_t index) {
    if (!tensorList || index >= tensorList->v.size()) return nullptr;
    return const_cast<aclTensor *>(tensorList->v[index]);
}

aclnnStatus aclGetViewShape(const aclTensor *t, int64_t **dims, uint64_t *num) {
    if (!t || !dims || !num) return ACLNN_ERR_PARAM_NULLPTR;
    *dims = const_cast<int64_t *>(t->viewDims.data());
    *num = t->viewDims.size();
    return ACLNN_SUCCESS;
}
aclnnStatus aclGetDataType(const aclTensor *t, aclDataType *dt) {
    if (!t || !dt) return ACLNN_ERR_PARAM_NULLPTR;
    *dt = t->dtype; return ACLNN_SUCCESS;
}
aclnnStatus aclGetFormat(const aclTensor *t, aclFormat *f) {
    if (!t || !f) return ACLNN_ERR_PARAM_NULLPTR;
    *f = t->format; return ACLNN_SUCCESS;
}
aclnnStatus aclGetRawTensorAddr(const aclTensor *t, void **addr) {
    if (!t || !addr) return ACLNN_ERR_PARAM_NULLPTR;
    *addr = t->data; return ACLNN_SUCCESS;
}
aclnnStatus aclSetRawTensorAddr(aclTensor *t, void *addr) {
    if (!t) return ACLNN_ERR_PARAM_NULLPTR;
    t->data = addr; return ACLNN_SUCCESS;
}

} // extern "C"
