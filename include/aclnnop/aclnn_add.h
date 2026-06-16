/* aclnn_add.h — shim self-declaration; signature matches CANN documentation: out = self + alpha * other */
#ifndef CANN_ON_GPU_ACLNN_ADD_H
#define CANN_ON_GPU_ACLNN_ADD_H

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

ACL_FUNC_VISIBILITY aclnnStatus aclnnAddGetWorkspaceSize(const aclTensor *self, const aclTensor *other,
                                                         const aclScalar *alpha, aclTensor *out,
                                                         uint64_t *workspaceSize, aclOpExecutor **executor);
ACL_FUNC_VISIBILITY aclnnStatus aclnnAdd(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
