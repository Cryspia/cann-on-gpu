/*
 * hccl.h — shim self-declaration (the public hccl/ header in this CANN beta only has types + partial control plane;
 * control plane + collective communication signatures are declared here per the official documentation).
 * Types/enums reuse the vendor hccl_types.h.
 */
#ifndef CANN_ON_GPU_HCCL_H
#define CANN_ON_GPU_HCCL_H

#include "hccl/hccl_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *aclrtStreamH;   // = aclrtStream (avoids pulling in acl header dependencies)

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo);
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm);
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm);
HcclResult HcclCommDestroy(HcclComm comm);

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStreamH stream);
HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStreamH stream);
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStreamH stream);
HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                         HcclComm comm, aclrtStreamH stream);
HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStreamH stream);
HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStreamH stream);
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                        const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStreamH stream);
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize);
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank);

#ifdef __cplusplus
}
#endif

#endif
