// HCCL -> single-rank stub (one Mac = one device). Built into libhccl.dylib (mirrors the CUDA backend's
// separate libhccl.so). On a single rank every collective degenerates to identity; with unified memory the
// send/recv buffers are host-addressable, so the copy is a plain memcpy. Multi-node is out of scope.
#include "hccl/hccl.h"
#include <cstring>
#include <cstdint>
#include <cstdlib>

namespace {
struct Comm { uint32_t rank = 0, size = 1; };
size_t dsize(HcclDataType t) {
    switch (t) {
        case HCCL_DATA_TYPE_INT8: case HCCL_DATA_TYPE_UINT8: case HCCL_DATA_TYPE_HIF8:
        case HCCL_DATA_TYPE_FP8E4M3: case HCCL_DATA_TYPE_FP8E5M2: case HCCL_DATA_TYPE_FP8E8M0: return 1;
        case HCCL_DATA_TYPE_INT16: case HCCL_DATA_TYPE_UINT16: case HCCL_DATA_TYPE_FP16: case HCCL_DATA_TYPE_BFP16: return 2;
        case HCCL_DATA_TYPE_INT32: case HCCL_DATA_TYPE_UINT32: case HCCL_DATA_TYPE_FP32: return 4;
        case HCCL_DATA_TYPE_INT64: case HCCL_DATA_TYPE_UINT64: case HCCL_DATA_TYPE_FP64: return 8;
        default: return 4;
    }
}
} // namespace

extern "C" {
HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo) { if (rootInfo) memset(rootInfo, 0, sizeof(*rootInfo)); return HCCL_SUCCESS; }
HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *, uint32_t rank, HcclComm *comm) {
    if (!comm) return HCCL_E_PARA; auto *c = new Comm(); c->rank = rank; c->size = nRanks ? nRanks : 1; *comm = (HcclComm)c; return HCCL_SUCCESS;
}
HcclResult HcclCommInitClusterInfo(const char *, uint32_t rank, HcclComm *comm) {
    if (!comm) return HCCL_E_PARA; auto *c = new Comm(); c->rank = rank; *comm = (HcclComm)c; return HCCL_SUCCESS;
}
HcclResult HcclCommDestroy(HcclComm comm) { delete (Comm *)comm; return HCCL_SUCCESS; }

HcclResult HcclAllReduce(void *s, void *r, uint64_t n, HcclDataType dt, HcclReduceOp, HcclComm, aclrtStreamH) {
    if (s && r && s != r) memcpy(r, s, n * dsize(dt)); return HCCL_SUCCESS;     // 1 rank: reduction = identity
}
HcclResult HcclAllGather(void *s, void *r, uint64_t n, HcclDataType dt, HcclComm, aclrtStreamH) {
    if (s && r && s != r) memcpy(r, s, n * dsize(dt)); return HCCL_SUCCESS;
}
HcclResult HcclReduceScatter(void *s, void *r, uint64_t n, HcclDataType dt, HcclReduceOp, HcclComm, aclrtStreamH) {
    if (s && r && s != r) memcpy(r, s, n * dsize(dt)); return HCCL_SUCCESS;
}
HcclResult HcclBroadcast(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStreamH) { return HCCL_SUCCESS; }  // root=self
HcclResult HcclSend(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStreamH) { return HCCL_SUCCESS; }
HcclResult HcclRecv(void *, uint64_t, HcclDataType, uint32_t, HcclComm, aclrtStreamH) { return HCCL_SUCCESS; }
HcclResult HcclAlltoAll(const void *s, uint64_t sc, HcclDataType st, const void *r, uint64_t, HcclDataType, HcclComm, aclrtStreamH) {
    if (s && r && s != r) memcpy((void *)r, s, sc * dsize(st)); return HCCL_SUCCESS;
}
HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize) { if (rankSize) *rankSize = comm ? ((Comm *)comm)->size : 1; return HCCL_SUCCESS; }
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank) { if (rank) *rank = comm ? ((Comm *)comm)->rank : 0; return HCCL_SUCCESS; }
} // extern "C"
