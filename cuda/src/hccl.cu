// HCCL → NCCL: control plane RootInfo = ncclUniqueId embedded directly (4108B holds 128B); data plane is 1:1.
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include "hccl/hccl.h"

static HcclResult from_nccl(ncclResult_t r) {
    switch (r) {
        case ncclSuccess: return HCCL_SUCCESS;
        case ncclInvalidArgument: case ncclInvalidUsage: return HCCL_E_PARA;
        case ncclSystemError: return HCCL_E_NETWORK;
        default: return HCCL_E_INTERNAL;
    }
}

static bool to_nccl_dtype(HcclDataType t, ncclDataType_t *o) {
    switch (t) {
        case HCCL_DATA_TYPE_INT8:   *o = ncclInt8; return true;
        case HCCL_DATA_TYPE_INT32:  *o = ncclInt32; return true;
        case HCCL_DATA_TYPE_FP16:   *o = ncclFloat16; return true;
        case HCCL_DATA_TYPE_FP32:   *o = ncclFloat32; return true;
        case HCCL_DATA_TYPE_INT64:  *o = ncclInt64; return true;
        case HCCL_DATA_TYPE_UINT64: *o = ncclUint64; return true;
        case HCCL_DATA_TYPE_UINT8:  *o = ncclUint8; return true;
        case HCCL_DATA_TYPE_UINT32: *o = ncclUint32; return true;
        case HCCL_DATA_TYPE_FP64:   *o = ncclFloat64; return true;
        case HCCL_DATA_TYPE_BFP16:  *o = ncclBfloat16; return true;
        default: return false;     // int16/int128/fp8 family has no NCCL equivalent
    }
}

static bool to_nccl_op(HcclReduceOp op, ncclRedOp_t *o) {
    switch (op) {
        case HCCL_REDUCE_SUM:  *o = ncclSum; return true;
        case HCCL_REDUCE_PROD: *o = ncclProd; return true;
        case HCCL_REDUCE_MAX:  *o = ncclMax; return true;
        case HCCL_REDUCE_MIN:  *o = ncclMin; return true;
        default: return false;
    }
}

extern "C" {

HcclResult HcclGetRootInfo(HcclRootInfo *rootInfo) {
    if (!rootInfo) return HCCL_E_PTR;
    static_assert(sizeof(ncclUniqueId) <= sizeof(rootInfo->internal), "uniqueId must fit RootInfo");
    return from_nccl(ncclGetUniqueId(reinterpret_cast<ncclUniqueId *>(rootInfo->internal)));
}

HcclResult HcclCommInitRootInfo(uint32_t nRanks, const HcclRootInfo *rootInfo, uint32_t rank, HcclComm *comm) {
    if (!rootInfo || !comm) return HCCL_E_PTR;
    ncclUniqueId id;
    memcpy(&id, rootInfo->internal, sizeof id);
    return from_nccl(ncclCommInitRank(reinterpret_cast<ncclComm_t *>(comm), (int)nRanks, id, (int)rank));
}

HcclResult HcclCommDestroy(HcclComm comm) { return from_nccl(ncclCommDestroy((ncclComm_t)comm)); }

// ranktable (cluster info JSON) → rank count: counts occurrences of "rank_id" (v1.0: one entry per device)
static int ranktable_nranks(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return -1; }
    char *buf = (char *)malloc(sz + 1); if (fread(buf, 1, sz, f) != (size_t)sz) { fclose(f); free(buf); return -1; }
    buf[sz] = 0; fclose(f);
    int n = 0; for (const char *p = buf; (p = strstr(p, "rank_id")); p += 7) ++n;
    free(buf);
    return n;
}

// HcclCommInitClusterInfo: parses the ranktable to obtain nRanks; distributes the uniqueId via an
//   out-of-band file (rank 0 writes, others poll), then calls ncclCommInitRank.
//   This allows Ascend distributed programs written against ranktable to run without modification
//   (the id file is copied across machines by the launch script).
//   The id file path is taken from env HCCL_ROOT_FILE; defaults to /tmp/hccl_cluster_root.bin.
HcclResult HcclCommInitClusterInfo(const char *clusterInfo, uint32_t rank, HcclComm *comm) {
    if (!clusterInfo || !comm) return HCCL_E_PTR;
    int nRanks = ranktable_nranks(clusterInfo);
    if (nRanks <= 0 || (int)rank >= nRanks) return HCCL_E_PARA;
    const char *idf = getenv("HCCL_ROOT_FILE"); if (!idf) idf = "/tmp/hccl_cluster_root.bin";
    ncclUniqueId id;
    if (rank == 0) {
        ncclResult_t r = ncclGetUniqueId(&id); if (r != ncclSuccess) return from_nccl(r);
        FILE *f = fopen(idf, "wb"); if (!f) return HCCL_E_NETWORK;
        fwrite(&id, sizeof id, 1, f); fclose(f);
    } else {
        bool got = false;
        for (int i = 0; i < 600 && !got; ++i) {
            FILE *f = fopen(idf, "rb");
            if (f) { if (fread(&id, sizeof id, 1, f) == 1) got = true; fclose(f); }
            if (!got) usleep(100 * 1000);
        }
        if (!got) return HCCL_E_NETWORK;
    }
    return from_nccl(ncclCommInitRank(reinterpret_cast<ncclComm_t *>(comm), nRanks, id, (int)rank));
}

HcclResult HcclAllReduce(void *sendBuf, void *recvBuf, uint64_t count, HcclDataType dataType,
                         HcclReduceOp op, HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt; ncclRedOp_t ro;
    if (!to_nccl_dtype(dataType, &dt) || !to_nccl_op(op, &ro)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclAllReduce(sendBuf, recvBuf, count, dt, ro, (ncclComm_t)comm, (cudaStream_t)stream));
}

HcclResult HcclAllGather(void *sendBuf, void *recvBuf, uint64_t sendCount, HcclDataType dataType,
                         HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt;
    if (!to_nccl_dtype(dataType, &dt)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclAllGather(sendBuf, recvBuf, sendCount, dt, (ncclComm_t)comm, (cudaStream_t)stream));
}

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
                             HcclReduceOp op, HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt; ncclRedOp_t ro;
    if (!to_nccl_dtype(dataType, &dt) || !to_nccl_op(op, &ro)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclReduceScatter(sendBuf, recvBuf, recvCount, dt, ro, (ncclComm_t)comm, (cudaStream_t)stream));
}

HcclResult HcclBroadcast(void *buf, uint64_t count, HcclDataType dataType, uint32_t root,
                         HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt;
    if (!to_nccl_dtype(dataType, &dt)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclBroadcast(buf, buf, count, dt, (int)root, (ncclComm_t)comm, (cudaStream_t)stream));
}

HcclResult HcclSend(void *sendBuf, uint64_t count, HcclDataType dataType, uint32_t destRank,
                    HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt;
    if (!to_nccl_dtype(dataType, &dt)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclSend(sendBuf, count, dt, (int)destRank, (ncclComm_t)comm, (cudaStream_t)stream));
}

HcclResult HcclRecv(void *recvBuf, uint64_t count, HcclDataType dataType, uint32_t srcRank,
                    HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t dt;
    if (!to_nccl_dtype(dataType, &dt)) return HCCL_E_NOT_SUPPORT;
    return from_nccl(ncclRecv(recvBuf, count, dt, (int)srcRank, (ncclComm_t)comm, (cudaStream_t)stream));
}

static int nccl_dtype_size(ncclDataType_t t) {
    switch (t) { case ncclInt8: case ncclUint8: return 1; case ncclFloat16: case ncclBfloat16: return 2;
        case ncclInt32: case ncclUint32: case ncclFloat32: return 4; default: return 8; }
}

// AllToAll: each rank sends one chunk to and receives one chunk from every rank (NCCL grouped send/recv)
HcclResult HcclAlltoAll(const void *sendBuf, uint64_t sendCount, HcclDataType sendType,
                        const void *recvBuf, uint64_t recvCount, HcclDataType recvType,
                        HcclComm comm, aclrtStreamH stream) {
    ncclDataType_t sdt, rdt;
    if (!to_nccl_dtype(sendType, &sdt) || !to_nccl_dtype(recvType, &rdt)) return HCCL_E_NOT_SUPPORT;
    int nranks = 0; ncclCommCount((ncclComm_t)comm, &nranks);
    int64_t ss = (int64_t)sendCount * nccl_dtype_size(sdt), rs = (int64_t)recvCount * nccl_dtype_size(rdt);
    const char *sp = (const char *)sendBuf; char *rp = (char *)recvBuf;
    ncclGroupStart();
    for (int r = 0; r < nranks; ++r) {
        ncclSend(sp + (int64_t)r * ss, sendCount, sdt, r, (ncclComm_t)comm, (cudaStream_t)stream);
        ncclRecv(rp + (int64_t)r * rs, recvCount, rdt, r, (ncclComm_t)comm, (cudaStream_t)stream);
    }
    return from_nccl(ncclGroupEnd());
}

HcclResult HcclGetRankSize(HcclComm comm, uint32_t *rankSize) {
    if (!rankSize) return HCCL_E_PTR;
    int n = 0; ncclResult_t r = ncclCommCount((ncclComm_t)comm, &n); *rankSize = (uint32_t)n; return from_nccl(r);
}
HcclResult HcclGetRankId(HcclComm comm, uint32_t *rank) {
    if (!rank) return HCCL_E_PTR;
    int r = 0; ncclResult_t s = ncclCommUserRank((ncclComm_t)comm, &r); *rank = (uint32_t)r; return from_nccl(s);
}

} // extern "C"
