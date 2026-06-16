// Two-machine full collective communication end-to-end verification (pure ACL + HCCL -> shim -> NCCL/RoCE).
// Covers AllReduce/AllGather/ReduceScatter/Broadcast/Send-Recv/AllToAll.
// Usage: hccl_collectives <rank> <nranks> <rootinfo_file>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "hccl/hccl.h"

#define CK(x) do { int _r=(int)(x); if(_r){fprintf(stderr,"FAIL %s:%d code=%d\n",__FILE__,__LINE__,_r);exit(1);} } while(0)
static int g_rank, g_nranks, g_bad = 0;
static HcclComm g_comm; static aclrtStream g_stream;

static void *dev(int64_t bytes) { void *d; CK(aclrtMalloc(&d, bytes, ACL_MEM_MALLOC_HUGE_FIRST)); return d; }
static void up(void *d, const std::vector<float> &h) { CK(aclrtMemcpy(d, h.size()*4, h.data(), h.size()*4, ACL_MEMCPY_HOST_TO_DEVICE)); }
static void down(std::vector<float> &h, void *d) { CK(aclrtMemcpy(h.data(), h.size()*4, d, h.size()*4, ACL_MEMCPY_DEVICE_TO_HOST)); }
static void check(const char *name, long bad) {
    if (bad) g_bad++;
    printf("[rank%d] %-14s %s\n", g_rank, name, bad ? "FAIL" : "PASS");
}

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s rank nranks idfile\n", argv[0]); return 1; }
    g_rank = atoi(argv[1]); g_nranks = atoi(argv[2]); const char *idf = argv[3];
    CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&g_stream));
    const char *ranktable = getenv("HCCL_RANKTABLE");
    if (ranktable) {            // ranktable (ClusterInfo) mode; nRanks is parsed from the ranktable
        CK(HcclCommInitClusterInfo(ranktable, g_rank, &g_comm));
        printf("[rank%d] ClusterInfo init via ranktable=%s\n", g_rank, ranktable);
    } else {                    // RootInfo mode
        HcclRootInfo root;
        if (g_rank == 0) { CK(HcclGetRootInfo(&root)); FILE *f = fopen(idf, "wb"); fwrite(&root, sizeof root, 1, f); fclose(f); }
        else { for (int i = 0; i < 600; i++) { FILE *f = fopen(idf, "rb"); if (f && fread(&root, sizeof root, 1, f) == 1) { fclose(f); goto got; } if (f) fclose(f); usleep(100000); } return 1; got:; }
        CK(HcclCommInitRootInfo(g_nranks, &root, g_rank, &g_comm));
    }
    { uint32_t rs = 0, ri = 0; CK(HcclGetRankSize(g_comm, &rs)); CK(HcclGetRankId(g_comm, &ri));
      if ((int)rs != g_nranks || (int)ri != g_rank) g_bad++; }

    const int64_t n = 4096;
    float sumAll = g_nranks * (g_nranks + 1) / 2.0f;

    // AllReduce SUM
    { std::vector<float> h(n, g_rank + 1); void *d = dev(n*4); up(d, h);
      CK(HcclAllReduce(d, d, n, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(h, d);
      long bad = 0; for (auto v : h) if (v != sumAll) bad++; check("AllReduce", bad); aclrtFree(d); }

    // AllGather：recv[i*n+k] = i+1
    { std::vector<float> s(n, g_rank + 1), r(n * g_nranks); void *ds = dev(n*4), *dr = dev(n*g_nranks*4); up(ds, s);
      CK(HcclAllGather(ds, dr, n, HCCL_DATA_TYPE_FP32, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(r, dr);
      long bad = 0; for (int i = 0; i < g_nranks; i++) for (int64_t k = 0; k < n; k++) if (r[i*n+k] != i+1) bad++;
      check("AllGather", bad); aclrtFree(ds); aclrtFree(dr); }

    // ReduceScatter SUM: send all (rank+1), each rank receives chunk = sum = sumAll
    { std::vector<float> s(n * g_nranks, g_rank + 1), r(n); void *ds = dev(n*g_nranks*4), *dr = dev(n*4); up(ds, s);
      CK(HcclReduceScatter(ds, dr, n, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(r, dr);
      long bad = 0; for (auto v : r) if (v != sumAll) bad++; check("ReduceScatter", bad); aclrtFree(ds); aclrtFree(dr); }

    // Broadcast root0: all ranks receive root0's value 1
    { std::vector<float> h(n, g_rank + 1); void *d = dev(n*4); up(d, h);
      CK(HcclBroadcast(d, n, HCCL_DATA_TYPE_FP32, 0, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(h, d);
      long bad = 0; for (auto v : h) if (v != 1.0f) bad++; check("Broadcast", bad); aclrtFree(d); }

    // Send/Recv: rank0 -> rank1 (value 7)
    { void *d = dev(n*4);
      if (g_rank == 0) { std::vector<float> h(n, 7.0f); up(d, h); CK(HcclSend(d, n, HCCL_DATA_TYPE_FP32, 1, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); check("Send", 0); }
      else if (g_rank == 1) { std::vector<float> h(n, 0); up(d, h); CK(HcclRecv(d, n, HCCL_DATA_TYPE_FP32, 0, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(h, d);
        long bad = 0; for (auto v : h) if (v != 7.0f) bad++; check("Recv", bad); }
      else check("Send/Recv", 0);
      aclrtFree(d); }

    // AllToAll: send[c*n+k] = rank*10+c (sent to rank c); recv[s*n+k] = s*10+rank
    { std::vector<float> s(n * g_nranks), r(n * g_nranks);
      for (int c = 0; c < g_nranks; c++) for (int64_t k = 0; k < n; k++) s[c*n+k] = g_rank * 10 + c;
      void *ds = dev(n*g_nranks*4), *dr = dev(n*g_nranks*4); up(ds, s);
      CK(HcclAlltoAll(ds, n, HCCL_DATA_TYPE_FP32, dr, n, HCCL_DATA_TYPE_FP32, g_comm, g_stream)); CK(aclrtSynchronizeStream(g_stream)); down(r, dr);
      long bad = 0; for (int sr = 0; sr < g_nranks; sr++) for (int64_t k = 0; k < n; k++) if (r[sr*n+k] != sr*10 + g_rank) bad++;
      check("AllToAll", bad); aclrtFree(ds); aclrtFree(dr); }

    CK(HcclCommDestroy(g_comm)); CK(aclrtDestroyStream(g_stream)); CK(aclrtResetDevice(0)); CK(aclFinalize());
    printf("[rank%d] == %s ==\n", g_rank, g_bad ? "SOME FAILED" : "ALL PASSED");
    return g_bad ? 1 : 0;
}
