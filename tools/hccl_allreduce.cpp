// End-to-end two-machine verification: client uses only AscendCL + HCCL API (no CUDA/NCCL headers),
// running shim libascendcl.so + libhccl.so. RootInfo distributed out-of-band via file (equivalent to ranktable exchange).
// Usage: hccl_allreduce <rank> <nranks> <rootinfo_file>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include "acl/acl.h"
#include "hccl/hccl.h"

#define CK(x) do { int _r = (int)(x); if (_r) { fprintf(stderr, "FAIL %s:%d code=%d\n", __FILE__, __LINE__, _r); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s rank nranks idfile\n", argv[0]); return 1; }
    int rank = atoi(argv[1]), nranks = atoi(argv[2]);
    const char *idf = argv[3];

    CK(aclInit(nullptr));
    CK(aclrtSetDevice(0));
    aclrtStream stream;
    CK(aclrtCreateStream(&stream));

    HcclRootInfo root;
    if (rank == 0) {
        CK(HcclGetRootInfo(&root));
        FILE *f = fopen(idf, "wb"); if (!f) { perror("idfile"); return 1; }
        fwrite(&root, sizeof root, 1, f); fclose(f);
        printf("[rank0] RootInfo written\n");
    } else {
        for (int i = 0; i < 600; i++) {
            FILE *f = fopen(idf, "rb");
            if (f && fread(&root, sizeof root, 1, f) == 1) { fclose(f); goto got; }
            if (f) fclose(f);
            usleep(100 * 1000);
        }
        fprintf(stderr, "[rank%d] RootInfo timeout\n", rank); return 1;
    got:;
    }

    HcclComm comm;
    CK(HcclCommInitRootInfo(nranks, &root, rank, &comm));
    printf("[rank%d] comm up\n", rank);

    const uint64_t n = 1 << 20;
    size_t bytes = n * sizeof(float);
    std::vector<float> h(n, (float)(rank + 1));
    void *d;
    CK(aclrtMalloc(&d, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMemcpy(d, bytes, h.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE));
    CK(HcclAllReduce(d, d, n, HCCL_DATA_TYPE_FP32, HCCL_REDUCE_SUM, comm, stream));
    CK(aclrtSynchronizeStream(stream));
    CK(aclrtMemcpy(h.data(), bytes, d, bytes, ACL_MEMCPY_DEVICE_TO_HOST));

    float expect = nranks * (nranks + 1) / 2.0f;
    long bad = 0;
    for (auto v : h) if (v != expect) bad++;
    printf("[rank%d] HCCL AllReduce expect=%g bad=%ld -> %s\n", rank, expect, bad, bad ? "FAIL" : "PASS");

    CK(HcclCommDestroy(comm));
    aclrtFree(d);
    CK(aclrtDestroyStream(stream));
    CK(aclrtResetDevice(0));
    CK(aclFinalize());
    return bad ? 1 : 0;
}
