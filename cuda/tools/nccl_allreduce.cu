// Minimal two-node NCCL AllReduce validation: uniqueId distributed out-of-band via file (simulates HCCL ranktable->uniqueId translation).
// Usage: nccl_allreduce <rank> <nranks> <uniqueid_file>
//   rank0 generates uniqueId, writes it to file, then waits; rank1 polls until the file appears. Verifies the full sum: sum(rank+1).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CK(x) do { auto _r = (x); if (_r) { fprintf(stderr, "FAIL %s:%d code=%d\n", __FILE__, __LINE__, (int)_r); exit(1); } } while (0)

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s rank nranks idfile\n", argv[0]); return 1; }
    int rank = atoi(argv[1]), nranks = atoi(argv[2]);
    const char *idf = argv[3];

    ncclUniqueId id;
    if (rank == 0) {
        CK(ncclGetUniqueId(&id));
        FILE *f = fopen(idf, "wb");
        if (!f) { perror("open idfile"); return 1; }
        fwrite(&id, sizeof id, 1, f);
        fclose(f);
        printf("[rank0] uniqueId written: %s\n", idf);
    } else {
        for (int i = 0; i < 600; i++) {           // wait for rank0 to write the file (out-of-band distribution via scp in the launch script)
            FILE *f = fopen(idf, "rb");
            if (f && fread(&id, sizeof id, 1, f) == 1) { fclose(f); goto got; }
            if (f) fclose(f);
            usleep(100 * 1000);
        }
        fprintf(stderr, "[rank%d] timeout waiting for uniqueId\n", rank);
        return 1;
    got:;
    }

    CK(cudaSetDevice(0));
    ncclComm_t comm;
    CK(ncclCommInitRank(&comm, nranks, id, rank));
    printf("[rank%d] comm up\n", rank);

    const int64_t n = 1 << 20;
    float *d;
    CK(cudaMalloc(&d, n * sizeof(float)));
    {
        float v = (float)(rank + 1);
        float *h = (float *)malloc(n * sizeof(float));
        for (int64_t i = 0; i < n; i++) h[i] = v;
        CK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
        free(h);
    }
    CK(ncclAllReduce(d, d, n, ncclFloat, ncclSum, comm, 0));
    CK(cudaStreamSynchronize(0));

    float expect = nranks * (nranks + 1) / 2.0f;
    float *h = (float *)malloc(n * sizeof(float));
    CK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));
    int64_t bad = 0;
    for (int64_t i = 0; i < n; i++) if (h[i] != expect) bad++;
    printf("[rank%d] expect=%g bad=%ld -> %s\n", rank, expect, (long)bad, bad ? "FAIL" : "PASS");
    free(h);
    cudaFree(d);
    ncclCommDestroy(comm);
    return bad ? 1 : 0;
}
