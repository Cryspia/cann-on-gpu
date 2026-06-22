// Shape / gather kernels (MSL): Permute (strided gather → contiguous) and Embedding (row gather).
// Permute/Embedding move elements by value, so they're keyed by element byte-size (b4/b2/b8), dtype-agnostic.
#include <metal_stdlib>
using namespace metal;

struct PermMeta { uint n; uint ndim; uint odims[8]; uint istr[8]; };   // istr[i] = input stride for OUTPUT axis i
struct EmbMeta  { uint n; uint H; };                                   // n = rows*H

template <typename T>
kernel void permute_k(device const T *in [[buffer(0)]], device T *out [[buffer(1)]],
                      constant PermMeta &m [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint rem = gid, off = 0;
    for (int d = int(m.ndim) - 1; d >= 0; --d) { uint id = rem % m.odims[d]; rem /= m.odims[d]; off += id * m.istr[d]; }
    out[gid] = in[off];
}
template [[host_name("perm_b4")]] kernel void permute_k<uint>(device const uint *, device uint *, constant PermMeta &, uint);
template [[host_name("perm_b2")]] kernel void permute_k<ushort>(device const ushort *, device ushort *, constant PermMeta &, uint);
template [[host_name("perm_b8")]] kernel void permute_k<ulong>(device const ulong *, device ulong *, constant PermMeta &, uint);

template <typename T, typename I>
kernel void embed_k(device const T *w [[buffer(0)]], device const I *ids [[buffer(1)]], device T *out [[buffer(2)]],
                    constant EmbMeta &m [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    if (gid >= m.n) return;
    uint row = gid / m.H, col = gid % m.H;
    uint id = (uint)ids[row];
    out[gid] = w[id * m.H + col];
}
template [[host_name("embed_b4_i64")]] kernel void embed_k<uint, long>(device const uint *, device const long *, device uint *, constant EmbMeta &, uint);
template [[host_name("embed_b4_i32")]] kernel void embed_k<uint, int>(device const uint *, device const int *, device uint *, constant EmbMeta &, uint);
template [[host_name("embed_b2_i64")]] kernel void embed_k<ushort, long>(device const ushort *, device const long *, device ushort *, constant EmbMeta &, uint);
template [[host_name("embed_b2_i32")]] kernel void embed_k<ushort, int>(device const ushort *, device const int *, device ushort *, constant EmbMeta &, uint);
