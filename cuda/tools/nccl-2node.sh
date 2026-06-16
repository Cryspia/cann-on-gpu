#!/usr/bin/env bash
# 2-node NCCL AllReduce verification: rank0 on this node, rank1 via ssh to node2.
# uniqueId distributed out-of-band via file + scp (mirrors HCCL ranktable->uniqueId control-plane translation).
# Source env.sh before running, or just run ./nccl-2node.sh directly.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-node2}"
IFACE="${NCCL_SOCKET_IFNAME:-eth0}"           # bootstrap via management NIC
HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"          # two RDMA NICs (aggregation affects performance only)
GID="${NCCL_IB_GID_INDEX:-3}"
ID=/tmp/nccl_uid.bin
BIN=./nccl_allreduce

[ -x "$BIN" ] || { echo "Missing $BIN: source ../../env.sh && nvcc -cudart static ... to build it"; exit 1; }
[ -n "${NCCL_DIR:-}" ] || source ../../env.sh
NCCL_LIB="$NCCL_DIR/lib/libnccl.so.2"
[ -f "$NCCL_LIB" ] || { echo "Cannot find $NCCL_LIB"; exit 1; }
export LD_LIBRARY_PATH="$NCCL_DIR/lib:${LD_LIBRARY_PATH:-}"

rm -f "$ID"; ssh "$REMOTE" "rm -f $ID"
scp -q "$BIN" "$REMOTE:/tmp/nccl_allreduce"
ssh "$REMOTE" "[ -f /tmp/libnccl.so.2 ]" || scp -q "$NCCL_LIB" "$REMOTE:/tmp/libnccl.so.2"

ENVS="NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID"

env $ENVS "$BIN" 0 2 "$ID" & R0=$!
for i in $(seq 1 100); do [ -s "$ID" ] && break; sleep 0.1; done
[ -s "$ID" ] || { echo "rank0 did not produce uniqueId"; kill $R0; exit 1; }
scp -q "$ID" "$REMOTE:$ID"

ssh "$REMOTE" "cd /tmp && LD_LIBRARY_PATH=/tmp $ENVS ./nccl_allreduce 1 2 $ID" & R1=$!

ok=0
wait $R0 || ok=1
wait $R1 || ok=1
[ $ok -eq 0 ] && echo "NCCL 2-NODE ALLREDUCE: PASSED" || echo "NCCL 2-NODE ALLREDUCE: FAILED"
exit $ok
