#!/usr/bin/env bash
# 2-node mc2 MatmulAllReduce over RoCE/HCCL. Dual-NIC aggregation for 200G: list both RoCE devices.
# Override the defaults below for your cluster (peer host, socket iface, RoCE HCAs, GID index):
#   REMOTE=<peer-host> NCCL_SOCKET_IFNAME=<iface> NCCL_IB_HCA=<hca0,hca1> NCCL_IB_GID_INDEX=<n> ./mc2-2node.sh
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-node2}"
IFACE="${NCCL_SOCKET_IFNAME:-eth0}"
HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"   # list both RoCE NICs → aggregate bandwidth
GID="${NCCL_IB_GID_INDEX:-3}"
ID=/tmp/mc2_root.bin
BIN=./mc2_2node
SHIM=../cuda/lib

[ -n "${NCCL_DIR:-}" ] || source ../env.sh
[ -x "$BIN" ] || { echo "Missing $BIN: compile mc2_2node.cpp first"; exit 1; }

rm -f "$ID"; ssh "$REMOTE" "rm -f $ID"
scp -q "$BIN" "$SHIM/libascendcl.so" "$SHIM/libhccl.so" "$REMOTE:/tmp/"
ssh "$REMOTE" "[ -f /tmp/libnccl.so.2 ]" || scp -q "$NCCL_DIR/lib/libnccl.so.2" "$REMOTE:/tmp/"
ssh "$REMOTE" "[ -f /tmp/libcudnn.so.9 ]" || scp -q "$CUDNN_DIR/lib/libcudnn.so.9" "$REMOTE:/tmp/"

ENVS="NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID"
export LD_LIBRARY_PATH="$SHIM:$NCCL_DIR/lib:${LD_LIBRARY_PATH:-}"

env $ENVS "$BIN" 0 2 "$ID" & R0=$!
for i in $(seq 1 100); do [ -s "$ID" ] && break; sleep 0.1; done
[ -s "$ID" ] || { echo "rank0 did not produce RootInfo"; kill $R0; exit 1; }
scp -q "$ID" "$REMOTE:$ID"

ssh "$REMOTE" "cd /tmp && LD_LIBRARY_PATH=/tmp $ENVS ./mc2_2node 1 2 $ID" & R1=$!

ok=0
wait $R0 || ok=1
wait $R1 || ok=1
[ $ok -eq 0 ] && echo "MC2 2-NODE MatmulAllReduce: PASSED" || echo "MC2 2-NODE MatmulAllReduce: FAILED"
exit $ok
