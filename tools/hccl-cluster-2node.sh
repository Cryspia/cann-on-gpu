#!/usr/bin/env bash
# 2-node HcclCommInitClusterInfo: ranktable-driven nRanks, uniqueId distributed out-of-band via file.
# Reuses the hccl_collectives client (HCCL_RANKTABLE triggers ClusterInfo mode) and runs all collectives.
set -euo pipefail
cd "$(dirname "$0")"

REMOTE="${REMOTE:-node2}"
IFACE="${NCCL_SOCKET_IFNAME:-eth0}"
HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1}"
GID="${NCCL_IB_GID_INDEX:-3}"
RT=/tmp/hccl_ranktable.json
ID=/tmp/hccl_cluster_root.bin
BIN=./hccl_collectives
SHIM=../cuda/lib

[ -n "${NCCL_DIR:-}" ] || source ../env.sh
[ -x "$BIN" ] || { echo "Missing $BIN"; exit 1; }

# Generate 2-rank ranktable (v1.0 style; shim counts nRanks from rank_id entries).
# The 10.0.0.x addresses below are placeholders — replace with your two nodes' actual IPs.
cat > "$RT" <<'JSON'
{
  "server_count": "2",
  "server_list": [
    {"server_id": "10.0.0.1", "device": [{"device_id": "0", "device_ip": "10.0.0.1", "rank_id": "0"}]},
    {"server_id": "10.0.0.2", "device": [{"device_id": "0", "device_ip": "10.0.0.2", "rank_id": "1"}]}
  ],
  "status": "completed", "version": "1.0"
}
JSON

rm -f "$ID"; ssh "$REMOTE" "rm -f $ID"
scp -q "$BIN" "$SHIM/libascendcl.so" "$SHIM/libhccl.so" "$RT" "$REMOTE:/tmp/"
ssh "$REMOTE" "[ -f /tmp/libnccl.so.2 ]" || scp -q "$NCCL_DIR/lib/libnccl.so.2" "$REMOTE:/tmp/"
ssh "$REMOTE" "[ -f /tmp/libcudnn.so.9 ]" || scp -q "$CUDNN_DIR/lib/libcudnn.so.9" "$REMOTE:/tmp/"

ENVS="NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID HCCL_RANKTABLE=$RT HCCL_ROOT_FILE=$ID"
export LD_LIBRARY_PATH="$SHIM:$NCCL_DIR/lib:${LD_LIBRARY_PATH:-}"

env $ENVS "$BIN" 0 2 "$ID" & R0=$!
for i in $(seq 1 100); do [ -s "$ID" ] && break; sleep 0.1; done
[ -s "$ID" ] || { echo "rank0 did not produce uniqueId"; kill $R0; exit 1; }
scp -q "$ID" "$REMOTE:$ID"

ssh "$REMOTE" "cd /tmp && LD_LIBRARY_PATH=/tmp NCCL_SOCKET_IFNAME=$IFACE NCCL_IB_HCA=$HCA NCCL_IB_GID_INDEX=$GID HCCL_RANKTABLE=/tmp/hccl_ranktable.json HCCL_ROOT_FILE=$ID ./hccl_collectives 1 2 $ID" & R1=$!

ok=0
wait $R0 || ok=1
wait $R1 || ok=1
[ $ok -eq 0 ] && echo "HCCL-SHIM 2-NODE CLUSTERINFO(ranktable): PASSED" || echo "HCCL-SHIM 2-NODE CLUSTERINFO(ranktable): FAILED"
exit $ok
