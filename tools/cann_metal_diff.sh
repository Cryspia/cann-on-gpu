#!/usr/bin/env bash
# 对拍 (differential test): run the SAME backend-agnostic ACL test binaries on the local Metal backend
# and on a peer's CUDA backend, and show their self-verification summaries side by side.
#
# The tests embed CPU/torch-style oracles and pass/fail internally, so "both PASS" means both backends
# satisfy the identical tolerance contract — cross-architecture agreement without needing Ascend hardware.
# For ops that lack a local oracle, this harness is the seam to compare Metal vs the verified CUDA reference.
#
# Usage:  tools/cann_metal_diff.sh [test_name ...]      (default: the currently-green Metal suite)
#   Set PEER=<host> and PEER_DIR=<path to this repo on that host> to also run the CUDA reference side.
set -uo pipefail
cd "$(dirname "$0")/.."
PEER="${PEER:-}"          # CUDA reference host (e.g. via ssh config); unset → local Metal results only
PEER_DIR="${PEER_DIR:-}"  # path to this repo on $PEER
TESTS="${*:-test_add test_compare test_runtime test_harness}"

echo "===================== local (Metal) ====================="
if [ -f env.sh ]; then source env.sh; fi
make -C metal >/dev/null 2>&1 || { echo "[FATAL] metal backend build failed"; exit 1; }
for t in $TESTS; do
    make -C tests BACKEND=../metal "bin/$t" >/dev/null 2>&1 \
        && printf "%-20s %s\n" "$t" "$(./tests/bin/$t 2>/dev/null | tail -1)" \
        || printf "%-20s %s\n" "$t" "[build/run failed — op(s) not implemented yet on Metal]"
done

if [ -z "$PEER" ] || [ -z "$PEER_DIR" ]; then
    echo "[info] set PEER=<host> and PEER_DIR=<repo path on peer> to also run the CUDA reference side"
    exit 0
fi
echo "===================== peer (CUDA) ====================="
ssh -o BatchMode=yes -o ConnectTimeout=15 "$PEER" "bash -lc '
    cd ${PEER_DIR} || exit 1
    source env.sh 2>/dev/null
    make -C tests >/dev/null 2>&1
    for t in ${TESTS}; do
        printf \"%-20s %s\n\" \"\$t\" \"\$(./tests/bin/\$t 2>/dev/null | tail -1)\"
    done
'" || echo "[WARN] peer unreachable — local Metal results stand on their own oracle"
