#!/usr/bin/env bash
# Differential test of the shim against an INDEPENDENT PyTorch oracle (CPU), reusing the
# cannsim_golden_check gen/check harness but producing the reference from torch_oracle.py
# instead of the Ascend cannsim. This is implementation-independent from our shim, so it catches
# "buggy-vs-buggy" errors that a hand-written CPU test reference (sharing the shim's formula) misses
# — the class of bug that hid the digamma asymptotic-sign error until real-Ascend golden exposed it.
#
# Unlike cannsim_golden.sh it needs no card simulation or upstream explorer unit — it runs in seconds
# and covers every op torch_oracle.py + cannsim_golden_check both know.
#
# Usage: torch_golden.sh [op ...]   (default: the special-function / elementary-math risk set)
set -uo pipefail
cd "$(dirname "$0")"
source ../env.sh 2>/dev/null || true
PY="${PY:-$HOME/miniforge3/envs/cann-gpu/bin/python}"
BIN=./cannsim_golden_check
TMP="${TMPDIR:-/tmp}/torch_golden"; mkdir -p "$TMP"

OPS=("$@")
[ ${#OPS[@]} -eq 0 ] && OPS=(sinc erfinv expm1 log1p exp2 lgamma digamma erfc gelu silu sigmoid tanh \
    asinh acosh atanh sinh cosh tan frac round rint trunc sign rsqrt reciprocal log2 log10 erf \
    relu exp sqrt abs neg log sin cos asin acos atan ceil floor add sub mul div max min power hypot fmod)

echo "[build] compiling comparison program"
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include cannsim_golden_check.cpp \
    -L../cuda/lib -lascendcl -Wl,-rpath,"$(cd ../cuda/lib && pwd)" -o "$BIN" || exit 1

pass=0; fail=0; skip=0; failed=""
for op in "${OPS[@]}"; do
    "$BIN" gen "$op" "$TMP/$op" >/dev/null 2>&1 || { echo "[skip] $op: harness has no gen path"; skip=$((skip+1)); continue; }
    "$PY" torch_oracle.py "$op" "$TMP/$op" "$TMP/$op.gold" 2>/dev/null || { echo "[skip] $op: no torch reference"; skip=$((skip+1)); continue; }
    line=$("$BIN" check "$op" "$TMP/$op" "$TMP/$op.gold" 2>&1 | tail -1)
    echo "$line"
    if echo "$line" | grep -q PASS; then pass=$((pass+1)); else fail=$((fail+1)); failed="$failed $op"; fi
done
echo "================ torch-oracle: $pass PASS, $fail FAIL, $skip skip ================"
[ -n "$failed" ] && echo "FAILED:$failed"
[ "$fail" -eq 0 ]
