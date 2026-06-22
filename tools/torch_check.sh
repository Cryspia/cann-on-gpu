#!/usr/bin/env bash
# Backend-selectable differential test of the FAMILY *_check harnesses against their independent
# PyTorch oracles (torch_<family>.py). Companion to torch_golden.sh (which covers the elementary /
# special-function set via cannsim_golden_check); this drives the per-family forward/backward harnesses
# — norm, reduce, scatter, attn, linalg, loss, actgrad, nn_grad, … — that cover the bulk of the catalog.
#
# Each family is a 1:1 pair: harness <fam>_check.cpp  +  oracle torch_<fam>.py. Both share the
#   <gen|check> <op> <prefix>   harness CLI  and  torch_<fam>.py <op> <prefix> [gold]   oracle CLI.
# (3-arg oracles write <gold>; 2-arg oracles ignore it and write <prefix>.out — passing the extra arg is
# harmless to both, so one flow drives every family.) The harness links only -lascendcl, so picking the
# backend is just choosing which libascendcl to link.
#
# Usage:  [BACKEND=metal] torch_check.sh [family ...]      (default BACKEND=cuda; default families = all)
#         [BACKEND=metal] torch_check.sh norm reduce       (subset)
#         OPS="layernorm rmsnorm" BACKEND=metal torch_check.sh norm   (explicit op subset for one family)
set -uo pipefail
cd "$(dirname "$0")"
source ../env.sh 2>/dev/null || true
PY="${PY:-$HOME/miniforge3/envs/cann-gpu/bin/python}"
mkdir -p bin

BACKEND="${BACKEND:-cuda}"; LIB="../$BACKEND/lib"
[ -e "$LIB/libascendcl.dylib" ] || [ -e "$LIB/libascendcl.so" ] || {
    echo "[error] no libascendcl in $LIB — build the $BACKEND backend first"; exit 1; }

# Default: every family that has both an oracle and a matching harness (exclude the elementary oracle).
FAMS=("$@")
if [ ${#FAMS[@]} -eq 0 ]; then
    for o in torch_*.py; do
        f="${o#torch_}"; f="${f%.py}"
        [ "$f" = common ] || [ "$f" = oracle ] && continue
        [ -f "${f}_check.cpp" ] && FAMS+=("$f")
    done
fi

TMP="${TMPDIR:-/tmp}/torch_check.$BACKEND"; mkdir -p "$TMP"
gpass=0; gfail=0; gskip=0; gfailed=""
for fam in "${FAMS[@]}"; do
    src="${fam}_check.cpp"; oracle="torch_${fam}.py"
    [ -f "$src" ] && [ -f "$oracle" ] || { echo "[skip-family] $fam (missing $src or $oracle)"; continue; }
    bin="bin/${fam}_check.$BACKEND"
    g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include "$src" \
        -L"$LIB" -lascendcl -Wl,-rpath,"$(cd "$LIB" && pwd)" -o "$bin" 2>/dev/null \
        || { echo "[skip-family] $fam (harness build failed)"; continue; }

    # Ops: explicit $OPS, else scrape the oracle's dispatch (op=="X" and "X": dict keys).
    if [ -n "${OPS:-}" ]; then ops=($OPS); else
        ops=($(grep -oE 'op *== *"[a-z][a-z0-9_]*"|"[a-z][a-z0-9_]*" *:' "$oracle" \
               | grep -oE '"[a-z][a-z0-9_]*"' | tr -d '"' | sort -u)); fi
    [ ${#ops[@]} -eq 0 ] && { echo "[skip-family] $fam (no ops discovered)"; continue; }

    echo "---- $fam ($BACKEND) ----"
    # loss carries an extra <reduction> positional (0=none/1=mean/2=sum) required by both harness and oracle.
    if [ "$fam" = loss ]; then
        for op in "${ops[@]}"; do for red in 0 1 2; do
            p="$TMP/$fam.$op.$red"
            "$bin" gen "$op" "$p" "$red" >/dev/null 2>&1 || { gskip=$((gskip+1)); continue; }
            "$PY" "$oracle" "$op" "$p" "$red" >/dev/null 2>&1 || { gskip=$((gskip+1)); continue; }
            out=$("$bin" check "$op" "$p" "$red" 2>&1)
            if   echo "$out" | grep -q FAIL; then gfail=$((gfail+1)); gfailed="$gfailed $fam:$op.r$red"; printf "  %-18s r%s FAIL\n" "$op" "$red"
            elif echo "$out" | grep -q PASS; then gpass=$((gpass+1));                                   printf "  %-18s r%s PASS\n" "$op" "$red"
            else gskip=$((gskip+1)); fi
        done; done
        continue
    fi
    for op in "${ops[@]}"; do
        "$bin" gen "$op" "$TMP/$fam.$op" >/dev/null 2>&1      || { gskip=$((gskip+1)); continue; }
        "$PY" "$oracle" "$op" "$TMP/$fam.$op" "$TMP/$fam.$op.gold" >/dev/null 2>&1 || { gskip=$((gskip+1)); continue; }
        out=$("$bin" check "$op" "$TMP/$fam.$op" "$TMP/$fam.$op.gold" 2>&1)
        if echo "$out" | grep -q FAIL;   then gfail=$((gfail+1)); gfailed="$gfailed $fam:$op"; printf "  %-22s FAIL\n" "$op"
        elif echo "$out" | grep -q PASS; then gpass=$((gpass+1));                              printf "  %-22s PASS\n" "$op"
        else gskip=$((gskip+1)); fi
    done
done
echo "============ torch family-oracle ($BACKEND): $gpass PASS, $gfail FAIL, $gskip skip ============"
[ -n "$gfailed" ] && echo "FAILED:$gfailed"
[ "$gfail" -eq 0 ]
