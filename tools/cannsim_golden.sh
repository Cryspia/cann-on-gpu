#!/usr/bin/env bash
# OPTIONAL real-Ascend second opinion. The default/primary verification is the fast local torch oracle
# (tools/torch_golden.sh) — it covers every op this golden loop does, in seconds, with no upstream dependency.
# This cannsim path is slow (~40-85s/op) and needs cann-api-explorer cloned with EXPLORER_DIR set; use it only
# to cross-check shim numerics against real Ascend (e.g. regression on a suspected Ascend-specific quirk).
#
# explorer->cannsim golden multi-op comparison loop:
#   1) cann-on-gpu generates deterministic fp32 inputs for each op (per-op shapes)
#   2) cann-api-explorer units read those inputs on cannsim card-free simulation,
#      compute the Ascend golden output and dump it
#   3) cann-on-gpu runs the corresponding shim aclnn op on the same inputs and
#      compares against the Ascend golden using normalized error
# Usage: cannsim_golden.sh [op ...]      no args runs the default op set
#        JOBS=4 cannsim_golden.sh ...    run multiple unit cannsim simulations in parallel (default 1)
#   Each op takes ~40-85s to build+simulate on cannsim; parallelism significantly reduces wall time.
set -uo pipefail
cd "$(dirname "$0")"
source ../env.sh 2>/dev/null || true

EXPLORER="${EXPLORER_DIR:-$(cd .. && pwd)/third_party/cann-api-explorer}"
if [ ! -d "$EXPLORER" ]; then
    echo "[ERR] cann-api-explorer not found: $EXPLORER" >&2
    echo "      This optional real-Ascend check needs the upstream explorer (hooks already upstreamed)." >&2
    echo "      For the fast default verification use tools/torch_golden.sh instead (no clone needed)." >&2
    echo "      To run cannsim anyway: git clone https://github.com/Cryspia/cann-api-explorer.git <dir> && EXPLORER_DIR=<dir> $0" >&2
    exit 1
fi
BIN=./cannsim_golden_check
JOBS="${JOBS:-1}"
TMP="${TMPDIR:-/tmp}/cannsim_golden"; mkdir -p "$TMP"

# op -> "subdirectory unit_name" (under examples/ascendc/<subdir>/<unit>; main.cpp must have a GOLDEN hook)
unit_of(){ case "$1" in
    add|sub|mul|div|relu|exp|sqrt|abs|neg) echo "vector $1";;
    rsqrt|reciprocal|max|min|reducesum)    echo "vector $1";;
    silu|gelu|sigmoid|tanh|erf|log|sin|cos|softmax|layernorm|rmsnorm) echo "highlevel $1";;
    # Note: logsoftmax skipped — the explorer unit's LogSoftMax uses log10 (not natural log, see its main.cpp comment), which differs from the standard shim ln by ln10≈2.303, making comparison meaningless
    acos|asin|atan|cosh|sinh|tan|ceil|floor|round|trunc|sign|erfc|frac|lgamma) echo "highlevel $1";;
    acosh|asinh|atanh|log2|log10|digamma)  echo "highlevel $1";;   # new Track-A units (upstream 172-unit expansion)
    rint)                                  echo "highlevel $1";;   # round-half-to-even == aclnnRound (rintf)
    ln)                                    echo "vector $1";;      # natural log == aclnnLog
    sum|mean|reduceprod)                   echo "highlevel $1";;   # full-axis reductions
    reducemax|reducemin)                   echo "vector $1";;
    shiftleft|shiftright)                  echo "vector $1";;      # int32 bit shift (a<<b / a>>b)
    is_finite|is_inf|is_nan)               echo "highlevel $1";;   # fp32 in, uint8 bool out (inf/nan injected)
    adds|subs|muls|divs|maxs|mins|leakyrelu|axpy) echo "vector $1";;   # scalar-elementwise (baked scalar 2.0 / 0.1)
    ascendquant|quantize)                  echo "highlevel $1";;   # fp32 in, int8 out, per-tensor scale/offset baked
    power|hypot|fmod)                      echo "highlevel $1";;
    sort|topk|deepnorm)                    echo "highlevel $1";;
    # Note: groupnorm skipped — the explorer unit's golden is always 0 for any input (only trivially correct
    #     for all-equal constant input; does not faithfully compute groupnorm). The shim is verified correct
    #     by CPU reference (out[0]=-1.4967); the explorer golden is unusable.
    matmul)                                echo "cube $1";;
    *) echo "";; esac; }

# op -> GOLDEN_IN_* environment variable mapping to export (based on <prefix> files)
export_inputs(){  # $1=op  $2=prefix
    local op="$1" pre="$2"
    unset GOLDEN_IN_X GOLDEN_IN_Y GOLDEN_IN_G GOLDEN_IN_B
    export GOLDEN_IN_X="$pre.x"
    case "$op" in
        add|sub|mul|div|matmul|max|min|power|hypot|fmod|shiftleft|shiftright) export GOLDEN_IN_Y="$pre.y";;
    esac
    case "$op" in
        layernorm|groupnorm) export GOLDEN_IN_G="$pre.g"; export GOLDEN_IN_B="$pre.b";;
        rmsnorm)   export GOLDEN_IN_G="$pre.g";;
        deepnorm)  export GOLDEN_IN_Y="$pre.y"; export GOLDEN_IN_G="$pre.g"; export GOLDEN_IN_B="$pre.b";;
    esac
}

OPS=("$@"); [ ${#OPS[@]} -eq 0 ] && OPS=(add mul relu exp silu gelu sigmoid tanh softmax layernorm rmsnorm matmul)

echo "[build] compiling comparison program"
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../include cannsim_golden_check.cpp \
    -L../cuda/lib -lascendcl -Wl,-rpath,"$(cd ../cuda/lib && pwd)" -o "$BIN" || exit 1

# ---- Phase 1: generate inputs for each op ----
declare -A UDIR
VALID=()
for op in "${OPS[@]}"; do
    read -r sub unit <<<"$(unit_of "$op")"
    [ -z "$unit" ] && { echo "[skip] unknown op: $op"; continue; }
    d="$EXPLORER/examples/ascendc/$sub/$unit"
    [ -d "$d" ] || { echo "[skip] unit directory not found: $d"; continue; }
    grep -q "GOLDEN_IN_X" "$d/main.cpp" 2>/dev/null || {
        echo "[skip] $op: $unit/main.cpp missing GOLDEN hook (needs upstreaming)"; continue; }
    "$BIN" gen "$op" "$TMP/$op" || { echo "[skip] $op: gen failed"; continue; }
    UDIR[$op]="$d"; VALID+=("$op")
done

# ---- Phase 2: cannsim simulation produces Ascend golden (parallelizable) ----
run_one_golden(){  # $1=op
    local op="$1" d="${UDIR[$op]}" out="$TMP/${op}_golden.bin"
    rm -f "$out"; export_inputs "$op" "$TMP/$op"; export GOLDEN_OUT="$out"
    bash "$EXPLORER/harness/run_one.sh" "$d" >"$TMP/${op}.simlog" 2>&1 || true
}
echo "[sim] explorer computing Ascend golden on cannsim (JOBS=$JOBS, ~several minutes per op)"
pids=(); running=0
for op in "${VALID[@]}"; do
    run_one_golden "$op" &
    pids+=($!); running=$((running+1))
    if [ "$running" -ge "$JOBS" ]; then wait -n 2>/dev/null || wait "${pids[0]}"; running=$((running-1)); fi
done
wait

# ---- Phase 3: shim comparison ----
PASS=0; FAIL=0
for op in "${VALID[@]}"; do
    out="$TMP/${op}_golden.bin"
    echo "================ $op ================"
    if [ ! -s "$out" ]; then
        echo "[FAIL] $op Ascend golden not produced (check $TMP/${op}.simlog and ${UDIR[$op]}/RESULT.md)"; FAIL=$((FAIL+1)); continue
    fi
    if "$BIN" check "$op" "$TMP/$op" "$out"; then PASS=$((PASS+1)); else FAIL=$((FAIL+1)); fi
done
echo "================ golden comparison summary: $PASS PASS, $FAIL FAIL ================"
[ "$FAIL" -eq 0 ]
