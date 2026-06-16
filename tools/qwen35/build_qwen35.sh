#!/usr/bin/env bash
# Build + run the Qwen3-Next ("Qwen3.5") hybrid-model adaptation cross-check on the cann-on-gpu shim.
#   1) export a standalone GatedDeltaNet layer and a tiny full Qwen3-Next from HF (random weights, no checkpoint needed)
#   2) compile the shim-backed C++ forwards
#   3) run both: per-stage / per-layer / logits comparison vs the HF reference
# Requires: cann-gpu conda env with torch + transformers>=5.12 (has Qwen3NextForCausalLM). CPU-only is sufficient.
set -uo pipefail
cd "$(dirname "$0")"
source ../../env.sh 2>/dev/null || true
PY="${PY:-$HOME/miniforge3/envs/cann-gpu/bin/python}"

echo "[lib] building backend (if needed)"
make -C ../../cuda >/dev/null || { echo "[FATAL] cuda lib build failed"; exit 1; }

echo "[export] GatedDeltaNet layer + tiny full Qwen3-Next (HF reference)"
"$PY" export_gdn.py      >/dev/null || { echo "[FATAL] export_gdn failed"; exit 1; }
"$PY" export_qwen35.py   >/dev/null || { echo "[FATAL] export_qwen35 failed"; exit 1; }

CXXFLAGS="-std=c++17 -O2 -I$ACL_INCLUDE -I../../include -I../../tests"
LDFLAGS="-L../../cuda/lib -lascendcl -Wl,-rpath,$(cd ../../cuda/lib && pwd)"
echo "[build] run_gdn / run_qwen35"
g++ $CXXFLAGS run_gdn.cpp    $LDFLAGS -o run_gdn    || exit 1
g++ $CXXFLAGS run_qwen35.cpp $LDFLAGS -o run_qwen35 || exit 1

echo "================ GatedDeltaNet linear-attention layer ================"
./run_gdn data_gdn
echo "================ full tiny Qwen3-Next forward ================"
./run_qwen35 data_full
