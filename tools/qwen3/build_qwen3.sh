#!/usr/bin/env bash
# End-to-end real open-source small-model demo: export Qwen3-0.6B -> compile shim forward harness -> run end-to-end cross-check.
# Usage: bash build_qwen3.sh [model_id] [prompt] [--hf]
#   Model weights download from ModelScope by default; pass --hf (or set QWEN_USE_HF=1) to download from HuggingFace.
#   First run downloads model weights and exports them to data/; skips export if data/ already exists (delete data/ to force re-export).
#   Requires: cann-gpu conda env with torch + transformers + safetensors (CPU-only is sufficient). ModelScope mode also needs the `modelscope` package (auto-installed below if missing).
set -uo pipefail
cd "$(dirname "$0")"
source ../../env.sh 2>/dev/null || true
PY="${PY:-$HOME/miniforge3/envs/cann-gpu/bin/python}"

# Separate positional args from the --hf flag
USE_HF=0; POS=()
for a in "$@"; do if [ "$a" = "--hf" ]; then USE_HF=1; else POS+=("$a"); fi; done
[ "${QWEN_USE_HF:-0}" = "1" ] && USE_HF=1
MODEL="${POS[0]:-Qwen/Qwen3-0.6B}"
PROMPT="${POS[1]:-The capital of France is}"

if [ ! -f data/config.txt ]; then
    if [ "$USE_HF" = "1" ]; then
        echo "[export] downloading + exporting $MODEL from HuggingFace (slow on first run)"
        EXTRA="--hf"
    else
        echo "[export] downloading + exporting $MODEL from ModelScope (slow on first run)"
        EXTRA=""
        if ! "$PY" -c "import modelscope" 2>/dev/null; then
            echo "[deps] installing modelscope into the conda env"
            "$PY" -m pip install -q modelscope || { echo "[FATAL] failed to install modelscope; rerun with --hf to use HuggingFace"; exit 1; }
        fi
    fi
    HF_HUB_DISABLE_PROGRESS_BARS=1 "$PY" export_qwen3.py "$MODEL" data "$PROMPT" $EXTRA || exit 1
else
    echo "[export] data/ already exists, skipping (delete data/ to force re-export)"
fi

echo "[build] compiling shim forward harness"
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../../include run_qwen3.cpp \
    -L../../cuda/lib -lascendcl -Wl,-rpath,"$(cd ../../cuda/lib && pwd)" -o run_qwen3 || exit 1

echo "[run] shim forward vs HF reference"
./run_qwen3 data
