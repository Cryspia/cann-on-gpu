#!/usr/bin/env bash
# Decode-loop full-loop CUDA Graph capture + replay + per-token latency measurement.
# Usage: bash build_decode_graph.sh [n_gen]
set -uo pipefail
cd "$(dirname "$0")"
source ../../env.sh 2>/dev/null || true
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I../../include decode_graph.cpp \
    -L../../cuda/lib -lascendcl -Wl,-rpath,"$(cd ../../cuda/lib && pwd)" -o decode_graph || exit 1
./decode_graph "${1:-32}"
