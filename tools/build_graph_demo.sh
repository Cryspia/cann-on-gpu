#!/usr/bin/env bash
# Build equivalent model container demo: cann_graph.h IR + serialization + executor, compared against direct op-by-op.
# Links only shim's libascendcl.so (same linking style as Ascend applications). Source ../env.sh first.
set -euo pipefail
cd "$(dirname "$0")"
: "${ACL_INCLUDE:?Run 'source ../env.sh' first}"
INC=../include; LIB=../cuda/lib
g++ -std=c++17 -O2 -I"$ACL_INCLUDE" -I"$INC" -I. graph_demo.cpp \
    -L"$LIB" -lascendcl -Wl,-rpath,"$(cd "$LIB" && pwd)" -o graph_demo
echo "built ./graph_demo"
./graph_demo
