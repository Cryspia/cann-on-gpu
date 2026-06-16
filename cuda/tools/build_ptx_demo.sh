#!/usr/bin/env bash
# Build the runtime PTX modular demo. Requires system CUDA 13 toolkit (nvrtc matching the driver) + libcuda.
set -euo pipefail
cd "$(dirname "$0")"
CUDA="${CUDA_TARGETS:-/usr/local/cuda-13.0/targets/sbsa-linux}"
g++ -std=c++17 -I"$CUDA/include" ptx_module_demo.cpp \
    -L/usr/lib/aarch64-linux-gnu -lcuda "$CUDA/lib/libnvrtc.so.13" \
    -Wl,-rpath,"$CUDA/lib" -o ptx_module_demo
echo "built ./ptx_module_demo"
./ptx_module_demo
