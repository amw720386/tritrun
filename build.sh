#!/usr/bin/env bash
# Build tritrun/libtritrun.so (needs the CUDA 12.x toolkit on PATH; edit the -gencode list to taste).
set -euo pipefail
cd "$(dirname "$0")"
nvcc -O3 -std=c++17 --use_fast_math -shared -Xcompiler -fPIC \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 \
  -o tritrun/libtritrun.so src/engine.cu
