#!/usr/bin/env bash
# Build tritrun/libtritrun.so (needs the CUDA 12.x toolkit on PATH; edit the -gencode list to taste).
# RTX 50-series (sm_120) code is added when nvcc is 12.8 or newer.
set -euo pipefail
cd "$(dirname "$0")"
EXTRA=""
read -r MAJ MIN <<< "$(nvcc --version | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1 \2/p')"
if [ "$MAJ" -gt 12 ] || { [ "$MAJ" -eq 12 ] && [ "$MIN" -ge 8 ]; }; then EXTRA="-gencode arch=compute_120,code=sm_120"; fi
nvcc -O3 -std=c++17 --use_fast_math -shared -Xcompiler -fPIC \
  -gencode arch=compute_75,code=sm_75 \
  -gencode arch=compute_86,code=sm_86 \
  -gencode arch=compute_89,code=sm_89 \
  -gencode arch=compute_90,code=sm_90 \
  -gencode arch=compute_90,code=compute_90 $EXTRA \
  -o tritrun/libtritrun.so src/engine.cu
