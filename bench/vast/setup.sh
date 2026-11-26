#!/usr/bin/env bash
# Prepare a rented CUDA 12.8+ machine (e.g. vast.ai, pytorch/pytorch:*-cuda12.8-*-devel image) to run the bench
# scripts without Modal: weights in /w, engine source in /src, fork binaries (CUDA 12.8 build) in /opt/prism.
# Run from the repository root: bash bench/vast/setup.sh
set -euo pipefail
ROOT=$(pwd)
apt-get update -qq && apt-get install -y -qq curl libgomp1 libcurl4 > /dev/null
pip install -q "huggingface_hub[hf_transfer]" numpy requests pyarrow
eval "$(python - <<'EOF'
import sys; sys.path.insert(0, "bench/vast"); sys.path.insert(0, "bench")
import modal_common as m
print(f'REPO={m.REPO} REV={m.REV} GGUF={m.GGUF} TAG={m.PRISM_TAG}')
EOF
)"
mkdir -p /w /opt/prism
ln -sfn "$ROOT/src" /src
URL="https://github.com/PrismML-Eng/llama.cpp/releases/download/$TAG/llama-$TAG-bin-linux-cuda-12.8-x64.tar.gz"
curl -sL "$URL" | tar xz -C /opt/prism --strip-components=1
HF_HUB_ENABLE_HF_TRANSFER=1 python - <<EOF
from huggingface_hub import hf_hub_download
for f in ("$GGUF", "Ternary-Bonsai-2.5-27B-PTQ1_0.gguf"):
    hf_hub_download("$REPO", f, revision="$REV", local_dir="/w")
EOF
ls -la /w; find /opt/prism -name llama-server -type f
nvcc --version | tail -2; nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader
