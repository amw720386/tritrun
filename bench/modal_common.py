"""Shared Modal pieces: the weights volume and the CUDA build/run image."""
import os, subprocess
import modal

VOL = modal.Volume.from_name("bonsai25-engine", create_if_missing=True)
REPO = "prism-ml/Ternary-Bonsai-2.5-27B-GGUF"
REV = "28a7093cc25bc151cc2b284c5570f12ce343fdcf"
GGUF = "Ternary-Bonsai-2.5-27B-PTQ1_0-mtp.gguf"
PRISM_TAG = "prism-b12178-03ec600"
PRISM_URL = ("https://github.com/PrismML-Eng/llama.cpp/releases/download/"
             f"{PRISM_TAG}/llama-{PRISM_TAG}-bin-linux-cuda-12.4-x64.tar.gz")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("curl", "git", "cmake", "ninja-build", "libgomp1", "libcurl4")
    .run_commands(f"mkdir -p /opt/prism && curl -sL {PRISM_URL} | tar xz -C /opt/prism --strip-components=1 || true",
                  "ls -R /opt/prism | head -60")
    .pip_install("huggingface_hub[hf_transfer]", "numpy", "requests")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

ARCHS = {"Tesla T4": "sm_75", "NVIDIA A10G": "sm_86", "NVIDIA A10": "sm_86", "NVIDIA L4": "sm_89"}
GPU = os.environ.get("GPU", "T4")                  # card for the single-card checks: GPU=L4 modal run ...

def gpu_arch():
    return ARCHS[subprocess.check_output("nvidia-smi --query-gpu=name --format=csv,noheader", shell=True, text=True).strip()]
