#!/usr/bin/env bash
# Runs ON a RunPod pod (runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2204 image,
# /workspace = persistent volume). Idempotent: safe to re-run after a restart.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export HF_HOME=/workspace/hf
export PATH="/root/.local/bin:$PATH"

cd /workspace
mkdir -p /workspace/hf /workspace/logs

if ! command -v rsync >/dev/null || ! command -v tmux >/dev/null; then
  apt-get update -qq && apt-get install -y -qq rsync tmux >/dev/null
fi
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi

# vLLM in its own venv on the volume. NOTE: the volume is a FUSE mount, so the
# first import after a start takes ~3-5 min (thousands of small reads); it is a
# one-time cost per server start, not per request.
# Host driver is CUDA 12.8 (driver 570). PyPI's vllm 0.29.0 pulls torch cu130, which
# fails at CUDA init ("driver too old"). Use the cu129 variant wheel + torch cu129
# (CUDA minor-version compatibility covers a 12.9 runtime on a 12.8 driver).
VLLM_WHL="https://github.com/vllm-project/vllm/releases/download/v0.29.0/vllm-0.29.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
if [ ! -x /workspace/vllm-venv/bin/vllm ] || ! /workspace/vllm-venv/bin/python -c "import torch; assert '+cu129' in torch.__version__" 2>/dev/null; then
  [ -d /workspace/vllm-venv ] || uv venv /workspace/vllm-venv --python 3.12 >/dev/null
  # --reinstall-package torch: an already-installed cu130 torch of the same version
  # otherwise satisfies the pin and is silently kept.
  uv pip install --python /workspace/vllm-venv/bin/python --torch-backend=cu129 \
    --reinstall-package torch --reinstall-package torchvision --reinstall-package torchaudio "$VLLM_WHL"
fi
# inductor (torch.compile) shells out to ninja + a C++ compiler
uv pip install --python /workspace/vllm-venv/bin/python ninja >/dev/null
command -v g++ >/dev/null || apt-get install -y -qq build-essential >/dev/null
ln -sf /workspace/vllm-venv/bin/ninja /usr/local/bin/ninja
# real CUDA init check in a fresh process (parent-process is_available() can lie)
/workspace/vllm-venv/bin/python -c "import vllm, torch; torch.zeros(1).cuda(); print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda-init OK')"

# model weights (public, no token needed) -> persistent cache
/workspace/vllm-venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
p = snapshot_download("Qwen/Qwen3-8B", allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
print("weights at", p)
EOF

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo BOOTSTRAP_OK
