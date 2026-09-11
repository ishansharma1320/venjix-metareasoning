#!/usr/bin/env bash
# Runs ON the pod under nohup. exp-v3 M2: the policy gate on each candidate
# substrate, in order, cheapest first (PLAN-exp-v3.md §4). Bar declared in §4a.
# Assumes bootstrap.sh has run (venv at /opt/vllm-venv, uv on PATH).
#   gate_v3.sh "<model1> <model2> ..."
set -uo pipefail
export PATH="/root/.local/bin:/opt/vllm-venv/bin:$PATH"
export HF_HOME=/workspace/hf
MODELS="${1:-Qwen/Qwen3-8B Qwen/Qwen3-30B-A3B-Instruct-2507 Qwen/Qwen3-32B}"
BAR="${2:-0.255}"
phase() { echo "PHASE:$1:$2 $(date -u +%H:%M:%S)"; }

cd /workspace/venjix-code
uv sync -q >/dev/null 2>&1 || true
for MODEL in $MODELS; do
  TAG=$(echo "$MODEL" | tr '/' '-')
  # weights (public) -> persistent cache; sequential large files, fine on the volume
  HF_HUB_OFFLINE=0 /opt/vllm-venv/bin/python - "$MODEL" <<'EOF' > "/workspace/logs/download-$TAG.log" 2>&1
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.py"]))
EOF
  grep -q "models--" "/workspace/logs/download-$TAG.log" && phase "download-$TAG" OK || { phase "download-$TAG" FAIL; tail -5 "/workspace/logs/download-$TAG.log"; continue; }

  tmux kill-session -t vllm 2>/dev/null; sleep 3
  bash /workspace/scripts/vllm_start.sh "$MODEL" 4096 0.92 > "/workspace/logs/vllm_start-$TAG.log" 2>&1 \
    && phase "vllm-$TAG" OK || { phase "vllm-$TAG" FAIL; grep -E "ERROR|Error" /workspace/logs/vllm.log | grep -v api_utils | tail -4; continue; }

  uv run python scripts/policy_gate.py --set experiments/exp-v3.json --backend vllm --model "$MODEL" \
    --success-bar "$BAR" --workers 20 --out runs/policy-gate > "/workspace/logs/gate-$TAG.log" 2>&1
  grep -E "^steps|POLICY GATE|report:" "/workspace/logs/gate-$TAG.log"
  grep -q "POLICY GATE: GREEN" "/workspace/logs/gate-$TAG.log" && phase "gate-$TAG" GREEN || phase "gate-$TAG" RED
done
tmux kill-session -t vllm 2>/dev/null
phase gates DONE
