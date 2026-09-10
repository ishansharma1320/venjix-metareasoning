#!/usr/bin/env bash
# Runs ON the pod. Launches a frozen experiment set against the local vLLM in tmux
# session "batch". Idempotent: resume-by-config-hash skips complete conditions.
#   pod_batch_start.sh [workers] [spec] [out]
set -euo pipefail
export PATH="/root/.local/bin:$PATH"
cd /workspace/venjix-code
WORKERS="${1:-96}"
SPEC="${2:-experiments/exp-v2.json}"
OUT="${3:-runs/exp-v2}"
curl -sf http://127.0.0.1:8000/v1/models >/dev/null || { echo "vLLM is not serving on :8000"; exit 1; }
if tmux has-session -t batch 2>/dev/null; then echo "batch session already running"; exit 0; fi
tmux new-session -d -s batch "uv run python -m venjix.batch --set $SPEC --backend vllm --workers $WORKERS --out $OUT 2>&1 | tee -a /workspace/logs/batch.log"
echo "batch launched: $SPEC -> $OUT with $WORKERS workers; log: /workspace/logs/batch.log"
