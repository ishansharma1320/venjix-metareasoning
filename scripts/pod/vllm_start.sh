#!/usr/bin/env bash
# Runs ON the pod. Starts vLLM on localhost only, inside tmux session "vllm".
# Serving config per docs/CLAUDE.md Amendment 9/11: bf16, seed 0, no reasoning
# parser (the client controls thinking via chat_template_kwargs; see llm.py).
set -euo pipefail
export HF_HOME=/workspace/hf
export HF_HUB_OFFLINE=1   # weights are cached; never block on hub round-trips
export PATH="/opt/vllm-venv/bin:$PATH"   # torch.compile/inductor needs `ninja` on PATH
if tmux has-session -t vllm 2>/dev/null; then echo "vllm session already running"; exit 0; fi
tmux new-session -d -s vllm "cd /workspace && \
  /opt/vllm-venv/bin/vllm serve Qwen/Qwen3-8B \
    --host 127.0.0.1 --port 8000 \
    --dtype bfloat16 --seed 0 \
    --max-model-len 8192 --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 \
    2>&1 | tee /workspace/logs/vllm.log"
echo "started; waiting for /v1/models ..."
for i in $(seq 1 180); do
  if curl -sf http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then echo VLLM_READY; exit 0; fi
  # fail fast if the server process died (tmux session gone) instead of waiting 15 min
  if ! tmux has-session -t vllm 2>/dev/null; then echo "vLLM process exited:"; tail -20 /workspace/logs/vllm.log; exit 1; fi
  sleep 5
done
echo "vLLM did not become ready in 15 min"; tail -40 /workspace/logs/vllm.log; exit 1
