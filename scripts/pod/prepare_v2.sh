#!/usr/bin/env bash
# Runs ON the pod under nohup. Full exp-v2 preparation, gates included, with
# PHASE markers the laptop-side watcher greps for. Stops at the first failure.
# Order: bootstrap -> vLLM -> constrained-decoding check -> world-model gate
#        -> policy gate (Amendment 11). NO experiment-set condition is run here.
set -uo pipefail
export PATH="/root/.local/bin:/workspace/vllm-venv/bin:$PATH"
export HF_HOME=/workspace/hf
cd /workspace
phase() { echo "PHASE:$1:$2 $(date -u +%H:%M:%S)"; }

bash /workspace/scripts/bootstrap.sh > /workspace/logs/bootstrap.log 2>&1 \
  && grep -q BOOTSTRAP_OK /workspace/logs/bootstrap.log \
  && phase bootstrap OK || { phase bootstrap FAIL; tail -15 /workspace/logs/bootstrap.log; exit 1; }
ln -sf /workspace/vllm-venv/bin/ninja /usr/local/bin/ninja

bash /workspace/scripts/vllm_start.sh > /workspace/logs/vllm_start.log 2>&1 \
  && phase vllm OK || { phase vllm FAIL; grep -E "ERROR|Error" /workspace/logs/vllm.log | grep -v api_utils | tail -6; exit 1; }

/workspace/vllm-venv/bin/python /workspace/scripts/regex_check.py > /workspace/logs/regex_check.log 2>&1
grep -q "^structured_outputs -> ('200'" /workspace/logs/regex_check.log \
  && phase regex OK || { phase regex FAIL; cat /workspace/logs/regex_check.log; exit 1; }

cd /workspace/venjix-code
uv sync -q >/dev/null 2>&1 || true
uv run python scripts/calibrate_world_model.py --backend vllm --model Qwen/Qwen3-8B --workers 64 --out runs/calibration \
  > /workspace/logs/gate_world.log 2>&1
grep -E "calm-state misprediction rate|VERDICT" /workspace/logs/gate_world.log
grep -q "VERDICT: GREEN" /workspace/logs/gate_world.log \
  && phase worldgate OK || { phase worldgate FAIL; tail -8 /workspace/logs/gate_world.log; exit 1; }

uv run python scripts/policy_gate.py --set experiments/exp-v2.json --backend vllm --workers 20 --out runs/policy-gate \
  > /workspace/logs/gate_policy.log 2>&1
grep -E "^steps|POLICY GATE|report:" /workspace/logs/gate_policy.log
grep -q "POLICY GATE: GREEN" /workspace/logs/gate_policy.log \
  && phase policygate OK || { phase policygate RED; exit 2; }
phase prepare DONE
