#!/bin/bash
set -u
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/new/rlvr
LOG="$ROOT/logs/diagnose.log"
mkdir -p "$ROOT/logs"
exec > "$LOG" 2>&1

pip install --no-index --no-deps --find-links "$ROOT/../EESD/wheels" peft bitsandbytes

export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$ROOT"
cd "$ROOT"

H=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
python diagnose.py \
  --model-config models/qwen25_7b.yaml \
  --problems "$H/hub/datasets--anjiangwei--CodeARC-Problems/snapshots/d2ee0ad485724668eabb893a965c681a9ce78031/problems.jsonl" \
  --invocations "$H/hub/datasets--anjiangwei--CodeARC-Invocations/snapshots/119512573bf96148cbb95e99cf7cdd21c06153e4/invocations.jsonl" \
  --output results/diagnose \
  --n-sources 3 \
  --num-generations 8 \
  --seed 0

echo "=== DONE $(date) ==="
