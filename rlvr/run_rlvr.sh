#!/bin/bash
set -u
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/new/rlvr
REWARD="${RLVR_REWARD:-evidence}"
OUTPUT="${RLVR_OUTPUT:-results/run}"
TEMP="${RLVR_TEMP:-0.8}"
ADAPTER="${RLVR_ADAPTER:-results/sft/adapter}"
LOG="$ROOT/logs/run_${REWARD}.log"
mkdir -p "$ROOT/logs"
exec > "$LOG" 2>&1

echo "=== START $(date) reward=$REWARD temp=$TEMP adapter=$ADAPTER output=$OUTPUT ==="

find "$ROOT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
export PYTHONDONTWRITEBYTECODE=1

echo "=== install deps ==="
pip install --no-index --no-deps --find-links "$ROOT/../EESD/wheels" peft bitsandbytes

export HF_HOME=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT"
cd "$ROOT"

H=/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache
rm -rf "$OUTPUT"
python train_grpo.py \
  --model-config models/qwen25_7b.yaml \
  --problems "$H/hub/datasets--anjiangwei--CodeARC-Problems/snapshots/d2ee0ad485724668eabb893a965c681a9ce78031/problems.jsonl" \
  --invocations "$H/hub/datasets--anjiangwei--CodeARC-Invocations/snapshots/119512573bf96148cbb95e99cf7cdd21c06153e4/invocations.jsonl" \
  --output "$OUTPUT" \
  --reward "$REWARD" \
  --previous-adapter "$ADAPTER" \
  --temperature "$TEMP" \
  --n-sources 50 \
  --num-generations 64 \
  --max-new-tokens 256 \
  --steps 5 \
  --learning-rate 2e-5 \
  --beta 0.04 \
  --seed 0

echo "=== DONE $(date) ==="
ls -la "$OUTPUT/" 2>&1
