#!/bin/bash
# EESD preflight on 1xH100: offline install deps -> Shapley scoring -> 1-step LoRA.
set -u
ROOT=/inspire/hdd/global_user/zhangjiaquan-253108540222/research/DiffRwkv/new/EESD
LOG="$ROOT/logs/qz/preflight-gpu.log"
exec > "$LOG" 2>&1

echo "=== START $(date) ==="

echo "=== 1. offline install peft/bitsandbytes ==="
pip install --no-index --no-deps --find-links "$ROOT/wheels" \
    peft bitsandbytes psutil tqdm safetensors
python -c "import torch, transformers, peft, bitsandbytes; print('torch', torch.__version__, '| tf', transformers.__version__, '| peft', peft.__version__, '| bnb', bitsandbytes.__version__)"
python -c "import torch; print('GPU', torch.cuda.get_device_name(0), 'devices', torch.cuda.device_count())"

echo "=== 2. HF cache reachability ==="
export HF_HOME="/inspire/hdd/global_user/zhangjiaquan-253108540222/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ls "$HF_HOME/hub/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/" || { echo "HF cache symlink NOT reachable"; exit 9; }

export PYTHONPATH="$ROOT/src"
cd "$ROOT"

echo "=== 3. Shapley scoring (2 trajectories) ==="
rm -rf preflight/shapley-out preflight/training
python scripts/score_eesd_shapley_relevance.py \
    --input preflight/corrections-shapley.jsonl \
    --model-config configs/models/qwen25_7b.yaml \
    --output preflight/shapley-out \
    --mode exact
echo "shapley exit=$?"

echo "=== 4. 1-step LoRA training (eesd_full) ==="
python scripts/run_eesd_weighted_sft.py \
    --input preflight/scored/scored-corrections.jsonl \
    --model-config configs/models/qwen25_7b.yaml \
    --rule eesd_full \
    --output preflight/training \
    --seed 1701 \
    --response-token-budget 64 \
    --max-steps 1 \
    --gradient-accumulation 1
echo "training exit=$?"

echo "=== DONE $(date) ==="
ls -la preflight/shapley-out/ preflight/training/ 2>&1
