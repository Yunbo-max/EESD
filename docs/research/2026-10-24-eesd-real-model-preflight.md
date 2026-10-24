# EESD Gate 2: actual model score-Gram preflight (NOT YET RUN)

**Date:** 2026-10-24  
**Scope:** the prospective update-admissibility research branch, not the original frozen final EESD experiment.  
**Verification status:** 23 CPU tests across Gate 0, Gate 1, score-Gram mathematics and bank validation passed on Python 3.13; Python compilation passed. **No GPU/Transformers/PEFT real-model execution has occurred.**

## Purpose

Replace invented gradient-Gram inputs in the categorical toy with real per-rollout score gradients of an LLM coding policy. Crucially:

- The policy must be the **same frozen checkpoint** that sampled the candidate patches.
- Score gradients are **full-response** sums of log probabilities of the actual sampled token IDs (including EOS if emitted); retokenizing code after generation changes the trajectory and is forbidden for the primary method.
- All base model weights are frozen by PEFT; the computed gradients live in the **trainable LoRA adapter subspace**, not all parameters of the pretrained model. This is a design choice requiring rank/target-layer ablation.
- The saved score Gram is K_ij = ⟨∇_LoRA log π(y_i|x), ∇_LoRA log π(y_j|x)⟩. The method never changes parameters while collecting it.
- Avoid transmitting actual high-dimensional score vectors: each candidate needs one teacher-forced backward pass; vectors are held in CPU memory temporarily to make the G×G Gram.

## Files

- src/eesd/score_gram_torch.py — conditional torch import, teacher-forced autoregressive shift, sequential per-candidate autograd and Gram.
- scripts/collect_update_score_gram.py — local-snapshot only CLI, strict no-hidden-label input schema and atomic exclusive output creation.
- tests/test_score_gram_torch.py — tiny causal model verifies exact score and Gram against direct autograd.
- tests/test_collect_update_score_gram.py — dry-run bank and privacy/split input checks.
- src/eesd/update_gate0.py — homogeneous Gram normalization prevents global gradient-unit scaling from affecting a gate decision.
- scripts/run_update_gate0.py — downstream geometry decision from the G×G Gram.

## Bank contract

The JSONL consumed by the collector must have exactly five fields:

    {"source_id":"demo","prompt_ids":[10,20],"candidate_ids":[[30,31],[32,33]],"before":[0,0],"after":[[1,0],[0,1]]}

This is purely illustrative; the example token IDs are not a real model tokenizer transcript. The pipeline expects a saved response-token ledger, not decoded strings. Source IDs must be unique; after rows must align with response order; hidden test labels are forbidden.

A dry run checks all rows without importing Transformers or loading a model:

    uv run python scripts/collect_update_score_gram.py \
      --input banks/locked_candidates.jsonl --dry-run

## Planned single-GPU collection

Start with **8 tasks and G=4** rather than committing to all 128 tasks. This small step measures memory, throughput and numerical rank. A later gate can expand to 128 sources × 8 candidates only if the measured cost is acceptable.

    uv sync --extra ml --extra test
    uv run python scripts/collect_update_score_gram.py \
      --input banks/locked_candidates.jsonl \
      --output banks/score_gram.jsonl \
      --model-path /absolute/path/to/pinned/local/qwen-coder-snapshot \
      --dtype bf16 --lora-rank 16 --seed 1701

For NVIDIA T4 with no native bf16 support, use --dtype fp16. If needed on supported hardware, append --nf4. If rollouts came from a previously trained adapter, pass --adapter-path /exact/pinned/adapter instead of initializing a new LoRA, or the score model differs from the sampling policy. The CLI requires installed local model snapshots; it does not perform network downloads.

The demo collector has **not** been run with real Transformers/PEFT; exact model-family compatibility, NF4 backward support and device placement still need to be checked in the first real GPU preflight. If the checkpoint was already quantized during generation, its quantization settings must match those used for scoring. Fresh LoRA is a diagnostic near the base policy, not a trained adapter.

To screen the created Gram bank:

    uv run python scripts/run_update_gate0.py \
      --input banks/score_gram.jsonl --output banks/gate0.jsonl

Output is create-once, recording decisions without hidden correctness. A separate trusted-oracle test collector is still needed for independent Gate 1 probes. Do not feed hidden benchmark evaluations into that collector.

## Cost and scientific controls

Per source, collecting G gradients entails G additional frozen-policy teacher-forced backward passes. A 128×8 bank means **1,024 backward passes**, plus sampling and test executions. Report walltime, peak GPU RAM, CPU memory, kernel conditioning and acceptance rates. If the full-response gradients are dominated by copied code and formatting, compare a **separately declared edited-token surrogate ablation**; do not conflate its gradients with unbiased full-response policy gradients.

Stop criteria before actual RL training:

1. Input provenance, model revision and sampling policy identity must be pinned.
2. If scores/Gram are frequently degenerate, numerical or semantic gate fails.
3. If 8-probe confidence audits reject >90% of cases, do not claim an efficient safety certificate.
4. If direction-targeted independent test probing cannot beat equally-budgeted random/disagreement tests in predicting hidden harm, park this idea.
5. Only then undertake the matched-update, matched-KL actual RL comparison against binary RLVR, pass-rate, CodeLift and VeRPO.

No additional performance results are claimed by writing this plan or collector.
