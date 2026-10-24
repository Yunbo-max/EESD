# EESD Gate 1: Falsifying an update, rather than rewarding another test

**Date:** 2026-10-24  
**Status:** CPU proof-of-concept and deterministic synthetic stress test. **No LLM gradients, no RL training, no benchmark claim.**  
**Parent:** [Gate 0 proposal](2026-10-24-eesd-update-admissibility-gate0.md); old lexical-EED history and prospective Shapley-EESD protocol are not superseded.

## Executive scientific decision

**Do not promote Gate 0 to the paper's main proposed method.** The public-test gradient-consensus heuristic is neither a guarantee of hidden correctness nor a sufficiently distinguishable standalone contribution. Existing work includes CodeLift (arXiv:2610.17224) on behavior-based test grouping; VERPO (arXiv:2609.06100) on locally reward-aligned gradients; *The Unlearnability Phenomenon in RLVR for Language Models* (ICML 2026) on cross-example gradient similarity; and *Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs* (arXiv:2609.03955) on counterexample tests. A possible next research question is whether **counterexamples targeted at a proposed learning update** provide value beyond equally expensive hard-test augmentation and reward recalibration.

## Counterexample 1: gradients agree; hidden test worsens

Consider three fully enumerated actions (candidate programs) sampled from a uniform categorical softmax policy. Their two public-test pass vectors are:

- candidate a: [1,1]
- candidate b: [0,1]
- candidate c: [0,0]

Suppose the hidden pass vector is [0,0,1]. The exact score gradients are the rows of I - 11ᵀ/3. Gate 0 accepts: leave-one-family-out gradient cosines are both 0.5. At an illustrative logit update with step size 0.3, the probability of the hidden-correct action c falls from 1/3 to 0.2986673711 (a **3.4666 percentage-point loss**). If the hidden-correct action were a instead, the same public evidence and same update would be beneficial. A public-only gate is observationally unable to distinguish these worlds.

The numerical counterexample is reproducible in scripts/audit_update_gate_counterexamples.py and requires no language model.

## Counterexample 2: iid versus deliberately inverted target distribution

Three independently seeded, **synthetic** 512-task categorical experiments (uniform 4-way policy, 4 public tests, 8 probe tests, 128 hidden tests per task, step 0.3):

| seed | Gate 0 accepted / 512 | harmful among accepted, IID | harmful among accepted, inverted target | passes Hoeffding LCB using 8 IID probes |
|---|---:|---:|---:|---:|
| 1701 | 193 | 3 / 193 | 189 / 193 | 4 / 512 |
| 1702 | 187 | 10 / 187 | 175 / 187 | 8 / 512 |
| 1703 | 191 | 11 / 191 | 180 / 191 | 2 / 512 |

The inverted distribution is constructed as q_hidden=1-q_public over candidate-specific Bernoulli success probabilities. This is a designed impossibility stressor, **not a measured coding-benchmark distribution shift**. Gate 0 is relatively favorable under the matching distribution but cannot diagnose the deliberately reversed target. The prior unnormalized version accepted 189 instead of 193 on seed 1701; the current version normalizes the score Gram and fixes scale-sensitive numerical tolerance. These two versions must not be mixed.

Running the new conservative audit with only eight IID probes leads to very few admissions, exposing its sample-complexity bottleneck. This is a **negative result on an elementary simulator**, not a claim that formal confidence bounds are universally unusable.

## A directional test score specific to the proposed update

With G frozen candidates, full-sequence rollout score vectors s_i = ∇log π(y_i|x), score Gram K_ij = ⟨s_i,s_j⟩, and a public-proposed direction d = Σ_i u_i s_i, define

    c_i = ((K u)_i - mean(K u)) / G
    z(t;d) = Σ_i c_i * pass(y_i, t)

The quantity z(t;d) equals the dot product between the proposed direction and the empirical probe-test policy-gradient estimator on the same frozen candidate bank. This is a **first-order finite-candidate proxy**, not the exact derivative of a general language model's expected hidden reward. It requires binary pass outcomes on test t obtained from a trusted oracle; syntactically executable but incorrect generated tests are not sound.

A **direction-targeted falsifier** may search a labeled probe bank for argmin_t z(t;d). Unlike simple test-disagreement ranking, two tests with the same number of passing candidates can have opposite signs of z because the proposed update boosts different candidates. This observation is implemented, not yet validated as a novel or useful training method. It overlaps with broad ideas in test generation and active verification.

## Adaptive counterexample search is not an independent confidence certificate

Split proposed probes into distinct sets **before** experiments:

1. Counterexample-search pool: may adapt to the frozen direction d and select the most negative observed z. Use it to falsify or identify defects; its adaptivity invalidates naive i.i.d. Hoeffding claims.
2. Independent audit pool: oracle-validated inputs sampled independently from a predeclared target-distribution proxy D, not used in direction construction or search. If each independent test yields z in [-B,B], where B=Σ_i max(c_i,0), the one-sided bound for the mean **conditional finite-candidate sensitivity** μ_D(d) is

    LCB_delta = mean(z_1,...,z_m) - B * sqrt(2 log(1/delta) / m).

Under the assumptions above, P(μ_D(d) >= LCB_delta | frozen candidates,d) >= 1-delta. This bound is classical Hoeffding, **not** a novel generalization theorem. It does not guarantee the true full-policy gradient, a successful finite learning step, real hidden-test generalization, or safety under arbitrary distribution shift. Duplication, dependence, post-hoc probe selection or wrong oracle outputs invalidate its interpretation. Distinct probe IDs are only a software check, not proof of independence.

A rough sample-complexity diagnostic: if μ_D/B = 0.2 and delta=0.05, positivity of this conservative lower bound requires more than 2log(20)/(0.2)^2 ≈ **150 independent probes** even when observed mean equals its expectation. Eight independent tests cannot justify the same confidence conclusion at such weak signal strength.

## Implemented code and verified state

- src/eesd/update_gate0.py: normalized Gram geometry; rejects nonfinite threshold inputs. Preserves raw gradient units in returned physical norms/gains and additionally returns normalized diagnostics.
- src/eesd/update_audit.py: direction-targeted oracle-probe score, worst counterexample and optional Hoeffding LCB, with explicit conditional interpretation.
- tests/test_update_gate0.py and tests/test_update_audit.py: **16 CPU tests passed locally** in a reconstructed minimal project (not full repository CI).
- scripts/audit_update_gate_counterexamples.py: deterministic uniform categorical experiment reproducing three-seed audit tables above.
- scripts/run_update_gate0.py: original create-once public Gate 0 JSONL runner, unchanged.

To reproduce in the source repository (Python environment with numpy and pytest):

    uv sync --extra test
    uv run pytest -q tests/test_update_gate0.py tests/test_update_audit.py
    uv run python scripts/audit_update_gate_counterexamples.py --seed 1701 --tasks 512
    uv run python scripts/audit_update_gate_counterexamples.py --seed 1702 --tasks 512
    uv run python scripts/audit_update_gate_counterexamples.py --seed 1703 --tasks 512

A connected GitHub write does not mean GitHub CI or real-model experiments have executed.

## Gate 2: prospective real-LLM experiment; no execution yet

**Data prerequisites.** Confirm that the repair dataset exposes a sound trusted reference implementation on train/development sources. If not, choose a coding TRAIN split with reliable oracle; do not borrow official EvalPlus/HumanEval/MBPP evaluation hidden tests as train-time probes. Separate all public, adaptive-search, independent-audit and benchmark-evaluation test IDs and execution hashes before seeing outcomes.

**Compute-limited pilot.** Qwen2.5-Coder-1.5B-Instruct, G=8, 128 source tasks, four public executions, fixed seed 1701. First measure whether exact full-sequence score-Gram computation is feasible (G score backward passes per source), and whether the Gate 0/1 signal is nondegenerate. Per-sample gradient computations in trainable LoRA space require a pinned model and adapter checkpoint; no simulated gradient can stand in for this. For reference-code problems, sample a bounded budget of independent and adversarially selected input tests with isolated sandbox execution.

**Ablations and matched budgets.** Compare public binary-GRPO, pass-rate, reproduced VeRPO, reproduced CodeLift, Gate 0, Gate 0 + random oracle probes, Gate 0 + maximum-disagreement oracle probes, Gate 0 + direction-targeted oracle probes, and optional independent-LCB gate. Match public+oracle execution counts, rollout response tokens, acceptance/update rate, actual KL and walltime. Use an equal-acceptance random gate as a direct control. Do not present first-order score improvements as downstream pass@1.

**Main falsification question.** At a fixed oracle-test budget and admission/parameter-update budget, does direction-targeted probing distinguish hidden-harmful updates better than random or disagreement-based probe selection? Report AUROC/AUPRC if nondegenerate, paired hidden directional utility, true per-task fix/regression after training, and all tasks skipped by the gate. If not better than controls, **park the branch**. If the LCB passes on <10% of tasks and cannot be calibrated at realistic budgets without violating its assumptions, keep it as a diagnostic rather than the main training objective.

## Related work and originality boundary

- VeRPO: https://arxiv.org/abs/2601.03525 (density-calibrated partial-pass RLVR).
- CodeLift: https://arxiv.org/abs/2610.17224 (behavior-based test grouping/dynamic curriculum).
- ACES: https://arxiv.org/abs/2604.03922 (test LOO-AUC consistency for code selection).
- VERPO: https://arxiv.org/abs/2609.06100 (evidence-modulated token credit and Fisher cost).
- Exploring Pass-Rate Reward: https://arxiv.org/abs/2605.02944 (gradient conflict under partial tests).
- The Unlearnability Phenomenon in RLVR for Language Models: https://proceedings.mlr.press/v306/chen26aq.html (ICML 2026 cross-example gradient similarity).
- Two-Stage Reinforcement Learning for Sound and Adversarial Test Generation in Code LLMs: https://arxiv.org/abs/2609.03955 (counterexample-based test generation).

**Research status:** the old effective-mass/shapley formulation has not been prospectively validated. This Gate 1 branch currently supplies a falsified public-consensus assumption, conditional mathematical diagnostic, and proposed real-model evaluation. It **does not** show that the new method improves coding agents.
