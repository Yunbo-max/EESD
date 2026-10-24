# EESD: Update Admissibility Gate 0 (research hypothesis)

**Date:** 2026-10-24  
**Status:** prospective; CPU invariants implemented; no real-model outcome or training result.  
**Main branch:** unchanged. This research branch does not supersede the frozen Shapley–EESD protocol or relabel the historical lexical-EED results.

## Why change the research question?

EESD asks when a coding agent should learn from an execution-guided correction. Its recorded lexical-relevance experiment has 4 positive, 1 tie, and 7 negative model-domain cells. The reported +5.4 pp DeepSeek/CodeARC gain belongs to that older pipeline, not to prospective Shapley-EESD or this gate.

The earlier RLVR redesign proposed the scalar discount λ = E(p)/n, where E(p) = 1 / Σp_j², alongside weighted test rewards and a backbone-dependent KL anchor. This is insufficient as the paper's central mechanism:

1. Equal relevance across four duplicated tests gives E(p)=4 and λ=1, although these may reflect a single independent functionality. High effective mass is not equivalent to verifier reliability.
2. A common positive scalar multiplying all group-relative advantages cannot fix a wrongly ordered reward ranking; it changes scale but not direction.
3. Under within-source mean centering, replacing repaired pass Y_ij with the signed transition Y_ij - b_j (original pass b_j) does not change per-test advantage, because b_j cancels.

VeRPO (arXiv:2601.03525) already handles dynamically weighted partial success. CodeLift (arXiv:2610.17224) explicitly handles test multiplicity, progress–regression imbalances and behavior-based grouping. ACES (arXiv:2604.03922) estimates discriminatory test weights with leave-one-out consistency. VERPO (arXiv:2609.06100) handles evidence-conditioned policy credit. Exploring Pass-Rate Reward (arXiv:2605.02944) identifies gradient conflicts from partial-pass rewards. RobustTests (Findings EMNLP 2026, ACL anthology 2026.findings-emnlp.1114) addresses test synthesis and redundancy. None of this proves that the gate below is new or superior.

## Proposed observable: can independent behavior groups support the same local update?

For one source, generate G corrected programs under one frozen policy. Record T public-test bits for the original program b_j and each corrected program Y_ij. Public tests with identical (original bit, G candidate bits) are placed in the same **observed-behavior group** c. This grouping is a borrowed CodeLift-like diagnostic; it **does not establish semantic independence**.

For every group c, form a candidate-level signed transition reward averaged within that group and its within-source centered, standard-deviation-floored advantage A_ic. Compute score gradients s_i = ∇θ log πθ(y_i|x) at the same frozen checkpoint. The associated family objective's empirical policy-gradient estimator is:

\[
g_c=\frac{1}{G}\sum_i A_{ic}s_i.
\]

The before-patch subtraction cancels under group centering; b_j instead determines progress/preservation labels and group provenance. Full-sequence score gradients define the policy-gradient primary. Gradients masked to edited tokens are a **surrogate-objective ablation**, not the unbiased gradient of the original full-sequence sampling policy.

We need only the **G×G** rollout gradient Gram matrix K_il = ⟨s_i,s_l⟩. The family Gram is:

\[
H_{cd}=\langle g_c,g_d\rangle=\frac{1}{G^2}A_{\cdot c}^{\top}K A_{\cdot d}.
\]

Find a minimum-norm convex combination of family gradients (the established MGDA operation):

\[
\alpha^*=\arg\min_{\alpha\in\Delta^{C}}\frac12\alpha^\top H\alpha,\qquad
d=\sum_c\alpha_c^*g_c=\sum_i u_i s_i,\qquad
u=A\alpha^*/G.
\]

The code returns rollout weights u for the candidate direction. It does not train a network.

### First-order statement (not a generalization theorem)

If d is the minimum-norm point of the observed gradients' convex hull, projection optimality gives ⟨g_c,d⟩≥‖d‖² for each included family c. A nonzero d is thus simultaneous first-order ascent for those **observed empirical family objectives**. The property is known from multiobjective optimization, not a novel mathematical theorem.

Checking this same in-sample condition as an acceptance gate would be circular. Accordingly, for each c, build d_-c without c and compute the leave-one-family-out alignment cosine:

\[
q_c=\frac{\langle g_c,d_{-c}\rangle}{\|g_c\|\|d_{-c}\|}.
\]

The initial diagnostic accepts only when at least two nonzero behavior families are present, the full d is nonzero, the dual solver gap is small, and every q_c > 0. Otherwise the method abstains and returns zero rollout weights. The same candidate bank is reused, so these folds are **not independent samples** and q_c is **not** a probabilistic reliability certificate.

### Impossibility boundary

Construct worlds W+ and W− with identical public outcomes, identical sampled corrections and identical model score gradients, but opposite unseen-test labels. Any decision algorithm using only the public observables must act identically in both worlds. Therefore no such public-only gate can guarantee hidden-test correctness without distributional assumptions or genuinely independent evaluations.

## Gate 0: completed software checks

Files:
- src/eesd/update_gate0.py: CPU implementation using G×G score Gram; abstains on absent or conflicting evidence.
- scripts/run_update_gate0.py: blind, create-once JSONL screening tool.
- tests/test_update_gate0.py: deterministic checks of common ascent, conflicting gradients, duplicate-test invariance and malformed kernel rejection.

Local development run: 6 unit tests passed on CPU. These are synthetic mathematical/software checks, **not scientific evidence of improved coding accuracy**. The connected GitHub branch should be checked again with the repository's full CI/tests.

Run:

    uv sync --extra test
    uv run pytest -q tests/test_update_gate0.py
    uv run python scripts/run_update_gate0.py --input source-bank.jsonl --output gate0-predictions.jsonl

Input JSONL, one source per line, requires exactly four keys: source_id (str), before (T binary bits), after (G×T binary bits), score_kernel (G×G symmetric PSD). Hidden-test labels are deliberately forbidden in that input. Example data is illustrative only:

    {"source_id":"demo","before":[0,1],"after":[[1,1],[0,1],[0,0]],"score_kernel":[[4,2,-6],[2,2,-4],[-6,-4,10]]}

The command does not compute score gradients or execute a model; the real rollout/test/gradient-Gram bank still needs to be generated.

## Prospective falsification schedule

**Signal, no training:** Pin one diagnostic model, begin with Qwen2.5-Coder-1.5B and 128 RunBugRun source tasks, G=8 rollout corrections per source, seed 1701, and exactly four public tests under the existing EESD test budget. A separate n=8 public-test ablation requires its own locked manifest. Construct real score-gradient Grams in the trainable LoRA subspace from the same frozen checkpoint. Document the extra G backward passes, memory, and wall time. Measure the fraction with ≥2 active observed families, admitted fraction, LOFO signs, effective degrees of freedom, and duplicate-stress sensitivity. Stop the method if usable support is negligible; never hide skipped tasks.

**Independent evaluation, no training:** First write/freeze every Gate 0 decision. Only then let a separate evaluator open the hidden outcomes and measure whether admitted directions align with hidden reward gradients. Compare with binary-GRPO and pass-rate-gradient directions, and test with a randomly permuted gate matched on admission fraction. This is a local first-order proxy, **not** post-training pass@1.

**Matched actual training:** Only proceed if the direction proxy discriminates false updates. Compare no-update, binary GRPO, pass-rate GRPO, reproduced VeRPO, reproduced CodeLift, old EESD ablations, minimum-norm aggregation without leave-one-family-out gating, gating without minimum-norm projection, and the full candidate. Preserve 500 fresh source evaluations, matched response-token and measured KL budgets, seed 1701 before seeds 1702/1703. Use Qwen/RunBugRun, DeepSeek/CodeARC, and Gemma/RunBugRun as the positive/moderate/harm-prevention trio. Report all-pass@1, fix versus regression, source-cluster paired confidence intervals, hidden/public gap, coverage, compute and walltime. Do not pool replay stress-test results with official benchmark accuracy.

**Failure modes:** no common direction despite valuable rare repairs; accidentally merged independent tests from coincidentally matching outcome vectors; high overhead of per-repair score gradients; correlation among public families; stale off-policy kernels after any checkpoint update; loss of ordinary policy-gradient validity with edited-token masking; reward overfitting to test suites. These can kill the branch rather than becoming post-hoc advantages.

**Novelty bar:** MGDA, test clustering and cross-validation alone are not contributions. A paper requires evidence that update-direction coherence predicts and reduces harmful *persistent learning* beyond reward reweighting, under realistic matched-budget coding benchmarks. Until then the claim is only a new falsifiable EESD research hypothesis.
