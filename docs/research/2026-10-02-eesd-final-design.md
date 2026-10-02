# EESD Final Repository Design

## Scientific decision

EESD studies when an LLM coding agent should learn from its own execution-guided correction. The parent failure is harmful self-distillation: a correction can pass visible tests yet be unsupported or regress on hidden behavior. Recorded one-round results are heterogeneous (4 positive, 1 tie, 7 negative model-domain cells), so the target is calibrated update trust, not unconditional self-correction.

The final method is prospective until rerun: exact Shapley attribution of four public executions on edited-token log-probability contrast; absolute attributions become relevance; Renyi-2 effective support controls Dirichlet evidence mass over improved/unchanged/regressed transitions; the positive posterior mean advantage weights correction SFT; a forward KL term anchors the student to the pre-update policy.

## Evidence status

- **Recorded evidence:** lexical-relevance EED mechanism and the prior 12-cell downstream matrix.
- **Implemented, not scientifically established:** exact Shapley relevance and the final axiomatic EESD pipeline.
- **Forbidden claim:** the recorded downstream gains validate the Shapley method.

## Gate sequence

1. **Software Gate:** unit tests, schema validation, dry-run plans, pinned revisions.
2. **Signal Gate:** 64 trajectories in each of three diagnostic cells; reject if the Shapley game span is degenerate or attribution is unstable.
3. **Mechanism Gate:** compare Shapley-EED against Shapley-fixed-mass and lexical-EED on hidden NLL, Brier, calibration, rank correlation, and concentration bins.
4. **Downstream Gate:** matched response-token budgets for no-update, equal-weight, final-correctness, scalar-confidence, fixed-mass Dirichlet, lexical-EED, Shapley-EED without KL, and full EESD.
5. **Expansion Gate:** only after the preceding gates pass, expand seeds, models, datasets, recursive rounds, EvalPlus, and LiveCodeBench.

## Frozen diagnostic cells

| Role | Model | Dataset | Reason |
|---|---|---|---|
| Positive preservation | DeepSeek-Coder-6.7B-Instruct | CodeARC | strongest recorded old-EESD gain (+5.4 pp) |
| Moderate signal | Qwen2.5-Coder-7B-Instruct | RunBugRun | positive but uncertain recorded gain (+2.4 pp) |
| Harm prevention | Gemma-3-4B-it | RunBugRun | recorded negative cell (-2.0 pp) |

Each downstream arm uses 500 fresh primary sources, seed 1701, identical prompts/candidates/executions, matched response-token budgets, and 10,000 paired source-cluster bootstrap draws. Seeds 1702 and 1703 are confirmation runs after the first gate passes.

## Expansion benchmarks

- RunBugRun: executable program repair and hidden-test generalization.
- CodeARC: interactive inductive program synthesis with differential execution feedback.
- EvalPlus HumanEval+ and MBPP+: external functional-correctness transfer only.
- LiveCodeBench release_v6: temporally filtered external coding transfer only.
- APPS Replay and CodeContests Replay: controlled stress tests; report as replay assessments, never official benchmark accuracy.

## Repository boundary

The repository contains the standalone `eesd` package, EESD scripts, frozen configs, tests, research contracts, recorded summaries, and the latest paper source. PBPF/APBPF history is not copied. Required execution helpers are migrated into `eesd.compat` or rewritten behind narrow interfaces. A provenance manifest maps every migrated file to source commit `d00969ce49bde6108aa4f53dccb09eeb150db601` and identifies the paper archive separately.

