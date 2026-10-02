# EESD Final Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a standalone, reproducible EESD research repository with frozen model/data contracts, complete baselines, tests, recorded evidence, and prospective final-method gates.

**Architecture:** Extract the EESD implementation into `src/eesd`, retain only required compatibility primitives, and organize execution as validate, signal, mechanism, downstream, and expansion stages. Configuration and result manifests distinguish recorded lexical-EED evidence from prospective Shapley-EESD experiments.

**Tech Stack:** Python 3.10+, NumPy, PyYAML; optional PyTorch/Transformers/PEFT/Datasets/Accelerate for GPU experiments; pytest for testing.

**Spec:** `docs/research/2026-10-02-eesd-final-design.md`

## Global Constraints

- Pin every model and benchmark/evaluator revision.
- Never relabel replay assessments as official benchmark accuracy.
- Never present lexical-EED recorded results as Shapley-EESD results.
- Match response-token budgets across training baselines.
- Preserve all predeclared cells and negative results.

## Review Focus

- Zero or near-zero Shapley attribution must not produce NaN or infinite mass.
- Fixed and effective mass comparisons must use identical relevance and prior.
- Missing dataset/model revisions must fail before GPU work starts.
- No-update must execute no optimizer step and consume zero training tokens.
- Recorded and prospective result schemas must be impossible to merge silently.

---

### Task 1: Standalone mathematical core

**Files:** create `src/eesd/{evidence,shapley_relevance,distillation}.py`; migrate focused tests.

**Interfaces:** produce exact Shapley values, relevance normalization, effective mass, Dirichlet posteriors, and training-rule weights.

- [ ] Migrate tests and verify they fail before package migration.
- [ ] Migrate the implementation with namespace-only changes.
- [ ] Run focused mathematical tests.

### Task 2: Experiment contracts and planner

**Files:** create pinned YAML configs, `src/eesd/contracts.py`, `src/eesd/planner.py`, `scripts/validate_experiment.py`, and tests.

**Interfaces:** consume YAML; produce validated stage/cell plans with exact model, dataset, budget, baseline, seed, and result-schema identities.

- [ ] Write failing contract tests.
- [ ] Implement strict validation and deterministic plan generation.
- [ ] Run contract tests and dry-run every stage.

### Task 3: Runners and baseline-complete pipeline

**Files:** migrate EESD runners and required helpers; add `scripts/run_stage.py` and tests.

**Interfaces:** consume validated plans; dispatch signal, mechanism, downstream, and expansion commands without launching them in dry-run mode.

- [ ] Write failing dispatch and matched-budget tests.
- [ ] Migrate/adapt the minimal runnable implementation.
- [ ] Run CLI tests and compile all scripts.

### Task 4: Evidence, paper, CI, and handoff

**Files:** add README, provenance/status documents, recorded summaries, paper source, GitHub Actions, and license.

**Interfaces:** make installed/implemented/tested/scientifically-established status explicit.

- [ ] Add repository integrity tests.
- [ ] Migrate artifacts and paper source with provenance.
- [ ] Run the full available suite, package build, and repository audit.

