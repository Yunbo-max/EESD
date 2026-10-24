# EESD Redesign Proposal: Evidence-Calibrated RLVR

Status: PROPOSED. This document is a prospective method redesign motivated by the
recorded experimental evidence. It introduces no completed validation and must not
be cited as a result. It supersedes neither the frozen final design
(`2026-10-02-eesd-final-design.md`) nor the recorded lexical-EED evidence.

## 1. What the recorded results actually tell us

Three recorded findings drive this redesign.

1. **The strength channel (`E(p)`) is doing little work.** In the matched-mass
   mechanism study (RQ2), effective mass and fixed mass agree in argmax on all 3,000
   examples and differ in NLL by only ~0.006; the appendix records that the NLL
   difference changes sign across seeds. In the history sweep (RQ1), fixed mass and
   EED improve almost in lockstep as `n` grows; the effective-mass increment is
   0.0046--0.0113 NLL. The Rényi-2 effective support was the headline of the
   original formulation, but the evidence does not support it as a reliable signal.

2. **Backbone ability is the dominant variable, and the method does not model it.**
   The 12-cell downstream matrix splits cleanly by backbone: DeepSeek-Coder-6.7B is
   positive in three of four domains, Qwen2.5-Coder-7B is mixed and not resolved,
   and Gemma-3-4B is negative in all four cells, with APPS Replay (-6.8 pp) and
   CodeContests Replay (-5.4 pp) significantly harmful. Self-distillation amplifies
   whatever the backbone produces; a weak backbone produces noisy corrections and
   the update makes it worse.

3. **Both relevance estimators have known failure modes.** The lexical kernel scores
   vocabulary overlap between the diff and the execution descriptor; it is a
   coincidence detector, not an evidence estimator. Exact Shapley attribution scores
   edited-token log-probability contrast, which risks degeneracy when the model's
   edit preference does not respond to adding or removing a public execution
   (`game_span` near zero), collapsing relevance to a uniform fallback.

## 2. The repositioning: RLVR is orthogonal to EESD, not a replacement

Reinforcement learning with verifiable rewards (RLVR: GRPO/PPO with execution
pass/fail as reward) and EESD answer different questions.

- **RLVR answers "what signal should drive learning"**: the execution outcome
  itself is a natural, verifiable reward; no correction, no relevance, no `E(p)` is
  required to form it.
- **EESD answers "how much of that signal is trustworthy"**: a correction that
  passes four public tests need not generalize to hidden ones.

The problem EESD was built to solve — over-trusting concentrated public evidence —
is exactly the known failure mode of RLVR (reward hacking / overfitting to the
verifier). RLVR therefore does not make EESD obsolete; it makes EESD's core idea
(evidence concentration controls trust) *more* valuable, because RLVR moves toward
the reward gradient more aggressively than weighted SFT and amplifies the same
overfitting the single-round Gemma cell already exhibits.

The redesign keeps the evidence-calibration idea but relocates it from a scalar SFT
coefficient to the RL advantage, where gradient scaling actually happens.

## 3. Proposed design: evidence-calibrated RLVR

### 3.1 New variables

Two new variables replace the scalar correction weight.

- **Advantage discount** `λ_i = E(p_i) / n` for rollout `i`, derived from the
  execution-outcome concentration of that rollout's public tests. Concentrated
  evidence (one decisive test) is discounted; broad evidence is not.
- **Test-level reliability** `w_j`, estimated per public test from leave-one-out
  outcome flips, replacing the uniform `sum`/`all` reward aggregation.

### 3.2 Reward construction

For each source, sample `G` programs and execute each on `n` public tests to obtain
an outcome matrix `(G x n)`. The reward is

```
r_i = sum_j  w_j * 1[test_j passes for program i]
```

where `w_j` is the test-level reliability. This turns the scalar `1[all pass]` into
a reliability-weighted vector reward, so a policy can learn which tests carry
signal instead of summing them blindly.

### 3.3 Advantage discount

`λ_i = E(p_i) / n` multiplies the advantage in the GRPO objective, so a solution
that passes only a single marginal test contributes less gradient than one that
passes all tests stably. This is the EESD strength channel moved to the place it
was designed for: gradient scaling under a reward signal.

### 3.4 Backbone-aware KL anchor

The KL coefficient is made a function of backbone ability:

```
β(ability) = β_0 * (ability / ability_ref)^(-γ)
```

with `ability` estimated from the backbone's own pass rate or calibration on the
target domain. A weak backbone (Gemma-class) gets a stronger anchor; below a
threshold, self-distillation is disabled and the arm reverts to `no_update`. This
directly encodes the strongest, most reproducible negative result in the 12-cell
matrix.

## 4. Risks and open questions

1. **Reward hacking is worse under RL, not better.** A reward built only from
   public tests will be exploited more aggressively than weighted SFT. The
   advantage discount only partially mitigates this. The real fix requires mixing in
   a hidden-test proxy (a held-out validation reward) or enforcing test-time
   diversity. This is an open problem that R1/TÜLU-3 do not fully solve and is a
   candidate differentiator if EESD's evidence idea is folded in.

2. **The `game_span` risk moves, it does not disappear.** Whether `E(p_i)` carries
   signal depends on whether execution outcomes vary across programs for a given
   source. If all `G` programs pass the same tests, `λ_i` is uniform and the design
   reduces to plain RLVR. The signal gate must report `game_span` on the outcome
   matrix, not on the log-probability contrast.

3. **Backbone `ability` is an estimate, not a measurement.** The proposed `β`
   schedule needs a calibration protocol (ECE or held-out pass rate) before it can
   be frozen; it is currently a placeholder functional form.

## 5. Relationship to the existing gate sequence

This proposal changes the *method*, not the discipline. It should enter the same
frozen gate order: Software -> Signal -> Mechanism -> Downstream -> Expansion, with
the outcome-matrix `game_span` replacing the log-probability `game_span` as the
signal-gate rejection diagnostic. The recorded lexical-EED downstream numbers
remain motivation only and must never be relabeled as evidence for this redesign.
