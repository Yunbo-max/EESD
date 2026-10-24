"""Prospective EESD update gate: public-test-family cross-fit in score-Gram space.

Only CPU mathematics. This is NOT evidence of hidden-test or model improvement.
"""
from __future__ import annotations

import numpy as np


def _project_simplex(z):
    if len(z) == 1:
        return np.ones(1)
    u = np.sort(z)[::-1]
    cumulative = np.cumsum(u)
    rho = np.flatnonzero(u - (cumulative - 1) / np.arange(1, len(u) + 1) > 0)[-1]
    return np.maximum(z - (cumulative[rho] - 1) / (rho + 1), 0)


def _minnorm_weights(h, *, steps=10000, tol=1e-11):
    """MGDA minimum-norm convex-hull weights; returns weights and dual gap."""
    h = (h + h.T) / 2
    eig = np.linalg.eigvalsh(h)
    if eig[0] < -1e-7:
        raise ValueError("gradient Gram matrix is not PSD")
    n = len(h)
    if n == 1:
        return np.ones(1), 0.0
    L = max(float(eig[-1]), 1e-15)
    alpha = np.full(n, 1.0 / n)
    z = alpha.copy()
    t = 1.0
    for _ in range(steps):
        nxt = _project_simplex(z - h @ z / L)
        if np.linalg.norm(nxt - alpha) <= tol:
            alpha = nxt
            break
        new_t = (1 + np.sqrt(1 + 4 * t * t)) / 2
        z = nxt + ((t - 1) / new_t) * (nxt - alpha)
        alpha, t = nxt, new_t
    norm2 = max(0.0, float(alpha @ h @ alpha))
    return alpha, max(0.0, norm2 - float(np.min(h @ alpha)))


def gate_update(before, after, score_kernel, *, std_floor=0.2, min_families=2,
                norm_floor=1e-9, dual_tolerance=1e-7):
    """Compute an update decision and rollout-score weights from public data.

    before: T original-program test pass bits
    after: G x T candidate-program pass bits
    score_kernel: G x G Gram of grad log pi(y_i|x) at a frozen policy
    The returned coefficients w define d=sum_i w_i * grad log pi(y_i|x).
    """
    b, y, k = map(np.asarray, (before, after, score_kernel))
    if b.ndim != 1 or y.ndim != 2 or y.shape[1] != len(b) or y.shape[0] < 2:
        raise ValueError("expected before[T] and after[G,T] with G>=2")
    if not np.isin(b, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("test outcomes must be binary")
    G, T = y.shape
    if not T or k.shape != (G, G) or not np.isfinite(k).all():
        raise ValueError("score_kernel must be finite G x G")
    if not np.allclose(k, k.T, atol=1e-7, rtol=0) or np.linalg.eigvalsh((k+k.T)/2)[0] < -1e-7:
        raise ValueError("score_kernel must be symmetric PSD")
    if std_floor <= 0 or min_families < 2 or norm_floor <= 0:
        raise ValueError("invalid gate thresholds")
    families_by_signature = {}
    for j in range(T):
        signature = (int(b[j]), tuple(int(v) for v in y[:, j]))
        families_by_signature.setdefault(signature, []).append(j)
    families = tuple(tuple(v) for v in families_by_signature.values())
    rewards = np.column_stack([(y[:, js]-b[list(js)]).mean(axis=1) for js in families])
    centered = rewards - rewards.mean(axis=0, keepdims=True)
    advantages = centered / np.maximum(centered.std(axis=0, keepdims=True), std_floor)
    h = (advantages.T @ k @ advantages) / (G * G)
    h = (h + h.T) / 2
    active = np.flatnonzero(np.sqrt(np.maximum(np.diag(h), 0)) > norm_floor)
    mixture = np.zeros(len(families))
    rollout_weights = np.zeros(G)
    lofo = [None] * len(families)
    gains = np.zeros(len(families))
    norm, gap = 0.0, 0.0
    reason = "insufficient_behavior_families"
    if len(active) >= min_families:
        sub = h[np.ix_(active, active)]
        alpha, gap = _minnorm_weights(sub)
        mixture[active] = alpha
        norm = float(np.sqrt(max(0.0, alpha @ sub @ alpha)))
        gains[active] = sub @ alpha
        for i, original_index in enumerate(active):
            remaining = np.array([j for j in range(len(active)) if j != i])
            smaller = sub[np.ix_(remaining, remaining)]
            w, _ = _minnorm_weights(smaller)
            denominator = float(np.sqrt(max(0.0, sub[i, i] * (w @ smaller @ w))))
            lofo[original_index] = float(sub[i, remaining] @ w / denominator) if denominator > norm_floor else 0.0
        if gap > dual_tolerance:
            reason = "solver_gap"
        elif norm <= norm_floor:
            reason = "no_common_ascent"
        elif any(lofo[j] <= 0 for j in active):
            reason = "cross_family_conflict"
        else:
            reason = "passed_diagnostic"
            rollout_weights = advantages @ mixture / G
    return {
        "schema": "eesd-gate0-prospective-v1", "accepted": reason == "passed_diagnostic",
        "reason": reason, "families": [list(f) for f in families],
        "active_families": active.tolist(), "family_mixture": mixture.tolist(),
        "rollout_weights": rollout_weights.tolist(), "lofo_cosines": lofo,
        "first_order_gains": gains.tolist(), "direction_norm": norm, "dual_gap": gap,
    }
