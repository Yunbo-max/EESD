"""Evidence-calibrated reward for RLVR.

The reward decomposes execution outcomes into two signals that the legacy
weighted-SFT formulation conflated:

  * test reliability ``w_j``  - how decisive each public test is (leave-one-out
    outcome flip), so weak/duplicate tests contribute less reward;
  * effective support ``E(p)`` - Renyi-2 concentration of a candidate's passing
    distribution, so a candidate that passes only one marginal test is discounted
    relative to one that passes tests broadly (anti reward-hacking).

``reward_i = (E_i / n) * sum_j w_j * 1[candidate_i passes test_j]``

All functions are pure numpy and unit-testable without a model or GPU.
"""

from __future__ import annotations

import numpy as np


def test_reliability(outcome_matrix: np.ndarray) -> np.ndarray:
    """Leave-one-out flip rate per test column.

    outcome_matrix: (G, n) boolean, True means PASS. Returns w of shape (n,)
    summing to 1. If no test is decisive (all flips are zero) falls back to
    uniform weights so reward does not collapse to zero.
    """
    matrix = np.asarray(outcome_matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("outcome_matrix must be a nonempty 2D boolean array")
    g, n = matrix.shape
    all_pass = matrix.all(axis=1)  # (G,) whether candidate passes every test
    flips = np.zeros(n, dtype=float)
    for j in range(n):
        without_j = matrix[:, np.arange(n) != j].all(axis=1)
        flips[j] = float((all_pass != without_j).sum())
    total = flips.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return flips / total


def effective_support(pass_vector: np.ndarray) -> float:
    """Renyi-2 effective support of a candidate's pass distribution.

    pass_vector: (n,) boolean. Returns E in [0, n]; E = number of passed tests
    when the passing distribution is uniform, and E = 1 when concentrated on a
    single test. Zero passed tests yields E = 0.
    """
    vector = np.asarray(pass_vector, dtype=bool)
    if vector.ndim != 1:
        raise ValueError("pass_vector must be 1D")
    count = int(vector.sum())
    if count == 0:
        return 0.0
    probabilities = vector.astype(float) / count
    return 1.0 / float(np.square(probabilities).sum())


def plain_reward(outcome_matrix: np.ndarray) -> np.ndarray:
    """Baseline: per-candidate pass fraction, no evidence weighting."""
    matrix = np.asarray(outcome_matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("outcome_matrix must be a nonempty 2D boolean array")
    return matrix.astype(float).mean(axis=1)


def evidence_calibrated_reward(outcome_matrix: np.ndarray) -> np.ndarray:
    """Per-candidate scalar reward from a (G, n) boolean outcome matrix.

    Returns reward of shape (G,) where each element is in [0, 1].
    """
    matrix = np.asarray(outcome_matrix, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("outcome_matrix must be a nonempty 2D boolean array")
    g, n = matrix.shape
    weights = test_reliability(matrix)
    weighted_pass = matrix.astype(float) @ weights  # (G,) in [0, 1]
    discount = np.array([effective_support(row) for row in matrix]) / n  # (G,)
    return weighted_pass * discount


def output_similarity(actual: str, expected: str) -> float:
    if actual == expected:
        return 1.0
    if not actual or not expected:
        return 0.0
    from difflib import SequenceMatcher

    return SequenceMatcher(None, actual, expected).ratio()


def soft_reward(scores: np.ndarray) -> np.ndarray:
    """Baseline soft reward: mean per-test score (exact pass = 1, close output = similarity)."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a nonempty 2D float array")
    return matrix.mean(axis=1)


def _effective_support_soft(score_vector: np.ndarray) -> float:
    total = float(score_vector.sum())
    if total <= 0:
        return 0.0
    probabilities = np.asarray(score_vector, dtype=float) / total
    return 1.0 / float(np.square(probabilities).sum())


def evidence_soft_reward(scores: np.ndarray) -> np.ndarray:
    """Evidence-calibrated soft reward: reliability-weighted score with an E(p) discount."""
    matrix = np.asarray(scores, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("scores must be a nonempty 2D float array")
    g, n = matrix.shape
    weights = test_reliability(matrix > 0.5)
    weighted = matrix @ weights  # (G,) in [0, 1]
    discount = np.array([_effective_support_soft(row) for row in matrix]) / n  # (G,)
    return weighted * discount
