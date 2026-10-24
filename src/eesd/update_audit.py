"""Prospective Gate 1: independent-oracle update-direction audit.

This is a finite-candidate, first-order sensitivity diagnostic. It provides a
Hoeffding lower bound on the mean probe sensitivity ONLY if probe inputs were
independent and identically distributed from the declared target test distribution,
were soundly labeled by a trusted oracle, and were not used to choose the update.
It is NOT a bound on hidden correctness or on the true policy-gradient objective.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def _coefficient_vector(score_kernel, rollout_weights):
    k = np.asarray(score_kernel, dtype=float)
    u = np.asarray(rollout_weights, dtype=float)
    if u.ndim != 1 or len(u) < 2 or not np.isfinite(u).all():
        raise ValueError('rollout_weights must be a finite vector with G >= 2')
    if k.shape != (len(u), len(u)) or not np.isfinite(k).all():
        raise ValueError('score_kernel must be a finite G x G matrix')
    if not np.allclose(k, k.T, atol=1e-7, rtol=0) or np.linalg.eigvalsh((k+k.T)/2)[0] < -1e-7:
        raise ValueError('score_kernel must be symmetric PSD')
    # c_i is the coefficient of a binary probe-test pass on candidate i.
    # The score-sensitivity baseline sum_i c_i = 0 is exact in finite samples.
    grad_dot_direction = k @ u
    return (grad_dot_direction - grad_dot_direction.mean()) / len(u)


def audit_update_probes(score_kernel, rollout_weights, probe_passes,
                        *, probe_ids: Sequence[str], delta: float = 0.05,
                        oracle_kind: str = 'reference_solution') -> dict:
    """Audit a fixed proposed direction on independent oracle-labeled probes.

    probe_passes: G candidates x M probe tests, 1 iff candidate matches oracle.
    Rejects duplicate IDs but cannot prove statistical independence or oracle
    soundness. Probe IDs must be distinct from public/hidden evaluation IDs by
    an external immutable manifest.
    """
    c = _coefficient_vector(score_kernel, rollout_weights)
    p = np.asarray(probe_passes)
    if p.ndim != 2 or p.shape[0] != len(c) or p.shape[1] < 1 or not np.isin(p, [0,1]).all():
        raise ValueError('probe_passes must be G x M binary pass indicators')
    if not 0 < delta < 1 or not math.isfinite(delta):
        raise ValueError('delta must be in (0,1)')
    if oracle_kind not in {'reference_solution', 'trusted_checker'}:
        raise ValueError('a documented trusted oracle type is required')
    ids = list(probe_ids)
    if (len(ids) != p.shape[1] or any(not isinstance(x,str) or not x for x in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError('unique nonempty probe IDs required')
    sensitivity = c @ p.astype(float)
    m = len(sensitivity)
    # For each binary pass vector y, z = sum_i c_i y_i lies in [-B, B].
    B = float(np.maximum(c, 0).sum())
    radius = B * math.sqrt(2*math.log(1/delta) / m)
    mean = float(np.mean(sensitivity))
    lcb = mean - radius
    return {
        'schema':'eesd-independent-oracle-audit-v1',
        'status':'first_order_finite_candidate_proxy_only',
        'accepted':bool(B > 0 and lcb > 0),
        'reason':('passed_lower_bound' if B > 0 and lcb > 0 else
                  'zero_direction' if B <= 0 else 'insufficient_independent_evidence'),
        'probe_count':m, 'oracle_kind':oracle_kind, 'delta':float(delta),
        'probe_sensitivity':sensitivity.tolist(),
        'mean_sensitivity':mean, 'sensitivity_bound':B,
        'hoeffding_radius':float(radius), 'lower_confidence_bound':float(lcb),
    }


def worst_counterexample_probe(score_kernel, rollout_weights, probe_passes,
                                *, probe_ids: Sequence[str]) -> dict:
    """Adaptively select strongest negative observed test, NOT a confidence bound.

    Independent audit probes must not be reused as adaptive search candidates.
    Sound input/output labels must be checked outside this purely numerical code.
    """
    c = _coefficient_vector(score_kernel, rollout_weights)
    p = np.asarray(probe_passes)
    ids = list(probe_ids)
    if (p.ndim != 2 or p.shape[0] != len(c) or p.shape[1] < 1 or
        not np.isin(p,[0,1]).all() or len(ids) != p.shape[1] or
        len(set(ids)) != len(ids) or any(not isinstance(x,str) or not x for x in ids)):
        raise ValueError('invalid counterexample probe bank')
    values = c @ p.astype(float)
    worst = int(np.argmin(values))
    return {'schema':'eesd-adaptive-falsifier-v1',
            'worst_probe_id':ids[worst], 'worst_sensitivity':float(values[worst]),
            'negative_probes':int(np.sum(values < 0)),
            'probe_count':len(ids),
            'status':'descriptive_adaptive_search_not_confidence_bound'}
