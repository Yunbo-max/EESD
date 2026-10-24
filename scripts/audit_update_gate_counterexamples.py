#!/usr/bin/env python3
"""Synthetic categorical-policy falsification of Gate 0; NOT a model benchmark.

Each task has a four-action softmax policy with exact score functions. Hidden
outcomes are NEVER passed to either public admission or probe decisions.
This is a reproducible directional-utility toy, not coding accuracy.
"""
from __future__ import annotations
import argparse
import json
import numpy as np
from eesd.update_gate0 import gate_update
from eesd.update_audit import audit_update_probes


def _softmax(x):
    w = np.exp(x - np.max(x))
    return w / w.sum()


def _gate0_demo():
    scores = np.eye(3) - 1. / 3.
    public = np.array([[1, 1], [0, 1], [0, 0]])
    result = gate_update([0, 0], public, scores @ scores.T)
    direction = np.asarray(result['rollout_weights']) @ scores
    hidden = np.array([0, 0, 1])
    eta = 0.3
    delta = float(_softmax(eta * direction) @ hidden - hidden.mean())
    return {'accepted': result['accepted'], 'cosines': result['lofo_cosines'],
            'hidden_before': float(hidden.mean()),
            'hidden_after': float(_softmax(eta * direction) @ hidden),
            'hidden_delta': delta}


def run(seed=1701, tasks=512, public_tests=4, hidden_tests=128,
        probe_tests=8, step=0.3):
    rng = np.random.default_rng(seed)
    g = 4
    s = np.eye(g) - 1/g
    k = s @ s.T
    out = []
    for task in range(tasks):
        # IID hypothesis: public and hidden sample from the same latent q.
        # Shifted hypothesis: hidden correctness anticorrelates with q.
        q = rng.uniform(.12, .88, g)
        y = rng.binomial(1, q[:, None], size=(g, public_tests))
        probe = rng.binomial(1, q[:, None], size=(g, probe_tests))
        probe_shift = rng.binomial(1, (1-q)[:, None], size=(g, probe_tests))
        hidden_iid = rng.binomial(1, q[:, None], size=(g, hidden_tests))
        hidden_shift = rng.binomial(1, (1-q)[:, None], size=(g, hidden_tests))
        gate = gate_update([0]*public_tests, y, k)
        weights = np.asarray(gate['rollout_weights'])
        d = weights @ s
        public_reward = y.mean(axis=1)
        d_base = (public_reward-public_reward.mean()) @ s/g
        gain_probes = (probe - probe.mean(axis=0)).T @ k @ weights /g
        probe_mean = float(gain_probes.mean())
        audit = audit_update_probes(k, weights, probe,
                      probe_ids=[f"{task}-iid-{j}" for j in range(probe_tests)])
        probe_shift_mean = float(((probe_shift-probe_shift.mean(axis=0)).T @ k @ weights /g).mean())
        probe_negative_fraction = float((gain_probes < -1e-12).mean())
        d_probe = (probe.mean(axis=1)-probe.mean()) @ s /g
        def measured(h, direction):
            return float(h.mean(axis=1) @ (_softmax(step * direction) - 1./g))
        out.append({'gate0': bool(gate['accepted']),
                    'gate1_in_dist': bool(gate['accepted'] and probe_mean > 0),
                    'gate1_confident_iid': bool(gate['accepted'] and audit['accepted']),
                    'gate1_shift_oracle': bool(gate['accepted'] and probe_shift_mean > 0),
                    'public_rew': float(public_reward.mean()),
                    'probe_mean_gain': probe_mean,
                    'probe_negative_frac': probe_negative_fraction,
                    'direction_norm': float(np.linalg.norm(d)),
                    'delta_gate_iid': measured(hidden_iid,d),
                    'delta_pass_iid': measured(hidden_iid,d_base),
                    'delta_probe_iid': measured(hidden_iid,d_probe),
                    'delta_gate_shift': measured(hidden_shift,d),
                    'delta_pass_shift': measured(hidden_shift,d_base),
                    'delta_probe_shift': measured(hidden_shift,d_probe),
                    'positive_gate_iid': bool(measured(hidden_iid,d)>0),
                    'positive_gate_shift': bool(measured(hidden_shift,d)>0)})
    accepted = [x for x in out if x['gate0']]
    accepted_probe = [x for x in out if x['gate1_in_dist']]
    accepted_confident = [x for x in out if x['gate1_confident_iid']]
    accepted_shift_probe = [x for x in out if x['gate1_shift_oracle']]
    return {'schema':'eesd-synthetic-falsification-v1','seed':seed,
            'caveat':'synthetic categorical bandits only; not model/benchmark evidence',
            'fixed_adversarial_counterexample': _gate0_demo(),
            'task_count':tasks, 'gate0_accepted':len(accepted),
            'gate0_rate': len(accepted)/tasks,
            'gate1_in_dist_accepted':len(accepted_probe),
            'gate1_lcb_accepted':len(accepted_confident),
            'gate1_oracle_shift_accepted':len(accepted_shift_probe),
            'gate1_in_dist_harm_rate_iid':float(np.mean([x['delta_gate_iid'] < -1e-12 for x in accepted_probe])) if accepted_probe else None,
            'gate1_in_dist_harm_rate_shift':float(np.mean([x['delta_gate_shift'] < -1e-12 for x in accepted_probe])) if accepted_probe else None,
            'gate1_oracle_shift_harm_rate_shift':float(np.mean([x['delta_gate_shift'] < -1e-12 for x in accepted_shift_probe])) if accepted_shift_probe else None,
            'gate1_in_dist_mean_delta_shift_all_sources':float(np.mean([x['delta_gate_shift'] for x in accepted_probe])*len(accepted_probe)/tasks),
            'gate1_oracle_shift_mean_delta_shift_all_sources':float(np.mean([x['delta_gate_shift'] for x in accepted_shift_probe])*len(accepted_shift_probe)/tasks) if accepted_shift_probe else 0.0,
            'gate0_accepted_harm_rate_iid': float(np.mean([x['delta_gate_iid'] < -1e-12 for x in accepted])) if accepted else None,
            'gate0_accepted_harm_rate_shift': float(np.mean([x['delta_gate_shift'] < -1e-12 for x in accepted])) if accepted else None,
            'mean_delta_pass_iid': float(np.mean([x['delta_pass_iid'] for x in out])),
            'mean_delta_gate_iid_all_sources': float(np.mean([x['delta_gate_iid'] for x in out])),
            'mean_delta_pass_shift': float(np.mean([x['delta_pass_shift'] for x in out])),
            'mean_delta_gate_shift_all_sources': float(np.mean([x['delta_gate_shift'] for x in out])),
            'avg_probe_negative_fraction_accepted': float(np.mean([x['probe_negative_frac'] for x in accepted])) if accepted else None,
            'corr_probe_mean_vs_gate_iid_delta_on_admitted': float(np.corrcoef([x['probe_mean_gain'] for x in accepted],[x['delta_gate_iid'] for x in accepted])[0,1]) if len(accepted)>=3 else None}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', type=int, default=512)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--output')
    args=p.parse_args()
    r=run(seed=args.seed,tasks=args.tasks)
    print(json.dumps(r,indent=2,sort_keys=True))
    if args.output:
        import pathlib
        path=pathlib.Path(args.output)
        if path.exists():raise FileExistsError(path)
        path.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
