import numpy as np
import pytest
from eesd.update_gate0 import gate_update
from eesd.update_audit import audit_update_probes, worst_counterexample_probe


def _toy():
    scores=np.eye(3)-np.ones((3,3))/3
    k=scores@scores.T
    gate=gate_update([0,0],[[1,1],[0,1],[0,0]],k)
    assert gate['accepted']
    return k,gate['rollout_weights']


def test_public_gate_accepts_but_oracle_counterexample_finds_harm():
    k,u=_toy()
    probe=np.array([[0],[0],[1]])
    chosen=worst_counterexample_probe(k,u,probe,probe_ids=['opposite'])
    assert chosen['worst_probe_id']=='opposite'
    assert chosen['worst_sensitivity'] < 0
    check=audit_update_probes(k,u,probe,probe_ids=['opposite'])
    assert not check['accepted']
    assert check['mean_sensitivity'] < 0


def test_update_specific_counterexample_beats_undirected_disagreement():
    k,u=_toy()
    # Each probe has the same number of passing candidate programs, yet
    # the proposed update helps one and harms the other.
    probes=np.array([[1,0],[0,0],[0,1]])
    result=worst_counterexample_probe(k,u,probes,probe_ids=['helped','harmed'])
    assert result['worst_probe_id']=='harmed'
    assert result['negative_probes']==1


def test_duplicate_probe_ids_rejected():
    k,u=_toy()
    with pytest.raises(ValueError,match='unique nonempty'):
        audit_update_probes(k,u,[[0,0],[0,0],[1,1]],probe_ids=['same','same'])


def test_oracle_reliability_is_a_stated_external_assumption():
    k,u=_toy()
    with pytest.raises(ValueError,match='trusted oracle'):
        audit_update_probes(k,u,[[1],[0],[0]],probe_ids=['p'],oracle_kind='unverified_model_judge')


def test_sensitivity_bound_is_valid_for_all_binary_patterns():
    import itertools
    k,u=_toy()
    samples=np.array(list(itertools.product([0,1],repeat=3))).T
    result=audit_update_probes(k,u,samples,probe_ids=[f'p{i}' for i in range(samples.shape[1])])
    assert all(abs(x)<=result['sensitivity_bound']+1e-12 for x in result['probe_sensitivity'])
    assert abs(np.mean(result['probe_sensitivity']))<1e-12


def test_too_few_independent_probes_abstains_despite_favorable_outcome():
    k,u=_toy()
    out=audit_update_probes(k,u,[[1],[0],[0]],probe_ids=['p'],delta=.05)
    assert out['mean_sensitivity']>0
    assert not out['accepted']
    assert out['lower_confidence_bound']<0


def test_many_independent_favorable_probes_can_pass_bound():
    k,u=_toy()
    probe=np.repeat(np.array([[1],[0],[0]]),50,axis=1)
    out=audit_update_probes(k,u,probe,probe_ids=[f'p{i}' for i in range(50)])
    assert out['accepted']
    # Repetition here is to verify numerical behavior. In a scientific
    # experiment, identical observations are NOT automatically independent.


def test_falsifier_does_not_synthesize_trusted_outcomes():
    k,u=_toy()
    with pytest.raises(ValueError,match='invalid counterexample'):
        worst_counterexample_probe(k,u,[[.2],[0],[1]],probe_ids=['invalid'])
