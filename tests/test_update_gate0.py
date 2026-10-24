import numpy as np
import pytest

from eesd.update_gate0 import gate_update, _minnorm_weights


def _sample():
    g = np.array([[2., 0.], [1., 1.], [-3., -1.]])
    return [0, 1], [[1, 1], [0, 1], [0, 0]], g @ g.T, g


def test_mgnorm_empirical_common_ascent():
    g = np.array([[1., 0.], [2., 1.], [1., 1.]])
    a, gap = _minnorm_weights(g @ g.T)
    d = a @ g
    assert gap < 1e-7
    assert np.all(g @ d >= d @ d - 1e-7)


def test_opposed_gradients_have_zero_direction():
    g = np.array([[1., 0.], [-1., 0.]])
    a, _ = _minnorm_weights(g @ g.T)
    np.testing.assert_allclose(a @ g, 0, atol=1e-8)


def test_aligned_cross_family_update_and_duplicate_invariance():
    b, y, k, g = _sample()
    a = gate_update(b, y, k)
    assert a["accepted"]
    assert np.all(np.array(a["lofo_cosines"]) > 0)
    assert np.linalg.norm(np.array(a["rollout_weights"]) @ g) > 0
    repeated_y = np.column_stack((np.tile(np.array(y)[:, :1], (1, 100)), np.array(y)[:, 1:]))
    duplicated = gate_update([0]*100+[1], repeated_y, k)
    np.testing.assert_allclose(a["rollout_weights"], duplicated["rollout_weights"])
    assert len(duplicated["families"]) == 2


def test_duplicate_only_has_no_fake_independent_evidence():
    g = np.array([[1., 0.], [0., 1.], [-1., -1.]])
    out = gate_update([0]*50, np.tile([[1], [0], [0]], (1, 50)), g @ g.T)
    assert not out["accepted"]
    assert out["reason"] == "insufficient_behavior_families"
    assert out["rollout_weights"] == [0., 0., 0.]


def test_conflicting_families_abstain():
    before, after, _, _ = _sample()
    g = np.array([[2., 0.], [0., 1.], [-2., -1.]])
    out = gate_update(before, [[1, 0], [0, 1], [0, 1]], g @ g.T)
    assert not out["accepted"]


def test_invalid_kernel_rejected():
    b, y, _, _ = _sample()
    with pytest.raises(ValueError, match="symmetric PSD"):
        gate_update(b, y, [[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]])


def test_gate_is_invariant_to_positive_global_score_rescaling():
    b, y, k, _ = _sample()
    baseline = gate_update(b, y, k)
    for scale in [1e-12, 1e-6, 1e6]:
        other = gate_update(b, y, k * scale * scale)
        assert other['accepted'] == baseline['accepted']
        assert other['reason'] == baseline['reason']
        np.testing.assert_allclose(other['rollout_weights'], baseline['rollout_weights'], atol=1e-8)
        np.testing.assert_allclose(other['lofo_cosines'], baseline['lofo_cosines'], atol=1e-8)


def test_invalid_nonfinite_thresholds_rejected():
    b, y, k, _ = _sample()
    for key, value in [('std_floor', float('nan')), ('std_floor', float('inf')),
                       ('dual_tolerance', float('nan')), ('dual_tolerance', -1.),
                       ('norm_floor', float('nan')), ('min_families', 2.3)]:
        with pytest.raises(ValueError, match='invalid gate thresholds'):
            gate_update(b, y, k, **{key: value})
