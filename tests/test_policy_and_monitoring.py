import numpy as np
from fraud_ds.policy import evaluate_threshold, optimize_threshold
from fraud_ds.monitoring import population_stability_index


def test_cost_calculation():
    y = [0, 0, 1, 1]
    p = [0.1, 0.8, 0.2, 0.9]
    r = evaluate_threshold(y, p, .5, fraud_miss_cost=100, false_decline_cost=10)
    assert r.false_positives == 1
    assert r.false_negatives == 1
    assert r.expected_cost == 110


def test_optimized_policy_returns_threshold():
    y = [0]*95 + [1]*5
    p = [0.01]*90 + [0.4]*5 + [0.2]*2 + [0.8]*3
    r = optimize_threshold(y, p, fraud_miss_cost=100, false_decline_cost=5)
    assert 0 <= r.threshold <= 1


def test_psi_low_for_same_distribution():
    x = np.linspace(0, 1, 1000)
    assert population_stability_index(x, x) < 1e-9


def test_psi_positive_for_shift():
    ref = np.linspace(0, 1, 1000)
    cur = np.linspace(0.5, 1.5, 1000)
    assert population_stability_index(ref, cur) > 0
