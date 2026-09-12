from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PolicyResult:
    threshold: float
    expected_cost: float
    false_positives: int
    false_negatives: int


def evaluate_threshold(y_true, prob, threshold: float, fraud_miss_cost: float, false_decline_cost: float) -> PolicyResult:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(prob)
    pred = (p >= threshold).astype(int)
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    cost = fn * fraud_miss_cost + fp * false_decline_cost
    return PolicyResult(float(threshold), float(cost), fp, fn)


def optimize_threshold(y_true, prob, fraud_miss_cost: float, false_decline_cost: float) -> PolicyResult:
    candidates = np.unique(np.quantile(np.asarray(prob), np.linspace(0.70, 0.999, 250)))
    results = [
        evaluate_threshold(y_true, prob, t, fraud_miss_cost, false_decline_cost)
        for t in candidates
    ]
    return min(results, key=lambda r: r.expected_cost)
