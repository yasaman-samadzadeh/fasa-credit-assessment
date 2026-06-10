"""Decision-policy invariants: ratings, costs, and out-of-fold threshold selection."""
import numpy as np
import pytest

from src.config import RATING_CUTOFFS
from src.policy import expected_cost, rating_validation, select_threshold, to_rating


def test_rating_boundaries():
    eps = 1e-9
    assert to_rating(0.0) == "Low"
    assert to_rating(RATING_CUTOFFS["Low"] - eps) == "Low"
    assert to_rating(RATING_CUTOFFS["Low"]) == "Medium"
    assert to_rating(RATING_CUTOFFS["Medium"] - eps) == "Medium"
    assert to_rating(RATING_CUTOFFS["Medium"]) == "High"
    assert to_rating(1.0) == "High"


def test_expected_cost_counts_fn_and_fp():
    y = np.array([1, 1, 0, 0])
    probs = np.array([0.9, 0.1, 0.9, 0.1])
    # threshold 0.5: one TP, one FN (p=0.1 default), one FP (p=0.9 healthy), one TN
    assert expected_cost(y, probs, 0.5, cost_fn=5.0, cost_fp=1.0) == 5.0 + 1.0


def test_expected_cost_extreme_thresholds():
    y = np.array([1, 0, 0, 0])
    probs = np.array([0.8, 0.2, 0.3, 0.4])
    # threshold ~0: everything flagged -> only FP costs; threshold ~1: only FN costs
    assert expected_cost(y, probs, 0.0, cost_fn=5.0, cost_fp=1.0) == 3.0
    assert expected_cost(y, probs, 1.01, cost_fn=5.0, cost_fp=1.0) == 5.0


def test_select_threshold_is_out_of_fold_and_stable():
    rng = np.random.default_rng(0)
    n = 500
    y = (rng.random(n) < 0.2).astype(int)
    # informative but noisy probabilities
    probs = np.clip(0.2 * y + rng.normal(0.15, 0.1, n), 0.01, 0.99)
    fold_ids = np.arange(n) % 5

    result = select_threshold(y, probs, fold_ids)
    assert 0.0 < result["threshold"] < 1.0
    assert len(result["per_fold"]) == 5
    assert 0.0 <= result["recall"] <= 1.0
    # the same inputs must give the same threshold (deterministic policy)
    again = select_threshold(y, probs, fold_ids)
    assert again["threshold"] == pytest.approx(result["threshold"])


def test_rating_validation_orders_bands():
    probs = np.array([0.02, 0.05, 0.15, 0.20, 0.40, 0.60])
    y = np.array([0, 0, 0, 1, 1, 1])
    table = rating_validation(y, probs)
    assert list(table.index) == ["Low", "Medium", "High"]
    assert table.loc["Low", "companies"] == 2
    assert table.loc["High", "observed_default_rate"] == 1.0
