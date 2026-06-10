"""Decision layer: turning a calibrated probability into actions.

Two separate instruments, because they answer two different questions:

1. Risk rating (Low / Medium / High) — a communication device for the analyst
   memo, defined by fixed cutoffs on the predicted default probability and
   validated against observed out-of-fold default rates per band.
2. Alert threshold — an operating point for "send to manual review", chosen to
   minimise expected cost under an asymmetric FN:FP cost ratio (a missed
   default costs principal; a false alarm costs review effort).

The threshold is selected *out-of-fold*: for each CV fold, the cost sweep runs
on the other four folds and the chosen threshold is evaluated on the held-out
fold (review finding S2 — selecting and scoring a threshold on the same rows
is mildly optimistic). The deployed threshold is the mean of the five choices,
and we report its fold-level dispersion as a stability check.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COST_FN, COST_FP, RATING_CUTOFFS, RATING_ORDER


def to_rating(p: float) -> str:
    if p < RATING_CUTOFFS["Low"]:
        return "Low"
    if p < RATING_CUTOFFS["Medium"]:
        return "Medium"
    return "High"


def rating_validation(y: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    """Observed default rate per rating band on OOF predictions.

    The bands are only legitimate if observed risk rises monotonically across
    Low -> Medium -> High; run.py asserts this before writing the deliverable.
    """
    bands = pd.Series([to_rating(p) for p in probs], name="risk_rating")
    table = (
        pd.DataFrame({"risk_rating": bands, "defaulted": y})
        .groupby("risk_rating", observed=True)
        .agg(companies=("defaulted", "size"), observed_default_rate=("defaulted", "mean"))
        .reindex(RATING_ORDER)
    )
    return table


def expected_cost(y: np.ndarray, probs: np.ndarray, threshold: float,
                  cost_fn: float = COST_FN, cost_fp: float = COST_FP) -> float:
    pred = probs >= threshold
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return fn * cost_fn + fp * cost_fp


def select_threshold(y: np.ndarray, probs: np.ndarray, fold_ids: np.ndarray,
                     grid: np.ndarray | None = None) -> dict:
    """Out-of-fold cost-minimising threshold (see module docstring)."""
    if grid is None:
        grid = np.linspace(0.02, 0.95, 187)

    per_fold = []
    for k in np.unique(fold_ids):
        sel_mask = fold_ids != k  # threshold chosen on the other folds...
        costs = [expected_cost(y[sel_mask], probs[sel_mask], t) for t in grid]
        t_k = float(grid[int(np.argmin(costs))])
        per_fold.append({
            "fold": int(k),
            "threshold": t_k,
            # ...and evaluated on the held-out fold
            "heldout_cost": expected_cost(y[fold_ids == k], probs[fold_ids == k], t_k),
        })

    folds = pd.DataFrame(per_fold)
    threshold = float(folds["threshold"].mean())
    pred = probs >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return {
        "threshold": threshold,
        "threshold_std": float(folds["threshold"].std()),
        "per_fold": folds,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "flagged_share": float(pred.mean()),
    }
