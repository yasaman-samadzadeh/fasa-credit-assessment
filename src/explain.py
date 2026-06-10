"""Per-company analyst-style explanations, grounded in the model itself.

Design choice: the explanations are *deterministic* and derived directly from
the fitted linear model. For a logistic regression, the exact contribution of
feature j to a company's log-odds is coef_j x standardized_value_j

"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import CATEGORICAL, NUMERIC, TEXT

_FMT = {
    "revenue_m": lambda v: f"{v:.0f}m",
    "ebitda_margin": lambda v: f"{v:.1%}",
    "debt_ratio": lambda v: f"{v:.2f}",
    "interest_coverage": lambda v: f"{v:.1f}x",
    "cash_ratio": lambda v: f"{v:.2f}",
    "years_in_operation": lambda v: f"{v:.0f} year" + ("" if round(v) == 1 else "s"),
    "employee_count": lambda v: f"{v:.0f}",
    "revenue_growth": lambda v: f"{v:+.1%}",
}

_PHRASES = {
    "revenue_m": ("small revenue base ({v} vs portfolio median {m})",
                  "sizeable revenue base ({v})"),
    "ebitda_margin": ("thin EBITDA margin ({v} vs median {m})",
                      "healthy EBITDA margin ({v})"),
    "debt_ratio": ("elevated leverage (debt ratio {v} vs median {m})",
                   "modest leverage (debt ratio {v})"),
    "interest_coverage": ("weak interest coverage ({v} vs median {m})",
                          "comfortable interest coverage ({v})"),
    "cash_ratio": ("tight liquidity (cash ratio {v} vs median {m})",
                   "solid liquidity (cash ratio {v})"),
    "years_in_operation": ("short operating history ({v})",
                           "long operating history ({v})"),
    "employee_count": ("small workforce ({v} employees)",
                       "established workforce ({v} employees)"),
    "revenue_growth": ("declining revenue ({v} YoY)",
                       "growing revenue ({v} YoY)"),
}


def contributions(pipe: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Signed log-odds contribution of every feature for every row.

    Exact for a linear model: contribution_ij = coef_j * transformed_value_ij.
    Positive = pushes the default probability up.
    """
    prep = pipe.named_steps["prep"]
    coefs = pipe.named_steps["clf"].coef_[0]
    Xt = prep.transform(X)
    names = prep.get_feature_names_out()
    return pd.DataFrame(Xt * coefs, columns=names, index=X.index)


def narrative_signals(text: str) -> tuple[list[str], list[str]]:
    """Extract the templated qualitative signals from a business description."""
    risks = [m.strip().rstrip(".") for m in re.findall(r"faces ([^.]+)", text)]
    positives = [m.strip().rstrip(".") for m in re.findall(r"benefits from ([^.]+)", text)]
    return risks, positives


def _numeric_phrase(feature: str, value: float, median: float, risk_up: bool) -> str:
    up, down = _PHRASES[feature]
    fmt = _FMT[feature]
    template = up if risk_up else down
    return template.format(v=fmt(value), m=fmt(median))


# Minimum |log-odds contribution| for a feature to be quoted in a memo. 0.10
# log-odds moves a 17% base-rate PD by roughly +/-1.5pp — anything smaller is
# portfolio-average noise and citing it would mislead the reader (a company at
# the median would otherwise get "concerns" invented for it).
MATERIALITY = 0.10


def build_explanation(row: pd.Series, contrib: pd.Series, medians: pd.Series,
                      probability: float, rating: str, top_k: int = 3) -> str:
    """One analyst memo for one company."""
    num_contrib = {f: contrib[f"num__{f}"] for f in NUMERIC}
    cat_contrib = {c: sum(v for n, v in contrib.items()
                          if n.startswith(f"cat__{c}_")) for c in CATEGORICAL}

    drivers = sorted(num_contrib.items(), key=lambda kv: kv[1], reverse=True)
    risk_drivers = [(f, c) for f, c in drivers if c > MATERIALITY][:top_k]
    mitigants = [(f, c) for f, c in reversed(drivers) if c < -MATERIALITY][:1]

    parts = []
    if risk_drivers:
        phrases = [_numeric_phrase(f, row[f], medians[f], risk_up=True)
                   for f, _ in risk_drivers]
        parts.append("Key concerns: " + "; ".join(phrases) + ".")
    else:
        parts.append("No material financial weaknesses against the portfolio.")
    if cat_contrib["sector"] > 0.05:
        parts.append(f"The {row['sector']} sector carries an above-average historical default rate.")
    if mitigants:
        phrases = [_numeric_phrase(f, row[f], medians[f], risk_up=False)
                   for f, _ in mitigants]
        parts.append("Main mitigant: " + "; ".join(phrases) + ".")

    risks, positives = narrative_signals(row[TEXT])
    if risks:
        parts.append("Narrative flags " + ", ".join(risks) + ".")
    if positives:
        parts.append("Narrative cites " + ", ".join(positives) + ".")

    headline = f"{rating} risk: estimated default probability {probability:.0%}."
    return headline + " " + " ".join(parts)


def build_all_explanations(pipe: Pipeline, scoring: pd.DataFrame, train: pd.DataFrame,
                           probabilities: np.ndarray, ratings: list[str]) -> list[str]:
    """Memos for every scoring company; medians come from the training portfolio."""
    medians = train[NUMERIC].median()
    contrib = contributions(pipe, scoring[NUMERIC + CATEGORICAL])
    return [
        build_explanation(scoring.iloc[i], contrib.iloc[i], medians,
                          probabilities[i], ratings[i])
        for i in range(len(scoring))
    ]
