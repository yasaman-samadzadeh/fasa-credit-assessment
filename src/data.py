"""Loading, validation and joining of the provided CSVs.
"""
from __future__ import annotations

import warnings

import pandas as pd
from scipy.stats import ks_2samp

from .config import CATEGORICAL, DATA_DIR, ID, NUMERIC, TARGET, TEXT


class DataValidationError(ValueError):
    """Raised when an input file violates the expected schema."""


def _require_columns(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing required columns {missing}")


def _require_unique_ids(df: pd.DataFrame, name: str) -> None:
    if df[ID].duplicated().any():
        dupes = df.loc[df[ID].duplicated(), ID].tolist()
        raise DataValidationError(f"{name}: duplicate {ID} values {dupes[:5]}")


def _check_missing(df: pd.DataFrame, cols: list[str], name: str) -> None:
    na = df[cols].isna().sum()
    bad = na[na > 0]
    if not bad.empty:
        raise DataValidationError(f"{name}: unexpected missing values:\n{bad}")


def load_training() -> pd.DataFrame:
    """Load and join the three training files on company_id (1:1 joins, validated)."""
    companies = pd.read_csv(DATA_DIR / "train_companies.csv")
    narratives = pd.read_csv(DATA_DIR / "train_narratives.csv")
    outcomes = pd.read_csv(DATA_DIR / "train_outcomes.csv")

    _require_columns(companies, [ID] + NUMERIC + CATEGORICAL, "train_companies")
    _require_columns(narratives, [ID, TEXT], "train_narratives")
    _require_columns(outcomes, [ID, TARGET], "train_outcomes")
    for df, name in [(companies, "train_companies"), (narratives, "train_narratives"),
                     (outcomes, "train_outcomes")]:
        _require_unique_ids(df, name)

    train = (
        companies
        .merge(narratives, on=ID, how="inner", validate="1:1")
        .merge(outcomes, on=ID, how="inner", validate="1:1")
        .sort_values(ID)
        .reset_index(drop=True)
    )
    if len(train) != len(companies):
        raise DataValidationError(
            f"join dropped rows: {len(companies)} companies -> {len(train)} joined"
        )

    _check_missing(train, NUMERIC + CATEGORICAL + [TARGET], "train (joined)")
    if not set(train[TARGET].unique()) <= {0, 1}:
        raise DataValidationError(f"target must be binary 0/1, got {train[TARGET].unique()}")
    return train


def load_scoring(train: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load the scoring file; if `train` is given, also run consistency checks."""
    scoring = pd.read_csv(DATA_DIR / "scoring_companies.csv")
    _require_columns(scoring, [ID] + NUMERIC + CATEGORICAL + [TEXT], "scoring_companies")
    _require_unique_ids(scoring, "scoring_companies")
    _check_missing(scoring, NUMERIC + CATEGORICAL, "scoring_companies")

    if train is not None:
        check_consistency(train, scoring)
    return scoring.sort_values(ID).reset_index(drop=True)


def check_consistency(train: pd.DataFrame, scoring: pd.DataFrame) -> pd.DataFrame:
   
    for col in CATEGORICAL:
        unseen = set(scoring[col]) - set(train[col])
        if unseen:
            warnings.warn(f"scoring has unseen {col} values {sorted(unseen)}; "
                          f"their one-hot columns will be all-zero", stacklevel=2)

    rows = []
    for col in NUMERIC:
        stat, pval = ks_2samp(train[col], scoring[col])
        rows.append({"feature": col, "ks_stat": stat, "p_value": pval})
    drift = pd.DataFrame(rows).sort_values("ks_stat", ascending=False).reset_index(drop=True)

    flagged = drift[drift["p_value"] < 0.01]
    if not flagged.empty:
        warnings.warn(
            "possible train->scoring distribution shift (KS p<0.01) in: "
            f"{flagged['feature'].tolist()}", stacklevel=2,
        )
    return drift
