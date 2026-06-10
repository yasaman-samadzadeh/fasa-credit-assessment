"""Preprocessing + model pipeline and its cross-validated evaluation.

Model choice (evidence in notebooks/03_experiments.ipynb):

"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import CATEGORICAL, LR_C, NUMERIC, RANDOM_STATE, make_cv, make_single_cv


def make_model() -> Pipeline:
    """The full scoring pipeline: impute -> scale -> one-hot -> logistic regression.

    Imputers are included although the provided files are complete: the pipeline
    is the production boundary and future scoring batches may have gaps.
    """
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(C=LR_C, max_iter=3000, random_state=RANDOM_STATE)),
    ])


def cv_metrics(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Mean +/- std over RepeatedStratifiedKFold(5x3) for the headline metrics."""
    scores = cross_validate(
        pipe, X, y, cv=make_cv(), n_jobs=-1,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                 "brier": "neg_brier_score"},
    )
    return {
        "roc_auc": scores["test_roc_auc"].mean(),
        "roc_auc_std": scores["test_roc_auc"].std(),
        "pr_auc": scores["test_pr_auc"].mean(),
        "pr_auc_std": scores["test_pr_auc"].std(),
        "brier": -scores["test_brier"].mean(),
        "n_fits": len(scores["test_roc_auc"]),
    }


def oof_predictions(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """One leak-free out-of-fold probability per training row, plus fold ids.

    Single-repeat stratified CV (a row can only have one OOF prediction). Fold
    ids let the decision policy select its threshold out-of-fold too (review
    finding S2): chosen on four folds, evaluated on the fifth.
    """
    cv = make_single_cv()
    probs = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    fold_ids = np.empty(len(y), dtype=int)
    for k, (_, val_idx) in enumerate(cv.split(X, y)):
        fold_ids[val_idx] = k
    return probs, fold_ids


def oof_report(y: pd.Series, probs: np.ndarray) -> dict:
    """Headline metrics on the OOF predictions (single split, for reference)."""
    return {
        "roc_auc": roc_auc_score(y, probs),
        "pr_auc": average_precision_score(y, probs),
        "brier": brier_score_loss(y, probs),
        "base_rate": float(np.mean(y)),
    }
