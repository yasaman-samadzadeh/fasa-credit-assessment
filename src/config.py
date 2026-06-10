
from pathlib import Path

from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold

#  Paths 
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

#  Schema
ID = "company_id"
TARGET = "defaulted"
TEXT = "business_description"

# Raw financials only. Evidence (notebook 03, ablation A)=
NUMERIC = [
    "revenue_m",
    "ebitda_margin",
    "debt_ratio",
    "interest_coverage",
    "cash_ratio",
    "years_in_operation",
    "employee_count",
    "revenue_growth",
]
CATEGORICAL = ["sector", "country"]   #country name is excluded during the training


#  Modelling 
RANDOM_STATE = 42

# L2-regularised logistic regression, C=0.05, *no* class weighting.
# Evidence: ablation B — LR (0.803) beats every booster/kernel/NN family tried

LR_C = 0.05

# 5x3 repeated stratified CV: stable mean +/- std on 1,000 rows, preserves the
# 17.4% default rate in every fold. 
def make_cv() -> RepeatedStratifiedKFold:
    return RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)


# Single-repeat CV used where one clean out-of-fold prediction per row is
# needed (threshold selection, band validation, error analysis).
def make_single_cv() -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


# Decision policy 
# Cost asymmetry for the alert flag: missing a real default (lost principal) is
# assumed 5x as costly as manually reviewing a healthy borrower. it can be also
# rerun with different cost ratios.
COST_FN = 5.0
COST_FP = 1.0

# Risk-rating cutoffs on the predicted default probability. Derived from the
# out-of-fold probabilities and validated in run.py
RATING_CUTOFFS = {"Low": 0.10, "Medium": 0.25}  # High = everything above

RATING_ORDER = ["Low", "Medium", "High"]
