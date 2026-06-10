"""End-to-end pipeline for the Fasanara AI Credit Risk Analyst Challenge.

    uv run python run.py            # fully offline, deterministic
    uv run python run.py --llm     # additionally polish memo prose via an LLM
                                    
Steps: validate data -> cross-validated evaluation -> decision-policy checks ->
fit on all training data -> score the 50 unseen companies -> write
outputs/predictions.csv with the required schema:

    company_id, predicted_default_probability, risk_rating, explanation

Every design choice is documented where it is implemented (src/*.py) and
evidenced in notebooks/02_experiments.ipynb.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from src.config import CATEGORICAL, ID, NUMERIC, OUT_DIR, TARGET
from src.data import check_consistency, load_scoring, load_training
from src.explain import build_all_explanations
from src.model import cv_metrics, make_model, oof_predictions, oof_report
from src.policy import rating_validation, select_threshold, to_rating

RULE = "-" * 72


def main(use_llm: bool = False) -> int:
    # 1. Data
    print(RULE)
    print("1. Loading & validating data")
    train = load_training()
    scoring = load_scoring()
    drift = check_consistency(train, scoring)
    print(f"   train: {len(train)} companies, default rate {train[TARGET].mean():.1%}")
    print(f"   scoring: {len(scoring)} companies")
    print(f"   largest train->scoring KS drift: "
          f"{drift.iloc[0]['feature']} (stat={drift.iloc[0]['ks_stat']:.3f}, "
          f"p={drift.iloc[0]['p_value']:.2f})")

    X = train[NUMERIC + CATEGORICAL]
    y = train[TARGET]

    # 2. Model evaluation
    print(RULE)
    print("2. Cross-validated performance (RepeatedStratifiedKFold 5x3)")
    pipe = make_model()
    m = cv_metrics(pipe, X, y)
    print(f"   ROC-AUC {m['roc_auc']:.3f} +/- {m['roc_auc_std']:.3f}   "
          f"PR-AUC {m['pr_auc']:.3f} +/- {m['pr_auc_std']:.3f}   "
          f"Brier {m['brier']:.3f}   ({m['n_fits']} fits)")

    probs_oof, fold_ids = oof_predictions(pipe, X, y)
    r = oof_report(y, probs_oof)
    print(f"   OOF single-repeat reference: ROC-AUC {r['roc_auc']:.3f}, "
          f"PR-AUC {r['pr_auc']:.3f} (base rate {r['base_rate']:.2f}), "
          f"Brier {r['brier']:.3f}")

    # 3. Decision policy 
    print(RULE)
    print("3. Decision policy on out-of-fold predictions")
    bands = rating_validation(y.to_numpy(), probs_oof)
    print(bands.to_string(float_format=lambda v: f"{v:.1%}"))
    rates = bands["observed_default_rate"].to_numpy()
    if not (np.diff(rates) > 0).all():
        print("   ERROR: rating bands are not monotone in observed default rate")
        return 1
    print("   rating bands validated: observed risk is monotone Low -> Medium -> High")

    th = select_threshold(y.to_numpy(), probs_oof, fold_ids)
    print(f"   alert threshold (cost FN:FP = 5:1, chosen out-of-fold): "
          f"{th['threshold']:.3f} +/- {th['threshold_std']:.3f}")
    print(f"   at that threshold: recall {th['recall']:.2f}, "
          f"precision {th['precision']:.2f}, flags {th['flagged_share']:.0%} of book")

    # 4. Final fit & scoring 
    print(RULE)
    print("4. Fitting on all training data & scoring the unseen companies")
    pipe.fit(X, y)
    p_scoring = pipe.predict_proba(scoring[NUMERIC + CATEGORICAL])[:, 1]
    ratings = [to_rating(p) for p in p_scoring]
    explanations = build_all_explanations(pipe, scoring, train, p_scoring, ratings)

    if use_llm:
        from src.llm import polish_memos
        explanations, n_polished = polish_memos(explanations)
        print(f"   LLM polish: {n_polished}/{len(explanations)} memos rewritten "
              f"({'no API key found in .env' if n_polished == 0 else 'facts validated, fallback on failure'})")

    predictions = scoring[[ID]].copy()
    predictions["predicted_default_probability"] = np.round(p_scoring, 4)
    predictions["risk_rating"] = ratings
    predictions["explanation"] = explanations

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / "predictions.csv"
    predictions.to_csv(out_path, index=False)

    dist = predictions["risk_rating"].value_counts().reindex(["Low", "Medium", "High"])
    print(f"   wrote {len(predictions)} rows -> {out_path}")
    print(f"   rating distribution: {dist.to_dict()}")

    print(RULE)
    print("5. Sample memos (highest and lowest predicted risk)")
    ordered = predictions.sort_values("predicted_default_probability", ascending=False)
    for row in (ordered.iloc[0], ordered.iloc[-1]):
        print(f"\n   [{row[ID]}] p={row['predicted_default_probability']}: "
              f"{row['explanation']}")
    print(RULE)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true",
                        help="polish memo prose with an LLM (key in .env; safe fallback)")
    args = parser.parse_args()
    sys.exit(main(use_llm=args.llm))
