"""Streamlit review UI for the credit-risk prototype.

    uv run streamlit run app.py

Three views for a credit committee:
1. Portfolio  — the 50 scored companies, filterable, downloadable.
2. Company    — one borrower's memo, score decomposition and narrative.
3. Policy lab — re-derive the alert threshold live under your own FN:FP costs.

The app contains no modelling logic of its own: it imports the same src/ modules
as run.py and the notebooks, so what you see here is exactly what ships in
outputs/predictions.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.config import CATEGORICAL, ID, NUMERIC, OUT_DIR, TARGET, TEXT
from src.data import load_scoring, load_training
from src.explain import build_all_explanations, contributions, narrative_signals
from src.model import cv_metrics, make_model, oof_predictions
from src.policy import expected_cost, rating_validation, to_rating

st.set_page_config(page_title="Fasanara Credit Risk", layout="wide")

BAND_COLOR = {"Low": "#55a868", "Medium": "#dd8452", "High": "#c44e52"}


@st.cache_data
def get_data():
    train = load_training()
    scoring = load_scoring()
    return train, scoring


@st.cache_resource
def get_artifacts():
    """Train once per session: fitted pipeline, OOF arrays, CV metrics, scored portfolio."""
    train, scoring = get_data()
    X, y = train[NUMERIC + CATEGORICAL], train[TARGET]

    pipe = make_model()
    metrics = cv_metrics(pipe, X, y)
    probs_oof, fold_ids = oof_predictions(pipe, X, y)
    pipe.fit(X, y)

    p_scoring = pipe.predict_proba(scoring[NUMERIC + CATEGORICAL])[:, 1]
    ratings = [to_rating(p) for p in p_scoring]
    memos = build_all_explanations(pipe, scoring, train, p_scoring, ratings)

    # If run.py already produced the deliverable (possibly with --llm polish),
    # display those memos so the app matches outputs/predictions.csv exactly.
    pred_path = OUT_DIR / "predictions.csv"
    if pred_path.exists():
        shipped = pd.read_csv(pred_path).set_index(ID)["explanation"]
        if set(scoring[ID]) == set(shipped.index):
            memos = shipped.loc[scoring[ID]].tolist()

    contrib = contributions(pipe, scoring[NUMERIC + CATEGORICAL])
    return pipe, metrics, probs_oof, fold_ids, p_scoring, ratings, memos, contrib


train, scoring = get_data()
pipe, metrics, probs_oof, fold_ids, p_scoring, ratings, memos, contrib = get_artifacts()
y = train[TARGET].to_numpy()

st.title("AI Credit Risk Analyst — prototype")
st.caption(
    f"Logistic regression on 8 financial ratios + sector/country · "
    f"CV ROC-AUC **{metrics['roc_auc']:.3f} ± {metrics['roc_auc_std']:.3f}** · "
    f"PR-AUC {metrics['pr_auc']:.3f} · Brier {metrics['brier']:.3f} · "
    f"trained on {len(train)} companies ({train[TARGET].mean():.1%} default rate)"
)

tab_portfolio, tab_company, tab_policy = st.tabs(
    ["Portfolio", "Company drill-down", "Policy lab"])

# --- 1. Portfolio -------------------------------------------------------------
with tab_portfolio:
    table = scoring[[ID, "company_name", "sector", "country"]].copy()
    table["P(default)"] = np.round(p_scoring, 4)
    table["rating"] = ratings

    pick = st.multiselect("Filter rating", ["Low", "Medium", "High"],
                          default=["Low", "Medium", "High"])
    view = (table[table["rating"].isin(pick)]
            .sort_values("P(default)", ascending=False)
            .reset_index(drop=True))

    c1, c2, c3 = st.columns(3)
    for col, band in zip((c1, c2, c3), ("Low", "Medium", "High")):
        col.metric(f"{band} risk", int((table["rating"] == band).sum()))

    st.dataframe(
        view.style.background_gradient(subset=["P(default)"], cmap="Reds", vmin=0, vmax=0.7),
        use_container_width=True, height=420)

    deliverable = pd.DataFrame({
        ID: scoring[ID],
        "predicted_default_probability": np.round(p_scoring, 4),
        "risk_rating": ratings,
        "explanation": memos,
    })
    st.download_button("Download predictions.csv",
                       deliverable.to_csv(index=False), "predictions.csv")

# --- 2. Company drill-down -----------------------------------------------------
with tab_company:
    labels = [f"{r[ID]} — {r['company_name']} ({r['sector']})"
              for _, r in scoring.iterrows()]
    idx = st.selectbox("Company", range(len(labels)), format_func=lambda i: labels[i])
    row = scoring.iloc[idx]
    rating = ratings[idx]

    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted default probability", f"{p_scoring[idx]:.1%}")
        st.markdown(
            f"<span style='background:{BAND_COLOR[rating]};color:white;"
            f"padding:4px 14px;border-radius:6px;font-weight:bold'>{rating} risk</span>",
            unsafe_allow_html=True)
        st.write("")
        fin = row[NUMERIC].to_frame("value")
        fin["portfolio median"] = train[NUMERIC].median()
        st.dataframe(fin.round(3), use_container_width=True)

    with c2:
        st.subheader("Analyst memo")
        st.info(memos[idx])

        st.subheader("Score decomposition (log-odds contributions)")
        num_contrib = pd.Series(
            {f: contrib.iloc[idx][f"num__{f}"] for f in NUMERIC}).sort_values()
        st.bar_chart(num_contrib, horizontal=True,
                     color="#c44e52" if num_contrib.iloc[-1] > 0 else "#4c72b0")

        risks, positives = narrative_signals(row[TEXT])
        st.subheader("Narrative")
        st.write(row[TEXT])
        if risks:
            st.write("Risk signals: " + " · ".join(f"`{r}`" for r in risks))
        if positives:
            st.write("Strengths: " + " · ".join(f"`{p}`" for p in positives))

# --- 3. Policy lab --------------------------------------------------------------
with tab_policy:
    st.write(
        "The alert threshold is a *business* decision: it trades missed defaults "
        "(lost principal) against review effort. Set your own cost ratio — the sweep "
        "below runs on out-of-fold predictions, never on training fits.")

    ratio = st.slider("Cost of a missed default, in multiples of a false alarm",
                      1.0, 20.0, 5.0, 0.5)
    grid = np.linspace(0.02, 0.95, 187)
    costs = [expected_cost(y, probs_oof, t, cost_fn=ratio, cost_fp=1.0) for t in grid]
    best_t = float(grid[int(np.argmin(costs))])

    pred = probs_oof >= best_t
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Optimal threshold", f"{best_t:.2f}")
    c2.metric("Defaults caught", f"{tp / (tp + fn):.0%}")
    c3.metric("Alert precision", f"{tp / (tp + fp):.0%}")
    c4.metric("Book flagged", f"{pred.mean():.0%}")

    st.line_chart(pd.DataFrame({"expected cost": costs}, index=grid))

    st.subheader("Rating bands (fixed, validated out-of-fold)")
    st.dataframe(rating_validation(y, probs_oof)
                 .style.format({"observed_default_rate": "{:.1%}"}))
