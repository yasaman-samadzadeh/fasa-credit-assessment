# fasa-credit-assessment

Fasanara AI Credit Risk Analyst Challenge — solution by Yasaman Samadzadeh.

A prototype credit-risk screening tool: it trains a probability-of-default model on
1,000 labelled companies, scores 50 unseen companies, and writes an analyst-style memo
for each one to `outputs/predictions.csv`
(`company_id, predicted_default_probability, risk_rating, explanation`).

## Quick start

```bash
pip install uv
uv sync                        # Python 3.12 + all dependencies into .venv

uv run python run.py           # the deliverable: validate -> evaluate -> score -> explain
uv run python run.py --llm    # optional: LLM-polished memo prose (key in .env, safe fallback)
uv run streamlit run app.py    # review UI: portfolio, drill-down, policy lab
uv run pytest                  # decision-policy & explanation invariants
```

The pipeline is deterministic (`random_state=42` everywhere) and fully offline by
default — no API keys needed for the core deliverable.

## How it works

```
data/*.csv ──► src/data.py ──► src/model.py ─────► src/policy.py ──► outputs/
              validation      impute+scale+OHE     risk ratings +     predictions.csv
              KS drift check  logistic regression  alert threshold
                                   │
                                   └─► src/explain.py ──► per-company analyst memos
                                        (exact linear attributions + narrative signals,
                                         optional LLM polish via src/llm.py)
```

1. **Join & validate** the three training files 1:1 on `company_id`.
2. **Model**: logistic regression (L2, C=0.05, unweighted) on the 8 raw financial
   ratios + one-hot `sector`/`country`; preprocessing inside the pipeline (leak-free).
3. **Evaluate**: `RepeatedStratifiedKFold(5×3)` → **ROC-AUC 0.806 ± 0.034**,
   PR-AUC 0.531 (vs 0.17 base rate), Brier 0.114.
4. **Decide**: probability → Low/Medium/High rating (cutoffs 0.10/0.25, validated
   monotone out-of-fold: observed default 4.3% → 15.4% → 43.6%) + a cost-based alert
   threshold (FN:FP = 5:1 → 0.175 ± 0.019, recall 0.75) selected out-of-fold.
5. **Explain**: each memo decomposes the company's own score (`coef × standardized
   value` — exact for a linear model), quantifies drivers against portfolio medians,
   and quotes the narrative's qualitative signals. A materiality floor stops the memo
   from inventing concerns for median companies.

## The story in notebooks

| Notebook | Question it answers |
|---|---|
| `notebooks/01_eda.ipynb` | What does the data say? Imbalance, drivers, the narrative-text verdict, dataset traps |
| `notebooks/02_experiments.ipynb` | Why this design? Ablations over features, model families, imbalance handling, calibration — one shared CV protocol |
| `notebooks/03_results.ipynb` | How good is it? OOF diagnostics, interpretability, decision policy, scored portfolio, limitations |

Notebooks import `src/` and never re-declare constants, so the narrative can never
drift from the deliverable.

## Design decisions & evidence

All numbers from `02_experiments.ipynb` (RepeatedStratifiedKFold 5×3, identical
features/CV per comparison).

| Decision | Evidence |
|---|---|
| **Logistic regression** over non-linear models | LR 0.806 ROC-AUC vs RandomForest 0.758, GradientBoosting 0.751, HistGB 0.719, SVM 0.747, KNN 0.753, NB 0.667 — the risk function is ~linear, so the best model is also the most auditable. |
| **Raw financials only** (no engineered ratios) | Raw 0.806 vs +engineered 0.800 — within noise; parsimony and interpretability win. |
| **Narratives excluded from the model** | Phrase flags 0.453 / MiniLM embeddings 0.488 AUC *alone* (≈ chance); adding them to the financials hurts (−0.02 to −0.06). The templated phrases occur largely independently of the label. Text goes to the explanation layer, where it adds analyst value. |
| **No class weighting / no resampling** | Unweighted 0.806 AUC, **Brier 0.114** vs weighted 0.803, Brier 0.178 — weighting distorts the probabilities for zero ranking gain. Imbalance is handled at the decision layer. |
| **No calibration wrapper** | Sigmoid/isotonic change Brier by ±0.001 — the unweighted model is already calibrated. |
| **No hyperparameter tuning** | A randomized search moved LR by ~0.002 AUC (within fold noise; the search score is selection-biased). Fixed C=0.05. |
| **Threshold selected out-of-fold** | Per fold: swept on the other four, evaluated held-out → 0.175 ± 0.019 (stable). Never graded on the rows that chose it. |
| **Deterministic memos by default** | Exact linear attributions cannot hallucinate numbers and run offline. `--llm` adds prose polish with the rating/probability validated verbatim and per-memo fallback. |
| **`company_id` only; `company_name` dropped** | Names are recycled between train and scoring with different identities — decoration, and a trap if used as a key. |

## Repository layout

```
run.py                   # entrypoint -> outputs/predictions.csv (+ printed report)
app.py                   # Streamlit review UI (portfolio / drill-down / policy lab)
src/
  config.py              # single source of truth: features, CV, cutoffs, costs
  data.py                # schema validation, 1:1 joins, KS drift check
  model.py               # pipeline + cross-validated evaluation
  policy.py              # ratings + out-of-fold cost-based alert threshold
  explain.py             # exact per-company attributions -> analyst memos
  llm.py                 # optional grounded LLM polish (.env keys, safe fallback)
notebooks/               # 01 EDA -> 02 experiments -> 03 results (all import src/)
tests/                   # policy & explanation invariants (8 tests)
outputs/predictions.csv  # the deliverable (50 companies)
```

## Bonus-idea coverage

Every optional enhancement from the brief, and where it lives:

| Bonus idea | Where |
|---|---|
| Text features from business descriptions | Tested as phrase flags — `02_experiments.ipynb` ablation A; used qualitatively in every memo (`src/explain.narrative_signals`) |
| Embeddings / richer NLP features | MiniLM sentence embeddings tested (alone and combined) — `02_experiments.ipynb` ablation A; verdict: no label signal |
| LLM / prompt-based explanations | `src/llm.py` + `run.py --llm`: grounded rewrite of the deterministic memo, facts validated verbatim, per-memo fallback |
| Streamlit app | `app.py` — portfolio, company drill-down with score decomposition, interactive policy lab |
| Model evaluation | RepeatedStratifiedKFold(5×3) leaderboard, OOF ROC/PR/reliability curves, Brier, band validation — `02`/`03` notebooks + `run.py` report |
| Feature importance / explainability | Global: coefficients + permutation importance (`03_results.ipynb`); local: exact per-company attributions in every memo |
| Project structure & reusable functions | `src/` package with single-source-of-truth config; notebooks and app import it, never re-declare |
| Input validation & error handling | `src/data.py`: schema/uniqueness/missingness/join-cardinality checks, unseen-category warnings, KS drift test; LLM layer fails safe |

## Validation & guardrails

- Boundary validation: required columns, unique IDs, no missing values, binary target,
  1:1 join cardinality (`src/data.py`).
- Train→scoring checks: unseen-category warning + per-feature KS drift test.
  *Finding: `ebitda_margin` shifts (KS 0.27, p<0.01 — scoring companies are more
  profitable on average); reported as model risk in the run output.*
- Rating bands asserted monotone in observed default rate before the deliverable is
  written; memo attributions tested to reconstruct the model's score exactly.

## Limitations & next steps

- **Synthetic data, no time dimension** — out-of-time validation impossible here; in
  production: validate on a future cohort, monitor drift (the `ebitda_margin` shift is
  exactly the signal to watch).
- **AUC ceiling ≈ 0.81** — the residual is deliberate label noise; chasing it would
  mean overfitting the generator, not learning credit risk.
- **Cost ratio (5:1) is an assumption** — the Streamlit *policy lab* re-derives the
  threshold live under any committee-supplied ratio.
- **Memos are templated by default** — grounded but uniform; `--llm` adds fluent prose
  with facts pinned. With more time: human-in-the-loop review queue and richer
  qualitative ingestion (filings, news) for the narrative layer.

## AI tool usage

Built with AI-assisted coding (Cursor). I own the problem framing, metric choice,
ablation design, decision-policy structure and the final model selection; AI tools
accelerated boilerplate, experiment execution and documentation, with every output
reviewed before inclusion.

---

### Original setup instructions

1. Fork this repository, clone your fork, create a branch.
2. `pip install uv && uv sync` — installs dependencies from `pyproject.toml`
   (Python 3.12+).
