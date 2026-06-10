"""Explanation-layer invariants: memos must be grounded, material and well-formed."""
import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL, NUMERIC, TARGET
from src.data import load_scoring, load_training
from src.explain import build_all_explanations, contributions, narrative_signals
from src.model import make_model
from src.policy import to_rating


@pytest.fixture(scope="module")
def fitted():
    train = load_training()
    scoring = load_scoring()
    pipe = make_model()
    pipe.fit(train[NUMERIC + CATEGORICAL], train[TARGET])
    probs = pipe.predict_proba(scoring[NUMERIC + CATEGORICAL])[:, 1]
    return train, scoring, pipe, probs


def test_narrative_signals_extraction():
    text = ("Acme operates in the retail sector. The business faces declining revenue. "
            "The business faces tight cash position. The company benefits from "
            "strong market position. Management focuses on efficiency.")
    risks, positives = narrative_signals(text)
    assert risks == ["declining revenue", "tight cash position"]
    assert positives == ["strong market position"]


def test_contributions_sum_to_logit(fitted):
    """The decomposition must reconstruct the model's own score exactly."""
    train, scoring, pipe, probs = fitted
    contrib = contributions(pipe, scoring[NUMERIC + CATEGORICAL])
    intercept = pipe.named_steps["clf"].intercept_[0]
    logit = np.log(probs / (1 - probs))
    np.testing.assert_allclose(contrib.sum(axis=1) + intercept, logit, atol=1e-8)


def test_memos_grounded_and_complete(fitted):
    train, scoring, pipe, probs = fitted
    ratings = [to_rating(p) for p in probs]
    memos = build_all_explanations(pipe, scoring, train, probs, ratings)

    assert len(memos) == len(scoring)
    for memo, p, rating in zip(memos, probs, ratings):
        assert memo.startswith(f"{rating} risk:")
        assert f"{p:.0%}" in memo          # the probability is quoted verbatim
        assert len(memo) > 60              # never an empty/stub explanation
