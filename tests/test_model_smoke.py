"""Smoke tests for the committed model.pkl: it loads and a prediction comes back
sane, so a library/pickle version mismatch fails here in CI rather than showing
up as a crash on demo day -- and categorical inputs are encoded correctly at
predict time.

That second part matters because the app predicts on a ONE-row frame, and
build_features turns each categorical column into a pandas `category` whose list
is just that one value (so ORD is code 0, not its training code 4). LightGBM
remaps such a frame onto the categories it stored at fit time, and these tests
pin that: a prediction must be identical however the categories happen to be
encoded, and must match a ground truth computed with no pandas involved.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.features import (
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    build_features,
)


def test_model_predicts_a_probability_between_0_and_1():
    model = joblib.load("model.pkl")

    row = pd.DataFrame(
        [
            {
                "airport": "ORD",
                "carrier_in": "AA",
                "carrier_out": "UA",
                "slack_min": 60,
                "arr_hour": 14,
                "day_of_week": 2,
                "month": 6,
                "distance_in": 800,
                TARGET_COLUMN: True,  # dummy; build_features requires the column, prediction ignores it
            }
        ]
    )
    features = build_features(row)

    prob = model.predict_proba(features[FEATURE_COLUMNS])[0, 1]
    assert 0.0 <= prob <= 1.0


# --- categorical encoding at predict time -------------------------------------

# airport, carrier_in, carrier_out, layover, arrival_hour, month, distance_in.
# Different airports and carrier pairs. A one-row frame codes every value as 0, so
# each case has at least one value whose real training code isn't 0 (ORD is 4, WN is
# 11, ...): a model that read the frame's own codes instead of remapping would give
# visibly different answers for every one of these.
CASES = [
    ("ORD", "AA", "UA", 60, 14, 6, 606),
    ("DEN", "DL", "WN", 45, 9, 12, 1200),
    ("CLT", "UA", "AA", 90, 20, 3, 800),
    ("ATL", "WN", "DL", 35, 17, 7, 300),
    ("DFW", "OO", "MQ", 120, 8, 10, 500),
]
CASE_IDS = [f"{a}-{ci}-to-{co}" for a, ci, co, *_ in CASES]


@pytest.fixture(scope="module")
def model():
    return joblib.load("model.pkl")


@pytest.fixture(scope="module")
def categories() -> dict:
    with open("model_categories.json") as f:
        return json.load(f)


def query(rows) -> pd.DataFrame:
    """Rows built and typed exactly the way the app does it (one row -> one-value categories)."""
    frame = pd.DataFrame(
        [
            {
                "airport": airport,
                "carrier_in": carrier_in,
                "carrier_out": carrier_out,
                "slack_min": layover,
                "arr_hour": hour,
                "day_of_week": 2,
                "month": month,
                "distance_in": distance,
                TARGET_COLUMN: True,
            }
            for airport, carrier_in, carrier_out, layover, hour, month, distance in rows
        ]
    )
    return build_features(frame)


def recategorized(features: pd.DataFrame, categories: dict, arrange) -> pd.DataFrame:
    """The same values, with each categorical column's category list set to `arrange(training list)`."""
    out = features.copy()
    for column in CATEGORICAL_COLUMNS:
        out[column] = pd.Categorical(out[column].astype(str), categories=arrange(categories[column]))
    return out


def predict(model, features: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(features[FEATURE_COLUMNS])[:, 1]


def test_model_categories_json_is_what_the_model_was_trained_on(model, categories):
    """model_categories.json feeds the dropdowns; the model remaps onto its own stored copy.
    They must agree on values and on order (position = integer code).

    Like the ground-truth test below, this reads sklearn internals
    (`calibrated_classifiers_[0].estimator`), so a version bump can break it on its own."""
    stored = model.calibrated_classifiers_[0].estimator.booster_.pandas_categorical

    assert [list(s) for s in stored] == [categories[c] for c in CATEGORICAL_COLUMNS]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_prediction_is_identical_however_the_categories_are_encoded(model, categories, case):
    single = query([case])
    assert [len(single[c].cat.categories) for c in CATEGORICAL_COLUMNS] == [1, 1, 1]  # the premise

    expected = predict(model, single)[0]
    encodings = {
        "training order": lambda cats: list(cats),
        "reversed": lambda cats: list(reversed(cats)),
        "rotated": lambda cats: list(cats[3:]) + list(cats[:3]),
    }
    for name, arrange in encodings.items():
        got = predict(model, recategorized(single, categories, arrange))[0]
        assert got == pytest.approx(expected, abs=1e-12), f"{name} encoding changed the prediction"


def test_batch_prediction_matches_each_row_predicted_alone(model, categories):
    batch = recategorized(query(CASES), categories, lambda cats: list(cats))
    together = predict(model, batch)

    for row, from_batch in zip(CASES, together):
        assert predict(model, query([row]))[0] == pytest.approx(from_batch, abs=1e-12)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_prediction_matches_a_ground_truth_computed_without_pandas(model, categories, case):
    """Hand-encode each category as its position in the training list, hand the raw
    booster a plain numpy row, and apply the calibrator -- no category dtype anywhere.

    NOTE: this reaches into sklearn internals -- `calibrated_classifiers_[0].estimator`
    and `.calibrators[0]` on the fitted CalibratedClassifierCV -- because that's the only
    way to run the booster and calibrator without pandas. Those aren't public API, so a
    scikit-learn (or LightGBM) version bump can break THIS test on its own, without the
    model being wrong. If it fails right after a bump, check the attribute names first;
    the encoding-equivalence, batch, and sensitivity tests use only the public predict_proba.
    """
    airport, carrier_in, carrier_out, layover, hour, month, distance = case
    row = {
        "slack_min": layover, "arr_hour": hour, "day_of_week": 2, "month": month, "distance_in": distance,
        "airport": airport, "carrier_in": carrier_in, "carrier_out": carrier_out,
    }
    vector = [row[c] for c in NUMERIC_COLUMNS] + [categories[c].index(row[c]) for c in CATEGORICAL_COLUMNS]

    calibrated = model.calibrated_classifiers_[0]
    raw = calibrated.estimator.booster_.predict(np.array([vector], dtype="float64"))
    truth = float(calibrated.calibrators[0].predict(raw)[0])

    assert predict(model, query([case]))[0] == pytest.approx(truth, abs=1e-12)


def test_model_actually_uses_the_categorical_features(model, categories):
    """Without this the equalities above could hold vacuously (a model that ignored
    airport and carrier would pass them all)."""
    rows = [
        (airport, carrier_in, "AA", 60, 14, 6, 800)
        for airport in categories["airport"]
        for carrier_in in categories["carrier_in"]
    ]
    probabilities = predict(model, recategorized(query(rows), categories, lambda cats: list(cats)))

    assert len(np.unique(probabilities.round(6))) > 5
