"""Smoke test for the committed model.pkl: loads it and checks a prediction
comes back sane, so a library/pickle version mismatch fails here in CI rather
than showing up as a crash on demo day."""

from __future__ import annotations

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features


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
