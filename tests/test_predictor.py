from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS
from src.predictor import MIN_COVERAGE_PAIRS, Backend, LookupTables


class FakeModel:
    """Stands in for the trained model and remembers what it was asked about."""

    def __init__(self, probability: float = 0.7):
        self.probability = probability
        self.seen: pd.DataFrame | None = None

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        self.seen = features
        return np.array([[1 - self.probability, self.probability]] * len(features))


def make_lookups() -> LookupTables:
    recovery = pd.DataFrame(
        {
            "airport": ["ORD", "ORD", "ATL"],
            "month": [7, 7, 7],
            "arr_block": ["late", "morning", "late"],
            "n_missed": [1000, 2000, 500],
            "median_wait_min": [286.0, 45.0, float("nan")],  # NaN: nothing was ever catchable
            "no_recovery_share": [0.92, 0.05, 1.0],
        }
    )
    comparison = pd.DataFrame(
        {
            "airport": ["ORD", "ATL", "ORD", "ATL", "ORD", "ORD"],
            "month": [7, 7, 7, 7, 7, 8],
            "layover_bucket": ["45-60", "45-60", "60-75", "60-75", "225-240", "60-75"],
            "n": [10, 20, 30, 40, 50, 60],
            "success_rate": [0.5, 0.6, 0.8, 0.9, 0.99, 0.7],
        }
    )
    coverage = pd.DataFrame(
        {
            "airport": ["ORD", "ORD", "ORD", "ORD", "ATL"],
            "carrier_in": ["AA", "AA", "DL", "UA", "DL"],
            "carrier_out": ["UA", "DL", "AA", "UA", "DL"],
            "n_pairs": [MIN_COVERAGE_PAIRS, MIN_COVERAGE_PAIRS - 1, 5000, 1, 800],
        }
    )
    origin_distances = pd.DataFrame(
        {"airport": ["ORD", "ORD", "ATL"], "origin": ["ATL", "BOS", "BOS"], "distance_in": [606, 867, 946]}
    )
    return LookupTables(
        recovery=recovery,
        comparison=comparison,
        coverage=coverage,
        origin_distances=origin_distances,
        categories={"airport": ["ATL", "ORD"], "carrier_in": ["AA", "DL", "UA"], "carrier_out": ["AA", "DL", "UA"]},
        carrier_names={"AA": "American Airlines"},
        origin_names={"BOS": "Boston Logan"},
    )


def make_backend(probability: float = 0.7) -> Backend:
    return Backend(FakeModel(probability), make_lookups())


INPUTS = {
    "airport": "ORD",
    "origin": "ATL",
    "carrier_in": "AA",
    "carrier_out": "UA",
    "month": 7,
    "layover_min": 60,
    "arrival_hour": 14,
}


# --- predict -------------------------------------------------------------------


def test_predict_returns_a_float_probability():
    probability = make_backend(0.7).predict(INPUTS)

    assert isinstance(probability, float)
    assert 0.0 <= probability <= 1.0
    assert probability == pytest.approx(0.7)


def test_predict_sends_the_model_the_selections_and_the_looked_up_distance():
    backend = make_backend()
    backend.predict(INPUTS)
    row = backend.model.seen.iloc[0]

    assert list(backend.model.seen.columns) == FEATURE_COLUMNS
    assert (row["airport"], row["carrier_in"], row["carrier_out"]) == ("ORD", "AA", "UA")
    assert row["slack_min"] == 60
    assert row["arr_hour"] == 14
    assert row["month"] == 7
    assert row["distance_in"] == 606  # ATL -> ORD, from the lookup, not from the caller
    assert row["day_of_week"] == 2  # fixed to Wednesday


def test_predict_uses_the_distance_for_the_selected_origin():
    backend = make_backend()
    backend.predict({**INPUTS, "origin": "BOS"})

    assert backend.model.seen.iloc[0]["distance_in"] == 867


def test_predict_rejects_a_route_the_data_has_no_flights_on():
    # The fake table has ATL -> ORD but no ORD -> ATL.
    with pytest.raises(ValueError, match="no route data for ORD -> ATL"):
        make_backend().predict({**INPUTS, "airport": "ATL", "origin": "ORD"})


# --- recovery ------------------------------------------------------------------


def test_recovery_returns_the_figures_for_the_arrival_time_block():
    backend = make_backend()

    late = backend.recovery("ORD", 7, arrival_hour=23)
    morning = backend.recovery("ORD", 7, arrival_hour=8)

    assert late == {"median_wait_min": 286.0, "no_recovery_share": 0.92, "n_missed": 1000}
    assert morning == {"median_wait_min": 45.0, "no_recovery_share": 0.05, "n_missed": 2000}


def test_recovery_is_empty_when_the_data_has_nothing_for_that_combination():
    backend = make_backend()

    assert backend.recovery("ORD", 7, arrival_hour=13) == {}  # midday: no row
    assert backend.recovery("ORD", 1, arrival_hour=23) == {}  # wrong month
    assert backend.recovery("DEN", 7, arrival_hour=23) == {}  # wrong airport


def test_recovery_keeps_nan_wait_when_no_missed_pair_had_a_later_flight():
    result = make_backend().recovery("ATL", 7, arrival_hour=22)

    assert result["no_recovery_share"] == 1.0
    assert math.isnan(result["median_wait_min"])


# --- comparison ----------------------------------------------------------------


def test_comparison_returns_one_row_per_airport_for_the_layover_bucket_and_month():
    rows = make_backend().comparison(layover_min=70, month=7)

    assert isinstance(rows, list)
    assert {(r["airport"], r["success_rate"]) for r in rows} == {("ORD", 0.8), ("ATL", 0.9)}
    assert all(r["layover_bucket"] == "60-75" and r["month"] == 7 for r in rows)


def test_comparison_buckets_are_left_inclusive():
    backend = make_backend()

    at_60 = backend.comparison(layover_min=60, month=7)
    at_59 = backend.comparison(layover_min=59, month=7)

    assert {r["layover_bucket"] for r in at_60} == {"60-75"}  # 60 starts its own bucket
    assert {r["layover_bucket"] for r in at_59} == {"45-60"}


def test_comparison_includes_the_240_minute_upper_bound():
    rows = make_backend().comparison(layover_min=240, month=7)

    assert [r["layover_bucket"] for r in rows] == ["225-240"]


def test_comparison_is_empty_for_a_month_with_no_data():
    assert make_backend().comparison(layover_min=60, month=1) == []


# --- coverage (FR-11) ----------------------------------------------------------


def test_a_combination_needs_the_minimum_number_of_pairs_to_be_covered():
    backend = make_backend()

    assert backend.is_covered("ORD", "AA", "UA")  # exactly the minimum
    assert not backend.is_covered("ORD", "AA", "DL")  # one short
    assert backend.is_covered("ORD", "DL", "AA")


def test_a_combination_absent_from_the_table_is_not_covered():
    backend = make_backend()

    assert backend.pair_count("ORD", "UA", "DL") == 0
    assert not backend.is_covered("ORD", "UA", "DL")


def test_coverage_is_per_airport():
    backend = make_backend()

    assert backend.pair_count("ATL", "DL", "DL") == 800
    assert backend.pair_count("ORD", "DL", "DL") == 0


def test_carrier_and_origin_options_are_limited_to_what_operates_at_the_airport():
    lookups = make_lookups()

    assert lookups.carriers_in_at("ORD") == ["AA", "DL", "UA"]
    assert lookups.carriers_out_at("ORD") == ["AA", "DL", "UA"]
    assert lookups.carriers_in_at("ATL") == ["DL"]
    assert lookups.carriers_out_at("ATL") == ["DL"]
    assert lookups.origins_at("ORD") == ["ATL", "BOS"]
    assert lookups.origins_at("ATL") == ["BOS"]
    assert lookups.airports() == ["ATL", "ORD"]


# --- the real artifacts --------------------------------------------------------


def test_backend_loads_the_committed_artifacts_and_answers_a_query():
    """The deployed entry point's path: model.pkl + data/lookups/*, no Streamlit."""
    backend = Backend.load()
    lookups = backend.lookups

    airport = lookups.airports()[0]
    query = {
        "airport": airport,
        "origin": lookups.origins_at(airport)[0],
        "carrier_in": lookups.carriers_in_at(airport)[0],
        "carrier_out": lookups.carriers_out_at(airport)[0],
        "month": 6,
        "layover_min": 90,
        "arrival_hour": 12,
    }

    assert 0.0 <= backend.predict(query) <= 1.0
    assert isinstance(backend.recovery(airport, 6, 12), dict)
    assert len(backend.comparison(90, 6)) == len(lookups.airports())  # one bar per airport
    assert backend.pair_count(airport, query["carrier_in"], query["carrier_out"]) >= 0
