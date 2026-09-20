from __future__ import annotations

import pandas as pd

from src.pairs import build_pairs

CLEANED_COLUMNS = [
    "flight_date",
    "carrier",
    "tail_number",
    "flight_number",
    "origin",
    "dest",
    "sched_dep",
    "dep",
    "sched_arr",
    "arr",
    "distance",
]


def make_cleaned(rows: list[dict]) -> pd.DataFrame:
    """Build a cleaned-schema DataFrame from partial rows, defaulting to a normal flight."""
    defaults = {
        "flight_date": pd.Timestamp("2025-01-06").date(),
        "carrier": "AA",
        "tail_number": "N1IN",
        "flight_number": 100,
        "origin": "SFO",
        "dest": "ORD",
        "sched_dep": pd.Timestamp("2025-01-06 09:00"),
        "dep": pd.Timestamp("2025-01-06 09:00"),
        "sched_arr": pd.Timestamp("2025-01-06 13:00"),
        "arr": pd.Timestamp("2025-01-06 13:00"),
        "distance": 1500,
    }
    full_rows = [{**defaults, **row} for row in rows]
    return pd.DataFrame(full_rows, columns=CLEANED_COLUMNS)


def arrival(**overrides) -> dict:
    row = {"dest": "ORD", "origin": "SFO", "flight_number": 1}
    row.update(overrides)
    return row


def departure(**overrides) -> dict:
    row = {"origin": "ORD", "dest": "JFK", "flight_number": 2}
    row.update(overrides)
    return row


def pairs_for(raw: pd.DataFrame, airport: str = "ORD", **kwargs) -> pd.DataFrame:
    """build_pairs with the same frame as both arrivals_source and departures_source
    -- there's no month boundary to worry about in these single-batch tests."""
    return build_pairs(raw, raw, airport, **kwargs)


# --- window boundaries ---------------------------------------------------------


def test_window_includes_lower_boundary_of_30_minutes():
    raw = make_cleaned(
        [
            arrival(sched_arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 13:30")),  # exactly 30 min
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 1
    assert pairs.iloc[0]["slack_min"] == 30


def test_window_includes_upper_boundary_of_240_minutes():
    raw = make_cleaned(
        [
            arrival(sched_arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 17:00")),  # exactly 240 min
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 1
    assert pairs.iloc[0]["slack_min"] == 240


def test_window_excludes_29_minutes():
    raw = make_cleaned(
        [
            arrival(sched_arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 13:29")),
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 0


def test_window_excludes_241_minutes():
    raw = make_cleaned(
        [
            arrival(sched_arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 17:01")),
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 0


# --- sampling cap ----------------------------------------------------------------


def test_cap_limits_candidates_per_arrival():
    rows = [arrival(sched_arr=pd.Timestamp("2025-01-06 13:00"))]
    for i in range(25):
        rows.append(
            departure(
                flight_number=200 + i,
                sched_dep=pd.Timestamp("2025-01-06 13:00") + pd.Timedelta(minutes=30 + i * 5),
            )
        )
    raw = make_cleaned(rows)

    pairs = pairs_for(raw, cap=20, seed=0)
    assert len(pairs) == 20


def test_cap_sample_is_deterministic_for_a_fixed_seed():
    rows = [arrival(sched_arr=pd.Timestamp("2025-01-06 13:00"))]
    for i in range(25):
        rows.append(
            departure(
                flight_number=200 + i,
                sched_dep=pd.Timestamp("2025-01-06 13:00") + pd.Timedelta(minutes=30 + i * 5),
            )
        )
    raw = make_cleaned(rows)

    first = pairs_for(raw, cap=20, seed=0)
    second = pairs_for(raw, cap=20, seed=0)
    pd.testing.assert_frame_equal(first, second)


# --- made/missed label -----------------------------------------------------------


def test_made_label_at_mct_boundary():
    raw = make_cleaned(
        [
            arrival(arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 13:30"), dep=pd.Timestamp("2025-01-06 13:30")),
        ]
    )
    # actual departure exactly arr + MCT (30 min, explicit) -> made
    pairs = pairs_for(raw, mct_min=30)
    assert pairs.iloc[0]["made"]


def test_missed_label_one_minute_short_of_mct():
    raw = make_cleaned(
        [
            arrival(arr=pd.Timestamp("2025-01-06 13:00")),
            departure(sched_dep=pd.Timestamp("2025-01-06 13:30"), dep=pd.Timestamp("2025-01-06 13:29")),
        ]
    )
    # actual departure one minute before arr + MCT (30 min, explicit) -> missed
    pairs = pairs_for(raw, mct_min=30)
    assert not pairs.iloc[0]["made"]


# --- airport isolation and cross-midnight matching --------------------------------


def test_pairs_never_cross_airports():
    raw = make_cleaned(
        [
            arrival(dest="ORD", sched_arr=pd.Timestamp("2025-01-06 13:00")),
            # A departure from a different airport (JFK), timed to be inside the
            # window if airport filtering were broken.
            departure(origin="JFK", sched_dep=pd.Timestamp("2025-01-06 13:30")),
            # A genuine ORD departure, also inside the window.
            departure(origin="ORD", flight_number=3, sched_dep=pd.Timestamp("2025-01-06 13:45")),
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 1
    assert (pairs["airport"] == "ORD").all()


def test_rolled_past_midnight_arrival_still_produces_candidates():
    # A red-eye: scheduled to land just after midnight, so sched_arr rolls onto
    # the next calendar day relative to the arrival's own flight_date. Matching
    # is by timestamp, not flight_date, so it must still pair with a departure
    # recorded under that next day's flight_date.
    raw = make_cleaned(
        [
            arrival(
                flight_date=pd.Timestamp("2025-01-05").date(),
                sched_dep=pd.Timestamp("2025-01-05 23:40"),
                sched_arr=pd.Timestamp("2025-01-06 00:10"),
            ),
            departure(
                flight_date=pd.Timestamp("2025-01-06").date(),
                sched_dep=pd.Timestamp("2025-01-06 00:40"),  # 30 min after sched_arr
            ),
        ]
    )
    pairs = pairs_for(raw)
    assert len(pairs) == 1
    assert pairs.iloc[0]["slack_min"] == 30
    # day_of_week/month come from the arrival's own flight_date (Jan 5), not the
    # departure's or the rolled sched_arr's date.
    assert pairs.iloc[0]["day_of_week"] == pd.Timestamp("2025-01-05").dayofweek
    assert pairs.iloc[0]["month"] == 1
