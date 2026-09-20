from __future__ import annotations

import pandas as pd

from src.cleaner import clean_month, drop_counts, parse_hhmm, push_past_midnight

RAW_COLUMNS = [
    "FL_DATE",
    "OP_UNIQUE_CARRIER",
    "TAIL_NUM",
    "OP_CARRIER_FL_NUM",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DEP_TIME",
    "CRS_ARR_TIME",
    "ARR_TIME",
    "CANCELLED",
    "DIVERTED",
    "DISTANCE",
]


def make_raw(rows: list[dict]) -> pd.DataFrame:
    """Build a raw-schema DataFrame from partial rows, defaulting to a normal flight."""
    defaults = {
        "FL_DATE": "1/6/2025 12:00:00 AM",
        "OP_UNIQUE_CARRIER": "AA",
        "TAIL_NUM": "N102NN",
        "OP_CARRIER_FL_NUM": 100,
        "ORIGIN": "SFO",
        "DEST": "JFK",
        "CRS_DEP_TIME": 900,
        "DEP_TIME": 900,
        "CRS_ARR_TIME": 1700,
        "ARR_TIME": 1700,
        "CANCELLED": 0.0,
        "DIVERTED": 0.0,
        "DISTANCE": 2586,
    }
    full_rows = [{**defaults, **row} for row in rows]
    return pd.DataFrame(full_rows, columns=RAW_COLUMNS)


# --- zero-padded time parsing -------------------------------------------------


def test_parse_hhmm_pads_short_times():
    flight_date = pd.Series([pd.Timestamp("2025-01-06")])
    assert parse_hhmm(pd.Series([28]), flight_date).iloc[0] == pd.Timestamp("2025-01-06 00:28")


def test_parse_hhmm_splits_hour_and_minute():
    flight_date = pd.Series([pd.Timestamp("2025-01-06")] * 2)
    result = parse_hhmm(pd.Series([829, 1321]), flight_date)
    assert result.iloc[0] == pd.Timestamp("2025-01-06 08:29")
    assert result.iloc[1] == pd.Timestamp("2025-01-06 13:21")


def test_parse_hhmm_treats_2400_as_midnight_next_day():
    flight_date = pd.Series([pd.Timestamp("2025-01-06")])
    result = parse_hhmm(pd.Series([2400]), flight_date)
    assert result.iloc[0] == pd.Timestamp("2025-01-07 00:00")


def test_parse_hhmm_leaves_missing_time_as_nat():
    flight_date = pd.Series([pd.Timestamp("2025-01-06")])
    result = parse_hhmm(pd.Series([None]), flight_date)
    assert pd.isna(result.iloc[0])


# --- midnight-crossing logic ---------------------------------------------------


def test_push_past_midnight_adds_a_day_when_arrival_precedes_departure():
    dep = pd.Series([pd.Timestamp("2025-01-06 23:50")])
    arr = pd.Series([pd.Timestamp("2025-01-06 00:15")])  # same date as dep, but earlier clock time
    result = push_past_midnight(dep, arr)
    assert result.iloc[0] == pd.Timestamp("2025-01-07 00:15")


def test_push_past_midnight_leaves_same_day_arrival_alone():
    dep = pd.Series([pd.Timestamp("2025-01-06 09:00")])
    arr = pd.Series([pd.Timestamp("2025-01-06 17:00")])
    result = push_past_midnight(dep, arr)
    assert result.iloc[0] == pd.Timestamp("2025-01-06 17:00")


def test_clean_month_rolls_overnight_flight_to_the_next_day():
    raw = make_raw(
        [
            {
                "CRS_DEP_TIME": 2350,
                "DEP_TIME": 2355,
                "CRS_ARR_TIME": 15,  # 00:15, scheduled to land after midnight
                "ARR_TIME": 20,  # 00:20 actual
            }
        ]
    )
    cleaned = clean_month(raw)
    row = cleaned.iloc[0]

    assert row["sched_dep"] == pd.Timestamp("2025-01-06 23:50")
    assert row["sched_arr"] == pd.Timestamp("2025-01-07 00:15")
    assert row["dep"] == pd.Timestamp("2025-01-06 23:55")
    assert row["arr"] == pd.Timestamp("2025-01-07 00:20")


# --- dropping cancelled and diverted rows --------------------------------------


def test_clean_month_drops_cancelled_and_diverted_rows():
    raw = make_raw(
        [
            {"OP_CARRIER_FL_NUM": 1},  # normal flight, kept
            {
                "OP_CARRIER_FL_NUM": 2,
                "CANCELLED": 1.0,
                "DEP_TIME": None,
                "ARR_TIME": None,
            },
            {
                "OP_CARRIER_FL_NUM": 3,
                "DIVERTED": 1.0,
                "ARR_TIME": None,
            },
        ]
    )
    cleaned = clean_month(raw)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["flight_number"] == 1


def test_drop_counts_breaks_down_by_reason():
    raw = make_raw(
        [
            {"OP_CARRIER_FL_NUM": 1},  # kept
            {"OP_CARRIER_FL_NUM": 2, "CANCELLED": 1.0, "DEP_TIME": None, "ARR_TIME": None},
            {"OP_CARRIER_FL_NUM": 3, "DIVERTED": 1.0, "ARR_TIME": None},
            {"OP_CARRIER_FL_NUM": 4, "ARR_TIME": None},  # unflagged but unlabelable
        ]
    )
    counts = drop_counts(raw)

    assert counts == {"cancelled": 1, "diverted": 1, "null_arrival_without_flag": 1}


def test_clean_month_drops_unflagged_rows_missing_an_actual_time():
    # Some BTS rows have no actual arrival time without being flagged cancelled or
    # diverted (a reporting gap seen in real data) -- drop those too, since they
    # can't be labeled either.
    raw = make_raw(
        [
            {"OP_CARRIER_FL_NUM": 1},
            {"OP_CARRIER_FL_NUM": 2, "ARR_TIME": None},
        ]
    )
    cleaned = clean_month(raw)

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["flight_number"] == 1


def test_clean_month_output_has_no_nulls_in_timestamp_columns():
    raw = make_raw(
        [
            {"OP_CARRIER_FL_NUM": 1},
            {"OP_CARRIER_FL_NUM": 2, "CANCELLED": 1.0, "DEP_TIME": None, "ARR_TIME": None},
        ]
    )
    cleaned = clean_month(raw)

    assert len(cleaned) == 1
    for column in ["sched_dep", "dep", "sched_arr", "arr"]:
        assert cleaned[column].notna().all()
