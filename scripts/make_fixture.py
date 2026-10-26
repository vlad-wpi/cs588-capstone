"""Build a small, deterministic slice of real BTS data for tests/test_pipeline_real_data.py
to run the real cleaner and pairs code against.

Why a contiguous window and not a random sample: pairs.py's whole job is matching
arrivals to departures leaving shortly after at the SAME airport. A random row
sample would destroy nearly every one of those pairs -- the fixture would have
flights but almost no connections, and the pairs-building tests would exercise
nothing. A contiguous window of calendar days keeps full, real days of traffic at
one airport, so the arrivals and departures that were actually scheduled to
connect are still both present.

Run: python3 scripts/make_fixture.py
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# --- the window -- edit these to change exactly what gets kept ------------------
# Every row where ORIGIN or DEST is AIRPORT, on a calendar FL_DATE in
# [START_DATE, START_DATE + WINDOW_DAYS - 1]. Chosen (see REQUIRED_PATTERNS below)
# because this window holds at least one of each pattern the real-data tests
# depend on, without a mass-cancellation event (e.g. a winter storm day) bloating
# the fixture with thousands of cancelled rows.
SOURCE_CSV = Path("data/raw_data/T_ONTIME_REPORTING_2025-01.csv")
AIRPORT = "ORD"
START_DATE = date(2025, 1, 16)
WINDOW_DAYS = 3
# ----------------------------------------------------------------------------------

END_DATE = START_DATE + timedelta(days=WINDOW_DAYS - 1)
FIXTURE_PATH = Path("tests/fixtures/ord_window.csv")

FL_DATE_FORMAT = "%m/%d/%Y %I:%M:%S %p"  # matches src/cleaner.py's own parsing of FL_DATE


def load_window(
    source_csv: Path = SOURCE_CSV,
    airport: str = AIRPORT,
    start: date = START_DATE,
    end: date = END_DATE,
) -> pd.DataFrame:
    """Every row where ORIGIN or DEST is `airport`, on a calendar day in [start, end].

    Read with dtype=str and written back untouched (see main()), so every cell that
    survives the filter keeps its exact original text -- this is a slice of the
    real file, not a reconstruction of one.
    """
    raw = pd.read_csv(source_csv, dtype=str)
    flight_date = pd.to_datetime(raw["FL_DATE"], format=FL_DATE_FORMAT).dt.date

    in_window = (flight_date >= start) & (flight_date <= end)
    at_airport = (raw["ORIGIN"] == airport) | (raw["DEST"] == airport)
    return raw.loc[in_window & at_airport].reset_index(drop=True)


def required_patterns(df: pd.DataFrame) -> dict[str, int]:
    """Row counts for each pattern the real-data tests need present in the fixture."""
    dep, arr = pd.to_numeric(df["DEP_TIME"], errors="coerce"), pd.to_numeric(df["ARR_TIME"], errors="coerce")
    crs_dep = pd.to_numeric(df["CRS_DEP_TIME"], errors="coerce")
    crs_arr = pd.to_numeric(df["CRS_ARR_TIME"], errors="coerce")

    return {
        "midnight crossing (actual ARR_TIME < DEP_TIME)": int((dep.notna() & arr.notna() & (arr < dep)).sum()),
        "red-eye (scheduled CRS_ARR_TIME < CRS_DEP_TIME, i.e. arrival after midnight)": int(
            (crs_dep.notna() & crs_arr.notna() & (crs_arr < crs_dep)).sum()
        ),
        "cancelled (CANCELLED == 1)": int((pd.to_numeric(df["CANCELLED"], errors="coerce") == 1).sum()),
        "diverted (DIVERTED == 1)": int((pd.to_numeric(df["DIVERTED"], errors="coerce") == 1).sum()),
        # The property test_no_nulls_in_the_timestamp_columns_after_drops actually
        # depends on: a row that would come out of clean_month with a null
        # sched_dep/dep/sched_arr/arr if dropping ever broke. Checked directly
        # (missing DEP_TIME or ARR_TIME) rather than assumed from "cancelled" or
        # "diverted" as a proxy -- in this window every diverted row still has
        # both actual times recorded, so the cancelled row is this fixture's ONLY
        # source of a missing time, and that's incidental, not guaranteed by
        # "cancelled" or "diverted" as a category.
        "missing actual time (DEP_TIME or ARR_TIME blank)": int((dep.isna() | arr.isna()).sum()),
    }


def confirm_patterns(df: pd.DataFrame) -> None:
    """Print each required pattern's count; raise if any is missing. A fixture
    window silently missing one of these would make the tests that depend on it
    exercise nothing, and pass for the wrong reason."""
    print(f"fixture window: {START_DATE} to {END_DATE} ({WINDOW_DAYS} days), airport = {AIRPORT}")

    counts = required_patterns(df)
    missing = []
    for label, count in counts.items():
        print(f"  {label}: {count} row(s)")
        if count < 1:
            missing.append(label)

    if missing:
        raise SystemExit(
            f"fixture window is missing required pattern(s): {missing} -- "
            "widen WINDOW_DAYS or move START_DATE and rerun"
        )


def main() -> None:
    window = load_window()
    print(f"{len(window):,} rows at {AIRPORT} across {WINDOW_DAYS} day(s)\n")
    confirm_patterns(window)

    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    window.to_csv(FIXTURE_PATH, index=False)

    size_kb = FIXTURE_PATH.stat().st_size / 1024
    print(f"\nwrote {len(window):,} rows to {FIXTURE_PATH} ({size_kb:,.1f} KB)")


if __name__ == "__main__":
    main()
