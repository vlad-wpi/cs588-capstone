"""Clean parquet -> labeled connection pairs, one airport at a time.

For a given airport, matches each arriving flight (dest == airport) with departing
flights (origin == airport) whose scheduled departure is min-to-max minutes after
the arrival's scheduled arrival, purely by timestamp -- independent of which
flight_date either row carries, so red-eye arrivals whose scheduled arrival rolls
past midnight still match against next-day departures. Candidates are capped per
arrival (a random, fixed-seed sample) since an uncapped join is combinatorial.
Each surviving pair is labeled made/missed from the actual times plus a minimum
connection time (MCT). Also carries the outbound flight's destination and
scheduled departure (dest_out, sched_dep_out), which lookups.py needs to find,
for a missed pair, the next departure to the same place.

Because arrivals and departures are matched by timestamp, an arrival near the end
of one month can connect to a departure recorded in the next month's parquet file
(a December 31 red-eye landing after midnight, connecting to a January 1
departure). `load_with_boundary` handles this by additionally loading the
following month's first calendar day as departure-only data.

Writes `data/pairs/pairs_<airport>.parquet`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLEAN_DIR = Path("data/clean")
PAIRS_DIR = Path("data/pairs")

MIN_SLACK_MIN = 30
MAX_SLACK_MIN = 240
MAX_CANDIDATES_PER_ARRIVAL = 20
DEFAULT_SEED = 0
DEFAULT_MCT_MIN = 45

OUTPUT_COLUMNS = [
    "airport",
    "carrier_in",
    "carrier_out",
    "slack_min",
    "arr_hour",
    "day_of_week",
    "month",
    "distance_in",
    "tail_in",
    "tail_out",
    "dest_out",
    "sched_dep_out",
    "made",
]


def _match_and_cap(
    arrivals: pd.DataFrame,
    departures: pd.DataFrame,
    min_slack_min: int,
    max_slack_min: int,
    cap: int,
    seed: int,
) -> pd.DataFrame:
    """For each arrival, find departures whose sched_dep falls min_slack_min to
    max_slack_min minutes after the arrival's sched_arr (by timestamp), then keep
    at most `cap` of them via a fixed-seed random sample.
    """
    if arrivals.empty or departures.empty:
        return pd.DataFrame()

    departures = departures.sort_values("sched_dep").reset_index(drop=True)
    dep_times = departures["sched_dep"].to_numpy()

    lo_offset = np.timedelta64(min_slack_min, "m")
    hi_offset = np.timedelta64(max_slack_min, "m")
    rng = np.random.default_rng(seed)

    arr_positions = []
    dep_positions = []
    for arr_pos, arr_time in enumerate(arrivals["sched_arr"].to_numpy()):
        lo = arr_time + lo_offset
        hi = arr_time + hi_offset
        i = np.searchsorted(dep_times, lo, side="left")
        j = np.searchsorted(dep_times, hi, side="right")
        if j <= i:
            continue

        window = np.arange(i, j)
        if len(window) > cap:
            window = rng.choice(window, size=cap, replace=False)

        arr_positions.extend([arr_pos] * len(window))
        dep_positions.extend(window.tolist())

    if not arr_positions:
        return pd.DataFrame()

    arr_part = arrivals.iloc[arr_positions].reset_index(drop=True).add_suffix("_in")
    dep_part = departures.iloc[dep_positions].reset_index(drop=True).add_suffix("_out")
    candidates = pd.concat([arr_part, dep_part], axis=1)
    candidates["slack_min"] = (
        candidates["sched_dep_out"] - candidates["sched_arr_in"]
    ) // pd.Timedelta(minutes=1)
    return candidates


def build_pairs(
    arrivals_source: pd.DataFrame,
    departures_source: pd.DataFrame,
    airport: str,
    mct_min: int = DEFAULT_MCT_MIN,
    min_slack_min: int = MIN_SLACK_MIN,
    max_slack_min: int = MAX_SLACK_MIN,
    cap: int = MAX_CANDIDATES_PER_ARRIVAL,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Build labeled connection pairs for one airport.

    `arrivals_source` and `departures_source` are separate so a caller can widen
    the departures side past a month boundary without treating that extra data as
    arrivals too (see `load_with_boundary`). Pass the same frame for both when
    there's no boundary to worry about.
    """
    arrivals = arrivals_source.loc[arrivals_source["dest"] == airport].reset_index(drop=True)
    departures = departures_source.loc[departures_source["origin"] == airport].reset_index(drop=True)

    candidates = _match_and_cap(arrivals, departures, min_slack_min, max_slack_min, cap, seed)
    if candidates.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    made = (candidates["arr_in"] + pd.Timedelta(minutes=mct_min)) <= candidates["dep_out"]
    flight_date_in = pd.to_datetime(candidates["flight_date_in"])

    return pd.DataFrame(
        {
            "airport": airport,
            "carrier_in": candidates["carrier_in"],
            "carrier_out": candidates["carrier_out"],
            "slack_min": candidates["slack_min"].astype("int64"),
            "arr_hour": candidates["sched_arr_in"].dt.hour,
            "day_of_week": flight_date_in.dt.dayofweek,
            "month": flight_date_in.dt.month,
            "distance_in": candidates["distance_in"],
            "tail_in": candidates["tail_number_in"],
            "tail_out": candidates["tail_number_out"],
            "dest_out": candidates["dest_out"],
            "sched_dep_out": candidates["sched_dep_out"],
            "made": made,
        }
    )[OUTPUT_COLUMNS].reset_index(drop=True)


def load_cleaned(months: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(CLEAN_DIR / f"flights_{month}.parquet") for month in months]
    return pd.concat(frames, ignore_index=True)


def _next_month_key(month_key: str) -> str:
    year, month = (int(part) for part in month_key.split("_"))
    month += 1
    if month > 12:
        month = 1
        year += 1
    return f"{year}_{month:02d}"


def load_with_boundary(months: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (arrivals_source, departures_source) for `months`.

    departures_source additionally includes the following month's first calendar
    day, if that month's file exists, so a late arrival in the last day of
    `months` can still match a departure recorded in the next month's parquet.
    """
    arrivals_source = load_cleaned(months)
    departures_source = arrivals_source

    next_month_path = CLEAN_DIR / f"flights_{_next_month_key(months[-1])}.parquet"
    if next_month_path.exists():
        next_month_df = pd.read_parquet(next_month_path)
        first_day = next_month_df["flight_date"].min()
        boundary = next_month_df.loc[next_month_df["flight_date"] == first_day]
        departures_source = pd.concat([arrivals_source, boundary], ignore_index=True)

    return arrivals_source, departures_source


AIRPORTS = ["ORD", "ATL", "DFW", "DEN", "CLT"]
ALL_MONTHS = [f"2025_{m:02d}" for m in range(1, 13)]

SLACK_BUCKETS = [
    ("30-45", 30, 45),
    ("45-60", 45, 60),
    ("60-90", 60, 90),
    ("90-120", 90, 120),
    ("120-180", 120, 180),
    ("180-240", 180, 241),  # upper bound exclusive here; 240 itself is included
]


def slack_bucket_table(pairs: pd.DataFrame) -> pd.DataFrame:
    """n and made rate per slack_min bucket, for reporting."""
    rows = []
    for label, lo, hi in SLACK_BUCKETS:
        mask = (pairs["slack_min"] >= lo) & (pairs["slack_min"] < hi)
        n = int(mask.sum())
        made_rate = pairs.loc[mask, "made"].mean() if n else float("nan")
        rows.append({"bucket": label, "n": n, "made_rate": made_rate})
    return pd.DataFrame(rows)


def main() -> None:
    cleaned_year = load_cleaned(ALL_MONTHS)
    PAIRS_DIR.mkdir(parents=True, exist_ok=True)

    all_pairs = []
    for airport in AIRPORTS:
        pairs = build_pairs(cleaned_year, cleaned_year, airport, mct_min=DEFAULT_MCT_MIN)
        out_path = PAIRS_DIR / f"pairs_{airport.lower()}.parquet"
        pairs.to_parquet(out_path, index=False)
        all_pairs.append(pairs)

        print(f"\n{airport}: {len(pairs):,} pairs, made_rate={pairs['made'].mean():.1%}, wrote {out_path}")
        print(slack_bucket_table(pairs).to_string(index=False))

    total = sum(len(p) for p in all_pairs)
    print(f"\ntotal pairs across {', '.join(AIRPORTS)}: {total:,}")


if __name__ == "__main__":
    main()
