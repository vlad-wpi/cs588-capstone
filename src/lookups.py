"""Labeled pairs + cleaned flights -> the small lookup tables the deployed app reads.

recovery, comparison, origin_distances, and coverage are all small (airport x month
x a handful of buckets, or airport x carrier pair) and are the only things, besides
model.pkl, that the app reads -- see schema.md.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from src.pairs import AIRPORTS

CLEAN_DIR = Path("data/clean")
PAIRS_DIR = Path("data/pairs")
LOOKUPS_DIR = Path("data/lookups")

ALL_MONTHS = [f"2025_{m:02d}" for m in range(1, 13)]

ARR_BLOCK_EDGES = [
    ("morning", 5, 10),
    ("midday", 11, 15),
    ("evening", 16, 20),
]  # anything not in one of these (21-23, 0-4) is "late"

LAYOVER_BUCKET_EDGES = list(range(30, 241, 15))  # 30, 45, 60, ..., 240

DEFAULT_MIN_RECOVERY_GAP_MIN = 30  # a "next departure" one minute later isn't catchable


PAIRS_LOOKUP_COLUMNS = ["airport", "month", "slack_min", "arr_hour", "dest_out", "sched_dep_out", "made"]
CLEANED_LOOKUP_COLUMNS = ["origin", "dest", "flight_date", "sched_dep"]


def load_pairs(columns: list[str] | None = PAIRS_LOOKUP_COLUMNS) -> pd.DataFrame:
    paths = sorted(PAIRS_DIR.glob("pairs_*.parquet"))
    if not paths:
        raise SystemExit(f"no pairs files found matching {PAIRS_DIR / 'pairs_*.parquet'}")
    return pd.concat((pd.read_parquet(p, columns=columns) for p in paths), ignore_index=True)


def load_cleaned(columns: list[str] | None = CLEANED_LOOKUP_COLUMNS) -> pd.DataFrame:
    frames = [pd.read_parquet(CLEAN_DIR / f"flights_{month}.parquet", columns=columns) for month in ALL_MONTHS]
    return pd.concat(frames, ignore_index=True)


def arr_block(arr_hour: pd.Series) -> pd.Series:
    """Bucket an hour-of-day (0-23) into morning/midday/evening/late."""
    block = pd.Series("late", index=arr_hour.index)
    for label, lo, hi in ARR_BLOCK_EDGES:
        block = block.mask(arr_hour.between(lo, hi), label)
    return block


def layover_bucket(slack_min: pd.Series) -> pd.Series:
    """15-minute buckets from 30 to 240, left-inclusive: '60-75' means
    60 <= slack_min < 75, matching how the label itself reads. The last bucket,
    '225-240', also includes 240 itself (pairs.py's own upper bound)."""
    labels = [f"{lo}-{hi}" for lo, hi in zip(LAYOVER_BUCKET_EDGES[:-1], LAYOVER_BUCKET_EDGES[1:])]
    bucket = pd.cut(slack_min, bins=LAYOVER_BUCKET_EDGES, labels=labels, right=False).astype("object")
    bucket[slack_min == LAYOVER_BUCKET_EDGES[-1]] = labels[-1]
    return bucket.astype(pd.CategoricalDtype(categories=labels, ordered=True))


def _next_departure_waits(
    missed: pd.DataFrame, cleaned: pd.DataFrame, min_recovery_gap_min: int = DEFAULT_MIN_RECOVERY_GAP_MIN
) -> pd.Series:
    """For each missed pair, minutes until the next same-day departure from
    `airport` to `dest_out` that leaves at least min_recovery_gap_min minutes
    after `sched_dep_out`; NaN if none exists that day.

    The gap matters: the nearest later flight by pure schedule adjacency can be
    one minute after the one that was missed, which nobody can actually board.
    min_recovery_gap_min is a floor on how soon a "recovery" can count, not a
    guess at how long it takes -- same spirit as pairs.py's MCT.

    Deliberately searches ALL departures from that airport (not the capped
    candidate set in pairs.py), since the true next flight to that destination
    can easily fall outside the 30-240 min connection window.
    """
    waits = pd.Series(np.nan, index=missed.index, dtype="float64")
    gap = np.timedelta64(min_recovery_gap_min, "m")

    for airport, group in missed.groupby("airport", observed=True):
        departures = cleaned.loc[cleaned["origin"] == airport, ["dest", "flight_date", "sched_dep"]]
        dep_times_by_key = {
            key: np.sort(sub["sched_dep"].to_numpy())
            for key, sub in departures.groupby(["dest", "flight_date"])
        }

        day = group["sched_dep_out"].dt.date
        for (dest, key_day), rows in group.groupby([group["dest_out"], day]):
            times = dep_times_by_key.get((dest, key_day))
            if times is None:
                continue

            query = rows["sched_dep_out"].to_numpy()
            pos = np.searchsorted(times, query + gap, side="left")
            valid = pos < len(times)

            result = np.full(len(rows), np.nan)
            result[valid] = (times[pos[valid]] - query[valid]) / np.timedelta64(1, "m")
            waits.loc[rows.index] = result

    return waits


def build_recovery(
    pairs: pd.DataFrame, cleaned: pd.DataFrame, min_recovery_gap_min: int = DEFAULT_MIN_RECOVERY_GAP_MIN
) -> pd.DataFrame:
    """Median recovery wait and no-same-day-recovery share, by airport x month x
    arrival time block, for pairs labeled missed."""
    missed = pairs.loc[~pairs["made"]].copy()
    missed["wait_min"] = _next_departure_waits(missed, cleaned, min_recovery_gap_min)
    missed["arr_block"] = arr_block(missed["arr_hour"])

    recovery = missed.groupby(["airport", "month", "arr_block"], observed=True).agg(
        n_missed=("wait_min", "size"),
        median_wait_min=("wait_min", "median"),
        no_recovery_share=("wait_min", lambda s: s.isna().mean()),
    )
    return recovery.reset_index()


def build_origin_distances(cleaned: pd.DataFrame, airports: list[str] = AIRPORTS) -> pd.DataFrame:
    """Distance in miles from each origin to each connecting airport, for every
    origin that actually flew there -- app.py's "flying in from" input looks up
    distance_in here instead of guessing at a single fixed value.

    A given (origin, airport) pair's distance is occasionally reported as two
    values a mile apart (rounding, not a data error) -- the mode is used rather
    than an arbitrary pick.
    """
    rows = []
    for airport in airports:
        arrivals = cleaned.loc[cleaned["dest"] == airport, ["origin", "distance"]]
        for origin, group in arrivals.groupby("origin"):
            distance = int(group["distance"].mode().iloc[0])
            rows.append({"airport": airport, "origin": origin, "distance_in": distance})
    return pd.DataFrame(rows)


def count_carrier_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pair count per (airport, carrier_in, carrier_out). A combination with no
    pairs at all is simply absent from the result, not a zero row."""
    return (
        pairs.groupby(["airport", "carrier_in", "carrier_out"])
        .size()
        .rename("n_pairs")
        .reset_index()
    )


def build_coverage(paths: list[Path] | None = None) -> pd.DataFrame:
    """Training pairs behind each (airport, carrier_in, carrier_out) -- app.py
    refuses to show a result for a combination with too few (FR-11).

    Reads one pairs file at a time, three columns each, so the full ~27.7M-row
    set never has to sit in memory (see the OOM notes in CLAUDE.md).
    """
    paths = paths if paths is not None else sorted(PAIRS_DIR.glob("pairs_*.parquet"))
    if not paths:
        raise SystemExit(f"no pairs files found matching {PAIRS_DIR / 'pairs_*.parquet'}")

    counts = pd.concat(
        (
            count_carrier_pairs(pd.read_parquet(path, columns=["airport", "carrier_in", "carrier_out"]))
            for path in paths
        ),
        ignore_index=True,
    )
    return counts.groupby(["airport", "carrier_in", "carrier_out"], as_index=False)["n_pairs"].sum()


def build_comparison(pairs: pd.DataFrame) -> pd.DataFrame:
    """Observed success rate by airport x month x 15-minute layover bucket."""
    labeled = pairs.copy()
    labeled["layover_bucket"] = layover_bucket(labeled["slack_min"])

    comparison = labeled.groupby(["airport", "month", "layover_bucket"], observed=True).agg(
        n=("made", "size"),
        success_rate=("made", "mean"),
    )
    return comparison.reset_index()


def main() -> None:
    pairs = load_pairs()
    cleaned = load_cleaned()
    LOOKUPS_DIR.mkdir(parents=True, exist_ok=True)

    recovery = build_recovery(pairs, cleaned)
    del cleaned
    gc.collect()

    recovery_path = LOOKUPS_DIR / "recovery.parquet"
    recovery.to_parquet(recovery_path, index=False)
    print(f"recovery: {len(recovery)} rows, wrote {recovery_path}")
    print(recovery.to_string(index=False))

    comparison = build_comparison(pairs)
    comparison_path = LOOKUPS_DIR / "comparison.parquet"
    comparison.to_parquet(comparison_path, index=False)
    print(f"\ncomparison: {len(comparison)} rows, wrote {comparison_path}")

    del pairs
    gc.collect()

    distance_cleaned = load_cleaned(columns=["origin", "dest", "distance"])
    origin_distances = build_origin_distances(distance_cleaned)
    origin_distances_path = LOOKUPS_DIR / "origin_distances.parquet"
    origin_distances.to_parquet(origin_distances_path, index=False)
    print(f"\norigin_distances: {len(origin_distances)} rows, wrote {origin_distances_path}")

    coverage = build_coverage()
    coverage_path = LOOKUPS_DIR / "coverage.parquet"
    coverage.to_parquet(coverage_path, index=False)
    print(f"\ncoverage: {len(coverage)} rows, wrote {coverage_path}")


if __name__ == "__main__":
    main()
