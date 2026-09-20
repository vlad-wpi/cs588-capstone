"""Raw BTS on-time CSVs -> clean, per-month parquet files.

Reads `data/raw_data/T_ONTIME_REPORTING_*.csv`, applies the rules in schema.md
(midnight-crossing timestamps, dropping cancelled/diverted flights, dropping the
trailing unnamed column), and writes `data/clean/flights_YYYY_MM.parquet`.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd

RAW_GLOB = "data/raw_data/T_ONTIME_REPORTING_*.csv"
CLEAN_DIR = Path("data/clean")

OUTPUT_COLUMNS = [
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


def parse_hhmm(raw_time: pd.Series, flight_date: pd.Series) -> pd.Series:
    """Build a timestamp from a bare-integer HHMM column and a date column.

    BTS times have no leading zeros (28 means 00:28) and use 2400 for midnight
    at the end of the day rather than 0000 the next day, so 2400 rolls into the
    next day rather than parsing as an invalid hour.
    """
    hhmm = pd.to_numeric(raw_time, errors="coerce")
    valid = hhmm.notna()

    hhmm_filled = hhmm.fillna(0)
    is_2400 = hhmm_filled == 2400
    hhmm_filled = hhmm_filled.mask(is_2400, 0)

    hour = (hhmm_filled // 100).astype(int)
    minute = (hhmm_filled % 100).astype(int)
    extra_day = is_2400.astype(int)

    offset = (
        pd.to_timedelta(hour, unit="h")
        + pd.to_timedelta(minute, unit="m")
        + pd.to_timedelta(extra_day, unit="D")
    )
    timestamp = flight_date + offset
    return timestamp.where(valid)


def push_past_midnight(dep: pd.Series, arr: pd.Series) -> pd.Series:
    """Add a day to arrivals that land before their own departure (schema.md's midnight-crossing rule)."""
    crossed = dep.notna() & arr.notna() & (arr < dep)
    return arr.where(~crossed, arr + pd.Timedelta(days=1))


def drop_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean masks for each reason a row gets dropped, aligned to df's index.

    `null_arrival_without_flag` covers rows with no actual departure/arrival time
    that aren't flagged cancelled or diverted either (a BTS reporting gap):
    schema.md's reason for dropping cancelled/diverted flights is that they can't
    be labeled without an actual arrival time, and that applies here too.
    """
    cancelled = pd.to_numeric(df["CANCELLED"], errors="coerce").fillna(0) == 1
    diverted = pd.to_numeric(df["DIVERTED"], errors="coerce").fillna(0) == 1
    unlabelable = df[["DEP_TIME", "ARR_TIME"]].isna().any(axis=1)
    null_arrival_without_flag = unlabelable & ~cancelled & ~diverted
    return {
        "cancelled": cancelled,
        "diverted": diverted,
        "null_arrival_without_flag": null_arrival_without_flag,
    }


def drop_counts(df: pd.DataFrame) -> dict[str, int]:
    """Per-reason row counts for the drop report."""
    return {reason: int(mask.sum()) for reason, mask in drop_masks(df).items()}


def clean_month(df: pd.DataFrame) -> pd.DataFrame:
    """Apply schema.md's cleaning rules to one month of raw BTS rows."""
    df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"Unnamed.*")]

    masks = drop_masks(df)
    keep = ~(masks["cancelled"] | masks["diverted"] | masks["null_arrival_without_flag"])
    df = df.loc[keep].reset_index(drop=True)

    flight_date = pd.to_datetime(df["FL_DATE"], format="%m/%d/%Y %I:%M:%S %p").dt.normalize()

    sched_dep = parse_hhmm(df["CRS_DEP_TIME"], flight_date)
    dep = parse_hhmm(df["DEP_TIME"], flight_date)
    sched_arr = push_past_midnight(sched_dep, parse_hhmm(df["CRS_ARR_TIME"], flight_date))
    arr = push_past_midnight(dep, parse_hhmm(df["ARR_TIME"], flight_date))

    return pd.DataFrame(
        {
            "flight_date": flight_date.dt.date,
            "carrier": df["OP_UNIQUE_CARRIER"],
            "tail_number": df["TAIL_NUM"],
            "flight_number": pd.to_numeric(df["OP_CARRIER_FL_NUM"], errors="coerce").astype("Int64"),
            "origin": df["ORIGIN"],
            "dest": df["DEST"],
            "sched_dep": sched_dep,
            "dep": dep,
            "sched_arr": sched_arr,
            "arr": arr,
            "distance": pd.to_numeric(df["DISTANCE"], errors="coerce").astype("Int64"),
        }
    )[OUTPUT_COLUMNS]


def month_key(path: Path) -> str:
    match = re.search(r"(\d{4})-(\d{2})", path.name)
    if not match:
        raise ValueError(f"can't find a YYYY-MM month in {path.name}")
    return f"{match.group(1)}_{match.group(2)}"


def main() -> None:
    paths = sorted(Path(p) for p in glob.glob(RAW_GLOB))
    if not paths:
        raise SystemExit(f"no raw files found matching {RAW_GLOB}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for path in paths:
        raw = pd.read_csv(path)
        counts = drop_counts(raw)
        cleaned = clean_month(raw)

        out_path = CLEAN_DIR / f"flights_{month_key(path)}.parquet"
        cleaned.to_parquet(out_path, index=False)

        report_rows.append(
            {
                "month": month_key(path),
                "rows_in": len(raw),
                "rows_out": len(cleaned),
                **counts,
            }
        )

        print(
            f"{path.name}: {len(raw):,} rows -> {len(cleaned):,} rows "
            f"(cancelled={counts['cancelled']:,}, diverted={counts['diverted']:,}, "
            f"null_arrival_without_flag={counts['null_arrival_without_flag']:,}), wrote {out_path}"
        )

    report = pd.DataFrame(report_rows)
    report_path = CLEAN_DIR / "drop_report.csv"
    report.to_csv(report_path, index=False)
    print(f"\nwrote drop report to {report_path}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
