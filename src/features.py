"""Labeled connection pairs -> model features.

Reads every `data/pairs/pairs_*.parquet` file and selects/types the columns the
model trains on. Categorical columns (airport, carrier_in, carrier_out) become
pandas `category` dtype rather than one-hot columns, so LightGBM can split on
them natively.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PAIRS_DIR = Path("data/pairs")

NUMERIC_DTYPES = {
    "slack_min": "int16",  # up to 240
    "arr_hour": "int8",  # 0-23
    "day_of_week": "int8",  # 0-6
    "month": "int8",  # 1-12
    "distance_in": "int16",  # miles; comfortably under int16's range
}
NUMERIC_COLUMNS = list(NUMERIC_DTYPES)
CATEGORICAL_COLUMNS = [
    "airport",
    "carrier_in",
    "carrier_out",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET_COLUMN = "made"


def load_pairs(pattern: str = "pairs_*.parquet") -> pd.DataFrame:
    paths = sorted(PAIRS_DIR.glob(pattern))
    if not paths:
        raise SystemExit(f"no pairs files found matching {PAIRS_DIR / pattern}")
    return pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True)


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Select and type the model-ready feature columns (+ target) from pairs rows."""
    features = pairs[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column, dtype in NUMERIC_DTYPES.items():
        features[column] = features[column].astype(dtype)
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].astype("category")
    return features


def main() -> None:
    pairs = load_pairs()
    features = build_features(pairs)
    print(f"{len(features):,} rows")
    print(features.dtypes)


if __name__ == "__main__":
    main()
