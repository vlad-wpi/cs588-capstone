"""Backend: the trained model plus the lookup tables, one method per figure the screen shows.

No Streamlit and no raw flight data -- the deployed app reads model.pkl and the small
files in data/lookups/, and this class is all it needs to answer a query. It is kept
apart from app.py so it can be tested without a running UI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from src.lookups import LOOKUPS_DIR, arr_block, layover_bucket

MODEL_PATH = Path("model.pkl")
CATEGORIES_PATH = Path("model_categories.json")

# FR-11: a carrier combination with fewer training pairs than this at the chosen
# airport isn't covered by the data, so the app says so instead of predicting.
MIN_COVERAGE_PAIRS = 100

# Not a user input, fixed to a neutral value: a sensitivity sweep found it moves the
# prediction by under 2 points across all 7 days, in every scenario tested -- a
# footnote, unlike distance_in (which is why that one is derived from a real input).
DEFAULT_DAY_OF_WEEK = 2  # Wednesday


@dataclass
class LookupTables:
    """Everything the app reads besides the model, plus the queries the screen needs."""

    recovery: pd.DataFrame
    comparison: pd.DataFrame
    coverage: pd.DataFrame
    origin_distances: pd.DataFrame
    categories: dict
    carrier_names: dict
    origin_names: dict

    @classmethod
    def load(
        cls, lookups_dir: Path = LOOKUPS_DIR, categories_path: Path = CATEGORIES_PATH
    ) -> LookupTables:
        carriers = pd.read_csv(lookups_dir / "carriers.csv")
        airports = pd.read_csv(lookups_dir / "airports.csv")
        return cls(
            recovery=pd.read_parquet(lookups_dir / "recovery.parquet"),
            comparison=pd.read_parquet(lookups_dir / "comparison.parquet"),
            coverage=pd.read_parquet(lookups_dir / "coverage.parquet"),
            origin_distances=pd.read_parquet(lookups_dir / "origin_distances.parquet"),
            categories=json.loads(Path(categories_path).read_text()),
            carrier_names=dict(zip(carriers["code"], carriers["name"])),
            origin_names=dict(zip(airports["code"], airports["name"])),
        )

    def airports(self) -> list[str]:
        """Connecting airports the model was trained on."""
        return list(self.categories["airport"])

    def origins_at(self, airport: str) -> list[str]:
        """Origins that actually fly into `airport` in the data."""
        rows = self.origin_distances
        return rows.loc[rows["airport"] == airport, "origin"].tolist()

    def carriers_in_at(self, airport: str) -> list[str]:
        """Arriving carriers with at least one pair at `airport`."""
        rows = self.coverage
        return sorted(rows.loc[rows["airport"] == airport, "carrier_in"].unique())

    def carriers_out_at(self, airport: str) -> list[str]:
        """Departing carriers with at least one pair at `airport`."""
        rows = self.coverage
        return sorted(rows.loc[rows["airport"] == airport, "carrier_out"].unique())

    def distance(self, airport: str, origin: str) -> int:
        rows = self.origin_distances
        match = rows.loc[(rows["airport"] == airport) & (rows["origin"] == origin), "distance_in"]
        if match.empty:
            raise ValueError(f"no route data for {origin} -> {airport}")
        return int(match.iloc[0])


class Backend:
    """The loaded model and lookup tables, answering one query per figure on screen."""

    def __init__(self, model, lookups: LookupTables):
        self.model = model
        self.lookups = lookups

    @classmethod
    def load(cls) -> Backend:
        return cls(joblib.load(MODEL_PATH), LookupTables.load())

    def predict(self, inputs: dict) -> float:
        """Probability (0-1) of making the connection described by `inputs`:
        airport, origin, carrier_in, carrier_out, month, layover_min, arrival_hour."""
        query = pd.DataFrame(
            [
                {
                    "airport": inputs["airport"],
                    "carrier_in": inputs["carrier_in"],
                    "carrier_out": inputs["carrier_out"],
                    "slack_min": inputs["layover_min"],
                    "arr_hour": inputs["arrival_hour"],
                    "day_of_week": DEFAULT_DAY_OF_WEEK,
                    "month": inputs["month"],
                    "distance_in": self.lookups.distance(inputs["airport"], inputs["origin"]),
                    TARGET_COLUMN: True,  # dummy; build_features needs the column, prediction ignores it
                }
            ]
        )
        features = build_features(query)
        return float(self.model.predict_proba(features[FEATURE_COLUMNS])[0, 1])

    def recovery(self, airport: str, month: int, arrival_hour: int) -> dict:
        """What a missed connection typically costs: {'median_wait_min',
        'no_recovery_share', 'n_missed'}, or {} if the data has nothing for this
        airport/month/time of day. `median_wait_min` covers only the missed pairs
        that had a later catchable flight, and is NaN if none did.

        Takes arrival_hour beyond (airport, month): the figure is keyed by arrival
        time block (morning/midday/evening/late), not just airport and month.
        """
        block = arr_block(pd.Series([arrival_hour])).iloc[0]
        rows = self.lookups.recovery
        match = rows.loc[(rows["airport"] == airport) & (rows["month"] == month) & (rows["arr_block"] == block)]
        if match.empty:
            return {}
        row = match.iloc[0]
        return {
            "median_wait_min": float(row["median_wait_min"]),
            "no_recovery_share": float(row["no_recovery_share"]),
            "n_missed": int(row["n_missed"]),
        }

    def comparison(self, layover_min: int, month: int) -> list:
        """Observed success rate at each airport for this layover's 15-minute
        bucket, as a list of {'airport', 'month', 'layover_bucket', 'n', 'success_rate'}.

        Takes month beyond layover_min: the figure is per month, not all-year.
        """
        bucket = layover_bucket(pd.Series([layover_min])).iloc[0]
        rows = self.lookups.comparison
        return rows.loc[(rows["month"] == month) & (rows["layover_bucket"] == bucket)].to_dict("records")

    def pair_count(self, airport: str, carrier_in: str, carrier_out: str) -> int:
        """Training pairs behind this carrier combination at this airport."""
        rows = self.lookups.coverage
        match = rows.loc[
            (rows["airport"] == airport) & (rows["carrier_in"] == carrier_in) & (rows["carrier_out"] == carrier_out),
            "n_pairs",
        ]
        return int(match.sum())

    def is_covered(self, airport: str, carrier_in: str, carrier_out: str) -> bool:
        """False if the data has too few pairs for this combination to say anything (FR-11)."""
        return self.pair_count(airport, carrier_in, carrier_out) >= MIN_COVERAGE_PAIRS
