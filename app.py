"""Streamlit UI: connection confidence estimator.

Loads model.pkl plus the small lookup tables and does nothing else -- no
training, no raw pairs/cleaned data. Crude layout on purpose.
"""

from __future__ import annotations

import json

import altair as alt
import joblib
import pandas as pd
import streamlit as st

from src.features import FEATURE_COLUMNS, TARGET_COLUMN, build_features
from src.lookups import arr_block, layover_bucket

MODEL_PATH = "model.pkl"
CATEGORIES_PATH = "model_categories.json"
CARRIERS_PATH = "data/lookups/carriers.csv"
ORIGIN_NAMES_PATH = "data/lookups/airports.csv"
ORIGIN_DISTANCES_PATH = "data/lookups/origin_distances.parquet"
RECOVERY_PATH = "data/lookups/recovery.parquet"
COMPARISON_PATH = "data/lookups/comparison.parquet"
COVERAGE_PATH = "data/lookups/coverage.parquet"

# FR-11: a carrier combination with fewer training pairs than this at the chosen
# airport isn't covered by the data, so the app says so instead of predicting.
MIN_COVERAGE_PAIRS = 100

FOOTER = "Results are historical estimates, not guarantees -- based on 2025 BTS on-time data."

MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

# Hardcoded, not sourced from a table: only five airports, unlikely to change often.
AIRPORT_NAMES = {
    "ATL": "Hartsfield-Jackson Atlanta",
    "CLT": "Charlotte Douglas",
    "DEN": "Denver",
    "DFW": "Dallas/Fort Worth",
    "ORD": "Chicago O'Hare",
}

# Not exposed as a UI input, fixed to a neutral value: a sensitivity sweep found
# it moves the prediction by under 2 points across all 7 days, in every scenario
# tested -- a footnote, unlike distance_in (which is why that one got a real input).
DEFAULT_DAY_OF_WEEK = 2  # Wednesday


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_categories() -> dict:
    with open(CATEGORIES_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_carrier_names() -> dict:
    carriers = pd.read_csv(CARRIERS_PATH)
    return dict(zip(carriers["code"], carriers["name"]))


@st.cache_resource
def load_origin_names() -> dict:
    airports = pd.read_csv(ORIGIN_NAMES_PATH)
    return dict(zip(airports["code"], airports["name"]))


@st.cache_data
def load_origin_distances() -> pd.DataFrame:
    return pd.read_parquet(ORIGIN_DISTANCES_PATH)


def code_label(code: str, names: dict) -> str:
    """'American Airlines (AA)', or just 'AA' if the code has no known name."""
    name = names.get(code)
    return f"{name} ({code})" if name else code


def search_codes(codes: list[str], names: dict, term: str) -> list[str]:
    """Plain substring filter over label-or-code, case-insensitive -- Streamlit's
    selectbox does fuzzy matching by default, which surfaces "Chippewa County"
    before any Chicago airport for "chi". Prefix matches on the label sort
    first, then other substring matches, alphabetically within each group."""
    term = term.strip().lower()
    if not term:
        return sorted(codes, key=lambda c: code_label(c, names).lower())

    matched = [c for c in codes if term in code_label(c, names).lower() or term in c.lower()]
    return sorted(
        matched,
        key=lambda c: (0 if code_label(c, names).lower().startswith(term) else 1, code_label(c, names).lower()),
    )


@st.cache_data
def load_lookups() -> tuple[pd.DataFrame, pd.DataFrame]:
    recovery = pd.read_parquet(RECOVERY_PATH)
    comparison = pd.read_parquet(COMPARISON_PATH)
    return recovery, comparison


@st.cache_data
def load_coverage() -> pd.DataFrame:
    return pd.read_parquet(COVERAGE_PATH)


model = load_model()
categories = load_categories()
carrier_names = load_carrier_names()
origin_names = load_origin_names()
origin_distances = load_origin_distances()
recovery, comparison = load_lookups()
coverage = load_coverage()

st.title("Connection Confidence")
st.caption("Estimated probability of making a connecting flight, from 2025 BTS on-time data.")

airport = st.selectbox(
    "Airport", categories["airport"], format_func=lambda c: code_label(c, AIRPORT_NAMES)
)

origins_here = origin_distances.loc[origin_distances["airport"] == airport]
origin_search = st.text_input("Search", "", placeholder="Type to filter")
origin_options = search_codes(origins_here["origin"].tolist(), origin_names, origin_search)

if not origin_options:
    st.warning(f"No airports match '{origin_search}'. Showing all instead.")
    origin_options = search_codes(origins_here["origin"].tolist(), origin_names, "")

origin = st.selectbox(
    "Flying in from", origin_options, format_func=lambda c: code_label(c, origin_names)
)
distance_in = int(origins_here.loc[origins_here["origin"] == origin, "distance_in"].iloc[0])

coverage_here = coverage.loc[coverage["airport"] == airport]
carrier_in = st.selectbox(
    "Arriving carrier",
    sorted(coverage_here["carrier_in"].unique()),
    format_func=lambda c: code_label(c, carrier_names),
)
carrier_out = st.selectbox(
    "Departing carrier",
    sorted(coverage_here["carrier_out"].unique()),
    format_func=lambda c: code_label(c, carrier_names),
)
month_idx = st.selectbox("Month", options=list(range(12)), format_func=lambda i: MONTH_NAMES[i])
month = month_idx + 1

layover = st.slider("Layover (minutes)", min_value=30, max_value=240, value=60, step=5)
arr_hour = st.slider("Arrival hour", min_value=0, max_value=23, value=12)

# --- coverage (FR-11) ---------------------------------------------------------

pair_count = int(
    coverage_here.loc[
        (coverage_here["carrier_in"] == carrier_in) & (coverage_here["carrier_out"] == carrier_out), "n_pairs"
    ].sum()
)
if pair_count < MIN_COVERAGE_PAIRS:
    st.info(
        f"This combination isn't covered by the data: {code_label(carrier_in, carrier_names)} arriving and "
        f"{code_label(carrier_out, carrier_names)} departing at {code_label(airport, AIRPORT_NAMES)} has "
        f"{pair_count} training pairs (fewer than {MIN_COVERAGE_PAIRS}), too few for a reliable estimate."
    )
    st.caption(FOOTER)
    st.stop()

# --- prediction ---------------------------------------------------------------

query = pd.DataFrame(
    [
        {
            "airport": airport,
            "carrier_in": carrier_in,
            "carrier_out": carrier_out,
            "slack_min": layover,
            "arr_hour": arr_hour,
            "day_of_week": DEFAULT_DAY_OF_WEEK,
            "month": month,
            "distance_in": distance_in,
            TARGET_COLUMN: True,  # dummy; build_features needs the column, prediction ignores it
        }
    ]
)
features = build_features(query)
prob = model.predict_proba(features[FEATURE_COLUMNS])[0, 1]
pct = prob * 100

st.markdown(f"# {pct:.0f}%")

if pct >= 80:
    st.success("Good chance of making it.")
elif pct >= 55:
    st.warning("Coin-flip territory -- could go either way.")
else:
    st.error("Likely to miss this connection.")

# --- recovery sentence ---------------------------------------------------------

block = arr_block(pd.Series([arr_hour])).iloc[0]
recovery_row = recovery.loc[
    (recovery["airport"] == airport) & (recovery["month"] == month) & (recovery["arr_block"] == block)
]

if len(recovery_row):
    wait = recovery_row["median_wait_min"].iloc[0]
    no_recovery = recovery_row["no_recovery_share"].iloc[0]

    sentence = (
        f"In {no_recovery:.0%} of similar missed connections at {airport} in {MONTH_NAMES[month_idx]}, "
        f"no later same-day flight was available."
    )
    if pd.notna(wait):
        wait_text = f"{wait / 60:.1f} hours" if wait > 90 else f"{wait:.0f} minutes"
        sentence += f" When one was, the typical wait was about {wait_text}."
    st.write(sentence)
else:
    st.write("No recovery data for this airport/month/time-of-day combination.")

# --- bar chart: success rate by airport at the selected layover ----------------

bucket = layover_bucket(pd.Series([layover])).iloc[0]
by_airport = comparison.loc[(comparison["month"] == month) & (comparison["layover_bucket"] == bucket)].copy()
by_airport["label"] = by_airport["airport"].map(lambda c: code_label(c, AIRPORT_NAMES))
by_airport["selected"] = by_airport["airport"] == airport

st.subheader(f"Success rate by airport, {MONTH_NAMES[month_idx]}, {bucket} min layover")

chart = (
    alt.Chart(by_airport)
    .mark_bar()
    .encode(
        x=alt.X("label:N", title="Airport", sort=None),
        y=alt.Y(
            "success_rate:Q",
            title="Success rate",
            axis=alt.Axis(format="%", values=[i / 10 for i in range(11)]),
            scale=alt.Scale(domain=[0, 1]),
        ),
        color=alt.Color(
            "selected:N",
            legend=None,
            scale=alt.Scale(domain=[False, True], range=["#4c78a8", "#e45756"]),
        ),
        tooltip=["label:N", alt.Tooltip("success_rate:Q", format=".1%")],
    )
)
st.altair_chart(chart, use_container_width=True)

st.caption(FOOTER)
