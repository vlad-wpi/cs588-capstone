"""Streamlit UI: connection confidence estimator.

The screen is the Frontend class below: it collects the selections and renders what
the Backend (src/predictor.py) returns. It holds no model and no data of its own --
no training, no raw pairs/cleaned data. Crude layout on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import altair as alt
import pandas as pd
import streamlit as st

from src.lookups import layover_bucket
from src.predictor import MIN_COVERAGE_PAIRS, Backend

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


@dataclass
class Result:
    """What the screen shows for a covered combination."""

    probability: float
    recovery: dict


class Frontend:
    """The single screen: holds the current selections and renders the result."""

    airport: str
    origin: str
    carrier_in: str
    carrier_out: str
    month: int
    layover_min: int
    arrival_hour: int

    def __init__(self, backend: Backend):
        self.backend = backend

    def get_inputs(self) -> dict:
        """Render the input widgets and return the current selections."""
        lookups = self.backend.lookups

        self.airport = st.selectbox(
            "Airport", lookups.airports(), format_func=lambda c: code_label(c, AIRPORT_NAMES)
        )

        origins = lookups.origins_at(self.airport)
        origin_search = st.text_input("Search", "", placeholder="Type to filter")
        origin_options = search_codes(origins, lookups.origin_names, origin_search)

        if not origin_options:
            st.warning(f"No airports match '{origin_search}'. Showing all instead.")
            origin_options = search_codes(origins, lookups.origin_names, "")

        self.origin = st.selectbox(
            "Flying in from", origin_options, format_func=lambda c: code_label(c, lookups.origin_names)
        )

        self.carrier_in = st.selectbox(
            "Arriving carrier",
            lookups.carriers_in_at(self.airport),
            format_func=lambda c: code_label(c, lookups.carrier_names),
        )
        self.carrier_out = st.selectbox(
            "Departing carrier",
            lookups.carriers_out_at(self.airport),
            format_func=lambda c: code_label(c, lookups.carrier_names),
        )
        month_idx = st.selectbox("Month", options=list(range(12)), format_func=lambda i: MONTH_NAMES[i])
        self.month = month_idx + 1

        self.layover_min = st.slider("Layover (minutes)", min_value=30, max_value=240, value=60, step=5)
        self.arrival_hour = st.slider("Arrival hour", min_value=0, max_value=23, value=12)

        return {
            "airport": self.airport,
            "origin": self.origin,
            "carrier_in": self.carrier_in,
            "carrier_out": self.carrier_out,
            "month": self.month,
            "layover_min": self.layover_min,
            "arrival_hour": self.arrival_hour,
        }

    def render_not_covered(self, pair_count: int) -> None:
        """FR-11: say so instead of showing a result."""
        names = self.backend.lookups.carrier_names
        st.info(
            f"This combination isn't covered by the data: {code_label(self.carrier_in, names)} arriving and "
            f"{code_label(self.carrier_out, names)} departing at {code_label(self.airport, AIRPORT_NAMES)} has "
            f"{pair_count} training pairs (fewer than {MIN_COVERAGE_PAIRS}), too few for a reliable estimate."
        )

    def render_result(self, r: Result) -> None:
        """The probability in large type, its plain-language band, and the recovery panel."""
        pct = r.probability * 100

        st.markdown(f"# {pct:.0f}%")

        if pct >= 80:
            st.success("Good chance of making it.")
        elif pct >= 55:
            st.warning("Coin-flip territory -- could go either way.")
        else:
            st.error("Likely to miss this connection.")

        if not r.recovery:
            st.write("No recovery data for this airport/month/time-of-day combination.")
            return

        wait = r.recovery["median_wait_min"]
        sentence = (
            f"In {r.recovery['no_recovery_share']:.0%} of similar missed connections at {self.airport} "
            f"in {MONTH_NAMES[self.month - 1]}, no later same-day flight was available."
        )
        if pd.notna(wait):
            wait_text = f"{wait / 60:.1f} hours" if wait > 90 else f"{wait:.0f} minutes"
            sentence += f" When one was, the typical wait was about {wait_text}."
        st.write(sentence)

    def render_chart(self, rows: list) -> None:
        """Success rate by airport at the selected layover, selected airport highlighted."""
        columns = ["airport", "month", "layover_bucket", "n", "success_rate"]
        by_airport = pd.DataFrame(rows, columns=columns)
        by_airport["label"] = by_airport["airport"].map(lambda c: code_label(c, AIRPORT_NAMES))
        by_airport["selected"] = by_airport["airport"] == self.airport

        bucket = layover_bucket(pd.Series([self.layover_min])).iloc[0]
        st.subheader(f"Success rate by airport, {MONTH_NAMES[self.month - 1]}, {bucket} min layover")

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

    def run(self) -> None:
        st.title("Connection Confidence")
        st.caption("Estimated probability of making a connecting flight, from 2025 BTS on-time data.")

        inputs = self.get_inputs()

        if self.backend.is_covered(self.airport, self.carrier_in, self.carrier_out):
            self.render_result(
                Result(
                    probability=self.backend.predict(inputs),
                    recovery=self.backend.recovery(self.airport, self.month, self.arrival_hour),
                )
            )
            self.render_chart(self.backend.comparison(self.layover_min, self.month))
        else:
            self.render_not_covered(self.backend.pair_count(self.airport, self.carrier_in, self.carrier_out))

        st.caption(FOOTER)


@st.cache_resource
def load_backend() -> Backend:
    return Backend.load()


def main() -> None:
    Frontend(load_backend()).run()


if __name__ == "__main__":
    main()
