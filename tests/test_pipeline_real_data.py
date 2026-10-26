"""System tests: the real cleaner and pairs code against a committed slice of
real BTS data (tests/fixtures/ord_window.csv -- see scripts/make_fixture.py).

Nobody knows the "right" cleaned output or pair count for three real days at
ORD, so these can't assert exact values the way the unit tests do. They assert
invariants instead: the ones schema.md already lists as a manual pre-handoff
checklist for the cleaner, plus a few of the same spirit for the pairs builder.

Marked @pytest.mark.realdata (registered in pytest.ini) and run on their own CI
workflow (.github/workflows/real-data-tests.yml), separate from tests.yml. The
point of the split: a red badge on tests.yml should mean the code is broken,
not that some property of a particular real day's traffic happened to shift.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.cleaner import OUTPUT_COLUMNS as CLEANED_COLUMNS
from src.cleaner import clean_month, drop_counts
from src.pairs import (
    DEFAULT_SEED,
    MAX_CANDIDATES_PER_ARRIVAL,
    MAX_SLACK_MIN,
    MIN_SLACK_MIN,
    _match_and_cap,
    build_pairs,
)

pytestmark = pytest.mark.realdata

FIXTURE_PATH = Path("tests/fixtures/ord_window.csv")
AIRPORT = "ORD"

# pairs.py's public output carries no arrival ID, so test_no_arrival_exceeds_the_
# candidate_cap reconstructs "which arrival does this pair belong to" from the
# best available proxy: every inbound-only column the output carries. Shared with
# test_arrival_proxy_key_is_one_to_one_with_ground_truth_identity below, which
# proves this key is actually safe to use that way for the committed fixture --
# if the fixture ever changes, that test is what catches this key silently
# starting to merge two distinct arrivals together.
ARRIVAL_PROXY_KEY = ["carrier_in", "tail_in", "arr_hour", "day_of_week", "month", "distance_in"]

# A real arrival's (carrier, flight_number, sched_arr) is a full-precision,
# unique identifier for that specific flight -- unlike arr_hour (hour only) in
# the proxy key above. Only available pre-output, via _match_and_cap's own
# arrival-side columns (suffixed _in), since build_pairs's public OUTPUT_COLUMNS
# drops flight_number_in and collapses sched_arr_in down to arr_hour.
GROUND_TRUTH_ARRIVAL_KEY = ["carrier_in", "flight_number_in", "sched_arr_in"]

# What clean_month() actually produces on real data: object dtype for flight_date
# (python date objects) and the plain string columns, nullable Int64 for the two
# integer columns (flight_number, distance), datetime64[ns] for the four
# timestamps. Checked here against real output, not assumed from schema.md's
# logical types (e.g. schema.md calls flight_date a "date", which is this dtype,
# not a pandas datetime64 column).
EXPECTED_DTYPES = {
    "flight_date": "object",
    "carrier": "object",
    "tail_number": "object",
    "flight_number": "Int64",
    "origin": "object",
    "dest": "object",
    "sched_dep": "datetime64[ns]",
    "dep": "datetime64[ns]",
    "sched_arr": "datetime64[ns]",
    "arr": "datetime64[ns]",
    "distance": "Int64",
}


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    assert FIXTURE_PATH.exists(), f"{FIXTURE_PATH} is missing -- run scripts/make_fixture.py"
    return pd.read_csv(FIXTURE_PATH)


@pytest.fixture(scope="module")
def cleaned(raw: pd.DataFrame) -> pd.DataFrame:
    return clean_month(raw)


@pytest.fixture(scope="module")
def pairs(cleaned: pd.DataFrame) -> pd.DataFrame:
    return build_pairs(cleaned, cleaned, AIRPORT)


# --- cleaner invariants (schema.md's "Checks before handoff") ------------------


def test_cleaned_output_has_the_expected_columns_in_order_and_dtype(cleaned):
    assert list(cleaned.columns) == CLEANED_COLUMNS
    assert cleaned.dtypes.astype(str).to_dict() == EXPECTED_DTYPES


def test_no_nulls_in_the_timestamp_columns_after_drops(cleaned):
    timestamp_columns = ["sched_dep", "dep", "sched_arr", "arr"]
    assert int(cleaned[timestamp_columns].isna().sum().sum()) == 0


def test_no_arrival_timestamp_earlier_than_its_own_departure(cleaned):
    # Both the scheduled pair and the actual pair -- clean_month pushes both
    # past midnight independently (see push_past_midnight in src/cleaner.py).
    assert int((cleaned["sched_arr"] < cleaned["sched_dep"]).sum()) == 0
    assert int((cleaned["arr"] < cleaned["dep"]).sum()) == 0


def test_cleaned_row_count_equals_fixture_rows_minus_dropped_rows(raw, cleaned):
    counts = drop_counts(raw)
    dropped_total = sum(counts.values())

    # Sanity on the fixture itself: scripts/make_fixture.py's window was chosen
    # specifically to contain at least one cancelled and one diverted row, so if
    # nothing was dropped, the fixture (or this check) is broken, not just quiet.
    assert dropped_total > 0, f"expected some dropped rows in the fixture, got drop_counts={counts}"
    assert len(cleaned) == len(raw) - dropped_total


# --- pairs invariants ------------------------------------------------------------


def test_every_pairs_layover_is_inside_the_configured_window(pairs):
    assert len(pairs) > 0, "fixture produced zero pairs -- nothing for this test file to check"
    assert pairs["slack_min"].between(MIN_SLACK_MIN, MAX_SLACK_MIN).all()


def test_both_legs_of_every_pair_are_at_the_connecting_airport(cleaned, pairs):
    # pairs.py's output doesn't carry the inbound leg's own origin or the outbound
    # leg's own origin directly (both are implied by "airport"), so this checks it
    # the other way: every leg a pair claims must trace back to a real row in
    # `cleaned` whose dest (inbound) or origin (outbound) is actually AIRPORT.
    arrivals_into = cleaned.loc[cleaned["dest"] == AIRPORT, ["carrier", "tail_number", "distance"]]
    arrival_keys = set(arrivals_into.itertuples(index=False, name=None))
    pair_inbound_keys = set(zip(pairs["carrier_in"], pairs["tail_in"], pairs["distance_in"]))
    assert pair_inbound_keys, "no inbound legs to check"
    assert pair_inbound_keys <= arrival_keys

    departures_from = cleaned.loc[cleaned["origin"] == AIRPORT, ["carrier", "tail_number", "sched_dep", "dest"]]
    departure_keys = set(departures_from.itertuples(index=False, name=None))
    pair_outbound_keys = set(zip(pairs["carrier_out"], pairs["tail_out"], pairs["sched_dep_out"], pairs["dest_out"]))
    assert pair_outbound_keys, "no outbound legs to check"
    assert pair_outbound_keys <= departure_keys


def test_arrival_proxy_key_is_one_to_one_with_ground_truth_arrival_identity(cleaned, pairs):
    """Validates ARRIVAL_PROXY_KEY itself, which test_no_arrival_exceeds_the_
    candidate_cap depends on to mean anything: if this key ever merged two
    distinct arrivals into one group, that test would silently stop checking
    the real per-arrival cap and just check a merged group's size instead. This
    test is what would catch that -- on a fixture change, a key collision here
    means the cap test's grouping is no longer trustworthy and needs revisiting
    before trusting it again.

    Proves it directly rather than just comparing group counts (which a
    coincidence could satisfy without the two keys partitioning the same rows
    the same way): every ground-truth arrival maps to exactly one proxy group,
    and every proxy group maps to exactly one ground-truth arrival -- a true
    bijection between the two partitions of the same pairs, not merely equal
    cardinality.
    """
    arrivals = cleaned.loc[cleaned["dest"] == AIRPORT].reset_index(drop=True)
    departures = cleaned.loc[cleaned["origin"] == AIRPORT].reset_index(drop=True)

    # The ground-truth key must itself be unique per real arrival row before
    # it's trusted as ground truth for anything else.
    identity_cols = ["carrier", "flight_number", "sched_arr"]
    assert arrivals.drop_duplicates(subset=identity_cols).shape[0] == len(arrivals), (
        "GROUND_TRUTH_ARRIVAL_KEY is not actually unique per arrival in this fixture -- "
        "it can't be used to validate the proxy key until it is"
    )

    # The same internal call build_pairs() makes, so `candidates` lines up
    # row-for-row with `pairs` and still carries flight_number_in/sched_arr_in,
    # which don't survive into build_pairs()'s public output.
    candidates = _match_and_cap(arrivals, departures, MIN_SLACK_MIN, MAX_SLACK_MIN, MAX_CANDIDATES_PER_ARRIVAL, DEFAULT_SEED)
    assert len(candidates) == len(pairs)  # same rows, same order -- required for the row-aligned comparison below

    ground_truth_group = candidates.groupby(GROUND_TRUTH_ARRIVAL_KEY, sort=False).ngroup()
    proxy_group = pairs.groupby(ARRIVAL_PROXY_KEY, sort=False).ngroup()

    ground_truth_to_proxy = pd.Series(proxy_group.values, index=ground_truth_group.values).groupby(level=0).nunique()
    proxy_to_ground_truth = pd.Series(ground_truth_group.values, index=proxy_group.values).groupby(level=0).nunique()

    assert (ground_truth_to_proxy == 1).all(), "a ground-truth arrival spans more than one proxy group"
    assert (proxy_to_ground_truth == 1).all(), "a proxy group merges more than one distinct arrival"


def test_no_arrival_exceeds_the_candidate_cap(pairs):
    # Deliberately excludes slack_min, carrier_out, tail_out, dest_out, and
    # sched_dep_out from ARRIVAL_PROXY_KEY -- those vary WITHIN one arrival's own
    # candidate set, so grouping on any of them would fragment a single arrival's
    # pairs into many spurious one-row groups instead of the one group a real
    # per-arrival cap check needs. See test_arrival_proxy_key_is_one_to_one_with_
    # ground_truth_arrival_identity above for why this key is trustworthy here.
    group_sizes = pairs.groupby(ARRIVAL_PROXY_KEY).size()
    assert int(group_sizes.max()) <= MAX_CANDIDATES_PER_ARRIVAL


def test_same_seed_produces_identical_output(cleaned):
    first = build_pairs(cleaned, cleaned, AIRPORT, seed=DEFAULT_SEED)
    second = build_pairs(cleaned, cleaned, AIRPORT, seed=DEFAULT_SEED)
    pd.testing.assert_frame_equal(first, second)
