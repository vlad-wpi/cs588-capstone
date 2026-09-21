from __future__ import annotations

import pandas as pd

from src.lookups import build_coverage, count_carrier_pairs


def as_dict(coverage: pd.DataFrame) -> dict:
    return {
        (row.airport, row.carrier_in, row.carrier_out): row.n_pairs
        for row in coverage.itertuples(index=False)
    }


def test_count_carrier_pairs_counts_each_combination():
    pairs = pd.DataFrame(
        {
            "airport": ["ORD"] * 5,
            "carrier_in": ["AA", "AA", "AA", "UA", "UA"],
            "carrier_out": ["UA", "UA", "DL", "AA", "AA"],
        }
    )
    counts = as_dict(count_carrier_pairs(pairs))

    assert counts == {("ORD", "AA", "UA"): 2, ("ORD", "AA", "DL"): 1, ("ORD", "UA", "AA"): 2}


def test_count_carrier_pairs_omits_combinations_with_no_pairs():
    pairs = pd.DataFrame({"airport": ["ORD"], "carrier_in": ["AA"], "carrier_out": ["UA"]})
    counts = as_dict(count_carrier_pairs(pairs))

    assert ("ORD", "UA", "AA") not in counts


def test_build_coverage_combines_files_and_sums_repeats(tmp_path):
    ord_pairs = pd.DataFrame(
        {"airport": ["ORD", "ORD", "ORD"], "carrier_in": ["AA", "AA", "UA"], "carrier_out": ["UA", "UA", "AA"]}
    )
    atl_pairs = pd.DataFrame({"airport": ["ATL", "ATL"], "carrier_in": ["DL", "DL"], "carrier_out": ["DL", "DL"]})
    # A second file for the same airport repeating a combination must add to its count.
    ord_more = pd.DataFrame({"airport": ["ORD"], "carrier_in": ["AA"], "carrier_out": ["UA"]})

    paths = []
    for name, frame in [("pairs_ord.parquet", ord_pairs), ("pairs_atl.parquet", atl_pairs), ("pairs_ord2.parquet", ord_more)]:
        path = tmp_path / name
        frame.to_parquet(path, index=False)
        paths.append(path)

    counts = as_dict(build_coverage(paths))

    assert counts == {("ORD", "AA", "UA"): 3, ("ORD", "UA", "AA"): 1, ("ATL", "DL", "DL"): 2}
