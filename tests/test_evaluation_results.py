"""Regression floors on the committed results/evaluation.json.

These thresholds are floors set with headroom BELOW the model's actual current
performance (see results/evaluation.md for the real numbers), not the values
themselves. This test guards against a worse model being committed, not a live
check of the current model -- it reads the committed record and does not
retrain or recompute anything. Regenerate the record with `python -m src.train`
(full dataset) when the model legitimately changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS_PATH = Path("results/evaluation.json")

MIN_MODEL_AUC = 0.78  # current run: 0.8125 -- headroom, not the live value
MAX_RELIABILITY_GAP = 0.05  # current run: 0.0370 (after calibration)

EXPECTED_PROVENANCE_KEYS = {
    "git_sha",
    "git_dirty",
    "generated_at_utc",
    "model_under_evaluation",
    "row_counts",
    "parameters",
    "versions",
}
EXPECTED_ROW_COUNT_KEYS = {"train", "calibration", "test"}
EXPECTED_PARAMETER_KEYS = {
    "mct_min",
    "slack_window_min",
    "candidate_cap_per_arrival",
    "pairs_sampling_seed",
    "train_subsample_seed",
    "subsample_target_rows",
    "train_months",
    "test_months",
}
EXPECTED_VERSION_KEYS = {"lightgbm", "scikit-learn", "pandas", "numpy"}


@pytest.fixture(scope="module")
def results() -> dict:
    assert RESULTS_PATH.exists(), f"{RESULTS_PATH} is missing -- run `python -m src.train` and commit it"
    return json.loads(RESULTS_PATH.read_text())


def test_model_beats_the_slack_threshold_baseline_auc(results):
    assert results["model"]["auc"] > results["baselines"]["slack_threshold"]["auc"]


def test_model_auc_clears_the_floor(results):
    assert results["model"]["auc"] >= MIN_MODEL_AUC


def test_model_beats_the_always_predict_made_baseline_accuracy(results):
    assert results["model"]["accuracy"] > results["baselines"]["always_predict_made"]["accuracy"]


def test_calibrated_reliability_gap_is_within_tolerance(results):
    after = results["reliability"]["after_calibration"]
    assert after, "after_calibration has no deciles to check"

    max_gap = max(abs(row["predicted_rate"] - row["observed_rate"]) for row in after)
    assert max_gap <= MAX_RELIABILITY_GAP


def test_provenance_block_is_present_and_complete(results):
    provenance = results["provenance"]
    assert EXPECTED_PROVENANCE_KEYS <= set(provenance)

    assert isinstance(provenance["git_sha"], str) and len(provenance["git_sha"]) == 40
    assert isinstance(provenance["git_dirty"], bool)
    assert isinstance(provenance["generated_at_utc"], str) and provenance["generated_at_utc"]

    model_under_eval = provenance["model_under_evaluation"]
    assert model_under_eval["is_deployed_model"] is False
    assert isinstance(model_under_eval["deployed_model_path"], str) and model_under_eval["deployed_model_path"]
    assert isinstance(model_under_eval["note"], str) and len(model_under_eval["note"]) > 20

    row_counts = provenance["row_counts"]
    assert set(row_counts) == EXPECTED_ROW_COUNT_KEYS
    assert all(isinstance(v, int) and v > 0 for v in row_counts.values())

    parameters = provenance["parameters"]
    assert set(parameters) == EXPECTED_PARAMETER_KEYS
    assert parameters["slack_window_min"]["min"] < parameters["slack_window_min"]["max"]
    assert parameters["train_months"] and parameters["test_months"]

    versions = provenance["versions"]
    assert set(versions) == {"pinned", "installed"}
    for kind in ("pinned", "installed"):
        assert set(versions[kind]) == EXPECTED_VERSION_KEYS
        assert all(isinstance(v, str) and v for v in versions[kind].values())


def test_installed_versions_match_pinned_versions(results):
    """A drift between what requirements.txt pins and what actually ran should
    fail loudly here, not get silently recorded as two numbers nobody compares."""
    versions = results["provenance"]["versions"]
    assert versions["installed"] == versions["pinned"]
