"""Fit a LightGBM connection-confidence classifier and report on the test set.

Also trains and saves the model actually used by the deployed app (app.py):
all twelve months, all five airports, calibrated -- see save_final_model.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

from src.features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, build_features, load_pairs

SUBSAMPLE_TARGET_ROWS = 5_000_000
SUBSAMPLE_SEED = 0

TRAIN_MONTHS = range(1, 11)  # January - October
TEST_MONTHS = range(11, 13)  # November - December

CALIB_FRACTION = 0.10  # held out for isotonic calibration, stratified across Jan-Oct

BASELINE_SLACK_THRESHOLD_MIN = 45
N_RELIABILITY_BUCKETS = 10

MODEL_PATH = Path("model.pkl")
CATEGORIES_PATH = Path("model_categories.json")


def stratified_subsample(features: pd.DataFrame, target_rows: int, seed: int) -> pd.DataFrame:
    """Subsample down to ~target_rows, preserving each airport's share of the data.

    A no-op once the data is already small enough (or for the eventual final run
    on the full training set, which should skip subsampling entirely).
    """
    if len(features) <= target_rows:
        return features

    frac = target_rows / len(features)
    parts = [
        group.sample(frac=frac, random_state=seed)
        for _, group in features.groupby("airport", observed=True)
    ]
    return pd.concat(parts, ignore_index=True)


def chronological_split(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train (Jan-Oct) vs. test (Nov-Dec). Neither side is subsampled here --
    the caller decides what, if anything, to subsample."""
    train = features.loc[features["month"].isin(TRAIN_MONTHS)]
    test = features.loc[features["month"].isin(TEST_MONTHS)]
    return train, test


def fit_calibration_split(
    train: pd.DataFrame, calib_frac: float = CALIB_FRACTION, seed: int = SUBSAMPLE_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the (already subsampled) training portion into a fit set and a
    calibration set, holding out calib_frac stratified across every (airport,
    month) in Jan-Oct -- not October alone, which carries its own seasonal
    offset relative to the rest of the year (see the airport x month analysis)."""
    calib_parts = [
        group.sample(frac=calib_frac, random_state=seed)
        for _, group in train.groupby(["airport", "month"], observed=True)
    ]
    calib_set = pd.concat(calib_parts, ignore_index=False)
    fit_set = train.drop(calib_set.index)
    return fit_set.reset_index(drop=True), calib_set.reset_index(drop=True)


def fit_model(train: pd.DataFrame) -> LGBMClassifier:
    model = LGBMClassifier(objective="binary", random_state=0)
    model.fit(train[FEATURE_COLUMNS], train["made"], categorical_feature=CATEGORICAL_COLUMNS)
    return model


def print_model_metrics(model: LGBMClassifier, test: pd.DataFrame, y_prob: np.ndarray) -> None:
    y_test = test["made"]
    y_pred = model.predict(test[FEATURE_COLUMNS])
    print("\n=== model ===")
    print(f"accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"AUC:      {roc_auc_score(y_test, y_prob):.4f}")


def print_baseline_metrics(test: pd.DataFrame) -> None:
    y_test = test["made"]

    print("\n=== baseline: always predict made ===")
    print(f"accuracy: {y_test.mean():.4f}")
    print("AUC:      n/a (a constant prediction has no ranking to score)")

    print(f"\n=== baseline: fixed {BASELINE_SLACK_THRESHOLD_MIN}-minute slack threshold ===")
    threshold_pred = test["slack_min"] >= BASELINE_SLACK_THRESHOLD_MIN
    print(f"accuracy: {accuracy_score(y_test, threshold_pred):.4f}")
    print(f"AUC (slack_min as the score): {roc_auc_score(y_test, test['slack_min']):.4f}")


def reliability_table(y_test: pd.Series, y_prob: np.ndarray, n_buckets: int = N_RELIABILITY_BUCKETS) -> pd.DataFrame:
    """Predicted vs. observed made rate per decile of predicted probability."""
    deciles = pd.qcut(y_prob, n_buckets, labels=False, duplicates="drop")
    by_bucket = pd.DataFrame({"decile": deciles, "predicted": y_prob, "observed": y_test.to_numpy()})
    return by_bucket.groupby("decile").agg(
        n=("observed", "size"),
        predicted_rate=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )


def print_feature_importance(model: LGBMClassifier) -> None:
    """Gain-based feature importance from the underlying booster."""
    booster = model.booster_
    gain = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=booster.feature_name(),
    ).sort_values(ascending=False)
    share = gain / gain.sum()

    print("\n=== feature importance (gain-based) ===")
    for name in gain.index:
        print(f"  {name:<12} gain={gain[name]:>16,.1f}  share={share[name]:.1%}")


def train_and_score_auc(fit_set: pd.DataFrame, test: pd.DataFrame, drop_columns: list[str]) -> float:
    """Fit a fresh model on fit_set minus drop_columns, return test-set AUC."""
    feature_columns = [c for c in FEATURE_COLUMNS if c not in drop_columns]
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c not in drop_columns]

    model = LGBMClassifier(objective="binary", random_state=0)
    model.fit(fit_set[feature_columns], fit_set["made"], categorical_feature=categorical_columns)
    y_prob = model.predict_proba(test[feature_columns])[:, 1]
    return roc_auc_score(test["made"], y_prob)


def run_ablation(fit_set: pd.DataFrame, test: pd.DataFrame) -> None:
    """Retrain with/without airport, month, and distance_in; gain-based importance
    is unreliable when features correlate, so this reports the actual AUC cost of
    dropping each."""
    runs = [
        ("all features", []),
        ("without airport", ["airport"]),
        ("without month", ["month"]),
        ("without distance_in", ["distance_in"]),
    ]
    print("\n=== ablation: AUC cost of dropping airport / month / distance_in ===")
    for label, drop_columns in runs:
        auc = train_and_score_auc(fit_set, test, drop_columns)
        print(f"  {label:<20} AUC={auc:.4f}")


def save_final_model(features: pd.DataFrame) -> None:
    """Train on everything -- all twelve months, all five airports -- calibrate,
    and write the artifacts app.py loads: model.pkl and model_categories.json
    (the dropdown options, taken from what the model actually saw in training).
    """
    print(f"loaded {len(features):,} pairs across {features['airport'].nunique()} airports")

    # A pandas categorical column keeps its full category list regardless of
    # which rows survive a split, so pull dropdown options before the split
    # rather than keeping `features` itself alive alongside fit_set/calib_set.
    categories = {column: sorted(features[column].cat.categories.tolist()) for column in CATEGORICAL_COLUMNS}

    fit_set, calib_set = fit_calibration_split(features)
    del features
    gc.collect()
    print(f"\nfinal model: fit {len(fit_set):,} rows, calibrate {len(calib_set):,} rows (all 12 months)")

    model = fit_model(fit_set)
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(calib_set[FEATURE_COLUMNS], calib_set["made"])

    dump(calibrated, MODEL_PATH)
    print(f"wrote {MODEL_PATH}")

    CATEGORIES_PATH.write_text(json.dumps(categories, indent=2))
    print(f"wrote {CATEGORIES_PATH}")


def main() -> None:
    # `--save-model` trains the final model in its own process, deliberately
    # separate from the dev-evaluation run below: on this machine's 7.7GB of
    # RAM, fitting on all 27.7M rows (unsampled) plus the dev pipeline's
    # in-memory splits at the same time gets OOM-killed. Run them as two
    # invocations instead of trying to free enough mid-process to fit both.
    if "--save-model" in sys.argv:
        # Passed straight through rather than bound to a local here: main()'s
        # own frame must not hold a second reference, or save_final_model's
        # `del features` mid-function won't actually free the memory.
        save_final_model(build_features(load_pairs()))
        return

    pairs = load_pairs()
    features = build_features(pairs)
    print(f"loaded {len(features):,} pairs across {features['airport'].nunique()} airports")

    train_full, test = chronological_split(features)
    print(f"train (Jan-Oct, full): {len(train_full):,} rows, test (Nov-Dec, full): {len(test):,} rows")

    train = stratified_subsample(train_full, SUBSAMPLE_TARGET_ROWS, SUBSAMPLE_SEED)
    print(f"training portion subsampled to {len(train):,} rows, stratified by airport (test left full-size)")

    fit_set, calib_set = fit_calibration_split(train)
    print(
        f"fit: {len(fit_set):,} rows, calibration: {len(calib_set):,} rows "
        f"({CALIB_FRACTION:.0%} stratified across airport x month, Jan-Oct)"
    )

    model = fit_model(fit_set)
    raw_prob = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    print_model_metrics(model, test, raw_prob)
    print_baseline_metrics(test)

    print("\n=== reliability BEFORE calibration ===")
    print(reliability_table(test["made"], raw_prob).to_string())

    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(calib_set[FEATURE_COLUMNS], calib_set["made"])
    calibrated_prob = calibrated.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    print("\n=== reliability AFTER isotonic calibration ===")
    print(reliability_table(test["made"], calibrated_prob).to_string())

    print_feature_importance(model)

    run_ablation(fit_set, test)


if __name__ == "__main__":
    main()
