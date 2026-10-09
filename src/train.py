"""Fit a LightGBM connection-confidence classifier and report on the test set.

Also trains and saves the model actually used by the deployed app (app.py):
all twelve months, all five airports, calibrated -- see save_final_model.

The dev-evaluation run (main(), without --save-model) also writes its results
to results/evaluation.json and results/evaluation.md -- see write_results.
"""

from __future__ import annotations

import gc
import importlib.metadata as importlib_metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score

from src import pairs as pairs_module
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

RESULTS_DIR = Path("results")
REQUIREMENTS_PATH = Path("requirements.txt")
TRACKED_PACKAGES = ["lightgbm", "scikit-learn", "pandas", "numpy"]


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


def compute_model_metrics(model: LGBMClassifier, test: pd.DataFrame, y_prob: np.ndarray) -> dict:
    y_test = test["made"]
    y_pred = model.predict(test[FEATURE_COLUMNS])
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_prob)),
    }


def print_model_metrics(metrics: dict) -> None:
    print("\n=== model ===")
    print(f"accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC:      {metrics['auc']:.4f}")


def compute_baseline_metrics(test: pd.DataFrame) -> dict:
    y_test = test["made"]
    threshold_pred = test["slack_min"] >= BASELINE_SLACK_THRESHOLD_MIN
    return {
        "always_predict_made": {"accuracy": float(y_test.mean())},
        "slack_threshold": {
            "threshold_min": BASELINE_SLACK_THRESHOLD_MIN,
            "accuracy": float(accuracy_score(y_test, threshold_pred)),
            "auc": float(roc_auc_score(y_test, test["slack_min"])),
        },
    }


def print_baseline_metrics(baselines: dict) -> None:
    always = baselines["always_predict_made"]
    print("\n=== baseline: always predict made ===")
    print(f"accuracy: {always['accuracy']:.4f}")
    print("AUC:      n/a (a constant prediction has no ranking to score)")

    threshold = baselines["slack_threshold"]
    print(f"\n=== baseline: fixed {threshold['threshold_min']}-minute slack threshold ===")
    print(f"accuracy: {threshold['accuracy']:.4f}")
    print(f"AUC (slack_min as the score): {threshold['auc']:.4f}")


def reliability_table(y_test: pd.Series, y_prob: np.ndarray, n_buckets: int = N_RELIABILITY_BUCKETS) -> pd.DataFrame:
    """Predicted vs. observed made rate per decile of predicted probability."""
    deciles = pd.qcut(y_prob, n_buckets, labels=False, duplicates="drop")
    by_bucket = pd.DataFrame({"decile": deciles, "predicted": y_prob, "observed": y_test.to_numpy()})
    return by_bucket.groupby("decile").agg(
        n=("observed", "size"),
        predicted_rate=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )


def reliability_records(table: pd.DataFrame) -> list[dict]:
    """JSON-safe records (native int/float, not numpy scalars) from a reliability_table() result."""
    records = table.reset_index().to_dict("records")
    return [{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()} for row in records]


def compute_feature_importance(model: LGBMClassifier) -> list[dict]:
    """Gain-based feature importance from the underlying booster."""
    booster = model.booster_
    gain = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=booster.feature_name(),
    ).sort_values(ascending=False)
    share = gain / gain.sum()
    return [{"feature": name, "gain": float(gain[name]), "share": float(share[name])} for name in gain.index]


def print_feature_importance(importance: list[dict]) -> None:
    print("\n=== feature importance (gain-based) ===")
    for row in importance:
        print(f"  {row['feature']:<12} gain={row['gain']:>16,.1f}  share={row['share']:.1%}")


def train_and_score_auc(fit_set: pd.DataFrame, test: pd.DataFrame, drop_columns: list[str]) -> float:
    """Fit a fresh model on fit_set minus drop_columns, return test-set AUC."""
    feature_columns = [c for c in FEATURE_COLUMNS if c not in drop_columns]
    categorical_columns = [c for c in CATEGORICAL_COLUMNS if c not in drop_columns]

    model = LGBMClassifier(objective="binary", random_state=0)
    model.fit(fit_set[feature_columns], fit_set["made"], categorical_feature=categorical_columns)
    y_prob = model.predict_proba(test[feature_columns])[:, 1]
    return roc_auc_score(test["made"], y_prob)


def compute_ablation(fit_set: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Retrain with/without airport, month, and distance_in; gain-based importance
    is unreliable when features correlate, so this reports the actual AUC cost of
    dropping each."""
    runs = [
        ("all features", []),
        ("without airport", ["airport"]),
        ("without month", ["month"]),
        ("without distance_in", ["distance_in"]),
    ]
    return {label: float(train_and_score_auc(fit_set, test, drop_columns)) for label, drop_columns in runs}


def print_ablation(ablation: dict) -> None:
    print("\n=== ablation: AUC cost of dropping airport / month / distance_in ===")
    for label, auc in ablation.items():
        print(f"  {label:<20} AUC={auc:.4f}")


# --- provenance ------------------------------------------------------------------


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def git_is_dirty() -> bool:
    """True if there are uncommitted changes to anything OTHER than results/.

    results/ is excluded on purpose: writing evaluation.json/.md always leaves
    results/ different from HEAD (it's new on the first run, modified on every
    run after), so including it would make git_dirty trivially true on every
    single run and tell a reader nothing. The question that actually matters
    for traceability is whether the CODE that produced these numbers was
    committed, not whether this run's own output has been staged yet.
    """
    status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":(exclude)results/"], text=True)
    return bool(status.strip())


def pinned_versions(packages: list[str] = TRACKED_PACKAGES, requirements_path: Path = REQUIREMENTS_PATH) -> dict:
    """Exact versions from requirements.txt -- the pinned source of truth for what
    SHOULD be installed. Compare against installed_versions() for what actually
    was; they should always agree, and build_provenance() records both so a
    drift between them is visible in the file rather than silently papered over."""
    pins = {}
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if "==" not in line or line.startswith("#"):
            continue
        name, version = line.split("==", 1)
        pins[name.strip()] = version.strip()

    missing = [p for p in packages if p not in pins]
    if missing:
        raise ValueError(f"{requirements_path} has no pin for {missing}")
    return {p: pins[p] for p in packages}


def installed_versions(packages: list[str] = TRACKED_PACKAGES) -> dict:
    """Exact versions actually loaded in this process, from importlib.metadata --
    what really produced these numbers, as opposed to pinned_versions()'s "should
    be installed" from requirements.txt."""
    return {p: importlib_metadata.version(p) for p in packages}


def build_provenance(fit_set: pd.DataFrame, calib_set: pd.DataFrame, test: pd.DataFrame) -> dict:
    return {
        "git_sha": git_sha(),
        "git_dirty": git_is_dirty(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_under_evaluation": {
            "is_deployed_model": False,
            "deployed_model_path": str(MODEL_PATH),
            "note": (
                "This evaluation measures the dev model: fit on Jan-Oct (subsampled, "
                "stratified by airport), calibrated on a held-out 10% slice of Jan-Oct, "
                "and scored on the full Nov-Dec test set. It is NOT model.pkl. The "
                "deployed model (`python -m src.train --save-model`) is fit on all "
                "twelve months with no chronological holdout, so it has no held-out "
                "data left to score honestly -- these figures stand in for it."
            ),
        },
        "row_counts": {
            "train": len(fit_set),
            "calibration": len(calib_set),
            "test": len(test),
        },
        "parameters": {
            "mct_min": pairs_module.DEFAULT_MCT_MIN,
            "slack_window_min": {"min": pairs_module.MIN_SLACK_MIN, "max": pairs_module.MAX_SLACK_MIN},
            "candidate_cap_per_arrival": pairs_module.MAX_CANDIDATES_PER_ARRIVAL,
            "pairs_sampling_seed": pairs_module.DEFAULT_SEED,
            "train_subsample_seed": SUBSAMPLE_SEED,
            "subsample_target_rows": SUBSAMPLE_TARGET_ROWS,
            "train_months": list(TRAIN_MONTHS),
            "test_months": list(TEST_MONTHS),
        },
        "versions": {
            "pinned": pinned_versions(),
            "installed": installed_versions(),
        },
    }


# --- results/ files ----------------------------------------------------------------


def render_markdown(results: dict) -> str:
    provenance = results["provenance"]
    params = provenance["parameters"]
    rows = provenance["row_counts"]
    model_under_eval = provenance["model_under_evaluation"]
    dirty_note = (
        " (dirty working tree outside results/ -- uncommitted changes were present)"
        if provenance["git_dirty"]
        else ""
    )

    pinned, installed = provenance["versions"]["pinned"], provenance["versions"]["installed"]
    version_lines = []
    for package in pinned:
        marker = "" if pinned[package] == installed[package] else "  **MISMATCH**"
        version_lines.append(f"  - {package}: pinned {pinned[package]}, installed {installed[package]}{marker}")

    lines = [
        "# Model evaluation",
        "",
        "Generated by `python -m src.train`. See `evaluation.json` for the machine-readable",
        "version of the same run, and `tests/test_evaluation_results.py` for the regression",
        "floors checked against it.",
        "",
        f"**Not the deployed model.** {model_under_eval['note']}",
        "",
        "## Provenance",
        "",
        f"- Commit: `{provenance['git_sha']}`{dirty_note}",
        f"- Generated: {provenance['generated_at_utc']}",
        f"- Rows: train {rows['train']:,} / calibration {rows['calibration']:,} / test {rows['test']:,}",
        f"- MCT: {params['mct_min']} min | slack window: {params['slack_window_min']['min']}-"
        f"{params['slack_window_min']['max']} min | candidate cap: {params['candidate_cap_per_arrival']} "
        f"per arrival | pairs sampling seed: {params['pairs_sampling_seed']} | train subsample seed: "
        f"{params['train_subsample_seed']} | subsample target: {params['subsample_target_rows']:,} rows",
        f"- Train months: {params['train_months']} | test months: {params['test_months']}",
        "- Versions:",
        *version_lines,
        "",
        "## Model vs. baselines",
        "",
        "| | accuracy | AUC |",
        "|---|---|---|",
        f"| Model | {results['model']['accuracy']:.4f} | {results['model']['auc']:.4f} |",
    ]

    always = results["baselines"]["always_predict_made"]
    threshold = results["baselines"]["slack_threshold"]
    lines += [
        f"| Always predict made | {always['accuracy']:.4f} | n/a |",
        f"| {threshold['threshold_min']}-min slack threshold | {threshold['accuracy']:.4f} | {threshold['auc']:.4f} |",
        "",
    ]

    for title, key in [("Before calibration", "before_calibration"), ("After isotonic calibration", "after_calibration")]:
        lines += [f"## Reliability: {title}", "", "| decile | n | predicted | observed | gap |", "|---|---|---|---|---|"]
        for row in results["reliability"][key]:
            gap = row["predicted_rate"] - row["observed_rate"]
            lines.append(
                f"| {row['decile']} | {row['n']:,} | {row['predicted_rate']:.4f} | "
                f"{row['observed_rate']:.4f} | {gap:+.4f} |"
            )
        lines.append("")

    lines += ["## Feature importance (gain-based)", "", "| feature | gain | share |", "|---|---|---|"]
    for row in results["feature_importance"]:
        lines.append(f"| {row['feature']} | {row['gain']:,.1f} | {row['share']:.1%} |")
    lines.append("")

    lines += ["## Ablation (AUC cost of dropping a feature)", "", "| | AUC |", "|---|---|"]
    for label, auc in results["ablation"].items():
        lines.append(f"| {label} | {auc:.4f} |")
    lines.append("")

    return "\n".join(lines)


def write_results(results: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / "evaluation.json"
    json_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {json_path}")

    md_path = RESULTS_DIR / "evaluation.md"
    md_path.write_text(render_markdown(results))
    print(f"wrote {md_path}")


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

    model_metrics = compute_model_metrics(model, test, raw_prob)
    print_model_metrics(model_metrics)

    baseline_metrics = compute_baseline_metrics(test)
    print_baseline_metrics(baseline_metrics)

    before_calibration = reliability_table(test["made"], raw_prob)
    print("\n=== reliability BEFORE calibration ===")
    print(before_calibration.to_string())

    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(calib_set[FEATURE_COLUMNS], calib_set["made"])
    calibrated_prob = calibrated.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    after_calibration = reliability_table(test["made"], calibrated_prob)
    print("\n=== reliability AFTER isotonic calibration ===")
    print(after_calibration.to_string())

    importance = compute_feature_importance(model)
    print_feature_importance(importance)

    ablation = compute_ablation(fit_set, test)
    print_ablation(ablation)

    results = {
        "provenance": build_provenance(fit_set, calib_set, test),
        "model": model_metrics,
        "baselines": baseline_metrics,
        "reliability": {
            "before_calibration": reliability_records(before_calibration),
            "after_calibration": reliability_records(after_calibration),
        },
        "feature_importance": importance,
        "ablation": ablation,
    }
    write_results(results)


if __name__ == "__main__":
    main()
