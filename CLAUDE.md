# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Connection Confidence

CS588 capstone. Estimates the probability of making an airline connection, from historical
BTS on-time data. Two-person team.

## How it works

BTS publishes scheduled and actual times for every reported domestic flight, but no
passengers, so connection outcomes do not exist in the source. We construct them: pair each
arriving flight with departures leaving shortly after, then use the actual times to label the
pair made or missed. Train a gradient boosted classifier on those labels.

All expensive work happens offline. The deployed app loads `model.pkl` plus the small lookup
tables in `data/lookups/` and does nothing else — no training, no raw data. It is split in two:
`src/predictor.py`'s `Backend` (the model and lookups, one method per figure on screen, no
Streamlit) and `app.py`'s `Frontend` (the widgets and rendering, no model or data of its own).
That split is the class diagram in the RAD.

## Current state

End-to-end prototype: every stage in the planned layout below exists and runs, for five
airports (ORD, ATL, DFW, DEN, CLT) and all twelve months of 2025.

- `data/raw_data/` — the twelve monthly BTS CSVs (`T_ONTIME_REPORTING_2025-*.csv`, ~7M rows
  total). `data/clean/` — `cleaner.py`'s output (one `flights_YYYY_MM.parquet` per month) plus
  `drop_report.csv` (per-month, per-reason drop counts). `data/pairs/` — `pairs.py`'s output,
  one `pairs_<airport>.parquet` per airport, ~27.7M pairs total across the five. None of this
  is committed (see `.gitignore` below).
- `data/lookups/` — `lookups.py`'s output: `recovery` (240 rows), `comparison` (840),
  `origin_distances` (838), and `coverage` (609, pair count per airport x carrier pair), plus the
  static `carriers.csv` / `airports.csv` name tables. Committed — see schema.md's Lookups schema.
- `model.pkl` and `model_categories.json` — `train.py --save-model`'s output: the calibrated
  classifier and its dropdown categories. Committed, at the repo root.
- `schema.md` — the interface between every stage (lives at the repo root, not
  `docs/schema.md`). Kept in sync with the code that implements it — update both together.
- `tests/` covers `cleaner.py`, `pairs.py`, the coverage counts in `lookups.py`, and the
  `Backend` against small hand-built DataFrames (the `Backend` tests use a fake model, so they
  need no Streamlit), plus smoke tests that load `model.pkl` and the committed lookups and check
  a prediction lands in [0, 1] (catches a library/pickle version mismatch in CI instead of on
  demo day).

Three things worth knowing before touching this code:

- **The app refuses to answer for thinly-covered carrier combinations (FR-11).** The carrier
  dropdowns list only carriers that operate at the selected airport, and a combination with
  fewer than 100 training pairs (`MIN_COVERAGE_PAIRS`, from `coverage.parquet`) gets a
  not-covered message instead of a probability, recovery sentence, or chart.

- **`pairs.py` matches by timestamp, not `flight_date`.** An earlier `flight_date`-grouped
  version silently produced zero candidate pairs for every red-eye arrival, since a
  same-`flight_date` departure is always scheduled before a rolled-forward `sched_arr`.
  Matching directly on `sched_arr`/`sched_dep` fixes that, which is also why `pairs.py` loads
  the following month's first calendar day as departure-only data (a Dec 31 red-eye can connect
  to a Jan 1 departure).
- **This machine has ~7.7GB of RAM, and LightGBM on the full 27.7M-row dataset is close to the
  ceiling.** `train.py`'s dev-evaluation path (`python -m src.train`, chronological split +
  ablation + reliability tables) and its final-model path (`python -m src.train --save-model`,
  all months/airports, no holdout) are two separate process invocations rather than one, and
  `features.py` uses the smallest int dtype that fits each column — both deliberate, to stay
  under that ceiling. `lookups.py` similarly projects to only the columns it needs and frees the
  cleaned-flights frame before building the comparison table.

## Layout

```
src/
  cleaner.py      raw BTS CSVs -> clean parquet (see schema.md)
  pairs.py        clean parquet -> labeled connection pairs
  features.py     pairs -> model features (native categorical dtype, no one-hot)
  train.py        fit, evaluate (chronological split, baselines, calibration, ablation),
                   and save the deployed model.pkl (--save-model)
  lookups.py      pairs + clean parquet -> recovery, comparison, origin_distances, coverage
  predictor.py    Backend: model.pkl + lookups -> predict / recovery / comparison / coverage
tests/
  test_cleaner.py, test_pairs.py, test_lookups.py, test_predictor.py, test_model_smoke.py
app.py            Streamlit UI: the Frontend class over Backend
data/             gitignored except data/lookups/ (see .gitignore)
model.pkl, model_categories.json   committed, at the repo root
schema.md         the interface between every stage above
```

## Conventions

- Every stage's output format is documented in `schema.md` — read it before touching the
  pipeline. Covers the cleaned-data schema, the pairs schema (candidate window, the MCT label,
  why matching is timestamp- not `flight_date`-based), and the lookups schema.
- Times stay in local time. Never convert time zones — `sched_dep`/`dep` are local to the
  origin and `sched_arr`/`arr` are local to the destination in the same row; this is
  intentional, see `schema.md` for why.
- `data/` is gitignored except `data/lookups/`. `model.pkl` and `model_categories.json` are
  committed at the repo root.
- Pin exact versions in `requirements.txt`. A version drift between training and deployment
  breaks the pickle load — this is also what `test_model_smoke.py` is a backstop for. CI runs on
  Python 3.11 (the deployment's version) rather than this machine's 3.8, so a pickle that only
  loads on 3.8 fails there. `pytest.ini` puts the repo root on the import path; without it bare
  `pytest` can't import `src` (only `python -m pytest` could).
- Train/test splits are chronological, never random. One arriving flight produces many pairs,
  so a random split leaks the same inbound flight across both sides.
- No hyperparameter tuning yet — `LGBMClassifier(objective="binary", random_state=0)`,
  defaults otherwise, everywhere in `train.py`.

## Commands

```bash
pytest                        # tests (also run by CI: .github/workflows/tests.yml, Python 3.11, every push/PR)
ruff check .                  # lint (not yet installed in this environment)
python -m src.cleaner         # data/raw_data/*.csv -> data/clean/*.parquet
python -m src.pairs           # data/clean/*.parquet -> data/pairs/pairs_<airport>.parquet
python -m src.lookups         # pairs + clean -> data/lookups/{recovery,comparison,origin_distances,coverage}.parquet
python -m src.train           # dev evaluation: chronological split, baselines, calibration, ablation
python -m src.train --save-model   # the deployed model: all months/airports -> model.pkl
streamlit run app.py          # local UI
```
