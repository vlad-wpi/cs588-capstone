# Cleaned Data Schema

The interface described in FR-1. `DataCleaner` writes this schema, `PairBuilder` reads it.
Once agreed, the two halves of the pipeline can be built in parallel against sample data.

## Schema

The cleaned-data schema below is `DataCleaner`'s output and `PairBuilder`'s input. The pairs
schema further down is `PairBuilder`'s output.

| Column | Type | Built from | Notes |
|---|---|---|---|
| `flight_date` | date | `FL_DATE` | Date the flight operated. |
| `carrier` | string | `OP_UNIQUE_CARRIER` | Operating carrier code. |
| `tail_number` | string | `TAIL_NUM` | Aircraft identifier. Not used yet; kept for possible same-aircraft analysis. |
| `flight_number` | int | `OP_CARRIER_FL_NUM` | Flight number. |
| `origin` | string | `ORIGIN` | Departure airport code. |
| `dest` | string | `DEST` | Arrival airport code. |
| `sched_dep` | datetime | `FL_DATE` + `CRS_DEP_TIME` | Scheduled departure, local time at origin. |
| `dep` | datetime | `FL_DATE` + `DEP_TIME` | Actual departure, local time at origin. |
| `sched_arr` | datetime | `FL_DATE` + `CRS_ARR_TIME` | Scheduled arrival, local time at destination. |
| `arr` | datetime | `FL_DATE` + `ARR_TIME` | Actual arrival, local time at destination. |
| `distance` | int | `DISTANCE` | Route distance in miles. |

## Rules

### Midnight crossings

Build full timestamps rather than keeping dates and times in separate fields. The source
stores times as bare integers, so zero-pad to four digits before splitting hours and minutes.

```
CRS_ARR_TIME = 28    means 00:28 the following day
ARR_TIME     = 2352  means 23:52 the same day
```

`2400` is also a valid value, meaning midnight at the end of the day rather than an hour of
24. Treat it as `0000` the following day rather than parsing it as an hour, or it will fail to
parse (or worse, silently parse as an invalid time) instead of rolling into the next date.

If an arrival time is earlier than its departure time on the same date, add one day to the
arrival. This turns every later comparison into ordinary datetime arithmetic with no special
cases.

### Cancelled and diverted flights

Drop both. Neither has an actual arrival time, so neither can be labeled.

A small number of rows have no actual departure or arrival time without being flagged
cancelled or diverted either — a BTS reporting gap, not a schema mismatch. Drop these too,
for the same reason: no actual arrival time means no label.

Log how many rows were dropped per month, broken down by reason (cancelled, diverted, or
null arrival without either flag) — `cleaner.py` writes this to `data/clean/drop_report.csv`,
and that report belongs in the final write-up.

### Time zones

Keep local time. Do not convert. Both flights in a connection pair are at the same airport,
so the offset cancels out of every comparison the project makes.

A single row therefore mixes two zones: `sched_dep` and `dep` are local to the origin,
`sched_arr` and `arr` are local to the destination. This is intentional. Do not "fix" it.

### Trailing column

The BTS exports carry an eighteenth unnamed column caused by a trailing comma on each line.
Drop it during cleaning. It is not a schema mismatch.

## Pairs schema

`PairBuilder` reads one or more cleaned monthly parquet files and, for a single connecting
airport at a time, produces one row per candidate connection.

| Column | Type | Notes |
|---|---|---|
| `airport` | string | The connecting airport (dest of the inbound flight, origin of the outbound flight). |
| `carrier_in` | string | Inbound flight's operating carrier. |
| `carrier_out` | string | Outbound flight's operating carrier. |
| `slack_min` | int | Scheduled outbound departure minus scheduled inbound arrival, in minutes. |
| `arr_hour` | int | Hour (0-23) of the inbound flight's *scheduled* arrival. Scheduled, not actual, so the feature can't leak the outcome it's predicting. |
| `day_of_week` | int | 0 (Monday) - 6 (Sunday), from the inbound flight's `flight_date`. |
| `month` | int | 1-12, from the inbound flight's `flight_date`. |
| `distance_in` | int | Inbound flight's route distance in miles. |
| `tail_in` | string | Inbound flight's aircraft tail number. |
| `tail_out` | string | Outbound flight's aircraft tail number. |
| `dest_out` | string | Outbound flight's destination airport. Not a model feature -- `lookups.py` uses it (with `sched_dep_out`) to find, for a missed pair, the next departure from `airport` to the same place. |
| `sched_dep_out` | datetime | Outbound flight's scheduled departure. Same reason as `dest_out`. |
| `made` | bool | Label: whether the connection was made (see below). |

### Candidate window and cap

A candidate pair is an inbound flight and an outbound flight at the same airport whose
scheduled departure is 30 to 240 minutes after the inbound flight's scheduled arrival
(inclusive on both ends), matched purely by timestamp. `flight_date` plays no part in the
match — an inbound flight whose scheduled arrival rolls past midnight (see Midnight crossings
above) keeps its own `flight_date`, but still matches outbound flights recorded under the next
day's `flight_date`, because what's compared is `sched_arr`/`sched_dep`, not which day either
row's `flight_date` says it is. (An earlier version of this rule matched on same `flight_date`;
that silently dropped every red-eye arrival, since a same-`flight_date` departure is always
scheduled before a rolled-forward `sched_arr` and so never lands in the window. Timestamp
matching fixes that.)

Because matching is by timestamp, an inbound flight near the end of one month's file can match
an outbound flight recorded in the next month's file (a December 31 red-eye landing after
midnight, connecting to a January 1 departure). `PairBuilder` handles this by loading the
following month's first calendar day alongside the target month whenever it's available, and
using it only as candidate outbound flights, never as inbound flights of its own.

One inbound flight can have many candidate outbound flights within that window. An uncapped
join is combinatorial — hundreds of millions of rows across all airports and months — so
candidates are capped at 20 outbound flights per inbound flight, chosen by a fixed-seed random
sample. Runs are reproducible, but the pair set is a sample of the true candidate set, not the
whole thing, for any inbound flight with more than 20 candidates.

### Made/missed label

A pair is labeled `made` if the inbound flight's actual arrival time, plus a minimum
connection time (MCT), is no later than the outbound flight's actual departure time:

```
made = (actual_arr_in + MCT) <= actual_dep_out
```

MCT defaults to 45 minutes, a rough approximation of a realistic hub connection time (time to
deplane, walk, and reboard) — not a measured value, and not the same at every airport. It's a
parameter, not a hardcoded constant: a planned sensitivity analysis will vary it across a range
of values once `lookups.py`'s recovery/comparison tables exist to inform a per-airport value.

## Lookups schema

`lookups.py` reads the pairs files (plus, for recovery, the full cleaned flights — not just the
capped candidate departures pairs.py kept) and writes two small tables. Both are keyed by
`airport` and `month`, small enough to commit, and are exactly what the deployed app reads
alongside `model.pkl` — see app.py below.

**`data/lookups/recovery.parquet`** — one row per (airport, month, arrival time block), built
from pairs labeled `missed`:

| Column | Type | Notes |
|---|---|---|
| `airport` | string | |
| `month` | int | 1-12 |
| `arr_block` | string | `morning` (05-10), `midday` (11-15), `evening` (16-20), or `late` (21-04), bucketed from the inbound flight's `arr_hour`. |
| `n_missed` | int | Missed pairs in this bucket. |
| `median_wait_min` | float | Median minutes from the missed outbound flight's `sched_dep_out` to the next *catchable* departure from `airport` to the same `dest_out` (see below), among pairs where such a departure existed later the same day. |
| `no_recovery_share` | float | Share of missed pairs in this bucket with no later same-day *catchable* departure to `dest_out` at all. |

The "next departure" search looks at every departure from that airport that day, not just the
20-capped candidates in the pairs file — the true next flight to a destination is often outside
the 30-240 min connection window pairs.py keeps.

A candidate next departure only counts if it leaves at least `min_recovery_gap_min` (default 30)
minutes after the missed flight's `sched_dep_out` — same spirit as pairs.py's MCT: the nearest
later flight by pure schedule adjacency can be one minute after the one that was missed, which
nobody can actually board. Without this floor, `median_wait_min` measures schedule density, not
a realistic recovery — confirmed on DEN/July/late-arrivals, where dropping the 1-minute-adjacent
"recoveries" moved the median from 10 minutes to 286 and the no-recovery share from 72% to 92%.

**`data/lookups/comparison.parquet`** — observed success rate by airport, month, and a 15-minute
layover bucket (`30-45`, `45-60`, ..., `225-240`, spanning the same 30-240 min range pairs.py
matches on):

| Column | Type | Notes |
|---|---|---|
| `airport` | string | |
| `month` | int | 1-12 |
| `layover_bucket` | string | e.g. `"60-75"`. |
| `n` | int | Pairs in this bucket. |
| `success_rate` | float | Mean of `made` in this bucket. |

**`data/lookups/carriers.csv`** — `code,name`, e.g. `AA,American Airlines`. Maps the two-letter
`OP_UNIQUE_CARRIER` codes in `model_categories.json` to display names for app.py's dropdowns
(the model itself is still trained and queried on the bare code). Sourced from BTS's
`L_UNIQUE_CARRIERS` support table on the TranStats download page, restricted to the carriers
that actually appear in our data (not the full historical table, which also lists carriers that
stopped reporting decades ago).

BTS reassigns carrier codes over time as regional carriers rebrand or shut down, so a lookup
mirror can be stale for a given code even when the table itself is authoritative — confirmed
one case here (`OH` was Comair until it stopped flying in 2012; PSA Airlines holds the code now,
and PSA is what `OH` means in our 2025 data) and corrected it, along with `MQ`'s 2014 rename
from American Eagle Airlines to Envoy Air. Re-verify against TranStats if new carrier codes
show up after a future retrain — `app.py` falls back to the bare code for anything missing from
this file rather than failing.

**`data/lookups/airports.csv`** — `code,name`, same shape and same fallback-to-bare-code
behavior as `carriers.csv`, but for the ~280 *origin* airports that feed the five connecting
airports (app.py's "flying in from" dropdown) — not just the five connecting airports
themselves, which stay in app.py's small hardcoded `AIRPORT_NAMES` dict. Built from BTS's
`T_MASTER_CORD` master coordinate table (a manually downloaded copy, not committed — it's a
large full reference file, not something to regenerate from our own cleaned data), filtered to
`AIRPORT_IS_LATEST = 1` and `AIRPORT_IS_CLOSED = 0` (an airport code can have several historical
rows, e.g. `AUS` has a current entry for Austin-Bergstrom and a closed one for the old Robert
Mueller Municipal — both flagged "latest" in the raw file, so the closed-airport filter is what
actually disambiguates). The label combines `DISPLAY_AIRPORT_CITY_NAME_FULL` and
`DISPLAY_AIRPORT_NAME` (e.g. "Boston" + "Logan International" -> "Boston Logan"), dropping
whichever of the two is redundant when one contains the other (so "Chicago" + "Chicago O'Hare
International" becomes "Chicago O'Hare", not "Chicago Chicago O'Hare"). One of our 283 origin
codes, `PBI`, has no row flagged `AIRPORT_IS_LATEST = 1` in the source file at all — both of its
rows (identical in every column, including `AIRPORT_IS_CLOSED = 0`) say `AIRPORT_IS_LATEST = 0`,
and this copy of `T_MASTER_CORD` has no date columns to break the tie by recency. For a code
like this, fall back to its non-closed rows as a group: if they all build the same label, use
it; only leave the code unmapped (falls back to the bare code in the dropdown, like any other
unmapped code) if the non-closed rows genuinely disagree on the name. `PBI` is the only one of
our 283 origin codes this fallback applies to — both of its identical rows agree, so it resolves
to "West Palm Beach" rather than going unmapped. No code in our data hit the disagreement case.

**`data/lookups/origin_distances.parquet`** — `airport,origin,distance_in`, one row per
(connecting airport, origin) pair that actually appears in the cleaned data (838 rows). Backs
app.py's "flying in from" input: `distance_in` isn't a question a traveler can answer directly,
but the airport they're flying from determines it. A given pair's distance is occasionally
reported as two values a mile apart across the year (rounding, not a data error) — the mode is
used rather than an arbitrary pick.

## Model artifacts

`train.py --save-model` trains on every airport and month (no chronological holdout — this is
the deployed model, not a dev-evaluation run), calibrates with isotonic regression on a 10%
stratified holdout, and writes two files to the repo root (committed, unlike `data/`):

- `model.pkl` — the calibrated classifier (`sklearn.calibration.CalibratedClassifierCV`
  wrapping the LightGBM model), via `joblib.dump`.
- `model_categories.json` — the `airport`/`carrier_in`/`carrier_out` values the model actually
  saw in training, i.e. app.py's dropdown options. Pulled from the training data's categorical
  dtype, not hand-maintained, so it can't drift from what the model knows.

## File layout

Raw files are the archive and are never edited. Cleaned output is Parquet, partitioned by
month, so a single month can be reprocessed or loaded on its own.

```
data/
  raw_data/   T_ONTIME_REPORTING_2025-01.csv ...   (never edited)
  clean/      flights_2025_01.parquet ...          (this schema)
  pairs/      pairs_ord.parquet ...
  lookups/    recovery.parquet, comparison.parquet
```

`data/` is gitignored except `data/lookups/`, which is committed — along with `model.pkl` and
`model_categories.json` at the repo root — since those are what the deployed app (`app.py`)
needs. Everything else in `data/` lives in the shared drive folder instead.

```python
df.to_parquet('data/clean/flights_2025_01.parquet')
df = pd.read_parquet('data/clean/flights_2025_01.parquet')
```

Parquet keeps column types, so parsed timestamps stay timestamps across reloads instead of
being re-parsed from strings every time. Files are roughly a fifth the size of the equivalent
CSV. Requires `pyarrow`.

## Source data as downloaded

Twelve monthly files, 18 columns each (17 selected plus the trailing artifact),
7,001,619 rows total.

| Month | Flights |
|---|---|
| 2025-01 | 539,747 |
| 2025-02 | 504,884 |
| 2025-03 | 600,872 |
| 2025-04 | 583,950 |
| 2025-05 | 605,648 |
| 2025-06 | 611,575 |
| 2025-07 | 631,428 |
| 2025-08 | 602,378 |
| 2025-09 | 562,439 |
| 2025-10 | 605,844 |
| 2025-11 | 570,550 |
| 2025-12 | 582,304 |

## Checks before handoff

- Every month writes the same columns in the same order and types.
- No nulls in `sched_dep`, `dep`, `sched_arr`, or `arr` after cancelled, diverted, and null-arrival-without-flag rows are dropped.
- No arrival timestamp earlier than its own departure timestamp.
- Row counts per month match the source counts above, less the dropped rows.
- Spot check a handful of known midnight-crossing flights by hand.
