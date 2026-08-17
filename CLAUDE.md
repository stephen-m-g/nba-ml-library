# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An NBA player injury-risk prediction project. Given a player-game, predict the probability they miss a game to injury in the next 1/3/10 games (`y_1game`/`y_3game`/`y_10game`). The live deliverable is a binary "elevated risk" flag built on top of those probabilities (see Models below) — not a raw probability output.

A separate "fatigue" model line (`src/fatigue_labels.py`, `models/fatigue/`) was tried and intentionally abandoned as a primary target — see the `project-model-scope-journey` memory for why. The code is kept, not deleted, as a documented negative result.

## Commands

```bash
# Environment: Windows venv at .venv/ — always invoke it explicitly, never bare `python`
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Run a notebook (they are plain .py scripts with # %% cell markers — Jupytext
# "light" format, runnable top-to-bottom or cell-by-cell in VS Code/Jupyter)
cd notebooks && ../.venv/Scripts/python.exe 07_train_all_models.py
```

There is no test suite, linter, or build step in this repo.

**Windows console gotcha**: player/team names contain non-ASCII characters (e.g. "Jokić"). Set `PYTHONIOENCODING=utf-8` when running scripts from a shell, or `print()` calls on unfiltered name columns will crash with `UnicodeEncodeError` on the default cp1252 console.

## Architecture

### Pipeline order (`notebooks/`)

Notebooks are numbered and meant to be read/run as an ordered build pipeline, each consuming the previous ones' output — they are not independent, unordered EDA scratch files:

1. `01`–`02`: EDA on the two raw sources (NBA Stats API via `nba_api`, and a Kaggle injury-transactions dataset).
2. `03`: builds `data/processed/game_labels.csv` — the y_1game/y_3game/y_10game labels, via `feature_engineering.build_labeled_dataset`.
3. `04`, `05`, `09`, `11`, `13`: build up `data/processed/features.csv` in layers (bio/rest/workload → injury history → BMI/cohort-backfill → travel).
4. `06`–`07`, `12`, `14`: train the injury models, v1 → v2 → v3 (each adds features and/or retunes hyperparameters; v3 is current).
5. `08`, `10`: evaluation (permutation importance, calibration curves, segment analysis, held-out test set).
6. `15`–`18`: the abandoned fatigue-model line (classification, then regression).
7. `19`: builds the "elevated risk" binary threshold on top of the v3 injury models — the actual current deliverable.

### `src/` module responsibilities

- **`data_loader.py`** — all raw data acquisition: `nba_api` pulls (player/team game logs, player bio, team advanced stats) and the Kaggle injury dataset via `kagglehub`. Caches API pulls to `data/raw/` as CSVs.
- **`feature_engineering.py`** — two things: (1) injury-label construction (name-matching the injury transaction log to player IDs, reconstructing out-intervals, computing y_Ngame — entry point is `build_labeled_dataset`), and (2) the shared feature functions (rest days, rolling workload/ACWR, cohort backfill, injury history, travel, bio/BMI) used by both model lines.
- **`fatigue_labels.py`** — the abandoned model line's label construction (`build_fatigue_labels` for classification, `build_fatigue_regression_targets` for regression). Reuses `feature_engineering`'s feature functions unchanged.
- **`training.py`** — model-agnostic training utilities: `FEATURE_COLUMNS` (the single source of truth for what columns a model consumes — must match what's actually in the features CSV) and `LABEL_COLUMNS`, `time_based_split`, `prepare_xy`/`prepare_xy_regression`, `tune_hyperparameters`, `train_classifier`/`train_regressor`.
- **`evaluation.py`** — permutation importance, calibration-curve binning, segment-based error analysis. Takes a fitted model + held-out X/y; never trains anything.

### Cross-cutting patterns worth knowing before touching this code

- **Leakage avoidance is the load-bearing design principle throughout.** Train/val/test is a *time-based* split by season (`time_based_split`), never random — same-player rows close in time are highly correlated. Anything fit on data (calibration, hyperparameter search, cohort-backfill baselines, the elevated-risk threshold) is fit on the training split only and applied unchanged to val/test. Rolling features (workload, rest) use `shift(1)` before rolling so the current/future game is never in its own feature.
- **`GAME_ID` and `SEASON_ID` are zero-padded strings in the NBA API** (e.g. `"0022400061"`) but look numeric, so `pd.read_csv` silently strips the leading zeros unless `dtype={"GAME_ID": str, "SEASON_ID": str}` is passed explicitly on every read. This has caused real, silent join failures before — always pass the dtype when reading any cached CSV or processed features file.
- **Notebook path handling**: every notebook starts with `sys.path.insert(...)` using `Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()`, so they work whether launched from the repo root or from inside `notebooks/`.
- **`possible_unlogged_gap`** (a column in `features.csv`): flags rows where the injury transaction log likely missed a real absence (confirmed case: a player's season-ending injury during the playoffs, which this data source doesn't reliably capture). Convention: train on these rows (label is usually right), always exclude them via `training.exclude_gap_rows` before *scoring* on val/test.
- **Model versioning**: `v1`/`v2`/`v3` suffixes on saved `.joblib` files mean iterative refinement, not replacement — older versions are kept for direct comparison, not cleanup targets.

### Models (`models/`)

- `models/injury_risk/` — the live model line. `{y_1game,y_3game,y_10game}_model_v3.joblib`, each a `CalibratedClassifierCV`-wrapped `HistGradientBoostingClassifier`. The "elevated risk" flag thresholds these at fixed values (0.034 / 0.087 / 0.235 — top quartile of training-set predictions, fit once, not periodically recalibrated; see the `project-elevated-risk-design` memory for why).
- `models/fatigue/` — the abandoned line, both classification (`z_Ngame`) and regression (`w_Ngame`) v1 models. Kept for reference, not extended further absent a specific reason to revisit.

### Persistent project memory

This project has an active auto-memory system (`C:\Users\steph\.claude\projects\C--dev-injury-predictor\memory\`, loaded automatically each session) recording *why* decisions were made — label definitions, the model-scope journey, threshold design reasoning, and known data-quality gaps that don't show up in the code itself. Check it before re-deriving a decision that's already been made and documented.
