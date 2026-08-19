# %% [markdown]
# # Build the live-reference snapshot
#
# Precomputes and freezes the two train-fit-once inputs the live
# single-player pipeline (src/live_features.py, notebooks/21) needs but must
# not compute itself on every request or every process start:
#
# - The injury out-intervals, reconstructed from the historical Kaggle
#   injury-transaction log via `build_labeled_dataset` — the same validated
#   pipeline notebooks 03/09 already use for the training labels and
#   `career_injury_count` feature. This is the piece with real staleness:
#   whatever `coverage_end` this run reports is the injury history the live
#   service will show to users as `injury_history_as_of` on every response,
#   until this notebook is deliberately re-run.
# - The workload cohort-backfill baseline (`fit_cohort_baseline`), fit on
#   TRAIN_SEASONS only — same leakage-avoidance principle as everywhere else
#   in this pipeline, extracted in this session out of `apply_cohort_backfill`
#   so the batch and live pipelines share one implementation.
#
# This is a manual, occasional step (re-run whenever the Kaggle dataset is
# deliberately refreshed) — never wired into the running backend service.
# Needs Kaggle API credentials available locally (same requirement notebooks
# 02/03 already have via kagglehub), unlike the live service itself, which
# never touches Kaggle at all.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import (
    load_kaggle_injury_data, get_static_players, season_string,
    fetch_and_cache_player_game_logs, fetch_and_cache_team_game_logs, RAW_DIR,
)
from src.live_reference_data import build_live_reference_snapshot, save_snapshot, DEFAULT_SNAPSHOT_PATH

pd.set_option("display.width", 120)

DATASET_SLUG = "jacquesoberweis/2016-2025-nba-injury-data"
TRAIN_SEASONS = ["22016", "22017", "22018", "22019", "22020", "22021"]

# %% [markdown]
# ## Load raw sources (all cached from earlier runs — no re-fetching except injury data)

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
team_log = fetch_and_cache_team_game_logs(seasons)
injury_raw = load_kaggle_injury_data(DATASET_SLUG)
static_players = get_static_players()
bio = pd.read_csv(RAW_DIR / "player_bio.csv")

print(f"player_log:  {player_log.shape}")
print(f"team_log:    {team_log.shape}")
print(f"injury_raw:  {injury_raw.shape}")
print(f"bio:         {bio.shape}")

# %% [markdown]
# ## Build the snapshot

# %%
ref = build_live_reference_snapshot(player_log, team_log, injury_raw, static_players, bio, TRAIN_SEASONS)

print(f"intervals for {len(ref.intervals)} players")
print(f"injury data coverage ends: {ref.coverage_end.date()}")
print(f"cohort baselines: {len(ref.cohort_baseline_map)} cohorts, overall fallback = {ref.overall_baseline:.2f} min")
print(ref.cohort_baseline_map.round(2))
print(f"BMI tercile edges: {ref.bmi_tercile_edges}")
print(f"snapshot built at: {ref.snapshot_built_at}")

# %% [markdown]
# ## Sanity checks before saving

# %%
assert len(ref.intervals) > 0, "no player intervals reconstructed"
assert len(ref.cohort_baseline_map) == 9, f"expected 9 cohorts (3 positions x 3 BMI terciles), got {len(ref.cohort_baseline_map)}"
assert not pd.isna(ref.overall_baseline)
assert len(ref.bmi_tercile_edges) == 4, f"expected 4 bin edges for 3 terciles, got {len(ref.bmi_tercile_edges)}"
print("sanity checks passed")

# %% [markdown]
# ## Save

# %%
save_snapshot(ref, DEFAULT_SNAPSHOT_PATH)
print(f"\nsaved snapshot to {DEFAULT_SNAPSHOT_PATH}")

# %% [markdown]
# ## Summary
#
# - `data/processed/live_reference_snapshot.joblib`: intervals +
#   coverage_end + cohort_baseline_map + overall_baseline + train_season_ids
#   + snapshot_built_at, loaded via `src.live_reference_data.load_snapshot()`.
# - Re-run this notebook manually whenever the Kaggle injury dataset is
#   deliberately refreshed — the running backend service loads this file
#   once at startup and never rebuilds it itself.
