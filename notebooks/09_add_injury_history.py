# %% [markdown]
# # Add prior-injury-history features
#
# Motivated by the evaluation in notebook 08: the model underperforms on
# veteran players (32+) despite them having the *lowest* missing-data rate
# on min_avg_chronic (the dominant feature) — so unlike the young-player gap
# (a data-availability problem), the veteran gap looks like a missing-signal
# problem. Nothing in the feature set captures a player's own injury
# history, even though it's one of the most well-established real risk
# factors in sports science. Adds career_injury_count and
# days_missed_last_365d, both derived from the same reconstructed
# out-intervals already used for the labels.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import (
    load_kaggle_injury_data, get_static_players, season_string,
    fetch_and_cache_player_game_logs, fetch_and_cache_team_game_logs,
)
from src.feature_engineering import build_labeled_dataset, compute_injury_history_features

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

DATASET_SLUG = "jacquesoberweis/2016-2025-nba-injury-data"

# %% [markdown]
# ## Recompute the intervals (same validated pipeline as notebook 03)
#
# Not re-deriving labels here, just reusing the intervals build_labeled_dataset
# already computes internally — now exposed as a third return value.

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
team_log = fetch_and_cache_team_game_logs(seasons)
injury_raw = load_kaggle_injury_data(DATASET_SLUG)
static_players = get_static_players()

labeled, diag, intervals = build_labeled_dataset(player_log, team_log, injury_raw, static_players)
assert len(labeled) == 211265, "label pipeline output changed — investigate before proceeding"
print(f"intervals for {len(intervals)} players")

# %% [markdown]
# ## Build and join the injury-history features

# %%
history = compute_injury_history_features(intervals, labeled)
print(history.describe())

features = pd.read_csv(OUT_DIR / "features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
history["GAME_ID"] = history["GAME_ID"].astype(str).str.zfill(10)
features = features.merge(history, on=["PLAYER_ID", "GAME_ID"], how="left")
print(f"\nrows: {features.shape}")
print(f"missing career_injury_count: {features['career_injury_count'].isna().sum()} (expect 0)")

# %% [markdown]
# ## Check: does injury history actually differ by age bucket the way we'd expect?
#
# If veterans really do carry more injury history than younger players (the
# premise for building this feature at all), that should show up directly.

# %%
from src.evaluation import bucket_age
features["age_bucket"] = bucket_age(features["age_years"])
print(features.groupby("age_bucket", observed=True)[["career_injury_count", "days_missed_last_365d"]].mean())

# %% [markdown]
# ## Sanity checks and save

# %%
assert features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0
assert len(features) == 211265
features = features.drop(columns=["age_bucket"])
features.to_csv(OUT_DIR / "features.csv", index=False)
print(f"\nsaved {len(features)} rows, {features.shape[1]} columns")
