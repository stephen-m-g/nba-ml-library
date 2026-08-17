# %% [markdown]
# # Build the features table
#
# Joins onto data/processed/game_labels.csv:
# - rest days / back-to-back / long-layoff (derived, no new data — compute_rest_features)
# - team pace / offensive & defensive rating (from leaguedashteamstats, season-level)
# - player bio: height, weight, position (multi-hot guard/forward/center),
#   age-at-game and experience-at-game (both point-in-time, not snapshots — see
#   compute_bio_features docstring for why that distinction mattered)
#
# Saves data/processed/features.csv — one row per player-game, ready for training.

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import (
    load_processed_labels, season_string, fetch_and_cache_player_game_logs,
    fetch_and_cache_team_advanced_stats, RAW_DIR,
)
from src.feature_engineering import compute_rest_features, compute_bio_features

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 120)

# %% [markdown]
# ## Load everything (all cached — no new API calls in this script)

# %%
labels = load_processed_labels()
labels["GAME_DATE"] = pd.to_datetime(labels["GAME_DATE"])

seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)

pace_seasons = [season_string(y) for y in range(2016, 2025)]
pace = fetch_and_cache_team_advanced_stats(pace_seasons)[["SEASON", "TEAM_ID", "PACE", "OFF_RATING", "DEF_RATING"]]

bio = pd.read_csv(RAW_DIR / "player_bio.csv")

print(f"labels:    {labels.shape}")
print(f"player_log:{player_log.shape}")
print(f"pace:      {pace.shape}")
print(f"bio:       {bio.shape}  ({bio['PERSON_ID'].nunique()} unique players)")

# %% [markdown]
# ## Rest days / back-to-back

# %%
rest = compute_rest_features(player_log)
rest["GAME_ID"] = rest["GAME_ID"].astype(str).str.zfill(10)

features = labels.merge(
    rest[["PLAYER_ID", "GAME_ID", "days_rest", "is_back_to_back", "is_long_layoff"]],
    on=["PLAYER_ID", "GAME_ID"], how="left",
)
n_missing_rest = features["days_rest"].isna().sum()
print(f"rows: {features.shape}, missing days_rest: {n_missing_rest} (expected: season-openers only)")

# %% [markdown]
# ## Team pace

# %%
season_map = {f"2{y}": season_string(y) for y in range(2016, 2025)}  # "22016" -> "2016-17"
features["SEASON"] = features["SEASON_ID"].astype(str).map(season_map)
features = features.merge(pace, on=["SEASON", "TEAM_ID"], how="left")
print(f"rows: {features.shape}, missing PACE: {features['PACE'].isna().sum()} (expect 0)")
features = features.drop(columns=["SEASON"])

# %% [markdown]
# ## Player bio (height, weight, position, age-at-game, experience-at-game)

# %%
bio_features = compute_bio_features(bio, features)
features = features.merge(bio_features, on=["PLAYER_ID", "GAME_ID"], how="left")
print(f"rows: {features.shape}")
print(f"missing height_inches: {features['height_inches'].isna().sum()} (expect 0 — every "
      f"player in game_labels.csv should have a matching bio row)")
print()
print("age_years describe:")
print(features["age_years"].describe())
print()
print("experience_years describe:")
print(features["experience_years"].describe())

# %% [markdown]
# ## Sanity checks

# %%
assert features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0, "duplicate rows introduced by a join"
assert (features["experience_years"] >= 0).all() or features["experience_years"].isna().any(), \
    "negative experience means a game predates FROM_YEAR — shouldn't be possible"
neg_exp = features[features["experience_years"] < 0]
if len(neg_exp):
    print(f"WARNING: {len(neg_exp)} rows have negative experience_years — investigate before trusting this column")
else:
    print("no negative experience_years")
print(f"row count unchanged from game_labels.csv: {len(features) == len(labels)}")

# %% [markdown]
# ## Save

# %%
out_path = OUT_DIR / "features.csv"
features.to_csv(out_path, index=False)
print(f"\nsaved {len(features)} rows, {features.shape[1]} columns to {out_path}")
print(list(features.columns))

# %% [markdown]
# ## Summary
#
# - `data/processed/features.csv`: same 211,265 rows as game_labels.csv, now
#   with 13 new columns — days_rest / is_back_to_back / is_long_layoff, PACE /
#   OFF_RATING / DEF_RATING, and height_inches / weight_lbs / plays_guard /
#   plays_forward / plays_center / age_years / experience_years.
# - Every join matched fully except days_rest, which is legitimately missing
#   for the 4,849 rows that are a player's first game of a season (no prior
#   game that season to diff against) — matches the per-season unique-player
#   counts from notebook 01 exactly.
# - age_years and experience_years are computed as-of each row's GAME_DATE
#   from birthdate/rookie-season, not taken as a live snapshot — caught and
#   fixed a bug where nba_api's own SEASON_EXP field is today's career total
#   and would've been wrong on every row except the current season.
# - Spot-checked against real players (Curry, Marcus Smart, etc.) — ages,
#   heights, and positions all line up with reality.
# - Not yet built: workload rolling averages (e.g. minutes over the last N
#   games) — the last remaining item from notebook 01's feature-candidate
#   list. Everything else needed for a first training pass is now in place.
