# %% [markdown]
# # Add workload and injury-carryover features
#
# Extends data/processed/features.csv with:
# - rolling workload (acute:chronic minutes ratio, schedule density) — the
#   "minutes restriction" signal
# - a cause-agnostic "returning from an extended absence" feature
# - a possible_unlogged_gap transparency flag on the labels themselves
#
# Background on the last two: investigating the question of whether a
# season-ending injury (e.g. an ACL tear late in one season) is correctly
# reflected in the labels for the games leading up to it turned up a real
# gap. Concrete confirmed case: Klay Thompson tore his ACL in Game 6 of the
# 2019 Finals and missed all of 2019-20 — but the Kaggle injury transaction
# log has no entry for it at all (it happened in the playoffs, which aren't
# in our game data, and the transaction appears to have never been filed).
# Quantified: 92 of 1,346 players (6.8%) have a season-plus game-log gap that
# the current y_10game doesn't flag beforehand. But the cause is genuinely
# mixed — established rotation players like Klay and Danilo Gallinari look
# like real missed injuries, but plenty of others (e.g. Darren Collison, who
# actually retired voluntarily around then) are normal roster churn, not
# injury. Decision: flag for visibility, don't relabel — see project memory
# for the full reasoning and the plan to eventually source better injury
# data (including playoff coverage) to close this gap for real.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_player_game_logs
from src.feature_engineering import (
    compute_workload_features, compute_extended_absence_features, flag_possible_unlogged_gap,
)

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Load

# %%
features = pd.read_csv(OUT_DIR / "features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
print(f"features: {features.shape}")

# %% [markdown]
# ## Rolling workload

# %%
workload = compute_workload_features(player_log)
workload["GAME_ID"] = workload["GAME_ID"].astype(str).str.zfill(10)
features = features.merge(
    workload[["PLAYER_ID", "GAME_ID", "min_avg_acute", "min_avg_chronic", "acute_chronic_ratio", "games_last_14d"]],
    on=["PLAYER_ID", "GAME_ID"], how="left",
)
print(f"rows: {features.shape}")
print(f"missing min_avg_acute (expect ~1300, players' first {3} games with no prior history): "
      f"{features['min_avg_acute'].isna().sum()}")
print(f"acute_chronic_ratio describe:\n{features['acute_chronic_ratio'].describe()}")

# %% [markdown]
# ## Extended-absence return flag

# %%
absence = compute_extended_absence_features(player_log)
absence["GAME_ID"] = absence["GAME_ID"].astype(str).str.zfill(10)
features = features.merge(
    absence[["PLAYER_ID", "GAME_ID", "games_since_extended_return", "is_returning_from_extended_absence"]],
    on=["PLAYER_ID", "GAME_ID"], how="left",
)
print(f"rows: {features.shape}")
print(f"rows flagged as an extended-absence return game: {features['is_returning_from_extended_absence'].sum()}")

# %% [markdown]
# ## Transparency flag on the labels

# %%
gap_flag = flag_possible_unlogged_gap(player_log, features)
gap_flag["GAME_ID"] = gap_flag["GAME_ID"].astype(str).str.zfill(10)
features = features.merge(gap_flag, on=["PLAYER_ID", "GAME_ID"], how="left")
print(f"rows: {features.shape}")
n_flagged = features["possible_unlogged_gap"].sum()
print(f"rows flagged possible_unlogged_gap: {n_flagged} ({n_flagged/len(features):.2%})")

# %% [markdown]
# ## Sanity checks

# %%
import numpy as np

assert features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0, "duplicate rows introduced by a join"
assert len(features) == 211265, "row count changed unexpectedly"
assert not np.isinf(features["acute_chronic_ratio"]).any(), "infinite ratio from a near-zero denominator"
print("no duplicate rows, no infinite ratios, row count unchanged")

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
# - features.csv now has 29 columns: the original 23 plus min_avg_acute,
#   min_avg_chronic, acute_chronic_ratio, games_last_14d,
#   games_since_extended_return, is_returning_from_extended_absence, and
#   possible_unlogged_gap.
# - Rolling workload uses only strictly-prior games (shift before rolling) —
#   nothing here uses the outcome of the game being labeled, so there's no
#   leakage from "this game's own minutes" into its own risk prediction.
# - possible_unlogged_gap is informational only — y_1game/y_3game/y_10game
#   are untouched. Filtering these ~7% of rows out (or modeling them
#   separately) is a call for the training phase, not made here.
