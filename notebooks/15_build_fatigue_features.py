# %% [markdown]
# # Fatigue model: build the features + labels table
#
# New model line, per the scope pivot: predicting performance decline
# (z_1game/z_3game/z_10game) instead of injury (y_1game/y_3game/y_10game).
# Reuses the *entire* existing feature set unchanged — workload, rest,
# travel, bio, injury history are all still legitimate fatigue predictors,
# only the target changes. See src/fatigue_labels.py for the label logic.
#
# The injury labels stay in this table too (not used as features or as the
# training target — kept specifically for a downstream check later: does
# this model's fatigue score actually correlate with real injury rates,
# even though it's never trained to predict injury directly).

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_player_game_logs
from src.fatigue_labels import build_fatigue_labels

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Build fatigue labels

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
fatigue_labels = build_fatigue_labels(player_log)
fatigue_labels["GAME_ID"] = fatigue_labels["GAME_ID"].astype(str).str.zfill(10)

print(f"eligible rows: {len(fatigue_labels)} / {len(player_log)} ({len(fatigue_labels)/len(player_log):.1%})")
print(f"class balance: z_1game={fatigue_labels['z_1game'].mean():.3f}, "
      f"z_3game={fatigue_labels['z_3game'].mean():.3f}, z_10game={fatigue_labels['z_10game'].mean():.3f}")

# %% [markdown]
# ## Join onto the existing feature set + keep injury labels for later validation

# %%
existing = pd.read_csv(OUT_DIR / "features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
injury_feature_cols = [c for c in existing.columns if c not in
                        ("y_1game", "y_3game", "y_10game", "possible_unlogged_gap")]

fatigue_label_cols = ["PLAYER_ID", "GAME_ID", "z_1game", "z_3game", "z_10game"]  # GAME_DATE/SEASON_ID come from `existing` instead
fatigue_features = fatigue_labels[fatigue_label_cols].merge(
    existing[injury_feature_cols], on=["PLAYER_ID", "GAME_ID"], how="inner",  # inner: only rows eligible for BOTH
)
# bring the injury labels back in too, purely for later validation, not as features/training target
fatigue_features = fatigue_features.merge(
    existing[["PLAYER_ID", "GAME_ID", "y_1game", "y_3game", "y_10game"]],
    on=["PLAYER_ID", "GAME_ID"], how="left",
)
print(f"\nrows after joining onto the shared feature set: {len(fatigue_features)} "
      f"(started from {len(fatigue_labels)} fatigue-eligible, {len(existing)} injury-eligible)")

# %% [markdown]
# ## Sanity checks and save

# %%
assert fatigue_features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0
assert fatigue_features["z_1game"].notna().all()
missing_feature_cols = [c for c in injury_feature_cols if fatigue_features[c].isna().any()
                         and c not in ("PLAYER_NAME", "TEAM_ABBREVIATION")]
print(f"feature columns with any missing values: {missing_feature_cols if missing_feature_cols else 'none'}")

out_path = OUT_DIR / "fatigue_features.csv"
fatigue_features.to_csv(out_path, index=False)
print(f"\nsaved {len(fatigue_features)} rows, {fatigue_features.shape[1]} columns to {out_path}")
print(list(fatigue_features.columns))
