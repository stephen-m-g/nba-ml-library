# %% [markdown]
# # Add travel-fatigue features
#
# travel_distance_last_game, travel_distance_last_14d, current_road_trip_length,
# days_since_home — team-level, computed from great-circle distance between
# arena coordinates. Skipping timezones (per user: "2-3 hours at absolute
# most, negligible") and teammate interactions — both noted as v3-refinement
# candidates if the model still needs work after this.
#
# Verified before trusting this: haversine distances checked against known
# real-world city-pair mileages (all within a few miles), and the three
# within-window arena changes (Detroit 2017-18, Golden State 2019-20, LA
# Clippers 2024-25 — confirmed via web search, all landed at a clean season
# boundary, not mid-season) checked to apply at exactly the right season.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_team_game_logs
from src.feature_engineering import compute_travel_features

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Load and compute

# %%
features = pd.read_csv(OUT_DIR / "features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
seasons = [season_string(y) for y in range(2016, 2026)]
team_log = fetch_and_cache_team_game_logs(seasons)

travel = compute_travel_features(team_log)
travel["GAME_ID"] = travel["GAME_ID"].astype(str).str.zfill(10)
print(f"missing values: \n{travel.isnull().sum()}")

# %% [markdown]
# ## Join

# %%
features = features.merge(
    travel[["TEAM_ID", "GAME_ID", "travel_distance_last_game", "travel_distance_last_14d",
            "current_road_trip_length", "days_since_home"]],
    on=["TEAM_ID", "GAME_ID"], how="left",
)
print(f"rows: {features.shape}")
print(f"missing travel_distance_last_game: {features['travel_distance_last_game'].isna().sum()}")
print(f"missing days_since_home: {features['days_since_home'].isna().sum()} "
      f"(expected — genuinely unknown for the very first tracked games of the dataset, Oct-Nov 2016)")

# %% [markdown]
# ## Sanity checks and save

# %%
assert features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0
assert len(features) == 211265
assert features["travel_distance_last_game"].isna().sum() == 0
features.to_csv(OUT_DIR / "features.csv", index=False)
print(f"\nsaved {len(features)} rows, {features.shape[1]} columns")
print(list(features.columns))
