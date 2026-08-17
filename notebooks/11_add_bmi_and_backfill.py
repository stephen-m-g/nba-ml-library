# %% [markdown]
# # Add BMI and cohort-backfilled workload
#
# Two changes to features.csv:
# - `bmi` (703 * lbs / inches^2) — height and weight in isolation don't
#   capture "mass relative to frame"; a lean 7-footer and a sturdy 6'6"
#   player can look similar on the raw numbers but have very different
#   injury profiles. Sanity-checked against a concrete example (7'0"/200lbs
#   -> BMI 19.9, 6'6"/250lbs -> BMI 28.9): clearly separates exactly the
#   cases it's meant to.
# - min_avg_acute / min_avg_chronic / acute_chronic_ratio backfilled for
#   early-career players using a cohort (position x BMI tercile) baseline,
#   blended in proportion to how much personal history exists yet. Motivated
#   by a concrete finding from evaluation: players 23-and-under are missing
#   min_avg_chronic 17% of the time (vs 3.7% for veterans) — a cold-start
#   problem for the single most important feature in every model. Verified
#   on a real rookie (Jordan Ford, debut 2023-10-25): his first game shows
#   the pure cohort baseline (16.5), then smoothly shifts toward his own
#   real data as it accumulates.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_player_game_logs, RAW_DIR
from src.feature_engineering import compute_bio_features, compute_workload_features, apply_cohort_backfill

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

TRAIN_SEASONS = ["22016", "22017", "22018", "22019", "22020", "22021"]

# %% [markdown]
# ## Load

# %%
features = pd.read_csv(OUT_DIR / "features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
bio = pd.read_csv(RAW_DIR / "player_bio.csv")
print(f"features (before): {features.shape}")

# %% [markdown]
# ## BMI — recompute bio features and merge in just the new column

# %%
bio_features = compute_bio_features(bio, features)
bio_features["GAME_ID"] = bio_features["GAME_ID"].astype(str).str.zfill(10)
features = features.merge(bio_features[["PLAYER_ID", "GAME_ID", "bmi"]], on=["PLAYER_ID", "GAME_ID"], how="left")
print(f"missing bmi: {features['bmi'].isna().sum()}")
print(features["bmi"].describe())

# %% [markdown]
# ## Cohort-backfilled workload — replace the existing acute/chronic/ratio columns

# %%
raw_workload = compute_workload_features(player_log)
backfilled = apply_cohort_backfill(raw_workload, player_log, bio, TRAIN_SEASONS)
backfilled["GAME_ID"] = backfilled["GAME_ID"].astype(str).str.zfill(10)

n_before = features["min_avg_chronic"].isna().sum()
features = features.drop(columns=["min_avg_acute", "min_avg_chronic", "acute_chronic_ratio"]).merge(
    backfilled[["PLAYER_ID", "GAME_ID", "min_avg_acute", "min_avg_chronic", "acute_chronic_ratio"]],
    on=["PLAYER_ID", "GAME_ID"], how="left",
)
n_after = features["min_avg_chronic"].isna().sum()
print(f"missing min_avg_chronic: {n_before} -> {n_after}")
print(f"missing min_avg_acute: {features['min_avg_acute'].isna().sum()}")
print(f"missing acute_chronic_ratio: {features['acute_chronic_ratio'].isna().sum()}")

# %% [markdown]
# ## Sanity checks and save

# %%
assert features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0
assert len(features) == 211265
assert features["min_avg_chronic"].isna().sum() == 0
features.to_csv(OUT_DIR / "features.csv", index=False)
print(f"\nsaved {len(features)} rows, {features.shape[1]} columns")
print(list(features.columns))
