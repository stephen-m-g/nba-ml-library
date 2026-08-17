# %% [markdown]
# # Add regression targets to the fatigue features table
#
# w_1game/w_3game/w_10game: continuous, signed deviation from personal
# baseline, averaged over the window — the follow-up to v1's weak
# classification result. Standard deviation already confirms the averaging
# is doing its job: 8.56 (1 game) -> 5.50 (3 games) -> 3.68 (10 games), and
# shrinking slower than pure random noise would, which is a good sign there's
# a real, persistent trend under there rather than nothing to predict.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_player_game_logs
from src.fatigue_labels import build_fatigue_regression_targets

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Build and join

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
targets = build_fatigue_regression_targets(player_log)
targets["GAME_ID"] = targets["GAME_ID"].astype(str).str.zfill(10)
print(f"targets: {targets.shape}")

existing = pd.read_csv(OUT_DIR / "fatigue_features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
fatigue_features = existing.merge(
    targets[["PLAYER_ID", "GAME_ID", "w_1game", "w_3game", "w_10game"]],
    on=["PLAYER_ID", "GAME_ID"], how="inner",  # inner: only rows eligible for both classification and regression targets
)
print(f"rows after join: {len(fatigue_features)} (started from {len(existing)} classification-eligible)")

# %% [markdown]
# ## Sanity checks and save

# %%
assert fatigue_features[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0
assert fatigue_features[["w_1game", "w_3game", "w_10game"]].isna().sum().sum() == 0
fatigue_features.to_csv(OUT_DIR / "fatigue_features.csv", index=False)
print(f"saved {len(fatigue_features)} rows, {fatigue_features.shape[1]} columns")
