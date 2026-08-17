# %% [markdown]
# # Build the processed, labeled dataset
#
# Runs the label-construction pipeline validated in
# `02_injury_data_eda.py` (now packaged as `build_labeled_dataset` in
# src/feature_engineering.py) and saves one clean table to data/processed/:
# one row per player-game actually played, with y_1game/y_3game/y_10game
# attached (strict definition). No engineered features yet (rest days, age,
# position, pace) — that's a separate pass, on purpose, so this step stays
# easy to review on its own.
#
# Coverage decision (confirmed): let the dataset cut off naturally wherever
# the injury data's coverage ends, rather than dropping all of 2024-25 for
# season-length consistency. See the conversation / project memory for why.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import (
    load_kaggle_injury_data, get_static_players, season_string,
    fetch_and_cache_player_game_logs, fetch_and_cache_team_game_logs,
)
from src.feature_engineering import build_labeled_dataset

OUT_DIR = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 120)

DATASET_SLUG = "jacquesoberweis/2016-2025-nba-injury-data"

# %% [markdown]
# ## Load raw sources (all cached from earlier runs — no re-fetching)

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
team_log = fetch_and_cache_team_game_logs(seasons)
injury_raw = load_kaggle_injury_data(DATASET_SLUG)
static_players = get_static_players()

print(f"player_log:  {player_log.shape}")
print(f"team_log:    {team_log.shape}")
print(f"injury_raw:  {injury_raw.shape}")

# %% [markdown]
# ## Run the pipeline

# %%
labeled, diag, intervals = build_labeled_dataset(player_log, team_log, injury_raw, static_players)

print(f"injury-log rows matched to a player: {diag['injury_rows_matched']}/{diag['injury_rows_total']} "
      f"({diag['match_rate']:.2%})")
print(f"injury data coverage ends: {diag['coverage_end'].date()}")
print()
print(f"eligible player-games: {diag['n_eligible']}/{diag['n_total_player_games']} ({diag['eligible_share']:.1%})")
print(f"most recent game_date retained: {diag['max_game_date_retained'].date()}")
print()
print(f"class balance — 1-game: {diag['pct_positive_1game']:.2%}  "
      f"3-game: {diag['pct_positive_3game']:.2%}  "
      f"10-game: {diag['pct_positive_10game']:.2%}")

# %% [markdown]
# ## Sanity checks before saving
#
# These numbers should match notebook 02's validated strict-definition
# results exactly (2.41% / 7.00% / 20.01%) — if they don't, something
# changed in the pipeline and needs investigating before this becomes the
# dataset everything else builds on.

# %%
assert labeled[["PLAYER_ID", "GAME_ID"]].duplicated().sum() == 0, "duplicate player-game rows"
assert labeled["y_10game"].sum() >= labeled["y_3game"].sum() >= labeled["y_1game"].sum(), \
    "a wider window should never have fewer positives than a narrower one"
print("no duplicate (player, game) rows; label windows nest as expected (1 <= 3 <= 10)")

# confirm the coverage cutoff behaved as decided: no 2025-26 games, some but not
# all of 2024-25 retained
season_counts = labeled["SEASON_ID"].value_counts().sort_index()
print("\nrows retained per season:")
print(season_counts)

# %% [markdown]
# ## Save

# %%
out_path = OUT_DIR / "game_labels.csv"
labeled.to_csv(out_path, index=False)
print(f"\nsaved {len(labeled)} rows to {out_path}")

summary = pd.DataFrame([{
    "injury_rows_matched": diag["injury_rows_matched"],
    "injury_rows_total": diag["injury_rows_total"],
    "match_rate": diag["match_rate"],
    "coverage_end": diag["coverage_end"],
    "n_eligible": diag["n_eligible"],
    "n_total_player_games": diag["n_total_player_games"],
    "eligible_share": diag["eligible_share"],
    "max_game_date_retained": diag["max_game_date_retained"],
    "pct_positive_1game": diag["pct_positive_1game"],
    "pct_positive_3game": diag["pct_positive_3game"],
    "pct_positive_10game": diag["pct_positive_10game"],
}])
summary.to_csv(OUT_DIR / "game_labels_build_summary.csv", index=False)

# %% [markdown]
# ## Summary
#
# - `data/processed/game_labels.csv`: one row per player-game actually
#   played, 2016-17 through the point where 2024-25 runs out of injury
#   coverage. Columns: PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION,
#   GAME_ID, GAME_DATE, SEASON_ID, y_1game, y_3game, y_10game.
# - No 2025-26 rows (zero injury coverage for that season); 2024-25 is
#   present but truncated where the source data runs out — by design,
#   confirmed with the user rather than assumed.
# - This is the labels-only table. Next step is feature_engineering.py
#   proper: joining in rest days/back-to-backs (already prototyped in
#   notebook 01), then age/position/team pace (need new API pulls).
