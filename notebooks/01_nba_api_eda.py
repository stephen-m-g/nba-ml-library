# %% [markdown]
# # NBA Stats API — Structure & Feature Candidate EDA
#
# Covers EDA items 2 (NBA stats structure), 4 (data quality, NBA side), and
# 5 (feature candidates). Injury-side items (1 and 3) are in
# `02_injury_data_eda.py`, pending the Kaggle dataset link.

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: we only save figures, no interactive display
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, fetch_and_cache_player_game_logs

OUT_DIR = Path("../data/exploration") if Path.cwd().name == "notebooks" else Path("data/exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load the 10 cached seasons (2016-17 .. 2025-26)

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
df = fetch_and_cache_player_game_logs(seasons)  # reads from data/raw cache, no re-fetch
print(f"shape: {df.shape}")
df.head()

# %% [markdown]
# ## Schema
#
# One row = one player's box score for one game **they appeared in**. There
# is no row for a game a player was on the roster for but did not play —
# DNP/injury/rest games are *absent*, not flagged. That means "games missed"
# can't be read off this table; it has to come from the injury dataset (or
# from diffing a player's game log against their team's full schedule).

# %%
print(df.dtypes)

# %%
season_summary = df.groupby("SEASON_ID").agg(
    team_games=("GAME_ID", "nunique"),
    players=("PLAYER_ID", "nunique"),
    player_game_rows=("GAME_ID", "size"),
).reset_index()
season_summary["SEASON"] = seasons  # SEASON_ID is like '22016'; keep human labels alongside
print(season_summary.to_string(index=False))
season_summary.to_csv(OUT_DIR / "season_summary.csv", index=False)

# %% [markdown]
# 2019-20 (1059 team-games) and 2020-21 (1080) are short: COVID stoppage +
# bubble restart, then a 72-game shortened season. The 2019-20 gap (Mar-Jul
# 2020) is a multi-month layoff, not "rest" in the normal sense — worth a
# flag feature so the model doesn't read it as a healthy rest window.

# %%
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(season_summary["SEASON"], season_summary["team_games"])
ax.set_ylabel("Team-games in season")
ax.set_title("Games per season (COVID-shortened seasons visible)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "games_per_season.png", dpi=120)
plt.close()

# %% [markdown]
# ## Data quality: nulls

# %%
null_counts = df.isnull().sum()
null_counts = null_counts[null_counts > 0]
print("Columns with nulls:")
print(null_counts)

# %% [markdown]
# FG_PCT/FG3_PCT/FT_PCT nulls are 0-attempt games (0/0), not missing data —
# expected and safe to fill as 0 or leave null-aware in feature engineering.

# %% [markdown]
# ## Feature candidates directly available here (workload / usage)

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df["MIN"].plot.hist(bins=50, ax=axes[0])
axes[0].set_title("Minutes played per game")
axes[0].set_xlabel("MIN")

games_per_player_season = df.groupby(["SEASON_ID", "PLAYER_ID"]).size()
games_per_player_season.plot.hist(bins=50, ax=axes[1])
axes[1].set_title("Games played per player-season")
axes[1].set_xlabel("count")
plt.tight_layout()
plt.savefig(OUT_DIR / "workload_distributions.png", dpi=120)
plt.close()

# %% [markdown]
# ## Feature candidate: rest days & back-to-backs
#
# Derived from consecutive GAME_DATE per player — not a raw column, but
# straightforward to build in feature_engineering.py.

# %%
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df_sorted = df.sort_values(["PLAYER_ID", "GAME_DATE"])
date_gap = df_sorted.groupby("PLAYER_ID")["GAME_DATE"].diff().dt.days
rest_days = date_gap - 1  # playing on consecutive calendar days = 0 days rest (a back-to-back)

print("Days of rest before a game, i.e. calendar-day gap minus 1 (all seasons pooled):")
print(rest_days.describe())
print(f"\n99th percentile: {rest_days.quantile(0.99):.0f} days")

b2b_share = (rest_days == 0).mean()
print(f"Share of games that are a back-to-back (0 days rest): {b2b_share:.1%}")

long_gap_share = (rest_days > 30).mean()
print(f"Share with >30 days rest (season boundaries, injuries, COVID gap, "
      f"or a player re-appearing after being out of the league): {long_gap_share:.2%}")

fig, ax = plt.subplots(figsize=(8, 4))
rest_days.clip(upper=10).plot.hist(bins=11, ax=ax)
ax.set_title("Days of rest before a game (clipped at 10)")
ax.set_xlabel("days")
plt.tight_layout()
plt.savefig(OUT_DIR / "rest_days_distribution.png", dpi=120)
plt.close()

# %% [markdown]
# `rest_days` mean/std are dragged hard by a long right tail — max is
# thousands of days, from season boundaries (offseason gap), the COVID
# stoppage, and a handful of players with multi-year gaps between
# appearances (injury absence the log itself can't distinguish from e.g.
# a G-League stint or retirement-and-return — that disambiguation has to
# come from the injury dataset). When this becomes a real feature: compute
# rest only within-season, and cap or bucket anything past ~2 weeks rather
# than feeding raw day counts.

# %% [markdown]
# ## Feature candidates NOT in this table (need other endpoints, not pulled yet)
#
# - **Age / position / experience** — `commonplayerinfo` (per-player bio
#   endpoint; slow to bulk-pull, one call per player)
# - **Team pace / advanced team context** — `leaguedashteamstats` with
#   `measure_type_detailed_defense="Advanced"` (bulk, one call per season)
# - **Injury history / current injury status** — the Kaggle dataset (this is
#   also where the y_1game/y_3game/y_10game labels come from)
#
# Deferring the bulk pulls above until feature_engineering.py, once the
# injury data defines which players/seasons/date ranges actually matter.

# %% [markdown]
# ## Summary
#
# - 10 seasons cached in `data/raw/player_game_log_<season>.csv`
#   (255,086 player-game rows total).
# - Grain matches the project spec: one row per player per game *played*.
#   Missed games must be sourced from the injury dataset or from
#   team-schedule diffing — this table alone can't tell you who sat out.
# - Box score columns are clean (nulls are all explainable 0-attempt cases).
# - Rest days / back-to-backs are cheap derived features; age, position,
#   and team pace need additional endpoint pulls later.
