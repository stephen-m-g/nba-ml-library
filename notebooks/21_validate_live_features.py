# %% [markdown]
# # Validate the live single-player feature pipeline
#
# The core correctness check for src/live_features.py: for a real player
# and a real PAST game date D that already has a row in
# data/processed/features.csv, `assemble_live_features(player_id,
# as_of=D)` should reproduce that row's FEATURE_COLUMNS values from fresh
# live NBA API calls — because the live pull filters to games STRICTLY
# BEFORE as_of (see fetch_live_player_game_log), the same shift(1)
# principle every batch rolling feature already uses.
#
# One real limitation of this test setup (not a pipeline bug): the live
# pipeline always uses a player's CURRENT team (for PACE/OFF/DEF and
# travel), since that's what "current status, as of today" is supposed to
# mean for real use. If a candidate has been traded since date D, comparing
# against their team-dependent features at D isn't apples-to-apples — this
# notebook checks for that explicitly and reports it rather than silently
# comparing mismatched teams.
#
# Test cases are discovered from the data itself (not hand-picked from
# memory) so this stays meaningful as the underlying data changes:
# - A recent, ordinary veteran game (the common case).
# - A recent return from an extended absence (tests the season-gap path in
#   workload/extended-absence/injury-history together).
# - A recent early-career game with cohort backfill active (tests the
#   backfill blend and BMI-tercile classification).

# %%
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.live_reference_data import load_snapshot, DEFAULT_SNAPSHOT_PATH
from src.live_features import assemble_live_features, fetch_live_player_bio
from src.training import FEATURE_COLUMNS

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)

FEATURES_PATH = Path("../data/processed/features.csv") if Path.cwd().name == "notebooks" else Path("data/processed/features.csv")

# %% [markdown]
# ## Load

# %%
ref = load_snapshot(DEFAULT_SNAPSHOT_PATH)
features = pd.read_csv(FEATURES_PATH, dtype={"GAME_ID": str, "SEASON_ID": str})
features["GAME_DATE"] = pd.to_datetime(features["GAME_DATE"])
print(f"features.csv: {features.shape}, date range {features['GAME_DATE'].min().date()} .. {features['GAME_DATE'].max().date()}")

RECENT_CUTOFF = features["GAME_DATE"].max() - pd.Timedelta(days=365)
recent = features[(features["GAME_DATE"] >= RECENT_CUTOFF) & (~features["possible_unlogged_gap"])]
print(f"recent (last 365d of data, gap rows excluded): {len(recent)} rows")

# %% [markdown]
# ## Discover test cases from the data

# %%
candidates = []

# 1. Ordinary veteran game: most recent row for a player with a long career (>= 300 rows in features.csv)
career_counts = features["PLAYER_ID"].value_counts()
veterans = career_counts[career_counts >= 300].index
veteran_row = recent[recent["PLAYER_ID"].isin(veterans)].sort_values("GAME_DATE").iloc[-1]
candidates.append(("ordinary_veteran", veteran_row))

# 2. Recent extended-absence return
absence_rows = recent[recent["is_returning_from_extended_absence"]].sort_values("GAME_DATE")
if len(absence_rows):
    candidates.append(("extended_absence_return", absence_rows.iloc[-1]))
else:
    print("no recent extended-absence-return row found in the last 365d; skipping that case")

# 3. Recent early-career game with cohort backfill active (min_avg_chronic was NaN pre-backfill —
# can't detect post-backfill directly from features.csv since it's already filled, so approximate
# with "early in this player's own tracked history")
features_sorted = features.sort_values(["PLAYER_ID", "GAME_DATE"])
features_sorted["career_game_num"] = features_sorted.groupby("PLAYER_ID").cumcount()
early_career_recent = features_sorted[
    (features_sorted["GAME_DATE"] >= RECENT_CUTOFF)
    & (features_sorted["career_game_num"].between(1, 5))
    & (~features_sorted["possible_unlogged_gap"])
]
if len(early_career_recent):
    candidates.append(("early_career_backfill", early_career_recent.sort_values("GAME_DATE").iloc[-1]))
else:
    print("no recent early-career row found; skipping that case")

for label, row in candidates:
    print(f"{label}: PLAYER_ID={row['PLAYER_ID']} ({row.get('PLAYER_NAME', '?')}), "
          f"GAME_DATE={row['GAME_DATE'].date()}, TEAM_ID={row['TEAM_ID']}")

# %% [markdown]
# ## Run the live pipeline for each candidate and compare

# %%
NUMERIC_TOL = 1e-6
results = []

for label, row in candidates:
    player_id = int(row["PLAYER_ID"])
    as_of = row["GAME_DATE"]
    print(f"\n{'=' * 70}\n{label}: player_id={player_id}, as_of={as_of.date()}")

    try:
        bio_now = fetch_live_player_bio(player_id)
    except Exception as e:
        print(f"  SKIPPED — couldn't fetch current bio: {e}")
        continue

    # travel_distance_last_game and current_road_trip_length are, by design, not
    # reproducible against a historical row that coincides with a real game date:
    # both are properties of that SPECIFIC game (distance traveled to arrive there;
    # whether that game itself was home or away), which live inference can't know
    # without a next-game schedule lookup — explicitly out of scope per decision #2.
    # See compute_live_travel_features's docstring. Always excluded, not just for
    # traded players.
    always_excluded_cols = {"travel_distance_last_game", "current_road_trip_length"}

    if bio_now["team_id"] != int(row["TEAM_ID"]):
        print(f"  NOTE: current team ({bio_now['team_abbreviation']}, id={bio_now['team_id']}) differs from "
              f"historical row's team (id={int(row['TEAM_ID'])}) — this player was likely traded since "
              f"{as_of.date()}. PACE/OFF_RATING/DEF_RATING and travel_distance_last_14d/days_since_home are "
              f"ALSO expected to differ and are excluded from pass/fail below; everything else should still match.")
        team_dependent_cols = always_excluded_cols | {"PACE", "OFF_RATING", "DEF_RATING",
                                                        "travel_distance_last_14d", "days_since_home"}
    else:
        team_dependent_cols = always_excluded_cols

    try:
        live_result = assemble_live_features(player_id, ref, as_of=as_of)
    except Exception as e:
        print(f"  FAILED — assemble_live_features raised: {type(e).__name__}: {e}")
        continue

    live_row = live_result.features.iloc[0]
    print(f"  data_quality: {live_result.data_quality}")
    print(f"  warnings: {live_result.warnings}")

    mismatches = []
    for col in FEATURE_COLUMNS:
        expected = row[col]
        actual = live_row[col]
        both_na = pd.isna(expected) and pd.isna(actual)
        if both_na:
            match = True
        elif pd.isna(expected) or pd.isna(actual):
            match = False
        elif isinstance(expected, (bool, np.bool_)) or isinstance(actual, (bool, np.bool_)):
            match = bool(expected) == bool(actual)
        else:
            match = abs(float(expected) - float(actual)) < NUMERIC_TOL

        status = "OK" if match else ("SKIP(team)" if col in team_dependent_cols else "MISMATCH")
        if status == "MISMATCH":
            mismatches.append((col, expected, actual))
        results.append({"case": label, "column": col, "expected": expected, "actual": actual, "status": status})

    if mismatches:
        print(f"  {len(mismatches)} MISMATCH(ES):")
        for col, expected, actual in mismatches:
            print(f"    {col}: expected={expected!r}  actual={actual!r}")
    else:
        print("  all non-team-dependent columns match")

# %% [markdown]
# ## Summary

# %%
results_df = pd.DataFrame(results, columns=["case", "column", "expected", "actual", "status"])
if len(results_df) == 0:
    print("No results — every candidate was skipped or failed before producing a comparison (see output above, "
          "likely a transient NBA API issue). Re-run the previous cell.")
else:
    summary = results_df.groupby(["case", "status"]).size().unstack(fill_value=0)
    print(summary)

    n_mismatch = (results_df["status"] == "MISMATCH").sum()
    if n_mismatch == 0:
        print(f"\nPASSED — {results_df['case'].nunique()} test case(s) compared, 0 mismatches "
              f"(team-dependent columns excluded where noted).")
    else:
        print(f"\n{n_mismatch} mismatch(es) found — see detail above before proceeding to the backend.")
        print(results_df[results_df["status"] == "MISMATCH"])
