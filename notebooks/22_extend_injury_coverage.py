# %% [markdown]
# # Extend injury coverage past the Kaggle source, league-wide
#
# The snapshot built by notebook 20 stops at the Kaggle injury dataset's
# own `coverage_end` (2025-01-12 as of writing) — everything after that is
# invisible to `career_injury_count` / `days_missed_last_365d`, the single
# most important feature in every version of this model.
#
# `src/live_features.py` already does a *capped, per-request* version of
# this (check a player's ~8 most recent missed games against the official
# injury reports). That's fine for "what happened this week" but can miss
# an entire earlier absence — confirmed real case: a player with 66 missed
# games in the gap window, only 8 of which the live path could check.
#
# This notebook does the systematic version: walk report DATES rather than
# players (one report download covers every team playing that day, so the
# whole league resolves in one pass), reconstruct intervals for everyone,
# and ADVANCE the snapshot's coverage_end for real. That in turn shrinks
# the live path's job to just the days since this last ran.
#
# Also produces a per-player **confidence** measure: the fraction of a
# player's missed games in this window that a report actually covered.
# That's the honest answer to "how much should I trust this player's
# injury history right now," and it's surfaced through the API.
#
# **Cost**: ~250 game dates, one PDF each (a few hundred KB, ~16 pages).
# Every fetched date is cached to `data/raw/injury_reports/` — including
# dates with no report — so re-runs are fast and interruptions resume
# cleanly. Re-run periodically (weekly is plenty); it is NOT wired into
# the running backend service.

# %%
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import season_string, get_static_players, fetch_and_cache_player_game_logs, fetch_and_cache_team_game_logs
from src.injury_backfill import build_status_index, build_supplemental_intervals, merge_intervals
from src.live_reference_data import load_snapshot, save_snapshot, DEFAULT_SNAPSHOT_PATH

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)

# Set to a small number to smoke-test the pipeline on the most recent N
# game dates before committing to the full ~250-date run; None = all.
LIMIT_DATES = None

# %% [markdown]
# ## Load the existing snapshot and the game logs

# %%
ref = load_snapshot()
base_coverage_end = ref.base_coverage_end or ref.coverage_end
print(f"snapshot coverage_end:      {ref.coverage_end.date()}")
print(f"original (Kaggle) coverage: {base_coverage_end.date()}")

seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
team_log = fetch_and_cache_team_game_logs(seasons)
static_players = get_static_players()

player_log["GAME_DATE"] = pd.to_datetime(player_log["GAME_DATE"])
team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

# Rebuild the supplement from the ORIGINAL Kaggle cutoff every time, not
# from the already-advanced coverage_end: report data can be revised, and
# rebuilding the whole window is cheap (everything is disk-cached) and
# idempotent, whereas incrementally appending would compound any earlier
# mistake and make the result depend on run history.
since = base_coverage_end
until = pd.Timestamp.now().normalize()

game_dates = sorted(d for d in team_log["GAME_DATE"].unique() if since < pd.Timestamp(d) <= until)
if LIMIT_DATES:
    game_dates = game_dates[-LIMIT_DATES:]
print(f"\nwindow: {since.date()} -> {until.date()}  ({len(game_dates)} game dates)")

# %% [markdown]
# ## Fetch and index every report in the window
#
# Cached per date, so this is only slow the first time.

# %%
status_index, teams_by_date, fetch_diag = build_status_index(game_dates, static_players)
print()
for k, v in fetch_diag.items():
    if k != "unmatched_names_sample":
        print(f"  {k}: {v}")
if fetch_diag["unmatched_names_sample"]:
    print(f"  unmatched sample: {fetch_diag['unmatched_names_sample'][:10]}")

# %% [markdown]
# ## Reconstruct intervals league-wide

# %%
supplement, coverage_df, diag = build_supplemental_intervals(
    player_log, team_log, status_index, teams_by_date, since=since, until=until,
)
print()
for k, v in diag.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## What the confidence distribution looks like
#
# Low confidence for a player means many of their missed games fell on
# dates no report covered — their injury history is a weaker signal, and
# the API says so rather than presenting it as equally certain.

# %%
if len(coverage_df):
    measurable = coverage_df[coverage_df["confidence"].notna()]
    print(f"players with a measurable confidence: {len(measurable)}/{len(coverage_df)}")
    print(measurable["confidence"].describe().round(3))
    print("\nlowest-confidence players with missed games:")
    print(measurable.nsmallest(5, "confidence")[
        ["PLAYER_ID", "missed_in_window", "resolved_games", "confidence", "confirmed_injury_games"]
    ].to_string(index=False))
    print("\nmost supplemental days out:")
    print(coverage_df.nlargest(5, "supplemental_days_out")[
        ["PLAYER_ID", "missed_games", "confidence", "supplemental_intervals", "supplemental_days_out"]
    ].to_string(index=False))

# %% [markdown]
# ## Sanity checks before saving

# %%
assert diag["total_missed_games"] > 0, "no missed games found — window or logs look wrong"
assert diag["overall_confidence"] is None or 0.0 <= diag["overall_confidence"] <= 1.0
# Every supplemental interval must fall strictly after the base cutoff —
# otherwise it would be double-counting an absence the Kaggle data already has.
for pid, (starts, ends) in supplement.items():
    assert all(pd.Timestamp(s) > since for s in starts), f"interval before cutoff for player {pid}"
print("sanity checks passed")

# %% [markdown]
# ## Merge into the snapshot and advance coverage_end

# %%
ref.intervals = merge_intervals(ref.intervals, supplement)
ref.base_coverage_end = base_coverage_end
ref.supplement_start = pd.Timestamp(diag["resolvable_from"]) if diag["resolvable_from"] else None
ref.coverage_end = max(pd.Timestamp(d) for d in game_dates) if game_dates else ref.coverage_end
ref.supplement_coverage = coverage_df
ref.supplement_diagnostics = {**fetch_diag, **diag,
                              "window_start": str(since.date()), "window_end": str(until.date())}
ref.supplement_built_at = pd.Timestamp.now()

save_snapshot(ref, DEFAULT_SNAPSHOT_PATH)
print(f"coverage_end advanced: {base_coverage_end.date()} -> {ref.coverage_end.date()}")
print(f"players with supplemental intervals: {len(supplement)}")

gap = ref.coverage_gap
if gap:
    print(f"\n!! KNOWN COVERAGE GAP: {gap[0].date()} .. {gap[1].date()} ({(gap[1]-gap[0]).days + 1} days)")
    print("   Neither source covers this span — the Kaggle data ends before it and the NBA's")
    print("   report CDN no longer retains it. Surfaced through the API rather than hidden.")
print(f"\nsaved snapshot to {DEFAULT_SNAPSHOT_PATH}")

# %% [markdown]
# ## Summary
#
# - `coverage_end` now reflects real report-derived coverage, not just the
#   Kaggle cutoff; `base_coverage_end` retains the original so both facts
#   stay separately answerable.
# - `supplement_coverage` carries per-player confidence, surfaced by the
#   API as `injury_data_confidence`.
# - Re-run this periodically. The live per-request check in
#   src/live_features.py automatically shrinks to cover only the days
#   since this last ran, since it keys off `ref.coverage_end`.
