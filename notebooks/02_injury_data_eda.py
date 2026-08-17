# %% [markdown]
# # Kaggle Injury Data EDA + First-Pass Label Construction
#
# Dataset: jacquesoberweis/2016-2025-nba-injury-data (via kagglehub).
# Covers EDA items 1 (injury dataset structure), 3 (class balance for the
# 1/3/10-game windows), and the injury-side half of 4 (data quality).
#
# This also builds a first-pass version of the actual y_1game/y_3game/y_10game
# labels, since "class balance" can't be answered without constructing them.
# The reusable pieces (name matching, interval reconstruction) live in
# src/feature_engineering.py; this notebook is the narrative + the numbers.

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.data_loader import (
    load_kaggle_injury_data, get_static_players, season_string,
    fetch_and_cache_player_game_logs, fetch_and_cache_team_game_logs,
)
from src.feature_engineering import (
    match_player_ids, categorize_injury_reason, build_out_intervals,
    clip_intervals_to_played_games, is_out_on_dates, NON_INJURY_CATEGORIES,
)

OUT_DIR = Path("../data/exploration") if Path.cwd().name == "notebooks" else Path("data/exploration")
OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 120)

DATASET_SLUG = "jacquesoberweis/2016-2025-nba-injury-data"

# %% [markdown]
# ## 1. Structure

# %%
inj = load_kaggle_injury_data(DATASET_SLUG)
inj["Date"] = pd.to_datetime(inj["Date"])
print(f"shape: {inj.shape}")
print(inj.dtypes)
print()
print(inj.head(5).to_string())

# %%
print("nulls:")
print(inj.isnull().sum())
print()
print("Acquired/Relinquished are mutually exclusive:",
      ((inj["Acquired"].notna()) & (inj["Relinquished"].notna())).sum(), "rows with both")
both_null = inj["Acquired"].isna() & inj["Relinquished"].isna()
print(f"{both_null.sum()} rows with neither (dropped)")
inj = inj[~both_null].copy()

# %% [markdown]
# ## Coverage gap — the headline data-quality finding
#
# The NBA-API pull (notebook 01) covers 2016-17 through 2025-26. This injury
# dataset does not, despite the "2016-2025" name.

# %%
print("injury data date range:", inj["Date"].min().date(), "to", inj["Date"].max().date())
print(inj["Date"].dt.to_period("M").value_counts().sort_index().tail(6))

# %% [markdown]
# Injury coverage stops **2025-01-12** — mid-way through the 2024-25 season.
# The back half of 2024-25 and all of 2025-26 (which the game log has fully)
# have no injury labels available from this source. Any label built from this
# join has to be restricted to windows fully inside that coverage; the rest
# just can't be labeled from this data, not "no injury happened."

# %% [markdown]
# ## 2. Reason categories
#
# Team is a nickname only ("Blazers", not "Portland Trail Blazers" or "POR") —
# matches nba_api's static `teams` "nickname" field if that join is ever needed.
# The real content is in Notes: almost all Relinquished rows are "placed on IL
# with <reason>", but the reason isn't always an injury.

# %%
placed = inj[inj["Relinquished"].notna()].copy()
placed["category"] = placed["Notes"].apply(categorize_injury_reason)
cat_counts = placed["category"].value_counts()
print(cat_counts)
print((cat_counts / len(placed) * 100).round(1))

fig, ax = plt.subplots(figsize=(8, 4))
cat_counts.plot.bar(ax=ax)
ax.set_title("IL placement reasons (Relinquished rows)")
ax.set_ylabel("count")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "injury_reason_categories.png", dpi=120)
plt.close()

covid_by_year = placed[placed["category"] == "covid_protocol"]["Date"].dt.year.value_counts().sort_index()
print("\ncovid_protocol rows by year (confirms concentration in 2020-21):")
print(covid_by_year)

# %% [markdown]
# `injury` (66.7%) and `unspecified` ("placed on IL" with no reason given,
# 18.1%) are the two buckets I'm treating as genuine injury. `covid_protocol`
# (6.2%, almost entirely 2021 — health & safety protocol quarantine, not an
# injury), `illness` (7.6%), `rest` (1.2%, load management), `personal` (0.2%)
# are excluded from the "strict" label definition below. This is a real
# framing choice, not a mechanical cleaning step — flagging it rather than
# deciding it silently.

# %% [markdown]
# ## 3. Matching player names to nba_api PLAYER_ID
#
# Raw exact-match only gets ~92%. Three fixable patterns account for most of
# the gap: diacritics ("Porziņģis" / accented letters), punctuation ("C.J."
# vs "CJ"), and parenthetical legal-name clarifications ("Reggie Jackson
# (Shon)"). Also: some cells hold multiple aliases for the *same* person
# separated by "/" (e.g. "Manu Ginobili / Emanuel Ginobili") — not different
# people, just inconsistent naming over time in the source data.

# %%
players = get_static_players()
inj["event_type"] = np.where(inj["Relinquished"].notna(), "out", "available")
inj["player_name_raw"] = inj["Relinquished"].fillna(inj["Acquired"])
inj["player_id"] = match_player_ids(inj["player_name_raw"], players)

match_rate = inj["player_id"].notna().mean()
print(f"match rate: {match_rate:.2%} ({inj['player_id'].notna().sum()}/{len(inj)})")

unmatched = inj[inj["player_id"].isna()]["player_name_raw"].drop_duplicates()
print(f"\n{len(unmatched)} unique unmatched names (saved to unmatched_injury_names.csv):")
print(unmatched.to_string())
unmatched.to_csv(OUT_DIR / "unmatched_injury_names.csv", index=False, header=["name"])

# %% [markdown]
# 99.76% match. Residual misses are mostly two-way/G-League players who were
# never in an actual NBA box score (so absent from nba_api's player list) —
# low-impact. One is a straight data-entry defect: a Relinquished cell whose
# value is literally the string `"Yes"`, not a name.

# %%
inj_matched = inj[inj["player_id"].notna()].copy()
inj_matched["player_id"] = inj_matched["player_id"].astype(int)
inj_matched["category"] = np.where(
    inj_matched["event_type"] == "out",
    inj_matched["Notes"].apply(categorize_injury_reason),
    "n/a",
)

# %% [markdown]
# ## 4. Reconstructing out-intervals
#
# State machine over each player's placed/activated events (see
# `build_out_intervals`): repeated same-type events are no-ops, so e.g. a
# duplicate "placed on IL" transaction filed two days into an existing
# absence doesn't reset anything. Spot-checked against Kristaps Porziņģis
# (most-placed player in the data, 96 events) — his ACL tear (Feb 2018, "out
# for season"), meniscus tear (Aug 2020), and a same-day-duplicate illness
# placement (Mar 2023) all reconstruct correctly as intervals.
#
# But pairing "placed" with whatever "activated" event comes next in the log
# is only as good as the log being complete, and it isn't always. Simply
# pairing sequential events closes Porziņģis's ACL-tear interval on
# 2020-01-21 — but that's the activation for an *unrelated* sore-knee stint
# more than a year later; his real ACL-recovery activation was never logged
# at all (he was traded from the Knicks to the Mavericks while out). The
# naive pairing overstates that absence by 3 months. See the clipping fix below.

# %%
INJURY_COVERAGE_END = inj["Date"].max()

intervals_strict = build_out_intervals(
    inj_matched[~inj_matched["category"].isin(NON_INJURY_CATEGORIES)], INJURY_COVERAGE_END
)
intervals_broad = build_out_intervals(inj_matched, INJURY_COVERAGE_END)
print(f"players with >=1 reconstructed out-interval: "
      f"{len(intervals_strict)} (strict) / {len(intervals_broad)} (broad)")

span_days_strict = np.concatenate([
    (ends - starts).astype("timedelta64[D]").astype(int)
    for starts, ends in intervals_strict.values() if len(starts)
])
print("\nstrict-definition interval length in days (before played-game clipping):")
print(pd.Series(span_days_strict).describe())

# %% [markdown]
# Max is ~3000 days — a right-censored interval (placed on IL, never a
# matching "activated" event before the data ends). 168 players have one.
# Checking those against the game log directly: 88 of them (52%!) have real
# logged appearances *after* their interval supposedly opened — e.g. Devin
# Harris shows as "out" starting 2018-03-17 but is back playing games just 2
# days later, and Greg Monroe's "open since 2018-01-31" interval sits under 4
# more years of appearances through 2022. This isn't a rare edge case: near
# the injury data's coverage edge (2025-01-12) the transaction log simply
# stops updating rather than the player being out for years, and even
# earlier in the data some "activated" transactions are just missing.
#
# Fix: an out-interval can't be valid across a date where the player-game log
# shows them actually playing — that's direct evidence, so it overrides the
# incomplete transaction log (`clip_intervals_to_played_games`).

# %%
seasons = [season_string(y) for y in range(2016, 2026)]
player_log = fetch_and_cache_player_game_logs(seasons)
team_log = fetch_and_cache_team_game_logs(seasons)
player_log["GAME_DATE"] = pd.to_datetime(player_log["GAME_DATE"])
team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

intervals_strict_raw = intervals_strict  # keep for the before/after comparison below
intervals_strict = clip_intervals_to_played_games(intervals_strict, player_log)
intervals_broad = clip_intervals_to_played_games(intervals_broad, player_log)

n_changed = sum(
    1 for pid, (s, e) in intervals_strict_raw.items()
    if not np.array_equal(e, intervals_strict[pid][1])
)
print(f"players with >=1 interval shortened by clipping: {n_changed} / {len(intervals_strict_raw)}")

span_days_strict_clipped = np.concatenate([
    (ends - starts).astype("timedelta64[D]").astype(int)
    for starts, ends in intervals_strict.values() if len(starts)
])
print("strict-definition interval length in days (after clipping):")
print(pd.Series(span_days_strict_clipped).describe())

porzingis_id = players[players["full_name"].str.contains("Porzi")]["id"].iloc[0]
raw_s, raw_e = intervals_strict_raw[porzingis_id]
clip_s, clip_e = intervals_strict[porzingis_id]
acl_i = next(i for i in range(len(raw_s)) if pd.Timestamp(raw_s[i]).year == 2018 and pd.Timestamp(raw_s[i]).month == 2)
print(f"\nPorziņģis ACL interval, naive pairing:  "
      f"{pd.Timestamp(raw_s[acl_i]).date()} to {pd.Timestamp(raw_e[acl_i]).date()}")
print(f"Porziņģis ACL interval, after clipping:  "
      f"{pd.Timestamp(clip_s[acl_i]).date()} to {pd.Timestamp(clip_e[acl_i]).date()} "
      f"(2019-10-23 is his actual season-debut game for Dallas)")

# %% [markdown]
# ## 5. Class balance for y_1game / y_3game / y_10game
#
# For each player-game actually played (from the game log), look up the
# team's next 1/3/10 *scheduled* games (from the full team schedule, not just
# games the player appeared in) and check whether the player has a
# reconstructed out-interval covering any of those dates.
#
# Two things bound which rows can get a label at all:
# - Needs 10 actual future scheduled games in our pulled window (only bites at
#   the very end of the 10-season pull).
# - The 10th of those game dates must be `<= INJURY_COVERAGE_END`, or the
#   injury data simply can't confirm/deny an injury that far out.
#
# Using the *same* eligible-row population (valid 10-game window) for all
# three labels, so the 1/3/10 percentages below are directly comparable to
# each other rather than each using a different denominator.
#
# Not handled here (first-pass, see feature_engineering.py TODOs): a next-N
# window can cross an offseason gap into the following season for
# late-season games — spot-checked that this doesn't false-positive (the
# offseason itself isn't inside any out-interval), but whether it *should*
# count toward "next 10 games" at all is a modeling decision, not a data bug.

# %%
team_log_sorted = team_log.sort_values(["TEAM_ID", "GAME_DATE"])
team_dates = {
    tid: grp["GAME_DATE"].values.astype("datetime64[ns]")
    for tid, grp in team_log_sorted.groupby("TEAM_ID")
}

# %%
def build_next10_index(player_log: pd.DataFrame, team_dates: dict) -> tuple[np.ndarray, np.ndarray]:
    """For every player_log row, the next 10 scheduled dates for that row's
    team after that row's game. Returns (dates[n,10] datetime64, valid[n] bool).
    """
    team_id_arr = player_log["TEAM_ID"].values
    game_date_arr = player_log["GAME_DATE"].values.astype("datetime64[ns]")
    result = np.full((len(player_log), 10), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = np.zeros(len(player_log), dtype=bool)

    for team_id, dates in team_dates.items():
        mask = team_id_arr == team_id
        if not mask.any():
            continue
        idx = np.searchsorted(dates, game_date_arr[mask], side="right")
        ok = (idx + 10) <= len(dates)
        positions = np.where(mask)[0][ok]
        gather_idx = idx[ok][:, None] + np.arange(10)[None, :]
        result[positions] = dates[gather_idx]
        valid[positions] = True
    return result, valid

next10_dates, has_next10 = build_next10_index(player_log, team_dates)
within_coverage = has_next10 & (next10_dates[:, -1] <= np.datetime64(INJURY_COVERAGE_END))
print(f"eligible rows: {within_coverage.sum()} / {len(player_log)} "
      f"({within_coverage.mean():.1%}) — rest excluded by the coverage cutoff above")

# %%
eligible = player_log.loc[within_coverage, ["PLAYER_ID", "TEAM_ID", "GAME_DATE", "SEASON_ID"]].copy()
eligible_future10 = next10_dates[within_coverage]


def compute_labels(intervals: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(eligible)
    flags = np.zeros((n, 10), dtype=bool)
    pid_arr = eligible["PLAYER_ID"].values
    for pid in np.unique(pid_arr):
        rows = np.where(pid_arr == pid)[0]
        fut = eligible_future10[rows]
        for k in range(10):
            flags[rows, k] = is_out_on_dates(intervals, pid, fut[:, k])
    y1 = flags[:, 0]
    y3 = flags[:, :3].any(axis=1)
    y10 = flags[:, :10].any(axis=1)
    return y1, y3, y10


y1_strict, y3_strict, y10_strict = compute_labels(intervals_strict)
y1_broad, y3_broad, y10_broad = compute_labels(intervals_broad)

summary = pd.DataFrame({
    "window": ["1-game", "3-game", "10-game"],
    "pct_positive_strict": [y1_strict.mean(), y3_strict.mean(), y10_strict.mean()],
    "pct_positive_broad": [y1_broad.mean(), y3_broad.mean(), y10_broad.mean()],
    "n_eligible": [within_coverage.sum()] * 3,
})
print(summary.to_string(index=False))
summary.to_csv(OUT_DIR / "class_balance_summary.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(3)
ax.bar(x - 0.2, summary["pct_positive_strict"] * 100, width=0.4, label="strict (injury only)")
ax.bar(x + 0.2, summary["pct_positive_broad"] * 100, width=0.4, label="broad (all IL placements)")
ax.set_xticks(x)
ax.set_xticklabels(summary["window"])
ax.set_ylabel("% of player-games")
ax.set_title("Class balance: y_1game / y_3game / y_10game")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "class_balance.png", dpi=120)
plt.close()

# %% [markdown]
# ## Class balance by season (data-consistency check)
#
# If injury-reporting practices changed materially over the 10 years, that
# would show up here as a level shift rather than season-to-season noise.

# %%
eligible["y1_strict"] = y1_strict
by_season = eligible.groupby("SEASON_ID")["y1_strict"].mean().sort_index()
print(by_season)

fig, ax = plt.subplots(figsize=(8, 4))
(by_season * 100).plot.bar(ax=ax)
ax.set_title("1-game miss rate (strict) by season")
ax.set_ylabel("% positive")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "class_balance_by_season.png", dpi=120)
plt.close()

# %% [markdown]
# ## Summary
#
# - Injury data covers 2016-10-25 -> **2025-01-12** only — a real gap against
#   the full 10-season (through 2025-26) game log. 82.8% of player-games end up
#   eligible for a label once that cutoff is applied; the rest aren't wrong,
#   just unlabelable from this source as-is.
# - 16,871 usable transaction rows, 99.76% matched to a PLAYER_ID after
#   normalizing diacritics/punctuation/parenthetical aliases.
# - IL placement reasons: 66.7% explicit injury, 18.1% unspecified (treated as
#   injury), and ~15% that isn't injury at all (illness, COVID protocol, rest,
#   personal) — excluding that 15% is the "strict" definition; including it is
#   "broad". Recommend strict, but this is a labeling decision the modeling
#   phase should own explicitly, not inherit silently from this notebook.
# - Naively pairing each "placed on IL" with whatever "activated" event comes
#   next in the log is wrong more often than expected: 416 of 1063 players
#   (39%) had at least one interval that overlapped a date they demonstrably
#   played (per the game log), usually because an activation transaction was
#   never logged. Clipping intervals to the player's own game log fixed all
#   of it (0 contradictions remain) and roughly **quartered** the class-balance
#   numbers — this was the single biggest lever in the whole notebook, bigger
#   than the reason-category (strict vs. broad) choice.
# - **Class balance, strict / broad** (n=211,265 eligible player-games):
#   - 1-game: **2.41% / 2.62%**
#   - 3-game: **7.00% / 7.60%**
#   - 10-game: **20.01% / 21.62%**
#   Season-by-season the 1-game rate stays in a 1.8%-3.0% band with no level
#   shift (see class_balance_by_season.png) — including the COVID season,
#   which is unremarkable once health-and-safety-protocol placements are
#   excluded under "strict". That's a good sign the labeling is consistent
#   across the 10 years, not an artifact of changing data-collection practices.
# - These are still first-pass numbers — known simplifications to revisit in
#   feature_engineering.py before they become real training labels: next-N
#   windows can cross an offseason boundary into the next season, trades
#   mid-injury-span aren't special-cased, and ~10 residual unmatched player
#   names (mostly fringe two-way players, one literal data defect) are
#   silently dropped. Good enough as-is to size the modeling problem though:
#   a rare-ish positive class, more imbalanced for 1-game than 10-game —
#   expect to need class-weighting or resampling.
