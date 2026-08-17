"""Build model features and labels from processed data.

Two groups of functions:
- Label construction (matching injury-log names to PLAYER_IDs, reconstructing
  out-intervals, computing y_1game/y_3game/y_10game) — see build_labeled_dataset,
  which is the single entry point notebooks/03 calls.
- Feature construction (rest days/back-to-backs, player bio/age/experience) —
  compute_rest_features and compute_bio_features, tied together in
  notebooks/04_build_features.py. Team pace doesn't need a function here since
  it's just a season+team join, done directly in that notebook.

Also: compute_workload_features (rolling/acute-chronic minutes load) and
compute_extended_absence_features (cause-agnostic long-gap flag), both tied
together in notebooks/05_build_workload_features.py.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

# Reasons an IL placement is not really an "injury" for labeling purposes.
NON_INJURY_CATEGORIES = {"covid_protocol", "rest", "personal", "suspension"}

LONG_LAYOFF_DAYS = 14  # threshold for the is_long_layoff flag in compute_rest_features


def compute_rest_features(player_log: pd.DataFrame) -> pd.DataFrame:
    """Per-player-game rest-day features derived from GAME_DATE (validated in
    notebooks/01_nba_api_eda.py). Computed *within season only* — a player's
    first game of a season gets NaN days_rest rather than a multi-month
    "rest" value spanning the offseason, which isn't meaningful rest.

    - days_rest: calendar-day gap to this player's previous game this season,
      minus 1 (playing on consecutive calendar days = 0 days rest, i.e. a
      back-to-back). NaN for a player's first game of each season.
    - is_back_to_back: days_rest == 0.
    - is_long_layoff: days_rest > LONG_LAYOFF_DAYS. Kept as its own flag
      rather than capping days_rest itself — a long gap (often a return
      from injury) is itself a meaningful signal, not noise to suppress.

    Returns PLAYER_ID, GAME_ID, GAME_DATE, days_rest, is_back_to_back,
    is_long_layoff — join onto game_labels.csv by (PLAYER_ID, GAME_ID).
    """
    df = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "SEASON_ID", "GAME_DATE"])

    date_gap = df.groupby(["PLAYER_ID", "SEASON_ID"])["GAME_DATE"].diff().dt.days
    df["days_rest"] = date_gap - 1
    df["is_back_to_back"] = df["days_rest"] == 0
    df["is_long_layoff"] = df["days_rest"] > LONG_LAYOFF_DAYS

    return df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "days_rest", "is_back_to_back", "is_long_layoff"]]


def normalize_name(name: str) -> str:
    """Fold a player name down for matching: strip parenthetical legal-name
    clarifications ("Reggie Jackson (Shon)"), diacritics ("Porziņģis" ->
    "Porzingis"), periods ("C.J." -> "CJ"), and common suffixes.
    """
    name = re.sub(r"\([^)]*\)", "", name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.replace(".", "")
    name = re.sub(r"\s+(Jr|Sr|II|III|IV|V)\.?$", "", name, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", name).strip().lower()


def match_player_ids(name_cells: pd.Series, static_players: pd.DataFrame) -> pd.Series:
    """Map a Series of name cells (possibly "Name A / Name B" aliases) to
    nba_api PLAYER_IDs via normalized exact match. Returns a Series of
    nullable Int64, same index as name_cells; unmatched entries are NaN.
    """
    name_to_id: dict[str, int] = {}
    for full_name, player_id in zip(static_players["full_name"], static_players["id"]):
        name_to_id.setdefault(normalize_name(full_name), player_id)

    def match_one(cell):
        if not isinstance(cell, str):
            return None
        for alias in cell.split("/"):
            pid = name_to_id.get(normalize_name(alias))
            if pid is not None:
                return pid
        return None

    return name_cells.apply(match_one).astype("Int64")


def categorize_injury_reason(note: str) -> str:
    """Bucket a transaction Note into a reason category. 'injury' and
    'unspecified' are treated as genuine injury; see NON_INJURY_CATEGORIES
    for the rest (COVID protocol, rest/load management, personal, suspension).
    """
    if not isinstance(note, str):
        return "unknown"
    n = note.lower()
    if "health and safety protocol" in n or "covid" in n:
        return "covid_protocol"
    if n.strip() in ("placed on il", "activated from il"):
        return "unspecified"
    if "rest" in n:
        return "rest"
    if "illness" in n or "flu" in n or "migraine" in n:
        return "illness"
    if any(kw in n for kw in ("not with team", "personal", "paternity", "family", "bereavement")):
        return "personal"
    if "suspen" in n:
        return "suspension"
    return "injury"


def build_out_intervals(
    events: pd.DataFrame, coverage_end: pd.Timestamp
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """events: columns player_id, Date, event_type ('out'/'available'), one row
    per placed/activated transaction already filtered to the categories that
    should count as "injured". Reconstructs each player's out-intervals via a
    state machine (repeated same-type events are no-ops, so double-logged
    "placed on IL" transactions don't reset anything). A still-open interval
    at the end of the data is right-censored at coverage_end + 1 day.

    Returns {player_id: (starts, ends)} as sorted, non-overlapping datetime64
    arrays suitable for a searchsorted-based "is this date inside an interval" check.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, grp in events.sort_values("Date").groupby("player_id"):
        starts, ends = [], []
        open_start = None
        for date, etype in zip(grp["Date"], grp["event_type"]):
            if etype == "out":
                if open_start is None:
                    open_start = date
            elif open_start is not None:
                starts.append(open_start)
                ends.append(date)
                open_start = None
        if open_start is not None:
            starts.append(open_start)
            ends.append(coverage_end + pd.Timedelta(days=1))
        out[pid] = (np.array(starts, dtype="datetime64[ns]"), np.array(ends, dtype="datetime64[ns]"))
    return out


def clip_intervals_to_played_games(
    intervals: dict[int, tuple[np.ndarray, np.ndarray]], player_log: pd.DataFrame
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """An out-interval can't be right if the player has a logged game
    appearance inside it — that's direct evidence they were available, and
    means the "activated" transaction is simply missing from the injury log
    (common right at the data's coverage edge, where the transaction log
    just stops updating rather than the player being out for years). Trims
    each interval's end to just before its earliest contradicting appearance.
    """
    play_dates_by_player = {
        pid: np.sort(grp["GAME_DATE"].values.astype("datetime64[ns]"))
        for pid, grp in player_log.groupby("PLAYER_ID")
    }
    clipped = {}
    for pid, (starts, ends) in intervals.items():
        play_dates = play_dates_by_player.get(pid)
        if play_dates is None or len(play_dates) == 0:
            clipped[pid] = (starts, ends)
            continue
        new_ends = ends.copy()
        for i, (s, e) in enumerate(zip(starts, ends)):
            in_range = play_dates[(play_dates >= s) & (play_dates < e)]
            if len(in_range):
                new_ends[i] = in_range.min()
        clipped[pid] = (starts, new_ends)
    return clipped


def compute_workload_features(
    player_log: pd.DataFrame,
    acute_window: int = 3,
    chronic_window: int = 15,
    density_window_days: int = 14,
) -> pd.DataFrame:
    """Rolling workload features from MIN (minutes played). Every column
    here is computed from STRICTLY PRIOR games (shift(1) before rolling) —
    none of it uses the current game's own minutes, since the point is
    "load heading into tonight's game," known before it's played, not what
    happened in it.

    Rolling windows span season boundaries (unlike compute_rest_features's
    days_rest) — a player's minutes trend from the end of last season is
    still meaningful context for their first few games of a new one, and
    unlike raw day-counts, a minutes average doesn't blow up across the
    offseason gap the way days_rest would.

    - min_avg_acute (mean MIN over the last `acute_window` games) and
      min_avg_chronic (mean MIN over the last `chronic_window` games): NaN
      until that many prior games exist.
    - acute_chronic_ratio: min_avg_acute / min_avg_chronic — the sports-
      science "Acute:Chronic Workload Ratio" (ACWR). A recent spike relative
      to a player's own baseline (ratio meaningfully above 1) is a stronger,
      more specific injury-risk signal in the literature than raw minutes
      alone: a player who always plays 38 minutes isn't newly at risk, but
      one who jumps from a 22-minute baseline to 38 might be. NaN wherever
      the chronic average isn't available or is 0.
    - games_last_{density_window_days}d: count of games played in the
      trailing calendar window (excluding this one) — catches condensed-
      schedule stretches that a pure game-count window can't distinguish
      from a normal pace (same number of recent games, played much closer together).

    Returns PLAYER_ID, GAME_ID, GAME_DATE, min_avg_acute, min_avg_chronic,
    acute_chronic_ratio, games_last_{density_window_days}d. Join onto
    features.csv by (PLAYER_ID, GAME_ID).
    """
    df = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "MIN"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    grp_min = df.groupby("PLAYER_ID")["MIN"]
    df["min_avg_acute"] = grp_min.transform(
        lambda s: s.shift(1).rolling(acute_window, min_periods=acute_window).mean()
    )
    df["min_avg_chronic"] = grp_min.transform(
        lambda s: s.shift(1).rolling(chronic_window, min_periods=chronic_window).mean()
    )
    df["acute_chronic_ratio"] = df["min_avg_acute"] / df["min_avg_chronic"].replace(0, np.nan)

    density_col = f"games_last_{density_window_days}d"

    def count_recent(g: pd.DataFrame) -> pd.Series:
        # must return a Series indexed like g, not a bare array — a bare
        # array here gets treated as one value per *group* and broadcast to
        # every row in it, rather than aligned per row (caught by testing
        # against real data: every row for a player showed the same list).
        s = g.set_index("GAME_DATE")["GAME_ID"]
        counts = s.rolling(f"{density_window_days}D", closed="left").count()
        return pd.Series(counts.values, index=g.index)

    df[density_col] = df.groupby("PLAYER_ID", group_keys=False)[["GAME_DATE", "GAME_ID"]].apply(count_recent)

    return df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "min_avg_acute", "min_avg_chronic",
               "acute_chronic_ratio", density_col]]


def compute_cohort(bio: pd.DataFrame) -> pd.DataFrame:
    """Position x BMI-tercile label per player, used only to build backfill
    cohorts in apply_cohort_backfill (not a model feature itself — the
    multi-hot position flags and continuous BMI stay the real features).
    Position priority center > forward > guard when multiple flags are set,
    just to get one label per player for grouping.
    """
    b = bio.rename(columns={"PERSON_ID": "PLAYER_ID"}).copy()
    feet_inches = b["HEIGHT"].str.split("-", expand=True).astype(float)
    height_inches = feet_inches[0] * 12 + feet_inches[1]
    bmi = 703 * b["WEIGHT"] / (height_inches ** 2)

    position = b["POSITION"].fillna("")
    primary_position = np.select(
        [position.str.contains("Center"), position.str.contains("Forward")],
        ["center", "forward"], default="guard",
    )
    bmi_tercile = pd.qcut(bmi, 3, labels=["low_bmi", "mid_bmi", "high_bmi"])
    return pd.DataFrame({
        "PLAYER_ID": b["PLAYER_ID"],
        "cohort": [f"{p}_{t}" if pd.notna(t) else np.nan for p, t in zip(primary_position, bmi_tercile)],
    })


def apply_cohort_backfill(
    workload: pd.DataFrame,
    player_log: pd.DataFrame,
    bio: pd.DataFrame,
    train_season_ids: list[str],
    acute_window: int = 3,
    chronic_window: int = 15,
) -> pd.DataFrame:
    """min_avg_acute/min_avg_chronic start out NaN simply because a player
    hasn't played enough games yet for a reliable rolling average — not
    because their workload is unknowable. Motivated by a concrete finding:
    players 23 and under are missing min_avg_chronic 17% of the time versus
    3.7% for veterans, and it's the single most important feature in every
    model — a real cold-start problem, the same kind recommendation systems
    hit with brand-new users who have no history yet.

    Fix: a shrinkage-weighted blend of the player's own partial history (an
    expanding average of their actual games so far) and a "typical early-
    career minutes" baseline from similar players — same primary position,
    similar BMI (see compute_cohort). At zero career games (a debut), the
    blend is 100% cohort baseline; by the time the real window is full, it's
    100% personal data (unchanged from compute_workload_features) — the
    cohort estimate's influence fades out smoothly as real history accumulates.

    Cohort baselines are computed using ONLY train_season_ids, the same
    leakage-avoidance principle used for calibration fitting elsewhere in
    this pipeline — never fit on val/test, only applied to it.

    Returns `workload` with min_avg_acute, min_avg_chronic, and
    acute_chronic_ratio backfilled where they were NaN (all other columns,
    and all rows that already had real values, untouched).
    """
    cohort = compute_cohort(bio)

    pl = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN"]].copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl = pl.sort_values(["PLAYER_ID", "GAME_DATE"])
    pl = pl.merge(cohort, on="PLAYER_ID", how="left")

    grp = pl.groupby("PLAYER_ID")["MIN"]
    pl["games_so_far"] = grp.cumcount()  # 0 on a player's debut game, counts strictly-prior games
    pl["personal_expanding_avg"] = grp.transform(lambda s: s.shift(1).expanding().mean())

    train_mask = pl["SEASON_ID"].astype(str).isin(train_season_ids)
    early_career = pl.loc[train_mask & (pl["games_so_far"] < chronic_window)]
    cohort_baseline_map = early_career.groupby("cohort")["MIN"].mean()
    overall_baseline = early_career["MIN"].mean()
    pl["cohort_baseline"] = pl["cohort"].map(cohort_baseline_map).fillna(overall_baseline)

    def blended(window: int) -> pd.Series:
        weight = (pl["games_so_far"] / window).clip(0, 1)
        personal = pl["personal_expanding_avg"].fillna(pl["cohort_baseline"])
        return weight * personal + (1 - weight) * pl["cohort_baseline"]

    pl["min_avg_acute_bf"] = blended(acute_window)
    pl["min_avg_chronic_bf"] = blended(chronic_window)
    pl["GAME_ID"] = pl["GAME_ID"].astype(str).str.zfill(10)

    result = workload.merge(
        pl[["PLAYER_ID", "GAME_ID", "min_avg_acute_bf", "min_avg_chronic_bf"]],
        on=["PLAYER_ID", "GAME_ID"], how="left",
    )
    result["min_avg_acute"] = result["min_avg_acute"].fillna(result["min_avg_acute_bf"])
    result["min_avg_chronic"] = result["min_avg_chronic"].fillna(result["min_avg_chronic_bf"])
    result["acute_chronic_ratio"] = result["acute_chronic_ratio"].fillna(
        result["min_avg_acute"] / result["min_avg_chronic"].replace(0, np.nan)
    )
    return result.drop(columns=["min_avg_acute_bf", "min_avg_chronic_bf"])


def compute_extended_absence_features(player_log: pd.DataFrame) -> pd.DataFrame:
    """Flags a player's return from a gap of one or more full skipped
    seasons — regardless of cause. Deliberately cause-agnostic: the injury
    transaction log doesn't reliably capture every real absence (confirmed
    concretely — Klay Thompson's 2019 Finals ACL tear was never logged as an
    IL transaction at all, likely because it happened in the playoffs), but
    plenty of season-plus gaps in the game log are normal roster churn
    (fringe/two-way players cycling out of the league), not injury. This
    doesn't claim to know which — it's just "did this player just have a
    long absence from NBA games," which is useful context either way and
    safe to compute without that causal judgment call.

    - games_since_extended_return: 0 on the first game back after such a
      gap, counting up for subsequent games; NaN if no such gap has occurred
      yet for this player in the data (including their whole career before
      their very first tracked gap, if any).
    - is_returning_from_extended_absence: games_since_extended_return == 0.

    Returns PLAYER_ID, GAME_ID, GAME_DATE, games_since_extended_return,
    is_returning_from_extended_absence.
    """
    season_idx_map = {f"2{y}": i for i, y in enumerate(range(2016, 2026))}

    df = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["season_idx"] = df["SEASON_ID"].astype(str).map(season_idx_map)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    def flag_group(season_idx: pd.Series) -> np.ndarray:
        vals = season_idx.values
        prev = np.concatenate([[vals[0]], vals[:-1]])
        is_return = (vals - prev) > 1
        is_return[0] = False  # no prior season to compare against on a player's first-ever game

        out = np.full(len(vals), np.nan)
        counter = np.nan
        for i in range(len(vals)):
            if is_return[i]:
                counter = 0
            elif not np.isnan(counter):
                counter += 1
            out[i] = counter
        return out

    df["games_since_extended_return"] = df.groupby("PLAYER_ID")["season_idx"].transform(
        lambda s: pd.Series(flag_group(s), index=s.index)
    )
    df["is_returning_from_extended_absence"] = df["games_since_extended_return"] == 0

    return df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "games_since_extended_return",
               "is_returning_from_extended_absence"]]


# Arena coordinates (lat, lon), keyed by TEAM_ABBREVIATION — verified against
# the abbreviations actually present in our cached team_game_log data.
ARENA_LOCATIONS = {
    "ATL": (33.7573, -84.3963), "BOS": (42.3662, -71.0621), "BKN": (40.6826, -73.9754),
    "CHA": (35.2251, -80.8392), "CHI": (41.8807, -87.6742), "CLE": (41.4965, -81.6882),
    "DAL": (32.7905, -96.8103), "DEN": (39.7487, -105.0077), "DET": (42.3410, -83.0552),
    "GSW": (37.7680, -122.3877), "HOU": (29.7508, -95.3621), "IND": (39.7640, -86.1555),
    "LAC": (33.9420, -118.3410), "LAL": (34.0430, -118.2673), "MEM": (35.1382, -90.0505),
    "MIA": (25.7814, -80.1870), "MIL": (43.0451, -87.9172), "MIN": (44.9795, -93.2760),
    "NOP": (29.9490, -90.0821), "NYK": (40.7505, -73.9934), "OKC": (35.4634, -97.5151),
    "ORL": (28.5392, -81.3839), "PHI": (39.9012, -75.1720), "PHX": (33.4457, -112.0712),
    "POR": (45.5316, -122.6668), "SAC": (38.5802, -121.4997), "SAS": (29.4269, -98.4375),
    "TOR": (43.6435, -79.3791), "UTA": (40.7683, -111.9011), "WAS": (38.8981, -77.0209),
}

# Three teams changed arenas within our 2016-2025 window — all confirmed (web
# search) to have landed cleanly at a season boundary, not mid-season. Maps
# (abbreviation, first season AT the new arena) -> the OLD location that
# applies to every season strictly before it.
ARENA_OVERRIDES_BEFORE = {
    ("DET", "22017"): (42.6963, -83.2453),   # The Palace of Auburn Hills, through 2016-17
    ("GSW", "22019"): (37.7502, -122.2030),  # Oracle Arena (Oakland), through 2018-19
    ("LAC", "22024"): (34.0430, -118.2673),  # shared Crypto.com Arena, through 2023-24
}


def _haversine_miles(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance in miles — the standard simplification used in
    sports travel-fatigue research (straight-line, not real flight paths,
    which we have no way to source)."""
    r = 3958.8  # Earth's radius in miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return r * 2 * np.arcsin(np.sqrt(a))


def compute_travel_features(team_log: pd.DataFrame, density_window_days: int = 14) -> pd.DataFrame:
    """Team-level travel-fatigue features, one row per team-game:

    - travel_distance_last_game: great-circle miles from the previous game's
      location to this one (0 for a same-city back-to-back, e.g. Lakers/Clippers).
    - travel_distance_last_14d: cumulative miles over the trailing window —
      blends distance and frequency together, the same way the acute:chronic
      workload features do (lots of long hops close together compounds; the
      same number of games spread over short hops doesn't).
    - current_road_trip_length: consecutive away games right now (0 on a
      home game). Exactly computable, not estimated — MATCHUP already tells
      us home vs away for every game.
    - days_since_home: calendar days since this team's last home game.

    Deliberately no attempt at a long-memory "does September travel still
    matter in December" model — same reasoning as the workload features:
    build the rolling-window version, let permutation importance say
    whether a longer window would have mattered, rather than presupposing
    an effect that might not be there.

    Returns TEAM_ID, GAME_ID, GAME_DATE, plus the four columns above — merge
    onto features.csv by (TEAM_ID, GAME_ID).
    """
    df = team_log[["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "SEASON_ID", "MATCHUP"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["SEASON_ID"] = df["SEASON_ID"].astype(str)
    df["is_home"] = df["MATCHUP"].str.contains("vs.")
    df = df.sort_values(["TEAM_ID", "GAME_DATE"])

    # location of THIS game: home team's arena if this team is home, else the
    # opponent's arena (parsed from "TEAM @ OPP" / "TEAM vs. OPP")
    opponent = df["MATCHUP"].str.extract(r"(?:vs\.|@)\s*(\w+)")[0]
    home_abbr = np.where(df["is_home"], df["TEAM_ABBREVIATION"], opponent)

    # explicit per-row lookup (small data — clarity over cleverness), applying
    # the pre-move location for the 3 teams that changed arenas mid-window
    locations = []
    for abbr, season in zip(home_abbr, df["SEASON_ID"]):
        loc = ARENA_LOCATIONS.get(abbr)
        for (o_abbr, o_before_season), o_loc in ARENA_OVERRIDES_BEFORE.items():
            if abbr == o_abbr and season < o_before_season:
                loc = o_loc
        locations.append(loc if loc else (np.nan, np.nan))
    df["game_lat"] = [loc[0] for loc in locations]
    df["game_lon"] = [loc[1] for loc in locations]

    grp = df.groupby("TEAM_ID")
    prev_lat = grp["game_lat"].shift(1)
    prev_lon = grp["game_lon"].shift(1)
    df["travel_distance_last_game"] = _haversine_miles(prev_lat, prev_lon, df["game_lat"], df["game_lon"]).fillna(0)

    def rolling_sum_by_date(g: pd.DataFrame) -> pd.Series:
        s = g.set_index("GAME_DATE")["travel_distance_last_game"]
        return pd.Series(s.rolling(f"{density_window_days}D", closed="left").sum().values, index=g.index)

    # a season-opener's trailing 14-day window is genuinely empty (the
    # offseason gap always exceeds 14 days) — pandas' time-based rolling
    # returns NaN for a truly empty window's sum rather than 0, but "zero
    # games in the last 14 days" is exactly what's true here, so fill it in
    df["travel_distance_last_14d"] = df.groupby("TEAM_ID", group_keys=False)[
        ["GAME_DATE", "travel_distance_last_game"]
    ].apply(rolling_sum_by_date).fillna(0)

    def road_trip_and_days_since_home(g: pd.DataFrame) -> pd.DataFrame:
        is_home = g["is_home"].values
        dates = g["GAME_DATE"].values
        road_trip = np.zeros(len(g), dtype=int)
        days_since = np.full(len(g), np.nan)
        streak = 0
        last_home_date = None
        for i in range(len(g)):
            road_trip[i] = 0 if is_home[i] else streak
            streak = 0 if is_home[i] else streak + 1
            if last_home_date is not None:
                days_since[i] = (dates[i] - last_home_date).astype("timedelta64[D]").astype(float)
            if is_home[i]:
                last_home_date = dates[i]
        return pd.DataFrame({"current_road_trip_length": road_trip, "days_since_home": days_since}, index=g.index)

    extra = df.groupby("TEAM_ID", group_keys=False)[["GAME_DATE", "is_home"]].apply(road_trip_and_days_since_home)
    df["current_road_trip_length"] = extra["current_road_trip_length"]
    df["days_since_home"] = extra["days_since_home"]

    return df[["TEAM_ID", "GAME_ID", "GAME_DATE", "travel_distance_last_game", "travel_distance_last_14d",
               "current_road_trip_length", "days_since_home"]]


def flag_possible_unlogged_gap(player_log: pd.DataFrame, game_rows: pd.DataFrame) -> pd.DataFrame:
    """Marks label rows from the season right before one of the extended
    absences described in compute_extended_absence_features — i.e. rows
    where y_1game/y_3game/y_10game might be a false negative because the
    injury transaction log missed whatever caused the player to disappear
    for a season or more (confirmed real example: Klay Thompson's 2019 ACL
    tear). Deliberately does NOT change the label values themselves — see
    the project notes on why a blanket "gap = injury" assumption would be
    wrong for a meaningful share of these cases (e.g. Darren Collison's
    entry in this same pattern is a voluntary retirement, not an injury).

    This is a transparency flag for filtering/inspection, not a fix. The
    real fix would be sourcing a more complete injury history (including
    playoff-time injuries) to replace the guess with a real label.

    Returns PLAYER_ID, GAME_ID, possible_unlogged_gap (bool).
    """
    season_idx_map = {f"2{y}": i for i, y in enumerate(range(2016, 2026))}

    pl = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID"]].copy()
    pl["GAME_DATE"] = pd.to_datetime(pl["GAME_DATE"])
    pl["season_idx"] = pl["SEASON_ID"].astype(str).map(season_idx_map)

    seasons_played = pl.groupby("PLAYER_ID")["season_idx"].apply(lambda s: sorted(s.unique()))
    flagged_player_seasons = set()
    for pid, seas in seasons_played.items():
        for i in range(len(seas) - 1):
            if seas[i + 1] - seas[i] > 1:
                flagged_player_seasons.add((pid, seas[i]))  # the season right before the gap

    rows = game_rows[["PLAYER_ID", "GAME_ID", "SEASON_ID"]].copy()
    rows["season_idx"] = rows["SEASON_ID"].astype(str).map(season_idx_map)
    rows["possible_unlogged_gap"] = [
        (pid, sidx) in flagged_player_seasons for pid, sidx in zip(rows["PLAYER_ID"], rows["season_idx"])
    ]
    return rows[["PLAYER_ID", "GAME_ID", "possible_unlogged_gap"]]


def compute_bio_features(bio: pd.DataFrame, game_rows: pd.DataFrame) -> pd.DataFrame:
    """Player bio features for each row in game_rows (needs PLAYER_ID,
    GAME_ID, GAME_DATE as datetime, SEASON_ID):

    - height_inches, weight_lbs, bmi, plays_guard/plays_forward/plays_center
      (multi-hot — a "Guard-Forward" gets both plays_guard and plays_forward):
      static per player. Height/weight/position don't have per-season history
      in this source, so the current listed values are applied across a
      player's whole career here — a standard, minor approximation.
    - bmi: standard imperial BMI (703 * lbs / inches^2). Height and weight in
      isolation don't capture "mass relative to frame" — a lean 7-footer at
      200 lbs and a sturdy 6'6" player at 250 lbs can have very different
      injury profiles despite neither raw number looking unusual on its own.
      A tree model *could* in principle rediscover this by combining height
      and weight splits on its own, but handing it the ratio directly makes
      an already-plausible risk factor easier to actually learn from the
      data we have, rather than hoping the model reconstructs it.
    - age_years, experience_years: computed AS OF each row's GAME_DATE, not
      taken as a snapshot. Age and tenure obviously change year to year, so
      using nba_api's own SEASON_EXP (today's career-total experience) or a
      static age would silently be wrong on every row except the current
      season — that's why get_player_bio() fetches FROM_YEAR (rookie season)
      instead, which stays correct for any historical game date.

    Returns PLAYER_ID, GAME_ID, height_inches, weight_lbs, bmi, plays_guard,
    plays_forward, plays_center, age_years, experience_years — merge onto
    game_labels.csv by (PLAYER_ID, GAME_ID).
    """
    b = bio.rename(columns={"PERSON_ID": "PLAYER_ID"}).copy()
    feet_inches = b["HEIGHT"].str.split("-", expand=True).astype(float)
    b["height_inches"] = feet_inches[0] * 12 + feet_inches[1]
    b["weight_lbs"] = b["WEIGHT"]
    b["bmi"] = 703 * b["weight_lbs"] / (b["height_inches"] ** 2)
    b["plays_guard"] = b["POSITION"].str.contains("Guard", na=False)
    b["plays_forward"] = b["POSITION"].str.contains("Forward", na=False)
    b["plays_center"] = b["POSITION"].str.contains("Center", na=False)
    b["birthdate"] = pd.to_datetime(b["BIRTHDATE"])

    df = game_rows[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID"]].merge(
        b[["PLAYER_ID", "height_inches", "weight_lbs", "bmi", "plays_guard", "plays_forward",
           "plays_center", "birthdate", "FROM_YEAR"]],
        on="PLAYER_ID", how="left",
    )
    df["age_years"] = (df["GAME_DATE"] - df["birthdate"]).dt.days / 365.25
    season_start_year = df["SEASON_ID"].astype(str).str[1:].astype(int)  # "22016" -> 2016
    df["experience_years"] = season_start_year - df["FROM_YEAR"]

    return df[["PLAYER_ID", "GAME_ID", "height_inches", "weight_lbs", "bmi", "plays_guard",
               "plays_forward", "plays_center", "age_years", "experience_years"]]


def is_out_on_dates(
    intervals: dict[int, tuple[np.ndarray, np.ndarray]], player_id: int, query_dates: np.ndarray
) -> np.ndarray:
    """Bool array, True where query_dates[i] falls inside one of player_id's out-intervals."""
    if player_id not in intervals:
        return np.zeros(len(query_dates), dtype=bool)
    starts, ends = intervals[player_id]
    if len(starts) == 0:
        return np.zeros(len(query_dates), dtype=bool)
    idx = np.searchsorted(starts, query_dates, side="right") - 1
    result = np.zeros(len(query_dates), dtype=bool)
    valid = idx >= 0
    result[valid] = query_dates[valid] < ends[idx[valid]]
    return result


def compute_injury_history_features(
    intervals: dict[int, tuple[np.ndarray, np.ndarray]], game_rows: pd.DataFrame, lookback_days: int = 365
) -> pd.DataFrame:
    """Prior injury history, from the same reconstructed out-intervals used
    for the labels themselves. Nothing in the feature set so far captures
    this — sports science treats prior injury as one of the strongest known
    predictors of future injury, and a veteran with a long injury history
    is a different risk profile than one with a clean one, even if their
    recent workload numbers look identical. Motivated by a concrete finding:
    the model underperforms specifically on veteran players (32+), despite
    them having the *lowest* missing-data rate on the single most important
    existing feature — so the gap likely isn't a data-availability problem
    the way it is for young players, it's a missing-signal problem.

    Only counts intervals that have already CLOSED (ended) before this row's
    game date — nothing here can leak information from the future.

    - career_injury_count: how many separate injury absences this player has
      already had, as of this game, within our data window (left-censored —
      doesn't know about injuries before 2016-17).
    - days_missed_last_{lookback_days}d: total days spent out across
      (possibly multiple) injury intervals in the trailing lookback_days
      before this game — recent burden, not just career total.

    Returns PLAYER_ID, GAME_ID, career_injury_count, days_missed_last_{N}d.
    """
    rows = game_rows[["PLAYER_ID", "GAME_ID", "GAME_DATE"]].copy()
    rows["GAME_DATE"] = pd.to_datetime(rows["GAME_DATE"])

    count_col = np.zeros(len(rows), dtype=int)
    days_col = np.zeros(len(rows), dtype=float)

    for player_id, grp in rows.groupby("PLAYER_ID"):
        starts, ends = intervals.get(player_id, (np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")))
        if len(starts) == 0:
            continue
        idx = grp.index.values
        dates = grp["GAME_DATE"].values.astype("datetime64[ns]")

        count_col[idx] = np.searchsorted(ends, dates, side="right")

        lookback_start = dates - np.timedelta64(lookback_days, "D")
        for i, (date, since) in enumerate(zip(dates, lookback_start)):
            overlap_start = np.maximum(starts, since)
            overlap_end = np.minimum(ends, date)
            overlap_days = np.clip((overlap_end - overlap_start).astype("timedelta64[D]").astype(float), 0, None)
            days_col[idx[i]] = overlap_days.sum()

    days_label = f"days_missed_last_{lookback_days}d"
    rows["career_injury_count"] = count_col
    rows[days_label] = days_col
    return rows[["PLAYER_ID", "GAME_ID", "career_injury_count", days_label]]


def next_team_game_dates(team_dates: np.ndarray, game_date: np.datetime64, n: int) -> np.ndarray | None:
    """The next n scheduled dates for a team strictly after game_date, from
    that team's full-season date array (see get_team_game_log). None if fewer
    than n remain in the data we have.
    """
    idx = np.searchsorted(team_dates, game_date, side="right")
    future = team_dates[idx : idx + n]
    return future if len(future) == n else None


def build_next_n_game_index(
    player_log: pd.DataFrame, team_dates: dict[int, np.ndarray], n: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized version of next_team_game_dates for a whole player_log at
    once. For every row, the next n scheduled dates for that row's team
    after that row's game. Returns (dates[n_rows, n] datetime64, valid[n_rows]
    bool) — valid is False where fewer than n future games exist in team_dates.
    """
    team_id_arr = player_log["TEAM_ID"].values
    game_date_arr = player_log["GAME_DATE"].values.astype("datetime64[ns]")
    result = np.full((len(player_log), n), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    valid = np.zeros(len(player_log), dtype=bool)

    for team_id, dates in team_dates.items():
        mask = team_id_arr == team_id
        if not mask.any():
            continue
        idx = np.searchsorted(dates, game_date_arr[mask], side="right")
        ok = (idx + n) <= len(dates)
        positions = np.where(mask)[0][ok]
        gather_idx = idx[ok][:, None] + np.arange(n)[None, :]
        result[positions] = dates[gather_idx]
        valid[positions] = True
    return result, valid


def build_labeled_dataset(
    player_log: pd.DataFrame,
    team_log: pd.DataFrame,
    injury_raw: pd.DataFrame,
    static_players: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict[int, tuple[np.ndarray, np.ndarray]]]:
    """The full label-construction pipeline validated in
    notebooks/02_injury_data_eda.py, packaged as one call: match injury-log
    names to PLAYER_ID, categorize placement reasons, reconstruct out-intervals
    under the strict (genuine-injury) definition, clip them against real
    game-log appearances, then compute y_1game/y_3game/y_10game for every
    player-game whose next-10-games window is fully inside the injury data's
    coverage window.

    Returns (labeled_df, diagnostics, intervals) — labeled_df has one row per
    eligible player-game (PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION,
    GAME_ID, GAME_DATE, SEASON_ID, y_1game, y_3game, y_10game); diagnostics is
    a dict of summary numbers worth reporting (match rate, eligible row
    count, etc); intervals is the strict-definition, played-game-clipped
    out-interval reconstruction (see build_out_intervals /
    clip_intervals_to_played_games) — exposed so other features (e.g. prior
    injury history) can reuse the same validated intervals without
    recomputing name-matching and interval reconstruction from scratch.
    """
    injury_raw = injury_raw.copy()
    injury_raw["Date"] = pd.to_datetime(injury_raw["Date"])
    injury_raw = injury_raw[~(injury_raw["Acquired"].isna() & injury_raw["Relinquished"].isna())]
    coverage_end = injury_raw["Date"].max()

    injury_raw["event_type"] = np.where(injury_raw["Relinquished"].notna(), "out", "available")
    injury_raw["player_name_raw"] = injury_raw["Relinquished"].fillna(injury_raw["Acquired"])
    injury_raw["player_id"] = match_player_ids(injury_raw["player_name_raw"], static_players)
    matched = injury_raw[injury_raw["player_id"].notna()].copy()
    matched["player_id"] = matched["player_id"].astype(int)
    matched["category"] = np.where(
        matched["event_type"] == "out",
        matched["Notes"].apply(categorize_injury_reason),
        "n/a",
    )

    strict_events = matched[~matched["category"].isin(NON_INJURY_CATEGORIES)]
    intervals = build_out_intervals(strict_events, coverage_end)

    player_log = player_log.copy()
    player_log["GAME_DATE"] = pd.to_datetime(player_log["GAME_DATE"])
    team_log = team_log.copy()
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

    intervals = clip_intervals_to_played_games(intervals, player_log)

    team_dates = {
        tid: grp.sort_values("GAME_DATE")["GAME_DATE"].values.astype("datetime64[ns]")
        for tid, grp in team_log.groupby("TEAM_ID")
    }
    next10_dates, has_next10 = build_next_n_game_index(player_log, team_dates, n=10)
    within_coverage = has_next10 & (next10_dates[:, -1] <= np.datetime64(coverage_end))

    id_cols = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "SEASON_ID"]
    eligible = player_log.loc[within_coverage, id_cols].copy()
    eligible_future10 = next10_dates[within_coverage]

    n_rows = len(eligible)
    flags = np.zeros((n_rows, 10), dtype=bool)
    pid_arr = eligible["PLAYER_ID"].values
    for pid in np.unique(pid_arr):
        rows = np.where(pid_arr == pid)[0]
        fut = eligible_future10[rows]
        for k in range(10):
            flags[rows, k] = is_out_on_dates(intervals, pid, fut[:, k])

    eligible["y_1game"] = flags[:, 0]
    eligible["y_3game"] = flags[:, :3].any(axis=1)
    eligible["y_10game"] = flags[:, :10].any(axis=1)
    eligible = eligible.reset_index(drop=True)

    diagnostics = {
        "coverage_end": coverage_end,
        "injury_rows_total": len(injury_raw),
        "injury_rows_matched": len(matched),
        "match_rate": len(matched) / len(injury_raw) if len(injury_raw) else float("nan"),
        "n_total_player_games": len(player_log),
        "n_eligible": int(within_coverage.sum()),
        "eligible_share": float(within_coverage.mean()),
        "max_game_date_retained": eligible["GAME_DATE"].max(),
        "pct_positive_1game": float(eligible["y_1game"].mean()),
        "pct_positive_3game": float(eligible["y_3game"].mean()),
        "pct_positive_10game": float(eligible["y_10game"].mean()),
    }
    return eligible, diagnostics, intervals
