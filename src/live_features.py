"""Assemble a single player's current feature row — the live analog of the
batch pipeline in src/feature_engineering.py, computed "as of today" rather
than as of a specific historical game (see project plan: decision #2).

Reuses the existing feature functions unchanged wherever "as of today"
means the same thing as "as of the next real game" (workload, extended
absence, injury history, bio, travel). Adapts only where it genuinely
doesn't (rest days, workload density) — each function below explains why.

Entry point: assemble_live_features(player_id, ref, as_of=None).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, TypeVar

import numpy as np
import pandas as pd

from src.data_loader import season_string, get_player_current_info, get_team_advanced_stats
from src.feature_engineering import (
    LONG_LAYOFF_DAYS,
    compute_workload_features,
    compute_extended_absence_features,
    compute_injury_history_features,
    compute_bio_features,
    compute_travel_features,
)
from src.live_reference_data import LiveReferenceData
from src.training import FEATURE_COLUMNS


class PlayerNotFoundError(Exception):
    """No player exists for the given player_id."""


class NoCurrentTeamError(Exception):
    """Player has no current NBA team (free agent, retired, inactive) —
    blocks the team-based features (PACE/OFF/DEF + all travel features) in
    a way training never saw for an isolated row, so this is refused
    rather than predicted through. See project plan's error table."""


class NbaApiUnavailableError(Exception):
    """A live NBA Stats API call failed or timed out."""


T = TypeVar("T")


def _retry(fn: Callable[[], T], attempts: int = 2, backoff_sec: float = 1.5) -> T:
    """Call fn() up to `attempts` times, sleeping backoff_sec between
    attempts, re-raising the last exception if every attempt fails. This is
    an unofficial, unauthenticated API (stats.nba.com) — connection resets
    and timeouts under load are routine, not exceptional, so one retry
    before treating a call as genuinely unavailable matches the project
    plan's error-handling design (503 NBA_API_UNAVAILABLE only after a
    retry, not on the first hiccup).
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if attempt < attempts - 1:
                time.sleep(backoff_sec)
    raise last_exc


# ---------------------------------------------------------------------------
# Season resolution — a calendar heuristic, not an API call. NBA seasons
# nominally start in October, but team/roster activity (and what most
# people mean by "this season") shifts by around August. Getting the exact
# month threshold slightly wrong costs nothing: every live puller below
# just falls back to the previous season if the nominal-current one has too
# few games, which is always true immediately after this heuristic flips
# for a season that hasn't tipped off yet — exactly the situation right now
# (2026-08-17 is deep 2026 offseason), so this path is genuinely exercised,
# not just theoretical.
# ---------------------------------------------------------------------------

def _nominal_start_year(as_of: pd.Timestamp) -> int:
    return as_of.year if as_of.month >= 8 else as_of.year - 1


def resolve_reference_seasons(as_of: pd.Timestamp, max_lookback: int = 4) -> list[str]:
    """Newest-first season strings to try, e.g. as_of=2026-08-17 ->
    ["2026-27", "2025-26", "2024-25", "2023-24"]."""
    start_year = _nominal_start_year(as_of)
    return [season_string(start_year - i) for i in range(max_lookback)]


def current_season_id(as_of: pd.Timestamp) -> str:
    """SEASON_ID form ('22026') of the same nominal season, matching the
    convention feature_engineering.py uses throughout (vs. data_loader's
    season_string's '2026-27' form, used for API calls)."""
    return f"2{_nominal_start_year(as_of)}"


def _season_string_to_id(season: str) -> str:
    """'2025-26' -> '22025'."""
    return f"2{season[:4]}"


# ---------------------------------------------------------------------------
# Live data pulls
# ---------------------------------------------------------------------------

def fetch_live_player_bio(player_id: int) -> dict:
    """Bio + CURRENT team in one call (see data_loader.get_player_current_info).
    Raises PlayerNotFoundError for an invalid player_id (empty response)."""
    try:
        info = _retry(lambda: get_player_current_info(player_id))
    except Exception as e:
        raise NbaApiUnavailableError(f"commonplayerinfo failed for player_id={player_id}: {e}") from e

    if len(info) == 0:
        raise PlayerNotFoundError(f"No player found for player_id={player_id}")

    row = info.iloc[0]
    team_id = int(row["TEAM_ID"]) if pd.notna(row["TEAM_ID"]) and row["TEAM_ID"] else None
    return {
        "player_id": int(row["PERSON_ID"]),
        "full_name": row["DISPLAY_FIRST_LAST"],
        "birthdate": pd.Timestamp(row["BIRTHDATE"]),
        "position": row["POSITION"] or "",
        "height": row["HEIGHT"],
        "weight": float(row["WEIGHT"]),  # CommonPlayerInfo returns this as a string, unlike
                                          # the batch pipeline's bio (numeric after pd.read_csv)
        "from_year": int(row["FROM_YEAR"]) if pd.notna(row["FROM_YEAR"]) else None,
        "team_id": team_id,
        "team_abbreviation": row["TEAM_ABBREVIATION"] or None,
        "team_name": row["TEAM_NAME"] or None,
        "roster_status": row["ROSTERSTATUS"],
    }


def fetch_live_player_game_log(player_id: int, as_of: pd.Timestamp, min_games: int = 20,
                                max_lookback: int = 4) -> pd.DataFrame:
    """Player's own recent games, newest-season-first, stopping once
    min_games real rows are accumulated (or lookback is exhausted — a
    genuine rookie may have fewer than min_games ever). Only games
    STRICTLY BEFORE as_of are kept — the same shift(1) principle every
    batch rolling feature already uses (never include the row's own
    game), which also means assemble_live_features(..., as_of=D)
    reproduces the exact game set a features.csv row for game date D was
    computed from. This matters for real "today" use too (a same-day
    already-played game shouldn't count as "prior" workload yet), and is
    essential for notebooks/21's historical-as_of validation, since the
    live API has no "as of a past date" parameter — it always returns
    everything up to the real present.

    Uses playergamelog.PlayerGameLog (one player, one season per call)
    rather than data_loader.get_player_game_log (whole league per call) —
    confirmed via a live smoke test this session that its real response
    already matches its declared schema exactly: SEASON_ID/GAME_DATE/MIN
    match the batch convention already; only Player_ID/Game_ID need
    renaming to PLAYER_ID/GAME_ID.
    """
    from nba_api.stats.endpoints import playergamelog

    as_of_date = as_of.normalize()
    frames, total = [], 0
    for season in resolve_reference_seasons(as_of, max_lookback):
        try:
            df = _retry(lambda s=season: playergamelog.PlayerGameLog(
                player_id=player_id, season=s, timeout=15
            ).get_data_frames()[0])
        except Exception as e:
            raise NbaApiUnavailableError(
                f"playergamelog failed for player_id={player_id}, season={season}: {e}"
            ) from e
        if len(df) == 0:
            continue
        df = df.rename(columns={"Player_ID": "PLAYER_ID", "Game_ID": "GAME_ID"})
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df["SEASON_ID"] = df["SEASON_ID"].astype(str)
        df = df[df["GAME_DATE"] < as_of_date]
        if len(df) == 0:
            continue
        frames.append(df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN"]])
        total += len(df)
        if total >= min_games:
            break

    if not frames:
        return pd.DataFrame(columns=["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN"])
    return pd.concat(frames, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)


def fetch_live_team_game_log(team_id: int, team_abbreviation: str, as_of: pd.Timestamp,
                              min_games: int = 14, max_lookback: int = 4) -> pd.DataFrame:
    """Team's recent schedule — same season-fallback/strictly-before-as_of
    truncation pattern as fetch_live_player_game_log, via teamgamelog.TeamGameLog
    (one team, one season per call) rather than data_loader.get_team_game_log
    (whole league per call). TeamGameLog doesn't return SEASON_ID or
    TEAM_ABBREVIATION — both synthesized here from context the caller
    already has (the season being requested; the abbreviation from the
    player's bio call), confirmed via live smoke test this session.
    """
    from nba_api.stats.endpoints import teamgamelog

    as_of_date = as_of.normalize()
    frames, total = [], 0
    for season in resolve_reference_seasons(as_of, max_lookback):
        try:
            df = _retry(lambda s=season: teamgamelog.TeamGameLog(
                team_id=team_id, season=s, timeout=15
            ).get_data_frames()[0])
        except Exception as e:
            raise NbaApiUnavailableError(
                f"teamgamelog failed for team_id={team_id}, season={season}: {e}"
            ) from e
        if len(df) == 0:
            continue
        df = df.rename(columns={"Team_ID": "TEAM_ID", "Game_ID": "GAME_ID"})
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], format="%b %d, %Y")
        df["SEASON_ID"] = _season_string_to_id(season)
        df["TEAM_ABBREVIATION"] = team_abbreviation
        df = df[df["GAME_DATE"] < as_of_date]
        if len(df) == 0:
            continue
        frames.append(df[["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "SEASON_ID", "MATCHUP"]])
        total += len(df)
        if total >= min_games:
            break

    if not frames:
        return pd.DataFrame(columns=["TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "SEASON_ID", "MATCHUP"])
    return pd.concat(frames, ignore_index=True).sort_values("GAME_DATE").reset_index(drop=True)


def fetch_live_team_advanced_stats(team_id: int, as_of: pd.Timestamp, max_lookback: int = 4) -> dict | None:
    """PACE/OFF_RATING/DEF_RATING for the player's OWN team (confirmed via
    notebooks/04_build_features.py that this is never the opponent's),
    current season with fallback. None if no lookback season has a row for
    this team_id at all (shouldn't happen for a real active team)."""
    for season in resolve_reference_seasons(as_of, max_lookback):
        try:
            stats = _retry(lambda s=season: get_team_advanced_stats(s))
        except Exception as e:
            raise NbaApiUnavailableError(f"leaguedashteamstats failed for season={season}: {e}") from e
        row = stats[stats["TEAM_ID"] == team_id]
        if len(row):
            r = row.iloc[0]
            return {"PACE": float(r["PACE"]), "OFF_RATING": float(r["OFF_RATING"]), "DEF_RATING": float(r["DEF_RATING"])}
    return None


# ---------------------------------------------------------------------------
# Live feature computation — one function per FEATURE_COLUMNS group
# ---------------------------------------------------------------------------

def compute_live_rest_features(player_game_log: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Adaptation needed, but a narrower one than it first looks:
    compute_rest_features groups by (PLAYER_ID, SEASON_ID) and diffs
    within-group, giving NaN for a season-opener (no real prior game *in
    that season*) — that's not a training-data quirk to route around, it's
    a deliberate modeling choice (see compute_rest_features's own
    docstring: a multi-month offseason gap isn't meaningful "rest," and the
    model was never trained on cross-season day-counts for this feature —
    every real days_rest value it saw was a within-season gap, however
    large). Feeding it a raw 623-day cross-season gap live would be a
    value far outside anything it learned from, not "more information."

    So this checks whether the player's last logged game falls in the
    SAME nominal season as as_of (current_season_id): if yes, compute a
    real within-season gap exactly as before; if the last game was an
    earlier season (or there is none), return NaN/False/False — the same
    "unknown, not meaningful rest" treatment as a real season-opener row.
    The season-crossing case doesn't go unrepresented: that's exactly what
    games_since_extended_return / is_returning_from_extended_absence
    exist for (see compute_live_extended_absence_features).
    """
    if len(player_game_log) == 0:
        return {"days_rest": np.nan, "is_back_to_back": False, "is_long_layoff": False}

    last_idx = player_game_log["GAME_DATE"].idxmax()
    last_game_date = player_game_log.loc[last_idx, "GAME_DATE"]
    last_game_season = str(player_game_log.loc[last_idx, "SEASON_ID"])

    if last_game_season != current_season_id(as_of):
        return {"days_rest": np.nan, "is_back_to_back": False, "is_long_layoff": False}

    # last_game_date < as_of is guaranteed (player_game_log is pre-filtered
    # to strictly before as_of — see fetch_live_player_game_log), so
    # days_rest is provably >= 0; the floor is kept as a cheap, explicit
    # invariant rather than a silent assumption on the caller's filtering.
    days_rest = max(int((as_of.normalize() - last_game_date).days) - 1, 0)
    return {
        "days_rest": float(days_rest),
        "is_back_to_back": days_rest == 0,
        "is_long_layoff": days_rest > LONG_LAYOFF_DAYS,
    }


def compute_live_workload_features(player_game_log: pd.DataFrame, as_of: pd.Timestamp,
                                    density_window_days: int = 14) -> dict:
    """min_avg_acute/min_avg_chronic/acute_chronic_ratio/games_last_14d:
    same synthetic-row trick as compute_live_bio_features and
    compute_live_injury_history_features — append one row dated as_of to
    player_game_log and call compute_workload_features (unchanged) on the
    result, reading the synthetic row's values.

    This isn't optional the way it looks for bio/injury-history: reading
    compute_workload_features's LAST REAL row (an earlier version of this
    function did) is off by one game, because that function's shift(1)
    already looks backward from each row — the last real row's own
    rolling average already excludes ITSELF, which is exactly wrong for
    "as of right now" (which should include it, since it already
    happened). Confirmed by notebooks/21 against real players: two
    players' min_avg_acute/min_avg_chronic were off by exactly one game's
    worth of minutes before this fix.

    The synthetic row also reproduces games_last_14d's real behavior
    exactly — including a subtlety worth naming: pandas' time-based
    rolling count returns NaN, not 0, for a window with literally no data
    in it (e.g. a player who hasn't played in a very long time), and that
    NaN is present in the actual training data the same way (no .fillna(0)
    in compute_workload_features's density column, unlike the analogous
    travel feature). Recomputing this by hand — an earlier version of this
    function did — always returns a real number and quietly feeds the
    model an input shape it never saw for that scenario.

    Also returns games_so_far and personal_avg_min (mean MIN over the REAL
    pulled log, excluding the synthetic row) — not FEATURE_COLUMNS
    themselves, but needed by apply_live_cohort_backfill.
    """
    if len(player_game_log) == 0:
        return {
            "min_avg_acute": np.nan, "min_avg_chronic": np.nan, "acute_chronic_ratio": np.nan,
            "games_last_14d": np.nan, "games_so_far": 0, "personal_avg_min": np.nan,
        }

    synthetic = pd.DataFrame({
        "PLAYER_ID": [player_game_log["PLAYER_ID"].iloc[0]], "GAME_ID": ["9999999999"],
        "GAME_DATE": [as_of.normalize()], "MIN": [np.nan],
    })
    extended = pd.concat([player_game_log, synthetic], ignore_index=True)
    workload = compute_workload_features(extended)
    last = workload.iloc[-1]
    density_col = f"games_last_{density_window_days}d"

    return {
        "min_avg_acute": last["min_avg_acute"],
        "min_avg_chronic": last["min_avg_chronic"],
        "acute_chronic_ratio": last["acute_chronic_ratio"],
        "games_last_14d": last[density_col],
        "games_so_far": len(player_game_log),
        "personal_avg_min": player_game_log["MIN"].mean(),
    }


def classify_live_cohort(position: str, bmi: float, bmi_tercile_edges: np.ndarray) -> str:
    """Live analog of feature_engineering.compute_cohort for one player:
    same position priority (center > forward > guard) and BMI-tercile
    bucketing, using the FROZEN tercile edges from the snapshot (fit on
    train data) rather than refitting qcut on a population of one. Outer
    edges are opened to +/-inf so a real BMI outside the historical
    training range still gets bucketed rather than falling out as NaN.
    """
    position = position or ""
    if "Center" in position:
        primary_position = "center"
    elif "Forward" in position:
        primary_position = "forward"
    else:
        primary_position = "guard"

    edges = np.asarray(bmi_tercile_edges, dtype=float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    tercile = pd.cut([bmi], bins=edges, labels=["low_bmi", "mid_bmi", "high_bmi"], include_lowest=True)[0]
    return f"{primary_position}_{tercile}"


def apply_live_cohort_backfill(workload: dict, games_so_far: int, personal_avg_min: float,
                                cohort_label: str, ref: LiveReferenceData,
                                acute_window: int = 3, chronic_window: int = 15) -> dict:
    """Live analog of feature_engineering.apply_cohort_backfill's blend,
    for one player's current values. Only ever changes min_avg_acute/
    min_avg_chronic/acute_chronic_ratio where the direct computation came
    back NaN — same .fillna() semantics as the batch version, so this is a
    no-op for any player with enough of their own history.

    personal_avg_min mirrors the batch pipeline's personal_expanding_avg
    (mean of strictly-prior MIN). For live use this is simply the mean MIN
    over whatever was pulled, since every pulled game is already strictly
    before as_of by construction. This function's output only ever matters when
    games_so_far < chronic_window — exactly the population for whom the
    live pull already captured their ENTIRE available history (a player
    with fewer than 15 career games can't have more history to miss), so
    this is not an approximation of the batch version, it's the same thing.
    """
    cohort_baseline = ref.cohort_baseline_map.get(cohort_label, ref.overall_baseline)
    if pd.isna(cohort_baseline):
        cohort_baseline = ref.overall_baseline

    def blended(window: int) -> float:
        weight = min(max(games_so_far / window, 0.0), 1.0)
        personal = personal_avg_min if pd.notna(personal_avg_min) else cohort_baseline
        return weight * personal + (1 - weight) * cohort_baseline

    result = dict(workload)
    used_backfill = False
    if pd.isna(result.get("min_avg_acute")):
        result["min_avg_acute"] = blended(acute_window)
        used_backfill = True
    if pd.isna(result.get("min_avg_chronic")):
        result["min_avg_chronic"] = blended(chronic_window)
        used_backfill = True
    if pd.isna(result.get("acute_chronic_ratio")):
        chronic = result["min_avg_chronic"]
        result["acute_chronic_ratio"] = (result["min_avg_acute"] / chronic) if chronic else np.nan

    result["cohort_backfill_used"] = used_backfill
    return result


def compute_live_extended_absence_features(player_game_log: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Same synthetic-row trick as compute_live_workload_features, for the
    same underlying reason, not the "purely as of the last row" version an
    earlier revision of this function used. is_returning_from_extended_absence
    is a flag on the RETURN game's own row — but player_game_log only
    contains games strictly before as_of (see fetch_live_player_game_log),
    so the return game itself (if as_of IS the return) is never in it, and
    reading the last REAL row instead answers "was the player's previous
    game a return from a gap," which is a different, generally False,
    question. Confirmed by notebooks/21: a real player whose batch row had
    is_returning_from_extended_absence=True came back False before this fix.
    The synthetic row needs SEASON_ID (current_season_id(as_of)) — the one
    thing this function's season-gap detection actually keys on.
    """
    if len(player_game_log) == 0:
        return {"games_since_extended_return": np.nan, "is_returning_from_extended_absence": False}

    synthetic = pd.DataFrame({
        "PLAYER_ID": [player_game_log["PLAYER_ID"].iloc[0]], "GAME_ID": ["9999999999"],
        "GAME_DATE": [as_of.normalize()], "SEASON_ID": [current_season_id(as_of)],
    })
    extended = pd.concat([player_game_log, synthetic], ignore_index=True)
    result = compute_extended_absence_features(extended)
    last = result.iloc[-1]
    return {
        "games_since_extended_return": last["games_since_extended_return"],
        "is_returning_from_extended_absence": bool(last["is_returning_from_extended_absence"]),
    }


def compute_live_injury_history_features(player_id: int, as_of: pd.Timestamp, ref: LiveReferenceData,
                                          lookback_days: int = 365) -> dict:
    """compute_injury_history_features already accepts arbitrary game_rows
    with a GAME_DATE column — it's already "as of an arbitrary date" by
    design (no adaptation needed beyond building one synthetic row)."""
    synthetic = pd.DataFrame({
        "PLAYER_ID": [player_id], "GAME_ID": ["9999999999"], "GAME_DATE": [as_of.normalize()],
    })
    result = compute_injury_history_features(ref.intervals, synthetic, lookback_days=lookback_days)
    row = result.iloc[0]
    return {
        "career_injury_count": int(row["career_injury_count"]),
        "days_missed_last_365d": float(row[f"days_missed_last_{lookback_days}d"]),
    }


def compute_live_bio_features(bio_raw: dict, as_of: pd.Timestamp, season_id: str) -> dict:
    """Same synthetic-row trick against compute_bio_features unchanged —
    age_years/experience_years fall out correctly as-of `as_of` for free."""
    bio_df = pd.DataFrame([{
        "PERSON_ID": bio_raw["player_id"], "HEIGHT": bio_raw["height"], "WEIGHT": bio_raw["weight"],
        "POSITION": bio_raw["position"], "BIRTHDATE": bio_raw["birthdate"], "FROM_YEAR": bio_raw["from_year"],
    }])
    game_rows = pd.DataFrame({
        "PLAYER_ID": [bio_raw["player_id"]], "GAME_ID": ["9999999999"],
        "GAME_DATE": [as_of.normalize()], "SEASON_ID": [season_id],
    })
    row = compute_bio_features(bio_df, game_rows).iloc[0]
    return {
        "height_inches": row["height_inches"], "weight_lbs": row["weight_lbs"], "bmi": row["bmi"],
        "plays_guard": bool(row["plays_guard"]), "plays_forward": bool(row["plays_forward"]),
        "plays_center": bool(row["plays_center"]), "age_years": row["age_years"],
        "experience_years": row["experience_years"],
    }


def compute_live_travel_features(team_game_log: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Two of these four columns are fully recoverable as-of as_of; two are
    NOT, for a reason worth being explicit about rather than discovering
    via a silent mismatch (which is exactly how this was first found, in
    notebooks/21, against real team schedules):

    - travel_distance_last_game and current_road_trip_length are, in the
      batch definition, properties of the SPECIFIC game happening on a
      row's own date (distance traveled to arrive there; whether that
      game itself is home or away). Computing them "as of today" would
      require knowing where/whether the team's NEXT game is — exactly the
      schedule lookup decision #2 rules out. Reinterpreted here as "as of
      the team's last COMPLETED game" instead (the closest well-defined
      analog without that lookup) — travel_distance_last_game is read
      straight off the last real row; current_road_trip_length is the
      last real row's own pre-game streak PLUS one if that game was away
      (0 if it was home), i.e. the completed streak length right now.
      This will legitimately differ from a training-data row for the
      exact date of a real game, by design, not a bug.
    - travel_distance_last_14d and days_since_home do NOT have this
      problem — both are naturally about games strictly BEFORE the
      evaluation point already (compute_travel_features' rolling window
      is closed='left', and days_since_home is computed before updating
      "last home date" for the current row), so both are recomputed here
      anchored at as_of rather than at the last real game's date. Reading
      them off the last real row instead (an earlier version of this
      function did) silently shrinks/misdates the window by however many
      days have passed since that game — same class of bug as
      games_last_14d, same fix shape, just direct here instead of via a
      synthetic row (a synthetic "as_of" row would need to know is_home,
      which is exactly the unknown this function can't have).
    """
    if len(team_game_log) == 0:
        return {
            "travel_distance_last_game": np.nan, "travel_distance_last_14d": np.nan,
            "current_road_trip_length": np.nan, "days_since_home": np.nan,
        }

    result = compute_travel_features(team_game_log)
    last = result.iloc[-1]

    as_of_date = as_of.normalize()
    window_start = as_of_date - pd.Timedelta(days=14)
    in_window = (team_game_log["GAME_DATE"] >= window_start) & (team_game_log["GAME_DATE"] < as_of_date)
    travel_distance_last_14d = result.loc[in_window.values, "travel_distance_last_game"].sum()

    is_home = team_game_log["MATCHUP"].str.contains("vs.").values
    last_is_home = bool(is_home[-1])
    current_road_trip_length = 0 if last_is_home else int(last["current_road_trip_length"]) + 1

    home_dates = team_game_log.loc[is_home, "GAME_DATE"]
    days_since_home = float((as_of_date - home_dates.max()).days) if len(home_dates) else np.nan

    return {
        "travel_distance_last_game": last["travel_distance_last_game"],
        "travel_distance_last_14d": travel_distance_last_14d,
        "current_road_trip_length": current_road_trip_length,
        "days_since_home": days_since_home,
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@dataclass
class LiveFeatureResult:
    features: pd.DataFrame          # exactly one row, columns == FEATURE_COLUMNS
    as_of: pd.Timestamp
    player_id: int
    player_name: str
    current_team: dict
    warnings: list[dict]
    data_quality: dict


def assemble_live_features(
    player_id: int, ref: LiveReferenceData, as_of: pd.Timestamp | None = None,
    team_stats_fetcher: Callable[[int, pd.Timestamp], dict | None] | None = None,
) -> LiveFeatureResult:
    """The one function the API layer calls. as_of defaults to "now" but
    stays an explicit parameter — not exposed on the public HTTP contract
    (decision #2: always "today," no client-controlled time travel), but
    essential for offline testing: notebooks/21 calls this with historical
    as_of values to diff against known-good rows in features.csv.

    team_stats_fetcher defaults to fetch_live_team_advanced_stats (a
    live, uncached call) — overridable so the backend service can inject a
    TTL-cached version (see backend/state.py) without this module needing
    to know anything about caching, which is a service-layer concern, not
    a feature-pipeline one.

    Raises PlayerNotFoundError / NoCurrentTeamError / NbaApiUnavailableError.
    """
    as_of = (pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now()).normalize()
    team_stats_fetcher = team_stats_fetcher or fetch_live_team_advanced_stats

    bio_raw = fetch_live_player_bio(player_id)
    # CommonPlayerInfo keeps a retired player's LAST team on TEAM_ID
    # indefinitely (confirmed live: Kobe Bryant, retired 2016, still
    # returns TEAM_ID=1610612747/LAL) — team_id truthiness alone doesn't
    # mean "currently on a roster." ROSTERSTATUS is the field that
    # actually distinguishes them ('Active' vs 'Inactive', confirmed live
    # for the same case) and is why it's fetched at all; this was a real
    # bug caught via the browser smoke test, not a hypothetical.
    if not bio_raw["team_id"] or bio_raw["roster_status"] != "Active":
        raise NoCurrentTeamError(
            f"player_id={player_id} ({bio_raw['full_name']}) has no current team "
            f"(roster_status={bio_raw['roster_status']!r})"
        )
    team_id, team_abbreviation = bio_raw["team_id"], bio_raw["team_abbreviation"]

    player_game_log = fetch_live_player_game_log(player_id, as_of)
    team_game_log = fetch_live_team_game_log(team_id, team_abbreviation, as_of)

    rest = compute_live_rest_features(player_game_log, as_of)
    bio_features = compute_live_bio_features(bio_raw, as_of, current_season_id(as_of))

    workload_raw = compute_live_workload_features(player_game_log, as_of)
    games_so_far = workload_raw.pop("games_so_far")
    personal_avg_min = workload_raw.pop("personal_avg_min")
    cohort_label = classify_live_cohort(bio_raw["position"], bio_features["bmi"], ref.bmi_tercile_edges)
    workload = apply_live_cohort_backfill(workload_raw, games_so_far, personal_avg_min, cohort_label, ref)
    cohort_backfill_used = workload.pop("cohort_backfill_used")

    extended_absence = compute_live_extended_absence_features(player_game_log, as_of)
    injury_history = compute_live_injury_history_features(player_id, as_of, ref)

    team_stats = team_stats_fetcher(team_id, as_of)
    if team_stats is None:
        raise NbaApiUnavailableError(f"no team advanced stats found for team_id={team_id} in any lookback season")

    travel = compute_live_travel_features(team_game_log, as_of)

    row = {**rest, **team_stats, **bio_features, **workload, **extended_absence, **injury_history, **travel}
    features = pd.DataFrame([row])[FEATURE_COLUMNS]

    warnings = []
    if cohort_backfill_used:
        warnings.append({"code": "COHORT_BACKFILL_USED",
                          "message": "Not enough recent games for a personal workload average; blended with a cohort baseline."})
    if extended_absence["is_returning_from_extended_absence"]:
        warnings.append({"code": "EXTENDED_ABSENCE",
                          "message": "This player is returning from a gap of a full season or more."})
    if games_so_far == 0:
        warnings.append({"code": "NO_GAME_HISTORY", "message": "No prior games found in the pulled window."})

    data_quality = {
        "last_game_played": player_game_log["GAME_DATE"].max().date().isoformat() if len(player_game_log) else None,
        "days_since_last_game": rest["days_rest"],
        "games_used_for_workload": int(games_so_far),
        "cohort_backfill_used": cohort_backfill_used,
        "extended_absence_return": bool(extended_absence["is_returning_from_extended_absence"]),
    }

    return LiveFeatureResult(
        features=features,
        as_of=as_of,
        player_id=player_id,
        player_name=bio_raw["full_name"],
        current_team={"team_id": team_id, "abbreviation": team_abbreviation, "full_name": bio_raw["team_name"]},
        warnings=warnings,
        data_quality=data_quality,
    )
