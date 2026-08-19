"""Load raw data from the Kaggle injury dataset and the NBA Stats API."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


# ---------------------------------------------------------------------------
# Kaggle injury dataset
# ---------------------------------------------------------------------------

def load_kaggle_injury_data(dataset_slug: str, file_name: str | None = None) -> pd.DataFrame:
    """Download (or reuse cached) a Kaggle dataset via kagglehub and load it.

    dataset_slug: the "owner/dataset-name" part of the Kaggle dataset URL,
        e.g. "loganlauton/nba-injury-stats-1951-2023".
    file_name: which CSV to load if the dataset has multiple files. If not
        given and the dataset has exactly one CSV, that one is used.
    """
    import kagglehub

    download_path = Path(kagglehub.dataset_download(dataset_slug))
    csv_files = sorted(download_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset at {download_path}")

    if file_name:
        match = next((f for f in csv_files if f.name == file_name), None)
        if match is None:
            available = ", ".join(f.name for f in csv_files)
            raise FileNotFoundError(f"'{file_name}' not found. Available files: {available}")
        return pd.read_csv(match)

    if len(csv_files) > 1:
        available = ", ".join(f.name for f in csv_files)
        print(f"Multiple CSVs found ({available}); loading '{csv_files[0].name}'. "
              f"Pass file_name= to pick a different one.")
    return pd.read_csv(csv_files[0])


def load_injury_csv(path: str | Path = RAW_DIR / "injuries.csv") -> pd.DataFrame:
    """Load a manually-downloaded Kaggle CSV (dropped into data/raw/) directly."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# NBA Stats API (via nba_api)
# ---------------------------------------------------------------------------

def season_string(start_year: int) -> str:
    """2016 -> '2016-17', matching nba_api's SEASON format."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _get_league_game_log(season: str, grain: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """grain: 'P' for one row per player per game, 'T' for one row per team per game."""
    from nba_api.stats.endpoints import leaguegamelog

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
        player_or_team_abbreviation=grain,
        timeout=60,
    )
    return log.get_data_frames()[0]


def get_player_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per player per game for a single season, e.g. season='2023-24'.

    Uses LeagueGameLog with player_or_team_abbreviation='P', which pulls the
    whole league in one call instead of looping per player.
    """
    return _get_league_game_log(season, "P", season_type)


def get_team_game_log(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per team per game for a single season — the full schedule
    (unlike the player log, this has no "did they play" gaps), used to work
    out a team's next N games from any given date.
    """
    return _get_league_game_log(season, "T", season_type)


def _fetch_and_cache_game_logs(
    seasons: list[str],
    grain: str,
    file_prefix: str,
    out_dir: Path | str = RAW_DIR,
    season_type: str = "Regular Season",
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    id_dtypes = {"GAME_ID": str, "SEASON_ID": str}  # zero-padded in the API; don't let read_csv strip leading zeros

    frames = []
    for season in seasons:
        cache_path = out_dir / f"{file_prefix}_{season}.csv"
        if cache_path.exists():
            frames.append(pd.read_csv(cache_path, dtype=id_dtypes))
            continue

        df = _get_league_game_log(season, grain, season_type)
        df.to_csv(cache_path, index=False)
        frames.append(df)
        time.sleep(sleep_sec)  # be polite to the (unofficial, unauthenticated) API

    return pd.concat(frames, ignore_index=True)


def fetch_and_cache_player_game_logs(
    seasons: list[str],
    out_dir: Path | str = RAW_DIR,
    season_type: str = "Regular Season",
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """Pull player game logs for each season, caching one CSV per season in
    out_dir so re-runs don't re-hit the NBA Stats API. Returns the concatenation.
    """
    return _fetch_and_cache_game_logs(
        seasons, "P", "player_game_log", out_dir, season_type, sleep_sec
    )


def fetch_and_cache_team_game_logs(
    seasons: list[str],
    out_dir: Path | str = RAW_DIR,
    season_type: str = "Regular Season",
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """Pull team game logs (full schedules) for each season, caching one CSV
    per season in out_dir. Returns the concatenation.
    """
    return _fetch_and_cache_game_logs(
        seasons, "T", "team_game_log", out_dir, season_type, sleep_sec
    )


def get_team_advanced_stats(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per team for a season: pace, offensive/defensive rating, etc.
    Season-level average (not per-game) — bulk call, all 30 teams at once.
    """
    from nba_api.stats.endpoints import leaguedashteamstats

    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense="Advanced",
        timeout=60,
    )
    return stats.get_data_frames()[0]


def fetch_and_cache_team_advanced_stats(
    seasons: list[str],
    out_dir: Path | str = RAW_DIR,
    season_type: str = "Regular Season",
    sleep_sec: float = 0.6,
) -> pd.DataFrame:
    """Pull team advanced stats (pace etc.) for each season, caching one CSV
    per season in out_dir. Returns the concatenation with a SEASON column added.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for season in seasons:
        cache_path = out_dir / f"team_advanced_stats_{season}.csv"
        if cache_path.exists():
            df = pd.read_csv(cache_path)
        else:
            df = get_team_advanced_stats(season, season_type)
            df.to_csv(cache_path, index=False)
            time.sleep(sleep_sec)
        df = df.copy()
        df["SEASON"] = season
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def get_player_bio(player_id: int) -> pd.DataFrame:
    """Bio info for one player: birthdate, position, height, weight, rookie
    season, draft year. No bulk endpoint for this — one API call per player.

    Deliberately does NOT keep SEASON_EXP: it's the player's career-total
    experience as of today, not as of any historical game, so it's wrong for
    every row except ones from the current season. FROM_YEAR (rookie season)
    is what lets feature_engineering compute experience-as-of-game-date
    correctly, and unlike DRAFT_YEAR it's populated for undrafted players too
    (~31% of this dataset's players went undrafted).
    """
    from nba_api.stats.endpoints import commonplayerinfo

    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30).get_data_frames()[0]
    return info[["PERSON_ID", "DISPLAY_FIRST_LAST", "BIRTHDATE", "POSITION", "HEIGHT", "WEIGHT",
                 "FROM_YEAR", "DRAFT_YEAR"]]


def get_player_current_info(player_id: int) -> pd.DataFrame:
    """Same CommonPlayerInfo call as get_player_bio(), widened to also
    include the player's CURRENT team and roster status. Needed for live
    single-player inference (current-team join for PACE/OFF/DEF and travel
    features), not by the batch pipeline — kept as a separate function
    rather than widening get_player_bio() itself, since notebooks 04/11
    already depend on that function's narrower column selection. Empty
    frame (0 rows) means an invalid player_id — callers decide how to
    handle that, this just returns whatever the API gives back.
    """
    from nba_api.stats.endpoints import commonplayerinfo

    info = commonplayerinfo.CommonPlayerInfo(player_id=player_id, timeout=30).get_data_frames()[0]
    return info[["PERSON_ID", "DISPLAY_FIRST_LAST", "BIRTHDATE", "POSITION", "HEIGHT", "WEIGHT",
                 "FROM_YEAR", "DRAFT_YEAR", "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "ROSTERSTATUS"]]


def fetch_and_cache_player_bios(
    player_ids: list[int],
    out_dir: Path | str = RAW_DIR,
    sleep_sec: float = 0.6,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    """Pull bio info for each player_id, one API call per player. Caches
    incrementally to a single CSV (checkpointed every `checkpoint_every`
    players) so an interrupted run resumes without re-fetching players
    already pulled, instead of redoing the whole ~1300-call pull from scratch.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "player_bio.csv"

    cached = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    have_ids = set(cached["PERSON_ID"]) if len(cached) else set()
    missing = [pid for pid in player_ids if pid not in have_ids]
    print(f"player bios: {len(have_ids)} cached, {len(missing)} to fetch")

    rows = [cached] if len(cached) else []
    failed = []
    for i, pid in enumerate(missing):
        try:
            rows.append(get_player_bio(pid))
        except Exception as e:
            print(f"  failed for player_id={pid}: {e}")
            failed.append(pid)
            continue
        time.sleep(sleep_sec)
        if (i + 1) % checkpoint_every == 0:
            pd.concat(rows, ignore_index=True).to_csv(cache_path, index=False)
            print(f"  ...{i + 1}/{len(missing)} fetched, checkpointed")

    result = pd.concat(rows, ignore_index=True) if rows else cached
    result.to_csv(cache_path, index=False)
    if failed:
        print(f"{len(failed)} players failed and were skipped: {failed}")
    return result


def load_processed_labels(path: Path | str = None) -> pd.DataFrame:
    """Load data/processed/game_labels.csv with GAME_ID/SEASON_ID forced to
    string (see the id_dtypes note in _fetch_and_cache_game_logs — same
    leading-zero issue applies here since this file was built by joining
    against those game logs).
    """
    path = Path(path) if path else RAW_DIR.parent / "processed" / "game_labels.csv"
    return pd.read_csv(path, dtype={"GAME_ID": str, "SEASON_ID": str})


def get_static_players() -> pd.DataFrame:
    """Static reference table: player_id, full_name, is_active, etc."""
    from nba_api.stats.static import players

    return pd.DataFrame(players.get_players())


def get_static_teams() -> pd.DataFrame:
    """Static reference table: team_id, full_name, abbreviation, etc."""
    from nba_api.stats.static import teams

    return pd.DataFrame(teams.get_teams())
