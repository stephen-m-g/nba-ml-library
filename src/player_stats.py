"""Live season and career per-game stat averages for one player — points,
rebounds, assists, steals, blocks, turnovers. Deliberately a much thinner
module than src/live_features.py: this is a direct passthrough of NBA API
numbers for display, not a model feature pipeline, so there's no
leakage-avoidance or rolling-window logic to get right here.

One playercareerstats.PlayerCareerStats call (per_mode36="PerGame") gives
both a season-by-season breakdown and career totals already averaged
per-game by the API itself — no manual game-log summing needed, unlike
src/live_features.py's workload features (which need custom rolling logic
this endpoint can't provide).
"""

from __future__ import annotations

import pandas as pd

from src.data_loader import retry_api_call, NbaApiUnavailableError

STAT_COLUMNS = ["PTS", "REB", "AST", "STL", "BLK", "TOV"]


class PlayerStatsNotFoundError(Exception):
    """No career regular-season stats at all for this player_id (e.g. drafted but never played)."""


def _stats_from_row(row: pd.Series) -> dict:
    return {stat.lower(): float(row[stat]) for stat in STAT_COLUMNS}


def fetch_player_stats(player_id: int) -> dict:
    """Returns {"season": {...} | None, "career": {...}}. `season` is the
    most recent season with any games on record (by SEASON_ID, sorted —
    not assumed to already be in order) — for a currently-active player
    that's the current season; for a retired player, whatever their last
    season was. None only if the player has zero season rows at all despite
    having career totals (shouldn't happen in practice, handled anyway
    rather than assumed away). Each stats dict also carries games_played
    (and career carries seasons_played) so the frontend can show what the
    averages are actually over.
    """
    from nba_api.stats.endpoints import playercareerstats

    try:
        stats = retry_api_call(lambda: playercareerstats.PlayerCareerStats(
            player_id=player_id, per_mode36="PerGame", timeout=30
        ))
    except Exception as e:
        raise NbaApiUnavailableError(f"playercareerstats failed for player_id={player_id}: {e}") from e

    season_df = stats.season_totals_regular_season.get_data_frame()
    career_df = stats.career_totals_regular_season.get_data_frame()

    if len(career_df) == 0:
        raise PlayerStatsNotFoundError(f"No career regular-season stats for player_id={player_id}")

    career_row = career_df.iloc[0]
    career = {
        "seasons_played": int(len(season_df)),
        "games_played": int(career_row["GP"]),
        **_stats_from_row(career_row),
    }

    season = None
    if len(season_df) > 0:
        season_row = season_df.sort_values("SEASON_ID").iloc[-1]
        season = {
            "season": season_row["SEASON_ID"],
            "games_played": int(season_row["GP"]),
            **_stats_from_row(season_row),
        }

    return {"season": season, "career": career}
