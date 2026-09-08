"""GET /players/{player_id}/stats — season and career per-game averages
(points/rebounds/assists/steals/blocks/turnovers), a thin passthrough of
NBA API numbers for display. Deliberately independent of the injury-risk
pipeline: no AppState models/reference needed, and no current-team
requirement — a retired player's career averages are perfectly valid to
show even though /risk correctly refuses to predict for them.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.data_loader import NbaApiUnavailableError
from src.player_stats import fetch_player_stats, PlayerStatsNotFoundError

from backend.schemas import CareerStats, PlayerStatsResponse, PlayerSummary, SeasonStats
from backend.state import AppState, get_state

router = APIRouter()


@router.get("/players/{player_id}/stats", response_model=PlayerStatsResponse)
def get_player_stats(player_id: int, state: AppState = Depends(get_state)) -> PlayerStatsResponse:
    # Static table lookup first: gives a fast 404 for a bogus id without
    # spending an NBA API call, and PlayerCareerStats' result sets don't
    # carry the player's name at all, so this is also where full_name
    # comes from for the response.
    player_row = state.static_players[state.static_players["id"] == player_id]
    if len(player_row) == 0:
        raise HTTPException(status_code=404, detail={
            "error_code": "PLAYER_NOT_FOUND", "message": f"No player found for player_id={player_id}",
            "player_id": player_id,
        })
    row = player_row.iloc[0]

    try:
        result = fetch_player_stats(player_id)
    except PlayerStatsNotFoundError as e:
        raise HTTPException(status_code=404, detail={
            "error_code": "PLAYER_STATS_NOT_FOUND", "message": str(e), "player_id": player_id,
        })
    except NbaApiUnavailableError as e:
        raise HTTPException(status_code=503, detail={
            "error_code": "NBA_API_UNAVAILABLE", "message": str(e), "player_id": player_id,
        })

    return PlayerStatsResponse(
        player=PlayerSummary(player_id=player_id, full_name=row["full_name"], is_active=bool(row["is_active"])),
        season=SeasonStats(**result["season"]) if result["season"] else None,
        career=CareerStats(**result["career"]),
    )
