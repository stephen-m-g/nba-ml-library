"""GET /players — served entirely from the cached static player table
(loaded once at startup, see backend/state.py), zero live NBA API calls
per request. Optional `q` filters by name so the same endpoint works both
for a "fetch once, filter client-side" frontend (today's plan) and a
future debounced server-side-filtered one, without a breaking change.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from backend.schemas import PlayerSummary, PlayersResponse
from backend.state import AppState, get_state

router = APIRouter()


@router.get("/players", response_model=PlayersResponse)
def list_players(
    q: str | None = Query(default=None, description="Case-insensitive substring filter on full_name"),
    active_only: bool = Query(default=False),
    state: AppState = Depends(get_state),
) -> PlayersResponse:
    df = state.static_players
    if active_only:
        df = df[df["is_active"]]
    if q:
        df = df[df["full_name"].str.contains(q, case=False, na=False, regex=False)]

    players = [
        PlayerSummary(player_id=int(row.id), full_name=row.full_name, is_active=bool(row.is_active))
        for row in df.itertuples()
    ]
    return PlayersResponse(players=players)
