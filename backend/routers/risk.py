"""GET /players/{player_id}/risk — the prediction endpoint. No
client-controlled `as_of` on the public contract: always "now," per
project plan decision #2 (the underlying src.live_features function keeps
as_of as a parameter purely for offline testing — see notebooks/21).
"""

from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from src.live_features import (
    assemble_live_features, PlayerNotFoundError, NoCurrentTeamError, NbaApiUnavailableError,
)
from src.predict import predict_elevated_risk

from backend.schemas import (
    RiskResponse, Predictions, PredictionForLabel, PlayerSummary, TeamSummary, DataQuality, Caveat,
)
from backend.state import AppState, get_state

router = APIRouter()


@router.get("/players/{player_id}/risk", response_model=RiskResponse)
def get_player_risk(player_id: int, state: AppState = Depends(get_state)) -> RiskResponse:
    as_of = pd.Timestamp.now().normalize()

    try:
        live_result = assemble_live_features(
            player_id, state.reference, as_of=as_of,
            team_stats_fetcher=state.get_team_stats_cached,
        )
    except PlayerNotFoundError as e:
        raise HTTPException(status_code=404, detail={
            "error_code": "PLAYER_NOT_FOUND", "message": str(e), "player_id": player_id,
        })
    except NoCurrentTeamError as e:
        raise HTTPException(status_code=422, detail={
            "error_code": "NO_CURRENT_TEAM", "message": str(e), "player_id": player_id,
        })
    except NbaApiUnavailableError as e:
        raise HTTPException(status_code=503, detail={
            "error_code": "NBA_API_UNAVAILABLE", "message": str(e), "player_id": player_id,
        })

    predictions = predict_elevated_risk(live_result.features, state.models)

    return RiskResponse(
        # every successful prediction required resolving a current team
        # (NoCurrentTeamError otherwise), so is_active is always true here
        player=PlayerSummary(player_id=live_result.player_id, full_name=live_result.player_name, is_active=True),
        current_team=TeamSummary(**live_result.current_team),
        as_of=live_result.as_of.date().isoformat(),
        predictions=Predictions(
            y_1game=PredictionForLabel(**predictions["y_1game"]),
            y_3game=PredictionForLabel(**predictions["y_3game"]),
            y_10game=PredictionForLabel(**predictions["y_10game"]),
        ),
        injury_history_as_of=state.reference.coverage_end.date().isoformat(),
        data_quality=DataQuality(**live_result.data_quality),
        caveats=[Caveat(**w) for w in live_result.warnings],
    )
