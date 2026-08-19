"""Pydantic request/response models — the API contract from the project
plan. `predictions` uses fixed y_1game/y_3game/y_10game fields (not a
generic dict) so the contract matches src.training.LABEL_COLUMNS exactly
and shows up explicitly in the auto-generated /docs schema.
"""

from __future__ import annotations

from pydantic import BaseModel


class PlayerSummary(BaseModel):
    player_id: int
    full_name: str
    is_active: bool


class PlayersResponse(BaseModel):
    players: list[PlayerSummary]


class TeamSummary(BaseModel):
    team_id: int
    abbreviation: str
    full_name: str


class PredictionForLabel(BaseModel):
    probability: float
    threshold: float
    elevated_risk: bool


class Predictions(BaseModel):
    y_1game: PredictionForLabel
    y_3game: PredictionForLabel
    y_10game: PredictionForLabel


class DataQuality(BaseModel):
    last_game_played: str | None
    days_since_last_game: float | None
    games_used_for_workload: int
    cohort_backfill_used: bool
    extended_absence_return: bool


class Caveat(BaseModel):
    code: str
    message: str


class RiskResponse(BaseModel):
    player: PlayerSummary
    current_team: TeamSummary
    as_of: str
    predictions: Predictions
    injury_history_as_of: str
    data_quality: DataQuality
    caveats: list[Caveat]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    reference_players: int
    injury_history_as_of: str
