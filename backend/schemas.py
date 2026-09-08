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
    recent_injury_checks: int
    recent_injury_confirmed: int
    # 0-1: share of this player's missed games since base_coverage_end that
    # an official injury report actually covered. None when no supplement
    # has been built, or the player had no missed games to resolve.
    injury_data_confidence: float | None


class Caveat(BaseModel):
    code: str
    message: str


class RiskResponse(BaseModel):
    player: PlayerSummary
    current_team: TeamSummary
    as_of: str
    # How current the injury history is. injury_history_as_of is the real,
    # effective coverage date (advanced by notebooks/22's report-derived
    # backfill); injury_history_source_as_of is the original Kaggle
    # dataset's own cutoff, kept separate so "how fresh" and "how much came
    # from the primary source" stay independently answerable rather than
    # one silently standing in for the other.
    injury_history_as_of: str
    injury_history_source_as_of: str | None
    # A span neither source covers (the Kaggle data ends before it; the
    # NBA's report CDN no longer retains it). Real and permanent when
    # present — reported rather than hidden behind the advanced
    # injury_history_as_of date.
    injury_history_gap_start: str | None
    injury_history_gap_end: str | None
    predictions: Predictions
    data_quality: DataQuality
    caveats: list[Caveat]


class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    reference_players: int
    injury_history_as_of: str
    injury_history_source_as_of: str | None
    injury_supplement_built_at: str | None
    injury_supplement_confidence: float | None


class SeasonStats(BaseModel):
    season: str
    games_played: int
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float


class CareerStats(BaseModel):
    seasons_played: int
    games_played: int
    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    tov: float


class PlayerStatsResponse(BaseModel):
    player: PlayerSummary
    season: SeasonStats | None
    career: CareerStats
