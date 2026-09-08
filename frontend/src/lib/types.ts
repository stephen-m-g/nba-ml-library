// Mirrors backend/schemas.py exactly — keep these two in sync by hand,
// there's no shared schema generation between the Python and TS sides yet.

export interface PlayerSummary {
  player_id: number;
  full_name: string;
  is_active: boolean;
}

export interface PlayersResponse {
  players: PlayerSummary[];
}

export interface TeamSummary {
  team_id: number;
  abbreviation: string;
  full_name: string;
}

export interface PredictionForLabel {
  probability: number;
  threshold: number;
  elevated_risk: boolean;
}

export interface Predictions {
  y_1game: PredictionForLabel;
  y_3game: PredictionForLabel;
  y_10game: PredictionForLabel;
}

export interface DataQuality {
  last_game_played: string | null;
  days_since_last_game: number | null;
  games_used_for_workload: number;
  cohort_backfill_used: boolean;
  extended_absence_return: boolean;
  recent_injury_checks: number;
  recent_injury_confirmed: number;
  injury_data_confidence: number | null;
}

export interface Caveat {
  code: string;
  message: string;
}

export interface RiskResponse {
  player: PlayerSummary;
  current_team: TeamSummary;
  as_of: string;
  predictions: Predictions;
  injury_history_as_of: string;
  injury_history_source_as_of: string | null;
  injury_history_gap_start: string | null;
  injury_history_gap_end: string | null;
  data_quality: DataQuality;
  caveats: Caveat[];
}

export interface ApiErrorDetail {
  error_code: string;
  message: string;
  player_id?: number;
}

export interface SeasonStats {
  season: string;
  games_played: number;
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  tov: number;
}

export interface CareerStats {
  seasons_played: number;
  games_played: number;
  pts: number;
  reb: number;
  ast: number;
  stl: number;
  blk: number;
  tov: number;
}

export interface PlayerStatsResponse {
  player: PlayerSummary;
  season: SeasonStats | null;
  career: CareerStats;
}
