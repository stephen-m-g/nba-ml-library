"""Environment-driven configuration — see project plan decision #4 (build
with eventual hosting in mind). Every path/setting has a sensible local-dev
default but comes from env vars, never a hardcoded absolute or Windows-only
path, so this can be reconfigured for a hosted environment without code
changes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    snapshot_path: Path = REPO_ROOT / "data" / "processed" / "live_reference_snapshot.joblib"
    models_dir: Path = REPO_ROOT / "models" / "injury_risk"

    # comma-separated list, e.g. "http://localhost:3000,https://myapp.vercel.app"
    cors_allowed_origins: str = "http://localhost:3000"

    # how long a team's PACE/OFF_RATING/DEF_RATING pull stays cached in
    # memory before being re-fetched — see backend/state.py. Season-level
    # aggregates barely move game to game, so a long TTL is fine.
    team_stats_cache_ttl_seconds: int = 21600  # 6h

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


settings = Settings()
