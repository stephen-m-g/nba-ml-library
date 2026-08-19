"""Typed access to what the service loads once at startup, plus the one
thing that's lazily fetched and TTL-cached in memory rather than loaded
upfront.

Loaded once at startup (see backend/main.py's lifespan) and read-only after
that: the 3 injury-risk models, the live-reference snapshot (injury
intervals + cohort baseline, frozen train-fit-once data — see
src/live_reference_data.py), and the static player table (for search).

Lazily fetched + TTL-cached: current-season team advanced stats
(PACE/OFF_RATING/DEF_RATING). Deliberately NOT loaded at startup — a
temporary NBA Stats API hiccup shouldn't be able to block the whole
service from reporting healthy, and season-level team stats barely change
game to game, so a coarse in-memory cache is enough (see
config.team_stats_cache_ttl_seconds).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import pandas as pd
from fastapi import Request

from src.data_loader import get_static_players
from src.live_reference_data import LiveReferenceData, load_snapshot
from src.live_features import fetch_live_team_advanced_stats
from src.predict import load_models

from backend.config import settings


@dataclass
class AppState:
    models: dict
    reference: LiveReferenceData
    static_players: pd.DataFrame
    _team_stats_cache: dict[int, tuple[float, dict]] = field(default_factory=dict)

    def get_team_stats_cached(self, team_id: int, as_of: pd.Timestamp) -> dict | None:
        cached = self._team_stats_cache.get(team_id)
        now = time.monotonic()
        if cached is not None and (now - cached[0]) < settings.team_stats_cache_ttl_seconds:
            return cached[1]

        stats = fetch_live_team_advanced_stats(team_id, as_of)
        if stats is not None:
            self._team_stats_cache[team_id] = (now, stats)
        return stats


def build_app_state() -> AppState:
    """Called once from backend/main.py's lifespan. Fails loudly and
    immediately if the snapshot or models are missing — a service that
    can't score predictions correctly shouldn't come up looking healthy.
    """
    if not settings.snapshot_path.exists():
        raise RuntimeError(
            f"No live reference snapshot at {settings.snapshot_path}. "
            f"Run notebooks/20_build_live_reference_data.py first."
        )
    reference = load_snapshot(settings.snapshot_path)
    models = load_models(settings.models_dir)
    static_players = get_static_players()
    return AppState(models=models, reference=reference, static_players=static_players)


def get_state(request: Request) -> AppState:
    return request.app.state.app_state
