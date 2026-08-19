"""FastAPI service entry point. Run from the repo root with:

    uvicorn backend.main:app --reload

(needs `pip install -e .` done once — see pyproject.toml — so `from src...`
imports resolve regardless of cwd or how uvicorn was launched.)
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.schemas import HealthResponse
from backend.state import build_app_state
from backend.routers import players, risk


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.app_state = build_app_state()
    yield


app = FastAPI(title="Injury Risk API", lifespan=lifespan)

# Real traffic goes through the Next.js route-handler proxy (server-to-server,
# no browser CORS involved at all) — this allow-list exists so /docs and
# manual testing work directly from a browser during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(players.router)
app.include_router(risk.router)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    state = app.state.app_state
    return HealthResponse(
        status="ok",
        models_loaded=len(state.models),
        reference_players=len(state.reference.intervals),
        injury_history_as_of=state.reference.coverage_end.date().isoformat(),
    )
