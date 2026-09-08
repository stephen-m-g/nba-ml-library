"""Runs on a non-cloud machine (e.g. your own PC) and forwards requests to
stats.nba.com on its behalf.

Why this exists: stats.nba.com sits behind bot-protection that silently
drops (read-timeout, not a clean 403) requests from datacenter/cloud IP
ranges (AWS, GCP, Azure, and every PaaS built on them — Render included).
It does not block ordinary residential connections. Rather than the
deployed backend calling stats.nba.com directly, it calls this relay
instead (via NBA_STATS_RELAY_URL — see backend/nba_relay.py), and this
relay makes the real call from wherever it happens to be running.

This is a byte-for-byte passthrough at the HTTP-request layer, not a raw
TCP/CONNECT proxy — deliberately, so it can sit behind a plain HTTPS
tunnel (e.g. `cloudflared tunnel --url http://localhost:8899`) with no
port-forwarding or raw-TCP tunnel support required.

Run locally with the SAME venv the backend already uses (fastapi/uvicorn/
requests are already installed for it):

    $env:RELAY_TOKEN="<a long random secret, must match backend's
        NBA_STATS_RELAY_TOKEN>"
    .venv\\Scripts\\python.exe -m uvicorn relay.main:app --host 0.0.0.0 --port 8899

Then expose it with a tunnel and point the deployed backend's
NBA_STATS_RELAY_URL at the tunnel's public URL. See relay/.env.example.
"""

from __future__ import annotations

import os

import requests
from fastapi import FastAPI, HTTPException, Request, Response
from nba_api.stats.library.http import STATS_HEADERS

RELAY_TOKEN = os.environ.get("RELAY_TOKEN")

app = FastAPI(title="NBA Stats Relay")


@app.get("/stats/{endpoint}")
def proxy_stats(endpoint: str, request: Request) -> Response:
    if not RELAY_TOKEN or request.headers.get("X-Relay-Token") != RELAY_TOKEN:
        raise HTTPException(status_code=403, detail="missing or invalid X-Relay-Token")

    upstream = requests.get(
        f"https://stats.nba.com/stats/{endpoint}",
        params=dict(request.query_params),
        headers=STATS_HEADERS,
        timeout=30,
    )
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("Content-Type", "application/json"),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "token_configured": bool(RELAY_TOKEN)}
