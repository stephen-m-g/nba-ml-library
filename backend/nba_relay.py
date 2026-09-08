"""Optionally redirects every nba_api stats.nba.com call through a relay
(see relay/main.py) instead of hitting stats.nba.com directly.

Why: stats.nba.com's bot-protection silently drops requests from
datacenter IPs (confirmed for Heroku, applies equally to Render and any
other cloud PaaS — see https://github.com/swar/nba_api/issues/320). It
does not block ordinary residential connections, so routing through a
relay running somewhere non-cloud (e.g. your own PC, exposed via
`cloudflared tunnel`) works around it without touching any of the actual
call sites in src/ — nba_api's HTTP layer is a single shared class
(NBAStatsHTTP), and patching its base_url/headers there covers every
current and future stats.nba.com call uniformly.

A no-op when NBA_STATS_RELAY_URL isn't set (i.e. always, in local dev) —
nba_api talks to stats.nba.com directly, exactly as before this existed.
"""

from __future__ import annotations

from backend.config import settings


def apply_relay_patch() -> None:
    if not settings.nba_stats_relay_url:
        return

    from nba_api.stats.library.http import NBAStatsHTTP, STATS_HEADERS

    relay_url = settings.nba_stats_relay_url.rstrip("/")
    NBAStatsHTTP.base_url = f"{relay_url}/stats/{{endpoint}}"
    NBAStatsHTTP.headers = {**STATS_HEADERS, "X-Relay-Token": settings.nba_stats_relay_token or ""}
