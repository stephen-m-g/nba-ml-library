import type { ApiErrorDetail, PlayersResponse, RiskResponse } from "./types";

// Server-only — never prefixed with NEXT_PUBLIC_, so this never reaches
// client bundles. Real browser traffic goes through this app's own
// /api/players and /api/players/[id]/risk route handlers, which read this
// same env var server-side; only Server Components (like the initial
// player-list fetch in app/page.tsx) call the backend directly.
const PY_API_BASE_URL = process.env.PY_API_BASE_URL ?? "http://localhost:8000";

export class BackendError extends Error {
  status: number;
  detail: ApiErrorDetail | string;

  constructor(status: number, detail: ApiErrorDetail | string) {
    super(typeof detail === "string" ? detail : detail.message);
    this.status = status;
    this.detail = detail;
  }
}

async function backendFetch<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${PY_API_BASE_URL}${path}`, { cache: "no-store" });
  } catch {
    throw new BackendError(503, "Could not reach the prediction service.");
  }

  if (!response.ok) {
    let detail: ApiErrorDetail | string = `Request failed with status ${response.status}`;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // non-JSON error body — keep the generic message
    }
    throw new BackendError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

export function fetchPlayers(): Promise<PlayersResponse> {
  return backendFetch<PlayersResponse>("/players");
}

export function fetchPlayerRisk(playerId: number): Promise<RiskResponse> {
  return backendFetch<RiskResponse>(`/players/${playerId}/risk`);
}
