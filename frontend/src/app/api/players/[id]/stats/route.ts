import { BackendError, fetchPlayerStats } from "@/lib/api";

// Same proxy pattern as ../risk/route.ts — keeps the backend URL out of
// client code. Independent of the risk lookup: this can succeed for a
// player /risk correctly refuses (e.g. a retired player has no current
// team for prediction purposes, but their career stats are still valid).
export async function GET(
  _request: Request,
  ctx: RouteContext<"/api/players/[id]/stats">
) {
  const { id } = await ctx.params;
  const playerId = Number(id);

  if (!Number.isInteger(playerId)) {
    return Response.json(
      { detail: { error_code: "INVALID_PLAYER_ID", message: `Not a valid player id: ${id}` } },
      { status: 400 }
    );
  }

  try {
    const data = await fetchPlayerStats(playerId);
    return Response.json(data);
  } catch (error) {
    if (error instanceof BackendError) {
      return Response.json({ detail: error.detail }, { status: error.status });
    }
    return Response.json({ detail: "Unexpected error." }, { status: 500 });
  }
}
