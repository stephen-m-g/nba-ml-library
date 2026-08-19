import { BackendError, fetchPlayerRisk } from "@/lib/api";

// The one genuinely dynamic, user-triggered call in this app — proxied so
// the backend's URL never reaches client code (see lib/api.ts). No
// as_of/date param accepted here: always "now," per the project plan
// (decision #2) — the underlying Python pipeline keeps as_of as a
// parameter purely for its own offline testing, never exposed over HTTP.
export async function GET(
  _request: Request,
  ctx: RouteContext<"/api/players/[id]/risk">
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
    const data = await fetchPlayerRisk(playerId);
    return Response.json(data);
  } catch (error) {
    if (error instanceof BackendError) {
      return Response.json({ detail: error.detail }, { status: error.status });
    }
    return Response.json({ detail: "Unexpected error." }, { status: 500 });
  }
}
