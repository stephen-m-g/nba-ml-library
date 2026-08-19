import { BackendError, fetchPlayers } from "@/lib/api";

// Not used by the current page (app/page.tsx fetches the player list
// directly from a Server Component instead — one less hop for the common
// case). Kept as a real, working proxy so a future switch to client-side
// or debounced server-side search has somewhere to call without a
// breaking change — see the project plan's API contract notes.
export async function GET() {
  try {
    const data = await fetchPlayers();
    return Response.json(data);
  } catch (error) {
    if (error instanceof BackendError) {
      return Response.json({ detail: error.detail }, { status: error.status });
    }
    return Response.json({ detail: "Unexpected error." }, { status: 500 });
  }
}
