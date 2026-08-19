import PlayerLookup from "@/components/PlayerLookup";
import { fetchPlayers } from "@/lib/api";

export default async function Home() {
  let players: Awaited<ReturnType<typeof fetchPlayers>>["players"] = [];
  let loadError: string | null = null;

  try {
    const response = await fetchPlayers();
    players = response.players;
  } catch {
    loadError = "Could not reach the prediction service. Is the backend running?";
  }

  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-12">
      <h1 className="text-2xl font-bold">NBA Injury Risk Lookup</h1>
      <p className="mt-1 text-sm text-gray-600">
        Search for a player to see their current elevated-risk status, computed live from today&apos;s data.
      </p>

      {loadError ? (
        <p className="mt-6 rounded border border-red-300 bg-red-50 px-4 py-3 text-red-800">{loadError}</p>
      ) : (
        <div className="mt-6">
          <PlayerLookup initialPlayers={players} />
        </div>
      )}
    </div>
  );
}
