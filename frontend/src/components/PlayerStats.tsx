"use client";

import { useEffect, useState } from "react";
import type { ApiErrorDetail, CareerStats, PlayerStatsResponse, SeasonStats } from "@/lib/types";

interface PlayerStatsProps {
  playerId: number;
}

type LoadState =
  | { status: "loading" }
  | { status: "error"; detail: ApiErrorDetail | string }
  | { status: "ready"; data: PlayerStatsResponse };

type Mode = "season" | "career";

// Only the per-game stat fields SeasonStats and CareerStats actually share.
// Deliberately not `keyof SeasonStats | keyof CareerStats`: that union also
// admits `season`/`seasons_played`, which exist on just one side each, so
// indexing the union with it isn't type-safe (and TS rejects it).
type StatKey = "pts" | "reb" | "ast" | "stl" | "blk" | "tov";

const STAT_LABELS: { key: StatKey; label: string }[] = [
  { key: "pts", label: "PTS" },
  { key: "reb", label: "REB" },
  { key: "ast", label: "AST" },
  { key: "stl", label: "STL" },
  { key: "blk", label: "BLK" },
  { key: "tov", label: "TOV" },
];

export default function PlayerStats({ playerId }: PlayerStatsProps) {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [mode, setMode] = useState<Mode>("season");

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });
    setMode("season");

    fetch(`/api/players/${playerId}/stats`)
      .then(async (res) => {
        const body = await res.json();
        if (cancelled) return;
        if (!res.ok) {
          setState({ status: "error", detail: body.detail ?? "Request failed." });
        } else {
          const data = body as PlayerStatsResponse;
          setState({ status: "ready", data });
          if (!data.season) setMode("career");
        }
      })
      .catch(() => {
        if (!cancelled) setState({ status: "error", detail: "Could not reach the server." });
      });

    return () => {
      cancelled = true;
    };
  }, [playerId]);

  if (state.status === "loading") {
    return <p className="mt-6 text-gray-500">Loading stats...</p>;
  }

  if (state.status === "error") {
    const detail = state.detail;
    const message = typeof detail === "string" ? detail : detail.message;
    return (
      <div className="mt-6 rounded border border-red-300 bg-red-50 px-4 py-3 text-red-800">
        <p className="text-sm">{message}</p>
      </div>
    );
  }

  const { data } = state;
  const active: SeasonStats | CareerStats | null = mode === "season" ? data.season : data.career;

  return (
    <div className="mt-6 space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold">Per-game averages</h3>
        <div className="inline-flex overflow-hidden rounded border border-gray-300 text-sm">
          <button
            type="button"
            onClick={() => setMode("season")}
            disabled={!data.season}
            className={`px-3 py-1 ${
              mode === "season" ? "bg-blue-600 text-white" : "bg-white text-gray-700 hover:bg-gray-100"
            } disabled:cursor-not-allowed disabled:opacity-40`}
          >
            Season
          </button>
          <button
            type="button"
            onClick={() => setMode("career")}
            className={`border-l border-gray-300 px-3 py-1 ${
              mode === "career" ? "bg-blue-600 text-white" : "bg-white text-gray-700 hover:bg-gray-100"
            }`}
          >
            Career
          </button>
        </div>
      </div>

      {!data.season && (
        <p className="text-xs text-gray-500">No games recorded this season — showing career averages.</p>
      )}

      {active && (
        <>
          <p className="text-xs text-gray-500">
            {mode === "season"
              ? `${(active as SeasonStats).season} season, ${active.games_played} games`
              : `${(active as CareerStats).seasons_played} seasons, ${active.games_played} games`}
          </p>
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-6">
            {STAT_LABELS.map(({ key, label }) => (
              <div key={key} className="rounded border border-gray-200 bg-gray-50 px-2 py-3 text-center">
                <p className="text-xl font-bold text-gray-900">{active[key].toFixed(1)}</p>
                <p className="text-xs text-gray-500">{label}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
