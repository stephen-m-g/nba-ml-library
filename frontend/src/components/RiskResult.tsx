"use client";

import { useEffect, useState } from "react";
import type { ApiErrorDetail, RiskResponse } from "@/lib/types";

interface RiskResultProps {
  playerId: number;
}

type LoadState =
  | { status: "loading" }
  | { status: "error"; detail: ApiErrorDetail | string }
  | { status: "ready"; data: RiskResponse };

const LABELS: { key: keyof RiskResponse["predictions"]; title: string }[] = [
  { key: "y_1game", title: "Next game" },
  { key: "y_3game", title: "Next 3 games" },
  { key: "y_10game", title: "Next 10 games" },
];

export default function RiskResult({ playerId }: RiskResultProps) {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    setState({ status: "loading" });

    fetch(`/api/players/${playerId}/risk`)
      .then(async (res) => {
        const body = await res.json();
        if (cancelled) return;
        if (!res.ok) {
          setState({ status: "error", detail: body.detail ?? "Request failed." });
        } else {
          setState({ status: "ready", data: body as RiskResponse });
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
    return <p className="mt-6 text-gray-500">Loading...</p>;
  }

  if (state.status === "error") {
    const detail = state.detail;
    const message = typeof detail === "string" ? detail : detail.message;
    const code = typeof detail === "string" ? null : detail.error_code;
    return (
      <div className="mt-6 rounded border border-red-300 bg-red-50 px-4 py-3 text-red-800">
        <p className="font-medium">{code ?? "Error"}</p>
        <p className="text-sm">{message}</p>
      </div>
    );
  }

  const { data } = state;

  return (
    <div className="mt-6 space-y-4">
      <div>
        <h2 className="text-xl font-semibold">{data.player.full_name}</h2>
        <p className="text-sm text-gray-600">
          {data.current_team.full_name} ({data.current_team.abbreviation}) — as of {data.as_of}
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {LABELS.map(({ key, title }) => {
          const pred = data.predictions[key];
          return (
            <div
              key={key}
              className={`rounded border px-4 py-3 ${
                pred.elevated_risk ? "border-red-300 bg-red-50" : "border-gray-200 bg-gray-50"
              }`}
            >
              <p className="text-sm font-medium text-gray-700">{title}</p>
              <p className={`text-2xl font-bold ${pred.elevated_risk ? "text-red-700" : "text-gray-900"}`}>
                {pred.elevated_risk ? "Elevated risk" : "Normal"}
              </p>
              <p className="text-xs text-gray-500">
                probability {(pred.probability * 100).toFixed(1)}% (threshold {(pred.threshold * 100).toFixed(1)}%)
              </p>
            </div>
          );
        })}
      </div>

      {data.caveats.length > 0 && (
        <ul className="space-y-1">
          {data.caveats.map((c) => (
            <li key={c.code} className="rounded bg-yellow-50 px-3 py-2 text-sm text-yellow-800">
              {c.message}
            </li>
          ))}
        </ul>
      )}

      <details className="rounded border border-gray-200 px-3 py-2 text-sm text-gray-600">
        <summary className="cursor-pointer font-medium">Data quality</summary>
        <ul className="mt-2 space-y-1">
          <li>Injury history current as of: {data.injury_history_as_of}</li>
          <li>Last game played: {data.data_quality.last_game_played ?? "no record"}</li>
          <li>Days since last game: {data.data_quality.days_since_last_game ?? "unknown"}</li>
          <li>Games used for workload average: {data.data_quality.games_used_for_workload}</li>
          <li>Cohort backfill used: {data.data_quality.cohort_backfill_used ? "yes" : "no"}</li>
        </ul>
      </details>
    </div>
  );
}
