"use client";

import { useState } from "react";
import PlayerSearch from "./PlayerSearch";
import RiskResult from "./RiskResult";
import PlayerStats from "./PlayerStats";
import type { PlayerSummary } from "@/lib/types";

interface PlayerLookupProps {
  initialPlayers: PlayerSummary[];
}

export default function PlayerLookup({ initialPlayers }: PlayerLookupProps) {
  const [selected, setSelected] = useState<PlayerSummary | null>(null);

  return (
    <div>
      <PlayerSearch players={initialPlayers} onSelect={setSelected} />
      {selected && (
        <>
          <RiskResult key={`risk-${selected.player_id}`} playerId={selected.player_id} />
          <div className="mt-6 border-t border-gray-200 pt-2">
            <PlayerStats key={`stats-${selected.player_id}`} playerId={selected.player_id} />
          </div>
        </>
      )}
    </div>
  );
}
