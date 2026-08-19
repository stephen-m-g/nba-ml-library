"use client";

import { useMemo, useRef, useState } from "react";
import type { PlayerSummary } from "@/lib/types";

interface PlayerSearchProps {
  players: PlayerSummary[];
  onSelect: (player: PlayerSummary) => void;
}

const MAX_RESULTS = 20;

// Plenty of real players have diacritics (Jokić, Dončić, Porziņģis...) that
// most people won't type — fold both sides to plain ASCII before matching,
// the same normalization src/feature_engineering.py::normalize_name already
// does on the Python side for injury-log name matching.
function foldToAscii(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/[̀-ͯ]/g, "")
    .toLowerCase();
}

export default function PlayerSearch({ players, onSelect }: PlayerSearchProps) {
  const [query, setQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const matches = useMemo(() => {
    const trimmed = foldToAscii(query.trim());
    if (!trimmed) return [];
    return players
      .filter((p) => foldToAscii(p.full_name).includes(trimmed))
      .sort((a, b) => Number(b.is_active) - Number(a.is_active))
      .slice(0, MAX_RESULTS);
  }, [players, query]);

  function handleSelect(player: PlayerSummary) {
    onSelect(player);
    setQuery(player.full_name);
    setIsOpen(false);
  }

  function handleBlur() {
    // Delay so a click on a dropdown item still registers before the
    // dropdown unmounts (an onMouseDown-based approach would also work,
    // this is simpler and fine for a functionality-first pass).
    setTimeout(() => setIsOpen(false), 150);
  }

  return (
    <div ref={containerRef} className="relative w-full max-w-md">
      <label htmlFor="player-search" className="mb-1 block text-sm font-medium">
        Player name
      </label>
      <input
        id="player-search"
        type="text"
        autoComplete="off"
        placeholder="Start typing a player's name..."
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setIsOpen(true);
        }}
        onFocus={() => setIsOpen(true)}
        onBlur={handleBlur}
        className="w-full rounded border border-gray-300 px-3 py-2 text-base focus:border-blue-500 focus:outline-none"
      />
      {isOpen && query.trim() && (
        <ul className="absolute z-10 mt-1 max-h-72 w-full overflow-y-auto rounded border border-gray-300 bg-white shadow-lg">
          {matches.length === 0 && (
            <li className="px-3 py-2 text-sm text-gray-500">No matching players</li>
          )}
          {matches.map((player) => (
            <li key={player.player_id}>
              <button
                type="button"
                onClick={() => handleSelect(player)}
                className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-gray-100"
              >
                <span>{player.full_name}</span>
                {!player.is_active && <span className="text-xs text-gray-400">retired</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
