"""League-wide, offline backfill of injury history from the NBA's official
injury reports — the systematic counterpart to src/live_features.py's
capped, per-request gap-fill.

Why this exists: the frozen Kaggle-derived snapshot stops at a fixed
coverage_end, and the live per-request check can only afford to probe a
handful of a player's most recent missed games. That's fine for "what
happened this week," but it can silently miss an entire earlier absence
(confirmed real case: a player with 66 missed games in the gap window,
only 8 of which the live path could check). This module instead walks
report DATES rather than players — one report download covers every team
playing that day, so the whole league gets resolved in one pass — and
produces intervals plus a per-player coverage/confidence measure.

Run via notebooks/22_extend_injury_coverage.py. The output extends the
snapshot's coverage_end for real, which in turn shrinks the live path's
job to just the days since the last rebuild.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.feature_engineering import normalize_name
from src.injury_reports import (
    fetch_report_for_date, teams_covered_by_report, categorize_report_reason,
    normalize_report_player_name, REPORT_CACHE_DIR,
)


MAX_RUN_GAP_DAYS = 30


def build_intervals_from_missed_dates(
    missed_dates, confirmed_injury_dates: set, team_game_dates,
    max_gap_days: int = MAX_RUN_GAP_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """Groups missed_dates into contiguous runs (no intervening team game
    the player DID play) and keeps a run only if AT LEAST ONE of its dates
    was confirmed an injury absence — the same "infer still-out between
    confirmed points" principle feature_engineering.py::build_out_intervals
    uses for the Kaggle transaction log. A run with no confirmed date is
    dropped, not guessed either way, keeping this biased toward
    undercounting career_injury_count rather than inventing absences.

    A run is ALSO broken when consecutive team games are more than
    max_gap_days apart, even though they're adjacent in the schedule.
    Without this, a player who missed games in one season and again in the
    next — with no games played in between — merges into a single
    absence spanning the offseason: a real bug caught in testing, which
    produced a 404-day "injury" for one player. 30 days comfortably
    exceeds any in-season gap (the All-Star break is ~1 week) while always
    splitting at an offseason.

    Shared by both the offline league-wide build (this module) and the
    live per-request gap-fill (src/live_features.py) so the two can't drift.
    """
    if len(missed_dates) == 0 or not confirmed_injury_dates:
        return np.array([], dtype="datetime64[ns]"), np.array([], dtype="datetime64[ns]")

    all_dates = sorted(set(team_game_dates))
    missed_set = set(missed_dates)
    missed_positions = sorted(i for i, d in enumerate(all_dates) if d in missed_set)

    starts, ends = [], []
    run, prev_pos = [], None

    def flush(current_run):
        if current_run and any(all_dates[p] in confirmed_injury_dates for p in current_run):
            starts.append(all_dates[current_run[0]])
            ends.append(all_dates[current_run[-1]] + pd.Timedelta(days=1))

    for pos in missed_positions:
        if run:
            schedule_broken = pos != prev_pos + 1
            date_gap = (all_dates[pos] - all_dates[prev_pos]).days
            if schedule_broken or date_gap > max_gap_days:
                flush(run)
                run = []
        run.append(pos)
        prev_pos = pos
    flush(run)

    return np.array(starts, dtype="datetime64[ns]"), np.array(ends, dtype="datetime64[ns]")


def _report_name_to_id_map(static_players: pd.DataFrame) -> dict[str, int]:
    name_to_id: dict[str, int] = {}
    for full_name, player_id in zip(static_players["full_name"], static_players["id"]):
        name_to_id.setdefault(normalize_name(full_name), int(player_id))
    return name_to_id


_report_name_to_normalized = normalize_report_player_name


def build_status_index(
    dates, static_players: pd.DataFrame, cache_dir=REPORT_CACHE_DIR, progress_every: int = 25,
) -> tuple[dict, dict, dict]:
    """Fetch (or read from cache) one report per date and index it.

    Returns:
      status_by_player_date: {(player_id, date): (status, reason_category)}
      teams_by_date:         {date: set of team abbreviations the report covered}
      diagnostics:           counts worth printing/saving

    Unmatched report names (a name the static player table doesn't know)
    are counted in diagnostics rather than silently dropped — a rising
    unmatched count is the signal that name matching, not the parser, has
    started failing.
    """
    name_to_id = _report_name_to_id_map(static_players)
    status_by_player_date: dict[tuple[int, pd.Timestamp], tuple[str, str]] = {}
    teams_by_date: dict[pd.Timestamp, set[str]] = {}

    dates = [pd.Timestamp(d).normalize() for d in dates]
    dates_with_report = 0
    unmatched_names: set[str] = set()
    total_rows = 0

    for i, d in enumerate(dates, start=1):
        report = fetch_report_for_date(d, cache_dir=cache_dir)
        teams_by_date[d] = teams_covered_by_report(report)
        if len(report):
            dates_with_report += 1
            total_rows += len(report)
            for name, status, reason in zip(report["PLAYER_NAME"], report["STATUS"], report["REASON"]):
                pid = name_to_id.get(_report_name_to_normalized(name))
                if pid is None:
                    unmatched_names.add(str(name))
                    continue
                status_by_player_date[(pid, d)] = (str(status), categorize_report_reason(reason))
        if progress_every and i % progress_every == 0:
            print(f"  ...{i}/{len(dates)} dates processed ({dates_with_report} with a report)")

    diagnostics = {
        "dates_requested": len(dates),
        "dates_with_report": dates_with_report,
        "report_rows_indexed": total_rows,
        "unmatched_name_count": len(unmatched_names),
        "unmatched_names_sample": sorted(unmatched_names)[:20],
    }
    return status_by_player_date, teams_by_date, diagnostics


def _player_team_timeline(player_rows: pd.DataFrame) -> list[tuple[str, pd.Timestamp]]:
    """[(team_abbreviation, first_game_date_with_that_team), ...] in order.
    Handles mid-window trades without needing a separate transactions feed —
    a change of TEAM_ABBREVIATION between consecutive played games IS the trade.
    """
    timeline = []
    last_team = None
    for team, date in zip(player_rows["TEAM_ABBREVIATION"], player_rows["GAME_DATE"]):
        if team != last_team:
            timeline.append((team, date))
            last_team = team
    return timeline


def build_supplemental_intervals(
    player_log: pd.DataFrame,
    team_log: pd.DataFrame,
    status_by_player_date: dict,
    teams_by_date: dict,
    since: pd.Timestamp,
    until: pd.Timestamp,
) -> tuple[dict, pd.DataFrame, dict]:
    """Per player: find the games their team played but they didn't (within
    the window, from their first appearance onward), classify each against
    the report index, and build injury intervals from confirmed runs.

    A missed date is counted RESOLVED only if a report for that date
    actually covered the player's team — that's the honest line for the
    confidence score. Within resolved dates:
      - listed Out with an Injury/Illness reason -> confirms an injury absence
      - listed Out with a rest/personal/G-League reason -> resolved, not injury
      - listed but not Out (Questionable/Available and still didn't play) ->
        resolved but ambiguous; deliberately NOT treated as confirmation
      - team covered, player not listed at all -> resolved, not injury
        (they weren't on the injury report — most likely a coach's decision)

    **Confidence is measured only over the RESOLVABLE window** — dates at
    or after the earliest date any report was found. Confirmed empirically
    (2026-08-25): the NBA's CDN retains injury-report PDFs for only ~8
    months, with a clean cutoff (145 consecutive dates returned nothing,
    then 104 consecutive dates all had reports). Dates before that cutoff
    are permanently unrecoverable from this source, so counting them as
    "unresolved" would blend a fixable gap with an unfixable one and
    produce a confidence number that understates how good the recent data
    actually is. They're reported separately as `missed_outside_window`.

    Returns (intervals, coverage_df, diagnostics). intervals maps
    player_id -> (starts, ends) for the SUPPLEMENT ONLY — merging with the
    frozen snapshot's own intervals is the caller's job.
    """
    player_log = player_log.copy()
    player_log["GAME_DATE"] = pd.to_datetime(player_log["GAME_DATE"])
    team_log = team_log.copy()
    team_log["GAME_DATE"] = pd.to_datetime(team_log["GAME_DATE"])

    since, until = pd.Timestamp(since).normalize(), pd.Timestamp(until).normalize()
    player_log = player_log[(player_log["GAME_DATE"] > since) & (player_log["GAME_DATE"] <= until)]
    team_log = team_log[(team_log["GAME_DATE"] > since) & (team_log["GAME_DATE"] <= until)]

    team_dates = {
        abbr: np.array(sorted(grp["GAME_DATE"].unique()))
        for abbr, grp in team_log.groupby("TEAM_ABBREVIATION")
    }

    # Earliest date any report was actually available — everything before
    # it is outside what this source can ever answer (see docstring).
    dates_with_coverage = sorted(d for d, teams in teams_by_date.items() if teams)
    resolvable_from = dates_with_coverage[0] if dates_with_coverage else None

    intervals: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    coverage_rows = []
    totals = {"missed": 0, "resolved": 0, "injury": 0, "non_injury": 0,
              "ambiguous": 0, "not_listed": 0, "outside_window": 0}

    for player_id, rows in player_log.sort_values("GAME_DATE").groupby("PLAYER_ID"):
        player_id = int(player_id)
        played = set(rows["GAME_DATE"])
        timeline = _player_team_timeline(rows)

        relevant_dates: list[pd.Timestamp] = []
        for idx, (team, stint_start) in enumerate(timeline):
            # A stint runs until the next team's first game (a trade), or to
            # the end of the window for the final stint. Deliberately NOT
            # capped at the player's own last game: a season-ending injury
            # has no "next game" to bound it, and the confirmation
            # requirement in build_intervals_from_missed_dates is what keeps
            # the open-ended tail from inventing absences.
            stint_end = timeline[idx + 1][1] if idx + 1 < len(timeline) else until
            dates = team_dates.get(team)
            if dates is None:
                continue
            relevant_dates.extend(
                pd.Timestamp(d) for d in dates if stint_start <= pd.Timestamp(d) <= stint_end
            )
        relevant_dates = sorted(set(relevant_dates))
        missed = [d for d in relevant_dates if d not in played]
        if not missed:
            continue

        player_teams = {team for team, _ in timeline}
        confirmed, resolved, in_window, outside_window = set(), 0, 0, 0
        counts = {"injury": 0, "non_injury": 0, "ambiguous": 0, "not_listed": 0}

        for d in missed:
            if resolvable_from is None or d < resolvable_from:
                outside_window += 1
                continue
            in_window += 1
            covered = teams_by_date.get(d, set()) & player_teams
            if not covered:
                continue  # no report covering this player's team that day -> unresolved
            resolved += 1
            entry = status_by_player_date.get((player_id, d))
            if entry is None:
                counts["not_listed"] += 1
            elif entry[0] == "Out" and entry[1] == "injury":
                confirmed.add(d)
                counts["injury"] += 1
            elif entry[0] == "Out":
                counts["non_injury"] += 1
            else:
                counts["ambiguous"] += 1

        starts, ends = build_intervals_from_missed_dates(missed, confirmed, relevant_dates)
        if len(starts):
            intervals[player_id] = (starts, ends)

        coverage_rows.append({
            "PLAYER_ID": player_id,
            "missed_games": len(missed),
            "missed_in_window": in_window,
            "missed_outside_window": outside_window,
            "resolved_games": resolved,
            # None, not 1.0, when a player has no missed games inside the
            # resolvable window: "nothing to measure" is a different fact
            # from "perfectly covered," and collapsing them would show a
            # confident green bar for a player we actually know nothing about.
            "confidence": (resolved / in_window) if in_window else None,
            "confirmed_injury_games": counts["injury"],
            "non_injury_games": counts["non_injury"],
            "ambiguous_games": counts["ambiguous"],
            "not_listed_games": counts["not_listed"],
            "supplemental_intervals": len(starts),
            "supplemental_days_out": int(sum((pd.Timestamp(e) - pd.Timestamp(s)).days for s, e in zip(starts, ends))),
        })

        totals["missed"] += len(missed)
        totals["resolved"] += resolved
        totals["outside_window"] += outside_window
        for k in counts:
            totals[k] += counts[k]

    coverage_df = pd.DataFrame(coverage_rows)
    missed_in_window = totals["missed"] - totals["outside_window"]
    diagnostics = {
        "players_with_missed_games": len(coverage_rows),
        "players_with_supplemental_intervals": len(intervals),
        "resolvable_from": str(resolvable_from.date()) if resolvable_from is not None else None,
        "total_missed_games": totals["missed"],
        "missed_games_in_window": missed_in_window,
        "missed_games_outside_window": totals["outside_window"],
        "total_resolved_games": totals["resolved"],
        "overall_confidence": (totals["resolved"] / missed_in_window) if missed_in_window else None,
        "confirmed_injury_games": totals["injury"],
        "non_injury_games": totals["non_injury"],
        "ambiguous_games": totals["ambiguous"],
        "not_listed_games": totals["not_listed"],
    }
    return intervals, coverage_df, diagnostics


def merge_intervals(
    base: dict[int, tuple[np.ndarray, np.ndarray]],
    supplement: dict[int, tuple[np.ndarray, np.ndarray]],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Union of two interval dicts, sorted by start per player. Returns a
    NEW dict — never mutates `base`, which is the shared frozen snapshot.
    Safe against overlap because the supplement is built strictly after the
    base's coverage_end, so the two can't describe the same absence twice.
    """
    merged = {pid: (starts.copy(), ends.copy()) for pid, (starts, ends) in base.items()}
    for pid, (starts, ends) in supplement.items():
        if pid in merged:
            all_starts = np.concatenate([merged[pid][0], starts])
            all_ends = np.concatenate([merged[pid][1], ends])
        else:
            all_starts, all_ends = starts, ends
        order = np.argsort(all_starts)
        merged[pid] = (all_starts[order], all_ends[order])
    return merged
