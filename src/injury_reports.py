"""Fetch and parse the NBA's official injury report PDFs
(ak-static.cms.nba.com) — a Java-free alternative to the `nbainjuries`
package (uses pdfplumber instead of tabula-py/JVM; see project memory for
why that swap was made).

This is NOT a historical-archive builder. It exists to fill the specific
gap between the frozen Kaggle-based snapshot's coverage_end and "today" at
LIVE inference time (see src/live_features.py::compute_live_injury_history_features),
bounded to one player's own recently-missed games — not a general-purpose
crawl of the report archive.

Report layout (confirmed against real reports, not assumed): one row per
player, with GameDate/GameTime/Matchup/Team only printed on the first row
of each group (blank/forward-filled after that), PlayerName as "Last,First"
(no space), and Reason text that can wrap across multiple lines vertically
CENTERED on the row (continuation words can appear both above and below
the row's own y-position) — a naive top-position row match misses this;
see _parse_page's banding approach.
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import requests

from src.data_loader import retry_api_call
from src.feature_engineering import normalize_name

REPORT_BASE_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{date}_{time}.pdf"

REPORT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "injury_reports"

REPORT_COLUMNS = ["GAME_DATE", "GAME_TIME", "MATCHUP", "TEAM", "PLAYER_NAME", "STATUS", "REASON",
                  "REPORT_TIMESTAMP"]

# Column x0 boundaries (points from page left edge), confirmed against a
# real report this session — generous margins around the observed values
# (GameDate~23, GameTime~120, Matchup~200, Team~264, PlayerName~426,
# CurrentStatus~587, Reason~667) rather than exact cutoffs, since a few
# points of layout drift between reports shouldn't misclassify a column.
COLUMN_BOUNDS = {
    "GAME_DATE": (0, 110),
    "GAME_TIME": (110, 195),
    "MATCHUP": (195, 260),
    "TEAM": (260, 420),
    "PLAYER_NAME": (420, 580),
    "STATUS": (580, 660),
    "REASON": (660, 1000),
}

HEADER_WORDS = {"GameDate", "GameTime", "Matchup", "Team", "PlayerName", "CurrentStatus", "Reason"}

# Reasons that don't count as a genuine injury under this project's
# "strict" definition (see project-injury-label-definition memory) — kept
# separate from feature_engineering.py::categorize_injury_reason, which is
# tuned for the Kaggle dataset's free-text Notes format, not this report's
# structured "Injury/Illness-<detail>" vs. plain-category Reason text.
NON_INJURY_REASON_PREFIXES = (
    "rest", "personal", "suspension", "not with team", "g league", "gleague",
    "health and safety", "covid", "return to competition",
)


def build_report_url(timestamp: datetime) -> str:
    date_part = timestamp.strftime("%Y-%m-%d")
    time_part = timestamp.strftime("%I_%M%p").upper()
    return REPORT_BASE_URL.format(date=date_part, time=time_part)


def candidate_report_timestamps(game_date, max_candidates: int = 4) -> list[datetime]:
    """Plausible report timestamps that might cover a given game date,
    ordered MOST-complete-first, not newest-calendar-first. Confirmed
    empirically (this session, same game date): the day-before 5:00pm
    report had 87 rows; the game-day 1:00pm report had 259 — the report
    accumulates as more teams submit through the day-before-deadline to
    game-day cycle, so the closest-to-tipoff snapshot is the most reliable
    one, not the earliest one that happens to exist. Game-day slots are
    tried first for that reason, day-before slots as fallback. This
    project doesn't model team-local timezones, and the exact minute
    varies, so this probes commonly-observed slots rather than computing
    one "correct" deadline timestamp — good enough for "was this player
    out," not a claim of completeness.

    max_candidates caps the list (default 4, the 3 game-day slots plus the
    single most likely day-before slot) — this runs live, per missed game,
    per player; every extra candidate is a potential extra PDF fetch, and
    fetch_injury_report is cached by exact timestamp, so widening this
    only helps when the first few genuinely don't exist, which is the
    uncommon case.
    """
    game_date = pd.Timestamp(game_date).normalize()
    day_before = game_date - timedelta(days=1)
    candidates = []
    for hour, minute in [(13, 0), (12, 30), (11, 0)]:
        candidates.append(game_date.replace(hour=hour, minute=minute).to_pydatetime())
    for hour, minute in [(17, 0), (16, 30), (16, 0), (15, 30), (15, 0), (13, 0), (12, 30)]:
        candidates.append(day_before.replace(hour=hour, minute=minute).to_pydatetime())
    return candidates[:max_candidates]


def _fetch_pdf_bytes(url: str) -> bytes | None:
    """None (not an error) for a 404 OR 403 — most candidate timestamps for
    a given date won't have a real report; that's expected, not a failure.
    Confirmed live (this session) that this specific host (S3-backed)
    returns 403 AccessDenied, not 404, for a nonexistent object — verified
    by requesting a deliberately-invalid timestamp (403) alongside a
    known-good one (200) and comparing; this is a bucket privacy setting
    (don't reveal via 404 whether a key exists), not rate-limiting or an
    access-policy block, so treating it as "no report" here is correct,
    not a workaround for something that should be investigated further.
    """
    def _get():
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code in (403, 404):
            return None
        resp.raise_for_status()
        return resp.content

    return retry_api_call(_get, attempts=2, backoff_sec=1.0)


def _classify_column(x0: float) -> str | None:
    for col, (lo, hi) in COLUMN_BOUNDS.items():
        if lo <= x0 < hi:
            return col
    return None


def _parse_page(page) -> list[dict]:
    """One dict per player row on this page. GAME_DATE/GAME_TIME/MATCHUP/
    TEAM are left as None where blank (forward-filled later, across the
    whole report, not just within a page — a team's list can span a page
    break). REASON is reconstructed from every word in the Reason column
    whose y-position falls in this row's "band" (the midpoint gap to the
    previous/next player row) rather than words matching the row's exact
    y-position, since wrapped Reason text is vertically centered and can
    sit above or below the row it belongs to.

    The page title ("Injury Report: <date> <time>") repeats on EVERY page
    and sits at x-positions that coincidentally overlap several of
    COLUMN_BOUNDS, so it has to be excluded explicitly. Two independent
    guards, because each alone was confirmed insufficient:

    1. Drop the title line by y-position. Keying this off the column
       header ("PlayerName") only works on page 1 — verified directly that
       continuation pages repeat the title but carry NO header row, so an
       early header-only version of this leaked title fragments
       ("04/12/26", "01:00") into the TEAM/PLAYER_NAME columns on pages
       2+, which then surfaced as ~46 unmatched "player names" in a
       league-wide run. Anchoring on the title's own "Report:" word
       instead works on every page.
    2. Require a comma in anything treated as a player-name anchor. The
       report always formats names "Last,First", and no title/footer
       fragment contains a comma — a cheap structural check that doesn't
       depend on layout coordinates holding steady at all.
    """
    words = page.extract_words()

    title_top = next((w["top"] for w in words if w["text"] == "Report:"), None)
    if title_top is not None:
        words = [w for w in words if abs(w["top"] - title_top) > 3]
    header_top = next((w["top"] for w in words if w["text"] == "PlayerName"), None)
    if header_top is not None:
        words = [w for w in words if w["top"] >= header_top]

    tagged = [
        {"col": col, "top": w["top"], "text": w["text"]}
        for w in words
        if (col := _classify_column(w["x0"])) is not None and w["text"] not in HEADER_WORDS
    ]

    anchors = sorted(
        (t for t in tagged if t["col"] == "PLAYER_NAME" and "," in t["text"]),
        key=lambda t: t["top"],
    )
    if not anchors:
        return []

    tops = [a["top"] for a in anchors]
    bounds = [
        (
            -1.0 if i == 0 else (tops[i - 1] + tops[i]) / 2,
            1e9 if i == len(tops) - 1 else (tops[i] + tops[i + 1]) / 2,
        )
        for i in range(len(tops))
    ]

    rows = [
        {"GAME_DATE": None, "GAME_TIME": None, "MATCHUP": None, "TEAM": None,
         "PLAYER_NAME": a["text"], "STATUS": None, "_reason_parts": []}
        for a in anchors
    ]

    for t in tagged:
        if t["col"] == "PLAYER_NAME":
            continue
        idx = next((i for i, (lo, hi) in enumerate(bounds) if lo <= t["top"] < hi), None)
        if idx is None:
            continue
        if t["col"] == "REASON":
            rows[idx]["_reason_parts"].append((t["top"], t["text"]))
        elif rows[idx][t["col"]] is None:
            rows[idx][t["col"]] = t["text"]

    for r in rows:
        r["REASON"] = " ".join(text for _, text in sorted(r.pop("_reason_parts")))
    return rows


@lru_cache(maxsize=256)
def fetch_injury_report(timestamp: datetime) -> pd.DataFrame | None:
    """One full report -> a DataFrame with columns GAME_DATE, GAME_TIME,
    MATCHUP, TEAM, PLAYER_NAME, STATUS, REASON — one row per player across
    every page, with GAME_DATE/GAME_TIME/MATCHUP/TEAM forward-filled
    (they're only printed once per group in the source PDF). None if no
    report exists at this exact timestamp (a normal, expected outcome for
    most candidate timestamps, not a failure) or the PDF has no
    recognizable player rows.
    """
    import pdfplumber

    pdf_bytes = _fetch_pdf_bytes(build_report_url(timestamp))
    if pdf_bytes is None:
        return None

    all_rows = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            all_rows.extend(_parse_page(page))

    if not all_rows:
        return None

    df = pd.DataFrame(all_rows)
    df[["GAME_DATE", "GAME_TIME", "MATCHUP", "TEAM"]] = df[["GAME_DATE", "GAME_TIME", "MATCHUP", "TEAM"]].ffill()
    df["REPORT_TIMESTAMP"] = timestamp
    return df


def categorize_report_reason(reason: str) -> str:
    """'injury' if the Reason text is the report's own "Injury/Illness"
    prefix (a strong, structured signal); 'non_injury' for known
    rest/personal/G-League/etc. categories; 'unknown' otherwise —
    deliberately NOT treated as injury by default, consistent with this
    project's preference for undercounting over injecting noise into
    career_injury_count (its single most important feature).
    """
    if not isinstance(reason, str) or not reason.strip():
        return "unknown"
    text = reason.strip().lower()
    if text.startswith("injury/illness") or text.startswith("injury"):
        return "injury"
    if any(text.startswith(prefix) for prefix in NON_INJURY_REASON_PREFIXES):
        return "non_injury"
    return "unknown"


def find_player_in_report(report: pd.DataFrame, player_full_name: str) -> pd.Series | None:
    """Matches by normalized name only (see normalize_report_player_name,
    which handles the glued-suffix quirk on top of the same diacritic
    folding used for Kaggle-name matching), not team — a specific player's
    full name is effectively unique within one report, and team-name
    matching would need a whole separate "AtlantaHawks" -> "ATL" mapping
    table for no real benefit here.
    """
    target = normalize_name(player_full_name)
    names_normalized = report["PLAYER_NAME"].apply(normalize_report_player_name)
    matches = report[names_normalized == target]
    return matches.iloc[0] if len(matches) else None


def fetch_report_for_date(game_date, cache_dir: Path | str = REPORT_CACHE_DIR,
                           use_cache: bool = True, max_candidates: int = 4) -> pd.DataFrame:
    """All report rows covering a specific GAME DATE, disk-cached one CSV
    per date. Used by the offline league-wide backfill
    (src/injury_backfill.py) — NOT by live per-request lookups, which use
    find_player_status_for_date instead.

    Filters to rows whose own GAME_DATE column equals game_date: a single
    report file can list more than one game date (a day-before report
    covers the next day's slate), so keying by the report's own stated
    game date is what actually makes "the report for date D" well-defined,
    rather than assuming the file's timestamp implies its contents.

    An EMPTY (0-row) result is a meaningful, cached outcome — "checked,
    no report available for this date" — not a failure, and it's written
    to the cache too so a re-run doesn't re-probe every dateless day (the
    All-Star break, offseason gaps, etc.). Deleting a cache file forces a
    refetch for that date.
    """
    game_date = pd.Timestamp(game_date).normalize()
    cache_path = Path(cache_dir) / f"report_{game_date.date().isoformat()}.csv"

    if use_cache and cache_path.exists():
        cached = pd.read_csv(cache_path, dtype=str)
        return cached.reindex(columns=REPORT_COLUMNS) if len(cached) else cached

    found = pd.DataFrame(columns=REPORT_COLUMNS)
    for ts in candidate_report_timestamps(game_date, max_candidates=max_candidates):
        try:
            report = fetch_injury_report(ts)
        except Exception:
            continue
        if report is None:
            continue
        report_dates = pd.to_datetime(report["GAME_DATE"], format="%m/%d/%Y", errors="coerce").dt.normalize()
        rows = report[report_dates == game_date]
        if len(rows):
            # candidate_report_timestamps is ordered most-complete-first, so
            # the first timestamp that actually covers this date is also the
            # best one available — no reason to keep probing older, sparser ones.
            found = rows.reindex(columns=REPORT_COLUMNS)
            break

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    found.to_csv(cache_path, index=False)
    return found


_GLUED_SUFFIX_RE = re.compile(r"^(.*[a-z])(Jr|Sr|II|III|IV|V)\.?$")


def normalize_report_player_name(last_comma_first: str) -> str:
    """Report name ("Bagley III,Marvin") -> the same normalized form
    feature_engineering.normalize_name produces for a static-table name
    ("Marvin Bagley III" -> "marvin bagley").

    The extra step over a plain split+normalize: pdfplumber returns the
    name as a single word, so a generational suffix arrives GLUED to the
    surname ("BagleyIII,Marvin", "HardawayJr.,Tim"). normalize_name only
    strips a suffix preceded by whitespace, so those silently failed to
    match — confirmed as ~33 unmatched names in a league-wide run, all of
    this exact shape. The suffix is split off here first.

    Deliberately NOT fixed by loosening normalize_name itself: that
    function is load-bearing for the validated Kaggle injury-label
    matching the trained models depend on, and this glued-suffix form is
    an artifact of PDF word extraction that only occurs here.
    """
    parts = str(last_comma_first).split(",", 1)
    if len(parts) != 2:
        return normalize_name(last_comma_first)
    last, first = parts[0].strip(), parts[1].strip()
    match = _GLUED_SUFFIX_RE.match(last)
    if match and len(match.group(1)) >= 2:
        last = f"{match.group(1)} {match.group(2)}"
    return normalize_name(f"{first} {last}")


def teams_covered_by_report(report: pd.DataFrame) -> set[str]:
    """Team abbreviations a report actually covers, parsed from MATCHUP
    ("ATL@MIA" -> {"ATL", "MIA"}). Load-bearing for the confidence score:
    a player missing from a report means "not on the injury report" (a real
    signal — likely a healthy scratch) only if their TEAM was covered by
    that report at all; otherwise it's a data gap and must not be read as
    evidence of anything. The TEAM column ("AtlantaHawks") can't serve this
    purpose without a separate full-name->abbreviation mapping table, which
    MATCHUP makes unnecessary.
    """
    if len(report) == 0 or "MATCHUP" not in report.columns:
        return set()
    teams: set[str] = set()
    for matchup in report["MATCHUP"].dropna():
        teams.update(part for part in str(matchup).replace("@", " ").replace("vs.", " ").split() if part)
    return teams


def find_player_status_for_date(player_full_name: str, game_date) -> pd.Series | None:
    """The main entry point src/live_features.py calls: probes
    candidate_report_timestamps for game_date (most-complete-first) and
    returns the player's row from the first report that lists them at
    all. None if no candidate report could be fetched, or none of the
    ones that did includes this player — a normal, common outcome (most
    players aren't on the injury report most days) that callers should
    treat as "no signal," not an error.
    """
    for ts in candidate_report_timestamps(game_date):
        report = fetch_injury_report(ts)
        if report is None:
            continue
        row = find_player_in_report(report, player_full_name)
        if row is not None:
            return row
    return None
