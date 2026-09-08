"""Build, save, and load the train-fit-once reference data the live
single-player pipeline (src/live_features.py) needs but must not compute
itself: the injury out-intervals (reconstructed from the historical Kaggle
transaction log) and the workload cohort-backfill baseline (fit on train
seasons only).

This is deliberately a snapshot-to-disk, not something recomputed at
service startup or per-request — see notebooks/20_build_live_reference_data.py
for the runnable driver and the reasoning for why. Re-run that notebook
manually whenever the Kaggle dataset is deliberately refreshed; nothing in
the running service ever calls build_live_reference_snapshot() itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineering import build_labeled_dataset, fit_cohort_baseline, compute_bmi_tercile_edges

DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "live_reference_snapshot.joblib"


@dataclass
class LiveReferenceData:
    """Frozen, train-fit-once inputs the live feature pipeline applies but
    never refits. `coverage_end` is surfaced directly in every API response
    as `injury_history_as_of` — the visible staleness caveat.

    The supplement_* fields are populated by
    notebooks/22_extend_injury_coverage.py (see src/injury_backfill.py),
    which extends injury coverage past the Kaggle source's own end using
    the NBA's official injury reports. When it runs, `coverage_end`
    ADVANCES to the supplemented date while `base_coverage_end` retains
    the original Kaggle cutoff — so "how current is this" and "how much of
    it came from the primary source" stay separately answerable rather
    than one silently overwriting the other. All default to None/empty so
    a snapshot built by notebook 20 alone (no supplement yet) is still
    valid, and so unpickling a snapshot written before these fields
    existed doesn't explode (see load_snapshot).
    """
    intervals: dict[int, tuple[np.ndarray, np.ndarray]]
    coverage_end: pd.Timestamp
    cohort_baseline_map: pd.Series
    overall_baseline: float
    bmi_tercile_edges: np.ndarray
    train_season_ids: list[str]
    snapshot_built_at: pd.Timestamp
    base_coverage_end: pd.Timestamp | None = None
    supplement_start: pd.Timestamp | None = None
    supplement_coverage: pd.DataFrame | None = None
    supplement_diagnostics: dict | None = None
    supplement_built_at: pd.Timestamp | None = None

    @property
    def coverage_gap(self) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """(start, end) of the span with NO injury coverage from either
        source, or None if there isn't one. Real and permanent as of
        writing: the Kaggle data stops at base_coverage_end, while the
        NBA's report CDN only retains ~8 months, so the stretch between
        them can't be recovered from either source. Surfaced through the
        API rather than papered over by simply advancing coverage_end.
        """
        if self.base_coverage_end is None or self.supplement_start is None:
            return None
        gap_start = self.base_coverage_end + pd.Timedelta(days=1)
        gap_end = self.supplement_start - pd.Timedelta(days=1)
        return (gap_start, gap_end) if gap_start <= gap_end else None

    def confidence_for(self, player_id: int) -> float | None:
        """Fraction of this player's missed games in the supplemented
        window that a report actually covered — None if no supplement has
        been built, or this player had no missed games in it (nothing to
        be uncertain about). See src/injury_backfill.py for exactly what
        counts as "resolved."
        """
        if self.supplement_coverage is None or len(self.supplement_coverage) == 0:
            return None
        row = self.supplement_coverage[self.supplement_coverage["PLAYER_ID"] == player_id]
        if len(row) == 0 or pd.isna(row.iloc[0]["confidence"]):
            return None
        return float(row.iloc[0]["confidence"])


def build_live_reference_snapshot(
    player_log: pd.DataFrame,
    team_log: pd.DataFrame,
    injury_raw: pd.DataFrame,
    static_players: pd.DataFrame,
    bio: pd.DataFrame,
    train_season_ids: list[str],
    chronic_window: int = 15,
) -> LiveReferenceData:
    """Pure assembly, no new logic beyond what notebooks 03/11 already do in
    batch form: build_labeled_dataset() gives the injury out-intervals (the
    same validated pipeline the training labels themselves come from);
    fit_cohort_baseline() gives the workload cold-start baseline.
    """
    _, diagnostics, intervals = build_labeled_dataset(player_log, team_log, injury_raw, static_players)
    cohort_baseline_map, overall_baseline = fit_cohort_baseline(
        player_log, bio, train_season_ids, chronic_window
    )
    bmi_tercile_edges = compute_bmi_tercile_edges(bio)
    return LiveReferenceData(
        intervals=intervals,
        coverage_end=pd.Timestamp(diagnostics["coverage_end"]),
        cohort_baseline_map=cohort_baseline_map,
        overall_baseline=overall_baseline,
        bmi_tercile_edges=bmi_tercile_edges,
        train_season_ids=train_season_ids,
        snapshot_built_at=pd.Timestamp.now(),
    )


def save_snapshot(ref: LiveReferenceData, path: Path | str = DEFAULT_SNAPSHOT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(ref, path)


def load_snapshot(path: Path | str = DEFAULT_SNAPSHOT_PATH) -> LiveReferenceData:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No live reference snapshot at {path}. Run notebooks/20_build_live_reference_data.py first."
        )
    ref = joblib.load(path)
    # Unpickling restores __dict__ directly without calling __init__, so a
    # snapshot written before the supplement_* fields existed comes back
    # missing them entirely rather than getting their defaults. Fill them
    # in here so older snapshots keep working instead of failing with an
    # AttributeError somewhere deep in the live pipeline.
    for field, default in [
        ("base_coverage_end", None), ("supplement_start", None), ("supplement_coverage", None),
        ("supplement_diagnostics", None), ("supplement_built_at", None),
    ]:
        if not hasattr(ref, field):
            setattr(ref, field, default)
    return ref
