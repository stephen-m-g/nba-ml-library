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
    """
    intervals: dict[int, tuple[np.ndarray, np.ndarray]]
    coverage_end: pd.Timestamp
    cohort_baseline_map: pd.Series
    overall_baseline: float
    bmi_tercile_edges: np.ndarray
    train_season_ids: list[str]
    snapshot_built_at: pd.Timestamp


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
    return joblib.load(path)
