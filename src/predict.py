"""Load the trained injury-risk models and apply the elevated-risk
thresholds to a live feature row. See project-elevated-risk-design memory:
thresholds are a fixed, absolute bar — fit once on training-set
predictions, never periodically re-quantiled.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV

from src.training import LABEL_COLUMNS, MODELS_DIR, prepare_X

# Full precision, from data/processed/elevated_risk_threshold_results.csv
# (notebooks/19_injury_elevated_risk_threshold.py) — top quartile of TRAIN
# predictions, fit once. This is their one importable home; do not
# re-derive or re-quantile these live.
ELEVATED_RISK_THRESHOLDS = {
    "y_1game": 0.03430758405019586,
    "y_3game": 0.0871784846202458,
    "y_10game": 0.23539369156565648,
}


def load_models(models_dir: Path = MODELS_DIR) -> dict[str, CalibratedClassifierCV]:
    """Load the three v3 injury-risk models once — call at service startup, not per-request."""
    return {label: joblib.load(Path(models_dir) / f"{label}_model_v3.joblib") for label in LABEL_COLUMNS}


def predict_elevated_risk(features: pd.DataFrame, models: dict[str, CalibratedClassifierCV]) -> dict[str, dict]:
    """features: one row, columns == training.FEATURE_COLUMNS (see
    src.live_features.assemble_live_features). Returns, per label:
    {"probability": ..., "threshold": ..., "elevated_risk": ...}.
    """
    X = prepare_X(features)
    results = {}
    for label, model in models.items():
        probability = float(model.predict_proba(X)[:, 1][0])
        threshold = ELEVATED_RISK_THRESHOLDS[label]
        results[label] = {
            "probability": probability,
            "threshold": threshold,
            "elevated_risk": probability >= threshold,
        }
    return results
