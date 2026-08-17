"""Evaluate trained models: permutation importance, calibration, error
analysis by segment. Everything here takes a fitted classifier + held-out
X/y rather than doing any fitting itself — this module never trains anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss


def compute_permutation_importance(
    clf, X_val: pd.DataFrame, y_val: pd.Series, scoring: str = "average_precision",
    n_repeats: int = 10, random_state: int = 0,
) -> pd.DataFrame:
    """How much each feature actually matters: shuffle one column at a time
    (breaking its relationship with the label while leaving everything else
    intact) and measure how much `scoring` drops. A feature that matters a
    lot causes a big drop when shuffled; a feature the model ignores causes
    ~no drop. Needed here specifically because HistGradientBoostingClassifier
    doesn't expose feature_importances_, and this approach also works
    correctly through the CalibratedClassifierCV wrapper (it only needs
    predict/predict_proba, not any tree-internal attributes).

    scoring="average_precision" (i.e. AUC-PR) rather than accuracy — accuracy
    on a ~2-20% positive-rate target is dominated by the majority class and
    would make every feature look nearly useless.

    Returns a DataFrame sorted by importance_mean descending: feature,
    importance_mean, importance_std.
    """
    result = permutation_importance(
        clf, X_val, y_val, scoring=scoring, n_repeats=n_repeats, random_state=random_state
    )
    return pd.DataFrame({
        "feature": X_val.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def compute_calibration_bins(
    y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Reliability-diagram data: bucket predictions into n_bins, compare
    mean predicted probability to actual observed positive rate in each
    bucket. strategy="quantile" (equal number of rows per bin) rather than
    "uniform" (equal-width probability ranges) — with a rare positive class,
    equal-width bins in the high-probability range would end up with almost
    no rows and a noisy/meaningless rate estimate.

    Returns predicted_mean, actual_mean, count per bin — a well-calibrated
    model has predicted_mean ~= actual_mean in every bin, not just on average.
    """
    actual, predicted = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    # calibration_curve doesn't return counts directly; recompute bin membership to get them
    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)) if strategy == "quantile" \
        else np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1]), 0, len(predicted) - 1)
    counts = pd.Series(bin_idx).value_counts().sort_index().reindex(range(len(predicted)), fill_value=0)
    return pd.DataFrame({"predicted_mean": predicted, "actual_mean": actual, "count": counts.values})


def evaluate_by_segment(
    y_true: pd.Series, y_prob: np.ndarray, segment: pd.Series, min_segment_size: int = 200
) -> pd.DataFrame:
    """AUC-ROC / AUC-PR / Brier / calibration ratio computed separately
    within each value of `segment` (e.g. an age bucket, position, or
    boolean flag). Segments smaller than min_segment_size are dropped —
    metrics on a handful of rows are mostly noise, especially AUC-based ones
    which need a reasonable number of positives to be meaningful at all.
    """
    df = pd.DataFrame({"y_true": y_true.values, "y_prob": y_prob, "segment": segment.values})
    rows = []
    for seg_value, grp in df.groupby("segment"):
        if len(grp) < min_segment_size or grp["y_true"].nunique() < 2:
            continue
        rows.append({
            "segment": seg_value,
            "n": len(grp),
            "positive_rate": grp["y_true"].mean(),
            "mean_predicted_prob": grp["y_prob"].mean(),
            "auc_roc": roc_auc_score(grp["y_true"], grp["y_prob"]),
            "auc_pr": average_precision_score(grp["y_true"], grp["y_prob"]),
            "brier": brier_score_loss(grp["y_true"], grp["y_prob"]),
        })
    return pd.DataFrame(rows)


def bucket_age(age: pd.Series) -> pd.Series:
    """Age in years -> a small number of named buckets, for segment analysis."""
    return pd.cut(
        age, bins=[0, 23, 27, 31, 100], labels=["<=23 (young)", "24-27", "28-31", "32+ (veteran)"]
    )
