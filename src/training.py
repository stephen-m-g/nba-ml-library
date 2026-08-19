"""Train the y_1game / y_3game / y_10game injury-probability models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "injury_risk"

FEATURE_COLUMNS = [
    "days_rest", "is_back_to_back", "is_long_layoff",
    "PACE", "OFF_RATING", "DEF_RATING",
    "height_inches", "weight_lbs", "bmi", "plays_guard", "plays_forward", "plays_center",
    "age_years", "experience_years",
    "min_avg_acute", "min_avg_chronic", "acute_chronic_ratio", "games_last_14d",
    "games_since_extended_return", "is_returning_from_extended_absence",
    "career_injury_count", "days_missed_last_365d",
    "travel_distance_last_game", "travel_distance_last_14d",
    "current_road_trip_length", "days_since_home",
]

LABEL_COLUMNS = ["y_1game", "y_3game", "y_10game"]


def time_based_split(
    features: pd.DataFrame, train_seasons: list[str], val_seasons: list[str], test_seasons: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by SEASON_ID rather than randomly. Consecutive games from the
    same player are highly correlated (age/team/height barely change game to
    game), so a random row-level split would let the model see near-
    duplicates of test rows during training — that inflates apparent
    performance in a way that wouldn't hold up on genuinely new data. A
    time-based split (train on earlier seasons, evaluate on later ones) also
    matches how this model would actually be used: predicting forward, not
    interpolating within a season it's already seen.
    """
    train = features[features["SEASON_ID"].isin(train_seasons)]
    val = features[features["SEASON_ID"].isin(val_seasons)]
    test = features[features["SEASON_ID"].isin(test_seasons)]
    return train, val, test


def exclude_gap_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop possible_unlogged_gap rows — use on val/test before scoring, not
    on train. The label on these rows is usually right but sometimes an
    undercount (see project memory / notebooks/05), which is fine as noisy
    training signal but shouldn't be allowed to distort how we judge the
    model's actual performance.
    """
    return df[~df["possible_unlogged_gap"]]


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix ready for sklearn: bool feature columns cast to float
    (the classifier handles this fine either way, but keeping dtypes
    uniform and explicit avoids surprises), NaNs left as-is —
    HistGradientBoostingClassifier handles missing values natively, no
    imputation needed. Shared by prepare_xy/prepare_xy_regression (batch,
    labeled) and src/predict.py (live, unlabeled single-row inference) so
    both stay in sync automatically rather than duplicating this logic.
    """
    X = df[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(float)
    return X


def prepare_xy(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """X, y ready for sklearn — see prepare_X for the X side."""
    X = prepare_X(df)
    y = df[label].astype(int)
    return X, y


def prepare_xy_regression(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, pd.Series]:
    """Same X prep as prepare_xy, but keeps y as a float — prepare_xy's
    y.astype(int) would silently truncate a continuous regression target
    (e.g. w_1game=-0.7 -> 0), which is wrong in a way that wouldn't raise
    an error, just quietly corrupt the target. Kept as a separate function
    rather than a flag on prepare_xy so that mistake isn't possible to make.
    """
    X = prepare_X(df)
    y = df[label].astype(float)
    return X, y


HYPERPARAM_SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 300],
    "max_depth": [None, 4, 6, 8],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [20, 50, 100],
    "l2_regularization": [0.0, 0.1, 1.0],
}


def tune_hyperparameters(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
    n_candidates: int = 25, random_state: int = 0,
) -> tuple[dict, pd.DataFrame]:
    """Random search over HYPERPARAM_SEARCH_SPACE, scored on AUC-PR against
    a genuinely held-out validation set — not internal k-fold cross-
    validation. Same reasoning as time_based_split: rows from the same
    player close in time are highly correlated, so random CV folds risk
    letting a model see near-duplicates of its own validation fold, which
    would make the search overly optimistic about candidates that just
    memorize player-specific quirks.

    Searches WITHOUT calibration — ranking metrics like AUC-PR only care
    about relative order, which calibration doesn't change (it rescales
    probability values, not rank), so skipping it here makes each candidate
    ~5x faster to fit. Calibration gets applied once, to the final winner.

    Returns (best_params, results_df) — results_df has every candidate
    tried and its val AUC-PR, so the search is inspectable, not a black box.
    """
    rng = np.random.RandomState(random_state)
    keys = list(HYPERPARAM_SEARCH_SPACE.keys())
    results = []
    best_score, best_params = -1.0, None

    for _ in range(n_candidates):
        params = {k: HYPERPARAM_SEARCH_SPACE[k][rng.randint(len(HYPERPARAM_SEARCH_SPACE[k]))] for k in keys}
        clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=random_state, **params)
        clf.fit(X_train, y_train)
        score = average_precision_score(y_val, clf.predict_proba(X_val)[:, 1])
        results.append({**params, "val_auc_pr": score})
        if score > best_score:
            best_score, best_params = score, params

    results_df = pd.DataFrame(results).sort_values("val_auc_pr", ascending=False).reset_index(drop=True)
    return best_params, results_df


def train_classifier(
    X_train: pd.DataFrame, y_train: pd.Series, class_weight: str | None = "balanced", random_state: int = 0,
    **hyperparams,
) -> CalibratedClassifierCV:
    """class_weight="balanced" (default) meaningfully improves the model's
    ability to RANK players by risk on this imbalanced data, confirmed by a
    real before/after test: AUC-ROC 0.500 -> 0.625 on the y_1game target.
    But reweighting the loss also badly distorts the actual probability
    values it outputs — same test, mean predicted probability came out at
    45% against a true 2.6% base rate. Wrapping in CalibratedClassifierCV
    (5-fold Platt scaling, done entirely within the training data — val/test
    stay untouched) fixes that: same test, mean predicted probability came
    back down to 2.3%, Brier score back in line with a trivial baseline,
    while keeping nearly all of the ranking improvement (AUC-ROC 0.614).
    Skipping this step would mean the model "works" by ranking metrics but
    outputs probabilities that don't mean what they claim to — not
    acceptable for a project whose whole output is meant to be a probability.
    """
    # isotonic vs sigmoid: tested both on y_10game — isotonic came out marginally
    # better (Brier 0.1665 vs 0.1669) and isotonic's flexibility needs a fair
    # amount of data to avoid overfitting, which our ~150K-row training set
    # comfortably supports. Neither method fixed the deeper issue that
    # predictions were too "compressed" relative to true risk — calibration
    # can only rescale the model's existing ranking, not manufacture
    # separation the base model isn't finding in the first place.
    base = HistGradientBoostingClassifier(class_weight=class_weight, random_state=random_state, **hyperparams)
    calibrated = CalibratedClassifierCV(base, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)
    return calibrated


def train_regressor(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 0, **hyperparams,
) -> HistGradientBoostingRegressor:
    """Same tree-based approach as train_classifier, for a continuous
    target. No class_weight (no classes to weight) and no calibration step
    (calibration is specifically about correcting predicted probability
    values to match true rates — meaningless for a regression output, which
    isn't a probability at all).
    """
    reg = HistGradientBoostingRegressor(random_state=random_state, **hyperparams)
    reg.fit(X_train, y_train)
    return reg
