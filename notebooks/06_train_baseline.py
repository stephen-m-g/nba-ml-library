# %% [markdown]
# # First-pass training: y_1game baseline
#
# Proof-of-concept run on the hardest of the three targets (1-game, ~2.4%
# positive) before committing to a full three-model training pass. Goals:
# confirm the pipeline works end to end, and get real numbers to ground the
# open decisions (split strategy, class-imbalance handling, gap-row
# handling) rather than deciding them in the abstract.
#
# Split: time-based by season, not random — see time_based_split's docstring
# in src/training.py for why. train=2016-17..2021-22 (6 seasons, 149,833
# rows), val=2022-23 (25,895 rows), test=2023-24+2024-25 (35,537 rows, held
# out untouched until final evaluation, not used here).

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import FEATURE_COLUMNS, time_based_split, prepare_xy, train_classifier, MODELS_DIR

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load and split

# %%
features = pd.read_csv("../data/processed/features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})

train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
test_seasons = ["22023", "22024"]  # held out, not touched in this notebook

train, val, test = time_based_split(features, train_seasons, val_seasons, test_seasons)
print(f"train: {len(train)} rows ({train['y_1game'].mean():.2%} positive)")
print(f"val:   {len(val)} rows ({val['y_1game'].mean():.2%} positive)")
print(f"test:  {len(test)} rows ({test['y_1game'].mean():.2%} positive) — held out, not used below")

# %% [markdown]
# ## Baseline: DummyClassifier
#
# Always predicts the training set's base rate, regardless of features. Any
# real model needs to beat this by a meaningful margin, or it isn't actually
# learning anything from the features.

# %%
X_train, y_train = prepare_xy(train, "y_1game")
X_val, y_val = prepare_xy(val, "y_1game")

dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
dummy_probs = dummy.predict_proba(X_val)[:, 1]

print(f"dummy AUC-ROC:  {roc_auc_score(y_val, dummy_probs):.3f}  (0.5 = no better than random)")
print(f"dummy AUC-PR:   {average_precision_score(y_val, dummy_probs):.3f}  (baseline = the positive rate, {y_val.mean():.3f})")
print(f"dummy Brier:    {brier_score_loss(y_val, dummy_probs):.4f}  (lower is better; 0 = perfect)")

# %% [markdown]
# ## First-pass model: HistGradientBoostingClassifier
#
# Gradient-boosted trees: the standard choice for tabular data like this —
# handles the mix of continuous (age, minutes) and boolean (position flags,
# back-to-back) features without preprocessing, handles the real NaNs in
# our data natively (no imputation step), and generally outperforms a
# single logistic regression on feature-engineered tabular data with
# nonlinear relationships (e.g. rest mattering differently depending on age).
#
# class_weight="balanced" reweights the loss to counteract the ~2.4% positive
# rate. First look at what that does *before* any calibration correction,
# since the size of the problem is worth seeing directly rather than fixing
# silently.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

uncalibrated = HistGradientBoostingClassifier(class_weight="balanced", random_state=0).fit(X_train, y_train)
uncalibrated_probs = uncalibrated.predict_proba(X_val)[:, 1]

print(f"true val positive rate:            {y_val.mean():.4f}")
print(f"uncalibrated mean predicted prob:  {uncalibrated_probs.mean():.4f}  <- badly distorted by class_weight")
print(f"uncalibrated AUC-ROC:  {roc_auc_score(y_val, uncalibrated_probs):.3f}")
print(f"uncalibrated AUC-PR:   {average_precision_score(y_val, uncalibrated_probs):.3f}")
print(f"uncalibrated Brier:    {brier_score_loss(y_val, uncalibrated_probs):.4f}  <- much worse than the dummy's "
      f"{brier_score_loss(y_val, dummy_probs):.4f}, because the probabilities themselves are wrong even "
      f"though the ranking improved")

# %% [markdown]
# class_weight="balanced" told the model "treat the classes as equally
# important" — it did, and as a direct result its raw output probabilities
# reflect a roughly-balanced world, not the real ~2.6% one. Ranking ability
# improved (AUC-ROC/AUC-PR both up), but the actual probability values are
# unusable as-is. Fix: `train_classifier` wraps the same model in
# CalibratedClassifierCV (5-fold Platt scaling, fit only on the training
# data — val stays untouched) to correct the output scale back to reality
# without giving up the ranking improvement.

# %%
clf = train_classifier(X_train, y_train, class_weight="balanced")
probs = clf.predict_proba(X_val)[:, 1]

print(f"calibrated mean predicted prob:  {probs.mean():.4f}  <- much closer to the true {y_val.mean():.4f}")
print(f"calibrated AUC-ROC:  {roc_auc_score(y_val, probs):.3f}")
print(f"calibrated AUC-PR:   {average_precision_score(y_val, probs):.3f}")
print(f"calibrated Brier:    {brier_score_loss(y_val, probs):.4f}  <- back in line with the dummy baseline")

# %% [markdown]
# ## Feature importance
#
# HistGradientBoostingClassifier doesn't expose feature_importances_ the way
# e.g. RandomForestClassifier does (and CalibratedClassifierCV wraps it,
# which would complicate direct access further even if it did) — that's a
# proper evaluation.py job (permutation importance: shuffle one column at a
# time, see how much performance drops), not done here.

# %% [markdown]
# ## Save

# %%
import joblib
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, MODELS_DIR / "y_1game_baseline.joblib")
print(f"saved to {MODELS_DIR / 'y_1game_baseline.joblib'}")
