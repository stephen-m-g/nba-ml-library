# %% [markdown]
# # Evaluate the three trained models
#
# Loads the saved models from notebooks/07 and digs into: which features are
# actually pulling weight (permutation importance), how trustworthy the
# probabilities are bin-by-bin (calibration/reliability diagrams, not just
# the single mean-vs-actual number from before), and whether performance
# holds up consistently across different kinds of players (segment analysis)
# rather than being an average that hides real blind spots.
#
# Still using validation (2022-23), not test — this is exploratory
# evaluation that might feed back into further tuning, so the held-out test
# set stays untouched until we're actually done iterating.

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import LABEL_COLUMNS, FEATURE_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy, MODELS_DIR
from src.evaluation import compute_permutation_importance, compute_calibration_bins, evaluate_by_segment, bucket_age

OUT_DIR = Path("../data/exploration") if Path.cwd().name == "notebooks" else Path("data/exploration")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Load

# %%
features = pd.read_csv("../data/processed/features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
_, val, _ = time_based_split(features, train_seasons, val_seasons, [])
val_scored = exclude_gap_rows(val)

models = {label: joblib.load(MODELS_DIR / f"{label}_model.joblib") for label in LABEL_COLUMNS}
print(f"val (scored): {len(val_scored)} rows")
print(f"models loaded: {list(models.keys())}")

# %% [markdown]
# ## Permutation importance
#
# Shuffle one feature at a time, see how much AUC-PR drops. A bigger drop
# means the model actually relies on that feature; ~0 means it's along for
# the ride.

# %%
importance_results = {}
for label in LABEL_COLUMNS:
    X_val, y_val = prepare_xy(val_scored, label)
    imp = compute_permutation_importance(models[label], X_val, y_val, n_repeats=5)
    importance_results[label] = imp
    print(f"\n{label} — top 8 features by importance:")
    print(imp.head(8).to_string(index=False))

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, LABEL_COLUMNS):
    top = importance_results[label].head(10).sort_values("importance_mean")
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_title(label)
    ax.set_xlabel("AUC-PR drop when shuffled")
plt.tight_layout()
plt.savefig(OUT_DIR / "permutation_importance.png", dpi=120)
plt.close()

# %% [markdown]
# ## Calibration (reliability diagrams)

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, LABEL_COLUMNS):
    X_val, y_val = prepare_xy(val_scored, label)
    probs = models[label].predict_proba(X_val)[:, 1]
    bins = compute_calibration_bins(y_val, probs, n_bins=8, strategy="quantile")
    print(f"\n{label} calibration bins:")
    print(bins.round(4).to_string(index=False))

    max_val = max(bins["predicted_mean"].max(), bins["actual_mean"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="perfect calibration")
    ax.plot(bins["predicted_mean"], bins["actual_mean"], "o-", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("actual positive rate")
    ax.set_title(label)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / "calibration_curves.png", dpi=120)
plt.close()

# %% [markdown]
# ## Segment analysis
#
# Does performance hold up across different kinds of players, or is the
# headline AUC hiding real unevenness? Checking: age bucket, position,
# back-to-back status. Using y_10game for this (the strongest model, so any
# unevenness found here is a real property of the signal, not just noise
# from a model that barely beats random overall).

# %%
X_val, y_val = prepare_xy(val_scored, "y_10game")
probs = models["y_10game"].predict_proba(X_val)[:, 1]

age_buckets = bucket_age(val_scored["age_years"])
print("By age bucket:")
print(evaluate_by_segment(y_val, probs, age_buckets).round(4).to_string(index=False))

print("\nBy position:")
position = np.select(
    [val_scored["plays_center"], val_scored["plays_forward"], val_scored["plays_guard"]],
    ["center", "forward", "guard"], default="unknown",
)
print(evaluate_by_segment(y_val, probs, pd.Series(position)).round(4).to_string(index=False))

print("\nBy back-to-back status:")
print(evaluate_by_segment(y_val, probs, val_scored["is_back_to_back"]).round(4).to_string(index=False))

# %% [markdown]
# ## Bonus diagnostic: performance on the excluded gap rows
#
# Not part of primary scoring, but worth a look — if possible_unlogged_gap
# is actually marking rows with less trustworthy labels, model performance
# there should look different (most likely worse, since the label itself is
# noisier) than on the clean validation set above.

# %%
gap_rows = val[val["possible_unlogged_gap"]]
if len(gap_rows) >= 50:
    X_gap, y_gap = prepare_xy(gap_rows, "y_10game")
    gap_probs = models["y_10game"].predict_proba(X_gap)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    print(f"gap rows (n={len(gap_rows)}): AUC-ROC={roc_auc_score(y_gap, gap_probs):.3f}, "
          f"AUC-PR={average_precision_score(y_gap, gap_probs):.3f}")
    print(f"clean val rows: AUC-ROC and AUC-PR from the main results above, for comparison")
else:
    print(f"only {len(gap_rows)} gap rows in val — too few for a meaningful comparison")
