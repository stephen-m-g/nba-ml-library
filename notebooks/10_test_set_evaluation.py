# %% [markdown]
# # Final test-set evaluation
#
# The one and only look at the held-out test set (2023-24 + 2024-25,
# 35,537 rows) — untouched through every round of training and tuning so
# far, specifically to give an honest, unbiased answer to "how good is
# this, really." No further tuning happens based on what's found here —
# that would defeat the point of holding it out in the first place.

# %%
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import LABEL_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy, MODELS_DIR
from src.evaluation import compute_calibration_bins

OUT_DIR = Path("../data/exploration") if Path.cwd().name == "notebooks" else Path("data/exploration")
pd.set_option("display.width", 120)

# %% [markdown]
# ## Load

# %%
features = pd.read_csv("../data/processed/features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})

train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
test_seasons = ["22023", "22024"]
_, _, test = time_based_split(features, train_seasons, val_seasons, test_seasons)
test_scored = exclude_gap_rows(test)

models = {label: joblib.load(MODELS_DIR / f"{label}_model.joblib") for label in LABEL_COLUMNS}
print(f"test: {len(test)} rows -> {len(test_scored)} after excluding possible_unlogged_gap")

# %% [markdown]
# ## Final metrics

# %%
results = []
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, label in zip(axes, LABEL_COLUMNS):
    X_test, y_test = prepare_xy(test_scored, label)
    probs = models[label].predict_proba(X_test)[:, 1]

    row = {
        "label": label,
        "test_positive_rate": y_test.mean(),
        "mean_predicted_prob": probs.mean(),
        "auc_roc": roc_auc_score(y_test, probs),
        "auc_pr": average_precision_score(y_test, probs),
        "brier": brier_score_loss(y_test, probs),
    }
    results.append(row)

    bins = compute_calibration_bins(y_test, probs, n_bins=8, strategy="quantile")
    max_val = max(bins["predicted_mean"].max(), bins["actual_mean"].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.4, label="perfect calibration")
    ax.plot(bins["predicted_mean"], bins["actual_mean"], "o-", label=label)
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("actual positive rate")
    ax.set_title(f"{label} (test set)")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(OUT_DIR / "test_calibration_curves.png", dpi=120)
plt.close()

results_df = pd.DataFrame(results)
print(results_df.round(4).to_string(index=False))

# %% [markdown]
# ## Compare against validation (sanity check, not tuning)
#
# Test performance should be in the same ballpark as validation — if it's
# dramatically worse, that would suggest the model was implicitly overfit
# to validation-era patterns despite never training on it directly (e.g.
# through repeated evaluation-driven feature choices). Not expecting a
# perfect match — different season, real basketball is noisy.

# %%
val_summary = pd.read_csv("../data/processed/training_results_summary.csv" if Path.cwd().name == "notebooks"
                           else "data/processed/training_results_summary.csv")
comparison = results_df[["label", "auc_roc", "auc_pr"]].merge(
    val_summary[["label", "model_auc_roc", "model_auc_pr"]], on="label"
).rename(columns={"model_auc_roc": "val_auc_roc", "model_auc_pr": "val_auc_pr",
                   "auc_roc": "test_auc_roc", "auc_pr": "test_auc_pr"})
print(comparison[["label", "val_auc_roc", "test_auc_roc", "val_auc_pr", "test_auc_pr"]].round(4).to_string(index=False))

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
results_df.to_csv(out_dir / "test_set_results.csv", index=False)
print(f"\nsaved to {out_dir / 'test_set_results.csv'}")
