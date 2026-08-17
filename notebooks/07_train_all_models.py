# %% [markdown]
# # Train all three models: y_1game, y_3game, y_10game
#
# Applies the recipe validated in 06_train_baseline.py (time-based split,
# class-weighted + calibrated HistGradientBoostingClassifier) to all three
# targets, with the gap-row handling decided this session: train on
# possible_unlogged_gap rows (noisy-but-mostly-right signal), exclude them
# from validation scoring (don't let a known-uncertain label distort how we
# judge real performance).

# %%
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import joblib

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import (
    LABEL_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy, train_classifier, MODELS_DIR,
)

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load and split

# %%
features = pd.read_csv("../data/processed/features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})

train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
test_seasons = ["22023", "22024"]  # held out entirely, not touched here

train, val, test = time_based_split(features, train_seasons, val_seasons, test_seasons)
val_scored = exclude_gap_rows(val)  # eval only, train keeps the gap rows
print(f"train: {len(train)} rows (includes possible_unlogged_gap rows)")
print(f"val:   {len(val)} rows -> {len(val_scored)} after excluding possible_unlogged_gap for scoring")
print(f"test:  {len(test)} rows — held out, not used below")

# %% [markdown]
# ## Train + evaluate each target

# %%
results = []
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for label in LABEL_COLUMNS:
    X_train, y_train = prepare_xy(train, label)
    X_val, y_val = prepare_xy(val_scored, label)

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_val)[:, 1]

    clf = train_classifier(X_train, y_train, class_weight="balanced")
    probs = clf.predict_proba(X_val)[:, 1]

    joblib.dump(clf, MODELS_DIR / f"{label}_model.joblib")

    results.append({
        "label": label,
        "train_positive_rate": y_train.mean(),
        "val_positive_rate": y_val.mean(),
        "mean_predicted_prob": probs.mean(),
        "dummy_auc_roc": roc_auc_score(y_val, dummy_probs),
        "model_auc_roc": roc_auc_score(y_val, probs),
        "dummy_auc_pr": average_precision_score(y_val, dummy_probs),
        "model_auc_pr": average_precision_score(y_val, probs),
        "dummy_brier": brier_score_loss(y_val, dummy_probs),
        "model_brier": brier_score_loss(y_val, probs),
    })
    print(f"{label}: done, saved to {MODELS_DIR / f'{label}_model.joblib'}")

results_df = pd.DataFrame(results)
print()
print(results_df.round(4).to_string(index=False))

# %% [markdown]
# ## Sanity checks

# %%
# mean predicted probability should track the true val rate reasonably closely
# for a well-calibrated model (won't be exact, but shouldn't be off by miles
# the way the uncalibrated version was in notebook 06)
for _, row in results_df.iterrows():
    ratio = row["mean_predicted_prob"] / row["val_positive_rate"]
    print(f"{row['label']}: predicted/actual rate ratio = {ratio:.2f} (1.0 = perfectly calibrated on average)")

# every model should beat its dummy baseline on ranking metrics, or something's wrong
for _, row in results_df.iterrows():
    assert row["model_auc_roc"] > row["dummy_auc_roc"], f"{row['label']}: model did not beat dummy on AUC-ROC"
    assert row["model_auc_pr"] > row["dummy_auc_pr"], f"{row['label']}: model did not beat dummy on AUC-PR"
print("\nall three models beat their dummy baseline on both ranking metrics")

# %% [markdown]
# ## Save results summary

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
results_df.to_csv(out_dir / "training_results_summary.csv", index=False)
print(f"saved summary to {out_dir / 'training_results_summary.csv'}")
