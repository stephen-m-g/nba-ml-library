# %% [markdown]
# # Fatigue model v1: train, evaluate, and check against injury (downstream)
#
# Proof of concept, matching the injury model's own v1 (notebook 07):
# default hyperparameters, no tuning yet — establish whether the concept
# holds up before investing in refinement. Same split, same feature
# preparation, same classifier + calibration approach — all reused directly
# from src/training.py, since FEATURE_COLUMNS is identical between the two
# models. Only the label (z_Ngame instead of y_Ngame) and the models
# directory (models/fatigue/ instead of models/injury_risk/) differ.
#
# Last step is the actual point of the hybrid design: this model is never
# trained on injury at all, so checking whether its predicted fatigue
# probability correlates with *actual* future injury rates is a genuine,
# non-circular test of whether it's capturing something real.

# %%
from pathlib import Path

import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import time_based_split, prepare_xy, train_classifier

FATIGUE_LABEL_COLUMNS = ["z_1game", "z_3game", "z_10game"]
MODELS_DIR = Path("../models/fatigue") if Path.cwd().name == "notebooks" else Path("models/fatigue")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
pd.set_option("display.width", 140)

# %% [markdown]
# ## Load and split (same season boundaries as the injury model, for consistency)

# %%
features = pd.read_csv("../data/processed/fatigue_features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/fatigue_features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
test_seasons = ["22023", "22024"]
train, val, test = time_based_split(features, train_seasons, val_seasons, test_seasons)
print(f"train: {len(train)}, val: {len(val)}, test: {len(test)}")

# %% [markdown]
# ## Train + evaluate each target

# %%
results = []
for label in FATIGUE_LABEL_COLUMNS:
    X_train, y_train = prepare_xy(train, label)
    X_val, y_val = prepare_xy(val, label)

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_val)[:, 1]

    clf = train_classifier(X_train, y_train, class_weight="balanced")
    probs = clf.predict_proba(X_val)[:, 1]

    joblib.dump(clf, MODELS_DIR / f"{label}_model_v1.joblib")

    results.append({
        "label": label, "val_positive_rate": y_val.mean(), "mean_predicted_prob": probs.mean(),
        "dummy_auc_roc": roc_auc_score(y_val, dummy_probs), "auc_roc": roc_auc_score(y_val, probs),
        "dummy_auc_pr": average_precision_score(y_val, dummy_probs), "auc_pr": average_precision_score(y_val, probs),
        "brier": brier_score_loss(y_val, probs),
    })
    print(f"{label}: done, saved to {MODELS_DIR / f'{label}_model_v1.joblib'}")

results_df = pd.DataFrame(results)
print()
print(results_df.round(4).to_string(index=False))

# %% [markdown]
# ## Sanity checks

# %%
for _, row in results_df.iterrows():
    assert row["auc_roc"] > row["dummy_auc_roc"], f"{row['label']}: did not beat dummy on AUC-ROC"
    assert row["auc_pr"] > row["dummy_auc_pr"], f"{row['label']}: did not beat dummy on AUC-PR"
print("all three beat their dummy baseline on both ranking metrics")

# %% [markdown]
# ## Downstream validation: does predicted fatigue correlate with real injury?
#
# Using z_10game's model (strongest signal, most data to work with) on the
# validation set. Bucket players into quartiles by predicted decline
# probability, then check the *actual* y_10game (real injury) rate within
# each bucket — this model never saw y_10game during training, so any real
# relationship here is genuine signal, not something it was fit to produce.

# %%
X_val, y_val = prepare_xy(val, "z_10game")
clf = joblib.load(MODELS_DIR / "z_10game_model_v1.joblib")
fatigue_probs = clf.predict_proba(X_val)[:, 1]

val_check = val.copy()
val_check["predicted_fatigue_prob"] = fatigue_probs
val_check["fatigue_quartile"] = pd.qcut(val_check["predicted_fatigue_prob"], 4,
                                         labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"])
injury_by_quartile = val_check.groupby("fatigue_quartile", observed=True)["y_10game"].agg(["mean", "count"])
print(injury_by_quartile)

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
results_df.to_csv(out_dir / "fatigue_training_results_v1.csv", index=False)
injury_by_quartile.to_csv(out_dir / "fatigue_vs_injury_validation.csv")
print("\nsaved training results and downstream validation")
