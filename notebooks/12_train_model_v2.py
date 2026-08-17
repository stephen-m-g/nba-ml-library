# %% [markdown]
# # Model v2: new features + hyperparameter tuning
#
# Combines everything from this session into one retrain: bmi,
# cohort-backfilled workload (min_avg_acute/chronic/acute_chronic_ratio),
# plus a hyperparameter search that's never been done — every model so far
# used scikit-learn's defaults.
#
# Saved as `{label}_model_v2.joblib`, alongside (not overwriting) the v1
# models, so the two can be compared directly.

# %%
from pathlib import Path

import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import (
    LABEL_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy, train_classifier,
    tune_hyperparameters, MODELS_DIR,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

pd.set_option("display.width", 120)

# %% [markdown]
# ## Load and split

# %%
features = pd.read_csv("../data/processed/features.csv" if Path.cwd().name == "notebooks"
                        else "data/processed/features.csv", dtype={"GAME_ID": str, "SEASON_ID": str})
train_seasons = ["22016", "22017", "22018", "22019", "22020", "22021"]
val_seasons = ["22022"]
test_seasons = ["22023", "22024"]
train, val, test = time_based_split(features, train_seasons, val_seasons, test_seasons)
val_scored = exclude_gap_rows(val)

# %% [markdown]
# ## Hyperparameter search
#
# Tuned on y_10game (the strongest, least-noisy signal of the three) and
# shared across all three targets rather than searching separately for
# each — a reasonable compromise given search cost, and y_10game's larger
# positive class makes its val AUC-PR estimate the least noisy to search
# against in the first place.

# %%
X_train, y_train = prepare_xy(train, "y_10game")
X_val, y_val = prepare_xy(val_scored, "y_10game")

best_params, search_results = tune_hyperparameters(X_train, y_train, X_val, y_val, n_candidates=25)
print("best params found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\ntop 5 candidates by val AUC-PR:")
print(search_results.head(5).to_string(index=False))
print(f"\nbest found val AUC-PR: {search_results['val_auc_pr'].max():.4f} "
      f"(range across all {len(search_results)} candidates tried: "
      f"{search_results['val_auc_pr'].min():.4f} - {search_results['val_auc_pr'].max():.4f})")

# %% [markdown]
# ## Train all three v2 models with the tuned hyperparameters

# %%
results = []
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for label in LABEL_COLUMNS:
    X_train, y_train = prepare_xy(train, label)
    X_val, y_val = prepare_xy(val_scored, label)

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_val)[:, 1]

    clf = train_classifier(X_train, y_train, class_weight="balanced", **best_params)
    probs = clf.predict_proba(X_val)[:, 1]

    joblib.dump(clf, MODELS_DIR / f"{label}_model_v2.joblib")

    results.append({
        "label": label,
        "val_positive_rate": y_val.mean(),
        "mean_predicted_prob": probs.mean(),
        "auc_roc": roc_auc_score(y_val, probs),
        "auc_pr": average_precision_score(y_val, probs),
        "brier": brier_score_loss(y_val, probs),
    })
    print(f"{label}: done, saved to {MODELS_DIR / f'{label}_model_v2.joblib'}")

v2_results = pd.DataFrame(results)

# %% [markdown]
# ## Compare v1 vs v2

# %%
v1_results = pd.read_csv("../data/processed/training_results_summary.csv" if Path.cwd().name == "notebooks"
                          else "data/processed/training_results_summary.csv")
v1_renamed = v1_results[["label", "model_auc_roc", "model_auc_pr", "mean_predicted_prob"]].rename(
    columns={"model_auc_roc": "auc_roc", "model_auc_pr": "auc_pr"}
)
comparison = v1_renamed.merge(
    v2_results[["label", "auc_roc", "auc_pr", "mean_predicted_prob"]], on="label", suffixes=("_v1", "_v2")
)
comparison["auc_roc_gain"] = comparison["auc_roc_v2"] - comparison["auc_roc_v1"]
comparison["auc_pr_gain"] = comparison["auc_pr_v2"] - comparison["auc_pr_v1"]
print(comparison.round(4).to_string(index=False))

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
v2_results.to_csv(out_dir / "training_results_summary_v2.csv", index=False)
search_results.to_csv(out_dir / "hyperparameter_search_results.csv", index=False)
print(f"\nsaved v2 results and full search log")
