# %% [markdown]
# # Model v3: travel features, then re-tune
#
# Two-step process, so the effect of the new features and the effect of
# re-tuning don't get tangled together:
# 1. Train with the travel features added, using v2's exact hyperparameters
#    — isolates what the new features alone are worth.
# 2. Re-run the hyperparameter search on this new feature set (a different
#    set of columns can call for a different tree structure), retrain, and
#    compare all three: v2 -> v3 (features only) -> v3 (retuned).

# %%
from pathlib import Path

import pandas as pd
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import (
    LABEL_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy, train_classifier,
    tune_hyperparameters, MODELS_DIR,
)

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
# ## Step 1: travel features with v2's hyperparameters (isolate the feature effect)

# %%
v2_search = pd.read_csv("../data/processed/hyperparameter_search_results.csv" if Path.cwd().name == "notebooks"
                         else "data/processed/hyperparameter_search_results.csv")
v2_best = v2_search.iloc[0]
v2_params = {
    "learning_rate": v2_best["learning_rate"],
    "max_iter": int(v2_best["max_iter"]),
    "max_depth": None if pd.isna(v2_best["max_depth"]) else int(v2_best["max_depth"]),
    "max_leaf_nodes": int(v2_best["max_leaf_nodes"]),
    "min_samples_leaf": int(v2_best["min_samples_leaf"]),
    "l2_regularization": v2_best["l2_regularization"],
}
print("v2 hyperparameters (reused for step 1):", v2_params)

step1_results = []
for label in LABEL_COLUMNS:
    X_train, y_train = prepare_xy(train, label)
    X_val, y_val = prepare_xy(val_scored, label)
    clf = train_classifier(X_train, y_train, class_weight="balanced", **v2_params)
    probs = clf.predict_proba(X_val)[:, 1]
    step1_results.append({
        "label": label, "auc_roc": roc_auc_score(y_val, probs),
        "auc_pr": average_precision_score(y_val, probs), "brier": brier_score_loss(y_val, probs),
    })
step1_df = pd.DataFrame(step1_results)
print("\nstep 1 (travel features, v2 hyperparameters):")
print(step1_df.round(4).to_string(index=False))

# %% [markdown]
# ## Step 2: re-tune hyperparameters on the new feature set

# %%
X_train, y_train = prepare_xy(train, "y_10game")
X_val, y_val = prepare_xy(val_scored, "y_10game")
v3_params, v3_search = tune_hyperparameters(X_train, y_train, X_val, y_val, n_candidates=25, random_state=1)
print("v3 (retuned) hyperparameters:", v3_params)
print(f"\ntop 5 candidates:")
print(v3_search.head(5).to_string(index=False))

# %% [markdown]
# ## Train final v3 models with retuned hyperparameters

# %%
MODELS_DIR.mkdir(parents=True, exist_ok=True)
v3_results = []
for label in LABEL_COLUMNS:
    X_train, y_train = prepare_xy(train, label)
    X_val, y_val = prepare_xy(val_scored, label)

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_probs = dummy.predict_proba(X_val)[:, 1]

    clf = train_classifier(X_train, y_train, class_weight="balanced", **v3_params)
    probs = clf.predict_proba(X_val)[:, 1]
    joblib.dump(clf, MODELS_DIR / f"{label}_model_v3.joblib")

    v3_results.append({
        "label": label, "val_positive_rate": y_val.mean(), "mean_predicted_prob": probs.mean(),
        "dummy_auc_roc": roc_auc_score(y_val, dummy_probs), "auc_roc": roc_auc_score(y_val, probs),
        "dummy_auc_pr": average_precision_score(y_val, dummy_probs), "auc_pr": average_precision_score(y_val, probs),
        "brier": brier_score_loss(y_val, probs),
    })
    print(f"{label}: saved to {MODELS_DIR / f'{label}_model_v3.joblib'}")

v3_df = pd.DataFrame(v3_results)

# %% [markdown]
# ## Full comparison: v2 -> v3 (features only) -> v3 (retuned)

# %%
v2_results = pd.read_csv("../data/processed/training_results_summary_v2.csv" if Path.cwd().name == "notebooks"
                          else "data/processed/training_results_summary_v2.csv")

comparison = v2_results[["label", "auc_roc", "auc_pr"]].rename(
    columns={"auc_roc": "v2_auc_roc", "auc_pr": "v2_auc_pr"}
).merge(
    step1_df[["label", "auc_roc", "auc_pr"]].rename(columns={"auc_roc": "v3_features_auc_roc", "auc_pr": "v3_features_auc_pr"}),
    on="label",
).merge(
    v3_df[["label", "auc_roc", "auc_pr"]].rename(columns={"auc_roc": "v3_final_auc_roc", "auc_pr": "v3_final_auc_pr"}),
    on="label",
)
print(comparison.round(4).to_string(index=False))

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
v3_df.to_csv(out_dir / "training_results_summary_v3.csv", index=False)
v3_search.to_csv(out_dir / "hyperparameter_search_results_v3.csv", index=False)
comparison.to_csv(out_dir / "v2_v3_comparison.csv", index=False)
print("\nsaved v3 results, search log, and full comparison")
