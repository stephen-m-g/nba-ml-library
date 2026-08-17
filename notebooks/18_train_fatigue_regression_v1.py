# %% [markdown]
# # Fatigue regression model v1: does averaging fix the noise problem?
#
# Direct follow-up to the classification v1 finding: predicting whether a
# single game crosses a decline threshold was weak (val AUC-ROC 0.52-0.54).
# This tests whether that was really a noise problem — predicting the
# *magnitude* of average deviation from baseline over the window, rather
# than a single-game threshold crossing. If R² comes back near 0, that's a
# real, informative answer too: it would mean the "dropoff" is close to
# genuinely random given what we can measure, exactly what needs testing
# honestly rather than assumed either way.

# %%
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import time_based_split, prepare_xy_regression, train_regressor

REGRESSION_TARGETS = ["w_1game", "w_3game", "w_10game"]
MODELS_DIR = Path("../models/fatigue") if Path.cwd().name == "notebooks" else Path("models/fatigue")
pd.set_option("display.width", 140)

# %% [markdown]
# ## Load and split

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
for label in REGRESSION_TARGETS:
    X_train, y_train = prepare_xy_regression(train, label)
    X_val, y_val = prepare_xy_regression(val, label)

    dummy = DummyRegressor(strategy="mean").fit(X_train, y_train)
    dummy_pred = dummy.predict(X_val)

    reg = train_regressor(X_train, y_train)
    pred = reg.predict(X_val)

    joblib.dump(reg, MODELS_DIR / f"{label}_model_v1.joblib")

    results.append({
        "label": label,
        "dummy_rmse": np.sqrt(mean_squared_error(y_val, dummy_pred)),
        "rmse": np.sqrt(mean_squared_error(y_val, pred)),
        "dummy_mae": mean_absolute_error(y_val, dummy_pred),
        "mae": mean_absolute_error(y_val, pred),
        "r2": r2_score(y_val, pred),  # dummy's r2 is ~0 by construction (predicting the mean), not worth a column
    })
    print(f"{label}: done, saved to {MODELS_DIR / f'{label}_model_v1.joblib'}")

results_df = pd.DataFrame(results)
print()
print(results_df.round(4).to_string(index=False))

# %% [markdown]
# ## Sanity checks
#
# R² is the key number here: it's the fraction of variance in actual
# performance deviation that the model explains. 0 = no better than always
# predicting the average (the dummy); 1 = perfect. Given how this project's
# entire framing has emphasized how much randomness is baked into single-
# game basketball outcomes, a small-but-positive R² would itself be a
# meaningful, honest result — not a disappointment to explain away.

# %%
# NOT a hard assert: unlike every classifier so far, it's a real, honest
# possibility that this model doesn't beat the dummy at all — that's the
# actual question this notebook is testing, not a bug to fail loudly on.
for _, row in results_df.iterrows():
    beat = row["rmse"] < row["dummy_rmse"]
    print(f"{row['label']}: {'beats' if beat else 'DOES NOT beat'} dummy on RMSE "
          f"({row['rmse']:.4f} vs {row['dummy_rmse']:.4f}), R²={row['r2']:.4f}")

# %% [markdown]
# ## Downstream validation: predicted dropoff vs. actual injury rate
#
# Same check as v1's classification model, same non-circularity (this
# model never saw injury during training) — bucket by predicted w_10game
# (more negative = worse predicted dropoff) and look at the actual y_10game
# (real injury) rate in each bucket.

# %%
X_val, y_val = prepare_xy_regression(val, "w_10game")
reg = joblib.load(MODELS_DIR / "w_10game_model_v1.joblib")
predicted_dropoff = reg.predict(X_val)

val_check = val.copy()
val_check["predicted_dropoff"] = predicted_dropoff
val_check["dropoff_quartile"] = pd.qcut(val_check["predicted_dropoff"], 4,
                                         labels=["Q1 (worst predicted)", "Q2", "Q3", "Q4 (best predicted)"])
injury_by_quartile = val_check.groupby("dropoff_quartile", observed=True)["y_10game"].agg(["mean", "count"])
print(injury_by_quartile)

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
results_df.to_csv(out_dir / "fatigue_regression_results_v1.csv", index=False)
injury_by_quartile.to_csv(out_dir / "fatigue_regression_vs_injury_validation.csv")
print("\nsaved regression results and downstream validation")
