# %% [markdown]
# # Injury model, redefined: "elevated risk" binary flag
#
# Returning to the injury model (validated across v1-v3, real monotonic
# signal throughout) with a simpler deliverable: instead of a raw
# probability, a binary "elevated risk" / "normal" flag. Threshold: top
# quartile of predicted probability — the same Q4 framing already used
# repeatedly for validation throughout this project (segment analysis,
# both fatigue-model downstream checks), not a new, untested cutoff.
#
# The threshold itself is fit on TRAIN predictions only, then applied as a
# fixed cutoff to val/test — the same leakage-avoidance principle as
# everywhere else in this pipeline (fit on train, apply unchanged
# elsewhere). Defining the cutoff adaptively on each new dataset would
# trivially force exactly 25% "positive" by construction on every dataset,
# which tests nothing; a fixed cutoff set once and applied forward is the
# only way to honestly check whether the operating point generalizes.

# %%
from pathlib import Path

import pandas as pd
import joblib

import sys
sys.path.insert(0, str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))
from src.training import LABEL_COLUMNS, time_based_split, exclude_gap_rows, prepare_xy

MODELS_DIR = Path("../models/injury_risk") if Path.cwd().name == "notebooks" else Path("models/injury_risk")
pd.set_option("display.width", 140)

QUANTILE = 0.75  # top quartile = "elevated risk"

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
test_scored = exclude_gap_rows(test)

# %% [markdown]
# ## Fit the threshold on train, apply to val and test

# %%
results = []
thresholds = {}
for label in LABEL_COLUMNS:
    clf = joblib.load(MODELS_DIR / f"{label}_model_v3.joblib")

    X_train, y_train = prepare_xy(train, label)
    train_probs = clf.predict_proba(X_train)[:, 1]
    threshold = pd.Series(train_probs).quantile(QUANTILE)
    thresholds[label] = threshold

    for split_name, split_df in [("val", val_scored), ("test", test_scored)]:
        X, y = prepare_xy(split_df, label)
        probs = clf.predict_proba(X)[:, 1]
        flagged = probs >= threshold

        base_rate = y.mean()
        flagged_rate = flagged.mean()
        precision = y[flagged].mean() if flagged.sum() > 0 else float("nan")
        recall = y[flagged].sum() / y.sum() if y.sum() > 0 else float("nan")
        lift = precision / base_rate if base_rate > 0 else float("nan")

        results.append({
            "label": label, "split": split_name, "threshold": threshold,
            "pct_flagged": flagged_rate, "base_rate": base_rate,
            "precision": precision, "recall": recall, "lift": lift,
        })

results_df = pd.DataFrame(results)
print("threshold (fit on train, top quartile of predicted probability):")
for label, t in thresholds.items():
    print(f"  {label}: {t:.4f}")
print()
print(results_df.round(4).to_string(index=False))

# %% [markdown]
# ## What this means
#
# - pct_flagged: what fraction of val/test actually got flagged by the
#   train-fit threshold — should stay reasonably close to 25% if the risk
#   distribution is stable across seasons; meaningful drift would itself be
#   a finding worth investigating.
# - precision: of the player-games flagged "elevated risk," what fraction
#   actually saw an injury.
# - recall: of the player-games that actually saw an injury, what fraction
#   were flagged in advance.
# - lift: how many times more likely a flagged game is to end in injury
#   than a random player-game — the clearest single number for "is this
#   flag actually informative."

# %% [markdown]
# ## Save

# %%
out_dir = Path("../data/processed") if Path.cwd().name == "notebooks" else Path("data/processed")
results_df.to_csv(out_dir / "elevated_risk_threshold_results.csv", index=False)
print(f"\nsaved to {out_dir / 'elevated_risk_threshold_results.csv'}")
