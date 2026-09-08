# NBA Injury Risk Predictor

Predicts the probability that an NBA player misses a game to injury within their next 1, 3, or 10 games, and surfaces that as a binary **elevated-risk flag** for a player's *current* status — not a raw probability score, and not a next-game forecast tied to a specific opponent.

A companion web app (Next.js + FastAPI) runs the trained models against live NBA data for any active player, but is **local-only by design** — see [Web app](#web-app-local-only-by-design) below for why.

## Overview

Given a player-game, the pipeline builds ~26 features (rest/schedule, workload load, bio, injury history, travel, team pace/rating context) and trains gradient-boosted classifiers to predict `y_1game` / `y_3game` / `y_10game` — whether the player misses a game to injury within that horizon. Each model is wrapped in isotonic probability calibration (injury events are rare — a few percent base rate — and raw classifier probabilities are badly overconfident without it), then thresholded against a fixed, top-quartile-of-training-predictions cutoff to produce the elevated-risk flag actually meant for use, rather than asking a raw probability to speak for itself.

**Leakage avoidance is the design principle threaded through everything here.** Train/val/test is split by season, not randomly — same-player rows close in time are highly correlated, and a random split would let the model see near-duplicates of its own test data. Every rolling feature (workload, rest) is computed with `shift(1)` so a game never sees its own outcome in its features. Every fit statistic — calibration, cohort baselines for missing bio data, the elevated-risk threshold itself — is fit once on the training split only and applied unchanged to validation/test.

## Results (held-out test set, 2016–2025 seasons)

| Horizon | Base rate | Flagged | Precision | Recall | Lift |
|---|---|---|---|---|---|
| Next 1 game | 2.8% | 40.9% of games | 4.3% | 64.3% | 1.57× |
| Next 3 games | 8.2% | 38.5% of games | 13.2% | 61.9% | 1.61× |
| Next 10 games | 23.2% | 38.0% of games | 34.9% | 57.1% | 1.50× |

Recall in the 57–64% range means the flag catches roughly three in five actual injury absences ahead of time, at 1.5–1.6× the base injury rate among the games it flags — a meaningfully informative signal for a genuinely noisy, hard-to-predict event, not a claim of certainty.

## Data sources

- **NBA Stats API** (`nba_api`) — player/team game logs, bio, and advanced team stats.
- **[2016–2025 NBA Injury Data](https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data)** (Kaggle) — the injury-transaction log used to construct labels and historical injury-history features. "Injury" is defined *strictly* (injury/unspecified only — illness, COVID, rest, and personal absences are excluded) so the label means what it says.
- **NBA's official injury-report PDFs** — parsed directly (`src/injury_reports.py`, pure-Python `pdfplumber`, no JVM) to extend injury-history coverage past the Kaggle dataset's cutoff for live inference, with a per-player confidence score reflecting how much of that extension window could actually be verified against a real report.

## Models

`models/injury_risk/` holds three iterations (v1 → v2 → v3, kept side by side for comparison, not cleanup targets) of `CalibratedClassifierCV`-wrapped `HistGradientBoostingClassifier`s — one per horizon. v3 is current. Fixed elevated-risk thresholds (`0.034` / `0.087` / `0.235`) were fit once as the top quartile of training-set predictions and are not periodically requantized — a real rise in league-wide injury rates should show up as *more* flags, not get normalized away.

A separate "fatigue" model line (`models/fatigue/`, predicting performance decline as an injury-risk proxy) was tried and intentionally abandoned in favor of the more direct injury-outcome framing above. The code and models are kept, not deleted, as a documented negative result.

## Repo structure

```
notebooks/       Numbered, ordered pipeline (EDA → labels → features → train → evaluate → live-serving support)
src/              Shared pipeline code: data loading, feature engineering, training, evaluation,
                  and the live-serving modules the web app depends on
models/           Trained model artifacts (all versions kept)
data/processed/   Built datasets + the live-reference snapshot (small artifacts, git-tracked)
backend/          FastAPI service — wraps src/ for live single-player inference
frontend/         Next.js app — player search, risk display, season/career stats
```

Notebooks are plain `.py` scripts (Jupytext "light" format — run top-to-bottom or cell-by-cell) meant to be read in order: `01`–`02` EDA, `03` labels, `04`–`13` feature layers, `06`–`14` model training (v1→v3), `08`/`10` evaluation, `15`–`18` the abandoned fatigue line, `19` the elevated-risk threshold, `20`–`22` live-serving support (reference-data snapshot, a validation harness that diffs live inference against known-good historical rows, and the injury-report backfill).

## Web app (local-only, by design)

The frontend/backend demonstrate the trained models against live data: search a player, see their elevated-risk flags computed fresh from today's NBA data, plus season/career per-game stats. It's deliberately **not deployed** — `stats.nba.com` silently blocks requests from cloud-hosted IPs (a known, structural limitation of the unofficial NBA Stats API, not specific to this project), and working around that means either an ongoing hosting cost or a fragile home-relay dependency for a project whose actual deliverable — trained, validated models and readable code — is already fully visible in this repo. Run it locally instead:

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

# run notebooks 01–22 in order to build the data, train the models,
# and produce the live-reference snapshot the backend needs

cd frontend
npm install
npm run dev:all   # backend (FastAPI, :8000) + frontend (Next.js, :3000) together
```

## Tech stack

Python (pandas, scikit-learn, `nba_api`), FastAPI, Next.js/TypeScript/React, and `pdfplumber` for PDF parsing. See [CLAUDE.md](CLAUDE.md) for full architecture notes, module responsibilities, and the reasoning behind key design decisions.
