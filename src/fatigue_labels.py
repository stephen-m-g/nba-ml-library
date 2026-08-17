"""Build the z_1game / z_3game / z_10game performance-decline labels for the
fatigue model — a separate model line from the injury-risk one (see
src/feature_engineering.py and src/training.py for that). Reuses the same
feature set (workload, rest, travel, bio, injury history); only the label
differs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_game_score(player_log: pd.DataFrame) -> pd.DataFrame:
    """John Hollinger's Game Score — a single-number, well-established
    composite of one game's box score production (roughly a simplified
    single-game PER), computed directly from columns already in player_log:

    GmSc = PTS + 0.4*FGM - 0.7*FGA - 0.4*(FTA-FTM) + 0.7*OREB + 0.3*DREB
           + STL + 0.7*AST + 0.7*BLK - 0.4*PF - TOV

    Also derives GAME_SCORE_PER36: minutes played is itself a workload
    decision (often the team's own fatigue management), not a symptom of
    fatigue — rate-adjusting to per-36-minutes isolates "how well did they
    play when they were out there" from "how many minutes did they get."
    Without this, a player given fewer minutes to manage their workload
    would show a lower raw Game Score even if nothing about the quality of
    their play changed — exactly the confound this model needs to avoid,
    since reduced minutes is already captured elsewhere as a feature, not
    something that should leak into the fatigue *target* too.

    NaN for MIN < 5 games — checked directly: a per-36 extrapolation from a
    token 1-2 minute appearance is mostly noise (one stray foul or a lucky
    three swings the extrapolated number wildly; the 627 rows with
    |GAME_SCORE_PER36| > 60 had a median of exactly 1 minute played). MIN < 5
    excludes only 6.8% of rows but catches 94% of that distortion, and
    per-36 variance drops sharply right around this cutoff (std 23 for
    0-5 minutes vs. 10 for 10-15 minutes) — a real reliability cliff, not an
    arbitrary line.

    Returns PLAYER_ID, GAME_ID, GAME_DATE, SEASON_ID, GAME_SCORE, GAME_SCORE_PER36.
    """
    df = player_log[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "MIN", "PTS", "FGM", "FGA",
                      "FTA", "FTM", "OREB", "DREB", "STL", "AST", "BLK", "PF", "TOV"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df["GAME_SCORE"] = (
        df["PTS"] + 0.4 * df["FGM"] - 0.7 * df["FGA"] - 0.4 * (df["FTA"] - df["FTM"])
        + 0.7 * df["OREB"] + 0.3 * df["DREB"] + df["STL"] + 0.7 * df["AST"] + 0.7 * df["BLK"]
        - 0.4 * df["PF"] - df["TOV"]
    )
    df["GAME_SCORE_PER36"] = np.where(df["MIN"] >= 5, df["GAME_SCORE"] * 36 / df["MIN"], np.nan)

    return df[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID", "GAME_SCORE", "GAME_SCORE_PER36"]]


def _prepare_game_score_with_baseline(
    player_log: pd.DataFrame, baseline_window: int, baseline_min_periods: int
) -> pd.DataFrame:
    """Shared prep for both fatigue targets (classification and regression):
    Game Score per-36 plus each player's own rolling PRIOR baseline (mean
    and std — shift-then-roll, same no-leakage pattern as
    compute_workload_features), sorted by (PLAYER_ID, GAME_DATE).
    """
    gs = compute_game_score(player_log).sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    grp = gs.groupby("PLAYER_ID")["GAME_SCORE_PER36"]
    gs["baseline_mean"] = grp.transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=baseline_min_periods).mean()
    )
    gs["baseline_std"] = grp.transform(
        lambda s: s.shift(1).rolling(baseline_window, min_periods=baseline_min_periods).std()
    )
    return gs


def build_fatigue_labels(
    player_log: pd.DataFrame,
    baseline_window: int = 15,
    baseline_min_periods: int = 5,
    decline_std_threshold: float = 2.0,
) -> pd.DataFrame:
    """z_1game/z_3game/z_10game: did this player's own performance drop
    meaningfully below their recent personal baseline in at least one of
    their next 1/3/10 games actually played?

    Deliberately structured around the player's own next games played (not
    the team's next scheduled games the way y_Ngame is) — a player who
    doesn't play at all in the window is a question the injury model
    already answers; this model is specifically about "when they do play,
    is production suffering," a distinct signal worth keeping separate
    rather than conflating the two.

    - decline: a future game's GAME_SCORE_PER36 falls more than
      `decline_std_threshold` standard deviations below that baseline —
      personalized per player (a naturally volatile scorer needs a bigger
      absolute drop to count than a highly consistent one), the same
      "relative to your own normal" logic as the acute:chronic workload ratio.
      Default of 2.0 chosen empirically, not arbitrarily: 1.0 produces a
      75.6% positive rate for z_10game, which is really just "ordinary
      game-to-game variance" (about 16% of any single game would qualify at
      that threshold under a rough normal approximation) rather than a
      meaningful outlier. 2.0 lands at 3.4% / 9.4% / 25.2% for
      z_1game/z_3game/z_10game — a real statistical outlier, and close
      enough to the injury model's own 2.4% / 7.0% / 20.0% to be
      comparably interpretable.

    Eligibility: requires both a valid baseline (>= baseline_min_periods
    prior games) and at least 10 more rows in this player's own log after
    the current one — rows without either are dropped, the same "don't
    label what we can't actually observe" principle as the injury pipeline's
    coverage cutoff.

    v1 result (classification, HistGradientBoostingClassifier): weak on its
    own terms (val AUC-ROC 0.52-0.54, barely past random) — single-game
    performance turns out to carry a lot of variance workload/rest/travel
    features can't see (matchup, shooting luck, blowouts). But a genuine,
    monotonic downstream relationship with actual injury rate showed up
    anyway (15.5% -> 26.4% actual injury rate across predicted-fatigue
    quartiles), which is why this stays in the codebase as a validated
    finding rather than getting deleted — see build_fatigue_regression_targets
    for the follow-up: averaging over the window instead of thresholding a
    single game, to test whether that noise was the real culprit.

    Returns PLAYER_ID, GAME_ID, GAME_DATE, SEASON_ID, z_1game, z_3game, z_10game.
    """
    gs = _prepare_game_score_with_baseline(player_log, baseline_window, baseline_min_periods)
    gs["decline_threshold"] = gs["baseline_mean"] - decline_std_threshold * gs["baseline_std"]

    games_remaining = gs.groupby("PLAYER_ID").cumcount(ascending=False)  # rows strictly after this one, per player
    has_baseline = gs["baseline_mean"].notna() & gs["baseline_std"].notna()

    grp_score = gs.groupby("PLAYER_ID")["GAME_SCORE_PER36"]
    future_declines = []
    for k in range(1, 11):
        future_score = grp_score.shift(-k)
        future_declines.append((future_score < gs["decline_threshold"]).fillna(False))
    future_declines = pd.concat(future_declines, axis=1, keys=range(1, 11))

    gs["z_1game"] = future_declines[1]
    gs["z_3game"] = future_declines[[1, 2, 3]].any(axis=1)
    gs["z_10game"] = future_declines[list(range(1, 11))].any(axis=1)

    eligible = has_baseline & (games_remaining >= 10)
    result = gs.loc[eligible, ["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID",
                                "z_1game", "z_3game", "z_10game"]].reset_index(drop=True)
    return result


def build_fatigue_regression_targets(
    player_log: pd.DataFrame,
    baseline_window: int = 15,
    baseline_min_periods: int = 5,
) -> pd.DataFrame:
    """w_1game/w_3game/w_10game: continuous, signed deviation of
    GAME_SCORE_PER36 from the player's own rolling baseline, AVERAGED over
    their next 1/3/10 games actually played. Negative = playing worse than
    their own normal (a "dropoff"), positive = playing better, 0 = at baseline.

    Averaging over the window — not a single game, not a threshold crossing
    — is the deliberate follow-up to build_fatigue_labels' weak result:
    single-game performance is noisy (matchup, shooting variance, blowouts),
    and averaging is a direct, principled way to cancel some of that out and
    test whether a real, smoother trend exists underneath it — the same
    logic that makes min_avg_chronic (a 15-game average) a far stronger
    feature than any single game's minutes would be on its own.

    Eligibility: same as build_fatigue_labels — valid baseline (>=
    baseline_min_periods prior games) and >= 10 more rows in this player's
    own log after the current one. Rows where every game in a window has no
    valid GAME_SCORE_PER36 (all sub-5-minute appearances — rare) are also
    dropped rather than left as an undefined target.

    Returns PLAYER_ID, GAME_ID, GAME_DATE, SEASON_ID, w_1game, w_3game, w_10game.
    """
    gs = _prepare_game_score_with_baseline(player_log, baseline_window, baseline_min_periods)

    games_remaining = gs.groupby("PLAYER_ID").cumcount(ascending=False)
    has_baseline = gs["baseline_mean"].notna()

    grp_score = gs.groupby("PLAYER_ID")["GAME_SCORE_PER36"]
    future_scores = pd.concat([grp_score.shift(-k) for k in range(1, 11)], axis=1, keys=range(1, 11))

    gs["w_1game"] = future_scores[1] - gs["baseline_mean"]
    gs["w_3game"] = future_scores[[1, 2, 3]].mean(axis=1) - gs["baseline_mean"]
    gs["w_10game"] = future_scores[list(range(1, 11))].mean(axis=1) - gs["baseline_mean"]

    eligible = has_baseline & (games_remaining >= 10)
    result = gs.loc[eligible, ["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON_ID",
                                "w_1game", "w_3game", "w_10game"]].reset_index(drop=True)
    result = result.dropna(subset=["w_1game", "w_3game", "w_10game"])
    return result
