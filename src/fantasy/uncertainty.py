"""Empirical, paired FPPG and games residuals from season-ahead forecasts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_residuals(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_parquet(path)
    rows = rows[rows["model"] == "factor_model"].copy()
    required = {"pred_fppg", "pred_games", "actual_fppg", "actual_games", "target_season"}
    if required - set(rows):
        raise ValueError(f"Backtest predictions lack {sorted(required - set(rows))}")
    # The unusual shortened 2019-20 and 2020-21 schedules are not comparable
    # to the upcoming 82-game season without a separate calendar adjustment.
    rows = rows[rows["target_season"] >= "2021-22"]
    rows["fppg_error"] = (rows["actual_fppg"] - rows["pred_fppg"]).fillna(0)
    rows["games_error"] = rows["actual_games"] - rows["pred_games"]
    return rows.dropna(subset=["pred_fppg", "pred_games", "games_error"])


def draw_joint_outcomes(
    predicted_fppg: float,
    predicted_games: float,
    residuals: pd.DataFrame,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap whole error pairs from a comparable FPPG cohort."""
    if len(residuals) < 100:
        raise ValueError("At least 100 season-ahead residual pairs are required")
    edges = np.quantile(residuals["pred_fppg"], [0, 0.2, 0.4, 0.6, 0.8, 1])
    cohort = np.searchsorted(edges[1:-1], predicted_fppg, side="right")
    historical_cohorts = np.searchsorted(
        edges[1:-1], residuals["pred_fppg"].to_numpy(), side="right"
    )
    pool = residuals.loc[historical_cohorts == cohort]
    if len(pool) < 100:
        pool = residuals
    indices = rng.integers(0, len(pool), count)
    sampled = pool.iloc[indices]
    games = np.clip(predicted_games + sampled["games_error"].to_numpy(), 0, 82)
    fppg = np.maximum(0, predicted_fppg + sampled["fppg_error"].to_numpy())
    return fppg, games


def interval_coverage(residuals: pd.DataFrame) -> pd.DataFrame:
    """Prequential 20th–80th coverage; each year uses only earlier residuals."""
    if residuals.empty:
        return pd.DataFrame(columns=["target_season", "n", "p20_p80_coverage"])
    reports = []
    for year in sorted(residuals["target_season"].unique()):
        train = residuals[residuals["target_season"] < year]
        valid = residuals[residuals["target_season"] == year]
        if len(train) < 100:
            continue
        rng = np.random.default_rng(int(year[:4]))
        covered = []
        for _, row in valid.iterrows():
            fppg, games = draw_joint_outcomes(
                float(row["pred_fppg"]), float(row["pred_games"]), train, 1000, rng
            )
            low, high = np.quantile(fppg * games, [0.2, 0.8])
            actual = float(row["actual_season_total"])
            covered.append(low <= actual <= high)
        reports.append({"target_season": year, "n": len(valid), "p20_p80_coverage": np.mean(covered)})
    return pd.DataFrame(reports)
