from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Settings
from .projection import build_baseline, finalize_box_score


@dataclass
class ValidationReport:
    integrity: pd.DataFrame
    backtest: pd.DataFrame

    @property
    def passed(self) -> bool:
        if self.integrity.empty:
            return False
        blocking = self.integrity[self.integrity["severity"] == "error"]
        return bool(blocking["passed"].all())


def integrity_checks(projections: pd.DataFrame, team_checks: pd.DataFrame) -> pd.DataFrame:
    checks: list[dict[str, object]] = []

    def add(name: str, passed: bool, detail: str, severity: str = "error") -> None:
        checks.append(
            {"check": name, "passed": bool(passed), "detail": detail, "severity": severity}
        )

    add("unique players", projections["player_id"].is_unique, "player_id must be unique")
    add("valid games", projections["expected_gp"].between(0, 82).all(), "expected GP in [0,82]")
    add("valid minutes", projections["projected_mpg"].between(0, 38.001).all(), "MPG in [0,38]")
    add("valid usage", projections["projected_usage"].between(0, 50).all(), "usage in [0,50]%")
    for made, attempted in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        add(
            f"{made} <= {attempted}", (projections[made] <= projections[attempted] + 1e-9).all(), ""
        )
    point_error = (
        (projections["pts"] - (2 * projections["fgm"] + projections["fg3m"] + projections["ftm"]))
        .abs()
        .max()
    )
    add("points reconcile", point_error < 1e-8, f"max error={point_error:.3g}")
    if not team_checks.empty:
        add(
            "team scheduled-minute capacity",
            team_checks["minutes_ok"].all(),
            f"{(~team_checks['minutes_ok']).sum()} teams exceed 240 expected minutes; forecasts are not rescaled",
            "warning",
        )
        add(
            "team usage",
            team_checks["usage_ok"].all(),
            f"{(~team_checks['usage_ok']).sum()} failures",
        )
    add("finite fantasy totals", np.isfinite(projections["expected_season_points"]).all(), "")
    return pd.DataFrame(checks)


def walk_forward_backtest(history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Backtest the transparent baseline, using no future-season data.

    Team/context overrides are intentionally excluded because only reproducible historical inputs belong here.
    """
    seasons = settings.historical_seasons
    results: list[dict[str, object]] = []
    for target_idx in range(2, len(seasons)):
        train_seasons = seasons[:target_idx]
        target = seasons[target_idx]
        train = history[history["season"].isin(train_seasons)].copy()
        actual = history[history["season"] == target].copy()
        if train.empty or actual.empty:
            continue
        local = settings.model_copy(deep=True)
        local.historical_seasons = train_seasons
        local.projection.season_weights = settings.projection.season_weights[:target_idx]
        roster = actual[["player_id", "player_name", "team_id"]].drop_duplicates("player_id")
        pred = build_baseline(train, roster, local)
        # No team constraint here: historical dataset may omit fringe players.
        pred = finalize_box_score(pred, local)
        joined = pred.merge(actual, on="player_id", suffixes=("_pred", "_actual"))
        if joined.empty:
            continue
        for stat in ["games", "mpg", "pts", "reb", "ast", "stl", "blk", "tov"]:
            pred_col = (
                "expected_gp"
                if stat == "games"
                else "projected_mpg"
                if stat == "mpg"
                else f"{stat}_pred"
            )
            actual_col = stat if stat in {"games", "mpg"} else f"{stat}_actual"
            error = joined[pred_col] - joined[actual_col]
            results.append(
                {
                    "target_season": target,
                    "metric": stat,
                    "n": len(joined),
                    "mae": error.abs().mean(),
                    "bias": error.mean(),
                }
            )
    return pd.DataFrame(results)


def validate(
    projections: pd.DataFrame, team_checks: pd.DataFrame, history: pd.DataFrame, settings: Settings
) -> ValidationReport:
    return ValidationReport(
        integrity_checks(projections, team_checks), walk_forward_backtest(history, settings)
    )
