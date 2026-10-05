"""Season-ahead ceilings and ranking diagnostics, kept out of live forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score

from .config import Settings
from .evaluation import BacktestResult, _fantasy_per_game, _score_rows

SHORT_SCHEDULE_SEASONS = {"2019-20", "2020-21"}


def oracle_backtest(
    backtest: BacktestResult, history: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Replace one or both factor forecasts with outcomes, only for diagnosis.

    An unplayed player's FPPG is unobservable. The FPPG-only oracle retains
    that player's original FPPG forecast rather than inferring a zero-game year.
    None of these outcome-informed forecasts may enter a live ranking.
    """
    factor = backtest.predictions[backtest.predictions.model == "factor_model"]
    reports = []
    for season, original in factor.groupby("target_season"):
        actual = history[history.season == season].copy()
        actual["actual_games"] = actual.games
        actual["actual_fppg"] = _fantasy_per_game(actual, settings)
        labels = actual[["player_id", "actual_games", "actual_fppg"]]
        replacement = float(backtest.metrics.loc[
            (backtest.metrics.target_season == season)
            & (backtest.metrics.model == "factor_model"), "replacement_fppg"
        ].iloc[0])
        for model, perfect_fppg, perfect_games in [
            ("actual_fppg_model_gp", True, False),
            ("model_fppg_actual_gp", False, True),
            ("actual_fppg_actual_gp", True, True),
        ]:
            rows = original.copy()
            if perfect_fppg:
                rows["pred_fppg"] = rows.actual_fppg.fillna(rows.pred_fppg)
            if perfect_games:
                rows["pred_games"] = rows.actual_games
            reports.append({
                "target_season": season, "model": model,
                "roster_source": original.roster_source.iloc[0],
                "replacement_fppg": replacement,
                **_score_rows(rows, labels, replacement),
            })
    return pd.DataFrame(reports)


def ranking_diagnostics(
    backtest: BacktestResult, history: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """NDCG, top-100 rank correlation and realized value regret.

    Missing forecast candidates receive the lowest possible predicted rank;
    ideal value is always computed against the full actual player universe.
    """
    reports = []
    for season, actual in history.groupby("season"):
        predicted = backtest.predictions[backtest.predictions.target_season == season]
        if predicted.empty:
            continue
        actual = actual.copy()
        actual["actual_fppg"] = _fantasy_per_game(actual, settings)
        replacement = float(backtest.metrics.loc[
            (backtest.metrics.target_season == season)
            & (backtest.metrics.model == "factor_model"), "replacement_fppg"
        ].iloc[0])
        actual["raw_points"] = actual.actual_fppg * actual.games
        actual["draft_value"] = (actual.actual_fppg - replacement) * actual.games
        for model, group in predicted.groupby("model"):
            pool = actual[["player_id", "raw_points", "draft_value"]].merge(
                group[["player_id", "pred_fppg", "pred_games"]],
                on="player_id", how="left", validate="one_to_one"
            )
            for board in ("raw_points", "draft_value"):
                scores = (pool.pred_fppg - (replacement if board == "draft_value" else 0)) * pool.pred_games
                scores = scores.fillna(-1e12)
                truth = pool[board].to_numpy(dtype=float)
                ideal_order = np.argsort(-truth, kind="stable")
                model_order = np.argsort(-scores.to_numpy(), kind="stable")
                top100 = ideal_order[: min(100, len(pool))]
                observed = np.isfinite(pool.pred_fppg.to_numpy()[top100])
                correlation = (
                    float(spearmanr(scores.to_numpy()[top100][observed], truth[top100][observed]).statistic)
                    if observed.sum() >= 3 else np.nan
                )
                for k in (30, 60):
                    cutoff = min(k, len(pool))
                    relevance = np.maximum(truth, 0)
                    reports.append({
                        "target_season": season, "model": model, "board": board,
                        "top_k": k,
                        "ndcg": float(ndcg_score(relevance.reshape(1, -1), scores.to_numpy().reshape(1, -1), k=cutoff)),
                        "value_regret": float(truth[ideal_order[:cutoff]].sum() - truth[model_order[:cutoff]].sum()),
                        "top100_rank_spearman": correlation,
                        "top100_pool_coverage": float(observed.mean()),
                    })
    return pd.DataFrame(reports)


def summarize_extended_topk(
    metrics: pd.DataFrame, settings: Settings, *, n_bootstrap: int = 3000
) -> pd.DataFrame:
    """Eight normal-length seasons, with season-block bootstrap intervals."""
    seasons = [
        season for season in sorted(metrics.target_season.unique())
        if season not in SHORT_SCHEDULE_SEASONS
    ][-8:]
    rng = np.random.default_rng(settings.random_seed)
    reports = []
    for model, group in metrics[metrics.target_season.isin(seasons)].groupby("model"):
        group = group.set_index("target_season").reindex(seasons).dropna(how="all")
        if group.empty:
            continue
        for board, prefix in (("raw_points", ""), ("draft_value", "vor_")):
            for k in (30, 60):
                values = group[f"{prefix}top{k}_hit_rate"].to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                if not len(values):
                    continue
                draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
                reports.append({
                    "model": model, "board": board, "top_k": k,
                    "seasons": ",".join(group.index.tolist()),
                    "n_seasons": len(values), "hit_rate": float(values.mean()),
                    "ci_low": float(np.quantile(draws, 0.025)),
                    "ci_high": float(np.quantile(draws, 0.975)),
                })
    return pd.DataFrame(reports)


def summarize_ranking_diagnostics(
    diagnostics: pd.DataFrame, settings: Settings, *, n_bootstrap: int = 3000
) -> pd.DataFrame:
    """Season-block uncertainty for the less volatile ranking-quality measures."""
    seasons = [
        season for season in sorted(diagnostics.target_season.unique())
        if season not in SHORT_SCHEDULE_SEASONS
    ][-8:]
    rng = np.random.default_rng(settings.random_seed + 1)
    reports = []
    for (model, board, k), group in diagnostics[
        diagnostics.target_season.isin(seasons)
    ].groupby(["model", "board", "top_k"]):
        report = {
            "model": model, "board": board, "top_k": k,
            "seasons": ",".join(sorted(group.target_season)),
            "n_seasons": len(group),
        }
        for metric in ("ndcg", "value_regret", "top100_rank_spearman"):
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            draws = rng.choice(values, size=(n_bootstrap, len(values)), replace=True).mean(axis=1)
            report[metric] = float(values.mean())
            report[f"{metric}_ci_low"] = float(np.quantile(draws, 0.025))
            report[f"{metric}_ci_high"] = float(np.quantile(draws, 0.975))
        reports.append(report)
    return pd.DataFrame(reports)
