"""Prequential top-K ranking experiment on end-to-end forecast outputs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .evaluation import BacktestResult, _fantasy_per_game
from .game_form import build_game_form

MODELS = ("factor_model", "previous_season", "three_year_weighted")


@dataclass
class ProbabilityBacktest:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def summarize_probability_targets(metrics: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Summarize the last four season-held-out probability-ranking folds."""
    recent = sorted(metrics.target_season.unique())[-4:]
    rows = []
    for (model, board, top), group in metrics[metrics.target_season.isin(recent)].groupby(
        ["model", "board", "top_k"]
    ):
        hits = int(group.hit_count.sum())
        possible = int(group.possible_hits.sum())
        rate = hits / possible
        minimum = getattr(settings.topk_targets, f"top{top}_min")
        stretch = getattr(settings.topk_targets, f"top{top}_stretch")
        rows.append({
            "model": model,
            "board": board,
            "top_k": top,
            "seasons": ",".join(sorted(group.target_season)),
            "hit_count": hits,
            "possible_hits": possible,
            "hit_rate": rate,
            "candidate_pool_ceiling": float(group.candidate_pool_ceiling.mean()),
            "minimum_target": minimum,
            "stretch_target": stretch,
            "additional_hits_to_minimum": max(0, ceil(minimum * possible) - hits),
            "target_status": (
                "below_minimum" if rate < minimum else
                "at_or_above_stretch" if rate >= stretch else "in_target_band"
            ),
        })
    return pd.DataFrame(rows)


def _features(backtest: BacktestResult, board: str) -> pd.DataFrame:
    predictions = backtest.predictions.copy()
    replacement = backtest.metrics.loc[
        backtest.metrics.model == "factor_model", ["target_season", "replacement_fppg"]
    ]
    predictions = predictions.merge(replacement, on="target_season", validate="many_to_one")
    if board == "draft_value":
        predictions["estimated_score"] = (
            predictions.pred_fppg - predictions.replacement_fppg
        ) * predictions.pred_games
    else:
        predictions["estimated_score"] = predictions.pred_fppg * predictions.pred_games
    predictions["score_rank_pct"] = predictions.groupby(
        ["target_season", "model"]
    )["estimated_score"].rank(pct=True)
    predictions["fppg_rank_pct"] = predictions.groupby(
        ["target_season", "model"]
    )["pred_fppg"].rank(pct=True)
    wide = predictions.pivot(
        index=["target_season", "player_id"],
        columns="model",
        values=["score_rank_pct", "fppg_rank_pct", "pred_games"],
    )
    wide.columns = [f"{stat}_{model}" for stat, model in wide.columns]
    required = [
        f"{stat}_{model}"
        for stat in ("score_rank_pct", "fppg_rank_pct", "pred_games")
        for model in MODELS
    ]
    return wide[required].dropna().reset_index()


def _trajectory_features(history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Build player features using only seasons before each draft cutoff."""
    history = history.copy()
    history["fppg"] = _fantasy_per_game(history, settings)
    for stat in ("fppg", "fga", "fta", "ast", "reb", "stl", "blk"):
        history[f"{stat}_per_min"] = history[stat] / history.mpg.clip(lower=1)
    base = [
        "age", "games", "mpg", "usage", "fppg", "fppg_per_min",
        "fga_per_min", "fta_per_min", "ast_per_min", "reb_per_min",
        "stl_per_min", "blk_per_min",
    ]
    rows = []
    seasons = settings.historical_seasons
    for index in range(1, len(seasons)):
        latest = history.loc[
            history.season == seasons[index - 1], ["player_id", *base]
        ].copy()
        latest["target_season"] = seasons[index]
        latest = latest.rename(columns={name: f"prior_{name}" for name in base})
        if index >= 2:
            earlier = history.loc[
                history.season == seasons[index - 2], ["player_id", *base]
            ].rename(columns={name: f"older_{name}" for name in base})
            latest = latest.merge(earlier, on="player_id", how="left", validate="one_to_one")
        else:
            for name in base:
                latest[f"older_{name}"] = float("nan")
        latest["has_two_year_history"] = latest.older_games.notna().astype(int)
        for name in ("games", "mpg", "usage", "fppg", "fppg_per_min"):
            latest[f"change_{name}"] = latest[f"prior_{name}"] - latest[f"older_{name}"]
        rows.append(latest.drop(columns=[f"older_{name}" for name in base]))
    return pd.concat(rows, ignore_index=True)


def walk_forward_topk_probabilities(
    backtest: BacktestResult,
    history: pd.DataFrame,
    settings: Settings,
    approach: str = "forecast_logistic",
    game_form: pd.DataFrame | None = None,
) -> ProbabilityBacktest:
    """Train only on earlier held-out seasons; no target-year label enters fitting.

    Model form and regularization are fixed in advance. This is a diagnostic
    experiment, not an in-sample choice of blend weights or a board override.
    """
    if approach not in {
        "forecast_logistic", "trajectory_boosted", "trajectory_logistic", "game_form_logistic"
    }:
        raise ValueError(f"Unknown ranking approach: {approach}")
    model_name = {
        "forecast_logistic": "topk_probability",
        "trajectory_boosted": "trajectory_boosted",
        "trajectory_logistic": "trajectory_logistic",
        "game_form_logistic": "game_form_logistic",
    }[approach]
    seasons = sorted(backtest.metrics.target_season.unique())
    form_seasons: set[str] = set()
    if approach == "game_form_logistic":
        if game_form is None:
            game_form = build_game_form(
                settings.paths.raw / "archive/espn_player_box.parquet", settings
            )
        form_seasons = set(game_form.target_season)
    reports = []
    details = []
    for board in ("raw_points", "draft_value"):
        frame = _features(backtest, board)
        if approach == "game_form_logistic":
            frame = frame.merge(
                game_form, on=["target_season", "player_id"],
                how="left", validate="one_to_one",
            )
        elif approach != "forecast_logistic":
            frame = frame.merge(
                _trajectory_features(history, settings),
                on=["target_season", "player_id"],
                how="left",
                validate="one_to_one",
            )
        feature_columns = frame.columns.drop(["target_season", "player_id"]).tolist()
        for top in (30, 60):
            truth = {}
            for season in seasons:
                actual = history[history.season == season].copy()
                actual["score"] = _fantasy_per_game(actual, settings) * actual.games
                if board == "draft_value":
                    replacement = backtest.metrics.loc[
                        (backtest.metrics.target_season == season)
                        & (backtest.metrics.model == "factor_model"),
                        "replacement_fppg",
                    ].iloc[0]
                    actual["score"] -= replacement * actual.games
                truth[season] = set(actual.nlargest(top, "score").player_id)
            local = frame.copy()
            local["actual_topk"] = [
                player_id in truth[season]
                for season, player_id in zip(local.target_season, local.player_id)
            ]
            for index, season in enumerate(seasons):
                if index < 3:
                    continue
                if approach == "game_form_logistic" and (
                    season not in form_seasons
                    or len(set(seasons[:index]) & form_seasons) < 3
                ):
                    continue
                train = local[local.target_season.isin(seasons[:index])]
                test = local[local.target_season == season].copy()
                if train.actual_topk.nunique() < 2 or len(test) < top:
                    continue
                if approach == "trajectory_boosted":
                    classifier = HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=80,
                        max_leaf_nodes=6,
                        min_samples_leaf=30,
                        l2_regularization=5.0,
                        random_state=settings.random_seed,
                    )
                elif approach in {"trajectory_logistic", "game_form_logistic"}:
                    classifier = make_pipeline(
                        SimpleImputer(strategy="median", add_indicator=True),
                        StandardScaler(),
                        LogisticRegression(C=0.25, max_iter=1000),
                    )
                else:
                    classifier = make_pipeline(
                        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)
                    )
                classifier.fit(train[feature_columns], train.actual_topk)
                test["estimated_probability"] = classifier.predict_proba(
                    test[feature_columns]
                )[:, 1]
                selected = set(test.nlargest(top, "estimated_probability").player_id)
                hits = len(selected & truth[season])
                reports.append({
                    "target_season": season,
                    "model": model_name,
                    "board": board,
                    "top_k": top,
                    "train_through": seasons[index - 1],
                    "hit_count": hits,
                    "possible_hits": top,
                    "hit_rate": hits / top,
                    "candidate_pool_ceiling": len(set(test.player_id) & truth[season]) / top,
                    "brier_score": float(((test.estimated_probability - test.actual_topk) ** 2).mean()),
                })
                details.append(test[[
                    "target_season", "player_id", "actual_topk", "estimated_probability"
                ]].assign(board=board, top_k=top, model=model_name))
    return ProbabilityBacktest(
        pd.DataFrame(reports),
        pd.concat(details, ignore_index=True) if details else pd.DataFrame(),
    )
