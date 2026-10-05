"""Season-ahead learned-lag ranking baseline with expanding-window fits."""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .evaluation import BacktestResult, _fantasy_per_game
from .ranker import ProbabilityBacktest


def _lag_features(history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    source = history.copy()
    source["fppg"] = _fantasy_per_game(source, settings)
    source["fppm"] = source.fppg / source.mpg.clip(lower=1)
    columns = ["age", "games", "mpg", "usage", "fppg", "fppm"]
    rows = []
    seasons = settings.historical_seasons
    for index in range(1, len(seasons)):
        target = pd.DataFrame({"player_id": source[source.season == seasons[index - 1]].player_id})
        target["target_season"] = seasons[index]
        for lag in (1, 2, 3):
            if index < lag:
                for name in columns:
                    target[f"lag{lag}_{name}"] = float("nan")
                continue
            prior = source.loc[
                source.season == seasons[index - lag], ["player_id", *columns]
            ].rename(columns={name: f"lag{lag}_{name}" for name in columns})
            target = target.merge(prior, on="player_id", how="left", validate="one_to_one")
        rows.append(target)
    return pd.concat(rows, ignore_index=True)


def walk_forward_learned_lags(
    backtest: BacktestResult, history: pd.DataFrame, settings: Settings
) -> ProbabilityBacktest:
    """Learn lag coefficients only from seasons preceding each evaluation season."""
    seasons = sorted(backtest.metrics.target_season.unique())
    candidates = backtest.predictions.loc[
        backtest.predictions.model == "factor_model", ["target_season", "player_id"]
    ]
    features = candidates.merge(
        _lag_features(history, settings),
        on=["target_season", "player_id"],
        how="left",
        validate="one_to_one",
    )
    feature_columns = features.columns.drop(["target_season", "player_id"]).tolist()
    reports = []
    details = []
    for board in ("raw_points", "draft_value"):
        frame = features.copy()
        truth = {}
        for season in seasons:
            actual = history[history.season == season].copy()
            replacement = float(backtest.metrics.loc[
                (backtest.metrics.target_season == season)
                & (backtest.metrics.model == "factor_model"), "replacement_fppg"
            ].iloc[0])
            actual["score"] = _fantasy_per_game(actual, settings) * actual.games
            if board == "draft_value":
                actual["score"] -= replacement * actual.games
            truth[season] = {
                top: set(actual.nlargest(top, "score").player_id) for top in (30, 60)
            }
            frame.loc[frame.target_season == season, "actual_score"] = frame.loc[
                frame.target_season == season, "player_id"
            ].map(actual.set_index("player_id").score).fillna(0)
        for index, season in enumerate(seasons):
            if index < 3:
                continue
            train = frame[frame.target_season.isin(seasons[:index])]
            test = frame[frame.target_season == season].copy()
            estimator = make_pipeline(
                SimpleImputer(strategy="median", add_indicator=True),
                StandardScaler(),
                Ridge(alpha=100.0),
            )
            estimator.fit(train[feature_columns], train.actual_score)
            test["estimated_score"] = estimator.predict(test[feature_columns])
            for top in (30, 60):
                selected = set(test.nlargest(top, "estimated_score").player_id)
                hits = len(selected & truth[season][top])
                reports.append({
                    "target_season": season,
                    "model": "learned_lag_ridge",
                    "board": board,
                    "top_k": top,
                    "train_through": seasons[index - 1],
                    "hit_count": hits,
                    "possible_hits": top,
                    "hit_rate": hits / top,
                    "candidate_pool_ceiling": len(set(test.player_id) & truth[season][top]) / top,
                    "score_mae": float((test.estimated_score - test.actual_score).abs().mean()),
                })
                details.append(test[["target_season", "player_id", "estimated_score"]].assign(
                    board=board, top_k=top, model="learned_lag_ridge"
                ))
    return ProbabilityBacktest(pd.DataFrame(reports), pd.concat(details, ignore_index=True))
