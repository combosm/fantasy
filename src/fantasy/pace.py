"""Team pace forecasts selected by season-ahead validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings

FEATURES = [
    "prior_team_pace",
    "roster_prior_pace",
    "roster_prior_mpg",
    "roster_prior_usage",
    "roster_prior_fga",
    "returning_minutes_share",
    "known_player_count",
]


@dataclass
class PaceResult:
    predictions: pd.DataFrame
    backtest: pd.DataFrame
    selected_model: str


def _team_pace(season: pd.DataFrame) -> pd.Series:
    return season.groupby("team_id")["team_pace"].median()


def build_pace_features(prior: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    prior = prior.copy()
    roster = roster.copy()
    prior["player_id"] = prior["player_id"].astype(str)
    prior["team_id"] = prior["team_id"].astype(str)
    roster["player_id"] = roster["player_id"].astype(str)
    roster["team_id"] = roster["team_id"].astype(str)
    prior = prior.drop_duplicates("player_id")
    roster = roster[["player_id", "team_id"]].drop_duplicates("player_id")
    joined = roster.merge(prior, on="player_id", suffixes=("_destination", ""))
    joined = joined.rename(columns={"team_id_destination": "destination_team_id"})
    prior_paces = _team_pace(prior)
    rows = []
    for team_id in sorted(roster["team_id"].unique()):
        players = joined[joined["destination_team_id"] == team_id]
        weights = players["mpg"] * players["games"]
        if len(players) and weights.sum() > 0:
            roster_pace = float(np.average(players["team_pace"], weights=weights))
            roster_mpg = float(np.average(players["mpg"], weights=weights))
            roster_usage = float(np.average(players["usage"], weights=weights))
            roster_fga = float(np.average(players["fga"], weights=weights))
            returning = float(weights[players["team_id"] == team_id].sum() / weights.sum())
        else:
            roster_pace = roster_mpg = roster_usage = roster_fga = np.nan
            returning = 0.0
        rows.append(
            {
                "team_id": team_id,
                "prior_team_pace": float(prior_paces.get(team_id, np.nan)),
                "roster_prior_pace": roster_pace,
                "roster_prior_mpg": roster_mpg,
                "roster_prior_usage": roster_usage,
                "roster_prior_fga": roster_fga,
                "returning_minutes_share": returning,
                "known_player_count": len(players),
            }
        )
    return pd.DataFrame(rows)


def forecast_pace(
    history: pd.DataFrame, current_roster: pd.DataFrame, settings: Settings
) -> PaceResult:
    seasons = settings.historical_seasons
    if len(seasons) == 1:
        current = build_pace_features(history, current_roster)
        current["projected_pace"] = current["prior_team_pace"].fillna(
            _team_pace(history).median()
        )
        return PaceResult(
            current[["team_id", "projected_pace"]],
            pd.DataFrame(columns=["target_season", "metric", "model", "n", "mae", "selected"]),
            "previous_season",
        )
    examples = []
    for index in range(1, len(seasons)):
        prior = history[history["season"] == seasons[index - 1]]
        actual = history[history["season"] == seasons[index]]
        # Do not expose target-season team assignments to historical features.
        features = build_pace_features(prior, prior[["player_id", "team_id"]])
        labels = _team_pace(actual).rename("actual_pace").reset_index()
        labeled = features.merge(labels, on="team_id", how="inner")
        labeled["target_season"] = seasons[index]
        examples.append(labeled)
    training = pd.concat(examples, ignore_index=True)
    reports = []
    for index in range(2, len(seasons)):
        train = training[training["target_season"].isin(seasons[1:index])]
        valid = training[training["target_season"] == seasons[index]]
        if train.empty or valid.empty:
            continue
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        model.fit(train[FEATURES], train["actual_pace"])
        for name, predicted in [
            ("ridge", model.predict(valid[FEATURES])),
            ("previous_season", valid["prior_team_pace"].fillna(train["actual_pace"].mean())),
        ]:
            reports.append(
                {
                    "target_season": seasons[index],
                    "metric": "pace",
                    "model": name,
                    "n": len(valid),
                    "mae": mean_absolute_error(valid["actual_pace"], predicted),
                }
            )
    backtest = pd.DataFrame(
        reports, columns=["target_season", "metric", "model", "n", "mae"]
    )
    if backtest.empty:
        selected = "previous_season"
    else:
        errors = backtest.groupby("model").apply(
            lambda group: np.average(group["mae"], weights=group["n"]), include_groups=False
        )
        selected = str(errors.idxmin())
    backtest["selected"] = backtest["model"] == selected
    latest = history[history["season"] == seasons[-1]]
    current = build_pace_features(latest, current_roster)
    if selected == "ridge":
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10))
        model.fit(training[FEATURES], training["actual_pace"])
        current["projected_pace"] = model.predict(current[FEATURES])
    else:
        current["projected_pace"] = current["prior_team_pace"].fillna(
            _team_pace(latest).median()
        )
    return PaceResult(current[["team_id", "projected_pace"]], backtest, selected)
