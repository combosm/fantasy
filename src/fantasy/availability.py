"""Expected games for rostered players, without treating league exits as injuries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings

FEATURES = [
    "age",
    "prior_games",
    "prior2_games",
    "prior3_games",
    "prior_mpg",
    "prior2_mpg",
    "prior3_mpg",
    "prior_minutes_load",
    "prior2_minutes_load",
    "prior3_minutes_load",
    "prior_usage",
    "observed_prior_seasons",
    "two_year_games_mean",
    "three_year_games_mean",
]
SHORT_SCHEDULE_SEASONS = {"2019-20", "2020-21"}


@dataclass
class AvailabilityResult:
    predictions: pd.DataFrame
    backtest: pd.DataFrame
    selected_model: str


def build_availability_features(
    history: pd.DataFrame, seasons: list[str], prior_index: int, roster: pd.DataFrame
) -> pd.DataFrame:
    prior = history[history["season"] == seasons[prior_index]].drop_duplicates("player_id")
    frame = roster[["player_id"]].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    prior = prior.copy()
    prior["player_id"] = prior["player_id"].astype(str)
    prior = prior.set_index("player_id")
    frame["age"] = frame["player_id"].map(prior["age"]) + 1
    frame["prior_usage"] = frame["player_id"].map(prior["usage"])
    for lag in range(1, 4):
        season_index = prior_index - lag + 1
        label = "prior" if lag == 1 else f"prior{lag}"
        if season_index < 0:
            frame[f"{label}_games"] = np.nan
            frame[f"{label}_mpg"] = np.nan
            frame[f"{label}_minutes_load"] = np.nan
            continue
        past_season = seasons[season_index]
        past = history[history["season"] == past_season].copy()
        past["player_id"] = past["player_id"].astype(str)
        past = past.drop_duplicates("player_id").set_index("player_id")
        games = frame["player_id"].map(past["games"])
        if past_season == "2020-21":
            games = (games * 82 / 72).clip(upper=82)
        elif past_season == "2019-20":
            # Team schedules varied in the bubble; without team game logs,
            # raw GP cannot distinguish schedule length from injury.
            games = games * np.nan
        frame[f"{label}_games"] = games
        frame[f"{label}_mpg"] = frame["player_id"].map(past["mpg"])
        frame[f"{label}_minutes_load"] = (
            frame[f"{label}_games"] * frame[f"{label}_mpg"]
        )
    game_columns = ["prior_games", "prior2_games", "prior3_games"]
    frame["observed_prior_seasons"] = frame[game_columns].notna().sum(axis=1)
    frame["two_year_games_mean"] = frame[game_columns[:2]].mean(axis=1)
    frame["three_year_games_mean"] = frame[game_columns].mean(axis=1)
    return frame


def _candidate(name: str):
    if name == "ridge":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=30),
        )
    if name == "gradient_boosting":
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            HistGradientBoostingRegressor(
                max_iter=180,
                learning_rate=0.05,
                max_leaf_nodes=15,
                min_samples_leaf=25,
                l2_regularization=1.0,
                random_state=42,
            ),
        )
    raise ValueError(name)


def forecast_availability(
    history: pd.DataFrame, roster: pd.DataFrame, settings: Settings
) -> AvailabilityResult:
    seasons = settings.historical_seasons
    current = build_availability_features(history, seasons, len(seasons) - 1, roster)
    if len(seasons) == 1:
        result = current[["player_id"]].copy()
        result["predicted_games"] = current["prior_games"].clip(0, 82)
        return AvailabilityResult(
            result,
            pd.DataFrame(columns=["target_season", "model", "n", "mae", "selected"]),
            "previous_season",
        )
    examples = []
    for index in range(1, len(seasons)):
        if seasons[index] in SHORT_SCHEDULE_SEASONS:
            continue
        prior = history[history["season"] == seasons[index - 1]]
        previous_roster = prior[["player_id", "team_id"]].drop_duplicates("player_id")
        features = build_availability_features(history, seasons, index - 1, previous_roster)
        actual = history[history["season"] == seasons[index]][["player_id", "games"]]
        actual = actual.copy()
        actual["player_id"] = actual["player_id"].astype(str)
        labeled = features.merge(actual.drop_duplicates("player_id"), on="player_id", how="left")
        # A missing target-season box-score row does not establish that the
        # player was rostered but injured. With carried-forward rosters it
        # includes retirees and players who left the league. Train GP
        # conditional on known target-season participation until dated
        # preseason membership can identify genuine zero-game seasons.
        labeled = labeled[labeled["games"].notna()].copy()
        labeled["target_games"] = labeled["games"]
        labeled["target_season"] = seasons[index]
        examples.append(labeled)
    training = pd.concat(examples, ignore_index=True)
    reports = []
    for index in range(2, len(seasons)):
        train = training[training["target_season"].isin(seasons[1:index])]
        valid = training[training["target_season"] == seasons[index]]
        if train.empty or valid.empty:
            continue
        for name in ["ridge", "gradient_boosting"]:
            model = _candidate(name)
            model.fit(train[FEATURES], train["target_games"])
            predicted = model.predict(valid[FEATURES]).clip(0, 82)
            reports.append(
                {"target_season": seasons[index], "model": name, "n": len(valid),
                 "mae": mean_absolute_error(valid["target_games"], predicted),
                 "mse": mean_squared_error(valid["target_games"], predicted)}
            )
        prior_valid = valid[valid["prior_games"].notna()]
        if not prior_valid.empty:
            reports.append(
                {"target_season": seasons[index], "model": "previous_season", "n": len(prior_valid),
                 "mae": mean_absolute_error(prior_valid["target_games"], prior_valid["prior_games"]),
                 "mse": mean_squared_error(prior_valid["target_games"], prior_valid["prior_games"])}
            )
    backtest = pd.DataFrame(reports, columns=["target_season", "model", "n", "mae", "mse"])
    if backtest.empty:
        chosen = "previous_season"
    else:
        errors = backtest.groupby("model").apply(
            lambda group: np.average(group["mse"], weights=group["n"]), include_groups=False
        )
        chosen = str(errors.idxmin())
    backtest["selected"] = backtest["model"] == chosen
    if chosen == "previous_season":
        predicted = current["prior_games"]
    else:
        model = _candidate(chosen)
        model.fit(training[FEATURES], training["target_games"])
        predicted = model.predict(current[FEATURES])
    result = current[["player_id"]].copy()
    result["predicted_games"] = np.clip(predicted, 0, 82)
    return AvailabilityResult(result, backtest, chosen)
