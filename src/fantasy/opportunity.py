"""Empirical, separate forecasts for healthy-game minutes and usage."""

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

OWN_FEATURES = [
    "age",
    "mpg",
    "usage",
    "fga36",
    "fg3a36",
    "fta36",
    "ast36",
    "reb36",
    "stl36",
    "blk36",
    "tov36",
    "three_share",
    "team_pace",
]
TEAM_FEATURES = [
    "teammate_count",
    "teammate_mpg_sum",
    "teammate_usage_minutes",
    "teammate_top3_usage",
    "teammate_max_mpg",
    "teammate_fga_sum",
    "teammate_ast_sum",
    "teammate_fg3a_sum",
    "teammate_reb_sum",
    "similar3_mpg_sum",
    "similar3_usage_minutes",
]
FEATURES = OWN_FEATURES + [
    "prior_games", "prior2_mpg", "prior2_usage", "prior2_games",
    "prior3_mpg", "prior3_usage", "prior3_games",
    "recent_mpg", "recent_usage",
    "destination_team_pace",
    "current_mpg_rank",
    "current_usage_rank",
    "current_mpg_share",
    "current_usage_share",
    *[f"current_{name}" for name in TEAM_FEATURES],
    *[f"previous_{name}" for name in TEAM_FEATURES],
    *[f"change_{name}" for name in TEAM_FEATURES],
]


@dataclass
class OpportunityResult:
    predictions: pd.DataFrame
    backtest: pd.DataFrame
    selected_models: dict[str, str]
    residual_scales: dict[str, float]


def _profiles(history: pd.DataFrame, season: str) -> pd.DataFrame:
    frame = history[history["season"] == season].copy()
    frame["player_id"] = frame["player_id"].astype(str)
    frame["team_id"] = frame["team_id"].astype(str)
    frame = frame.drop_duplicates("player_id")
    minutes = frame["mpg"].clip(lower=1)
    for stat in ["fga", "fg3a", "fta", "ast", "reb", "stl", "blk", "tov"]:
        frame[f"{stat}36"] = frame[stat] * 36 / minutes
    frame["three_share"] = frame["fg3a"] / frame["fga"].clip(lower=1)
    return frame


def _teammate_summary(
    player_id: str,
    own_style: np.ndarray,
    teammates: pd.DataFrame,
    style_lookup: dict[str, np.ndarray],
) -> dict[str, float]:
    peers = teammates[teammates["player_id"] != player_id]
    if peers.empty:
        return dict.fromkeys(TEAM_FEATURES, 0.0)
    similarity = np.array(
        [np.linalg.norm(style_lookup[str(pid)] - own_style) for pid in peers["player_id"]]
    )
    nearest = peers.iloc[np.argsort(similarity)[:3]]
    return {
        "teammate_count": float(len(peers)),
        "teammate_mpg_sum": float(peers["mpg"].sum()),
        "teammate_usage_minutes": float((peers["usage"] * peers["mpg"]).sum()),
        "teammate_top3_usage": float(peers["usage"].nlargest(3).sum()),
        "teammate_max_mpg": float(peers["mpg"].max()),
        "teammate_fga_sum": float(peers["fga"].sum()),
        "teammate_ast_sum": float(peers["ast"].sum()),
        "teammate_fg3a_sum": float(peers["fg3a"].sum()),
        "teammate_reb_sum": float(peers["reb"].sum()),
        "similar3_mpg_sum": float(nearest["mpg"].sum()),
        "similar3_usage_minutes": float((nearest["usage"] * nearest["mpg"]).sum()),
    }


def build_features(prior: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Use only prior-season player statistics and the known destination roster."""
    prior = prior.copy()
    prior["player_id"] = prior["player_id"].astype(str)
    prior["team_id"] = prior["team_id"].astype(str)
    roster = roster[["player_id", "team_id"]].copy()
    roster["player_id"] = roster["player_id"].astype(str)
    roster["team_id"] = roster["team_id"].astype(str)
    roster = roster.drop_duplicates("player_id")
    joined = roster.merge(prior, on="player_id", how="inner", suffixes=("_destination", ""))
    joined = joined.rename(columns={"team_id_destination": "destination_team_id"})
    if joined.empty:
        return pd.DataFrame(columns=["player_id", *FEATURES])
    style_cols = ["usage", "ast36", "reb36", "fg3a36", "blk36"]
    raw_style = prior[style_cols].replace([np.inf, -np.inf], np.nan)
    style_median = raw_style.median().fillna(0)
    style_std = raw_style.std().replace(0, 1).fillna(1)
    normalized = ((raw_style.fillna(style_median) - style_median) / style_std).to_numpy()
    style_lookup = dict(zip(prior["player_id"], normalized))
    current_groups = {key: group for key, group in joined.groupby("destination_team_id")}
    prior_groups = {key: group for key, group in prior.groupby("team_id")}
    team_pace = prior.groupby("team_id")["team_pace"].median()
    rows: list[dict[str, float | str]] = []
    for _, player in joined.iterrows():
        pid = str(player["player_id"])
        destination = str(player["destination_team_id"])
        original = str(player["team_id"])
        own_style = style_lookup[pid]
        current = _teammate_summary(pid, own_style, current_groups[destination], style_lookup)
        previous = _teammate_summary(
            pid, own_style, prior_groups.get(original, prior.iloc[0:0]), style_lookup
        )
        item: dict[str, float | str] = {"player_id": pid}
        for feature in OWN_FEATURES:
            item[feature] = float(player[feature])
        item["prior_games"] = float(player["games"])
        item["destination_team_pace"] = float(team_pace.get(destination, player["team_pace"]))
        teammates_on_destination = current_groups[destination]
        item["current_mpg_rank"] = float(
            1 + (teammates_on_destination["mpg"] > player["mpg"]).sum()
        )
        item["current_usage_rank"] = float(
            1 + (teammates_on_destination["usage"] > player["usage"]).sum()
        )
        item["current_mpg_share"] = float(
            player["mpg"] / max(teammates_on_destination["mpg"].sum(), 1)
        )
        item["current_usage_share"] = float(
            player["usage"]
            * player["mpg"]
            / max((teammates_on_destination["usage"] * teammates_on_destination["mpg"]).sum(), 1)
        )
        for name in TEAM_FEATURES:
            item[f"current_{name}"] = current[name]
            item[f"previous_{name}"] = previous[name]
            item[f"change_{name}"] = current[name] - previous[name]
        rows.append(item)
    return pd.DataFrame(rows)


def _add_role_history(
    features: pd.DataFrame, history: pd.DataFrame, seasons: list[str], prior_index: int
) -> pd.DataFrame:
    """Expose pre-cutoff role history, including a short disrupted season."""
    out = features.copy()
    for lag in (2, 3):
        index = prior_index - lag + 1
        label = f"prior{lag}"
        if index < 0:
            for field in ("mpg", "usage", "games"):
                out[f"{label}_{field}"] = np.nan
            continue
        past = history[history.season == seasons[index]][
            ["player_id", "mpg", "usage", "games"]
        ].drop_duplicates("player_id").copy()
        past["player_id"] = past.player_id.astype(str)
        out = out.merge(
            past.rename(columns={field: f"{label}_{field}" for field in ("mpg", "usage", "games")}),
            on="player_id", how="left", validate="one_to_one"
        )
    for field in ("mpg", "usage"):
        numerator = out[field] * out.prior_games
        denominator = out.prior_games.copy()
        for lag in (2, 3):
            games = out[f"prior{lag}_games"].fillna(0)
            numerator = numerator + out[f"prior{lag}_{field}"].fillna(0) * games
            denominator = denominator + games
        out[f"recent_{field}"] = numerator / denominator.clip(lower=1)
    return out


def _candidate(name: str):
    if name in {"ridge", "ridge_change", "ridge_recent_change"}:
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(), Ridge(alpha=30)
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


def forecast_opportunity(
    history: pd.DataFrame, current_roster: pd.DataFrame, settings: Settings,
    current_prior_extra: pd.DataFrame | None = None,
) -> OpportunityResult:
    seasons = settings.historical_seasons
    current_history = (
        pd.concat([history, current_prior_extra], ignore_index=True)
        if current_prior_extra is not None and not current_prior_extra.empty
        else history
    )
    if len(seasons) == 1:
        current_prior = _profiles(current_history, seasons[-1])
        current_features = _add_role_history(
            build_features(current_prior, current_roster), history, seasons, 0
        )
        predictions = current_features[["player_id"]].copy()
        predictions["disrupted_role"] = False
        predictions["stable_starter_role"] = False
        predictions["established_starter_role"] = False
        for target, source in [("mpg", "mpg"), ("usage", "usage")]:
            predictions[f"predicted_{target}"] = current_features[source]
        return OpportunityResult(
            predictions,
            pd.DataFrame(columns=["target_season", "metric", "model", "n", "mae", "selected"]),
            {target: "previous_season" for target in ["mpg", "usage"]},
            {},
        )
    examples: list[pd.DataFrame] = []
    for index in range(1, len(seasons)):
        prior = _profiles(history, seasons[index - 1])
        prior["prior_fppg"] = sum(
            prior[stat] * weight for stat, weight in settings.scoring.model_dump().items()
        )
        prior["prior_fppg_rank"] = prior.prior_fppg.rank(ascending=False, method="first")
        actual = _profiles(history, seasons[index])
        # The eventual team is unavailable at the preseason cutoff. Until a
        # dated snapshot exists, train on the prior roster carried forward.
        roster = prior[["player_id", "team_id"]]
        features = _add_role_history(build_features(prior, roster), history, seasons, index - 1)
        features = features.merge(
            prior[["player_id", "prior_fppg_rank"]], on="player_id", how="left"
        )
        labeled = features.merge(
        actual[["player_id", "mpg", "usage", "games"]].rename(
                columns={"mpg": "target_mpg", "usage": "target_usage", "games": "target_games"}
            ),
            on="player_id",
            how="inner",
        )
        labeled = labeled[labeled["target_games"] >= 1].copy()
        labeled["target_season"] = seasons[index]
        examples.append(labeled)
    training = pd.concat(examples, ignore_index=True)
    reports: list[dict[str, float | str | int]] = []
    selected: dict[str, str] = {}
    disrupted_selected: dict[str, str] = {}
    stable_starter_mpg_model = ""
    established_starter_mpg_model = ""
    scales: dict[str, float] = {}
    predicted_models: dict[str, object] = {}
    disrupted_models: dict[str, object] = {}
    for target in ["mpg", "usage"]:
        target_col = f"target_{target}"
        feature_names = FEATURES
        baseline_col = target
        for validation_index in range(2, len(seasons)):
            train_seasons = set(seasons[1:validation_index])
            validation_season = seasons[validation_index]
            train = training[training["target_season"].isin(train_seasons)]
            valid = training[training["target_season"] == validation_season]
            relevant = valid.prior_fppg_rank <= settings.league.drafted_players
            disrupted = (
                relevant & valid.prior_games.lt(50) & valid.prior2_games.ge(40)
            )
            stable_starter = (
                disrupted
                & valid.prior2_mpg.ge(30)
                & valid.prior3_mpg.ge(30)
                & valid.prior3_games.ge(40)
            )
            established_starter = (
                relevant
                & valid.mpg.ge(30)
                & valid.prior2_mpg.ge(30)
                & valid.prior2_games.ge(40)
            )
            for name in ["ridge", "ridge_change", "ridge_recent_change", "gradient_boosting"]:
                model = _candidate(name)
                train_target = (
                    train[target_col] - train[
                        f"recent_{target}" if name == "ridge_recent_change" else baseline_col
                    ] if name in {"ridge_change", "ridge_recent_change"} else train[target_col]
                )
                model.fit(train[feature_names], train_target)
                predicted = model.predict(valid[feature_names])
                if name in {"ridge_change", "ridge_recent_change"}:
                    predicted += valid[
                        f"recent_{target}" if name == "ridge_recent_change" else baseline_col
                    ].to_numpy()
                reports.append(
                    {
                        "target_season": validation_season,
                        "metric": target,
                        "model": name,
                        "n": len(valid),
                        "mae": mean_absolute_error(valid[target_col], predicted),
                        "mse": mean_squared_error(valid[target_col], predicted),
                        "n_relevant": int(relevant.sum()),
                        "draft_relevant_mse": mean_squared_error(
                            valid.loc[relevant, target_col], predicted[relevant.to_numpy()]
                        ) if relevant.any() else np.nan,
                        "n_disrupted": int(disrupted.sum()),
                        "disrupted_mse": mean_squared_error(
                            valid.loc[disrupted, target_col], predicted[disrupted.to_numpy()]
                        ) if disrupted.any() else np.nan,
                        "disrupted_bias": float(
                            (valid.loc[disrupted, target_col].to_numpy()
                             - predicted[disrupted.to_numpy()]).mean()
                        ) if disrupted.any() else np.nan,
                        "n_stable_starter": int(stable_starter.sum()),
                        "stable_starter_mse": mean_squared_error(
                            valid.loc[stable_starter, target_col],
                            predicted[stable_starter.to_numpy()],
                        ) if stable_starter.any() else np.nan,
                        "n_established_starter": int(established_starter.sum()),
                        "established_starter_mse": mean_squared_error(
                            valid.loc[established_starter, target_col],
                            predicted[established_starter.to_numpy()],
                        ) if established_starter.any() else np.nan,
                    }
                )
            reports.append(
                {
                    "target_season": validation_season,
                    "metric": target,
                    "model": "previous_season",
                    "n": len(valid),
                    "mae": mean_absolute_error(valid[target_col], valid[baseline_col]),
                    "mse": mean_squared_error(valid[target_col], valid[baseline_col]),
                    "n_relevant": int(relevant.sum()),
                    "draft_relevant_mse": mean_squared_error(
                        valid.loc[relevant, target_col], valid.loc[relevant, baseline_col]
                    ) if relevant.any() else np.nan,
                    "n_disrupted": int(disrupted.sum()),
                    "disrupted_mse": mean_squared_error(
                        valid.loc[disrupted, target_col], valid.loc[disrupted, baseline_col]
                    ) if disrupted.any() else np.nan,
                    "disrupted_bias": float(
                        (valid.loc[disrupted, target_col]
                         - valid.loc[disrupted, baseline_col]).mean()
                    ) if disrupted.any() else np.nan,
                    "n_stable_starter": int(stable_starter.sum()),
                    "stable_starter_mse": mean_squared_error(
                        valid.loc[stable_starter, target_col],
                        valid.loc[stable_starter, baseline_col],
                    ) if stable_starter.any() else np.nan,
                    "n_established_starter": int(established_starter.sum()),
                    "established_starter_mse": mean_squared_error(
                        valid.loc[established_starter, target_col],
                        valid.loc[established_starter, baseline_col],
                    ) if established_starter.any() else np.nan,
                }
            )
            reports.append(
                {
                    "target_season": validation_season,
                    "metric": target,
                    "model": "recent_three_year",
                    "n": len(valid),
                    "mae": mean_absolute_error(valid[target_col], valid[f"recent_{target}"]),
                    "mse": mean_squared_error(valid[target_col], valid[f"recent_{target}"]),
                    "n_relevant": int(relevant.sum()),
                    "draft_relevant_mse": mean_squared_error(
                        valid.loc[relevant, target_col], valid.loc[relevant, f"recent_{target}"]
                    ) if relevant.any() else np.nan,
                    "n_disrupted": int(disrupted.sum()),
                    "disrupted_mse": mean_squared_error(
                        valid.loc[disrupted, target_col], valid.loc[disrupted, f"recent_{target}"]
                    ) if disrupted.any() else np.nan,
                    "disrupted_bias": float(
                        (valid.loc[disrupted, target_col]
                         - valid.loc[disrupted, f"recent_{target}"]).mean()
                    ) if disrupted.any() else np.nan,
                    "n_stable_starter": int(stable_starter.sum()),
                    "stable_starter_mse": mean_squared_error(
                        valid.loc[stable_starter, target_col],
                        valid.loc[stable_starter, f"recent_{target}"],
                    ) if stable_starter.any() else np.nan,
                    "n_established_starter": int(established_starter.sum()),
                    "established_starter_mse": mean_squared_error(
                        valid.loc[established_starter, target_col],
                        valid.loc[established_starter, f"recent_{target}"],
                    ) if established_starter.any() else np.nan,
                }
            )
        target_reports = [r for r in reports if r["metric"] == target]
        candidate_errors = {}
        if target_reports:
            for name in [
                "ridge", "ridge_change", "ridge_recent_change", "gradient_boosting",
                "previous_season", "recent_three_year",
            ]:
                candidate_errors[name] = np.average(
                    [r["draft_relevant_mse"] for r in target_reports if r["model"] == name],
                    weights=[r["n_relevant"] for r in target_reports if r["model"] == name],
                )
        chosen = min(candidate_errors, key=candidate_errors.get) if target_reports else "previous_season"
        selected[target] = chosen
        disrupted_count = sum(
            r["n_disrupted"] for r in target_reports if r["model"] == chosen
        )
        disrupted_errors = {
            name: np.average(
                [r["disrupted_mse"] for r in target_reports
                 if r["model"] == name and r["n_disrupted"] > 0],
                weights=[r["n_disrupted"] for r in target_reports
                         if r["model"] == name and r["n_disrupted"] > 0],
            )
            for name in candidate_errors
        } if disrupted_count >= 10 else {}
        disrupted_chosen = min(disrupted_errors, key=disrupted_errors.get) if disrupted_errors else chosen
        disrupted_selected[target] = disrupted_chosen
        if target == "mpg":
            established_count = sum(
                r["n_established_starter"] for r in target_reports if r["model"] == chosen
            )
            if established_count >= 20:
                established_errors = {
                    name: np.average(
                        [r["established_starter_mse"] for r in target_reports
                         if r["model"] == name and r["n_established_starter"] > 0],
                        weights=[r["n_established_starter"] for r in target_reports
                                 if r["model"] == name and r["n_established_starter"] > 0],
                    )
                    for name in candidate_errors
                }
                established_starter_mpg_model = min(
                    established_errors, key=established_errors.get
                )
                # A tiny pooled MSE edge is not enough to replace a stable
                # starter's observed active-game role. Compare paired season
                # errors and prefer that simpler baseline within one SE.
                best = established_starter_mpg_model
                paired = pd.DataFrame(
                    [r for r in target_reports if r["model"] in {best, "previous_season"}]
                ).pivot(index="target_season", columns="model", values="established_starter_mse")
                paired = paired.dropna()
                if best != "previous_season" and not paired.empty:
                    difference = paired["previous_season"] - paired[best]
                    standard_error = (
                        float(difference.std(ddof=1) / np.sqrt(len(difference)))
                        if len(difference) >= 2 else float("inf")
                    )
                    if established_errors["previous_season"] - established_errors[best] <= standard_error:
                        established_starter_mpg_model = "previous_season"
            else:
                established_starter_mpg_model = chosen
            stable_count = sum(
                r["n_stable_starter"] for r in target_reports if r["model"] == chosen
            )
            if stable_count >= 20:
                stable_errors = {
                    name: np.average(
                        [r["stable_starter_mse"] for r in target_reports
                         if r["model"] == name and r["n_stable_starter"] > 0],
                        weights=[r["n_stable_starter"] for r in target_reports
                                 if r["model"] == name and r["n_stable_starter"] > 0],
                    )
                    for name in candidate_errors
                }
                stable_starter_mpg_model = min(stable_errors, key=stable_errors.get)
            else:
                stable_starter_mpg_model = disrupted_chosen
        if disrupted_chosen not in {"previous_season", "recent_three_year"}:
            disrupted_model = _candidate(disrupted_chosen)
            disrupted_target = (
                training[target_col] - training[
                    f"recent_{target}" if disrupted_chosen == "ridge_recent_change" else baseline_col
                ] if disrupted_chosen in {"ridge_change", "ridge_recent_change"}
                else training[target_col]
            )
            disrupted_model.fit(training[feature_names], disrupted_target)
            disrupted_models[target] = disrupted_model
        if chosen in {"previous_season", "recent_three_year"}:
            predicted_models[target] = None
            baseline = baseline_col if chosen == "previous_season" else f"recent_{target}"
            residual = training[target_col] - training[baseline]
        else:
            model = _candidate(chosen)
            train_target = (
                training[target_col] - training[
                    f"recent_{target}" if chosen == "ridge_recent_change" else baseline_col
                ] if chosen in {"ridge_change", "ridge_recent_change"} else training[target_col]
            )
            model.fit(training[feature_names], train_target)
            predicted_models[target] = model
            fitted = model.predict(training[feature_names])
            if chosen in {"ridge_change", "ridge_recent_change"}:
                fitted += training[
                    f"recent_{target}" if chosen == "ridge_recent_change" else baseline_col
                ].to_numpy()
            residual = training[target_col] - fitted
        scales[target] = float(np.sqrt(np.mean(np.square(residual))))
    current_prior = _profiles(current_history, seasons[-1])
    current_prior["prior_fppg"] = sum(
        current_prior[stat] * weight for stat, weight in settings.scoring.model_dump().items()
    )
    current_prior["prior_fppg_rank"] = current_prior.prior_fppg.rank(
        ascending=False, method="first"
    )
    current_features = _add_role_history(
        build_features(current_prior, current_roster), history, seasons, len(seasons) - 1
    )
    current_features = current_features.merge(
        current_prior[["player_id", "prior_fppg_rank"]], on="player_id", how="left",
        validate="one_to_one",
    )
    disrupted_current = (
        current_features.prior_fppg_rank.le(settings.league.drafted_players)
        & current_features.prior_games.lt(50)
        & current_features.prior2_games.ge(40)
    )
    stable_starter_current = (
        disrupted_current
        & current_features.prior2_mpg.ge(30)
        & current_features.prior3_mpg.ge(30)
        & current_features.prior3_games.ge(40)
    )
    established_starter_current = (
        current_features.prior_fppg_rank.le(settings.league.drafted_players)
        & current_features.mpg.ge(30)
        & current_features.prior2_mpg.ge(30)
        & current_features.prior2_games.ge(40)
    )
    predictions = current_features[["player_id"]].copy()
    predictions["disrupted_role"] = disrupted_current.to_numpy()
    predictions["stable_starter_role"] = stable_starter_current.to_numpy()
    predictions["established_starter_role"] = established_starter_current.to_numpy()
    for target, model in predicted_models.items():
        predictions[f"predicted_{target}"] = (
            current_features[
                f"recent_{target}" if selected[target] == "recent_three_year" else target
            ]
            if model is None
            else model.predict(current_features[FEATURES])
            + (
                current_features[
                    f"recent_{target}" if selected[target] == "ridge_recent_change" else target
                ].to_numpy()
                if selected[target] in {"ridge_change", "ridge_recent_change"} else 0
            )
        )
        disrupted_chosen = disrupted_selected[target]
        if disrupted_chosen != selected[target] and disrupted_current.any():
            if disrupted_chosen in {"previous_season", "recent_three_year"}:
                alternate = current_features[
                    f"recent_{target}" if disrupted_chosen == "recent_three_year" else target
                ].to_numpy()
            else:
                alternate = disrupted_models[target].predict(current_features[FEATURES])
                if disrupted_chosen in {"ridge_change", "ridge_recent_change"}:
                    alternate += current_features[
                        f"recent_{target}" if disrupted_chosen == "ridge_recent_change" else target
                    ].to_numpy()
            predictions.loc[disrupted_current, f"predicted_{target}"] = alternate[disrupted_current.to_numpy()]
    if established_starter_current.any():
        if established_starter_mpg_model == "previous_season":
            established_mpg = current_features.mpg.to_numpy()
        elif established_starter_mpg_model == "recent_three_year":
            established_mpg = current_features.recent_mpg.to_numpy()
        else:
            established_model = _candidate(established_starter_mpg_model)
            established_target = (
                training.target_mpg - training[
                    "recent_mpg" if established_starter_mpg_model == "ridge_recent_change" else "mpg"
                ]
                if established_starter_mpg_model in {"ridge_change", "ridge_recent_change"}
                else training.target_mpg
            )
            established_model.fit(training[FEATURES], established_target)
            established_mpg = established_model.predict(current_features[FEATURES])
            if established_starter_mpg_model in {"ridge_change", "ridge_recent_change"}:
                established_mpg += current_features[
                    "recent_mpg" if established_starter_mpg_model == "ridge_recent_change" else "mpg"
                ].to_numpy()
        predictions.loc[established_starter_current, "predicted_mpg"] = established_mpg[
            established_starter_current.to_numpy()
        ]
    if stable_starter_current.any():
        if stable_starter_mpg_model == "recent_three_year":
            stable_mpg = current_features.recent_mpg.to_numpy()
        elif stable_starter_mpg_model == "previous_season":
            stable_mpg = current_features.mpg.to_numpy()
        else:
            stable_model = _candidate(stable_starter_mpg_model)
            stable_target = (
                training.target_mpg - training[
                    "recent_mpg" if stable_starter_mpg_model == "ridge_recent_change" else "mpg"
                ]
                if stable_starter_mpg_model in {"ridge_change", "ridge_recent_change"}
                else training.target_mpg
            )
            stable_model.fit(training[FEATURES], stable_target)
            stable_mpg = stable_model.predict(current_features[FEATURES])
            if stable_starter_mpg_model in {"ridge_change", "ridge_recent_change"}:
                stable_mpg += current_features[
                    "recent_mpg" if stable_starter_mpg_model == "ridge_recent_change" else "mpg"
                ].to_numpy()
        predictions.loc[stable_starter_current, "predicted_mpg"] = stable_mpg[
            stable_starter_current.to_numpy()
        ]
    predictions["predicted_mpg"] = predictions["predicted_mpg"].clip(0, 38)
    predictions["predicted_usage"] = predictions["predicted_usage"].clip(0, 50)
    backtest = pd.DataFrame(
        reports, columns=[
            "target_season", "metric", "model", "n", "mae", "mse",
            "n_relevant", "draft_relevant_mse", "n_disrupted", "disrupted_mse",
            "disrupted_bias", "n_stable_starter", "stable_starter_mse",
            "n_established_starter", "established_starter_mse",
        ]
    )
    backtest["selected"] = (
        backtest.apply(lambda row: row["model"] == selected[row["metric"]], axis=1)
        if not backtest.empty
        else pd.Series(dtype=bool)
    )
    selected.update({f"{target}_disrupted": model for target, model in disrupted_selected.items()})
    selected["mpg_stable_starter"] = stable_starter_mpg_model
    selected["mpg_established_starter"] = established_starter_mpg_model
    return OpportunityResult(predictions, backtest, selected, scales)
