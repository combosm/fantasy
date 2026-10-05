"""Empirical NBA debut priors by official draft slot and age.

These are fallbacks for drafted rookies without source-league box scores, not
claims that their individual skills are known. Historical undrafted and
non-playing draftees are not represented by the training target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings

TARGETS = [
    "games", "mpg", "usage", "reb", "ast", "stl", "blk", "tov",
    "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
]


def rookie_training_rows(history: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    actual = history.copy()
    actual["draft_year"] = actual.season.str[:4].astype(int)
    actual["player_id"] = actual.player_id.astype(str)
    picks = draft[["draft_year", "player_id", "draft_pick"]].copy()
    picks["player_id"] = picks.player_id.astype(str)
    rows = actual.merge(picks, on=["draft_year", "player_id"], how="inner")
    rows = rows.dropna(subset=["age", "draft_pick", *TARGETS])
    return rows[rows.draft_pick.between(1, 60)].copy()


def _features(frame: pd.DataFrame) -> np.ndarray:
    pick = np.log1p(frame.draft_pick.to_numpy(dtype=float))
    age = frame.age.to_numpy(dtype=float)
    return np.column_stack([pick, pick**2, age, age * pick])


def _fit(training: pd.DataFrame):
    if len(training) < 100 or training.draft_year.nunique() < 4:
        raise ValueError("At least 100 drafted NBA debuts across four years are needed")
    model = make_pipeline(StandardScaler(), Ridge(alpha=30))
    model.fit(_features(training), training[TARGETS])
    return model


def forecast_drafted_rookies(
    history: pd.DataFrame, draft: pd.DataFrame, roster: pd.DataFrame,
    settings: Settings,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return synthetic, auditable prior-season rows for new current draftees."""
    training = rookie_training_rows(history, draft)
    model = _fit(training)
    current_year = int(settings.target_season[:4])
    picks = draft[draft.draft_year == current_year][["player_id", "draft_pick"]].copy()
    picks["player_id"] = picks.player_id.astype(str)
    current = roster.copy()
    current["player_id"] = current.player_id.astype(str)
    current = current.merge(picks, on="player_id", how="inner")
    if current.empty:
        return pd.DataFrame(), pd.DataFrame()
    if "age" not in current:
        current["age"] = training.age.median()
    current["age"] = current.age.fillna(training.age.median())
    predicted = pd.DataFrame(model.predict(_features(current)), columns=TARGETS)
    predicted["games"] = predicted.games.clip(0, 82)
    predicted["mpg"] = predicted.mpg.clip(0, 38)
    predicted["usage"] = predicted.usage.clip(0, 50)
    for stat in TARGETS[3:]:
        predicted[stat] = predicted[stat].clip(lower=0)
    for made, attempted in (("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")):
        predicted[made] = np.minimum(predicted[made], predicted[attempted])
    predicted["fg3a"] = np.minimum(predicted.fg3a, predicted.fga)
    predicted["fg3m"] = np.minimum(predicted.fg3m, predicted.fg3a)
    predicted["fgm"] = np.maximum(predicted.fgm, predicted.fg3m)
    predicted["pts"] = 2 * predicted.fgm + predicted.fg3m + predicted.ftm
    latest = history[history.season == settings.historical_seasons[-1]]
    paces = latest.groupby("team_id").team_pace.median()
    league_pace = float(latest.team_pace.median())
    result = predicted.copy()
    result.insert(0, "player_id", current.player_id.to_numpy())
    result.insert(1, "player_name", current.player_name.to_numpy())
    result.insert(2, "team_id", current.team_id.astype(str).to_numpy())
    result.insert(3, "season", settings.historical_seasons[-1])
    result["age"] = current.age.to_numpy(dtype=float) - 1
    result["team_pace"] = result.team_id.map(paces).fillna(league_pace)
    result["rookie_prior"] = True
    quality = pd.DataFrame({
        "category": "empirical draft-slot prior", "entity_id": current.player_id,
        "passed": True,
        "detail": "NBA-debut regression from historical draft slot and age; source-league stats unavailable",
    })
    return result, quality


def rookie_walk_forward(history: pd.DataFrame, draft: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Evaluate the cold-start prior by holding out one draft class at a time."""
    rows = rookie_training_rows(history, draft)
    reports = []
    weights = settings.scoring.model_dump()
    for year in sorted(rows.draft_year.unique()):
        train = rows[rows.draft_year < year]
        valid = rows[rows.draft_year == year]
        if len(train) < 100 or train.draft_year.nunique() < 4 or valid.empty:
            continue
        model = _fit(train)
        predicted = pd.DataFrame(model.predict(_features(valid)), columns=TARGETS)
        predicted["games"] = predicted.games.clip(0, 82)
        for stat in TARGETS[3:]:
            predicted[stat] = predicted[stat].clip(lower=0)
        predicted["fg3a"] = np.minimum(predicted.fg3a, predicted.fga)
        predicted["fg3m"] = np.minimum(predicted.fg3m, predicted.fg3a)
        predicted["fgm"] = np.maximum(predicted.fgm, predicted.fg3m)
        predicted["ftm"] = np.minimum(predicted.ftm, predicted.fta)
        predicted["pts"] = 2 * predicted.fgm + predicted.fg3m + predicted.ftm
        actual_fppg = sum(valid[stat].to_numpy() * weight for stat, weight in weights.items())
        predicted_fppg = sum(predicted[stat].to_numpy() * weight for stat, weight in weights.items())
        reports.append({
            "draft_year": year, "n": len(valid),
            "games_mae": float(np.abs(predicted.games.to_numpy() - valid.games.to_numpy()).mean()),
            "fppg_mae": float(np.abs(predicted_fppg - actual_fppg).mean()),
            "season_total_mae": float(np.abs(
                predicted_fppg * predicted.games.to_numpy()
                - actual_fppg * valid.games.to_numpy()
            ).mean()),
        })
    return pd.DataFrame(reports)
