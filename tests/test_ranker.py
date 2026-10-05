from __future__ import annotations

import pandas as pd
import pytest

from fantasy.config import load_settings
from fantasy.evaluation import BacktestResult
from fantasy.game_form import EVENTS, FORM_COLUMNS, build_game_form
from fantasy.learned_lag import walk_forward_learned_lags
from fantasy.ranker import walk_forward_topk_probabilities


@pytest.mark.parametrize(
    "approach", [
        "forecast_logistic", "trajectory_boosted", "trajectory_logistic",
        "game_form_logistic",
    ]
)
def test_probability_ranker_never_trains_on_target_season_labels(approach):
    settings = load_settings()
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    history_rows = []
    forecast_rows = []
    for season in seasons:
        for number in range(90):
            player_id = str(number)
            history_rows.append({
                "season": season, "player_id": player_id, "games": 1,
                "pts": float(90 - number), "age": 25.0,
                "mpg": 30.0, "usage": 20.0,
            })
            for model in ("factor_model", "previous_season", "three_year_weighted"):
                forecast_rows.append({
                    "target_season": season, "player_id": player_id,
                    "model": model, "pred_fppg": float(90 - number),
                    "pred_games": 1.0,
                })
    history = pd.DataFrame(history_rows)
    for stat in settings.scoring.model_dump():
        if stat not in history:
            history[stat] = 0.0
    backtest = BacktestResult(
        metrics=pd.DataFrame({
            "target_season": seasons,
            "model": ["factor_model"] * len(seasons),
            "replacement_fppg": [0.0] * len(seasons),
        }),
        predictions=pd.DataFrame(forecast_rows),
    )
    form = pd.DataFrame([
        {"target_season": season, "player_id": str(number), **{
            column: float(number % 3) for column in FORM_COLUMNS
        }}
        for season in seasons for number in range(90)
    ])
    before = walk_forward_topk_probabilities(
        backtest, history, settings, approach=approach, game_form=form
    )
    changed = history.copy()
    mask = changed.season == seasons[-1]
    changed.loc[mask, "pts"] = 100 - changed.loc[mask, "pts"]
    after = walk_forward_topk_probabilities(
        backtest, changed, settings, approach=approach, game_form=form
    )
    pd.testing.assert_frame_equal(
        before.predictions.drop(columns="actual_topk"),
        after.predictions.drop(columns="actual_topk"),
    )
    assert len(before.metrics) == 4  # raw/draft, top 30/top 60
    assert set(before.metrics.train_through) == {seasons[-2]}


def test_learned_lag_ranker_uses_only_prior_seasons():
    settings = load_settings()
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    warmup = ["2018-19", "2019-20", "2020-21"]
    history = pd.DataFrame([
        {
            "season": season, "player_id": str(number), "games": 70,
            "age": 25.0, "mpg": 30.0, "usage": 20.0,
            "pts": float(90 - number),
        }
        for season in [*warmup, *seasons] for number in range(90)
    ])
    for stat in settings.scoring.model_dump():
        if stat not in history:
            history[stat] = 0.0
    backtest = BacktestResult(
        metrics=pd.DataFrame({
            "target_season": seasons,
            "model": ["factor_model"] * len(seasons),
            "replacement_fppg": [20.0] * len(seasons),
        }),
        predictions=pd.DataFrame([
            {"target_season": season, "player_id": str(number), "model": "factor_model"}
            for season in seasons for number in range(90)
        ]),
    )
    before = walk_forward_learned_lags(backtest, history, settings)
    changed = history.copy()
    changed.loc[changed.season == seasons[-1], "pts"] *= -1
    after = walk_forward_learned_lags(backtest, changed, settings)
    pd.testing.assert_frame_equal(before.predictions, after.predictions)
    assert set(before.metrics.train_through) == {seasons[-2]}


def test_game_form_uses_only_prior_regular_season_games(tmp_path):
    settings = load_settings()
    rows = []
    for number in range(1, 26):
        row = {stat: 0 for stat in EVENTS.values()}
        row.update({
            "season": 2025, "game_id": f"00224{number:05d}",
            "player_id": 123, "played": 1, "starter": int(number > 5),
            "minutes_played": "30:00", "assistedShooter": 0,
            "pts": 20 if number > 5 else 10,
        })
        rows.append(row)
    later = rows[-1].copy()
    later.update({"season": 2026, "game_id": "0022500001", "pts": 200})
    playoff = rows[-1].copy()
    playoff.update({"game_id": "0042400001", "pts": 200})
    path = tmp_path / "game_logs.parquet"
    pd.DataFrame([*rows, later, playoff]).to_parquet(path, index=False)
    form = build_game_form(path, settings, minimum_games=1).set_index(
        ["target_season", "player_id"]
    )
    prior = form.loc[("2025-26", "123")]
    assert prior.form_fppg_change == 2
    assert prior.form_starter_share == 1
    assert prior.form_last_game_gap == 0
