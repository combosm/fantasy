from __future__ import annotations

import pandas as pd
import pytest

from fantasy.config import load_settings
from fantasy.diagnostics import oracle_backtest, ranking_diagnostics, summarize_extended_topk
from fantasy.evaluation import BacktestResult
from fantasy.market import load_market_rankings
from fantasy.sanity import build_sanity_review


def test_oracle_and_regret_detect_a_misranked_star():
    settings = load_settings()
    players = [str(i) for i in range(60)]
    history = pd.DataFrame({
        "season": ["2025-26"] * 60, "player_id": players,
        "games": [70] * 60, "pts": [100 - i for i in range(60)],
    })
    for stat in settings.scoring.model_dump():
        if stat != "pts":
            history[stat] = 0.0
    predictions = pd.DataFrame({
        "target_season": ["2025-26"] * 60, "model": ["factor_model"] * 60,
        "roster_source": ["previous_season_carryforward"] * 60,
        "player_id": players, "pred_fppg": [1] + [100 - i for i in range(1, 60)],
        "pred_games": [70] * 60, "actual_fppg": [100 - i for i in range(60)],
        "actual_games": [70] * 60,
    })
    metrics = pd.DataFrame([{
        "target_season": "2025-26", "model": "factor_model", "replacement_fppg": 0,
    }])
    backtest = BacktestResult(metrics, predictions)
    oracle = oracle_backtest(backtest, history, settings)
    perfect = oracle[oracle.model == "actual_fppg_model_gp"].iloc[0]
    assert perfect.top30_hit_rate == 1.0
    diagnostics = ranking_diagnostics(backtest, history, settings)
    raw30 = diagnostics[(diagnostics.board == "raw_points") & (diagnostics.top_k == 30)].iloc[0]
    assert raw30.value_regret > 0
    assert raw30.ndcg < 1


def test_extended_summary_excludes_short_schedules_and_bootstraps_by_season():
    settings = load_settings()
    seasons = settings.historical_seasons[1:]
    metrics = pd.DataFrame({
        "target_season": seasons, "model": ["factor_model"] * len(seasons),
        "top30_hit_rate": [0.5] * len(seasons),
        "top60_hit_rate": [0.6] * len(seasons),
        "vor_top30_hit_rate": [0.7] * len(seasons),
        "vor_top60_hit_rate": [0.8] * len(seasons),
    })
    summary = summarize_extended_topk(metrics, settings, n_bootstrap=100)
    assert set(summary.n_seasons) == {8}
    assert all("2019-20" not in value and "2020-21" not in value for value in summary.seasons)
    assert summary.ci_low.le(summary.hit_rate).all()
    assert summary.ci_high.ge(summary.hit_rate).all()


def test_market_rankings_reject_post_cutoff_dates(tmp_path):
    path = tmp_path / "market.csv"
    pd.DataFrame([{
        "season": "2025-26", "player_name": "Test", "rank": 1,
        "source": "test", "published_at": "2025-10-21", "url": "https://example.com",
    }]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="strictly preseason"):
        load_market_rankings(path)


def test_sanity_review_flags_low_raw_rank_with_still_draftable_value():
    settings = load_settings()
    history = pd.DataFrame([{
        "season": "2025-26", "player_id": "1", "games": 70,
        "mpg": 34, "pts": 45,
        **{stat: 0 for stat in settings.scoring.model_dump() if stat != "pts"},
    }])
    board = pd.DataFrame([{
        "player_id": "1", "player_name": "Test Star", "team_id": "1",
        "raw_rank": 130, "draft_rank": 80, "fppg": 32,
        "expected_gp": 40, "projected_mpg": 25,
        "expected_season_points": 1280,
    }])
    review = build_sanity_review(board, history, settings).iloc[0]
    assert review.requires_review
    assert "prior top-60 FPPG" in review.review_reason
