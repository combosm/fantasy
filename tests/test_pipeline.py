from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fantasy.audit import audit_rankings
from fantasy.availability import build_availability_features
from fantasy.config import load_settings
from fantasy.data import generate_fixture, normalize_archive_player_season
from fantasy.evaluation import _score_rows, summarize_topk_targets, walk_forward_full
from fantasy.opportunity import _profiles, build_features
from fantasy.projection import (
    COUNTING,
    assign_ranks_and_tiers,
    build_baseline,
    constrain_teams,
    finalize_box_score,
    project,
)
from fantasy.uncertainty import draw_joint_outcomes
from fantasy.validation import integrity_checks, walk_forward_backtest


def fixture_settings(tmp_path: Path):
    base = yaml.safe_load(Path("config/default.yml").read_text())
    base["simulation_count"] = 250
    base["top_n"] = 30
    base["paths"] = {
        "raw": str(tmp_path / "raw"),
        "processed": str(tmp_path / "processed"),
        "inputs": str(tmp_path / "inputs"),
        "output": str(tmp_path / "output"),
    }
    (tmp_path / "inputs").mkdir()
    for file, header in {
        "external_rankings.csv": "source,published_at,player_name,rank\n",
        "overrides.csv": "entity_type,entity_id,target,operation,value,reason,source,reviewed_at,confidence\n",
        "player_context.csv": "player_id,team_id,projected_mpg,projected_usage,role_security,creator_dependency,health_multiplier,shutdown_risk,notes\n",
        "team_context.csv": "team_id,projected_pace,coach_pace,rotation_size,creator_quality_change,three_rate_change,assist_rate_change,notes\n",
    }.items():
        (tmp_path / "inputs" / file).write_text(header)
    config = tmp_path / "settings.yml"
    config.write_text(yaml.safe_dump(base))
    return load_settings(config)


def test_end_to_end_projection_is_coherent(tmp_path):
    settings = fixture_settings(tmp_path)
    generate_fixture(settings)
    result = project(settings)
    projected = result.projections
    assert len(projected) == 30
    assert len(result.draft_board) == 30
    assert projected["our_rank"].tolist() == list(range(1, 31))
    assert result.draft_board["our_rank"].tolist() == list(range(1, 31))
    assert result.team_checks["minutes_ok"].all()
    assert result.team_checks["usage_ok"].all()
    assert projected["projected_usage"].between(0, 50).all()
    assert np.allclose(
        projected["projected_possessions"],
        projected["projected_mpg"] * projected["projected_pace"] / 48,
    )
    assert {"mpg", "usage"}.issubset(set(result.opportunity_backtest["metric"]))
    assert {"games"} == set(result.availability_backtest["metric"])
    assert {"pace"} == set(result.pace_backtest["metric"])
    assert np.allclose(
        projected["pts"], 2 * projected["fgm"] + projected["fg3m"] + projected["ftm"]
    )
    assert integrity_checks(projected, result.team_checks)["passed"].all()


def test_rankings_are_audit_only(tmp_path):
    settings = fixture_settings(tmp_path)
    generate_fixture(settings)
    result = project(settings)
    before = result.projections[["player_id", "expected_season_points"]].copy()
    names = result.projections["player_name"].head(3).tolist()
    rankings = pd.DataFrame(
        {
            "source": ["ESPN"] * 3,
            "published_at": ["2026-09-28"] * 3,
            "player_name": names,
            "rank": [250, 249, 248],
        }
    )
    rankings["name_key"] = rankings["player_name"].str.lower().str.replace(" ", "", regex=False)
    audited, _ = audit_rankings(result.projections, rankings)
    after = audited[["player_id", "expected_season_points"]]
    pd.testing.assert_frame_equal(before.reset_index(drop=True), after.reset_index(drop=True))
    assert audited["audit_flag"].any()


def test_context_override_changes_primitive_not_rank_directly(tmp_path):
    settings = fixture_settings(tmp_path)
    generate_fixture(settings)
    pd.DataFrame(
        [
            {
                "entity_type": "player",
                "entity_id": "10000",
                "target": "projected_mpg",
                "operation": "multiply",
                "value": 1.05,
                "reason": "confirmed larger role",
                "source": "test",
                "reviewed_at": "2026-09-28",
                "confidence": 0.8,
            }
        ]
    ).to_csv(settings.paths.inputs / "overrides.csv", index=False)
    result = project(settings)
    assert not result.adjustments.empty
    assert "expected_season_points" not in result.adjustments["target"].tolist()


def test_walk_forward_backtest_has_no_future_leakage_columns(tmp_path):
    settings = fixture_settings(tmp_path)
    history, _ = generate_fixture(settings)
    report = walk_forward_backtest(history, settings)
    assert not report.empty
    assert {"games", "mpg", "pts", "reb", "ast", "stl", "blk", "tov"}.issubset(
        set(report["metric"])
    )


def test_players_not_on_current_roster_are_excluded(tmp_path):
    settings = fixture_settings(tmp_path)
    history, _ = generate_fixture(settings)
    retired = history.iloc[[0]].copy()
    retired["player_id"] = "retired-player"
    retired["player_name"] = "Retired Player"
    history = pd.concat([history, retired], ignore_index=True)
    history.to_parquet(settings.paths.processed / "player_seasons.parquet", index=False)
    result = project(settings)
    assert "retired-player" not in set(result.projections["player_id"])


def test_new_high_usage_teammate_changes_roster_features(tmp_path):
    settings = fixture_settings(tmp_path)
    history, rosters = generate_fixture(settings)
    prior = _profiles(history, settings.historical_seasons[-1])
    original = build_features(prior, rosters).set_index("player_id").loc["10000"]
    moving_id = "20000"
    rosters.loc[rosters.player_id == moving_id, "team_id"] = "100"
    revised = build_features(prior, rosters).set_index("player_id").loc["10000"]
    assert revised.current_teammate_usage_minutes > original.current_teammate_usage_minutes
    assert revised.current_teammate_mpg_sum > original.current_teammate_mpg_sum


def test_games_override_does_not_change_minutes_or_per_game_stats(tmp_path):
    settings = fixture_settings(tmp_path)
    generate_fixture(settings)
    before = project(settings).projections.set_index("player_id").loc["10000"]
    pd.DataFrame(
        [{
            "entity_type": "player", "entity_id": "10000", "target": "expected_gp",
            "operation": "multiply", "value": 0.9, "reason": "test availability",
            "source": "test", "reviewed_at": "2026-09-28", "confidence": 0.8,
        }]
    ).to_csv(settings.paths.inputs / "overrides.csv", index=False)
    after = project(settings).projections.set_index("player_id").loc["10000"]
    assert after.expected_gp < before.expected_gp
    assert after.projected_mpg == before.projected_mpg
    assert after.projected_usage == before.projected_usage
    assert after.fppg == before.fppg
    assert after.expected_season_points < before.expected_season_points


def test_possessions_and_usage_drive_box_score_separately(tmp_path):
    settings = fixture_settings(tmp_path)
    history, rosters = generate_fixture(settings)
    baseline = build_baseline(history, rosters, settings).iloc[[0]].copy()
    baseline["projected_mpg"] = 30.0
    baseline["projected_pace"] = 100.0
    baseline["projected_usage"] = 25.0
    base = finalize_box_score(baseline, settings).iloc[0]
    faster = baseline.copy()
    faster["projected_pace"] = 105.0
    fast = finalize_box_score(faster, settings).iloc[0]
    assert np.isclose(fast.projected_possessions / base.projected_possessions, 1.05)
    assert np.isclose(fast.reb / base.reb, 1.05)
    assert np.isclose(fast.ast / base.ast, 1.05)
    higher_usage = baseline.copy()
    higher_usage["projected_usage"] = 30.0
    used = finalize_box_score(higher_usage, settings).iloc[0]
    assert np.isclose(used.fga / base.fga, 1.2)
    assert np.isclose(used.fta / base.fta, 1.2)
    assert used.pts > base.pts
    assert np.isclose(used.reb, base.reb)
    assert np.isclose(used.ast, base.ast)


def test_real_nba_box_scores_reconstruct_without_shot_volume_bias():
    """Observed usage/pace/minutes must recover observed events, not lose ~14%."""
    settings = load_settings()
    path = settings.paths.processed / "player_seasons.parquet"
    if not path.exists():
        pytest.skip("Cached NBA history is not available")
    actual = pd.read_parquet(path)
    actual = actual[(actual.games >= 20) & (actual.mpg > 0) & (actual.fga > 0)
                    & (actual.fg3a > 0) & (actual.fta > 0) & (actual.usage > 0)].copy()
    baseline = actual[["player_id", "team_id"]].copy()
    baseline["projected_mpg"] = actual["mpg"]
    baseline["projected_pace"] = actual["team_pace"]
    baseline["base_pace"] = actual["team_pace"]
    baseline["projected_usage"] = actual["usage"]
    baseline["baseline_usage"] = actual["usage"]
    baseline["expected_gp"] = actual["games"]
    for stat in COUNTING:
        baseline[f"{stat}_per_min"] = actual[stat] / actual["mpg"]
    for made, attempted in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        baseline[f"{made}_pct"] = actual[made] / actual[attempted]
    reconstructed = finalize_box_score(baseline, settings)
    assert np.abs(reconstructed["fga"] - actual["fga"]).mean() < 1e-8
    assert np.abs(reconstructed["fta"] - actual["fta"]).mean() < 1e-8
    assert np.abs(reconstructed["tov"] - actual["tov"]).mean() < 1e-8
    assert abs((reconstructed["pts"] - actual["pts"]).mean()) < 0.05


def test_end_to_end_backtest_uses_no_target_season_stats(tmp_path):
    settings = fixture_settings(tmp_path)
    history, _ = generate_fixture(settings)
    before = walk_forward_full(history, settings)
    assert set(before.metrics["model"]) == {
        "previous_season", "three_year_weighted", "factor_model"
    }
    assert set(before.metrics["roster_source"]) == {"previous_season_carryforward"}
    assert before.metrics["vor_mae"].notna().all()
    altered = history.copy()
    target = settings.historical_seasons[-1]
    altered.loc[altered.season == target, "pts"] += 100
    altered.loc[altered.season == target, "team_id"] = "future-only-team"
    after = walk_forward_full(altered, settings)
    key = (before.predictions.target_season == target) & (
        before.predictions.model == "factor_model"
    )
    other = (after.predictions.target_season == target) & (
        after.predictions.model == "factor_model"
    )
    pd.testing.assert_series_equal(
        before.predictions.loc[key, "pred_season_total"].reset_index(drop=True),
        after.predictions.loc[other, "pred_season_total"].reset_index(drop=True),
    )


def test_topk_hit_rates_count_unprojected_actual_players():
    actual = pd.DataFrame({
        "player_id": [str(i) for i in range(60)],
        "actual_games": [70] * 60,
        "actual_fppg": [100 - i for i in range(60)],
    })
    predicted = actual[actual.player_id != "0"].copy().reset_index(drop=True)
    predicted["pred_games"] = predicted["actual_games"]
    predicted["pred_fppg"] = predicted["actual_fppg"]
    score = _score_rows(predicted, actual, replacement_fppg=0)
    assert score["top30_hit_rate"] == 29 / 30
    assert score["top30_pool_ceiling"] == 29 / 30
    assert score["vor_top30_hit_rate"] == 29 / 30
    assert score["top60_hit_rate"] == 59 / 60


def test_topk_target_summary_tracks_separate_bands_and_boards():
    settings = load_settings()
    metrics = pd.DataFrame({
        "target_season": ["2024-25", "2025-26"],
        "model": ["factor_model", "factor_model"],
        "top30_hit_rate": [24 / 30, 25 / 30],
        "top60_hit_rate": [42 / 60, 44 / 60],
        "vor_top30_hit_rate": [26 / 30, 26 / 30],
        "vor_top60_hit_rate": [43 / 60, 43 / 60],
        "top30_pool_ceiling": [1, 1],
        "top60_pool_ceiling": [1, 1],
        "vor_top30_pool_ceiling": [1, 1],
        "vor_top60_pool_ceiling": [1, 1],
    })
    summary = summarize_topk_targets(metrics, settings).set_index(["board", "top_k"])
    assert summary.loc[("raw_points", 30), "hit_count"] == 49
    assert summary.loc[("raw_points", 30), "target_status"] == "in_target_band"
    assert summary.loc[("raw_points", 60), "target_status"] == "in_target_band"
    assert summary.loc[("draft_value", 30), "target_status"] == "at_or_above_stretch"
    assert summary.loc[("draft_value", 60), "target_status"] == "in_target_band"


def test_documented_injury_context_changes_games_only(tmp_path):
    settings = fixture_settings(tmp_path)
    generate_fixture(settings)
    before = project(settings).projections.set_index("player_id").loc["10000"]
    pd.DataFrame([{
        "player_id": "10000", "expected_gp": 60,
        "status": "known return", "reviewed_at": "2026-09-28",
        "source": "https://example.com/injury-report",
        "notes": "Explicit scenario reviewed for test",
    }]).to_csv(settings.paths.inputs / "injury_context.csv", index=False)
    after = project(settings).projections.set_index("player_id").loc["10000"]
    assert after.expected_gp == 60
    assert after.projected_mpg == before.projected_mpg
    assert after.fppg == before.fppg


def test_replacement_rank_values_healthy_games_only(tmp_path):
    settings = fixture_settings(tmp_path)
    settings.replacement_fppg = 20
    candidates = pd.DataFrame([
        {"player_id": "a", "fppg": 50, "expected_gp": 40, "expected_season_points": 2000},
        {"player_id": "b", "fppg": 30, "expected_gp": 75, "expected_season_points": 2250},
    ])
    raw = assign_ranks_and_tiers(candidates, settings, "raw")
    draft = assign_ranks_and_tiers(candidates, settings, "draft")
    assert raw["player_id"].tolist() == ["b", "a"]
    assert draft["player_id"].tolist() == ["a", "b"]
    assert draft["draft_value"].tolist() == [1200, 750]
    assert set(draft["draft_value_metric"]) == {"points_above_waiver_replacement"}


def test_eight_team_default_uses_105th_fppg_as_waiver_line(tmp_path):
    settings = fixture_settings(tmp_path)
    settings.top_n = 110
    players = pd.DataFrame({
        "player_id": [str(i) for i in range(110)],
        "projected_mpg": [25.0] * 110,
        "expected_gp": [70.0] * 110,
        "fppg": [110.0 - i for i in range(110)],
        "expected_season_points": [(110.0 - i) * 70 for i in range(110)],
    })
    ranked = assign_ranks_and_tiers(players, settings, "draft")
    assert ranked.replacement_fppg.iloc[0] == 6.0
    assert ranked.drafted_players.iloc[0] == 104
    assert ranked.loc[ranked.draft_rank == 105, "draft_value"].iloc[0] == 0


def test_archive_totals_become_exact_per_game_rates():
    frame = pd.DataFrame([{
        "player_id": 1, "player_name": "Test", "team_id": 100, "age": 25,
        "gp": 50, "min": 1500, "usg_pct": 0.25, "pace": 100,
        "pts": 1000, "reb": 300, "ast": 200, "stl": 50, "blk": 25,
        "tov": 100, "fgm": 350, "fga": 700, "fg3m": 100,
        "fg3a": 250, "ftm": 200, "fta": 250,
    }])
    converted = normalize_archive_player_season(frame, "2015-16").iloc[0]
    assert converted.games == 50
    assert converted.mpg == 30
    assert converted.usage == 25
    assert converted.pts == 20
    assert converted.fga == 14


def test_empirical_uncertainty_keeps_games_and_fppg_errors_paired():
    residuals = pd.DataFrame({
        "pred_fppg": [20] * 200,
        "fppg_error": [-5, 5] * 100,
        "games_error": [-20, 20] * 100,
    })
    fppg, games = draw_joint_outcomes(20, 50, residuals, 1000, np.random.default_rng(1))
    assert set(zip(fppg, games)) == {(15, 30), (25, 70)}


def test_team_capacity_reconciliation_preserves_games_and_usage():
    team = pd.DataFrame([
        {"team_id": "1", "player_id": str(i), "player_name": str(i),
         "projected_mpg": 30.0, "expected_gp": 82, "projected_usage": 20.0}
        for i in range(9)
    ])
    reconciled, checks, changes = constrain_teams(team)
    assert np.isclose(reconciled.projected_mpg.sum(), 240)
    assert np.isclose(reconciled.expected_gp.sum(), team.expected_gp.sum())
    assert reconciled.projected_usage.equals(team.projected_usage)
    assert checks.minutes_ok.all()
    assert len(changes) == len(team)


def test_shortened_schedule_does_not_look_like_injury():
    history = pd.DataFrame([
        {"season": "2019-20", "player_id": "a", "age": 25,
         "games": 68, "mpg": 30, "usage": 20},
        {"season": "2020-21", "player_id": "a", "age": 26,
         "games": 72, "mpg": 30, "usage": 20},
    ])
    roster = pd.DataFrame([{"player_id": "a"}])
    features = build_availability_features(
        history, ["2019-20", "2020-21"], 1, roster
    ).iloc[0]
    assert features.prior_games == 82
    assert np.isnan(features.prior2_games)
