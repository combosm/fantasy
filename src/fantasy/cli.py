from __future__ import annotations

import argparse
import sys

import pandas as pd

from .audit import audit_rankings, load_external_rankings
from .config import load_settings
from .data import extend_history_from_archive, generate_fixture, ingest_draft_history, ingest_live
from .diagnostics import (
    oracle_backtest,
    ranking_diagnostics,
    summarize_extended_topk,
    summarize_ranking_diagnostics,
)
from .evaluation import load_roster_snapshots, summarize_topk_targets, walk_forward_full
from .export import export_workbook
from .game_form import FORM_COLUMNS, build_game_form
from .learned_lag import walk_forward_learned_lags
from .market import load_market_rankings, market_backtest
from .projection import project
from .ranker import summarize_probability_targets, walk_forward_topk_probabilities
from .rookie import rookie_walk_forward
from .sanity import build_sanity_review
from .uncertainty import interval_coverage, load_residuals
from .validation import validate

BOARD_COLUMNS = [
    "our_rank", "player_name", "team_id", "raw_rank", "draft_rank",
    "drafted_in_model", "expected_gp", "projected_mpg", "projected_usage",
    "projected_pace", "fppg", "expected_season_points", "draft_value",
    "replacement_fppg", "season_points_p20", "season_points_p80",
    "pts", "reb", "ast", "stl", "blk", "fg3m", "tov",
    "fgm", "fga", "fg_pct", "fg3_pct", "fta", "ftm", "ft_pct",
]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="fantasy", description="NBA fantasy projection pipeline")
    root.add_argument("--config", default="config/default.yml")
    sub = root.add_subparsers(dest="command", required=True)
    ingest = sub.add_parser("ingest", help="Download and normalize public NBA data")
    ingest.add_argument(
        "--fixture", action="store_true", help="Generate deterministic local fixture data"
    )
    sub.add_parser("project", help="Build projections and cached tabular outputs")
    sub.add_parser("validate", help="Run integrity checks and walk-forward backtests")
    sub.add_parser("backtest", help="Evaluate complete season forecasts against naive baselines")
    sub.add_parser("check-uncertainty", help="Check historical 20th–80th interval coverage")
    sub.add_parser("extend-history", help="Append cached public NBA season archives")
    sub.add_parser("ingest-drafts", help="Cache NBA draft slots for empirical rookie priors")
    sub.add_parser("export", help="Build projections and export the Excel workbook")
    all_cmd = sub.add_parser("all", help="Run ingestion, projections, validation and export")
    all_cmd.add_argument("--fixture", action="store_true", help="Use deterministic fixture data")
    return root


def _run_projection(settings):
    result = project(settings)
    rankings = load_external_rankings(settings.paths.inputs / "external_rankings.csv")
    all_audited, unmatched = audit_rankings(result.all_projections, rankings)
    audited = all_audited.head(settings.top_n).copy()
    history = pd.read_parquet(settings.paths.processed / "player_seasons.parquet")
    report = validate(audited, result.team_checks, history, settings)
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    audited.to_parquet(settings.paths.processed / "projections.parquet", index=False)
    all_audited.to_parquet(settings.paths.processed / "all_projections.parquet", index=False)
    result.draft_board.to_parquet(
        settings.paths.processed / "draft_board.parquet", index=False
    )
    settings.paths.output.mkdir(parents=True, exist_ok=True)
    audited[BOARD_COLUMNS].to_csv(
        settings.paths.output / f"raw_points_board_{settings.target_season}.csv", index=False
    )
    result.draft_board[BOARD_COLUMNS].to_csv(
        settings.paths.output / f"draft_value_board_{settings.target_season}.csv", index=False
    )
    result.team_checks.to_csv(settings.paths.processed / "team_checks.csv", index=False)
    result.rotation_detail.to_csv(settings.paths.processed / "team_rotations.csv", index=False)
    result.adjustments.to_csv(settings.paths.processed / "adjustments.csv", index=False)
    result.healthy_rotations.to_csv(settings.paths.processed / "healthy_rotations.csv", index=False)
    result.data_quality.to_csv(settings.paths.processed / "model_data_quality.csv", index=False)
    build_sanity_review(all_audited, history, settings).to_csv(
        settings.paths.processed / "sanity_review.csv", index=False
    )
    game_form_path = settings.paths.raw / "archive/espn_player_box.parquet"
    if game_form_path.exists():
        form = build_game_form(game_form_path, settings)
        current_form = form[form.target_season == settings.target_season]
        review = audited[[
            "player_id", "player_name", "team_id", "raw_rank", "draft_rank",
            "fppg", "expected_gp", "expected_season_points", "draft_value",
        ]].merge(current_form, on="player_id", how="left", validate="one_to_one")
        review["form_data_available"] = review[FORM_COLUMNS].notna().all(axis=1)
        review.to_csv(settings.paths.processed / "game_form_review.csv", index=False)
    report.integrity.to_csv(settings.paths.processed / "integrity.csv", index=False)
    report.backtest.to_csv(settings.paths.processed / "backtest.csv", index=False)
    return result, audited, rankings, unmatched, report


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    settings = load_settings(args.config)
    if args.command == "ingest":
        generate_fixture(settings) if args.fixture else ingest_live(settings)
        return 0
    if args.command == "extend-history":
        history = extend_history_from_archive(settings)
        print(f"{len(history)} player-seasons across {history.season.nunique()} seasons")
        return 0
    if args.command == "ingest-drafts":
        draft = ingest_draft_history(settings)
        print(f"{len(draft)} NBA draft picks across {draft.draft_year.nunique()} years")
        return 0
    if args.command == "check-uncertainty":
        coverage = interval_coverage(
            load_residuals(settings.paths.processed / "end_to_end_predictions.parquet")
        )
        coverage.to_csv(settings.paths.processed / "uncertainty_coverage.csv", index=False)
        print(coverage.round(3).to_string(index=False))
        return 0
    if args.command == "backtest":
        history = pd.read_parquet(settings.paths.processed / "player_seasons.parquet")
        snapshots = load_roster_snapshots(settings.paths.inputs / "preseason_rosters.csv")
        result = walk_forward_full(
            history, settings, snapshots, replacement_fppg=settings.replacement_fppg
        )
        settings.paths.processed.mkdir(parents=True, exist_ok=True)
        result.metrics.to_csv(settings.paths.processed / "end_to_end_backtest.csv", index=False)
        market_rows = load_market_rankings(settings.paths.inputs / "market_preseason_rankings.csv")
        market_metrics, market_matches = market_backtest(history, market_rows, settings)
        if not market_metrics.empty:
            market_metrics.to_csv(settings.paths.processed / "market_backtest.csv", index=False)
            market_matches.to_csv(settings.paths.processed / "market_match_audit.csv", index=False)
        draft_path = settings.paths.processed / "draft_history.parquet"
        if draft_path.exists():
            rookie_walk_forward(history, pd.read_parquet(draft_path), settings).to_csv(
                settings.paths.processed / "rookie_backtest.csv", index=False
            )
        oracles = oracle_backtest(result, history, settings)
        oracles.to_csv(settings.paths.processed / "oracle_backtest.csv", index=False)
        diagnostics = ranking_diagnostics(result, history, settings)
        diagnostics.to_csv(settings.paths.processed / "ranking_diagnostics.csv", index=False)
        summarize_ranking_diagnostics(diagnostics, settings).to_csv(
            settings.paths.processed / "ranking_diagnostics_summary.csv", index=False
        )
        extended = summarize_extended_topk(
            pd.concat([result.metrics, oracles], ignore_index=True), settings
        )
        extended.to_csv(settings.paths.processed / "extended_topk_summary.csv", index=False)
        targets = summarize_topk_targets(result.metrics, settings)
        probability = walk_forward_topk_probabilities(result, history, settings)
        trajectory = walk_forward_topk_probabilities(
            result, history, settings, approach="trajectory_boosted"
        )
        trajectory_linear = walk_forward_topk_probabilities(
            result, history, settings, approach="trajectory_logistic"
        )
        game_form_path = settings.paths.raw / "archive/espn_player_box.parquet"
        game_form = (
            walk_forward_topk_probabilities(
                result, history, settings, approach="game_form_logistic"
            ) if game_form_path.exists() else None
        )
        if game_form is not None:
            game_form.metrics.to_csv(
                settings.paths.processed / "game_form_backtest.csv", index=False
            )
            game_form.predictions.to_parquet(
                settings.paths.processed / "game_form_predictions.parquet", index=False
            )
        learned_lag = walk_forward_learned_lags(result, history, settings)
        learned_lag.metrics.to_csv(
            settings.paths.processed / "learned_lag_backtest.csv", index=False
        )
        learned_lag.predictions.to_parquet(
            settings.paths.processed / "learned_lag_predictions.parquet", index=False
        )
        ranker_metrics = pd.concat(
            [probability.metrics, trajectory.metrics, trajectory_linear.metrics],
            ignore_index=True,
        )
        ranker_predictions = pd.concat(
            [probability.predictions, trajectory.predictions, trajectory_linear.predictions],
            ignore_index=True,
        )
        ranker_metrics.to_csv(
            settings.paths.processed / "topk_probability_backtest.csv", index=False
        )
        ranker_predictions.to_parquet(
            settings.paths.processed / "topk_probability_predictions.parquet", index=False
        )
        if not ranker_metrics.empty:
            targets = pd.concat(
                [
                    targets,
                    summarize_probability_targets(ranker_metrics, settings),
                    summarize_probability_targets(learned_lag.metrics, settings),
                    *(
                        [summarize_probability_targets(game_form.metrics, settings)]
                        if game_form is not None else []
                    ),
                ],
                ignore_index=True,
            )
        targets.to_csv(settings.paths.processed / "topk_target_summary.csv", index=False)
        result.predictions.to_parquet(
            settings.paths.processed / "end_to_end_predictions.parquet", index=False
        )
        coverage = interval_coverage(
            load_residuals(settings.paths.processed / "end_to_end_predictions.parquet")
        )
        coverage.to_csv(settings.paths.processed / "uncertainty_coverage.csv", index=False)
        print(result.metrics.round(3).to_string(index=False))
        print("\nEight-normal-season overlap and oracle diagnostics:")
        print(extended.round(3).to_string(index=False))
        if not market_metrics.empty:
            print("\nDated ESPN preseason points-ranking benchmark (single source, not consensus):")
            print(market_metrics.round(3).to_string(index=False))
        print("\nTop-30 / top-60 target summary (last four seasons):")
        print(targets.round(3).to_string(index=False))
        if not coverage.empty:
            print("\nPrequential 20–80% interval coverage:")
            print(coverage.round(3).to_string(index=False))
        return 0
    if args.command == "all":
        generate_fixture(settings) if args.fixture else ingest_live(settings)
    if args.command in {"project", "validate", "export", "all"}:
        required = settings.paths.processed / "player_seasons.parquet"
        if not required.exists():
            raise FileNotFoundError("No processed player data. Run `fantasy ingest` first.")
        result, audited, rankings, unmatched, report = _run_projection(settings)
        if args.command in {"export", "all"}:
            path = export_workbook(
                settings.paths.output / "fantasy_draft_board.xlsx",
                audited,
                rankings,
                unmatched,
                result.adjustments,
                result.team_checks,
                result.rotation_detail,
                result.healthy_rotations,
                pd.concat(
                    [result.opportunity_backtest, result.availability_backtest, result.pace_backtest],
                    ignore_index=True,
                ),
                result.data_quality,
                report,
                settings,
            )
            print(path)
        if args.command == "validate":
            print(report.integrity.to_string(index=False))
        if args.command == "project":
            replacement = float(result.draft_board["replacement_fppg"].iloc[0])
            print(
                f"{settings.league.teams}-team draft value: "
                f"{settings.league.drafted_players} drafted spots; "
                f"waiver baseline {replacement:.2f} FPPG"
            )
            for label, board, value in [
                ("Raw season points", audited, "expected_season_points"),
                ("Draft value over replacement", result.draft_board, "draft_value"),
            ]:
                print(f"\n{label}")
                print(board[["our_rank", "player_name", value]].head(15).to_string(index=False))
        return 0 if report.passed else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
