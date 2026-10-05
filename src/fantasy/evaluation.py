"""Season-ahead evaluation with an explicit, dated roster information set."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from .availability import forecast_availability
from .config import Settings
from .opportunity import forecast_opportunity
from .pace import forecast_pace
from .projection import build_baseline, finalize_box_score

OPENING_DATES = {
    "2022-23": "2022-10-18",
    "2023-24": "2023-10-24",
    "2024-25": "2024-10-22",
    "2025-26": "2025-10-21",
}


@dataclass
class BacktestResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame


def summarize_topk_targets(metrics: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Compare recent walk-forward top-K overlap with explicit target bands."""
    recent = sorted(metrics["target_season"].unique())[-4:]
    subset = metrics[metrics["target_season"].isin(recent)]
    rows = []
    for model, group in subset.groupby("model"):
        for board, prefix in [("raw_points", ""), ("draft_value", "vor_")]:
            for k in (30, 60):
                rate_column = f"{prefix}top{k}_hit_rate"
                ceiling_column = f"{prefix}top{k}_pool_ceiling"
                opportunities = k * len(group)
                hits = round(float((group[rate_column] * k).sum()))
                lower = getattr(settings.topk_targets, f"top{k}_min")
                stretch = getattr(settings.topk_targets, f"top{k}_stretch")
                hit_rate = hits / opportunities
                rows.append({
                    "model": model,
                    "board": board,
                    "top_k": k,
                    "seasons": ",".join(recent),
                    "hit_count": hits,
                    "possible_hits": opportunities,
                    "hit_rate": hit_rate,
                    "candidate_pool_ceiling": float(group[ceiling_column].mean()),
                    "minimum_target": lower,
                    "stretch_target": stretch,
                    "additional_hits_to_minimum": max(0, ceil(lower * opportunities) - hits),
                    "target_status": (
                        "below_minimum" if hit_rate < lower
                        else "at_or_above_stretch" if hit_rate >= stretch
                        else "in_target_band"
                    ),
                })
    return pd.DataFrame(rows)


def _fantasy_per_game(frame: pd.DataFrame, settings: Settings) -> pd.Series:
    return sum(frame[stat] * weight for stat, weight in settings.scoring.model_dump().items())


def _roster_for_fold(
    prior: pd.DataFrame, target: str, snapshots: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    if not snapshots.empty and target in set(snapshots["season"]):
        if target not in OPENING_DATES:
            raise ValueError(f"No verified opening date configured for {target}")
        roster = snapshots[snapshots["season"] == target].copy()
        if roster["player_id"].duplicated().any():
            raise ValueError(f"Duplicate player in {target} preseason roster")
        dates = pd.to_datetime(roster["as_of"], errors="raise")
        if (dates >= pd.Timestamp(OPENING_DATES[target])).any():
            raise ValueError(f"{target} roster snapshot is not strictly pre-opening-night")
        return roster[["player_id", "team_id"]], "dated_preseason_snapshot"
    # This fallback is available at the cutoff and cannot leak target-season
    # outcomes. It does miss summer trades, signings, and incoming rookies.
    return (
        prior[["player_id", "team_id"]].drop_duplicates("player_id"),
        "previous_season_carryforward",
    )


def _three_year_baseline(history: pd.DataFrame, target_index: int, seasons: list[str], settings: Settings) -> pd.DataFrame:
    recent = seasons[max(0, target_index - 3):target_index]
    rows = history[history["season"].isin(recent)].copy()
    weights = {season: float(index + 1) for index, season in enumerate(recent)}
    rows["weight"] = rows["season"].map(weights)
    rows["prior_fppg"] = _fantasy_per_game(rows, settings)
    rows["weighted_fppg"] = rows["prior_fppg"] * rows["weight"]
    rows["weighted_games"] = rows["games"] * rows["weight"]
    grouped = rows.groupby("player_id", as_index=False)[
        ["weighted_fppg", "weighted_games", "weight"]
    ].sum()
    grouped["pred_fppg"] = grouped["weighted_fppg"] / grouped["weight"]
    grouped["pred_games"] = grouped["weighted_games"] / grouped["weight"]
    return grouped[["player_id", "pred_fppg", "pred_games"]]


def _model_forecast(
    history: pd.DataFrame, roster: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    base = build_baseline(history, roster, settings)
    pace = forecast_pace(history, roster, settings).predictions.set_index("team_id")
    base["projected_pace"] = base["team_id"].map(pace["projected_pace"]).fillna(
        base["base_pace"]
    )
    opportunity = forecast_opportunity(history, roster, settings).predictions.set_index("player_id")
    availability = forecast_availability(history, roster, settings).predictions.set_index("player_id")
    for output, source in [
        ("projected_mpg", "predicted_mpg"),
        ("projected_usage", "predicted_usage"),
    ]:
        base[output] = base["player_id"].map(opportunity[source]).fillna(base[output])
    base["expected_gp"] = base["player_id"].map(availability["predicted_games"]).fillna(
        base["expected_gp"]
    )
    projected = finalize_box_score(base, settings)
    return projected.rename(
        columns={"fppg": "pred_fppg", "expected_gp": "pred_games"}
    )[["player_id", "pred_fppg", "pred_games"]]


def _score_rows(
    rows: pd.DataFrame,
    actual_universe: pd.DataFrame,
    replacement_fppg: float,
) -> dict[str, float]:
    observed = rows["actual_games"] > 0
    pred_total = rows["pred_fppg"] * rows["pred_games"]
    actual_total = rows["actual_fppg"].fillna(0) * rows["actual_games"]
    count = len(rows)
    metrics = {
        "n": count,
        "fppg_mae": float((rows.loc[observed, "pred_fppg"] - rows.loc[observed, "actual_fppg"]).abs().mean()),
        "games_mae": float((rows["pred_games"] - rows["actual_games"]).abs().mean()),
        "season_total_mae": float((pred_total - actual_total).abs().mean()),
        "rank_spearman": float(spearmanr(pred_total, actual_total).statistic),
    }
    universe = actual_universe.copy()
    universe["raw_total"] = universe["actual_fppg"] * universe["actual_games"]
    universe["vor_total"] = (
        universe["actual_fppg"] - replacement_fppg
    ) * universe["actual_games"]
    for top in (12, 30, 50, 60):
        k = min(top, len(universe))
        predicted_ids = set(rows.loc[pred_total.nlargest(k).index, "player_id"])
        actual_ids = set(universe.nlargest(k, "raw_total")["player_id"])
        metrics[f"top{top}_hit_rate"] = len(predicted_ids & actual_ids) / k
        metrics[f"top{top}_pool_ceiling"] = len(set(rows["player_id"]) & actual_ids) / k
    pred_vor = (rows["pred_fppg"] - replacement_fppg) * rows["pred_games"]
    actual_vor = (rows["actual_fppg"].fillna(0) - replacement_fppg) * rows["actual_games"]
    metrics["vor_mae"] = float((pred_vor - actual_vor).abs().mean())
    metrics["vor_rank_spearman"] = float(spearmanr(pred_vor, actual_vor).statistic)
    for top in (12, 30, 50, 60):
        k = min(top, len(universe))
        predicted_ids = set(rows.loc[pred_vor.nlargest(k).index, "player_id"])
        actual_ids = set(universe.nlargest(k, "vor_total")["player_id"])
        metrics[f"vor_top{top}_hit_rate"] = len(predicted_ids & actual_ids) / k
        metrics[f"vor_top{top}_pool_ceiling"] = len(set(rows["player_id"]) & actual_ids) / k
    return metrics


def walk_forward_full(
    history: pd.DataFrame,
    settings: Settings,
    roster_snapshots: pd.DataFrame | None = None,
    replacement_fppg: float | None = None,
) -> BacktestResult:
    """Evaluate the whole factor-to-fantasy pipeline without target-season inputs.

    A provided roster snapshot must predate opening night. Without one, the
    previous season's roster is carried forward and explicitly labeled.
    """
    seasons = settings.historical_seasons
    snapshots = roster_snapshots if roster_snapshots is not None else pd.DataFrame()
    if not snapshots.empty:
        required = {"season", "player_id", "team_id", "as_of", "source"}
        missing = required - set(snapshots)
        if missing:
            raise ValueError(f"Roster snapshots missing columns: {sorted(missing)}")
    reports = []
    details = []
    for index in range(1, len(seasons)):
        target = seasons[index]
        past = history[history["season"].isin(seasons[:index])].copy()
        prior = history[history["season"] == seasons[index - 1]].copy()
        actual = history[history["season"] == target].copy()
        roster, roster_source = _roster_for_fold(prior, target, snapshots)
        local = settings.model_copy(deep=True)
        local.historical_seasons = seasons[:index]
        local.projection.season_weights = settings.projection.season_weights[:index]
        local.target_season = target
        last = prior[["player_id", "games"]].copy()
        last["pred_fppg"] = _fantasy_per_game(prior, settings)
        last = last.rename(columns={"games": "pred_games"})
        three = _three_year_baseline(past, index, seasons, settings)
        model = _model_forecast(past, roster, local)
        fold_replacement = replacement_fppg
        if fold_replacement is None:
            prior_rates = last["pred_fppg"].sort_values(ascending=False)
            index_at_waivers = min(settings.league.drafted_players, len(prior_rates) - 1)
            fold_replacement = float(prior_rates.iloc[index_at_waivers])
        labels = actual[["player_id", "games"]].rename(columns={"games": "actual_games"})
        labels["actual_fppg"] = _fantasy_per_game(actual, settings)
        for name, forecast in [
            ("previous_season", last),
            ("three_year_weighted", three),
            ("factor_model", model),
        ]:
            rows = roster[["player_id"]].merge(forecast, on="player_id", how="inner")
            rows = rows.merge(labels, on="player_id", how="left")
            rows["actual_games"] = rows["actual_games"].fillna(0)
            rows["target_season"] = target
            rows["model"] = name
            rows["roster_source"] = roster_source
            rows["pred_season_total"] = rows["pred_fppg"] * rows["pred_games"]
            rows["actual_season_total"] = rows["actual_fppg"].fillna(0) * rows[
                "actual_games"
            ]
            reports.append(
                {
                    "target_season": target,
                    "model": name,
                    "roster_source": roster_source,
                    "roster_size": len(roster),
                    "coverage": len(rows) / len(roster),
                    "replacement_fppg": fold_replacement,
                    **_score_rows(rows, labels, fold_replacement),
                }
            )
            details.append(rows)
    return BacktestResult(pd.DataFrame(reports), pd.concat(details, ignore_index=True))


def load_roster_snapshots(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"player_id": str, "team_id": str}).dropna(how="all")
