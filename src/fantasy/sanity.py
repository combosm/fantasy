"""Explain low rankings of historically strong players without auto-overriding them."""

from __future__ import annotations

import pandas as pd

from .config import Settings
from .evaluation import _fantasy_per_game


def build_sanity_review(board: pd.DataFrame, history: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    recent = history[history.season.isin(settings.historical_seasons[-3:])].copy()
    recent["historical_fppg"] = _fantasy_per_game(recent, settings)
    latest = recent[recent.season == settings.historical_seasons[-1]].copy()
    latest["prior_fppg_rank"] = latest.historical_fppg.rank(ascending=False, method="min")
    latest = latest.rename(columns={
        "historical_fppg": "prior_fppg", "games": "prior_gp", "mpg": "prior_mpg"
    })[["player_id", "prior_fppg", "prior_fppg_rank", "prior_gp", "prior_mpg"]]
    three_year = recent.groupby("player_id", as_index=False).agg(
        three_year_fppg=("historical_fppg", "mean"),
        three_year_gp=("games", "mean"),
    )
    columns = [
        "player_id", "player_name", "team_id", "raw_rank", "draft_rank",
        "fppg", "expected_gp", "projected_mpg", "expected_season_points",
    ]
    for optional in ("external_median_rank", "audit_flag"):
        if optional in board:
            columns.append(optional)
    review = board[columns].merge(latest, on="player_id", how="left").merge(
        three_year, on="player_id", how="left"
    )
    injury_path = settings.paths.inputs / "injury_context.csv"
    documented = (
        set(pd.read_csv(injury_path, dtype={"player_id": str}).player_id.dropna())
        if injury_path.exists() else set()
    )
    review["documented_injury_context"] = review.player_id.astype(str).isin(documented)
    review["fppg_change_from_prior"] = review.fppg - review.prior_fppg
    review["mpg_change_from_prior"] = review.projected_mpg - review.prior_mpg
    review["gp_change_from_three_year"] = review.expected_gp - review.three_year_gp
    review["market_rank_gap"] = (
        review.raw_rank - review.external_median_rank
        if "external_median_rank" in review else float("nan")
    )
    reasons = []
    for _, row in review.iterrows():
        flags = []
        if row.raw_rank > 120 and row.prior_fppg_rank <= 60:
            flags.append("prior top-60 FPPG but raw rank below 120")
        if (
            "external_median_rank" in review
            and pd.notna(row.external_median_rank)
            and row.external_median_rank <= 100
            and row.market_rank_gap >= 40
        ):
            flags.append("raw rank at least 40 places below dated market top 100")
        if row.raw_rank > 120 and row.draft_rank <= settings.league.drafted_players:
            flags.append("draftable by replacement value but low raw total")
        if row.prior_mpg >= 20 and row.fppg < 0.8 * row.prior_fppg:
            flags.append("projected FPPG more than 20% below last season")
        if row.prior_mpg >= 25 and row.projected_mpg < 0.8 * row.prior_mpg:
            flags.append("projected MPG more than 20% below last season")
        if row.raw_rank > 120 and row.expected_gp < 50 and row.three_year_gp >= 55:
            flags.append("low GP despite 55+ three-year GP mean")
        reasons.append("; ".join(flags))
    review["review_reason"] = reasons
    review["requires_review"] = review.review_reason.ne("")
    review["market_top100_gap"] = review.market_rank_gap.where(
        review.external_median_rank.le(100), 0
    ) if "external_median_rank" in review else 0
    return review.sort_values(
        ["requires_review", "market_top100_gap", "raw_rank"],
        ascending=[False, False, False],
    )
