"""Dated preseason rankings as an external benchmark, never a model input."""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from .config import Settings
from .evaluation import _fantasy_per_game

OPENING_DATES = {
    "2022-23": "2022-10-18",
    "2023-24": "2023-10-24",
    "2024-25": "2024-10-22",
    "2025-26": "2025-10-21",
}
SOURCE_NAME_CORRECTIONS = {
    "benedictmathurin": "bennedictmathurin",
    "shaedonsharp": "shaedonsharpe",
    "paulobanchero": "paolobanchero",
    "nicolasclaxton": "nicclaxton",
}


def _name_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value)).casefold()
    words = "".join(ch if ch.isalnum() else " " for ch in normalized).split()
    while words and words[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        words.pop()
    return "".join(words)


def load_market_rankings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows = pd.read_csv(path, dtype={"season": str})
    if rows.empty:
        return rows
    required = {"season", "player_name", "rank", "source", "published_at", "url"}
    if required - set(rows):
        raise ValueError(f"Market rankings missing columns: {sorted(required - set(rows))}")
    for season, group in rows.groupby("season"):
        if season not in OPENING_DATES:
            raise ValueError(f"Opening date not verified for {season}")
        if (pd.to_datetime(group.published_at) >= pd.Timestamp(OPENING_DATES[season])).any():
            raise ValueError(f"{season} rankings are not strictly preseason")
        if group["rank"].duplicated().any():
            raise ValueError(f"{season} rankings repeat a rank")
    return rows


def market_backtest(
    history: pd.DataFrame, rankings: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score a points-rank baseline against both outcomes without score blending."""
    if rankings.empty:
        return pd.DataFrame(), pd.DataFrame()
    names = history[["player_id", "player_name"]].drop_duplicates().copy()
    names["name_key"] = names.player_name.map(_name_key)
    ambiguities = names.groupby("name_key").player_id.nunique()
    safe_names = names[~names.name_key.isin(ambiguities[ambiguities > 1].index)]
    lookup = safe_names.drop_duplicates("name_key").set_index("name_key").player_id
    matched = rankings.copy()
    matched["name_key"] = matched.player_name.map(_name_key).replace(SOURCE_NAME_CORRECTIONS)
    matched["player_id"] = matched.name_key.map(lookup)
    matched["match_status"] = matched.player_id.notna().map({True: "matched", False: "unmatched"})
    reports = []
    for season, group in matched.groupby("season"):
        actual = history[history.season == season].copy()
        if actual.empty:
            continue
        actual["actual_fppg"] = _fantasy_per_game(actual, settings)
        actual["raw_points"] = actual.actual_fppg * actual.games
        prior = history[history.season == settings.historical_seasons[
            settings.historical_seasons.index(season) - 1
        ]].copy()
        prior["prior_fppg"] = _fantasy_per_game(prior, settings)
        ordered = prior.prior_fppg.sort_values(ascending=False)
        replacement = float(ordered.iloc[min(settings.league.drafted_players, len(ordered) - 1)])
        actual["draft_value"] = (actual.actual_fppg - replacement) * actual.games
        eligible = group[group.player_id.notna()].sort_values("rank")
        report = {
            "target_season": season, "model": "espn_preseason_points_rank",
            "source": group.source.iloc[0], "published_at": group.published_at.iloc[0],
            "ranked_players": len(group), "matched_players": len(eligible),
            "match_rate": len(eligible) / len(group),
        }
        for board, prefix in (("raw_points", ""), ("draft_value", "vor_")):
            for k in (30, 60):
                actual_ids = set(actual.nlargest(k, board).player_id)
                picks = set(eligible.head(k).player_id)
                report[f"{prefix}top{k}_hit_rate"] = len(picks & actual_ids) / k
                report[f"{prefix}top{k}_pool_ceiling"] = len(set(eligible.player_id) & actual_ids) / k
                selected_value = actual.set_index("player_id")[board].reindex(list(picks)).fillna(0).sum()
                report[f"{prefix}top{k}_value_regret"] = float(
                    actual.nlargest(k, board)[board].sum() - selected_value
                )
        reports.append(report)
    return pd.DataFrame(reports), matched
