"""Pre-draft late-season form from regular-season player game logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import Settings

RECENT_GAMES = 20  # Exploratory window fixed before fold evaluation, not tuned on outcomes.
EVENTS = {
    "pts": "pts",
    "fg3m": "fg3mplyr",
    "fgm": "fgmplyr",
    "fga": "fgaplyr",
    "ftm": "ftmplyr",
    "fta": "ftaplyr",
    "reb": "rebounder",
    "ast": "assister",
    "stl": "stlr",
    "blk": "blockplyr",
    "tov": "tov1",
}
FORM_COLUMNS = [
    "form_mpg_change", "form_fppg_change", "form_fppm_change",
    "form_fga_change", "form_ast_change", "form_reb_change",
    "form_starter_share", "form_last_game_gap", "form_assisted_share",
]


def build_game_form(
    path: Path, settings: Settings, *, minimum_games: int = 900
) -> pd.DataFrame:
    """Use the final 20 played games of each *previous* regular season.

    Only seasons with at least 900 distinct regular-season games enter the
    training/evaluation set. This excludes the archive's partial older logs.
    """
    needed = [
        "season", "game_id", "player_id", "played", "starter", "minutes_played",
        "assistedShooter", *set(EVENTS.values()),
    ]
    box = pd.read_parquet(path, columns=needed)
    box = box[box.game_id.astype(str).str.startswith("002") & (box.played == 1)].copy()
    complete = box.groupby("season").game_id.nunique()
    box = box[box.season.isin(complete[complete >= minimum_games].index)].copy()
    clock = box.minutes_played.str.split(":", n=1, expand=True)
    box["minutes"] = pd.to_numeric(clock[0], errors="coerce") + (
        pd.to_numeric(clock[1], errors="coerce") / 60
    )
    box = box[box.minutes > 0].copy()
    box["player_id"] = box.player_id.astype(str)
    box["game_id"] = box.game_id.astype(str)
    box["game_order"] = box.game_id.str[-4:].astype(int)
    box["fppg"] = sum(
        pd.to_numeric(box[source], errors="coerce").fillna(0) * weight
        for stat, source in EVENTS.items()
        if (weight := getattr(settings.scoring, stat)) != 0
    )
    box["fppm"] = box.fppg / box.minutes
    box["assisted_share"] = box.assistedShooter / box.fgmplyr.clip(lower=1)
    box = box.sort_values(["season", "player_id", "game_order"])
    keys = ["season", "player_id"]
    measures = {
        "minutes": "mpg", "fppg": "fppg", "fppm": "fppm",
        "fgaplyr": "fga", "assister": "ast", "rebounder": "reb",
    }
    full = box.groupby(keys).agg({name: "mean" for name in measures}).rename(
        columns={name: f"full_{alias}" for name, alias in measures.items()}
    )
    recent_rows = box.groupby(keys, group_keys=False).tail(RECENT_GAMES)
    recent = recent_rows.groupby(keys).agg({
        **{name: "mean" for name in measures},
        "starter": "mean", "assisted_share": "mean", "game_order": "max",
    }).rename(columns={
        **{name: f"recent_{alias}" for name, alias in measures.items()},
        "starter": "form_starter_share",
        "assisted_share": "form_assisted_share",
        "game_order": "last_game_order",
    })
    result = full.join(recent).reset_index()
    for alias in measures.values():
        result[f"form_{alias}_change"] = result[f"recent_{alias}"] - result[f"full_{alias}"]
    result["form_last_game_gap"] = result.season.map(
        box.groupby("season").game_order.max()
    ) - result.last_game_order
    result["target_season"] = result.season.map(
        lambda year: f"{int(year)}-{(int(year) + 1) % 100:02d}"
    )
    return result[["target_season", "player_id", *FORM_COLUMNS]]
