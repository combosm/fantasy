from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .availability import forecast_availability
from .config import Settings
from .opportunity import forecast_opportunity
from .pace import forecast_pace
from .rookie import forecast_drafted_rookies
from .uncertainty import draw_joint_outcomes, load_residuals

COUNTING = ["reb", "ast", "stl", "blk", "tov", "fga", "fg3a", "fta"]
RATE_UNCERTAINTY = {
    "reb": 0.08,
    "ast": 0.10,
    "stl": 0.13,
    "blk": 0.15,
    "tov": 0.10,
    "fga": 0.07,
    "fg3a": 0.09,
    "fta": 0.11,
}


@dataclass
class ProjectionResult:
    projections: pd.DataFrame
    draft_board: pd.DataFrame
    team_checks: pd.DataFrame
    rotation_detail: pd.DataFrame
    adjustments: pd.DataFrame
    simulations: pd.DataFrame
    data_quality: pd.DataFrame
    healthy_rotations: pd.DataFrame
    opportunity_backtest: pd.DataFrame
    availability_backtest: pd.DataFrame
    pace_backtest: pd.DataFrame
    all_projections: pd.DataFrame


def _read_optional(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path, dtype={"player_id": str, "team_id": str, "entity_id": str})
    return df.dropna(how="all")


def _weighted(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any() or weights[valid].sum() == 0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def append_rookie_translations(
    history: pd.DataFrame, rosters: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Translate source-league per-game lines for current players with no NBA history.

    The deliberately conservative factors are transparent defaults. They can be replaced by
    walk-forward learned factors once multiple historical draft classes are supplied.
    """
    path = settings.paths.inputs / "rookie_source_stats.csv"
    known = set(history["player_id"].astype(str))
    current = set(rosters["player_id"].astype(str)) if not rosters.empty else known
    quality: list[dict[str, object]] = []
    rookie_frames: list[pd.DataFrame] = []
    automatic_path = settings.paths.processed / "rookie_source_stats.parquet"
    if automatic_path.exists():
        rookie_frames.append(pd.read_parquet(automatic_path))
    if path.exists() and path.stat().st_size > 0:
        manual = pd.read_csv(path, dtype={"player_id": str, "team_id": str}).dropna(how="all")
        if not manual.empty:
            manual["source_priority"] = 1
            rookie_frames.append(manual)
    rookies = pd.concat(rookie_frames, ignore_index=True) if rookie_frames else pd.DataFrame()
    if not rookies.empty:
        rookies["player_id"] = rookies["player_id"].astype(str)
        if "source_priority" not in rookies:
            rookies["source_priority"] = 0
        else:
            rookies["source_priority"] = rookies["source_priority"].fillna(0)
        rookies = rookies.sort_values("source_priority").drop_duplicates("player_id", keep="last")
    translated: list[dict[str, object]] = []
    factors = {
        "reb": 0.84,
        "ast": 0.74,
        "stl": 0.68,
        "blk": 0.70,
        "tov": 0.90,
        "fga": 0.80,
        "fg3a": 0.88,
        "fta": 0.72,
    }
    for _, source in rookies.iterrows():
        player_id = str(source["player_id"])
        if player_id in known or player_id not in current:
            continue
        source_mpg = max(float(source["minutes"]), 1.0)
        draft_pick = float(source.get("draft_pick", 60) or 60)
        nba_mpg = float(np.clip(30 - 0.34 * (draft_pick - 1), 8, 30))
        rates = {
            stat: float(source[stat]) / source_mpg * factor * nba_mpg
            for stat, factor in factors.items()
        }
        fg_pct = (float(source["fgm"]) / float(source["fga"])) if float(source["fga"]) else 0.45
        fg3_pct = float(source["fg3m"]) / float(source["fg3a"]) if float(source["fg3a"]) else 0.34
        ft_pct = (float(source["ftm"]) / float(source["fta"])) if float(source["fta"]) else 0.76
        rates["fg3a"] = min(rates["fg3a"], rates["fga"])
        fg3m = rates["fg3a"] * (0.6 * fg3_pct + 0.4 * 0.355)
        fgm = max(fg3m, rates["fga"] * (0.6 * fg_pct + 0.4 * 0.465))
        ftm = rates["fta"] * (0.7 * ft_pct + 0.3 * 0.78)
        translated.append(
            {
                "season": settings.historical_seasons[-1],
                "player_id": player_id,
                "player_name": source["player_name"],
                "team_id": str(source["team_id"]),
                "age": float(source["age"]) - 1,
                "games": min(int(source["games"]), 72),
                "mpg": nba_mpg,
                "team_pace": 100.0,
                "usage": float(np.clip(16 + rates["fga"] * 0.45, 14, 28)),
                "pts": 2 * fgm + fg3m + ftm,
                "reb": rates["reb"],
                "ast": rates["ast"],
                "stl": rates["stl"],
                "blk": rates["blk"],
                "tov": rates["tov"],
                "fgm": fgm,
                "fga": rates["fga"],
                "fg3m": fg3m,
                "fg3a": rates["fg3a"],
                "ftm": ftm,
                "fta": rates["fta"],
                "rookie_prior": True,
            }
        )
        quality.append(
            {
                "category": "rookie translation",
                "entity_id": player_id,
                "passed": True,
                "detail": f"Translated {source.get('source_league', 'source')} stats with widened uncertainty",
            }
        )
    supplied = set(rookies["player_id"].astype(str)) if not rookies.empty else set()
    prior_ids: set[str] = set()
    draft_path = settings.paths.processed / "draft_history.parquet"
    if draft_path.exists() and not rosters.empty:
        draft = pd.read_parquet(draft_path)
        eligible = rosters[
            ~rosters.player_id.astype(str).isin(known | supplied)
        ].copy()
        prior_rows, prior_quality = forecast_drafted_rookies(history, draft, eligible, settings)
        if not prior_rows.empty:
            translated.extend(prior_rows.to_dict("records"))
            quality.extend(prior_quality.to_dict("records"))
            prior_ids = set(prior_rows.player_id.astype(str))
    for player_id in sorted(current - known - supplied - prior_ids):
        player = rosters.loc[rosters["player_id"].astype(str) == player_id].iloc[0]
        quality.append(
            {
                "category": "missing projection",
                "entity_id": player_id,
                "passed": False,
                "detail": f"{player['player_name']} has no NBA history or rookie source statistics",
            }
        )
    if translated:
        history = pd.concat([history, pd.DataFrame(translated)], ignore_index=True)
    return history, pd.DataFrame(quality)


def build_baseline(
    history: pd.DataFrame, rosters: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    df = history.copy()
    df["player_id"] = df["player_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    season_weight = dict(zip(settings.historical_seasons, settings.projection.season_weights))
    df["season_weight"] = df["season"].map(season_weight).fillna(0.0)
    df["total_minutes"] = df["games"] * df["mpg"]
    df["model_weight"] = df["season_weight"] * np.sqrt(df["total_minutes"].clip(lower=1))

    latest = (
        df.sort_values("season").groupby("player_id", as_index=False).tail(1).set_index("player_id")
    )
    roster_map = rosters.set_index("player_id") if not rosters.empty else pd.DataFrame()
    league_per_minute = {
        stat: (df[stat] * df["games"]).sum() / df["total_minutes"].sum() for stat in COUNTING
    }
    rows: list[dict[str, float | str]] = []
    for player_id, group in df.groupby("player_id"):
        last = latest.loc[player_id]
        target_age = float(last["age"] + 1)
        team_id = (
            str(roster_map.loc[player_id, "team_id"])
            if player_id in roster_map.index
            else str(last["team_id"])
        )
        name = str(last["player_name"])
        total_hist_minutes = float(group["total_minutes"].sum())
        recency = group["season_weight"] * group["games"].clip(lower=1)
        mpg = _weighted(group["mpg"], recency)
        availability = _weighted(group["games"] / 82, group["season_weight"])
        age_gp_penalty = max(target_age - 30, 0) * 0.006
        expected_gp = np.clip(82 * (availability - age_gp_penalty), 25, 82)
        base_pace = _weighted(group["team_pace"], group["model_weight"])
        usage = _weighted(group["usage"], group["model_weight"])
        row: dict[str, float | str] = {
            "player_id": player_id,
            "player_name": name,
            "team_id": team_id,
            "age": target_age,
            "expected_gp": expected_gp,
            "projected_mpg": np.clip(mpg, 4, 38),
            "base_pace": base_pace,
            "projected_pace": base_pace,
            "projected_usage": usage,
            "baseline_usage": usage,
            "historical_minutes": total_hist_minutes,
            "is_rookie_prior": bool(group.get("rookie_prior", pd.Series(False, index=group.index)).fillna(False).any()),
        }
        for stat in COUNTING:
            rates = group[stat] / group["mpg"].replace(0, np.nan)
            own = _weighted(rates, group["model_weight"])
            shrink = settings.projection.shrinkage_minutes
            shrunk = (own * total_hist_minutes + league_per_minute[stat] * shrink) / (
                total_hist_minutes + shrink
            )
            row[f"{stat}_per_min"] = shrunk
        for made, attempted in [("fgm", "fga"), ("fg3m", "fg3a"), ("ftm", "fta")]:
            makes = float((group[made] * group["games"] * group["season_weight"]).sum())
            attempts = float((group[attempted] * group["games"] * group["season_weight"]).sum())
            league_pct = float((df[made] * df["games"]).sum() / (df[attempted] * df["games"]).sum())
            prior = settings.projection.shrinkage_shots
            row[f"{made}_pct"] = (makes + prior * league_pct) / (attempts + prior)
        rows.append(row)
    return pd.DataFrame(rows)


def apply_current_team_pace(
    base: pd.DataFrame, history: pd.DataFrame, rosters: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply an independently validated team-level pace forecast."""
    out = base.copy()
    result = forecast_pace(history, rosters, settings)
    team_pace = result.predictions.set_index("team_id")["projected_pace"]
    out["projected_pace"] = out["team_id"].map(team_pace).fillna(out["base_pace"])
    return out, result.backtest


def apply_opportunity_forecasts(
    base: pd.DataFrame, full_history: pd.DataFrame, rosters: pd.DataFrame, settings: Settings,
    rookie_profiles: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    result = forecast_opportunity(
        full_history, rosters, settings, current_prior_extra=rookie_profiles
    )
    out = base.copy()
    matches = result.predictions.set_index("player_id")
    changes: list[dict[str, object]] = []
    for idx, player in out.iterrows():
        pid = str(player["player_id"])
        if bool(player.get("is_rookie_prior", False)):
            continue
        if pid not in matches.index:
            continue
        for field, predicted in [
            ("projected_mpg", "predicted_mpg"),
            ("projected_usage", "predicted_usage"),
        ]:
            old = float(out.at[idx, field])
            new = float(matches.at[pid, predicted])
            out.at[idx, field] = new
            metric = {"projected_mpg": "mpg", "projected_usage": "usage"}[field]
            selected_key = (
                "mpg_stable_starter"
                if metric == "mpg" and bool(matches.at[pid, "stable_starter_role"])
                else "mpg_established_starter"
                if metric == "mpg" and bool(matches.at[pid, "established_starter_role"])
                else metric + (
                    "_disrupted" if bool(matches.at[pid, "disrupted_role"]) else ""
                )
            )
            changes.append(
                _change(
                    player,
                    field,
                    old,
                    new,
                    f"{result.selected_models[selected_key]} model selected by past-season validation",
                )
            )
    return out, result.backtest, changes


def apply_availability_forecasts(
    base: pd.DataFrame, full_history: pd.DataFrame, rosters: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    result = forecast_availability(full_history, rosters, settings)
    out = base.copy()
    matches = result.predictions.set_index("player_id")["predicted_games"]
    changes = []
    for idx, player in out.iterrows():
        pid = str(player["player_id"])
        if pid not in matches.index or pd.isna(matches.at[pid]):
            continue
        if bool(player.get("is_rookie_prior", False)):
            continue
        old = float(out.at[idx, "expected_gp"])
        new = float(matches.at[pid])
        out.at[idx, "expected_gp"] = new
        changes.append(
            _change(
                player, "expected_gp", old, new,
                f"{result.selected_model} availability model; selected by past-season MSE",
            )
        )
    backtest = result.backtest.assign(metric="games")
    return out, backtest, changes


def apply_injury_context(
    base: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply only documented, dated human judgments about known injuries."""
    out = base.copy()
    context = _read_optional(settings.paths.inputs / "injury_context.csv", ["player_id"])
    if context.empty:
        return out, []
    required = {"player_id", "expected_gp", "status", "reviewed_at", "source", "notes"}
    missing = required - set(context)
    if missing:
        raise ValueError(f"Injury context missing columns: {sorted(missing)}")
    if context["player_id"].duplicated().any():
        raise ValueError("Injury context has duplicate player IDs")
    changes = []
    for _, item in context.iterrows():
        pid = str(item["player_id"])
        matches = out.index[out["player_id"].astype(str) == pid]
        if len(matches) != 1:
            raise ValueError(f"Injury-context player {pid} has no unique projection")
        if any(pd.isna(item[field]) or not str(item[field]).strip() for field in
               ["status", "reviewed_at", "source", "notes"]):
            raise ValueError(f"Injury context for {pid} needs status, date, source and notes")
        reviewed = pd.to_datetime(item["reviewed_at"], errors="raise")
        if reviewed > pd.Timestamp.now():
            raise ValueError(f"Injury context for {pid} has a future review date")
        gp = float(item["expected_gp"])
        if not 0 <= gp <= 82:
            raise ValueError(f"Injury context for {pid} has invalid expected GP")
        idx = matches[0]
        old = float(out.at[idx, "expected_gp"])
        out.at[idx, "expected_gp"] = gp
        change = _change(
            out.loc[idx], "expected_gp", old, gp,
            f"injury review: {item['status']}; {item['notes']}",
        )
        change.update({"source": item["source"], "reviewed_at": item["reviewed_at"]})
        changes.append(change)
    return out, changes


def apply_team_context(
    base: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = base.copy()
    changes: list[dict[str, object]] = []
    context = _read_optional(settings.paths.inputs / "team_context.csv", ["team_id"])
    if context.empty:
        return out, changes
    context = context.set_index("team_id")
    for idx, row in out.iterrows():
        team_id = str(row["team_id"])
        if team_id not in context.index:
            continue
        ctx = context.loc[team_id]
        if pd.notna(ctx.get("projected_pace")):
            old = out.at[idx, "projected_pace"]
            out.at[idx, "projected_pace"] = float(ctx["projected_pace"])
            changes.append(
                _change(row, "projected_pace", old, out.at[idx, "projected_pace"], "team context")
            )
    return out, changes


def apply_player_context(
    base: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = base.copy()
    changes: list[dict[str, object]] = []
    context = _read_optional(settings.paths.inputs / "player_context.csv", ["player_id"])
    if context.empty:
        out["role_security"] = 0.8
        return out, changes
    context = context.set_index("player_id")
    for idx, row in out.iterrows():
        pid = str(row["player_id"])
        if pid not in context.index:
            continue
        ctx = context.loc[pid]
        for field in ["projected_mpg", "projected_usage"]:
            if pd.notna(ctx.get(field)):
                old = out.at[idx, field]
                out.at[idx, field] = float(ctx[field])
                changes.append(
                    _change(
                        row, field, old, out.at[idx, field], str(ctx.get("notes", "player context"))
                    )
                )
        for field, default in [
            ("role_security", 0.8),
        ]:
            out.at[idx, field] = float(
                ctx.get(field, default) if pd.notna(ctx.get(field, np.nan)) else default
            )
        if pd.notna(ctx.get("health_multiplier")):
            old = out.at[idx, "expected_gp"]
            out.at[idx, "expected_gp"] = np.clip(old * float(ctx["health_multiplier"]), 0, 82)
            changes.append(
                _change(
                    row,
                    "expected_gp",
                    old,
                    out.at[idx, "expected_gp"],
                    str(ctx.get("notes", "health")),
                )
            )
    for field, default in [
        ("role_security", 0.8),
    ]:
        if field not in out:
            out[field] = default
        else:
            out[field] = out[field].fillna(default)
    return out, changes


def apply_healthy_rotations(
    base: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    path = settings.paths.inputs / "healthy_rotations.csv"
    out = base.copy()
    if not path.exists() or path.stat().st_size == 0:
        return out, pd.DataFrame(), []
    rotation = pd.read_csv(path, dtype={"team_id": str, "player_id": str})
    if rotation.empty:
        return out, rotation, []
    if rotation.duplicated(["team_id", "player_id"]).any():
        raise ValueError("Healthy rotation has duplicate team/player rows")
    rotation["healthy_mpg"] = pd.to_numeric(rotation["healthy_mpg"], errors="raise")
    if not rotation["healthy_mpg"].between(0, 38).all():
        raise ValueError("Healthy rotation player minutes must be between 0 and 38")
    for team_id, group in rotation.groupby("team_id"):
        if not np.isclose(group["healthy_mpg"].sum(), 240.0, atol=1e-6):
            raise ValueError(f"Healthy rotation for {team_id} must total 240 minutes")
    roster_ids = set(out["player_id"].astype(str))
    if not set(rotation["player_id"]).issubset(roster_ids):
        raise ValueError("Healthy rotation contains a player without a current projection")
    changes: list[dict[str, object]] = []
    out.loc[out["team_id"].isin(rotation["team_id"]), "projected_mpg"] = 0.0
    for _, item in rotation.iterrows():
        idx = out.index[out["player_id"].astype(str) == item["player_id"]][0]
        old = base.at[idx, "projected_mpg"]
        out.at[idx, "projected_mpg"] = float(item["healthy_mpg"])
        change = _change(
            out.loc[idx], "projected_mpg", old, out.at[idx, "projected_mpg"], item["reason"]
        )
        change["source"] = item["source"]
        changes.append(change)
    return out, rotation, changes


def _change(row: pd.Series, target: str, old: float, new: float, reason: str) -> dict[str, object]:
    return {
        "entity_type": "player",
        "entity_id": row["player_id"],
        "player_name": row["player_name"],
        "target": target,
        "baseline": old,
        "final": new,
        "reason": reason,
    }


def apply_overrides(
    base: pd.DataFrame, settings: Settings
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    out = base.copy()
    changes: list[dict[str, object]] = []
    overrides = _read_optional(settings.paths.inputs / "overrides.csv", ["entity_id"])
    if overrides.empty:
        return out, changes
    allowed = set(out.columns) - {"player_id", "player_name", "team_id"}
    for _, override in overrides.iterrows():
        target = str(override["target"])
        if target not in allowed:
            raise ValueError(f"Override target is not a primitive model field: {target}")
        selector = "player_id" if override["entity_type"] == "player" else "team_id"
        matches = out.index[out[selector].astype(str) == str(override["entity_id"])]
        for idx in matches:
            old = float(out.at[idx, target])
            value = float(override["value"])
            operation = override["operation"]
            new = (
                value
                if operation == "set"
                else old + value
                if operation == "delta"
                else old * value
            )
            if target.endswith("_per_min") or target in {
                "projected_mpg",
                "projected_usage",
                "expected_gp",
            }:
                ratio = new / old if old else 1
                if (
                    not settings.projection.min_context_multiplier
                    <= ratio
                    <= settings.projection.max_context_multiplier
                ):
                    raise ValueError(
                        f"Override for {override['entity_id']} {target} exceeds safety bounds"
                    )
            out.at[idx, target] = new
            item = _change(out.loc[idx], target, old, new, str(override["reason"]))
            item.update(
                {
                    "source": override["source"],
                    "reviewed_at": override["reviewed_at"],
                    "confidence": override["confidence"],
                }
            )
            changes.append(item)
    return out, changes


def constrain_teams(
    df: pd.DataFrame,
    healthy_rotations: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Reconcile excess scheduled team minutes without positional quotas."""
    out = df.copy()
    checks: list[dict[str, object]] = []
    changes: list[dict[str, object]] = []
    for team_id, team in out.groupby("team_id"):
        before = float((team["projected_mpg"] * team["expected_gp"] / 82).sum())
        factor = min(1.0, 240.0 / before) if before > 0 else 1.0
        if factor < 1.0:
            for idx, player in team.iterrows():
                old = float(player["projected_mpg"])
                new = old * factor
                out.at[idx, "projected_mpg"] = new
                changes.append(_change(
                    player, "projected_mpg", old, new,
                    f"team {team_id} season-minute capacity reconciliation ({factor:.4f})",
                ))
        reconciled = out.loc[team.index]
        healthy_minutes_sum = float(reconciled["projected_mpg"].sum())
        expected_team_mpg = float(
            (reconciled["projected_mpg"] * reconciled["expected_gp"] / 82).sum()
        )
        checks.append(
            {
                "team_id": team_id,
                "players": len(team),
                "pre_reconciliation_expected_mpg": before,
                "reconciliation_factor": factor,
                "sum_player_mpg": healthy_minutes_sum,
                "expected_team_mpg": expected_team_mpg,
                "capacity_gap": 240 - expected_team_mpg,
                "minutes_ok": expected_team_mpg <= 240.01,
                "usage_ok": team["projected_usage"].between(0, 50).all(),
            }
        )
    return out, pd.DataFrame(checks), changes


def finalize_box_score(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df.copy()
    # Team possessions while this player is on court.
    out["projected_possessions"] = out["projected_mpg"] * out["projected_pace"] / 48
    out["expected_season_possessions"] = out["projected_possessions"] * out["expected_gp"]
    historical_possessions_per_min = out["base_pace"].clip(lower=1) / 48
    for stat in ["reb", "ast", "stl", "blk"]:
        out[stat] = (
            out[f"{stat}_per_min"]
            / historical_possessions_per_min
            * out["projected_possessions"]
        )
    # Anchor shot events to observed plays per minute. NBA usage and PACE do
    # not have interchangeable denominators: offensive rebounds permit more
    # FGA + possession-ending FTA + TOV than team possessions. Apply the
    # forecast usage *change* to observed play volume instead of multiplying
    # usage by on-court possessions.
    fga_rate = out["fga_per_min"].clip(lower=0)
    fta_rate = out["fta_per_min"].clip(lower=0)
    tov_rate = out["tov_per_min"].clip(lower=0)
    terminal_rate = (fga_rate + 0.44 * fta_rate + tov_rate).replace(0, np.nan)
    usage_change = (
        out["projected_usage"] / out["baseline_usage"].replace(0, np.nan)
    ).fillna(1.0)
    out["projected_plays"] = (
        terminal_rate.fillna(0)
        * out["projected_mpg"]
        * out["projected_pace"]
        / out["base_pace"].clip(lower=1)
        * usage_change
    )
    out["plays_per_possession"] = out["projected_plays"] / out[
        "projected_possessions"
    ].replace(0, np.nan)
    terminal_events = out["projected_plays"]
    out["fga"] = (terminal_events * fga_rate / terminal_rate).fillna(0)
    out["fta"] = (terminal_events * fta_rate / terminal_rate).fillna(0)
    out["tov"] = (terminal_events * tov_rate / terminal_rate).fillna(0)
    out["fg3a"] = out["fga"] * (
        out["fg3a_per_min"] / out["fga_per_min"].replace(0, np.nan)
    ).clip(0, 1).fillna(0)
    out["fg3a"] = np.minimum(out["fg3a"], out["fga"])
    out["fg3m"] = out["fg3a"] * out["fg3m_pct"]
    out["fgm"] = np.maximum(out["fg3m"], out["fga"] * out["fgm_pct"])
    out["ftm"] = out["fta"] * out["ftm_pct"]
    out["pts"] = 2 * out["fgm"] + out["fg3m"] + out["ftm"]
    out["fg_pct"] = out["fgm"] / out["fga"].replace(0, np.nan)
    out["fg3_pct"] = out["fg3m"] / out["fg3a"].replace(0, np.nan)
    out["ft_pct"] = out["ftm"] / out["fta"].replace(0, np.nan)
    scoring = settings.scoring.model_dump()
    out["fppg"] = sum(out[stat] * weight for stat, weight in scoring.items())
    out["expected_season_points"] = out["fppg"] * out["expected_gp"]
    return out


def simulate(df: pd.DataFrame, settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    residuals = load_residuals(settings.paths.processed / "end_to_end_predictions.parquet")
    if len(residuals) >= 100:
        rng = np.random.default_rng(settings.random_seed)
        summaries = []
        sample_rows = []
        for _, row in df.iterrows():
            fppg, games = draw_joint_outcomes(
                float(row["fppg"]), float(row["expected_gp"]),
                residuals, settings.simulation_count, rng,
            )
            totals = fppg * games
            summaries.append({
                "player_id": row["player_id"],
                "season_points_p20": np.quantile(totals, 0.20),
                "season_points_median": np.median(totals),
                "season_points_p80": np.quantile(totals, 0.80),
                "simulated_mean": totals.mean(),
                "gp_p20": np.quantile(games, 0.20),
                "gp_median": np.median(games),
                "gp_p80": np.quantile(games, 0.80),
                "uncertainty_source": "paired_walk_forward_residuals",
            })
            sample_rows.append(pd.DataFrame({
                "player_id": row["player_id"], "draw": np.arange(settings.simulation_count),
                "games": games, "season_points": totals,
            }))
        return pd.DataFrame(summaries), pd.concat(sample_rows, ignore_index=True)
    rng = np.random.default_rng(settings.random_seed)
    summaries: list[dict[str, float | str]] = []
    sample_rows: list[pd.DataFrame] = []
    n = settings.simulation_count
    weights = settings.scoring.model_dump()
    for _, row in df.iterrows():
        p = np.clip(row["expected_gp"] / 82, 0.01, 0.995)
        concentration = 30
        availability = rng.beta(p * concentration, (1 - p) * concentration, n)
        games = rng.binomial(82, availability)
        rookie_uncertainty = 0.05 if row["historical_minutes"] < 1500 else 0.0
        role_sd = 0.04 + (1 - float(row.get("role_security", 0.8))) * 0.10 + rookie_uncertainty
        minutes_factor = np.clip(rng.normal(1, role_sd, n), 0.65, 1.20)
        fantasy_pg = np.zeros(n)
        generated: dict[str, np.ndarray] = {}
        for stat in COUNTING:
            sd = RATE_UNCERTAINTY[stat]
            generated[stat] = row[stat] * minutes_factor * rng.lognormal(-sd * sd / 2, sd, n)
        generated["fg3a"] = np.minimum(generated["fg3a"], generated["fga"])
        for made, attempted, pct_col in [
            ("fgm", "fga", "fgm_pct"),
            ("fg3m", "fg3a", "fg3m_pct"),
            ("ftm", "fta", "ftm_pct"),
        ]:
            attempt = generated[attempted]
            pct_sd = 0.018 if made == "fgm" else 0.025 if made == "fg3m" else 0.016
            pct = np.clip(rng.normal(row[pct_col], pct_sd, n), 0.15, 0.95)
            generated[made] = attempt * pct
        generated["fgm"] = np.maximum(generated["fgm"], generated["fg3m"])
        generated["pts"] = 2 * generated["fgm"] + generated["fg3m"] + generated["ftm"]
        for stat, weight in weights.items():
            fantasy_pg += generated[stat] * weight
        totals = fantasy_pg * games
        summaries.append(
            {
                "player_id": row["player_id"],
                "season_points_p20": np.quantile(totals, 0.20),
                "season_points_median": np.median(totals),
                "season_points_p80": np.quantile(totals, 0.80),
                "simulated_mean": totals.mean(),
                "gp_p20": np.quantile(games, 0.20),
                "gp_median": np.median(games),
                "gp_p80": np.quantile(games, 0.80),
                "uncertainty_source": "uncalibrated_fixture_fallback",
            }
        )
        sample_rows.append(
            pd.DataFrame(
                {
                    "player_id": row["player_id"],
                    "draw": np.arange(n),
                    "games": games,
                    "season_points": totals,
                }
            )
        )
    return pd.DataFrame(summaries), pd.concat(sample_rows, ignore_index=True)


def add_draft_values(df: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    out = df.copy()
    if settings.replacement_fppg is None:
        available = out.loc[
            (out["projected_mpg"] > 0) & (out["expected_gp"] > 0), "fppg"
        ].sort_values(ascending=False)
        if available.empty:
            raise ValueError("Cannot estimate replacement level from an empty player pool")
        index = min(settings.league.drafted_players, len(available) - 1)
        replacement = float(available.iloc[index])
        out["replacement_source"] = (
            "projected_undrafted_pool" if len(available) > settings.league.drafted_players
            else "incomplete_pool_lower_bound"
        )
    else:
        replacement = settings.replacement_fppg
        out["replacement_source"] = "manual_config"
    out["replacement_fppg"] = replacement
    out["draft_value"] = (out["fppg"] - replacement) * out["expected_gp"]
    out["draft_value_metric"] = "points_above_waiver_replacement"
    out["drafted_players"] = settings.league.drafted_players
    out["raw_rank"] = out["expected_season_points"].rank(
        ascending=False, method="first"
    ).astype(int)
    out["draft_rank"] = out["draft_value"].rank(
        ascending=False, method="first"
    ).astype(int)
    out["drafted_in_model"] = out["draft_rank"] <= settings.league.drafted_players
    return out


def assign_ranks_and_tiers(
    df: pd.DataFrame, settings: Settings, metric: str = "raw"
) -> pd.DataFrame:
    if metric not in {"raw", "draft"}:
        raise ValueError(f"Unknown board metric: {metric}")
    out = df if {"raw_rank", "draft_rank"}.issubset(df) else add_draft_values(df, settings)
    value_column = "expected_season_points" if metric == "raw" else "draft_value"
    rank_column = "raw_rank" if metric == "raw" else "draft_rank"
    out = out.sort_values(rank_column).head(settings.top_n).copy()
    out["our_rank"] = out[rank_column]
    out["board_type"] = metric
    tier = 1
    leader = None
    tiers: list[int] = []
    for points in out[value_column]:
        if leader is None:
            leader = points
        elif points < leader * (1 - settings.projection.tier_drop):
            tier += 1
            leader = points
        tiers.append(tier)
    out["our_tier"] = tiers
    return out


def project(settings: Settings) -> ProjectionResult:
    full_history = pd.read_parquet(settings.paths.processed / "player_seasons.parquet")
    history = full_history.copy()
    roster_path = settings.paths.processed / "current_rosters.parquet"
    rosters = pd.read_parquet(roster_path) if roster_path.exists() else pd.DataFrame()
    history, data_quality = append_rookie_translations(history, rosters, settings)
    if not rosters.empty:
        current_ids = set(rosters["player_id"].astype(str))
        history = history[history["player_id"].astype(str).isin(current_ids)].copy()
    base = build_baseline(history, rosters, settings)
    base, pace_backtest = apply_current_team_pace(base, full_history, rosters, settings)
    current, opportunity_backtest, a = apply_opportunity_forecasts(
        base, full_history, rosters, settings,
        rookie_profiles=history[history.get("rookie_prior", pd.Series(False, index=history.index)).fillna(False)],
    )
    current, availability_backtest, availability_changes = apply_availability_forecasts(
        current, full_history, rosters, settings
    )
    current, b = apply_team_context(current, settings)
    current, healthy_rotations, c = apply_healthy_rotations(current, settings)
    current, d = apply_player_context(current, settings)
    current, injury_changes = apply_injury_context(current, settings)
    current, e = apply_overrides(current, settings)
    current, checks, reconciliation_changes = constrain_teams(current, healthy_rotations)
    rotation_detail = current[
        ["team_id", "player_id", "player_name", "expected_gp", "projected_mpg", "projected_usage"]
    ].copy()
    rotation_detail = rotation_detail.merge(
        rosters[["player_id", "position"]] if "position" in rosters else rosters[["player_id"]],
        on="player_id",
        how="left",
    )
    rotation_detail["season_minutes"] = (
        rotation_detail["projected_mpg"] * rotation_detail["expected_gp"]
    )
    if not healthy_rotations.empty:
        rotation_detail = rotation_detail.merge(
            healthy_rotations[["player_id", "healthy_mpg"]], on="player_id", how="left"
        )
    else:
        rotation_detail["healthy_mpg"] = np.nan
    rotation_detail = rotation_detail.sort_values(
        ["team_id", "projected_mpg"], ascending=[True, False]
    )
    current = finalize_box_score(current, settings)
    summary, samples = simulate(current, settings)
    current = current.merge(summary, on="player_id", how="left")
    current = add_draft_values(current, settings)
    full_settings = settings.model_copy(update={"top_n": len(current)})
    all_projections = assign_ranks_and_tiers(current, full_settings, "raw")
    raw_board = assign_ranks_and_tiers(current, settings, "raw")
    draft_board = assign_ranks_and_tiers(current, settings, "draft")
    return ProjectionResult(
        raw_board,
        draft_board,
        checks,
        rotation_detail,
        pd.DataFrame(
            a + availability_changes + injury_changes + b + c + d + e
            + reconciliation_changes
        ),
        samples,
        data_quality,
        healthy_rotations,
        opportunity_backtest,
        availability_backtest,
        pace_backtest,
        all_projections,
    )
