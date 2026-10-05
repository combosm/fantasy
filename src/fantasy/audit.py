from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def normalize_name(value: str) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def load_external_rankings(path: Path) -> pd.DataFrame:
    columns = ["source", "published_at", "player_name", "rank"]
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=columns)
    rankings = pd.read_csv(path)
    if rankings.empty:
        return pd.DataFrame(columns=columns)
    missing = set(columns) - set(rankings.columns)
    if missing:
        raise ValueError(f"External rankings missing columns: {sorted(missing)}")
    optional = [column for column in ("player_id", "source_url") if column in rankings]
    rankings = rankings[columns + optional].dropna(subset=["source", "player_name", "rank"]).copy()
    rankings["rank"] = pd.to_numeric(rankings["rank"], errors="raise").astype(int)
    if "player_id" in rankings:
        rankings["player_id"] = rankings["player_id"].astype(str)
    rankings["name_key"] = rankings["player_name"].map(normalize_name)
    return rankings


def audit_rankings(
    projections: pd.DataFrame, rankings: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = projections.copy()
    base["name_key"] = base["player_name"].map(normalize_name)
    if rankings.empty:
        base["external_median_rank"] = np.nan
        base["external_tier"] = np.nan
        base["rank_difference"] = np.nan
        base["tier_difference"] = np.nan
        base["audit_flag"] = False
        return base.drop(columns="name_key"), pd.DataFrame(columns=["player_name", "issue"])

    duplicate_keys = set(base.loc[base.duplicated("name_key", keep=False), "name_key"])
    matched = rankings.copy()
    ids = set(base.player_id.astype(str))
    if "player_id" in matched:
        matched["matched_player_id"] = matched.player_id.where(matched.player_id.isin(ids))
    else:
        matched["matched_player_id"] = np.nan
    name_ids = base.loc[~base.name_key.isin(duplicate_keys)].set_index("name_key").player_id.astype(str)
    matched["matched_player_id"] = matched.matched_player_id.fillna(
        matched.name_key.map(name_ids)
    )
    medians = matched.groupby("matched_player_id")["rank"].median()
    out = base.copy()
    out["external_median_rank"] = out.player_id.astype(str).map(medians)
    tier_bounds = out.groupby("our_tier")["our_rank"].agg(["min", "max"]).reset_index()

    def external_tier(rank: float) -> float:
        if pd.isna(rank):
            return np.nan
        matches = tier_bounds[(tier_bounds["min"] <= rank) & (tier_bounds["max"] >= rank)]
        if not matches.empty:
            return float(matches.iloc[0]["our_tier"])
        return float(tier_bounds.iloc[-1]["our_tier"] + 1)

    out["external_tier"] = out["external_median_rank"].map(external_tier)
    out["rank_difference"] = out["our_rank"] - out["external_median_rank"]
    out["tier_difference"] = out["our_tier"] - out["external_tier"]
    threshold = np.select([out["our_rank"] <= 50, out["our_rank"] <= 150], [8, 15], default=25)
    out["audit_flag"] = out["external_median_rank"].notna() & (
        (out["tier_difference"].abs() >= 2) | (out["rank_difference"].abs() > threshold)
    )
    unmatched = matched.loc[matched.matched_player_id.isna(), ["player_name", "source"]].copy()
    unmatched["issue"] = "External ranking name did not match a projected player"
    return out.drop(columns="name_key"), unmatched


def build_audit_queue(audited: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "player_id",
        "player_name",
        "our_rank",
        "our_tier",
        "external_median_rank",
        "external_tier",
        "rank_difference",
        "tier_difference",
        "audit_status",
        "audit_explanation",
    ]
    flagged = audited[audited["audit_flag"]].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    flagged["audit_status"] = "Needs review"
    flagged["audit_explanation"] = ""
    return flagged[columns]
