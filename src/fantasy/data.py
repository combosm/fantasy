from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Settings

NBA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}

TEAM_IDS = [
    1610612737,
    1610612738,
    1610612739,
    1610612740,
    1610612741,
    1610612742,
    1610612743,
    1610612744,
    1610612745,
    1610612746,
    1610612747,
    1610612748,
    1610612749,
    1610612750,
    1610612751,
    1610612752,
    1610612753,
    1610612754,
    1610612755,
    1610612756,
    1610612757,
    1610612758,
    1610612759,
    1610612760,
    1610612761,
    1610612762,
    1610612763,
    1610612764,
    1610612765,
    1610612766,
]


def nba_result_frame(payload: dict[str, Any], index: int = 0) -> pd.DataFrame:
    sets = payload.get("resultSets") or payload.get("resultSet")
    if isinstance(sets, dict):
        sets = [sets]
    result = sets[index]
    return pd.DataFrame(result["rowSet"], columns=result["headers"])


class CachedNBAClient:
    def __init__(self, raw_dir: Path, max_age_hours: int) -> None:
        self.raw_dir = raw_dir
        self.max_age = max_age_hours * 3600
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(NBA_HEADERS)

    def get(self, endpoint: str, params: dict[str, Any], cache_key: str) -> dict[str, Any]:
        path = self.raw_dir / f"{cache_key}.json"
        if path.exists() and time.time() - path.stat().st_mtime <= self.max_age:
            return json.loads(path.read_text())
        url = f"https://stats.nba.com/stats/{endpoint}"
        response = self.session.get(url, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
        path.write_text(json.dumps(payload))
        time.sleep(0.7)
        return payload


BASE_PARAMS = {
    "College": "",
    "Conference": "",
    "Country": "",
    "DateFrom": "",
    "DateTo": "",
    "Division": "",
    "DraftPick": "",
    "DraftYear": "",
    "GameScope": "",
    "GameSegment": "",
    "Height": "",
    "LastNGames": 0,
    "LeagueID": "00",
    "Location": "",
    "Month": 0,
    "OpponentTeamID": 0,
    "Outcome": "",
    "PORound": 0,
    "PaceAdjust": "N",
    "PerMode": "PerGame",
    "Period": 0,
    "PlayerExperience": "",
    "PlayerPosition": "",
    "PlusMinus": "N",
    "Rank": "N",
    "SeasonSegment": "",
    "SeasonType": "Regular Season",
    "ShotClockRange": "",
    "StarterBench": "",
    "TeamID": 0,
    "VsConference": "",
    "VsDivision": "",
    "Weight": "",
}


def ingest_live(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch NBA historical player stats and target-season rosters.

    NBA Stats endpoints are cached. If a live refresh fails, a prior cached response remains usable.
    """
    client = CachedNBAClient(settings.paths.raw, settings.cache_max_age_hours)
    frames: list[pd.DataFrame] = []
    archived_frames: list[pd.DataFrame] = []
    for season in settings.historical_seasons:
        archive_path = settings.paths.raw / "archive" / f"players_{int(season[:4]) + 1}.parquet"
        if archive_path.exists():
            archived_frames.append(
                normalize_archive_player_season(pd.read_parquet(archive_path), season)
            )
            continue
        params = {**BASE_PARAMS, "Season": season, "MeasureType": "Base"}
        base = nba_result_frame(
            client.get("leaguedashplayerstats", params, f"players_base_{season}")
        )
        params["MeasureType"] = "Advanced"
        advanced = nba_result_frame(
            client.get("leaguedashplayerstats", params, f"players_advanced_{season}")
        )
        keep = [c for c in ["PLAYER_ID", "USG_PCT", "PACE"] if c in advanced]
        frame = base.merge(advanced[keep], on="PLAYER_ID", how="left")
        frame["season"] = season
        frames.append(frame)

    normalized_live = normalize_nba_history(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    history = pd.concat([normalized_live, *archived_frames], ignore_index=True)
    history = history.sort_values(["season", "player_id"]).reset_index(drop=True)
    roster_frames: list[pd.DataFrame] = []
    for team_id in TEAM_IDS:
        params = {"LeagueID": "00", "Season": settings.target_season, "TeamID": team_id}
        try:
            roster = nba_result_frame(
                client.get("commonteamroster", params, f"roster_{settings.target_season}_{team_id}")
            )
        except (requests.RequestException, KeyError, IndexError, ValueError):
            continue
        if not roster.empty:
            roster["TEAM_ID"] = team_id
            roster_frames.append(roster)
    rosters = (
        normalize_rosters(pd.concat(roster_frames, ignore_index=True))
        if roster_frames
        else pd.DataFrame()
    )
    rookie_stats = ingest_summer_rookies(client, settings)
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    history.to_parquet(settings.paths.processed / "player_seasons.parquet", index=False)
    rosters.to_parquet(settings.paths.processed / "current_rosters.parquet", index=False)
    rookie_stats.to_parquet(settings.paths.processed / "rookie_source_stats.parquet", index=False)
    return history, rosters


def ingest_summer_rookies(client: CachedNBAClient, settings: Settings) -> pd.DataFrame:
    params = {
        **BASE_PARAMS,
        "Season": settings.target_season,
        "SeasonType": "Summer League",
        "MeasureType": "Base",
    }
    try:
        summer = nba_result_frame(
            client.get("leaguedashplayerstats", params, f"players_summer_{settings.target_season}")
        )
    except (requests.RequestException, KeyError, IndexError, ValueError):
        return pd.DataFrame()
    if summer.empty:
        return pd.DataFrame()
    draft_year = settings.target_season.split("-")[0]
    draft_params = {
        "LeagueID": "00",
        "Season": draft_year,
        "College": "",
        "TeamID": 0,
        "TopX": "",
        "OverallPick": "",
        "RoundNum": "",
        "RoundPick": "",
    }
    try:
        draft = nba_result_frame(client.get("drafthistory", draft_params, f"draft_{draft_year}"))
        pick_map = draft.set_index("PERSON_ID")["OVERALL_PICK"].to_dict()
    except (requests.RequestException, KeyError, IndexError, ValueError):
        pick_map = {}
    mapping = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "AGE": "age",
        "GP": "games",
        "MIN": "minutes",
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "FGM": "fgm",
        "FGA": "fga",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FTM": "ftm",
        "FTA": "fta",
    }
    out = summer.rename(columns=mapping)
    out = out[[c for c in mapping.values() if c in out]].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["source_league"] = "NBA Summer League"
    out["draft_pick"] = pd.to_numeric(summer["PLAYER_ID"].map(pick_map), errors="coerce").fillna(60)
    return out


def ingest_draft_history(settings: Settings) -> pd.DataFrame:
    """Cache official NBA draft slots for empirical rookie priors."""
    client = CachedNBAClient(settings.paths.raw, settings.cache_max_age_hours)
    years = sorted({int(season[:4]) for season in settings.historical_seasons} | {
        int(settings.target_season[:4])
    })
    frames = []
    for year in years:
        params = {
            "LeagueID": "00", "Season": str(year), "College": "", "TeamID": 0,
            "TopX": "", "OverallPick": "", "RoundNum": "", "RoundPick": "",
        }
        draft = nba_result_frame(client.get("drafthistory", params, f"draft_{year}"))
        if draft.empty:
            raise ValueError(f"NBA draft-history endpoint returned no picks for {year}")
        frame = draft[["PERSON_ID", "PLAYER_NAME", "OVERALL_PICK"]].rename(columns={
            "PERSON_ID": "player_id", "PLAYER_NAME": "player_name",
            "OVERALL_PICK": "draft_pick",
        })
        frame["draft_year"] = year
        frame["player_id"] = frame.player_id.astype(str)
        frame["draft_pick"] = pd.to_numeric(frame.draft_pick, errors="raise")
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    if result.duplicated(["draft_year", "player_id"]).any():
        raise ValueError("Official draft history contains duplicate player/year rows")
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    result.to_parquet(settings.paths.processed / "draft_history.parquet", index=False)
    return result


def normalize_nba_history(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "PLAYER_ID": "player_id",
        "PLAYER_NAME": "player_name",
        "TEAM_ID": "team_id",
        "AGE": "age",
        "GP": "games",
        "MIN": "mpg",
        "PTS": "pts",
        "REB": "reb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "FGM": "fgm",
        "FGA": "fga",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FTM": "ftm",
        "FTA": "fta",
        "USG_PCT": "usage",
        "PACE": "team_pace",
    }
    out = df.rename(columns=mapping)
    required = list(mapping.values()) + ["season"]
    for col in required:
        if col not in out:
            out[col] = np.nan
    out = out[required].copy()
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    out["usage"] = pd.to_numeric(out["usage"], errors="coerce") * 100
    out["team_pace"] = pd.to_numeric(out["team_pace"], errors="coerce")
    out["team_pace"] = out["team_pace"].fillna(100.0)
    return out


def normalize_archive_player_season(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """Map the public season-total archive to our per-game NBA Stats schema."""
    required = {
        "player_id", "player_name", "team_id", "age", "gp", "min", "usg_pct",
        "pace", "pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga",
        "fg3m", "fg3a", "ftm", "fta",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Archive {season} is missing {sorted(missing)}")
    if frame["player_id"].duplicated().any():
        raise ValueError(f"Archive {season} has duplicate player IDs")
    if not frame["gp"].gt(0).all():
        raise ValueError(f"Archive {season} has non-positive games")
    out = frame[["player_id", "player_name", "team_id", "age", "gp", "min", "usg_pct", "pace"]].copy()
    out = out.rename(columns={"gp": "games", "usg_pct": "usage", "pace": "team_pace"})
    out["season"] = season
    out["mpg"] = frame["min"] / frame["gp"]
    out = out.drop(columns="min")
    out["usage"] *= 100
    for stat in ["pts", "reb", "ast", "stl", "blk", "tov", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta"]:
        out[stat] = frame[stat] / frame["gp"]
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    return out


def extend_history_from_archive(settings: Settings) -> pd.DataFrame:
    """Add downloaded 2015-16 through 2020-21 seasons without replacing NBA.com rows."""
    destination = settings.paths.processed / "player_seasons.parquet"
    if not destination.exists():
        raise FileNotFoundError("Run the original NBA ingestion before extending history")
    current = pd.read_parquet(destination)
    archive_dir = settings.paths.raw / "archive"
    frames = [current]
    for season in settings.historical_seasons:
        if season in set(current["season"]):
            continue
        end_year = int(season[:4]) + 1
        path = archive_dir / f"players_{end_year}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}; download from https://github.com/llimllib/nba_data/tree/main/data"
            )
        frames.append(normalize_archive_player_season(pd.read_parquet(path), season))
    joined = pd.concat(frames, ignore_index=True)
    if joined.duplicated(["season", "player_id"]).any():
        raise ValueError("Combined history has duplicate player-season records")
    joined = joined.sort_values(["season", "player_id"]).reset_index(drop=True)
    joined.to_parquet(destination, index=False)
    return joined


def normalize_rosters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.rename(
        columns={"PLAYER_ID": "player_id", "PLAYER": "player_name", "TEAM_ID": "team_id"}
    )
    cols = [c for c in ["player_id", "player_name", "team_id", "POSITION", "AGE"] if c in out]
    out = out[cols].copy()
    out.columns = [c.lower() for c in out.columns]
    out["player_id"] = out["player_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    return out.drop_duplicates("player_id")


def generate_fixture(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic, mathematically valid data for local validation and demos."""
    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    roster_rows: list[dict[str, Any]] = []
    team_ids = ["100", "200", "300"]
    for team_no, team_id in enumerate(team_ids):
        for player_no in range(12):
            pid = f"{team_id}{player_no:02d}"
            name = f"Fixture {chr(65 + team_no)}{player_no + 1}"
            roster_rows.append(
                {
                    "player_id": pid,
                    "player_name": name,
                    "team_id": team_id,
                    "position": "G" if player_no < 4 else "F" if player_no < 8 else "C",
                }
            )
            talent = 1.45 - player_no * 0.055 + team_no * 0.01
            for season_i, season in enumerate(settings.historical_seasons):
                age = 20 + player_no + season_i
                mpg = max(7, 32 - player_no * 1.8 + rng.normal(0, 0.8))
                games = int(np.clip(72 - max(age - 31, 0) * 1.4 + rng.normal(0, 5), 35, 82))
                fga = mpg * 0.43 * talent
                fg3a = fga * (0.28 + 0.025 * (player_no % 4))
                fg_pct = np.clip(0.47 + rng.normal(0, 0.012), 0.40, 0.58)
                fg3_pct = np.clip(0.36 + rng.normal(0, 0.014), 0.29, 0.44)
                fgm = fga * fg_pct
                fg3m = fg3a * fg3_pct
                fta = fga * 0.24
                ftm = fta * np.clip(0.78 + rng.normal(0, 0.02), 0.62, 0.92)
                rows.append(
                    {
                        "season": season,
                        "player_id": pid,
                        "player_name": name,
                        "team_id": team_id,
                        "age": age,
                        "games": games,
                        "mpg": mpg,
                        "team_pace": 98 + team_no * 2,
                        "usage": 25 - player_no * 0.7,
                        "pts": 2 * fgm + fg3m + ftm,
                        "reb": mpg * (0.12 + 0.014 * player_no),
                        "ast": mpg * max(0.04, 0.20 - player_no * 0.01),
                        "stl": mpg * 0.025,
                        "blk": mpg * (0.008 + player_no * 0.002),
                        "tov": mpg * max(0.025, 0.09 - player_no * 0.004),
                        "fgm": fgm,
                        "fga": fga,
                        "fg3m": fg3m,
                        "fg3a": fg3a,
                        "ftm": ftm,
                        "fta": fta,
                    }
                )
    history = pd.DataFrame(rows)
    rosters = pd.DataFrame(roster_rows)
    settings.paths.processed.mkdir(parents=True, exist_ok=True)
    history.to_parquet(settings.paths.processed / "player_seasons.parquet", index=False)
    rosters.to_parquet(settings.paths.processed / "current_rosters.parquet", index=False)
    return history, rosters
