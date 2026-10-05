from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

STAT_COLUMNS = [
    "pts",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
]


class Scoring(BaseModel):
    pts: float = 1
    fg3m: float = 1
    fgm: float = 2
    fga: float = -1
    ftm: float = 1
    fta: float = -1
    reb: float = 1
    ast: float = 2
    stl: float = 4
    blk: float = 4
    tov: float = -2


class ProjectionOptions(BaseModel):
    season_weights: list[float]
    shrinkage_minutes: float = Field(gt=0)
    shrinkage_shots: float = Field(gt=0)
    max_context_multiplier: float = Field(ge=1)
    min_context_multiplier: float = Field(gt=0, le=1)
    tier_drop: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_weights(self) -> ProjectionOptions:
        if not self.season_weights or sum(self.season_weights) <= 0:
            raise ValueError("season_weights must have a positive sum")
        return self


class Paths(BaseModel):
    raw: Path
    processed: Path
    inputs: Path
    output: Path


class League(BaseModel):
    teams: int = Field(gt=0)
    active_slots: int = Field(gt=0)
    bench_slots: int = Field(ge=0)

    @property
    def drafted_players(self) -> int:
        return self.teams * (self.active_slots + self.bench_slots)


class TopKTargets(BaseModel):
    top30_min: float = Field(ge=0, le=1)
    top30_stretch: float = Field(ge=0, le=1)
    top60_min: float = Field(ge=0, le=1)
    top60_stretch: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bands(self) -> TopKTargets:
        if self.top30_min > self.top30_stretch or self.top60_min > self.top60_stretch:
            raise ValueError("Top-K minimums cannot exceed stretch targets")
        return self


class Settings(BaseModel):
    target_season: str
    historical_seasons: list[str]
    top_n: int = Field(gt=0)
    simulation_count: int = Field(ge=100)
    random_seed: int
    cache_max_age_hours: int = Field(ge=0)
    scoring: Scoring
    projection: ProjectionOptions
    paths: Paths
    league: League
    topk_targets: TopKTargets
    replacement_fppg: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_seasons(self) -> Settings:
        if len(self.projection.season_weights) != len(self.historical_seasons):
            raise ValueError("season_weights must align with historical_seasons")
        return self


def load_settings(path: str | Path = "config/default.yml") -> Settings:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text())
    settings = Settings.model_validate(data)
    root = config_path.resolve().parent.parent
    for field in ("raw", "processed", "inputs", "output"):
        value = getattr(settings.paths, field)
        if not value.is_absolute():
            setattr(settings.paths, field, root / value)
    return settings


OverrideOperation = Literal["set", "delta", "multiply"]
