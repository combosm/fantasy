from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class PlayerSeason(BaseModel):
    season: str
    player_id: str
    player_name: str
    team_id: str
    age: float = Field(gt=15, lt=50)
    games: int = Field(ge=0, le=82)
    minutes: float = Field(ge=0)
    team_pace: float = Field(gt=70, lt=130)
    usage: float = Field(ge=0, le=50)
    pts: float = Field(ge=0)
    reb: float = Field(ge=0)
    ast: float = Field(ge=0)
    stl: float = Field(ge=0)
    blk: float = Field(ge=0)
    tov: float = Field(ge=0)
    fgm: float = Field(ge=0)
    fga: float = Field(ge=0)
    fg3m: float = Field(ge=0)
    fg3a: float = Field(ge=0)
    ftm: float = Field(ge=0)
    fta: float = Field(ge=0)
    ast_pct: float | None = None
    reb_pct: float | None = None
    assisted_fg_rate: float | None = None
    catch_shoot_3a: float | None = None
    pullup_3a: float | None = None

    @model_validator(mode="after")
    def validate_box_score(self) -> PlayerSeason:
        if self.fgm > self.fga or self.fg3m > self.fg3a or self.ftm > self.fta:
            raise ValueError("makes cannot exceed attempts")
        if self.fg3m > self.fgm or self.fg3a > self.fga:
            raise ValueError("three-point values cannot exceed field-goal values")
        return self


class Override(BaseModel):
    entity_type: Literal["player", "team"]
    entity_id: str
    target: str
    operation: Literal["set", "delta", "multiply"]
    value: float
    reason: str = Field(min_length=3)
    source: str = Field(min_length=1)
    reviewed_at: str
    confidence: float = Field(ge=0, le=1)
