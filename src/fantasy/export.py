from __future__ import annotations

from pathlib import Path

import pandas as pd

from .audit import build_audit_queue
from .config import Settings
from .validation import ValidationReport


def _autosize(writer: pd.ExcelWriter, sheet: str, df: pd.DataFrame, max_width: int = 42) -> None:
    worksheet = writer.sheets[sheet]
    worksheet.freeze_panes(1, 0)
    worksheet.autofilter(0, 0, max(len(df), 1), max(len(df.columns) - 1, 0))
    for idx, column in enumerate(df.columns):
        values = df[column].astype(str) if not df.empty else pd.Series(dtype=str)
        measured = values.str.len().max()
        content_width = int(measured) + 2 if pd.notna(measured) else 0
        width = min(max(len(str(column)) + 2, content_width), max_width)
        worksheet.set_column(idx, idx, width)


def export_workbook(
    path: Path,
    projections: pd.DataFrame,
    external: pd.DataFrame,
    unmatched: pd.DataFrame,
    adjustments: pd.DataFrame,
    team_checks: pd.DataFrame,
    rotation_detail: pd.DataFrame,
    healthy_rotations: pd.DataFrame,
    factor_backtest: pd.DataFrame,
    model_quality: pd.DataFrame,
    validation: ValidationReport,
    settings: Settings,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    board_cols = [
        "our_rank",
        "our_tier",
        "player_name",
        "team_id",
        "age",
        "expected_gp",
        "projected_mpg",
        "fppg",
        "expected_season_points",
        "season_points_p20",
        "season_points_median",
        "season_points_p80",
        "external_median_rank",
        "external_tier",
        "audit_flag",
    ]
    stat_cols = [
        "player_id",
        "player_name",
        "team_id",
        "age",
        "expected_gp",
        "projected_mpg",
        "projected_usage",
        "projected_pace",
        "projected_possessions",
        "expected_season_possessions",
        "pts",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "fgm",
        "fga",
        "fg_pct",
        "fg3m",
        "fg3a",
        "fg3_pct",
        "ftm",
        "fta",
        "ft_pct",
        "fppg",
        "expected_season_points",
    ]
    methodology = pd.DataFrame(
        [
            {"item": "Target season", "value": settings.target_season},
            {"item": "Ranking basis", "value": "Expected games × ESPN fantasy points per game"},
            {
                "item": "Rotation minutes",
                "value": "Player MPG is forecast when active from player and current-roster features; no positional quotas",
            },
            {
                "item": "Usage",
                "value": "Player usage is separately forecast from role and roster features",
            },
            {"item": "Games played", "value": "Forecast separately; does not alter active-game MPG"},
            {"item": "Age", "value": "Input to MPG, usage and GP forecasts; no hand-set age multipliers on box-score rates"},
            {"item": "Pace", "value": "Team-level season forecast selected by walk-forward MAE"},
            {"item": "Possessions", "value": "MPG × projected team pace / 48"},
            {"item": "Attempts", "value": "Usage × possessions, split by historical FGA/FTA/TOV mix"},
            {"item": "Team capacity", "value": "Expected minute excess is an audit warning, never a silent adjustment"},
            {"item": "External rankings", "value": "Audit only; never used as projection inputs"},
            {
                "item": "Simulation",
                "value": f"{settings.simulation_count} draws; seed {settings.random_seed}",
            },
            {
                "item": "Tier rule",
                "value": f"New tier > {settings.projection.tier_drop:.0%} below tier leader",
            },
            {"item": "Scoring", "value": str(settings.scoring.model_dump())},
        ]
    )
    audit_queue = build_audit_queue(projections)
    data_quality = pd.concat(
        [
            validation.integrity.assign(category="integrity"),
            model_quality.rename(columns={"entity_id": "check"})[
                ["check", "passed", "detail", "category"]
            ]
            if not model_quality.empty
            else pd.DataFrame(),
            unmatched.rename(columns={"player_name": "check", "issue": "detail"}).assign(
                passed=False, category="unmatched ranking"
            )[["check", "passed", "detail", "category"]]
            if not unmatched.empty
            else pd.DataFrame(),
        ],
        ignore_index=True,
    )

    sheets = {
        "Draft Board": projections[[c for c in board_cols if c in projections]].copy(),
        "Full Projections": projections[[c for c in stat_cols if c in projections]].copy(),
        "Team Rotations": rotation_detail,
        "Team Checks": team_checks,
        "Healthy Rotations": healthy_rotations,
        "Context Adjustments": adjustments,
        "External Rankings": external,
        "Audit Queue": audit_queue,
        "Baseline Backtest": validation.backtest,
        "Factor Backtest": factor_backtest,
        "Methodology": methodology,
        "Data Quality": data_quality,
    }
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        workbook = writer.book
        header = workbook.add_format({"bold": True, "bg_color": "#1F4E78", "font_color": "white"})
        decimal = workbook.add_format({"num_format": "0.00"})
        integer = workbook.add_format({"num_format": "0"})
        warning = workbook.add_format({"bg_color": "#FCE4D6"})
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            worksheet = writer.sheets[name]
            worksheet.set_row(0, None, header)
            _autosize(writer, name, frame)
            for idx, col in enumerate(frame.columns):
                if pd.api.types.is_float_dtype(frame[col]):
                    worksheet.set_column(idx, idx, None, decimal)
                elif pd.api.types.is_integer_dtype(frame[col]):
                    worksheet.set_column(idx, idx, None, integer)
            if name == "Draft Board" and "audit_flag" in frame:
                flag_col = frame.columns.get_loc("audit_flag")
                worksheet.conditional_format(
                    1,
                    0,
                    len(frame),
                    len(frame.columns) - 1,
                    {
                        "type": "formula",
                        "criteria": f"=${_excel_col(flag_col)}2=TRUE",
                        "format": warning,
                    },
                )
            if name in {"Team Checks", "Data Quality"} and len(frame):
                status_col = "minutes_ok" if name == "Team Checks" else "passed"
                if status_col in frame:
                    flag_col = frame.columns.get_loc(status_col)
                    worksheet.conditional_format(
                        1,
                        0,
                        len(frame),
                        len(frame.columns) - 1,
                        {
                            "type": "formula",
                            "criteria": f"=${_excel_col(flag_col)}2=FALSE",
                            "format": warning,
                        },
                    )
    return path


def _excel_col(index: int) -> str:
    value = index + 1
    result = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        result = chr(65 + remainder) + result
    return result
