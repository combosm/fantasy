from __future__ import annotations

import pandas as pd

from fantasy.config import load_settings
from fantasy.rookie import forecast_drafted_rookies, rookie_training_rows, rookie_walk_forward


def test_empirical_draft_fallback_uses_only_completed_draft_classes():
    settings = load_settings()
    records = []
    picks = []
    for year in range(2015, 2026):
        for pick in range(1, 31):
            player_id = str(year * 100 + pick)
            minutes = 31 - pick / 2
            row = {
                "season": f"{year}-{str(year + 1)[-2:]}",
                "player_id": player_id, "player_name": f"Player {player_id}",
                "team_id": "1", "age": 19 + pick % 4,
                "games": 72 - pick / 2, "mpg": minutes,
                "usage": 25 - pick / 4, "team_pace": 100,
                "reb": minutes / 6, "ast": minutes / 10,
                "stl": 1, "blk": 0.5, "tov": 2,
                "fgm": minutes / 6, "fga": minutes / 3,
                "fg3m": 1, "fg3a": 3, "ftm": 2, "fta": 3,
            }
            row["pts"] = 2 * row["fgm"] + row["fg3m"] + row["ftm"]
            records.append(row)
            picks.append({"draft_year": year, "player_id": player_id, "draft_pick": pick})
    picks.append({"draft_year": 2026, "player_id": "new", "draft_pick": 3})
    history = pd.DataFrame(records)
    draft = pd.DataFrame(picks)
    roster = pd.DataFrame([{
        "player_id": "new", "player_name": "New Rookie",
        "team_id": "1", "age": 20,
    }])
    assert len(rookie_training_rows(history, draft)) == 330
    prior, quality = forecast_drafted_rookies(history, draft, roster, settings)
    assert len(prior) == 1
    assert prior.iloc[0].rookie_prior
    assert 0 < prior.iloc[0].games <= 82
    assert prior.iloc[0].fg3m <= prior.iloc[0].fg3a <= prior.iloc[0].fga
    assert quality.passed.all()
    assert not rookie_walk_forward(history, draft, settings).empty
