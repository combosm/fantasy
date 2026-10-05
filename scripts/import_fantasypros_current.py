"""Freeze a dated FantasyPros ESPN-points ECR HTML snapshot for audit only.

Usage: python scripts/import_fantasypros_current.py SNAPSHOT.html YYYY-MM-DD
Source: https://www.fantasypros.com/nba/rankings/overall-points-espn.php
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

SOURCE_URL = "https://www.fantasypros.com/nba/rankings/overall-points-espn.php"
ENTRY = re.compile(r'\{"player_id":\d+,"player_name":.*?"rank_ecr":"\d+".*?\}')


def main(path: str, date: str) -> None:
    html = Path(path).read_text()
    records = [json.loads(match.group()) for match in ENTRY.finditer(html)]
    if len(records) < 200:
        raise ValueError("Too few embedded player rankings in the snapshot")
    ranked = pd.DataFrame(records)
    if ranked.rank_ecr.duplicated().any():
        raise ValueError("Duplicate ECR rank in snapshot")
    output = pd.DataFrame({
        "source": "FantasyPros ESPN points ECR",
        "published_at": date,
        "player_name": ranked.player_name,
        "rank": ranked.rank_ecr.astype(int),
        "source_url": SOURCE_URL,
        "player_id": ranked.player_nbacom_id.astype(str),
    }).sort_values("rank")
    destination = Path("data/inputs/external_rankings.csv")
    output.to_csv(destination, index=False)
    print(f"Wrote {len(output)} audit-only ranks to {destination}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
