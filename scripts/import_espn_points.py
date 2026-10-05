"""Extract dated ESPN preseason H2H-points rankings from downloaded PDFs.

Usage: python scripts/import_espn_points.py PDF_2022 PDF_2023 PDF_2024 PDF_2025
The PDFs must be the exact source URLs below. This command does not download or
silently update historical rankings; archive the PDFs before running it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

SOURCES = [
    ("2022-23", "2022-10-12", "https://g.espncdn.com/s/fbadraftkit/2022-2023/FBA2022CSpts.pdf", 200),
    ("2023-24", "2023-10-20", "https://g.espncdn.com/s/fbadraftkit/2023-2024/FBA2024top200PTS.pdf", 200),
    ("2024-25", "2024-10-18", "https://g.espncdn.com/s/fbadraftkit/2024-25/FBA24ptsCS.pdf", 150),
    ("2025-26", "2025-10-20", "https://g.espncdn.com/s/fbadraftkit/2025-26/fbacheatsheetPts.pdf", 150),
]
ENTRY = re.compile(r"(?<!\d)(\d{1,3})\.\s+(.+?),\s+([A-Za-z]{2,4})(?=\s|$)")


def extract(path: Path, expected_count: int) -> pd.DataFrame:
    page = subprocess.check_output(["pdftotext", "-layout", str(path), "-"], text=True)
    rows = []
    for line in page.splitlines():
        for match in ENTRY.finditer(line):
            rows.append({"rank": int(match.group(1)), "player_name": match.group(2).strip()})
    result = pd.DataFrame(rows).sort_values("rank")
    if result["rank"].tolist() != list(range(1, expected_count + 1)):
        raise ValueError(f"{path}: ranks are not exactly 1 through {expected_count}")
    if result.player_name.duplicated().any():
        raise ValueError(f"{path}: duplicate player name")
    return result


def main(paths: list[str]) -> None:
    if len(paths) != len(SOURCES):
        raise SystemExit(__doc__)
    frames = []
    for path, (season, date, url, count) in zip(paths, SOURCES):
        frame = extract(Path(path), count)
        frame.insert(0, "season", season)
        frame["source"] = "ESPN H2H points preseason cheat sheet"
        frame["published_at"] = date
        frame["url"] = url
        frames.append(frame)
    destination = Path("data/inputs/market_preseason_rankings.csv")
    pd.concat(frames, ignore_index=True).to_csv(destination, index=False)
    print(f"Wrote {sum(len(frame) for frame in frames)} dated rankings to {destination}")


if __name__ == "__main__":
    main(sys.argv[1:])
