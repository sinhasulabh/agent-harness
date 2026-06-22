"""Read the structured log CSV and sample a random batch of lines."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .config import SamplingConfig


def read_logs(path: str | Path) -> list[dict[str, str]]:
    """Read every row of the structured log CSV into a list of dicts."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No log rows found in {path}")
    return rows


def sample_logs(
    rows: list[dict[str, str]], sampling: SamplingConfig
) -> list[dict[str, str]]:
    """Pick a random number of lines (within the configured bounds) to analyze."""
    rng = random.Random(sampling.seed)

    upper = min(sampling.max_size, len(rows))
    lower = min(sampling.min_size, upper)
    count = rng.randint(lower, upper)

    if sampling.strategy == "random":
        return rng.sample(rows, count)

    # contiguous: a random consecutive window keeps correlated events together
    start = rng.randint(0, len(rows) - count)
    return rows[start : start + count]


def format_logs(rows: list[dict[str, str]]) -> str:
    """Render rows as `[LineId] <Time> <Level> <Content>` lines for the prompt."""
    lines = []
    for row in rows:
        line_id = row.get("LineId", "?")
        time = row.get("Time", "")
        level = row.get("Level", "")
        content = row.get("Content", "")
        lines.append(f"[{line_id}] {time}  {level}  {content}")
    return "\n".join(lines)
