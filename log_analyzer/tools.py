"""Provider-neutral log-exploration tools.

Each tool is authored exactly **once** here: a canonical JSON Schema (the
`parameters`) describing its inputs, plus a Python implementation that runs
against the full set of log rows. The schemas are deliberately plain JSON
Schema — the common denominator that every provider's tool/function-calling
format is built on — so each provider only has to wrap them in its own envelope
(see `tools.md` / the provider adapters), never redefine them.

The tools let the model explore the *whole* log file, not just the random sample
it was shown in the prompt: grep across every line, pull a context window around
an interesting LineId, or get a level histogram over a range.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

#: Canonical tool definitions. `name` + `description` + JSON-Schema `parameters`.
#: Providers translate these into their own shapes (Claude `input_schema`,
#: OpenAI `function.parameters`, Gemini `function_declarations`).
TOOL_SCHEMAS: list[dict] = [
    {
        "name": "grep_logs",
        "description": (
            "Search the Content of every log line for a regular-expression "
            "pattern, optionally restricted to a single log level. Returns the "
            "matching lines (LineId, Time, Level, Content)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regular expression matched (case-insensitively) against each line's Content.",
                },
                "level": {
                    "type": "string",
                    "description": "Optional log level to filter by, e.g. 'error', 'notice', 'warn'.",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_window",
        "description": (
            "Return the log lines surrounding a given LineId — n lines before and "
            "n lines after — to see the context around an event."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "center_line_id": {
                    "type": "integer",
                    "description": "The LineId to center the window on.",
                },
                "n": {
                    "type": "integer",
                    "description": "Number of lines to include on each side of the center line.",
                },
            },
            "required": ["center_line_id", "n"],
            "additionalProperties": False,
        },
    },
    {
        "name": "count_by_level",
        "description": (
            "Count log lines grouped by level, optionally restricted to an "
            "inclusive LineId range [start, end]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start": {
                    "type": "integer",
                    "description": "Optional inclusive lower LineId bound.",
                },
                "end": {
                    "type": "integer",
                    "description": "Optional inclusive upper LineId bound.",
                },
            },
            "additionalProperties": False,
        },
    },
]

#: Cap on how many rows a single tool call returns, to keep tool results from
#: blowing up the context window. Truncation is reported in the payload.
_MAX_ROWS = 50


def _line_id(row: dict[str, str]) -> int | None:
    try:
        return int(row["LineId"])
    except (KeyError, TypeError, ValueError):
        return None


def _slim(row: dict[str, str]) -> dict[str, str]:
    """Project a row down to the fields the model needs back from a tool."""
    return {k: row.get(k, "") for k in ("LineId", "Time", "Level", "Content")}


@dataclass
class LogToolkit:
    """Executes the canonical tools against a fixed set of log rows.

    Construct one per run with the full CSV (`rows`), then `dispatch(name, args)`
    for each tool call the model makes. The result is a JSON string suitable as
    a tool-result payload for any provider.
    """

    rows: list[dict[str, str]]

    def grep_logs(self, pattern: str, level: str | None = None) -> dict:
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            return {"error": f"invalid regex: {exc}"}

        wanted_level = level.lower() if level else None
        matches = [
            _slim(r)
            for r in self.rows
            if rx.search(r.get("Content", ""))
            and (wanted_level is None or r.get("Level", "").lower() == wanted_level)
        ]
        return {
            "match_count": len(matches),
            "returned": min(len(matches), _MAX_ROWS),
            "truncated": len(matches) > _MAX_ROWS,
            "matches": matches[:_MAX_ROWS],
        }

    def get_window(self, center_line_id: int, n: int) -> dict:
        n = max(0, int(n))
        lo, hi = int(center_line_id) - n, int(center_line_id) + n
        window = [
            _slim(r)
            for r in self.rows
            if (lid := _line_id(r)) is not None and lo <= lid <= hi
        ]
        window.sort(key=lambda r: int(r["LineId"]))
        return {
            "center_line_id": int(center_line_id),
            "n": n,
            "returned": len(window),
            "lines": window,
        }

    def count_by_level(self, start: int | None = None, end: int | None = None) -> dict:
        counts: dict[str, int] = {}
        considered = 0
        for r in self.rows:
            lid = _line_id(r)
            if start is not None and (lid is None or lid < start):
                continue
            if end is not None and (lid is None or lid > end):
                continue
            considered += 1
            level = r.get("Level", "") or "(none)"
            counts[level] = counts.get(level, 0) + 1
        return {
            "start": start,
            "end": end,
            "total_lines": considered,
            "counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        }

    def dispatch(self, name: str, args: dict) -> str:
        """Run a tool call by name and return its result as a JSON string.

        Unknown tools and bad arguments come back as an `{"error": ...}` payload
        rather than raising, so the model can read the error and recover.
        """
        handlers = {
            "grep_logs": self.grep_logs,
            "get_window": self.get_window,
            "count_by_level": self.count_by_level,
        }
        handler = handlers.get(name)
        if handler is None:
            return json.dumps({"error": f"unknown tool: {name!r}"})
        try:
            return json.dumps(handler(**(args or {})))
        except TypeError as exc:
            return json.dumps({"error": f"bad arguments for {name}: {exc}"})
