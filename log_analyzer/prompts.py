"""Prompt text shared by every provider, so they're judged on the same input."""

from __future__ import annotations

import json

from .models import LogAnalysis

SYSTEM_PROMPT = (
    "You are a senior site reliability engineer triaging server logs. "
    "You are given a batch of structured log lines and must identify the most "
    "likely root cause of any problem they describe. Base your analysis only on "
    "the lines provided. Cite the exact LineId values that support your reasoning. "
    "Be precise and actionable."
)

# Appended to SYSTEM_PROMPT when the provider runs in tool mode. It overrides the
# "only on the lines provided" instruction above, since tools let the model reach
# the rest of the file.
TOOL_SYSTEM_SUFFIX = (
    "\n\nThe batch above is a small random sample of a larger log file. You have "
    "tools to investigate the rest of the file before concluding:\n"
    "- grep_logs(pattern, level?): search every line's Content by regex, "
    "optionally filtered by level.\n"
    "- get_window(center_line_id, n): see the n lines before and after a LineId "
    "for context around an event.\n"
    "- count_by_level(start?, end?): get a level histogram, optionally over a "
    "LineId range.\n\n"
    "Use these tools to confirm or refine your hypothesis, then stop calling tools "
    "and give your final analysis. You may cite any LineId you discovered via the "
    "tools in evidence_line_ids — not just lines from the original sample."
)


def schema_hint() -> str:
    """A compact JSON-schema description, for providers without native schema support."""
    return json.dumps(LogAnalysis.model_json_schema(), indent=2)


def build_user_prompt(log_text: str) -> str:
    return (
        "Analyze the following log lines and return your assessment.\n\n"
        "Each line is formatted as: [LineId] <Time> <Level> <Content>\n\n"
        "----- BEGIN LOGS -----\n"
        f"{log_text}\n"
        "----- END LOGS -----\n\n"
        "Use the LineId values when citing evidence in `evidence_line_ids`."
    )
