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
