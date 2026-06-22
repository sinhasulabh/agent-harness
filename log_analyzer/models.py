"""The structured response schema shared by every provider."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Severity = Literal["low", "medium", "high", "critical"]


class LogAnalysis(BaseModel):
    """Root-cause analysis of a batch of log lines.

    The field set and descriptions are reused across all providers — Claude's
    structured outputs, Gemini's response schema, and the prompt-level contract
    for the NVIDIA endpoint all derive from this one model.
    """

    severity: Severity = Field(
        description="Overall severity of the situation described by the logs.",
    )
    suspected_cause: str = Field(
        description="A one-sentence hypothesis for the most likely root cause.",
    )
    evidence_line_ids: list[int] = Field(
        description="The LineId values of the log lines that support the hypothesis.",
    )
    next_step: str = Field(
        description="A concrete, actionable next step to investigate or remediate.",
    )
