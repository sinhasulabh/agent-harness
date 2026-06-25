"""Tie the pieces together: read -> sample -> analyze."""

from __future__ import annotations

from dataclasses import dataclass

from .agent_loop import StopReason
from .config import Config
from .log_reader import format_logs, read_logs, sample_logs
from .models import LogAnalysis
from .providers import get_provider
from .tools import LogToolkit
from .validation import ValidationResult, validate_analysis


@dataclass
class AnalysisRun:
    provider: str
    model: str
    sampled_line_ids: list[str]
    sample_count: int
    tools_used: bool
    analysis: LogAnalysis | None
    validation: ValidationResult | None
    stop_reason: StopReason
    steps_used: int
    tokens_used: int


def run_analysis(config: Config, provider_name: str | None = None) -> AnalysisRun:
    provider_name = provider_name or config.provider

    rows = read_logs(config.log_file)
    sampled = sample_logs(rows, config.sampling)
    log_text = format_logs(sampled)

    provider = get_provider(provider_name, config)
    # The toolkit exposes the full log file; tool-capable providers explore it,
    # plain providers ignore it and analyze only the sampled lines.
    toolkit = LogToolkit(rows)
    outcome = provider.analyze(log_text, toolkit)

    # Verify the model's cited evidence is grounded in the actual log rows (only
    # when there is an analysis to check). With tools, the model may legitimately
    # cite lines beyond the sample, so grounding is relaxed to the whole file.
    validation = (
        validate_analysis(outcome.analysis, sampled, rows, tools_used=provider.uses_tools)
        if outcome.analysis is not None
        else None
    )

    return AnalysisRun(
        provider=provider_name,
        model=provider.model,
        sampled_line_ids=[r.get("LineId", "?") for r in sampled],
        sample_count=len(sampled),
        tools_used=provider.uses_tools,
        analysis=outcome.analysis,
        validation=validation,
        stop_reason=outcome.stop_reason,
        steps_used=outcome.steps_used,
        tokens_used=outcome.tokens_used,
    )
