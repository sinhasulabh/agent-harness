"""Tie the pieces together: read -> sample -> analyze."""

from __future__ import annotations

from dataclasses import dataclass

from .config import Config
from .log_reader import format_logs, read_logs, sample_logs
from .models import LogAnalysis
from .providers import get_provider
from .validation import ValidationResult, validate_analysis


@dataclass
class AnalysisRun:
    provider: str
    model: str
    sampled_line_ids: list[str]
    sample_count: int
    analysis: LogAnalysis
    validation: ValidationResult


def run_analysis(config: Config, provider_name: str | None = None) -> AnalysisRun:
    provider_name = provider_name or config.provider

    rows = read_logs(config.log_file)
    sampled = sample_logs(rows, config.sampling)
    log_text = format_logs(sampled)

    provider = get_provider(provider_name, config)
    analysis = provider.analyze(log_text)

    # Verify the model's cited evidence is grounded in the actual log rows.
    validation = validate_analysis(analysis, sampled, rows)

    return AnalysisRun(
        provider=provider_name,
        model=provider.model,
        sampled_line_ids=[r.get("LineId", "?") for r in sampled],
        sample_count=len(sampled),
        analysis=analysis,
        validation=validation,
    )
