"""Load configuration from config.yaml (plus API keys from the environment)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SamplingConfig:
    min_size: int = 1
    max_size: int = 25
    strategy: str = "contiguous"  # "contiguous" | "random"
    seed: int | None = None


@dataclass
class Budgets:
    """Stop conditions for the agent loop — a policy, not a single integer.

    The loop checks these every iteration, before the next model call.
    """

    max_steps: int = 6  # max model round-trips in the tool loop
    max_tokens: int = 60_000  # cumulative input+output tokens across the loop
    max_seconds: float | None = None  # optional wall-clock cap (None = off)


@dataclass
class Config:
    provider: str
    log_file: Path
    sampling: SamplingConfig
    models: dict[str, str]
    budgets: Budgets
    max_tokens: int = 4096

    def model_for(self, provider: str | None = None) -> str:
        provider = provider or self.provider
        try:
            return self.models[provider]
        except KeyError:
            raise ValueError(
                f"No model configured for provider {provider!r}. "
                f"Known providers: {', '.join(sorted(self.models))}."
            )


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}

    sampling_raw = raw.get("sampling", {}) or {}
    sampling = SamplingConfig(
        min_size=int(sampling_raw.get("min_size", 1)),
        max_size=int(sampling_raw.get("max_size", 25)),
        strategy=str(sampling_raw.get("strategy", "contiguous")),
        seed=sampling_raw.get("seed"),
    )
    if sampling.min_size < 1:
        raise ValueError("sampling.min_size must be >= 1")
    if sampling.max_size < sampling.min_size:
        raise ValueError("sampling.max_size must be >= sampling.min_size")
    if sampling.strategy not in ("contiguous", "random"):
        raise ValueError("sampling.strategy must be 'contiguous' or 'random'")

    budgets_raw = raw.get("budgets", {}) or {}
    max_seconds = budgets_raw.get("max_seconds")
    budgets = Budgets(
        max_steps=int(budgets_raw.get("max_steps", 6)),
        max_tokens=int(budgets_raw.get("max_tokens", 60_000)),
        max_seconds=float(max_seconds) if max_seconds is not None else None,
    )
    if budgets.max_steps < 1:
        raise ValueError("budgets.max_steps must be >= 1")
    if budgets.max_tokens < 1:
        raise ValueError("budgets.max_tokens must be >= 1")
    if budgets.max_seconds is not None and budgets.max_seconds <= 0:
        raise ValueError("budgets.max_seconds must be > 0 (or null to disable)")

    # Resolve log_file relative to the config file's directory if not absolute.
    log_file = Path(raw.get("log_file", "Apache_2k.log_structured_1.csv"))
    if not log_file.is_absolute():
        log_file = (path.parent / log_file).resolve()

    return Config(
        provider=str(raw.get("provider", "claude")),
        log_file=log_file,
        sampling=sampling,
        models=dict(raw.get("models", {})),
        budgets=budgets,
        max_tokens=int(raw.get("max_tokens", 4096)),
    )
