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
class Config:
    provider: str
    log_file: Path
    sampling: SamplingConfig
    models: dict[str, str]
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

    # Resolve log_file relative to the config file's directory if not absolute.
    log_file = Path(raw.get("log_file", "Apache_2k.log_structured_1.csv"))
    if not log_file.is_absolute():
        log_file = (path.parent / log_file).resolve()

    return Config(
        provider=str(raw.get("provider", "claude")),
        log_file=log_file,
        sampling=sampling,
        models=dict(raw.get("models", {})),
        max_tokens=int(raw.get("max_tokens", 4096)),
    )
