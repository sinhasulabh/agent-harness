"""Provider registry. Imports are lazy so you only need the SDK you actually use."""

from __future__ import annotations

from ..config import Config
from .base import LLMProvider


def get_provider(name: str, config: Config) -> LLMProvider:
    """Construct the provider for `name`, importing its SDK lazily."""
    name = name.lower()
    if name == "claude":
        from .claude import ClaudeProvider

        return ClaudeProvider(config)
    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(config)
    if name == "nvidia":
        from .nvidia import NvidiaProvider

        return NvidiaProvider(config)
    raise ValueError(
        f"Unknown provider {name!r}. Choose one of: claude, gemini, nvidia."
    )


__all__ = ["LLMProvider", "get_provider"]
