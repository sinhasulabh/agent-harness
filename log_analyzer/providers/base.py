"""The interface every provider implements."""

from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING

from ..config import Config
from ..models import LogAnalysis

if TYPE_CHECKING:
    from ..tools import LogToolkit


class LLMProvider(abc.ABC):
    """A provider turns a batch of formatted log lines into a `LogAnalysis`."""

    #: Environment variable holding this provider's API key.
    api_key_env: str = ""

    #: Whether this provider explores the full log via the tool registry.
    #: Plain providers leave this False; tool-capable ones set it True (see
    #: `ToolCallingProvider`). `analyzer` uses it to choose grounding semantics.
    uses_tools: bool = False

    def __init__(self, config: Config) -> None:
        self.config = config
        self.model = config.model_for(self.name)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """The provider key used in config (claude / gemini / nvidia)."""

    def _require_key(self) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(
                f"Missing API key: set {self.api_key_env} in your environment "
                f"or .env file to use the {self.name!r} provider."
            )
        return key

    @abc.abstractmethod
    def analyze(self, log_text: str, toolkit: "LogToolkit") -> LogAnalysis:
        """Return a parsed analysis of the formatted logs.

        `toolkit` exposes the full log file for tool-capable providers; plain
        providers ignore it and analyze only the sampled `log_text`.
        """


class ToolCallingProvider(LLMProvider):
    """Base for providers that explore the full log via the tool registry.

    Subclass this (instead of `LLMProvider` directly) to opt into tool calling.
    The provider runs the tool loop inside its own `analyze()` using the canonical
    schemas/executor in `tools.py` and the per-provider rendering helpers in
    `tool_calling.py`. Shared loop scaffolding can be hoisted here once a second
    tool-capable provider exists.
    """

    uses_tools: bool = True
