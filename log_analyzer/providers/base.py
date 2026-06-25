"""The interface every provider implements."""

from __future__ import annotations

import abc
import os
from typing import TYPE_CHECKING

from ..agent_loop import LoopOutcome, ToolCall, run_agent_loop
from ..config import Config
from ..models import LogAnalysis

if TYPE_CHECKING:
    from ..tools import LogToolkit


class LLMProvider(abc.ABC):
    """A provider turns a batch of formatted log lines into a `LoopOutcome`."""

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
    def analyze(self, log_text: str, toolkit: "LogToolkit") -> LoopOutcome:
        """Analyze the logs and return a `LoopOutcome` (analysis + cost + stop reason).

        `toolkit` exposes the full log file for tool-capable providers; plain
        providers ignore it and analyze only the sampled `log_text`.
        """


class ToolCallingProvider(LLMProvider, abc.ABC):
    """Base for providers that explore the full log via the agent loop.

    Subclass this (instead of `LLMProvider`) to opt into tool calling. The shared
    `run_agent_loop` ([agent_loop.py]) drives the explore→conclude cycle; the
    subclass only implements the dialect **hooks** below — the loop itself never
    knows which provider it's driving. `claude.py` / `nvidia.py` are the templates.
    """

    uses_tools: bool = True

    def analyze(self, log_text: str, toolkit: "LogToolkit") -> LoopOutcome:
        return run_agent_loop(self, toolkit, log_text, self.config.budgets)

    # --- dialect hooks (see agent_loop.AgentHooks) ---

    @abc.abstractmethod
    def seed_messages(self, log_text: str) -> list:
        """Build the initial message list for this provider."""

    @abc.abstractmethod
    def call_model(self, messages: list, use_tools: bool):
        """One model round-trip; `use_tools=False` for the final conclude call."""

    @abc.abstractmethod
    def usage(self, response) -> tuple[int, int]:
        """Extract `(input_tokens, output_tokens)` from a response."""

    @abc.abstractmethod
    def is_refusal(self, response) -> bool:
        """True if the model declined."""

    @abc.abstractmethod
    def is_truncated(self, response) -> bool:
        """True if this single response stopped on max_tokens."""

    @abc.abstractmethod
    def assistant_message(self, response):
        """The assistant turn to append back to `messages`."""

    @abc.abstractmethod
    def extract_tool_calls(self, response) -> list[ToolCall]:
        """Normalize the model's tool requests into `ToolCall`s (empty if none)."""

    @abc.abstractmethod
    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list:
        """Turn executed tool results into the message(s) to append.

        Batched (not per-call) because Anthropic wants all results in one user
        message while OpenAI wants one `tool` message each.
        """

    @abc.abstractmethod
    def extract_final(self, response) -> LogAnalysis | None:
        """Parse a final `LogAnalysis` out of a response, or `None`."""

    @abc.abstractmethod
    def conclude_nudge(self):
        """A user turn instructing the model to stop and emit the final answer."""
