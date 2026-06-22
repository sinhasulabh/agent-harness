"""Claude provider — official Anthropic SDK with structured outputs."""

from __future__ import annotations

import anthropic

from ..models import LogAnalysis
from ..prompts import SYSTEM_PROMPT, build_user_prompt
from .base import LLMProvider


class ClaudeProvider(LLMProvider):
    api_key_env = "ANTHROPIC_API_KEY"

    @property
    def name(self) -> str:
        return "claude"

    def __init__(self, config) -> None:
        super().__init__(config)
        # The SDK reads ANTHROPIC_API_KEY from the environment; _require_key()
        # gives a clearer error if it's missing.
        self._require_key()
        self.client = anthropic.Anthropic()

    def analyze(self, log_text: str) -> LogAnalysis:
        # messages.parse() validates the response against the Pydantic model and
        # returns it on `parsed_output`. Adaptive thinking lets Claude reason as
        # much as the task needs without a fixed token budget.
        message = self.client.messages.parse(
            model=self.model,
            max_tokens=self.config.max_tokens,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(log_text)}],
            output_format=LogAnalysis,
        )

        if message.stop_reason == "refusal":
            raise RuntimeError(
                "Claude declined to analyze these logs (stop_reason=refusal)."
            )

        if message.parsed_output is None:
            raise RuntimeError("Claude returned no parseable structured output.")

        return message.parsed_output
