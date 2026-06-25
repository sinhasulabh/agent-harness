"""Claude provider — Anthropic dialect hooks for the shared agent loop.

The explore→conclude cycle lives in `agent_loop.run_agent_loop`; this class only
translates that loop's hook calls into Anthropic Messages-API wire format. The
final `LogAnalysis` is read out of the model's own text (the tool-mode system
prompt instructs it to emit a bare JSON object), so the loop stays single-phase
and identical across providers.
"""

from __future__ import annotations

import logging

import anthropic

from ..agent_loop import ToolCall, parse_log_analysis
from ..models import LogAnalysis
from ..prompts import (
    CONCLUDE_NUDGE,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_SUFFIX,
    build_user_prompt,
    final_answer_instruction,
)
from ..tool_calling import to_anthropic_tools
from .base import ToolCallingProvider

log = logging.getLogger(__name__)


class ClaudeProvider(ToolCallingProvider):
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
        self._system = SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX + final_answer_instruction()
        self._tools = to_anthropic_tools()
        # Adaptive thinking exists on the 4.6+ Opus/Sonnet family and Fable, but
        # is rejected (400) by Haiku 4.5 and older models. Ask the Models API
        # rather than hardcoding a model list, so swapping models stays safe.
        self._thinking = self._thinking_kwargs()
        log.debug("thinking config for %s: %s", self.model, self._thinking or "off")

    def _thinking_kwargs(self) -> dict:
        """Return ``{"thinking": {"type": "adaptive"}}`` if the model supports it, else ``{}``.

        Models metadata calls are free (no tokens billed). On any error we fall
        back to omitting thinking, which every model accepts.
        """
        try:
            caps = self.client.models.retrieve(self.model).capabilities
            if caps.thinking.types.adaptive.supported:
                return {"thinking": {"type": "adaptive"}}
        except Exception:  # noqa: BLE001 - network/shape change -> safe default
            pass
        return {}

    # --- agent-loop hooks (Anthropic dialect) ---

    def seed_messages(self, log_text: str) -> list:
        return [{"role": "user", "content": build_user_prompt(log_text)}]

    def call_model(self, messages: list, use_tools: bool):
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.config.max_tokens,
            system=self._system,
            messages=messages,
            **self._thinking,
        )
        if use_tools:
            kwargs["tools"] = self._tools
        return self.client.messages.create(**kwargs)

    def usage(self, response) -> tuple[int, int]:
        u = response.usage
        return (u.input_tokens, u.output_tokens)

    def is_refusal(self, response) -> bool:
        return response.stop_reason == "refusal"

    def is_truncated(self, response) -> bool:
        return response.stop_reason == "max_tokens"

    def assistant_message(self, response):
        # Echo the assistant turn back verbatim (thinking blocks included).
        return {"role": "assistant", "content": response.content}

    def extract_tool_calls(self, response) -> list[ToolCall]:
        return [
            ToolCall(id=b.id, name=b.name, args=b.input)
            for b in response.content
            if b.type == "tool_use"
        ]

    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list:
        # Anthropic wants every tool_result for a turn in ONE user message.
        blocks = [
            {"type": "tool_result", "tool_use_id": call.id, "content": output}
            for call, output in results
        ]
        return [{"role": "user", "content": blocks}]

    def extract_final(self, response) -> LogAnalysis | None:
        text = "".join(b.text for b in response.content if b.type == "text")
        return parse_log_analysis(text)

    def conclude_nudge(self):
        return {"role": "user", "content": CONCLUDE_NUDGE}
