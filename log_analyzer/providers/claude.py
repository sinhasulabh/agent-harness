"""Claude provider — Anthropic SDK with a tool-calling loop + structured output.

Claude runs in **tool mode**: it first explores the full log file through the
`grep_logs` / `get_window` / `count_by_level` tools, then concludes with a
schema-validated `LogAnalysis`. The two phases are deliberate — a plain tool loop
(`messages.create`) for exploration, then `messages.parse` for the structured
answer — which keeps each step on a well-supported SDK path. The NVIDIA provider
uses the same two-phase template in OpenAI's function-calling dialect.
"""

from __future__ import annotations

import logging

import anthropic

from ..models import LogAnalysis
from ..prompts import SYSTEM_PROMPT, TOOL_SYSTEM_SUFFIX, build_user_prompt
from ..tool_calling import MAX_TOOL_ITERATIONS, to_anthropic_tools
from ..tools import LogToolkit
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
        self._system = SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX
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

    @staticmethod
    def _tool_results(content, toolkit: LogToolkit) -> list[dict]:
        """Run every tool_use block in `content` and return tool_result blocks."""
        return [
            {
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": toolkit.dispatch(block.name, block.input),
            }
            for block in content
            if block.type == "tool_use"
        ]

    def analyze(self, log_text: str, toolkit: LogToolkit) -> LogAnalysis:
        log.debug(
            "analyze() start: sample=%d chars, max_iterations=%d",
            len(log_text),
            MAX_TOOL_ITERATIONS,
        )
        messages: list[dict] = [
            {"role": "user", "content": build_user_prompt(log_text)}
        ]
        log.debug("seeded conversation with user prompt; messages=%d", len(messages))

        # Phase 1 — explore: let Claude call tools to investigate the full log.
        for i in range(MAX_TOOL_ITERATIONS):
            log.debug("[phase1 iter %d] calling messages.create() ...", i + 1)
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                **self._thinking,
                system=self._system,
                tools=self._tools,
                messages=messages,
            )
            log.debug(
                "[phase1 iter %d] stop_reason=%s blocks=%s",
                i + 1,
                response.stop_reason,
                [b.type for b in response.content],
            )
            if response.stop_reason == "refusal":
                log.debug("[phase1 iter %d] refusal -> raising", i + 1)
                raise RuntimeError(
                    "Claude declined to analyze these logs (stop_reason=refusal)."
                )
            # Echo the assistant turn back verbatim (thinking blocks included).
            messages.append({"role": "assistant", "content": response.content})
            log.debug("[phase1 iter %d] appended assistant turn; messages=%d", i + 1, len(messages))
            if response.stop_reason != "tool_use":
                log.debug("[phase1 iter %d] no tool_use -> exploration complete", i + 1)
                break
            tool_calls = [
                (b.name, b.input) for b in response.content if b.type == "tool_use"
            ]
            log.debug("[phase1 iter %d] tool calls: %s", i + 1, tool_calls)
            tool_results = self._tool_results(response.content, toolkit)
            for tr in tool_results:
                log.debug(
                    "[phase1 iter %d] result for %s -> %s",
                    i + 1,
                    tr["tool_use_id"],
                    tr["content"],
                )
            messages.append({"role": "user", "content": tool_results})
            log.debug(
                "[phase1 iter %d] appended %d tool result(s); messages=%d",
                i + 1,
                len(tool_results),
                len(messages),
            )

        # Phase 2 — conclude: ask for the structured analysis. Allow one stray
        # tool round in case the model wants to look once more before answering.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Based on your investigation above, provide your final "
                    "structured analysis now. Do not call any more tools."
                ),
            }
        )
        log.debug("[phase2] appended conclude nudge; messages=%d", len(messages))
        for j in range(2):
            log.debug("[phase2 attempt %d] calling messages.parse() ...", j + 1)
            message = self.client.messages.parse(
                model=self.model,
                max_tokens=self.config.max_tokens,
                **self._thinking,
                system=self._system,
                tools=self._tools,
                messages=messages,
                output_format=LogAnalysis,
            )
            log.debug(
                "[phase2 attempt %d] stop_reason=%s parsed=%s",
                j + 1,
                message.stop_reason,
                message.parsed_output is not None,
            )
            if message.stop_reason == "refusal":
                log.debug("[phase2 attempt %d] refusal -> raising", j + 1)
                raise RuntimeError(
                    "Claude declined to analyze these logs (stop_reason=refusal)."
                )
            if message.parsed_output is not None:
                log.debug("[phase2 attempt %d] got structured analysis -> returning", j + 1)
                return message.parsed_output
            # The model asked for another tool instead of concluding; satisfy it.
            tool_results = self._tool_results(message.content, toolkit)
            log.debug(
                "[phase2 attempt %d] no structured output; stray tool calls=%d",
                j + 1,
                len(tool_results),
            )
            if not tool_results:
                break
            messages.append({"role": "assistant", "content": message.content})
            messages.append({"role": "user", "content": tool_results})

        log.debug("exhausted attempts -> raising no-parseable-output")
        raise RuntimeError("Claude returned no parseable structured output.")
