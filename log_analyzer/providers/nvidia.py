"""NVIDIA NIM provider (Llama 3.3 70B) — OpenAI dialect hooks for the shared agent loop.

The explore→conclude cycle lives in `agent_loop.run_agent_loop`; this class only
translates that loop's hook calls into OpenAI-compatible chat-completions wire
format. NIM has no native schema enforcement, so the final `LogAnalysis` is read
out of the model's text (the system prompt spells out the JSON schema) — the same
single-phase shape the Claude provider uses, just a different dialect.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from ..agent_loop import ToolCall, parse_log_analysis
from ..models import LogAnalysis
from ..prompts import (
    CONCLUDE_NUDGE,
    SYSTEM_PROMPT,
    TOOL_SYSTEM_SUFFIX,
    build_user_prompt,
    final_answer_instruction,
)
from ..tool_calling import to_openai_tools
from .base import ToolCallingProvider

log = logging.getLogger(__name__)

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(ToolCallingProvider):
    api_key_env = "NVIDIA_API_KEY"

    @property
    def name(self) -> str:
        return "nvidia"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.client = OpenAI(
            base_url=NVIDIA_BASE_URL,
            api_key=self._require_key(),
        )
        self._system = SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX + final_answer_instruction()
        self._tools = to_openai_tools()

    # --- agent-loop hooks (OpenAI dialect) ---

    def seed_messages(self, log_text: str) -> list:
        return [
            {"role": "system", "content": self._system},
            {"role": "user", "content": build_user_prompt(log_text)},
        ]

    def call_model(self, messages: list, use_tools: bool):
        kwargs: dict = dict(
            model=self.model,
            max_tokens=self.config.max_tokens,
            messages=messages,
        )
        if use_tools:
            kwargs["tools"] = self._tools
        return self.client.chat.completions.create(**kwargs)

    def usage(self, response) -> tuple[int, int]:
        u = getattr(response, "usage", None)
        if u is None:
            return (0, 0)
        return (u.prompt_tokens, u.completion_tokens)

    def is_refusal(self, response) -> bool:
        return response.choices[0].finish_reason == "content_filter"

    def is_truncated(self, response) -> bool:
        return response.choices[0].finish_reason == "length"

    def assistant_message(self, response):
        # The response message object carries the tool_calls; append it verbatim.
        return response.choices[0].message

    def extract_tool_calls(self, response) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for tc in response.choices[0].message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                log.debug("could not parse args for %s: %r", tc.function.name, tc.function.arguments)
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        return calls

    def format_tool_results(self, results: list[tuple[ToolCall, str]]) -> list:
        # OpenAI wants one `tool` message per tool call.
        return [
            {"role": "tool", "tool_call_id": call.id, "content": output}
            for call, output in results
        ]

    def extract_final(self, response) -> LogAnalysis | None:
        return parse_log_analysis(response.choices[0].message.content)

    def conclude_nudge(self):
        return {"role": "user", "content": CONCLUDE_NUDGE}
