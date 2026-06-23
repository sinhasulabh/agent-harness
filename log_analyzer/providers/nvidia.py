"""NVIDIA NIM provider (Kimi K2) — OpenAI-compatible endpoint via the openai SDK.

Mirrors the Claude tool flow, translated to OpenAI's function-calling wire format:
the model first explores the full log through the `grep_logs` / `get_window` /
`count_by_level` tools, then concludes with a JSON object validated against
`LogAnalysis`. NIM has no native schema enforcement, so the conclude step uses
JSON mode (`response_format={"type": "json_object"}`) plus the schema spelled out
in the prompt — the same approach the non-tool version used.
"""

from __future__ import annotations

import json
import logging

from openai import OpenAI

from ..models import LogAnalysis
from ..prompts import SYSTEM_PROMPT, TOOL_SYSTEM_SUFFIX, build_user_prompt, schema_hint
from ..tool_calling import MAX_TOOL_ITERATIONS, to_openai_tools
from ..tools import LogToolkit
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
        self._explore_system = SYSTEM_PROMPT + TOOL_SYSTEM_SUFFIX
        self._tools = to_openai_tools()
        self._conclude_instruction = (
            "Based on your investigation above, respond now with a single JSON "
            "object that conforms to this JSON schema (no markdown, no extra "
            "text):\n" + schema_hint()
        )

    @staticmethod
    def _log_usage(tag: str, completion) -> None:
        usage = getattr(completion, "usage", None)
        if usage is not None:
            log.debug(
                "%s usage: prompt=%s completion=%s total=%s tokens",
                tag,
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

    def analyze(self, log_text: str, toolkit: LogToolkit) -> LogAnalysis:
        log.debug(
            "analyze() start: model=%s sample=%d chars, max_iterations=%d",
            self.model,
            len(log_text),
            MAX_TOOL_ITERATIONS,
        )
        messages: list = [
            {"role": "system", "content": self._explore_system},
            {"role": "user", "content": build_user_prompt(log_text)},
        ]
        log.debug("seeded conversation (system + user); messages=%d", len(messages))

        # Phase 1 — explore: let Kimi call tools to investigate the full log.
        for i in range(MAX_TOOL_ITERATIONS):
            log.debug("[phase1 iter %d] calling chat.completions.create(tools=...) ...", i + 1)
            completion = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.config.max_tokens,
                tools=self._tools,
                messages=messages,
            )
            choice = completion.choices[0]
            msg = choice.message
            self._log_usage(f"[phase1 iter {i + 1}]", completion)
            log.debug(
                "[phase1 iter %d] finish_reason=%s tool_calls=%d content=%r",
                i + 1,
                choice.finish_reason,
                len(msg.tool_calls or []),
                (msg.content or "")[:160],
            )
            # Echo the assistant turn back verbatim (carries the tool_calls).
            messages.append(msg)
            if not msg.tool_calls:
                log.debug("[phase1 iter %d] no tool_calls -> exploration complete", i + 1)
                break
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    log.debug(
                        "[phase1 iter %d] could not parse args for %s: %r",
                        i + 1,
                        tc.function.name,
                        tc.function.arguments,
                    )
                    args = {}
                log.debug(
                    "[phase1 iter %d] tool call %s(%s) id=%s",
                    i + 1,
                    tc.function.name,
                    args,
                    tc.id,
                )
                result = toolkit.dispatch(tc.function.name, args)
                log.debug("[phase1 iter %d] result for %s -> %s", i + 1, tc.id, result)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )
            log.debug("[phase1 iter %d] appended tool result(s); messages=%d", i + 1, len(messages))

        # Phase 2 — conclude: force a schema-conforming JSON object, no tools.
        messages.append({"role": "user", "content": self._conclude_instruction})
        log.debug("[phase2] appended conclude instruction; messages=%d", len(messages))
        log.debug("[phase2] calling chat.completions.create(response_format=json_object) ...")
        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=messages,
        )
        self._log_usage("[phase2]", completion)
        content = completion.choices[0].message.content
        log.debug("[phase2] raw content=%r", (content or "")[:300])
        if not content:
            raise RuntimeError("NVIDIA endpoint returned an empty response.")
        analysis = LogAnalysis.model_validate_json(content)
        log.debug("[phase2] validated LogAnalysis -> returning")
        return analysis
