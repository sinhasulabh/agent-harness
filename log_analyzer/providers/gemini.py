"""Gemini provider — google-genai SDK with a native response schema."""

from __future__ import annotations

from google import genai
from google.genai import types

from ..agent_loop import LoopOutcome, StopReason
from ..models import LogAnalysis
from ..prompts import SYSTEM_PROMPT, build_user_prompt
from ..tools import LogToolkit
from .base import LLMProvider


class GeminiProvider(LLMProvider):
    api_key_env = "GEMINI_API_KEY"

    @property
    def name(self) -> str:
        return "gemini"

    def __init__(self, config) -> None:
        super().__init__(config)
        self.client = genai.Client(api_key=self._require_key())

    def analyze(self, log_text: str, toolkit: LogToolkit) -> LoopOutcome:
        # `toolkit` is unused: this provider is plain (no tool loop), so it
        # analyzes only the sampled lines and returns a single-step outcome.
        # Passing the Pydantic model as response_schema makes Gemini return JSON
        # matching the schema; `response.parsed` is a LogAnalysis instance.
        response = self.client.models.generate_content(
            model=self.model,
            contents=build_user_prompt(log_text),
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=self.config.max_tokens,
                response_mime_type="application/json",
                response_schema=LogAnalysis,
            ),
        )

        parsed = response.parsed
        if not isinstance(parsed, LogAnalysis):
            # Fall back to validating the raw JSON text if the SDK didn't parse it.
            if response.text:
                parsed = LogAnalysis.model_validate_json(response.text)
            else:
                raise RuntimeError("Gemini returned no parseable structured output.")

        meta = getattr(response, "usage_metadata", None)
        tokens = getattr(meta, "total_token_count", 0) or 0
        return LoopOutcome(parsed, StopReason.COMPLETED, steps_used=1, tokens_used=tokens)
