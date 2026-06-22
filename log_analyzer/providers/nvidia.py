"""NVIDIA NIM provider (Kimi K2) — OpenAI-compatible endpoint via the openai SDK."""

from __future__ import annotations

from openai import OpenAI

from ..models import LogAnalysis
from ..prompts import SYSTEM_PROMPT, build_user_prompt, schema_hint
from .base import LLMProvider

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"


class NvidiaProvider(LLMProvider):
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

    def analyze(self, log_text: str) -> LogAnalysis:
        # NIM exposes an OpenAI-compatible API. We request JSON mode and spell out
        # the target schema in the prompt, then validate the response ourselves.
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            "Respond with a single JSON object that conforms to this JSON schema "
            "(no markdown, no extra text):\n"
            f"{schema_hint()}"
        )

        completion = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": build_user_prompt(log_text)},
            ],
        )

        content = completion.choices[0].message.content
        if not content:
            raise RuntimeError("NVIDIA endpoint returned an empty response.")
        return LogAnalysis.model_validate_json(content)
