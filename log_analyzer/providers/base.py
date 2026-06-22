"""The interface every provider implements."""

from __future__ import annotations

import abc
import os

from ..config import Config
from ..models import LogAnalysis


class LLMProvider(abc.ABC):
    """A provider turns a batch of formatted log lines into a `LogAnalysis`."""

    #: Environment variable holding this provider's API key.
    api_key_env: str = ""

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
    def analyze(self, log_text: str) -> LogAnalysis:
        """Send the formatted logs to the model and return a parsed analysis."""
