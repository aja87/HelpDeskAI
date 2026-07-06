"""LLM adapters for RAG rewriting and answer generation."""

from __future__ import annotations

import os
import time
from typing import Protocol


class RagLlm(Protocol):
    """Minimal LLM interface used by the RAG pipeline."""

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        """Return a text completion for the provided prompt."""


class MissingAnthropicKeyError(RuntimeError):
    """Raised when the Anthropic API key required for Claude is missing."""


class TransientLlmError(RuntimeError):
    """Raised when the LLM provider is temporarily unavailable."""


class ClaudeLlm:
    """Anthropic Claude adapter."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_retries: int = 2,
        base_delay_s: float = 0.8,
    ) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingAnthropicKeyError(
                "ANTHROPIC_API_KEY is required to run the real RAG pipeline"
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key)
        self.max_retries = max_retries
        self.base_delay_s = base_delay_s

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        """Generate text with Claude Messages API."""
        for attempt in range(self.max_retries + 1):
            try:
                message = self._client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                break
            except Exception as exc:
                if not _is_transient_provider_error(exc):
                    raise
                if attempt >= self.max_retries:
                    raise TransientLlmError(_transient_error_message(exc)) from exc
                time.sleep(self.base_delay_s * (2**attempt))
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()


def _is_transient_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {408, 409, 429, 500, 502, 503, 504, 529}:
        return True
    name = type(exc).__name__.lower()
    transient_names = (
        "apiconnectionerror",
        "apitimerouterror",
        "apitomeouterror",
        "apitimeouterror",
        "internalservererror",
        "ratelimiterror",
    )
    if any(item in name for item in transient_names):
        return True
    message = str(exc).lower()
    return "overload" in message or "overloaded" in message or "temporarily unavailable" in message


def _transient_error_message(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None)
    if status_code:
        return f"LLM provider temporarily unavailable after retries (status {status_code})."
    return "LLM provider temporarily unavailable after retries."
