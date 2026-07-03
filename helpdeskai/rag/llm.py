"""LLM adapters for RAG rewriting and answer generation."""

from __future__ import annotations

import os
from typing import Protocol


class RagLlm(Protocol):
    """Minimal LLM interface used by the RAG pipeline."""

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        """Return a text completion for the provided prompt."""


class MissingAnthropicKeyError(RuntimeError):
    """Raised when the Anthropic API key required for Claude is missing."""


class ClaudeLlm:
    """Anthropic Claude adapter."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise MissingAnthropicKeyError(
                "ANTHROPIC_API_KEY is required to run the real RAG pipeline"
            )
        from anthropic import Anthropic

        self._client = Anthropic(api_key=self.api_key)

    def complete(self, prompt: str, *, model: str, max_tokens: int, temperature: float) -> str:
        """Generate text with Claude Messages API."""
        message = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()
