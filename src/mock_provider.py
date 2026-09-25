"""Mock provider for Lab 02."""

from __future__ import annotations

from src.models import LLMResponse
from src.provider import BaseProvider


class MockProvider(BaseProvider):
    """Deterministic provider used for testing and local development."""

    def generate(self, prompt: str) -> LLMResponse:
        """Return a deterministic mock response."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            raise ValueError("Prompt cannot be empty.")

        text = (
            f"Mock response for '{normalized_prompt}'. "
            "This deterministic output is intended for testing and local development."
        )
        return LLMResponse(text=text, provider="mock", model="mock-model")
