"""Optional OpenAI provider implementation for Lab 02."""

from __future__ import annotations

from openai import OpenAI

from src.models import LLMResponse
from src.provider import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider implementation backed by the OpenAI Python SDK."""

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        """Generate a response using the OpenAI SDK."""
        if not isinstance(prompt, str):
            raise TypeError("Prompt must be a string.")
        if not self.api_key:
            raise ValueError("OpenAI API key is missing.")
        if not self.model:
            raise ValueError("OpenAI model is missing.")

        try:
            client = OpenAI(api_key=self.api_key)
            response = client.responses.create(model=self.model, input=prompt)
        except Exception as exc:  # pragma: no cover - runtime integration path
            raise RuntimeError("OpenAI provider request failed.") from exc

        text = ""
        if hasattr(response, "output_text") and response.output_text:
            text = response.output_text
        elif hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, list):
                    chunks = []
                    for item in content:
                        text_value = getattr(item, "text", None)
                        if text_value:
                            chunks.append(text_value)
                    text = "".join(chunks)
                elif isinstance(content, str):
                    text = content

        if not text and hasattr(response, "output"):
            for item in response.output:
                if getattr(item, "type", None) == "message":
                    for part in getattr(item, "content", []):
                        text_value = getattr(part, "text", None)
                        if text_value:
                            text = text_value
                            break
                    if text:
                        break

        if not text:
            text = "No response content returned."

        usage = None
        if hasattr(response, "usage") and response.usage is not None:
            usage_obj = response.usage
            if hasattr(usage_obj, "model_dump"):
                usage = usage_obj.model_dump()
            else:
                usage = dict(usage_obj)

        return LLMResponse(text=text, provider="openai", model=self.model, usage=usage)
