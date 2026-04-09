"""Pluggable LLM providers for intelligent memory processing.

Hierarchy:
    BaseLLMProvider (ABC)
    ├── OpenAILLMProvider   — OpenAI Chat Completions API
    ├── OllamaLLMProvider   — Local Ollama inference
    └── CallableLLMProvider — Wrap any user-supplied function

All providers expose a single ``complete()`` method that takes a system
prompt and user message, returning the model's text response. This keeps
the LLM interface minimal and decoupled from any specific SDK.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from neuragram.core.exceptions import BackendNotAvailableError, EngramError


class LLMError(EngramError):
    """Raised when an LLM call fails."""


@dataclass
class LLMResponse:
    """Structured response from an LLM call."""

    text: str
    model: str = ""
    usage: dict[str, int] | None = None


class BaseLLMProvider(ABC):
    """Interface for all LLM providers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The model identifier being used."""

    @abstractmethod
    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            system_prompt: System-level instruction.
            user_message: The user's input.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens in the response.

        Returns:
            LLMResponse with the model's text output.
        """

    async def complete_json(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Complete and parse the response as JSON.

        Raises LLMError if the response is not valid JSON.
        """
        response = await self.complete(
            system_prompt=system_prompt + "\n\nYou MUST respond with valid JSON only. No markdown, no explanation.",
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (```json and ```)
            lines = [line for line in lines if not line.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"LLM response is not valid JSON: {exc}\nResponse: {text[:500]}"
            ) from exc


class OpenAILLMProvider(BaseLLMProvider):
    """LLM provider using OpenAI Chat Completions API.

    Requires: pip install neuragram[openai]
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise BackendNotAvailableError(
                "openai",
                "openai package not installed. Run: pip install neuragram[openai]",
            ) from exc

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            text = response.choices[0].message.content or ""
            usage = {}
            if response.usage:
                usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            return LLMResponse(text=text, model=self._model, usage=usage)
        except Exception as exc:
            raise LLMError(f"OpenAI completion failed: {exc}") from exc


class OllamaLLMProvider(BaseLLMProvider):
    """LLM provider using local Ollama inference.

    Requires: pip install neuragram[ollama]
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
    ) -> None:
        try:
            import ollama  # noqa: F401
        except ImportError as exc:
            raise BackendNotAvailableError(
                "ollama",
                "ollama package not installed. Run: pip install neuragram[ollama]",
            ) from exc

        from ollama import AsyncClient

        self._client = AsyncClient(host=base_url)
        self._model = model

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        try:
            response = await self._client.chat(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                options={"temperature": temperature, "num_predict": max_tokens},
            )
            text = response["message"]["content"]
            return LLMResponse(text=text, model=self._model)
        except Exception as exc:
            raise LLMError(f"Ollama completion failed: {exc}") from exc


class CallableLLMProvider(BaseLLMProvider):
    """Wrap any async callable as an LLM provider.

    Useful for testing or integrating custom LLM backends.

    Example::

        async def my_llm(system: str, user: str) -> str:
            return "extracted memory"

        provider = CallableLLMProvider(my_llm, model_name="custom")
    """

    def __init__(
        self,
        func: Callable[[str, str], Awaitable[str]],
        model_name: str = "callable",
    ) -> None:
        self._func = func
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    async def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        try:
            text = await self._func(system_prompt, user_message)
            return LLMResponse(text=text, model=self._model_name)
        except Exception as exc:
            raise LLMError(f"Callable LLM failed: {exc}") from exc


def create_llm_provider(
    provider_name: str,
    model: str = "",
    **kwargs: Any,
) -> BaseLLMProvider:
    """Factory function to create an LLM provider by name.

    Args:
        provider_name: One of "openai", "ollama".
        model: Model name/identifier.
        **kwargs: Provider-specific options (api_key, base_url, etc.).
    """
    if provider_name == "openai":
        return OpenAILLMProvider(
            model=model or "gpt-4o-mini",
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
        )

    if provider_name == "ollama":
        return OllamaLLMProvider(
            model=model or "llama3.2",
            base_url=kwargs.get("base_url", "http://localhost:11434"),
        )

    raise BackendNotAvailableError(
        provider_name, f"Unknown LLM provider: {provider_name}"
    )
