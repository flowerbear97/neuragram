"""Pluggable embedding providers.

Hierarchy:
    BaseEmbeddingProvider (ABC)
    ├── NullEmbeddingProvider   — returns empty vectors, zero external deps
    ├── LocalEmbeddingProvider  — sentence-transformers, fully offline
    └── OpenAIEmbeddingProvider — OpenAI API
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engram.core.exceptions import BackendNotAvailableError, EmbeddingError


class BaseEmbeddingProvider(ABC):
    """Interface for all embedding providers."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimensionality of the vectors this provider produces."""

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts. Default loops over embed_text."""


class NullEmbeddingProvider(BaseEmbeddingProvider):
    """Returns zero-vectors. Used when embedding is disabled ("none" mode).

    Allows the system to function with keyword-only retrieval.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        return [0.0] * self._dimension

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self._dimension for _ in texts]


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Offline embedding via sentence-transformers.

    Requires: pip install engram[local]
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise BackendNotAvailableError(
                "local",
                "sentence-transformers not installed. Run: pip install engram[local]",
            ) from exc

        self._model = SentenceTransformer(model_name, device=device)
        self._dimension = self._model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    @property
    def dimension(self) -> int:
        return int(self._dimension)

    async def embed_text(self, text: str) -> list[float]:
        try:
            vector = self._model.encode(text, normalize_embeddings=True)
            return vector.tolist()
        except Exception as exc:
            raise EmbeddingError(f"Local embedding failed: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(texts, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        except Exception as exc:
            raise EmbeddingError(f"Local batch embedding failed: {exc}") from exc


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """Embedding via OpenAI API.

    Requires: pip install engram[openai]
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        dimension: int = 1536,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise BackendNotAvailableError(
                "openai",
                "openai package not installed. Run: pip install engram[openai]",
            ) from exc

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        try:
            response = await self._client.embeddings.create(
                input=[text], model=self._model
            )
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        try:
            response = await self._client.embeddings.create(
                input=texts, model=self._model
            )
            return [item.embedding for item in response.data]
        except Exception as exc:
            raise EmbeddingError(f"OpenAI batch embedding failed: {exc}") from exc


def create_embedding_provider(
    provider_name: str,
    dimension: int = 384,
    model: str = "",
    **kwargs: object,
) -> BaseEmbeddingProvider:
    """Factory function to create an embedding provider by name.

    Args:
        provider_name: One of "none", "local", "openai".
        dimension: Vector dimension (used by NullEmbeddingProvider).
        model: Model name/identifier for the provider.
        **kwargs: Additional provider-specific options.
    """
    if provider_name == "none":
        return NullEmbeddingProvider(dimension=dimension)

    if provider_name == "local":
        return LocalEmbeddingProvider(
            model_name=model or "all-MiniLM-L6-v2",
            device=kwargs.get("device"),  # type: ignore[arg-type]
        )

    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            model=model or "text-embedding-3-small",
            api_key=kwargs.get("api_key"),  # type: ignore[arg-type]
            dimension=dimension,
        )

    raise BackendNotAvailableError(
        provider_name, f"Unknown embedding provider: {provider_name}"
    )
