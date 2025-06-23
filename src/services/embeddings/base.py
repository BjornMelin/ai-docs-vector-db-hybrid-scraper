import typing

"""Base embedding provider interface."""

from abc import ABC
from abc import abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs):
        """Initialize embedding provider.

        Args:
            model_name: Name of the embedding model
            **kwargs: Additional provider-specific arguments
        """
        self.model_name = model_name
        self.dimensions: int = 0

    @abstractmethod
    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings for texts.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size for processing

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider."""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup provider resources."""
        pass

    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        """Get cost per token for this provider."""
        pass

    @property
    @abstractmethod
    def max_tokens_per_request(self) -> int:
        """Get maximum tokens per request."""
        pass
