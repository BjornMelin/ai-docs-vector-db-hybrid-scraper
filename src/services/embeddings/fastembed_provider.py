"""FastEmbed provider for local embedding generation."""

import logging
from typing import Any, ClassVar

import numpy as np

from src.services.errors import EmbeddingServiceError
from src.services.monitoring.metrics import get_metrics_registry

from .base import EmbeddingProvider


try:
    from fastembed import SparseTextEmbedding, TextEmbedding
except ImportError:
    TextEmbedding = None
    SparseTextEmbedding = None


logger = logging.getLogger(__name__)


class FastEmbedProvider(EmbeddingProvider):
    """FastEmbed provider for local embeddings."""

    # Model configurations with dimensions and descriptions
    SUPPORTED_MODELS: ClassVar[dict[str, dict[str, Any]]] = {
        "BAAI/bge-small-en-v1.5": {
            "dimensions": 384,
            "description": "Small English model, good balance of speed and quality",
            "max_tokens": 512,
        },
        "BAAI/bge-base-en-v1.5": {
            "dimensions": 768,
            "description": "Base English model, better quality than small",
            "max_tokens": 512,
        },
        "BAAI/bge-large-en-v1.5": {
            "dimensions": 1024,
            "description": "Large English model, best quality",
            "max_tokens": 512,
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "dimensions": 384,
            "description": "Fast and lightweight model",
            "max_tokens": 256,
        },
        "snowflake/snowflake-arctic-embed-xs": {
            "dimensions": 384,
            "description": "Extra small Arctic model",
            "max_tokens": 512,
        },
        "snowflake/snowflake-arctic-embed-s": {
            "dimensions": 384,
            "description": "Small Arctic model",
            "max_tokens": 512,
        },
        "snowflake/snowflake-arctic-embed-m": {
            "dimensions": 768,
            "description": "Medium Arctic model",
            "max_tokens": 512,
        },
        "snowflake/snowflake-arctic-embed-l": {
            "dimensions": 1024,
            "description": "Large Arctic model",
            "max_tokens": 512,
        },
        "jinaai/jina-embeddings-v2-small-en": {
            "dimensions": 512,
            "description": "Jina small model, supports longer context",
            "max_tokens": 8192,
        },
        "jinaai/jina-embeddings-v2-base-en": {
            "dimensions": 768,
            "description": "Jina base model, supports longer context",
            "max_tokens": 8192,
        },
    }

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize FastEmbed provider.

        Args:
            model_name: Name of the FastEmbed model

        """
        if model_name not in self.SUPPORTED_MODELS:
            msg = (
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self.SUPPORTED_MODELS.keys())}"
            )
            raise EmbeddingServiceError(msg)

        super().__init__(model_name)
        self._model: TextEmbedding | None = None
        self._sparse_model: SparseTextEmbedding | None = None
        self._initialized = False

        # Set model configuration
        config = self.SUPPORTED_MODELS[model_name]
        self.dimensions = config["dimensions"]
        self._max_tokens = config["max_tokens"]
        self._description = config["description"]

        # Default sparse model for hybrid search
        self._sparse_model_name = "prithvida/Splade_PP_en_v1"

        # Initialize metrics if available
        try:
            self.metrics_registry = get_metrics_registry()
        except (AttributeError, ImportError, RuntimeError, TypeError):
            logger.warning(
                "Metrics registry not available - embedding monitoring disabled"
            )
            self.metrics_registry = None

    async def initialize(self) -> None:
        """Initialize FastEmbed model."""
        if self._initialized:
            return

        try:
            # FastEmbed is synchronous, so we initialize it here
            self._model = TextEmbedding(self.model_name)
            self._initialized = True
            logger.info(
                "FastEmbed initialized with model %s (%s)",
                self.model_name,
                self._description,
            )
        except Exception as e:
            msg = f"Failed to initialize FastEmbed: {e}"
            raise EmbeddingServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup FastEmbed resources."""
        self._model = None
        self._initialized = False
        logger.info("FastEmbed resources cleaned up")

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings locally.

        Args:
            texts: List of texts to embed
            batch_size: Batch size (ignored, FastEmbed handles batching)

        Returns:
            List of embedding vectors

        """
        # Monitor embedding generation
        if self.metrics_registry:
            decorator = self.metrics_registry.monitor_embedding_generation(
                provider="fastembed", model=self.model_name
            )

            async def _monitored_generation():
                return await self._execute_embedding_generation(texts, batch_size)

            return await decorator(_monitored_generation)()
        return await self._execute_embedding_generation(texts, batch_size)

    async def _execute_embedding_generation(
        self, texts: list[str], _batch_size: int | None = None
    ) -> list[list[float]]:
        """Execute the actual embedding generation."""
        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

        if not texts:
            return []

        try:
            # FastEmbed handles batching internally
            embeddings_iter = self._model.embed(texts)

            # Convert to list of lists
            embeddings = []
            for embedding in embeddings_iter:
                if isinstance(embedding, np.ndarray):
                    embeddings.append(embedding.tolist())
                else:
                    embeddings.append(list(embedding))

            logger.debug("Generated %d embeddings locally", len(embeddings))

        except Exception as e:
            msg = f"Failed to generate embeddings: {e}"
            raise EmbeddingServiceError(msg) from e

        else:
            return embeddings

    async def generate_sparse_embeddings(
        self, texts: list[str]
    ) -> list[dict[str, Any]]:
        """Generate sparse embeddings for hybrid search.

        Args:
            texts: List of texts to embed

        Returns:
            List of sparse embeddings with indices and values

        """
        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

        if SparseTextEmbedding is None:
            msg = "Sparse embedding support not available"
            raise EmbeddingServiceError(msg)

        try:
            # Initialize sparse model if needed
            if self._sparse_model is None:
                self._sparse_model = SparseTextEmbedding(self._sparse_model_name)
                logger.info("Initialized sparse model: %s", self._sparse_model_name)

            # Generate sparse embeddings
            sparse_embeddings = []
            for result in self._sparse_model.embed(texts):
                sparse_data = {
                    "indices": result.indices.tolist(),
                    "values": result.values.tolist(),
                }
                sparse_embeddings.append(sparse_data)

        except Exception as e:
            msg = f"Sparse embedding generation failed: {e}"
            raise EmbeddingServiceError(msg) from e

        else:
            return sparse_embeddings

    @property
    def cost_per_token(self) -> float:
        """Get cost per token (0 for local models)."""
        return 0.0

    @property
    def max_tokens_per_request(self) -> int:
        """Get maximum tokens per request."""
        return self._max_tokens

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available FastEmbed models.

        Returns:
            List of model names

        """
        return list(cls.SUPPORTED_MODELS.keys())

    @classmethod
    def get_model_info(cls, model_name: str) -> dict:
        """Get information about a specific model.

        Args:
            model_name: Model name

        Returns:
            Model information dictionary

        """
        if model_name not in cls.SUPPORTED_MODELS:
            msg = f"Unknown model: {model_name}"
            raise ValueError(msg)
        return cls.SUPPORTED_MODELS[model_name]
