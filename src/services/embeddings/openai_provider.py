"""OpenAI embedding provider with batch support."""

import logging
from typing import ClassVar

from openai import AsyncOpenAI

from ..errors import EmbeddingServiceError
from .base import EmbeddingProvider

logger = logging.getLogger(__name__)


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with batch processing."""

    _model_configs: ClassVar[dict[str, dict[str, int | float]]] = {
        "text-embedding-3-small": {
            "max_dimensions": 1536,
            "cost_per_million": 0.02,  # $0.02 per 1M tokens
            "max_tokens": 8191,
        },
        "text-embedding-3-large": {
            "max_dimensions": 3072,
            "cost_per_million": 0.13,  # $0.13 per 1M tokens
            "max_tokens": 8191,
        },
        "text-embedding-ada-002": {
            "max_dimensions": 1536,
            "cost_per_million": 0.10,  # $0.10 per 1M tokens
            "max_tokens": 8191,
        },
    }

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Model name (text-embedding-3-small, text-embedding-3-large)
            dimensions: Optional dimensions for text-embedding-3-* models
        """
        super().__init__(model_name)
        self.api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._dimensions = dimensions
        self._initialized = False

        if model_name not in self._model_configs:
            raise EmbeddingServiceError(
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self._model_configs.keys())}"
            )

        # Set dimensions
        config = self._model_configs[model_name]
        if dimensions:
            if dimensions > config["max_dimensions"]:
                raise EmbeddingServiceError(
                    f"Dimensions {dimensions} exceeds max {config['max_dimensions']} "
                    f"for {model_name}"
                )
            self.dimensions = dimensions
        else:
            self.dimensions = config["max_dimensions"]

    async def initialize(self) -> None:
        """Initialize OpenAI client."""
        if self._initialized:
            return

        try:
            self._client = AsyncOpenAI(api_key=self.api_key)
            self._initialized = True
            logger.info(f"OpenAI client initialized with model {self.model_name}")
        except Exception as e:
            raise EmbeddingServiceError(f"Failed to initialize OpenAI client: {e}")

    async def cleanup(self) -> None:
        """Cleanup OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None
            self._initialized = False
            logger.info("OpenAI client closed")

    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings with batching.

        Args:
            texts: List of texts to embed
            batch_size: Batch size (default: 100)

        Returns:
            List of embedding vectors
        """
        if not self._initialized:
            raise EmbeddingServiceError("Provider not initialized")

        if not texts:
            return []

        batch_size = batch_size or 100
        embeddings = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]

                # Build request parameters
                params = {
                    "input": batch,
                    "model": self.model_name,
                }

                # Add dimensions for text-embedding-3-* models
                if self.model_name.startswith("text-embedding-3-") and self._dimensions:
                    params["dimensions"] = self._dimensions

                # Generate embeddings
                response = await self._client.embeddings.create(**params)

                # Extract embeddings in order
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)

                logger.debug(
                    f"Generated embeddings for batch {i // batch_size + 1} "
                    f"({len(batch)} texts)"
                )

            return embeddings

        except Exception as e:
            raise EmbeddingServiceError(f"Failed to generate embeddings: {e}")

    @property
    def cost_per_token(self) -> float:
        """Get cost per token."""
        config = self._model_configs[self.model_name]
        return config["cost_per_million"] / 1_000_000

    @property
    def max_tokens_per_request(self) -> int:
        """Get maximum tokens per request."""
        return self._model_configs[self.model_name]["max_tokens"]

    async def generate_embeddings_batch_api(
        self, texts: list[str], custom_ids: list[str] | None = None
    ) -> str:
        """Generate embeddings using OpenAI Batch API for 50% cost savings.

        Note: This is for asynchronous batch processing with 24-hour completion.
        For immediate results, use generate_embeddings() instead.

        Args:
            texts: List of texts to embed
            custom_ids: Optional custom IDs for each text

        Returns:
            Batch job ID for status checking
        """
        if not self._initialized:
            raise EmbeddingServiceError("Provider not initialized")

        if not custom_ids:
            custom_ids = [f"text-{i}" for i in range(len(texts))]

        try:
            # Create JSONL content for batch
            import json
            import tempfile

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                for i, (text, custom_id) in enumerate(
                    zip(texts, custom_ids, strict=False)
                ):
                    request = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/embeddings",
                        "body": {
                            "model": self.model_name,
                            "input": text,
                        },
                    }

                    if (
                        self.model_name.startswith("text-embedding-3-")
                        and self._dimensions
                    ):
                        request["body"]["dimensions"] = self._dimensions

                    f.write(json.dumps(request) + "\n")

                temp_file = f.name

            # Upload file
            with open(temp_file, "rb") as f:
                file_response = await self._client.files.create(file=f, purpose="batch")

            # Create batch
            batch_response = await self._client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/embeddings",
                completion_window="24h",
            )

            logger.info(f"Created batch job {batch_response.id} for {len(texts)} texts")
            return batch_response.id

        except Exception as e:
            raise EmbeddingServiceError(f"Failed to create batch job: {e}")
        finally:
            # Clean up temp file
            import os

            if "temp_file" in locals():
                os.unlink(temp_file)
