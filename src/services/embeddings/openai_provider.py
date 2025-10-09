"""OpenAI embedding provider with batch support."""

import contextlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import trace_function
from src.services.observability.tracking import record_ai_operation
from src.services.utilities.rate_limiter import RateLimitManager

from .base import EmbeddingProvider


if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from src.infrastructure.client_manager import ClientManager

logger = logging.getLogger(__name__)


def _raise_openai_api_key_not_configured() -> None:
    """Raise EmbeddingServiceError for missing OpenAI API key."""
    msg = "OpenAI API key not configured"
    raise EmbeddingServiceError(msg)


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
        client_manager: "ClientManager",
        model_name: str = "text-embedding-3-small",
        dimensions: int | None = None,
        rate_limiter: RateLimitManager | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            client_manager: ClientManager instance for dependency injection
            model_name: Model name (text-embedding-3-small, text-embedding-3-large)
            dimensions: Optional dimensions for text-embedding-3-* models
            rate_limiter: Optional RateLimitManager instance
        """

        super().__init__(model_name)
        self.api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._dimensions = dimensions
        self._initialized = False
        self.rate_limiter = rate_limiter
        self._client_manager = client_manager

        if model_name not in self._model_configs:
            msg = (
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self._model_configs.keys())}"
            )
            raise EmbeddingServiceError(msg)

        # Set dimensions
        config = self._model_configs[model_name]
        if dimensions:
            if dimensions > config["max_dimensions"]:
                msg = (
                    f"Dimensions {dimensions} exceeds max {config['max_dimensions']} "
                    f"for {model_name}"
                )
                raise EmbeddingServiceError(msg)
            self.dimensions = dimensions
        else:
            self.dimensions = config["max_dimensions"]

    @trace_function()
    async def initialize(self) -> None:
        """Initialize OpenAI client using ClientManager."""

        if self._initialized:
            return

        try:
            # Get client from ClientManager
            self._client = await self._client_manager.get_openai_client()

            if self._client is None:
                _raise_openai_api_key_not_configured()

            self._initialized = True
            logger.info("OpenAI client initialized with model %s", self.model_name)
        except Exception as e:
            msg = f"Failed to initialize OpenAI client: {e}"
            raise EmbeddingServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup OpenAI client (delegated to ClientManager)."""
        # Note: ClientManager handles client cleanup, we just reset our reference
        self._client = None
        self._initialized = False

    @trace_function()
    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings with batching and rate limiting.

        Args:
            texts: Text strings to embed (max ~8191 tokens each)
            batch_size: Texts per API call (default: 100)

        Returns:
            List of embedding vectors in same order as input

        Raises:
            EmbeddingServiceError: If not initialized or API call fails
        """

        start = time.perf_counter()
        success = True
        total_tokens = sum(len(text.split()) for text in texts)

        try:
            embeddings = await self._execute_embedding_generation(texts, batch_size)
            return embeddings
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            cost = self._calculate_cost(total_tokens)
            record_ai_operation(
                operation_type="embedding",
                provider="openai",
                model=self.model_name,
                duration_s=duration,
                tokens=total_tokens or None,
                cost_usd=cost or None,
                success=success,
            )

    async def _execute_embedding_generation(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Execute the actual embedding generation."""

        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

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

                # Generate embeddings with rate limiting
                response = await self._generate_embeddings_with_rate_limit(params)

                # Extract embeddings in order
                batch_embeddings = [embedding.embedding for embedding in response.data]
                embeddings.extend(batch_embeddings)

                logger.debug(
                    "Generated embeddings for batch %d to %d (%d texts)",
                    i + 1,
                    min(i + batch_size, len(texts)),
                    len(batch),
                )

        except Exception as e:
            logger.exception("Failed to generate embeddings for %d texts", len(texts))

            # Provide specific error messages based on error type
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg.lower():
                msg = (
                    f"OpenAI rate limit exceeded. Please try again later or reduce "
                    f"batch size. Error: {e}"
                )
                raise EmbeddingServiceError(msg) from e
            if "insufficient_quota" in error_msg.lower():
                msg = (
                    f"OpenAI API quota exceeded. Please check your billing. Error: {e}"
                )
                raise EmbeddingServiceError(msg) from e
            if "invalid_api_key" in error_msg.lower():
                msg = (
                    f"Invalid OpenAI API key. Please check your configuration. "
                    f"Error: {e}"
                )
                raise EmbeddingServiceError(msg) from e
            if "context_length_exceeded" in error_msg.lower():
                msg = (
                    f"Text too long for model "
                    f"{self.model_name}. Max tokens: {self.max_tokens_per_request}. "
                    f"Error: {e}"
                )
                raise EmbeddingServiceError(msg) from e
            msg = f"Failed to generate embeddings: {e}"
            raise EmbeddingServiceError(msg) from e

        return embeddings

    async def _generate_embeddings_with_rate_limit(self, params: dict[str, Any]) -> Any:
        """Generate embeddings with rate limiting.

        Args:
            params: Parameters for embeddings.create

        Returns:
            OpenAI embeddings response
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire("openai")
        return await self._client.embeddings.create(**params)

    @property
    def cost_per_token(self) -> float:
        """Get cost per token."""
        config = self._model_configs[self.model_name]
        return config["cost_per_million"] / 1_000_000

    @property
    def max_tokens_per_request(self) -> int:
        """Get maximum tokens per request."""
        return self._model_configs[self.model_name]["max_tokens"]

    def _calculate_cost(self, token_count: int) -> float:
        """Calculate cost for given token count.

        Args:
            token_count: Number of tokens

        Returns:
            Cost in USD

        """
        return token_count * self.cost_per_token

    async def generate_embeddings_batch_api(
        self, texts: list[str], custom_ids: list[str] | None = None
    ) -> str:
        """Submit embeddings for batch processing with 50% cost reduction.

        Batch API processes within 24 hours with significant cost savings.
        Ideal for large-scale, non-time-critical embedding generation.

        Args:
            texts: Text strings to embed
            custom_ids: Optional IDs for matching results (auto-generated if None)

        Returns:
            Batch job ID for status checking and result retrieval

        Raises:
            EmbeddingServiceError: If not initialized or batch creation fails

        Note:
            Results available within 24 hours with 50% cost savings.
            No rate limits for batch processing.
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

        if not custom_ids:
            custom_ids = [f"text-{i}" for i in range(len(texts))]

        try:
            # Create JSONL content for batch
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as f:
                temp_file = f.name
                for _i, (text, custom_id) in enumerate(
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

                # Ensure all data is written to disk before closing
                f.flush()
                os.fsync(f.fileno())

            # Upload file with rate limiting
            with Path(temp_file).open("rb") as f:
                file_response = await self._upload_file_with_rate_limit(f, "batch")

            # Create batch with rate limiting
            batch_response = await self._create_batch_with_rate_limit(
                file_response.id, "/v1/embeddings", "24h"
            )

            logger.info(
                "Created batch job %s for %d texts", batch_response.id, len(texts)
            )

        except Exception as e:
            msg = f"Failed to create batch job: {e}"
            raise EmbeddingServiceError(msg) from e
        else:
            return batch_response.id
        finally:
            # Clean up temp file
            if "temp_file" in locals() and temp_file and Path(temp_file).exists():
                with contextlib.suppress(OSError):
                    Path(temp_file).unlink()

    async def _upload_file_with_rate_limit(self, file: Any, purpose: str) -> Any:
        """Upload file with rate limiting.

        Args:
            file: File object to upload
            purpose: Purpose of the file ("batch")

        Returns:
            File response from OpenAI
        """

        if self.rate_limiter:
            await self.rate_limiter.acquire("openai")
        return await self._client.files.create(file=file, purpose=purpose)

    async def _create_batch_with_rate_limit(
        self, input_file_id: str, endpoint: str, completion_window: str
    ) -> Any:
        """Create batch with rate limiting.

        Args:
            input_file_id: ID of uploaded file
            endpoint: API endpoint for batch
            completion_window: Time window for completion

        Returns:
            Batch response from OpenAI
        """
        if self.rate_limiter:
            await self.rate_limiter.acquire("openai")
        return await self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
        )
