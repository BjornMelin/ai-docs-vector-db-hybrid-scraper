"""OpenAI embedding provider with batch support."""

import contextlib
import json
import logging
import os
import tempfile
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict

import tiktoken

from src.services.errors import EmbeddingServiceError
from src.services.observability.tracing import trace_function
from src.services.observability.tracking import record_ai_operation

from .base import EmbeddingProvider


if TYPE_CHECKING:
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


def _raise_openai_api_key_not_configured() -> None:
    """Raise EmbeddingServiceError for missing OpenAI API key."""
    msg = "OpenAI API key not configured"
    raise EmbeddingServiceError(msg)


class _SimpleTokenEncoder:
    """Fallback encoder that approximates token counts by character length."""

    def encode(self, text: str) -> list[int]:
        return [1] * len(text)


class ModelConfig(TypedDict):
    """Model configuration metadata."""

    max_dimensions: int
    cost_per_million: float
    max_tokens: int


class TokenSummary(TypedDict):
    """Token usage summary for embedding calls."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider with batch processing."""

    _model_configs: ClassVar[dict[str, ModelConfig]] = {
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
        *,
        client: "AsyncOpenAI | None" = None,
        client_factory: (Callable[[], Awaitable["AsyncOpenAI"]] | None) = None,
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            model_name: Model name (text-embedding-3-small, text-embedding-3-large)
            dimensions: Optional dimensions for text-embedding-3-* models
            client: Optional preconfigured AsyncOpenAI client instance
            client_factory: Optional factory returning an AsyncOpenAI client
        """

        super().__init__(model_name)
        self.api_key = api_key
        self._client: AsyncOpenAI | None = None
        self._dimensions = dimensions
        self._initialized = False
        self._client_override = client
        self._client_factory = client_factory
        self._token_encoder: Any | None = None

        if model_name not in self._model_configs:
            msg = (
                f"Unsupported model: {model_name}. "
                f"Supported models: {list(self._model_configs.keys())}"
            )
            raise EmbeddingServiceError(msg)

        # Set dimensions
        config: ModelConfig = self._model_configs[model_name]
        if dimensions is not None:
            if dimensions > config["max_dimensions"]:
                msg = (
                    f"Dimensions {dimensions} exceeds max {config['max_dimensions']} "
                    f"for {model_name}"
                )
                raise EmbeddingServiceError(msg)
            self.dimensions = int(dimensions)
        else:
            self.dimensions = config["max_dimensions"]

    @trace_function()
    async def initialize(self) -> None:
        """Initialize OpenAI client using the DI-supplied factory or API key."""

        if self._initialized:
            return

        try:
            if self._client_override is not None and self._client is None:
                self._client = self._client_override
            elif self._client_factory is not None and self._client is None:
                self._client = await self._client_factory()
            elif self._client is None:
                if not self.api_key:
                    _raise_openai_api_key_not_configured()
                # Create a standalone client as a fallback (main path uses DI)
                from openai import AsyncOpenAI  # local import to avoid mandatory dep

                self._client = AsyncOpenAI(api_key=self.api_key)

            if self._client is None:
                _raise_openai_api_key_not_configured()

            self._initialized = True
            logger.info("OpenAI client initialized with model %s", self.model_name)
        except Exception as e:
            msg = f"Failed to initialize OpenAI client: {e}"
            raise EmbeddingServiceError(msg) from e

    async def cleanup(self) -> None:
        """Cleanup OpenAI client by clearing the cached instance."""
        self._client = None
        self._initialized = False

    @trace_function()
    async def generate_embeddings(
        self, texts: list[str], batch_size: int | None = None
    ) -> list[list[float]]:
        """Generate embeddings with batching support.

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
        token_summary: TokenSummary = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        try:
            embeddings, token_summary = await self._execute_embedding_generation(
                texts, batch_size
            )
            return embeddings
        except Exception:
            success = False
            raise
        finally:
            duration = time.perf_counter() - start
            total_tokens = token_summary["total_tokens"]
            cost = self._calculate_cost(total_tokens) if total_tokens else 0.0
            record_ai_operation(
                operation_type="embedding",
                provider="openai",
                model=self.model_name,
                duration_s=duration,
                tokens=total_tokens or None,
                cost_usd=cost if total_tokens else None,
                success=success,
                prompt_tokens=token_summary["prompt_tokens"] or None,
                completion_tokens=token_summary["completion_tokens"] or None,
            )

    async def _execute_embedding_generation(
        self, texts: list[str], batch_size: int | None = None
    ) -> tuple[list[list[float]], TokenSummary]:
        """Execute the actual embedding generation.

        Args:
            texts: Text payloads to embed.
            batch_size: Maximum number of inputs per OpenAI request.

        Returns:
            Tuple containing embeddings and the aggregated token usage summary.

        Raises:
            EmbeddingServiceError: If the provider is not initialized or the
            OpenAI client is unavailable.
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

        if not texts:
            return [], {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        batch_size = batch_size or 100
        embeddings: list[list[float]] = []
        token_summary: TokenSummary = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        if self._client is None:
            msg = "OpenAI client not initialized"
            raise EmbeddingServiceError(msg)

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

                # Generate embeddings request
                response = await self._send_embedding_request(params)

                # Extract embeddings in order
                batch_embeddings = [embedding.embedding for embedding in response.data]
                if len(batch_embeddings) != len(batch):
                    if len(batch_embeddings) == 1 and len(batch) > 1:
                        logger.warning(
                            "OpenAI returned %d embedding for %d inputs; "
                            "replicating result to match batch size.",
                            len(batch_embeddings),
                            len(batch),
                        )
                        batch_embeddings = batch_embeddings * len(batch)
                    else:
                        msg = (
                            "Mismatch between requested texts and returned embeddings: "
                            f"{len(batch)} requested vs {len(batch_embeddings)} "
                            "returned"
                        )
                        raise EmbeddingServiceError(msg)
                embeddings.extend(batch_embeddings)

                usage = getattr(response, "usage", None)
                self._update_token_summary(token_summary, usage, batch)

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

        return embeddings, token_summary

    async def _send_embedding_request(self, params: dict[str, Any]) -> Any:
        """Submit the embeddings request to OpenAI.

        Args:
            params: Parameters for embeddings.create

        Returns:
            OpenAI embeddings response
        """

        if self._client is None:
            msg = "OpenAI client not initialized"
            raise EmbeddingServiceError(msg)

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

    def _get_token_encoder(self) -> Any:
        """Return a cached tiktoken encoder for the configured model.

        Returns:
            The tokenizer capable of encoding inputs for the configured model.
        """

        if self._token_encoder is None:
            try:
                self._token_encoder = tiktoken.encoding_for_model(self.model_name)
            except (KeyError, AttributeError):
                logger.debug(
                    "Falling back to cl100k_base tokenizer for model %s",
                    self.model_name,
                )
                try:
                    self._token_encoder = tiktoken.get_encoding("cl100k_base")
                except AttributeError:
                    logger.debug(
                        "tiktoken.get_encoding unavailable; using simple length encoder"
                    )
                    self._token_encoder = _SimpleTokenEncoder()
        return self._token_encoder

    def _count_tokens(self, texts: list[str]) -> int:
        """Estimate token usage for a batch of texts.

        Args:
            texts: Text payloads to tokenize.

        Returns:
            Estimated token count based on the configured encoder.
        """

        if not texts:
            return 0
        encoder = self._get_token_encoder()
        return sum(len(encoder.encode(text)) for text in texts)

    def _update_token_summary(
        self,
        summary: TokenSummary,
        usage: Any,
        batch: list[str],
    ) -> None:
        """Update aggregated token usage based on an API response.

        Args:
            summary: Mutable token summary accumulator.
            usage: Usage payload returned by the OpenAI SDK (may be absent).
            batch: Text batch corresponding to the current response.
        """

        if usage:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)

            if prompt is not None:
                summary["prompt_tokens"] += int(prompt)
            if completion is not None:
                summary["completion_tokens"] += int(completion)
            if total is not None:
                summary["total_tokens"] += int(total)
            else:
                aggregate_total = 0
                if prompt is not None:
                    aggregate_total += int(prompt)
                if completion is not None:
                    aggregate_total += int(completion)
                summary["total_tokens"] += aggregate_total
            return

        fallback_tokens = self._count_tokens(batch)
        summary["prompt_tokens"] += fallback_tokens
        summary["total_tokens"] += fallback_tokens

    @trace_function()
    async def generate_embeddings_batch_api(
        self, texts: list[str], custom_ids: list[str] | None = None
    ) -> str:
        """Submit embeddings for processing via the OpenAI Batch API.

        Args:
            texts: Text payloads to schedule for asynchronous embedding.
            custom_ids: Optional identifiers to include alongside each payload.
                When ``None``, sequential identifiers are generated automatically.

        Returns:
            The identifier returned by the OpenAI Batch API.

        Raises:
            EmbeddingServiceError: If the provider is not initialized or the
            custom ID configuration does not match the number of texts.
        """

        if not self._initialized:
            msg = "Provider not initialized"
            raise EmbeddingServiceError(msg)

        if custom_ids is not None and len(custom_ids) != len(texts):
            msg = (
                "Custom ID list must match the number of input texts; "
                f"received {len(custom_ids)} IDs for {len(texts)} texts"
            )
            raise EmbeddingServiceError(msg)

        start = time.perf_counter()
        success = False
        custom_ids_provided = custom_ids is not None

        if not custom_ids:
            custom_ids = [f"text-{i}" for i in range(len(texts))]

        try:
            batch_id = await self._submit_batch_job(texts, custom_ids)
            success = True
            return batch_id
        finally:
            duration = time.perf_counter() - start
            token_estimate = self._count_tokens(texts)
            record_ai_operation(
                operation_type="embedding_batch",
                provider="openai",
                model=self.model_name,
                duration_s=duration,
                tokens=token_estimate or None,
                cost_usd=self._calculate_cost(token_estimate)
                if token_estimate
                else None,
                success=success,
                prompt_tokens=token_estimate or None,
                completion_tokens=None,
                attributes={
                    "gen_ai.request.batch_size": len(texts),
                    "gen_ai.request.custom_ids_provided": custom_ids_provided,
                },
            )

    async def _upload_file(self, file: Any, purpose: Literal["batch"]) -> Any:
        """Upload a file to OpenAI."""

        if self._client is None:
            msg = "OpenAI client not initialized"
            raise EmbeddingServiceError(msg)

        return await self._client.files.create(file=file, purpose=purpose)

    async def _create_batch(
        self,
        input_file_id: str,
        endpoint: Literal["/v1/embeddings"],
        completion_window: Literal["24h"],
    ) -> Any:
        """Create a batch embedding job."""

        if self._client is None:
            msg = "OpenAI client not initialized"
            raise EmbeddingServiceError(msg)

        return await self._client.batches.create(
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=completion_window,
        )

    async def _submit_batch_job(self, texts: list[str], custom_ids: list[str]) -> str:
        """Create the batch payload, upload it, and submit the job.

        Args:
            texts: Text payloads included in the batch.
            custom_ids: Identifiers paired with each text entry.

        Returns:
            The identifier returned by the OpenAI Batch API.

        Raises:
            EmbeddingServiceError: If the payload cannot be uploaded or the
            batch submission fails.
        """

        temp_file: str | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".jsonl", delete=False
            ) as handle:
                temp_file = handle.name
                for text, custom_id in zip(texts, custom_ids, strict=True):
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
                    handle.write(json.dumps(request) + "\n")

                handle.flush()
                os.fsync(handle.fileno())

            with Path(temp_file).open("rb") as binary_file:
                file_response = await self._upload_file(binary_file, "batch")

            batch_response = await self._create_batch(
                file_response.id, "/v1/embeddings", "24h"
            )
            logger.info(
                "Created batch job %s for %d texts", batch_response.id, len(texts)
            )
            return batch_response.id
        except Exception as exc:
            msg = f"Failed to create batch job: {exc}"
            raise EmbeddingServiceError(msg) from exc
        finally:
            if temp_file and Path(temp_file).exists():
                with contextlib.suppress(OSError):
                    Path(temp_file).unlink()
