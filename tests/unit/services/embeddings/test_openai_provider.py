"""Tests for the OpenAI embedding provider with deterministic stubs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from src.services.embeddings.openai_provider import OpenAIEmbeddingProvider
from src.services.errors import EmbeddingServiceError


class _StubAsyncClient:
    """Minimal AsyncOpenAI stub supporting embeddings and lifecycle."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the stub async client."""
        self.kwargs = kwargs
        self.closed = False

        async def _raise(*_args: Any, **_kwargs: Any) -> None:
            """Raise an assertion error."""
            raise AssertionError("Unexpected API invocation in test")

        self.embeddings = SimpleNamespace(create=_raise)
        self.files = SimpleNamespace(create=_raise)
        self.batches = SimpleNamespace(create=_raise)

    async def close(self) -> None:
        """Close the stub async client."""
        self.closed = True


class _EmbeddingStub:
    """Simple payload mimicking OpenAI embedding data entry."""

    def __init__(self, vector: list[float]) -> None:
        """Initialize the embedding stub."""
        self.embedding = vector


def _build_response(
    vectors: list[list[float]], usage: SimpleNamespace | None
) -> SimpleNamespace:
    """Create a response object compatible with the provider expectations."""
    return SimpleNamespace(
        data=[_EmbeddingStub(vector) for vector in vectors],
        usage=usage,
        model="text-embedding-3-small",
    )


@pytest.fixture
def async_client_calls(monkeypatch: pytest.MonkeyPatch) -> list[_StubAsyncClient]:
    """Patch AsyncOpenAI constructor to capture created client instances."""
    clients: list[_StubAsyncClient] = []

    def _factory(**kwargs: Any) -> _StubAsyncClient:
        """Factory function to create a stub async client."""
        client = _StubAsyncClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.AsyncOpenAI",
        _factory,
        raising=False,
    )

    return clients


def _prepare_initialized_provider() -> OpenAIEmbeddingProvider:
    """Prepare an initialized provider.

    This test-only helper bypasses the provider's normal initialize() flow
    by directly setting _client and _initialized to put the provider into
    a ready state for testing.
    """
    provider = OpenAIEmbeddingProvider(api_key="sk-test")
    cast(Any, provider)._client = _StubAsyncClient()  # pylint: disable=protected-access
    cast(Any, provider)._initialized = True  # pylint: disable=protected-access
    return provider


@pytest.mark.asyncio
async def test_initialize_configures_client(
    async_client_calls: list[_StubAsyncClient],
) -> None:
    """Initialization should instantiate AsyncOpenAI with expected kwargs."""
    provider = OpenAIEmbeddingProvider(
        api_key="sk-test",
        max_retries=5,
        timeout=3.5,
    )

    await provider.initialize()

    assert provider._initialized is True  # pylint: disable=protected-access
    assert len(async_client_calls) == 1
    assert async_client_calls[0].kwargs["api_key"] == "sk-test"
    assert async_client_calls[0].kwargs["max_retries"] == 5
    assert async_client_calls[0].kwargs["timeout"] == 3.5


@pytest.mark.asyncio
async def test_initialize_without_api_key_fails(
    async_client_calls: list[_StubAsyncClient],
) -> None:
    """Missing API key should raise a descriptive EmbeddingServiceError."""
    provider = OpenAIEmbeddingProvider(api_key="")

    with pytest.raises(EmbeddingServiceError, match="API key not configured"):
        await provider.initialize()

    assert not async_client_calls


@pytest.mark.asyncio
async def test_initialize_handles_constructor_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AsyncOpenAI constructor failure should surface as EmbeddingServiceError."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test")

    class _Boom(Exception):
        """Boom exception."""

    def _raise_constructor(**_k: Any) -> None:
        """Raise the boom exception."""
        raise _Boom("boom")

    monkeypatch.setattr(
        "src.services.embeddings.openai_provider.AsyncOpenAI",
        _raise_constructor,
        raising=False,
    )

    with pytest.raises(
        EmbeddingServiceError, match="Failed to initialize OpenAI client"
    ):
        await provider.initialize()


@pytest.mark.asyncio
async def test_cleanup_closes_client(
    async_client_calls: list[_StubAsyncClient],
) -> None:
    """Cleanup should close the async client and reset internal state."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test")
    await provider.initialize()
    client = cast(_StubAsyncClient, provider._client)

    await provider.cleanup()

    assert client.closed is True
    assert provider._client is None
    assert provider._initialized is False  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_generate_embeddings_requires_initialization() -> None:
    """Provider must be initialized before generating embeddings."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test")

    with pytest.raises(EmbeddingServiceError, match="Provider not initialized"):
        await provider.generate_embeddings(["hello"])


@pytest.mark.asyncio
async def test_generate_embeddings_empty_payload_short_circuits(
    ai_operation_calls: list[dict[str, Any]],
) -> None:
    """Empty payload should return immediately and emit telemetry with zero tokens."""
    provider = _prepare_initialized_provider()

    embeddings = await provider.generate_embeddings([])

    assert embeddings == []
    assert ai_operation_calls
    payload = ai_operation_calls[0]
    assert payload["tokens"] is None
    assert payload["success"] is True


@pytest.mark.asyncio
async def test_generate_embeddings_batches_aggregate_usage(
    monkeypatch: pytest.MonkeyPatch,
    ai_operation_calls: list[dict[str, Any]],
) -> None:
    """Batched generation should aggregate usage from API responses."""
    provider = _prepare_initialized_provider()
    calls: list[dict[str, Any]] = []

    responses = [
        _build_response(
            [[0.1, 0.2], [0.3, 0.4]],
            SimpleNamespace(prompt_tokens=8, completion_tokens=0, total_tokens=8),
        ),
        _build_response([[0.5, 0.6]], None),  # Force fallback token counting
    ]

    async def _fake_send(self: OpenAIEmbeddingProvider, params: dict[str, Any]) -> Any:
        calls.append(params)
        return responses.pop(0)

    monkeypatch.setattr(
        OpenAIEmbeddingProvider,
        "_send_embedding_request",
        _fake_send,
        raising=False,
    )

    vectors = await provider.generate_embeddings(
        ["alpha", "beta", "gamma"], batch_size=2
    )

    assert vectors == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    assert len(calls) == 2
    assert calls[0]["input"] == ["alpha", "beta"]
    assert calls[1]["input"] == ["gamma"]

    telemetry = ai_operation_calls[0]
    # Usage should include explicit 8 tokens plus fallback of len("gamma") == 5
    assert telemetry["tokens"] == 13
    assert telemetry["success"] is True
    assert telemetry["cost_usd"] == pytest.approx(13 * provider.cost_per_token)


@pytest.mark.asyncio
async def test_generate_embeddings_error_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Low-level errors should surface as EmbeddingServiceError with helpful message."""
    provider = _prepare_initialized_provider()

    async def _raise(_self: OpenAIEmbeddingProvider, _params: dict[str, Any]) -> Any:
        raise RuntimeError("rate_limit_exceeded: slow down")

    monkeypatch.setattr(
        OpenAIEmbeddingProvider,
        "_send_embedding_request",
        _raise,
        raising=False,
    )

    with pytest.raises(EmbeddingServiceError, match="rate limit exceeded"):
        await provider.generate_embeddings(["hello"], batch_size=1)


@pytest.mark.asyncio
async def test_generate_embeddings_raises_context_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Context length errors should propagate a descriptive message."""
    provider = _prepare_initialized_provider()

    async def _raise(_self: OpenAIEmbeddingProvider, _params: dict[str, Any]) -> Any:
        raise RuntimeError("context_length_exceeded: too long")

    monkeypatch.setattr(
        OpenAIEmbeddingProvider,
        "_send_embedding_request",
        _raise,
        raising=False,
    )

    with pytest.raises(EmbeddingServiceError, match="Text too long"):
        await provider.generate_embeddings(["x" * 9000])


@pytest.mark.asyncio
async def test_generate_embeddings_error_invalid_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid API key errors should be mapped clearly."""
    provider = _prepare_initialized_provider()

    async def _raise(_self: OpenAIEmbeddingProvider, _params: dict[str, Any]) -> Any:
        raise RuntimeError("invalid_api_key: bad")

    monkeypatch.setattr(
        OpenAIEmbeddingProvider, "_send_embedding_request", _raise, raising=False
    )

    with pytest.raises(EmbeddingServiceError, match="Invalid OpenAI API key"):
        await provider.generate_embeddings(["hello"], batch_size=1)


@pytest.mark.asyncio
async def test_generate_embeddings_error_quota(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quota errors should be mapped clearly."""
    provider = _prepare_initialized_provider()

    async def _raise(_self: OpenAIEmbeddingProvider, _params: dict[str, Any]) -> Any:
        raise RuntimeError("insufficient_quota: exhausted")

    monkeypatch.setattr(
        OpenAIEmbeddingProvider, "_send_embedding_request", _raise, raising=False
    )

    with pytest.raises(EmbeddingServiceError, match="quota"):
        await provider.generate_embeddings(["hello"], batch_size=1)


def test_tokenizer_fallback_when_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_get_token_encoder should fall back when encoding_for_model fails."""
    provider = OpenAIEmbeddingProvider(api_key="sk-test")

    class _Enc:
        """Encoding stub."""

        def encode(self, text: str) -> list[int]:
            """Encode the text."""
            return [ord(c) & 0x7F for c in text]

    def _raise_key(*_a: Any, **_k: Any) -> None:
        """Raise a key error."""
        raise KeyError("nope")

    monkeypatch.setattr("tiktoken.encoding_for_model", _raise_key, raising=False)
    monkeypatch.setattr("tiktoken.get_encoding", lambda _n: _Enc(), raising=False)

    enc = provider._get_token_encoder()  # pylint: disable=protected-access
    assert len(enc.encode("abc")) == 3


@pytest.mark.asyncio
async def test_generate_embeddings_batch_api_submits_job(
    monkeypatch: pytest.MonkeyPatch,
    ai_operation_calls: list[dict[str, Any]],
) -> None:
    """Batch API should upload job and return identifier."""
    provider = _prepare_initialized_provider()

    async def _fake_submit(
        self: OpenAIEmbeddingProvider, texts: list[str], custom_ids: list[str]
    ) -> str:  # pragma: no cover - signature used in patching
        """Fake submit."""
        assert len(texts) == 2
        assert texts == ["alpha", "beta"]
        assert custom_ids == ["item-0", "item-1"]
        return "batch-123"

    monkeypatch.setattr(
        OpenAIEmbeddingProvider,
        "_submit_batch_job",
        _fake_submit,
        raising=False,
    )

    batch_id = await provider.generate_embeddings_batch_api(
        ["alpha", "beta"], custom_ids=["item-0", "item-1"]
    )

    assert batch_id == "batch-123"
    assert ai_operation_calls
    telemetry = ai_operation_calls[0]
    assert telemetry["operation_type"] == "embedding_batch"
    assert telemetry["success"] is True
    assert telemetry["attributes"]["gen_ai.request.batch_size"] == 2


@pytest.mark.asyncio
async def test_generate_embeddings_batch_api_custom_ids_validation() -> None:
    """Mismatch between custom ids and payload should raise an error."""
    provider = _prepare_initialized_provider()

    with pytest.raises(EmbeddingServiceError, match="Custom ID list must match"):
        await provider.generate_embeddings_batch_api(
            ["alpha", "beta"], custom_ids=["one"]
        )


@pytest.mark.asyncio
async def test_submit_batch_job_writes_and_cleans_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """_submit_batch_job should upload, create batch, and cleanup temp file."""
    provider = _prepare_initialized_provider()

    class _Obj:
        """Object stub."""

        def __init__(self, oid: str) -> None:
            """Initialize the object stub."""
            self.id = oid

    async def _fake_upload(*, file, purpose):
        """Fake upload."""
        assert purpose == "batch"
        assert file is not None
        return _Obj("file-1")

    async def _fake_create(input_file_id: str, endpoint: str, completion_window: str):
        """Fake create."""
        assert input_file_id == "file-1"
        assert endpoint == "/v1/embeddings"
        assert completion_window == "24h"
        return _Obj("batch-xyz")

    # Patch client methods on provider
    client = cast(Any, provider._client)  # pylint: disable=protected-access
    client.files = SimpleNamespace(create=_fake_upload)
    client.batches = SimpleNamespace(create=_fake_create)

    # Exercise the job submission path
    batch_id = await provider._submit_batch_job(["a", "b"], ["x", "y"])  # type: ignore[attr-defined]  # pylint: disable=protected-access
    assert batch_id == "batch-xyz"


@pytest.mark.asyncio
async def test_submit_batch_job_error_maps_to_embedding_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Errors during batch submission should raise EmbeddingServiceError."""
    provider = _prepare_initialized_provider()

    async def _boom(**_k: Any) -> None:
        """Raise a runtime error."""
        raise RuntimeError("upload failed")

    client = cast(Any, provider._client)  # pylint: disable=protected-access
    client.files = SimpleNamespace(create=_boom)
    client.batches = SimpleNamespace(create=lambda **_k: None)  # never reached

    with pytest.raises(EmbeddingServiceError, match="Failed to create batch job"):
        await provider._submit_batch_job(["a"], ["x"])  # type: ignore[attr-defined]  # pylint: disable=protected-access
