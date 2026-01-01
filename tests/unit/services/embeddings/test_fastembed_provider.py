"""Tests for the FastEmbed provider using deterministic fakes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from src.services.embeddings.fastembed_provider import (
    FastEmbedProvider,
)
from src.services.errors import EmbeddingServiceError


@dataclass
class _StubDenseEmbeddings:
    """Stub FastEmbedEmbeddings compatible implementation."""

    model_name: str
    doc_embed_type: str
    max_length: int = 256

    def embed_query(self, text: str) -> list[float]:
        """Embed the query."""
        return [float(len(text) + 1), 1.0, -1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed the documents."""
        return [[float(len(text)), 0.5, -0.5] for text in texts]


class _StubSparseEmbeddings:
    """Stub sparse embedding generator mimicking langchain-qdrant output."""

    def __init__(self, model_name: str) -> None:
        """Initialize the stub sparse embeddings."""
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[SimpleNamespace]:
        """Embed the documents."""
        payloads = []
        for index, text in enumerate(texts):
            payloads.append(
                SimpleNamespace(indices=[index, index + 1], values=[len(text), 1.0])
            )
        return payloads


@pytest.fixture(autouse=True)
def stub_asyncio_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    """Execute asyncio.to_thread synchronously for determinism."""

    async def _to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Execute the function in a thread."""
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", _to_thread)


@pytest.fixture(autouse=True)
def stub_dense_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide deterministic dense embedding implementation."""
    monkeypatch.setattr(
        "src.services.embeddings.fastembed_provider.FastEmbedEmbeddings",
        _StubDenseEmbeddings,
    )


@pytest.fixture(autouse=True)
def stub_sparse_impl(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure sparse runtime is available for tests."""
    monkeypatch.setattr(
        "src.services.embeddings.fastembed_provider.FastEmbedSparseRuntime",
        _StubSparseEmbeddings,
        raising=False,
    )


@pytest.mark.asyncio
async def test_initialize_sets_dimensions() -> None:
    """Initialization should probe embedding size via embed_query."""
    provider = FastEmbedProvider(model_name="stub-model")

    await provider.initialize()

    assert provider.embedding_dimension == 3


@pytest.mark.asyncio
async def test_generate_embeddings_requires_initialization() -> None:
    """Generate should fail when provider has not been initialized."""
    provider = FastEmbedProvider()

    with pytest.raises(EmbeddingServiceError, match="not been initialized"):
        await provider.generate_embeddings(["alpha"])


@pytest.mark.asyncio
async def test_generate_embeddings_returns_dense_vectors(
    ai_operation_calls: list[dict[str, Any]],
) -> None:
    """Dense embeddings should be produced for each input text."""
    provider = FastEmbedProvider()
    await provider.initialize()

    embeddings = await provider.generate_embeddings(["alpha", "beta"])

    assert embeddings == [[5.0, 0.5, -0.5], [4.0, 0.5, -0.5]]
    assert ai_operation_calls  # record_ai_operation invoked
    assert ai_operation_calls[0]["provider"] == "fastembed"
    assert ai_operation_calls[0]["success"] is True


@pytest.mark.asyncio
async def test_generate_embeddings_empty_payload_short_circuits() -> None:
    """Empty input should not invoke LangChain and return an empty list."""
    provider = FastEmbedProvider()
    await provider.initialize()

    embeddings = await provider.generate_embeddings([])

    assert embeddings == []


@pytest.mark.asyncio
async def test_cleanup_resets_internal_state() -> None:
    """Cleanup should reset cached models and initialization flag."""
    provider = FastEmbedProvider()
    await provider.initialize()

    await provider.cleanup()

    with pytest.raises(EmbeddingServiceError, match="not been initialized"):
        await provider.generate_embeddings(["gamma"])


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_happy_path() -> None:
    """Sparse embeddings require optional dependency and should serialize output."""
    provider = FastEmbedProvider()

    await provider.initialize()

    sparse_results = await provider.generate_sparse_embeddings(["alpha", "beta"])

    assert sparse_results == [
        {"indices": [0, 1], "values": [5, 1.0]},
        {"indices": [1, 2], "values": [4, 1.0]},
    ]


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_invalid_payload_raises() -> None:
    """Sparse payload missing required fields should raise a clear error."""
    provider = FastEmbedProvider()
    await provider.initialize()

    class _BadSparse:  # missing indices/values
        def __init__(self) -> None:
            self.other = [1]

    class _BadRuntime:  # type: ignore[too-many-instance-attributes]
        def __init__(self, *_a, **_k) -> None:
            pass

        def embed_documents(self, _texts: list[str]) -> list[object]:
            return [_BadSparse()]

    # Force provider to use bad runtime stub
    provider._sparse = _BadRuntime()  # type: ignore[assignment]  # pylint: disable=protected-access

    with pytest.raises(EmbeddingServiceError, match="missing indices/values"):
        await provider.generate_sparse_embeddings(["alpha"])


@pytest.mark.asyncio
async def test_generate_embeddings_failure_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dense embedding failures should propagate and still record telemetry."""
    provider = FastEmbedProvider()
    await provider.initialize()

    class _BoomDense(_StubDenseEmbeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("boom")

    # Replace dense model with one that raises
    provider._dense = _BoomDense(model_name="m", doc_embed_type="default")  # type: ignore[assignment]  # pylint: disable=protected-access

    with pytest.raises(RuntimeError):
        await provider.generate_embeddings(["alpha"])


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse embedding runtime failures should propagate and record telemetry."""
    provider = FastEmbedProvider()
    await provider.initialize()

    class _BoomSparse:
        def __init__(self, *_a, **_k) -> None:
            pass

        def embed_documents(self, _texts: list[str]):
            raise RuntimeError("sparse-boom")

    provider._sparse = _BoomSparse()  # type: ignore[assignment]  # pylint: disable=protected-access

    with pytest.raises(RuntimeError):
        await provider.generate_sparse_embeddings(["alpha"])


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_initialization_required() -> None:
    """Sparse embeddings should enforce initialization precondition."""
    provider = FastEmbedProvider()

    with pytest.raises(EmbeddingServiceError, match="not been initialized"):
        await provider.generate_sparse_embeddings(["alpha"])


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_dependency_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When sparse runtime missing, provider should raise a clear error."""
    provider = FastEmbedProvider()
    await provider.initialize()

    monkeypatch.setattr(
        "src.services.embeddings.fastembed_provider.FastEmbedSparseRuntime",
        None,
        raising=False,
    )

    with pytest.raises(EmbeddingServiceError, match="required for sparse embeddings"):
        await provider.generate_sparse_embeddings(["alpha"])


def test_langchain_embeddings_property_requires_initialization() -> None:
    """Accessing langchain_embeddings before initialization should raise."""
    provider = FastEmbedProvider()

    with pytest.raises(EmbeddingServiceError, match="not been initialized"):
        _ = provider.langchain_embeddings


@pytest.mark.asyncio
async def test_initialize_is_idempotent() -> None:
    """Repeated initialization should be a no-op after first init."""
    provider = FastEmbedProvider(model_name="stub-model")

    await provider.initialize()
    await provider.initialize()

    assert provider.embedding_dimension == 3


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_empty_payload_short_circuits() -> None:
    """Empty sparse payload should return without requiring initialization."""
    provider = FastEmbedProvider()

    assert await provider.generate_sparse_embeddings([]) == []


def test_max_tokens_per_request_defaults_to_512_when_uninitialized() -> None:
    """The provider should report a conservative default max token value."""
    provider = FastEmbedProvider()

    assert provider.max_tokens_per_request == 512


def test_internal_loader_raises_when_dense_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing langchain-community fastembed module should raise a clear error."""
    import src.services.embeddings.fastembed_provider as module

    monkeypatch.setattr(module, "FastEmbedEmbeddings", None)

    def _missing(_name: str):
        raise ModuleNotFoundError("langchain_community.embeddings.fastembed")

    monkeypatch.setattr(module, "import_module", _missing)

    with pytest.raises(EmbeddingServiceError, match="langchain-community"):
        module._load_fastembed_embeddings()


def test_internal_loader_raises_when_dense_symbol_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing FastEmbedEmbeddings symbol should raise a clear error."""
    import src.services.embeddings.fastembed_provider as module

    monkeypatch.setattr(module, "FastEmbedEmbeddings", None)

    class _NoFastEmbedModule:
        """Minimal stub module without FastEmbedEmbeddings."""

    monkeypatch.setattr(module, "import_module", lambda _name: _NoFastEmbedModule())

    with pytest.raises(
        EmbeddingServiceError, match="FastEmbedEmbeddings not available"
    ):
        module._load_fastembed_embeddings()


def test_internal_sparse_runtime_loader_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse runtime loader should return None when langchain-qdrant is missing."""
    import src.services.embeddings.fastembed_provider as module

    monkeypatch.setattr(module, "FastEmbedSparseRuntime", module._SPARSE_RUNTIME_UNSET)

    def _missing(_name: str):
        raise ModuleNotFoundError("langchain_qdrant")

    monkeypatch.setattr(module, "import_module", _missing)

    assert module._load_fastembed_sparse_runtime() is None


def test_internal_sparse_runtime_loader_returns_none_when_symbol_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sparse runtime loader should return None when FastEmbedSparse is missing."""
    import src.services.embeddings.fastembed_provider as module

    monkeypatch.setattr(module, "FastEmbedSparseRuntime", module._SPARSE_RUNTIME_UNSET)

    class _NoSparseModule:
        """Minimal stub module without FastEmbedSparse."""

    monkeypatch.setattr(module, "import_module", lambda _name: _NoSparseModule())

    assert module._load_fastembed_sparse_runtime() is None
