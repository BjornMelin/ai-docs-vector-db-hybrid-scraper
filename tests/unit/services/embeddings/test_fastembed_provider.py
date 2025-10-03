"""Sanity checks for the LangChain-backed FastEmbed provider."""

import pytest

from src.services.embeddings.fastembed_provider import FastEmbedProvider


@pytest.mark.asyncio
async def test_generate_embeddings_returns_dense_vectors() -> None:
    provider = FastEmbedProvider()
    await provider.initialize()

    embeddings = await provider.generate_embeddings(["alpha", "beta"])

    assert len(embeddings) == 2
    assert all(isinstance(vector, list) for vector in embeddings)
    assert all(len(vector) == provider.embedding_dimension for vector in embeddings)


@pytest.mark.asyncio
async def test_generate_sparse_embeddings_optional_dependency() -> None:
    provider = FastEmbedProvider()
    await provider.initialize()

    try:
        sparse_vectors = await provider.generate_sparse_embeddings(["hybrid"])
    except Exception as exc:  # pragma: no cover - dependency guard
        pytest.skip(f"Sparse embeddings unavailable: {exc}")

    assert len(sparse_vectors) == 1
    payload = sparse_vectors[0]
    assert set(payload.keys()) == {"indices", "values"}
    assert len(payload["indices"]) == len(payload["values"])


@pytest.mark.asyncio
async def test_cleanup_releases_resources() -> None:
    provider = FastEmbedProvider()
    await provider.initialize()
    await provider.cleanup()

    from src.services.errors import EmbeddingServiceError

    with pytest.raises(EmbeddingServiceError):
        await provider.generate_embeddings(["gamma"])
