"""Validation tests for MCP tool request models."""

import pytest
from pydantic import ValidationError

from src.config.models import (
    ChunkingStrategy,
    FusionAlgorithm,
    SearchAccuracy,
    SearchStrategy,
    VectorType,
)
from src.mcp_tools.models.requests import (
    BatchRequest,
    DocumentRequest,
    EmbeddingRequest,
)
from src.models.search import SearchRequest


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "collection": "docs", "limit": 10, "offset": 0},
        {"query": "docs", "collection": "", "limit": 10, "offset": 0},
    ],
)
def test_search_request_rejects_blank_fields(payload):
    """SearchRequest should reject blank query or collection values."""
    with pytest.raises(ValidationError):
        SearchRequest(**payload)


@pytest.mark.parametrize(
    "limit",
    [0, 1001],
)
def test_search_request_enforces_limit_bounds(limit):
    """SearchRequest.limit must stay within the documented bounds."""
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=limit, offset=0)


def test_search_request_rejects_malicious_filters() -> None:
    """Filter keys and values should be sanitized for injection patterns."""

    with pytest.raises(ValidationError):
        SearchRequest(query="danger", filters={"$illegal": "ok"})

    with pytest.raises(ValidationError):
        SearchRequest(query="danger", filters={"status": "DROP TABLE"})

    with pytest.raises(ValidationError):
        SearchRequest(query="danger", filters={"tags": [str(i) for i in range(101)]})


def test_search_request_validates_vectors() -> None:
    """Dense and sparse vectors must respect dimensional constraints."""

    with pytest.raises(ValidationError):
        SearchRequest(query="vector", query_vector=[])

    with pytest.raises(ValidationError):
        SearchRequest(
            query="vector",
            query_vector=[0.1, 0.2],
            force_dimension=3,
        )

    with pytest.raises(ValidationError):
        SearchRequest(
            query="vector",
            vector_type=VectorType.SPARSE,
        )

    with pytest.raises(ValidationError):
        SearchRequest(
            query="vector",
            vector_type=VectorType.SPARSE,
            sparse_vector={-1: 0.5},
        )


def test_search_request_validates_strategy_alignment() -> None:
    """Search strategy and vector type combinations must remain consistent."""

    with pytest.raises(ValidationError):
        SearchRequest(
            query="strategy",
            search_strategy=SearchStrategy.SPARSE,
            vector_type=VectorType.DENSE,
        )


@pytest.mark.parametrize(
    "strategy, fusion, accuracy",
    [
        (SearchStrategy.DENSE, FusionAlgorithm.RRF, SearchAccuracy.BALANCED),
        (SearchStrategy.SPARSE, FusionAlgorithm.WEIGHTED, SearchAccuracy.ACCURATE),
    ],
)
def test_search_request_accepts_supported_enums(strategy, fusion, accuracy):
    """Enumerated configuration options should round-trip correctly."""
    request = SearchRequest(
        query="hello",
        search_strategy=strategy,
        fusion_algorithm=fusion,
        search_accuracy=accuracy,
        limit=10,
        offset=0,
    )

    assert request.search_strategy is strategy
    assert request.fusion_algorithm is fusion
    assert request.search_accuracy is accuracy


@pytest.mark.parametrize("chunk_size", [99, 4001])
def test_document_request_validates_chunk_size(chunk_size):
    """DocumentRequest enforces chunk size bounds from configuration."""
    with pytest.raises(ValidationError):
        DocumentRequest(url="https://example.com", chunk_size=chunk_size)


@pytest.mark.parametrize("batch_size", [0, 101])
def test_embedding_request_validates_batch_size(batch_size):
    """EmbeddingRequest should enforce documented batch size limits."""
    with pytest.raises(ValidationError):
        EmbeddingRequest(texts=["one"], batch_size=batch_size)


@pytest.mark.parametrize(
    "payload",
    [
        {"urls": None, "collection": "docs"},
        {"urls": ["https://example.com"], "max_concurrent": 0},
    ],
)
def test_batch_request_validates_inputs(payload):
    """BatchRequest validates list contents and concurrency limits."""
    with pytest.raises(ValidationError):
        BatchRequest(**payload)


def test_document_request_defaults_apply():
    """DocumentRequest should apply default collection and chunking strategy."""
    request = DocumentRequest(url="https://example.com/document")

    assert request.collection == "documentation"
    assert request.chunk_strategy is ChunkingStrategy.ENHANCED
    assert request.token_chunk_size == 600
    assert request.token_chunk_overlap == 120
    assert request.token_model == "cl100k_base"  # noqa: S105 - public encoding key
    assert request.json_max_chars == 20000
    assert request.enable_semantic_html_segmentation is True
    assert request.normalize_html_text is True
