"""Validation tests for MCP tool request models."""

import pytest
from pydantic import ValidationError

from src.config import (
    ChunkingStrategy,
    FusionAlgorithm,
    SearchAccuracy,
    SearchStrategy,
)
from src.mcp_tools.models.requests import (
    BatchRequest,
    DocumentRequest,
    EmbeddingRequest,
    FilteredSearchRequest,
    SearchRequest,
)


@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "collection": "docs"},
        {"query": "docs", "collection": ""},
    ],
)
def test_search_request_rejects_blank_fields(payload):
    """SearchRequest should reject blank query or collection values."""
    with pytest.raises(ValidationError):
        SearchRequest(**payload)


@pytest.mark.parametrize(
    "limit",
    [0, 101],
)
def test_search_request_enforces_limit_bounds(limit):
    """SearchRequest.limit must stay within the documented bounds."""
    with pytest.raises(ValidationError):
        SearchRequest(query="test", limit=limit)


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
        strategy=strategy,
        fusion_algorithm=fusion,
        search_accuracy=accuracy,
    )

    assert request.strategy is strategy
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


def test_filtered_search_request_requires_filters():
    """FilteredSearchRequest must include at least one filter."""
    invalid_payload: dict[str, object] = {"query": "hello", "filters": None}
    with pytest.raises(ValidationError):
        FilteredSearchRequest.model_validate(invalid_payload)


def test_document_request_defaults_apply():
    """DocumentRequest should apply default collection and chunking strategy."""
    request = DocumentRequest(url="https://example.com/document")

    assert request.collection == "documentation"
    assert request.chunk_strategy is ChunkingStrategy.ENHANCED
