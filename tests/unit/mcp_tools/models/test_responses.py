"""Unit tests for MCP response models."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from src.contracts.retrieval import SearchRecord
from src.mcp_tools.models.responses import (
    EmbeddingGenerationResponse,
    GenericDictResponse,
    OperationStatus,
)


class TestSearchRecord:
    """Search record contract coverage."""

    def test_minimal_valid_result(self) -> None:
        """SearchRecord should accept minimal payloads."""
        result = SearchRecord(
            id="doc_123",
            content="This is the document content",
            score=0.95,
        )
        assert result.id == "doc_123"
        assert result.content == "This is the document content"
        assert result.score == 0.95
        assert result.url is None
        assert result.title is None
        assert result.metadata is None

    def test_all_fields(self) -> None:
        """SearchRecord should preserve optional metadata."""
        metadata = {
            "author": "John Doe",
            "created_at": "2024-01-01",
            "tags": ["python", "tutorial"],
        }
        result = SearchRecord(
            id="doc_456",
            content="Advanced Python tutorial content",
            score=0.87,
            url="https://example.com/tutorials/python",
            title="Advanced Python Tutorial",
            metadata=metadata,
        )
        assert result.id == "doc_456"
        assert result.url == "https://example.com/tutorials/python"
        assert result.title == "Advanced Python Tutorial"
        metadata_result = result.metadata
        assert metadata_result is not None
        assert metadata_result == metadata
        assert metadata_result["tags"] == ["python", "tutorial"]

    def test_missing_required_fields(self) -> None:
        """SearchRecord should enforce required fields."""
        with pytest.raises(ValidationError):
            SearchRecord.model_validate({"content": "test", "score": 0.5})
        with pytest.raises(ValidationError):
            SearchRecord.model_validate({"id": "123", "score": 0.5})
        with pytest.raises(ValidationError):
            SearchRecord.model_validate({"id": "123", "content": "test"})

    def test_score_validation(self) -> None:
        """Score should accept floats >= 0."""
        SearchRecord(id="1", content="test", score=0.0)
        SearchRecord(id="2", content="test", score=1.5)
        with pytest.raises(ValidationError):
            SearchRecord(id="3", content="test", score=-0.1)

    def test_metadata_flexibility(self) -> None:
        """Metadata should allow arbitrary structures."""
        complex_metadata: dict[str, Any] = {
            "nested": {"level": 2, "items": [1, 2, 3]},
            "array": ["a", "b", "c"],
            "metrics": {"score": 42},
        }
        result = SearchRecord(
            id="meta",
            content="test",
            score=0.5,
            metadata=complex_metadata,
        )
        metadata_result = result.metadata
        assert metadata_result is not None
        assert metadata_result == complex_metadata

    def test_serialization(self) -> None:
        """SearchRecord should serialize to dict/JSON."""
        metadata = {
            "author": "Jane",
            "tags": ["docs"],
        }
        result = SearchRecord(
            id="doc_789",
            content="Content",
            score=0.42,
            metadata=metadata,
        )
        data = result.model_dump()
        assert data["metadata"] == metadata
        json_data = json.loads(result.model_dump_json())
        assert json_data["metadata"]["tags"] == ["docs"]


class TestOperationStatus:
    """OperationStatus and dictionary response helpers."""

    def test_status_defaults(self) -> None:
        """OperationStatus should accept minimal payloads."""
        status = OperationStatus(status="success")
        assert status.status == "success"
        assert status.message is None
        assert status.details is None

    def test_status_with_details(self) -> None:
        """OperationStatus should preserve details."""
        status = OperationStatus(
            status="error",
            message="failed",
            details={"attempts": 2},
        )
        dumped = status.model_dump()
        assert dumped["message"] == "failed"
        assert dumped["details"] == {"attempts": 2}

    def test_generic_dict_response(self) -> None:
        """GenericDictResponse should allow arbitrary keys."""
        payload = GenericDictResponse.model_validate(
            {"value": "ok", "diagnostics": {"span": "abc"}}
        )
        dumped = payload.model_dump()
        assert dumped["value"] == "ok"
        assert dumped["diagnostics"]["span"] == "abc"


class TestEmbeddingGenerationResponse:
    """EmbeddingGenerationResponse coverage."""

    def test_dense_only_payload(self) -> None:
        """Dense embeddings should serialize as lists."""
        response = EmbeddingGenerationResponse(
            embeddings=[[1.0, 0.5]],
            model="local-model",
            provider="fastembed",
            total_tokens=42,
        )
        data = response.model_dump()
        assert data["embeddings"] == [[1.0, 0.5]]
        assert data["model"] == "local-model"
        assert data["provider"] == "fastembed"
        assert data["total_tokens"] == 42

    def test_hybrid_payload(self) -> None:
        """Sparse embeddings should remain optional."""
        response = EmbeddingGenerationResponse(
            embeddings=[[0.1, 0.2]],
            sparse_embeddings=[[0.3, 0.4]],
        )
        assert response.sparse_embeddings == [[0.3, 0.4]]

    def test_extra_fields(self) -> None:
        """Extra keys should be preserved due to allow extra config."""
        response = EmbeddingGenerationResponse(
            embeddings=[[0.0]],
            cost_estimate=1.23,
            model="model",
            provider="provider",
        )
        dumped = response.model_dump()
        assert dumped["cost_estimate"] == 1.23
