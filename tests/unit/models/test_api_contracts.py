"""Unit tests for canonical API contract models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pytest
from pydantic import ValidationError

from src.config.models import SearchStrategy, VectorType
from src.contracts.documents import (
    DocumentListResponse,
    DocumentOperationResponse,
    DocumentRecord,
    DocumentUpsertRequest,
)
from src.contracts.retrieval import SearchRecord, SearchResponse
from src.models.search import SearchRequest


@pytest.fixture
def base_search_payload() -> dict[str, object]:
    """Provide a minimal search payload for reuse."""

    return {"query": "search docs", "collection": "documentation"}


class TestSearchRequest:
    """SearchRequest behaviour and validation."""

    def test_from_input_string_payload(
        self, base_search_payload: Mapping[str, object]
    ) -> None:
        """The helper should normalise string payloads."""

        request = SearchRequest.from_input(
            "vector databases", collection="guides", limit=25
        )

        assert request.query == "vector databases"
        assert request.collection == "guides"
        assert request.limit == 25
        assert request.search_strategy == SearchStrategy.HYBRID

    def test_from_input_overrides_existing_request(
        self, base_search_payload: Mapping[str, object]
    ) -> None:
        """Passing an existing model applies overrides immutably."""

        original = SearchRequest.model_validate(base_search_payload)
        updated = SearchRequest.from_input(original, limit=50, enable_rag=True)

        assert updated is not original
        assert updated.limit == 50
        assert updated.enable_rag is True
        assert original.limit == 10

    @pytest.mark.parametrize(
        "vector_type, sparse_vector",
        [
            (VectorType.SPARSE, None),
            (VectorType.HYBRID, None),
        ],
    )
    def test_sparse_vector_required_for_sparse_types(
        self,
        vector_type: VectorType,
        sparse_vector: dict[int, float] | None,
        base_search_payload: Mapping[str, object],
    ) -> None:
        """Sparse-compatible vector types must include a sparse_vector payload."""

        payload = dict(base_search_payload)
        payload["vector_type"] = vector_type
        payload["sparse_vector"] = sparse_vector

        with pytest.raises(ValueError, match="sparse_vector is required"):
            SearchRequest.model_validate(payload)

    def test_strategy_vector_type_mismatch_rejected(
        self, base_search_payload: Mapping[str, object]
    ) -> None:
        """Dense strategy cannot run with a sparse-only vector type."""

        payload = dict(base_search_payload)
        payload["vector_type"] = VectorType.SPARSE
        payload["sparse_vector"] = {0: 1.0}
        payload["search_strategy"] = SearchStrategy.DENSE

        with pytest.raises(ValueError, match="Dense search_strategy"):
            SearchRequest.model_validate(payload)

    def test_filter_key_validation_rejects_invalid_names(
        self, base_search_payload: Mapping[str, object]
    ) -> None:
        """Filter keys must match the allowed pattern."""

        payload = dict(base_search_payload)
        payload["filters"] = {"invalid key": "value"}

        with pytest.raises(ValueError, match="Invalid filter key"):
            SearchRequest.model_validate(payload)

    def test_filter_value_validation_rejects_dangerous_content(
        self,
        base_search_payload: Mapping[str, object],
    ) -> None:
        """Dangerous patterns inside filter values are blocked."""

        payload = dict(base_search_payload)
        payload["filters"] = {"title": "DROP TABLE users"}

        with pytest.raises(ValueError, match="dangerous pattern"):
            SearchRequest.model_validate(payload)

    def test_query_vector_dimension_must_match_force_dimension(
        self,
        base_search_payload: Mapping[str, object],
    ) -> None:
        """force_dimension requires the dense vector to have matching size."""

        payload = dict(base_search_payload)
        payload["query_vector"] = [0.1, 0.2]
        payload["force_dimension"] = 3

        with pytest.raises(ValueError, match="force_dimension"):
            SearchRequest.model_validate(payload)


class TestDocumentContracts:
    """Document contract payloads."""

    def test_document_upsert_request_defaults(self) -> None:
        """DocumentUpsertRequest applies sensible defaults."""

        payload = DocumentUpsertRequest(content="Doc body")

        assert payload.collection == "documentation"
        assert payload.metadata is None

    def test_document_operation_response_defaults(self) -> None:
        """DocumentOperationResponse returns success by default."""

        response = DocumentOperationResponse(id="doc-123", message="stored")

        assert response.status == "success"
        assert response.message == "stored"

    def test_document_list_response_structure(self) -> None:
        """DocumentListResponse normalises nested document records."""

        record = DocumentRecord(id="doc-1", content="body", metadata={"source": "test"})
        response = DocumentListResponse(
            documents=[record],
            count=1,
            limit=25,
            next_offset="cursor-2",
        )

        assert response.count == 1
        assert response.documents[0].id == "doc-1"
        assert response.next_offset == "cursor-2"


@dataclass
class _MatchStub:
    """Minimal stub to emulate vector store matches in tests."""

    id: str
    payload: dict[str, object]
    score: float = 0.0
    normalized_score: float | None = None
    collection: str | None = None


class TestSearchRecord:
    """SearchRecord normalisation logic."""

    def test_from_payload_handles_plain_dict(self) -> None:
        """Missing identifiers are generated automatically."""

        payload = SearchRecord.from_payload({"content": "hello world", "score": 0.5})

        assert isinstance(payload.id, str) and payload.id
        assert payload.content == "hello world"
        assert payload.score == pytest.approx(0.5)

    def test_parse_list_normalises_iterable(self) -> None:
        """parse_list accepts heterogeneous payloads."""

        result = SearchRecord.parse_list(
            [
                {"id": "d1", "content": "doc", "score": 0.9},
                "fallback content",
            ],
            default_collection="docs",
        )

        assert len(result) == 2
        assert result[0].id == "d1"
        assert result[1].content == "fallback content"
        assert result[1].collection == "docs"

    def test_from_vector_match_extracts_payload_fields(self) -> None:
        """Vector matches are converted to SearchRecord."""

        match = _MatchStub(
            id="match-1",
            payload={"content": "body text", "collection": "docs"},
            score=0.42,
            normalized_score=0.84,
        )

        record = SearchRecord.from_vector_match(match, collection_name="fallback")

        assert record.id == "match-1"
        assert record.content == "body text"
        assert record.score == pytest.approx(0.84)
        assert record.collection == "docs"

    def test_from_payload_rejects_unsupported_type(self) -> None:
        """Unsupported payload types raise TypeError."""

        with pytest.raises(TypeError):
            SearchRecord.from_payload(object())


class TestSearchResponse:
    """SearchResponse envelope behaviour."""

    def test_requires_mandatory_fields(self) -> None:
        """Validation fails when mandatory fields are missing."""

        with pytest.raises(ValidationError):
            SearchResponse.model_validate({"records": []})

    def test_forbids_extra_fields(self) -> None:
        """Extra attributes are rejected by the schema."""

        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SearchResponse.model_validate(
                {
                    "records": [],
                    "total_results": 0,
                    "query": "q",
                    "processing_time_ms": 10.0,
                    "unexpected": "value",
                }
            )
