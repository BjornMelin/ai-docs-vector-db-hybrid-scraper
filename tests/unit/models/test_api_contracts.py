"""Smoke tests for modern request/response data models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.models import SearchRequest
from src.models.document_processing import (
    ContentFilter,
    DocumentMetadata,
    ProcessedDocument,
)


class TestSearchRequest:
    """Validate the unified search request model."""

    def test_defaults_provide_balanced_hybrid_search(self) -> None:
        request = SearchRequest.model_validate({"query": "install vector db"})

        payload = request.model_dump()
        assert payload["collection"] == "documentation"
        assert payload["search_strategy"] == "hybrid"
        assert payload["search_accuracy"] == "balanced"
        assert request.enable_reranking is True
        assert payload["limit"] == 10

    def test_from_input_normalizes_string_payload(self) -> None:
        request = SearchRequest.from_input("deep learning", limit=5)

        assert request.query == "deep learning"
        assert request.limit == 5
        assert request.offset == 0

    def test_forbids_extra_attributes(self) -> None:
        with pytest.raises(ValidationError):
            SearchRequest.model_validate({"query": "hello", "unknown": 1})


class TestDocumentProcessingModels:
    """Validate document processing metadata models."""

    def test_document_metadata_defaults(self) -> None:
        metadata = DocumentMetadata(url="https://docs.example.com")

        metadata_dump = metadata.model_dump()
        assert metadata_dump["word_count"] == 0
        assert metadata_dump["has_code"] is False
        assert metadata_dump["chunking_strategy"] == "enhanced"

    def test_processed_document_requires_metadata(self) -> None:
        metadata = DocumentMetadata(url="https://docs.example.com")
        document = ProcessedDocument(id="doc-1", content="body", metadata=metadata)

        document_dump = document.model_dump()
        assert document_dump["status"] == "pending"
        assert document_dump["chunks"] == []

    def test_content_filter_enforces_duplicate_ratio_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ContentFilter.model_validate({"max_duplicate_ratio": 1.5})
