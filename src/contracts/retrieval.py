"""Canonical retrieval data models shared across services and MCP tooling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:  # pragma: no cover - import cycles avoided at runtime
    from src.services.vector_db.types import VectorMatch


class SearchRecord(BaseModel):
    """Normalized representation of a search hit.

    The model intentionally keeps its surface minimal and stable so higher-level
    services, CLI utilities, and MCP tools can rely on a single schema. Extra
    provider-specific fields are permitted via Pydantic's ``extra='allow'`` to
    avoid brittle wrappers or bespoke DTOs downstream.
    """

    id: str = Field(..., description="Unique identifier for the document")
    content: str = Field(..., description="Document content or snippet")
    score: float = Field(..., ge=0.0, description="Relevance score")
    url: str | None = Field(default=None, description="Document URL")
    title: str | None = Field(default=None, description="Document title")
    metadata: dict[str, Any] | None = Field(
        default=None, description="Additional metadata supplied by providers"
    )
    content_type: str | None = Field(
        default=None, description="Detected content type for analytics"
    )
    content_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Content confidence score"
    )
    quality_overall: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall quality score"
    )
    quality_completeness: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Completeness quality score"
    )
    quality_relevance: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Relevance quality score"
    )
    quality_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Confidence in quality scoring"
    )
    content_intelligence_analyzed: bool | None = Field(
        default=None, description="Flag indicating content intelligence analysis"
    )
    collection: str | None = Field(
        default=None, description="Source collection identifier"
    )
    raw_score: float | None = Field(
        default=None, description="Unnormalized similarity score"
    )
    normalized_score: float | None = Field(
        default=None, description="Normalized similarity score"
    )
    group_id: str | None = Field(
        default=None, description="Grouping identifier such as doc_id"
    )
    group_rank: int | None = Field(
        default=None, ge=1, description="Rank of the result within its group"
    )
    grouping_applied: bool | None = Field(
        default=None, description="Whether server-side grouping was applied"
    )

    model_config = ConfigDict(extra="allow")

    @classmethod
    def from_payload(cls, payload: Any) -> SearchRecord:
        """Coerce arbitrary payloads into a :class:`SearchRecord`.

        Args:
            payload: A dictionary, existing :class:`SearchRecord`, or object that
                can be coerced into the canonical form.

        Returns:
            A validated :class:`SearchRecord` instance.

        Raises:
            TypeError: If the payload type cannot be normalized.
        """

        if isinstance(payload, cls):
            return payload
        if isinstance(payload, dict):
            normalized_payload = payload.copy()
            normalized_payload.setdefault("id", str(uuid4()))
            normalized_payload.setdefault("content", "")
            normalized_payload.setdefault("score", 0.0)
            return cls.model_validate(normalized_payload)
        msg = f"Unsupported search record payload type: {type(payload)!r}"
        raise TypeError(msg)

    @classmethod
    def parse_list(cls, payloads: Iterable[Any]) -> list[SearchRecord]:
        """Normalize a sequence of payloads into search records."""

        return [cls.from_payload(item) for item in payloads]

    @classmethod
    def from_vector_match(
        cls,
        match: VectorMatch,
        *,
        collection_name: str,
    ) -> SearchRecord:
        """Create a :class:`SearchRecord` from a vector store match.

        Args:
            match: Vector store match returned by the Qdrant client or LangChain.
            collection_name: Name of the collection the match was sourced from.

        Returns:
            Canonical :class:`SearchRecord` representation of the match.
        """

        payload: dict[str, Any] = dict(getattr(match, "payload", {}) or {})
        group_info_raw = payload.get("_grouping")
        group_info: Mapping[str, Any] | None = (
            dict(group_info_raw) if isinstance(group_info_raw, Mapping) else None
        )

        def _coerce_float(value: Any) -> float | None:
            if isinstance(value, int | float):
                return float(value)
            return None

        def _coerce_bool(value: Any) -> bool | None:
            if isinstance(value, bool):
                return value
            return None

        def _coerce_str(value: Any) -> str | None:
            if isinstance(value, str):
                return value
            return None

        normalized_score = _coerce_float(getattr(match, "normalized_score", None))
        raw_score = (
            _coerce_float(getattr(match, "raw_score", None))
            or _coerce_float(getattr(match, "score", 0.0))
            or 0.0
        )
        score = normalized_score if normalized_score is not None else raw_score
        collection_value = (
            _coerce_str(payload.get("collection"))
            or _coerce_str(payload.get("_collection"))
            or _coerce_str(getattr(match, "collection", None))
            or collection_name
        )
        group_mapping = group_info or {}
        metadata = payload or None

        return cls(
            id=str(match.id),
            content=(
                _coerce_str(payload.get("content"))
                or _coerce_str(payload.get("text"))
                or _coerce_str(payload.get("page_content"))
                or ""
            ),
            title=(
                _coerce_str(payload.get("title"))
                or _coerce_str(payload.get("name"))
                or None
            ),
            url=_coerce_str(payload.get("url")),
            metadata=metadata,
            score=score,
            raw_score=raw_score,
            normalized_score=normalized_score,
            collection=collection_value,
            content_type=_coerce_str(payload.get("content_type")),
            content_confidence=_coerce_float(payload.get("content_confidence")),
            quality_overall=_coerce_float(payload.get("quality_overall")),
            quality_completeness=_coerce_float(payload.get("quality_completeness")),
            quality_relevance=_coerce_float(payload.get("quality_relevance")),
            quality_confidence=_coerce_float(payload.get("quality_confidence")),
            content_intelligence_analyzed=_coerce_bool(
                payload.get("content_intelligence_analyzed")
            ),
            group_id=(
                _coerce_str(group_mapping.get("group_id"))
                or _coerce_str(payload.get("doc_id"))
            ),
            group_rank=(
                group_mapping.get("rank")
                if isinstance(group_mapping.get("rank"), int)
                else None
            ),
            grouping_applied=_coerce_bool(group_mapping.get("applied")),
        )


class SearchResponse(BaseModel):
    """Canonical response payload for search operations.

    Attributes mirror orchestrator outputs so HTTP handlers, CLI utilities,
    and MCP tools rely on the same envelope.
    """

    records: list[SearchRecord] = Field(
        default_factory=list, description="Search results in canonical form"
    )
    total_results: int = Field(
        default=0, ge=0, description="Number of records returned"
    )
    query: str = Field(..., description="Processed query text")
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Latency measured in milliseconds"
    )
    expanded_query: str | None = Field(
        default=None, description="Expanded query when query expansion applied"
    )
    features_used: list[str] = Field(
        default_factory=list, description="Features engaged during search"
    )
    grouping_applied: bool = Field(
        default=False, description="Whether server-side grouping was applied"
    )
    generated_answer: str | None = Field(
        default=None, description="Generated RAG answer when requested"
    )
    answer_confidence: float | None = Field(
        default=None, description="Confidence score for the generated answer"
    )
    answer_sources: list[dict[str, Any]] | None = Field(
        default=None, description="Sources that support the generated answer"
    )

    model_config = ConfigDict(extra="forbid")


__all__ = ["SearchRecord", "SearchResponse"]
