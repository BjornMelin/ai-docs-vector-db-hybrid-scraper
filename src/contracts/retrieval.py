"""Canonical retrieval data models shared across services and MCP tooling."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


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


__all__ = ["SearchRecord"]
