"""Canonical document API contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field


class DocumentUpsertRequest(BaseModel):
    """Request payload for creating or updating a document."""

    content: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Raw document content to ingest.",
    )
    metadata: Mapping[str, Any] | None = Field(
        default=None,
        description="Optional metadata stored alongside the document.",
    )
    collection: str = Field(
        default="documentation",
        min_length=1,
        description="Destination collection name.",
    )


class DocumentOperationResponse(BaseModel):
    """Standard operation response for document mutations."""

    id: str = Field(..., description="Unique identifier of the document.")
    status: str = Field(default="success", description="Operation status indicator.")
    message: str = Field(..., description="Human-readable status message.")


class DocumentRecord(BaseModel):
    """Canonical representation of a stored document."""

    id: str = Field(..., description="Unique identifier of the document.")
    content: str | None = Field(
        default=None, description="Primary textual content of the document."
    )
    metadata: Mapping[str, Any] | None = Field(
        default=None,
        description="Arbitrary metadata payload returned by the vector store.",
    )
    collection: str | None = Field(
        default=None,
        description="Collection that owns the document.",
    )


class DocumentListResponse(BaseModel):
    """Paginated response for listing documents."""

    documents: list[DocumentRecord] = Field(
        default_factory=list, description="Documents returned for the page."
    )
    count: int = Field(
        default=0,
        ge=0,
        description="Number of documents in the current response.",
    )
    limit: int = Field(
        ...,
        ge=1,
        description="Maximum number of documents requested per page.",
    )
    next_offset: str | None = Field(
        default=None,
        description="Opaque cursor to fetch the next page of documents.",
    )
