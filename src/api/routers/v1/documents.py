"""Versioned document management API router."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_service_dependency
from src.contracts.documents import (
    DocumentListResponse,
    DocumentOperationResponse,
    DocumentRecord,
    DocumentUpsertRequest,
)
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()

VectorServiceDependency = Annotated[
    VectorStoreService,
    Depends(get_vector_service_dependency),
]


@router.post("/documents", response_model=DocumentOperationResponse)
async def add_document(
    request: DocumentUpsertRequest,
    vector_service: VectorServiceDependency,
) -> DocumentOperationResponse:
    """Insert a document into the configured vector collection."""

    try:
        collection = request.collection
        document_id = await vector_service.add_document(
            collection,
            request.content,
            metadata=_maybe_to_dict(request.metadata),
        )
    except Exception as exc:  # pragma: no cover - service-level failures
        logger.exception("Failed to add document")
        raise HTTPException(
            status_code=500,
            detail="Failed to add document to the vector store.",
        ) from exc

    return DocumentOperationResponse(
        id=document_id,
        message="Document added successfully.",
    )


@router.get("/documents/{document_id}", response_model=DocumentRecord)
async def get_document(
    document_id: str,
    vector_service: VectorServiceDependency,
    collection: str = Query(
        default="documentation",
        description="Collection that stores the document.",
    ),
) -> DocumentRecord:
    """Fetch a document payload by identifier."""

    try:
        payload = await vector_service.get_document(collection, document_id)
    except Exception as exc:  # pragma: no cover - service-level failures
        logger.exception(
            "Failed to retrieve document", extra={"document_id": document_id}
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve document from the vector store.",
        ) from exc

    if payload is None:
        raise HTTPException(status_code=404, detail="Document not found.")
    return _to_document_record(payload, collection)


@router.delete("/documents/{document_id}", response_model=DocumentOperationResponse)
async def delete_document(
    document_id: str,
    vector_service: VectorServiceDependency,
    collection: str = Query(
        default="documentation",
        description="Collection that stores the document.",
    ),
) -> DocumentOperationResponse:
    """Delete a document by identifier."""

    try:
        deleted = await vector_service.delete_document(collection, document_id)
    except Exception as exc:  # pragma: no cover - service-level failures
        logger.exception(
            "Failed to delete document", extra={"document_id": document_id}
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document from the vector store.",
        ) from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return DocumentOperationResponse(
        id=document_id,
        message="Document deleted successfully.",
    )


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    vector_service: VectorServiceDependency,
    collection: str = Query(
        default="documentation",
        description="Collection that stores the documents.",
    ),
    limit: int = Query(
        default=25,
        ge=1,
        le=1000,
        description="Maximum number of documents to return.",
    ),
    offset: str | None = Query(
        default=None,
        description="Opaque cursor used to continue listing results.",
    ),
) -> DocumentListResponse:
    """List documents within a collection."""

    try:
        documents, next_offset = await vector_service.list_documents(
            collection,
            limit=limit,
            offset=offset,
        )
    except Exception as exc:  # pragma: no cover - service-level failures
        logger.exception("Failed to list documents", extra={"collection": collection})
        raise HTTPException(
            status_code=500,
            detail="Failed to list documents from the vector store.",
        ) from exc

    records = [_to_document_record(doc, collection) for doc in documents]
    return DocumentListResponse(
        documents=records,
        count=len(records),
        limit=limit,
        next_offset=next_offset,
    )


@router.get("/collections", response_model=dict[str, list[str] | int])
async def list_collections(
    vector_service: VectorServiceDependency,
) -> dict[str, Any]:
    """Enumerate available vector store collections."""

    try:
        collections = await vector_service.list_collections()
    except Exception as exc:  # pragma: no cover - service-level failures
        logger.exception("Failed to list collections")
        raise HTTPException(
            status_code=500,
            detail="Failed to list vector store collections.",
        ) from exc

    return {"collections": collections, "count": len(collections)}


def _maybe_to_dict(metadata: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Convert metadata to a concrete dictionary if provided."""

    if metadata is None:
        return None
    return dict(metadata)


def _to_document_record(
    payload: Mapping[str, Any],
    collection: str,
) -> DocumentRecord:
    """Normalize raw payloads returned by the vector service."""

    metadata = dict(payload)
    content = metadata.pop("content", None)
    document_id = metadata.pop("id", None)
    metadata["collection"] = metadata.get("collection", collection)
    return DocumentRecord(
        id=document_id or "",
        content=content,
        metadata=metadata or None,
        collection=collection,
    )
