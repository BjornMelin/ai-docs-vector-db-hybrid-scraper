"""Versioned document management API router."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from src.api.dependencies import get_vector_service_dependency
from src.api.routers.v1.service_helpers import execute_service_call
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
    """Insert a document into the configured vector collection.

    Args:
        request: Canonical document upsert payload.
        vector_service: Vector store dependency resolved from DI.

    Returns:
        Operation response containing the new document identifier.
    """
    collection = request.collection
    document_id = await execute_service_call(
        operation="documents.add",
        logger=logger,
        coroutine_factory=lambda: vector_service.add_document(
            collection,
            request.content,
            metadata=_maybe_to_dict(request.metadata),
        ),
        error_detail="Failed to add document to the vector store.",
        extra={"collection": collection},
    )

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
    """Fetch a document payload by identifier.

    Args:
        document_id: Identifier of the document to load.
        vector_service: Vector store dependency resolved from DI.
        collection: Collection that stores the document.

    Returns:
        Canonical document record constructed from the vector payload.
    """

    payload = await execute_service_call(
        operation="documents.get",
        logger=logger,
        coroutine_factory=lambda: vector_service.get_document(collection, document_id),
        error_detail="Failed to retrieve document from the vector store.",
        extra={"collection": collection, "document_id": document_id},
    )

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
    """Delete a document by identifier.

    Args:
        document_id: Identifier of the document to delete.
        vector_service: Vector store dependency resolved from DI.
        collection: Collection that stores the document.

    Returns:
        Operation response describing the deletion outcome.
    """

    deleted = await execute_service_call(
        operation="documents.delete",
        logger=logger,
        coroutine_factory=lambda: vector_service.delete_document(
            collection, document_id
        ),
        error_detail="Failed to delete document from the vector store.",
        extra={"collection": collection, "document_id": document_id},
    )

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
    """List documents within a collection.

    Args:
        vector_service: Vector store dependency resolved from DI.
        collection: Collection that stores the documents.
        limit: Maximum number of documents requested.
        offset: Optional pagination cursor.

    Returns:
        Paginated list of canonical document records.
    """
    documents, next_offset = await execute_service_call(
        operation="documents.list",
        logger=logger,
        coroutine_factory=lambda: vector_service.list_documents(
            collection,
            limit=limit,
            offset=offset,
        ),
        error_detail="Failed to list documents from the vector store.",
        extra={"collection": collection},
    )

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
    """Enumerate available vector store collections.

    Args:
        vector_service: Vector store dependency resolved from DI.

    Returns:
        Mapping containing collection names and total count.
    """

    collections = await execute_service_call(
        operation="documents.list_collections",
        logger=logger,
        coroutine_factory=vector_service.list_collections,
        error_detail="Failed to list vector store collections.",
    )

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
