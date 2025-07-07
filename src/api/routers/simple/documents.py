"""Simple mode documents API router.

Simplified document management endpoints optimized for solo developers.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from src.architecture.service_factory import get_service


logger = logging.getLogger(__name__)

router = APIRouter()


class SimpleDocumentRequest(BaseModel):
    """Simplified document request for simple mode."""

    content: str = Field(
        ..., min_length=1, max_length=10000, description="Document content"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Document metadata"
    )
    collection_name: str = Field(default="documents", description="Collection name")


class SimpleDocumentResponse(BaseModel):
    """Simplified document response for simple mode."""

    id: str = Field(..., description="Document ID")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")


@router.post("/documents", response_model=SimpleDocumentResponse)
async def add_document(request: SimpleDocumentRequest) -> SimpleDocumentResponse:
    """Add a single document to the collection.

    This endpoint provides basic document indexing without advanced features
    like batch processing or advanced content analysis.
    """
    try:
        return await _add_document_to_service(request)
    except Exception as e:
        logger.exception("Document addition failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _add_document_to_service(
    request: SimpleDocumentRequest,
) -> SimpleDocumentResponse:
    """Add document to vector database service."""
    # Get vector database service
    vector_db_service = await get_service("vector_db_service")

    # Process document (simplified)
    document_id = await vector_db_service.add_document(
        content=request.content,
        metadata=request.metadata,
        collection_name=request.collection_name,
    )

    return SimpleDocumentResponse(
        id=document_id,
        status="success",
        message="Document added successfully",
    )


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, Any]:
    """Get a document by ID."""
    try:
        document = await _get_document_from_service(document_id, collection_name)
    except Exception as e:
        logger.exception("Document retrieval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def _get_document_from_service(
    document_id: str, collection_name: str
) -> dict[str, Any] | None:
    """Get document from vector database service."""
    vector_db_service = await get_service("vector_db_service")
    return await vector_db_service.get_document(
        document_id=document_id,
        collection_name=collection_name,
    )


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, str]:
    """Delete a document by ID."""
    try:
        success = await _delete_document_from_service(document_id, collection_name)
    except Exception as e:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": "Document deleted successfully"}


async def _delete_document_from_service(document_id: str, collection_name: str) -> bool:
    """Delete document from vector database service."""
    vector_db_service = await get_service("vector_db_service")
    return await vector_db_service.delete_document(
        document_id=document_id,
        collection_name=collection_name,
    )


@router.get("/documents")
async def list_documents(
    collection_name: str = Query(default="documents", description="Collection name"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> dict[str, Any]:
    """List documents in a collection (simplified)."""
    try:
        return await _list_documents_from_service(collection_name, limit, offset)
    except Exception as e:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_documents_from_service(
    collection_name: str, limit: int, offset: int
) -> dict[str, Any]:
    """List documents from vector database service."""
    vector_db_service = await get_service("vector_db_service")

    documents = await vector_db_service.list_documents(
        collection_name=collection_name,
        limit=limit,
        offset=offset,
    )

    return {
        "documents": documents,
        "count": len(documents),
        "limit": limit,
        "offset": offset,
    }


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """List available collections."""
    try:
        return await _list_collections_from_service()
    except Exception as e:
        logger.exception("Collection listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_collections_from_service() -> dict[str, Any]:
    """List collections from vector database service."""
    vector_db_service = await get_service("vector_db_service")
    collections = await vector_db_service.list_collections()

    return {
        "collections": collections,
        "count": len(collections),
    }


@router.get("/documents/health")
async def documents_health() -> dict[str, Any]:
    """Get documents service health status."""
    try:
        return await _check_documents_health()
    except Exception as e:
        logger.exception("Documents health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _check_documents_health() -> dict[str, Any]:
    """Check documents service health."""
    vector_db_service = await get_service("vector_db_service")
    collections = await vector_db_service.list_collections()

    return {
        "status": "healthy",
        "service_type": "simple",
        "collections_count": len(collections),
    }
