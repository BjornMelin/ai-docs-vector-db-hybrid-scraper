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

    except Exception as e:
        logger.exception("Document addition failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


def _raise_document_not_found():
    raise HTTPException(status_code=404, detail="Document not found")


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, Any]:
    """Get a document by ID."""
    try:
        vector_db_service = await get_service("vector_db_service")

        document = await vector_db_service.get_document(
            document_id=document_id,
            collection_name=collection_name,
        )

        if not document:
            _raise_document_not_found()
        else:
            return document

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document retrieval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, str]:
    """Delete a document by ID."""
    try:
        vector_db_service = await get_service("vector_db_service")

        success = await vector_db_service.delete_document(
            document_id=document_id,
            collection_name=collection_name,
        )

        if not success:
            _raise_document_not_found()
        else:
            return {"status": "success", "message": "Document deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents")
async def list_documents(
    collection_name: str = Query(default="documents", description="Collection name"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> dict[str, Any]:
    """List documents in a collection (simplified)."""
    try:
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

    except Exception as e:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/collections")
async def list_collections() -> dict[str, Any]:
    """List available collections."""
    try:
        vector_db_service = await get_service("vector_db_service")

        collections = await vector_db_service.list_collections()

        return {
            "collections": collections,
            "count": len(collections),
        }

    except Exception as e:
        logger.exception("Collection listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/documents/health")
async def documents_health() -> dict[str, Any]:
    """Get documents service health status."""
    try:
        vector_db_service = await get_service("vector_db_service")

        # Simple health check
        collections = await vector_db_service.list_collections()

        return {
            "status": "healthy",
            "service_type": "simple",
            "collections_count": len(collections),
        }

    except Exception as e:
        logger.exception("Documents health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }
