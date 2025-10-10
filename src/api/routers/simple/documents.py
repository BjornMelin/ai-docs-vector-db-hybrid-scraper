"""Simple mode documents API router."""

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_vector_service_dependency
from src.services.vector_db.service import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()


VectorServiceDependency = Annotated[
    VectorStoreService, Depends(get_vector_service_dependency)
]


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
async def add_document(
    request: SimpleDocumentRequest,
    vector_service: VectorServiceDependency,
) -> SimpleDocumentResponse:
    """Add a single document to the collection.

    This endpoint provides basic document indexing without advanced features
    like batch processing or advanced content analysis.
    """

    try:
        return await _add_document_to_service(request, vector_service)
    except Exception as e:
        logger.exception("Document addition failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _add_document_to_service(
    request: SimpleDocumentRequest, vector_db_service: VectorStoreService
) -> SimpleDocumentResponse:
    """Add document to vector database service."""

    document_id = await vector_db_service.add_document(
        request.collection_name,
        request.content,
        metadata=request.metadata or None,
    )

    return SimpleDocumentResponse(
        id=document_id,
        status="success",
        message="Document added successfully",
    )


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    vector_service: VectorServiceDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, Any]:
    """Get a document by ID."""

    try:
        document = await _get_document_from_service(
            document_id, collection_name, vector_service
        )
    except Exception as e:
        logger.exception("Document retrieval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def _get_document_from_service(
    document_id: str, collection_name: str, vector_db_service: VectorStoreService
) -> dict[str, Any] | None:
    """Get document from vector database service."""

    document = await vector_db_service.get_document(collection_name, document_id)
    return dict(document) if document else None


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    vector_service: VectorServiceDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, str]:
    """Delete a document by ID."""

    try:
        success = await _delete_document_from_service(
            document_id, collection_name, vector_service
        )
    except Exception as e:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": "Document deleted successfully"}


async def _delete_document_from_service(
    document_id: str, collection_name: str, vector_db_service: VectorStoreService
) -> bool:
    """Delete document from vector database service."""

    return await vector_db_service.delete_document(collection_name, document_id)


@router.get("/documents")
async def list_documents(
    vector_service: VectorServiceDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    offset: str | None = Query(
        default=None, description="Opaque pagination offset token"
    ),
) -> dict[str, Any]:
    """List documents in a collection (simplified)."""

    try:
        return await _list_documents_from_service(
            collection_name, limit, offset, vector_service
        )
    except Exception as e:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_documents_from_service(
    collection_name: str,
    limit: int,
    offset: str | None,
    vector_db_service: VectorStoreService,
) -> dict[str, Any]:
    """List documents from vector database service."""

    documents, next_offset = await vector_db_service.list_documents(
        collection_name,
        limit=limit,
        offset=offset,
    )
    normalized_docs = [dict(document) for document in documents]

    return {
        "documents": normalized_docs,
        "count": len(normalized_docs),
        "limit": limit,
        "next_offset": next_offset,
    }


@router.get("/collections")
async def list_collections(
    vector_service: VectorServiceDependency,
) -> dict[str, Any]:
    """List available collections."""

    try:
        return await _list_collections_from_service(vector_service)
    except Exception as e:
        logger.exception("Collection listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_collections_from_service(
    vector_db_service: VectorStoreService,
) -> dict[str, Any]:
    """List collections from vector database service."""

    collections = await vector_db_service.list_collections()

    return {
        "collections": collections,
        "count": len(collections),
    }
