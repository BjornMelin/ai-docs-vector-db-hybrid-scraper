"""Simple mode documents API router."""

import logging
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.architecture.service_factory import (
    ModeAwareServiceFactory,
    get_request_service_factory,
)
from src.services.vector_db import VectorStoreService


logger = logging.getLogger(__name__)

router = APIRouter()


FactoryDependency = Annotated[
    ModeAwareServiceFactory, Depends(get_request_service_factory)
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
    factory: FactoryDependency,
) -> SimpleDocumentResponse:
    """Add a single document to the collection.

    This endpoint provides basic document indexing without advanced features
    like batch processing or advanced content analysis.
    """
    try:
        return await _add_document_to_service(request, factory)
    except Exception as e:
        logger.exception("Document addition failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _add_document_to_service(
    request: SimpleDocumentRequest, factory: ModeAwareServiceFactory
) -> SimpleDocumentResponse:
    """Add document to vector database service."""
    vector_db_service = await _get_vector_store_service(factory)

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
    factory: FactoryDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, Any]:
    """Get a document by ID."""
    try:
        document = await _get_document_from_service(
            document_id, collection_name, factory
        )
    except Exception as e:
        logger.exception("Document retrieval failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    return document


async def _get_document_from_service(
    document_id: str, collection_name: str, factory: ModeAwareServiceFactory
) -> dict[str, Any] | None:
    """Get document from vector database service."""
    vector_db_service = await _get_vector_store_service(factory)
    document = await vector_db_service.get_document(collection_name, document_id)
    return dict(document) if document else None


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    factory: FactoryDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
) -> dict[str, str]:
    """Delete a document by ID."""
    try:
        success = await _delete_document_from_service(
            document_id, collection_name, factory
        )
    except Exception as e:
        logger.exception("Document deletion failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "success", "message": "Document deleted successfully"}


async def _delete_document_from_service(
    document_id: str, collection_name: str, factory: ModeAwareServiceFactory
) -> bool:
    """Delete document from vector database service."""
    vector_db_service = await _get_vector_store_service(factory)
    return await vector_db_service.delete_document(collection_name, document_id)


@router.get("/documents")
async def list_documents(
    factory: FactoryDependency,
    collection_name: str = Query(default="documents", description="Collection name"),
    limit: int = Query(default=10, ge=1, le=50, description="Maximum results"),
    offset: str | None = Query(
        default=None, description="Opaque pagination offset token"
    ),
) -> dict[str, Any]:
    """List documents in a collection (simplified)."""
    try:
        return await _list_documents_from_service(
            collection_name, limit, offset, factory
        )
    except Exception as e:
        logger.exception("Document listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_documents_from_service(
    collection_name: str,
    limit: int,
    offset: str | None,
    factory: ModeAwareServiceFactory,
) -> dict[str, Any]:
    """List documents from vector database service."""
    vector_db_service = await _get_vector_store_service(factory)

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
async def list_collections(factory: FactoryDependency) -> dict[str, Any]:
    """List available collections."""
    try:
        return await _list_collections_from_service(factory)
    except Exception as e:
        logger.exception("Collection listing failed")
        raise HTTPException(status_code=500, detail=str(e)) from e


async def _list_collections_from_service(
    factory: ModeAwareServiceFactory,
) -> dict[str, Any]:
    """List collections from vector database service."""
    vector_db_service = await _get_vector_store_service(factory)
    collections = await vector_db_service.list_collections()

    return {
        "collections": collections,
        "count": len(collections),
    }


@router.get("/documents/health")
async def documents_health(factory: FactoryDependency) -> dict[str, Any]:
    """Get documents service health status."""
    try:
        return await _check_documents_health(factory)
    except Exception as e:
        logger.exception("Documents health check failed")
        return {
            "status": "unhealthy",
            "error": str(e),
        }


async def _check_documents_health(factory: ModeAwareServiceFactory) -> dict[str, Any]:
    """Check documents service health."""
    vector_db_service = await _get_vector_store_service(factory)
    collections = await vector_db_service.list_collections()

    return {
        "status": "healthy",
        "service_type": "simple",
        "collections_count": len(collections),
    }


async def _get_vector_store_service(
    factory: ModeAwareServiceFactory,
) -> VectorStoreService:
    """Resolve the vector store service with runtime type protection."""

    service = await factory.get_service("vector_db_service")
    if not isinstance(service, VectorStoreService):  # pragma: no cover - safety
        msg = "Vector DB service is not a VectorStoreService instance"
        raise TypeError(msg)
    return cast(VectorStoreService, service)
