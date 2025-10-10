"""Document management tools for MCP server."""

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from fastmcp import Context

from src.chunking import DocumentChunker
from src.config.models import ChunkingConfig, ChunkingStrategy
from src.mcp_tools.models.requests import BatchRequest, DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse, DocumentBatchResponse
from src.security.ml_security import MLSecurityValidator
from src.services.cache.manager import CacheManager
from src.services.dependencies import (
    get_cache_manager,
    get_content_intelligence_service,
    get_crawl_manager,
    get_vector_store_service,
)
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import CollectionSchema, TextDocument


logger = logging.getLogger(__name__)


def _coerce_add_document_response(value: Any) -> AddDocumentResponse | None:
    """Convert cached payloads into :class:`AddDocumentResponse` instances."""

    if isinstance(value, AddDocumentResponse):
        return value

    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return AddDocumentResponse(**cast(Mapping[str, Any], value))

    return None


def _raise_scrape_error(url: str) -> None:
    """Raise ValueError for scraping failure."""

    msg = f"Failed to scrape {url}"
    raise ValueError(msg)


async def _resolve_vector_service(
    vector_service: VectorStoreService | None = None,
) -> VectorStoreService:
    """Return an initialized VectorStoreService instance."""

    service = vector_service or await get_vector_store_service()
    if hasattr(service, "is_initialized") and not service.is_initialized():
        initializer = getattr(service, "initialize", None)
        if callable(initializer):
            result = initializer()
            if asyncio.iscoroutine(result):
                await result
    return service


async def _resolve_cache_manager(
    cache_manager: CacheManager | None = None,
) -> CacheManager:
    """Return the cache manager instance, resolving from the container when needed."""

    resolved = await get_cache_manager(cache_manager)
    if not isinstance(resolved, CacheManager):
        raise RuntimeError("Resolved cache manager has unexpected type")
    return resolved


async def _run_content_intelligence(
    service: Any,
    crawl_result: dict[str, Any],
    request: DocumentRequest,
    doc_id: str,
    ctx: Context,
):
    """Execute content intelligence analysis when the service is available."""

    if not service:
        await ctx.debug("Content Intelligence Service not available")
        return None

    try:
        analysis_result = await service.analyze_content(
            content=crawl_result["content"],
            url=request.url,
            title=crawl_result.get("title")
            or crawl_result["metadata"].get("title", ""),
            raw_html=crawl_result.get("raw_html"),
            confidence_threshold=0.7,
        )
    except (asyncio.CancelledError, TimeoutError, RuntimeError) as exc:
        await ctx.warning(f"Content intelligence analysis error for {doc_id}: {exc}")
        return None

    if not getattr(analysis_result, "success", False):
        await ctx.warning(f"Content intelligence analysis failed for {doc_id}")
        return None

    enriched_content = getattr(analysis_result, "enriched_content", None)
    if enriched_content is None:
        await ctx.warning(
            f"Content intelligence returned no content for {doc_id} despite success"
        )
        return None

    await ctx.info(
        f"Content intelligence analysis completed for {doc_id}: "
        f"type={enriched_content.classification.primary_type.value}, "
        f"quality={enriched_content.quality_score.overall_score:.2f}"
    )
    return enriched_content


async def _scrape_document(
    request: DocumentRequest,
    doc_id: str,
    ctx: Context,
    *,
    crawl_manager: Any | None = None,
    content_service: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Scrape the target URL and optionally enrich the content."""

    resolved_crawl_manager = await get_crawl_manager(crawl_manager)
    resolved_crawl_manager = cast(Any, resolved_crawl_manager)

    await ctx.debug(f"Scraping URL for document {doc_id} via UnifiedBrowserManager")
    crawl_result = await resolved_crawl_manager.scrape_url(request.url)
    if (
        not crawl_result
        or not crawl_result.get("success")
        or not crawl_result.get("content")
    ):
        await ctx.error(f"Failed to scrape {request.url}")
        _raise_scrape_error(request.url)

    content_intelligence = content_service
    try:
        content_intelligence = await get_content_intelligence_service(
            content_intelligence
        )
    except Exception as exc:  # pragma: no cover - optional component
        logger.info(
            "Content intelligence unavailable for %s: %s", doc_id, exc, exc_info=exc
        )
        content_intelligence = None
    enriched_content = await _run_content_intelligence(
        content_intelligence,
        crawl_result,
        request,
        doc_id,
        ctx,
    )
    return crawl_result, enriched_content


async def _chunk_document(
    request: DocumentRequest,
    crawl_result: dict[str, Any],
    enriched_content: Any | None,
    doc_id: str,
    ctx: Context,
) -> list[dict[str, Any]]:
    """Chunk the scraped document using the configured strategy."""

    chunk_config = ChunkingConfig(
        strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
    )

    if enriched_content and enriched_content.classification:
        content_type = enriched_content.classification.primary_type.value
        if (
            content_type in {"code", "reference"}
            and request.chunk_strategy == ChunkingStrategy.BASIC
        ):
            chunk_config.strategy = ChunkingStrategy.ENHANCED
            await ctx.debug(
                f"Upgraded chunking strategy to ENHANCED for {content_type} content"
            )

    chunker = DocumentChunker(chunk_config)
    chunks = chunker.chunk_content(
        content=crawl_result["content"],
        title=crawl_result.get("title") or crawl_result["metadata"].get("title", ""),
        url=crawl_result.get("url", request.url),
    )
    await ctx.debug(f"Created {len(chunks)} chunks for document {doc_id}")
    return chunks


def _build_text_documents(
    chunks: list[dict[str, Any]],
    crawl_result: dict[str, Any],
    request: DocumentRequest,
    enriched_content: Any | None,
    doc_id: str,
) -> list[TextDocument]:
    """Convert chunk data into TextDocument payloads."""

    if not chunks:
        msg = f"No chunks generated for {request.url}"
        raise ValueError(msg)

    documents: list[TextDocument] = []
    total_chunks = len(chunks)
    base_title = crawl_result.get("title") or crawl_result["metadata"].get("title", "")

    current_time = datetime.now(UTC).isoformat()
    for index, chunk in enumerate(chunks):
        payload: dict[str, Any] = {
            "content": chunk["content"],
            "url": request.url,
            "title": base_title,
            "chunk_index": index,
            "total_chunks": total_chunks,
            "tier_used": crawl_result.get("tier_used", "unknown"),
            "quality_score": crawl_result.get("quality_score", 0.0),
            **chunk.get("metadata", {}),
        }

        payload.setdefault("doc_id", doc_id)
        payload["chunk_id"] = payload.get("chunk_id", index)
        payload.setdefault("tenant", request.collection or "default")
        payload.setdefault("source", payload.get("source") or request.url)
        payload.setdefault("created_at", current_time)

        if enriched_content:
            payload.update(
                {
                    "content_type": (
                        enriched_content.classification.primary_type.value
                    ),
                    "content_confidence": (
                        enriched_content.classification.confidence_scores.get(
                            enriched_content.classification.primary_type, 0.0
                        )
                    ),
                    "quality_overall": enriched_content.quality_score.overall_score,
                    "quality_completeness": enriched_content.quality_score.completeness,
                    "quality_relevance": enriched_content.quality_score.relevance,
                    "quality_confidence": enriched_content.quality_score.confidence,
                    "ci_word_count": enriched_content.metadata.word_count,
                    "ci_char_count": enriched_content.metadata.char_count,
                    "ci_language": enriched_content.metadata.language,
                    "ci_semantic_tags": enriched_content.metadata.semantic_tags,
                    "content_intelligence_analyzed": True,
                }
            )
            if enriched_content.classification.secondary_types:
                payload["secondary_content_types"] = [
                    content_type.value
                    for content_type in enriched_content.classification.secondary_types
                ]
        else:
            payload["content_intelligence_analyzed"] = False

        documents.append(
            TextDocument(
                id=str(uuid4()),
                content=chunk["content"],
                metadata=payload,
            )
        )

    return documents


async def _persist_documents(
    vector_service: VectorStoreService,
    collection: str,
    strategy: ChunkingStrategy,
    documents_to_upsert: list[TextDocument],
) -> None:
    """Ensure collection existence and persist documents."""

    schema = CollectionSchema(
        name=collection,
        vector_size=vector_service.embedding_dimension,
        distance="cosine",
        requires_sparse=(strategy != ChunkingStrategy.BASIC),
    )
    await vector_service.ensure_collection(schema)
    await vector_service.upsert_documents(collection, documents_to_upsert)


def _build_ingestion_response(
    request: DocumentRequest,
    crawl_result: dict[str, Any],
    chunk_count: int,
    vector_service: VectorStoreService,
    enriched_content: Any | None,
) -> AddDocumentResponse:
    """Create the structured response for the ingestion flow."""

    response_kwargs: dict[str, Any] = {
        "url": request.url,
        "title": crawl_result.get("title") or crawl_result["metadata"].get("title", ""),
        "chunks_created": chunk_count,
        "collection": request.collection,
        "chunking_strategy": request.chunk_strategy.value,
        "embedding_dimensions": vector_service.embedding_dimension,
    }

    if enriched_content:
        response_kwargs.update(
            {
                "content_type": (enriched_content.classification.primary_type.value),
                "quality_score": enriched_content.quality_score.overall_score,
                "content_intelligence_analyzed": True,
            }
        )

    return AddDocumentResponse(**response_kwargs)


def register_tools(
    mcp,
    vector_service: VectorStoreService | None = None,
    cache_manager: CacheManager | None = None,
    crawl_manager: Any | None = None,
    content_intelligence_service: Any | None = None,
) -> None:
    """Register document management tools with the MCP server."""

    @mcp.tool()
    async def add_document(
        request: DocumentRequest, ctx: Context
    ) -> AddDocumentResponse:
        """Add a document to the vector database with smart chunking.

        Crawls the URL, applies the selected chunking strategy, generates
        embeddings, and stores in the specified collection.
        """
        doc_id = str(uuid4())
        await ctx.info(f"Processing document {doc_id}: {request.url}")

        try:
            service = await _resolve_vector_service(vector_service)
            resolved_cache = await _resolve_cache_manager(cache_manager)

            request.url = MLSecurityValidator.from_unified_config().validate_url(
                request.url
            )

            cache_key = f"doc:{request.url}"
            cached_value = await resolved_cache.get(cache_key)
            cached_response = _coerce_add_document_response(cached_value)
            if cached_response is not None:
                await ctx.debug(f"Document {doc_id} found in cache")
                return cached_response

            crawl_result, enriched_content = await _scrape_document(
                request,
                doc_id,
                ctx,
                crawl_manager=crawl_manager,
                content_service=content_intelligence_service,
            )

            chunks = await _chunk_document(
                request,
                crawl_result,
                enriched_content,
                doc_id,
                ctx,
            )
            await _persist_documents(
                service,
                request.collection,
                request.chunk_strategy,
                _build_text_documents(
                    chunks,
                    crawl_result,
                    request,
                    enriched_content,
                    doc_id,
                ),
            )

            result = _build_ingestion_response(
                request,
                crawl_result,
                len(chunks),
                service,
                enriched_content,
            )

            await resolved_cache.set(cache_key, result.model_dump(), ttl=86400)

            message = (
                f"Document {doc_id} processed successfully: "
                f"{len(chunks)} chunks created in collection {request.collection}"
            )
            result_content_type = getattr(result, "content_type", None)
            if result_content_type:
                message += (
                    f" (type: {result_content_type}, "
                    f"quality: {getattr(result, 'quality_score', 0.0):.2f})"
                )
            await ctx.info(message)
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors to MCP clients
            await ctx.error(f"Failed to process document {doc_id}: {exc}")
            logger.exception("Failed to add document")
            raise
        return result

    @mcp.tool()
    async def add_documents_batch(
        request: BatchRequest, ctx: Context
    ) -> DocumentBatchResponse:
        """Add multiple documents in batch with optimized processing.

        Processes multiple URLs concurrently with rate limiting and
        progress tracking.
        """

        successes: list[AddDocumentResponse] = []
        failures: list[str] = []
        total_urls = len(request.urls)

        # Process URLs in batches
        semaphore = asyncio.Semaphore(request.max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                try:
                    # Validate URL first
                    security_validator = MLSecurityValidator.from_unified_config()
                    validated_url = security_validator.validate_url(url)

                    doc_request = DocumentRequest(
                        url=validated_url,
                        collection=request.collection,
                    )
                    result = await add_document(doc_request, ctx)
                    successes.append(result)
                except (ConnectionError, OSError, PermissionError):
                    failures.append(url)

        # Process all URLs concurrently
        await asyncio.gather(
            *[process_url(url) for url in request.urls],
            return_exceptions=True,
        )

        successes.sort(key=lambda x: x.chunks_created, reverse=True)
        return DocumentBatchResponse(
            successful=successes,
            failed=failures,
            total=total_urls,
        )
