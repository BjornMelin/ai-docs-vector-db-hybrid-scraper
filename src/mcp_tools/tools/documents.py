"""Document management tools for MCP server."""

import asyncio
import json
import logging
from collections.abc import Mapping
from typing import Any, cast
from uuid import uuid4

from fastmcp import Context
from langchain_core.documents import Document

from src.config.models import ChunkingConfig, ChunkingStrategy
from src.mcp_tools.models.requests import BatchRequest, DocumentRequest
from src.mcp_tools.models.responses import AddDocumentResponse, DocumentBatchResponse
from src.security.ml_security import MLSecurityValidator
from src.services.cache.manager import CacheManager
from src.services.crawling.normalization import (
    normalize_crawler_output,
    resolve_chunk_inputs,
)
from src.services.document_chunking import chunk_to_documents, infer_document_kind
from src.services.vector_db.document_builder import (
    build_params_from_crawl,
    build_text_documents,
)
from src.services.vector_db.service import VectorStoreService
from src.services.vector_db.types import CollectionSchema, TextDocument


logger = logging.getLogger(__name__)


def _coerce_add_document_response(value: Any) -> AddDocumentResponse | None:
    """Convert cached payloads into :class:`AddDocumentResponse` instances."""

    if isinstance(value, AddDocumentResponse):
        return value

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None

    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return AddDocumentResponse(**cast(Mapping[str, Any], value))

    return None


def _raise_scrape_error(url: str) -> None:
    """Raise ValueError for scraping failure."""

    msg = f"Failed to scrape {url}"
    raise ValueError(msg)


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
            content=crawl_result.get("content") or {},
            url=request.url,
            title=crawl_result.get("title")
            or crawl_result.get("metadata", {}).get("title", ""),
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
    crawl_manager: Any | None,
    content_service: Any | None = None,
) -> tuple[dict[str, Any], Any | None]:
    """Scrape the target URL and optionally enrich the content."""

    if crawl_manager is None:
        msg = (
            "UnifiedBrowserManager is unavailable; ensure browser features are enabled"
        )
        await ctx.error(msg)
        raise RuntimeError(msg)

    resolved_crawl_manager = cast(Any, crawl_manager)

    await ctx.debug(f"Scraping URL for document {doc_id} via UnifiedBrowserManager")
    raw_result = await resolved_crawl_manager.scrape_url(request.url)
    crawl_result = normalize_crawler_output(raw_result, fallback_url=request.url)

    has_chunkable_content = False
    content_block = crawl_result.get("content")
    if isinstance(content_block, Mapping):
        has_chunkable_content = any(
            isinstance(content_block.get(key), str) and content_block[key].strip()
            for key in ("markdown", "html", "text")
        )
    elif isinstance(content_block, str):
        has_chunkable_content = bool(content_block.strip())

    if not crawl_result.get("success") or not has_chunkable_content:
        await ctx.error(f"Failed to scrape {request.url}")
        _raise_scrape_error(request.url)

    content_intelligence = content_service
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
) -> list[Document]:
    """Chunk the scraped document using the configured strategy."""

    chunk_config = ChunkingConfig(
        strategy=request.chunk_strategy,
        chunk_size=request.chunk_size,
        chunk_overlap=request.chunk_overlap,
        token_chunk_size=request.token_chunk_size,
        token_chunk_overlap=request.token_chunk_overlap,
        token_model=request.token_model,
        json_max_chars=request.json_max_chars,
        enable_semantic_html_segmentation=request.enable_semantic_html_segmentation,
        normalize_html_text=request.normalize_html_text,
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

    raw_content, metadata_payload, kind_hint = resolve_chunk_inputs(
        crawl_result,
        fallback_url=request.url,
    )

    documents = chunk_to_documents(
        raw_content,
        metadata_payload,
        infer_document_kind(
            metadata_payload,
            default=kind_hint or "text",
        ),
        chunk_config,
    )
    await ctx.debug(f"Created {len(documents)} chunks for document {doc_id}")
    return documents


def _build_text_documents(
    chunks: list[Document],
    crawl_result: dict[str, Any],
    request: DocumentRequest,
    enriched_content: Any | None,
    doc_id: str,
) -> list[TextDocument]:
    """Convert chunk data into TextDocument payloads."""

    if not chunks:
        msg = f"No chunks generated for {request.url}"
        raise ValueError(msg)

    params = build_params_from_crawl(
        crawl_result,
        fallback_url=request.url,
        tenant=request.collection or "default",
        doc_id=doc_id,
        enriched_content=enriched_content,
    )
    return build_text_documents(chunks, params)


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
    *,
    vector_service: VectorStoreService,
    cache_manager: CacheManager,
    crawl_manager: Any,
    content_intelligence_service: Any,
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
            service = vector_service
            resolved_cache = cache_manager

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

            await resolved_cache.set(
                cache_key, result.model_dump(mode="json"), ttl=86400
            )

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
