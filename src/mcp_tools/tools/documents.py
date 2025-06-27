"""Document management tools for MCP server."""

import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import uuid4


if TYPE_CHECKING:
    from fastmcp import Context
else:
    # Use a protocol for testing to avoid FastMCP import issues
    from typing import Protocol

    class Context(Protocol):
        async def info(self, msg: str) -> None: ...
        async def debug(self, msg: str) -> None: ...
        async def warning(self, msg: str) -> None: ...
        async def error(self, msg: str) -> None: ...


from src.config import ChunkingConfig, ChunkingStrategy

from ...chunking import DocumentChunker
from ...infrastructure.client_manager import ClientManager
from ...security import MLSecurityValidator as SecurityValidator
from ..models.requests import BatchRequest, DocumentRequest


logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register document management tools with the MCP server."""

    from ..models.responses import AddDocumentResponse, DocumentBatchResponse

    @mcp.tool()
    async def add_document(
        request: DocumentRequest, ctx: Context
    ) -> AddDocumentResponse:
        """
        Add a document to the vector database with smart chunking.

        Crawls the URL, applies the selected chunking strategy, generates
        embeddings, and stores in the specified collection.
        """
        doc_id = str(uuid4())
        await ctx.info(f"Processing document {doc_id}: {request.url}")

        try:
            # Validate URL using SecurityValidator
            security_validator = SecurityValidator.from_unified_config()
            validated_url = security_validator.validate_url(request.url)
            request.url = validated_url
            # Get services from client manager
            cache_manager = await client_manager.get_cache_manager()
            crawl_manager = await client_manager.get_crawl_manager()
            embedding_manager = await client_manager.get_embedding_manager()
            qdrant_service = await client_manager.get_qdrant_service()

            # Check cache for existing document
            cache_key = f"doc:{request.url}"
            cached = await cache_manager.get(cache_key)
            if cached:
                await ctx.debug(f"Document {doc_id} found in cache")
                return AddDocumentResponse(**cached)

            # Scrape the URL using 5-tier UnifiedBrowserManager
            await ctx.debug(
                f"Scraping URL for document {doc_id} via UnifiedBrowserManager"
            )
            crawl_result = await crawl_manager.scrape_url(request.url)
            if (
                not crawl_result
                or not crawl_result.get("success")
                or not crawl_result.get("content")
            ):

                def _raise_scrape_error():
                    raise ValueError(f"Failed to scrape {request.url}")

                await ctx.error(f"Failed to scrape {request.url}")
                _raise_scrape_error()

            # Apply Content Intelligence analysis
            enriched_content = None
            content_intelligence_service = (
                await client_manager.get_content_intelligence_service()
            )
            if content_intelligence_service:
                await ctx.debug(f"Analyzing content intelligence for document {doc_id}")
                try:
                    analysis_result = (
                        await content_intelligence_service.analyze_content(
                            content=crawl_result["content"],
                            url=request.url,
                            title=crawl_result.get("title")
                            or crawl_result["metadata"].get("title", ""),
                            raw_html=crawl_result.get("raw_html"),
                            confidence_threshold=0.7,
                        )
                    )
                    if analysis_result.success:
                        enriched_content = analysis_result.enriched_content
                        await ctx.info(
                            f"Content intelligence analysis completed for {doc_id}: "
                            f"type={enriched_content.classification.primary_type.value}, "
                            f"quality={enriched_content.quality_score.overall_score:.2f}"
                        )
                    else:
                        await ctx.warning(
                            f"Content intelligence analysis failed for {doc_id}"
                        )
                except Exception:
                    await ctx.warning(
                        f"Content intelligence analysis error for {doc_id}: {e}"
                    )
            else:
                await ctx.debug("Content Intelligence Service not available")

            # Configure chunking (enhanced with content intelligence insights)
            chunk_config = ChunkingConfig(
                strategy=request.chunk_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )

            # Optimize chunking strategy based on content type
            if enriched_content and enriched_content.classification:
                content_type = enriched_content.classification.primary_type.value
                if (
                    content_type in ["code", "reference"]
                    and request.chunk_strategy == ChunkingStrategy.BASIC
                ):
                    chunk_config.strategy = ChunkingStrategy.SEMANTIC
                    await ctx.debug(
                        f"Upgraded chunking strategy to SEMANTIC for {content_type} content"
                    )

            # Chunk the document
            await ctx.debug(
                f"Chunking document {doc_id} with strategy {chunk_config.strategy}"
            )
            chunker = DocumentChunker(chunk_config)
            chunks = chunker.chunk_content(
                content=crawl_result["content"],
                title=crawl_result["title"]
                or crawl_result["metadata"].get("title", ""),
                url=crawl_result["url"],
            )
            await ctx.debug(f"Created {len(chunks)} chunks for document {doc_id}")

            # Generate embeddings for chunks
            texts = [chunk["content"] for chunk in chunks]
            await ctx.debug(f"Generating embeddings for {len(texts)} chunks")
            embeddings_result = await embedding_manager.generate_embeddings(texts)
            embeddings = embeddings_result.embeddings

            # Prepare points for insertion
            points = []
            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=False)
            ):
                # Base payload
                payload = {
                    "content": chunk["content"],
                    "url": request.url,
                    "title": crawl_result["title"]
                    or crawl_result["metadata"].get("title", ""),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "tier_used": crawl_result.get("tier_used", "unknown"),
                    "quality_score": crawl_result.get("quality_score", 0.0),
                    **chunk.get("metadata", {}),
                }

                # Enhance with content intelligence metadata if available
                if enriched_content:
                    payload.update(
                        {
                            "content_type": enriched_content.classification.primary_type.value,
                            "content_confidence": enriched_content.classification.confidence_scores.get(
                                enriched_content.classification.primary_type, 0.0
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

                    # Add secondary content types if available
                    if enriched_content.classification.secondary_types:
                        payload["secondary_content_types"] = [
                            ct.value
                            for ct in enriched_content.classification.secondary_types
                        ]
                else:
                    payload["content_intelligence_analyzed"] = False

                point = {
                    "id": str(uuid4()),
                    "vector": embedding,
                    "payload": payload,
                }
                points.append(point)

            # Ensure collection exists
            await qdrant_service.create_collection(
                collection_name=request.collection,
                vector_size=len(embeddings[0]),
                distance="Cosine",
                sparse_vector_name="sparse"
                if request.chunk_strategy != ChunkingStrategy.BASIC
                else None,
                enable_quantization=True,
            )

            # Insert points
            await qdrant_service.upsert_points(
                collection_name=request.collection,
                points=points,
            )

            # Prepare response with content intelligence insights
            response_kwargs = {
                "url": request.url,
                "title": crawl_result["title"]
                or crawl_result["metadata"].get("title", ""),
                "chunks_created": len(chunks),
                "collection": request.collection,
                "chunking_strategy": request.chunk_strategy.value,
                "embedding_dimensions": len(embeddings[0]),
            }

            # Add content intelligence metadata to response if available
            if enriched_content:
                response_kwargs.update(
                    {
                        "content_type": enriched_content.classification.primary_type.value,
                        "quality_score": enriched_content.quality_score.overall_score,
                        "content_intelligence_analyzed": True,
                    }
                )

            result = AddDocumentResponse(**response_kwargs)

            # Cache result
            await cache_manager.set(cache_key, result.model_dump(), ttl=86400)

            # Enhanced completion message with content intelligence info
            completion_msg = (
                f"Document {doc_id} processed successfully: "
                f"{len(chunks)} chunks created in collection {request.collection}"
            )
            if enriched_content:
                completion_msg += (
                    f" (type: {enriched_content.classification.primary_type.value}, "
                    f"quality: {enriched_content.quality_score.overall_score:.2f})"
                )
            await ctx.info(completion_msg)

        except Exception:
            await ctx.error(f"Failed to process document {doc_id}: {e}")
            logger.exception("Failed to add document")
            raise
        else:
            return result

    @mcp.tool()
    async def add_documents_batch(
        request: BatchRequest, ctx: Context
    ) -> DocumentBatchResponse:
        """
        Add multiple documents in batch with optimized processing.

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
                    security_validator = SecurityValidator.from_unified_config()
                    validated_url = security_validator.validate_url(url)

                    doc_request = DocumentRequest(
                        url=validated_url,
                        collection=request.collection,
                    )
                    result = await add_document(doc_request, ctx)
                    successes.append(result)
                except Exception:
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
