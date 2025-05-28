"""Document management tools for MCP server."""

import asyncio
import logging
from typing import Any
from uuid import uuid4

from fastmcp import Context

from ...chunking import ChunkingConfig
from ...chunking import EnhancedChunker
from ...config.enums import ChunkingStrategy
from ...infrastructure.client_manager import ClientManager
from ...security import SecurityValidator
from ..models.requests import BatchRequest
from ..models.requests import DocumentRequest

logger = logging.getLogger(__name__)


def register_tools(mcp, client_manager: ClientManager):
    """Register document management tools with the MCP server."""

    @mcp.tool()
    async def add_document(request: DocumentRequest, ctx: Context) -> dict[str, Any]:
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
            # Check cache for existing document
            cache_key = f"doc:{request.url}"
            cached = await client_manager.cache_manager.get(cache_key)
            if cached:
                await ctx.debug(f"Document {doc_id} found in cache")
                return cached

            # Crawl the URL
            await ctx.debug(f"Crawling URL for document {doc_id}")
            crawl_result = await client_manager.crawl_manager.crawl_single(request.url)
            if not crawl_result or not crawl_result.markdown:
                await ctx.error(f"Failed to crawl {request.url}")
                raise ValueError(f"Failed to crawl {request.url}")

            # Configure chunking
            chunk_config = ChunkingConfig(
                strategy=request.chunk_strategy,
                chunk_size=request.chunk_size,
                chunk_overlap=request.chunk_overlap,
            )

            # Chunk the document
            await ctx.debug(
                f"Chunking document {doc_id} with strategy {request.chunk_strategy}"
            )
            chunker = EnhancedChunker(chunk_config)
            chunks = chunker.chunk_content(
                content=crawl_result.markdown,
                title=crawl_result.metadata.get("title", ""),
                url=crawl_result.metadata.get("url", request.url),
            )
            await ctx.debug(f"Created {len(chunks)} chunks for document {doc_id}")

            # Generate embeddings for chunks
            texts = [chunk["content"] for chunk in chunks]
            await ctx.debug(f"Generating embeddings for {len(texts)} chunks")
            embeddings = await client_manager.embedding_manager.generate_embeddings(
                texts
            )

            # Prepare points for insertion
            points = []
            for i, (chunk, embedding) in enumerate(
                zip(chunks, embeddings, strict=False)
            ):
                point = {
                    "id": str(uuid4()),
                    "vector": embedding,
                    "payload": {
                        "content": chunk["content"],
                        "url": request.url,
                        "title": crawl_result.metadata.get("title", ""),
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        **chunk.get("metadata", {}),
                    },
                }
                points.append(point)

            # Ensure collection exists
            await client_manager.qdrant_service.create_collection(
                collection_name=request.collection,
                vector_size=len(embeddings[0]),
                distance="Cosine",
                sparse_vector_name="sparse"
                if request.chunk_strategy != ChunkingStrategy.BASIC
                else None,
                enable_quantization=True,
            )

            # Insert points
            await client_manager.qdrant_service.upsert_points(
                collection_name=request.collection,
                points=points,
            )

            # Prepare response
            result = {
                "url": request.url,
                "title": crawl_result.metadata.get("title", ""),
                "chunks_created": len(chunks),
                "collection": request.collection,
                "chunking_strategy": request.chunk_strategy.value,
                "embedding_dimensions": len(embeddings[0]),
            }

            # Cache result
            await client_manager.cache_manager.set(cache_key, result, ttl=86400)

            await ctx.info(
                f"Document {doc_id} processed successfully: "
                f"{len(chunks)} chunks created in collection {request.collection}"
            )

            return result

        except Exception as e:
            await ctx.error(f"Failed to process document {doc_id}: {e}")
            logger.error(f"Failed to add document: {e}")
            raise

    @mcp.tool()
    async def add_documents_batch(request: BatchRequest) -> dict[str, Any]:
        """
        Add multiple documents in batch with optimized processing.

        Processes multiple URLs concurrently with rate limiting and
        progress tracking.
        """
        results = {
            "successful": [],
            "failed": [],
            "total": len(request.urls),
        }

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
                    result = await add_document(doc_request)
                    results["successful"].append(result)
                except Exception as e:
                    results["failed"].append(
                        {
                            "url": url,
                            "error": str(e),
                        }
                    )

        # Process all URLs concurrently
        await asyncio.gather(
            *[process_url(url) for url in request.urls],
            return_exceptions=True,
        )

        return results
