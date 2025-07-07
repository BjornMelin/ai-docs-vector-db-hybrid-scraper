"""Simple mode search service implementation.

Optimized for solo developer use with minimal dependencies and complexity.
Focuses on core vector search functionality without advanced features.
"""

import logging
from typing import Any

from src.architecture.features import conditional_feature
from src.architecture.service_factory import BaseService
from src.models.vector_search import BasicSearchRequest, SearchResponse
from src.services.embeddings.manager import EmbeddingManager
from src.services.vector_db.service import QdrantService


logger = logging.getLogger(__name__)


class SimpleSearchService(BaseService):
    """Simplified search service for solo developer use.

    Features:
    - Basic vector search only
    - No advanced reranking or query expansion
    - Simple caching strategy
    - Minimal resource usage
    """

    def __init__(self, config=None, client_manager=None):
        super().__init__()
        # Import here to avoid circular imports
        from src.config import get_config  # noqa: PLC0415
        from src.infrastructure.client_manager import ClientManager  # noqa: PLC0415

        self.config = config or get_config()
        self.client_manager = client_manager or ClientManager()
        self.max_results = 25  # Reduced from enterprise 1000
        self.enable_reranking = False
        self.enable_hybrid_search = False
        self.enable_query_expansion = False
        self.enable_personalization = False
        self._cache: dict[str, Any] = {}  # Simple in-memory cache
        self.cache_size_limit = 100  # Simple cache size limit

    async def initialize(self) -> None:
        """Initialize the simple search service."""
        logger.info("Initializing simple search service")

        # Initialize basic vector search only
        try:
            self.vector_db = QdrantService(self.config, self.client_manager)
            await self.vector_db.initialize()

            self._mark_initialized()
            logger.info("Simple search service initialized successfully")

        except Exception:
            logger.exception("Failed to initialize simple search service")
            raise

    async def cleanup(self) -> None:
        """Clean up search service resources."""
        if hasattr(self, "vector_db"):
            await self.vector_db.cleanup()

        self._cache.clear()
        self._mark_cleanup()
        logger.info("Simple search service cleaned up")

    def get_service_name(self) -> str:
        """Get the service name."""
        return "simple_search_service"

    async def search(self, request: BasicSearchRequest) -> SearchResponse:
        """Perform basic vector search.

        Args:
            request: Search request

        Returns:
            Search response with results
        """
        # Check cache first
        cache_key = f"search:{hash(request.query)}"
        if cache_key in self._cache:
            logger.debug("Cache hit for query: %s...", request.query[:50])
            return self._cache[cache_key]

        # Limit results for simple mode
        limited_request = request.model_copy()
        limited_request.limit = min(request.limit, self.max_results)

        try:
            # Perform basic vector search
            results = await self._perform_vector_search(limited_request)

            response = SearchResponse(
                query=request.query,
                results=results,
                total_count=len(results),
                search_type="vector",
                processing_time_ms=0,  # Simplified - no timing
            )

            # Cache result (with size limit)
            self._cache_result(cache_key, response)

        except Exception as e:
            logger.exception("Search failed")
            return SearchResponse(
                query=request.query,
                results=[],
                total_count=0,
                search_type="vector",
                processing_time_ms=0,
                error=str(e),
            )

        else:
            return response

    async def _perform_vector_search(
        self, request: BasicSearchRequest
    ) -> list[dict[str, Any]]:
        """Perform basic vector search without advanced features."""
        # Generate embedding for query

        embedding_manager = EmbeddingManager(self.config, self.client_manager)
        embeddings = await embedding_manager.generate_embeddings([request.query])
        query_embedding = embeddings[0]  # Get first (and only) embedding

        # Perform vector search using filtered search with empty filters
        return await self.vector_db.filtered_search(
            collection_name=request.collection_name,
            query_vector=query_embedding,
            filters={},  # No filters for simple search
            limit=request.limit,
        )

    def _cache_result(self, key: str, response: SearchResponse) -> None:
        """Cache search result with simple LRU-like behavior."""
        # Simple cache size management
        if len(self._cache) >= self.cache_size_limit:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[key] = response

    @conditional_feature("enable_hybrid_search", fallback_value=[])
    async def hybrid_search(self, _request: BasicSearchRequest) -> list[dict[str, Any]]:
        """Hybrid search - disabled in simple mode."""
        return []

    @conditional_feature("enable_query_expansion", fallback_value=None)
    async def expand_query(self, query: str) -> str:
        """Query expansion - disabled in simple mode."""
        return query

    @conditional_feature("enable_reranking", fallback_value=None)
    async def rerank_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Result reranking - disabled in simple mode."""
        return results

    def get_search_stats(self) -> dict[str, Any]:
        """Get simple search statistics."""
        return {
            "service_type": "simple",
            "cache_size": len(self._cache),
            "max_results": self.max_results,
            "features": {
                "reranking": self.enable_reranking,
                "hybrid_search": self.enable_hybrid_search,
                "query_expansion": self.enable_query_expansion,
                "personalization": self.enable_personalization,
            },
        }
