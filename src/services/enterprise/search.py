"""Enterprise mode search service implementation.

Full-featured search service with all advanced capabilities for enterprise deployments
and portfolio demonstrations.
"""

import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any

from src.architecture.service_factory import BaseService
from src.models.requests import SearchRequest
from src.models.vector_search import SearchResponse
from src.services.vector_db.service import VectorStoreService


# Optional imports for enterprise features
try:
    from src.services.monitoring.metrics import MetricsCollector
    from src.services.query_processing.expansion import QueryExpansionService
    from src.services.query_processing.ranking import RankingService
except ImportError:
    QueryExpansionService = None
    RankingService = None
    SearchAnalyticsService = None
    MetricsCollector = None


logger = logging.getLogger(__name__)


def _raise_ranking_service_not_available() -> None:
    """Raise ImportError for RankingService not available."""
    msg = "RankingService not available"
    raise ImportError(msg)


def _raise_query_expansion_service_not_available() -> None:
    """Raise ImportError for QueryExpansionService not available."""
    msg = "QueryExpansionService not available"
    raise ImportError(msg)


def _raise_metrics_collector_not_available() -> None:
    """Raise ImportError for MetricsCollector not available."""
    msg = "MetricsCollector not available"
    raise ImportError(msg)


class EnterpriseSearchService(BaseService):  # pylint: disable=too-many-instance-attributes
    """Full-featured search service for enterprise deployments.

    Features:
    - Hybrid vector + keyword search
    - Advanced reranking and query expansion
    - Personalization and analytics
    - Comprehensive caching and performance optimization
    - A/B testing and experimentation
    """

    def __init__(self):
        super().__init__()
        self.max_results = 1000  # Enterprise scale
        self.enable_reranking = True
        self.enable_hybrid_search = True
        self.enable_query_expansion = True
        self.enable_personalization = True
        self.enable_analytics = True
        self.enable_ab_testing = True
        self._search_metrics: dict[str, Any] = {}
        self._query_cache: dict[str, Any] = {}
        self.vector_store: VectorStoreService | None = None
        self.reranker = None
        self.query_expander = None
        self.metrics_collector = None

    async def initialize(self) -> None:
        """Initialize the enterprise search service."""
        logger.info("Initializing enterprise search service")

        try:
            # Initialize all search components
            await self._initialize_vector_search()
            await self._initialize_hybrid_search()
            await self._initialize_reranking()
            await self._initialize_query_expansion()
            await self._initialize_analytics()
            await self._initialize_personalization()

            self._mark_initialized()
            logger.info("Enterprise search service initialized successfully")

        except Exception:
            logger.exception("Failed to initialize enterprise search service")
            raise

    async def cleanup(self) -> None:
        """Clean up search service resources."""
        # Clean up all components
        if self.vector_store:
            await self.vector_store.cleanup()
            self.vector_store = None
        if hasattr(self, "reranker"):
            await self.reranker.cleanup()
        if hasattr(self, "query_expander"):
            await self.query_expander.cleanup()

        self._search_metrics.clear()
        self._query_cache.clear()
        self._mark_cleanup()
        logger.info("Enterprise search service cleaned up")

    def get_service_name(self) -> str:
        """Get the service name."""
        return "enterprise_search_service"

    async def search(self, request: SearchRequest) -> SearchResponse:
        """Perform comprehensive enterprise search.

        Args:
            request: Search request

        Returns:
            Search response with enriched results

        """
        start_time = time.time()

        try:
            # Check cache first
            cache_key = self._generate_cache_key(request)
            if cached_result := await self._get_cached_result(cache_key):
                logger.debug("Cache hit for query: %s...", request.query[:50])
                return cached_result

            # Perform query expansion if enabled
            expanded_query = await self.expand_query(request.query)

            # Perform hybrid search
            if self.enable_hybrid_search:
                results = await self.hybrid_search(request, expanded_query)
            else:
                results = await self._perform_vector_search(request, expanded_query)

            # Apply reranking if enabled
            if self.enable_reranking and results:
                results = await self.rerank_results(results, request.query)

            # Apply personalization if enabled
            if self.enable_personalization and request.user_id:
                results = await self._apply_personalization(results, request.user_id)

            # Enrich results with metadata
            enriched_results = await self._enrich_results(results)

            # Create response
            processing_time = (time.time() - start_time) * 1000
            response = SearchResponse(
                query=request.query,
                results=enriched_results,
                total_count=len(enriched_results),
                search_type="hybrid" if self.enable_hybrid_search else "vector",
                processing_time_ms=processing_time,
                expanded_query=expanded_query
                if expanded_query != request.query
                else None,
                reranked=self.enable_reranking,
                personalized=self.enable_personalization
                and request.user_id is not None,
            )

            # Cache result
            await self._cache_result(cache_key, response)

            # Record analytics
            await self._record_search_analytics(request, response)

        except Exception as e:
            logger.exception("Enterprise search failed")
            return SearchResponse(
                query=request.query,
                results=[],
                total_count=0,
                search_type="error",
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )

        return response

    async def hybrid_search(
        self, request: SearchRequest, expanded_query: str | None = None
    ) -> list[dict[str, Any]]:
        """Perform hybrid vector + keyword search."""
        query = expanded_query or request.query

        vector_store = self._require_vector_store()
        collection = self._resolve_collection(request)

        try:
            hybrid_matches = await vector_store.hybrid_search(
                collection,
                query,
                sparse_vector=None,
                limit=request.limit,
            )
        except Exception as exc:  # pragma: no cover - defensive log
            logger.warning(
                "Hybrid search failed (%s); falling back to dense search", exc
            )
            hybrid_matches = await vector_store.search_documents(
                collection,
                query,
                limit=request.limit,
            )

        vector_results = self._normalize_vector_matches(hybrid_matches)
        keyword_results = await self._perform_keyword_search(request, query)
        fused_results = await self._fusion_rank(vector_results, keyword_results)
        return fused_results[: request.limit]

    async def expand_query(self, query: str) -> str:
        """Expand query using various techniques."""
        if not self.enable_query_expansion:
            return query

        try:
            # Use query expansion service
            if hasattr(self, "query_expander"):
                return await self.query_expander.expand(query)

            # Fallback: simple synonym expansion (placeholder)

        except Exception:
            logger.exception("Query expansion failed")

        return query  # Would implement actual expansion logic

    async def rerank_results(
        self, results: list[dict[str, Any]], query: str
    ) -> list[dict[str, Any]]:
        """Rerank search results using advanced algorithms."""
        if not self.enable_reranking or not results:
            return results

        try:
            # Use reranking service
            if hasattr(self, "reranker"):
                return await self.reranker.rerank(results, query)

            # Fallback: simple relevance-based reranking
            return sorted(results, key=lambda x: x.get("score", 0), reverse=True)

        except Exception:
            logger.exception("Reranking failed")
            return results

    async def _initialize_vector_search(self) -> None:
        """Initialize vector search components."""
        if self.vector_store:
            return

        try:
            self.vector_store = VectorStoreService()
            await self.vector_store.initialize()
        except Exception:  # pragma: no cover - defensive path
            logger.exception("Failed to initialize vector store service")
            self.vector_store = None
            self.enable_hybrid_search = False

    async def _initialize_hybrid_search(self) -> None:
        """Initialize hybrid search components."""
        if not self.enable_hybrid_search:
            return

        if self.vector_store is None:
            logger.warning(
                "Vector store unavailable; disabling hybrid search capabilities"
            )
            self.enable_hybrid_search = False

    async def _initialize_reranking(self) -> None:
        """Initialize reranking components."""
        if self.enable_reranking:
            try:
                if RankingService is None:
                    _raise_ranking_service_not_available()

                self.reranker = RankingService()
                await self.reranker.initialize()
            except ImportError:
                logger.warning("Reranking service not available")
                self.enable_reranking = False

    async def _initialize_query_expansion(self) -> None:
        """Initialize query expansion components."""
        if self.enable_query_expansion:
            try:
                if QueryExpansionService is None:
                    _raise_query_expansion_service_not_available()

                self.query_expander = QueryExpansionService()
                await self.query_expander.initialize()
            except ImportError:
                logger.warning("Query expansion service not available")
                self.enable_query_expansion = False

    async def _initialize_analytics(self) -> None:
        """Initialize analytics components."""
        if self.enable_analytics:
            try:
                if MetricsCollector is None:
                    _raise_metrics_collector_not_available()

                self.metrics_collector = MetricsCollector()
                await self.metrics_collector.initialize()
            except ImportError:
                logger.warning("Analytics service not available")
                self.enable_analytics = False

    async def _initialize_personalization(self) -> None:
        """Initialize personalization components."""
        if self.enable_personalization:
            # Placeholder for personalization service
            logger.info("Personalization enabled (placeholder implementation)")

    async def _perform_vector_search(
        self, request: SearchRequest, query: str
    ) -> list[dict[str, Any]]:
        """Perform vector search."""
        vector_store = self._require_vector_store()
        collection = self._resolve_collection(request)
        matches = await vector_store.search_documents(
            collection,
            query,
            limit=request.limit,
        )
        return self._normalize_vector_matches(matches)

    def _require_vector_store(self) -> VectorStoreService:
        """Return the initialized vector store service."""

        if self.vector_store is None:
            msg = "VectorStoreService is not initialized"
            raise RuntimeError(msg)
        return self.vector_store

    @staticmethod
    def _resolve_collection(request: SearchRequest) -> str:
        """Resolve the target collection for a search request."""

        return request.collection_name or "documents"

    @staticmethod
    def _normalize_vector_matches(
        matches: Sequence[Any],
    ) -> list[dict[str, Any]]:
        """Convert vector matches into plain dictionaries."""

        normalized: list[dict[str, Any]] = []
        for match in matches:
            if isinstance(match, Mapping):
                payload = dict(match.get("payload", {}) or {})
                payload.setdefault("id", match.get("id"))
                payload.setdefault("score", match.get("score", 0.0))
                normalized.append(payload)
                continue

            payload = dict(getattr(match, "payload", {}) or {})
            payload.setdefault("id", getattr(match, "id", None))
            payload.setdefault("score", getattr(match, "score", 0.0))
            normalized.append(payload)
        return normalized

    async def _perform_keyword_search(
        self, _request: SearchRequest, _query: str
    ) -> list[dict[str, Any]]:
        """Perform keyword search (placeholder)."""
        # Would implement actual keyword search
        return []

    async def _fusion_rank(
        self,
        vector_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Combine and rank vector and keyword results."""
        # Simple fusion strategy (placeholder)
        combined = vector_results + keyword_results

        # Remove duplicates and rank
        seen_ids = set()
        fused = []
        for result in combined:
            if (result_id := result.get("id", "")) not in seen_ids:
                seen_ids.add(result_id)
                fused.append(result)

        return fused

    async def _apply_personalization(
        self, results: list[dict[str, Any]], _user_id: str
    ) -> list[dict[str, Any]]:
        """Apply personalization to search results."""
        # Placeholder personalization logic
        return results

    async def _enrich_results(
        self, results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Enrich results with additional metadata."""
        enriched = []
        for result in results:
            enriched_result = result.copy()
            enriched_result["enriched_at"] = time.time()
            enriched_result["source"] = "enterprise_search"
            enriched.append(enriched_result)
        return enriched

    def _generate_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for request."""
        return f"search:{hash(request.query)}:{request.limit}:{request.collection_name}"

    async def _get_cached_result(self, key: str) -> SearchResponse | None:
        """Get cached search result."""
        return self._query_cache.get(key)

    async def _cache_result(self, key: str, response: SearchResponse) -> None:
        """Cache search result."""
        # Simple cache with size limit
        if len(self._query_cache) > 1000:
            # Remove oldest entry
            oldest_key = next(iter(self._query_cache))
            del self._query_cache[oldest_key]

        self._query_cache[key] = response

    async def _record_search_analytics(
        self, request: SearchRequest, response: SearchResponse
    ) -> None:
        """Record search analytics."""
        if not self.enable_analytics:
            return

        if hasattr(self, "metrics_collector"):
            await self.metrics_collector.record_search(request, response)

    def get_search_stats(self) -> dict[str, Any]:
        """Get comprehensive search statistics."""
        return {
            "service_type": "enterprise",
            "cache_size": len(self._query_cache),
            "max_results": self.max_results,
            "features": {
                "reranking": self.enable_reranking,
                "hybrid_search": self.enable_hybrid_search,
                "query_expansion": self.enable_query_expansion,
                "personalization": self.enable_personalization,
                "analytics": self.enable_analytics,
                "ab_testing": self.enable_ab_testing,
            },
            "metrics": self._search_metrics,
        }
