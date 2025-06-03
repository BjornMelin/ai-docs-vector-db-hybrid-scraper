"""Search interceptor for canary deployment traffic routing.

This module provides a wrapper around QdrantSearch that intercepts search requests
and routes them based on active canary deployments.
"""

import logging
import time
from typing import Any

from ...config import UnifiedConfig
from ..deployment.canary_router import CanaryRouter
from .search import QdrantSearch

logger = logging.getLogger(__name__)


class SearchInterceptor:
    """Intercepts search requests for canary routing."""

    def __init__(
        self,
        search_service: QdrantSearch,
        router: CanaryRouter | None,
        config: UnifiedConfig,
        redis_client=None,
    ):
        """Initialize search interceptor.

        Args:
            search_service: Underlying search service
            router: Canary router for traffic decisions
            config: Unified configuration
            redis_client: Optional Redis client for event publishing
        """
        self._search = search_service
        self._router = router
        self._config = config
        self._redis_client = redis_client
        self._metrics_enabled = getattr(
            config.performance, "enable_canary_metrics", True
        )
        self._event_stream_key = "search:events"

    async def _publish_search_event(
        self,
        event_type: str,
        collection_name: str,
        target_collection: str,
        is_canary: bool,
        deployment_id: str | None,
        latency_ms: float | None = None,
        is_error: bool = False,
        error_message: str | None = None,
    ) -> None:
        """Publish search event to Redis Stream.

        Args:
            event_type: Type of event (e.g., "search_routed", "search_completed")
            collection_name: Original collection/alias name
            target_collection: Actual collection used for search
            is_canary: Whether this was routed to canary
            deployment_id: Canary deployment ID if applicable
            latency_ms: Search latency in milliseconds
            is_error: Whether search failed
            error_message: Error message if applicable
        """
        if not self._redis_client:
            return

        try:
            event_data = {
                "type": event_type,
                "collection_name": collection_name,
                "target_collection": target_collection,
                "is_canary": str(is_canary),
                "timestamp": str(time.time()),
            }

            if deployment_id:
                event_data["deployment_id"] = deployment_id
            if latency_ms is not None:
                event_data["latency_ms"] = str(latency_ms)
            if is_error:
                event_data["is_error"] = "true"
            if error_message:
                event_data["error_message"] = error_message

            await self._redis_client.xadd(self._event_stream_key, event_data)

        except Exception as e:
            logger.warning(f"Failed to publish search event: {e}")

    async def hybrid_search(
        self,
        collection_name: str,
        query_vector: list[float],
        sparse_vector: dict[int, float] | None = None,
        limit: int = 10,
        score_threshold: float = 0.0,
        fusion_type: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search with canary routing.

        Args:
            collection_name: Collection or alias to search
            query_vector: Dense embedding vector
            sparse_vector: Optional sparse vector
            limit: Maximum results to return
            score_threshold: Minimum score threshold
            fusion_type: Fusion method ("rrf" or "dbsf")
            search_accuracy: Accuracy level
            user_id: Optional user ID for consistent routing
            request_id: Optional request ID for tracking

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails
        """
        start_time = time.time()
        target_collection = collection_name
        route_decision = None

        try:
            # Check for canary routing if router is available
            if self._router:
                try:
                    route_decision = await self._router.get_route_decision(
                        alias=collection_name,
                        user_id=user_id,
                        request_id=request_id,
                        use_sticky_sessions=True,
                    )
                    target_collection = route_decision.collection_name

                    logger.debug(
                        f"Canary routing decision: {collection_name} -> {target_collection} "
                        f"(canary={route_decision.is_canary}, percentage={route_decision.canary_percentage})"
                    )

                    # Publish routing event
                    await self._publish_search_event(
                        event_type="search_routed",
                        collection_name=collection_name,
                        target_collection=target_collection,
                        is_canary=route_decision.is_canary,
                        deployment_id=route_decision.deployment_id,
                    )
                except Exception as e:
                    logger.warning(
                        f"Canary routing failed, using default collection: {e}"
                    )
                    # Fall back to original collection if routing fails
                    target_collection = collection_name
                    route_decision = None

            # Perform actual search
            results = await self._search.hybrid_search(
                collection_name=target_collection,
                query_vector=query_vector,
                sparse_vector=sparse_vector,
                limit=limit,
                score_threshold=score_threshold,
                fusion_type=fusion_type,
                search_accuracy=search_accuracy,
            )

            # Record successful metrics
            latency_ms = (time.time() - start_time) * 1000

            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                try:
                    await self._router.record_request_metrics(
                        deployment_id=route_decision.deployment_id,
                        collection_name=target_collection,
                        latency_ms=latency_ms,
                        is_error=False,
                    )
                except Exception as e:
                    logger.warning(f"Failed to record success metrics: {e}")

            # Publish search completed event
            if route_decision:
                await self._publish_search_event(
                    event_type="search_completed",
                    collection_name=collection_name,
                    target_collection=target_collection,
                    is_canary=route_decision.is_canary,
                    deployment_id=route_decision.deployment_id,
                    latency_ms=latency_ms,
                    is_error=False,
                )

            return results

        except Exception as e:
            # Record error metrics
            latency_ms = (time.time() - start_time) * 1000

            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                try:
                    await self._router.record_request_metrics(
                        deployment_id=route_decision.deployment_id,
                        collection_name=target_collection,
                        latency_ms=latency_ms,
                        is_error=True,
                    )
                except Exception as metrics_error:
                    logger.warning(f"Failed to record error metrics: {metrics_error}")

            # Publish search error event
            if route_decision:
                await self._publish_search_event(
                    event_type="search_failed",
                    collection_name=collection_name,
                    target_collection=target_collection,
                    is_canary=route_decision.is_canary,
                    deployment_id=route_decision.deployment_id,
                    latency_ms=latency_ms,
                    is_error=True,
                    error_message=str(e),
                )

            raise e

    async def multi_stage_search(
        self,
        collection_name: str,
        stages: list[dict[str, Any]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform multi-stage search with canary routing.

        Args:
            collection_name: Collection or alias to search
            stages: List of search stages
            limit: Final number of results
            fusion_algorithm: Fusion method
            search_accuracy: Accuracy level
            user_id: Optional user ID for consistent routing
            request_id: Optional request ID for tracking

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails
        """
        start_time = time.time()
        target_collection = collection_name
        route_decision = None

        try:
            # Check for canary routing
            if self._router:
                try:
                    route_decision = await self._router.get_route_decision(
                        alias=collection_name,
                        user_id=user_id,
                        request_id=request_id,
                        use_sticky_sessions=True,
                    )
                    target_collection = route_decision.collection_name
                except Exception as e:
                    logger.warning(
                        f"Canary routing failed, using default collection: {e}"
                    )
                    target_collection = collection_name
                    route_decision = None

            # Perform actual search
            results = await self._search.multi_stage_search(
                collection_name=target_collection,
                stages=stages,
                limit=limit,
                fusion_algorithm=fusion_algorithm,
                search_accuracy=search_accuracy,
            )

            # Record metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                try:
                    latency_ms = (time.time() - start_time) * 1000
                    await self._router.record_request_metrics(
                        deployment_id=route_decision.deployment_id,
                        collection_name=target_collection,
                        latency_ms=latency_ms,
                        is_error=False,
                    )
                except Exception as e:
                    logger.warning(f"Failed to record success metrics: {e}")

            return results

        except Exception as e:
            # Record error metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                latency_ms = (time.time() - start_time) * 1000
                await self._router.record_request_metrics(
                    deployment_id=route_decision.deployment_id,
                    collection_name=target_collection,
                    latency_ms=latency_ms,
                    is_error=True,
                )

            raise e

    async def hyde_search(
        self,
        collection_name: str,
        query: str,
        query_embedding: list[float],
        hypothetical_embeddings: list[list[float]],
        limit: int = 10,
        fusion_algorithm: str = "rrf",
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search using HyDE with canary routing.

        Args:
            collection_name: Collection or alias to search
            query: Original query text
            query_embedding: Original query embedding
            hypothetical_embeddings: List of hypothetical document embeddings
            limit: Number of results
            fusion_algorithm: Fusion method
            search_accuracy: Accuracy level
            user_id: Optional user ID for consistent routing
            request_id: Optional request ID for tracking

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails
        """
        start_time = time.time()
        target_collection = collection_name
        route_decision = None

        try:
            # Check for canary routing
            if self._router:
                route_decision = await self._router.get_route_decision(
                    alias=collection_name,
                    user_id=user_id,
                    request_id=request_id,
                    use_sticky_sessions=True,
                )
                target_collection = route_decision.collection_name

            # Perform actual search
            results = await self._search.hyde_search(
                collection_name=target_collection,
                query=query,
                query_embedding=query_embedding,
                hypothetical_embeddings=hypothetical_embeddings,
                limit=limit,
                fusion_algorithm=fusion_algorithm,
                search_accuracy=search_accuracy,
            )

            # Record metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                latency_ms = (time.time() - start_time) * 1000
                await self._router.record_request_metrics(
                    deployment_id=route_decision.deployment_id,
                    collection_name=target_collection,
                    latency_ms=latency_ms,
                    is_error=False,
                )

            return results

        except Exception as e:
            # Record error metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                latency_ms = (time.time() - start_time) * 1000
                await self._router.record_request_metrics(
                    deployment_id=route_decision.deployment_id,
                    collection_name=target_collection,
                    latency_ms=latency_ms,
                    is_error=True,
                )

            raise e

    async def filtered_search(
        self,
        collection_name: str,
        query_vector: list[float],
        filters: dict[str, Any],
        limit: int = 10,
        search_accuracy: str = "balanced",
        user_id: str | None = None,
        request_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Optimized filtered search with canary routing.

        Args:
            collection_name: Collection or alias to search
            query_vector: Query vector
            filters: Filters to apply
            limit: Number of results
            search_accuracy: Accuracy level
            user_id: Optional user ID for consistent routing
            request_id: Optional request ID for tracking

        Returns:
            List of search results

        Raises:
            QdrantServiceError: If search fails
        """
        start_time = time.time()
        target_collection = collection_name
        route_decision = None

        try:
            # Check for canary routing
            if self._router:
                route_decision = await self._router.get_route_decision(
                    alias=collection_name,
                    user_id=user_id,
                    request_id=request_id,
                    use_sticky_sessions=True,
                )
                target_collection = route_decision.collection_name

            # Perform actual search
            results = await self._search.filtered_search(
                collection_name=target_collection,
                query_vector=query_vector,
                filters=filters,
                limit=limit,
                search_accuracy=search_accuracy,
            )

            # Record metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                latency_ms = (time.time() - start_time) * 1000
                await self._router.record_request_metrics(
                    deployment_id=route_decision.deployment_id,
                    collection_name=target_collection,
                    latency_ms=latency_ms,
                    is_error=False,
                )

            return results

        except Exception as e:
            # Record error metrics
            if (
                self._metrics_enabled
                and self._router
                and route_decision
                and route_decision.deployment_id
            ):
                latency_ms = (time.time() - start_time) * 1000
                await self._router.record_request_metrics(
                    deployment_id=route_decision.deployment_id,
                    collection_name=target_collection,
                    latency_ms=latency_ms,
                    is_error=True,
                )

            raise e
