"""Canary routing interceptor for search requests.

This module provides application-level traffic routing for canary deployments,
allowing gradual traffic shifts between old and new collections.
"""

import hashlib
import logging
import random
import time
from dataclasses import dataclass
from dataclasses import field

from ...config import UnifiedConfig
from ..cache.dragonfly_cache import DragonflyCache

logger = logging.getLogger(__name__)


@dataclass
class RouteDecision:
    """Result of routing decision for a request."""

    collection_name: str
    is_canary: bool
    canary_percentage: float | None = None
    deployment_id: str | None = None
    routing_key: str | None = None


@dataclass
class CanaryRoute:
    """Routing configuration for a canary deployment."""

    deployment_id: str
    alias: str
    old_collection: str
    new_collection: str
    percentage: float
    status: str
    updated_at: float = field(default_factory=time.time)


class CanaryRouter:
    """Routes search requests based on active canary deployments."""

    def __init__(
        self,
        cache: DragonflyCache,
        config: UnifiedConfig,
    ):
        """Initialize canary router.

        Args:
            cache: DragonflyDB cache for storing routing state
            config: Unified configuration
        """
        self.cache = cache
        self.config = config
        self._route_cache_prefix = "canary:routes:"
        self._metrics_prefix = "canary:metrics:"
        self._sticky_sessions_prefix = "canary:sticky:"
        self._route_ttl = 3600  # 1 hour cache for routes
        self._sticky_ttl = 86400  # 24 hour sticky sessions

    async def update_route(
        self,
        deployment_id: str,
        alias: str,
        old_collection: str,
        new_collection: str,
        percentage: float,
        status: str = "running",
    ) -> bool:
        """Update routing configuration for a canary deployment.

        Args:
            deployment_id: ID of the canary deployment
            alias: Alias being updated
            old_collection: Original collection name
            new_collection: New collection being deployed
            percentage: Traffic percentage for new collection (0-100)
            status: Deployment status

        Returns:
            bool: True if update successful
        """
        try:
            route = CanaryRoute(
                deployment_id=deployment_id,
                alias=alias,
                old_collection=old_collection,
                new_collection=new_collection,
                percentage=percentage,
                status=status,
            )

            # Store route configuration
            route_key = f"{self._route_cache_prefix}{alias}"
            route_data = {
                "deployment_id": route.deployment_id,
                "alias": route.alias,
                "old_collection": route.old_collection,
                "new_collection": route.new_collection,
                "percentage": route.percentage,
                "status": route.status,
                "updated_at": route.updated_at,
            }

            await self.cache.set(route_key, route_data, ttl=self._route_ttl)

            logger.info(
                f"Updated canary route for {alias}: {percentage}% to {new_collection}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to update canary route: {e}")
            return False

    async def remove_route(self, alias: str) -> bool:
        """Remove routing configuration for an alias.

        Args:
            alias: Alias to remove routing for

        Returns:
            bool: True if removal successful
        """
        try:
            route_key = f"{self._route_cache_prefix}{alias}"
            await self.cache.delete(route_key)

            # Clear any sticky sessions for this alias
            pattern = f"{self._sticky_sessions_prefix}{alias}:*"
            sticky_keys = await self.cache.scan_keys(pattern)
            for key in sticky_keys:
                await self.cache.delete(key)

            logger.info(f"Removed canary route for {alias}")
            return True

        except Exception as e:
            logger.error(f"Failed to remove canary route: {e}")
            return False

    async def get_route_decision(
        self,
        alias: str,
        user_id: str | None = None,
        request_id: str | None = None,
        use_sticky_sessions: bool = True,
    ) -> RouteDecision:
        """Decide which collection to route a search request to.

        Args:
            alias: Alias being searched
            user_id: Optional user identifier for consistent routing
            request_id: Optional request identifier
            use_sticky_sessions: Whether to use sticky sessions for users

        Returns:
            RouteDecision: Routing decision with target collection
        """
        try:
            # Check for active canary route
            route_key = f"{self._route_cache_prefix}{alias}"
            route_data = await self.cache.get(route_key)

            if not route_data or route_data.get("status") != "running":
                # No active canary, use alias directly
                return RouteDecision(
                    collection_name=alias,
                    is_canary=False,
                )

            # Extract route configuration
            route = CanaryRoute(**route_data)

            # Generate routing key for consistent hashing
            routing_key = self._generate_routing_key(
                user_id=user_id,
                request_id=request_id,
                alias=alias,
            )

            # Check sticky session if enabled
            if use_sticky_sessions and user_id:
                sticky_decision = await self._check_sticky_session(
                    alias=alias,
                    user_id=user_id,
                    route=route,
                )
                if sticky_decision:
                    return sticky_decision

            # Make routing decision based on percentage
            target_collection = self._make_routing_decision(
                routing_key=routing_key,
                percentage=route.percentage,
                old_collection=route.old_collection,
                new_collection=route.new_collection,
            )

            # Store sticky session if enabled
            if use_sticky_sessions and user_id:
                await self._store_sticky_session(
                    alias=alias,
                    user_id=user_id,
                    target_collection=target_collection,
                    route=route,
                )

            return RouteDecision(
                collection_name=target_collection,
                is_canary=target_collection == route.new_collection,
                canary_percentage=route.percentage,
                deployment_id=route.deployment_id,
                routing_key=routing_key,
            )

        except Exception as e:
            logger.error(f"Error making route decision: {e}")
            # Fallback to alias on error
            return RouteDecision(
                collection_name=alias,
                is_canary=False,
            )

    async def record_request_metrics(
        self,
        deployment_id: str,
        collection_name: str,
        latency_ms: float,
        is_error: bool = False,
    ) -> None:
        """Record metrics for a routed request.

        Args:
            deployment_id: Canary deployment ID
            collection_name: Collection that served the request
            latency_ms: Request latency in milliseconds
            is_error: Whether the request resulted in an error
        """
        try:
            # Metrics key structure: canary:metrics:{deployment_id}:{collection}:{metric_type}
            timestamp = int(time.time())
            bucket_key = f"{timestamp // 60}"  # 1-minute buckets

            # Record latency
            latency_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:latency:{bucket_key}"
            await self.cache.client.lpush(latency_key, str(latency_ms))
            await self.cache.client.expire(latency_key, 7200)  # 2 hour retention

            # Record request count
            count_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:count:{bucket_key}"
            await self.cache.client.incr(count_key)
            await self.cache.client.expire(count_key, 7200)

            # Record error count if applicable
            if is_error:
                error_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:errors:{bucket_key}"
                await self.cache.client.incr(error_key)
                await self.cache.client.expire(error_key, 7200)

        except Exception as e:
            logger.warning(f"Failed to record canary metrics: {e}")

    async def get_collection_metrics(
        self,
        deployment_id: str,
        collection_name: str,
        duration_minutes: int = 10,
    ) -> dict[str, float]:
        """Get aggregated metrics for a collection in a canary deployment.

        Args:
            deployment_id: Canary deployment ID
            collection_name: Collection to get metrics for
            duration_minutes: Time window for metrics (default: 10 minutes)

        Returns:
            dict: Metrics including latency and error rate
        """
        try:
            current_time = int(time.time())
            start_time = current_time - (duration_minutes * 60)

            total_requests = 0
            total_errors = 0
            latencies = []

            # Iterate through time buckets
            for timestamp in range(start_time, current_time + 60, 60):
                bucket_key = f"{timestamp // 60}"

                # Get request count
                count_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:count:{bucket_key}"
                count = await self.cache.client.get(count_key)
                if count:
                    total_requests += int(count)

                # Get error count
                error_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:errors:{bucket_key}"
                errors = await self.cache.client.get(error_key)
                if errors:
                    total_errors += int(errors)

                # Get latencies
                latency_key = f"{self._metrics_prefix}{deployment_id}:{collection_name}:latency:{bucket_key}"
                bucket_latencies = await self.cache.client.lrange(latency_key, 0, -1)
                if bucket_latencies:
                    latencies.extend([float(lat) for lat in bucket_latencies])

            # Calculate metrics
            error_rate = (total_errors / total_requests) if total_requests > 0 else 0.0
            avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
            p95_latency = (
                sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0.0
            )

            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": error_rate,
                "avg_latency": avg_latency,
                "p95_latency": p95_latency,
                "latency": p95_latency,  # For compatibility with existing code
            }

        except Exception as e:
            logger.error(f"Failed to get collection metrics: {e}")
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "avg_latency": 0.0,
                "p95_latency": 0.0,
                "latency": 0.0,
            }

    def _generate_routing_key(
        self,
        user_id: str | None,
        request_id: str | None,
        alias: str,
    ) -> str:
        """Generate consistent routing key for a request.

        Args:
            user_id: Optional user identifier
            request_id: Optional request identifier
            alias: Alias being accessed

        Returns:
            str: Hashed routing key
        """
        # Use user_id if available for consistent user experience
        if user_id:
            key_source = f"{alias}:{user_id}"
        elif request_id:
            key_source = f"{alias}:{request_id}"
        else:
            # Random key for anonymous requests
            key_source = f"{alias}:{random.random()}"

        return hashlib.md5(key_source.encode()).hexdigest()

    def _make_routing_decision(
        self,
        routing_key: str,
        percentage: float,
        old_collection: str,
        new_collection: str,
    ) -> str:
        """Make routing decision based on consistent hashing.

        Args:
            routing_key: Hashed routing key
            percentage: Percentage of traffic for new collection
            old_collection: Original collection name
            new_collection: New collection name

        Returns:
            str: Target collection name
        """
        # Convert hex routing key to number in range [0, 100)
        key_value = int(routing_key[:8], 16) % 100

        # Route to new collection if within percentage threshold
        if key_value < percentage:
            return new_collection
        else:
            return old_collection

    async def _check_sticky_session(
        self,
        alias: str,
        user_id: str,
        route: CanaryRoute,
    ) -> RouteDecision | None:
        """Check if user has a sticky session for this alias.

        Args:
            alias: Alias being accessed
            user_id: User identifier
            route: Current canary route configuration

        Returns:
            RouteDecision if sticky session exists, None otherwise
        """
        try:
            sticky_key = f"{self._sticky_sessions_prefix}{alias}:{user_id}"
            sticky_data = await self.cache.get(sticky_key)

            if sticky_data:
                # Validate sticky session is still valid
                target_collection = sticky_data.get("collection")
                if target_collection in [route.old_collection, route.new_collection]:
                    return RouteDecision(
                        collection_name=target_collection,
                        is_canary=target_collection == route.new_collection,
                        canary_percentage=route.percentage,
                        deployment_id=route.deployment_id,
                        routing_key=user_id,
                    )

        except Exception as e:
            logger.warning(f"Failed to check sticky session: {e}")

        return None

    async def _store_sticky_session(
        self,
        alias: str,
        user_id: str,
        target_collection: str,
        route: CanaryRoute,
    ) -> None:
        """Store sticky session for consistent user routing.

        Args:
            alias: Alias being accessed
            user_id: User identifier
            target_collection: Collection user is routed to
            route: Current canary route configuration
        """
        try:
            sticky_key = f"{self._sticky_sessions_prefix}{alias}:{user_id}"
            sticky_data = {
                "collection": target_collection,
                "deployment_id": route.deployment_id,
                "created_at": time.time(),
            }

            await self.cache.set(sticky_key, sticky_data, ttl=self._sticky_ttl)

        except Exception as e:
            logger.warning(f"Failed to store sticky session: {e}")
