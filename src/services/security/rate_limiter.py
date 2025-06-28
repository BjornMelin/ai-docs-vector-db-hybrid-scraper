#!/usr/bin/env python3
"""Advanced rate limiting with Redis backend and fallback mechanisms.

This module implements production-grade rate limiting using Redis for distributed
rate limiting with local cache fallback for high availability.

Features:
- Distributed rate limiting with Redis backend
- Local cache fallback when Redis is unavailable
- Sliding window algorithm for accurate rate limiting
- Burst traffic support with configurable factors
- Thread-safe operations with atomic Redis operations
- Comprehensive logging and monitoring integration
"""

import asyncio
import logging
import time


from typing import Dict

try:
    import redis.asyncio as redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


from src.config.security import SecurityConfig


logger = logging.getLogger(__name__)


class DistributedRateLimiter:
    """Production-grade rate limiting with Redis backend and local fallback.

    This class provides comprehensive rate limiting capabilities suitable for
    production deployment:

    - Uses Redis for distributed rate limiting across multiple instances
    - Falls back to local memory cache when Redis is unavailable
    - Implements sliding window algorithm for accurate rate limiting
    - Supports burst traffic with configurable burst factors
    - Provides atomic operations to prevent race conditions
    - Includes comprehensive error handling and logging
    """

    def __init__(
        self,
        redis_url: str | None = None,
        security_config: SecurityConfig | None = None,
    ):
        """Initialize distributed rate limiter.

        Args:
            redis_url: Redis connection URL. If None, only local fallback is used.
            security_config: Security configuration for rate limiting settings.
        """
        self.security_config = security_config or SecurityConfig()
        self.redis_client: redis.Redis | None = None
        self.local_cache: Dict[str, Dict[str, float]] = {}  # Fallback cache
        self._cache_lock = asyncio.Lock()

        # Initialize Redis client if available and URL provided
        if REDIS_AVAILABLE and redis_url:
            try:
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                    health_check_interval=30,
                )
                logger.info(f"Redis rate limiter initialized: {redis_url}")  # TODO: Convert f-string to logging format
            except Exception as e:
                logger.warning(
                    f"Failed to initialize Redis: {e}. Using local fallback only."
                )
                self.redis_client = None
        else:
            if not REDIS_AVAILABLE:
                logger.warning(
                    "Redis not available. Install redis package for distributed rate limiting."
                )
            logger.info("Rate limiter initialized with local cache fallback only")

    async def check_rate_limit(
        self, identifier: str, limit: int, window: int, burst_factor: float = 1.5
    ) -> tuple[bool, dict]:
        """Check rate limit using sliding window algorithm.

        Args:
            identifier: Client identifier (IP, user ID, API key hash)
            limit: Max requests per window
            window: Time window in seconds
            burst_factor: Allow burst above normal limit (default: 1.5)

        Returns:
            Tuple of (is_allowed, rate_limit_info)
            - is_allowed: Whether request is allowed
            - rate_limit_info: Dictionary with rate limit status info
        """
        current_time = time.time()
        burst_limit = int(limit * burst_factor)

        # Try Redis first, fallback to local cache
        if self.redis_client:
            try:
                return await self._check_rate_limit_redis(
                    identifier, limit, window, burst_limit, current_time
                )
            except Exception as e:
                logger.warning(
                    f"Redis rate limit check failed: {e}. Using local fallback."
                )

        # Local fallback
        return await self._check_rate_limit_local(
            identifier, limit, window, burst_limit, current_time
        )

    async def _check_rate_limit_redis(
        self,
        identifier: str,
        limit: int,
        window: int,
        burst_limit: int,
        current_time: float,
    ) -> tuple[bool, dict]:
        """Check rate limit using Redis with sliding window algorithm.

        Args:
            identifier: Client identifier
            limit: Base rate limit
            window: Time window in seconds
            burst_limit: Maximum burst requests allowed
            current_time: Current timestamp

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        key = f"rate_limit:{identifier}:{window}"
        window_start = current_time - window

        # Use pipeline for atomic operations
        pipe = self.redis_client.pipeline()

        # Remove old entries outside the window
        pipe.zremrangebyscore(key, 0, window_start)

        # Count current requests in window
        pipe.zcard(key)

        # Add current request
        pipe.zadd(key, {str(current_time): current_time})

        # Set expiration
        pipe.expire(key, window + 10)  # Extra time to handle edge cases

        # Execute pipeline
        results = await pipe.execute()
        current_requests = results[1]

        # Calculate rate limit info
        is_allowed = current_requests < burst_limit
        requests_remaining = max(0, burst_limit - current_requests - 1)

        rate_limit_info = {
            "limit": limit,
            "burst_limit": burst_limit,
            "remaining": requests_remaining,
            "reset_time": int(current_time + window),
            "current_requests": current_requests,
            "window_seconds": window,
            "backend": "redis",
        }

        if not is_allowed:
            # Remove the request we just added since it's not allowed
            await self.redis_client.zrem(key, str(current_time))

        logger.debug(
            f"Redis rate limit check: {identifier} - "
            f"{current_requests}/{burst_limit} requests, allowed: {is_allowed}"
        )

        return is_allowed, rate_limit_info

    async def _check_rate_limit_local(
        self,
        identifier: str,
        limit: int,
        window: int,
        burst_limit: int,
        current_time: float,
    ) -> tuple[bool, dict]:
        """Local fallback rate limiting using in-memory cache.

        Args:
            identifier: Client identifier
            limit: Base rate limit
            window: Time window in seconds
            burst_limit: Maximum burst requests allowed
            current_time: Current timestamp

        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        async with self._cache_lock:
            window_start = current_time - window

            # Initialize or get existing request history
            if identifier not in self.local_cache:
                self.local_cache[identifier] = {}

            request_history = self.local_cache[identifier]

            # Clean old requests outside window
            expired_keys = [
                timestamp
                for timestamp in request_history
                if float(timestamp) < window_start
            ]
            for key in expired_keys:
                del request_history[key]

            # Count current requests
            current_requests = len(request_history)
            is_allowed = current_requests < burst_limit

            if is_allowed:
                # Add current request
                request_history[str(current_time)] = current_time
                current_requests += 1

            requests_remaining = max(0, burst_limit - current_requests)

            rate_limit_info = {
                "limit": limit,
                "burst_limit": burst_limit,
                "remaining": requests_remaining,
                "reset_time": int(current_time + window),
                "current_requests": current_requests,
                "window_seconds": window,
                "backend": "local",
            }

            logger.debug(
                f"Local rate limit check: {identifier} - "
                f"{current_requests}/{burst_limit} requests, allowed: {is_allowed}"
            )

            return is_allowed, rate_limit_info

    async def get_rate_limit_status(self, identifier: str, window: int = 60) -> dict:
        """Get current rate limit status for an identifier.

        Args:
            identifier: Client identifier
            window: Time window to check

        Returns:
            Dictionary with current rate limit status
        """
        current_time = time.time()

        if self.redis_client:
            try:
                key = f"rate_limit:{identifier}:{window}"
                window_start = current_time - window

                # Clean and count current requests
                pipe = self.redis_client.pipeline()
                pipe.zremrangebyscore(key, 0, window_start)
                pipe.zcard(key)
                results = await pipe.execute()

                current_requests = results[1]

                return {
                    "identifier": identifier,
                    "current_requests": current_requests,
                    "window_seconds": window,
                    "backend": "redis",
                    "timestamp": current_time,
                }
            except Exception as e:
                logger.warning(f"Failed to get Redis rate limit status: {e}")  # TODO: Convert f-string to logging format

        # Local fallback
        async with self._cache_lock:
            if identifier not in self.local_cache:
                current_requests = 0
            else:
                window_start = current_time - window
                request_history = self.local_cache[identifier]
                current_requests = sum(
                    1
                    for timestamp in request_history
                    if float(timestamp) >= window_start
                )

            return {
                "identifier": identifier,
                "current_requests": current_requests,
                "window_seconds": window,
                "backend": "local",
                "timestamp": current_time,
            }

    async def reset_rate_limit(self, identifier: str, window: int = 60) -> bool:
        """Reset rate limit for a specific identifier (admin function).

        Args:
            identifier: Client identifier to reset
            window: Time window to reset

        Returns:
            True if reset was successful
        """
        try:
            if self.redis_client:
                key = f"rate_limit:{identifier}:{window}"
                await self.redis_client.delete(key)
                logger.info(f"Reset Redis rate limit for {identifier}")  # TODO: Convert f-string to logging format

            # Also reset local cache
            async with self._cache_lock:
                if identifier in self.local_cache:
                    del self.local_cache[identifier]
                    logger.info(f"Reset local rate limit for {identifier}")  # TODO: Convert f-string to logging format

            return True
        except Exception as e:
            logger.exception(f"Failed to reset rate limit for {identifier}: {e}")
            return False

    async def cleanup_expired_entries(self, max_age_hours: int = 24) -> int:
        """Clean up expired entries from local cache.

        Args:
            max_age_hours: Maximum age in hours for cache entries

        Returns:
            Number of entries cleaned up
        """
        if not self.local_cache:
            return 0

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        cutoff_time = current_time - max_age_seconds

        cleanup_count = 0

        async with self._cache_lock:
            identifiers_to_remove = []

            for identifier, request_history in self.local_cache.items():
                # Remove old requests from history
                expired_keys = [
                    timestamp
                    for timestamp in request_history
                    if float(timestamp) < cutoff_time
                ]

                for key in expired_keys:
                    del request_history[key]
                    cleanup_count += 1

                # Remove identifier if no recent requests
                if not request_history:
                    identifiers_to_remove.append(identifier)

            # Remove empty identifiers
            for identifier in identifiers_to_remove:
                del self.local_cache[identifier]

        if cleanup_count > 0:
            logger.info(f"Cleaned up {cleanup_count} expired rate limit entries")  # TODO: Convert f-string to logging format

        return cleanup_count

    async def get_health_status(self) -> dict:
        """Get health status of the rate limiter.

        Returns:
            Dictionary with health status information
        """
        status = {
            "local_cache_enabled": True,
            "local_cache_entries": len(self.local_cache),
            "redis_enabled": self.redis_client is not None,
            "redis_healthy": False,
        }

        if self.redis_client:
            try:
                await self.redis_client.ping()
                status["redis_healthy"] = True
            except Exception as e:
                logger.warning(f"Redis health check failed: {e}")  # TODO: Convert f-string to logging format
                status["redis_error"] = str(e)

        return status

    async def close(self):
        """Close Redis connection if open."""
        if self.redis_client:
            try:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            except Exception as e:
                logger.warning(f"Error closing Redis connection: {e}")  # TODO: Convert f-string to logging format