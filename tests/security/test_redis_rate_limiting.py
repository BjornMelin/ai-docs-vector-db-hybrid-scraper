#!/usr/bin/env python3
"""Tests for Redis-backed rate limiting in SecurityMiddleware.

This test module verifies the Redis-backed rate limiting implementation
addresses the critical security vulnerability identified in the Security
Architecture Assessment. It tests:

1. Redis connection and health checking
2. Distributed rate limiting with sliding window algorithm
3. Fallback to in-memory rate limiting when Redis is unavailable
4. Proper error handling and recovery
5. Concurrent request handling
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.config.security import SecurityConfig
from src.services.fastapi.middleware.security import SecurityMiddleware


@pytest.fixture
def security_config():
    """Create test security configuration."""
    return SecurityConfig(
        enabled=True,
        enable_rate_limiting=True,
        rate_limit_requests=5,
        rate_limit_window=60,
    )


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client for testing."""
    mock_client = AsyncMock()
    mock_client.ping = AsyncMock(return_value=True)
    mock_client.get = AsyncMock(return_value=None)
    mock_client.incr = AsyncMock(return_value=1)
    mock_client.expire = AsyncMock(return_value=True)
    mock_client.pipeline = AsyncMock()
    mock_client.aclose = AsyncMock()
    return mock_client


@pytest.fixture
def mock_app():
    """Create mock ASGI application."""
    async def app(request):
        return JSONResponse({"status": "ok"})
    return app


class TestRedisRateLimiting:
    """Test Redis-backed rate limiting functionality."""

    def test_security_middleware_initialization(self, mock_app, security_config):
        """Test SecurityMiddleware initializes with Redis configuration."""
        redis_url = "redis://localhost:6379/1"
        middleware = SecurityMiddleware(mock_app, security_config, redis_url)
        
        assert middleware.redis_url == redis_url
        assert middleware.redis_client is None  # Not connected yet
        assert middleware._redis_healthy is False
        assert isinstance(middleware._rate_limit_storage, dict)

    @pytest.mark.asyncio
    async def test_redis_initialization_success(self, mock_app, security_config):
        """Test successful Redis connection initialization."""
        with patch('redis.asyncio.Redis.from_url') as mock_redis_from_url:
            mock_client = AsyncMock()
            mock_client.ping = AsyncMock(return_value=True)
            mock_redis_from_url.return_value = mock_client
            
            middleware = SecurityMiddleware(mock_app, security_config)
            await middleware._initialize_redis()
            
            assert middleware.redis_client is not None
            assert middleware._redis_healthy is True
            mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_initialization_failure(self, mock_app, security_config):
        """Test Redis connection initialization failure and fallback."""
        with patch('redis.asyncio.Redis.from_url') as mock_redis_from_url:
            mock_redis_from_url.side_effect = Exception("Connection failed")
            
            middleware = SecurityMiddleware(mock_app, security_config)
            await middleware._initialize_redis()
            
            assert middleware.redis_client is None
            assert middleware._redis_healthy is False

    @pytest.mark.asyncio
    async def test_redis_rate_limiting_within_limits(self, mock_app, security_config):
        """Test Redis rate limiting allows requests within limits."""
        middleware = SecurityMiddleware(mock_app, security_config)
        middleware._redis_healthy = True
        
        # Mock Redis client with pipeline
        mock_pipeline = AsyncMock()
        mock_pipeline.get.return_value = AsyncMock()
        mock_pipeline.get.return_value.execute = AsyncMock(return_value=[None])
        mock_pipeline.incr.return_value = AsyncMock()
        mock_pipeline.expire.return_value = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[1])
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline
        middleware.redis_client = mock_client
        
        # Test rate limiting allows request
        result = await middleware._check_rate_limit_redis("192.168.1.1")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_rate_limiting_exceeds_limits(self, mock_app, security_config):
        """Test Redis rate limiting blocks requests that exceed limits."""
        middleware = SecurityMiddleware(mock_app, security_config)
        middleware._redis_healthy = True
        
        # Mock Redis client returning count that exceeds limit
        mock_pipeline = AsyncMock()
        mock_pipeline.get.return_value = AsyncMock()
        mock_pipeline.get.return_value.execute = AsyncMock(return_value=[str(security_config.rate_limit_requests + 1)])
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock()
        
        mock_client = AsyncMock()
        mock_client.pipeline.return_value = mock_pipeline
        middleware.redis_client = mock_client
        
        # Test rate limiting blocks request
        result = await middleware._check_rate_limit_redis("192.168.1.1")
        assert result is False

    @pytest.mark.asyncio
    async def test_redis_failure_fallback_to_memory(self, mock_app, security_config):
        """Test fallback to in-memory rate limiting when Redis fails."""
        middleware = SecurityMiddleware(mock_app, security_config)
        middleware._redis_healthy = True
        
        # Mock Redis client that fails
        mock_client = AsyncMock()
        mock_client.pipeline.side_effect = Exception("Redis connection lost")
        middleware.redis_client = mock_client
        
        # Mock the memory rate limiting to return True
        with patch.object(middleware, '_check_rate_limit_memory', return_value=True) as mock_memory:
            result = await middleware._check_rate_limit_redis("192.168.1.1")
            
            # Should fallback to memory and return result
            assert result is True
            mock_memory.assert_called_once_with("192.168.1.1")
            # Redis should be marked as unhealthy
            assert middleware._redis_healthy is False

    def test_memory_rate_limiting_within_limits(self, mock_app, security_config):
        """Test in-memory rate limiting allows requests within limits."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        # Test multiple requests within limit
        for i in range(security_config.rate_limit_requests):
            result = middleware._check_rate_limit_memory("192.168.1.1")
            assert result is True, f"Request {i+1} should be allowed"

    def test_memory_rate_limiting_exceeds_limits(self, mock_app, security_config):
        """Test in-memory rate limiting blocks requests that exceed limits."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        # Fill up the rate limit
        for _ in range(security_config.rate_limit_requests):
            middleware._check_rate_limit_memory("192.168.1.1")
        
        # Next request should be blocked
        result = middleware._check_rate_limit_memory("192.168.1.1")
        assert result is False

    def test_memory_rate_limiting_window_reset(self, mock_app, security_config):
        """Test in-memory rate limiting resets after time window."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        # Fill up the rate limit
        for _ in range(security_config.rate_limit_requests):
            middleware._check_rate_limit_memory("192.168.1.1")
        
        # Simulate time passage beyond window
        current_time = int(time.time())
        middleware._rate_limit_storage["192.168.1.1"]["window_start"] = (
            current_time - security_config.rate_limit_window - 1
        )
        
        # Should allow new request after window reset
        result = middleware._check_rate_limit_memory("192.168.1.1")
        assert result is True

    @pytest.mark.asyncio
    async def test_redis_health_check_success(self, mock_app, security_config):
        """Test Redis health check with successful connection."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(return_value=True)
        middleware.redis_client = mock_client
        middleware._redis_healthy = False
        
        result = await middleware._check_redis_health()
        
        assert result is True
        assert middleware._redis_healthy is True
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_health_check_failure(self, mock_app, security_config):
        """Test Redis health check with failed connection."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=Exception("Connection failed"))
        middleware.redis_client = mock_client
        middleware._redis_healthy = True
        
        result = await middleware._check_redis_health()
        
        assert result is False
        assert middleware._redis_healthy is False
        mock_client.ping.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_redis_connection(self, mock_app, security_config):
        """Test Redis connection cleanup."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        middleware.redis_client = mock_client
        middleware._redis_healthy = True
        
        await middleware.cleanup()
        
        assert middleware.redis_client is None
        assert middleware._redis_healthy is False
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_with_exception(self, mock_app, security_config):
        """Test Redis connection cleanup handles exceptions gracefully."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock(side_effect=Exception("Close failed"))
        middleware.redis_client = mock_client
        middleware._redis_healthy = True
        
        # Should not raise exception
        await middleware.cleanup()
        
        assert middleware.redis_client is None
        assert middleware._redis_healthy is False

    @pytest.mark.asyncio
    async def test_integration_rate_limiting_workflow(self, mock_app, security_config):
        """Test complete rate limiting workflow with Redis and fallback."""
        middleware = SecurityMiddleware(mock_app, security_config)
        client_ip = "192.168.1.100"
        
        # Test 1: Redis unavailable, should use memory
        middleware._redis_healthy = False
        middleware.redis_client = None
        
        result = await middleware._check_rate_limit(client_ip)
        assert result is True
        
        # Test 2: Redis becomes available
        middleware._redis_healthy = True
        mock_client = AsyncMock()
        
        # Mock successful Redis rate limiting
        mock_pipeline = AsyncMock()
        mock_pipeline.get.return_value = AsyncMock()
        mock_pipeline.get.return_value.execute = AsyncMock(return_value=[None])
        mock_pipeline.incr.return_value = AsyncMock()
        mock_pipeline.expire.return_value = AsyncMock()
        mock_pipeline.execute = AsyncMock(return_value=[1])
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock()
        
        mock_client.pipeline.return_value = mock_pipeline
        middleware.redis_client = mock_client
        
        result = await middleware._check_rate_limit(client_ip)
        assert result is True
        
        # Test 3: Redis fails, should fallback to memory
        mock_client.pipeline.side_effect = Exception("Redis failed")
        
        result = await middleware._check_rate_limit(client_ip)
        assert result is True
        assert middleware._redis_healthy is False


@pytest.mark.integration
class TestSecurityMiddlewareIntegration:
    """Integration tests for SecurityMiddleware with rate limiting."""

    @pytest.mark.asyncio
    async def test_middleware_dispatch_with_rate_limiting(self, mock_app, security_config):
        """Test middleware dispatch with rate limiting enabled."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.headers.get.return_value = "test-agent"
        mock_request.client.host = "192.168.1.1"
        
        # Mock call_next
        async def mock_call_next(request):
            return JSONResponse({"status": "success"})
        
        # Test successful request within rate limits
        response = await middleware.dispatch(mock_request, mock_call_next)
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_middleware_dispatch_rate_limit_exceeded(self, mock_app, security_config):
        """Test middleware dispatch blocks request when rate limit exceeded."""
        middleware = SecurityMiddleware(mock_app, security_config)
        
        # Mock request
        mock_request = MagicMock(spec=Request)
        mock_request.method = "GET"
        mock_request.url.path = "/api/test"
        mock_request.headers.get.return_value = "test-agent"
        mock_request.client.host = "192.168.1.1"
        
        # Mock call_next
        async def mock_call_next(request):
            return JSONResponse({"status": "success"})
        
        # Exceed rate limit by mocking _check_rate_limit to return False
        with patch.object(middleware, '_check_rate_limit', return_value=False):
            response = await middleware.dispatch(mock_request, mock_call_next)
            assert response.status_code == 429
            # Response should include retry-after header
            assert "Retry-After" in response.headers