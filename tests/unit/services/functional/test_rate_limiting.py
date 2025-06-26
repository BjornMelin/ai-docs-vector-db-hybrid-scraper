"""Tests for functional rate limiting service."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config import Config
from src.services.functional.rate_limiting import (
    TokenBucket,
    acquire_rate_limit,
    get_rate_limit_status,
    get_rate_limiter,
    handle_api_response,
)


@pytest.fixture
def mock_config():
    """Mock configuration for rate limiting tests."""
    config = MagicMock(spec=Config)
    config.performance = MagicMock()
    config.performance.default_rate_limits = {
        "openai": {"max_calls": 60, "time_window": 60},
        "firecrawl": {"max_calls": 100, "time_window": 60},
    }
    return config


class TestTokenBucket:
    """Test TokenBucket rate limiting implementation."""

    def test_token_bucket_initialization(self):
        """Test token bucket initialization."""
        bucket = TokenBucket(max_calls=60, time_window=60, burst_multiplier=1.5)

        assert bucket.max_calls == 60
        assert bucket.time_window == 60
        assert bucket.max_tokens == 90  # 60 * 1.5
        assert bucket.tokens == 90
        assert bucket.refill_rate == 1.0  # 60 calls / 60 seconds

    @pytest.mark.asyncio
    async def test_token_acquisition_success(self):
        """Test successful token acquisition."""
        bucket = TokenBucket(max_calls=60, time_window=60)

        # Should succeed immediately
        await bucket.acquire(1)
        assert bucket.tokens == 89  # 90 - 1

    @pytest.mark.asyncio
    async def test_token_acquisition_exceeds_capacity(self):
        """Test token acquisition that exceeds bucket capacity."""
        bucket = TokenBucket(max_calls=60, time_window=60, burst_multiplier=1.5)

        # Should raise HTTPException for requests exceeding capacity
        with pytest.raises(Exception):  # HTTPException
            await bucket.acquire(100)  # Exceeds max_tokens (90)

    @pytest.mark.asyncio
    async def test_token_refill(self):
        """Test token bucket refill mechanism."""
        bucket = TokenBucket(max_calls=60, time_window=60)

        # Consume all tokens
        await bucket.acquire(90)
        assert bucket.tokens == 0

        # Simulate time passage (mock time.time())
        import time

        original_time = time.time
        mock_time = 0

        def mock_time_func():
            return mock_time

        time.time = mock_time_func
        bucket.last_refill = 0

        try:
            # Advance time by 30 seconds
            mock_time = 30

            # Try to acquire token (should trigger refill)
            await bucket.acquire(1)

            # Should have refilled 30 tokens (30 seconds * 1 token/second)
            # But we consumed 1, so should have 29 remaining
            assert bucket.tokens == 29
        finally:
            time.time = original_time


class TestRateLimitingFunctions:
    """Test rate limiting functional interface."""

    @pytest.mark.asyncio
    async def test_get_rate_limiter(self, mock_config):
        """Test rate limiter creation and retrieval."""
        # Clear any existing limiters
        from src.services.functional.rate_limiting import _rate_limiters

        _rate_limiters.clear()

        # Get limiter for openai
        limiter = await get_rate_limiter("openai", config=mock_config)

        assert isinstance(limiter, TokenBucket)
        assert limiter.max_calls == 60
        assert limiter.time_window == 60

        # Getting same limiter should return same instance
        limiter2 = await get_rate_limiter("openai", config=mock_config)
        assert limiter is limiter2

    @pytest.mark.asyncio
    async def test_get_rate_limiter_unknown_provider(self, mock_config):
        """Test rate limiter creation for unknown provider."""
        # Clear any existing limiters
        from src.services.functional.rate_limiting import _rate_limiters

        _rate_limiters.clear()

        # Should raise HTTPException for unknown provider
        with pytest.raises(Exception):  # HTTPException
            await get_rate_limiter("unknown_provider", config=mock_config)

    @pytest.mark.asyncio
    async def test_acquire_rate_limit(self, mock_config):
        """Test rate limit acquisition."""
        # Clear any existing limiters
        from src.services.functional.rate_limiting import _rate_limiters

        _rate_limiters.clear()

        # Should succeed
        await acquire_rate_limit("openai", tokens=1, config=mock_config)

        # Limiter should have been created and token consumed
        assert "openai" in _rate_limiters
        limiter = _rate_limiters["openai"]
        assert limiter.tokens == 89  # 90 - 1

    @pytest.mark.asyncio
    async def test_get_rate_limit_status(self, mock_config):
        """Test rate limit status retrieval."""
        # Clear any existing limiters
        from src.services.functional.rate_limiting import _rate_limiters

        _rate_limiters.clear()

        # Create a limiter
        await acquire_rate_limit("openai", tokens=10, config=mock_config)

        # Get status
        status = await get_rate_limit_status("openai", config=mock_config)

        assert status["provider"] == "openai"
        assert status["max_tokens"] == 90
        assert status["current_tokens"] == 80  # 90 - 10
        assert status["utilization"] == pytest.approx(10 / 90, rel=1e-2)

    @pytest.mark.asyncio
    async def test_handle_api_response_success(self, mock_config):
        """Test API response handling for successful calls."""
        # This is a no-op for basic TokenBucket, but should not raise
        await handle_api_response("openai", 200, config=mock_config)

    @pytest.mark.asyncio
    async def test_handle_api_response_rate_limit(self, mock_config):
        """Test API response handling for rate limit errors."""
        # This is a no-op for basic TokenBucket, but should not raise
        await handle_api_response("openai", 429, config=mock_config)


class TestRateLimitingDecorators:
    """Test rate limiting decorators and context managers."""

    @pytest.mark.asyncio
    async def test_rate_limited_decorator(self, mock_config):
        """Test rate limited decorator."""
        from src.services.functional.rate_limiting import _rate_limiters, rate_limited

        # Clear any existing limiters
        _rate_limiters.clear()

        # Mock the get_config dependency
        async def mock_get_config():
            return mock_config

        # Patch dependencies for decorator test
        import src.services.functional.rate_limiting as rl_module

        original_get_config = rl_module.get_config

        try:
            rl_module.get_config = mock_get_config

            @rate_limited("openai", "embeddings")
            async def test_function():
                return "success"

            # Should succeed and create rate limiter
            result = await test_function()
            assert result == "success"

            # Rate limiter should have been created
            assert "openai:embeddings" in _rate_limiters

        finally:
            rl_module.get_config = original_get_config

    @pytest.mark.asyncio
    async def test_rate_limit_context_manager(self, mock_config):
        """Test rate limit context manager."""
        from src.services.functional.rate_limiting import (
            RateLimitContext,
            _rate_limiters,
        )

        # Clear any existing limiters
        _rate_limiters.clear()

        # Mock the acquire_rate_limit function
        async def mock_acquire_rate_limit(provider, endpoint, tokens):
            # Simulate rate limiting
            limiter = TokenBucket(60, 60)
            await limiter.acquire(tokens)
            _rate_limiters[f"{provider}:{endpoint}"] = limiter

        # Patch the function
        import src.services.functional.rate_limiting as rl_module

        original_acquire = rl_module.acquire_rate_limit

        try:
            rl_module.acquire_rate_limit = mock_acquire_rate_limit

            async with RateLimitContext("openai", "embeddings", 2):
                # Should execute without error
                pass

            # Rate limiter should have been used
            assert "openai:embeddings" in _rate_limiters

        finally:
            rl_module.acquire_rate_limit = original_acquire
