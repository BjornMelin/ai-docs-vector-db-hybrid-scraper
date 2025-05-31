"""Tests for rate limiting utilities."""

import asyncio
import time
from unittest.mock import patch

import pytest
from src.config.models import PerformanceConfig
from src.config.models import UnifiedConfig
from src.services.errors import APIError
from src.services.utilities.rate_limiter import AdaptiveRateLimiter
from src.services.utilities.rate_limiter import RateLimiter
from src.services.utilities.rate_limiter import RateLimitManager


@pytest.fixture
def config():
    """Create test configuration with rate limits."""
    return UnifiedConfig(
        performance=PerformanceConfig(
            default_rate_limits={
                "openai": {"max_calls": 60, "time_window": 60},
                "firecrawl": {"max_calls": 100, "time_window": 60},
                "test_provider": {"max_calls": 10, "time_window": 1},
            }
        )
    )


@pytest.fixture
def rate_limiter():
    """Create a basic rate limiter for testing."""
    return RateLimiter(max_calls=10, time_window=1)  # 10 calls per second


@pytest.fixture
def rate_limit_manager(config):
    """Create a rate limit manager."""
    return RateLimitManager(config)


class TestRateLimiter:
    """Test RateLimiter functionality."""

    def test_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_calls=100, time_window=60, burst_multiplier=2.0)

        assert limiter.max_calls == 100
        assert limiter.time_window == 60
        assert limiter.burst_multiplier == 2.0
        assert limiter.max_tokens == 200  # 100 * 2.0
        assert limiter.tokens == 200
        assert limiter.refill_rate == 100 / 60

    @pytest.mark.asyncio
    async def test_acquire_single_token(self, rate_limiter):
        """Test acquiring a single token."""
        initial_tokens = rate_limiter.tokens

        await rate_limiter.acquire(1)

        assert rate_limiter.tokens < initial_tokens
        # Should have consumed approximately 1 token (with some refill)
        assert rate_limiter.tokens >= initial_tokens - 1.1

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens(self, rate_limiter):
        """Test acquiring multiple tokens."""
        await rate_limiter.acquire(5)

        # Should have consumed approximately 5 tokens
        assert rate_limiter.tokens < rate_limiter.max_tokens - 4

    @pytest.mark.asyncio
    async def test_token_refill(self, rate_limiter):
        """Test token refill over time."""
        # Consume all tokens
        rate_limiter.tokens = 0
        rate_limiter.last_refill = time.time()

        # Wait for refill
        await asyncio.sleep(0.1)  # 0.1 second = 1 token refill

        await rate_limiter.acquire(0)  # Trigger refill calculation

        # Should have refilled approximately 1 token
        assert 0.8 <= rate_limiter.tokens <= 1.2

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self, rate_limiter):
        """Test that rate limiting is enforced."""
        # Consume all tokens
        rate_limiter.tokens = 0
        rate_limiter.last_refill = time.time()

        # Try to acquire tokens - should wait
        start_time = time.time()
        await rate_limiter.acquire(2)  # Need 2 tokens, should wait ~0.2s
        elapsed = time.time() - start_time

        # Should have waited approximately 0.2 seconds
        assert 0.15 <= elapsed <= 0.3

    @pytest.mark.asyncio
    async def test_burst_capacity(self, rate_limiter):
        """Test burst capacity allows exceeding normal rate temporarily."""
        # Should be able to acquire up to burst capacity immediately
        burst_capacity = int(rate_limiter.max_calls * rate_limiter.burst_multiplier)

        for _ in range(burst_capacity):
            await rate_limiter.acquire(1)

        # Tokens should be depleted
        assert rate_limiter.tokens < 1

    @pytest.mark.asyncio
    async def test_exceed_bucket_capacity(self, rate_limiter):
        """Test error when requesting more tokens than bucket capacity."""
        with pytest.raises(APIError, match="exceeds bucket capacity"):
            await rate_limiter.acquire(rate_limiter.max_tokens + 1)

    @pytest.mark.asyncio
    async def test_concurrent_access(self, rate_limiter):
        """Test thread-safe concurrent access."""
        # Set specific token count
        rate_limiter.tokens = 10
        rate_limiter.last_refill = time.time()

        # Acquire tokens concurrently
        tasks = [rate_limiter.acquire(1) for _ in range(5)]
        await asyncio.gather(*tasks)

        # Should have consumed 5 tokens (plus small refill)
        assert rate_limiter.tokens <= 5.5


class TestRateLimitManager:
    """Test RateLimitManager functionality."""

    def test_initialization(self, config):
        """Test rate limit manager initialization."""
        manager = RateLimitManager(config)

        assert manager.default_limits == config.performance.default_rate_limits
        assert len(manager.limiters) == 0

    def test_get_limiter_creates_new(self, rate_limit_manager):
        """Test getting a limiter creates it if not exists."""
        limiter = rate_limit_manager.get_limiter("openai")

        assert isinstance(limiter, RateLimiter)
        assert limiter.max_calls == 60
        assert limiter.time_window == 60
        assert "openai" in rate_limit_manager.limiters

    def test_get_limiter_returns_existing(self, rate_limit_manager):
        """Test getting existing limiter returns same instance."""
        limiter1 = rate_limit_manager.get_limiter("openai")
        limiter2 = rate_limit_manager.get_limiter("openai")

        assert limiter1 is limiter2

    def test_get_limiter_with_endpoint(self, rate_limit_manager):
        """Test getting limiter with specific endpoint."""
        limiter = rate_limit_manager.get_limiter("openai", "embeddings")

        assert "openai:embeddings" in rate_limit_manager.limiters
        assert limiter.max_calls == 60

    def test_get_limiter_unknown_provider(self, rate_limit_manager):
        """Test error for unknown provider."""
        with pytest.raises(ValueError, match="No rate limits configured"):
            rate_limit_manager.get_limiter("unknown_provider")

    @pytest.mark.asyncio
    async def test_acquire_through_manager(self, rate_limit_manager):
        """Test acquiring tokens through manager."""
        with patch.object(RateLimiter, "acquire") as mock_acquire:
            mock_acquire.return_value = None

            await rate_limit_manager.acquire("openai", tokens=2)

            mock_acquire.assert_called_once_with(2)

    @pytest.mark.asyncio
    async def test_acquire_with_endpoint(self, rate_limit_manager):
        """Test acquiring tokens for specific endpoint."""
        await rate_limit_manager.acquire("openai", "embeddings", tokens=1)

        # Should have created endpoint-specific limiter
        assert "openai:embeddings" in rate_limit_manager.limiters

    def test_multiple_providers(self, rate_limit_manager):
        """Test managing multiple providers."""
        openai_limiter = rate_limit_manager.get_limiter("openai")
        firecrawl_limiter = rate_limit_manager.get_limiter("firecrawl")

        assert openai_limiter.max_calls == 60
        assert firecrawl_limiter.max_calls == 100
        assert openai_limiter is not firecrawl_limiter


class TestAdaptiveRateLimiter:
    """Test AdaptiveRateLimiter functionality."""

    def test_initialization(self):
        """Test adaptive rate limiter initialization."""
        limiter = AdaptiveRateLimiter(
            initial_max_calls=100, time_window=60, min_rate=0.2, max_rate=1.5
        )

        assert limiter.initial_max_calls == 100
        assert limiter.min_rate == 0.2
        assert limiter.max_rate == 1.5
        assert limiter.adjustment_factor == 1.0

    @pytest.mark.asyncio
    async def test_handle_rate_limit_response(self):
        """Test handling 429 rate limit response."""
        limiter = AdaptiveRateLimiter()
        initial_factor = limiter.adjustment_factor

        await limiter.handle_response(429)

        # Should reduce rate by 50%
        assert limiter.adjustment_factor == initial_factor * 0.5

    @pytest.mark.asyncio
    async def test_handle_success_response(self):
        """Test handling successful response."""
        limiter = AdaptiveRateLimiter()
        limiter.adjustment_factor = 0.5  # Start low

        await limiter.handle_response(200)

        # Should increase rate by 5%
        assert limiter.adjustment_factor == 0.5 * 1.05

    @pytest.mark.asyncio
    async def test_min_rate_enforcement(self):
        """Test minimum rate enforcement."""
        limiter = AdaptiveRateLimiter(min_rate=0.2)
        limiter.adjustment_factor = 0.3

        await limiter.handle_response(429)  # Should reduce by 50%

        # Should be capped at min_rate
        assert limiter.adjustment_factor == 0.2

    @pytest.mark.asyncio
    async def test_max_rate_enforcement(self):
        """Test maximum rate enforcement."""
        limiter = AdaptiveRateLimiter(max_rate=1.5)
        limiter.adjustment_factor = 1.45

        await limiter.handle_response(200)  # Should increase by 5%

        # Should be capped at max_rate
        assert limiter.adjustment_factor == 1.5

    @pytest.mark.asyncio
    async def test_handle_response_with_headers(self):
        """Test handling response with rate limit headers."""
        limiter = AdaptiveRateLimiter(initial_max_calls=60)

        headers = {
            "x-ratelimit-remaining": "40",
            "x-ratelimit-limit": "100",
            "x-ratelimit-reset": "1234567890",
        }

        await limiter.handle_response(200, headers)

        # Should adjust max_calls based on headers
        # Since adjustment_factor increases by 5% (from 1.0 to 1.05)
        # max_calls = 100 * 1.05 = 105
        assert limiter.max_calls == 105  # limit * adjustment_factor
        assert limiter.max_tokens == int(105 * limiter.burst_multiplier)

    @pytest.mark.asyncio
    async def test_handle_response_invalid_headers(self):
        """Test handling response with invalid headers."""
        limiter = AdaptiveRateLimiter()
        initial_max_calls = limiter.max_calls

        headers = {
            "x-ratelimit-remaining": "invalid",
            "x-ratelimit-limit": "also-invalid",
        }

        # Should not crash and maintain previous values
        await limiter.handle_response(200, headers)

        assert limiter.max_calls == initial_max_calls

    @pytest.mark.asyncio
    async def test_refill_rate_update(self):
        """Test that refill rate updates with max_calls."""
        limiter = AdaptiveRateLimiter(initial_max_calls=60, time_window=60)
        initial_refill = limiter.refill_rate

        headers = {"x-ratelimit-limit": "120", "x-ratelimit-remaining": "100"}
        await limiter.handle_response(200, headers)

        # Refill rate should have updated
        assert limiter.refill_rate != initial_refill
        assert limiter.refill_rate == limiter.max_calls / limiter.time_window

    @pytest.mark.asyncio
    async def test_concurrent_response_handling(self):
        """Test thread-safe concurrent response handling."""
        limiter = AdaptiveRateLimiter()

        # Handle multiple responses concurrently
        tasks = [
            limiter.handle_response(200),
            limiter.handle_response(429),
            limiter.handle_response(200),
        ]
        await asyncio.gather(*tasks)

        # Should have processed all without errors
        assert 0.1 <= limiter.adjustment_factor <= 2.0
