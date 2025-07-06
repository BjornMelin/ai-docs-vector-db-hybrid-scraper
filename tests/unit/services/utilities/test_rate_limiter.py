"""Tests for rate limiting utilities."""

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.config import Config
from src.services.errors import APIError
from src.services.utilities.rate_limiter import (
    AdaptiveRateLimiter,
    RateLimiter,
    RateLimitManager,
)


class TestRateLimiter:
    """Tests for RateLimiter class."""

    def test_init_default_params(self):
        """Test rate limiter initialization with default parameters."""
        limiter = RateLimiter(max_calls=60)

        assert limiter.max_calls == 60
        assert limiter.time_window == 60
        assert limiter.burst_multiplier == 1.5
        assert limiter.max_tokens == 90  # 60 * 1.5
        assert limiter.tokens == 90
        assert limiter.refill_rate == 1.0  # 60/60
        assert limiter.last_refill > 0

    def test_init_custom_params(self):
        """Test rate limiter initialization with custom parameters."""
        limiter = RateLimiter(
            max_calls=100,
            time_window=30,
            burst_multiplier=2.0,
        )

        assert limiter.max_calls == 100
        assert limiter.time_window == 30
        assert limiter.burst_multiplier == 2.0
        assert limiter.max_tokens == 200  # 100 * 2.0
        assert limiter.tokens == 200
        assert abs(limiter.refill_rate - (100 / 30)) < 0.0001  # 100/30

    @pytest.mark.asyncio
    async def test_acquire_single_token_success(self):
        """Test acquiring a single token successfully."""
        limiter = RateLimiter(max_calls=60)
        initial_tokens = limiter.tokens

        await limiter.acquire(1)

        assert limiter.tokens == initial_tokens - 1

    @pytest.mark.asyncio
    async def test_acquire_multiple_tokens_success(self):
        """Test acquiring multiple tokens successfully."""
        limiter = RateLimiter(max_calls=60)
        initial_tokens = limiter.tokens

        await limiter.acquire(5)

        assert limiter.tokens == initial_tokens - 5

    @pytest.mark.asyncio
    async def test_acquire_exceeds_capacity_error(self):
        """Test acquiring more tokens than bucket capacity."""
        limiter = RateLimiter(max_calls=60)  # max_tokens = 90

        with pytest.raises(APIError) as exc_info:
            await limiter.acquire(100)

        assert "exceeds bucket capacity" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acquire_with_refill(self):
        """Test token refill during acquisition."""
        limiter = RateLimiter(max_calls=60, time_window=1)  # 60 tokens per second

        # Consume some tokens
        await limiter.acquire(30)
        initial_tokens = limiter.tokens

        # Wait for some refill
        await asyncio.sleep(0.1)  # 0.1 seconds = 6 tokens refill

        await limiter.acquire(1)

        # Should have refilled approximately 6 tokens minus the 1 we acquired
        assert limiter.tokens > initial_tokens - 1

    @pytest.mark.asyncio
    async def test_acquire_wait_for_refill(self):
        """Test waiting for token refill when insufficient tokens."""
        limiter = RateLimiter(max_calls=10, time_window=1)  # 10 tokens per second

        # Consume all tokens
        await limiter.acquire(15)  # Uses all 15 burst tokens

        start_time = time.time()

        # This should wait for refill
        await limiter.acquire(5)

        elapsed = time.time() - start_time

        # Should have waited approximately 0.5 seconds for 5 tokens
        assert elapsed >= 0.4  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_acquire_concurrent_access(self):
        """Test concurrent token acquisition."""
        limiter = RateLimiter(max_calls=60)
        initial_tokens = limiter.tokens

        # Create multiple concurrent acquisitions
        tasks = [limiter.acquire(1) for _ in range(10)]
        await asyncio.gather(*tasks)

        # Allow for small timing variations due to refill
        assert abs(limiter.tokens - (initial_tokens - 10)) < 1.0

    @pytest.mark.asyncio
    async def test_token_refill_capped_at_maximum(self):
        """Test that token refill is capped at maximum capacity."""
        limiter = RateLimiter(max_calls=60, time_window=1)

        # Consume some tokens
        await limiter.acquire(30)

        # Manually set last_refill to simulate long time passage
        limiter.last_refill = time.time() - 100  # 100 seconds ago

        # Acquire one token to trigger refill
        await limiter.acquire(1)

        # Tokens should be capped at max_tokens minus the one we just acquired
        assert limiter.tokens == limiter.max_tokens - 1

    @pytest.mark.asyncio
    async def test_refill_rate_calculation(self):
        """Test token refill rate calculation."""
        limiter = RateLimiter(max_calls=120, time_window=60)

        # Refill rate should be 2 tokens per second
        assert limiter.refill_rate == 2.0

        # Test with different parameters
        limiter2 = RateLimiter(max_calls=30, time_window=10)
        assert limiter2.refill_rate == 3.0

    @pytest.mark.asyncio
    async def test_acquire_exact_capacity(self):
        """Test acquiring exactly the bucket capacity."""
        limiter = RateLimiter(max_calls=60)  # max_tokens = 90

        await limiter.acquire(90)

        assert limiter.tokens == 0

    @pytest.mark.asyncio
    async def test_acquire_with_fractional_refill(self):
        """Test token acquisition with fractional refill amounts."""
        limiter = RateLimiter(max_calls=100, time_window=3)  # ~33.33 tokens per second

        # Consume tokens
        await limiter.acquire(50)
        tokens_after_first = limiter.tokens

        # Wait a short time for partial refill
        await asyncio.sleep(0.03)  # Should refill ~1 token

        await limiter.acquire(1)

        # Should have approximately the same tokens (refill ~= acquired)
        assert abs(limiter.tokens - (tokens_after_first - 1)) <= 2


class TestRateLimitManager:
    """Tests for RateLimitManager class."""

    @pytest.fixture
    def mock_config(self):
        """Create mock unified config."""
        config = MagicMock(spec=Config)
        mock_performance = MagicMock()
        mock_performance.default_rate_limits = {
            "openai": {"max_calls": 60, "time_window": 60},
            "firecrawl": {"max_calls": 100, "time_window": 60},
            "qdrant": {"max_calls": 1000, "time_window": 60},
        }
        config.performance = mock_performance
        return config

    def test_init(self, mock_config):
        """Test rate limit manager initialization."""
        manager = RateLimitManager(mock_config)

        assert manager.limiters == {}
        assert manager.default_limits == mock_config.performance.default_rate_limits

    def test_get_limiter_new_provider(self, mock_config):
        """Test getting limiter for new provider."""
        manager = RateLimitManager(mock_config)

        limiter = manager.get_limiter("openai")

        assert isinstance(limiter, RateLimiter)
        assert limiter.max_calls == 60
        assert limiter.time_window == 60
        assert "openai" in manager.limiters

    def test_get_limiter_with_endpoint(self, mock_config):
        """Test getting limiter for provider with specific endpoint."""
        manager = RateLimitManager(mock_config)

        limiter = manager.get_limiter("openai", "embeddings")

        assert isinstance(limiter, RateLimiter)
        assert "openai:embeddings" in manager.limiters

    def test_get_limiter_cached(self, mock_config):
        """Test getting cached limiter."""
        manager = RateLimitManager(mock_config)

        limiter1 = manager.get_limiter("openai")
        limiter2 = manager.get_limiter("openai")

        assert limiter1 is limiter2

    def test_get_limiter_unknown_provider(self, mock_config):
        """Test getting limiter for unknown provider."""
        manager = RateLimitManager(mock_config)

        with pytest.raises(ValueError) as exc_info:
            manager.get_limiter("unknown_provider")

        assert "No rate limits configured" in str(exc_info.value)
        assert "unknown_provider" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_acquire_success(self, mock_config):
        """Test successful token acquisition through manager."""
        manager = RateLimitManager(mock_config)

        # This should create and use a limiter
        await manager.acquire("openai", tokens=5)

        # Verify limiter was created and tokens consumed
        limiter = manager.limiters["openai"]
        assert limiter.max_calls == 60
        # Can't check exact tokens due to potential refill, but limiter should exist

    @pytest.mark.asyncio
    async def test_acquire_with_endpoint(self, mock_config):
        """Test token acquisition with specific endpoint."""
        manager = RateLimitManager(mock_config)

        await manager.acquire("openai", "chat", tokens=3)

        assert "openai:chat" in manager.limiters

    @pytest.mark.asyncio
    async def test_acquire_unknown_provider(self, mock_config):
        """Test acquisition with unknown provider."""
        manager = RateLimitManager(mock_config)

        with pytest.raises(ValueError):
            await manager.acquire("unknown_provider")

    def test_multiple_providers(self, mock_config):
        """Test managing multiple providers."""
        manager = RateLimitManager(mock_config)

        openai_limiter = manager.get_limiter("openai")
        firecrawl_limiter = manager.get_limiter("firecrawl")
        qdrant_limiter = manager.get_limiter("qdrant")

        assert len(manager.limiters) == 3
        assert openai_limiter.max_calls == 60
        assert firecrawl_limiter.max_calls == 100
        assert qdrant_limiter.max_calls == 1000

    def test_provider_endpoint_combinations(self, mock_config):
        """Test different provider and endpoint combinations."""
        manager = RateLimitManager(mock_config)

        # Same provider, different endpoints
        limiter1 = manager.get_limiter("openai", "embeddings")
        limiter2 = manager.get_limiter("openai", "chat")
        limiter3 = manager.get_limiter("openai")  # No endpoint

        assert limiter1 is not limiter2
        assert limiter1 is not limiter3
        assert limiter2 is not limiter3
        assert len(manager.limiters) == 3


class TestAdaptiveRateLimiter:
    """Tests for AdaptiveRateLimiter class."""

    def test_init_default_params(self):
        """Test adaptive rate limiter initialization with default parameters."""
        limiter = AdaptiveRateLimiter()

        assert limiter.initial_max_calls == 60
        assert limiter.max_calls == 60
        assert limiter.time_window == 60
        assert limiter.min_rate == 0.1
        assert limiter.max_rate == 2.0
        assert limiter.adjustment_factor == 1.0

    def test_init_custom_params(self):
        """Test adaptive rate limiter initialization with custom parameters."""
        limiter = AdaptiveRateLimiter(
            initial_max_calls=100,
            time_window=30,
            min_rate=0.2,
            max_rate=3.0,
        )

        assert limiter.initial_max_calls == 100
        assert limiter.max_calls == 100
        assert limiter.time_window == 30
        assert limiter.min_rate == 0.2
        assert limiter.max_rate == 3.0

    @pytest.mark.asyncio
    async def test_handle_response_rate_limited(self):
        """Test handling rate limited response (429)."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)
        initial_factor = limiter.adjustment_factor

        await limiter.handle_response(429)

        # Should reduce adjustment factor by 50%
        assert limiter.adjustment_factor == initial_factor * 0.5
        assert limiter.adjustment_factor >= limiter.min_rate

    @pytest.mark.asyncio
    async def test_handle_response_success(self):
        """Test handling successful response."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)
        # Start with reduced rate
        limiter.adjustment_factor = 0.5

        await limiter.handle_response(200)

        # Should increase adjustment factor by 5%
        assert limiter.adjustment_factor == 0.5 * 1.05

    @pytest.mark.asyncio
    async def test_handle_response_multiple_rate_limits(self):
        """Test multiple rate limit responses."""
        limiter = AdaptiveRateLimiter()

        # Multiple 429 responses should keep reducing rate
        await limiter.handle_response(429)
        factor_after_first = limiter.adjustment_factor

        await limiter.handle_response(429)
        factor_after_second = limiter.adjustment_factor

        assert factor_after_second < factor_after_first
        assert factor_after_second >= limiter.min_rate

    @pytest.mark.asyncio
    async def test_handle_response_rate_recovery(self):
        """Test rate recovery after successful responses."""
        limiter = AdaptiveRateLimiter()

        # First get rate limited
        await limiter.handle_response(429)
        low_factor = limiter.adjustment_factor

        # Then get several successful responses
        for _ in range(20):
            await limiter.handle_response(200)

        # Rate should have recovered somewhat
        assert limiter.adjustment_factor > low_factor
        assert limiter.adjustment_factor <= limiter.max_rate

    @pytest.mark.asyncio
    async def test_handle_response_with_rate_limit_headers(self):
        """Test handling response with rate limit headers."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)

        headers = {
            "x-ratelimit-remaining": "30",
            "x-ratelimit-limit": "60",
            "x-ratelimit-reset": "1234567890",
        }

        await limiter.handle_response(200, headers)

        # Should adjust max_calls based on actual limit and adjustment factor
        # Since it's a success (200), adjustment factor is increased to 1.05
        expected_calls = int(60 * limiter.adjustment_factor)
        assert limiter.max_calls == expected_calls

    @pytest.mark.asyncio
    async def test_handle_response_invalid_headers(self):
        """Test handling response with invalid rate limit headers."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)
        initial_max_calls = limiter.max_calls

        headers = {
            "x-ratelimit-remaining": "invalid",
            "x-ratelimit-limit": "not_a_number",
        }

        await limiter.handle_response(200, headers)

        # Should ignore invalid headers and not change max_calls
        assert limiter.max_calls == initial_max_calls

    @pytest.mark.asyncio
    async def test_handle_response_partial_headers(self):
        """Test handling response with partial rate limit headers."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)

        # Only remaining header, missing limit
        headers = {"x-ratelimit-remaining": "30"}

        await limiter.handle_response(200, headers)

        # Should not adjust based on incomplete headers
        # but should still adjust the factor for successful response
        assert limiter.adjustment_factor == 1.05

    @pytest.mark.asyncio
    async def test_adjustment_factor_bounds(self):
        """Test that adjustment factor stays within bounds."""
        limiter = AdaptiveRateLimiter(min_rate=0.1, max_rate=2.0)

        # Test lower bound
        limiter.adjustment_factor = 0.2
        await limiter.handle_response(429)  # Should reduce by 50%
        assert limiter.adjustment_factor == 0.1  # Clamped to min_rate

        # Test upper bound
        limiter.adjustment_factor = 1.9
        for _ in range(5):
            await limiter.handle_response(200)  # Each increases by 5%
        assert limiter.adjustment_factor <= 2.0  # Clamped to max_rate

    @pytest.mark.asyncio
    async def test_refill_rate_updates_with_adjustment(self):
        """Test that refill rate updates when max_calls changes."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100, time_window=60)
        initial_adjustment_factor = limiter.adjustment_factor

        # Provide rate limit headers to trigger max_calls update
        headers = {
            "x-ratelimit-remaining": "30",
            "x-ratelimit-limit": "60",
        }

        # Force a rate limit response to trigger adjustment
        await limiter.handle_response(429, headers)

        # Verify adjustment factor changed
        assert limiter.adjustment_factor < initial_adjustment_factor

        # Since headers were provided and adjustment factor changed,
        # max_calls should update
        expected_max_calls = int(60 * limiter.adjustment_factor)
        assert limiter.max_calls == expected_max_calls
        assert limiter.refill_rate == limiter.max_calls / limiter.time_window

    @pytest.mark.asyncio
    async def test_concurrent_response_handling(self):
        """Test concurrent response handling thread safety."""
        limiter = AdaptiveRateLimiter()

        # Handle multiple responses concurrently
        tasks = [
            limiter.handle_response(200),
            limiter.handle_response(429),
            limiter.handle_response(200),
            limiter.handle_response(200),
        ]

        await asyncio.gather(*tasks)

        # Should have valid adjustment factor
        assert 0.1 <= limiter.adjustment_factor <= 2.0

    @pytest.mark.asyncio
    async def test_rate_limit_header_priority(self):
        """Test that rate limit headers take priority over status codes."""
        limiter = AdaptiveRateLimiter(initial_max_calls=100)

        # Successful status but low remaining in headers
        headers = {
            "x-ratelimit-remaining": "5",
            "x-ratelimit-limit": "50",
        }

        await limiter.handle_response(200, headers)

        # Should use header information to set max_calls
        assert limiter.max_calls == int(50 * limiter.adjustment_factor)

    def test_inheritance_from_rate_limiter(self):
        """Test that AdaptiveRateLimiter properly inherits from RateLimiter."""
        limiter = AdaptiveRateLimiter()

        # Should have all RateLimiter attributes
        assert hasattr(limiter, "max_calls")
        assert hasattr(limiter, "time_window")
        assert hasattr(limiter, "tokens")
        assert hasattr(limiter, "refill_rate")
        assert hasattr(limiter, "_lock")

        # Should be able to call parent methods
        assert callable(limiter.acquire)

    @pytest.mark.asyncio
    async def test_adaptive_acquire_functionality(self):
        """Test that adaptive limiter can still acquire tokens."""
        limiter = AdaptiveRateLimiter(initial_max_calls=60)
        initial_tokens = limiter.tokens

        await limiter.acquire(5)

        assert limiter.tokens == initial_tokens - 5
