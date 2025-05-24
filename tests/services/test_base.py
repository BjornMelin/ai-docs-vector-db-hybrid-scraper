"""Tests for base service functionality."""

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from src.services.base import BaseService
from src.services.errors import APIError


class MockService(BaseService):
    """Mock service for testing base functionality."""

    async def initialize(self) -> None:
        """Initialize mock service."""
        self._client = MagicMock()
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup mock service."""
        self._client = None
        self._initialized = False


class TestBaseService:
    """Test base service functionality."""

    @pytest.fixture
    def mock_service(self):
        """Create mock service instance."""
        return MockService()

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_service):
        """Test async context manager."""
        assert not mock_service._initialized

        async with mock_service.context() as service:
            assert service._initialized
            assert service._client is not None

        assert not service._initialized
        assert service._client is None

    @pytest.mark.asyncio
    async def test_context_manager_error(self, mock_service):
        """Test context manager handles errors."""
        # Mock initialize to raise error
        mock_service.initialize = AsyncMock(side_effect=Exception("Init error"))

        with pytest.raises(Exception, match="Init error"):
            async with mock_service.context():
                pass

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self, mock_service):
        """Test retry with backoff succeeds on first try."""
        mock_func = AsyncMock(return_value="success")

        result = await mock_service._retry_with_backoff(
            mock_func, "arg1", kwarg="value"
        )

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg="value")

    @pytest.mark.asyncio
    async def test_retry_with_backoff_retry(self, mock_service):
        """Test retry with backoff retries on failure."""
        mock_func = AsyncMock(
            side_effect=[Exception("Try 1"), Exception("Try 2"), "success"]
        )

        result = await mock_service._retry_with_backoff(
            mock_func,
            max_retries=3,
            base_delay=0.01,  # Fast for testing
        )

        assert result == "success"
        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_all_fail(self, mock_service):
        """Test retry with backoff fails after max retries."""
        mock_func = AsyncMock(side_effect=Exception("Always fails"))

        with pytest.raises(APIError, match="API call failed after 3 attempts"):
            await mock_service._retry_with_backoff(
                mock_func, max_retries=3, base_delay=0.01
            )

        assert mock_func.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exponential_backoff(self, mock_service):
        """Test exponential backoff timing."""
        mock_func = AsyncMock(
            side_effect=[Exception("Try 1"), Exception("Try 2"), "success"]
        )

        # Track sleep calls
        sleep_calls = []
        original_sleep = asyncio.sleep

        async def mock_sleep(delay):
            sleep_calls.append(delay)
            await original_sleep(0.001)  # Small actual delay for test

        asyncio.sleep = mock_sleep

        try:
            await mock_service._retry_with_backoff(
                mock_func, max_retries=3, base_delay=1.0
            )

            # Check exponential backoff: 1s, 2s
            assert len(sleep_calls) == 2
            assert sleep_calls[0] == 1.0
            assert sleep_calls[1] == 2.0
        finally:
            asyncio.sleep = original_sleep

    def test_validate_initialized(self, mock_service):
        """Test initialization validation."""
        # Not initialized
        with pytest.raises(APIError, match="not initialized"):
            mock_service._validate_initialized()

        # After initialization
        mock_service._initialized = True
        mock_service._validate_initialized()  # Should not raise

    @pytest.mark.asyncio
    async def test_cleanup_on_context_exit(self, mock_service):
        """Test cleanup is called on context exit even with error."""
        mock_service.cleanup = AsyncMock()

        # Normal exit
        async with mock_service.context():
            pass
        mock_service.cleanup.assert_called_once()

        # Exit with error
        mock_service.cleanup.reset_mock()
        with pytest.raises(ValueError):
            async with mock_service.context():
                raise ValueError("Test error")
        mock_service.cleanup.assert_called_once()
