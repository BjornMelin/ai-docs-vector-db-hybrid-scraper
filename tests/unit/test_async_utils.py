"""Unit tests for async utilities."""

import asyncio

import pytest

from src.utils.async_utils import gather_limited, gather_with_taskgroup


class TestAsyncUtils:
    """Test async utility functions."""

    @pytest.mark.asyncio
    async def test_gather_with_taskgroup_basic(self):
        """Test basic gather_with_taskgroup functionality."""

        async def task(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        results = await gather_with_taskgroup(
            task(1),
            task(2),
            task(3),
        )

        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_gather_with_taskgroup_return_exceptions(self):
        """Test gather_with_taskgroup with return_exceptions=True."""

        async def good_task(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        async def bad_task() -> int:
            await asyncio.sleep(0.01)
            msg = "Test error"
            raise ValueError(msg)

        results = await gather_with_taskgroup(
            good_task(1),
            bad_task(),
            good_task(3),
            return_exceptions=True,
        )

        assert results[0] == 2
        assert isinstance(results[1], ValueError)
        assert str(results[1]) == "Test error"
        assert results[2] == 6

    @pytest.mark.asyncio
    async def test_gather_with_taskgroup_raises_exception_group(self):
        """Test that gather_with_taskgroup raises ExceptionGroup when return_exceptions=False."""

        async def good_task(value: int) -> int:
            await asyncio.sleep(0.01)
            return value * 2

        async def bad_task() -> int:
            await asyncio.sleep(0.01)
            msg = "Test error"
            raise ValueError(msg)

        with pytest.raises(ExceptionGroup) as exc_info:
            await gather_with_taskgroup(
                good_task(1),
                bad_task(),
                good_task(3),
                return_exceptions=False,
            )

        # Check that the ExceptionGroup contains our ValueError
        assert len(exc_info.value.exceptions) == 1
        assert isinstance(exc_info.value.exceptions[0], ValueError)
        assert str(exc_info.value.exceptions[0]) == "Test error"

    @pytest.mark.asyncio
    async def test_gather_with_taskgroup_empty(self):
        """Test gather_with_taskgroup with no coroutines."""
        results = await gather_with_taskgroup()
        assert results == []

    @pytest.mark.asyncio
    async def test_gather_limited_basic(self):
        """Test gather_limited with concurrency control."""

        counter = 0
        max_concurrent = 0

        async def task(value: int) -> int:
            nonlocal counter, max_concurrent
            counter += 1
            max_concurrent = max(max_concurrent, counter)
            await asyncio.sleep(0.05)
            counter -= 1
            return value * 2

        # Create 10 tasks but limit to 3 concurrent
        tasks = [task(i) for i in range(10)]
        results = await gather_limited(*tasks, limit=3)

        assert results == [i * 2 for i in range(10)]
        assert max_concurrent <= 3

    @pytest.mark.asyncio
    async def test_gather_limited_with_exceptions(self):
        """Test gather_limited with exceptions and return_exceptions=True."""

        async def task(value: int) -> int:
            await asyncio.sleep(0.01)
            if value == 5:
                msg = f"Error on {value}"
                raise ValueError(msg)
            return value * 2

        tasks = [task(i) for i in range(10)]
        results = await gather_limited(*tasks, limit=3, return_exceptions=True)

        # Check all results except index 5
        for i in range(10):
            if i == 5:
                assert isinstance(results[i], ValueError)
                assert str(results[i]) == "Error on 5"
            else:
                assert results[i] == i * 2

    @pytest.mark.asyncio
    async def test_migration_pattern_comparison(self):
        """Test that the new pattern produces the same results as asyncio.gather."""

        async def task(value: int) -> int:
            await asyncio.sleep(0.01)
            if value == 2:
                msg = f"Error on {value}"
                raise ValueError(msg)
            return value * 2

        # Old pattern with asyncio.gather
        tasks_old = [task(i) for i in range(5)]
        results_old = await asyncio.gather(*tasks_old, return_exceptions=True)

        # New pattern with gather_with_taskgroup
        tasks_new = [task(i) for i in range(5)]
        results_new = await gather_with_taskgroup(*tasks_new, return_exceptions=True)

        # Results should be identical
        assert len(results_old) == len(results_new)
        for old, new in zip(results_old, results_new, strict=False):
            if isinstance(old, Exception) and isinstance(new, Exception):
                assert type(old) is type(new)
                assert str(old) == str(new)
            else:
                assert old == new
