"""Async utilities for structured concurrency with TaskGroup.

This module provides utilities for migrating from asyncio.gather to TaskGroup
with proper exception handling and backwards compatibility patterns.
"""

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar


T = TypeVar("T")


async def gather_with_taskgroup(
    *coros: Coroutine[Any, Any, T],
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Execute multiple coroutines concurrently using TaskGroup.

    This function provides a migration path from asyncio.gather to TaskGroup
    with similar behavior for exception handling.

    Args:
        *coros: Coroutines to execute concurrently
        return_exceptions: If True, exceptions are returned as results rather than
            raised

    Returns:
        List of results in the same order as the input coroutines

    Raises:
        ExceptionGroup: If any coroutine raises and return_exceptions is False
    """
    if not coros:
        return []

    results: list[T | BaseException] = [None] * len(coros)  # type: ignore[assignment]

    async def run_with_index(idx: int, coro: Coroutine[Any, Any, T]) -> None:
        """Run coroutine and store result at specific index."""
        try:
            results[idx] = await coro
        except BaseException as e:
            if return_exceptions:
                results[idx] = e
            else:
                raise

    if return_exceptions:
        # When return_exceptions=True, we need to catch all exceptions
        try:
            async with asyncio.TaskGroup() as tg:
                for idx, coro in enumerate(coros):
                    tg.create_task(run_with_index(idx, coro))
        except* (
            TimeoutError,
            asyncio.CancelledError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as eg:
            # With return_exceptions=True, exceptions are already stored in results
            # We just need to handle any unexpected errors
            for e in eg.exceptions:
                if not isinstance(e, Exception | asyncio.CancelledError):
                    # Re-raise system exceptions
                    raise e from None
    else:
        # When return_exceptions=False, let TaskGroup handle exceptions normally
        async with asyncio.TaskGroup() as tg:
            for idx, coro in enumerate(coros):
                tg.create_task(run_with_index(idx, coro))

    return results


async def gather_limited(
    *coros: Coroutine[Any, Any, T],
    limit: int,
    return_exceptions: bool = False,
) -> list[T | BaseException]:
    """Execute coroutines with concurrency limit using TaskGroup.

    Args:
        *coros: Coroutines to execute
        limit: Maximum number of concurrent tasks
        return_exceptions: If True, exceptions are returned as results

    Returns:
        List of results in the same order as input
    """
    if not coros:
        return []

    results: list[T | BaseException] = [None] * len(coros)  # type: ignore[assignment]
    semaphore = asyncio.Semaphore(limit)

    async def run_with_limit(idx: int, coro: Coroutine[Any, Any, T]) -> None:
        """Run coroutine with semaphore limit."""
        async with semaphore:
            try:
                results[idx] = await coro
            except BaseException as e:
                if return_exceptions:
                    results[idx] = e
                else:
                    raise

    if return_exceptions:
        try:
            async with asyncio.TaskGroup() as tg:
                for idx, coro in enumerate(coros):
                    tg.create_task(run_with_limit(idx, coro))
        except* (
            TimeoutError,
            asyncio.CancelledError,
            RuntimeError,
            ValueError,
            TypeError,
        ) as eg:
            # Handle any unexpected errors
            for e in eg.exceptions:
                if not isinstance(e, Exception | asyncio.CancelledError):
                    raise e from None
    else:
        async with asyncio.TaskGroup() as tg:
            for idx, coro in enumerate(coros):
                tg.create_task(run_with_limit(idx, coro))

    return results
