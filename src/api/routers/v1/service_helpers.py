"""Shared utilities for versioned API routers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from fastapi import HTTPException


T = TypeVar("T")


async def execute_service_call(  # noqa: UP047
    *,
    operation: str,
    logger: logging.Logger,
    coroutine_factory: Callable[[], Awaitable[T]],
    error_detail: str,
    extra: dict[str, Any] | None = None,
) -> T:
    """Execute a service coroutine and normalize unexpected failures.

    Args:
        operation: Descriptive operation name for logging.
        logger: Module logger used to emit diagnostic messages.
        coroutine_factory: Zero-argument callable returning the awaited coroutine.
        error_detail: Message exposed to API consumers when an unexpected error occurs.
        extra: Optional structured logging metadata.

    Returns:
        The awaited result from ``coroutine_factory``.

    Raises:
        HTTPException: When the underlying coroutine raises ``HTTPException`` or an
            unexpected error occurs. Unexpected failures are logged and converted to a
            500 response with ``error_detail`` as the message.
    """
    try:
        return await coroutine_factory()
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.exception("%s failed", operation, extra=extra)
        raise HTTPException(status_code=500, detail=error_detail) from exc
