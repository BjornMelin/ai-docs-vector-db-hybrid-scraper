"""Circuit breaker decorators shared across services."""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.services.errors import ExternalServiceError, NetworkError, RateLimitError


try:  # pragma: no cover - optional purgatory integration
    from purgatory.domain.model import OpenedState  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    OpenedState = type(
        "OpenedState",
        (Exception,),
        {
            "__doc__": "Fallback OpenedState when purgatory is unavailable.",
        },
    )


logger = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])
_LEGACY_ARGS_NOTICE_EMITTED = False


def _build_breaker_kwargs(
    failure_threshold: int,
    recovery_timeout: float,
) -> dict[str, object]:
    """Translate legacy decorator arguments into purgatory parameters."""

    return {
        "threshold": failure_threshold,
        "ttl": recovery_timeout,
    }


async def _call_with_circuit_breaker(
    service_name: str,
    breaker_kwargs: dict[str, object],
    func,
    *args,
    **kwargs,
):
    """Execute an async callable inside a purgatory circuit breaker."""

    manager = await _get_circuit_breaker_manager()
    breaker = await manager.get_breaker(service_name, **breaker_kwargs)
    try:
        async with breaker:
            return await func(*args, **kwargs)
    except OpenedState as exc:
        msg = f"Circuit breaker is open for {service_name}"
        raise ExternalServiceError(
            msg,
            context={"service": service_name, "state": "open"},
        ) from exc


def circuit_breaker(  # pylint: disable=too-many-arguments
    service_name: str | None = None,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    half_open_max_calls: int = 3,  # Retained for backward compatibility
    expected_exceptions: tuple[type[Exception], ...] | None = None,
    *,
    expected_exception: type[Exception] | None = None,
    enable_adaptive_timeout: bool = True,
    enable_metrics: bool = True,
) -> Callable[[F], F]:
    """Wrap an async function with the shared purgatory circuit breaker."""

    if _legacy_params_supplied(
        half_open_max_calls=half_open_max_calls,
        expected_exceptions=expected_exceptions,
        expected_exception=expected_exception,
        enable_adaptive_timeout=enable_adaptive_timeout,
        enable_metrics=enable_metrics,
    ):
        _emit_legacy_args_notice()

    breaker_kwargs = _build_breaker_kwargs(failure_threshold, recovery_timeout)

    def decorator(func: F) -> F:
        breaker_name = service_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def _invoke(*inner_args, **inner_kwargs):
                return await func(*inner_args, **inner_kwargs)

            return await _call_with_circuit_breaker(
                breaker_name,
                breaker_kwargs,
                _invoke,
                *args,
                **kwargs,
            )

        async def _status() -> dict[str, Any]:
            manager = await _get_circuit_breaker_manager()
            return await manager.get_breaker_status(breaker_name)

        async def _reset() -> bool:
            manager = await _get_circuit_breaker_manager()
            return await manager.reset_breaker(breaker_name)

        wrapper.circuit_breaker_name = breaker_name  # type: ignore[attr-defined]
        wrapper.get_circuit_status = _status  # type: ignore[attr-defined]
        wrapper.reset_circuit = _reset  # type: ignore[attr-defined]
        return wrapper  # type: ignore[misc]

    return decorator


def tenacity_circuit_breaker(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    service_name: str | None = None,
    max_attempts: int = 3,
    wait_multiplier: float = 1.0,
    wait_max: float = 10.0,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exceptions: tuple[type[Exception], ...] = (
        ExternalServiceError,
        NetworkError,
        RateLimitError,
    ),
) -> Callable[[F], F]:
    """Combine Tenacity retries with the shared purgatory circuit breaker."""

    breaker_kwargs = _build_breaker_kwargs(failure_threshold, recovery_timeout)

    def decorator(func: F) -> F:
        breaker_name = service_name or f"{func.__module__}.{func.__name__}"

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            async def _invoke(*inner_args, **inner_kwargs):
                return await func(*inner_args, **inner_kwargs)

            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(multiplier=wait_multiplier, max=wait_max),
                retry=retry_if_exception_type(expected_exceptions),
                reraise=True,
            ):
                with attempt:
                    return await _call_with_circuit_breaker(
                        breaker_name,
                        breaker_kwargs,
                        _invoke,
                        *args,
                        **kwargs,
                    )

        async def _status() -> dict[str, Any]:
            manager = await _get_circuit_breaker_manager()
            return await manager.get_breaker_status(breaker_name)

        async def _reset() -> bool:
            manager = await _get_circuit_breaker_manager()
            return await manager.reset_breaker(breaker_name)

        wrapper.circuit_breaker_name = breaker_name  # type: ignore[attr-defined]
        wrapper.get_circuit_status = _status  # type: ignore[attr-defined]
        wrapper.reset_circuit = _reset  # type: ignore[attr-defined]
        return wrapper  # type: ignore[misc]

    return decorator


async def _get_circuit_breaker_manager():
    """Resolve the circuit breaker manager lazily to avoid import cycles."""

    from src.services.circuit_breaker.provider import (  # pylint: disable=import-outside-toplevel
        get_circuit_breaker_manager,
    )

    return await get_circuit_breaker_manager()


def _legacy_params_supplied(
    *,
    half_open_max_calls: int,
    expected_exceptions: tuple[type[Exception], ...] | None,
    expected_exception: type[Exception] | None,
    enable_adaptive_timeout: bool,
    enable_metrics: bool,
) -> bool:
    """Return True when ignored legacy parameters were provided."""

    return any(
        (
            half_open_max_calls != 3,
            expected_exceptions is not None,
            expected_exception is not None,
            not enable_adaptive_timeout,
            not enable_metrics,
        )
    )


def _emit_legacy_args_notice() -> None:
    """Log once when callers rely on ignored legacy parameters."""

    global _LEGACY_ARGS_NOTICE_EMITTED  # pylint: disable=global-statement
    if _LEGACY_ARGS_NOTICE_EMITTED:
        return
    logger.warning(
        "Legacy circuit_breaker parameters (half_open_max_calls, expected_exceptions, "
        "expected_exception, enable_adaptive_timeout, enable_metrics) are ignored. "
        "Please update call sites to the new configuration semantics."
    )
    _LEGACY_ARGS_NOTICE_EMITTED = True


__all__ = [
    "circuit_breaker",
    "tenacity_circuit_breaker",
]
