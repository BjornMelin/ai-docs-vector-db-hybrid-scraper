"""Namespace package for shared test fixtures."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING


__all__ = [
    "async_fixtures",
    "async_isolation",
    "configuration",
    "factories",
    "parallel_config",
    "test_data",
]


if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from . import (
        async_fixtures,
        async_isolation,
        configuration,
        factories,
        parallel_config,
        test_data,
    )


def __getattr__(name: str) -> ModuleType:
    """Lazily import fixture modules to avoid plugin rewrite conflicts."""
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
