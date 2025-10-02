"""Compatibility helpers for optional dependencies.

Provides a single import point for the optional ``pydantic_ai`` package
without raising an ``ImportError`` at module import time.

This file intentionally has no external runtime side effects.
"""

from __future__ import annotations

import logging
from importlib import import_module
from typing import Any


logger = logging.getLogger(__name__)


def load_pydantic_ai() -> tuple[bool, Any, Any]:
    """Return availability and key classes from ``pydantic_ai``.

    The function never raises. On success it returns a tuple
    ``(True, Agent, RunContext)``. If the package is missing or does not
    expose the expected attributes, it returns ``(False, None, None)``.

    Returns:
        tuple[bool, Any, Any]: Availability flag, Agent class, RunContext
        class.
    """

    try:
        module = import_module("pydantic_ai")
        agent_cls = module.Agent
        run_ctx_cls = module.RunContext
        return True, agent_cls, run_ctx_cls
    except (ModuleNotFoundError, ImportError, AttributeError) as exc:
        logger.debug("pydantic_ai not available: %s", exc)
        return False, None, None


__all__ = ["load_pydantic_ai"]
