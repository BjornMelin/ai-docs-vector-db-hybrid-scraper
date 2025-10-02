"""Shared strategy façade.

This module exposes a single, stable function used by tests and callers to
choose a coordination strategy. Implementation is delegated to the
canonical logic in :mod:`src.services.agents.coordination` to avoid
duplication (DRY).
"""

from __future__ import annotations

# Import types and the canonical chooser from the coordination module.
from src.services.agents.coordination import (
    CoordinationStrategy,
    TaskDefinition,
    TaskPriority,  # re-exported types for test convenience
    choose_coordination_strategy as _choose,
)


def choose_coordination_strategy(
    tasks: list[TaskDefinition],
) -> CoordinationStrategy:
    """Proxy to the canonical strategy selection logic.

    Args:
        tasks: Task definitions to evaluate.

    Returns:
        CoordinationStrategy: Selected strategy.
    """
    return _choose(tasks)


__all__ = [
    "choose_coordination_strategy",
    "TaskDefinition",
    "TaskPriority",
    "CoordinationStrategy",
]
