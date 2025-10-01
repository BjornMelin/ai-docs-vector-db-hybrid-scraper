"""Tests for coordination strategy selection heuristics."""

from src.services.agents._shared import choose_coordination_strategy
from src.services.agents.coordination import (
    CoordinationStrategy,
    TaskDefinition,
    TaskPriority,
)


def _task(
    index: int,
    duration_ms: float = 500.0,
    dependencies: list[str] | None = None,
    priority: TaskPriority = TaskPriority.NORMAL,
) -> TaskDefinition:
    """Construct a synthetic task definition for heuristic checks."""
    return TaskDefinition(
        task_id=f"t{index}",
        description="",
        priority=priority,
        estimated_duration_ms=duration_ms,
        dependencies=dependencies or [],
        required_capabilities=[],
        input_data={},
    )


def test_single_task_sequential() -> None:
    """Fallback to sequential strategy when only one task exists."""
    tasks = [_task(1)]
    assert choose_coordination_strategy(tasks) == CoordinationStrategy.SEQUENTIAL


def test_many_deps_pipeline() -> None:
    """Prefer pipeline strategy when linear dependencies dominate."""
    tasks = [
        _task(1, dependencies=["t0"]),
        _task(2, dependencies=["t1"]),
        _task(3, dependencies=["t2"]),
        _task(4, dependencies=[]),
    ]
    assert choose_coordination_strategy(tasks) == CoordinationStrategy.PIPELINE


def test_majority_high_priority_hierarchical() -> None:
    """Select hierarchical planning when high-priority tasks dominate."""
    tasks = [
        _task(1, priority=TaskPriority.HIGH),
        _task(2, priority=TaskPriority.CRITICAL),
        _task(3),
    ]
    assert choose_coordination_strategy(tasks) == CoordinationStrategy.HIERARCHICAL


def test_short_tasks_parallel() -> None:
    """Choose parallel execution when tasks are uniformly short."""
    tasks = [
        _task(1, duration_ms=100.0),
        _task(2, duration_ms=200.0),
        _task(3, duration_ms=300.0),
    ]
    assert choose_coordination_strategy(tasks) == CoordinationStrategy.PARALLEL
