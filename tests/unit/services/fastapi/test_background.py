"""Tests for FastAPI background task utilities."""

from __future__ import annotations

import asyncio

import pytest

from src.services.fastapi.background import (
    BackgroundTaskManager,
    TaskOptions,
    TaskPriority,
    TaskStatus,
    cleanup_task_manager,
    get_task_manager,
    initialize_task_manager,
    submit_managed_task,
)


async def _async_identity(value: int) -> int:
    """Return the provided value asynchronously."""
    await asyncio.sleep(0)
    return value


def _sync_identity(value: str) -> str:
    """Return the provided value synchronously."""
    return value


@pytest.mark.asyncio
async def test_submit_task_executes_and_returns_result() -> None:
    """BackgroundTaskManager executes submitted coroutine functions."""
    manager = BackgroundTaskManager(max_workers=1)
    await manager.start()
    try:
        task_id = await manager.submit_task(_async_identity, 7)
        result = await manager.wait_for_task(task_id, timeout=1.0)

        assert task_id
        assert result.status is TaskStatus.COMPLETED
        assert result.result == 7
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_submit_task_respects_task_options() -> None:
    """TaskOptions control identifiers and retry configuration."""
    manager = BackgroundTaskManager(max_workers=1)
    await manager.start()
    try:
        options = TaskOptions(
            task_id="custom-task-1",
            priority=TaskPriority.HIGH,
            max_retries=5,
            timeout=0.5,
        )

        task_id = await manager.submit_task(_sync_identity, "data", options=options)
        result = await manager.wait_for_task(task_id, timeout=1.0)

        assert task_id == "custom-task-1"
        assert result.status is TaskStatus.COMPLETED
        assert result.result == "data"
        with manager._task_lock:  # noqa: SLF001 - validated internal state
            task = manager._tasks[task_id]
        assert task.priority is TaskPriority.HIGH
        assert task.max_retries == 5
        assert task.timeout == 0.5
    finally:
        await manager.stop()


@pytest.mark.asyncio
async def test_submit_task_requires_running_manager() -> None:
    """Submitting before start raises a runtime error."""
    manager = BackgroundTaskManager(max_workers=1)

    with pytest.raises(RuntimeError, match="Task manager is not running"):
        await manager.submit_task(_sync_identity, "value")


@pytest.mark.asyncio
async def test_submit_managed_task_priority_override() -> None:
    """submit_managed_task overrides priority settings when requested."""
    await cleanup_task_manager()
    await initialize_task_manager(max_workers=1)

    try:
        options = TaskOptions(
            task_id="singleton-task",
            priority=TaskPriority.LOW,
            timeout=0.5,
        )
        task_id = await submit_managed_task(
            _sync_identity,
            "singleton",
            priority=TaskPriority.CRITICAL,
            options=options,
        )
        manager = get_task_manager()
        result = await manager.wait_for_task(task_id, timeout=1.0)

        assert task_id == "singleton-task"
        assert result.status is TaskStatus.COMPLETED
        assert result.result == "singleton"
        with manager._task_lock:  # noqa: SLF001 - validated internal state
            task = manager._tasks[task_id]
        assert task.priority is TaskPriority.CRITICAL
    finally:
        await cleanup_task_manager()
