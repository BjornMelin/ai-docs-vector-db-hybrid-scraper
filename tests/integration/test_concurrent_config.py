"""Integration exercises for concurrent configuration operations."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from src.config import ConfigManager, get_degradation_handler


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"openai": {"api_key": "sk-initial"}}),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def manager(config_file: Path) -> Iterator[ConfigManager]:
    mgr = ConfigManager(
        config_file=config_file,
        enable_file_watching=False,
        enable_graceful_degradation=True,
    )
    try:
        yield mgr
    finally:
        get_degradation_handler().reset()


@pytest.mark.asyncio
async def test_concurrent_reload_requests(
    manager: ConfigManager, config_file: Path
) -> None:
    """Multiple concurrent reload requests should succeed without race conditions."""

    async def mutate_and_reload(counter: int) -> None:
        payload = {"openai": {"api_key": f"sk-{counter}"}}
        config_file.write_text(json.dumps(payload), encoding="utf-8")
        await manager.reload_config_async()

    await asyncio.gather(*(mutate_and_reload(i) for i in range(5)))
    final_key = manager.get_config().openai.api_key
    assert final_key and final_key.startswith("sk-")


def test_parallel_readers(manager: ConfigManager) -> None:
    """Parallel access from multiple threads should return consistent config objects."""

    with ThreadPoolExecutor(max_workers=8) as executor:
        configs = list(executor.map(lambda _: manager.get_config(), range(32)))

    assert all(cfg.app_name for cfg in configs)
    assert len({id(cfg) for cfg in configs}) == len(configs)


def test_degradation_handler_reset(manager: ConfigManager) -> None:
    """The global degradation handler should reset after recording failures."""

    handler = get_degradation_handler()
    for idx in range(6):
        handler.record_failure("reload_config", RuntimeError("fail"), {"attempt": idx})
    assert handler.degradation_active is True

    handler.reset()
    assert handler.degradation_active is False
    assert list(handler.iter_records()) == []
