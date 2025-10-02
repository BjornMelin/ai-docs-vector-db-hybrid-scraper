"""Benchmarks targeting the configuration reloader hot paths."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Iterator
from pathlib import Path
from typing import Any

import pytest

from src.config import get_config, set_config
from src.config.reloader import ConfigReloader, ReloadTrigger


def run_async(coro: Awaitable[Any]) -> Any:
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture(scope="module")
def config_file(tmp_path_factory) -> Path:
    config_path = Path(tmp_path_factory.mktemp("reload")) / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "app_name": "reload-benchmark",
                "version": "1.0.0",
                "mode": "simple",
                "environment": "testing",
            }
        ),
        encoding="utf-8",
    )
    return config_path


@pytest.fixture
def reloader() -> Iterator[ConfigReloader]:
    original = get_config()
    reloader = ConfigReloader()
    try:
        yield reloader
    finally:
        set_config(original)


def test_reload_latency(benchmark, config_file: Path, reloader: ConfigReloader):
    async def reload_once() -> None:
        await reloader.reload_config(
            trigger=ReloadTrigger.MANUAL,
            config_source=config_file,
            force=True,
        )

    benchmark(lambda: run_async(reload_once()))


def test_reload_history_growth(benchmark, config_file: Path, reloader: ConfigReloader):
    async def reload_multiple() -> None:
        for _ in range(5):
            await reloader.reload_config(
                trigger=ReloadTrigger.FILE_WATCH,
                config_source=config_file,
            )

    benchmark(lambda: run_async(reload_multiple()))
    history = reloader.get_reload_history(limit=5)
    assert len(history) == 5
