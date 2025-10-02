"""Focused benchmarks covering configuration load and validation costs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.config import Config, get_config, set_config


@pytest.fixture(scope="module")
def config_payload() -> dict[str, Any]:
    """Provide a representative configuration payload for benchmarks."""

    return {
        "app_name": "benchmark",
        "version": "1.0.0",
        "mode": "simple",
        "environment": "testing",
        "debug": True,
        "log_level": "INFO",
        "embedding_provider": "fastembed",
        "crawl_provider": "crawl4ai",
        "openai_api_key": "sk-test-123456789",
        "qdrant_url": "http://localhost:6333",
    }


def test_config_instantiation_performance(benchmark, config_payload):
    """Benchmark time taken to instantiate the Config model with validation."""

    benchmark(lambda: Config(**config_payload))


def test_config_serialisation_round_trip(benchmark, config_payload):
    """Benchmark serialisation followed by deserialisation."""

    config = Config(**config_payload)

    def round_trip() -> Config:
        dumped = config.model_dump_json()
        return Config.model_validate_json(dumped)

    benchmark(round_trip)


def test_config_reload_from_file(benchmark, config_payload, tmp_path: Path):
    """Benchmark loading configuration from disk via ``set_config``."""

    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config_payload), encoding="utf-8")

    def reload() -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as tmp_file:
            tmp_file.write(json.dumps(config_payload))
            tmp_file.flush()
            new_config = Config.model_validate_json(
                Path(tmp_file.name).read_text(encoding="utf-8")
            )
            set_config(new_config)

    benchmark(reload)
    set_config(get_config(force_reload=True))
