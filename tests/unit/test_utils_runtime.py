"""Runtime utility tests covering GPU fallbacks and health checks."""

from __future__ import annotations

import types
from typing import cast

import pytest

from src.config import Config
from src.utils.gpu import get_gpu_stats, is_gpu_available, safe_gpu_operation
from src.utils.health_checks import ServiceHealthChecker
from tests._helpers.config import make_test_settings


def test_gpu_helpers_fall_back_to_cpu() -> None:
    """GPU helpers should degrade gracefully when CUDA is unavailable."""
    assert is_gpu_available() is False

    def _primary() -> str:
        raise RuntimeError("gpu unavailable")

    def _fallback() -> str:
        return "cpu"

    assert safe_gpu_operation(_primary, fallback=_fallback) == "cpu"


def test_gpu_stats_structure() -> None:
    """GPU stats should expose the detection flags required by observability."""
    stats = get_gpu_stats()
    expected_keys = {
        "gpu_available",
        "torch_available",
        "cuda_available",
        "device_count",
        "devices",
    }
    assert expected_keys <= set(stats)
    assert isinstance(stats["devices"], list)


def test_health_checks_report_unconfigured_services(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Health checks should surface configuration issues without network calls."""
    settings = make_test_settings(tmp_path_factory.mktemp("health"))

    openai_status = ServiceHealthChecker.check_openai_connection(settings)
    assert openai_status["connected"] is False
    assert openai_status["error"]
    assert "OpenAI not configured" in str(openai_status["error"])

    firecrawl_status = ServiceHealthChecker.check_firecrawl_connection(settings)
    assert firecrawl_status["connected"] is False
    assert firecrawl_status["error"]
    assert "Firecrawl not configured" in str(firecrawl_status["error"])

    dragonfly_stub = types.SimpleNamespace(
        cache=types.SimpleNamespace(
            enable_dragonfly_cache=False, dragonfly_url="redis://localhost:6379/0"
        )
    )
    dragonfly_status = ServiceHealthChecker.check_dragonfly_connection(
        cast(Config, dragonfly_stub)
    )
    assert dragonfly_status["connected"] is False
    assert dragonfly_status["error"]
    assert "not enabled" in str(dragonfly_status["error"])
