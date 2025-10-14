"""Pytest configuration for the stress testing suite."""

from __future__ import annotations

import pytest


STRESS_MARKERS: dict[str, str] = {
    "resource_exhaustion": "mark test as resource exhaustion test",
    "breaking_point": "mark test as breaking point identification test",
    "chaos": "mark test as chaos engineering test",
    "recovery": "mark test as recovery validation test",
    "slow": "mark test as slow-running test",
}


def pytest_configure(config: pytest.Config) -> None:
    """Register stress testing markers with pytest."""
    for marker, description in STRESS_MARKERS.items():
        config.addinivalue_line("markers", f"{marker}: {description}")
