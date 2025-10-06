"""Shared CI fixtures for parallel execution tests."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from .test_isolation import IsolatedTestResources


@pytest.fixture
def isolated_resources() -> Generator[IsolatedTestResources, None, None]:
    """Provide isolated resource manager for CI tests."""
    resources = IsolatedTestResources()
    try:
        yield resources
    finally:
        resources.cleanup()


@pytest.fixture
def isolated_temp_dir(isolated_resources: IsolatedTestResources) -> Path:
    """Return a worker-specific temporary directory."""
    return isolated_resources.get_isolated_temp_dir()


@pytest.fixture
def isolated_port(isolated_resources: IsolatedTestResources) -> int:
    """Return a worker-specific port assignment."""
    return isolated_resources.get_isolated_port()
