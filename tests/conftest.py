"""Shared pytest configuration for the repository's unit suites."""

from __future__ import annotations

import inspect


pytest_plugins = [
    "tests.fixtures.async_fixtures",
    "tests.fixtures.async_isolation",
    "tests.fixtures.configuration",
    "tests.fixtures.external_services",
    "tests.fixtures.mock_factories",
    "tests.fixtures.observability",
    "tests.fixtures.test_data_observability",
    "tests.fixtures.test_utils_observability",
    "tests.fixtures.parallel_config",
    "tests.fixtures.test_data",
    "tests.plugins.random_seed",
    "tests.fixtures.http_mocks",
]


def pytest_pycollect_makeitem(collector, name, obj):
    """Prevent pytest from treating custom exception helpers as test classes."""

    if inspect.isclass(obj) and name == "TestError":
        return []

    return None
