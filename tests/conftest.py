"""Shared pytest configuration for the repository's unit suites."""

from __future__ import annotations

pytest_plugins = [
    "tests.fixtures.async_fixtures",
    "tests.fixtures.async_isolation",
    "tests.fixtures.configuration",
    "tests.fixtures.external_services",
    "tests.fixtures.mock_factories",
    "tests.fixtures.parallel_config",
    "tests.fixtures.test_data",
    "tests.plugins.random_seed",
    "tests.fixtures.http_mocks",
]
