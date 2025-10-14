"""Shared pytest configuration for the repository's unit suites."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest


_STUBS_PATH = Path(__file__).resolve().parent / "stubs"
if str(_STUBS_PATH) not in sys.path:
    sys.path.insert(0, str(_STUBS_PATH))

pytest_plugins = [
    "pytest_asyncio",
    "tests.fixtures.async_fixtures",
    "tests.fixtures.async_isolation",
    "tests.fixtures.configuration",
    "tests.fixtures.factories",
    "tests.fixtures.observability",
    "tests.fixtures.test_data_observability",
    "tests.fixtures.test_utils_observability",
    "tests.fixtures.parallel_config",
    "tests.fixtures.redis",
    "tests.fixtures.test_data",
]

_ALLOWED_SKIP_PREFIXES: tuple[str, ...] = (
    "need --runslow option to run",
    "respx is not installed",
    "Sparse embeddings unavailable",
    "need RUN_LOAD_TESTS=1 to run load tests",
)


def pytest_pycollect_makeitem(collector, name, obj):
    """Prevent pytest from treating custom exception helpers as test classes."""

    if inspect.isclass(obj) and name == "TestError":
        return []

    return None


def _extract_skip_reason(report) -> str:
    """Map a pytest report to the human-readable skip reason string."""

    longrepr = getattr(report, "longrepr", None)
    if not longrepr:
        return ""

    if isinstance(longrepr, str):
        return longrepr

    if isinstance(longrepr, tuple) and len(longrepr) >= 3:
        reason = longrepr[2]
        return (
            reason.replace("Skipped: ", "").strip() if isinstance(reason, str) else ""
        )

    reprcrash = getattr(longrepr, "reprcrash", None)
    if reprcrash:
        message = getattr(reprcrash, "message", "")
        return str(message).replace("Skipped: ", "").strip()

    return ""


def _is_skip_allowed(reason: str, prefixes: Iterable[str]) -> bool:
    """Return True when the recorded skip reason is part of the allowlist."""

    return any(reason.startswith(prefix) for prefix in prefixes)


def pytest_terminal_summary(terminalreporter) -> None:
    """Fail the test run when an unexpected skip reason is encountered."""

    skipped_reports = terminalreporter.stats.get("skipped", [])
    unexpected = []
    for report in skipped_reports:
        reason = _extract_skip_reason(report)
        if not _is_skip_allowed(reason, _ALLOWED_SKIP_PREFIXES):
            unexpected.append((report.nodeid, reason))

    if unexpected:
        summary_lines = "\n".join(
            f"- {nodeid}: {reason or 'no skip reason provided'}"
            for nodeid, reason in unexpected
        )
        pytest.exit(f"Unexpected skip reasons detected:\n{summary_lines}")
