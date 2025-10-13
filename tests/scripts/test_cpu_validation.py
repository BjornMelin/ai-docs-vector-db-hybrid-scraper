"""Tests for the CPU validation harness."""

from __future__ import annotations

import pytest

from scripts.validation import cpu_validation


def test_cpu_validation_passes() -> None:
    """The harness should succeed when scientific libraries operate correctly."""

    report = cpu_validation.run_cpu_validation()
    assert report.status == "passed"
    check_names = {check.name for check in report.checks}
    assert {"numpy-linear-algebra", "scipy-linalg"}.issubset(check_names)


def test_cpu_validation_handles_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Individual check failures should be reflected in the aggregate report."""

    def _broken_numpy_checks() -> tuple[
        cpu_validation.CheckResult, str, dict[str, object]
    ]:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(cpu_validation, "_numpy_checks", _broken_numpy_checks)

    report = cpu_validation.run_cpu_validation()
    assert report.status == "failed"
    assert any(check.status == "failed" for check in report.checks)
