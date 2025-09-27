"""Minimal pytest plugin to seed global RNGs deterministically."""

from __future__ import annotations

import random
from typing import Any

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register options that mirror pytest-randomly for deterministic seeding."""
    group = parser.getgroup("random-seed")
    group.addoption(
        "--randomly-seed",
        action="store",
        type=int,
        dest="randomly_seed",
        help="Set a deterministic random seed for Python and NumPy RNGs.",
    )
    group.addoption(
        "--randomly-dont-reset-seed",
        action="store_true",
        dest="randomly_keep_seed",
        help="Preserve the seed across tests instead of reseeding per test.",
    )


def _seed_all(seed: int) -> None:
    """Seed Python and optional NumPy RNGs."""
    random.seed(seed)
    try:
        import numpy.random as np_random  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel

        np_random.seed(seed)
    except ImportError:  # pragma: no cover - numpy optional in unit tests
        pass


def pytest_configure(config: pytest.Config) -> None:
    """Apply the configured seed and track plugin state for fixtures."""
    seed: int | None = config.getoption("randomly_seed")
    if seed is None:
        return
    _seed_all(seed)
    config._randomly_state = {  # type: ignore[attr-defined]
        "seed": seed,
        "keep": bool(config.getoption("randomly_keep_seed")),
    }


@pytest.fixture(autouse=True)
def _reseed_between_tests(pytestconfig: pytest.Config) -> None:
    """Reseed RNGs before each test when reset mode is enabled."""
    state: dict[str, Any] | None = getattr(  # type: ignore[attr-defined]
        pytestconfig, "_randomly_state", None
    )
    if not state or state["keep"]:
        return
    _seed_all(int(state["seed"]))
