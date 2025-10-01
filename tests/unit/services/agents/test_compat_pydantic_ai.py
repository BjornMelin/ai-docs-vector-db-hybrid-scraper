"""Compatibility tests for optional pydantic-ai integration."""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, cast

import pytest

from src.services.agents._compat import load_pydantic_ai


def test_load_pydantic_ai_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return sentinel values when optional dependency is missing."""
    monkeypatch.setitem(sys.modules, "pydantic_ai", None)
    available, agent_cls, run_context_cls = load_pydantic_ai()
    assert available is False
    assert agent_cls is None and run_context_cls is None


def test_load_pydantic_ai_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Surface the imported classes when dependency is present."""
    fake = cast(Any, ModuleType("pydantic_ai"))
    fake.Agent = object
    fake.RunContext = object
    monkeypatch.setitem(sys.modules, "pydantic_ai", fake)

    available, agent_cls, run_context_cls = load_pydantic_ai()
    assert available is True
    assert agent_cls is object and run_context_cls is object
