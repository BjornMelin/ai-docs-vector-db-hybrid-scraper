"""Shared fixtures for embeddings service tests."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import suppress
from typing import Any

import pytest


class _DeterministicEncoding:
    """Minimal tiktoken-compatible encoder returning deterministic token ids."""

    def encode(self, text: str) -> list[int]:
        """Encode text into pseudo-token ids using character ordinals."""
        return [ord(char) % 127 for char in text]


@pytest.fixture(autouse=True)
def stub_tiktoken(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Provide deterministic tokenizer for all tests."""
    encoding = _DeterministicEncoding()

    monkeypatch.setattr(
        "tiktoken.encoding_for_model",
        lambda _model: encoding,
        raising=False,
    )
    monkeypatch.setattr(
        "tiktoken.get_encoding",
        lambda _name: encoding,
        raising=False,
    )

    yield


@pytest.fixture
def ai_operation_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Capture calls to record_ai_operation for assertions."""
    calls: list[dict[str, Any]] = []

    def _record_ai_operation(**payload: Any) -> None:
        calls.append(payload)

    # Patch the source and the provider-local bindings to capture all calls
    paths = [
        "src.services.observability.tracking.record_ai_operation",
        "src.services.embeddings.openai_provider.record_ai_operation",
        "src.services.embeddings.fastembed_provider.record_ai_operation",
    ]
    for path in paths:
        # Defensive: some modules may not be imported yet
        with suppress(Exception):  # pragma: no cover - safe best-effort patch
            monkeypatch.setattr(path, _record_ai_operation, raising=False)

    return calls
