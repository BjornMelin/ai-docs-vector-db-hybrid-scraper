"""Tests for the RAGGenerator configuration validation helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from src.services.rag.generator import RAGGenerator
from src.services.rag.models import RAGConfig


class _AsyncRetriever:
    """Minimal retriever stub exposing an asynchronous ``ainvoke`` method."""

    async def ainvoke(self, query: str, /) -> list[str]:
        return [query]


class _SyncRetriever:
    """Retriever stub missing the asynchronous ``ainvoke`` contract."""

    def ainvoke(self, query: str, /) -> list[str]:
        return [query]


@pytest.mark.asyncio()
async def test_validate_configuration_accepts_async_retriever() -> None:
    """validate_configuration should pass when the retriever exposes ``ainvoke``."""

    generator = RAGGenerator(RAGConfig(), cast(Any, _AsyncRetriever()))

    await generator.validate_configuration()


@pytest.mark.asyncio()
async def test_validate_configuration_rejects_sync_retriever() -> None:
    """validate_configuration should raise when ``ainvoke`` is not async."""

    generator = RAGGenerator(RAGConfig(), cast(Any, _SyncRetriever()))

    with pytest.raises(TypeError):
        await generator.validate_configuration()
