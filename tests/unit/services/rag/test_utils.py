"""Tests for RAG service utility functions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.rag import utils
from src.services.rag.models import RAGConfig


class TestBuildDefaultRagConfig:
    """Unit tests for build_default_rag_config."""

    def test_extracts_standard_config_fields(self) -> None:
        """Maps expected fields from the global config into RAGConfig."""
        config = SimpleNamespace(
            rag=SimpleNamespace(
                model="gpt-4-turbo",
                temperature=0.5,
                max_tokens=800,
                max_results_for_context=10,
                include_sources=False,
                include_confidence_score=False,
            )
        )

        result = utils.build_default_rag_config(cast(Any, config))

        assert isinstance(result, RAGConfig)
        assert result.model == "gpt-4-turbo"
        assert result.temperature == 0.5
        assert result.max_tokens == 800
        assert result.retriever_top_k == 10
        assert result.include_sources is False
        assert result.confidence_from_scores is False

    def test_uses_defaults_for_missing_optional_fields(self) -> None:
        """Falls back to defaults when optional config attrs are missing."""
        config = SimpleNamespace(
            rag=SimpleNamespace(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=600,
                include_sources=True,
            )
        )

        result = utils.build_default_rag_config(cast(Any, config))

        assert result.retriever_top_k == 5
        assert result.confidence_from_scores is True


class TestInitialiseRagGenerator:
    """Unit tests for initialise_rag_generator."""

    @pytest.mark.asyncio
    async def test_constructs_retriever_and_initializes_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Initialise should wire retriever + generator and call initialize()."""
        config = SimpleNamespace(
            rag=SimpleNamespace(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=600,
                max_results_for_context=7,
                include_sources=True,
                include_confidence_score=True,
            ),
            qdrant=SimpleNamespace(collection_name="documents"),
        )
        vector_store = MagicMock()

        retriever = MagicMock()
        generator = MagicMock()
        generator.initialize = AsyncMock()

        def _fake_retriever_ctor(*, vector_service, collection: str, k: int):
            assert vector_service is vector_store
            assert collection == "documents"
            assert k == 7
            return retriever

        def _fake_generator_ctor(rag_config: RAGConfig, injected_retriever):
            assert injected_retriever is retriever
            assert rag_config.model == "gpt-4o-mini"
            return generator

        monkeypatch.setattr(utils, "VectorServiceRetriever", _fake_retriever_ctor)
        monkeypatch.setattr(utils, "RAGGenerator", _fake_generator_ctor)

        got_generator, got_config = await utils.initialise_rag_generator(
            cast(Any, config), cast(Any, vector_store)
        )

        assert got_generator is generator
        assert got_config.retriever_top_k == 7
        generator.initialize.assert_awaited_once()
