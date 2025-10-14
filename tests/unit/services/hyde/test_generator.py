"""Tests for the HyDE hypothetical document generator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.errors import EmbeddingServiceError
from src.services.hyde.config import HyDEConfig, HyDEPromptConfig
from src.services.hyde.generator import GenerationResult, HypotheticalDocumentGenerator


@pytest.fixture
def hyde_config() -> HyDEConfig:
    """Provide HyDE configuration with test defaults."""
    return HyDEConfig(
        num_generations=2,
        generation_temperature=0.7,
        max_generation_tokens=64,
        generation_model="gpt-4o-mini",
        generation_timeout_seconds=5,
        parallel_generation=False,
        min_generation_length=10,
        filter_duplicates=True,
        diversity_threshold=0.1,
    )


@pytest.fixture
def prompt_config() -> HyDEPromptConfig:
    """Provide HyDE prompt configuration defaults."""
    return HyDEPromptConfig()


@pytest.fixture
def async_openai_constructor(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    """Mock AsyncOpenAI client with complete usage metadata."""
    client = AsyncMock()
    client.responses.create = AsyncMock(
        return_value=MagicMock(
            output_text=(
                "Generated answer with enough tokens for this validation coverage "
                "path to retain meaningful content"
            ),
            usage=MagicMock(
                input_tokens=21,
                output_tokens=21,
                total_tokens=42,
            ),
        )
    )
    client.close = AsyncMock()

    factory = MagicMock(return_value=client)
    monkeypatch.setattr(
        "src.services.hyde.generator.AsyncOpenAI",
        factory,
    )
    return client


@pytest.fixture
def async_openai_without_total_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """Mock AsyncOpenAI client with missing total_tokens field."""
    client = AsyncMock()
    client.responses.create = AsyncMock(
        return_value=MagicMock(
            output_text=(
                "Generated answer with sufficient length to survive downstream "
                "post processing thresholds during validation"
            ),
            usage=MagicMock(
                input_tokens=10,
                output_tokens=6,
                total_tokens=None,
            ),
        )
    )
    client.close = AsyncMock()

    factory = MagicMock(return_value=client)
    monkeypatch.setattr(
        "src.services.hyde.generator.AsyncOpenAI",
        factory,
    )
    return client


class TestGenerationResultModel:
    """Tests for GenerationResult model serialization."""

    def test_model_dump_and_restore(self) -> None:
        """Verify GenerationResult serializes and deserializes correctly."""
        result = GenerationResult(
            documents=["doc"],
            generation_time=0.5,
            tokens_used=10,
            cost_estimate=0.002,
            cached=True,
            diversity_score=0.2,
        )

        payload = result.model_dump()
        restored = GenerationResult.model_validate(payload)

        assert restored.documents == ["doc"]
        assert restored.cached is True
        assert restored.tokens_used == 10


class TestHypotheticalDocumentGenerator:
    """Tests for HypotheticalDocumentGenerator lifecycle and generation."""

    def test_init_defaults(
        self, hyde_config: HyDEConfig, prompt_config: HyDEPromptConfig
    ) -> None:
        """Verify generator initializes with config and starts uninitialized."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key="sk-test",
        )

        assert generator.config is hyde_config
        assert generator.prompt_config is prompt_config
        assert generator.is_initialized() is False

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        async_openai_constructor: AsyncMock,
    ) -> None:
        """Verify generator initializes AsyncOpenAI client successfully."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key="sk-test",
        )

        await generator.initialize()

        assert generator.is_initialized() is True
        async_openai_constructor.responses.create.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_initialize_missing_api_key(
        self, hyde_config: HyDEConfig, prompt_config: HyDEPromptConfig
    ) -> None:
        """Verify initialization raises when API key is missing."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key=None,
        )

        with pytest.raises(
            EmbeddingServiceError, match="OpenAI API key not configured"
        ):
            await generator.initialize()

    @pytest.mark.asyncio
    async def test_generate_documents(
        self,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        async_openai_constructor: AsyncMock,
    ) -> None:
        """Verify generation produces documents with token usage tracking."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key="sk-test",
        )
        await generator.initialize()

        result = await generator.generate_documents("What is HyDE?")

        assert result.documents
        assert result.tokens_used > 0
        async_openai_constructor.responses.create.assert_awaited()

    @pytest.mark.asyncio
    async def test_generate_documents_uses_usage_fallback(
        self,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        async_openai_without_total_tokens: AsyncMock,
    ) -> None:
        """Verify fallback calculation when total_tokens missing from API."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key="sk-test",
        )
        await generator.initialize()

        result = await generator.generate_documents("Explain HyDE")

        assert result.tokens_used == 16
        async_openai_without_total_tokens.responses.create.assert_awaited()

    @pytest.mark.asyncio
    async def test_cleanup_closes_client(
        self,
        hyde_config: HyDEConfig,
        prompt_config: HyDEPromptConfig,
        async_openai_constructor: AsyncMock,
    ) -> None:
        """Verify cleanup closes AsyncOpenAI client properly."""
        generator = HypotheticalDocumentGenerator(
            config=hyde_config,
            prompt_config=prompt_config,
            api_key="sk-test",
        )
        await generator.initialize()
        await generator.cleanup()

        async_openai_constructor.close.assert_awaited_once()
